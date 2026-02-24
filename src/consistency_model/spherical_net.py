"""
Spherical U-Net wrapper for the consistency model.

Wraps a graph-based U-Net so it can be used as a drop-in replacement for the
diffusers UNet2DModel inside the Consistency class. All consistency training
logic continues to operate on [B, C, H, W] tensors — the grid-to-graph and
graph-to-grid conversions happen entirely inside this module.

Architecture (structurally matches diffusers UNet2DModel):
  - **Encoder**: 2 × GraphResNetBlock per level (matches layers_per_block=2).
  - **Mid-block**: GraphResNetBlock → GraphSelfAttention → GraphResNetBlock.
  - **Decoder**: 2 × GraphResNetBlock per level with skip connections.
  - **Time conditioning**: Additive projection after first conv in each
    ResNet block (matches diffusers ``"default"`` time_embedding_norm).
  - **Normalization**: GroupNorm everywhere (train/eval invariant).
  - **Activation**: SiLU throughout.
  - **Residual connections**: Learned shortcut when channels differ.

Convolution operator — **DirectNeighConv** (v6):
  Uses a direct 9-tap spatial gather (self + 8 surrounding vertices) with
  ``nn.Linear(9 * F_in, F_out)``, analogous to a 3×3 Conv2d but with proper
  spherical topology:
    - **Longitude**: circular wrapping (East ↔ West boundary stitched).
    - **Poles**: reflection with 180° longitude shift (crossing the pole
      maps to the antipodal longitude, E/W direction reverses).
  This provides 9 anisotropic directional weights per filter, eliminating the
  isotropy bottleneck of the previous Chebyshev polynomial approach (K=3 had
  only 3 isotropic spectral weights with a T₀=identity shortcut).

Key design decisions:
  1. DirectNeighConv operates on a precomputed ``neigh_orders`` index array,
     making convolution equivalent to a 3×3 Conv2d on the equiangular grid
     but with correct spherical boundary handling.
  2. Neighbour indices are precomputed once at __init__ time and registered
     as buffers so they live on the correct device automatically.
  3. Pooling/upsampling use F.avg_pool2d / F.interpolate (image domain),
     with UpsampleConv2d for post-upsample smoothing.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight output container (matches diffusers UNet2DOutput interface)
# ---------------------------------------------------------------------------
@dataclass
class SphericalUNetOutput:
    """Drop-in replacement for ``diffusers.models.unet_2d.UNet2DOutput``."""
    sample: torch.Tensor


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (same idea as in the original DDPM U-Net)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Produce a sinusoidal embedding of scalar time values.

    Given a batch of time values ``t`` of shape ``[B]``, returns embeddings
    of shape ``[B, dim]`` using interleaved sin/cos frequencies, exactly as
    in *Attention Is All You Need* positional encodings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] tensor of time values.

        Returns:
            [B, dim] sinusoidal embedding.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Graph Laplacian utilities (vendored from DeepSphere to avoid hard dep)
# ---------------------------------------------------------------------------

def _scipy_csr_to_sparse_tensor(csr_mat):
    """Convert a scipy CSR sparse matrix to a torch sparse tensor."""
    coo = csr_mat.tocoo()
    indices = torch.LongTensor(np.array([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def _prepare_laplacian(laplacian):
    """Scale the Laplacian eigenvalues to [-1, 1] for Chebyshev filters."""

    def _estimate_lmax(lap, tol=5e-3):
        lmax = sparse.linalg.eigsh(
            lap, k=1, tol=tol,
            ncv=min(lap.shape[0], 10),
            return_eigenvectors=False,
        )[0]
        return lmax * (1 + 2 * tol)

    def _scale(L, lmax):
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L = L * (2.0 / lmax)
        L = L - I
        return L

    lmax = _estimate_lmax(laplacian)
    laplacian = _scale(laplacian, lmax)
    return _scipy_csr_to_sparse_tensor(laplacian)


def build_equiangular_laplacians(n_lat: int, n_lon: int, depth: int,
                                 laplacian_type: str = "combinatorial"):
    """Build a list of graph Laplacians for an equiangular grid at multiple
    resolutions (one per U-Net level).

    Uses PyGSP's ``SphereEquiangular`` graph.  Each pooling step halves
    both spatial dimensions via 2×2 average pooling.

    Requires the DeepSphere fork of PyGSP::

        pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP

    Args:
        n_lat: Number of latitude points (must be divisible by 2^(depth-1)).
        n_lon: Number of longitude points (must be divisible by 2^(depth-1)).
        depth: Number of U-Net levels (= number of Laplacians needed).
        laplacian_type: ``"combinatorial"`` or ``"normalized"``.

    Returns:
        Tuple of (laps, grid_dims) where:
          - ``laps`` is a list of sparse Laplacians ordered from *coarsest* to *finest*.
          - ``grid_dims`` is a list of (H, W) tuples in the same order.
    """
    try:
        from pygsp.graphs.sphereequiangular import SphereEquiangular
    except ImportError:
        raise ImportError(
            "Could not import SphereEquiangular from PyGSP. "
            "You need the DeepSphere fork:\n"
            "  pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP"
        )

    factor = 2 ** (depth - 1)
    if n_lat % factor != 0 or n_lon % factor != 0:
        raise ValueError(
            f"Grid {n_lat}×{n_lon} is not divisible by 2^(depth-1) = {factor}. "
            f"Either reduce spherical_depth or adjust the grid dimensions."
        )

    laps = []
    dims = []
    lat, lon = n_lat, n_lon
    for _ in range(depth):
        G = SphereEquiangular(size=(lat, lon))
        G.compute_laplacian(laplacian_type)
        laps.append(_prepare_laplacian(G.L))
        dims.append((lat, lon))
        lat //= 2
        lon //= 2
    return laps[::-1], dims[::-1]  # coarsest first, finest last


def build_healpix_laplacians(n_pixels: int, depth: int,
                             laplacian_type: str = "combinatorial"):
    """Build Laplacians for a HEALPix grid.

    Args:
        n_pixels: Total number of HEALPix pixels (= 12 * nside^2).
        depth:    Number of U-Net levels.
        laplacian_type: ``"combinatorial"`` or ``"normalized"``.

    Returns:
        Tuple of (laps, grid_dims) where:
          - ``laps`` is a list of sparse Laplacians ordered from *coarsest* to *finest*.
          - ``grid_dims`` is a list of (V,) tuples (1-D vertex counts) in the same order.
    """
    try:
        from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
    except ImportError:
        raise ImportError(
            "Could not import SphereHealpix from PyGSP. "
            "You need the DeepSphere fork:\n"
            "  pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP"
        )

    laps = []
    dims = []
    nside = int(math.sqrt(n_pixels / 12))
    for i in range(depth):
        subdiv = nside // (2 ** i)
        G = SphereHealpix(subdiv, nest=True, k=20)
        G.compute_laplacian(laplacian_type)
        laps.append(_prepare_laplacian(G.L))
        n_pix_level = 12 * subdiv * subdiv
        dims.append((n_pix_level,))
    return laps[::-1], dims[::-1]  # coarsest first, finest last


# ---------------------------------------------------------------------------
# Direct spatial neighbourhood for equiangular grids (v6)
# ---------------------------------------------------------------------------

def build_equiangular_neighbours(H: int, W: int) -> np.ndarray:
    """Build neighbour index array for an equiangular grid with spherical topology.

    Each vertex (i, j) has 9 neighbours in a fixed canonical order::

        [self, N, NE, E, SE, S, SW, W, NW]

    Boundary handling:
      - **Longitude** (j axis): circular wrapping — the East and West
        boundaries are stitched together.
      - **Latitude poles** (i = 0 or i = H−1): reflection across the pole
        with a 180° longitude shift.  Crossing the pole maps (i, j) to
        (i, (j + W//2) % W) and reverses the East/West sense, so NE at
        the original point becomes NW at the reflected point and vice versa.

    Args:
        H: Number of latitude points.
        W: Number of longitude points.

    Returns:
        neigh_orders: ``[H*W, 9]`` int64 array of flat vertex indices.
    """
    neigh = np.zeros((H * W, 9), dtype=np.int64)

    for i in range(H):
        for j in range(W):
            v = i * W + j

            # -- Longitude wrapping (circular) --
            je = (j + 1) % W   # east
            jw = (j - 1) % W   # west

            # Self
            neigh[v, 0] = v

            # East / West (always circular)
            neigh[v, 3] = i * W + je
            neigh[v, 7] = i * W + jw

            # -- North neighbours --
            if i > 0:
                i_n = i - 1
                neigh[v, 1] = i_n * W + j    # N
                neigh[v, 2] = i_n * W + je   # NE
                neigh[v, 8] = i_n * W + jw   # NW
            else:
                # Pole reflection: cross north pole → same row, lon + 180°
                j_refl = (j + W // 2) % W
                neigh[v, 1] = j_refl                        # N  (row 0)
                neigh[v, 2] = (j_refl - 1) % W              # NE (E/W flips)
                neigh[v, 8] = (j_refl + 1) % W              # NW (E/W flips)

            # -- South neighbours --
            if i < H - 1:
                i_s = i + 1
                neigh[v, 5] = i_s * W + j    # S
                neigh[v, 4] = i_s * W + je   # SE
                neigh[v, 6] = i_s * W + jw   # SW
            else:
                # Pole reflection: cross south pole → same row, lon + 180°
                j_refl = (j + W // 2) % W
                last = (H - 1) * W
                neigh[v, 5] = last + j_refl                  # S  (row H-1)
                neigh[v, 4] = last + (j_refl - 1) % W       # SE (E/W flips)
                neigh[v, 6] = last + (j_refl + 1) % W       # SW (E/W flips)

    return neigh


def build_equiangular_graph(n_lat: int, n_lon: int, depth: int):
    """Build neighbour-order arrays and grid dimensions for an equiangular
    grid at multiple resolutions (one per U-Net level).

    Each pooling step halves both spatial dimensions via 2×2 average pooling.

    Args:
        n_lat: Number of latitude points (must be divisible by 2^(depth-1)).
        n_lon: Number of longitude points (must be divisible by 2^(depth-1)).
        depth: Number of U-Net levels.

    Returns:
        Tuple of (neigh_orders_list, grid_dims) where:
          - ``neigh_orders_list`` is a list of ``[V, 9]`` int64 tensors
            ordered from *coarsest* to *finest*.
          - ``grid_dims`` is a list of (H, W) tuples in the same order.
    """
    factor = 2 ** (depth - 1)
    if n_lat % factor != 0 or n_lon % factor != 0:
        raise ValueError(
            f"Grid {n_lat}×{n_lon} is not divisible by 2^(depth-1) = {factor}. "
            f"Either reduce spherical_depth or adjust the grid dimensions."
        )

    all_neigh = []
    dims = []
    lat, lon = n_lat, n_lon
    for _ in range(depth):
        neigh = build_equiangular_neighbours(lat, lon)
        all_neigh.append(torch.from_numpy(neigh))
        dims.append((lat, lon))
        lat //= 2
        lon //= 2
    # Return coarsest first, finest last (matching Laplacian convention)
    return all_neigh[::-1], dims[::-1]


# ---------------------------------------------------------------------------
# Chebyshev graph convolution (vendored & simplified from DeepSphere)
# ---------------------------------------------------------------------------

def _cheb_conv(laplacian, x, weight):
    """Chebyshev polynomial graph convolution.

    Computes the Chebyshev polynomial basis T_0(L)x, T_1(L)x, …, T_{K-1}(L)x
    where L is the (rescaled) graph Laplacian, then linearly mixes them.

    Args:
        laplacian: [V, V] sparse tensor (rescaled to [-1, 1]).
        x:         [B, V, F_in] input signal.
        weight:    [K, F_in, F_out] filter weights (K = polynomial order).

    Returns:
        [B, V, F_out] filtered signal.
    """
    B, V, Fin = x.shape
    K, _, Fout = weight.shape

    # Flatten batch & feature dims so we can do a single sparse matmul
    x0 = x.permute(1, 2, 0).contiguous().view(V, Fin * B)  # V × Fin*B

    polynomials = [x0.unsqueeze(0)]  # T_0

    if K >= 2:
        x1 = torch.sparse.mm(laplacian, x0)                 # T_1
        polynomials.append(x1.unsqueeze(0))
        for _ in range(2, K):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0    # T_k recurrence
            polynomials.append(x2.unsqueeze(0))
            x0, x1 = x1, x2

    out = torch.cat(polynomials, dim=0)             # K × V × Fin*B
    out = out.view(K, V, Fin, B)                     # K × V × Fin × B
    out = out.permute(3, 1, 2, 0).contiguous()       # B × V × Fin × K
    out = out.view(B * V, Fin * K)
    out = out.matmul(weight.view(Fin * K, Fout))     # B*V × Fout
    out = out.view(B, V, Fout)
    return out


class ChebConvLayer(nn.Module):
    """Single Chebyshev graph convolution layer with optional bias."""

    def __init__(self, in_ch: int, out_ch: int, K: int):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_ch, out_ch))
        self.bias = nn.Parameter(torch.Tensor(out_ch))
        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.in_channels * self.K))
        self.weight.data.normal_(0, std)
        self.bias.data.fill_(0.01)

    def forward(self, lap, x):
        return _cheb_conv(lap, x, self.weight) + self.bias


# ---------------------------------------------------------------------------
# Direct spatial neighbourhood convolution (v6)
# ---------------------------------------------------------------------------

class DirectNeighConv(nn.Module):
    """Direct 9-tap spatial neighbourhood convolution on equiangular grid.

    Gathers 9 neighbours (self + 8 surrounding) per vertex and applies a
    learned linear transform, analogous to a 3×3 Conv2d but with proper
    spherical topology (circular longitude wrapping, pole reflection).

    Each of the 9 spatial positions has independent weights → **anisotropic**.
    This provides 9 directional degrees of freedom per filter, compared to
    ChebConv's 3 isotropic spectral weights.

    Args:
        in_ch:  Input feature channels.
        out_ch: Output feature channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight = nn.Linear(9 * in_ch, out_ch)

    def forward(self, neigh_orders, x):
        """Apply the direct spatial convolution.

        Args:
            neigh_orders: ``[V, 9]`` int64 tensor of neighbour indices.
            x:            ``[B, V, F_in]`` graph signal.

        Returns:
            ``[B, V, F_out]`` filtered signal.
        """
        # Gather neighbours: [B, V, 9, F_in]
        mat = x[:, neigh_orders]
        B, V = mat.shape[0], mat.shape[1]
        # Flatten spatial neighbourhood: [B, V, 9 * F_in]
        mat = mat.reshape(B, V, 9 * self.in_ch)
        # Apply linear: [B, V, F_out]
        return self.weight(mat)


# ---------------------------------------------------------------------------
# Building blocks for the Spherical U-Net
# ---------------------------------------------------------------------------

def _gn_num_groups(num_channels: int, preferred: int = 32) -> int:
    """Pick the largest feasible group count ≤ *preferred* that divides
    *num_channels*.  Falls back to 1 (≡ LayerNorm) for primes."""
    for g in (preferred, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1  # pragma: no cover


class SphericalChebBN(nn.Module):
    """ChebConv → GroupNorm → SiLU (no activation variant available too).

    Uses GroupNorm instead of BatchNorm so that behaviour is identical in
    train and eval mode — no running-statistics mismatch.
    """

    def __init__(self, in_ch, out_ch, lap, K, activation=True):
        super().__init__()
        self.register_buffer("lap", lap)
        self.conv = ChebConvLayer(in_ch, out_ch, K)
        self.norm = nn.GroupNorm(_gn_num_groups(out_ch), out_ch)
        self.activation = activation

    def forward(self, x):
        x = self.conv(self.lap, x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.activation:
            x = F.silu(x)
        return x


# ---------------------------------------------------------------------------
# Graph ResNet block — matches diffusers ResnetBlock2D logic
# ---------------------------------------------------------------------------

class GraphResNetBlock(nn.Module):
    """Graph-based ResNet block matching the 2D UNet ResnetBlock2D.

    Forward path:
        GN → SiLU → DirectNeighConv → time_proj(t) additive → GN → SiLU → DirectNeighConv
    Residual path:
        x  → (optional 1×1 shortcut if in_ch ≠ out_ch) → add to main path

    This exactly mirrors the ``ResnetBlock2D`` from diffusers:
      - Pre-norm (GroupNorm before convolution)
      - Additive time injection between the two convolutions
      - Learned residual shortcut when channel dimensions differ
      - SiLU activation throughout

    Args:
        in_ch:       Input feature channels.
        out_ch:      Output feature channels.
        neigh_orders: ``[V, 9]`` int64 tensor of neighbour indices.
        time_dim:    Dimensionality of the time embedding vector.
    """

    def __init__(self, in_ch: int, out_ch: int, neigh_orders, time_dim: int):
        super().__init__()

        # --- main path ---
        self.norm1 = nn.GroupNorm(_gn_num_groups(in_ch), in_ch)
        self.conv1 = DirectNeighConv(in_ch, out_ch)
        self.register_buffer("neigh_orders", neigh_orders)

        # Additive time projection (matches diffusers "default" mode)
        self.time_emb_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(_gn_num_groups(out_ch), out_ch)
        self.conv2 = DirectNeighConv(out_ch, out_ch)

        # --- residual shortcut ---
        if in_ch != out_ch:
            self.shortcut = nn.Linear(in_ch, out_ch)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x:     [B, V, F_in] graph signal.
            t_emb: [B, D_time] time embedding vector.
        Returns:
            [B, V, F_out] graph signal.
        """
        residual = x

        # 1) GN → SiLU → DirectNeighConv
        h = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = F.silu(h)
        h = self.conv1(self.neigh_orders, h)

        # 2) Additive time injection: project t_emb → [B, out_ch] → add
        t = F.silu(t_emb)
        t = self.time_emb_proj(t).unsqueeze(1)  # [B, 1, out_ch]
        h = h + t

        # 3) GN → SiLU → DirectNeighConv
        h = self.norm2(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = F.silu(h)
        h = self.conv2(self.neigh_orders, h)

        # 4) Residual connection
        return h + self.shortcut(residual)


# ---------------------------------------------------------------------------
# Graph self-attention — matches diffusers AttentionBlock
# ---------------------------------------------------------------------------

class GraphSelfAttention(nn.Module):
    """Multi-head self-attention for graph signals.

    Mirrors the diffusers ``AttentionBlock`` used in the 2D UNet mid-block:
      GN → Q,K,V linear projections → scaled dot-product attention →
      output projection → residual add.

    At the coarsest level (22×45 = 990 vertices) full self-attention is
    computationally cheap.

    Args:
        channels:  Feature dimension of the input.
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )

        self.group_norm = nn.GroupNorm(_gn_num_groups(channels), channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        """
        Args:
            x: [B, V, C] graph signal.
        Returns:
            [B, V, C] attention-refined graph signal.
        """
        residual = x
        B, V, C = x.shape

        # GroupNorm (expects [B, C, V])
        x = self.group_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Q, K, V projections → [B, num_heads, V, head_dim]
        q = self.to_q(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # [B, heads, V, head_dim]

        # Reshape back → [B, V, C]
        attn = attn.transpose(1, 2).contiguous().view(B, V, C)
        attn = self.to_out(attn)

        return attn + residual


class AvgPool2dGraph(nn.Module):
    """2-D spatial average pooling for equiangular graph signals.

    Reshapes [B, V, F] → [B, F, H, W], applies 2×2 average pooling,
    and re-flattens to [B, V/4, F].  This is the correct pooling for
    equiangular grids where the vertex ordering is row-major
    (lat × lon).  HEALPix grids would need 1-D pooling instead.
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        # x: [B, V, C]  where V = H * W
        B, V, C = x.shape
        x = x.view(B, self.h, self.w, C)   # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)          # [B, C, H, W]
        x = F.avg_pool2d(x, kernel_size=2) # [B, C, H/2, W/2]
        x = x.permute(0, 2, 3, 1)          # [B, H/2, W/2, C]
        return x.contiguous().view(B, -1, C)  # [B, V/4, C]


class AvgUnpool2dGraph(nn.Module):
    """2-D spatial upsampling for equiangular graph signals.

    Reshapes [B, V, F] → [B, F, H, W], applies 2× bilinear interpolation
    in both spatial dimensions, and re-flattens to [B, V*4, F].

    Uses bilinear mode instead of nearest-neighbour to avoid 2×2 block
    artefacts that Chebyshev graph convolutions cannot smooth
    (isotropic filters lack direction-specific kernels).
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        B, V, C = x.shape
        x = x.view(B, self.h, self.w, C)
        x = x.permute(0, 3, 1, 2)          # [B, C, H, W]
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        x = x.permute(0, 2, 3, 1)          # [B, 2H, 2W, C]
        return x.contiguous().view(B, -1, C)


class UpsampleConv2d(nn.Module):
    """Post-upsample Conv2d smoothing for graph signals.

    Operates in the 2-D image domain: reshapes [B, V, C] → [B, C, H, W],
    applies a 3×3 Conv2d with GroupNorm + SiLU, then reshapes back.

    This replaces the isotropic ChebConv (SphericalChebBN) that was unable
    to smooth nearest-neighbour / bilinear artefacts.  Conv2d has 9
    direction-specific weights per channel pair (vs. ChebConv's 3 isotropic
    spectral weights), matching the approach in diffusers' UpBlock2D.

    The AvgPool/Unpool layers already use F.avg_pool2d / F.interpolate in
    image domain, so applying Conv2d here is consistent with the existing
    design.

    Args:
        channels: Feature channels.
        h:        Height of the upsampled grid.
        w:        Width of the upsampled grid.
    """

    def __init__(self, channels: int, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=1, padding_mode="circular")
        self.norm = nn.GroupNorm(_gn_num_groups(channels), channels)

    def forward(self, x):
        """
        Args:
            x: [B, V, C] graph signal where V = H × W.
        Returns:
            [B, V, C] smoothed graph signal.
        """
        B, V, C = x.shape
        x = x.view(B, self.h, self.w, C)
        x = x.permute(0, 3, 1, 2)              # [B, C, H, W]
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = x.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]
        return x.view(B, V, C)


# ---------------------------------------------------------------------------
# Configurable Spherical Encoder / Decoder
# ---------------------------------------------------------------------------

class SphericalEncoder(nn.Module):
    """Configurable encoder with 2 ResNet blocks per level (matching 2D UNet).

    Each level applies **two** ``GraphResNetBlock``s — the first may change
    channels (in_ch → out_ch), the second keeps the same width (out_ch →
    out_ch).  This matches the diffusers ``DownBlock2D`` which stacks
    ``layers_per_block=2`` ResNet blocks at every resolution level.

    Args:
        channel_list:      E.g. ``[3, 128, 128, 256, 256]`` → 3 features in,
                           then 128 at level-0, 128 at level-1 (after pool), etc.
        neigh_orders_list: List of ``[V, 9]`` int64 tensors from *coarsest*
                           to *finest*.
        grid_dims:         List of (H, W) tuples at each resolution, from
                           *coarsest* to *finest*.
        time_dim:          Dimensionality of the time embedding for FiLM.
    """

    def __init__(self, channel_list: List[int], neigh_orders_list: List,
                 grid_dims: List[Tuple[int, int]] = None,
                 time_dim: int = 16):
        super().__init__()
        self.depth = len(channel_list) - 1  # number of encoder levels
        assert len(neigh_orders_list) >= self.depth, (
            f"Need at least {self.depth} neigh_orders, got {len(neigh_orders_list)}"
        )

        # Pooling layers (between levels, not before the first level)
        self.pools = nn.ModuleList()
        for i in range(1, self.depth):
            h, w = grid_dims[-i]
            self.pools.append(AvgPool2dGraph(h, w))

        # 2 ResNet blocks per level  (matching layers_per_block=2)
        self.level_blocks = nn.ModuleList()
        for i in range(self.depth):
            no_idx = -(i + 1)  # finest → coarser
            neigh = neigh_orders_list[no_idx]
            # block-0: may change channels (in → out)
            # block-1: keeps width (out → out)
            b0 = GraphResNetBlock(channel_list[i], channel_list[i + 1], neigh, time_dim)
            b1 = GraphResNetBlock(channel_list[i + 1], channel_list[i + 1], neigh, time_dim)
            self.level_blocks.append(nn.ModuleList([b0, b1]))

    def forward(self, x, t_emb):
        """Returns list of encoder outputs from *deepest* to *shallowest*.

        Args:
            x:     [B, V, F_in] graph signal.
            t_emb: [B, D_time] time embedding vector.
        """
        enc_outputs = []
        for i, blocks in enumerate(self.level_blocks):
            if i > 0:
                x = self.pools[i - 1](x)
            for blk in blocks:
                x = blk(x, t_emb)
            enc_outputs.append(x)
        # Return deepest first (for decoder concat order)
        return list(reversed(enc_outputs))


class SphericalDecoder(nn.Module):
    """Configurable decoder with 2 ResNet blocks per level + skip connections.

    Mirrors the 2D UNet ``UpBlock2D``: bilinear-unpool → Conv2d smooth →
    concat with skip → 2 × GraphResNetBlock.

    Post-upsample smoothing uses ``UpsampleConv2d`` (3×3 Conv2d with circular
    padding) for direction-specific smoothing after bilinear interpolation.

    Args:
        channel_list:      Encoder channel list *reversed* (deepest to shallowest).
                           E.g. if encoder was [3,128,128,256], this is [256,128,128].
        out_channels:      Number of output features in the final layer.
        neigh_orders_list: Same list as encoder (coarsest-to-finest).
        grid_dims:         List of (H, W) tuples at each resolution, from
                           *coarsest* to *finest*.
        time_dim:          Dimensionality of the time embedding for FiLM.
    """

    def __init__(self, channel_list: List[int], out_channels: int,
                 neigh_orders_list: List,
                 grid_dims: List[Tuple[int, int]] = None,
                 time_dim: int = 16):
        super().__init__()
        self.depth = len(channel_list) - 1

        # Unpooling layers
        self.unpools = nn.ModuleList()
        for i in range(self.depth):
            coarse_idx = -(self.depth - i + 1)
            h, w = grid_dims[coarse_idx]
            self.unpools.append(AvgUnpool2dGraph(h, w))

        # Post-upsample convolutions: Conv2d for direction-specific smoothing
        # after bilinear interpolation.  Operates in image domain [B, C, H, W]
        # — consistent with the pooling/unpooling layers.
        self.upsample_convs = nn.ModuleList()
        for i in range(self.depth):
            ch = channel_list[i]
            fine_idx = -(self.depth - i)          # target (finer) resolution
            h_fine, w_fine = grid_dims[fine_idx]  # dims after unpool
            self.upsample_convs.append(UpsampleConv2d(ch, h_fine, w_fine))

        # 2 ResNet blocks per decoder level
        self.level_blocks = nn.ModuleList()
        for i in range(self.depth):
            in_ch = channel_list[i] + channel_list[i + 1]  # concat with skip
            out_ch = channel_list[i + 1]
            no_idx = -(self.depth - i)  # move towards finer
            neigh = neigh_orders_list[no_idx]
            # block-0: absorbs skip (in_ch → out_ch), block-1: (out_ch → out_ch)
            b0 = GraphResNetBlock(in_ch, out_ch, neigh, time_dim)
            b1 = GraphResNetBlock(out_ch, out_ch, neigh, time_dim)
            self.level_blocks.append(nn.ModuleList([b0, b1]))

        # Final output projection at full resolution
        # GN + SiLU + DirectNeighConv (like diffusers conv_norm_out + conv_out)
        self.final_norm = nn.GroupNorm(_gn_num_groups(channel_list[-1]), channel_list[-1])
        self.final = DirectNeighConv(channel_list[-1], out_channels)
        self.register_buffer("final_neigh", neigh_orders_list[-1])

    def forward(self, enc_outputs, t_emb):
        """
        Args:
            enc_outputs: list of encoder features, deepest first.
                         enc_outputs[0] = deepest, enc_outputs[-1] = shallowest.
            t_emb: [B, D_time] time embedding vector.
        """
        x = enc_outputs[0]
        for i, blocks in enumerate(self.level_blocks):
            x = self.unpools[i](x)
            x = self.upsample_convs[i](x)   # Conv2d smooth after bilinear upsample
            skip = enc_outputs[i + 1]
            x = torch.cat([x, skip], dim=2)
            for blk in blocks:
                x = blk(x, t_emb)
        # Final output projection with GN + SiLU
        x = self.final_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.silu(x)
        x = self.final(self.final_neigh, x)
        return x


# ---------------------------------------------------------------------------
# Full Spherical U-Net (with mid block and per-level time conditioning)
# ---------------------------------------------------------------------------

class SphericalUNetCore(nn.Module):
    """Graph-based spherical U-Net matching the 2D UNet structure.

    Architecture (identical to diffusers UNet2DModel):
      - **Encoder**: 2 ResNet blocks per level, avg-pool between levels.
      - **Mid-block**: ResNet → Self-Attention → ResNet (at coarsest res).
      - **Decoder**: unpool + skip-concat → 2 ResNet blocks per level.

    Args:
        in_features:       Number of input features per vertex (= image channels).
        out_features:      Number of output features per vertex.
        channel_list:      Feature widths at each encoder level, e.g. ``[128, 128, 256, 256]``.
        neigh_orders_list: Neighbour-order arrays from coarsest to finest.
        grid_dims:         List of (H, W) tuples at each resolution, from
                           *coarsest* to *finest*.
        time_dim:          Dimensionality of the time embedding.
        attn_heads:        Number of self-attention heads in the mid-block.
    """

    def __init__(self, in_features: int, out_features: int,
                 channel_list: List[int], neigh_orders_list: List,
                 grid_dims: List[Tuple[int, int]] = None,
                 time_dim: int = 16,
                 attn_heads: int = 1):
        super().__init__()
        enc_channels = [in_features] + list(channel_list)
        self.encoder = SphericalEncoder(
            enc_channels, neigh_orders_list, grid_dims=grid_dims, time_dim=time_dim
        )

        # Mid-block: ResNet → Self-Attention → ResNet (matching 2D UNet)
        deepest_ch = channel_list[-1]
        self.mid_resnet1 = GraphResNetBlock(
            deepest_ch, deepest_ch, neigh_orders_list[0], time_dim
        )
        self.mid_attention = GraphSelfAttention(
            deepest_ch, num_heads=attn_heads
        )
        self.mid_resnet2 = GraphResNetBlock(
            deepest_ch, deepest_ch, neigh_orders_list[0], time_dim
        )

        dec_channels = list(reversed(channel_list))
        self.decoder = SphericalDecoder(
            dec_channels, out_features, neigh_orders_list,
            grid_dims=grid_dims, time_dim=time_dim
        )

    def forward(self, x, t_emb):
        """
        Args:
            x:     [B, V, F_in] graph signal.
            t_emb: [B, D_time] time embedding vector.
        """
        enc_out = self.encoder(x, t_emb)
        # enc_out[0] is the deepest — apply mid block
        h = self.mid_resnet1(enc_out[0], t_emb)
        h = self.mid_attention(h)
        h = self.mid_resnet2(h, t_emb)
        enc_out[0] = h
        return self.decoder(enc_out, t_emb)


# ---------------------------------------------------------------------------
# The public wrapper: SphericalUNetWrapper
# ---------------------------------------------------------------------------

class SphericalUNetWrapper(nn.Module):
    """Full wrapper that makes the SphericalUNet compatible with the
    consistency model's ``model(images, times)`` calling convention.

    Externally this module operates on **[B, C, H, W]** image tensors and
    returns a ``SphericalUNetOutput`` whose ``.sample`` field is also
    ``[B, C, H, W]``.  Internally it:

    1. Reshapes ``[B, C, H, W]`` → ``[B, V, C]``  (V = H × W).
    2. Computes a sinusoidal time embedding and passes it to the graph
       U-Net where it is injected at every encoder/decoder level via
       FiLM (Feature-wise Linear Modulation).
    3. Runs the ``SphericalUNetCore`` graph U-Net.
    4. Reshapes the output ``[B, V, C_out]`` → ``[B, C_out, H, W]``.

    Args:
        in_channels:  Number of image channels (e.g. 3 for pr/t2m/tcc).
        out_channels: Number of output channels.
        image_height: Height of the (padded) input images.
        image_width:  Width of the (padded) input images.
        channel_list: Feature widths at each encoder level.
        spherical_sampling: ``"equiangular"`` (only supported sampling).
        spherical_depth: Number of U-Net levels.
        laplacian_type: Unused (kept for backward-compatible CLI args).
        chebyshev_K:  Unused (kept for backward-compatible CLI args).
        time_emb_dim: Dimensionality of the sinusoidal time embedding.
        use_coord_channels: If ``True``, append 4 coordinate channels
            (sin/cos of latitude and longitude) to the input features.
            This breaks the longitude equivariance so that the model can
            learn geographically pinned features (e.g. orographic cold
            spots over the Andes / Himalayas).  The coordinate channels
            are injected *inside* the wrapper — the external interface
            (noise, sampling, loss) stays unchanged at ``in_channels``.
    """

    N_COORD_CHANNELS = 4  # sin(lat), cos(lat), sin(lon), cos(lon)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_height: int,
        image_width: int,
        channel_list: Tuple[int, ...] = (128, 128, 256, 256),
        spherical_sampling: str = "equiangular",
        spherical_depth: int = 4,
        laplacian_type: str = "combinatorial",
        chebyshev_K: int = 3,
        time_emb_dim: int = 64,
        use_coord_channels: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = image_height
        self.W = image_width
        self.time_emb_dim = time_emb_dim
        self.use_coord_channels = use_coord_channels

        # ---- Build neighbour-order arrays for DirectNeighConv ----
        if spherical_sampling != "equiangular":
            raise ValueError(
                f"DirectNeighConv only supports 'equiangular' sampling, "
                f"got '{spherical_sampling}'."
            )
        neigh_orders_list, grid_dims = build_equiangular_graph(
            image_height, image_width, spherical_depth
        )

        # Store neighbour orders as buffers (moved to device automatically)
        self._neigh_list = []  # placeholder list
        for i, no in enumerate(neigh_orders_list):
            self.register_buffer(f"_neigh_{i}", no)
            self._neigh_list.append(None)

        self.n_levels = len(neigh_orders_list)
        self.grid_dims = grid_dims  # stored for reference

        # ---- Time embedding MLP ----
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        # Project time embedding to a richer representation for FiLM
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ---- Coordinate channels (geographic position encoding) ----
        n_coord = self.N_COORD_CHANNELS if use_coord_channels else 0
        if use_coord_channels:
            self._build_coord_features(image_height, image_width)

        # ---- Spherical U-Net core ----
        # Input features = image channels + optional coordinate channels
        # (time injected via FiLM, coordinates via input concatenation)
        self.unet = SphericalUNetCore(
            in_features=in_channels + n_coord,
            out_features=out_channels,
            channel_list=list(channel_list[:spherical_depth]),
            neigh_orders_list=neigh_orders_list,
            grid_dims=grid_dims,
            time_dim=time_emb_dim,
        )

    def _build_coord_features(self, H: int, W: int):
        """Precompute geographic coordinate features as model buffers.

        Creates four channels encoding position on the sphere:
          - sin(lat), cos(lat)  — latitude encoding
          - sin(lon), cos(lon)  — longitude encoding (no discontinuity)

        All values lie in [-1, 1], matching the data normalisation range.
        The coordinates are stored as a flat ``[V, 4]`` buffer that is
        concatenated with the graph signal in :meth:`forward`.
        """
        # Latitude: evenly spaced from +90° to -90° (north → south)
        lat_rad = torch.linspace(math.pi / 2, -math.pi / 2, H)
        # Longitude: evenly spaced from 0 to just under 2π
        lon_rad = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)

        lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing='ij')

        # Stack as [V, 4] — graph domain (row-major flattening)
        coords = torch.stack([
            lat_grid.reshape(-1),              # sin(lat) — hemisphere
            torch.cos(lat_grid.reshape(-1)),   # cos(lat) — distance from pole
            torch.sin(lon_grid.reshape(-1)),   # sin(lon)
            torch.cos(lon_grid.reshape(-1)),   # cos(lon)
        ], dim=1)  # [V, 4]

        # sin(lat_grid) is just lat_grid since lat_grid is already the angle?
        # No — lat_grid *is* in radians; we want sin/cos of those angles.
        coords[:, 0] = torch.sin(lat_grid.reshape(-1))

        self.register_buffer('_coord_features', coords)  # [V, 4]

    def _get_neigh_orders(self):
        """Retrieve neighbour-order buffers (ensures correct device)."""
        return [getattr(self, f"_neigh_{i}") for i in range(self.n_levels)]

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> SphericalUNetOutput:
        """
        Args:
            images: [B, C, H, W] noisy image batch.
            times:  [B] or [1] diffusion time values.

        Returns:
            SphericalUNetOutput with ``.sample`` of shape [B, C_out, H, W].
        """
        B, C, H, W = images.shape
        V = H * W

        # 1. Image → graph signal: [B, C, H, W] → [B, V, C]
        x = images.permute(0, 2, 3, 1).reshape(B, V, C)

        # 1b. Append coordinate channels if enabled: [B, V, C] → [B, V, C+4]
        if self.use_coord_channels:
            coords = self._coord_features.unsqueeze(0).expand(B, -1, -1)  # [B, V, 4]
            x = torch.cat([x, coords], dim=2)

        # 2. Time embedding: [B] → [B, D_time]
        if times.dim() == 0:
            times = times.unsqueeze(0)
        if times.shape[0] == 1 and B > 1:
            times = times.expand(B)
        t_emb = self.time_embed(times)         # [B, D_time]
        t_emb = self.time_mlp(t_emb)           # [B, D_time]

        # 3. Update neigh_orders in the U-Net to use correct device buffers
        neigh_list = self._get_neigh_orders()
        self._update_unet_neigh_orders(neigh_list)

        # 4. Forward through spherical U-Net: [B, V, C_out]
        x = self.unet(x, t_emb)

        # 5. Graph signal → image: [B, V, C_out] → [B, C_out, H, W]
        x = x.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return SphericalUNetOutput(sample=x)

    def _update_unet_neigh_orders(self, neigh_list):
        """Push device-correct neighbour-order buffers into every block.

        This is necessary because the neigh_orders are registered as buffers
        on *this* wrapper module, but the ResNet blocks inside the U-Net have
        their own copies registered at __init__ time.  After ``.to(device)``
        the wrapper's buffers move but the inner copies may not.  This method
        synchronizes them.
        """
        # Walk through encoder (2 blocks per level)
        enc = self.unet.encoder
        for i, level_blocks in enumerate(enc.level_blocks):
            no_idx = -(i + 1)
            for blk in level_blocks:
                blk.neigh_orders = neigh_list[no_idx]

        # Mid block: 2 ResNets + attention (all at coarsest)
        self.unet.mid_resnet1.neigh_orders = neigh_list[0]
        self.unet.mid_resnet2.neigh_orders = neigh_list[0]

        # Walk through decoder (2 blocks per level)
        dec = self.unet.decoder
        for i, level_blocks in enumerate(dec.level_blocks):
            no_idx = -(dec.depth - i)
            for blk in level_blocks:
                blk.neigh_orders = neigh_list[no_idx]
            # Note: upsample_convs are Conv2d — no neigh_orders needed
        dec.final_neigh = neigh_list[-1]
