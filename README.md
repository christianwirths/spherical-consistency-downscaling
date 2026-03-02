# Spherical Consistency Downscaling



Downscaling of global climate fields using a spherical graph consistency model. The model learns to stochastically map coarse-resolution ESM output (∼4°) to higher-resolution ERA5 fields (1°) for precipitation, temperature, and total cloud cover simultaneously. To represent Earth's spherical structure, we utilize a spherical graph neural network backbone operating natively on the $S^2$ manifold. This approach avoids polar singularities and edge distortions inherent to standard latitude-longitude projections.

<img src="figures/downscaling_comparison.gif" alt="Coarse Reconstruction" width="100%">

## Method

The backbone is a spherical graph U-Net (Ronneberger et al., 2015; Zhao et al., 2019) operating natively on the sphere, avoiding the polar distortion of regular-grid convolutions. Spatial convolutions use **DirectNeighConv** — a 9-tap spatial gather with correct spherical boundary handling (circular longitude wrapping, pole reflection with 180° longitude shift), providing the same anisotropic expressiveness as a standard 3×3 Conv2d. Training follows the consistency model framework (Song et al., 2023) and especially its application for climate downscaling tasks (Hess et al., 2025), enabling single-step conditional sampling via stroke guidance at a fixed noise time $t^* = 0.468$. Time conditioning uses sinusoidal embeddings (Ho et al., 2020) injected additively into each ResNet block. Coordinate channels (sin/cos of lat/lon) are appended to break longitude equivariance.

## Usage

**Training:**
```bash
python main.py --name "spherical_3channel_v7" \
    -dm consistency --backbone_type spherical \
    --spherical_depth 4 --spherical_channels 128 128 256 256 \
    --use_coord_channels --n_epochs 100 --batch_size 1
```

**Conditional sampling:**
```bash
python spherical_sampling.py \
    -dm consistency --backbone_type spherical \
    --mode conditional --checkpoint best_consistency_model.ckpt \
    --use_ema --sample_time 0.468
```

Data (ERA5, checkpoints) are not included and must be placed in `data/`.

## Tests

Run unit tests with 

```bash
python -m unittest discover tests/
```

## References

- This implementation builds on the repository: https://github.com/p-hss/consistency-climate-downscaling 

- P. Hess et al. (2025), Consistent and physically based climate downscaling — https://www.nature.com/articles/s42256-025-00980-5
- Y. Song et al. (2023), Consistency Models — https://arxiv.org/abs/2303.01469
- Y. Song et al. (2021), Score-based generative models — https://arxiv.org/abs/2011.13456
- T. Karras et al. (2022), EDM — https://arxiv.org/abs/2206.00364
- J. Ho et al. (2020), Denoising Diffusion Probabilistic Models — https://arxiv.org/abs/2006.11239
- M. Defferrard et al. (2020), DeepSphere: a graph-based spherical CNN — https://arxiv.org/abs/2012.15000
- F. Zhao et al. (2019), Spherical U-Net on Unstructured Meshes — https://arxiv.org/abs/1904.00906
- O. Ronneberger et al. (2015), U-Net — https://arxiv.org/abs/1505.04597
- Bischoff & Deck (2023) — https://arxiv.org/abs/2305.01822