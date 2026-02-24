# Spherical Consistency Downscaling



Statistical downscaling of global climate fields using a spherical graph consistency model. The model learns to stochastically map coarse-resolution ESM output (∼4°) to high-resolution ERA5 fields (1°) for precipitation, temperature, and total cloud cover simultaneously.

<img src="figures/downscaling_comparison.gif" alt="Coarse Reconstruction" width="100%">

## Method

The backbone is a spherical graph U-Net operating natively on the sphere, avoiding the polar distortion of regular-grid convolutions. Training follows the consistency model framework (Song et al., 2023) and especially its application for climate downscaling tasks (Hess at al. 2025), enabling single-step conditional sampling via stroke guidance at a fixed noise time $t^* = 0.468$ (as in Hess et al. 2025). Coordinate channels (sin/cos of lat/lon) are appended to break longitude equivariance.

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

## References

- P. Hess et al. (2025) — https://www.nature.com/articles/s42256-025-00980-5
- Y. Song et al. (2023), Consistency Models — https://arxiv.org/abs/2303.01469
- Y. Song et al. (2021), Score-based generative models — https://arxiv.org/abs/2011.13456
- Bischoff & Deck (2023) — https://arxiv.org/abs/2305.01822
- T. Karras et al. (2022), EDM — https://arxiv.org/abs/2206.00364