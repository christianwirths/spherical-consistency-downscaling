"""
Spherical sampling script for the spherical graph consistency model.

This script loads a trained spherical consistency model checkpoint and
generates samples either unconditionally or conditionally (stroke-guidance).

Usage examples
--------------
Unconditional sampling (from pure noise):
    python spherical_sampling.py -dm consistency --backbone_type spherical \
        --mode unconditional --checkpoint last_sperical.ckpt \
        --num_samples 16 --use_ema

Conditional sampling (stroke-guidance from ESM input):
    python spherical_sampling.py -dm consistency --backbone_type spherical \
        --mode conditional --checkpoint last_sperical.ckpt \
        --esm_filename era5_1deg_merged_short.nc \
        --test_start 1990 --test_end 1991 \
        --sample_time 0.468 --use_ema
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader

from src.configuration import parse_command_line, Config
from src.consistency_model.inference import ConsistencyInference as CMInference
from src.data import GeoDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Spherical config setup (mirrors main.py)
# ---------------------------------------------------------------------------

def setup_spherical_config(config: Config):
    """Configure spherical dimensions.

    Reads the target NetCDF file to determine (H, W), auto-crops latitude
    if needed for divisibility, and disables padding.
    """
    ds = xr.open_dataset(f"{config.data_path}/{config.target_filename}")
    raw_h = len(ds.latitude)
    raw_w = len(ds.longitude)
    ds.close()

    if config.crop_data_latitude != (None, None):
        raw_h = config.crop_data_latitude[1] - config.crop_data_latitude[0]
    if config.crop_data_longitude != (None, None):
        raw_w = config.crop_data_longitude[1] - config.crop_data_longitude[0]

    factor = 2 ** (config.spherical_depth - 1)

    # Auto-crop latitude for divisibility (symmetric, from both poles)
    if raw_h % factor != 0:
        target_h = (raw_h // factor) * factor
        crop_total = raw_h - target_h
        crop_top = crop_total // 2
        crop_bot = raw_h - (crop_total - crop_top)
        if config.crop_data_latitude == (None, None):
            config.crop_data_latitude = (crop_top, crop_bot)
        else:
            config.crop_data_latitude = (
                config.crop_data_latitude[0] + crop_top,
                config.crop_data_latitude[1] - (crop_total - crop_top),
            )
        raw_h = target_h
        print(f"[Spherical] Auto-cropped latitude to {raw_h} "
              f"(removed {crop_total} polar rows for divisibility by {factor})")

    if raw_w % factor != 0:
        target_w = (raw_w // factor) * factor
        crop_total = raw_w - target_w
        crop_left = crop_total // 2
        crop_right = raw_w - (crop_total - crop_left)
        if config.crop_data_longitude == (None, None):
            config.crop_data_longitude = (crop_left, crop_right)
        else:
            config.crop_data_longitude = (
                config.crop_data_longitude[0] + crop_left,
                config.crop_data_longitude[1] - (crop_total - crop_left),
            )
        raw_w = target_w
        print(f"[Spherical] Auto-cropped longitude to {raw_w} "
              f"(removed {crop_total} cols for divisibility by {factor})")

    # No padding — spherical graph encodes boundaries natively
    config.pad_input = (0, 0, 0, 0)
    config.sample_dimension = (raw_h, raw_w)

    print(f"[Spherical] Grid: {raw_h}×{raw_w}, depth: {config.spherical_depth}, "
          f"channels: {config.spherical_channels}, K: {config.chebyshev_K}")
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_dataset_to_config(ds, config):
    """Crop an xarray Dataset/DataArray to match config.crop_data_latitude/longitude.

    After auto-cropping the grid for divisibility, the model produces outputs
    of size (cropped_H, cropped_W).  The reference datasets used for xarray
    coordinate alignment must match, otherwise ``convert_to_xarray`` will
    raise a dimension-size conflict.
    """
    lat_crop = getattr(config, "crop_data_latitude", (None, None))
    lon_crop = getattr(config, "crop_data_longitude", (None, None))

    if lat_crop != (None, None) and lat_crop is not None:
        ds = ds.isel(latitude=slice(lat_crop[0], lat_crop[1]))
    if lon_crop != (None, None) and lon_crop is not None:
        ds = ds.isel(longitude=slice(lon_crop[0], lon_crop[1]))
    return ds


# ---------------------------------------------------------------------------
# Sampling modes
# ---------------------------------------------------------------------------

def run_unconditional(config: Config, inf: CMInference,
                      num_samples: int = 16, steps: int = 1,
                      use_ema: bool = True,
                      output_file: str = None):
    """Generate unconditional samples from pure noise."""

    # Reference datasets for inverse transforms + xarray conversion
    # training_target: training period data (1940-1990) for computing statistics
    full_data = xr.open_dataset(f"{config.data_path}/{config.target_filename}")
    full_data = _crop_dataset_to_config(full_data, config)
    inf.training_target = full_data.sel(
        time=slice(str(config.train_start), str(config.train_end)))
    inf.test_input = full_data  # For coordinate alignment

    samples = inf.run(
        convert_to_xarray=True,
        inverse_transform=True,
        num_samples=num_samples,
        steps=steps,
        use_ema=use_ema,
    )

    out = output_file or "./results/samples/spherical_unconditional.nc"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    samples["generated"].to_netcdf(out)
    print(f"✓ Unconditional samples saved to {out}")
    return samples


def run_conditional(config: Config, inf: CMInference,
                    sample_times: list = None,
                    steps: int = 1,
                    use_ema: bool = True,
                    output_file: str = None,
                    seed: int = 42):
    """Generate conditional samples via stroke-guidance from ESM input."""

    if sample_times is None:
        sample_times = [0.468]

    # Disable lazy loading for faster iteration
    config.lazy = False

    # Build test dataset from ESM data
    test_data = GeoDataset(
        stage="test",
        dataset_name="ESM",
        config=config,
        transform_esm_with_target_reference=True,
    )
    print(f"Test dataset: {test_data.num_samples} samples")

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )

    # Reference datasets for inverse transforms
    # training_target: training period data (1940-1990) for computing statistics
    full_era5 = xr.open_dataset(f"{config.data_path}/{config.target_filename}")
    full_era5 = _crop_dataset_to_config(full_era5, config)
    inf.training_target = full_era5.sel(
        time=slice(str(config.train_start), str(config.train_end)))

    # Test input for xarray coordinate alignment
    esm_ds = xr.open_dataset(f"{config.data_path}/{config.esm_filename}")
    esm_ds = _crop_dataset_to_config(esm_ds, config)
    esm_slice = esm_ds[config.predict_variable].sel(
        time=slice(str(config.test_start), str(config.test_end)))
    inf.test_input = esm_slice

    samples = inf.run_stroke_guidance(
        convert_to_xarray=True,
        inverse_transform=True,
        use_ema=use_ema,
        esm_dataloader=dataloader,
        sample_times=sample_times,
    )

    out = output_file or "./results/samples/spherical_conditional.nc"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    samples["generated"].to_netcdf(out)
    print(f"✓ Conditional samples saved to {out}")

    # Optionally save conditions + raw ESM
    out_cond = out.replace(".nc", "_conditions.nc")
    samples["conditions"].to_netcdf(out_cond)
    out_esm = out.replace(".nc", "_esm.nc")
    samples["esm"].to_netcdf(out_esm)
    print(f"  Conditions: {out_cond}")
    print(f"  Raw ESM:    {out_esm}")

    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_sampling_args():
    """Parse all arguments: config + sampling-specific.
    
    Uses two-stage parsing to avoid conflicts:
    1. Pre-parse sampling args to extract them
    2. Let parse_command_line() handle the rest
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--mode", type=str, default="unconditional",
                        choices=["unconditional", "conditional"],
                        help="Sampling mode")
    parser.add_argument("--checkpoint", type=str, default="last_sperical.ckpt",
                        help="Checkpoint filename (inside results/ or checkpoint_path)")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of unconditional samples to generate")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of denoising steps")
    parser.add_argument("--sample_time", type=float, nargs="+", default=[0.468],
                        help="Noising time(s) for conditional sampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .nc file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Parse only sampling args, leave the rest for parse_command_line()
    args, remaining = parser.parse_known_args()
    
    # Replace sys.argv with remaining args so parse_command_line() doesn't see sampling args
    sys.argv = [sys.argv[0]] + remaining
    
    return args


def main():
    """Main entry point for spherical sampling."""

    # Parse sampling args first and strip them from sys.argv
    sargs = parse_sampling_args()
    
    # Now parse training config (won't see sampling args)
    config = parse_command_line()

    set_seed(sargs.seed)

    # Checkpoint discovery (use proper spherical checkpoint directory)
    # Set the checkpoint_path to the spherical model directory (v7 with coordinate channels)
    config.checkpoint_path = "./data/checkpoints/ckpt_spherical_3channel_v7_coordchannels"
    
    # Set num_batches (required by inference.run() but not in default config)
    if not hasattr(config, 'num_batches') or config.num_batches is None:
        config.num_batches = 1  # Generate num_samples in a single batch

    # Set up spherical dimensions (no padding, validate grid)
    if getattr(config, "backbone_type", "unet2d") == "spherical":
        setup_spherical_config(config)

    # Build inference object
    inf = CMInference(config)

    # Load model from checkpoint
    print(f"Loading checkpoint: {sargs.checkpoint}")
    inf.load_model(checkpoint_fname=sargs.checkpoint)
    print(f"✓ Model loaded on {inf.device}")

    # Dispatch to sampling mode
    if sargs.mode == "unconditional":
        run_unconditional(
            config, inf,
            num_samples=sargs.num_samples,
            steps=sargs.steps,
            use_ema=config.use_ema,
            output_file=sargs.output,
        )
    elif sargs.mode == "conditional":
        assert config.esm_filename is not None, \
            "Conditional sampling requires --esm_filename"
        run_conditional(
            config, inf,
            sample_times=sargs.sample_time,
            steps=sargs.steps,
            use_ema=config.use_ema,
            output_file=sargs.output,
            seed=sargs.seed,
        )

    print("Done.")


if __name__ == "__main__":
    main()
