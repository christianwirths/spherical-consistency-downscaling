from src.configuration import parse_command_line
from src.training import training
from src.sde_model.model import SDEModel
from src.utils.utils import get_checkpoint_path, create_paths, get_date_time, get_latest_checkpoint, get_date_time
from src.consistency_model.model import Consistency
import xarray as xr
import os


def _setup_spherical_dimensions(config):
    """Set image dimensions for the spherical backbone.

    The spherical U-Net needs to know (H, W) at construction time to build
    graph Laplacians.  Both dimensions must be divisible by
    ``2^(spherical_depth - 1)`` for the 2×2 spatial pooling to work at
    every level.

    No spatial padding is applied — the spherical graph Laplacian already
    encodes proper wrap-around / polar connectivity.  This function:
      1. Probes the target NetCDF file for the raw (lat, lon) shape.
      2. Applies any latitude/longitude cropping from the config.
      3. If the resulting dimensions are NOT compatible with the requested
         depth, **auto-crops** latitude symmetrically (from both poles) to
         the nearest smaller compatible size.
      4. Sets ``config.pad_input = (0, 0, 0, 0)`` (no padding).
      5. Sets ``config.sample_dimension = (H, W)``.
    """
    ds = xr.open_dataset(f"{config.data_path}/{config.target_filename}")
    raw_h = len(ds.latitude)
    raw_w = len(ds.longitude)
    ds.close()

    # Account for user-specified cropping
    if config.crop_data_latitude != (None, None):
        raw_h = config.crop_data_latitude[1] - config.crop_data_latitude[0]
    if config.crop_data_longitude != (None, None):
        raw_w = config.crop_data_longitude[1] - config.crop_data_longitude[0]

    # ---- Auto-crop to satisfy divisibility requirement ----
    factor = 2 ** (config.spherical_depth - 1)

    if raw_h % factor != 0:
        target_h = (raw_h // factor) * factor          # largest multiple ≤ raw_h
        crop_total = raw_h - target_h                  # rows to remove
        crop_top = crop_total // 2
        crop_bot = raw_h - (crop_total - crop_top)
        # Merge with any existing user crop
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

    # No padding — the spherical graph handles boundaries natively
    config.pad_input = (0, 0, 0, 0)
    config.sample_dimension = (raw_h, raw_w)

    print(f"[Spherical] Grid dimensions: {raw_h}×{raw_w} (no padding)")
    print(f"[Spherical] Depth: {config.spherical_depth}, "
          f"channels: {config.spherical_channels} (DirectNeighConv, 9-tap)")


def main():
    """ Main executable to start training from command line. """

    config = parse_command_line()

    create_paths(config)
    
    
    # Use the --name CLI argument for the checkpoint directory so that
    # SLURM-requeued tasks always write to the correct checkpoint dir
    # (even if main.py is edited between requeues).
    checkpoint_name = config.name  # e.g. "spherical_3channel_v6_directneigh"
    if checkpoint_name is not None: 
        config.checkpoint_path = f'{config.checkpoint_path}/ckpt_{checkpoint_name}'
        os.makedirs(config.checkpoint_path, exist_ok=True)
        date_time = get_date_time()
        config.date_time = date_time
        resume_ckpt_path = get_latest_checkpoint(config.checkpoint_path)
        # if resume_ckpt_path is not None:
        #     print(f'Resuming training from checkpoint: {resume_ckpt_path}')
        # else:
        #     print('No checkpoint found. Starting training from scratch.')
    else:
        get_checkpoint_path(config)
        resume_ckpt_path = None

    if config.diffusion_model == 'consistency':
        # For the spherical backbone, compute padded dimensions before model construction
        if getattr(config, 'backbone_type', 'unet2d') == 'spherical':
            _setup_spherical_dimensions(config)
            # Spherical backbone benefits from clipped output to keep
            # predictions in [-1, 1] (the diffusers UNet2D achieves this
            # implicitly via GroupNorm; we must enforce it explicitly).
            config.clip_output = True
        model = Consistency(config)

    else:
        model = SDEModel(config)



    training(config, model, verbose=False, resume_ckpt_path=resume_ckpt_path)

    print("Training finished.")


if __name__ == "__main__":
    main()