from dataclasses import asdict
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import get_dataloaders
from src.configuration import Config
from src.utils.utils import MyProgressBar

# Free up GPU memory after each epoch
class EmptyCacheCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()  # free unused GPU memory at the end of each epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()  # optional, if validation accumulates memory


def training(config: Config,
             model: pl.LightningModule,
             verbose=True,
             resume_ckpt_path=None):
    """ Main training function.

    Args:
        config: Configuration containing hyperparameters and file paths
        model: The diffusion model to be trained
        verbose: Prints the trainig configuration
        resume_ckpt_path: Resumes training from a saved model checkpoint if provided.
    """

    pl.seed_everything(42, workers=True)

    if verbose:
        print(f'saving checkpoints at: {config.checkpoint_path}')
        print(json.dumps(asdict(config), sort_keys=False, indent=4))

    # save model checkpoints to disk
    checkpoints = ModelCheckpoint(
                    dirpath=config.checkpoint_path, 
                    save_top_k=2, 
                    monitor="val_loss",
                    save_last=True)
    callbacks = [checkpoints]

    # custom progress bar
    progressbar = MyProgressBar(config=config)
    callbacks.append(progressbar)

    # Log training statistics to Tensorboard
    tb_logger = TensorBoardLogger(config.tensorboard_path,
                                  name=config.name,
                                  default_hp_metric=False,
                                  version=config.date_time)
    model.config.tensorboard_path = config.tensorboard_path

    callbacks.append(EmptyCacheCallback())
    # Initialize trainer instance
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         callbacks=callbacks,
                         deterministic=False,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         gradient_clip_val=config.grad_clip_norm,
                         logger=tb_logger)
    info_lines = [
        "",
        "=" * 80,
        " Dataloader Configuration",
        "-" * 80,
        f" n_workers       : {config.n_workers}",
        "=" * 80,
        "",
    ]
    print("\n".join(info_lines))
    # Get dataloaders
    dataloaders = get_dataloaders(config,
                                  n_workers=config.n_workers,
                                  use_mnist=config.use_mnist)

    # Train the diffusion model
    trainer.fit(model,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'],
                ckpt_path=resume_ckpt_path) 

    # Save best performing model
    trainer.save_checkpoint(f"{config.checkpoint_path}/best_{config.diffusion_model}_model.ckpt")

    return None