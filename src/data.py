from typing import List, Optional, Type, Union, Tuple

import torch 
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from dataclasses import dataclass

from src.utils.transforms import apply_transforms


def compute_spherical_pad(raw_h: int, raw_w: int, depth: int):
    """Compute symmetric padding so that both spatial dimensions are
    divisible by ``2**depth``.

    .. deprecated::
        The spherical U-Net no longer uses spatial padding.  With 2x2-aware
        pooling, both dimensions only need to be divisible by
        ``2**(depth-1)``.  Use the raw grid directly and choose a depth
        that is compatible.  This function is kept for backward
        compatibility with the CNN (UNet2D) path only.

    Args:
        raw_h: Unpadded height (latitude).
        raw_w: Unpadded width (longitude).
        depth: Number of U-Net levels (= number of pooling steps).

    Returns:
        4-tuple ``(pad_left, pad_right, pad_top, pad_bottom)`` suitable for
        ``geo_pad``.
    """
    factor = 2 ** depth
    # target sizes: smallest multiples of factor >= raw dimension
    target_h = ((raw_h + factor - 1) // factor) * factor
    target_w = ((raw_w + factor - 1) // factor) * factor

    pad_h = target_h - raw_h
    pad_w = target_w - raw_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)

def get_dataloaders(config,
                    n_workers=1,
                    use_mnist=False,
                    ):
    """Prepares dataloaders for training and validation.

    Args:
        config: configuration containing file paths and hyperparameters.
        n_workers: number of worker processes
        use_mnist: use MNIST dataset instead

    Returns:
        Training and validation dataloaders.
     
    """

    if use_mnist:
        train_dataset= MNISTDataset(config, train=True)
        val_dataset = MNISTDataset(config, train=False)
    else:
        train_dataset = GeoDataset("train", "ERA5", config)
        val_dataset = GeoDataset("valid", "ERA5", config)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=n_workers,
                                      persistent_workers=True if n_workers > 0 else False,
                                      pin_memory=True)

    dataloaders['val'] = DataLoader(val_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    persistent_workers=True if n_workers > 0 else False,
                                    pin_memory=True)

    return dataloaders

def geo_pad(tensor, pad_config):
    """ Applies geographic padding to a 2D tensor.

    For the spherical backbone, pad_config is (0,0,0,0) — no padding is applied
    since the graph Laplacian handles spherical connectivity natively.

    Args:
        tensor: [channels, latitude, longitude] tensor to be padded
        pad_config: Tuple of 4 integers specifying the padding (left, right, top, bottom)

    Returns:
        Padded tensor (or original tensor when padding is all zeros).
    """
    if all(p == 0 for p in pad_config):
        return tensor
    pad = torch.nn.CircularPad2d((pad_config[0], pad_config[1], 0, 0))
    tensor = pad(tensor)
    pad = torch.nn.ReflectionPad2d((0, 0, pad_config[2], pad_config[3]))
    return pad(tensor)

class GeoDataset(torch.utils.data.Dataset):
    """ Dataset for ESM simulation and ERA5 reanalysis"""
    
    def __init__(self,
                 stage: str,
                 dataset_name: str,
                 config: dataclass,
                 epsilon: Optional[float]=0.0001,
                 transform_esm_with_target_reference: Optional[bool]=False):
        """ 
            stage: Train, valid or test.
            dataset_name: Either ESM or ERA5.
            config: Model configuration dataclass
            epsilon: Small constant for the log transform
            transform_esm_with_target_reference: Use target dataset to tranform the ESM data.
        """
        self.stage = stage
        self.config = config
        self.transforms = config.transforms
        self.epsilon = epsilon
        self.transform_esm_with_target_reference = transform_esm_with_target_reference

        self.target = None
        self.target_reference = None
        self.climate_model = None
        self.data = None

        if config.lazy:
            self.cache = False
            self.chunks = {'time': 1}
        else:
            self.cache = True
            self.chunks = None

        assert(stage in ['train', 'valid', 'test', 'proj']), "stage needs to be train, valid or test"

        self.splits = {
                "train": [str(config.train_start), str(config.train_end)],
                "valid": [str(config.valid_start), str(config.valid_end)],
                "test":  [str(config.test_start), str(config.test_end)],
                "proj":  ['2015', '2050'],
        }

        self.pad_config = config.pad_input
        #self.pad = torch.nn.ZeroPad2d(config.pad_input)
        self.pad = geo_pad

        assert(dataset_name in ['ESM', 'ERA5']), f"Dataset name {dataset_name} not supported"

        if dataset_name == "ERA5":
            self.prepare_target_data()

        elif dataset_name == "ESM":
            self.prepare_climate_model_data()
        

    def load_data(self, filename, is_reference=False):
        """ Loads data from file and applies some preprocessing.

        Args:
            is_reference: Loads data from the training period to be used as reference for transformations.
        """

        data_path: str = self.config.data_path + '/' + filename
        target = xr.open_dataset(data_path,
                               cache=self.cache, chunks=self.chunks)


        
        # assert len(list(target.keys())) <= 1, "more than one variable detected in target dataset."
        # self.config.predict_variable = list(target.keys())[0] 

        # target = target[self.config.predict_variable]

        # Extract fields and concatenate along a new channel dimension
        data_vars = []
        for var in self.config.predict_variable:
            data_vars.append(target[var])
        target = xr.concat(data_vars, dim='channel')
        target = target.transpose('time','channel', 'latitude', 'longitude')


        if is_reference:
            target = target.sel(time=slice(self.splits['train'][0],
                                           self.splits['train'][1]))
        else:
            target = target.sel(time=slice(self.splits[self.stage][0],
                                           self.splits[self.stage][1]))

        if self.config.crop_data_latitude != (None,None):
            target = target.isel(latitude=slice(self.config.crop_data_latitude[0],
                                                self.config.crop_data_latitude[1]))

        if self.config.crop_data_longitude != (None,None):
            target = target.isel(longitude=slice(self.config.crop_data_longitude[0],
                                                 self.config.crop_data_longitude[1]))

        if self.config.use_float16:
            target = target.astype(np.float16)

        return target
        

    def prepare_climate_model_data(self):
        """ Calls the climate model data loading and applies transformations.  """

        self.climate_model = self.load_data(self.config.esm_filename)
        if self.transform_esm_with_target_reference:
            climate_model_reference = self.load_data(self.config.target_filename, is_reference=True)
        else:
            climate_model_reference = self.load_data(self.config.esm_filename, is_reference=True)
        self.num_samples = len(self.climate_model.time.values)
        self.data = apply_transforms(self.climate_model, climate_model_reference, self.config)


    def prepare_target_data(self):
        """ Calls the target data loading and applies transformations.  """

        self.target = self.load_data(self.config.target_filename)
        self.target_reference = self.load_data(self.config.target_filename, is_reference=True)
        self.num_samples = len(self.target.time.values)
        self.data = apply_transforms(self.target, self.target_reference, self.config)


    def __getitem__(self, index):

        y = torch.from_numpy(self.data.isel(time=index).values).float()

        #y = self.pad(y)
        y = self.pad(y, self.pad_config)

        return y

    def __len__(self):
        return self.num_samples


