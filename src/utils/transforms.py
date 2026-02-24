import numpy as np
import xarray as xr


def apply_transforms(data: xr.DataArray,
                     data_ref: xr.DataArray,
                     config) -> xr.DataArray:
    """ Apply a sequence of transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants

    Returns:
        The transformed data
    
    """
    for channel in range(data.shape[1]):
        var_name = config.predict_variable[channel]
        config.transforms = config.transforms_per_variable[var_name]
        config.epsilon = config.epsilon_per_variable[var_name]

        if 'log' in config.transforms:
            data[:, channel, :, :] = log_transform(data[:, channel, :, :]   , config.epsilon)
            data_ref[:, channel, :, :] = log_transform(data_ref[:, channel, :, :], config.epsilon)

        if 'standardize' in config.transforms:
            data[:, channel, :, :] = standardize(data[:, channel, :, :], data_ref[:, channel, :, :])
            data_ref[:, channel, :, :] = standardize(data_ref[:, channel, :, :], data_ref[:, channel, :, :])

        if 'normalize' in config.transforms:
            data[:, channel, :, :] = norm_transform(data[:, channel, :, :], data_ref[:, channel, :, :])

        if 'normalize_minus1_to_plus1' in config.transforms:
            data[:, channel, :, :] = norm_minus1_to_plus1_transform(data[:, channel, :, :], data_ref[:, channel, :, :])
        
    return data   


def apply_inverse_transforms(data: xr.DataArray,
                            data_ref: xr.DataArray,
                            config) -> xr.DataArray:
    """ Apply a sequence of inverse transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set (raw, untransformed)
        config: Conifguration dataclass of transforms and constants
    
    Returns:
        The data tranformed back to the physical space
    """
    data_ref_ = data_ref.copy(deep=True)  # Deep copy to avoid mutating input

    for channel in range(3):
        var_name = config.predict_variable[channel]
        config.transforms = config.transforms_per_variable[var_name]
        config.epsilon = config.epsilon_per_variable[var_name]

        if "log" in config.transforms:
            data_ref_[var_name][:, :, :] = log_transform(data_ref_[var_name][:, :, :], config.epsilon)

        if "standardize" in config.transforms:
            data_ref_std = data_ref_.copy(deep=True)
            data_ref_std[var_name][:, :, :] = standardize(data_ref_[var_name][:, :, :], data_ref_[var_name][:, :, :]) 

        if "normalize_minus1_to_plus1" in config.transforms:
            if "standardize" in config.transforms:
                data[var_name][:, :, :] = inv_norm_minus1_to_plus1_transform(data[var_name][:, :, :], data_ref_std[var_name][:, :, :])
            else:
                data[var_name][:, :, :] = inv_norm_minus1_to_plus1_transform(data[var_name][:, :, :], data_ref_[var_name][:, :, :])
            
        if "standardize" in config.transforms:
            data[var_name][:, :, :] = inv_standardize(data[var_name][:, :, :], data_ref_[var_name][:, :, :])
        if "log" in config.transforms:
            data[var_name][:, :, :] = inv_log_transform(data[var_name][:, :, :], config.epsilon)

    return data


def log_transform(x, epsilon):
    return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    return np.exp(x + np.log(epsilon)) - epsilon


def standardize(x, x_ref):
    return (x - x_ref.mean(dim='time'))/x_ref.std(dim='time')


def inv_standardize(x, x_ref):
    x = x*x_ref.std(dim='time')
    x = x + x_ref.mean(dim='time')
    return x


def norm_transform(x, x_ref):
    return (x - x_ref.min(dim='time'))/(x_ref.max(dim='time') - x_ref.min(dim='time'))


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max(dim='time') - x_ref.min(dim='time')) + x_ref.min(dim='time')


def norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    if use_quantiles: 
        x = (x - x_ref.quantile(1-q_max,dim='time'))/(x_ref.quantile(q_max,dim='time') - x_ref.quantile(1-q_max,dim='time'))
    else:
        x = (x - x_ref.min())/(x_ref.max() - x_ref.min())
    x = x*2 - 1
    return x 


def inv_norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    x = (x + 1)/2
    if use_quantiles: 
        x = x * (x_ref.quantile(q_max) - x_ref.quantile(1-q_max)) + x_ref.quantile(1-q_max)
    else:
        x = x * (x_ref.max() - x_ref.min()) + x_ref.min()
    return x


