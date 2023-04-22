"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models, including
Valid Prediction Time (VPT), Valid Prediction Distance (VPD), etc.
"""
import torch


def recon_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].generation_len] - target[:, :kwargs['args'].generation_len]) ** 2
    sequence_pixel_mse = torch.mean(full_pixel_mses, dim=(1, 2, 3))
    return torch.mean(sequence_pixel_mse), torch.std(sequence_pixel_mse)
