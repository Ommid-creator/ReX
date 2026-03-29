#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import torch as tt
from scipy.ndimage import gaussian_filter


def __split_groups(neg_mask):
    # for some reason, it's much faster to do this on the cpu with numpy
    # than it is to use tensor_split
    return np.split(neg_mask, np.where(np.diff(neg_mask) > 1)[0] + 1)


def spectral_occlusion(
    mask: tt.Tensor, data: tt.Tensor, noise=0.02, device: str | tt.device = "cpu"
):
    """Linear interpolated occlusion for spectral data, with optional added noise.

    @param mask: boolean valued NDArray
    @param data: data to be occluded
    @param noise: parameter for optional gaussian noise.
        Set to 0.0 if you want simple linear interpolation

    @return torch.Tensor
    """
    neg_mask = tt.where(mask == 0)[0]
    split = __split_groups(neg_mask.detach().cpu().numpy())

    # strangely, this all seems to run faster if we do it on the cpu.
    # TODO Needs further investigation
    local_data = np.copy(data.detach().cpu().numpy())

    for s in split:
        if len(s) <= 1:
            return tt.from_numpy(local_data).to(device)
        start = s[0]
        stop = s[-1]
        dstart = data[:, :, start][0][0].item()
        dstop = data[:, :, stop][0][0].item()
        interp = tt.linspace(dstart, dstop, stop - start)
        if noise > 0.0:
            interp += np.random.normal(0, noise, len(interp))

        local_data[0, 0, start:stop] = interp

    return tt.from_numpy(local_data).to(device)


# Occlusions such as beach, sky, and other context-based occlusions


# Medical-based occlusions could be a CT scan of a healthy patient
def context_occlusion(mask: tt.Tensor, data: tt.Tensor, context: tt.Tensor, noise=0.5):
    """Context based occlusion with optional added noise.

    @param mask: boolean valued NDArray
    @param data: data to be occluded
    @param context: data to be used as occlusion e.g. CT scan of a healthy patient or a road
    @param noise: parameter for optional gaussian noise.
        Set to 0.0 for no noise

    @return torch.Tensor
    """
    if noise > 0.0:
        device = data.device
        context = context.to("cpu")  # As gaussian_filter expects cpu bound
        context = tt.tensor(gaussian_filter(context, sigma=noise), dtype=tt.float32).to(
            device
        )
    return tt.where(not mask, context, data)


def blur_occlusion(
    mask: tt.Tensor, data: tt.Tensor, sigma: float = 5.0
) -> tt.Tensor:
    """Gaussian blur occlusion: masked regions are replaced with a blurred
    version of the entire image, so edges between masked and visible regions
    are smoothed rather than hard-cut to a constant colour.

    @param mask: boolean tensor, True = keep, False = occlude
    @param data: image tensor of shape (1, C, H, W)
    @param sigma: standard deviation for gaussian blur kernel
    @return: torch.Tensor with masked regions replaced by blurred pixels
    """
    device = data.device
    np_data = data.detach().cpu().numpy()  # (1, C, H, W)
    blurred = np.zeros_like(np_data)
    for c in range(np_data.shape[1]):
        blurred[0, c] = gaussian_filter(np_data[0, c], sigma=sigma)
    blurred_tensor = tt.from_numpy(blurred).to(device)
    np_mask = mask.detach().cpu().numpy().astype(bool)
    result = np.where(np_mask, np_data, blurred)
    return tt.from_numpy(result).to(device)


def median_occlusion(
    mask: tt.Tensor, data: tt.Tensor
) -> tt.Tensor:
    """Median colour occlusion: masked regions are replaced with the
    per-channel median value of the entire image, giving a neutral
    fill that matches the overall colour distribution of the scene.

    @param mask: boolean tensor, True = keep, False = occlude
    @param data: image tensor of shape (1, C, H, W)
    @return: torch.Tensor with masked regions replaced by per-channel median
    """
    device = data.device
    np_data = data.detach().cpu().numpy()  # (1, C, H, W)
    result = np_data.copy()
    np_mask = mask.detach().cpu().numpy().astype(bool)
    for c in range(np_data.shape[1]):
        if np_mask.ndim == 4:
            chan_mask = np_mask[0, c]
        elif np_mask.ndim == 3:
            chan_mask = np_mask[c]
        else:
            chan_mask = np_mask
        median_val = float(np.median(np_data[0, c]))
        result[0, c] = np.where(chan_mask, np_data[0, c], median_val)
    return tt.from_numpy(result).to(device)
