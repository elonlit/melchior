#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, C, H, W)
#         window_size: window size
#         h_w: Height of window
#         w_w: Width of window
#     Returns:
#         local window features (num_windows*B, window_size*window_size, C)
#     """
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
#     windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
#     return windows

# Partitions input signal into windows
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, L)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size, C)
    """
    B, C, L = x.shape
    x = x.view(B, C, L // window_size, window_size)
    windows = x.permute(0, 2, 3, 1).reshape(-1, window_size, C)
    return windows

# Reverses the window partitioning
def window_reverse(windows, window_size, L):
    """
    Args:
        windows: local window features (num_windows*B, window_size, C)
        window_size: Window size
        L: Length of sequence
    Returns:
        x: (B, C, L)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.reshape(B, L // window_size, window_size, -1)
    x = x.permute(0, 3, 1, 2).reshape(B, windows.shape[2], L)
    return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data = torch.randn(128,1,4096).to(device)
data = torch.randn(128, 1, 4096).to(device)
windows = window_partition(data, 64)
print(windows.shape)
reconstructed_data = window_reverse(windows, 64, 4096)
print(reconstructed_data.shape)

# print(window_partition(data, 64).shape)
# print(medmamba_t(data).shape)