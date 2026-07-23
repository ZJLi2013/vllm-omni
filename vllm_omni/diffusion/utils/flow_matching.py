# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import torch
from torch import nn


def safe_linalg_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    # ROCm PyTorch wheels are built without CPU LAPACK.
    if matrix.device.type == "cpu" and not torch._C.has_lapack:
        return torch.from_numpy(np.linalg.solve(matrix.numpy(), rhs.numpy()))
    return torch.linalg.solve(matrix, rhs)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B, T, embedding_dim)
        return enc
