"""Wavelength-Aware Patch Embedding.

* ``PatchEmbed``:            e_j = LayerNorm(W_e p_j + b_e)                     in R^d
* ``LinearWavelengthEmbed``: w_j = LayerNorm(W_w lambda_j + b_w)              in R^d
* ``FourierWavelengthEmbed``:w_j = [sin(2 pi W_f lambda_j), cos(2 pi W_f lambda_j)]  (learnable W_f)
* ``SinusoidalPositionEmbed``: ablation without wavelength information -- standard
  index-based sinusoidal positional encoding that ignores the physical wavelength.

The token is t_j = Concat(e_j, w_j) in R^{2d}; the concatenation happens in the model.
All wavelength embeddings take the *normalised* central wavelength of each patch.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_size, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # (B, N, P) -> (B, N, d)
        return self.norm(self.proj(patches))


class LinearWavelengthEmbed(nn.Module):
    """Note: LayerNorm(w*lambda) is identical for every lambda > 0, so the bias must
    be non-zero (and is re-initialised after the model's generic init) and the
    coordinate is centred to [-1, 1] for the embedding to encode position."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proj.weight, std=1.0)
        nn.init.normal_(self.proj.bias, std=1.0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # (N,) -> (N, d)
        return self.norm(self.proj((2.0 * coords - 1.0).unsqueeze(-1)))


class FourierWavelengthEmbed(nn.Module):
    """NeRF-style Fourier features with learnable frequencies W_f.

    With ``num_frequencies = dim // 2`` the output is exactly the concatenated
    concatenated sin/cos vector. If a different number of frequencies is requested
    (hyper-parameter sweep 128/256/512 at fixed d) a linear map projects 2F -> d.
    """

    def __init__(self, dim: int, num_frequencies: Optional[int] = None, sigma: float = 10.0):
        super().__init__()
        f = num_frequencies or dim // 2
        self.freqs = nn.Parameter(torch.randn(f) * sigma)
        self.proj = nn.Identity() if 2 * f == dim else nn.Linear(2 * f, dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # (N,) -> (N, d)
        ang = 2.0 * math.pi * coords.float().unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class SinusoidalPositionEmbed(nn.Module):
    """Fixed Transformer positional encoding over patch *indices* (ablation)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:  # (N,) -> (N, d)
        n = coords.shape[0]
        half = self.dim // 2
        pos = torch.arange(n, device=coords.device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(half, device=coords.device, dtype=torch.float32)
                        * (-math.log(10000.0) / max(half, 1)))
        pe = torch.zeros(n, self.dim, device=coords.device)
        pe[:, 0:2 * half:2] = torch.sin(pos * div)
        pe[:, 1:2 * half:2] = torch.cos(pos * div)
        return pe


def build_wavelength_embed(kind: str, dim: int, num_frequencies: Optional[int] = None,
                           sigma: float = 10.0) -> nn.Module:
    kind = kind.lower()
    if kind == "fourier":
        return FourierWavelengthEmbed(dim, num_frequencies, sigma)
    if kind == "linear":
        return LinearWavelengthEmbed(dim)
    if kind in ("sinusoidal", "positional", "none"):
        return SinusoidalPositionEmbed(dim)
    raise ValueError(f"unknown wavelength embedding: {kind}")
