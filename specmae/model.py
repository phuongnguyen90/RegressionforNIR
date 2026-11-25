"""Asymmetric Transformer masked autoencoder.

Encoder: wavelength-aware tokens t_j = [e_j ; w_j] (2d) -> linear to d_e -> L_e
pre-LN Transformer blocks, applied to the *visible* tokens only.

Decoder: encoded visible tokens are projected to d_d and scattered back to their
patch positions; masked positions get a shared learnable mask token. Every
position is then concatenated with a decoder-side wavelength embedding
(m'_j = [m_j ; w_j] in R^{2 d_d}, and the same for visible tokens so both kinds
of token live in one space), fused back to d_d, passed through L_d blocks and a
linear head to P values per patch.

No index-based positional encoding is used: the wavelength embedding is the
only source of position information (except in the "sinusoidal" ablation).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig
from .embedding import LinearWavelengthEmbed, PatchEmbed, build_wavelength_embed


def _block(dim: int, heads: int, mlp_ratio: float, dropout: float) -> nn.Module:
    # Pre-LN block: Z = MHSA(LN(Z)) + Z ; Z = FFN(LN(Z)) + Z
    return nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio), dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)


class SpecMultiMAE(nn.Module):
    def __init__(self, wavelengths: Sequence[float], cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        wl = torch.as_tensor(np.asarray(wavelengths, dtype=np.float64), dtype=torch.float32)
        if wl.ndim != 1:
            raise ValueError("wavelengths must be 1-D")
        self.D = int(wl.shape[0])
        self.P = int(cfg.patch_size)
        self.N = math.ceil(self.D / self.P)
        self.pad = self.N * self.P - self.D

        centers = torch.stack([wl[j * self.P: min((j + 1) * self.P, self.D)].mean() for j in range(self.N)])
        lo, hi = cfg.wl_ref_range
        self.register_buffer("wavelengths", wl, persistent=False)
        self.register_buffer("patch_wavelengths", centers, persistent=False)
        self.register_buffer("patch_coords", (centers - lo) / (hi - lo), persistent=False)

        d, de, dd = cfg.embed_dim, cfg.encoder_dim, cfg.decoder_dim
        # ---- encoder
        self.patch_embed = PatchEmbed(self.P, d)
        self.enc_wl_embed = build_wavelength_embed(cfg.wavelength_embed, d, cfg.fourier_frequencies, cfg.fourier_sigma)
        self.enc_in = nn.Linear(2 * d, de)
        self.encoder = nn.ModuleList([_block(de, cfg.encoder_heads, cfg.mlp_ratio, cfg.dropout)
                                      for _ in range(cfg.encoder_depth)])
        self.enc_norm = nn.LayerNorm(de)
        # ---- decoder
        self.dec_embed = nn.Linear(de, dd)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
        dec_freqs = None if cfg.fourier_frequencies is None else max(1, cfg.fourier_frequencies * dd // d)
        self.dec_wl_embed = build_wavelength_embed(cfg.wavelength_embed, dd, dec_freqs, cfg.fourier_sigma)
        self.dec_in = nn.Linear(2 * dd, dd)
        self.decoder = nn.ModuleList([_block(dd, cfg.decoder_heads, cfg.mlp_ratio, cfg.dropout)
                                      for _ in range(cfg.decoder_depth)])
        self.dec_norm = nn.LayerNorm(dd)
        self.head = nn.Linear(dd, self.P)

        self.apply(self._init_weights)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():  # undo the zero-bias init where it would erase position information
            if isinstance(m, LinearWavelengthEmbed):
                m.reset_parameters()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------ shapes
    @property
    def n_patches(self) -> int:
        return self.N

    def patchify(self, x: torch.Tensor) -> torch.Tensor:  # (B, D) -> (B, N, P)
        if self.pad:
            x = torch.cat([x, x[:, -1:].expand(-1, self.pad)], dim=1)
        return x.reshape(x.shape[0], self.N, self.P)

    def unpatchify(self, p: torch.Tensor) -> torch.Tensor:  # (B, N, P) -> (B, D)
        return p.reshape(p.shape[0], self.N * self.P)[:, : self.D]

    def pixel_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """Patch mask (B, N) -> wavelength mask (B, D)."""
        return patch_mask.repeat_interleave(self.P, dim=1)[:, : self.D]

    def pixel_segments(self, seg_ids: torch.Tensor) -> torch.Tensor:
        """Per-patch segment ids (B, N; -1 visible) -> per-wavelength ids (B, D)."""
        return seg_ids.repeat_interleave(self.P, dim=1)[:, : self.D]

    # ------------------------------------------------------------ forward
    def forward_encoder(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        tokens = torch.cat([
            self.patch_embed(self.patchify(x)),
            self.enc_wl_embed(self.patch_coords).unsqueeze(0).expand(B, -1, -1).to(x.dtype),
        ], dim=-1)
        tokens = self.enc_in(tokens)

        n_vis = (~mask).sum(dim=1)
        if bool((n_vis != n_vis[0]).any()):
            raise ValueError("all spectra in a batch must have the same number of visible patches")
        n_keep = int(n_vis[0])
        # visible patches first, in their original order
        order = torch.argsort(mask.long() * self.N + torch.arange(self.N, device=mask.device), dim=1)
        keep = order[:, :n_keep]
        z = torch.gather(tokens, 1, keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        for blk in self.encoder:
            z = blk(z)
        return self.enc_norm(z), keep

    def forward_decoder(self, z: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z = self.dec_embed(z)
        dd = z.shape[-1]
        full = self.mask_token.to(z.dtype).expand(B, self.N, dd)
        full = full.scatter(1, keep.unsqueeze(-1).expand(-1, -1, dd), z)
        w = self.dec_wl_embed(self.patch_coords).unsqueeze(0).expand(B, -1, -1).to(full.dtype)
        h = self.dec_in(torch.cat([full, w], dim=-1))
        for blk in self.decoder:
            h = blk(h)
        return self.unpatchify(self.head(self.dec_norm(h)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, D) spectra, mask: (B, N) bool (True = masked). Returns full reconstruction (B, D)."""
        z, keep = self.forward_encoder(x, mask)
        return self.forward_decoder(z, keep)
