"""Multi-Mask Ensemble Inference.

For every test spectrum, M independent masks are drawn with the training masking
procedure and the spectrum is reconstructed M times. Outputs:

  R_avg(l) = mean_m S^(m)(l)
  E(l)     = mean_m (S(l) - S^(m)(l))^2          ("mse", default)
           | var_m S^(m)(l) (unbiased)           ("variance", alternative)
  R(l)     = S(l) - R_avg(l)

``aggregation="full"`` (default) averages the complete reconstructions.
``aggregation="masked"`` instead averages, per wavelength, only the members in
which that wavelength was masked (falls back to all members for wavelengths
never masked), because the decoder is never trained on visible positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import InferenceConfig
from .masking import MaskSampler
from .model import SpecMultiMAE


@dataclass
class MMIEResult:
    r_avg: np.ndarray     # (n, D) ensemble-average reconstruction
    error: np.ndarray     # (n, D) error map E
    residual: np.ndarray  # (n, D) residual R = S - R_avg


@torch.no_grad()
def mmie(model: SpecMultiMAE, x: np.ndarray, sampler: MaskSampler, cfg: InferenceConfig,
         rng: np.random.Generator, device: Optional[torch.device] = None,
         ensemble_size: Optional[int] = None) -> MMIEResult:
    model.eval()
    device = device or next(model.parameters()).device
    M = int(ensemble_size or cfg.ensemble_size)
    if M < 1:
        raise ValueError("ensemble size must be >= 1")
    x = np.asarray(x, dtype=np.float32)
    n, D = x.shape
    r_avg = np.empty((n, D), np.float32)
    err = np.empty((n, D), np.float32)
    per = max(1, cfg.chunk_size // M)

    for i in range(0, n, per):
        xb = torch.from_numpy(x[i:i + per]).to(device)
        b = xb.shape[0]
        mask_np, _ = sampler.sample(b * M, rng)
        mask = torch.from_numpy(mask_np).to(device)
        pred = model(xb.repeat_interleave(M, dim=0), mask).float().view(b, M, D)
        target = xb.unsqueeze(1)
        sq = (target - pred) ** 2

        if cfg.aggregation == "full":
            avg = pred.mean(1)
            e_mse = sq.mean(1)
        elif cfg.aggregation == "masked":
            pm = model.pixel_mask(mask).view(b, M, D).float()
            cnt = pm.sum(1)
            has = cnt > 0
            avg = torch.where(has, (pred * pm).sum(1) / cnt.clamp(min=1), pred.mean(1))
            e_mse = torch.where(has, (sq * pm).sum(1) / cnt.clamp(min=1), sq.mean(1))
        else:
            raise ValueError(f"unknown aggregation: {cfg.aggregation}")

        if cfg.error_map == "mse":
            e = e_mse
        elif cfg.error_map == "variance":
            e = pred.var(1, unbiased=True) if M > 1 else torch.zeros_like(avg)
        else:
            raise ValueError(f"unknown error map: {cfg.error_map}")

        r_avg[i:i + b] = avg.cpu().numpy()
        err[i:i + b] = e.cpu().numpy()

    return MMIEResult(r_avg=r_avg, error=err, residual=x - r_avg)
