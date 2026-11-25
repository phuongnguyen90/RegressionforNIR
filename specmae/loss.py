"""Training objectives.

CASRL:  L = L_rec + lambda * L_grad
  L_rec  = sum_k w_k * mean_{j in I_k} (S_j - S^_j)^2,   w_k = (l_k / mean_k l_k)^beta
  L_grad = mean_{j in M} ((S_j - S_{j-1}) - (S^_j - S^_{j-1}))^2

Both are computed on masked wavelength positions only (padding is excluded).
L_rec is averaged over the batch; L_grad is averaged over all masked positions of
the batch (1/|M| normalisation).

``masked_mse`` is the plain MAE objective used for the "w/o CASRL" ablation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import LossConfig


def segment_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, seg_ids: torch.Tensor,
                                beta: float = 1.0, normalize_weights: bool = False) -> torch.Tensor:
    """Default: sum_k w_k * MSE_k. Because the weights sum to ~K, the term
    grows with the number of segments; ``normalize_weights=True`` divides by
    sum_k w_k so masks with different K (e.g. CMSM vs. random masking) are on the
    same scale."""
    sq = (pred - target) ** 2
    masked = seg_ids >= 0
    if not bool(masked.any()):
        return sq.new_zeros(())
    k = int(seg_ids.max().item()) + 1
    idx = seg_ids.clamp(min=0)
    m = masked.to(sq.dtype)
    seg_sum = sq.new_zeros(sq.shape[0], k).scatter_add_(1, idx, sq * m)
    seg_len = sq.new_zeros(sq.shape[0], k).scatter_add_(1, idx, m)
    valid = seg_len > 0
    seg_mse = seg_sum / seg_len.clamp(min=1.0)
    mean_len = seg_len.sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
    w = torch.where(valid, (seg_len / mean_len.clamp(min=1e-12)).pow(beta), torch.zeros_like(seg_len))
    per_sample = (w * seg_mse).sum(1)
    if normalize_weights:
        per_sample = per_sample / w.sum(1).clamp(min=1e-12)
    return per_sample.mean()


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    d_t = target[:, 1:] - target[:, :-1]
    d_p = pred[:, 1:] - pred[:, :-1]
    m = masked[:, 1:].to(pred.dtype)
    return (((d_t - d_p) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    m = masked.to(pred.dtype)
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, seg_ids: torch.Tensor,
                 cfg: LossConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    """pred/target: (B, D); seg_ids: (B, D) per-wavelength segment id, -1 = visible."""
    pred = pred.float()
    target = target.float()
    masked = seg_ids >= 0
    if cfg.name == "casrl":
        rec = segment_reconstruction_loss(pred, target, seg_ids, cfg.beta, cfg.normalize_weights)
        grad = gradient_consistency_loss(pred, target, masked)
        total = rec + cfg.grad_weight * grad
        return total, {"loss": float(total.detach()), "rec": float(rec.detach()), "grad": float(grad.detach())}
    if cfg.name == "mse":
        total = masked_mse(pred, target, masked)
        return total, {"loss": float(total.detach())}
    raise ValueError(f"unknown loss: {cfg.name}")
