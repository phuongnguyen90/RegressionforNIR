"""Hàm mất mát tái tạo nhận biết tính liên tục phổ CASRL (Section 4.4, công thức
24-27).

L = L_rec + lambda * L_grad
L_rec  = sum_k w_k * (1/|I_k|) sum_{j in I_k} (S_j - S^_j)^2,   w_k = (l_k /
l_bar)^beta L_grad = (1/|M|) sum_{j in M} (dS_j - dS^_j)^2,  dS_j = S_j - S_{j-1}
Các đoạn I_k ở mức bước sóng được suy ra từ các đoạn mảnh của CMSM (nhân với P, kẹp
theo D). beta điều khiển mức ưu tiên cho đoạn dài; mặc định 1.0 (chỉnh trong
cấu hình).
"""

from __future__ import annotations

import torch


def segments_to_wavelength(segs, P: int, D: int):
    out = []
    for s, e in segs:
        a, b = s * P, min(e * P, D)
        if b > a:
            out.append((a, b))
    return out


def casrl_loss(
    x,
    x_hat,
    patch_mask,
    segs_per_sample,
    P: int,
    lam: float = 0.5,
    beta: float = 1.0,
    vanilla: bool = False,
):
    """x, x_hat: (B, D); patch_mask: bool (B, N); segs_per_sample: danh sách đoạn
    mảnh.
    vanilla=True: MSE thường trên vị trí bị che (biến thể "không CASRL")."""
    B, D = x.shape
    wmask = patch_mask.repeat_interleave(P, 1)[:, :D].float()
    sq = (x - x_hat) ** 2
    if vanilla:
        return (sq * wmask).sum() / wmask.sum().clamp_min(1)
    rec = x.new_zeros(())
    for b in range(B):
        segs = segments_to_wavelength(segs_per_sample[b], P, D)
        lens = torch.tensor([e - s for s, e in segs], dtype=x.dtype, device=x.device)
        w = (lens / lens.mean()) ** beta
        means = torch.stack([sq[b, s:e].mean() for s, e in segs])
        rec = rec + (w * means).sum()
    rec = rec / B
    dx = x[:, 1:] - x[:, :-1]
    dxh = x_hat[:, 1:] - x_hat[:, :-1]
    gm = wmask[:, 1:]
    grad = (((dx - dxh) ** 2) * gm).sum() / gm.sum().clamp_min(1)
    return rec + lam * grad
