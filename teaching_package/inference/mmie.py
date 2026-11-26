"""Suy luận tổ hợp đa mặt nạ MMIE (Algorithm 2, công thức 13-17) và điểm bất thường
(công thức 28).

Với mỗi phổ kiểm tra, sinh M mặt nạ CMSM độc lập, tái tạo M lần, rồi tính
  R_avg(l) = trung bình các phổ tái tạo                         (14)
  E(l)     = (1/M) sum_m (S(l) - S^(m)(l))^2   [MSE, dùng trong mọi thí nghiệm]  (15)
  R(l)     = S(l) - R_avg(l)                                      (17)
  A(S)     = (1/D) sum_l E(l)                                     (28)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from masking.cmsm import batch_masks, scattered_patch_mask


@torch.no_grad()
def mmie(
    model,
    X: np.ndarray,
    wl: np.ndarray,
    cfg: dict,
    M: int | None = None,
    device="cpu",
    seed: int = 123,
    chunk: int = 4096,
    masked_only: bool | None = None,
):
    """Trả về dict với R_avg, E, R (n, D) và score (n,).

    masked_only (mặc định lấy từ cfg["inference"]["masked_only"], True): lấy trung
    bình E và R_avg tại mỗi bước sóng trên các lượt mà bước sóng đó bị che, là các
    vị trí mô hình được huấn luyện để tái tạo. Đặt False để lấy trung bình trên phổ
    tái tạo đầy đủ theo công thức 14-15."""
    if masked_only is None:
        masked_only = cfg["inference"].get("masked_only", True)
    model.eval()
    M = M or cfg["inference"]["M"]
    P = cfg["model"]["P"]
    rng = np.random.default_rng(seed)
    wlt = torch.as_tensor(wl, dtype=torch.float32, device=device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, D = Xt.shape
    N = math.ceil(D / P)
    R_avg = torch.zeros(n, D, device=device)
    E = torch.zeros(n, D, device=device)
    per = max(1, chunk // M)  # số phổ mỗi lượt để lô = per * M
    for i in range(0, n, per):
        x = Xt[i : i + per]
        b = len(x)
        xr = x.repeat_interleave(M, 0)
        if cfg["masking"].get("strategy", "cmsm") == "scattered":
            m, _ = scattered_patch_mask(
                b * M, N, cfg["masking"]["mask_ratio"], rng, device
            )
        else:
            m, _ = batch_masks(b * M, N, cfg["masking"], rng, device)
        xh = model(xr, wlt, m).view(b, M, D)
        if masked_only:
            # chỉ lấy giá trị tái tạo tại các bước sóng bị che ở từng lượt (nơi mô
            # hình thực sự phải suy luận từ ngữ cảnh; tại vị trí nhìn thấy đầu ra
            # không được huấn luyện)
            w = m.repeat_interleave(P, 1)[:, :D].view(b, M, D).float()
            cnt = w.sum(1)
            R_avg[i : i + b] = torch.where(
                cnt > 0, (w * xh).sum(1) / cnt.clamp_min(1), x
            )
            E[i : i + b] = (w * (x[:, None, :] - xh) ** 2).sum(1) / cnt.clamp_min(1)
        else:
            R_avg[i : i + b] = xh.mean(1)
            E[i : i + b] = ((x[:, None, :] - xh) ** 2).mean(1)
    R = Xt - R_avg
    return {
        "R_avg": R_avg.cpu().numpy(),
        "E": E.cpu().numpy(),
        "R": R.cpu().numpy(),
        "score": E.mean(1).cpu().numpy(),
    }


def prominence(
    R: np.ndarray, E: np.ndarray, alpha: float = 0.5, normalize: bool = True
):
    """P(l) = alpha |R(l)| + (1 - alpha) E(l)  (công thức 30).
    normalize=True chia mỗi phổ cho giá trị lớn nhất của nó để các ngưỡng chiều cao
    và độ nổi bật (0,10 / 0,08) có cùng thang trên mọi bộ dữ liệu."""
    Pm = alpha * np.abs(R) + (1 - alpha) * E
    if normalize:
        Pm = Pm / np.maximum(Pm.max(axis=-1, keepdims=True), 1e-12)
    return Pm
