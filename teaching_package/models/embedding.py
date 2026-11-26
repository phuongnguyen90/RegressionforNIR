"""Nhúng mảnh phổ có nhận biết bước sóng (Section 4.1 của bài báo, mục 2.2.1 của báo
cáo).

e_j = LayerNorm(W_e p_j + b_e)                     (5)
lambda_j = trung bình bước sóng trong mảnh j         (6)
w_j = LayerNorm(W_w lambda_j + b_w)       [linear]   (7)
w_j = [sin(2 pi W_f lambda_j), cos(2 pi W_f lambda_j)]  [fourier, W_f học được]  (8)
t_j = Concat(e_j, w_j) in R^{2d}                    (9)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, P: int, d: int):
        super().__init__()
        self.P = P
        self.proj = nn.Linear(P, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, patches):  # (B, N, P)
        return self.norm(self.proj(patches))


class WavelengthEmbed(nn.Module):
    """Ánh xạ bước sóng trung tâm của mảnh (nm) sang véc-tơ d chiều.

    mode = "fourier" (mặc định), "linear", hoặc "sinusoidal" (mã hóa vị trí sin
    chuẩn theo chỉ số mảnh, dùng cho biến thể "không nhúng bước sóng" trong nghiên
    cứu loại bỏ). Bước sóng được đổi sang micromet (lambda/1000) trước khi nhúng để
    giữ ý nghĩa tuyệt đối, nhờ đó các máy đo có dải khác nhau vẫn dùng chung một
    thang.
    """

    def __init__(
        self,
        d: int,
        mode: str = "fourier",
        n_freq: int | None = None,
        freq_scale: float = 10.0,
    ):
        super().__init__()
        self.mode, self.d = mode, d
        if mode == "fourier":
            n_freq = n_freq or d // 2
            self.W_f = nn.Parameter(torch.randn(n_freq) * freq_scale)
            self.out = nn.Identity() if 2 * n_freq == d else nn.Linear(2 * n_freq, d)
        elif mode == "linear":
            self.lin = nn.Linear(1, d)
            self.norm = nn.LayerNorm(d)
        elif mode != "sinusoidal":
            raise ValueError(mode)

    def forward(self, lam_nm: torch.Tensor):  # (N,) hoặc (B, N)
        if self.mode == "sinusoidal":
            N = lam_nm.shape[-1]
            pos = torch.arange(N, device=lam_nm.device, dtype=torch.float32)[:, None]
            div = torch.exp(
                torch.arange(0, self.d, 2, device=lam_nm.device).float()
                * (-math.log(10000.0) / self.d)
            )
            pe = torch.zeros(N, self.d, device=lam_nm.device)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            return pe.expand(*lam_nm.shape, self.d) if lam_nm.dim() > 1 else pe
        lam = (lam_nm / 1000.0).unsqueeze(-1)  # micromet
        if self.mode == "linear":
            return self.norm(self.lin(lam))
        ang = 2 * math.pi * lam * self.W_f
        return self.out(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


def patchify(x: torch.Tensor, P: int):
    """(B, D) -> (B, N, P), đệm cuối bằng giá trị biên khi D không chia hết cho P."""
    B, D = x.shape
    N = math.ceil(D / P)
    pad = N * P - D
    if pad:
        x = torch.cat([x, x[:, -1:].expand(B, pad)], dim=1)
    return x.view(B, N, P)


def unpatchify(p: torch.Tensor, D: int):
    B, N, P = p.shape
    return p.reshape(B, N * P)[:, :D]


def patch_centers(wl: torch.Tensor, P: int):
    """Bước sóng trung tâm của từng mảnh (công thức 6); đệm bằng bước sóng cuối nếu
    cần."""
    D = wl.shape[0]
    N = math.ceil(D / P)
    pad = N * P - D
    if pad:
        step = (wl[-1] - wl[-2]) if D > 1 else wl.new_tensor(1.0)
        wl = torch.cat(
            [wl, wl[-1] + step * torch.arange(1, pad + 1, device=wl.device)]
        )
    return wl.view(N, P).mean(dim=1)
