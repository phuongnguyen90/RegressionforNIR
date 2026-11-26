"""Che mặt nạ đa đoạn liên tục CMSM (Algorithm 1 của bài báo, mục 2.2.2 của báo cáo).

Mặt nạ được sinh ở mức mảnh phổ (patch): mỗi đoạn là một dãy mảnh liền kề, độ dài
lấy đều trong [l_min, l_max] mảnh; các đoạn không chồng lấn; dừng khi số mảnh bị che
đạt floor(r * N). Hàm trả về cả mặt nạ mức mảnh lẫn danh sách đoạn để hàm mất mát
CASRL dùng trọng số theo độ dài đoạn.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def cmsm_patch_mask(
    N: int,
    r: float = 0.75,
    l_min: int = 8,
    l_max: int = 64,
    K_range=(4, 8),
    rng: np.random.Generator | None = None,
    max_tries: int = 10_000,
):
    """Sinh một mặt nạ CMSM cho chuỗi N mảnh.

    Trả về (mask, segments): mask là mảng bool (N,), True = bị che;
    segments là danh sách (start, end) theo chỉ số mảnh, end không bao gồm.

    Ghi chú cài đặt:
    - K lấy đều trong K_range và là số đoạn tối đa; khi chưa đạt tỉ lệ che, các đoạn
      được nối dài sang hai phía (vẫn không chồng lấn) cho tới khi đạt đúng tỉ lệ r,
      nên mọi phổ trong lô có cùng số mảnh nhìn thấy.
    - l_min, l_max được kẹp theo N để dùng được với phổ ngắn (vd. FDS-NIR các máy
      cầm tay, MSMI-NIR sau nội suy 32 điểm).
    """
    rng = rng or np.random.default_rng()
    target = max(1, math.floor(r * N))
    l_max_eff = max(1, min(l_max, N))
    l_min_eff = max(1, min(l_min, l_max_eff, max(1, target // K_range[0])))
    K = int(rng.integers(K_range[0], K_range[1] + 1))
    mask = np.zeros(N, dtype=bool)
    segs: list[list[int]] = []
    count, tries = 0, 0
    while count < target and len(segs) < K and tries < max_tries:
        tries += 1
        s = int(rng.integers(0, N))
        l = int(rng.integers(l_min_eff, l_max_eff + 1))
        e = min(s + l, N)
        e = min(e, s + (target - count))  # không vượt tỉ lệ che mục tiêu
        if e <= s or mask[s:e].any():
            continue
        mask[s:e] = True
        segs.append([s, e])
        count += e - s
    # nối dài các đoạn nếu chưa đạt tỉ lệ (khi đã đủ K đoạn hoặc hết lượt thử)
    while count < target:
        grown = False
        for seg in rng.permutation(len(segs)) if segs else []:
            s, e = segs[seg]
            for side in rng.permutation(2):
                if count >= target:
                    break
                if side == 0 and s > 0 and not mask[s - 1]:
                    s -= 1
                    mask[s] = True
                    count += 1
                    grown = True
                elif side == 1 and e < N and not mask[e]:
                    mask[e] = True
                    e += 1
                    count += 1
                    grown = True
            segs[seg] = [s, e]
        if not grown:  # không thể nối dài: thêm đoạn mới ở vùng trống
            free = np.flatnonzero(~mask)
            j = int(rng.choice(free))
            mask[j] = True
            segs.append([j, j + 1])
            count += 1
    # gộp các đoạn kề nhau thành một đoạn liên tục
    segs = _merge(sorted(segs))
    return mask, segs


def _merge(segs):
    out = []
    for s, e in segs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def batch_masks(B: int, N: int, cfg: dict, rng: np.random.Generator, device="cpu"):
    """Sinh B mặt nạ độc lập. Trả về tensor bool (B, N) và danh sách đoạn cho từng
    mẫu."""
    masks, segs = [], []
    for _ in range(B):
        m, s = cmsm_patch_mask(
            N,
            cfg.get("mask_ratio", 0.75),
            cfg.get("l_min", 8),
            cfg.get("l_max", 64),
            tuple(cfg.get("K_range", (4, 8))),
            rng,
        )
        masks.append(m)
        segs.append(s)
    return torch.as_tensor(np.stack(masks), device=device), segs


def scattered_patch_mask(
    B: int, N: int, r: float, rng: np.random.Generator, device="cpu"
):
    """Che ngẫu nhiên rời rạc kiểu MAE gốc (biến thể "không CMSM" trong nghiên cứu
    loại bỏ)."""
    k = max(1, math.floor(r * N))
    masks = np.zeros((B, N), dtype=bool)
    for b in range(B):
        masks[b, rng.choice(N, k, replace=False)] = True
    segs = [[(int(j), int(j) + 1) for j in np.flatnonzero(m)] for m in masks]
    return torch.as_tensor(masks, device=device), segs


def patch_to_wavelength_mask(patch_mask: torch.Tensor, P: int, D: int):
    """Mở rộng mặt nạ mức mảnh (B, N) thành mặt nạ mức bước sóng (B, D)."""
    return patch_mask.repeat_interleave(P, dim=1)[:, :D]
