"""Dò đỉnh trên điểm nổi bật P(l) để định vị bước sóng hấp thụ (Section 4.5.2).

Ràng buộc: chiều cao tối thiểu, độ nổi bật tối thiểu, khoảng cách tối thiểu giữa hai
đỉnh tính theo nm (FDS-NIR, VF-PST-NIR: 10 nm, 0,10, 0,08; MSMI-NIR: 100 nm, 0,12,
0,10), sau đó xếp hạng theo P và triệt tiêu không cực đại (NMS) trong cùng một dải.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_peaks(
    Pm: np.ndarray,
    wl: np.ndarray,
    min_sep_nm: float = 10.0,
    min_height: float = 0.10,
    min_prominence: float = 0.08,
    nms_nm: float | None = None,
    top_k: int | None = None,
):
    """Pm: (D,) điểm nổi bật đã chuẩn hóa; wl: (D,) nm.
    Trả về danh sách (bước_sóng_nm, độ_tin_cậy) theo thứ tự giảm dần."""
    step = float(np.median(np.diff(wl)))
    dist = max(1, int(round(min_sep_nm / step)))
    idx, _ = find_peaks(
        Pm, height=min_height, prominence=min_prominence, distance=dist
    )
    order = idx[np.argsort(-Pm[idx])]
    nms_nm = nms_nm or min_sep_nm
    keep = []
    for j in order:
        if all(abs(wl[j] - wl[k]) > nms_nm for k in keep):
            keep.append(j)
    if top_k:
        keep = keep[:top_k]
    conf = Pm[keep] / max(Pm[keep].max(), 1e-12) if keep else np.array([])
    return [(float(wl[j]), float(c)) for j, c in zip(keep, conf)]


def select_wavelengths(P_mean: np.ndarray, n_select: int = 100):
    """Chọn n_select chỉ số bước sóng có điểm nổi bật trung bình cao nhất (giai đoạn
    2, hồi quy).
    P_mean là trung bình P(l) trên các phổ có chứa chất cần định lượng thuộc tập
    huấn luyện.
    """
    return np.sort(np.argsort(-P_mean)[:n_select])
