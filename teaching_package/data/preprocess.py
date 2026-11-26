"""Toán tử tiền xử lý Phi (mục 3.2 của báo cáo; Section 5.4 của bài báo).

Bốn bước áp dụng giống nhau cho mọi bộ dữ liệu:
  1. Lấy mẫu lại về lưới bước sóng đều bằng nội suy tuyến tính.
  2. Làm trơn Savitzky-Golay (cửa sổ 15, bậc 2).
  3. SNV (Standard Normal Variate) cho từng phổ.
  4. Chuẩn hóa min-max từng phổ về [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def resample_uniform(X: np.ndarray, wl: np.ndarray, n_points: int | None = None):
    """Nội suy tuyến tính mỗi phổ lên lưới bước sóng đều.

    X: (n, D) ma trận phổ; wl: (D,) bước sóng tăng dần (nm).
    n_points: số điểm của lưới mới (mặc định giữ nguyên D).
    """
    wl = np.asarray(wl, dtype=float)
    order = np.argsort(wl)
    wl, X = wl[order], X[:, order]
    n_points = n_points or len(wl)
    grid = np.linspace(wl[0], wl[-1], n_points)
    Xr = np.vstack([np.interp(grid, wl, x) for x in X])
    return Xr, grid


def savgol(X: np.ndarray, window: int = 15, poly: int = 2):
    if X.shape[1] < window:  # phổ quá ngắn (vd. MSMI-NIR 8 băng trước nội suy)
        return X.copy()
    return savgol_filter(X, window_length=window, polyorder=poly, axis=1)


def snv(X: np.ndarray):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def minmax(X: np.ndarray):
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return (X - lo) / np.where(hi - lo == 0, 1.0, hi - lo)


def phi(
    X: np.ndarray,
    wl: np.ndarray,
    n_points: int | None = None,
    sg_window: int = 15,
    sg_poly: int = 2,
):
    """Toán tử Phi đầy đủ. Trả về (X_đã_xử_lý, lưới_bước_sóng)."""
    Xr, grid = resample_uniform(X, wl, n_points)
    Xs = savgol(Xr, sg_window, sg_poly)
    return minmax(snv(Xs)).astype(np.float32), grid
