"""Chemometric preprocessing, applied per spectrum in this order:

1. linear resampling onto a uniform wavelength grid (optional);
2. Savitzky-Golay smoothing (window 15, order 2);
3. Standard Normal Variate;
4. min-max scaling to [0, 1].

Every step is per-spectrum, so nothing is fitted and there is no train/test leakage.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .config import PreprocessConfig


def resample(X: np.ndarray, wl: np.ndarray, n_points: int | None = None,
             step_nm: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if n_points is None and step_nm is None:
        return X, wl
    if n_points is not None:
        grid = np.linspace(wl.min(), wl.max(), int(n_points))
    else:
        grid = np.arange(wl.min(), wl.max() + 1e-9, float(step_nm))
        grid = np.unique(np.clip(grid, wl.min(), wl.max()))  # float rounding can overshoot wl.max()
    f = interp1d(wl, X, axis=1, kind="linear", assume_sorted=True)
    return f(grid).astype(np.float32), grid


def savgol(X: np.ndarray, window: int, order: int) -> np.ndarray:
    D = X.shape[1]
    if window <= 0:
        return X
    window = min(window, D if D % 2 == 1 else D - 1)
    if window % 2 == 0:
        window -= 1
    if window <= order:
        return X
    return savgol_filter(X, window, order, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def minmax(X: np.ndarray) -> np.ndarray:
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return (X - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def preprocess(X: np.ndarray, wl: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    wl = np.asarray(wl, dtype=np.float64)
    order = np.argsort(wl)
    X, wl = X[:, order], wl[order]
    X, wl = resample(X, wl, cfg.resample_points, cfg.resample_step_nm)
    X = savgol(np.asarray(X, dtype=np.float64), cfg.savgol_window, cfg.savgol_order)
    if cfg.snv:
        X = snv(X)
    if cfg.minmax:
        X = minmax(X)
    return X.astype(np.float32), wl
