"""Downstream tasks.

* anomaly score  A = mean_l E(l)
* threshold      Youden's J on a labelled validation split, or mu + k*sigma
                 on validation normals
* localisation   P(l) = alpha*|R(l)| + (1-alpha)*E(l), max-normalised, then peak
                 detection (height / prominence / min separation in nm), ranking by
                 score and interval non-maximum suppression.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import roc_curve

from .config import LocalizationConfig


def anomaly_score(error_map: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """A = mean_l E(l) by default; ``reduce="max"`` uses the largest error instead."""
    e = np.asarray(error_map)
    if reduce == "mean":
        return e.mean(axis=-1)
    if reduce == "max":
        return e.max(axis=-1)
    raise ValueError(f"unknown score reduction: {reduce}")


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        raise ValueError("Youden's J needs both normal and anomalous validation samples")
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thr = roc_curve(labels, scores)
    t = thr[int(np.argmax(tpr - fpr))]
    if not np.isfinite(t):  # sklearn's first threshold is +inf
        return float(np.max(scores))
    # roc_curve labels "score >= t" positive; the decision rule is "score > tau",
    # so place tau midway between t and the next lower observed score.
    lower = scores[scores < t]
    return float((t + lower.max()) / 2) if lower.size else float(np.nextafter(t, -np.inf))


def sigma_threshold(normal_scores: np.ndarray, k: float = 3.0) -> float:
    s = np.asarray(normal_scores, dtype=np.float64)
    return float(s.mean() + k * s.std())


@dataclass
class Peak:
    index: int
    wavelength: float
    confidence: float   # max-normalised prominence score at the peak
    prominence: float   # topographic prominence on the normalised score
    left_nm: float      # half-height interval of the peak
    right_nm: float

    def to_dict(self) -> dict:
        return asdict(self)


def prominence_score(residual: np.ndarray, error: np.ndarray, alpha: float,
                     component_norm: str = "none") -> np.ndarray:
    a = np.abs(residual)
    e = np.asarray(error)
    if component_norm == "max":
        a = a / (a.max(axis=-1, keepdims=True) + 1e-12)
        e = e / (e.max(axis=-1, keepdims=True) + 1e-12)
    elif component_norm != "none":
        raise ValueError(f"unknown component_norm: {component_norm}")
    return alpha * a + (1.0 - alpha) * e


def localize(residual: np.ndarray, error: np.ndarray, wavelengths: np.ndarray,
             cfg: LocalizationConfig, top_k: Optional[int] = None) -> List[Peak]:
    """Ranked candidate anomalous wavelengths for one spectrum."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    p = prominence_score(residual, error, cfg.alpha, cfg.component_norm).astype(np.float64)
    pn = p / (p.max() + 1e-12)
    spacing = float(np.median(np.abs(np.diff(wl)))) if wl.size > 1 else 1.0
    distance = max(1, int(round(cfg.min_separation_nm / max(spacing, 1e-12))))

    peaks, props = find_peaks(pn, height=cfg.min_height, prominence=cfg.min_prominence, distance=distance)
    if peaks.size == 0:  # always return at least the global maximum
        peaks, props = find_peaks(np.concatenate([[-np.inf], pn, [-np.inf]]))
        peaks = peaks - 1
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(pn))])
        peaks = peaks[np.argsort(-pn[peaks])][:1]
        prom = np.array([pn[peaks[0]] - pn.min()])
    else:
        prom = props["prominences"]

    _, _, left_ips, right_ips = peak_widths(pn, peaks, rel_height=0.5)
    idx = np.arange(wl.size, dtype=np.float64)
    left_nm = np.interp(left_ips, idx, wl)
    right_nm = np.interp(right_ips, idx, wl)

    order = np.argsort(-pn[peaks])
    kept: List[Peak] = []
    for o in order:  # non-maximum suppression on the half-height intervals
        lo, hi = sorted((float(left_nm[o]), float(right_nm[o])))
        if any(lo <= k.right_nm and k.left_nm <= hi for k in kept):
            continue
        kept.append(Peak(int(peaks[o]), float(wl[peaks[o]]), float(pn[peaks[o]]), float(prom[o]), lo, hi))
    return kept[: (top_k or cfg.top_k)]


def localize_batch(residual: np.ndarray, error: np.ndarray, wavelengths: np.ndarray,
                   cfg: LocalizationConfig) -> List[List[Peak]]:
    return [localize(r, e, wavelengths, cfg) for r, e in zip(residual, error)]
