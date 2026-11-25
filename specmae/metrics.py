"""Evaluation metrics.

Detection: AUROC, AUPRC, and Precision/Recall/F1 at a fixed threshold.

Localisation, per anomalous test spectrum with ground-truth bands B (nm intervals):
* IoU            -- interval IoU between the top-1 peak's half-height interval and
                    the best-matching ground-truth band;
* Peak F1        -- a peak is a hit if it lies inside some band widened by +/- tol;
                    precision = hit peaks / peaks, recall = bands hit / bands;
* Mean peak error-- distance (nm) from the top-1 peak to the nearest band
                    (0 if inside);
* Top-3 hit rate -- 1 if any of the top-3 peaks hits a band (+/- tol).
All are averaged over spectra.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .downstream import Peak

Band = Tuple[float, float]


def detection_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pred = (scores > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": float(threshold)}
    if labels.min() != labels.max():
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    return out


def interval_iou(a: Band, b: Band) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _hits(wl: float, band: Band, tol: float) -> bool:
    return band[0] - tol <= wl <= band[1] + tol


def _dist(wl: float, band: Band) -> float:
    return max(0.0, band[0] - wl, wl - band[1])


def localization_metrics(peaks_per_sample: Sequence[List[Peak]],
                         bands_per_sample: Sequence[Optional[List[Band]]],
                         tolerance_nm: float) -> Dict[str, float]:
    ious, f1s, errs, top3 = [], [], [], []
    for peaks, bands in zip(peaks_per_sample, bands_per_sample):
        if not bands or not peaks:
            continue
        top = peaks[0]
        ious.append(max(interval_iou((top.left_nm, top.right_nm), b) for b in bands))
        errs.append(min(_dist(top.wavelength, b) for b in bands))
        hit_peaks = sum(any(_hits(p.wavelength, b, tolerance_nm) for b in bands) for p in peaks)
        hit_bands = sum(any(_hits(p.wavelength, b, tolerance_nm) for p in peaks) for b in bands)
        prec = hit_peaks / len(peaks)
        rec = hit_bands / len(bands)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        top3.append(float(any(_hits(p.wavelength, b, tolerance_nm) for p in peaks[:3] for b in bands)))
    if not ious:
        return {"n_localized": 0}
    return {
        "iou": float(np.mean(ious)),
        "peak_f1": float(np.mean(f1s)),
        "mean_peak_error_nm": float(np.mean(errs)),
        "top3_hit_rate": float(np.mean(top3)),
        "n_localized": len(ious),
    }


def aggregate(results: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and std of every numeric key across folds/runs."""
    results = list(results)
    keys = sorted({k for r in results for k, v in r.items() if isinstance(v, (int, float))})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in results if k in r], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n": int(vals.size)}
    return out
