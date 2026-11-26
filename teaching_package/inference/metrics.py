"""Độ đo đánh giá (Section 5.5, mục 3.3.4 của báo cáo).

Phát hiện: AUROC, AUPRC; Precision, Recall, F1 tại ngưỡng Youden J chọn trên tập
kiểm định. Định vị (so với dải hấp thụ tham chiếu lấy từ tài liệu, dung sai +/- tol
nm):
  IoU         giao/hợp giữa vùng dự đoán (hợp các [l* - tol, l* + tol]) và hợp các
  dải tham chiếu Peak F1     đỉnh dự đoán đúng nếu nằm trong một dải tham chiếu nới
  rộng tol Mean peak error  khoảng cách (nm) từ tâm mỗi dải tham chiếu tới đỉnh dự
  đoán gần nhất Top-3 hit   tỉ lệ dải tham chiếu có ít nhất một trong ba đỉnh đầu
  rơi vào
Cách tính cụ thể của IoU và Peak F1 được định nghĩa trong tệp này. Hồi quy: R^2, RMSE.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def youden_threshold(y_true, score):
    fpr, tpr, thr = roc_curve(y_true, score)
    return float(thr[np.argmax(tpr - fpr)])


def detection_metrics(y_true, score, threshold):
    pred = (score > threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, score),
        "AUPRC": average_precision_score(y_true, score),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def _union_len(intervals):
    iv = sorted(intervals)
    tot, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


def localization_metrics(peaks_nm, ref_bands, tol=15.0):
    """peaks_nm: danh sách bước sóng đỉnh theo thứ tự tin cậy; ref_bands: [[lo, hi],
    ...]."""
    if not ref_bands:
        return {}
    if not peaks_nm:
        return {
            "IoU": 0.0,
            "PeakF1": 0.0,
            "MeanPeakError": float("nan"),
            "Top3Hit": 0.0,
        }
    pred = [(p - tol, p + tol) for p in peaks_nm]
    inter = 0.0
    for a, b in pred:
        for lo, hi in ref_bands:
            inter += max(0.0, min(b, hi) - max(a, lo))
    inter = min(inter, _union_len(pred))
    union = _union_len(pred + [tuple(r) for r in ref_bands])
    in_band = [
        any(lo - tol <= p <= hi + tol for lo, hi in ref_bands) for p in peaks_nm
    ]
    prec = np.mean(in_band)
    hit = [any(lo - tol <= p <= hi + tol for p in peaks_nm) for lo, hi in ref_bands]
    rec = np.mean(hit)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    err = np.mean(
        [min(abs(p - (lo + hi) / 2) for p in peaks_nm) for lo, hi in ref_bands]
    )
    top3 = np.mean(
        [any(lo - tol <= p <= hi + tol for p in peaks_nm[:3]) for lo, hi in ref_bands]
    )
    return {
        "IoU": inter / union if union else 0.0,
        "PeakF1": f1,
        "MeanPeakError": float(err),
        "Top3Hit": float(top3),
    }


def regression_metrics(y, yhat):
    return {
        "R2": r2_score(y, yhat),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
    }
