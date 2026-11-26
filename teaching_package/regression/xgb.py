"""Giai đoạn 2: hồi quy hàm lượng bằng XGBoost (mục 3.3.1 và 3.8 của báo cáo; Table
15 của bài báo).

Cùng một bộ hồi quy g được huấn luyện ở hai cấu hình:
  - "full": toàn bộ D kênh phổ;
  - "selected": n_select (80-120) bước sóng có điểm nổi bật cao nhất do
  SpecMultiMAE-AD trả về.
Tập bước sóng được chọn chỉ từ dữ liệu huấn luyện của từng fold để tránh rò rỉ thông
tin.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from inference.localize import select_wavelengths
from inference.metrics import regression_metrics

DEFAULT_XGB = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
)


def cv_regression(
    X,
    y,
    P_all=None,
    n_select=100,
    folds=5,
    groups=None,
    seed=0,
    xgb_params=None,
    positive_only=True,
):
    """X: (n, D) phổ đã tiền xử lý; y: (n,) hàm lượng một chất; P_all: (n, D) điểm
    nổi bật
    của SpecMultiMAE-AD cho từng phổ (None -> chỉ chạy cấu hình toàn phổ).
    positive_only=True: chỉ hồi quy trên các mẫu có dư lượng (y > 0)."""
    params = {**DEFAULT_XGB, **(xgb_params or {}), "random_state": seed}
    keep = np.isfinite(y) & ((y > 0) if positive_only else True)
    X, y = X[keep], y[keep]
    P_all = None if P_all is None else P_all[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    splitter = (
        GroupKFold(folds)
        if g is not None
        else KFold(folds, shuffle=True, random_state=seed)
    )
    res = {"full": [], "selected": []}
    yhat = {"full": np.zeros_like(y), "selected": np.zeros_like(y)}
    for tr, te in splitter.split(X, y, g):
        m = XGBRegressor(**params).fit(X[tr], y[tr])
        yhat["full"][te] = m.predict(X[te])
        if P_all is not None:
            sel = select_wavelengths(P_all[tr].mean(0), n_select)
            m2 = XGBRegressor(**params).fit(X[tr][:, sel], y[tr])
            yhat["selected"][te] = m2.predict(X[te][:, sel])
    out = {"full": regression_metrics(y, yhat["full"]), "n": int(len(y))}
    if P_all is not None:
        out["selected"] = regression_metrics(y, yhat["selected"])
        out["selected_idx_last_fold"] = sel.tolist()
    return out
