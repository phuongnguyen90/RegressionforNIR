"""Gợi ý lời giải một số bài tập của ba tình huống (dành cho giảng viên).

Chạy sau khi đã chạy tình huống 2:  python teaching/bai_tap_goi_y.py   (khoảng 2 phút)
"""

# %% Chuẩn bị chung
import os
import sys

import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.preprocess import minmax, phi, snv  # noqa: E402
from data.synthetic import make_dataset  # noqa: E402
from inference.mmie import mmie  # noqa: E402
from models.specmultimae import SpecMultiMAE  # noqa: E402

OUT = os.path.join(ROOT, "teaching", "output")
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# %% Bài tập 1.1 và 1.2: vai trò của SNV và số thành phần PLS
ds1 = make_dataset(n=1200, D=400, seed=1)
X_sg = savgol_filter(ds1.X, 15, 2, axis=1)
variants = {
    "SG + SNV + min-max": minmax(snv(X_sg)),
    "SG + min-max (bỏ SNV)": minmax(X_sg),
}
ai = ds1.analytes.index("Imidacloprid")
keep = ds1.y[:, ai] > 0
y = ds1.y[keep, ai]
print("Bài tập 1.1, Imidacloprid, PLSR 10 thành phần:")
for name, Xv in variants.items():
    yhat = cross_val_predict(PLSRegression(10), Xv[keep], y, cv=kf).ravel()
    print(f"  {name:22s} R2 = {r2_score(y, yhat):.3f}")
print("Bài tập 1.2, R2 theo số thành phần PLS:")
Xp = variants["SG + SNV + min-max"][keep]
for k in [2, 5, 10, 20, 50]:
    yhat = cross_val_predict(PLSRegression(k), Xp, y, cv=kf).ravel()
    print(f"  {k:3d} thành phần: R2 = {r2_score(y, yhat):.3f}")

# %% Bài tập 2.1: ảnh hưởng của số mặt nạ M khi suy luận (dùng mô hình đã lưu ở TH2)
ds = make_dataset(n=1500, D=400, seed=0)
X, wl = phi(ds.X, ds.wl)
test = np.load(f"{OUT}/th2_test_idx.npy")
model = SpecMultiMAE(
    P=16,
    d=64,
    d_enc=64,
    enc_depth=2,
    enc_heads=4,
    d_dec=32,
    dec_depth=1,
    dec_heads=4,
    n_freq=32,
)
model.load_state_dict(torch.load(f"{OUT}/th2_model.pt"))
cfg = {
    "model": {"P": 16},
    "inference": {"alpha": 0.5},
    "masking": dict(mask_ratio=0.75, l_min=2, l_max=8, K_range=(4, 8)),
}
print("Bài tập 2.1, AUROC theo số mặt nạ M:")
for M in [1, 4, 16, 64]:
    out = mmie(model, X[test], wl, cfg, M=M)
    print(
        f"  M = {M:3d}: AUROC = {roc_auc_score(~ds.normal[test], out['score']):.3f}"
    )

# %% Bài tập 3.1: R2 theo số bước sóng giữ lại (Imidacloprid)
Pm = np.load(f"{OUT}/th2_prominence_test.npy")
Xt, Yt = X[test], ds.y[test]
ai = ds.analytes.index("Imidacloprid")
keep = Yt[:, ai] > 0
Xa, ya, Pa = Xt[keep], Yt[keep, ai], Pm[keep]
params = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=0,
)
print("Bài tập 3.1, Imidacloprid, R2 theo số bước sóng chọn:")
for n_sel in [10, 40, 100, 200, 400]:
    yhat = np.zeros_like(ya)
    for tr, te in kf.split(Xa):
        sel = np.sort(np.argsort(-Pa[tr].mean(axis=0))[:n_sel])
        yhat[te] = (
            XGBRegressor(**params).fit(Xa[tr][:, sel], ya[tr]).predict(Xa[te][:, sel])
        )
    print(f"  {n_sel:3d} bước sóng: R2 = {r2_score(ya, yhat):.3f}")
