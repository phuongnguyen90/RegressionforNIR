"""Một số phương pháp đối chứng có thể cài đặt lại trực tiếp (Isolation Forest, bộ
tự mã hóa tái
tạo AE, Deep SVDD cho phát hiện; PLS-VIP cho định vị), dùng cùng giao thức chia fold
với run_cv.py.

Các đối chứng còn lại trong báo cáo (MAE-NIR, CSSFA, Anomaly Transformer, NIRsViT,
SMART-NIR, PLS-DA, SVM, Balanced Random Forest, mRMR+GA, SHAP trên MAE-NIR) được
chạy bằng mã nguồn công bố của tác giả gốc hoặc thư viện chuẩn và không đóng gói lại
ở đây.

  python scripts/baselines.py --data runs/synth.npz --config configs/smoke_test.yaml
  --out runs/baselines
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from _common import device, dump, load_config
from data.io import load_npz
from inference.localize import detect_peaks
from inference.metrics import (
    detection_metrics,
    localization_metrics,
    youden_threshold,
)


def mlp(D, h=(256, 64)):
    enc = nn.Sequential(nn.Linear(D, h[0]), nn.GELU(), nn.Linear(h[0], h[1]))
    dec = nn.Sequential(nn.Linear(h[1], h[0]), nn.GELU(), nn.Linear(h[0], D))
    return enc, dec


def fit_ae(X, epochs=100, dev="cpu", seed=0):
    torch.manual_seed(seed)
    enc, dec = mlp(X.shape[1])
    m = nn.Sequential(enc, dec).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    Xt = torch.as_tensor(X, device=dev)
    for _ in range(epochs):
        for i in torch.randperm(len(Xt)).split(128):
            loss = ((m(Xt[i]) - Xt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return (
        lambda Z: (
            (m(torch.as_tensor(Z, device=dev)) - torch.as_tensor(Z, device=dev)) ** 2
        )
        .mean(1)
        .detach()
        .cpu()
        .numpy()
    )


def fit_svdd(X, epochs=100, dev="cpu", seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(X.shape[1], 256, bias=False),
        nn.GELU(),
        nn.Linear(256, 32, bias=False),
    ).to(
        dev
    )  # không hệ số tự do (Ruff 2018)
    Xt = torch.as_tensor(X, device=dev)
    with torch.no_grad():
        c = net(Xt).mean(0)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for i in torch.randperm(len(Xt)).split(128):
            loss = ((net(Xt[i]) - c) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return (
        lambda Z: ((net(torch.as_tensor(Z, device=dev)) - c) ** 2)
        .sum(1)
        .detach()
        .cpu()
        .numpy()
    )


def vip_scores(pls: PLSRegression):
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = w.shape[0]
    ss = np.diag(t.T @ t @ q.T @ q).reshape(-1)
    wn = (w / np.linalg.norm(w, axis=0)) ** 2
    return np.sqrt(p * (wn @ ss) / ss.sum())


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=100)
a = ap.parse_args()
cfg = load_config(a.config)
ds = load_npz(a.data)
dev = device()
groups = ds.group if ds.group is not None else np.arange(ds.n)
norm_idx, anom_idx = np.flatnonzero(ds.normal), np.flatnonzero(~ds.normal)
E = cfg["eval"]
res = {"IsolationForest": [], "AE": [], "DeepSVDD": []}
rng = np.random.default_rng(0)
for fold, (tr_n, te_n) in enumerate(
    GroupKFold(E["folds"]).split(norm_idx, groups=groups[norm_idx])
):
    tr, te_norm = norm_idx[tr_n], norm_idx[te_n]
    vn = rng.choice(te_norm, max(1, int(E["val_frac"] * len(te_norm))), replace=False)
    va = rng.choice(
        anom_idx, max(1, int(E["val_frac"] * len(anom_idx))), replace=False
    )
    val = np.concatenate([vn, va])
    test = np.concatenate([np.setdiff1d(te_norm, vn), np.setdiff1d(anom_idx, va)])
    scorers = {
        "IsolationForest": (lambda f: (lambda Z: -f.score_samples(Z)))(
            IsolationForest(n_estimators=300, random_state=fold).fit(ds.X[tr])
        ),
        "AE": fit_ae(ds.X[tr], a.epochs, dev, fold),
        "DeepSVDD": fit_svdd(ds.X[tr], a.epochs, dev, fold),
    }
    for name, f in scorers.items():
        thr = youden_threshold(~ds.normal[val], f(ds.X[val]))
        res[name].append(detection_metrics(~ds.normal[test], f(ds.X[test]), thr))
summary = {
    n: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for n, v in res.items()
}

# PLS-VIP cho định vị: khớp PLS giữa phổ và hàm lượng từng chất, đỉnh trên đường VIP
# chuẩn hóa
pk = cfg["peaks"]
loc = {}
if ds.y is not None:
    for ai, name in enumerate(ds.analytes):
        y = ds.y[:, ai]
        if (y > 0).sum() < 10 or name not in ds.ref_bands:
            continue
        pls = PLSRegression(n_components=min(10, ds.X.shape[1] - 1)).fit(ds.X, y)
        v = vip_scores(pls)
        v = v / v.max()
        peaks = detect_peaks(
            v,
            ds.wl,
            pk["min_sep_nm"],
            pk["min_height"],
            pk["min_prominence"],
            top_k=pk["top_k"],
        )
        loc[name] = localization_metrics(
            [p for p, _ in peaks], ds.ref_bands[name], pk["tol_nm"]
        )
summary["PLS-VIP_localization"] = loc
dump(summary, f"{a.out}/baselines.json")
for n, v in summary.items():
    print(n, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
