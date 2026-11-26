"""Bước 3-6: toàn bộ khung hai giai đoạn với kiểm định chéo (mục 3.3.3 của báo cáo).

Với mỗi lần chạy (seed) và mỗi fold:
  - phổ nền chia 5 fold theo nhóm (mẫu vật lý); huấn luyện trên 4 fold phổ nền;
  - tập kiểm định = 10% phổ nền giữ lại + 10% phổ bất thường: dừng sớm (chỉ phần phổ
  nền) và
    chọn ngưỡng Youden J; được loại khỏi tập kiểm tra;
  - tập kiểm tra = phần còn lại của fold phổ nền + toàn bộ phổ bất thường còn lại;
  - suy luận MMIE -> điểm bất thường, bản đồ lỗi, phần dư, điểm nổi bật;
  - đánh giá phát hiện và định vị.
Sau đó chạy hồi quy XGBoost toàn phổ và trên tập bước sóng chọn, cho từng chất.

  python scripts/run_cv.py --data runs/synth.npz --config configs/smoke_test.yaml
  --out runs/smoke
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from _common import device, dump, load_config
from data.io import load_npz
from inference.localize import detect_peaks
from inference.metrics import (
    detection_metrics,
    localization_metrics,
    youden_threshold,
)
from inference.mmie import mmie, prominence
from models.trainer import train
from regression.xgb import cv_regression

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--set", nargs="*", default=[], help="ghi đè cấu hình, vd. inference.M=1"
)
ap.add_argument("--skip-regression", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config, a.set)
ds = load_npz(a.data)
dev = device()
os.makedirs(a.out, exist_ok=True)
dump(cfg, f"{a.out}/config_used.json")
groups = ds.group if ds.group is not None else np.arange(ds.n)
norm_idx = np.flatnonzero(ds.normal)
anom_idx = np.flatnonzero(~ds.normal)
E = cfg["eval"]
results = []
P_all = np.zeros_like(
    ds.X
)  # điểm nổi bật cho mọi phổ (lấy từ fold mà phổ đó thuộc tập kiểm tra)
P_cnt = np.zeros(ds.n)

for run in range(E["runs"]):
    seed = 1000 * run
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(E["folds"])
    for fold, (tr_n, te_n) in enumerate(gkf.split(norm_idx, groups=groups[norm_idx])):
        tr = norm_idx[tr_n]
        te_norm = norm_idx[te_n]
        val_norm = rng.choice(
            te_norm, max(1, int(E["val_frac"] * len(te_norm))), replace=False
        )
        val_anom = rng.choice(
            anom_idx, max(1, int(E["val_frac"] * len(anom_idx))), replace=False
        )
        test = np.concatenate(
            [np.setdiff1d(te_norm, val_norm), np.setdiff1d(anom_idx, val_anom)]
        )
        val = np.concatenate([val_norm, val_anom])
        model, hist = train(ds.X[tr], ds.X[val_norm], ds.wl, cfg, dev, seed + fold)
        torch.save(model.state_dict(), f"{a.out}/model_run{run}_fold{fold}.pt")
        out_val = mmie(model, ds.X[val], ds.wl, cfg, device=dev, seed=seed + 7)
        thr = youden_threshold(~ds.normal[val], out_val["score"])
        out = mmie(model, ds.X[test], ds.wl, cfg, device=dev, seed=seed + 11)
        det = detection_metrics(~ds.normal[test], out["score"], thr)
        Pm = prominence(out["R"], out["E"], cfg["inference"]["alpha"])
        P_all[test] += Pm
        P_cnt[test] += 1
        # định vị: với mỗi chất, trung bình chỉ số trên các phổ bất thường chứa chất
        # đó
        loc = {}
        pk = cfg["peaks"]
        if ds.y is not None and ds.ref_bands:
            for ai, name in enumerate(ds.analytes):
                sel = [
                    k
                    for k, i in enumerate(test)
                    if not ds.normal[i] and ds.y[i, ai] > 0
                ]
                if not sel or name not in ds.ref_bands:
                    continue
                ms = []
                for k in sel:
                    peaks = detect_peaks(
                        Pm[k],
                        ds.wl,
                        pk["min_sep_nm"],
                        pk["min_height"],
                        pk["min_prominence"],
                        top_k=pk["top_k"],
                    )
                    ms.append(
                        localization_metrics(
                            [p for p, _ in peaks], ds.ref_bands[name], pk["tol_nm"]
                        )
                    )
                loc[name] = {m: float(np.nanmean([x[m] for x in ms])) for m in ms[0]}
        rec = {
            "run": run,
            "fold": fold,
            "threshold": thr,
            "detection": det,
            "localization": loc,
            "epochs_trained": len(hist),
        }
        results.append(rec)
        print(
            json.dumps(
                {"run": run, "fold": fold, **{k: round(v, 4) for k, v in det.items()}}
            )
        )


def agg(key):
    vals = [r["detection"][key] for r in results]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


summary = {"detection": {k: agg(k) for k in results[0]["detection"]}}
loc_names = sorted({n for r in results for n in r["localization"]})
summary["localization"] = {
    n: {
        m: float(
            np.nanmean(
                [r["localization"][n][m] for r in results if n in r["localization"]]
            )
        )
        for m in ["IoU", "PeakF1", "MeanPeakError", "Top3Hit"]
    }
    for n in loc_names
}

P_all = P_all / np.maximum(P_cnt, 1)[:, None]
np.save(f"{a.out}/prominence.npy", P_all)
if not a.skip_regression and ds.y is not None:
    rc = cfg["regression"]
    summary["regression"] = {}
    for ai, name in enumerate(ds.analytes):
        if name in rc.get("exclude", []):
            continue
        y = ds.y[:, ai].astype(float)
        if (y > 0).sum() < 2 * E["folds"]:
            continue
        r = cv_regression(
            ds.X,
            y,
            P_all,
            rc["n_select"],
            E["folds"],
            groups,
            positive_only=rc["positive_only"],
        )
        summary["regression"][name] = r
        print(
            f"{name:16s} toàn phổ R2={r['full']['R2']:.3f} "
            f"RMSE={r['full']['RMSE']:.3f} | "
            f"bước sóng chọn R2={r['selected']['R2']:.3f} "
            f"RMSE={r['selected']['RMSE']:.3f}"
        )
dump({"folds": results, "summary": summary}, f"{a.out}/results.json")
print(json.dumps(summary["detection"], indent=1))
