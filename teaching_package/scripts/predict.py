"""Dùng mô hình đã huấn luyện cho một tệp phổ mới: điểm bất thường, đỉnh hấp thụ, và
(tùy chọn)
dự đoán hàm lượng bằng bộ hồi quy XGBoost đã lưu.

  python scripts/predict.py --data runs/new.npz --config configs/vf_pst_nir.yaml \
      --model runs/vf/model_run0_fold0.pt --threshold 0.0123 --out runs/pred.json
"""

import argparse

import numpy as np
import torch

from _common import device, dump, load_config
from data.io import load_npz
from inference.localize import detect_peaks
from inference.mmie import mmie, prominence
from models.specmultimae import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--threshold", type=float)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = load_config(a.config)
ds = load_npz(a.data)
dev = device()
model = build_model(cfg).to(dev)
model.load_state_dict(torch.load(a.model, map_location=dev))
out = mmie(model, ds.X, ds.wl, cfg, device=dev)
Pm = prominence(out["R"], out["E"], cfg["inference"]["alpha"])
pk = cfg["peaks"]
rows = []
for i in range(ds.n):
    peaks = detect_peaks(
        Pm[i],
        ds.wl,
        pk["min_sep_nm"],
        pk["min_height"],
        pk["min_prominence"],
        top_k=pk["top_k"],
    )
    rows.append(
        {
            "index": i,
            "score": float(out["score"][i]),
            "anomalous": (
                None if a.threshold is None else bool(out["score"][i] > a.threshold)
            ),
            "peaks_nm": peaks,
        }
    )
dump(rows, a.out)
np.save(a.out.replace(".json", "_prominence.npy"), Pm)
print(f"đã ghi {a.out} ({ds.n} phổ)")
