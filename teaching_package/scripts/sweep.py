"""Khảo sát siêu tham số (mục 3.7; Table 8 của bài báo): thay đổi từng siêu tham số
một,
giữ nguyên các siêu tham số khác ở giá trị mặc định.

  python scripts/sweep.py --data runs/vf.npz --config configs/vf_pst_nir.yaml --out
  runs/sweep
"""

import argparse
import os
import subprocess
import sys

GRID = {
    "model.P": [16, 32, 64],
    "masking.mask_ratio": [0.60, 0.75, 0.85],
    "masking.K_range": ["[2,2]", "[4,8]", "[10,10]"],
    "inference.M": [1, 32, 64, 128],
    "model.n_freq": [128, 256, 512],
    "loss.lambda": [0.1, 0.5, 1.0],
    "inference.alpha": [0.3, 0.5, 0.7],
    "model.wl_mode": ["linear", "fourier"],
}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--params", nargs="*")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
for key, vals in GRID.items():
    if a.params and key not in a.params:
        continue
    for v in vals:
        tag = f"{key}={v}".replace("[", "").replace("]", "").replace(",", "-")
        print(f"=== {tag} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                f"{here}/run_cv.py",
                "--data",
                a.data,
                "--config",
                a.config,
                "--out",
                f"{a.out}/{tag}",
                "--skip-regression",
                "--set",
                f"{key}={v}",
            ],
            check=True,
        )
