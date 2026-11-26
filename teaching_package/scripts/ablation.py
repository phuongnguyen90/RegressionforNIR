"""Nghiên cứu loại bỏ thành phần (mục 3.6; Tables 6-7 của bài báo).

Chạy run_cv.py cho năm biến thể bằng cách ghi đè cấu hình:
  full            mô hình đầy đủ
  no_cmsm         che ngẫu nhiên rời rạc (masking.strategy=scattered)
  no_mmie         một mặt nạ khi suy luận (inference.M=1)
  no_wavelength   mã hóa vị trí sin chuẩn (model.wl_mode=sinusoidal)
  no_casrl        MSE thường (loss.vanilla=true)

  python scripts/ablation.py --data runs/synth.npz --config configs/smoke_test.yaml
  --out runs/ablation
"""

import argparse
import os
import subprocess
import sys

VARIANTS = {
    "full": [],
    "no_cmsm": ["masking.strategy=scattered"],
    "no_mmie": ["inference.M=1"],
    "no_wavelength": ["model.wl_mode=sinusoidal"],
    "no_casrl": ["loss.vanilla=true"],
}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
for name, ov in VARIANTS.items():
    if a.only and name not in a.only:
        continue
    print(f"=== {name} ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            f"{here}/run_cv.py",
            "--data",
            a.data,
            "--config",
            a.config,
            "--out",
            f"{a.out}/{name}",
            "--skip-regression",
            "--set",
            *ov,
        ],
        check=True,
    )
