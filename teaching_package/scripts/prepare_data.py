"""Bước 1-2: chuẩn bị dữ liệu và tiền xử lý Phi.

Ví dụ:
  # dữ liệu mô phỏng để chạy thử
  python scripts/prepare_data.py --synthetic --n 3000 --out runs/synth.npz
  # dữ liệu thật dạng bảng
  python scripts/prepare_data.py --spectra spectra.csv --meta meta.csv \
      --analytes Thiamethoxam Permethrin --normal-col normal --group-col sample_id \
      --ref-bands configs/ref_bands_vf_pst_nir.json --out runs/vf.npz --config
      configs/vf_pst_nir.yaml
"""

import argparse
import json

from _common import load_config, ROOT  # noqa: F401
from data.io import from_csv, save_npz
from data.preprocess import phi
from data.synthetic import make_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--D", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--spectra")
ap.add_argument("--meta")
ap.add_argument("--analytes", nargs="*", default=[])
ap.add_argument("--normal-col", default="normal")
ap.add_argument("--group-col")
ap.add_argument("--ref-bands", help="tệp json {chất: [[lo, hi], ...]}")
ap.add_argument("--config", default=f"{ROOT}/configs/default.yaml")
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = load_config(a.config)
if a.synthetic:
    ds = make_dataset(n=a.n, D=a.D, seed=a.seed)
else:
    ds = from_csv(a.spectra, a.meta, a.analytes, a.normal_col, a.group_col)
    if a.ref_bands:
        ds.ref_bands = json.load(open(a.ref_bands, encoding="utf8"))
pc = cfg["preprocess"]
ds.X, ds.wl = phi(ds.X, ds.wl, pc["n_points"], pc["sg_window"], pc["sg_poly"])
save_npz(a.out, ds)
print(
    f"đã lưu {a.out}: {ds.n} phổ x {ds.X.shape[1]} kênh, "
    f"{int(ds.normal.sum())} phổ nền, "
    f"chất: {ds.analytes}"
)
