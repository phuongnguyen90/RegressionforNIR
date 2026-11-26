"""Đọc, ghi dữ liệu phổ ở định dạng thống nhất của gói mã nguồn.

Một bộ dữ liệu được lưu thành một tệp .npz với các khóa:
  X          (n, D)  ma trận phổ thô
  wl         (D,)    véc-tơ bước sóng (nm)
  normal     (n,)    1 = phổ nền (không chứa chất cần định lượng), 0 = phổ bất thường
  y          (n, A)  hàm lượng của A chất (NaN nếu không đo); có thể vắng mặt
  analytes   (A,)    tên các chất
  group      (n,)    nhãn nhóm dùng khi chia fold (mẫu vật lý / thiết bị); có thể
  vắng mặt ref_bands  dict    {tên chất: [[lo_nm, hi_nm], ...]} dải hấp thụ tham
  chiếu (json)

Hàm from_csv giúp chuyển dữ liệu dạng bảng (mỗi cột là một bước sóng) sang .npz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpectralDataset:
    X: np.ndarray
    wl: np.ndarray
    normal: np.ndarray
    y: np.ndarray | None = None
    analytes: list[str] = field(default_factory=list)
    group: np.ndarray | None = None
    ref_bands: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.X.shape[0]

    def subset(self, idx):
        return SpectralDataset(
            self.X[idx],
            self.wl,
            self.normal[idx],
            None if self.y is None else self.y[idx],
            self.analytes,
            None if self.group is None else self.group[idx],
            self.ref_bands,
        )


def save_npz(path: str, ds: SpectralDataset):
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        X=ds.X,
        wl=ds.wl,
        normal=ds.normal.astype(np.int8),
        y=np.array([]) if ds.y is None else ds.y,
        analytes=np.array(ds.analytes, dtype=object),
        group=np.array([]) if ds.group is None else ds.group,
        ref_bands=json.dumps(ds.ref_bands),
    )


def load_npz(path: str) -> SpectralDataset:
    z = np.load(path, allow_pickle=True)
    y = z["y"] if z["y"].size else None
    group = z["group"] if z["group"].size else None
    return SpectralDataset(
        X=z["X"].astype(np.float32),
        wl=z["wl"].astype(float),
        normal=z["normal"].astype(bool),
        y=y,
        analytes=[str(a) for a in z["analytes"]],
        group=group,
        ref_bands=json.loads(str(z["ref_bands"])),
    )


def from_csv(
    spectra_csv: str,
    meta_csv: str,
    analyte_cols: list[str],
    normal_col: str = "normal",
    group_col: str | None = None,
) -> SpectralDataset:
    """spectra_csv: hàng = mẫu, cột = bước sóng (tiêu đề cột là số nm).
    meta_csv: cùng thứ tự hàng, chứa cột phổ nền và các cột hàm lượng."""
    import pandas as pd

    S = pd.read_csv(spectra_csv)
    M = pd.read_csv(meta_csv)
    wl = S.columns.astype(float).to_numpy()
    y = M[analyte_cols].to_numpy(dtype=float) if analyte_cols else None
    group = M[group_col].to_numpy() if group_col else None
    return SpectralDataset(
        S.to_numpy(np.float32),
        wl,
        M[normal_col].to_numpy().astype(bool),
        y,
        list(analyte_cols),
        group,
    )
