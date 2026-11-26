"""Sinh bộ dữ liệu phổ NIR mô phỏng dùng để chạy thử gói mã nguồn và cho các tình
huống học tập.

Bộ dữ liệu VF-PST-NIR không công khai (dữ liệu của dự án 24/HĐ-SKHCN). Để sinh viên
và người dùng có thể chạy toàn bộ quy trình, mô-đun này tạo phổ có cấu trúc tương
tự: nền gồm các dải hấp thụ rộng của nước (~1450, ~1940 nm) và carbohydrate thay đổi
theo "loại rau quả", cộng thêm dải hấp thụ hẹp của từng "hoạt chất" với cường độ tỉ
lệ với hàm lượng, kèm nhiễu và hiệu ứng tán xạ (nhân, cộng). Đây là dữ liệu mô
phỏng, KHÔNG dùng để báo cáo kết quả khoa học.
"""

from __future__ import annotations

import numpy as np

from .io import SpectralDataset

# Dải hấp thụ tham chiếu (nm) của các hoạt chất mô phỏng, đặt theo vùng nhóm chức ở
# Bảng 3.8.
ANALYTE_BANDS = {
    "Thiamethoxam": (1400, 1500),
    "Permethrin": (1720, 1750),
    "Azoxystrobin": (1360, 1480),
    "Imidacloprid": (1480, 1600),
    "Cypermethrin": (1680, 1750),
}


def _gauss(wl, c, w):
    return np.exp(-0.5 * ((wl - c) / w) ** 2)


def make_dataset(
    n: int = 3000,
    D: int = 512,
    wl_range=(900.0, 2500.0),
    n_types: int = 4,
    analytes: list[str] | None = None,
    frac_normal: float = 0.4,
    noise: float = 0.004,
    amp: float = 0.12,
    seed: int = 0,
) -> SpectralDataset:
    rng = np.random.default_rng(seed)
    analytes = analytes or list(ANALYTE_BANDS)
    wl = np.linspace(*wl_range, D)
    # nền theo loại rau quả
    type_params = [
        dict(
            water=rng.uniform(0.8, 1.2),
            carb=rng.uniform(0.2, 0.6),
            slope=rng.uniform(-0.2, 0.2),
        )
        for _ in range(n_types)
    ]
    types = rng.integers(0, n_types, n)
    X = np.empty((n, D), np.float32)
    y = np.zeros((n, len(analytes)))
    normal = rng.random(n) < frac_normal
    for i in range(n):
        p = type_params[types[i]]
        base = (
            p["water"] * (0.6 * _gauss(wl, 1450, 60) + 1.0 * _gauss(wl, 1940, 80))
            + p["carb"] * (_gauss(wl, 2100, 70) + 0.4 * _gauss(wl, 1200, 50))
            + p["slope"] * (wl - wl.mean()) / np.ptp(wl)
        )
        base *= rng.normal(1.0, 0.03)
        if not normal[i]:
            k = rng.integers(1, 3)  # mỗi mẫu bất thường chứa 1-2 hoạt chất
            for a in rng.choice(len(analytes), k, replace=False):
                conc = rng.gamma(2.0, 0.5)
                lo, hi = ANALYTE_BANDS.get(analytes[a], (1500, 1600))
                c, w = (lo + hi) / 2, (hi - lo) / 4
                base += amp * conc * _gauss(wl, c, w)
                y[i, a] = conc
        X[i] = (
            base + rng.normal(0, 0.02) + rng.normal(0, noise, D)
        )  # dịch nền cộng + nhiễu đầu dò
    ref = {a: [list(ANALYTE_BANDS.get(a, (1500, 1600)))] for a in analytes}
    return SpectralDataset(
        X, wl, normal, y, analytes, group=np.arange(n), ref_bands=ref
    )
