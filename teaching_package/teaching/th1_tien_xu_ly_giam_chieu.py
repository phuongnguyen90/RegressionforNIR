"""Tình huống học tập 1: Tiền xử lý và giảm chiều dữ liệu phổ NIR.

Chạy toàn bộ:  python teaching/th1_tien_xu_ly_giam_chieu.py
Hoặc chạy từng khối "# %%" trong VS Code, Spyder, Jupyter.
Kết quả: hình trong teaching/output/, số liệu in ra màn hình.
"""

# %% Bước 1. Nạp thư viện và sinh dữ liệu
import os
import sys

import matplotlib

matplotlib.use("Agg")  # bỏ dòng này nếu muốn hiện cửa sổ hình
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.synthetic import make_dataset  # noqa: E402

OUT = os.path.join(ROOT, "teaching", "output")
os.makedirs(OUT, exist_ok=True)

ds = make_dataset(n=1200, D=400, seed=1)  # 1.200 phổ, 400 bước sóng, 5 hoạt chất
X, wl, y_all = ds.X, ds.wl, ds.y
print("Ma trận phổ X:", X.shape, " véc-tơ bước sóng:", wl[0], "->", wl[-1], "nm")
print("Hoạt chất:", ds.analytes)
print("Số phổ nền (không có hoạt chất):", int(ds.normal.sum()))
print("Số mẫu có dư lượng theo hoạt chất:", (y_all > 0).sum(axis=0))

# %% Bước 2. Quan sát dữ liệu: phổ thô và tương quan giữa các bước sóng
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(wl, X[:40].T, lw=0.6)
ax[0].set_xlabel("Bước sóng (nm)")
ax[0].set_ylabel("Độ hấp thụ")
ax[0].set_title("40 phổ thô")
C = np.corrcoef(X.T)  # ma trận tương quan 400 x 400 giữa các bước sóng
im = ax[1].imshow(
    C, extent=[wl[0], wl[-1], wl[-1], wl[0]], cmap="RdBu_r", vmin=-1, vmax=1
)
ax[1].set_title("Tương quan giữa các bước sóng")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
plt.savefig(f"{OUT}/th1_du_lieu.png", dpi=150)
plt.close()
r_adj = np.mean(np.diag(C, 1))  # tương quan giữa bước sóng j và j+1
print(f"Tương quan trung bình giữa hai bước sóng kề nhau: {r_adj:.3f}")


# %% Bước 3. Tự cài đặt các phép tiền xử lý
def snv(X):
    """Standard Normal Variate: chuẩn hóa từng phổ theo trung bình, độ lệch chuẩn."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    return (X - mu) / sd


def minmax(X):
    """Đưa từng phổ về đoạn [0, 1]."""
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    return (X - lo) / (hi - lo)


X_sg = savgol_filter(X, window_length=15, polyorder=2, axis=1)  # làm trơn
X_snv = snv(X_sg)
Xp = minmax(X_snv)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for a, Z, t in zip(
    ax, [X, X_snv, Xp], ["Phổ thô", "Savitzky-Golay + SNV", "Min-max"]
):
    a.plot(wl, Z[:30].T, lw=0.6)
    a.set_title(t)
    a.set_xlabel("Bước sóng (nm)")
plt.tight_layout()
plt.savefig(f"{OUT}/th1_tien_xu_ly.png", dpi=150)
plt.close()

# %% Bước 4. PCA: dữ liệu 400 chiều thực chất có bao nhiêu chiều thông tin?
pca = PCA().fit(Xp)
cum = np.cumsum(pca.explained_variance_ratio_)
k99 = int(np.searchsorted(cum, 0.99) + 1)
print(f"Số thành phần chính giải thích 99% phương sai: {k99} / {Xp.shape[1]}")
T = pca.transform(Xp)[:, :2]  # điểm số trên hai thành phần chính đầu tiên
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(np.arange(1, 21), cum[:20], "o-")
ax[0].set_xlabel("Số thành phần chính")
ax[0].set_ylabel("Phương sai tích lũy")
ax[0].grid(alpha=0.3)
ax[1].scatter(*T[~ds.normal].T, s=6, alpha=0.5, label="Có hoạt chất")
ax[1].scatter(*T[ds.normal].T, s=6, alpha=0.5, label="Phổ nền")
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/th1_pca.png", dpi=150)
plt.close()

# %% Bước 5. PLSR làm mô hình cơ sở, kiểm định chéo 5 phần viết tường minh
print("\nPLSR 10 thành phần, kiểm định chéo 5 phần:")
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for ai, name in enumerate(ds.analytes):
    keep = y_all[:, ai] > 0  # chỉ hồi quy trên các mẫu có dư lượng
    Xa, ya = Xp[keep], y_all[keep, ai]
    yhat = np.zeros_like(ya)
    for tr, te in kf.split(Xa):
        pls = PLSRegression(n_components=10).fit(Xa[tr], ya[tr])
        yhat[te] = pls.predict(Xa[te]).ravel()
    r2 = r2_score(ya, yhat)
    rmse = np.sqrt(mean_squared_error(ya, yhat))
    print(f"  {name:14s} n={len(ya):4d}  R2={r2:.3f}  RMSE={rmse:.3f}")
    if name == "Imidacloprid":
        plt.figure(figsize=(4, 4))
        plt.scatter(ya, yhat, s=8)
        lim = [0, ya.max()]
        plt.plot(lim, lim, "k--", lw=1)
        plt.xlabel("Hàm lượng đo")
        plt.ylabel("Hàm lượng dự đoán")
        plt.title(f"PLSR, {name}: R² = {r2:.3f}")
        plt.tight_layout()
        plt.savefig(f"{OUT}/th1_plsr.png", dpi=150)
        plt.close()
print(f"\nHình đã lưu trong {OUT}")
