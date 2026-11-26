"""Tình huống học tập 3: Hồi quy hàm lượng và đánh giá mô hình.

Cần chạy tình huống 2 trước (dùng điểm nổi bật đã lưu trong teaching/output/).
Chạy toàn bộ:  python teaching/th3_hoi_quy_danh_gia.py   (khoảng 1 phút trên CPU)
"""

# %% Bước 1. Nạp thư viện, dữ liệu và điểm nổi bật từ tình huống 2
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.preprocess import phi  # noqa: E402
from data.synthetic import ANALYTE_BANDS, make_dataset  # noqa: E402

OUT = os.path.join(ROOT, "teaching", "output")
ds = make_dataset(n=1500, D=400, seed=0)  # cùng dữ liệu, cùng seed với tình huống 2
X, wl = phi(ds.X, ds.wl)
test = np.load(f"{OUT}/th2_test_idx.npy")  # các phổ đã có điểm nổi bật
Pm = np.load(f"{OUT}/th2_prominence_test.npy")  # (số phổ kiểm tra, 400)
X, Y = X[test], ds.y[test]  # từ đây chỉ làm việc trên các phổ này
N_SELECT = 40  # số bước sóng giữ lại (10% của 400 kênh)
PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=0,
)
kf = KFold(n_splits=5, shuffle=True, random_state=0)


def metrics(y, yhat):
    return r2_score(y, yhat), np.sqrt(mean_squared_error(y, yhat))


# %% Bước 2. Hồi quy trên toàn phổ và trên bước sóng chọn trong từng fold
results = {}
for ai, name in enumerate(ds.analytes):
    keep = Y[:, ai] > 0  # mẫu có dư lượng của hoạt chất này
    Xa, ya, Pa = X[keep], Y[keep, ai], Pm[keep]
    pred_full, pred_sel = np.zeros_like(ya), np.zeros_like(ya)
    for tr, te in kf.split(Xa):
        # (a) toàn phổ: 400 biến đầu vào
        m = XGBRegressor(**PARAMS).fit(Xa[tr], ya[tr])
        pred_full[te] = m.predict(Xa[te])
        # (b) chọn N_SELECT bước sóng có điểm nổi bật trung bình cao nhất,
        #     CHỈ dùng phổ huấn luyện của fold này để tránh rò rỉ dữ liệu
        sel = np.sort(np.argsort(-Pa[tr].mean(axis=0))[:N_SELECT])
        m2 = XGBRegressor(**PARAMS).fit(Xa[tr][:, sel], ya[tr])
        pred_sel[te] = m2.predict(Xa[te][:, sel])
    results[name] = dict(
        n=len(ya),
        full=metrics(ya, pred_full),
        sel=metrics(ya, pred_sel),
        last_sel=sel,
        y=ya,
        yhat=pred_sel,
    )

h = ["Hoạt chất", "n", "R2 toàn", "RMSE", "R2 chọn", "RMSE"]
print(f"{h[0]:14s} {h[1]:>4s} {h[2]:>8s} {h[3]:>6s} {h[4]:>8s} {h[5]:>6s}")
for name, r in results.items():
    (a, b), (c, d) = r["full"], r["sel"]
    print(f"{name:14s} {r['n']:4d} {a:8.3f} {b:6.3f} {c:8.3f} {d:6.3f}")

# %% Bước 3. Trực quan hóa: R2 hai cấu hình, và các bước sóng được chọn
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.arange(len(results))
ax.bar(x - 0.2, [r["full"][0] for r in results.values()], 0.4, label="Toàn phổ")
ax.bar(
    x + 0.2,
    [r["sel"][0] for r in results.values()],
    0.4,
    label=f"{N_SELECT} bước sóng",
)
ax.set_xticks(x)
ax.set_xticklabels(list(results), rotation=20)
ax.set_ylabel("R²")
ax.set_ylim(0.6, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/th3_hoi_quy.png", dpi=150)
plt.close()

name = "Imidacloprid"
lo, hi = ANALYTE_BANDS[name]
sel = results[name]["last_sel"]
plt.figure(figsize=(8, 3))
plt.plot(wl, X.mean(axis=0), color="k", lw=1, label="Phổ trung bình")
plt.scatter(
    wl[sel], X.mean(axis=0)[sel], color="r", s=10, label="Bước sóng được chọn"
)
plt.axvspan(lo, hi, color="orange", alpha=0.2, label="Dải hấp thụ tham chiếu")
plt.xlabel("Bước sóng (nm)")
plt.legend()
plt.title(f"{N_SELECT} bước sóng được chọn cho {name}")
plt.tight_layout()
plt.savefig(f"{OUT}/th3_buoc_song_chon.png", dpi=150)
plt.close()

# %% Bước 4. Minh họa rò rỉ dữ liệu: chọn bước sóng theo tương quan với y
# Cách SAI: chọn bước sóng trên toàn bộ dữ liệu rồi mới kiểm định chéo.
# Cách ĐÚNG: chọn lại trong từng fold, chỉ trên phần huấn luyện.
rng = np.random.default_rng(0)
Xn = rng.normal(size=(150, 2000))  # 2.000 biến hoàn toàn ngẫu nhiên
yn = rng.normal(size=150)  # biến đích không liên quan đến X


def top_corr(Xs, ys, k=20):
    c = np.abs([np.corrcoef(Xs[:, j], ys)[0, 1] for j in range(Xs.shape[1])])
    return np.argsort(-c)[:k]


sel_all = top_corr(Xn, yn)  # SAI: đã nhìn thấy y của tập kiểm tra
p_wrong, p_right = np.zeros_like(yn), np.zeros_like(yn)
for tr, te in kf.split(Xn):
    m = XGBRegressor(**PARAMS).fit(Xn[tr][:, sel_all], yn[tr])
    p_wrong[te] = m.predict(Xn[te][:, sel_all])
    s = top_corr(Xn[tr], yn[tr])  # ĐÚNG
    m = XGBRegressor(**PARAMS).fit(Xn[tr][:, s], yn[tr])
    p_right[te] = m.predict(Xn[te][:, s])
print(f"\nChọn biến trước khi chia fold (SAI): R2 = {r2_score(yn, p_wrong):.3f}")
print(f"Chọn biến trong từng fold (ĐÚNG):    R2 = {r2_score(yn, p_right):.3f}")
print(f"Hình đã lưu trong {OUT}")
