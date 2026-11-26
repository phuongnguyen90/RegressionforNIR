"""Tình huống học tập 2: Học biểu diễn tự giám sát bằng bộ tự mã hóa mặt nạ.

Chạy toàn bộ:  python teaching/th2_hoc_tu_giam_sat_mae.py   (khoảng 2 phút trên CPU)
Hoặc chạy từng khối "# %%". Có thể đổi các tham số ở Bước 1 để làm bài tập.
"""

# %% Bước 1. Nạp thư viện, tham số thí nghiệm
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.preprocess import phi  # noqa: E402
from data.synthetic import ANALYTE_BANDS, make_dataset  # noqa: E402
from inference.localize import detect_peaks  # noqa: E402
from inference.mmie import mmie, prominence  # noqa: E402
from masking.cmsm import batch_masks, cmsm_patch_mask, scattered_patch_mask  # noqa
from models.casrl import casrl_loss  # noqa: E402
from models.specmultimae import SpecMultiMAE  # noqa: E402

OUT = os.path.join(ROOT, "teaching", "output")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)

P = 16  # kích thước mảnh phổ (số bước sóng mỗi mảnh)
MASK = dict(strategy="cmsm", mask_ratio=0.75, l_min=2, l_max=8, K_range=(4, 8))
EPOCHS = 120
M_INFER = 16  # số mặt nạ khi suy luận MMIE (bài tập: thử M = 1)
cfg = {"model": {"P": P}, "masking": MASK, "inference": {"M": M_INFER, "alpha": 0.5}}

# %% Bước 2. Dữ liệu: tiền xử lý và chia tập
ds = make_dataset(n=1500, D=400, seed=0)
X, wl = phi(ds.X, ds.wl)  # Savitzky-Golay, SNV, min-max
nidx = rng.permutation(np.flatnonzero(ds.normal))  # chỉ số phổ nền
aidx = np.flatnonzero(~ds.normal)  # chỉ số phổ có hoạt chất
tr, va, te = nidx[:450], nidx[450:500], nidx[500:]
print(
    f"Huấn luyện {len(tr)} phổ nền, kiểm định {len(va)}, kiểm tra {len(te)} phổ nền"
)
print(f"và {len(aidx)} phổ có hoạt chất (chỉ dùng khi kiểm tra)")

# %% Bước 3. Quan sát mặt nạ: CMSM so với che rời rạc
N = math.ceil(X.shape[1] / P)  # số mảnh của một phổ
m_c, segs = cmsm_patch_mask(N, 0.75, 2, 8, (4, 8), np.random.default_rng(3))
m_s, _ = scattered_patch_mask(1, N, 0.75, np.random.default_rng(3))
print(f"{N} mảnh; CMSM che {m_c.sum()} mảnh thành {len(segs)} đoạn: {segs}")
fig, ax = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
for a, m, t in zip(
    ax, [m_c, m_s[0].numpy()], ["CMSM (đoạn liên tục)", "Che rời rạc"]
):
    a.plot(wl, X[tr[0]], color="k", lw=1)
    for j in np.flatnonzero(m):  # tô xám các mảnh bị che
        a.axvspan(wl[j * P], wl[min((j + 1) * P, len(wl)) - 1], color="0.8")
    a.set_title(t)
ax[1].set_xlabel("Bước sóng (nm)")
plt.tight_layout()
plt.savefig(f"{OUT}/th2_mat_na.png", dpi=150)
plt.close()

# %% Bước 4. Dựng mô hình thu nhỏ và kiểm tra một lượt lan truyền
model = SpecMultiMAE(
    P=P,
    d=64,
    d_enc=64,
    enc_depth=2,
    enc_heads=4,
    d_dec=32,
    dec_depth=1,
    dec_heads=4,
    n_freq=32,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Số tham số của mô hình: {n_params:,}")
Xtr = torch.tensor(X[tr])
Xva = torch.tensor(X[va])
wl_t = torch.tensor(wl, dtype=torch.float32)
mask, segs_b = batch_masks(4, N, MASK, rng)  # 4 mặt nạ cho 4 phổ
x_hat = model(Xtr[:4], wl_t, mask)
print("Đầu vào:", tuple(Xtr[:4].shape), " mặt nạ:", tuple(mask.shape), end="")
print(" đầu ra:", tuple(x_hat.shape))
loss = casrl_loss(Xtr[:4], x_hat, mask, segs_b, P, lam=0.5, beta=1.0)
print(f"CASRL khi chưa huấn luyện: {loss.item():.4f}")

# %% Bước 5. Huấn luyện tự giám sát chỉ trên phổ nền
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
best, best_state, history = float("inf"), None, []
for epoch in range(EPOCHS):
    model.train()
    for idx in torch.randperm(len(Xtr)).split(128):  # lô 128 phổ
        x = Xtr[idx]
        mask, segs_b = batch_masks(len(x), N, MASK, rng)  # mặt nạ mới mỗi lô
        x_hat = model(x, wl_t, mask)
        loss = casrl_loss(x, x_hat, mask, segs_b, P, lam=0.5, beta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():  # sai số tái tạo trên phổ nền kiểm định, tại vị trí bị che
        mask, _ = batch_masks(len(Xva), N, MASK, rng)
        w = mask.repeat_interleave(P, 1)[:, : X.shape[1]].float()
        val = (((Xva - model(Xva, wl_t, mask)) ** 2) * w).sum() / w.sum()
    history.append(val.item())
    if val < best:  # giữ lại trọng số tốt nhất (dừng sớm đơn giản)
        best = val.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch + 1:3d}  loss {loss.item():.5f}  val {val.item():.6f}")
model.load_state_dict(best_state)
plt.figure(figsize=(5, 3.2))
plt.semilogy(history)
plt.xlabel("Epoch")
plt.ylabel("MSE tái tạo (kiểm định)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/th2_huan_luyen.png", dpi=150)
plt.close()

# %% Bước 6. Suy luận tổ hợp đa mặt nạ và đánh giá phát hiện
test = np.concatenate([te, aidx])
y_true = ~ds.normal[test]  # True = phổ có hoạt chất
out = mmie(model, X[test], wl, cfg, M=M_INFER)
score = out["score"]  # điểm bất thường A = trung bình E(lambda)
auc = roc_auc_score(y_true, score)
fpr, tpr, thr = roc_curve(y_true, score)
k = np.argmax(tpr - fpr)  # ngưỡng Youden J
print(f"AUROC = {auc:.3f}; ngưỡng Youden = {thr[k]:.5f}", end=" ")
print(f"(TPR {tpr[k]:.2f}, FPR {fpr[k]:.2f})")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
bins = np.linspace(0, np.quantile(score, 0.99), 50)
ax[0].hist(score[~y_true], bins, alpha=0.6, density=True, label="Phổ nền")
ax[0].hist(score[y_true], bins, alpha=0.6, density=True, label="Có hoạt chất")
ax[0].axvline(thr[k], color="k", ls="--")
ax[0].set_xlabel("Điểm bất thường")
ax[0].set_ylabel("Mật độ")
ax[0].legend()
ax[1].plot(fpr, tpr)
ax[1].plot([0, 1], [0, 1], "k:")
ax[1].set_xlabel("FPR")
ax[1].set_ylabel("TPR")
ax[1].set_title(f"ROC, AUROC = {auc:.3f}")
plt.tight_layout()
plt.savefig(f"{OUT}/th2_phat_hien.png", dpi=150)
plt.close()

# %% Bước 7. Định vị bước sóng hấp thụ trên một phổ chứa Imidacloprid
Pm = prominence(out["R"], out["E"], alpha=0.5)  # điểm nổi bật P(lambda)
ai = ds.analytes.index("Imidacloprid")
lo, hi = ANALYTE_BANDS["Imidacloprid"]
k = int(np.flatnonzero(y_true & (ds.y[test, ai] > 1.0))[0])
peaks = detect_peaks(
    Pm[k], wl, min_sep_nm=10, min_height=0.10, min_prominence=0.08, top_k=3
)
print(
    "Ba đỉnh nổi bật nhất (nm, độ tin cậy):",
    [(round(p), round(c, 2)) for p, c in peaks],
)
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(wl, X[test][k], label="Phổ đo")
ax[0].plot(wl, out["R_avg"][k], "--", label="Tái tạo (MMIE)")
ax[0].legend()
ax[0].set_ylabel("Giá trị chuẩn hóa")
ax[1].plot(wl, Pm[k], color="C2")
ax[1].set_ylabel("Điểm nổi bật P(λ)")
ax[1].set_xlabel("Bước sóng (nm)")
for a in ax:
    a.axvspan(lo, hi, color="orange", alpha=0.2)
for p, _ in peaks:
    ax[1].axvline(p, color="r", lw=0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/th2_dinh_vi.png", dpi=150)
plt.close()

# %% Bước 8. Đánh giá định vị trên mọi phổ chứa từng hoạt chất
print("Tỉ lệ phổ có ít nhất một trong ba đỉnh đầu nằm trong dải hấp thụ (±15 nm):")
for ai, name in enumerate(ds.analytes):
    lo, hi = ANALYTE_BANDS[name]
    rows = np.flatnonzero(y_true & (ds.y[test, ai] > 0))
    hit = 0
    for r in rows:
        pk = detect_peaks(Pm[r], wl, 10, 0.10, 0.08, top_k=3)
        hit += any(lo - 15 <= p <= hi + 15 for p, _ in pk)
    print(f"  {name:14s} {hit / len(rows):.2f}  ({len(rows)} phổ)")

# %% Bước 9. Lưu kết quả cho tình huống 3
torch.save(model.state_dict(), f"{OUT}/th2_model.pt")
np.save(f"{OUT}/th2_prominence_test.npy", Pm)
np.save(f"{OUT}/th2_test_idx.npy", test)
print(f"Đã lưu mô hình và điểm nổi bật trong {OUT}")
