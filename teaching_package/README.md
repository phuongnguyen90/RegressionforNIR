# SpecMultiMAE-AD: gói mã nguồn giảng dạy và hồi quy hàm lượng

Gói mã nguồn bàn giao kèm Báo cáo tổng kết đề tài NCKH cấp trường *Nghiên cứu phát triển kiến trúc học sâu xác định hàm lượng các chất trên dữ liệu phổ cận hồng ngoại* (Trường Đại học Phạm Văn Đồng, 2026), cài đặt theo kiến trúc đã công bố trong bài báo:

> V.H. Nguyen, M.N. Phan, T.H.P. Nguyen, *SpecMultiMAE-AD: A self-supervised masked autoencoder for anomaly detection and localization in NIR spectra*, Applied Soft Computing 204 (2027) 116756.

Gói này là bản thu gọn của SpecMultiMAE-AD ở thư mục gốc của kho (`specmae/`), được tổ chức để sinh viên đọc và chạy được trên máy cấu hình thấp. Gói bổ sung giai đoạn hồi quy hàm lượng bằng XGBoost và ba tình huống học tập của học phần Trí tuệ nhân tạo. Mọi lệnh dưới đây chạy từ thư mục `teaching_package/`.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `data/` | đọc, ghi dữ liệu (`io.py`), toán tử tiền xử lý Φ (`preprocess.py`), sinh dữ liệu mô phỏng (`synthetic.py`) |
| `models/` | nhúng mảnh phổ có nhận biết bước sóng (`embedding.py`), bộ mã hóa và giải mã Transformer bất đối xứng (`specmultimae.py`), hàm mất mát CASRL (`casrl.py`), vòng huấn luyện (`trainer.py`) |
| `masking/` | che mặt nạ đa đoạn liên tục CMSM, Algorithm 1 (`cmsm.py`) |
| `inference/` | MMIE, Algorithm 2: bản đồ lỗi, phần dư, điểm bất thường, điểm nổi bật (`mmie.py`); dò đỉnh và chọn bước sóng (`localize.py`); độ đo (`metrics.py`) |
| `regression/` | hồi quy XGBoost trên toàn phổ và trên tập bước sóng chọn (`xgb.py`) |
| `scripts/` | chuẩn bị dữ liệu, kiểm định chéo toàn khung, loại bỏ thành phần, khảo sát siêu tham số, đối chứng, dự đoán |
| `configs/` | cấu hình mặc định và cho ba bộ dữ liệu; dải hấp thụ tham chiếu của 11 hoạt chất |
| `teaching/` | mã nguồn ba tình huống học tập của tài liệu tích hợp giảng dạy học phần Trí tuệ nhân tạo |
| `tests/` | kiểm thử đơn vị |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
```

Cấu hình đầy đủ (12 lớp mã hóa, d = 512) nên chạy trên GPU; bài báo ghi nhận bộ nhớ GPU đỉnh 2,35 GB với kích thước lô 256. Cấu hình `configs/smoke_test.yaml` chạy được trên CPU trong vài phút.

## Quy trình sáu bước

```bash
# 1-2. Chuẩn bị dữ liệu và tiền xử lý Φ (lấy mẫu lại, Savitzky-Golay 15/2, SNV, min-max)
python scripts/prepare_data.py --spectra spectra.csv --meta meta.csv \
    --analytes Thiamethoxam Permethrin Metalaxyl Azoxystrobin Imidacloprid Difenoconazole \
               Cypermethrin Chlorothalonil Indoxacarb Abamectin Clothianidin \
    --normal-col normal --group-col sample_id \
    --ref-bands configs/ref_bands_vf_pst_nir.json \
    --config configs/vf_pst_nir.yaml --out runs/vf.npz

# 3-6. Huấn luyện tự giám sát trên phổ nền, suy luận MMIE, dò đỉnh, hồi quy XGBoost (5 fold x 3 lần chạy)
python scripts/run_cv.py --data runs/vf.npz --config configs/vf_pst_nir.yaml --out runs/vf

# Nghiên cứu loại bỏ thành phần và khảo sát siêu tham số
python scripts/ablation.py --data runs/vf.npz --config configs/vf_pst_nir.yaml --out runs/vf_ablation
python scripts/sweep.py    --data runs/vf.npz --config configs/vf_pst_nir.yaml --out runs/vf_sweep

# Đối chứng cài đặt lại được (Isolation Forest, AE, Deep SVDD, PLS-VIP)
python scripts/baselines.py --data runs/vf.npz --config configs/vf_pst_nir.yaml --out runs/vf_baselines

# Dùng mô hình đã huấn luyện cho phổ mới
python scripts/predict.py --data runs/new.npz --config configs/vf_pst_nir.yaml \
    --model runs/vf/model_run0_fold0.pt --threshold <ngưỡng trong results.json> --out runs/pred.json
```

Chạy thử nhanh toàn bộ quy trình trên dữ liệu mô phỏng:

```bash
python scripts/prepare_data.py --synthetic --n 1500 --D 400 --out runs/synth.npz --config configs/smoke_test.yaml
python scripts/run_cv.py --data runs/synth.npz --config configs/smoke_test.yaml --out runs/smoke
```

Định dạng dữ liệu đầu vào: `spectra.csv` có mỗi hàng là một phổ và mỗi cột là một bước sóng (tiêu đề cột là số nm). `meta.csv` có cùng thứ tự hàng, gồm cột `normal` (1 = phổ nền, không phát hiện dư lượng của bất kỳ hoạt chất nào), các cột hàm lượng và cột mã mẫu vật lý dùng khi chia fold.

Đầu ra của `run_cv.py` gồm `results.json` (độ đo theo từng fold và tổng hợp), `prominence.npy` (điểm nổi bật P(λ) của mọi phổ) và trọng số `model_run*_fold*.pt`.

## Đối chiếu với bài báo

| Thành phần | Bài báo | Tệp |
|---|---|---|
| Nhúng mảnh, nhúng bước sóng tuyến tính hoặc Fourier | công thức 4-9 | `models/embedding.py` |
| CMSM | Algorithm 1, công thức 10-12 | `masking/cmsm.py` |
| Bộ mã hóa, giải mã bất đối xứng | công thức 18-23 | `models/specmultimae.py` |
| CASRL | công thức 24-27 | `models/casrl.py` |
| MMIE, điểm bất thường | Algorithm 2, công thức 13-17, 28 | `inference/mmie.py` |
| Điểm nổi bật, dò đỉnh, NMS | công thức 29-30, Section 4.5.2 | `inference/mmie.py`, `inference/localize.py` |
| Giao thức đánh giá, ngưỡng Youden J | Section 5.4-5.5 | `scripts/run_cv.py`, `inference/metrics.py` |
| Hồi quy XGBoost toàn phổ và phổ chọn | Table 15 | `regression/xgb.py` |
| Loại bỏ thành phần, siêu tham số | Tables 6-8 | `scripts/ablation.py`, `scripts/sweep.py` |

## Chi tiết cài đặt

1. **Kích thước bộ mã hóa và giải mã.** Token 1.024 chiều sau ghép được chiếu tuyến tính về 512 trước khi vào bộ mã hóa. Bộ giải mã có độ rộng 256, gồm hai nửa 128 chiều (token và nhúng bước sóng).
2. **Thang bước sóng của nhúng Fourier.** Bước sóng được đổi sang micromet để giữ ý nghĩa tuyệt đối giữa các thiết bị; tần số học được khởi tạo chuẩn với độ lệch 10.
3. **Số đoạn K trong CMSM.** K là số đoạn tối đa; các đoạn được nối dài cho tới khi đạt đúng tỉ lệ che, nên mọi phổ trong lô có cùng số mảnh nhìn thấy. Cận độ dài đoạn tự co lại với phổ ngắn.
4. **β của CASRL** mặc định 1,0, chỉnh được trong cấu hình.
5. **Tổng hợp MMIE.** Mặc định lấy trung bình tại mỗi bước sóng trên các lượt bước sóng đó bị che (`inference.masked_only: true`), là các vị trí mô hình được huấn luyện để tái tạo. Đặt `false` để lấy trung bình trên phổ tái tạo đầy đủ theo công thức 14-15.
6. **Chuẩn hóa điểm nổi bật.** P(λ) được chia cho giá trị lớn nhất của từng phổ để các ngưỡng chiều cao 0,10 và độ nổi bật 0,08 có cùng thang trên mọi bộ dữ liệu.
7. **Chọn bước sóng cho hồi quy.** Tập 80-120 bước sóng (mặc định 100) là các bước sóng có P(λ) trung bình cao nhất trên các phổ chứa hoạt chất thuộc tập huấn luyện của từng fold, nên không dùng thông tin của tập kiểm tra. Hồi quy chạy trên các mẫu có dư lượng của hoạt chất đó. Metalaxyl bị loại như trong bài báo.
8. **Độ đo định vị.** IoU, Peak F1, sai số đỉnh và Top-3 được định nghĩa trong `inference/metrics.py`.
9. **MSMI-NIR.** Lưới 32 điểm được chia mảnh với P = 4, cho 8 mảnh, mỗi mảnh ứng với một băng gốc của cảm biến (`configs/msmi_nir.yaml`).
10. **Đối chứng.** `scripts/baselines.py` cài lại Isolation Forest, AE, Deep SVDD và PLS-VIP theo cùng giao thức chia fold. Các đối chứng còn lại dùng mã nguồn của tác giả gốc hoặc thư viện chuẩn.

## Dữ liệu

FDS-NIR và MSMI-NIR là bộ dữ liệu công khai (xem tài liệu tham khảo của báo cáo). VF-PST-NIR thuộc dự án 24/HĐ-SKHCN của Sở Khoa học và Công nghệ thành phố Đà Nẵng, không phân phối kèm gói này. Dữ liệu mô phỏng trong `data/synthetic.py` chỉ dùng để chạy thử và giảng dạy, không dùng để báo cáo kết quả khoa học.
