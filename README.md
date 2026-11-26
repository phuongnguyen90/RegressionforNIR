# SpecMultiMAE-AD

A self-supervised masked autoencoder for **anomaly detection** and **anomalous absorption-wavelength localization** in 1-D near-infrared (NIR) spectra.

This repository implements the architecture published in:

> V.H. Nguyen, M.N. Phan, T.H.P. Nguyen, *SpecMultiMAE-AD: A self-supervised masked autoencoder for anomaly detection and localization in NIR spectra*, Applied Soft Computing 204 (2027) 116756. https://doi.org/10.1016/j.asoc.2026.116756

It has two parts:
- `specmae/`: the full implementation (this README);
- `teaching_package/`: a compact version for teaching, which adds the downstream **content regression** stage (XGBoost on the full spectrum vs. on the wavelengths selected by SpecMultiMAE-AD) and three learning scenarios. See [`teaching_package/README.md`](teaching_package/README.md).

The model is trained **only on normal spectra**; it needs no anomaly labels, no annotations and no chemical reference libraries. It learns to reconstruct hidden bands of normal spectra, so a foreign substance shows up in two ways:
- the spectrum's reconstruction error rises (detection);
- residual peaks appear at the wavelengths where that substance absorbs (localization).

## How it works

```
spectrum S (D channels)
  │  split into N = ⌈D/P⌉ patches of P channels
  ▼
wavelength-aware tokens  t_j = [ LayerNorm(W·p_j) ; wavelength embedding of patch j ]
  │  CMSM: hide K contiguous bands of patches (ratio r)
  ▼
encoder (deep Transformer)        ← visible tokens only
  ▼
decoder (shallow Transformer)     ← encoded tokens + learnable mask tokens, each with its wavelength embedding
  ▼
reconstruction Ŝ (D channels)  ──  training loss on hidden positions only (CASRL)

inference (MMIE): reconstruct each test spectrum under M random masks
  → average reconstruction R_avg, error map E(λ), residual R(λ) = S − R_avg
  → anomaly score A, and ranked anomalous wavelengths
```

| Component | What it does | Module |
|---|---|---|
| **Wavelength-aware patch embedding** | Each patch is linearly projected (+LayerNorm) and concatenated with an embedding of its central wavelength λ_j. The embedding is learnable Fourier features `[sin 2πW_fλ, cos 2πW_fλ]` (default) or a linear map. This gives tokens physical wavelength positions instead of generic index positions. | `specmae/embedding.py` |
| **CMSM** (contiguous multi-segment masking) | Hides K non-overlapping contiguous bands of patches instead of scattered patches, mimicking the band structure of NIR absorption. | `specmae/masking.py` |
| **Asymmetric encoder–decoder** | Deep, wide pre-LN Transformer encoder on visible tokens; shallow, narrower decoder over the full sequence; linear head back to P values per patch. | `specmae/model.py` |
| **CASRL** (contiguous-aware spectral reconstruction loss) | `L = L_rec + λ·L_grad`. `L_rec = Σ_k w_k · MSE(segment k)` with `w_k = (l_k / mean l)^β`, so longer hidden bands weigh more. `L_grad` matches first differences (local slopes) on hidden positions. | `specmae/loss.py` |
| **MMIE** (multi-mask ensemble inference) | M independent CMSM masks per test spectrum. Outputs `R_avg = mean_m Ŝ⁽ᵐ⁾`, `E(λ) = mean_m (S − Ŝ⁽ᵐ⁾)²`, `R(λ) = S − R_avg`. | `specmae/inference.py` |
| **Detection** | Score `A = mean_λ E(λ)`; anomalous if `A > τ`. τ is chosen with Youden's J on a labelled validation split, or as μ + kσ of normal validation scores. | `specmae/downstream.py` |
| **Localization** | Prominence `P(λ) = α·|R(λ)| + (1−α)·E(λ)`, max-normalized. Peaks are detected with height, prominence and minimum-separation (nm) constraints, ranked by score, then de-duplicated by interval non-maximum suppression. | `specmae/downstream.py` |
| **Preprocessing** | Per spectrum: optional uniform resampling → Savitzky–Golay smoothing (window 15, order 2) → SNV → min–max to [0, 1]. Nothing is fitted, so nothing leaks between train and test. | `specmae/preprocessing.py` |
| **Evaluation** | One-class K-fold protocol, detection and localization metrics. | `specmae/data.py`, `specmae/metrics.py` |
| **Downstream regression** | XGBoost trained on the full spectrum and on the top-ranked wavelengths (80–120, default 100) selected from the prominence score within each training fold; R² and RMSE per analyte. | `teaching_package/regression/xgb.py`, `teaching_package/scripts/run_cv.py` |
| **Training** | AdamW, linear warm-up + cosine schedule, mixed precision on CUDA, early stopping on validation reconstruction MSE. | `specmae/train.py` |

`specmae/pipeline.py` wraps all of this in one class, `SpecMultiMAEAD`; `specmae/config.py` holds every setting as dataclasses loaded from JSON.

## Installation

Python ≥ 3.9 and PyTorch ≥ 1.12.

```bash
git clone https://github.com/phuongnguyen90/RegressionforNIR.git && cd RegressionforNIR
conda create -n specmae -c conda-forge python=3.11 numpy scipy scikit-learn pandas openpyxl pytest
conda activate specmae
pip install torch --index-url https://download.pytorch.org/whl/cu126   # choose the build for your CUDA / CPU
pytest -q                                                               # unit tests + short end-to-end run
```

**On Windows, always `conda activate` the environment** rather than calling its `python.exe` directly. Without activation, the env's `Library\bin` folder is not on PATH; scipy's BLAS/LAPACK DLLs then fail to load, and Python crashes with error `0xc06d007f`.

## Quick start

All commands are run from the repository root.

```bash
# Synthetic spectra with injected absorption bands: checks that everything runs end to end
python run_experiment.py --config configs/synthetic.json

# Your data (see "Data format"); 5-fold one-class cross-validation
python run_experiment.py --config configs/vf_pst.json --data path/to/spectra.csv

# Override any setting with dotted keys; run only some folds
python run_experiment.py --config configs/vf_pst.json --set train.epochs=200 inference.ensemble_size=128 --folds 0,1
```

Each run writes these files to `runs/<name>_<variant>_<timestamp>/`:
- `config.json`: the exact configuration used.
- `log.txt`: the training log.
- `fold_results.csv`: metrics for each fold and run.
- `summary.json`: mean ± std, overall and per unit (e.g. per instrument).
- `test_scores.csv`: score, prediction, top-3 peak wavelengths and ground-truth bands for every test spectrum.
- `model_*.pt`: checkpoints, written only with `--save-models`.

### Python API

```python
from specmae import ExperimentConfig, SpecMultiMAEAD, preprocess

cfg = ExperimentConfig.load("configs/vf_pst.json")
X, wl = preprocess(X_raw, wavelengths_nm, cfg.preprocess)           # (n, D) and (D,) in nm

det = SpecMultiMAEAD(wl, cfg).fit(X_train_normal, X_val_normal)     # self-supervised, normals only
det.calibrate(det.score(X_val), y_val)                              # threshold (Youden's J by default)

res = det.infer(X_test)            # MMIE: res.r_avg, res.error (E), res.residual (R)
scores = det.anomaly_score(res)
pred = det.predict(scores)         # 1 = anomalous
peaks = det.localize(res)          # per spectrum: ranked Peak(wavelength, confidence, left_nm, right_nm, ...)

det.save("model.pt"); det = SpecMultiMAEAD.load("model.pt")
```

## Evaluation protocol

`specmae.data.one_class_folds` works as follows:

1. **Folds.** Normal spectra are split into K folds (default 5). When `data.group_col` gives a physical-sample id, the split uses GroupKFold, so repeated scans of one item never cross folds.
2. **Train / held out.** For fold i, the model trains on the normals of the other folds. The held-out normals and all anomalous spectra are divided into two parts:
   - **validation** (`val_frac`, default 10%): its normals drive early stopping; normals and anomalies together set the decision threshold;
   - **test**: everything else.

   The test part always keeps at least one normal and one anomaly. With groups, anomalies are split by physical sample too.
3. **Anomalous spectra are never used for training.**
4. **Threshold fallbacks.** If the validation part has no anomalies, the threshold is μ + kσ of normal validation scores. If it has no normals, it is μ + kσ of training-normal scores; this is optimistic and the runner warns about it.
5. **Repeats.** `runs` repeats the whole protocol with different seeds.

### Metrics

**Detection** (`detection_metrics`): AUROC, AUPRC, and precision / recall / F1 at the calibrated threshold.

**Localization** (`localization_metrics`) is computed on anomalous test spectra that have ground-truth bands (nm intervals), then averaged over spectra:
- **IoU**: interval IoU between the top-ranked peak's half-height interval and the best-matching band.
- **Peak F1**: a peak is a hit if it lies inside a band widened by ± `localization.tolerance_nm`. Precision = hit peaks / peaks; recall = bands hit / bands.
- **Mean peak error**: distance (nm) from the top peak to the nearest band, 0 if inside.
- **Top-3 hit rate**: share of spectra where at least one of the top-3 peaks hits a band (± tolerance).

## Data format

`data.source = "table"` reads CSV, XLSX or Parquet with one row per spectrum.

| Setting | Meaning |
|---|---|
| wavelength columns | Every column whose name parses as a number (nm), optionally after `data.wavelength_prefix`; or list them in `data.wavelength_cols`. |
| `label_col` + `normal_values` | A spectrum is normal if its label is in `normal_values`, compared both as text and as a number (`0` matches `0.0`). Rows with an empty label are dropped. |
| `anomaly_cols` | Alternative to `label_col`: a spectrum is anomalous if any of these columns is > 0 (e.g. contaminant concentrations). |
| `bands_file` | JSON `{contaminant: [[lo_nm, hi_nm], ...]}` of known absorption bands, used only for localization metrics. The contaminants in a spectrum come from `anomaly_cols` (columns > 0) or `contaminant_col` (names separated by `contaminant_sep`). |
| `group_col` | Physical-sample id, which keeps repeated scans of one item together. |
| `split_col` / `subsets` | Train and evaluate a separate model per value of a column, or per file (e.g. one per instrument with a different wavelength range). `--data` cannot be combined with `subsets`. |

`data.source = "npz"` expects arrays `X`, `wavelengths` and `labels`, plus optional `groups` and `bands`. `data.source = "synthetic"` uses `make_synthetic`.

**Missing values** never reach the model:
- channels that are missing in every spectrum are dropped;
- spectra with ≤ 5% missing channels are linearly interpolated;
- spectra with more missing channels are dropped. Every case is reported as a warning.

## Configuration

Settings are JSON files in `configs/`, mirroring the dataclasses in `specmae/config.py`. Key settings and their defaults:

| Key | Default | Notes |
|---|---|---|
| `model.patch_size` | 32 | Needs ≥ 2 patches per spectrum. Use a smaller P for short spectra (e.g. P = 4 for 32 points). |
| `model.embed_dim` / `encoder_dim` / `decoder_dim` | 512 / 512 / 256 | Token = [content (d) ; wavelength (d)] = 2d, projected to the encoder width. |
| `model.encoder_depth` / `decoder_depth` | 12 / 4 | 8 heads each. |
| `model.wavelength_embed` | `fourier` | `fourier` (learnable frequencies, `fourier_frequencies=256`), `linear`, or `sinusoidal` (plain index positions, no wavelength information). |
| `model.wl_ref_range` | [300, 2700] nm | Fixed range mapping wavelengths to [0, 1], identical for every instrument, so coordinates stay absolute. |
| `mask.strategy` | `cmsm` | `cmsm` or `random` (scattered patches). |
| `mask.ratio`, `k_min..k_max`, `l_min..l_max` | 0.75, 4–8, 8–64 patches | Bounds shrink automatically when they do not fit the spectrum. |
| `loss.name` | `casrl` | `casrl` or `mse` (plain masked MSE). |
| `loss.grad_weight` (λ), `loss.beta` | 0.5, 1.0 | |
| `loss.normalize_weights` | false | See "Implementation notes". |
| `train.*` | AdamW 1e-4, wd 0.05, batch 256, ≤ 800 epochs, 10 warm-up epochs, patience 50 | Use a smaller batch for datasets with few normal spectra. |
| `inference.ensemble_size` (M) | 64 | M = 1 is single-pass inference. |
| `inference.aggregation` | `full` | `full` or `masked`. |
| `inference.error_map` | `mse` | `mse` or `variance` (ensemble variance). |
| `inference.normal_reference` | `none` | `zscore`: optional per-wavelength standardization of E and \|R\| with statistics of normal training spectra. |
| `inference.score_reduce` | `mean` | `mean` or `max`. |
| `threshold.method` | `youden` | `youden` or `sigma` (μ + k·σ). |
| `localization.*` | α = 0.5; separation 10 nm; height 0.10; prominence 0.08; tolerance 15 nm | For coarse or interpolated spectra, use a larger separation and higher thresholds. |

### Variants (`--ablation`)

Each variant removes or replaces one component:

| Variant | Change |
|---|---|
| `no_cmsm` | random scattered masking |
| `no_mmie` | M = 1 |
| `no_wavelength` | index-based sinusoidal positions |
| `linear_wavelength` | linear wavelength embedding |
| `no_casrl` | plain masked MSE |

## Implementation notes

1. **Masking sampler.** CMSM masks exactly ⌊r·N⌋ patches (clamped to [1, N−1]) in exactly K segments, separated by at least one visible patch:
   - K ~ U{k_min..k_max}, reduced when it does not fit;
   - segment lengths fall within [l_min, l_max], with the bounds shrunk to fit N.

   A fixed masked count keeps every spectrum in a batch at the same number of visible tokens. The procedure always terminates, even for short spectra.
2. **Decoder tokens.** Encoded visible tokens are projected to the decoder width, and mask tokens fill the hidden positions. A decoder-side wavelength embedding is concatenated to *every* position and fused back to the decoder width. Wavelength embeddings are the only positional information.
3. **Segment weighting.** `L_rec = Σ_k w_k·MSE_k` has weights summing to about K, so its scale grows with the number of segments. Scattered masking produces many more segments than CMSM. When comparing masking strategies, set `loss.normalize_weights=true` (divide by Σ w_k) so both are on the same loss scale.
4. **Ensemble aggregation.** `full` averages the complete reconstructions of all M passes. `masked` averages each wavelength only over the passes in which it was hidden; the decoder is never trained on visible positions, so this can be cleaner. It falls back to all passes for wavelengths never hidden.
5. **Linear wavelength embedding.** LayerNorm(w·λ) is identical for every λ > 0, so the coordinate is centred to [−1, 1] and the bias is initialized non-zero. Otherwise all patches would start with the same embedding.

### Optional per-wavelength calibration

`inference.normal_reference=zscore` standardizes E and |R| per wavelength with mean and std statistics from normal training spectra (`reference_size` of them, stored in the checkpoint); `inference.score_reduce=max` scores a spectrum by its largest error instead of the mean. Both are off by default.

```bash
python run_experiment.py --config configs/vf_pst.json --data spectra.csv \
    --set inference.normal_reference=zscore inference.score_reduce=max
```

## Provided configurations

| File | Use |
|---|---|
| `configs/synthetic.json` | Small model on synthetic data; runs in minutes. |
| `configs/vf_pst.json` | High-resolution spectra (2,136 channels), contaminant-concentration columns as labels, known bands in `configs/bands_vf_pst.json`. |
| `configs/msmi.json` | 8-band sensor resampled to 32 points, with P = 4 and coarser localization settings. |
| `configs/fds.json` | One model per instrument (`data.subsets`), batch size 32 for small normal sets. |

**The data paths and column names in the dataset configs are placeholders; adapt them to your files.** By default they point to `data/` in the repository root, which is git-ignored, so datasets are never committed.

`configs/bands_vf_pst.json` lists typical NIR absorption ranges of the functional groups of 11 pesticides. These ranges are approximate reference values, not measured spectra of the pure compounds.

## Hardware

The default model (12 + 4 layers, width 512 / 256, batch 256, 2,136 channels) is meant to run on a GPU; the paper reports a peak GPU memory of 2.35 GB at batch size 256. For small GPUs, lower `train.batch_size`, `model.embed_dim` / `encoder_dim` or `encoder_depth`, or raise `model.patch_size`. `inference.chunk_size` caps the forward passes per inference batch.
