#!/usr/bin/env python
"""Train and evaluate SpecMultiMAE-AD with the one-class K-fold protocol.

Examples
--------
  python run_experiment.py --config configs/synthetic.json
  python run_experiment.py --config configs/vf_pst.json --data ../data/VF-PST-NIR/ALL.csv
  python run_experiment.py --config configs/vf_pst.json --ablation no_cmsm
  python run_experiment.py --config configs/vf_pst.json --set inference.ensemble_size=128 train.epochs=200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specmae.config import ExperimentConfig  # noqa: E402
from specmae.data import SpectralData, load_data, one_class_folds  # noqa: E402
from specmae.downstream import localize_batch, sigma_threshold  # noqa: E402
from specmae.metrics import aggregate, detection_metrics, localization_metrics  # noqa: E402
from specmae.pipeline import SpecMultiMAEAD  # noqa: E402
from specmae.preprocessing import preprocess  # noqa: E402

# Ablation variants: each removes or replaces one component.
ABLATIONS = {
    "none": {},
    "no_cmsm": {"mask.strategy": "random"},
    "no_mmie": {"inference.ensemble_size": 1},
    "no_wavelength": {"model.wavelength_embed": "sinusoidal"},
    "linear_wavelength": {"model.wavelength_embed": "linear"},
    "no_casrl": {"loss.name": "mse"},
}


def _parse_value(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def run_unit(data: SpectralData, cfg: ExperimentConfig, out_dir: Path, folds_sel, save_models: bool, log):
    X, wl = preprocess(data.X, data.wavelengths, cfg.preprocess)
    y = data.labels
    log(f"[{data.name}] {len(X)} spectra x {X.shape[1]} channels "
        f"({wl.min():.0f}-{wl.max():.0f} nm); normal {int((y == 0).sum())}, anomalous {int((y == 1).sum())}")
    rows, sample_rows = [], []
    for run in range(cfg.runs):
        seed = cfg.seed + run
        for fold in one_class_folds(y, cfg.n_folds, cfg.val_frac, data.groups, seed=seed):
            if folds_sel and fold.index not in folds_sel:
                continue
            fseed = seed * 1000 + fold.index
            torch.manual_seed(fseed)
            t0 = time.time()
            det = SpecMultiMAEAD(wl, cfg)
            log(f"[{data.name}] run {run} fold {fold.index}: train {len(fold.train)}, val {len(fold.val)}, "
                f"test {len(fold.test)}, patches {det.model.n_patches}, masked/spectrum {det.sampler.n_masked}")
            y_val = y[fold.val]
            det.fit(X[fold.train], X[fold.val[y_val == 0]], seed=fseed, log=log)
            train_time = time.time() - t0

            s_val = det.anomaly_score(det.infer(X[fold.val], seed=fseed + 1)) if len(fold.val) else np.array([])
            if len(np.unique(y_val)) == 2:
                det.calibrate(s_val, y_val)
            elif (y_val == 0).any():
                det.threshold = sigma_threshold(s_val[y_val == 0], cfg.threshold.k)
                log("[eval] validation split has no anomalies; using mu + k*sigma on validation normals")
            else:
                s_tr = det.anomaly_score(det.infer(X[fold.train], seed=fseed + 3))
                det.threshold = sigma_threshold(s_tr, cfg.threshold.k)
                log("[eval] WARNING: validation split has no normals; using mu + k*sigma on TRAINING normals "
                    "(optimistic, scores on training data are low)")

            t1 = time.time()
            res = det.infer(X[fold.test], seed=fseed + 2)
            infer_ms = 1000 * (time.time() - t1) / max(1, len(fold.test))
            s_test = det.anomaly_score(res)
            y_test = y[fold.test]
            m = detection_metrics(s_test, y_test, det.threshold)

            peaks_all = localize_batch(res.residual, res.error, wl, cfg.localization)
            if data.bands is not None:
                idx = [i for i in range(len(fold.test)) if y_test[i] == 1 and data.bands[fold.test[i]]]
                if idx:
                    m.update(localization_metrics([peaks_all[i] for i in idx],
                                                  [data.bands[fold.test[i]] for i in idx],
                                                  cfg.localization.tolerance_nm))
            best_epoch = min(det.history, key=lambda h: h["val_mse"])["epoch"] if det.history else -1
            m.update(unit=data.name, run=run, fold=fold.index, n_train=len(fold.train), n_test=len(fold.test),
                     best_epoch=best_epoch, train_min=train_time / 60, infer_ms_per_sample=infer_ms)
            rows.append(m)
            log(f"[{data.name}] run {run} fold {fold.index}: AUROC {m.get('auroc', float('nan')):.4f} "
                f"AUPRC {m.get('auprc', float('nan')):.4f} F1 {m['f1']:.4f}"
                + (f" IoU {m['iou']:.3f} PeakF1 {m['peak_f1']:.3f} Top3 {m['top3_hit_rate']:.3f}" if "iou" in m else ""))

            for j, i in enumerate(fold.test):
                pk = peaks_all[j][:3]
                sample_rows.append({
                    "unit": data.name, "run": run, "fold": fold.index, "index": int(i), "label": int(y[i]),
                    "score": float(s_test[j]), "pred": int(s_test[j] > det.threshold),
                    "peaks_nm": ";".join(f"{p.wavelength:.1f}" for p in pk),
                    "peak_conf": ";".join(f"{p.confidence:.3f}" for p in pk),
                    "gt_bands": "" if data.bands is None or not data.bands[i] else
                    ";".join(f"{a:.0f}-{b:.0f}" for a, b in data.bands[i]),
                })
            if save_models:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in data.name)
                det.save(out_dir / f"model_{safe}_run{run}_fold{fold.index}.pt")
    return rows, sample_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", help="override data.path")
    ap.add_argument("--ablation", default="none", choices=sorted(ABLATIONS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. train.epochs=100 mask.ratio=0.6")
    ap.add_argument("--folds", help="comma-separated fold indices to run (default: all)")
    ap.add_argument("--output", help="output directory (default: <output_dir>/<name>_<ablation>_<time>)")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.data:
        if cfg.data.subsets:
            ap.error("--data cannot be used with a config that defines data.subsets (each subset has its own "
                     "'path'); edit the subset paths in the config or override them with --set")
        cfg.data.path = args.data
    for k, v in ABLATIONS[args.ablation].items():
        cfg.override(k, v)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.override(k, _parse_value(v))

    out = Path(args.output or Path(cfg.output_dir) / f"{cfg.name}_{args.ablation}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    log_file = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    folds_sel = {int(f) for f in args.folds.split(",")} if args.folds else None
    all_rows, all_samples = [], []
    for unit in load_data(cfg.data):
        rows, samples = run_unit(unit, cfg, out, folds_sel, args.save_models, log)
        all_rows += rows
        all_samples += samples

    pd.DataFrame(all_rows).to_csv(out / "fold_results.csv", index=False)
    pd.DataFrame(all_samples).to_csv(out / "test_scores.csv", index=False)
    summary = {"overall": aggregate(all_rows),
               "per_unit": {u: aggregate(r for r in all_rows if r["unit"] == u)
                            for u in sorted({r["unit"] for r in all_rows})}}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"\n=== {cfg.name} / ablation={args.ablation} ({len(all_rows)} fold-runs) ===")
    for k in ["auroc", "auprc", "precision", "recall", "f1", "iou", "peak_f1", "mean_peak_error_nm", "top3_hit_rate"]:
        if k in summary["overall"]:
            s = summary["overall"][k]
            log(f"{k:>20s}: {s['mean']:.4f} +/- {s['std']:.4f}")
    log(f"results written to {out}")
    log_file.close()


if __name__ == "__main__":
    main()
