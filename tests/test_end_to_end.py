"""Short end-to-end run on synthetic spectra: training must reduce the loss and the
detector must separate injected absorption bands from normal spectra."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specmae.config import ExperimentConfig  # noqa: E402
from specmae.data import make_synthetic, one_class_folds  # noqa: E402
from specmae.downstream import anomaly_score  # noqa: E402
from specmae.metrics import detection_metrics  # noqa: E402
from specmae.pipeline import SpecMultiMAEAD  # noqa: E402
from specmae.preprocessing import preprocess  # noqa: E402


def test_train_and_detect(tmp_path):
    torch.manual_seed(0)
    cfg = ExperimentConfig.load(Path(__file__).resolve().parents[1] / "configs" / "synthetic.json")
    cfg.train.epochs = 25
    cfg.train.patience = 25
    cfg.train.device = "cpu"
    cfg.inference.ensemble_size = 8
    d = make_synthetic(n_normal=300, n_anomalous=120, n_channels=256, amp_range=(0.2, 0.3), seed=1)
    X, wl = preprocess(d.X, d.wavelengths, cfg.preprocess)
    fold = next(one_class_folds(d.labels, 5, 0.1, seed=0))

    det = SpecMultiMAEAD(wl, cfg, device="cpu")
    det.fit(X[fold.train], X[fold.val[d.labels[fold.val] == 0]], log=lambda m: None)
    assert det.history[-1]["loss"] < det.history[0]["loss"]

    det.calibrate(det.score(X[fold.val]), d.labels[fold.val])
    m = detection_metrics(det.score(X[fold.test]), d.labels[fold.test], det.threshold)
    # a short CPU run is far from converged (~0.6 AUROC at 25 epochs, ~0.74 at 150); only require signal
    assert m["auroc"] > 0.55, m

    # the reconstruction error must concentrate inside the injected absorption bands
    yt = d.labels[fold.test]
    r = det.infer(X[fold.test])
    in_band = np.zeros_like(r.error, dtype=bool)
    for j, i in enumerate(fold.test):
        if d.bands[i]:
            lo, hi = d.bands[i][0]
            in_band[j] = (wl >= lo) & (wl <= hi)
    out_band = (yt[:, None] == 1) & ~in_band
    assert r.error[in_band].mean() > 1.2 * r.error[out_band].mean()

    res = det.infer(X[fold.test][:4])
    peaks = det.localize(res)
    assert len(peaks) == 4 and all(len(p) >= 1 for p in peaks)

    # optional per-wavelength normal standardisation
    det.cfg.inference.normal_reference = "zscore"
    det.fit_reference(X[fold.train])
    rz = det.infer(X[fold.test][:6])
    assert rz.error.shape == (6, X.shape[1]) and (rz.error >= 0).all()
    assert np.allclose(rz.r_avg, det.infer(X[fold.test][:6], raw=True).r_avg)

    det.save(tmp_path / "m.pt")
    det2 = SpecMultiMAEAD.load(tmp_path / "m.pt", device="cpu")
    s1 = anomaly_score(det.infer(X[:3], seed=5).error)
    s2 = anomaly_score(det2.infer(X[:3], seed=5).error)
    assert np.allclose(s1, s2, atol=1e-5)
