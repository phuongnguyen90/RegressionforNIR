"""High-level detector: fit on normal spectra, then score / threshold / localise."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import ExperimentConfig
from .downstream import Peak, anomaly_score, localize_batch, sigma_threshold, youden_threshold
from .inference import MMIEResult, mmie
from .masking import MaskSampler
from .model import SpecMultiMAE
from .train import resolve_device, train_model


class SpecMultiMAEAD:
    """SpecMultiMAE-AD on spectra that are already preprocessed (see ``preprocessing.preprocess``)."""

    def __init__(self, wavelengths: np.ndarray, cfg: ExperimentConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.device = resolve_device(device or cfg.train.device)
        self.model = SpecMultiMAE(self.wavelengths, cfg.model).to(self.device)
        self.sampler = MaskSampler(cfg.mask, self.model.n_patches)
        self.threshold: Optional[float] = None
        self.history: list = []
        self.reference: Optional[Dict[str, np.ndarray]] = None  # per-wavelength normal statistics

    # ----------------------------------------------------------- training
    def fit(self, X_train: np.ndarray, X_val_normal: Optional[np.ndarray] = None, seed: int = 0,
            log: Callable[[str], None] = print) -> "SpecMultiMAEAD":
        self.history = train_model(self.model, self.sampler, X_train, X_val_normal, self.cfg.loss,
                                   self.cfg.train, self.device, seed=seed, log=log)
        if self.cfg.inference.normal_reference == "zscore":
            self.fit_reference(X_train, seed=seed + 7)
        return self

    def fit_reference(self, X_normal: np.ndarray, seed: int = 0) -> None:
        """Per-wavelength mean/std of E and |R| over normal spectra (``normal_reference="zscore"``)."""
        X_normal = np.asarray(X_normal)
        n = min(len(X_normal), self.cfg.inference.reference_size)
        idx = np.random.default_rng(seed).choice(len(X_normal), size=n, replace=False)
        ref = self.infer(X_normal[idx], seed=seed, raw=True)
        a = np.abs(ref.residual)
        self.reference = {"mu_e": ref.error.mean(0), "sd_e": ref.error.std(0) + 1e-8,
                          "mu_r": a.mean(0), "sd_r": a.std(0) + 1e-8}

    # ----------------------------------------------------------- inference
    def infer(self, X: np.ndarray, seed: int = 0, ensemble_size: Optional[int] = None,
              raw: bool = False) -> MMIEResult:
        """MMIE outputs. With ``normal_reference="zscore"`` (and ``raw=False``) ``error`` and
        ``residual`` are standardised per wavelength (clipped at 0; residual keeps its sign);
        ``r_avg`` is always the raw ensemble reconstruction."""
        res = mmie(self.model, X, self.sampler, self.cfg.inference, np.random.default_rng(seed),
                   self.device, ensemble_size)
        mode = self.cfg.inference.normal_reference
        if raw or mode == "none":
            return res
        if mode != "zscore":
            raise ValueError(f"unknown normal_reference: {mode}")
        if self.reference is None:
            raise RuntimeError("normal_reference='zscore' needs fit() or fit_reference() first")
        ref = self.reference
        z_e = np.clip((res.error - ref["mu_e"]) / ref["sd_e"], 0, None)
        z_r = np.clip((np.abs(res.residual) - ref["mu_r"]) / ref["sd_r"], 0, None) * np.sign(res.residual)
        return MMIEResult(r_avg=res.r_avg, error=z_e.astype(np.float32), residual=z_r.astype(np.float32))

    def anomaly_score(self, result: MMIEResult) -> np.ndarray:
        return anomaly_score(result.error, self.cfg.inference.score_reduce)

    def score(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        return self.anomaly_score(self.infer(X, seed))

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> float:
        labels = np.asarray(labels).astype(int)
        if self.cfg.threshold.method == "youden":
            self.threshold = youden_threshold(scores, labels)
        elif self.cfg.threshold.method == "sigma":
            self.threshold = sigma_threshold(np.asarray(scores)[labels == 0], self.cfg.threshold.k)
        else:
            raise ValueError(f"unknown threshold method: {self.cfg.threshold.method}")
        return self.threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("call calibrate() first")
        return (np.asarray(scores) > self.threshold).astype(int)

    def localize(self, result: MMIEResult) -> List[List[Peak]]:
        return localize_batch(result.residual, result.error, self.wavelengths, self.cfg.localization)

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        torch.save({"config": self.cfg.to_dict(), "wavelengths": self.wavelengths,
                    "state_dict": self.model.state_dict(), "threshold": self.threshold,
                    "reference": self.reference}, path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "SpecMultiMAEAD":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 1.13
            ckpt = torch.load(path, map_location="cpu")
        obj = cls(ckpt["wavelengths"], ExperimentConfig.from_dict(ckpt["config"]), device)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.threshold = ckpt.get("threshold")
        obj.reference = ckpt.get("reference")
        return obj
