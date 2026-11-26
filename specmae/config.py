"""Configuration dataclasses for SpecMultiMAE-AD.

Defaults (P=32, d=512, Fourier frequencies=256, encoder 12x8 heads x 512,
decoder 4x8 heads x 256, r=0.75, K=4-8, l in [8, 64] patches, lambda=0.5,
alpha=0.5, M=64, AdamW 1e-4 / wd 0.05, cosine + 10 warm-up epochs, batch 256,
<=800 epochs, early-stopping patience 50).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    patch_size: int = 32                 # P
    embed_dim: int = 512                 # d (content and wavelength embeddings each; token = 2d)
    wavelength_embed: str = "fourier"    # "fourier" | "linear" | "sinusoidal" (ablation: index-based PE)
    fourier_frequencies: int = 256       # number of learnable frequencies (d/2 by default)
    fourier_sigma: float = 10.0          # std of the initial frequencies (cycles per unit of normalised wavelength)
    wl_ref_range: List[float] = field(default_factory=lambda: [300.0, 2700.0])  # nm -> [0,1] normalisation, fixed for all instruments
    encoder_dim: int = 512               # d_e
    encoder_depth: int = 12              # L_e
    encoder_heads: int = 8
    decoder_dim: int = 256               # d_d
    decoder_depth: int = 4               # L_d
    decoder_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class MaskConfig:
    strategy: str = "cmsm"   # "cmsm" | "random" (ablation: scattered patch masking)
    ratio: float = 0.75      # r
    k_min: int = 4           # number of segments K ~ U{k_min..k_max}
    k_max: int = 8
    l_min: int = 8           # segment length bounds, in patches
    l_max: int = 64
    require_gap: bool = True  # keep >=1 visible patch between segments so the K segments stay distinct


@dataclass
class LossConfig:
    name: str = "casrl"       # "casrl" | "mse" (ablation)
    grad_weight: float = 0.5  # lambda in L = L_rec + lambda * L_grad
    beta: float = 1.0         # segment-length exponent in w_k = (l_k / mean l)^beta (beta=1: weight proportional to length)
    normalize_weights: bool = False  # False = sum_k w_k*MSE_k (scales with K); True = divide by sum_k w_k


@dataclass
class TrainConfig:
    epochs: int = 800
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    patience: int = 50        # early stopping on validation reconstruction MSE
    grad_clip: float = 1.0    # 0 disables
    amp: bool = True          # mixed precision on CUDA
    num_workers: int = 0
    device: str = "auto"
    log_every: int = 10
    val_batch_size: int = 1024


@dataclass
class InferenceConfig:
    ensemble_size: int = 64      # M
    aggregation: str = "full"    # "full": average full reconstructions; "masked": per wavelength, average only members in which it was masked
    error_map: str = "mse"       # "mse" (default) | "variance"
    chunk_size: int = 4096       # max forward passes per GPU batch (spectra x M)
    # Optional calibration (off by default): "zscore" standardises E and |R| per wavelength
    # with statistics of normal training spectra before scoring and localisation.
    normal_reference: str = "none"  # "none" (raw maps) | "zscore"
    reference_size: int = 1000       # normal training spectra used for the reference statistics
    score_reduce: str = "mean"       # "mean" (A = mean_l E) | "max"


@dataclass
class ThresholdConfig:
    method: str = "youden"   # "youden" | "sigma" (tau = mu + k*sigma on validation normals)
    k: float = 3.0


@dataclass
class LocalizationConfig:
    alpha: float = 0.5               # P = alpha*|R| + (1-alpha)*E
    component_norm: str = "none"     # "none" | "max" (scale |R| and E to [0,1] per spectrum before mixing)
    min_separation_nm: float = 10.0
    min_height: float = 0.10         # on the max-normalised prominence score
    min_prominence: float = 0.08
    top_k: int = 10
    tolerance_nm: float = 15.0       # +/- window for peak matching metrics


@dataclass
class PreprocessConfig:
    resample_points: Optional[int] = None    # uniform grid with this many points (e.g. 32 for MSMI-NIR)
    resample_step_nm: Optional[float] = None # or uniform grid with this step
    savgol_window: int = 15                  # 0 disables
    savgol_order: int = 2
    snv: bool = True
    minmax: bool = True


@dataclass
class DataConfig:
    source: str = "table"            # "table" (csv/xlsx/parquet) | "npz" | "synthetic"
    path: str = ""
    label_col: Optional[str] = None  # column whose values in `normal_values` mark normal spectra
    normal_values: List[Any] = field(default_factory=lambda: [0])
    anomaly_cols: List[str] = field(default_factory=list)  # alternative: anomalous if any of these columns > 0 (e.g. pesticide concentrations)
    contaminant_col: Optional[str] = None   # alternative source of contaminant names for band lookup
    contaminant_sep: str = ","
    group_col: Optional[str] = None  # physical-sample id: folds never split a group
    split_col: Optional[str] = None  # train/evaluate a separate model per value (e.g. instrument for FDS-NIR)
    wavelength_cols: Optional[List[str]] = None  # default: every column whose name parses as a number
    wavelength_prefix: str = ""
    bands_file: Optional[str] = None  # JSON {contaminant: [[lo_nm, hi_nm], ...]} with ground-truth absorption bands
    subsets: List[Dict[str, Any]] = field(default_factory=list)  # list of DataConfig overrides, each loaded and evaluated separately
    synthetic: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "specmae"
    seed: int = 0
    runs: int = 1          # independent runs (seeds) per fold
    n_folds: int = 5
    val_frac: float = 0.1  # fraction of held-out normal fold and of anomalies used for threshold selection
    output_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def override(self, dotted: str, value: Any) -> None:
        """Set e.g. ``override("mask.strategy", "random")``."""
        obj: Any = self
        *parents, last = dotted.split(".")
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, last):
            raise KeyError(f"unknown config key: {dotted}")
        setattr(obj, last, value)


def _from_dict(cls, d: Dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    defaults = cls()
    for name, value in d.items():
        current = getattr(defaults, name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(current), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
