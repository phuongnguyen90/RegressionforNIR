"""Data containers, loaders, a synthetic generator and the one-class fold protocol.

Fold protocol:
normal spectra are split into K folds (GroupKFold when a physical-sample id is
given). For fold i: train = normals of the other folds; the held-out normal fold
and all anomalies are split into a labelled validation part (``val_frac`` of each,
used for threshold selection and early stopping on its normals) and the test part.
Anomalous spectra are never used for training.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .config import DataConfig

Band = Tuple[float, float]


@dataclass
class SpectralData:
    X: np.ndarray                       # (n, D) raw spectra
    wavelengths: np.ndarray             # (D,) nm
    labels: np.ndarray                  # (n,) 0 = normal, 1 = anomalous
    groups: Optional[np.ndarray] = None  # (n,) physical-sample ids
    bands: Optional[List[Optional[List[Band]]]] = None  # per-sample ground-truth absorption bands
    meta: Optional[pd.DataFrame] = None
    name: str = "data"

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, idx: np.ndarray, name: Optional[str] = None) -> "SpectralData":
        idx = np.asarray(idx)
        return SpectralData(
            X=self.X[idx], wavelengths=self.wavelengths, labels=self.labels[idx],
            groups=None if self.groups is None else self.groups[idx],
            bands=None if self.bands is None else [self.bands[i] for i in idx],
            meta=None if self.meta is None else self.meta.iloc[idx].reset_index(drop=True),
            name=name or self.name)


# ---------------------------------------------------------------- loading
def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _load_bands(path: Optional[str]) -> Dict[str, List[Band]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    table = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        table[k.strip().lower()] = [tuple(map(float, b)) for b in v]
    return table


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {path}")


def load_table(cfg: DataConfig) -> SpectralData:
    path = Path(cfg.path)
    df = _read_table(path)
    prefix = cfg.wavelength_prefix
    if cfg.wavelength_cols:
        wl_cols = list(cfg.wavelength_cols)
    else:
        wl_cols = [c for c in df.columns
                   if str(c).startswith(prefix) and _to_float(str(c)[len(prefix):]) is not None]
    if len(wl_cols) < 2:
        raise ValueError(f"{path}: could not find wavelength columns (got {wl_cols[:5]})")
    wl = np.array([float(str(c)[len(prefix):]) for c in wl_cols])
    order = np.argsort(wl)
    X = df[wl_cols].to_numpy(dtype=np.float64)[:, order]
    wl = wl[order]

    if cfg.label_col:
        col = df[cfg.label_col]
        missing = col.isna().to_numpy()
        if missing.any():
            warnings.warn(f"{path}: dropping {int(missing.sum())} rows with an empty '{cfg.label_col}' label")
            df = df.loc[~missing].reset_index(drop=True)
            X = X[~missing]
            col = df[cfg.label_col]
        normal = col.astype(str).str.strip().isin([str(v).strip() for v in cfg.normal_values])
        # numeric labels: 0 must match 0.0 (pandas reads int columns with gaps as float)
        num_values = [f for f in (_to_float(str(v)) for v in cfg.normal_values) if f is not None]
        if num_values:
            normal |= pd.to_numeric(col, errors="coerce").isin(num_values)
        if not normal.any():
            raise ValueError(f"{path}: no spectrum has '{cfg.label_col}' in normal_values={cfg.normal_values}; "
                             f"values found: {sorted(map(str, col.unique()))[:10]}")
        labels = (~normal).to_numpy().astype(int)
    elif cfg.anomaly_cols:
        labels = (df[cfg.anomaly_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy() > 0) \
            .any(axis=1).astype(int)
    else:
        raise ValueError("set data.label_col or data.anomaly_cols")

    groups = df[cfg.group_col].to_numpy() if cfg.group_col else None
    table = _load_bands(cfg.bands_file)
    bands: Optional[List[Optional[List[Band]]]] = None
    if table:
        bands = []
        present = None
        if cfg.anomaly_cols:
            present = df[cfg.anomaly_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy() > 0
        for i in range(len(df)):
            if labels[i] == 0:
                bands.append(None)
                continue
            if present is not None:
                names = [c for c, p in zip(cfg.anomaly_cols, present[i]) if p]
            elif cfg.contaminant_col:
                names = [s for s in str(df[cfg.contaminant_col].iloc[i]).split(cfg.contaminant_sep) if s.strip()]
            else:
                names = []
            b = [band for n in names for band in table.get(str(n).strip().lower(), [])]
            bands.append(b or None)

    meta = df.drop(columns=wl_cols).reset_index(drop=True)
    return SpectralData(X, wl, labels, groups, bands, meta, name=path.stem)


def load_npz(cfg: DataConfig) -> SpectralData:
    z = np.load(cfg.path, allow_pickle=True)
    return SpectralData(
        X=z["X"].astype(np.float64), wavelengths=z["wavelengths"].astype(np.float64),
        labels=z["labels"].astype(int), groups=z["groups"] if "groups" in z else None,
        bands=list(z["bands"]) if "bands" in z else None, name=Path(cfg.path).stem)


def load_data(cfg: DataConfig) -> List[SpectralData]:
    """Returns one ``SpectralData`` per evaluation unit (subset / split value)."""
    if cfg.subsets:
        out = []
        for sub in cfg.subsets:
            sub = dict(sub)
            name = sub.pop("name", None)
            child = replace(cfg, subsets=[], **sub)
            for d in load_data(child):
                if name:
                    d.name = name
                out.append(d)
        return out

    if cfg.source == "synthetic":
        data = make_synthetic(**cfg.synthetic)
    elif cfg.source == "npz":
        data = load_npz(cfg)
    elif cfg.source == "table":
        data = load_table(cfg)
    else:
        raise ValueError(f"unknown data source: {cfg.source}")

    if cfg.split_col:
        if data.meta is None or cfg.split_col not in data.meta:
            raise KeyError(f"split column {cfg.split_col!r} not found")
        out = []
        for value in pd.unique(data.meta[cfg.split_col]):
            idx = np.flatnonzero((data.meta[cfg.split_col] == value).to_numpy())
            out.append(clean_nonfinite(data.subset(idx, name=f"{data.name}[{value}]")))
        return out
    return [clean_nonfinite(data)]


def clean_nonfinite(data: SpectralData, max_row_gap_frac: float = 0.05) -> SpectralData:
    """Remove NaN/inf so they never reach preprocessing or training.

    1. channels that are non-finite in every spectrum are dropped (e.g. outside an
       instrument's wavelength coverage);
    2. spectra with a few non-finite channels (<= ``max_row_gap_frac``) are
       linearly interpolated along wavelength;
    3. spectra with more missing channels are dropped.
    """
    X = np.asarray(data.X, dtype=np.float64)
    bad = ~np.isfinite(X)
    if not bad.any():
        return data
    keep_ch = ~bad.all(axis=0)
    if (~keep_ch).any():
        warnings.warn(f"[{data.name}] dropping {int((~keep_ch).sum())} channels with no finite values")
    X, wl = X[:, keep_ch], data.wavelengths[keep_ch]
    bad = ~np.isfinite(X)
    n_bad = bad.sum(axis=1)
    drop = n_bad > max_row_gap_frac * X.shape[1]
    for i in np.flatnonzero((n_bad > 0) & ~drop):
        ok = ~bad[i]
        X[i, ~ok] = np.interp(wl[~ok], wl[ok], X[i, ok])
    n_fixed = int(((n_bad > 0) & ~drop).sum())
    if n_fixed:
        warnings.warn(f"[{data.name}] interpolated missing channels in {n_fixed} spectra")
    if drop.any():
        warnings.warn(f"[{data.name}] dropping {int(drop.sum())} spectra with > {max_row_gap_frac:.0%} missing channels")
    out = data.subset(np.flatnonzero(~drop))
    out.X = X[~drop]
    out.wavelengths = wl
    return out


# ---------------------------------------------------------------- synthetic
def make_synthetic(n_normal: int = 800, n_anomalous: int = 400, n_channels: int = 512,
                   wl_min: float = 900.0, wl_max: float = 1700.0,
                   contaminants: Optional[Dict[str, Tuple[float, float]]] = None,
                   amp_range: Tuple[float, float] = (0.08, 0.20), noise: float = 0.003,
                   seed: int = 0) -> SpectralData:
    """Smooth NIR-like spectra; anomalies add a Gaussian absorption band of a 'contaminant'.

    Ground-truth band of a contaminant with centre c and width s is [c - 2s, c + 2s].
    """
    rng = np.random.default_rng(seed)
    contaminants = contaminants or {"A": (1100.0, 12.0), "B": (1350.0, 12.0), "C": (1600.0, 12.0)}
    wl = np.linspace(wl_min, wl_max, n_channels)
    span = wl_max - wl_min
    base = [(wl_min + 0.10 * span, 0.08 * span), (wl_min + 0.45 * span, 0.06 * span),
            (wl_min + 0.70 * span, 0.10 * span)]

    def g(c, s):
        return np.exp(-0.5 * ((wl - c) / s) ** 2)

    n = n_normal + n_anomalous
    X = np.empty((n, n_channels))
    for i in range(n):
        y = rng.uniform(0.2, 0.4) + rng.uniform(-0.1, 0.1) * (wl - wl_min) / span
        for c, s in base:
            y += rng.uniform(0.4, 0.9) * g(c + rng.normal(0, 0.005 * span), s * rng.uniform(0.9, 1.1))
        X[i] = y + rng.normal(0, noise, n_channels)

    labels = np.r_[np.zeros(n_normal, int), np.ones(n_anomalous, int)]
    names = list(contaminants)
    which = [""] * n_normal
    bands: List[Optional[List[Band]]] = [None] * n_normal
    for i in range(n_normal, n):
        name = names[int(rng.integers(len(names)))]
        c, s = contaminants[name]
        X[i] += rng.uniform(*amp_range) * g(c, s)
        which.append(name)
        bands.append([(c - 2 * s, c + 2 * s)])
    meta = pd.DataFrame({"contaminant": which})
    return SpectralData(X, wl, labels, None, bands, meta, name="synthetic")


# ---------------------------------------------------------------- folds
@dataclass
class Fold:
    index: int
    train: np.ndarray       # normals only
    val: np.ndarray         # labelled: held-out normals + anomalies (threshold selection)
    test: np.ndarray        # held-out normals + anomalies


def one_class_folds(labels: np.ndarray, n_splits: int = 5, val_frac: float = 0.1,
                    groups: Optional[np.ndarray] = None, seed: int = 0) -> Iterator[Fold]:
    labels = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    anom = np.flatnonzero(labels == 1)
    if groups is not None:
        splits = GroupKFold(n_splits=n_splits).split(normal, groups=np.asarray(groups)[normal])
    else:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(normal)

    g = None if groups is None else np.asarray(groups)
    for i, (tr, ho) in enumerate(splits):
        val_n, test_n = split_val_test(normal[ho], val_frac, rng, g, what=f"fold {i} normals")
        val_a, test_a = split_val_test(anom, val_frac, rng, g, what=f"fold {i} anomalies")
        yield Fold(
            index=i,
            train=np.sort(normal[tr]),
            val=np.sort(np.r_[val_n, val_a]).astype(np.int64),
            test=np.sort(np.r_[test_n, test_a]).astype(np.int64),
        )


def split_val_test(idx: np.ndarray, frac: float, rng: np.random.Generator,
                   groups: Optional[np.ndarray] = None, what: str = "samples") -> Tuple[np.ndarray, np.ndarray]:
    """Send about ``frac`` of ``idx`` to validation and the rest to test.

    Validation gets at least one unit and test always keeps at least one unit. A
    unit is one spectrum, or one physical sample (group) when ``groups`` is given,
    so scans of the same item never end up on both sides.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx, idx
    if groups is None:
        units = [np.array([j]) for j in rng.permutation(idx)]
    else:
        g = groups[idx]
        units = [idx[g == u] for u in rng.permutation(np.unique(g))]
    if len(units) < 2:
        warnings.warn(f"{what}: only one sample/group, so it goes to the test split and validation gets none")
        return idx[:0], idx
    want = max(1, int(round(frac * idx.size)))
    val, count = [], 0
    for u in units[:-1]:
        if count >= want:
            break
        val.append(u)
        count += u.size
    val_idx = np.concatenate(val)
    return val_idx, np.setdiff1d(idx, val_idx)
