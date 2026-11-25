"""Masking strategies.

Masks are defined on *patches* (the encoder drops whole patches); the wavelength
mask m in {0,1}^D is the patch mask expanded by the patch size.

``cmsm_mask`` -- Contiguous Multi-Segment Masking: hides K contiguous bands of
patches. The sampler always terminates and keeps batches rectangular:

* exactly ``floor(r*N)`` patches are masked (clamped to [1, N-1]), so every
  spectrum in a batch has the same number of visible tokens;
* exactly K segments are produced (K ~ U{k_min..k_max}, reduced when infeasible),
  with segment lengths drawn within [l_min, l_max] (bounds shrunk to fit N) and
  separated by >=1 visible patch.

``random_mask`` -- standard scattered MAE masking at the same ratio (ablation
"w/o CMSM"); its segments are the maximal runs of masked patches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import MaskConfig

Segment = Tuple[int, int]  # half-open [start, end) in patch indices


@dataclass
class PatchMask:
    mask: np.ndarray            # (N,) bool, True = masked
    segments: List[Segment]


def target_masked_count(n_patches: int, ratio: float) -> int:
    if n_patches < 2:
        raise ValueError(
            f"Spectrum has only {n_patches} patch(es); masking needs >= 2. "
            "Use a smaller patch size (model.patch_size) or resample to more points.")
    return int(min(max(math.floor(ratio * n_patches), 1), n_patches - 1))


def _bounded_composition(total: int, k: int, lo: int, hi: int, rng: np.random.Generator) -> np.ndarray:
    """Random k-part composition of ``total`` with every part in [lo, hi]."""
    lengths = np.full(k, lo, dtype=np.int64)
    rem = total - k * lo
    while rem > 0:
        open_parts = np.flatnonzero(lengths < hi)
        i = int(rng.choice(open_parts))
        add = int(rng.integers(1, min(hi - lengths[i], rem) + 1))
        lengths[i] += add
        rem -= add
    return lengths


def segments_from_mask(mask: np.ndarray) -> List[Segment]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def cmsm_mask(n_patches: int, ratio: float, k_range: Tuple[int, int], l_range: Tuple[int, int],
              rng: np.random.Generator, require_gap: bool = True) -> PatchMask:
    n = n_patches
    target = target_masked_count(n, ratio)
    n_vis = n - target
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    k_cap = min(target, n_vis + 1) if require_gap else target
    k = max(1, min(k, k_cap))

    lo = max(1, min(l_range[0], target // k))
    hi = max(lo, min(l_range[1], target))
    if k * hi < target:
        hi = math.ceil(target / k)
    lengths = _bounded_composition(target, k, lo, hi, rng)
    rng.shuffle(lengths)

    min_gap = 1 if require_gap else 0
    free = n_vis - min_gap * (k - 1)
    gaps = rng.multinomial(free, np.full(k + 1, 1.0 / (k + 1)))
    gaps[1:-1] += min_gap

    mask = np.zeros(n, dtype=bool)
    segments: List[Segment] = []
    pos = int(gaps[0])
    for i in range(k):
        s, e = pos, pos + int(lengths[i])
        mask[s:e] = True
        segments.append((s, e))
        pos = e + int(gaps[i + 1])
    if not require_gap:  # adjacent segments merge into one band
        segments = segments_from_mask(mask)
    return PatchMask(mask, segments)


def random_mask(n_patches: int, ratio: float, rng: np.random.Generator) -> PatchMask:
    target = target_masked_count(n_patches, ratio)
    mask = np.zeros(n_patches, dtype=bool)
    mask[rng.choice(n_patches, size=target, replace=False)] = True
    return PatchMask(mask, segments_from_mask(mask))


class MaskSampler:
    """Draws batches of patch masks plus per-patch segment ids (-1 = visible)."""

    def __init__(self, cfg: MaskConfig, n_patches: int):
        self.cfg = cfg
        self.n_patches = n_patches
        self.n_masked = target_masked_count(n_patches, cfg.ratio)

    def sample_one(self, rng: np.random.Generator) -> PatchMask:
        c = self.cfg
        if c.strategy == "cmsm":
            return cmsm_mask(self.n_patches, c.ratio, (c.k_min, c.k_max), (c.l_min, c.l_max),
                             rng, c.require_gap)
        if c.strategy == "random":
            return random_mask(self.n_patches, c.ratio, rng)
        raise ValueError(f"unknown masking strategy: {c.strategy}")

    def sample(self, batch: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        masks = np.zeros((batch, self.n_patches), dtype=bool)
        seg_ids = np.full((batch, self.n_patches), -1, dtype=np.int64)
        for b in range(batch):
            pm = self.sample_one(rng)
            masks[b] = pm.mask
            for k, (s, e) in enumerate(pm.segments):
                seg_ids[b, s:e] = k
        return masks, seg_ids
