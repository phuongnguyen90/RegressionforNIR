"""Kiểm tra nhanh các thành phần cốt lõi: python -m pytest -q tests"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masking.cmsm import cmsm_patch_mask  # noqa: E402
from models.casrl import casrl_loss  # noqa: E402
from models.specmultimae import SpecMultiMAE  # noqa: E402
from data.preprocess import phi  # noqa: E402


def test_cmsm_ratio_and_no_overlap():
    rng = np.random.default_rng(0)
    for N in [8, 67, 200]:
        for _ in range(50):
            m, segs = cmsm_patch_mask(N, 0.75, 8, 64, (4, 8), rng)
            assert m.sum() == max(1, int(0.75 * N))
            cover = np.zeros(N, int)
            for s, e in segs:
                cover[s:e] += 1
            assert cover.max() == 1 and (cover.astype(bool) == m).all()


def test_forward_and_loss_shapes():
    torch.manual_seed(0)
    model = SpecMultiMAE(
        P=16,
        d=32,
        d_enc=32,
        enc_depth=1,
        enc_heads=4,
        d_dec=16,
        dec_depth=1,
        dec_heads=4,
        n_freq=16,
    )
    B, D = 4, 150
    x = torch.rand(B, D)
    wl = torch.linspace(900, 2500, D)
    N = int(np.ceil(D / 16))
    rng = np.random.default_rng(1)
    ms = [cmsm_patch_mask(N, 0.75, 2, 4, (2, 3), rng) for _ in range(B)]
    mask = torch.as_tensor(np.stack([m for m, _ in ms]))
    segs = [s for _, s in ms]
    xh = model(x, wl, mask)
    assert xh.shape == (B, D)
    loss = casrl_loss(x, xh, mask, segs, 16)
    loss.backward()
    assert torch.isfinite(loss)


def test_phi_range():
    X = np.random.rand(5, 100) + np.linspace(0, 1, 100)
    Xp, grid = phi(X, np.linspace(1000, 2000, 100))
    assert (
        np.allclose(Xp.min(1), 0) and np.allclose(Xp.max(1), 1) and len(grid) == 100
    )
