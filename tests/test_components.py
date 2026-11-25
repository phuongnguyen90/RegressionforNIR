"""Unit tests for the independently testable components (run: ``pytest -q`` from ``code/``)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specmae.config import ExperimentConfig, LocalizationConfig, LossConfig, MaskConfig, ModelConfig  # noqa: E402
from specmae.data import make_synthetic, one_class_folds  # noqa: E402
from specmae.downstream import localize, youden_threshold  # noqa: E402
from specmae.embedding import FourierWavelengthEmbed, LinearWavelengthEmbed, SinusoidalPositionEmbed  # noqa: E402
from specmae.inference import mmie  # noqa: E402
from specmae.loss import compute_loss, gradient_consistency_loss, segment_reconstruction_loss  # noqa: E402
from specmae.masking import MaskSampler, cmsm_mask, random_mask, segments_from_mask  # noqa: E402
from specmae.metrics import interval_iou, localization_metrics  # noqa: E402
from specmae.model import SpecMultiMAE  # noqa: E402
from specmae.preprocessing import preprocess  # noqa: E402
from specmae.config import InferenceConfig, PreprocessConfig  # noqa: E402

TINY = ModelConfig(patch_size=8, embed_dim=32, fourier_frequencies=16, encoder_dim=32, encoder_depth=2,
                   encoder_heads=4, decoder_dim=16, decoder_depth=1, decoder_heads=4)


# ------------------------------------------------------------------ masking
@pytest.mark.parametrize("n", [2, 5, 20, 67, 200])
@pytest.mark.parametrize("ratio", [0.5, 0.75, 0.85])
def test_cmsm_exact_ratio_and_segments(n, ratio):
    rng = np.random.default_rng(0)
    for _ in range(50):
        pm = cmsm_mask(n, ratio, (4, 8), (8, 64), rng)
        target = min(max(int(np.floor(ratio * n)), 1), n - 1)
        assert pm.mask.sum() == target
        assert 1 <= len(pm.segments) <= 8
        assert segments_from_mask(pm.mask) == pm.segments  # non-overlapping, separated, contiguous
        assert sum(e - s for s, e in pm.segments) == target


def test_cmsm_respects_length_bounds_when_feasible():
    rng = np.random.default_rng(1)
    for _ in range(100):
        pm = cmsm_mask(400, 0.75, (4, 8), (8, 64), rng)
        lengths = [e - s for s, e in pm.segments]
        assert 4 <= len(lengths) <= 8
        assert min(lengths) >= 8
        assert max(lengths) <= 75  # hi is relaxed only when K * l_max < target


def test_random_mask_and_sampler_shapes():
    rng = np.random.default_rng(0)
    pm = random_mask(30, 0.75, rng)
    assert pm.mask.sum() == 22
    s = MaskSampler(MaskConfig(), 67)
    m, seg = s.sample(7, rng)
    assert m.shape == seg.shape == (7, 67)
    assert (m.sum(1) == s.n_masked).all()
    assert ((seg >= 0) == m).all()


def test_too_few_patches_raises():
    with pytest.raises(ValueError):
        MaskSampler(MaskConfig(), 1)


# ------------------------------------------------------------------ embeddings
def test_embedding_shapes():
    c = torch.linspace(0, 1, 10)
    assert FourierWavelengthEmbed(32).forward(c).shape == (10, 32)
    assert FourierWavelengthEmbed(32, num_frequencies=8).forward(c).shape == (10, 32)
    assert LinearWavelengthEmbed(32).forward(c).shape == (10, 32)
    assert SinusoidalPositionEmbed(32).forward(c).shape == (10, 32)


# ------------------------------------------------------------------ model
@pytest.mark.parametrize("kind", ["fourier", "linear", "sinusoidal"])
def test_model_shapes_with_padding(kind):
    cfg = ModelConfig(**{**TINY.__dict__, "wavelength_embed": kind})
    D = 101  # not divisible by P -> padding
    model = SpecMultiMAE(np.linspace(900, 1700, D), cfg)
    assert model.n_patches == 13
    s = MaskSampler(MaskConfig(k_min=2, k_max=3, l_min=1, l_max=4), model.n_patches)
    m, _ = s.sample(4, np.random.default_rng(0))
    out = model(torch.rand(4, D), torch.from_numpy(m))
    assert out.shape == (4, D)


def test_encoder_never_sees_masked_values():
    torch.manual_seed(0)
    D = 96
    model = SpecMultiMAE(np.linspace(900, 1700, D), TINY).eval()
    s = MaskSampler(MaskConfig(k_min=2, k_max=3, l_min=1, l_max=4), model.n_patches)
    m, _ = s.sample(3, np.random.default_rng(0))
    mask = torch.from_numpy(m)
    x = torch.rand(3, D)
    x2 = x.clone()
    pix = model.pixel_mask(mask)
    x2[pix] = torch.rand(int(pix.sum())) * 10  # change only masked values
    with torch.no_grad():
        assert torch.allclose(model(x, mask), model(x2, mask), atol=1e-5)


# ------------------------------------------------------------------ loss
def test_casrl_zero_for_perfect_and_positive_otherwise():
    t = torch.rand(2, 40)
    seg = torch.full((2, 40), -1, dtype=torch.long)
    seg[:, 5:10] = 0
    seg[:, 20:35] = 1
    loss, parts = compute_loss(t.clone(), t, seg, LossConfig())
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    p = t.clone()
    p[:, 0:3] += 1.0  # error only on visible positions not adjacent to a segment -> no loss
    assert compute_loss(p, t, seg, LossConfig())[0].item() == pytest.approx(0.0, abs=1e-12)
    p = t + 0.1
    loss, parts = compute_loss(p, t, seg, LossConfig())
    assert parts["grad"] == pytest.approx(0.0, abs=1e-10)  # constant offset keeps slopes
    assert parts["rec"] > 0


def test_segment_weights_emphasise_long_segments():
    t = torch.zeros(1, 30)
    seg = torch.full((1, 30), -1, dtype=torch.long)
    seg[0, 0:5] = 0     # short
    seg[0, 10:30] = 1   # long
    p_short = t.clone(); p_short[0, 0:5] = 1.0
    p_long = t.clone(); p_long[0, 10:30] = 1.0
    l_short = segment_reconstruction_loss(p_short, t, seg, beta=1.0)
    l_long = segment_reconstruction_loss(p_long, t, seg, beta=1.0)
    assert l_long > l_short
    # beta = 0 -> unweighted sum of per-segment MSEs
    assert segment_reconstruction_loss(p_short, t, seg, 0.0).item() == pytest.approx(1.0)


def test_gradient_loss_penalises_spikes():
    t = torch.zeros(1, 20)
    masked = torch.ones(1, 20, dtype=torch.bool)
    spike = t.clone(); spike[0, 10] = 1.0
    assert gradient_consistency_loss(spike, t, masked) > 0


# ------------------------------------------------------------------ inference
@pytest.mark.parametrize("agg", ["full", "masked"])
def test_mmie_shapes_and_consistency(agg):
    D = 64
    model = SpecMultiMAE(np.linspace(900, 1700, D), TINY)
    s = MaskSampler(MaskConfig(k_min=1, k_max=2, l_min=1, l_max=3), model.n_patches)
    x = np.random.default_rng(0).random((5, D)).astype(np.float32)
    res = mmie(model, x, s, InferenceConfig(ensemble_size=4, aggregation=agg, chunk_size=8),
               np.random.default_rng(0), torch.device("cpu"))
    assert res.r_avg.shape == res.error.shape == res.residual.shape == (5, D)
    assert np.allclose(res.residual, x - res.r_avg, atol=1e-6)
    assert (res.error >= 0).all()
    if agg == "full":  # E = mean_m (S - S^m)^2 >= (S - mean_m S^m)^2
        assert (res.error + 1e-6 >= res.residual ** 2).all()


# ------------------------------------------------------------------ downstream / metrics
def test_localize_finds_injected_peak():
    wl = np.linspace(900, 1700, 400)
    e = 0.001 + 0.5 * np.exp(-0.5 * ((wl - 1350) / 8) ** 2) + 0.2 * np.exp(-0.5 * ((wl - 1100) / 8) ** 2)
    r = np.sqrt(e)
    peaks = localize(r, e, wl, LocalizationConfig())
    assert abs(peaks[0].wavelength - 1350) < 5
    assert any(abs(p.wavelength - 1100) < 5 for p in peaks)
    assert peaks[0].left_nm < 1350 < peaks[0].right_nm
    m = localization_metrics([peaks], [[(1330, 1370)]], 10)
    assert m["top3_hit_rate"] == 1.0 and m["mean_peak_error_nm"] == 0.0 and m["iou"] > 0


def test_youden_and_iou():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1])
    t = youden_threshold(scores, labels)
    assert 0.3 <= t <= 0.8
    assert interval_iou((0, 10), (5, 15)) == pytest.approx(1 / 3)


# ------------------------------------------------------------------ data
def test_one_class_folds_protocol():
    d = make_synthetic(n_normal=100, n_anomalous=40, n_channels=64)
    seen_test_normals = []
    for f in one_class_folds(d.labels, 5, 0.1, seed=0):
        assert (d.labels[f.train] == 0).all()                        # train on normals only
        assert not set(f.train) & set(f.val) and not set(f.train) & set(f.test) and not set(f.val) & set(f.test)
        assert set(d.labels[f.val]) == {0, 1}
        seen_test_normals += list(f.test[d.labels[f.test] == 0]) + list(f.val[d.labels[f.val] == 0])
    assert sorted(seen_test_normals) == list(range(100))             # every normal held out exactly once


def test_group_folds_keep_groups_together():
    labels = np.zeros(50, int)
    groups = np.repeat(np.arange(10), 5)
    for f in one_class_folds(labels, 5, 0.1, groups=groups):
        assert not set(groups[f.train]) & set(groups[np.r_[f.val, f.test]])


def test_preprocess_range():
    d = make_synthetic(n_normal=10, n_anomalous=0, n_channels=128)
    X, wl = preprocess(d.X, d.wavelengths, PreprocessConfig(resample_points=32))
    assert X.shape == (10, 32) and wl.shape == (32,)
    assert np.allclose(X.min(1), 0) and np.allclose(X.max(1), 1)


# ------------------------------------------------------------------ review fixes
def test_numeric_labels_with_gaps_match_normal_values(tmp_path):
    import pandas as pd
    from specmae.config import DataConfig
    from specmae.data import load_table
    df = pd.DataFrame({"label": [0, 1, None, 0], "900": [1, 2, 3, 4], "910": [2, 3, 4, 5]})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)  # 'label' is read back as float: 0.0, 1.0, NaN, 0.0
    with pytest.warns(UserWarning):
        d = load_table(DataConfig(path=str(p), label_col="label", normal_values=["0"]))
    assert d.labels.tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        load_table(DataConfig(path=str(p), label_col="label", normal_values=["pure"]))


def test_clean_nonfinite_interpolates_drops():
    from specmae.data import SpectralData, clean_nonfinite
    X = np.tile(np.linspace(0, 1, 40), (3, 1))
    X[:, 0] = np.nan            # channel missing everywhere -> dropped
    X[0, 10] = np.nan           # single gap -> interpolated
    X[1, 5:30] = np.nan         # mostly missing -> spectrum dropped
    d = SpectralData(X, np.arange(40.0), np.array([0, 0, 1]))
    with pytest.warns(UserWarning):
        c = clean_nonfinite(d)
    assert c.X.shape == (2, 39) and np.isfinite(c.X).all()
    assert c.labels.tolist() == [0, 1]
    assert c.X[0, 9] == pytest.approx(10 / 39)


def test_casrl_normalize_weights_removes_k_scaling():
    t = torch.zeros(1, 40)
    p = torch.ones(1, 40)       # MSE = 1 on every segment
    seg2 = torch.full((1, 40), -1, dtype=torch.long); seg2[0, 0:10] = 0; seg2[0, 20:30] = 1
    seg4 = torch.full((1, 40), -1, dtype=torch.long)
    for k, s in enumerate(range(0, 40, 10)):
        seg4[0, s:s + 5] = k
    assert segment_reconstruction_loss(p, t, seg2).item() == pytest.approx(2.0)   # unnormalised: ~K * MSE
    assert segment_reconstruction_loss(p, t, seg4).item() == pytest.approx(4.0)
    assert segment_reconstruction_loss(p, t, seg2, normalize_weights=True).item() == pytest.approx(1.0)
    assert segment_reconstruction_loss(p, t, seg4, normalize_weights=True).item() == pytest.approx(1.0)


def test_resample_step_never_exceeds_range():
    wl = np.linspace(900, 1700, 801)
    X = np.random.default_rng(0).random((2, 801))
    Xr, grid = preprocess(X, wl, PreprocessConfig(resample_step_nm=0.1, savgol_window=0))
    assert grid.max() <= 1700 and grid.min() >= 900 and Xr.shape[1] == grid.size


def test_linear_wavelength_embedding_is_position_dependent_at_init():
    cfg = ModelConfig(**{**TINY.__dict__, "wavelength_embed": "linear"})
    model = SpecMultiMAE(np.linspace(900, 1700, 96), cfg)
    with torch.no_grad():
        w = model.enc_wl_embed(model.patch_coords)
    assert (w[0] - w[-1]).abs().max() > 1e-2
    assert (w[0] - w[len(w) // 2]).abs().max() > 1e-3


def test_small_folds_keep_normals_in_test_and_groups_together():
    from specmae.data import split_val_test
    labels = np.r_[np.zeros(10, int), np.ones(4, int)]
    for f in one_class_folds(labels, 5, 0.1, seed=0):   # held-out normal folds of size 2
        assert (labels[f.test] == 0).any() and (labels[f.test] == 1).any()
    groups = np.repeat(np.arange(6), 3)
    val, test = split_val_test(np.arange(18), 0.3, np.random.default_rng(0), groups)
    assert not set(groups[val]) & set(groups[test]) and len(test) > 0 and len(val) > 0
    with pytest.warns(UserWarning):
        val, test = split_val_test(np.array([7]), 0.1, np.random.default_rng(0))
    assert len(val) == 0 and test.tolist() == [7]


def test_config_roundtrip(tmp_path):
    cfg = ExperimentConfig()
    cfg.override("mask.strategy", "random")
    p = tmp_path / "c.json"
    cfg.save(p)
    assert ExperimentConfig.load(p).mask.strategy == "random"
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"bogus": 1})
