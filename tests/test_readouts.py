"""Tests for `concept_swap.readouts`: z-scoring, template/ridge orientation decoders, normalized effect."""

import numpy as np

from concept_swap import readouts as R


def _tuned(theta: np.ndarray, prefs: np.ndarray, noise: float, rng) -> np.ndarray:
    """Von-Mises-like population response to orientations `theta` (deg) of units with preferred angles `prefs`."""
    d = np.radians(theta[:, None] - prefs[None])
    return 10 * np.exp(2 * (np.cos(d) - 1)) + noise * rng.normal(size=d.shape)


def test_template_decoder_recovers_orientation_per_cell():
    rng = np.random.default_rng(0)
    theta = np.tile(np.arange(72) * 5.0, 2)
    cell = np.repeat([0, 1], 72)
    prefs = np.linspace(0, 360, 14, endpoint=False)
    Z = _tuned(theta, prefs, 0.05, rng) * np.where(cell[:, None] == 1, 2.0, 1.0)  # cell 1 has a different gain
    dec = R.TemplateDecoder().fit(Z, theta, cell)
    Zt = _tuned(theta, prefs, 0.05, rng) * np.where(cell[:, None] == 1, 2.0, 1.0)
    assert (np.abs(R.circ_dist(dec.predict(Zt, cell), theta)) <= 10).mean() > 0.95


def test_template_correlations_are_what_predict_maximises():
    """`correlations` exposes the per-template Pearson r of one cell: a template correlates 1 with itself, and the
    argmax over templates is exactly the label `predict` returns."""
    rng = np.random.default_rng(0)
    theta, cell = np.arange(72) * 5.0, np.zeros(72, int)
    Z = _tuned(theta, np.linspace(0, 360, 14, endpoint=False), 0.05, rng)
    dec = R.TemplateDecoder().fit(Z, theta, cell)
    th, r = dec.correlations(Z, 0)
    np.testing.assert_allclose(np.diag(r), 1, atol=1e-9)
    np.testing.assert_array_equal(th[r.argmax(1)], dec.predict(Z, cell))


def test_ridge_decoder_recovers_orientation():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 360, 2000)
    X = _tuned(theta, np.linspace(0, 360, 40, endpoint=False), 0.1, rng)
    dec = R.RidgeDecoder().fit(X[:1000], theta[:1000])
    assert np.median(np.abs(R.circ_dist(dec.predict(X[1000:]), theta[1000:]))) < 5


def test_zscore_roundtrip():
    X = np.random.default_rng(0).normal(5, 3, size=(100, 4))
    z = R.ZScore().fit(X)
    assert np.allclose(z(X).mean(0), 0) and np.allclose(z(X).std(0), 1)


def test_normalized_effect_extremes_and_denominator_filter():
    rng = np.random.default_rng(0)
    rb, rs = rng.normal(size=(50, 6)), rng.normal(size=(50, 6))
    assert np.isclose(R.normalized_effect(rb, rb, rs), 0) and np.isclose(R.normalized_effect(rs, rb, rs), 1)
    assert np.isclose(R.normalized_effect((rb + rs) / 2, rb, rs), 0.5)
    keep = R.denominator_mask(rb, rs, q=0.1)
    assert keep.sum() == 45  # the 10% of pairs with the smallest |rs - rb|^2 are dropped
