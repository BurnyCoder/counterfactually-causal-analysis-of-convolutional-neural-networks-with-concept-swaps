"""Tests for `concept_swap.stimuli`: arc/line geometry, grids and minimal pairs."""

import numpy as np

from concept_swap.config import Config
from concept_swap import stimuli as S


def _centroid(mask: np.ndarray) -> np.ndarray:
    """(x, y) centre of mass of a stroke mask."""
    ys, xs = np.nonzero(mask > 0.5)
    return np.array([xs.mean(), ys.mean()])


def test_arc_passes_through_center_and_bends_toward_theta():
    """The arc midpoint sits on the RF centre and the arc bulges away from direction theta (its centre of
    curvature lies along theta), so the stroke centroid is displaced from the midpoint along u(theta)."""
    for theta in (0, 90, 135, 270):
        m = S.stroke_mask(S.Stim("arc", theta, 40, 3, 0, 0, 0), span_deg=120)
        assert m[S.CENTER, S.CENTER] > 0.5
        d = _centroid(m) - S.CENTER
        u = np.array([np.cos(np.radians(theta)), -np.sin(np.radians(theta))])  # (x, y_down) unit vector
        assert np.dot(d / np.linalg.norm(d), u) > 0.95


def test_line_is_centered_and_perpendicular_to_theta():
    """A line stimulus is centred on the RF centre and runs along the arc's tangent (perpendicular to theta)."""
    m = S.stroke_mask(S.Stim("line", 30, 40, 3, 0, 0, 0), span_deg=120)
    assert np.allclose(_centroid(m), S.CENTER, atol=1.0)
    ys, xs = np.nonzero(m > 0.5)
    direction = np.linalg.svd(np.stack([xs, ys], 1) - _centroid(m))[2][0]  # principal axis of the stroke
    u = np.array([np.cos(np.radians(30)), -np.sin(np.radians(30))])
    assert abs(np.dot(direction, u)) < 0.05


def test_compose_polarity_and_range():
    """Polarity 0 = black stroke on white, 1 = white stroke on black; values stay in [0, 1]."""
    m = S.stroke_mask(S.Stim("arc", 0, 40, 5, 0, 0, 0), span_deg=120)
    dark, light = S.compose(m, 0), S.compose(m, 1)
    assert dark[0, 0].tolist() == [1, 1, 1] and light[0, 0].tolist() == [0, 0, 0]
    assert dark[S.CENTER, S.CENTER].max() < 0.2 and light[S.CENTER, S.CENTER].min() > 0.8
    assert dark.min() >= 0 and dark.max() <= 1


def test_grid_and_minimal_pairs():
    """Grid has n_theta x cells rows; pairs share their nuisance cell and differ by >= min_dtheta."""
    cfg = Config(n_theta=24, radii=(20, 40), widths=(3,), sources_per_base=3, min_dtheta=30)
    g = S.grid(cfg, seed=1)
    assert len(g) == 24 * 2 * 1 * 2 and g.groupby("cell")[["dy", "dx"]].nunique().max().max() == 1
    p = S.minimal_pairs(g, cfg, seed=1)
    assert len(p) == len(g) * 3
    assert (g.loc[p.base, "cell"].values == g.loc[p.source, "cell"].values).all()
    assert (np.abs(p.dtheta) >= 30).all() and (p.base != p.source).all()


def test_nuisance_swap_pairs_change_only_one_factor():
    """Each stimulus's partner has the same orientation and differs only in the requested nuisance factor; the
    mapping is a permutation of the rows (cyclic over the factor's levels)."""
    g = S.grid(Config(n_theta=24, radii=(20, 40, 60), widths=(3, 5)), seed=1)
    for factor in ("radius", "width", "polarity"):
        j = S.nuisance_swap_pairs(g, factor)
        assert sorted(j.tolist()) == list(range(len(g)))
        a, b = g, g.iloc[j]
        assert (a.theta.values == b.theta.values).all() and (a[factor].values != b[factor].values).all()
        for other in {"radius", "width", "polarity"} - {factor}:
            assert (a[other].values == b[other].values).all()


def test_circ_diff():
    assert S.circ_diff(10, 350) == 20 and S.circ_diff(350, 10) == -20 and abs(S.circ_diff(0, 180)) == 180
