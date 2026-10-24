"""Tests for `concept_swap.stats`: pigeonhole bootstrap, permutation p-values, decision rule, Holm."""

import numpy as np

from concept_swap import stats as St


def test_pigeonhole_weights_are_products_of_counts():
    base, src = np.array([0, 0, 1, 2]), np.array([3, 4, 3, 5])
    W = St.pigeonhole_weights(base, src, B=200, rng=np.random.default_rng(0))
    assert W.shape == (200, 4) and np.allclose(W.mean(0), 1, atol=0.25)  # E[count]=1 for each factor


def test_cluster_weights_share_one_count_per_cluster():
    """Every element of a cluster gets that cluster's draw count, and the counts of the distinct clusters sum to their
    number; the pigeonhole weights are the product of a base and a source cluster draw (same RNG order)."""
    ids = np.array([0, 0, 1, 2, 2, 2])
    W = St.cluster_weights(ids, B=100, rng=np.random.default_rng(0))
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 5]).all() and (W[:, [0, 2, 3]].sum(1) == 3).all()
    base, src, rng = np.array([0, 0, 1, 2]), np.array([3, 4, 3, 5]), np.random.default_rng(1)
    expected = St.cluster_weights(base, 50, rng) * St.cluster_weights(src, 50, rng)
    np.testing.assert_array_equal(St.pigeonhole_weights(base, src, 50, np.random.default_rng(1)), expected)


def test_bootstrap_ci_covers_true_rate_for_independent_pairs():
    rng = np.random.default_rng(0)
    succ = rng.random(4000) < 0.3
    ids = np.arange(4000)
    W = St.pigeonhole_weights(ids, ids + 4000, B=500, rng=rng)
    lo, hi = St.percentile_ci(St.weighted_rates(succ[None], W)[:, 0])
    assert lo < 0.3 < hi and hi - lo < 0.06


def test_random_rows_is_the_seeded_sorted_draw_inside_the_mask():
    """The example-selection rule: k distinct rows where `mask` holds, drawn with default_rng(seed), sorted (the
    same rule that picked the README's six H1 successes)."""
    mask = np.random.default_rng(1).random(500) < 0.2
    rows = St.random_rows(mask, 6, seed=0)
    expected = np.sort(np.random.default_rng(0).choice(np.flatnonzero(mask), 6, replace=False))
    np.testing.assert_array_equal(rows, expected)
    assert mask[rows].all() and (np.diff(rows) > 0).all()


def test_phipson_smyth_p():
    assert St.perm_p(1.0, np.zeros(99)) == 0.01 and St.perm_p(0.0, np.ones(99)) == 1.0


def test_h1_decision_rule():
    rule = dict(min_rate=0.5, sesoi=0.10, falsify_rate=0.2, falsify_margin=0.05)
    assert St.decide(0.8, 0.3, (0.4, 0.6), (0.3, 0.6), **rule) == "supported"
    assert St.decide(0.15, 0.1, (0.0, 0.1), (0.0, 0.1), **rule) == "falsified"
    assert St.decide(0.4, 0.3, (-0.02, 0.2), (0.0, 0.2), **rule) == "inconclusive"


def test_holm():
    p = St.holm([0.01, 0.04, 0.03])
    assert np.allclose(p, [0.03, 0.06, 0.06])
