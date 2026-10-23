"""Tests for `concept_swap.families`: published families, null-set samplers, Fisher scores."""

import numpy as np

from concept_swap import families as Fm
from concept_swap.config import Config
from concept_swap.model import branch_slices


def test_published_families_and_branch_specialisation():
    """Families are read from the Distill JSONs; all 3b curve units lie in the 5x5 branch (Voss et al. 2021)."""
    fam = Fm.load_family(Config().data_dir, "mixed3b")
    assert fam.neurons == [379, 406, 385, 343, 342, 388, 324, 340, 330, 349] and fam.tuning.shape == (10, 720)
    s = branch_slices("mixed3b")["5x5"]
    assert all(s.start <= n < s.stop for n in fam.neurons)
    assert len(Fm.load_family(Config().data_dir, "mixed4a").neurons) == 14


def test_delta_matched_sets_match_histogram_and_exclude_family():
    rng = np.random.default_rng(0)
    delta = rng.exponential(size=480)
    family = [379, 406, 385, 343, 342, 388, 324, 340, 330, 349]
    sets = Fm.delta_matched_sets(delta, family, n_sets=50, rng=rng)
    edges = np.quantile(delta, np.linspace(0, 1, 11)[1:-1])
    hist = lambda idx: np.bincount(np.digitize(delta[idx], edges), minlength=10)
    assert sets.shape == (50, 10) and not np.isin(sets, family).any()
    assert all((hist(s) == hist(family)).all() and len(set(s)) == 10 for s in sets)


def test_rank_matched_sets_track_family_delta_in_the_tail():
    """Post-hoc null: each family unit is matched to one of its nearest non-family channels by delta, so the null's
    delta distribution follows the family even when the family sits at the extreme tail (where deciles cannot)."""
    rng = np.random.default_rng(0)
    delta = np.arange(480, dtype=float)  # channel c has delta c
    family = list(range(470, 480))  # the 10 largest deltas
    sets = Fm.rank_matched_sets(delta, family, n_sets=50, rng=rng, pool=10)
    assert sets.shape == (50, 10) and not np.isin(sets, family).any() and all(len(set(s)) == 10 for s in sets)
    assert sets.min() >= 450  # drawn from the 20 channels just below the family, not from the whole top decile
    assert list(Fm.top_nonfamily(delta, family, 3)) == [469, 468, 467]


def test_pool_sets_draws_without_replacement_from_pool():
    sets = Fm.pool_sets(np.arange(320, 416), size=10, n_sets=30, rng=np.random.default_rng(0))
    assert sets.shape == (30, 10) and sets.min() >= 320 and sets.max() < 416
    assert all(len(set(s)) == 10 for s in sets)


def test_fisher_score_ranks_informative_feature_first():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    X = rng.normal(size=(400, 5))
    X[:, 3] += 3 * y  # only feature 3 depends on the class
    assert np.argmax(Fm.fisher_scores(X, y)) == 3
