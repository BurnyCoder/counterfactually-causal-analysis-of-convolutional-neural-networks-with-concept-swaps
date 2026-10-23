"""Neuron families under test and the control neuron sets they are compared against.

Local: loads the Circuits authors' published curve families (data/circuits/*.json), samples null neuron sets
(activation-change-matched, branch-matched, random) and computes Fisher scores for data-driven selection.
Global context: the family is the explanation's claimed mediator; the nulls answer "is this specific set special,
or would any set whose activations change as much do the same?" (Huang et al. 2023 random baseline, made harder).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Family:
    """A published neuron family: layer name, channel indices, and the authors' 720-point tuning curves."""

    layer: str
    neurons: list[int]
    tuning: np.ndarray  # (len(neurons), 720), sampled every 0.5 deg in the authors' orientation convention


def load_family(data_dir: Path, layer: str) -> Family:
    """Read `data/circuits/curves-<layer>.json` (copied from github.com/distillpub/post--circuits-curve-detectors)."""
    d = json.loads((data_dir / "circuits" / f"curves-{layer}.json").read_text())  # {"layer", "neurons", "activations"}
    # tuning rows follow the authors' neuron order; JSON keys are strings
    return Family(layer, d["neurons"], np.array([d["activations"][str(n)] for n in d["neurons"]], np.float64))


def delta_matched_sets(delta: np.ndarray, family: list[int], n_sets: int, rng: np.random.Generator,
                       n_bins: int = 10) -> np.ndarray:
    """Random non-family channel sets whose activation-change histogram (over `n_bins` quantile bins of `delta`)
    equals the family's.

    Local: `delta[c]` = mean squared change of channel c between base and source (selection split). For each set and
    each bin, draw as many non-family channels from that bin as the family has there (without replacement).
    Global context: a patch can only matter through channels that differ between base and source, so matching
    this histogram removes the trivial advantage of the family over random channels that barely change.
    """
    edges = np.quantile(delta, np.linspace(0, 1, n_bins + 1)[1:-1])  # interior decile edges
    bins = np.digitize(delta, edges)  # bin index per channel
    need = np.bincount(bins[family], minlength=n_bins)  # family's histogram
    pools = [np.setdiff1d(np.flatnonzero(bins == b), family) for b in range(n_bins)]  # non-family channels per bin
    # one set = for every bin, as many distinct channels from that bin's pool as the family has there
    return np.stack([np.concatenate([rng.choice(pools[b], need[b], replace=False) for b in range(n_bins) if need[b]])
                     for _ in range(n_sets)])


def rank_matched_sets(delta: np.ndarray, family: list[int], n_sets: int, rng: np.random.Generator,
                      pool: int = 10) -> np.ndarray:
    """Post-hoc, stricter null: for every family unit, one distinct non-family channel among the `pool` channels
    closest to it in delta.

    Local: unlike decile matching, this follows the family into the extreme tail (all 10 curve units lie above the
    96th delta percentile, so a decile null mostly draws channels that change less). Global context: added after the
    confirmatory run in response to code review; reported as post-hoc in reports/posthoc_robustness.md.
    """
    others = np.setdiff1d(np.arange(len(delta)), family)  # candidate channels
    near = [others[np.argsort(np.abs(delta[others] - delta[f]))[:pool]] for f in family]  # nearest by delta, per unit
    sets = []
    for _ in range(n_sets):
        chosen: list[int] = []
        for i in rng.permutation(len(family)):  # random order so no unit always gets first pick
            cand = np.setdiff1d(near[i], chosen)  # keep channels distinct within a set
            chosen.append(int(rng.choice(cand if len(cand) else np.setdiff1d(others, chosen))))
        sets.append(chosen)
    return np.array(sets)


def top_nonfamily(delta: np.ndarray, family: list[int], k: int) -> np.ndarray:
    """The k non-family channels with the largest delta (the strongest single competitor set), in descending order."""
    order = np.argsort(-delta)  # largest delta first
    return order[~np.isin(order, family)][:k]  # boolean mask keeps the order (np.setdiff1d would re-sort)


def pool_sets(pool: np.ndarray, size: int, n_sets: int, rng: np.random.Generator) -> np.ndarray:
    """`n_sets` random subsets of `pool`, each of `size` distinct channels (branch-matched / random nulls)."""
    return np.stack([rng.choice(pool, size, replace=False) for _ in range(n_sets)])


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature Fisher score: between-class over within-class variance, sum_k n_k (mu_k-mu)^2 / sum_k n_k s_k^2.

    Local: Gu, Li & Han 2011 (https://arxiv.org/abs/1202.3725); used for neuron selection by Baeumel et al. 2025
    (https://arxiv.org/abs/2508.02513). Global context: picks the data-driven "best" orientation neurons as an
    upper-bound reference for the human (Circuits) explanation.
    """
    mu = X.mean(0)  # overall mean per feature
    classes = np.unique(y)
    n = np.array([(y == k).sum() for k in classes])[:, None]  # class sizes n_k, shaped for broadcasting
    mus = np.stack([X[y == k].mean(0) for k in classes])  # class means mu_k
    var = np.stack([X[y == k].var(0) for k in classes])  # class variances s_k^2
    return (n * (mus - mu) ** 2).sum(0) / np.maximum((n * var).sum(0), 1e-12)  # between / within (guard 0 variance)
