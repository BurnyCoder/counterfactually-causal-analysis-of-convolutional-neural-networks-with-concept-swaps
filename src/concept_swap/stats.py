"""Statistics: two-way (pigeonhole) bootstrap over crossed base/source stimuli, null quantiles, p-values,
the pre-registered decision rule and Holm correction.

Local: small numpy/statsmodels helpers operating on per-pair outcome matrices of shape (n_sets, n_pairs).
Global context: pairs share base and source images, so outcomes are not independent; resampling base images and
source images separately (Owen 2007, "The pigeonhole bootstrap", https://doi.org/10.1214/07-AOAS122) gives
honest CIs. Null quantiles are recomputed inside every bootstrap replicate so their uncertainty is included.
"""

import numpy as np
from statsmodels.stats.multitest import multipletests  # https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html


def pigeonhole_weights(base: np.ndarray, src: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, n_pairs) bootstrap weights: resample base ids and source ids independently with replacement; a pair's
    weight is (times its base was drawn) x (times its source was drawn)."""
    ub, bi = np.unique(base, return_inverse=True)
    us, si = np.unique(src, return_inverse=True)
    cb = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(B)])
    cs = np.stack([np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)) for _ in range(B)])
    return (cb[:, bi] * cs[:, si]).astype(np.float64)


def weighted_rates(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted mean of each row of `values` (n_sets, n_pairs) under each bootstrap weight row: (B, n_sets)."""
    return (W @ values.T.astype(np.float64)) / W.sum(1, keepdims=True)


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval (Efron & Tibshirani 1993)."""
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def perm_p(obs: float, null: np.ndarray) -> float:
    """One-sided Monte-Carlo p-value (1 + #{null >= obs}) / (1 + M) (Phipson & Smyth 2010, https://doi.org/10.2202/1544-6115.1585)."""
    return float((1 + np.sum(np.asarray(null) >= obs)) / (1 + len(null)))


def margin_vs_null(target: np.ndarray, null: np.ndarray, W: np.ndarray, q: float = 0.95) -> dict:
    """Effect of the target set minus the q-quantile of a null distribution of sets, with a pigeonhole CI.

    target: (n_pairs,) outcomes of the explanation's set; null: (n_sets, n_pairs) outcomes of control sets.
    Returns point estimates, the bootstrap CI of the margin D = rate - q95(null), and the permutation p-value.
    """
    rate, null_rates = float(target.mean()), null.mean(1)
    rb = weighted_rates(np.vstack([target, null]), W)  # (B, 1 + n_sets)
    D = rb[:, 0] - np.quantile(rb[:, 1:], q, axis=1)
    return dict(rate=rate, rate_ci=percentile_ci(rb[:, 0]), null_mean=float(null_rates.mean()),
                null_q95=float(np.quantile(null_rates, q)), margin=rate - float(np.quantile(null_rates, q)),
                margin_ci=percentile_ci(D), p=perm_p(rate, null_rates), n_pairs=int(len(target)), n_sets=int(len(null)))


def decide(rate: float, null_q95: float, ci_margin: tuple, ci_margin_branch: tuple, *, min_rate: float,
           sesoi: float, falsify_rate: float, falsify_margin: float) -> str:
    """Pre-registered three-way verdict for H1 (see reports/preregistration.md)."""
    if rate >= min_rate and rate - null_q95 >= sesoi and ci_margin[0] > 0 and ci_margin_branch[0] > 0:
        return "supported"
    if ci_margin[1] < falsify_margin or rate < falsify_rate:
        return "falsified"
    return "inconclusive"


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control across secondary hypotheses)."""
    return multipletests(pvals, method="holm")[1]
