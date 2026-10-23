"""Figures for the reports, paper and README (matplotlib, static PNG + one GIF).

Local: one function per figure; a shared light style with the validated categorical palette of the dataviz
guidance (blue = the explanation's neuron set, orange = activation-change-matched null, aqua = branch-matched null,
gray = descriptive references), thin marks, recessive grid, one y-axis per panel.
Global context: every figure is generated from `results/*.json` / in-memory arrays by the phases, never by hand.
"""

import matplotlib

matplotlib.use("Agg")  # file output only (no display needed): https://matplotlib.org/stable/users/explain/figure/backends.html
import matplotlib.pyplot as plt
import numpy as np

from concept_swap.config import Config

BLUE, ORANGE, AQUA, GRAY, INK, INK2, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
                     "figure.dpi": 150})


def _save(cfg: Config, fig, name: str) -> None:
    """Write figures/<name>.png and close the figure."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.figures_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_radial_replication(cfg: Config, ours: dict, pub: dict, res: dict) -> None:
    """Small multiples of 3b radial tuning (ours vs published, each scaled to max 1) + per-layer mean r vs null."""
    fam = res["mixed3b"]["neurons"]
    n = ours["mixed3b"].shape[1]
    ang = np.radians(np.arange(n + 1) * 360 / n)  # closed curve
    fig, axes = plt.subplots(2, 5, subplot_kw={"projection": "polar"}, figsize=(11, 4.8))
    for k, ax in enumerate(axes.flat):
        for curve, color, ls, lab in ((pub["mixed3b"][k], GRAY, "--", "published (Cammarata et al. 2020)"),
                                      (ours["mixed3b"][k], BLUE, "-", "ours")):
            c = curve / max(curve.max(), 1e-9)
            ax.plot(ang, np.r_[c, c[:1]], color=color, ls=ls, lw=1.5, label=lab)
        ax.set_title(f"3b:{fam[k]}  r={res['mixed3b']['r'][k]:.2f}", fontsize=8, pad=8)
        ax.set_yticklabels([]), ax.set_xticklabels([]), ax.grid(color="#e4e3df", lw=0.5)
    axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0, 1.5), fontsize=8, frameon=False, ncol=2)
    fig.suptitle("E1: radial orientation tuning of the mixed3b curve family (activation scaled to max = 1)", y=1.06)
    _save(cfg, fig, "e1_radial_3b")
    fig, ax = plt.subplots(figsize=(5, 2.4))
    layers = ["mixed3a", "mixed3b", "mixed4a"]
    for i, l in enumerate(layers):
        ax.plot([res[l]["perm_null_mean"], res[l]["perm_null_q95"]], [i, i], color=GRAY, lw=6, alpha=0.5,
                solid_capstyle="round", label="label-permutation null (mean to q95)" if i == 0 else None)
        ax.plot(res[l]["mean_r"], i, "o", color=BLUE, ms=8, label="ours vs published" if i == 0 else None)
    ax.set_yticks(range(3), [f"{l}\n(fit layer)" if l == res["alignment"]["fit_layer"] else l for l in layers])
    ax.set_xlabel("mean Pearson r of tuning curves"), ax.set_xlim(-0.3, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(cfg, fig, "e1_replication_summary")


def plot_null(cfg: Config, name: str, title: str, metric: str, target: dict, nulls: dict, refs: dict) -> None:
    """Histogram of each null's per-set metric (small multiples, shared x) with the explanation's set (95% CI band)
    and descriptive reference sets marked."""
    fig, axes = plt.subplots(len(nulls), 1, figsize=(6.5, 1.9 * len(nulls)), sharex=True, squeeze=False)
    colors = [ORANGE, AQUA, GRAY]
    for ax, (label, vals), color in zip(axes[:, 0], nulls.items(), colors):
        ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(np.quantile(vals, 0.95), color=color, ls=":", lw=1.5)
        ax.axvspan(*target["ci"], color=BLUE, alpha=0.15, lw=0)
        ax.axvline(target["value"], color=BLUE, lw=2)
        ax.set_ylabel("# sets"), ax.set_title(f"{label} (n={len(vals)} sets; dotted = q95)", fontsize=8, loc="left")
    top = axes[0, 0]
    ymax = top.get_ylim()[1]
    top.text(target["value"], ymax * 0.95, f" {target['label']}\n {target['value']:.2f}", color=INK, fontsize=8, va="top")
    for k, (lab, v) in enumerate(refs.items()):
        top.plot(v, ymax * (0.55 - 0.13 * k), "v", color=INK2, ms=6)
        top.text(v, ymax * (0.55 - 0.13 * k), f" {lab}", color=INK2, fontsize=7, va="center")
    allv = np.concatenate([np.ravel(v) for v in nulls.values()] + [np.ravel(list(refs.values())), np.ravel(target["ci"])])
    axes[-1, 0].set_xlim(min(0.0, allv.min()) - 0.02, max(1.0, allv.max()) + 0.02)  # metric lives on [0, 1] (effect may dip below 0)
    axes[-1, 0].set_xlabel(metric)
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    _save(cfg, fig, name)


def plot_points(cfg: Config, name: str, title: str, xlabels: list, series: dict, ylabel: str, ylim=(0, 1)) -> None:
    """Point estimates with 95% CI error bars for one or more series over categorical x (e.g. |dtheta| bins, K)."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(len(xlabels))
    for k, ((lab, (y, lo, hi)), color) in enumerate(zip(series.items(), [BLUE, ORANGE, AQUA, GRAY])):
        off = (k - (len(series) - 1) / 2) * 0.12
        y, lo, hi = map(np.asarray, (y, lo, hi))
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="o-", color=color, ms=5, lw=1.5, capsize=2, label=lab)
    ax.set_xticks(x, xlabels), ax.set_ylabel(ylabel), ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    _save(cfg, fig, name)


def plot_effects(cfg: Config, name: str, title: str, rows: list[dict], xlabel: str) -> None:
    """One row per hypothesis: explanation's effect (blue dot + CI) vs null q95 (orange tick) and null mean (gray)."""
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1))
    for i, r in enumerate(rows):
        ax.plot(r["ci"], [i, i], color=BLUE, lw=2)
        ax.plot(r["value"], i, "o", color=BLUE, ms=7, label="curve family (95% CI)" if i == 0 else None)
        ax.plot(r["null_q95"], i, "|", color=ORANGE, ms=14, mew=2, label="null q95" if i == 0 else None)
        ax.plot(r["null_mean"], i, "|", color=GRAY, ms=10, mew=2, label="null mean" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows]), ax.invert_yaxis()
    ax.set_xlabel(xlabel), ax.set_xlim(min(0.0, ax.get_xlim()[0]), 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.1 / len(rows) * 4), ncol=3)
    ax.set_title(title, fontsize=10, loc="left")
    _save(cfg, fig, name)
