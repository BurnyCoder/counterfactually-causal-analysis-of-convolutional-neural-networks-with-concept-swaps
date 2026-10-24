"""Figures for the reports, paper and README (matplotlib, static PNGs; the GIF is written by `phases/hero.py`).

Local: one function per figure; a shared light style with the validated categorical palette of the dataviz
guidance (blue = the explanation's neuron set, orange = activation-change-matched null, aqua = branch-matched null,
gray = descriptive references), thin marks, recessive grid, one y-axis per panel.
Global context: every figure is generated from `results/*.json` / in-memory arrays by the phases, never by hand.
"""

import matplotlib

matplotlib.use("Agg")  # file output only (no display needed): https://matplotlib.org/stable/users/explain/figure/backends.html
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from concept_swap.config import Config

BLUE, ORANGE, AQUA, GRAY, INK, INK2, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
# Sequential = one hue light -> dark (the dataviz blue ramp, starting at the surface so zero recedes); diverging =
# blue <-> red poles around a neutral gray midpoint (dataviz reference palette).
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef9a99", "#b3302f"])
ROWS = (GRAY, ORANGE, BLUE)  # clean base, clean source, swapped base (same roles as the hero figure)
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
                     "figure.dpi": 150})


def _save(cfg: Config, fig, name: str) -> None:
    """Write figures/<name>.png and close the figure."""
    path = cfg.figures_dir / f"{name}.png"  # `name` may contain a subfolder, e.g. "examples/success_153"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")  # tight: keep outside legends and titles
    plt.close(fig)  # free memory; figures are never shown interactively


def bare(ax) -> None:
    """Strip ticks, grid and frame from an image panel."""
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _icons(axes, icons, labels, size: float = 6.5) -> None:
    """Feature-visualization icons with their neuron label underneath (the Circuits papers' way of naming a unit)."""
    for ax, im, lab in zip(axes, icons, labels):
        ax.imshow(im), bare(ax), ax.set_xlabel(lab, fontsize=size, labelpad=2, color=INK2)


def _decoder_panel(ax, th, corr, dec, colors, tol, band=None) -> None:
    """Pearson r of readout vectors with the cell's clean templates vs template orientation, decoded label as a dot;
    `band` = centre of the +-tol success window (circular)."""
    if band is not None:  # success window around decode(clean source); shifted copies make it wrap past 0/360
        for lo in (band - tol - 360, band - tol, band - tol + 360):
            ax.axvspan(lo, lo + 2 * tol, color=ORANGE, alpha=0.12, lw=0)  # clipped to [0, 360] by set_xlim below
    for c, d, col in zip(corr, dec, colors):
        ax.plot(th, c, color=col, lw=1.5)
        ax.plot(d, c[np.argmin(np.abs(th - d))], "o", color=col, ms=6, mec=SURFACE, mew=1.5)  # surface ring
    ax.set_xlim(0, 360), ax.set_ylim(-1.05, 1.05), ax.set_xticks([0, 90, 180, 270, 360])
    ax.tick_params(labelsize=7)


def plot_featvis_grid(cfg: Config, name: str, groups: list[tuple[str, np.ndarray, list[str]]]) -> None:
    """One row of feature-visualization icons per neuron family (Cammarata et al. 2020 show families this way)."""
    n = max(len(icons) for _, icons, _ in groups)  # columns = the largest family
    fig, axes = plt.subplots(len(groups), n, figsize=(1.05 * n, 1.45 * len(groups)), squeeze=False)
    for row, (title, icons, labels) in zip(axes, groups):
        _icons(row, icons, labels)
        for ax in row[len(icons):]:
            ax.axis("off")  # unused cells of the shorter family
        row[0].set_title(title, fontsize=9, loc="left", color=INK)
    _save(cfg, fig, name)


def plot_swap_flow(cfg: Config, name: str, ex: dict, icons3b, labels3b, icons4a, labels4a) -> None:
    """One pair flowing through the network. Rows: clean base, clean source, swapped base. Columns: input image,
    the 10 mixed3b curve-unit maps (swapped 11x11 window boxed), the 14 mixed4a curve-unit readouts, and the
    decoder's correlation with each clean template. Units are shown by feature visualizations, ordered by preferred
    orientation."""
    fig = plt.figure(figsize=(21, 7.6))
    gs = fig.add_gridspec(5, 16, height_ratios=[1.05, 1, 1, 1, 0.07], hspace=0.34, wspace=0.07,
                          width_ratios=[1.6, 0.1] + [1] * 10 + [0.3, 5.0, 0.5, 3.0])
    vmax, vdiv = max(float(ex["maps"].max()), 1e-6), max(float(np.abs(ex["strip"]).max()), 1e-6)  # shared scales
    ax3 = [fig.add_subplot(gs[0, 2 + j]) for j in range(10)]
    _icons(ax3, icons3b, labels3b)
    sub = gs[0, 13].subgridspec(1, len(icons4a), wspace=0.1)
    ax4 = [fig.add_subplot(sub[0, j]) for j in range(len(icons4a))]
    _icons(ax4, icons4a, labels4a, size=6)
    ax3[0].set_title("10 mixed3b curve units (feature visualization, unit, E1 preferred θ) and their activation maps; "
                     "box = the swapped 11×11 window", fontsize=8, loc="left")
    ax4[0].set_title("14 mixed4a curve units (4a: unit, preferred θ): pre-ReLU readout at (6, 6)", fontsize=8, loc="left")
    lo, hi = ex["box"]
    x0, w = ex["px_box"]
    for i in range(3):
        r, swapped = i + 1, i == 2
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(ex["images"][i]), bare(ax)
        ax.add_patch(Rectangle((x0 - 0.5, x0 - 0.5), w, w, fill=False, ec=ORANGE if swapped else INK2, lw=1, ls="-" if swapped else "--"))
        ax.add_patch(Rectangle((-0.07, 0), 0.035, 1, transform=ax.transAxes, color=ROWS[i], clip_on=False))  # row key
        ax.text(-0.11, 0.5, ex["row_labels"][i], transform=ax.transAxes, ha="right", va="center", fontsize=8, color=INK)
        for j in range(10):
            am = fig.add_subplot(gs[r, 2 + j])
            im = am.imshow(ex["maps"][i, j], cmap=SEQ, vmin=0, vmax=vmax)
            bare(am), am.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo, fill=False, lw=0.9,
                                             ec=ORANGE if swapped else INK2, ls="-" if swapped else "--"))
        a = fig.add_subplot(gs[r, 13])
        im2 = a.imshow(ex["strip"][i][None], cmap=DIV, vmin=-vdiv, vmax=vdiv, aspect="auto")
        bare(a)
        for k in range(1, ex["strip"].shape[1]):
            a.axvline(k - 0.5, color=SURFACE, lw=2)  # 2px surface gap between cells
        ad = fig.add_subplot(gs[r, 15])
        _decoder_panel(ad, ex["th"], ex["corr"][[1, 2]] if swapped else ex["corr"][[i]], ex["dec"][[1, 2]] if swapped else ex["dec"][[i]],
                       [ORANGE, BLUE] if swapped else [ROWS[i]], ex["tol"], band=ex["dec"][1] if swapped else None)
        ad.set_title(f"decoded {ex['dec'][i]:.0f}°" + (f": {ex['verdict']} (orange band = source decode ± {ex['tol']:.0f}°)" if swapped else ""),
                     fontsize=8, loc="left")
        ad.set_ylabel("Pearson r", fontsize=7)
        ad.tick_params(labelbottom=swapped)
    ad.set_xlabel("template orientation θ (deg)", fontsize=7)
    fig.add_subplot(gs[0, 15]).axis("off")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[4, 2:12]), orientation="horizontal")
    cb.set_label("mixed3b activation (post-ReLU), σ above the selection-split mean; below the mean drawn as 0", fontsize=7)
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(gs[4, 13]), orientation="horizontal")
    cb2.set_label("mixed4a readout, z (σ, selection split)", fontsize=7)
    for c in (cb, cb2):
        c.ax.tick_params(labelsize=7), c.outline.set_visible(False)
    fig.suptitle(f"{ex['title']}\n{ex['view']}", fontsize=10, x=0.02, ha="left", y=0.99)
    _save(cfg, fig, name)


def plot_examples_overview(cfg: Config, name: str, rows: list[dict], unit_labels: list[str]) -> None:
    """One row per example pair: base and source images, the 14 mixed4a readouts (base / source / swapped) and the
    decoder curves, so successful and failed swaps can be compared at a glance."""
    n = len(rows)
    fig = plt.figure(figsize=(12.5, 1.5 * n + 1.0))
    gs = fig.add_gridspec(n + 1, 4, width_ratios=[1, 1, 3.4, 3.4], height_ratios=[1] * n + [0.1], wspace=0.3, hspace=0.6)
    vdiv = max(float(np.abs(ex["strip"]).max()) for ex in rows)
    for i, ex in enumerate(rows):
        for k, who in enumerate(("base", "source")):  # input images with their row-colour key
            a = fig.add_subplot(gs[i, k])
            a.imshow(ex["images"][k]), bare(a), a.set_title(f"{who} θ = {ex['theta'][k]:.0f}°", fontsize=7)
            a.add_patch(Rectangle((-0.06, 0), 0.04, 1, transform=a.transAxes, color=ROWS[k], clip_on=False))
            if k == 0:
                a.text(-0.14, 0.5, ex["label"], transform=a.transAxes, ha="right", va="center", fontsize=8)
        a = fig.add_subplot(gs[i, 2])
        im = a.imshow(ex["strip"], cmap=DIV, vmin=-vdiv, vmax=vdiv, aspect="auto")
        a.set_yticks(range(3), ["base", "source", "swapped"], fontsize=7), a.grid(False)
        a.set_xticks(range(len(unit_labels)), unit_labels if i == n - 1 else [], fontsize=6)  # units under the last row
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        for k in range(1, ex["strip"].shape[1]):
            a.axvline(k - 0.5, color=SURFACE, lw=2)
        for k in (0.5, 1.5):
            a.axhline(k, color=SURFACE, lw=2)
        if i == 0:
            a.set_title("14 mixed4a curve units (ordered by preferred θ), z", fontsize=8, loc="left")
        ad = fig.add_subplot(gs[i, 3])
        _decoder_panel(ad, ex["th"], ex["corr"], ex["dec"], ROWS, ex["tol"], band=ex["dec"][1])
        ad.set_title(f"{ex['verdict']}: swapped base decodes {ex['dec'][2]:.0f}°, source {ex['dec'][1]:.0f}°", fontsize=8, loc="left")
        ad.tick_params(labelbottom=i == n - 1)
    ad.set_xlabel("template orientation θ (deg); dots = decoded θ; band = source decode ± 10°", fontsize=7)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[n, 2]), orientation="horizontal")
    cb.ax.tick_params(labelsize=7), cb.outline.set_visible(False)
    cb.set_label("mixed4a pre-ReLU readout at (6, 6), z (σ, selection split); columns = 4a unit, preferred θ", fontsize=7)
    lax = fig.add_subplot(gs[n, 3])
    lax.axis("off")
    lax.legend([plt.Line2D([], [], color=c, lw=2) for c in ROWS], ["clean base", "clean source", "swapped base"],
               loc="center", ncol=3, frameon=False, fontsize=8)
    _save(cfg, fig, name)


def plot_radial_replication(cfg: Config, ours: dict, pub: dict, res: dict) -> None:
    """Small multiples of 3b radial tuning (ours vs published, each scaled to max 1) + per-layer mean r vs null."""
    fam = res["mixed3b"]["neurons"]
    n = ours["mixed3b"].shape[1]  # orientations on the grid
    ang = np.radians(np.arange(n + 1) * 360 / n)  # closed curve
    fig, axes = plt.subplots(2, 5, subplot_kw={"projection": "polar"}, figsize=(11, 4.8))  # one polar panel per unit
    for k, ax in enumerate(axes.flat):
        for curve, color, ls, lab in ((pub["mixed3b"][k], GRAY, "--", "published (Cammarata et al. 2020)"),
                                      (ours["mixed3b"][k], BLUE, "-", "ours")):
            c = curve / max(curve.max(), 1e-9)  # scale to max 1 (shape comparison; units differ between sources)
            ax.plot(ang, np.r_[c, c[:1]], color=color, ls=ls, lw=1.5, label=lab)  # repeat first point to close
        ax.set_title(f"3b:{fam[k]}  r={res['mixed3b']['r'][k]:.2f}", fontsize=8, pad=8)
        ax.set_yticklabels([]), ax.set_xticklabels([]), ax.grid(color="#e4e3df", lw=0.5)  # recessive grid, no tick clutter
    axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0, 1.5), fontsize=8, frameon=False, ncol=2)
    fig.suptitle("E1: radial orientation tuning of the mixed3b curve family (activation scaled to max = 1)", y=1.06)
    _save(cfg, fig, "e1_radial_3b")
    fig, ax = plt.subplots(figsize=(5, 2.4))  # second figure: mean r per layer vs its permutation null
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
    allv = np.concatenate([np.ravel(v) for v in nulls.values()] + [np.ravel(list(refs.values())), np.ravel(target["ci"])])
    lo, hi = min(0.0, allv.min()) - 0.02, max(1.0, allv.max()) + 0.02  # metric lives on [0, 1] (effect may dip below 0)
    edges = np.linspace(lo, hi, 161)  # shared bins over the axis: a null piled up near 0 still draws a visible bar
    fig, axes = plt.subplots(len(nulls), 1, figsize=(6.5, 1.9 * len(nulls)), sharex=True, squeeze=False)
    colors = [ORANGE, AQUA, GRAY]  # fixed categorical order: decile null, branch null, random
    for ax, (label, vals), color in zip(axes[:, 0], nulls.items(), colors):
        ax.hist(vals, bins=edges, color=color, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)  # surface-coloured gaps
        ax.axvline(np.quantile(vals, 0.95), color=color, ls=":", lw=1.5)  # the null's q95 bar
        ax.axvspan(*target["ci"], color=BLUE, alpha=0.15, lw=0)  # family 95% CI
        ax.axvline(target["value"], color=BLUE, lw=2)  # family point estimate
        ax.set_ylabel("# sets"), ax.set_title(f"{label} (n={len(vals)} sets; dotted = q95)", fontsize=8, loc="left")
    top = axes[0, 0]  # annotations go on the first panel only
    ymax = top.get_ylim()[1]
    top.text(target["value"], ymax * 0.95, f" {target['label']}\n {target['value']:.2f}", color=INK, fontsize=8, va="top")
    for k, (lab, v) in enumerate(refs.items()):  # reference sets as labelled markers, staggered vertically
        top.plot(v, ymax * (0.55 - 0.13 * k), "v", color=INK2, ms=6)
        top.text(v, ymax * (0.55 - 0.13 * k), f" {lab}", color=INK2, fontsize=7, va="center")
    axes[-1, 0].set_xlim(lo, hi)  # shared x axis = the histogram range
    axes[-1, 0].set_xlabel(metric)
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    _save(cfg, fig, name)


def plot_points(cfg: Config, name: str, title: str, xlabels: list, series: dict, ylabel: str, ylim=(0, 1),
                x=None, xlabel: str = "") -> None:
    """Point estimates with 95% CI error bars for one or more series; x is categorical unless numeric `x` is given
    (numeric x keeps unequal spacing honest, e.g. subset sizes K = 1, 2, 3, 5, 7, 10)."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    x = np.arange(len(xlabels)) if x is None else np.asarray(x, float)
    for k, ((lab, (y, lo, hi)), color) in enumerate(zip(series.items(), [BLUE, ORANGE, AQUA, GRAY])):
        off = (k - (len(series) - 1) / 2) * 0.06 * (x.max() - x.min()) / max(len(x) - 1, 1)  # small dodge, x-scaled
        y, lo, hi = map(np.asarray, (y, lo, hi))
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="o-", color=color, ms=5, lw=1.5, capsize=2, label=lab)  # asymmetric CI
    ax.set_xticks(x, xlabels), ax.set_ylabel(ylabel), ax.set_ylim(*ylim), ax.set_xlabel(xlabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)  # legend only when there is more than one series
    ax.set_title(title, fontsize=10, loc="left")
    _save(cfg, fig, name)


def plot_effects(cfg: Config, name: str, title: str, rows: list[dict], xlabel: str) -> None:
    """One row per hypothesis: explanation's effect (blue dot + CI) vs null q95 (orange tick) and null mean (gray)."""
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1))  # height grows with the number of tests
    for i, r in enumerate(rows):
        ax.plot(r["ci"], [i, i], color=BLUE, lw=2)  # CI as a horizontal segment
        ax.plot(r["value"], i, "o", color=BLUE, ms=7, label="curve family (95% CI)" if i == 0 else None)
        ax.plot(r["null_q95"], i, "|", color=ORANGE, ms=14, mew=2, label="null q95" if i == 0 else None)
        ax.plot(r["null_mean"], i, "|", color=GRAY, ms=10, mew=2, label="null mean" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows]), ax.invert_yaxis()  # first test on top
    ax.set_xlabel(xlabel), ax.set_xlim(min(0.0, ax.get_xlim()[0]), 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.1 / len(rows) * 4), ncol=3)  # below axis
    ax.set_title(title, fontsize=10, loc="left")
    _save(cfg, fig, name)
