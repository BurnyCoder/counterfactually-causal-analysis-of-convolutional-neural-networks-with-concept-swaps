"""README hero figure and orientation-sweep GIF for one fixed, rule-chosen example (not selected for success).

Local: base = arc with theta 0, radius 45, width 3, dark-on-light, from the confirmatory split; the figure uses the
source with theta 90 deg; the GIF sweeps the source over all orientations. Each frame shows the 14 mixed4a curve units
(z-scores) for the clean base, the patched base (3b curve family copied from the source) and the clean source.
Global context: an illustration of the interchange intervention measured statistically in E2.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter  # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html

from concept_swap import families as Fm, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import Readout, make_split, swap_readouts
from concept_swap.plots import BLUE, GRAY, INK2, ORANGE, _save, bare

log = get_logger(__name__)


def hero_pair(g) -> tuple[int, int]:
    """Row indices (base theta 0, source theta 90) of the fixed example in the cell radius 45, width 3, dark-on-light.

    The rule is fixed in code, not chosen by outcome; the example phase draws the same pair."""
    cell = g[(g.radius == 45) & (g.width == 3) & (g.polarity == 0)]  # fixed example cell (rule, not outcome, based)
    return int(cell.index[cell.theta == 0][0]), int(cell.index[cell.theta == 90][0])


def run(cfg: Config, model) -> None:
    """Write figures/hero_concept_swap.png and figures/concept_swap_sweep.gif."""
    fam3b, fam4a = Fm.load_family(cfg.data_dir, "mixed3b").neurons, np.array(Fm.load_family(cfg.data_dir, "mixed4a").neurons)
    sel, test = make_split(cfg, model, "mixed4a", S.grid(cfg, cfg.selection_seed)), make_split(cfg, model, "mixed4a", S.grid(cfg, cfg.confirm_seed))
    rd = Readout.fit(sel, fam4a, sel.grid.theta.values)  # the same primary readout as H1
    g = test.grid
    b, _ = hero_pair(g)  # base: theta = 0
    srcs = g.index.values[g.cell == g.cell[b]]  # every orientation in the base's cell, theta = 0 ... 355
    sw = swap_readouts(model, test, rd, np.full(len(srcs), b), srcs, fam3b)  # one swap per source
    z_b, z_s, z_p, dec_p, dec_s = sw["z_b"][0], sw["z_s"], sw["z_p"], sw["dec_p"], sw["dec_s"]  # z-vectors, decodes
    imgs = S.render(g.loc[srcs], cfg.arc_span_deg).permute(0, 2, 3, 1).numpy()  # HWC images for imshow
    x = np.arange(len(fam4a))  # bar positions, one per 4a curve unit

    def draw(fig, axes, k):
        """Draw frame k: base image, source image k, and the three readout bar groups."""
        for ax in axes:
            ax.clear()  # the GIF redraws every frame from scratch
        axes[0].imshow(imgs[0]), axes[0].set_title(f"base: θ = {g.theta[b]:.0f}°", fontsize=9)  # srcs[0] is the base itself
        axes[1].imshow(imgs[k]), axes[1].set_title(f"source: θ = {g.theta[srcs[k]]:.0f}°", fontsize=9)
        for ax in axes[:2]:
            bare(ax)  # plain image panels
        w = 0.27  # bar width for three side-by-side groups
        axes[2].bar(x - w, z_b, w, color=GRAY, label=f"clean base → decoded {sw['dec_b'][0]:.0f}°")
        axes[2].bar(x, z_p[k], w, color=BLUE, label=f"base with 3b curve family from source → decoded {dec_p[k]:.0f}°")
        axes[2].bar(x + w, z_s[k], w, color=ORANGE, label=f"clean source → decoded {dec_s[k]:.0f}°")
        axes[2].set_xticks(x, [f"4a:{n}" for n in fam4a], rotation=60, fontsize=7)
        axes[2].set_ylabel("z-scored pre-ReLU activation"), axes[2].legend(frameon=False, fontsize=7, loc="upper left")
        axes[2].set_ylim(min(z_b.min(), z_p.min(), z_s.min()) - 0.5, max(z_b.max(), z_p.max(), z_s.max()) + 2.5)  # fixed range across frames
        fig.suptitle("Concept swap: copy the 10 mixed3b curve-unit maps from the source into the base, read the 14 mixed4a curve units",
                     fontsize=9, color=INK2)

    def layout():
        """Figure with two image panels, a spacer column and a wide bar-chart panel."""
        fig = plt.figure(figsize=(10, 3.6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.08, 3.2])
        return fig, [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])]

    fig, axes = layout()
    draw(fig, axes, int(np.flatnonzero(g.theta.values[srcs] == 90)[0]))  # the pre-chosen theta = 90 source
    _save(cfg, fig, "hero_concept_swap")
    fig, axes = layout()
    anim = FuncAnimation(fig, lambda k: draw(fig, axes, k), frames=range(0, len(srcs), 2))  # every 10 degrees
    anim.save(cfg.figures_dir / "concept_swap_sweep.gif", writer=PillowWriter(fps=4))
    plt.close(fig)
    th = g.theta.values[srcs]  # source orientations 0 ... 355
    far, ok = np.abs(S.circ_diff(th, g.theta[b])) >= cfg.min_dtheta, np.abs(S.circ_diff(dec_s, th)) <= cfg.tol_deg  # pair rules
    hit = far & ok & (np.abs(S.circ_diff(dec_p, dec_s)) <= cfg.tol_deg)  # H1's success criterion
    log.info("hero figure + GIF written; example base %d, decoded patched orientations %s; decoded clean sources %s; "
             "%d of %d sources >= %d deg from the base decode within %g deg when clean; %d succeed: %s", b,
             dec_p.round().tolist(), dec_s.round().tolist(), (far & ok).sum(), far.sum(), cfg.min_dtheta, cfg.tol_deg,
             hit.sum(), th[hit].tolist())
