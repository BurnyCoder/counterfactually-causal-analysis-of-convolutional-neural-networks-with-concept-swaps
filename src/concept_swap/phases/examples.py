"""Example-swap figures: highlighted H1 pairs shown as they flow through the network (output-only, descriptive).

Local: recomputes the H1 per-pair outcomes on the confirmatory split (the same patch and decoder as E2), checks the
IIA against results/e2_synthetic.json, draws random successes and random failures (`stats.random_rows`, not chosen
by outcome size) plus the README hero pair, and for each writes a figure with the exact input images, the full
mixed3b maps of the 10 curve units (swapped window boxed), the 14 mixed4a readouts and the decoder's template
correlations. Neurons are shown by Lucent feature visualizations, as in the Circuits papers (Cammarata et al. 2020,
https://distill.pub/2020/circuits/curve-detectors/). Global context: illustrates the statistics of E2; it adds no
statistic and changes no reported number. Writes figures/{neurons_featvis, examples_overview, examples/*}.png and
reports/generated/examples.md.
"""

import json

import numpy as np
import torch

from concept_swap import families as Fm, featvis as V, model as M, plots as P, stats as St, stimuli as S
from concept_swap.config import ROOT, Config
from concept_swap.log import get_logger
from concept_swap.phases.common import Readout, make_split, swap_readouts
from concept_swap.phases.hero import hero_pair

log = get_logger(__name__)
STRIDE = 8  # mixed3b is 28x28 on a 224-px input; position j has its RF centre at pixel 8j+7 (model.READOUT_POS)


def _load(cfg: Config, name: str) -> dict:
    """An earlier phase's results JSON (E1 preferred orientations, E2's H1 IIA)."""
    return json.loads((cfg.results_dir / f"{name}.json").read_text())["results"]


@torch.no_grad()
def _full_maps(cfg: Config, model, grid, rows: np.ndarray) -> torch.Tensor:
    """Full post-ReLU mixed3b maps (len(rows), 480, 28, 28) of the given stimuli, from a normal forward pass."""
    X = S.render(grid.loc[rows], cfg.arc_span_deg).to(cfg.device)  # the exact input images
    with M.capture(model, ["mixed3b"]) as acts:
        model(M.preprocess(X))
    return acts["mixed3b"]


def run(cfg: Config, model) -> None:
    """Select the example pairs, verify that the figures show exactly what was scored, and draw them."""
    e1, h1 = _load(cfg, "e1_replication"), _load(cfg, "e2_synthetic")["H1"]
    fam3b, fam4a = Fm.load_family(cfg.data_dir, "mixed3b").neurons, Fm.load_family(cfg.data_dir, "mixed4a").neurons
    o3, o4 = (np.argsort(e1[l]["pref_theta_ours"]) for l in ("mixed3b", "mixed4a"))  # units by preferred orientation
    u3, u4 = np.array(fam3b)[o3], np.array(fam4a)[o4]
    sel, test = (make_split(cfg, model, "mixed4a", S.grid(cfg, s)) for s in (cfg.selection_seed, cfg.confirm_seed))
    rd = Readout.fit(sel, np.array(fam4a), sel.grid.theta.values)  # the primary H1 readout, fitted on selection
    g, tol = test.grid, cfg.tol_deg
    pairs = S.minimal_pairs(g, cfg, cfg.confirm_seed)  # the scored H1 pairs (same seed as E2)
    b, s = pairs.base.values, pairs.source.values
    sw = swap_readouts(model, test, rd, b, s, fam3b)
    ok = lambda a, t: np.abs(S.circ_diff(a, t)) <= tol  # within the decoder tolerance
    elig = ok(sw["dec_b"], g.theta.values[b]) & ok(sw["dec_s"], g.theta.values[s])  # both clean images decode
    succ = ok(sw["dec_p"], sw["dec_s"])  # Huang et al.'s success: swapped base decodes as the clean source
    iia = float(succ[elig].mean())
    log.info("examples: H1 family IIA %.6f on %d eligible pairs (results/e2_synthetic.json: %.6f)", iia, elig.sum(), h1["iia"]["family"])
    assert np.isclose(iia, h1["iia"]["family"]), "recomputed H1 outcomes differ from the confirmatory run"
    n_succ, n_fail = cfg.example_counts
    picks = [("success", int(r)) for r in St.random_rows(elig & succ, n_succ, cfg.example_seed)]
    picks += [("failure", int(r)) for r in St.random_rows(elig & ~succ, n_fail, cfg.example_seed)]
    log.info("examples: random rows (seed %d) %s", cfg.example_seed, picks)
    hb, hs = hero_pair(g)
    bases = np.array([b[r] for _, r in picks] + [hb])
    srcs = np.array([s[r] for _, r in picks] + [hs])
    kinds = [k for k, _ in picks] + ["hero"]
    rows = [r for _, r in picks] + [None]

    ex = swap_readouts(model, test, rd, bases, srcs, fam3b)  # the examples' own swaps (hero pair included)
    comp = swap_readouts(model, test, rd, bases, srcs, np.setdiff1d(np.arange(480), fam3b))["dec_p"]  # other 470
    full_b, full_s = _full_maps(cfg, model, g, bases), _full_maps(cfg, model, g, srcs)
    lo, hi = M.window_bounds("mixed4a")
    win = test.col.windows[torch.as_tensor(bases)]
    assert torch.allclose(M.crop_window(full_b, "mixed4a").cpu(), win, atol=1e-3 * win.abs().max().item()), "crop != cached window"
    full_p = full_b.clone()
    full_p[:, fam3b, lo:hi, lo:hi] = full_s[:, fam3b, lo:hi, lo:hi]  # the intervention: 10 maps, swapped window only
    p = M.READOUT_POS["mixed4a"]
    z_full = rd.z(M.run_block(model, "mixed4a", full_p, pre_relu=True)[:, fam4a, p, p].cpu().numpy())
    assert np.allclose(z_full, ex["z_p"], atol=1e-3), "full-map swap != scored cropped-window swap"
    log.info("examples: full-map swap matches the scored readout (max |dz| %.2e)", np.abs(z_full - ex["z_p"]).max())

    W = sel.col.windows[:, fam3b]  # selection-split windows of the family: sigma units for the maps
    mu, sd = W.mean((0, 2, 3)).numpy(), W.std((0, 2, 3)).numpy()
    sig = lambda f: np.clip((f[:, fam3b].cpu().numpy() - mu[:, None, None]) / sd[:, None, None], 0, None)[:, o3]

    log.info("examples: rendering %d feature visualizations (%d steps)", len(u3) + len(u4), cfg.featvis_steps)
    icons = V.render_units(M.load_inceptionv1(cfg.device, cfg.lrn_mode, redirected_relu=True),
                           [("mixed3b", int(n)) for n in u3] + [("mixed4a", int(n)) for n in u4], cfg.featvis_steps, cfg.example_seed)
    c, h3, h4 = S.CENTER, M.rf_halfwidth(model, "mixed3b", cfg.device), M.rf_halfwidth(model, "mixed4a", cfg.device)
    log.info("examples: icons cropped to the effective RF (99%% gradient mass, 5x5 branch): +-%d px (3b), +-%d px (4a)", h3, h4)
    ic3, ic4 = icons[:len(u3), c - h3:c + h3 + 1, c - h3:c + h3 + 1], icons[len(u3):, c - h4:c + h4 + 1, c - h4:c + h4 + 1]
    pref3, pref4 = np.array(e1["mixed3b"]["pref_theta_ours"])[o3], np.array(e1["mixed4a"]["pref_theta_ours"])[o4]
    lab3 = [f"3b:{n}\n{t:.0f}°" for n, t in zip(u3, pref3)]
    lab4 = [f"4a:{n}\n{t:.0f}°" for n, t in zip(u4, pref4)]
    P.plot_featvis_grid(cfg, "neurons_featvis", [("mixed3b curve units (the swapped family)", ic3, lab3),
                                                 ("mixed4a curve units (the readout)", ic4, lab4)])
    lab4_short = [f"{n}\n{t:.0f}°" for n, t in zip(u4, pref4)]  # narrow columns in the flow figures

    # View: the central 3b positions (and their input pixels) that contain every stimulus of the grid. An arc's
    # farthest point lies r*sin(span/2) along its tangent from the midpoint, plus half its width and the jitter.
    q = M.READOUT_POS["mixed3b"]  # 3b position of the RF centre (pixel 111)
    reach = max(cfg.radii) * np.sin(np.radians(cfg.arc_span_deg / 2)) + max(cfg.widths) / 2 + cfg.jitter_px
    m = int(np.ceil((reach - STRIDE / 2) / STRIDE))  # positions each side whose 8-px cells cover the reach
    v0, v1, p0, p1 = q - m, q + m + 1, STRIDE * (q - m) + 3, STRIDE * (q + m + 1) + 3  # map rows, input pixels
    maps_b, maps_s, maps_p = (sig(f)[..., v0:v1, v0:v1] for f in (full_b, full_s, full_p))
    every = np.stack([sig(f) for f in (full_b, full_s, full_p)])  # all drawn maps, full 28x28
    shown, border = every[..., v0:v1, v0:v1].sum() / every.sum(), 1 - every[..., 2:-2, 2:-2].sum() / every.sum()
    log.info("examples: view = 3b rows %d-%d (input px %d-%d) holds %.4f of the drawn sigma mass; the outer 2-position "
             "border ring holds %.4f", v0, v1 - 1, p0, p1 - 1, shown, border)
    imgs_b, imgs_s = (S.render(g.loc[r], cfg.arc_span_deg).permute(0, 2, 3, 1).numpy()[:, p0:p1, p0:p1] for r in (bases, srcs))
    pol = {0: "dark-on-light", 1: "light-on-dark"}
    overview, table = [], []
    for i, (kind, row) in enumerate(zip(kinds, rows)):
        tb, ts = g.theta[bases[i]], g.theta[srcs[i]]
        dec = np.array([ex["dec_b"][i], ex["dec_s"][i], ex["dec_p"][i]])
        th, corr = rd.dec.correlations(np.stack([ex["z_b"][i], ex["z_s"][i], ex["z_p"][i]]), ex["cell"][i])
        strip = np.stack([ex["z_b"][i], ex["z_s"][i], ex["z_p"][i]])[:, o4]
        verdict = "success" if ok(dec[2], dec[1]) else "failure"
        eff = float(((ex["z_p"][i] - ex["z_b"][i]) * (ex["z_s"][i] - ex["z_b"][i])).sum() / ((ex["z_s"][i] - ex["z_b"][i]) ** 2).sum())
        nuis = f"radius {g.radius[bases[i]]}, width {g.width[bases[i]]}, {pol[g.polarity[bases[i]]]}"
        pool = {"success": f"{succ[elig].sum()} successes", "failure": f"{(~succ[elig]).sum()} failures"}.get(kind)
        head = (f"H1 {kind}: pair row {row}, drawn at random (seed {cfg.example_seed}) from the {pool} among {elig.sum()} eligible pairs"
                if row is not None else "README example: fixed rule (hero.py), not one of the scored pairs")
        flow = dict(title=f"{head}\nbase θ = {tb:.0f}°, source θ = {ts:.0f}°, {nuis}; per-pair effect {eff:.2f}",
                    row_labels=[f"clean base\nθ = {tb:.0f}°", f"clean source\nθ = {ts:.0f}°",
                                "swapped base\n(base pixels; the 10\ncurve-unit maps in the\nbox come from the source)"],
                    images=np.stack([imgs_b[i], imgs_s[i], imgs_b[i]]), maps=np.stack([maps_b[i], maps_s[i], maps_p[i]]),
                    box=(lo - v0, hi - v0), px_box=(STRIDE * lo + 3 - p0, STRIDE * (hi - lo)), strip=strip, th=th,
                    corr=corr, dec=dec, tol=tol, verdict=verdict,
                    view=(f"input: central {p1 - p0} px of 224 (holds every stimulus); maps: central {v1 - v0}×{v1 - v0} of "
                          f"28×28 ({shown:.0%} of the drawn activation; {border:.0%} lies in the map's outer 2-position border)"))
        name = f"examples/{kind}_{row}" if row is not None else "examples/hero_pair"
        P.plot_swap_flow(cfg, name, flow, ic3, lab3, ic4, lab4_short)
        if row is not None:
            overview.append(dict(flow, label=f"{kind}\npair {row}", theta=(tb, ts)))
        table.append(f"| {kind} | {row if row is not None else '—'} | {tb:.0f}° | {ts:.0f}° | {nuis} | {dec[0]:.0f}° | {dec[1]:.0f}° | "
                     f"{dec[2]:.0f}° | {verdict} | {eff:.2f} | {comp[i]:.0f}° ({'success' if ok(comp[i], dec[1]) else 'failure'}) | "
                     f"[{name}.png](../../figures/{name}.png) |")
        log.info("examples: %s row %s base %.0f source %.0f | decoded base %.0f source %.0f swapped %.0f -> %s | effect %.3f | complement %.0f",
                 kind, row, tb, ts, *dec, verdict, eff, comp[i])
    P.plot_examples_overview(cfg, "examples_overview", overview, lab4_short)
    out = ROOT / "reports" / "generated" / ("examples_quick.md" if cfg.quick else "examples.md")  # quick never overwrites
    out.write_text("\n".join([
        "# Example swaps (generated)", "",
        "Generated by `concept_swap.phases.examples` from the confirmatory split. Do not edit by hand. Rows are drawn at "
        f"random (`stats.random_rows`, seed {cfg.example_seed}) among eligible H1 successes and failures; the hero pair "
        "is the README's rule-chosen example. Effect = this pair's projection of the swapped readout's move onto the "
        "base→source direction (0 = no change, 1 = equals the source).", "",
        "| kind | pair row | base θ | source θ | cell | clean base → | clean source → | swapped base → | verdict | effect | other 470 channels → | figure |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|", *table, ""]), encoding="utf-8", newline="\n")
    log.info("examples: wrote %d flow figures, the overview, the feature-visualization grid and %s", len(kinds), out.name)
