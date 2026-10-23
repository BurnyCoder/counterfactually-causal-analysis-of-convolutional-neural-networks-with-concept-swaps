"""E2 (interventional, synthetic stimuli): interchange interventions on the Circuits curve families.

Local: H1 (primary) swaps the 10 mixed3b curve units from a source arc into a base arc that differs only in
orientation and asks whether the 14 published mixed4a curve units then encode the source orientation. Secondary:
the same with a ridge readout over all 508 mixed4a channels, H2 curvature swap (line <-> arc of equal tangent),
H4 3a->3b (Curve Circuits claim), stability across jitter seeds and the LRN sensitivity run. Exploratory: |dtheta|
bins, dose-response, top-K by activation change, mean ablation, same-orientation sham and a family-rotation swap.
Global context: Huang et al. 2023's intervention mode (https://arxiv.org/abs/2309.10312) for a CNN neuron family.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import torch

from concept_swap import families as Fm, interventions as I, model as M, stats as St, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import interchange_experiment, make_split, save_json, success, summarize

log = get_logger(__name__)


def dtheta_bins(cfg: Config, ex: dict, rng) -> dict:
    """Family IIA per |dtheta| bin with pigeonhole 95% CIs (exploratory)."""
    e, p = ex["elig"], ex["pairs"]
    succ = success(ex, ex["outs"]["family"])
    null = success(ex, ex["nulls"]["delta_matched"]).mean(0)
    edges, out = [30, 60, 90, 120, 150, 181], {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = e & (np.abs(p.dtheta.values) >= lo) & (np.abs(p.dtheta.values) < hi)
        W = St.pigeonhole_weights(p.base.values[m], p.source.values[m], cfg.n_boot, rng)
        out[f"{lo}-{min(hi, 180)}"] = dict(iia=float(succ[m].mean()), ci=St.percentile_ci(St.weighted_rates(succ[m][None], W)[:, 0]),
                                         null_mean=float(null[m].mean()), n=int(m.sum()))
    return out


def dose_response(cfg: Config, ex: dict, n_sub: int = 100) -> dict:
    """IIA and normalized effect for random K-subsets of the family (descriptive)."""
    fam, e, k, out = ex["named"]["family"], ex["elig"], ex["elig"] & ex["keep"], {}
    rng = np.random.default_rng(cfg.confirm_seed)
    for K in (1, 2, 3, 5, 7, len(fam)):
        subs = list(combinations(fam, K))
        subs = [subs[i] for i in rng.choice(len(subs), min(n_sub, len(subs)), replace=False)]
        o = ex["run"](subs)
        iia = success(ex, o)[:, e].mean(1)
        eff = o[:, k, 1].sum(1) / ex["den"][k].sum()
        out[str(K)] = dict(iia_mean=float(iia.mean()), iia_range=St.percentile_ci(iia), effect_mean=float(eff.mean()),
                           effect_range=St.percentile_ci(eff), n_subsets=len(subs))
    return out


def delta_ranked(ex: dict) -> dict:
    """Exploratory (pilot-motivated): IIA when patching the top-K channels by selection-split activation change,
    with and without forcing the family in. Measures how distributed the orientation signal is."""
    order, fam, e = np.argsort(-ex["delta"]), list(ex["named"]["family"]), ex["elig"]
    rest = [c for c in order if c not in fam]
    out = {}
    for K in (10, 20, 40, 80, 160, 320, 480):
        sets = [order[:K], np.array(fam + rest[:K - len(fam)])]
        o = success(ex, ex["run"](sets))[:, e].mean(1)
        out[str(K)] = dict(top_delta=float(o[0]), family_plus_top_delta=float(o[1]))
    return out


def mean_ablation(cfg: Config, ex: dict, sel_windows, rng) -> dict:
    """Necessity-style check: replace the family (or null sets) by its selection-split mean window and ask how often
    the decoded orientation of the base survives (exploratory)."""
    mean = sel_windows.mean(0)
    e = ex["elig"]
    p = ex["pairs"]
    b = p.base.values
    sets = [ex["named"]["family"], ex["named"]["complement"]] + list(Fm.delta_matched_sets(ex["delta"], list(ex["named"]["family"]), 200, rng))
    o = ex["run"](sets, base=b, src=b, mode="mean", mean=mean)
    keep = success(ex, o, target="base")[:, e].mean(1)
    return dict(base_retained_family=float(keep[0]), base_retained_complement=float(keep[1]),
                base_retained_delta_null_mean=float(keep[2:].mean()), base_retained_delta_null_q05=float(np.quantile(keep[2:], 0.05)))


def family_rotation(ex: dict, pref: list[float], fam: list[int]) -> dict:
    """Exploratory J-lens/SAE-style activation-space concept swap: give each curve unit the map of the unit whose
    preferred orientation is k steps further around the circle (donor = the base itself). If downstream reads the
    family as an orientation code, the decoded orientation should shift by the preferred-orientation difference."""
    order = np.argsort(pref)
    srt, ps = np.array(fam)[order], np.array(pref)[order]
    e, p = ex["elig"], ex["pairs"]
    b = p.base.values
    out = {}
    for k in range(1, len(fam)):
        o = ex["run"]([srt], base=b, src=b, src_sets=[np.roll(srt, -k)])[0]
        shift_pred = float(np.degrees(np.angle(np.exp(1j * np.radians(ps - np.roll(ps, -k))).mean())))
        shift_obs = S.circ_diff(o[e, 0], ex["clean"][b][e])
        out[str(k)] = dict(pred_shift=shift_pred, obs_circ_mean=float(np.degrees(np.angle(np.exp(1j * np.radians(shift_obs)).mean()))),
                           frac_within_30_of_pred=float((np.abs(S.circ_diff(shift_obs, shift_pred)) <= 30).mean()),
                           frac_unchanged=float((np.abs(shift_obs) <= 10).mean()))
    return out


def sham(cfg: Config, model, ex: dict, test, stab) -> float:
    """Same-orientation sham: source = same (cell, theta) from another jitter split; the decoded base orientation
    should survive (fraction reported)."""
    W = torch.cat([test.col.windows, stab.col.windows])  # source rows are offset by len(test)
    n = len(test.grid)
    e = ex["elig"]
    b = ex["pairs"].base.values[e]
    rd = ex["readout"]
    reduce = lambda Rr: rd.predict_z(rd.z(Rr), test.grid.cell.values[b])
    th = I.patched_readouts(model, "mixed4a", W, b, b + n, [ex["named"]["family"]], rd.out_idx, reduce)[0]
    return float((np.abs(S.circ_diff(th, ex["clean"][b])) <= cfg.tol_deg).mean())


def curvature(cfg: Config, model, fam3b: list[int], fam4a: np.ndarray, rng, **kw) -> dict:
    """H2: base = straight line, source = arc with the same midpoint tangent (and the reverse). Label 0 = line,
    180 = arc (so the circular tolerance test doubles as class equality); templates compare line vs arc within the
    same (nuisance cell, theta)."""
    def split(seed):
        g = pd.concat([S.grid(cfg, seed, "line"), S.grid(cfg, seed, "arc")], ignore_index=True)
        g["cell"] = g.cell * 1000 + (g.theta / (360 / cfg.n_theta)).round().astype(int)  # cell x theta
        return make_split(cfg, model, "mixed4a", g)

    def pairs(g):
        n = len(g) // 2  # first half lines, second half arcs, same row order
        return pd.DataFrame(dict(base=np.r_[np.arange(n), np.arange(n) + n], source=np.r_[np.arange(n) + n, np.arange(n)],
                                 dtheta=0.0))

    sel, test = split(cfg.selection_seed), split(cfg.confirm_seed)
    lab = lambda g: np.where(g.kind.values == "arc", 180.0, 0.0)
    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, pairs(test.grid), pairs(sel.grid), fam3b, fam4a, rng,
                                cfg.n_null_sets // 2, labels_sel=lab(sel.grid), labels_test=lab(test.grid), **kw)
    out = summarize(cfg, ex, rng, "H2: curvature swap (line <-> arc), 3b curve family -> 4a curve family")
    d = ex["pairs"]
    for direction, m in (("line_to_arc", d.base.values < len(test.grid) // 2), ("arc_to_line", d.base.values >= len(test.grid) // 2)):
        out[f"iia_{direction}"] = float(success(ex, ex["outs"]["family"])[ex["elig"] & m].mean())
    return out


def run(cfg: Config, model, rng: np.random.Generator, e1: dict) -> dict:
    """Run all E2 analyses; save `results/e2_synthetic.json` and figures."""
    fam = {l: Fm.load_family(cfg.data_dir, l).neurons for l in ("mixed3a", "mixed3b", "mixed4a")}
    fam4a = np.array(fam["mixed4a"])
    g_sel, g_test = S.grid(cfg, cfg.selection_seed), S.grid(cfg, cfg.confirm_seed)
    p_sel, p_test = S.minimal_pairs(g_sel, cfg, cfg.selection_seed), S.minimal_pairs(g_test, cfg, cfg.confirm_seed)
    sel, test = make_split(cfg, model, "mixed4a", g_sel), make_split(cfg, model, "mixed4a", g_test)
    res = {}

    log.info("H1 (primary): interchange of the 3b curve family, readout = 4a curve family (template decoder)")
    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam["mixed3b"], fam4a, rng, cfg.n_null_sets)
    h1 = res["H1"] = summarize(cfg, ex, rng, "H1: 3b curve family -> 4a curve family")
    h1["closer_to_source"] = float((np.abs(S.circ_diff(ex["outs"]["family"][:, 0], ex["clean"][p_test.source.values]))
                                    < np.abs(S.circ_diff(ex["outs"]["family"][:, 0], ex["clean"][p_test.base.values])))[ex["elig"]].mean())
    h1["dtheta_bins"] = dtheta_bins(cfg, ex, rng)
    h1["dose_response"] = dose_response(cfg, ex)
    h1["delta_ranked"] = delta_ranked(ex)
    h1["mean_ablation"] = mean_ablation(cfg, ex, sel.col.windows, rng)
    h1["family_rotation"] = family_rotation(ex, e1["mixed3b"]["pref_theta_ours"], fam["mixed3b"])
    h1["delta_percentile_of_family"] = [float((ex["delta"] < ex["delta"][n]).mean()) for n in fam["mixed3b"]]
    stab = make_split(cfg, model, "mixed4a", S.grid(cfg, cfg.stability_seeds[0]))
    h1["sham_same_theta_base_retained"] = sham(cfg, model, ex, test, stab)
    log.info("H1 exploratory: closer-to-source %.3f, sham base-retained %.3f, mean-ablation %s", h1["closer_to_source"],
             h1["sham_same_theta_base_retained"], h1["mean_ablation"])

    log.info("H1 secondary: ridge readout over all 508 mixed4a channels (non-circular)")
    exr = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam["mixed3b"], np.arange(508), rng,
                                 cfg.n_null_sets // 4, kind="ridge")
    res["H1_ridge_all4a"] = summarize(cfg, exr, rng, "H1 (secondary): 3b curve family -> ridge over all 4a channels")
    del exr

    log.info("H2: curvature swap")
    res["H2"] = curvature(cfg, model, fam["mixed3b"], fam4a, rng)

    log.info("H4: 3a curve family -> 3b curve family (Curve Circuits claim)")
    sel3, test3 = make_split(cfg, model, "mixed3b", g_sel), make_split(cfg, model, "mixed3b", g_test)
    ex4 = interchange_experiment(cfg, model, "mixed3b", sel3, test3, p_test, p_sel, fam["mixed3a"], np.array(fam["mixed3b"]), rng,
                                 cfg.n_null_sets)
    res["H4"] = summarize(cfg, ex4, rng, "H4: 3a curve family -> 3b curve family")
    del ex4, sel3, test3

    for seed in cfg.stability_seeds:
        log.info("Stability: H1 on jitter seed %d", seed)
        g = S.grid(cfg, seed)
        exs = interchange_experiment(cfg, model, "mixed4a", sel, make_split(cfg, model, "mixed4a", g), S.minimal_pairs(g, cfg, seed),
                                     p_sel, fam["mixed3b"], fam4a, rng, cfg.n_null_sets // 4)
        res[f"H1_seed{seed}"] = summarize(cfg, exs, rng, f"H1 stability (jitter seed {seed})")
    del exs

    log.info("Sensitivity: H1 with Lucent's default LRN constants")
    lucent = M.load_inceptionv1(cfg.device, "lucent")
    exl = interchange_experiment(cfg, lucent, "mixed4a", make_split(cfg, lucent, "mixed4a", g_sel), make_split(cfg, lucent, "mixed4a", g_test),
                                 p_test, p_sel, fam["mixed3b"], fam4a, rng, cfg.n_null_sets // 4)
    res["H1_lrn_lucent"] = summarize(cfg, exl, rng, "H1 sensitivity (Lucent default LRN)")
    save_json(cfg, "e2_synthetic", res)
    return res
