"""Third-round audit phase (NOT pre-registered; post-hoc, exploratory checks added by the third independent audit).

Local: three checks of interpretive claims, whose overturning outcomes were committed to reports/posthoc_robustness.md
before this phase first ran on confirmatory data:
(g) the causal channel ranking redone under the whole-layer ridge readout (is the family special there too?);
(h) the H5 blur comparison redone on matched angles (does removing interpolation blur increase the family's effect?);
(i) additivity of channel-set patches resolved by mixed4a branch (how much of it is architectural?).
Global context: results go to results/posthoc_audit2.json and are discussed in reports/posthoc_robustness.md.
"""

from dataclasses import replace

import numpy as np

from concept_swap import families as Fm, interventions as I, model as M, natural as N, stats as St, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.audit import NONE
from concept_swap.phases.common import interchange_experiment, make_split, save_json, success
from concept_swap.phases.e3_natural import rotation_experiment
from concept_swap.phases.posthoc import cos_sin_effects, decoded_iia

log = get_logger(__name__)


def ridge_effect_ranked(cfg: Config, model, exs: dict, sel, exr: dict, test, fam: list[int], rng) -> dict:
    """(g) Rank every mixed3b channel by its single-channel effect in the ridge decoder's (cos, sin) output on the
    selection split (`exs`), then score the family, the top-10 and the top-10 non-family on the confirmatory pairs
    (`exr`). The ridge analogue of `audit.effect_ranked`, which used the 14-unit curve readout."""
    num, den, _ = cos_sin_effects(exs, model, sel, [[c] for c in range(sel.col.windows.shape[1])])  # selection only
    order = np.argsort(-(num.sum(1) / den.sum()))  # channels by decreasing single-channel (cos, sin) effect
    top, top_nf = order[:len(fam)], order[~np.isin(order, fam)][:len(fam)]  # top-10, and top-10 without the family
    num, den, R = cos_sin_effects(exr, model, test, [np.array(fam), top, top_nf])  # confirmatory eligible pairs
    p, e = exr["pairs"], exr["elig"]
    W = St.pigeonhole_weights(p.base.values[e], p.source.values[e], cfg.n_boot, rng)  # crossed bootstrap (Owen 2007)
    eff_b = (W @ num.T) / (W @ den)[:, None]  # (B, 3) bootstrap ratio-of-sums effects
    names = ["family", "top10_by_effect", "top10_nonfamily_by_effect"]
    return dict(family_ranks=sorted(int(np.flatnonzero(order == c)[0]) + 1 for c in fam), top10_by_effect=top.tolist(),
                top10_nonfamily_by_effect=top_nf.tolist(), overlap_top10_family=len(set(top) & set(fam)),
                effect=dict(zip(names, (num.sum(1) / den.sum()).tolist())), iia=dict(zip(names, decoded_iia(exr, R, cfg.tol_deg).tolist())),
                effect_diff_ci={n: St.percentile_ci(eff_b[:, 0] - eff_b[:, j]) for j, n in enumerate(names) if j}, n_pairs=int(len(den)))


def additivity_by_branch(model, exr: dict, test, fam4a: np.ndarray) -> dict:
    """(i) Relative non-additivity sum|D_F + D_C - D_all|^2 / sum|D_all|^2 of the z-scored mixed4a readout, where D_S
    is the change caused by patching set S (family F, complement C, all channels) into the base, per mixed4a branch and
    for the 6 1x1-branch and 8 5x5-branch curve readout units. 0 = exactly additive. Also the family's and complement's
    normalized effects per group (eligible pairs, no denominator filter), whose sum is 1 when patches add exactly."""
    rd, e, p = exr["readout"], exr["elig"], exr["pairs"]  # ridge experiment: z-statistics for all 508 channels
    b, s = p.base.values[e], p.source.values[e]
    named = exr["named"]
    z0, zF, zC, zA = I.patched_readouts(model, "mixed4a", test.col.windows, b, s, [[], named["family"], named["complement"], named["all"]],
                                        rd.out_idx, rd.z)  # patching nothing / F / C / everything, same code path
    dF, dC, dA = zF - z0, zC - z0, zA - z0
    groups = {k: np.arange(sl.start, sl.stop) for k, sl in M.branch_slices("mixed4a").items()}  # 4a branches
    groups |= {f"curve_units_{b}": fam4a[np.isin(fam4a, groups[b])] for b in ("1x1", "5x5")} | {"curve_units_all": fam4a}
    ratio = lambda a, g: float((a[:, g] ** 2).sum() / (dA[:, g] ** 2).sum())  # squared size relative to the full change
    proj = lambda d, g: float((d[:, g] * dA[:, g]).sum() / (dA[:, g] ** 2).sum())  # normalized effect within the group
    return {k: dict(n_channels=int(len(g)), nonadditivity=ratio(dF + dC - dA, g), effect_family=proj(dF, g),
                    effect_complement=proj(dC, g)) for k, g in groups.items()} | dict(n_pairs=int(len(b)))


def h5_blur_matched(cfg: Config, model, fam3b, fam4a, rng, sel_paths: list, test_paths: list, angles=(-90, 90, 180)) -> dict:
    """(h) The family's H5 effect and IIA per angle with bilinear-capable rotations ("original", as in the pre-registered
    H5) and with exact transposes of square crops ("exact"), on the same curve-present photos and the same angles, plus
    a photo-cluster bootstrap CI of the pooled effect difference. Only the family is patched (no nulls)."""
    c = replace(cfg, rotations=tuple(angles))  # the exact run's angles, so both modes have the same pairs and templates
    runs = {mode: rotation_experiment(c, model, fam3b, fam4a, rng, sel_paths, test_paths, square=sq, **NONE)["ex"]
            for mode, sq in (("original", False), ("exact", True))}
    p = runs["original"]["pairs"]
    assert p.equals(runs["exact"]["pairs"]), "both modes must use the same photos, angles and pair order"
    W = St.cluster_weights(p.base.values, cfg.n_boot, rng)  # resample photos (base row = photo); shared by both modes
    out, eff_b = {}, {}
    for mode, ex in runs.items():
        o, e, k = ex["outs"]["family"], ex["elig"], ex["elig"] & ex["keep"]
        succ = success(ex, o)  # per-pair IIA success of the family patch
        eff = lambda m: float(o[m, 1].sum() / ex["den"][m].sum())  # ratio-of-sums effect over mask m
        at = lambda a: p.dtheta.values == a  # pairs with rotation angle a
        out[mode] = dict(effect=eff(k), iia=float(succ[e].mean()),
                         by_angle={str(a): dict(effect=eff(k & at(a)), iia=float(succ[e & at(a)].mean())) for a in angles})
        eff_b[mode] = (W[:, k] @ o[k, 1]) / (W[:, k] @ ex["den"][k])  # bootstrap effects under the shared photo draws
    return out | dict(effect_diff=out["exact"]["effect"] - out["original"]["effect"],
                      effect_diff_ci=St.percentile_ci(eff_b["exact"] - eff_b["original"]),
                      n_images=int(p.base.nunique()), n_pairs=int(len(p)), angles=list(angles))


def run(cfg: Config, model, rng: np.random.Generator) -> dict:
    """Run the third-round audit checks on the E2/E3 splits; save results/posthoc_audit2.json."""
    fam3b, fam4a = Fm.load_family(cfg.data_dir, "mixed3b").neurons, np.array(Fm.load_family(cfg.data_dir, "mixed4a").neurons)
    g_sel, g_test = S.grid(cfg, cfg.selection_seed), S.grid(cfg, cfg.confirm_seed)  # the E2 splits
    p_sel, p_test = S.minimal_pairs(g_sel, cfg, cfg.selection_seed), S.minimal_pairs(g_test, cfg, cfg.confirm_seed)
    sel, test = make_split(cfg, model, "mixed4a", g_sel), make_split(cfg, model, "mixed4a", g_test)
    n4a = test.col.readout.shape[1]  # 508 mixed4a channels
    ridge = lambda t, pt: interchange_experiment(cfg, model, "mixed4a", sel, t, pt, p_sel, fam3b, np.arange(n4a), rng, 0,
                                                 kind="ridge", **NONE)  # ridge readout fitted on selection, no nulls
    exs, exr = ridge(sel, p_sel), ridge(test, p_test)  # selection pairs (ranking) and confirmatory pairs (scoring)
    res = {"ridge_effect_ranked": ridge_effect_ranked(cfg, model, exs, sel, exr, test, fam3b, rng)}
    log.info("audit2 (g) ridge-readout causal ranking: %s", res["ridge_effect_ranked"])
    res["additivity_by_branch"] = additivity_by_branch(model, exr, test, fam4a)
    log.info("audit2 (i) additivity by mixed4a branch: %s", res["additivity_by_branch"])
    del exs, exr  # free per-set outcome arrays before the natural-image check
    paths = N.list_images(N.ensure_imagenette(cfg))  # checksum-verified Imagenette validation images
    sel_paths, test_paths = N.image_halves(cfg, paths)  # the same halves as E3
    res["h5_blur_matched"] = h5_blur_matched(cfg, model, fam3b, fam4a, rng, sel_paths, test_paths)
    log.info("audit2 (h) H5 blur on matched angles: %s", res["h5_blur_matched"])
    save_json(cfg, "posthoc_audit2", res)
    return res
