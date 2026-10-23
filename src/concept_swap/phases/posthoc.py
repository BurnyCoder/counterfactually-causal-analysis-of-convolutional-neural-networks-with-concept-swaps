"""Post-hoc robustness phase (NOT pre-registered; added after the confirmatory run in response to code review).

Local: re-runs every hypothesis against a stricter **rank-matched** null (each curve unit matched to one of its nearest
non-family channels by activation change) and the most-changing competitor set (the 10 non-family channels with
the largest activation change; the audit phase later found a stronger, effect-ranked competitor); recomputes the H1-ridge effect in the ridge decoder's (cos, sin) output space; and
repeats H5 with blur-free rotations (square crops, exact 90/180/270-degree transposes).
Global context: the pre-registered decile-matched null leaves the family at the extreme tail of its bin, and the
H1-ridge effect and H5 rotations had interpretation caveats; this phase tests whether the conclusions survive them.
"""

from dataclasses import replace

import numpy as np

from concept_swap import families as Fm, interventions as I, natural as N, stats as St, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import interchange_experiment, make_split, save_json, summarize
from concept_swap.phases.e2_synthetic import curvature
from concept_swap.phases.e3_natural import composited, halves, rotation

log = get_logger(__name__)
RANK = dict(null_kinds=("rank_matched",))  # only the post-hoc null (the pre-registered nulls are in results/e2, e3)


def compact(r: dict) -> dict:
    """Keep the numbers the post-hoc report needs from a `summarize` output."""
    keys = [k for k in r if k.startswith(("iia_vs_", "effect_vs_"))]  # margins vs every null that was run
    return dict(label=r["label"], iia=r["iia"], effect=r["effect"], n_eligible=r["n_eligible"], **{k: r[k] for k in keys})


def cos_sin_effects(ex: dict, model, test, sets: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patch `sets` into the eligible pairs of a ridge-readout experiment and work in the decoder's (cos, sin) output
    space: returns projection numerators (n_sets, n_pairs), denominators |cs_source - cs_base|^2 (n_pairs,) and the
    clean source's (cos, sin) (n_pairs, 2), so callers can form effects and decoded-angle IIA."""
    rd, e, p = ex["readout"], ex["elig"], ex["pairs"]
    cs = lambda R: rd.dec.m.predict(rd.z(R))  # (n, 2) predicted (cos, sin)
    clean = cs(test.col.readout.numpy()[:, rd.out_idx])  # (cos, sin) predictions for every clean test stimulus
    b, s = p.base.values[e], p.source.values[e]  # eligible pairs only
    R = I.patched_readouts(model, "mixed4a", test.col.windows, b, s, sets, rd.out_idx, cs)  # (n_sets, n_pairs, 2)
    cb, csrc = clean[b], clean[s]  # clean base / source (cos, sin)
    return ((R - cb) * (csrc - cb)).sum(-1), ((csrc - cb) ** 2).sum(-1), R  # numerators, denominators, patched (cos, sin)


def ridge_output_effect(cfg: Config, ex: dict, model, test, rng, n_sets: int) -> dict:
    """H1-ridge effect measured in the ridge decoder's (cos theta, sin theta) output space, i.e. on the decoded
    orientation itself, for the family vs rank-matched and decile-matched null sets (eligible pairs only)."""
    fam = list(ex["named"]["family"])
    sets = ([np.array(fam)] + list(Fm.rank_matched_sets(ex["delta"], fam, n_sets, rng))  # [family, rank nulls..., decile nulls...]
            + list(Fm.delta_matched_sets(ex["delta"], fam, n_sets, rng)))
    num, den, _ = cos_sin_effects(ex, model, test, sets)
    eff = num.sum(1) / den.sum()  # ratio-of-sums effect per set
    b, s = ex["pairs"].base.values[ex["elig"]], ex["pairs"].source.values[ex["elig"]]  # eligible pairs
    W = St.pigeonhole_weights(b, s, cfg.n_boot, rng)  # crossed bootstrap over base and source images
    eff_b = (W @ num.T) / (W @ den)[:, None]  # (B, n_sets_total) bootstrap effects
    res = {}
    for name, sl in (("rank_matched", slice(1, 1 + n_sets)), ("delta_matched", slice(1 + n_sets, None))):
        res[name] = dict(rate=float(eff[0]), rate_ci=St.percentile_ci(eff_b[:, 0]), null_mean=float(eff[sl].mean()),
                         null_q95=float(np.quantile(eff[sl], 0.95)),
                         margin_ci=St.percentile_ci(eff_b[:, 0] - np.quantile(eff_b[:, sl], 0.95, axis=1)), p=St.perm_p(eff[0], eff[sl]))
    return res


def run(cfg: Config, model, rng: np.random.Generator) -> dict:
    """Run all post-hoc checks; save results/posthoc_robustness.json."""
    fam = {l: Fm.load_family(cfg.data_dir, l).neurons for l in ("mixed3a", "mixed3b", "mixed4a")}
    fam4a = np.array(fam["mixed4a"])
    n = cfg.n_null_sets // 2  # 1000 sets per post-hoc null
    g_sel, g_test = S.grid(cfg, cfg.selection_seed), S.grid(cfg, cfg.confirm_seed)  # the same splits as E2
    p_sel, p_test = S.minimal_pairs(g_sel, cfg, cfg.selection_seed), S.minimal_pairs(g_test, cfg, cfg.confirm_seed)
    sel, test = make_split(cfg, model, "mixed4a", g_sel), make_split(cfg, model, "mixed4a", g_test)
    res = {}  # test name -> compact result
    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam["mixed3b"], fam4a, rng, n, **RANK)
    res["H1"] = compact(summarize(cfg, ex, rng, "H1 vs rank-matched null"))
    exr = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam["mixed3b"], np.arange(508), rng, n // 2,
                                 kind="ridge", **RANK)
    res["H1_ridge_all4a"] = compact(summarize(cfg, exr, rng, "H1-ridge vs rank-matched null (z-space effect)"))
    res["H1_ridge_all4a"]["effect_in_cos_sin_space"] = ridge_output_effect(cfg, exr, model, test, rng, n // 2)
    log.info("post-hoc H1-ridge (cos, sin)-space effect: %s", res["H1_ridge_all4a"]["effect_in_cos_sin_space"])
    del ex, exr  # free per-set outcome arrays before the next tests
    res["H2"] = compact(curvature(cfg, model, fam["mixed3b"], fam4a, rng, **RANK))  # same H2 code, stricter null
    sel3, test3 = make_split(cfg, model, "mixed3b", g_sel), make_split(cfg, model, "mixed3b", g_test)  # 3a windows for H4
    ex4 = interchange_experiment(cfg, model, "mixed3b", sel3, test3, p_test, p_sel, fam["mixed3a"], np.array(fam["mixed3b"]), rng, n, **RANK)
    res["H4"] = compact(summarize(cfg, ex4, rng, "H4 vs rank-matched null"))
    del ex4, sel3, test3
    paths = N.list_images(N.ensure_imagenette(cfg))  # checksum-verified Imagenette validation images
    sel_paths, test_paths = halves(cfg, paths[::20] if cfg.quick else paths)  # same subsampling as E3 in --quick
    res["composited"] = compact(composited(cfg, model, fam["mixed3b"], fam4a, rng, sel_paths, **RANK))
    res["H5"] = compact(rotation(cfg, model, fam["mixed3b"], fam4a, rng, sel_paths, test_paths, **RANK))
    exact = replace(cfg, rotations=(-90, 90, 180))  # PIL rotates square images by these angles with exact transposes
    res["H5_exact_rotations"] = compact(rotation(exact, model, fam["mixed3b"], fam4a, rng, sel_paths, test_paths, square=True,
                                                 null_kinds=("delta_matched", "rank_matched")))
    save_json(cfg, "posthoc_robustness", res)
    return res
