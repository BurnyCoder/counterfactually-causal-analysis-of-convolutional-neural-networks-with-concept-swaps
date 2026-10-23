"""Post-hoc audit phase (NOT pre-registered; added after an independent scientific audit of the confirmatory run).

Local: six checks of interpretive claims that the pre-registered analyses did not test directly:
(a) a causal, data-driven competitor: channels ranked by single-channel patching effect on the selection split;
(b) the complement's effect on the ridge-decoded orientation (is the code "redundant" under the whole-layer readout?);
(c) H2 per swap direction against that direction's own null; (d) mean ablation against the stricter rank-matched
null; (e) a cross-nuisance sham (same orientation, other radius / width / polarity); (f) whether the relevance
test's null ablations match the family's ablation size (variance).
Global context: results go to results/posthoc_audit.json and are discussed in reports/posthoc_robustness.md. The
overturning outcomes of (a)-(d) were set in the audit plan before this phase ran, but were first committed together
with its results, so git does not timestamp them; (e) and (f) have no overturning rule.
"""

import numpy as np

from concept_swap import families as Fm, interventions as I, natural as N, stats as St, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import interchange_experiment, make_split, save_json, success
from concept_swap.phases.e2_synthetic import curvature, mean_ablation
from concept_swap.phases.e3_natural import halves, mixed3b_moments
from concept_swap.phases.posthoc import cos_sin_effects, decoded_iia

log = get_logger(__name__)
NONE = dict(null_kinds=())  # score only the named sets: no null sampling, so no RNG draws


def effect_ranked(cfg: Config, model, sel, ex: dict, p_sel, fam: list[int], fam4a: np.ndarray, rng) -> dict:
    """(a) Rank every mixed3b channel by its single-channel patching effect on the selection split, then score the
    family against the top-10 by effect and the top-10 non-family by effect on the confirmatory H1 pairs.

    Local: only the selection split is used for ranking. It differs from the confirmatory split only by a +-2 px jitter
    per nuisance cell, so the ranking is close to in-sample for the confirmatory stimuli, which favours the competitor.
    Global context: a causal analogue of Fisher selection (Baeumel et al. 2025, https://arxiv.org/abs/2508.02513)
    and the strongest competitor set for the claim that the published family is special.
    """
    exs = interchange_experiment(cfg, model, "mixed4a", sel, sel, p_sel, p_sel, fam, fam4a, rng, 0, **NONE)  # selection only
    ks = exs["elig"] & exs["keep"]  # selection pairs used for the effect
    eff1 = exs["run"]([[c] for c in range(sel.col.windows.shape[1])])[:, ks, 1].sum(1) / exs["den"][ks].sum()  # per channel
    order = np.argsort(-eff1)  # channels by decreasing single-channel effect
    top, top_nf = order[:len(fam)], order[~np.isin(order, fam)][:len(fam)]  # top-10, and top-10 without the family
    e, k, p = ex["elig"], ex["elig"] & ex["keep"], ex["pairs"]  # confirmatory eligibility / effect masks and pairs
    o = ex["run"]([np.array(fam), top, top_nf])  # (3, n_pairs, 2) confirmatory outcomes
    W = St.pigeonhole_weights(p.base.values[k], p.source.values[k], cfg.n_boot, rng)  # crossed bootstrap (effect pairs)
    eff_b = (W @ o[:, k, 1].T) / (W @ ex["den"][k])[:, None]  # (B, 3) bootstrap ratio-of-sums effects
    names = ["family", "top10_by_effect", "top10_nonfamily_by_effect"]
    topk = {str(K): float(success(ex, ex["run"]([order[:K]]))[0, e].mean()) for K in (10, 20, 40, 80, 160)}  # IIA of top-K
    return dict(family_ranks=sorted(int(np.flatnonzero(order == c)[0]) + 1 for c in fam), top10_by_effect=top.tolist(),
                top10_nonfamily_by_effect=top_nf.tolist(), overlap_top10_family=len(set(top) & set(fam)),
                selection_single_effect_family_mean=float(eff1[fam].mean()),
                iia=dict(zip(names, success(ex, o)[:, e].mean(1).tolist())),
                effect=dict(zip(names, (o[:, k, 1].sum(1) / ex["den"][k].sum()).tolist())),
                effect_diff_ci={n: St.percentile_ci(eff_b[:, 0] - eff_b[:, j]) for j, n in enumerate(names) if j},
                iia_topk_by_effect=topk)


def ridge_complement(cfg: Config, model, sel, test, p_sel, p_test, fam: list[int], rng) -> dict:
    """(b) Family vs complement under the pre-registered whole-layer ridge readout, in the decoder's (cos, sin)
    output space (effect) and on the decoded angle (IIA)."""
    exr = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam, np.arange(test.col.readout.shape[1]), rng, 0,
                                 kind="ridge", **NONE)
    num, den, R = cos_sin_effects(exr, model, test, [exr["named"]["family"], exr["named"]["complement"]])
    iia = decoded_iia(exr, R, cfg.tol_deg)  # (2,) family, complement
    return dict(cos_sin_effect=dict(family=float(num[0].sum() / den.sum()), complement=float(num[1].sum() / den.sum())),
                iia=dict(family=float(iia[0]), complement=float(iia[1])), n_pairs=int(len(den)))


def nuisance_sham(cfg: Config, model, test, ex: dict) -> dict:
    """(e) Cross-nuisance sham: the source has the base's orientation but another radius, width or polarity; the
    fraction of stimuli whose decoded orientation survives the swap (family, complement, all channels)."""
    rd, cells, b = ex["readout"], test.grid.cell.values, np.arange(len(test.grid))  # every stimulus is a base once
    sets, out = [ex["named"][n] for n in ("family", "complement", "all")], {}
    for factor in ("radius", "width", "polarity"):
        src = S.nuisance_swap_pairs(test.grid, factor)  # same theta, only `factor` changed
        th = I.patched_readouts(model, "mixed4a", test.col.windows, b, src, sets, rd.out_idx,
                                lambda R: rd.predict_z(rd.z(R), cells))  # decoded with the base's cell templates
        m = ex["acc"] & ex["acc"][src]  # both clean stimuli decode correctly (the H1 eligibility rule)
        kept = (np.abs(S.circ_diff(th[:, m], ex["clean"][m])) <= cfg.tol_deg).mean(1)  # base orientation retained
        out[factor] = dict(zip(("family", "complement", "all"), kept.tolist()), n=int(m.sum()))
    return out


def relevance_size_check(cfg: Config, model, fam: list[int], rng, sel_paths: list, n_sets: int) -> dict:
    """(f) Is the relevance null matched on ablation size? Family variance percentiles and family / null mean
    variance, for decile-matched sets drawn exactly as `e3_natural.relevance` draws them."""
    _, var = mixed3b_moments(cfg, model, sel_paths)  # the variance `relevance` matches on
    null = var[Fm.delta_matched_sets(var, fam, n_sets, rng)].mean(1)  # mean variance of each null set
    return dict(family_var_percentiles=[float((var < var[c]).mean()) for c in fam], family_mean_var=float(var[fam].mean()),
                null_mean_var_mean=float(null.mean()), null_mean_var_q95=float(np.quantile(null, 0.95)),
                family_over_null_ratio=float(var[fam].mean() / null.mean()))


def run(cfg: Config, model, rng: np.random.Generator) -> dict:
    """Run the audit checks on the same splits as E2/E3; save results/posthoc_audit.json."""
    fam3b, fam4a = Fm.load_family(cfg.data_dir, "mixed3b").neurons, np.array(Fm.load_family(cfg.data_dir, "mixed4a").neurons)
    g_sel, g_test = S.grid(cfg, cfg.selection_seed), S.grid(cfg, cfg.confirm_seed)  # the E2 splits
    p_sel, p_test = S.minimal_pairs(g_sel, cfg, cfg.selection_seed), S.minimal_pairs(g_test, cfg, cfg.confirm_seed)
    sel, test = make_split(cfg, model, "mixed4a", g_sel), make_split(cfg, model, "mixed4a", g_test)
    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam3b, fam4a, rng, 0, **NONE)  # H1 pairs
    res = {"effect_ranked": effect_ranked(cfg, model, sel, ex, p_sel, fam3b, fam4a, rng)}
    log.info("audit (a) effect-ranked competitor: %s", res["effect_ranked"])
    res["nuisance_sham"] = nuisance_sham(cfg, model, test, ex)
    log.info("audit (e) cross-nuisance sham, base orientation retained: %s", res["nuisance_sham"])
    abl = mean_ablation(cfg, ex, sel.col.windows, rng, sampler=Fm.rank_matched_sets)
    res["mean_ablation_rank_null"] = {k.replace("delta_null", "rank_null"): v for k, v in abl.items()}  # name the null used
    log.info("audit (d) mean ablation vs rank-matched null: %s", res["mean_ablation_rank_null"])
    res["ridge_complement"] = ridge_complement(cfg, model, sel, test, p_sel, p_test, fam3b, rng)
    log.info("audit (b) ridge readout, family vs complement: %s", res["ridge_complement"])
    del ex  # free the cached outcome arrays before H2
    h2 = curvature(cfg, model, fam3b, fam4a, rng, null_kinds=("delta_matched",))  # same H2 code, decile-matched null
    res["H2_by_direction"] = {k: v for k, v in h2.items() if k.startswith(("iia_line", "iia_arc", "line_to", "arc_to"))}
    log.info("audit (c) H2 per direction: %s", res["H2_by_direction"])
    paths = N.list_images(N.ensure_imagenette(cfg))  # checksum-verified Imagenette validation images
    sel_paths, _ = halves(cfg, paths[::20] if cfg.quick else paths)  # same subsampling as E3 in --quick
    res["relevance_size_check"] = relevance_size_check(cfg, model, fam3b, rng, sel_paths, 20 if cfg.quick else 100)
    log.info("audit (f) relevance null ablation size: %s", res["relevance_size_check"])
    save_json(cfg, "posthoc_audit", res)
    return res
