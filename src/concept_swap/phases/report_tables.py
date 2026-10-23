"""Report phase: turn results/*.json into Markdown tables (reports/generated/results.md) and the summary figure.

Local: no computation beyond formatting and the pre-registered Holm correction across secondary hypotheses.
Global context: every number quoted in reports/*.md and the paper is copied from this generated file, so the prose
can be re-checked against the JSONs mechanically.
"""

import json

import numpy as np

from concept_swap.config import Config, ROOT
from concept_swap.log import get_logger
from concept_swap.plots import plot_effects, plot_null, plot_points
from concept_swap.stats import holm

log = get_logger(__name__)
SECONDARY = ("H1_ridge_all4a", "H2", "H4", "composited", "H5")  # Holm family (reports/preregistration.md)
SHORT = {"H1": "H1 3b->4a curves (primary)", "H1_ridge_all4a": "H1 3b->ridge(all 4a)", "H2": "H2 curvature swap", "H4": "H4 3a->3b curves",
         "H1_lrn_lucent": "H1, Lucent LRN", "composited": "H1, natural backgrounds", "H5": "H5 natural rotation"}  # figure labels


def _f(x, d=3) -> str:
    """Fixed-point number with d decimals (one formatting rule for every table and macro)."""
    return f"{x:.{d}f}"


def _ci(t) -> str:
    """A (low, high) interval as "[low, high]" with 3 decimals."""
    return f"[{t[0]:.3f}, {t[1]:.3f}]"


def hypothesis_rows(res: dict) -> list[str]:
    """One Markdown row per hypothesis result dict (IIA and continuous effect vs both nulls)."""
    rows = []
    for key, r in res.items():  # d/b: IIA vs decile/branch nulls; e/eb: effect vs decile/branch nulls
        d, b, e, eb = r["iia_vs_delta_matched"], r["iia_vs_branch_matched"], r["effect_vs_delta_matched"], r["effect_vs_branch_matched"]
        rows.append(f"| {key} | {r['label']} | {r['readout']} | {_f(r['clean_acc'])} | {r['n_eligible']} | {_f(d['rate'])} {_ci(d['rate_ci'])} "
                    f"| {_f(d['null_q95'])} | {_f(d['margin'])} {_ci(d['margin_ci'])} | {d['p']:.2g} | {_f(b['null_q95'])} | {b['p']:.2g} "
                    f"| {_f(e['rate'])} {_ci(e['rate_ci'])} | {_f(e['null_q95'])} | {_f(e['margin'])} {_ci(e['margin_ci'])} | {e['p']:.2g} "
                    f"| {_f(eb['null_q95'])} | {r.get('verdict_secondary', r['verdict'])} |")
    return rows


MACRO = {"H1": "HOne", "H1_ridge_all4a": "HRidge", "H2": "HTwo", "H4": "HFour", "composited": "HComp", "H5": "HFive",
         "H1_seed3": "HSeedThree", "H1_seed4": "HSeedFour", "H1_lrn_lucent": "HLrn"}  # LaTeX-safe prefixes


def latex_numbers(e1: dict, e2: dict, e3: dict, hyp: dict) -> str:
    """`\\newcommand` definitions for every number the paper quotes (reports/generated/numbers.tex)."""
    m = {}  # macro name -> formatted value
    for l, n in (("mixed3a", "ThreeA"), ("mixed3b", "ThreeB"), ("mixed4a", "FourA")):  # E1 numbers per layer
        m[f"EOneR{n}"], m[f"EOneNull{n}"] = _f(e1[l]["mean_r"]), _f(e1[l]["perm_null_q95"])  # replication r, null q95
        m[f"EOneLucentR{n}"] = _f(e1["lrn_lucent_mean_r"][l])  # same correlation with Lucent's default LRN
    for k, p in MACRO.items():  # one block of macros per hypothesis, e.g. \HOneIIA, \HOneEffNullQ
        if k not in hyp:
            continue  # e.g. --quick runs use other stability-seed keys
        h, d, e = hyp[k], hyp[k]["iia_vs_delta_matched"], hyp[k]["effect_vs_delta_matched"]
        m |= {f"{p}IIA": _f(d["rate"]), f"{p}IIALo": _f(d["rate_ci"][0]), f"{p}IIAHi": _f(d["rate_ci"][1]), f"{p}NullQ": _f(d["null_q95"]),
              f"{p}BranchQ": _f(h["iia_vs_branch_matched"]["null_q95"]), f"{p}P": f"{d['p']:.2g}", f"{p}Eff": _f(e["rate"]),
              f"{p}EffLo": _f(e["rate_ci"][0]), f"{p}EffHi": _f(e["rate_ci"][1]), f"{p}EffNullQ": _f(e["null_q95"]),
              f"{p}EffBranchQ": _f(h["effect_vs_branch_matched"]["null_q95"]), f"{p}EffMarginLo": _f(e["margin_ci"][0]),
              f"{p}EffP": f"{e['p']:.2g}", f"{p}Compl": _f(h["iia"]["complement"]), f"{p}All": _f(h["iia"]["all"]),
              f"{p}Fisher": _f(h["iia"]["fisher_topk"]), f"{p}Clean": _f(h["clean_acc"]), f"{p}NElig": str(h["n_eligible"]),
              f"{p}ComplEff": _f(h["effect"]["complement"]), f"{p}Verdict": h.get("verdict_secondary", h["verdict"]),
              f"{p}Holm": f"{h['holm_p_effect']:.2g}" if "holm_p_effect" in h else "--"}
    h1, rel = e2["H1"], e3["relevance"]  # exploratory H1 analyses and the natural-image relevance check
    m |= {"HOneCloser": _f(h1["closer_to_source"]), "HOneSham": _f(h1["sham_same_theta_base_retained"]),
          "HOneAblFam": _f(h1["mean_ablation"]["base_retained_family"]), "HOneAblNull": _f(h1["mean_ablation"]["base_retained_delta_null_mean"]),
          "HOneAblCompl": _f(h1["mean_ablation"]["base_retained_complement"]), "HTwoLineToArc": _f(e2["H2"]["iia_line_to_arc"]),
          "HTwoArcToLine": _f(e2["H2"]["iia_arc_to_line"]), "HFiveNImg": str(e3["H5"]["n_images_test"]),
          "HFiveAbsentEff": _f(e3["H5"]["curve_absent_contrast"]["effect"]), "HFiveAbsentIIA": _f(e3["H5"]["curve_absent_contrast"]["iia"]),
          "RelFlipFam": _f(rel["flip_family"]), "RelFlipNull": _f(rel["flip_null_mean"]), "RelFlipQ": _f(rel["flip_null_q95"]),
          "RelFlipP": f"{rel['flip_p']:.2g}", "RelKLFam": _f(rel["kl_family"], 4), "RelKLNull": _f(rel["kl_null_mean"], 4),
          "RelKLP": f"{rel['kl_p']:.2g}", "RelN": str(rel["n_images"])}
    words = {"1": "One", "2": "Two", "3": "Three", "5": "Five", "7": "Seven", "10": "Ten", "20": "Twenty", "40": "Forty",
             "80": "Eighty", "160": "OneSixty"}  # digits are not allowed in LaTeX macro names
    m |= {f"DoseEff{words[k]}": _f(v["effect_mean"]) for k, v in h1["dose_response"].items()}
    m |= {f"DoseIIA{words[k]}": _f(v["iia_mean"]) for k, v in h1["dose_response"].items()}
    m |= {f"TopK{words[k]}": _f(v["top_delta"]) for k, v in h1["delta_ranked"].items() if k in words}
    m |= {f"DTheta{chr(65 + i)}": _f(v["iia"]) for i, v in enumerate(h1["dtheta_bins"].values())}  # A = 30-60 ... E = 150-180
    rot = h1["family_rotation"]["1"]  # one-step family rotation (quoted in the paper)
    m |= {"RotPredOne": f"{rot['pred_shift']:.0f}", "RotObsOne": f"{rot['obs_circ_mean']:.0f}",
          "HFiveIIAOneEighty": _f(e3["H5"]["iia_by_angle"]["180"]), "HFiveIIAFortyFive": _f(e3["H5"]["iia_by_angle"]["45"])}
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"  # one \newcommand per line


PH_MACRO = {"H1": "PHOne", "H1_ridge_all4a": "PHRidge", "H2": "PHTwo", "H4": "PHFour", "composited": "PHComp", "H5": "PHFive",
            "H5_exact_rotations": "PHFiveExact"}  # LaTeX prefixes of the post-hoc numbers


def posthoc_section(ph: dict) -> tuple[list[str], dict]:
    """Markdown lines and LaTeX macros for results/posthoc_robustness.json (post-hoc, not pre-registered)."""
    L = ["", "## Post-hoc robustness (NOT pre-registered; added after code review)", "",
         "Rank-matched null: each curve unit is matched to one of its 10 nearest non-family channels by activation change "
         "(1000 sets; 500 for the ridge readout). Top-Δ non-family: the 10 non-family channels with the largest activation change. "
         "H5-exact: square crops rotated by -90/90/180 deg (exact transposes, no interpolation blur).", "",
         "| key | IIA family | rank-null q95 (IIA) | effect family [CI] | rank-null q95 (effect) | effect margin [CI] | p | top-Δ non-family IIA / effect |",
         "|---|---|---|---|---|---|---|---|"]
    m = {}
    for k, r in ph.items():  # one table row and macro block per post-hoc test
        d, e = r["iia_vs_rank_matched"], r["effect_vs_rank_matched"]
        L.append(f"| {k} | {_f(d['rate'])} | {_f(d['null_q95'])} | {_f(e['rate'])} {_ci(e['rate_ci'])} | {_f(e['null_q95'])} | "
                 f"{_f(e['margin'])} {_ci(e['margin_ci'])} | {e['p']:.2g} | {_f(r['iia']['top_delta_nonfamily'])} / {_f(r['effect']['top_delta_nonfamily'])} |")
        p = PH_MACRO[k]
        m |= {f"{p}IIA": _f(d["rate"]), f"{p}IIANullQ": _f(d["null_q95"]), f"{p}Eff": _f(e["rate"]), f"{p}EffNullQ": _f(e["null_q95"]),
              f"{p}MarginLo": _f(e["margin_ci"][0]), f"{p}TopIIA": _f(r["iia"]["top_delta_nonfamily"]),
              f"{p}TopEff": _f(r["effect"]["top_delta_nonfamily"]), f"{p}Compl": _f(r["iia"]["complement"])}
    cs = ph["H1_ridge_all4a"]["effect_in_cos_sin_space"]  # ridge effect on the decoded orientation itself
    L += ["", f"H1-ridge effect in the decoder's (cos, sin) output space: family {_f(cs['rank_matched']['rate'])} "
              f"{_ci(cs['rank_matched']['rate_ci'])}; rank-null q95 {_f(cs['rank_matched']['null_q95'])} (p={cs['rank_matched']['p']:.2g}); "
              f"decile-null q95 {_f(cs['delta_matched']['null_q95'])} (p={cs['delta_matched']['p']:.2g})."]
    ex = ph["H5_exact_rotations"]["effect_vs_delta_matched"]  # blur-free H5 also compared with the decile null
    L += [f"H5-exact vs the decile-matched null: effect q95 {_f(ex['null_q95'])}, margin {_ci(ex['margin_ci'])}, p={ex['p']:.2g}."]
    m |= {"PHRidgeCSEff": _f(cs["rank_matched"]["rate"]), "PHRidgeCSRankQ": _f(cs["rank_matched"]["null_q95"]),
          "PHRidgeCSDeltaQ": _f(cs["delta_matched"]["null_q95"]), "PHFiveExactDeltaQ": _f(ex["null_q95"])}
    return L, m


def audit_section(a: dict) -> tuple[list[str], dict]:
    """Markdown lines and LaTeX macros for results/posthoc_audit.json (post-hoc checks added by the independent audit)."""
    er, rc, sh, ab, h2, rs = (a[k] for k in ("effect_ranked", "ridge_complement", "nuisance_sham", "mean_ablation_rank_null",
                                             "H2_by_direction", "relevance_size_check"))
    L = ["", "## Independent audit checks (NOT pre-registered; post-hoc, exploratory)", "",
         f"Effect-ranked channels: all mixed3b channels ranked by single-channel patching effect on the selection split. Family ranks: "
         f"{er['family_ranks']}; top-10 by effect: {er['top10_by_effect']} ({er['overlap_top10_family']} of 10 are family units).", "",
         "| set (ranked on selection, scored on confirmatory H1 pairs) | IIA | effect | effect difference, family minus set [CI] |", "|---|---|---|---|"]
    L += [f"| {n} | {_f(er['iia'][n])} | {_f(er['effect'][n])} | {_ci(er['effect_diff_ci'][n]) if n in er['effect_diff_ci'] else '—'} |" for n in er["iia"]]
    L += ["", "| K (top channels by single-channel effect) | IIA |", "|---|---|"] + [f"| {k} | {_f(v)} |" for k, v in er["iia_topk_by_effect"].items()]
    L += ["", f"- Ridge readout over all mixed4a channels, family vs complement (470 channels), {rc['n_pairs']} eligible pairs: effect in the "
              f"decoder's (cos, sin) space {_f(rc['cos_sin_effect']['family'])} vs {_f(rc['cos_sin_effect']['complement'])}; decoded-angle IIA "
              f"{_f(rc['iia']['family'])} vs {_f(rc['iia']['complement'])}"]
    for d in ("line_to_arc", "arc_to_line"):  # H2 against each direction's own decile-matched null
        r = h2[f"{d}_vs_delta_matched"]
        L.append(f"- H2 {d.replace('_to_', '->')}: IIA family {_f(h2[f'iia_{d}'])} vs null mean {_f(r['null_iia_mean'])}, q95 {_f(r['null_iia_q95'])} "
                 f"(p={r['iia_p']:.2g}); effect family {_f(r['effect_family'])} vs null q95 {_f(r['null_effect_q95'])} (p={r['effect_p']:.2g})")
    L += [f"- Mean ablation vs rank-matched null (200 sets), base orientation retained: family {_f(ab['base_retained_family'])}, "
          f"complement {_f(ab['base_retained_complement'])}, rank-null mean {_f(ab['base_retained_rank_null_mean'])} (q05 {_f(ab['base_retained_rank_null_q05'])})",
          "- Cross-nuisance sham (source = same orientation, another radius / width / polarity), base orientation retained: "
          + "; ".join(f"{f}: family {_f(v['family'])}, complement {_f(v['complement'])}, all {_f(v['all'])} (n {v['n']})" for f, v in sh.items()),
          f"- Relevance null size check: family variance percentiles {[round(v, 2) for v in rs['family_var_percentiles']]}; family mean variance / "
          f"matched-null mean variance = {_f(rs['family_over_null_ratio'], 2)}"]
    la, al = h2["line_to_arc_vs_delta_matched"], h2["arc_to_line_vs_delta_matched"]
    m = {"AuTopOverlap": str(er["overlap_top10_family"]), "AuFamRankMax": str(max(er["family_ranks"])),
         "AuTopEff": _f(er["effect"]["top10_by_effect"]), "AuTopIIA": _f(er["iia"]["top10_by_effect"]),
         "AuTopNFEff": _f(er["effect"]["top10_nonfamily_by_effect"]), "AuTopNFIIA": _f(er["iia"]["top10_nonfamily_by_effect"]),
         "AuDiffNFLo": _f(er["effect_diff_ci"]["top10_nonfamily_by_effect"][0]), "AuDiffNFHi": _f(er["effect_diff_ci"]["top10_nonfamily_by_effect"][1]),
         "AuRidgeFamCS": _f(rc["cos_sin_effect"]["family"]), "AuRidgeComplCS": _f(rc["cos_sin_effect"]["complement"]),
         "AuRidgeFamIIA": _f(rc["iia"]["family"]), "AuRidgeComplIIA": _f(rc["iia"]["complement"]),
         "AuLineArcNullMean": _f(la["null_iia_mean"]), "AuLineArcNullQ": _f(la["null_iia_q95"]), "AuLineArcP": f"{la['iia_p']:.2g}",
         "AuLineArcEff": _f(la["effect_family"]), "AuLineArcEffNullQ": _f(la["null_effect_q95"]),
         "AuArcLineEff": _f(al["effect_family"]), "AuArcLineEffNullQ": _f(al["null_effect_q95"]),
         "AuAblRankNull": _f(ab["base_retained_rank_null_mean"]), "AuAblRankNullQ": _f(ab["base_retained_rank_null_q05"]),
         "AuShamMin": _f(min(v["family"] for v in sh.values())), "AuRelVarRatio": _f(rs["family_over_null_ratio"], 2)}
    words = {"10": "Ten", "20": "Twenty", "40": "Forty", "80": "Eighty", "160": "OneSixty"}  # digits are not allowed in macro names
    m |= {f"AuTopK{words[k]}": _f(v) for k, v in er["iia_topk_by_effect"].items()}
    return L, m


def audit2_section(a: dict) -> tuple[list[str], dict]:
    """Markdown lines and LaTeX macros for results/posthoc_audit2.json (post-hoc checks added by the third audit)."""
    rr, ad, h5 = a["ridge_effect_ranked"], a["additivity_by_branch"], a["h5_blur_matched"]
    L = ["", "## Third-audit checks (NOT pre-registered; post-hoc, exploratory)", "",
         "Ridge-readout causal ranking: all mixed3b channels ranked by single-channel effect in the ridge decoder's (cos, sin) "
         f"output on the selection split. Family ranks: {rr['family_ranks']}; top-10: {rr['top10_by_effect']} "
         f"({rr['overlap_top10_family']} of 10 are family units).", "",
         "| set (ranked on selection, scored on confirmatory pairs) | decoded-angle IIA | (cos, sin) effect | effect difference, family minus set [CI] |",
         "|---|---|---|---|"]
    L += [f"| {n} | {_f(rr['iia'][n])} | {_f(rr['effect'][n])} | {_ci(rr['effect_diff_ci'][n]) if n in rr['effect_diff_ci'] else '—'} |" for n in rr["iia"]]
    groups = {k: v for k, v in ad.items() if isinstance(v, dict)}  # skip the n_pairs entry
    L += ["", f"Additivity by mixed4a group ({ad['n_pairs']} H1 pairs, z-scored pre-ReLU readout): non-additivity = "
              "sum|D_F + D_C - D_all|^2 / sum|D_all|^2 (0 = exactly additive); effects are projections on D_all within the group.", "",
          "| group | channels | non-additivity | effect family | effect complement | sum |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['n_channels']} | {v['nonadditivity']:.2g} | {_f(v['effect_family'])} | {_f(v['effect_complement'])} | "
          f"{_f(v['effect_family'] + v['effect_complement'])} |" for k, v in groups.items()]
    ang = h5["angles"]
    L += ["", f"H5 on matched angles ({', '.join(map(str, ang))} deg; {h5['n_images']} curve-present confirmatory photos, {h5['n_pairs']} pairs; "
              "family only). Original = rotations as in the pre-registered H5 (180 deg is an exact transpose, +-90 deg are bilinear on "
              "non-square images); exact = square crops, exact transposes at every angle.", "",
          "| rotations | effect | IIA | " + " | ".join(f"effect / IIA at {x} deg" for x in ang) + " |", "|---|---|---|" + "---|" * len(ang)]
    L += [f"| {m} | {_f(h5[m]['effect'])} | {_f(h5[m]['iia'])} | " + " | ".join(f"{_f(v['effect'])} / {_f(v['iia'])}" for v in h5[m]["by_angle"].values())
          + " |" for m in ("original", "exact")]
    L += ["", f"Effect difference, exact minus original: {_f(h5['effect_diff'])} {_ci(h5['effect_diff_ci'])} (photo-cluster bootstrap)."]
    m = {"TrRidgeOverlap": str(rr["overlap_top10_family"]), "TrRidgeFamRankMax": str(max(rr["family_ranks"])),
         "TrRidgeFamEff": _f(rr["effect"]["family"]), "TrRidgeFamIIA": _f(rr["iia"]["family"]), "TrRidgeTopEff": _f(rr["effect"]["top10_by_effect"]),
         "TrRidgeTopNFEff": _f(rr["effect"]["top10_nonfamily_by_effect"]), "TrRidgeTopNFIIA": _f(rr["iia"]["top10_nonfamily_by_effect"]),
         "TrRidgeDiffNFLo": _f(rr["effect_diff_ci"]["top10_nonfamily_by_effect"][0]),
         "TrRidgeDiffNFHi": _f(rr["effect_diff_ci"]["top10_nonfamily_by_effect"][1]),
         "TrBlurOrigEff": _f(h5["original"]["effect"]), "TrBlurExactEff": _f(h5["exact"]["effect"]), "TrBlurOrigIIA": _f(h5["original"]["iia"]),
         "TrBlurExactIIA": _f(h5["exact"]["iia"]), "TrBlurDiff": _f(h5["effect_diff"]), "TrBlurDiffLo": _f(h5["effect_diff_ci"][0]),
         "TrBlurDiffHi": _f(h5["effect_diff_ci"][1])}
    names = {"1x1": "One", "3x3": "Three", "5x5": "Five", "pool_reduce": "Pool", "curve_units_1x1": "CurveOne",
             "curve_units_5x5": "CurveFive", "curve_units_all": "Curve"}  # digits are not allowed in macro names
    m |= {f"TrNonAdd{names[k]}": _f(v["nonadditivity"]) for k, v in groups.items()}
    return L, m


def figures(cfg: Config, e2: dict, e3: dict) -> None:
    """All E2/E3 figures, drawn from the result JSONs (null distributions, |dtheta| bins, dose-response)."""
    nulls = lambda h, m: {"activation-change-matched null": h[f"null_{m}"]["delta_matched"],  # per-set null values
                          "branch-matched null": h[f"null_{m}"]["branch_matched"], "random-10 (descriptive)": h[f"null_{m}"]["random"]}
    refs = lambda h, m: {"Fisher top-k": h[m]["fisher_topk"], "complement": h[m]["complement"], "all channels": h[m]["all"]}
    for key, h, name in (("H1", e2["H1"], "e2_h1"), ("H2", e2["H2"], "e2_h2"), ("H4", e2["H4"], "e2_h4"), ("H5", e3["H5"], "e3_h5")):
        for m, metric, axis in (("iia", "IIA", "IIA (interchange intervention accuracy)"),
                                ("effect", "effect", "normalized effect (0 = none, 1 = source)")):  # two figures per test
            r = h[f"{m}_vs_delta_matched"]  # the family's value and CI
            plot_null(cfg, f"{name}_{m}_null", f"{SHORT.get(key, key)}: {metric} of the curve family vs null neuron sets", axis,
                      dict(value=r["rate"], ci=r["rate_ci"], label="curve family"), nulls(h, m), refs(h, m))
    bins = e2["H1"]["dtheta_bins"]
    col = lambda d, k: [v[k] for v in d.values()]  # column k of a dict of row dicts
    plot_points(cfg, "e2_h1_dtheta", "H1 by orientation difference |Δθ| (95% pigeonhole CI)", [f"{k}°" for k in bins],
                {"curve family": (col(bins, "iia"), [v["ci"][0] for v in bins.values()], [v["ci"][1] for v in bins.values()]),
                 "activation-change-matched null (mean)": (col(bins, "null_mean"),) * 3}, "IIA", xlabel="|Δθ| bin")
    dr = e2["H1"]["dose_response"]
    plot_points(cfg, "e2_h1_dose", "Dose-response over random K-subsets of the curve family (2.5–97.5% range over subsets)", list(dr),
                {"IIA": (col(dr, "iia_mean"), [v["iia_range"][0] for v in dr.values()], [v["iia_range"][1] for v in dr.values()]),
                 "normalized effect": (col(dr, "effect_mean"), [v["effect_range"][0] for v in dr.values()], [v["effect_range"][1] for v in dr.values()])},
                "value", x=[int(k) for k in dr], xlabel="K (number of curve units patched)")
    kr = e2["H1"]["delta_ranked"]
    plot_points(cfg, "e2_h1_topk", "IIA when patching the top-K mixed3b channels by activation change", list(kr),
                {"top-K by activation change": (col(kr, "top_delta"),) * 3, "curve family + top-(K-10)": (col(kr, "family_plus_top_delta"),) * 3},
                "IIA", x=np.log2([int(k) for k in kr]), xlabel="K (log scale)")


def run(cfg: Config) -> dict:
    """Write reports/generated/{results.md,numbers.tex}, figures/summary_{iia,effect}.png and the E2/E3 figures from the
    result JSONs."""
    load = lambda n: json.loads((cfg.results_dir / f"{n}.json").read_text())["results"]  # payload without config
    e1, e2, e3 = load("e1_replication"), load("e2_synthetic"), load("e3_natural")
    hyp = {k: v for k, v in e2.items() if k.startswith("H")} | {"composited": e3["composited"], "H5": e3["H5"]}  # all tests
    p_adj = holm([hyp[k]["effect_vs_delta_matched"]["p"] for k in SECONDARY])  # Holm over the 5 secondary hypotheses
    for k, p in zip(SECONDARY, p_adj):  # pre-registered secondary criterion
        h, e = hyp[k], hyp[k]["effect_vs_delta_matched"]
        h["holm_p_effect"] = float(p)
        h["verdict_secondary"] = "supported" if (e["margin_ci"][0] > 0 and p < 0.05) else "not supported"
    L = ["# Generated results tables", "", f"Generated by `concept_swap.phases.report_tables` from `{cfg.results_dir.relative_to(ROOT).as_posix()}/*.json`. Do not edit by hand.", ""]
    L += ["## E1: replication of the published tuning curves", "",
          f"Orientation convention fitted on mixed3a only: reflection {e1['alignment']['reflection']}, shift {e1['alignment']['shift_deg']:.0f} deg.", "",
          "| layer | n units | mean r (ours vs published) | permutation-null mean | null q95 | p |", "|---|---|---|---|---|---|"]
    fit = lambda l: " (fit layer: in-sample)" if l == e1["alignment"]["fit_layer"] else ""  # the convention was fitted on this layer
    L += [f"| {l}{fit(l)} | {len(e1[l]['r'])} | {_f(e1[l]['mean_r'])} | {_f(e1[l]['perm_null_mean'])} | {_f(e1[l]['perm_null_q95'])} | {e1[l]['perm_p']:.2g} |"
          for l in ("mixed3a", "mixed3b", "mixed4a")]
    L += ["", "LRN sensitivity (Lucent's default LRN constants, same stimuli and convention), mean r: "
          + ", ".join(f"{l} {_f(v)}" for l, v in e1["lrn_lucent_mean_r"].items())]
    L += ["", "| 3b unit | r | preferred theta (ours, deg) | AUROC vs other orientations | AUROC vs straight lines |", "|---|---|---|---|---|"]
    L += [f"| {s['neuron']} | {_f(r)} | {s['pref_theta']:.0f} | {_f(s['auroc_vs_other_orient'])} | {_f(s['auroc_vs_lines'])} |"
          for s, r in zip(e1["selectivity_3b"], e1["mixed3b"]["r"])]
    L += ["", "## Interchange interventions (all hypotheses)", "",
          "IIA = fraction of eligible pairs whose patched base decodes as the source; effect = normalized projection effect "
          "(0 = no change, 1 = source). Nulls: delta = activation-change-matched, branch = branch-matched. p = Phipson-Smyth "
          "permutation p vs the null sets. Verdict: H1 uses the pre-registered three-way rule; the others use the secondary criterion "
          "(effect margin CI > 0 and Holm-adjusted p < 0.05). The `verdict` field inside results/*.json applies the H1 rule to "
          "every test; for secondary tests use the verdicts in this table.", "",
          "| key | hypothesis | readout | clean acc | n elig. | IIA family [95% CI] | delta-null q95 | IIA margin [CI] | p | branch-null q95 | p (branch) "
          "| effect family [CI] | delta-null q95 | effect margin [CI] | p | branch-null q95 | verdict |", "|" + "---|" * 17]
    L += hypothesis_rows(hyp)
    L += ["", "Holm-adjusted p (effect vs delta-matched null): " + ", ".join(f"{k} {hyp[k]['holm_p_effect']:.2g}" for k in SECONDARY), "",
          "## Reference neuron sets (IIA / effect)", "", "| key | Fisher top-k | Jaccard(Fisher, family) | complement | all |", "|---|---|---|---|---|"]
    L += [f"| {k} | {_f(r['iia']['fisher_topk'])} / {_f(r['effect']['fisher_topk'])} | {_f(r['fisher_jaccard'], 2)} | "
          f"{_f(r['iia']['complement'])} / {_f(r['effect']['complement'])} | {_f(r['iia']['all'])} / {_f(r['effect']['all'])} |" for k, r in hyp.items()]
    h1 = e2["H1"]
    L += ["", "## H1 exploratory analyses", "", f"- closer-to-source rate (decoded orientation nearer the source than the base): {_f(h1['closer_to_source'])}",
          f"- same-orientation sham, base orientation retained: {_f(h1['sham_same_theta_base_retained'])}",
          f"- mean ablation, base orientation retained: family {_f(h1['mean_ablation']['base_retained_family'])}, complement "
          f"{_f(h1['mean_ablation']['base_retained_complement'])}, delta-null mean {_f(h1['mean_ablation']['base_retained_delta_null_mean'])} "
          f"(q05 {_f(h1['mean_ablation']['base_retained_delta_null_q05'])})",
          f"- activation-change percentile of each family unit: {[round(v, 3) for v in h1['delta_percentile_of_family']]}", "",
          "| abs(dtheta) bin | n | IIA family [CI] | delta-null mean |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {_f(v['iia'])} {_ci(v['ci'])} | {_f(v['null_mean'])} |" for k, v in h1["dtheta_bins"].items()]
    L += ["", "| K (family subset size) | n subsets | IIA mean (2.5-97.5%) | effect mean (2.5-97.5%) |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n_subsets']} | {_f(v['iia_mean'])} {_ci(v['iia_range'])} | {_f(v['effect_mean'])} {_ci(v['effect_range'])} |"
          for k, v in h1["dose_response"].items()]
    L += ["", "| K (top channels by activation change) | IIA top-K | IIA family + top-(K-10) |", "|---|---|---|"]
    L += [f"| {k} | {_f(v['top_delta'])} | {_f(v['family_plus_top_delta'])} |" for k, v in h1["delta_ranked"].items()]
    L += ["", "| family rotation k | predicted shift (deg) | observed circular-mean shift | within 30 deg of prediction | unchanged |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['pred_shift']:.0f} | {v['obs_circ_mean']:.0f} | {_f(v['frac_within_30_of_pred'])} | {_f(v['frac_unchanged'])} |"
          for k, v in h1["family_rotation"].items()]
    h2, h5, rel = e2["H2"], e3["H5"], e3["relevance"]
    L += ["", "## H2 / H5 / relevance details", "", f"- H2 IIA by direction: line->arc {_f(h2['iia_line_to_arc'])}, arc->line {_f(h2['iia_arc_to_line'])}",
          f"- H5: {h5['n_images_test']} curve-present confirmatory images (curve-family score > {h5['threshold']:.1f}); IIA by rotation: "
          + ", ".join(f"{a} deg {_f(v)}" for a, v in h5["iia_by_angle"].items()),
          f"- H5 curve-absent contrast (bottom quartile, family only): IIA {_f(h5['curve_absent_contrast']['iia'])}, effect "
          f"{_f(h5['curve_absent_contrast']['effect'])} (n pairs {h5['curve_absent_contrast']['n_pairs']})",
          f"- Relevance (mean ablation over the whole map, {rel['n_images']} images): top-1 flip rate family {_f(rel['flip_family'])} vs null mean "
          f"{_f(rel['flip_null_mean'])} (q95 {_f(rel['flip_null_q95'])}, p={rel['flip_p']:.2g}); KL family {_f(rel['kl_family'], 4)} vs null mean "
          f"{_f(rel['kl_null_mean'], 4)} (q95 {_f(rel['kl_null_q95'], 4)}, p={rel['kl_p']:.2g})"]
    out = ROOT / "reports" / "generated" / ("results_quick.md" if cfg.quick else "results.md")  # quick runs never overwrite
    out.parent.mkdir(parents=True, exist_ok=True)
    macros = latex_numbers(e1, e2, e3, hyp)  # numbers the paper \input-s
    if (cfg.results_dir / "posthoc_robustness.json").exists():  # optional post-hoc section (main.py --phase posthoc)
        ph_lines, ph_macros = posthoc_section(load("posthoc_robustness"))
        L += ph_lines
        macros += "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in ph_macros.items()) + "\n"
    if (cfg.results_dir / "posthoc_audit.json").exists():  # optional audit section (main.py --phase audit)
        au_lines, au_macros = audit_section(load("posthoc_audit"))
        L += au_lines
        macros += "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in au_macros.items()) + "\n"
    if (cfg.results_dir / "posthoc_audit2.json").exists():  # optional third-audit section (main.py --phase audit2)
        a2_lines, a2_macros = audit2_section(load("posthoc_audit2"))
        L += a2_lines
        macros += "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in a2_macros.items()) + "\n"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")  # LF on every OS
    (out.parent / ("numbers_quick.tex" if cfg.quick else "numbers.tex")).write_text(macros, encoding="utf-8", newline="\n")
    # summary figures: one row per test, IIA first, then the same rows with the continuous effect
    rows = [dict(label=SHORT.get(k, k.replace("H1_seed", "H1, jitter seed ")), value=v["iia_vs_delta_matched"]["rate"], ci=v["iia_vs_delta_matched"]["rate_ci"],
                 null_q95=v["iia_vs_delta_matched"]["null_q95"], null_mean=v["iia_vs_delta_matched"]["null_mean"]) for k, v in hyp.items()]
    plot_effects(cfg, "summary_iia", "Interchange accuracy of the curve family vs activation-change-matched null", rows, "IIA")
    rows = [dict(r, value=hyp[k]["effect_vs_delta_matched"]["rate"], ci=hyp[k]["effect_vs_delta_matched"]["rate_ci"],
                 null_q95=hyp[k]["effect_vs_delta_matched"]["null_q95"], null_mean=hyp[k]["effect_vs_delta_matched"]["null_mean"])
            for r, k in zip(rows, hyp)]
    plot_effects(cfg, "summary_effect", "Normalized patching effect of the curve family vs activation-change-matched null", rows,
                 "normalized effect (0 = none, 1 = source)")
    figures(cfg, e2, e3)
    log.info("wrote %s", out.relative_to(ROOT).as_posix())
    return hyp
