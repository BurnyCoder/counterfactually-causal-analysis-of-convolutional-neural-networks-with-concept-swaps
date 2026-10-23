"""Building blocks shared by the experiment phases: result saving, stimulus splits, readouts, and the generic
interchange-intervention experiment (one function reused for H1, H2, H4, robustness and sensitivity runs).

Local: `interchange_experiment` fits a readout on a selection split, then patches the explanation's neuron set,
reference sets and null sets into the confirmatory pairs and returns per-set, per-pair outcomes; `summarize` turns
them into the pre-registered statistics. Global context: implements the analysis plan in reports/preregistration.md.
"""

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from concept_swap import families as Fm, interventions as I, model as M, readouts as R, stats as St, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger

log = get_logger(__name__)
RULE = dict(min_rate=0.5, sesoi=0.10, falsify_rate=0.2, falsify_margin=0.05)  # pre-registered H1 thresholds


def _jsonable(o):
    """Convert numpy scalars/arrays and tuples recursively so json.dump accepts them."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}  # JSON keys must be strings
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]  # tuples (e.g. CIs) become lists
    if isinstance(o, np.ndarray):
        return o.tolist()  # arrays -> nested lists of Python numbers
    if isinstance(o, np.generic):
        return o.item()  # numpy scalar -> Python scalar
    return o


def save_json(cfg: Config, name: str, obj: dict) -> None:
    """Write results/<name>.json with the run config and a timestamp attached."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(created=datetime.now().isoformat(timespec="seconds"), config=cfg.as_log_dict(), results=_jsonable(obj))
    (cfg.results_dir / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")  # LF on all OSes
    log.info("saved %s/%s.json", cfg.as_log_dict()["results_dir"], name)  # repo-relative path in the log


@dataclass
class Split:
    """A stimulus table plus its cached windows/readouts for one readout layer."""

    grid: pd.DataFrame  # one row per stimulus (kind, theta, radius, width, polarity, cell, jitter)
    col: I.Collected  # cached RF windows and clean readouts, row-aligned with `grid`


def make_split(cfg: Config, model, layer: str, grid: pd.DataFrame, backgrounds: np.ndarray | None = None) -> Split:
    """Render `grid` (optionally over natural backgrounds) and cache the windows feeding `layer`'s readout."""
    X = S.render(grid, cfg.arc_span_deg, backgrounds)  # (N, 3, 224, 224) images in [0, 1]
    return Split(grid.reset_index(drop=True), I.collect(model, X, layer, cfg.device, cfg.batch_size))  # 0..N-1 row index


def channel_delta(windows, base: np.ndarray, src: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Mean squared base->source change of every channel over the window (selection split), chunked for memory."""
    tot, base, src = 0.0, np.array(base), np.array(src)  # writable copies (pandas .values are read-only views)
    for i in range(0, len(base), chunk):  # chunks bound the temporary (chunk, C, 11, 11) difference tensor
        tot = tot + ((windows[src[i:i + chunk]] - windows[base[i:i + chunk]]) ** 2).sum((0, 2, 3)).double()  # per channel
    return (tot / len(base)).numpy()  # mean over pairs, summed over window positions


def branch_pool(layer_in: str, family: list[int]) -> np.ndarray:
    """Channels of the branch(es) that contain the family, minus the family (branch-matched null pool)."""
    sl = M.branch_slices(layer_in)  # branch -> channel slice of the concatenated layer
    chans = np.concatenate([np.arange(sl[b].start, sl[b].stop) for b in {M.branch_of(layer_in, n) for n in family}])
    return np.setdiff1d(chans, family)  # never draw the family itself


@dataclass
class Readout:
    """Orientation readout tau: z-score the chosen output channels, then decode with a nearest-template decoder
    (per nuisance cell; primary) or a ridge (cos, sin) decoder (secondary, all channels). Fitted on selection only."""

    out_idx: np.ndarray  # readout-layer channels used (14 curve units, or all 508)
    z: R.ZScore  # selection-split standardisation
    dec: object  # TemplateDecoder or RidgeDecoder
    kind: str  # "template" or "ridge"

    @classmethod
    def fit(cls, sel: Split, out_idx: np.ndarray, labels: np.ndarray, kind: str = "template") -> "Readout":
        """Fit z-statistics and the decoder on the clean selection-split readouts (no confirmatory data)."""
        Z = sel.col.readout.numpy()[:, out_idx]  # clean pre-ReLU readouts of the chosen channels
        z = R.ZScore().fit(Z)
        dec = R.TemplateDecoder().fit(z(Z), labels, sel.grid.cell.values) if kind == "template" else R.RidgeDecoder().fit(z(Z), labels)
        return cls(np.asarray(out_idx), z, dec, kind)

    def predict_z(self, Z: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Decode already z-scored readouts (the template decoder also needs each row's nuisance cell)."""
        return self.dec.predict(Z, cell) if self.kind == "template" else self.dec.predict(Z)

    def decode(self, readout: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Decode raw readouts (all channels) of clean or patched stimuli."""
        return self.predict_z(self.z(readout[:, self.out_idx]), cell)


def interchange_experiment(cfg: Config, model, layer: str, sel: Split, test: Split, pairs: pd.DataFrame,
                           sel_pairs: pd.DataFrame, family: list[int], out_idx: np.ndarray, rng: np.random.Generator,
                           n_sets: int, kind: str = "template", labels_sel: np.ndarray | None = None,
                           labels_test: np.ndarray | None = None, readout: "Readout | None" = None,
                           null_kinds: tuple = ("delta_matched", "branch_matched", "random")) -> dict:
    """Patch the explanation's set, reference sets and null sets from source into base; return per-pair outcomes.

    Outcome per pair and set: [decoded label of the patched base, projection numerator <zp - zb, zs - zb>].
    Success (computed in `summarize`) = decoded label within tol of decode(clean source): Huang et al.'s
    tau(M_{a<-z}(b)) = tau(M(s)). Eligible pairs: clean base and clean source both decode correctly.
    Nulls (`null_kinds`): activation-change-matched and branch-matched (n_sets each), random (descriptive, n_sets // 4),
    and the post-hoc rank-matched null; the post-hoc `top_delta_nonfamily` reference set is always evaluated.
    """
    lab_sel = sel.grid.theta.values if labels_sel is None else labels_sel  # labels the decoder is fitted on
    lab = test.grid.theta.values if labels_test is None else labels_test  # true labels of the test stimuli
    rd = readout or Readout.fit(sel, out_idx, lab_sel, kind)  # a pre-built readout is used by the natural-image H5
    clean = rd.decode(test.col.readout.numpy(), test.grid.cell.values)  # decoded label of every clean test stimulus
    acc = np.abs(S.circ_diff(clean, lab)) <= cfg.tol_deg  # is each clean stimulus decoded correctly?
    b, s = pairs.base.values, pairs.source.values  # row indices of base and source per pair
    zb, zs = (rd.z(test.col.readout.numpy()[i][:, rd.out_idx]) for i in (b, s))  # clean z-vectors of base and source
    cell_b = test.grid.cell.values[b]  # nuisance cell of each base (selects its templates)

    def reduce(Rr: np.ndarray, zb=zb, zs=zs, cell_b=cell_b) -> np.ndarray:
        """Patched readouts (n_pairs, len(out_idx)) -> [decoded label, projection numerator] per pair."""
        zp = rd.z(Rr)  # z-score with selection statistics
        return np.c_[rd.predict_z(zp, cell_b), ((zp - zb) * (zs - zb)).sum(1)]

    def run(sets, base=b, src=s, **kw) -> np.ndarray:
        """(n_sets, n_pairs, 2) outcomes for arbitrary channel sets (reused for dose-response, rotation, ...)."""
        return I.patched_readouts(model, layer, test.col.windows, base, src, list(sets), rd.out_idx, reduce, **kw)

    delta = channel_delta(sel.col.windows, sel_pairs.base.values, sel_pairs.source.values)  # selection split only
    n_in, mid = sel.col.windows.shape[1], sel.col.windows.shape[-1] // 2  # channel count; window centre index
    others = np.setdiff1d(np.arange(n_in), family)  # every non-family channel (the complement)
    fisher = np.argsort(-Fm.fisher_scores(sel.col.windows[:, :, mid, mid].numpy(), lab_sel))[:len(family)]  # data-driven top-k
    named = {"family": np.array(family), "fisher_topk": fisher, "complement": others, "all": np.arange(n_in),
             "top_delta_nonfamily": Fm.top_nonfamily(delta, family, len(family))}
    samplers = {"delta_matched": lambda: Fm.delta_matched_sets(delta, family, n_sets, rng),  # drawn lazily, in this order
                "branch_matched": lambda: Fm.pool_sets(branch_pool(I.PREV[layer], family), len(family), n_sets, rng),
                "random": lambda: Fm.pool_sets(others, len(family), max(n_sets // 4, 20), rng),
                "rank_matched": lambda: Fm.rank_matched_sets(delta, family, n_sets, rng)}
    null_sets = {k: samplers[k]() for k in null_kinds}  # only the requested nulls consume RNG draws
    outs = dict(zip(named, run(named.values())))  # name -> (n_pairs, 2)
    nulls = {k: run(v) for k, v in null_sets.items()}  # null kind -> (n_sets, n_pairs, 2)
    elig = acc[b] & acc[s]  # both clean stimuli decode correctly (same filter for every set)
    log.info("%s readout (%s): clean decoding accuracy %.3f; pairs %d, eligible %d", layer, kind, acc.mean(), len(b), elig.sum())
    return dict(acc=acc, elig=elig, keep=R.denominator_mask(zb, zs), den=((zs - zb) ** 2).sum(1), outs=outs, nulls=nulls,
                named=named, delta=delta, fisher=fisher, pairs=pairs, readout=rd, clean=clean, run=run, tol=cfg.tol_deg)


def success(ex: dict, o: np.ndarray, target: str = "source") -> np.ndarray:
    """Per-pair success of outcome array `o` (..., n_pairs, 2): decoded label equals decode(clean target)."""
    t = ex["pairs"].source.values if target == "source" else ex["pairs"].base.values  # whose clean decode to match
    return np.abs(S.circ_diff(o[..., 0], ex["clean"][t])) <= ex["tol"]  # (n_pairs,) broadcasts over leading set axis


def summarize(cfg: Config, ex: dict, rng: np.random.Generator, label: str) -> dict:
    """Pre-registered statistics: IIA and normalized effect of each named set, margins vs each null with pigeonhole
    CIs and permutation p-values, and the three-way H1 verdict."""
    e, k = ex["elig"], ex["elig"] & ex["keep"]  # IIA uses eligible pairs; the effect also drops tiny denominators
    pairs = ex["pairs"]
    W = St.pigeonhole_weights(pairs.base.values[e], pairs.source.values[e], cfg.n_boot, rng)  # bootstrap weights (IIA)
    Wk = St.pigeonhole_weights(pairs.base.values[k], pairs.source.values[k], cfg.n_boot, rng)  # bootstrap weights (effect)
    den_k = ex["den"][k]  # |zs - zb|^2 of the kept pairs
    fam = ex["outs"]["family"]
    out = dict(label=label, readout=ex["readout"].kind, clean_acc=float(ex["acc"].mean()), n_pairs=int(len(e)),
               n_eligible=int(e.sum()), n_effect_pairs=int(k.sum()), fisher_topk=ex["fisher"].tolist(),
               fisher_jaccard=len(set(ex["fisher"]) & set(ex["named"]["family"])) / len(set(ex["fisher"]) | set(ex["named"]["family"])),
               iia={n: float(success(ex, o)[e].mean()) for n, o in ex["outs"].items()},  # IIA of every named set
               effect={n: float(o[k, 1].sum() / den_k.sum()) for n, o in ex["outs"].items()})  # ratio-of-sums effect
    for null, nv in ex["nulls"].items():
        out[f"iia_vs_{null}"] = St.margin_vs_null(success(ex, fam)[e], success(ex, nv)[:, e], W)  # IIA margin, CI, p
        num = np.vstack([fam[k, 1], nv[:, k, 1]])  # (1 + n_sets, n_kept) projection numerators
        eff, eff_b = num.sum(1) / den_k.sum(), (Wk @ num.T) / (Wk @ den_k)[:, None]  # point and bootstrap ratios
        out[f"effect_vs_{null}"] = dict(rate=float(eff[0]), rate_ci=St.percentile_ci(eff_b[:, 0]), null_mean=float(eff[1:].mean()),
                                        null_q95=float(np.quantile(eff[1:], 0.95)), margin=float(eff[0] - np.quantile(eff[1:], 0.95)),
                                        margin_ci=St.percentile_ci(eff_b[:, 0] - np.quantile(eff_b[:, 1:], 0.95, axis=1)),
                                        p=St.perm_p(eff[0], eff[1:]), n_pairs=int(k.sum()), n_sets=int(len(nv)))
    first = next(iter(ex["nulls"]))  # primary null of this run (delta_matched in the pre-registered runs)
    d, br = out[f"iia_vs_{first}"], out.get("iia_vs_branch_matched", out[f"iia_vs_{first}"])
    out["verdict"] = St.decide(d["rate"], d["null_q95"], d["margin_ci"], br["margin_ci"], **RULE)  # three-way H1 rule
    out["null_iia"] = {n: success(ex, v)[:, e].mean(1).tolist() for n, v in ex["nulls"].items()}  # for histograms
    out["null_effect"] = {n: (v[:, k, 1].sum(1) / den_k.sum()).tolist() for n, v in ex["nulls"].items()}
    log.info("%s: IIA family %.3f [%.3f, %.3f] | %s-null q95 %.3f (p=%.2g) | other nulls q95 %s | "
             "Fisher %.3f, complement %.3f, all %.3f, top-delta non-family %.3f | effect family %.3f vs %s-null q95 %.3f | verdict: %s",
             label, d["rate"], *d["rate_ci"], first, d["null_q95"], d["p"],
             {k[7:]: round(v["null_q95"], 3) for k, v in out.items() if k.startswith("iia_vs_") and k != f"iia_vs_{first}"}, out["iia"]["fisher_topk"],
             out["iia"]["complement"], out["iia"]["all"], out["iia"]["top_delta_nonfamily"], out["effect"]["family"], first,
             out[f"effect_vs_{first}"]["null_q95"], out["verdict"])
    return out
