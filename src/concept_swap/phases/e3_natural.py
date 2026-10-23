"""E3 (natural images, Imagenette val): does the curve family's causal role survive natural image statistics?

Local: (a) H1 rerun with arcs composited onto natural photo backgrounds; (b) H5, a natural counterfactual: the base
is a photo crop and the source is the SAME photo rotated about the readout RF centre, so the "concept" being swapped
is the orientation of whatever curves the unit sees; success = the patched base's 4a curve-family vector is most
correlated with the rotated version (among the photo's clean versions); restricted to "curve-present" crops (top 25%
3b curve-family response, threshold from the selection half); (c) exploratory relevance: mean-ablating the family and
measuring top-1 flips of the full network's predictions vs activation-matched null ablations.
Global context: rotating the input should rotate the curve-family code (Olah et al. 2020, "Naturally Occurring
Equivariance", https://distill.pub/2020/circuits/equivariance/); here we ask whether that code is causally used.
"""

from dataclasses import replace  # copy a frozen Config with a few fields changed

import numpy as np
import pandas as pd
import torch

from concept_swap import families as Fm, interventions as I, model as M, natural as N, readouts as R, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import Readout, Split, interchange_experiment, save_json, summarize

log = get_logger(__name__)


def halves(cfg: Config, paths: list) -> tuple[list, list]:
    """Seeded split of the validation images into a selection half and a confirmatory half."""
    perm = np.random.default_rng(cfg.selection_seed).permutation(len(paths))  # fixed shuffle
    return [paths[i] for i in perm[::2]], [paths[i] for i in perm[1::2]]  # even positions / odd positions


def as_tensor(X: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) float array -> (N, 3, H, W) tensor."""
    return torch.from_numpy(X).permute(0, 3, 1, 2)  # channels-first, as PyTorch convolutions expect


def composited(cfg: Config, model, fam3b, fam4a, rng, bg_paths: list, n_rep: int = 2, **kw) -> dict:
    """H1 robustness: arcs drawn over natural backgrounds; one background per nuisance cell (shared by every
    orientation in that cell, so pairs still differ only in theta), same backgrounds in selection and test splits."""
    c2 = replace(cfg, sources_per_base=2)  # 2 sources per base keeps the pair count at 4608 with twice the cells
    bgs = N.load(bg_paths[: n_rep * len(cfg.radii) * len(cfg.widths) * 2])  # one photo per (replicate, nuisance cell)

    def split(seed):
        """n_rep replicated grids (distinct jitter) with cell ids offset per replicate, rendered over the photos."""
        g = pd.concat([S.grid(cfg, seed * 1000 + 500 + r).assign(cell=lambda d, r=r: d.cell + r * len(d.cell.unique()))
                       for r in range(n_rep)], ignore_index=True)
        X = S.render(g, cfg.arc_span_deg, bgs[g.cell.values])  # each stimulus over its cell's background
        return Split(g, I.collect(model, X, "mixed4a", cfg.device, cfg.batch_size)), S.minimal_pairs(g, c2, seed)

    (sel, p_sel), (test, p_test) = split(cfg.selection_seed), split(cfg.confirm_seed)
    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam3b, fam4a, rng, cfg.n_null_sets // 2, **kw)
    return summarize(cfg, ex, rng, "H1 on natural backgrounds (arcs composited onto Imagenette crops)")


def family_score(cfg: Config, model, paths: list, fam3b) -> np.ndarray:
    """Max 3b curve-family activation at the readout RF centre for each clean crop (curve-presence score)."""
    out = []
    for i in range(0, len(paths), 256):  # 256 photos at a time bounds host memory
        c = I.collect(model, as_tensor(N.load(paths[i:i + 256])), "mixed4a", cfg.device, cfg.batch_size)
        out.append(c.windows[:, fam3b, 5, 5].max(1).values.numpy())  # window index 5 = 3b position 13
    return np.concatenate(out)


def rotation_split(cfg: Config, model, paths: list, square: bool = False) -> tuple[Split, pd.DataFrame]:
    """Clean versions (angle 0 + each rotation) of every image, and (base = angle 0, source = rotated) pairs."""
    angles = [0, *cfg.rotations]
    g = pd.DataFrame([dict(image=i, angle=a, theta=a % 360, cell=i) for i in range(len(paths)) for a in angles])  # cell = photo
    X = np.stack([N.crop(paths[r.image], r.angle, square) for r in g.itertuples()])  # square: exact 90/180/270 rotations
    sp = Split(g, I.collect(model, as_tensor(X), "mixed4a", cfg.device, cfg.batch_size))
    idx = {(r.image, r.angle): k for k, r in enumerate(g.itertuples())}  # (photo, angle) -> row
    pairs = pd.DataFrame([dict(base=idx[(i, 0)], source=idx[(i, a)], dtheta=float(a)) for i in range(len(paths)) for a in cfg.rotations])
    return sp, pairs


def rotation(cfg: Config, model, fam3b, fam4a, rng, sel_paths: list, test_paths: list, square: bool = False, **kw) -> dict:
    """H5: natural rotation counterfactual on curve-present crops (+ exploratory curve-absent contrast)."""
    s_sel, s_test = family_score(cfg, model, sel_paths, fam3b), family_score(cfg, model, test_paths, fam3b)
    thr, low = np.quantile(s_sel, 0.75), np.quantile(s_sel, 0.25)  # thresholds from the selection half only
    pick = lambda ps, m: [p for p, k in zip(ps, m) if k]  # paths where mask m is True
    sel, p_sel = rotation_split(cfg, model, pick(sel_paths, s_sel > thr), square)
    test, p_test = rotation_split(cfg, model, pick(test_paths, s_test > thr), square)
    z = R.ZScore().fit(sel.col.readout.numpy()[:, fam4a])  # z statistics from the selection half only

    def readout_for(sp):
        """Nearest-version decoder: templates are each photo's own clean versions (cell = photo, label = angle)."""
        return Readout(np.asarray(fam4a), z, R.TemplateDecoder().fit(z(sp.col.readout.numpy()[:, fam4a]), sp.grid.theta.values,
                                                                      sp.grid.cell.values), "template")

    ex = interchange_experiment(cfg, model, "mixed4a", sel, test, p_test, p_sel, fam3b, fam4a, rng, cfg.n_null_sets // 2,
                                readout=readout_for(test), **kw)
    out = summarize(cfg, ex, rng, "H5: natural rotation counterfactual (curve-present Imagenette crops)")
    out.update(threshold=float(thr), n_images_test=int(len(test.grid) // (1 + len(cfg.rotations))),
               iia_by_angle={str(a): float(((np.abs(S.circ_diff(ex["outs"]["family"][:, 0], ex["clean"][p_test.source.values])) <= cfg.tol_deg)
                                             [ex["elig"] & (p_test.dtheta.values == a)]).mean()) for a in cfg.rotations})
    absent, p_abs = rotation_split(cfg, model, pick(test_paths, s_test <= low), square)  # many crops score exactly 0 (ReLU)
    rd = readout_for(absent)
    zb, zs = (z(absent.col.readout.numpy()[i][:, fam4a]) for i in (p_abs.base.values, p_abs.source.values))
    o = I.patched_readouts(model, "mixed4a", absent.col.windows, p_abs.base.values, p_abs.source.values, [fam3b], fam4a,
                           lambda Rr: np.c_[rd.predict_z(z(Rr), absent.grid.cell.values[p_abs.base.values]), ((z(Rr) - zb) * (zs - zb)).sum(1)])[0]
    clean = rd.decode(absent.col.readout.numpy(), absent.grid.cell.values)  # clean decodes of the curve-absent photos
    out["curve_absent_contrast"] = dict(iia=float((np.abs(S.circ_diff(o[:, 0], clean[p_abs.source.values])) <= cfg.tol_deg).mean()),
                                        effect=float(o[:, 1].sum() / ((zs - zb) ** 2).sum()), n_pairs=int(len(p_abs)))
    log.info("H5 exploratory curve-absent contrast: %s", out["curve_absent_contrast"])
    return out


@torch.no_grad()
def mixed3b_moments(cfg: Config, model, paths: list) -> tuple[torch.Tensor, np.ndarray]:
    """Mean mixed3b map (480, 28, 28) over `paths` and each channel's variance averaged over positions (the size of a
    mean ablation of that channel). Used by `relevance` and by the post-hoc audit's variance diagnostic."""
    s1 = s2 = 0.0  # running sums of mixed3b and its square
    for i in range(0, len(paths), cfg.batch_size):
        with M.capture(model, ["mixed3b"]) as a:
            model(M.preprocess(as_tensor(N.load(paths[i:i + cfg.batch_size])).to(cfg.device)))
        s1, s2 = s1 + a["mixed3b"].double().sum(0), s2 + (a["mixed3b"].double() ** 2).sum(0)
    var = (s2 / len(paths) - (s1 / len(paths)) ** 2).mean((1, 2)).cpu().numpy()  # E[x^2] - E[x]^2, averaged over positions
    return (s1 / len(paths)).float(), var


@torch.no_grad()
def relevance(cfg: Config, model, fam3b, rng, sel_paths: list, test_paths: list, n_sets: int = 100) -> dict:
    """Exploratory: mean-ablate the 3b curve family over the whole map and count top-1 changes of the network's
    1008-way prediction (no label mapping needed) and the mean KL(clean || ablated), vs activation-matched nulls."""
    mean, var = mixed3b_moments(cfg, model, sel_paths)  # selection-half mean map (ablation value) and per-channel variance
    sets = [np.array(fam3b)] + list(Fm.delta_matched_sets(var, fam3b, n_sets, rng))  # family + variance-matched nulls
    X = as_tensor(N.load(test_paths))  # confirmatory photos
    flips, kls = np.zeros(len(sets)), np.zeros(len(sets))  # accumulated over batches
    for i in range(0, len(X), cfg.batch_size):
        x = M.preprocess(X[i:i + cfg.batch_size].to(cfg.device))
        p0 = model(x)  # Lucent's InceptionV1 ends in softmax2: class probabilities
        for k, ch in enumerate(sets):
            ch_t = torch.as_tensor(ch, device=cfg.device)

            def ablate(o, ch_t=ch_t):
                """Forward-hook edit: replace channels ch_t of mixed3b by their selection-half mean maps."""
                o = o.clone()  # never modify the hooked tensor in place
                o[:, ch_t] = mean[ch_t]
                return o

            with M.patch(model, "mixed3b", ablate):
                p = model(x)
            flips[k] += (p.argmax(1) != p0.argmax(1)).sum().item()  # top-1 changes
            kls[k] += (p0 * (torch.log(p0 + 1e-12) - torch.log(p + 1e-12))).sum(1).sum().item()  # KL(clean || ablated)
    flips, kls = flips / len(X), kls / len(X)  # per-photo rates
    return dict(n_images=len(X), flip_family=float(flips[0]), flip_null_mean=float(flips[1:].mean()), flip_null_q95=float(np.quantile(flips[1:], 0.95)),
                flip_p=float((1 + (flips[1:] >= flips[0]).sum()) / (1 + n_sets)), kl_family=float(kls[0]), kl_null_mean=float(kls[1:].mean()),
                kl_null_q95=float(np.quantile(kls[1:], 0.95)), kl_p=float((1 + (kls[1:] >= kls[0]).sum()) / (1 + n_sets)))


def run(cfg: Config, model, rng: np.random.Generator) -> dict:
    """Run E3; save `results/e3_natural.json` (figures are drawn by the report phase)."""
    paths = N.list_images(N.ensure_imagenette(cfg))  # checksum-verified download on first use
    if cfg.quick:
        paths = paths[::20]  # ~200 images for the smoke test
    sel_paths, test_paths = halves(cfg, paths)
    fam3b = Fm.load_family(cfg.data_dir, "mixed3b").neurons
    fam4a = np.array(Fm.load_family(cfg.data_dir, "mixed4a").neurons)
    log.info("E3: %d validation images (%d selection / %d confirmatory)", len(paths), len(sel_paths), len(test_paths))
    res = {"composited": composited(cfg, model, fam3b, fam4a, rng, sel_paths)}  # backgrounds from the selection half
    res["H5"] = rotation(cfg, model, fam3b, fam4a, rng, sel_paths, test_paths)
    res["relevance"] = relevance(cfg, model, fam3b, rng, sel_paths, test_paths, n_sets=20 if cfg.quick else 100)
    log.info("E3 relevance (mean ablation): %s", res["relevance"])
    res["image_ids_test"] = [p.relative_to(p.parents[2]).as_posix() for p in test_paths]  # e.g. val/n01440764/x.JPEG
    save_json(cfg, "e3_natural", res)
    return res
