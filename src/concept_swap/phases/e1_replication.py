"""E1 (observational): replicate the Circuits curve-family tuning curves with our model port and renderer.

Local: measures radial tuning (activation vs orientation) of the published 3a/3b/4a curve families on the selection
split, aligns our orientation convention to the authors' with ONE global reflection+shift fitted on 3a only, then
tests the correlation with the published curves on 3b and 4a against a neuron-label permutation null. Also reports
orientation selectivity (AUROC) of each 3b unit vs other orientations and vs straight lines.
Global context: if the published tuning is not reproduced, any intervention result could be an artefact of our
port, renderer or orientation convention; this phase is therefore the gate for E2/E3 (Huang et al.'s "observational
mode", https://arxiv.org/abs/2309.10312).
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

from concept_swap import families as Fm, model as M, stimuli as S
from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.phases.common import save_json
from concept_swap.plots import plot_radial_replication

log = get_logger(__name__)
LAYERS = {"mixed3a": 13, "mixed3b": 13, "mixed4a": 6}  # readout position per layer (RF centre at pixel 111)


@torch.no_grad()
def center_acts(model, X: torch.Tensor, device: str, batch: int) -> dict[str, np.ndarray]:
    """Post-ReLU activations of all channels of mixed3a/3b/4a at their readout positions, (N, C) per layer."""
    out = {l: [] for l in LAYERS}
    for i in range(0, len(X), batch):
        with M.capture(model, list(LAYERS)) as a:
            model(M.preprocess(X[i:i + batch].to(device)))
        for l, p in LAYERS.items():
            out[l].append(a[l][:, :, p, p].cpu())
    return {l: torch.cat(v).numpy() for l, v in out.items()}


def tuning(acts: np.ndarray, theta: np.ndarray, n_theta: int) -> np.ndarray:
    """(C, n_theta) mean activation per orientation, averaged over nuisance cells."""
    grid = np.arange(n_theta) * 360 / n_theta
    return np.stack([acts[theta == t].mean(0) for t in grid], 1)


def corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching rows of A and B."""
    A, B = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def published_on_grid(fam: Fm.Family, n_theta: int) -> np.ndarray:
    """Authors' 720-point (0.5 deg) curves subsampled to our n_theta grid."""
    return fam.tuning[:, :: 720 // n_theta]


def to_published(ours: np.ndarray, refl: int, shift: int) -> np.ndarray:
    """Re-index our tuning curves (C, n) so column j holds our orientation index (refl*j + shift) mod n,
    i.e. the orientation that corresponds to published orientation j under the fitted convention."""
    n = ours.shape[1]
    return ours[:, (refl * np.arange(n) + shift) % n]


def align(ours: np.ndarray, pub: np.ndarray) -> tuple[int, int, float]:
    """Best (reflection, shift in grid steps) of `to_published`, maximising the mean correlation over neurons."""
    best = max(((refl, sh, corr_rows(to_published(ours, refl, sh), pub).mean())
                for refl in (1, -1) for sh in range(ours.shape[1])), key=lambda t: t[2])
    return int(best[0]), int(best[1]), float(best[2])


def run(cfg: Config, model, rng: np.random.Generator) -> dict:
    """Run E1 on the selection split; returns and saves `results/e1_replication.json`."""
    arcs, lines = S.grid(cfg, cfg.selection_seed), S.grid(cfg, cfg.selection_seed, kind="line")
    acts = center_acts(model, S.render(arcs, cfg.arc_span_deg), cfg.device, cfg.batch_size)
    line_acts = center_acts(model, S.render(lines, cfg.arc_span_deg), cfg.device, cfg.batch_size)["mixed3b"]
    fams = {l: Fm.load_family(cfg.data_dir, l) for l in LAYERS}
    ours = {l: tuning(acts[l][:, f.neurons], arcs.theta.values, cfg.n_theta) for l, f in fams.items()}
    pub = {l: published_on_grid(f, cfg.n_theta) for l, f in fams.items()}
    refl, shift, r3a = align(ours["mixed3a"], pub["mixed3a"])  # convention fitted on 3a ONLY
    log.info("E1 orientation convention fitted on 3a: reflection=%d shift=%d deg (mean r=%.3f)", refl, shift * 360 // cfg.n_theta, r3a)
    res = dict(alignment=dict(reflection=refl, shift_deg=shift * 360 / cfg.n_theta, fit_layer="mixed3a", fit_r=r3a))
    for l in LAYERS:
        aligned = to_published(ours[l], refl, shift)
        r = corr_rows(aligned, pub[l])
        perm = np.array([corr_rows(aligned, pub[l][rng.permutation(len(r))]).mean() for _ in range(10_000)])
        p = float((1 + (perm >= r.mean()).sum()) / (1 + len(perm)))  # Phipson & Smyth one-sided p
        res[l] = dict(neurons=fams[l].neurons, r=r.round(4).tolist(), mean_r=float(r.mean()), perm_null_mean=float(perm.mean()),
                      perm_null_q95=float(np.quantile(perm, 0.95)), perm_p=p,
                      pref_theta_ours=(ours[l].argmax(1) * 360 / cfg.n_theta).tolist())
        log.info("E1 %s: mean r with published tuning = %.3f (label-permutation null mean %.3f, q95 %.3f, p=%.2g)",
                 l, r.mean(), perm.mean(), np.quantile(perm, 0.95), p)
    # Selectivity of each 3b curve unit: AUROC for "arc within +-45 deg of preferred" vs other arcs, and vs lines.
    auc = []
    for k, n in enumerate(fams["mixed3b"].neurons):
        pref = res["mixed3b"]["pref_theta_ours"][k]
        near = np.abs(S.circ_diff(arcs.theta.values, pref)) <= 45
        a_pos, a_neg = acts["mixed3b"][near, n], acts["mixed3b"][~near, n]
        lab = lambda neg: np.r_[np.ones(len(a_pos)), np.zeros(len(neg))]
        auc.append(dict(neuron=n, pref_theta=pref, auroc_vs_other_orient=float(roc_auc_score(lab(a_neg), np.r_[a_pos, a_neg])),
                        auroc_vs_lines=float(roc_auc_score(lab(line_acts[:, n]), np.r_[a_pos, line_acts[:, n]])),
                        mean_arc_pref=float(a_pos.mean()), mean_line=float(line_acts[:, n].mean())))
    res["selectivity_3b"] = auc
    log.info("E1 3b AUROC vs other orientations: %s", [round(a["auroc_vs_other_orient"], 3) for a in auc])
    log.info("E1 3b AUROC vs straight lines:     %s", [round(a["auroc_vs_lines"], 3) for a in auc])
    plot_radial_replication(cfg, {l: to_published(ours[l], refl, shift) for l in LAYERS}, pub, res)
    save_json(cfg, "e1_replication", res)
    return res
