"""Export phase: build `hf_export/` for the Hugging Face dataset (synthetic stimuli, per-pair H1 outcomes, results).

Local: recomputes the H1 per-pair outcomes for the family and reference sets (the same deterministic computation as
E2, without the null sets) and writes CSV tables, the rendered synthetic stimuli as PNGs, the result JSONs and a
dataset card. Global context: lets anyone re-analyse the swaps without a GPU. No Imagenette pixels are exported
(ImageNet-derived); natural-image results are included only as aggregate JSON plus file names.
"""

import shutil  # file copies: https://docs.python.org/3/library/shutil.html

import numpy as np
import pandas as pd
from PIL import Image

from concept_swap import families as Fm, interventions as I, stimuli as S
from concept_swap.config import ROOT, Config
from concept_swap.log import get_logger
from concept_swap.phases.common import Readout, make_split

log = get_logger(__name__)


def run(cfg: Config, model) -> None:
    """Write hf_export/{data/*.csv, images/<split>/*.png, results/*, README.md} (hf_export_quick/ for --quick)."""
    out_dir = ROOT / ("hf_export_quick" if cfg.quick else "hf_export")  # smoke tests never overwrite the real export
    fam3b, fam4a = Fm.load_family(cfg.data_dir, "mixed3b").neurons, np.array(Fm.load_family(cfg.data_dir, "mixed4a").neurons)
    splits = {"selection": cfg.selection_seed, "confirmatory": cfg.confirm_seed}
    sp = {k: make_split(cfg, model, "mixed4a", S.grid(cfg, s)) for k, s in splits.items()}  # same splits as E2
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    pd.concat([v.grid.assign(split=k, idx=v.grid.index) for k, v in sp.items()]).to_csv(out_dir / "data" / "stimuli.csv", index=False)
    for k, v in sp.items():  # rendered stimuli, 8-bit PNG
        d = out_dir / "images" / k
        d.mkdir(parents=True, exist_ok=True)
        X = S.render(v.grid, cfg.arc_span_deg).permute(0, 2, 3, 1).numpy()  # (N, 224, 224, 3) in [0, 1]
        for i, x in enumerate(X):
            Image.fromarray((x * 255).round().astype(np.uint8)).save(d / f"{i:04d}.png")  # file name = row idx
    sel, test = sp["selection"], sp["confirmatory"]
    rd = Readout.fit(sel, fam4a, sel.grid.theta.values)  # primary readout, selection-fitted as in E2
    pairs = S.minimal_pairs(test.grid, cfg, cfg.confirm_seed)  # identical pairs to E2 (same seed)
    b, s = pairs.base.values, pairs.source.values
    clean = rd.decode(test.col.readout.numpy(), test.grid.cell.values)  # clean decodes
    zb, zs = (rd.z(test.col.readout.numpy()[i][:, fam4a]) for i in (b, s))
    out = pairs.assign(clean_decode_base=clean[b], clean_decode_source=clean[s],
                       eligible=(np.abs(S.circ_diff(clean[b], test.grid.theta.values[b])) <= cfg.tol_deg)
                       & (np.abs(S.circ_diff(clean[s], test.grid.theta.values[s])) <= cfg.tol_deg),
                       effect_denominator=((zs - zb) ** 2).sum(1))
    sets = {"family": fam3b, "complement": np.setdiff1d(np.arange(480), fam3b), "all": np.arange(480)}
    for name, ch in sets.items():  # per-pair outcomes of each reference set
        Rp = I.patched_readouts(model, "mixed4a", test.col.windows, b, s, [ch], fam4a, lambda R: R)[0]
        zp = rd.z(Rp)
        dec = rd.predict_z(zp, test.grid.cell.values[b])
        out[f"{name}_decode"] = dec
        out[f"{name}_success"] = np.abs(S.circ_diff(dec, clean[s])) <= cfg.tol_deg  # Huang et al.'s success criterion
        out[f"{name}_effect_numerator"] = ((zp - zb) * (zs - zb)).sum(1)
    out.to_csv(out_dir / "data" / "pairs_h1_confirmatory.csv", index=False)
    log.info("export: H1 family IIA on eligible pairs = %.3f (must match results/e2_synthetic.json)",
             out.family_success[out.eligible].mean())  # consistency check against the confirmatory run
    (out_dir / "results").mkdir(exist_ok=True)
    for f in list(cfg.results_dir.glob("*.json")) + [ROOT / "reports" / "generated" / "results.md", ROOT / "reports" / "preregistration.md"]:
        shutil.copy(f, out_dir / "results" / f.name)  # aggregate results travel with the data
    shutil.copy(ROOT / "docs" / "hf_dataset_card.md", out_dir / "README.md")  # dataset card
    log.info("export written to %s/", out_dir.name)
