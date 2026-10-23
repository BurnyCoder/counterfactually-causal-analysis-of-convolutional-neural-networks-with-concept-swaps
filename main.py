"""Pipeline entry point: run the experiment phases in order.

Local: parses `--quick` (smoke test on tiny grids) and `--phase` (run one phase; later phases read earlier results
from results/*.json), sets up timestamped logging, loads InceptionV1 once and calls each phase.
Global context: `uv run python main.py` reproduces the pre-registered numbers, figures and tables; the post-hoc
numbers in reports/ and the paper also need `--phase posthoc`, then `--phase audit`, then `--phase report`.

    E1 replication   -> results/e1_replication.json  (observational gate: do we reproduce the published tuning?)
    E2 synthetic     -> results/e2_synthetic.json    (H1 primary, H1 ridge, H2, H4, stability, LRN sensitivity)
    E3 natural       -> results/e3_natural.json      (composited arcs, H5 rotation counterfactual, relevance)
    hero             -> figures/hero_concept_swap.png, figures/concept_swap_sweep.gif (README illustration)
    report           -> reports/generated/{results.md,numbers.tex} + figures/summary_*.png (generated from the JSONs)
    posthoc          -> results/posthoc_robustness.json (post-hoc, NOT pre-registered: stricter null, ridge output space, exact rotations)
    audit            -> results/posthoc_audit.json (post-hoc, NOT pre-registered: checks added by the independent audit)
    export           -> hf_export/ (Hugging Face dataset: stimuli, per-pair H1 outcomes, results; run explicitly)
"""

import argparse  # command-line flags: https://docs.python.org/3/library/argparse.html
import json

import numpy as np
import torch

from concept_swap.config import load_config
from concept_swap.log import get_logger, setup_logging
from concept_swap.model import load_inceptionv1
from concept_swap.phases import audit, e1_replication, e2_synthetic, e3_natural, export_hf, hero, posthoc, report_tables

PHASES = ("e1", "e2", "e3", "hero", "report")  # run in this order by `--phase all` (the default)


def main() -> None:
    """Parse flags, configure logging and seeds, load the model once, and run the requested phases.

    Local: phases share one numpy Generator so null sets and bootstraps are reproducible for a given phase order.
    Global context: the only executable entry point of the repository.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test: small grids, few null sets, results/quick/")
    ap.add_argument("--phase", choices=(*PHASES, "posthoc", "audit", "export", "all"), default="all")
    args = ap.parse_args()
    cfg = load_config()  # .env -> Config
    cfg = cfg.with_quick() if args.quick else cfg  # shrunken grids + offset seeds for smoke tests
    path = setup_logging(cfg.log_dir)  # logs/<timestamp>.log plus the terminal
    log = get_logger("main")
    log.info("log file: logs/%s", path.name)  # file name only: no absolute paths in logs
    log.info("config: %s", json.dumps(cfg.as_log_dict()))  # the full run configuration, for reproducibility
    rng = np.random.default_rng(cfg.confirm_seed)  # single RNG stream for null sets and bootstraps
    torch.manual_seed(cfg.confirm_seed)  # torch RNG (used by rf_center's random images)
    run = lambda name: args.phase in (name, "all")  # should this phase run?
    needs_model = any(run(p) for p in PHASES[:4]) or args.phase in ("posthoc", "audit", "export")  # report reads JSON only
    model = load_inceptionv1(cfg.device, cfg.lrn_mode) if needs_model else None
    load = lambda n: json.loads((cfg.results_dir / f"{n}.json").read_text())["results"]  # earlier phase's results
    e1 = e1_replication.run(cfg, model, rng) if run("e1") else None
    if run("e2"):
        e2_synthetic.run(cfg, model, rng, e1 or load("e1_replication"))  # E2 needs E1's preferred orientations
    if run("e3"):
        e3_natural.run(cfg, model, rng)
    if run("hero"):
        hero.run(cfg, model)
    if run("report"):
        report_tables.run(cfg)
    if args.phase == "posthoc":  # explicit only: post-hoc robustness analyses
        posthoc.run(cfg, model, rng)
    if args.phase == "audit":  # explicit only: checks added by the independent audit (post-hoc)
        audit.run(cfg, model, rng)
    if args.phase == "export":  # explicit only: writes the Hugging Face upload folder
        export_hf.run(cfg, model)
    log.info("done")


if __name__ == "__main__":
    main()
