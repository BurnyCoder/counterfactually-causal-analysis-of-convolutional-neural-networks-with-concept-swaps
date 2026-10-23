"""Pipeline entry point: run the experiment phases in order.

Local: parses `--quick` (smoke test on tiny grids) and `--phase` (run one phase; later phases read earlier results
from results/*.json), sets up timestamped logging, loads InceptionV1 once and calls each phase.
Global context: `uv run python main.py` reproduces every number, figure and table in reports/ and the paper.

    E1 replication   -> results/e1_replication.json  (observational gate: do we reproduce the published tuning?)
    E2 synthetic     -> results/e2_synthetic.json    (H1 primary, H1 ridge, H2, H4, stability, LRN sensitivity)
    E3 natural       -> results/e3_natural.json      (composited arcs, H5 rotation counterfactual, relevance)
    report           -> reports/generated/*.md + figures/summary_effects.png (tables generated from the JSONs)
"""

import argparse
import json

import numpy as np
import torch

from concept_swap.config import load_config
from concept_swap.log import get_logger, setup_logging
from concept_swap.model import load_inceptionv1
from concept_swap.phases import e1_replication, e2_synthetic, e3_natural, report_tables

PHASES = ("e1", "e2", "e3", "report")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test: small grids, few null sets, results/quick/")
    ap.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    args = ap.parse_args()
    cfg = load_config()
    cfg = cfg.with_quick() if args.quick else cfg
    path = setup_logging(cfg.log_dir)
    log = get_logger("main")
    log.info("log file: %s", path)
    log.info("config: %s", json.dumps(cfg.as_log_dict()))
    rng = np.random.default_rng(cfg.confirm_seed)  # single RNG stream for null sets and bootstraps
    torch.manual_seed(cfg.confirm_seed)
    run = lambda name: args.phase in (name, "all")
    model = load_inceptionv1(cfg.device, cfg.lrn_mode) if any(run(p) for p in PHASES[:3]) else None
    load = lambda n: json.loads((cfg.results_dir / f"{n}.json").read_text())["results"]
    e1 = e1_replication.run(cfg, model, rng) if run("e1") else None
    if run("e2"):
        e2_synthetic.run(cfg, model, rng, e1 or load("e1_replication"))
    if run("e3"):
        e3_natural.run(cfg, model, rng)
    if run("report"):
        report_tables.run(cfg)
    log.info("done")


if __name__ == "__main__":
    main()
