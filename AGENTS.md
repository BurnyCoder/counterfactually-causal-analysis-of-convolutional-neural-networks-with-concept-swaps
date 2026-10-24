# AGENTS.md: notes for coding agents working in this repo

## Layout
Read `docs/architecture.md` for the module graph and data flow.
- `main.py` only wires phases together.
- Experiment logic lives in `src/concept_swap/phases/`.
- Reusable mechanics live in the library modules: `config`, `log`, `model`, `stimuli`, `interventions`, `readouts`,
  `families`, `stats`, `natural`, `featvis`, `plots`.

## Commands
```bash
uv sync                                  # install (Python 3.12, torch cu130 from the PyTorch index)
uv run pytest -q                         # 41 tests; model tests download inception5h.pth on first use
uv run python main.py --quick            # smoke test of the whole pipeline (offset seeds, results/quick/)
uv run python main.py                    # full run (writes results/, figures/, reports/generated/results.md)
uv run python main.py --phase report     # regenerate tables/figures from existing results/*.json
uv run python main.py --phase examples   # example-swap figures + reports/generated/examples.md (output-only)
uv run python main.py --phase audit      # post-hoc audit checks (results/posthoc_audit.json); then --phase report
uv run python main.py --phase audit2     # third-audit checks (results/posthoc_audit2.json); then --phase report
```

## Rules that protect the science
- **`reports/preregistration.md` is frozen.**
  - Do not change hypotheses, thresholds, null sizes, seeds or the primary metric. Report any deviation in
    `reports/` as a deviation, with its reason.
  - New analyses are **exploratory** and must be labelled that way.
- **Never run interventional analyses on the confirmatory seeds (2, 3, 4) or the confirmatory Imagenette half during
  development.** Use `--quick` (seeds 101–104; photos from the real selection half only, `natural.image_halves`) or
  the pilot seeds (0, 100).
- **Numbers in reports and the paper come from `reports/generated/results.md`**, which `report_tables.py` writes.
  Regenerate that file; never edit it by hand.
- **The interventions rely on invariants enforced in `tests/test_model.py`:** the cropped-window readout equals the
  full forward pass, and the patch hooks are identities. Keep these tests passing when touching `model.py`.

## Conventions
- Configuration comes from `.env` (template: `.env.example`) through `config.load_config()`. Prefer config values
  over magic numbers in phases; the remaining constants (e.g. 10,000 E1 permutations, 200 ablation sets, 100
  dose-response subsets) are documented in the pre-registration or the reports.
- Logging goes through `concept_swap.log.get_logger`, which writes to both `logs/<timestamp>.log` and stdout.
- Every module and library/phase function has a docstring with a local explanation and its global role. Cite the source (paper,
  library doc) next to any non-obvious line.
- Imagenette images must never be committed or uploaded, because they are ImageNet-derived. Only file names may
  be stored.
