# Pilot (development only; excluded from every reported test)

These are the as-run scratch scripts and outputs that informed the design choices disclosed in
[`../preregistration.md`](../preregistration.md). They use only pilot jitter seeds (0 and 100) and are not part of
the pipeline. Run them from the repo root, for example `uv run python reports/pilot/pilot_h1.py`.

| File | What it did | Used for |
|---|---|---|
| `pilot_tuning_radii.py` / `.log` | 3a/3b/4a tuning vs the published curves for radii 10–80 px (seed 0). This log is a deterministic re-run that reproduces the original output. | Choosing radii {30, 45, 60, 80}, where 3b r = 0.93–0.97 and 4a r = 0.76–0.85 (radius 10: 0.88 and 0.61). Also showed that the orientation convention (reflection −1, shift 180°) is the same for every layer. |
| `pilot_h1.py` / `.log` | The H1 IIA procedure on pilot seeds: selection 0, test 100, with 200 null sets per null and 500 bootstrap replicates (no continuous effect, no random null). | Runtime estimate. The precision analysis (IIA CI half-width 0.022 at 4608 pairs). The disclosed pilot result (family IIA 0.162, null q95 0.001). |

Only the interpretation-free numbers above entered the design. The H1 decision thresholds were fixed in the
approved plan before the pilot and were not changed afterwards.
