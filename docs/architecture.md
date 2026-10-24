# Code architecture and data flow

`main.py` is the only entry point. It loads the configuration and the model once, then calls the phases in order.
- E1, E2 and E3 each write `results/<phase>.json`; E1 also draws its figures.
- `hero` writes the README figure and GIF.
- `examples` writes the example-swap figures and `reports/generated/examples.md` (output-only).
- `report` writes the generated tables, LaTeX macros and the E2/E3/summary figures from the JSONs.
- `posthoc`, `audit`, `audit2` and `export` run only when named. They write `results/posthoc_robustness.json`,
  `results/posthoc_audit.json`, `results/posthoc_audit2.json` and `hf_export/`.
- Library modules hold no experiment logic.

```mermaid
graph TD
  ENV[.env / .env.example] --> CFG[config.py<br/>frozen Config]
  CFG --> MAIN[main.py]
  MAIN --> LOG[log.py<br/>logs/&lt;timestamp&gt;.log + terminal]
  MAIN --> MODEL[model.py<br/>Lucent InceptionV1, TF-exact LRN,<br/>capture/patch hooks, run_block, cropped readout]
  MAIN --> E1[phases/e1_replication.py]
  MAIN --> E2[phases/e2_synthetic.py]
  MAIN --> E3[phases/e3_natural.py]
  MAIN --> REP[phases/report_tables.py]
  MAIN --> HERO[phases/hero.py<br/>README figure + GIF]
  MAIN --> EXA[phases/examples.py<br/>example-swap figures]
  MAIN -.->|only when named| PH[phases/posthoc.py<br/>rank-matched null, exact rotations]
  MAIN -.->|only when named| AU[phases/audit.py<br/>effect-ranked competitor, ridge complement,<br/>H2 per direction, nuisance sham]
  MAIN -.->|only when named| AU2[phases/audit2.py<br/>ridge-readout causal ranking,<br/>additivity by 4a branch, H5 blur on matched angles]
  MAIN -.->|only when named| EXP[phases/export_hf.py<br/>hf_export/]
  AU --> COMMON
  AU --> PH
  AU --> E2
  AU --> E3
  AU2 --> COMMON
  AU2 --> AU
  AU2 --> PH
  AU2 --> E3
  PH --> COMMON
  PH --> E2
  PH --> E3
  HERO --> COMMON
  EXA --> COMMON
  EXA --> HERO
  EXA --> FV[featvis.py<br/>Lucent feature visualizations]
  FV --> MODEL
  EXP --> COMMON
  E1 --> STIM[stimuli.py<br/>arcs/lines, grids, minimal pairs]
  E1 --> COMMON
  E2 --> COMMON[phases/common.py<br/>interchange_experiment, summarize]
  E3 --> COMMON
  E3 --> NAT[natural.py<br/>Imagenette download, crops, rotations]
  COMMON --> STIM
  COMMON --> INT[interventions.py<br/>cached RF windows, batched swaps/ablations]
  COMMON --> FAM[families.py<br/>published families, null sets, Fisher]
  COMMON --> RO[readouts.py<br/>z-score, template/ridge decoders, effect]
  COMMON --> ST[stats.py<br/>pigeonhole bootstrap, p-values, decision rule]
  INT --> MODEL
  FAM --> DATA[(data/circuits/*.json<br/>published tuning)]
  E1 --> PLOT[plots.py]
  HERO --> PLOT
  EXA --> PLOT
  REP --> PLOT
  E1 --> R1[(results/e1_replication.json)]
  E2 --> R2[(results/e2_synthetic.json)]
  E3 --> R3[(results/e3_natural.json)]
  R1 --> REP
  R2 --> REP
  R3 --> REP
  PH --> R4[(results/posthoc_robustness.json)]
  R4 --> REP
  AU --> R5[(results/posthoc_audit.json)]
  R5 --> REP
  AU2 --> R6[(results/posthoc_audit2.json)]
  R6 --> REP
  REP --> MD[(reports/generated/results.md<br/>numbers.tex)]
```

## What happens when you run `uv run python main.py`

1. **Config and logging.** `.env` is read, and the full config is logged to `logs/<timestamp>.log` and the terminal.
2. **Model.** Lucent's InceptionV1 is loaded. The weights (`inception5h.pth`, 28 MB) are downloaded once by
   torch.hub. Its `F.local_response_norm` calls are redirected to TF-exact constants.
3. **E1, observational replication (13 s including start-up).**
   - Render 1152 arcs and 1152 lines from the selection split.
   - Run them through the network and read `mixed3a`/`mixed3b` at (13,13) and `mixed4a` at (6,6).
   - Average activation per orientation to get tuning curves.
   - Fit one orientation convention on 3a, then correlate 3b and 4a with the published curves against a
     label-permutation null.
   - Compute each 3b unit's AUROC against other orientations and against lines.
4. **E2, synthetic interventions (about 40 minutes on an RTX 5070 Laptop).**
   - Render the selection and confirmatory splits and cache, per image, the 11×11×480 `mixed3b` window that feeds
     `mixed4a` unit (6,6).
   - Fit z-scores and decoder templates on the selection split.
   - For each neuron set (the family, Fisher top-10, the complement, all channels, and thousands of null sets),
     overwrite those channels in the base window with the source window's values.
   - Recompute only `maxpool4 → mixed4a` on the window, then decode the orientation.
   - Repeat the same machinery for the ridge readout, H2 (lines ↔ arcs), H4 (3a → 3b windows), the stability seeds
     and the Lucent-LRN model.
5. **E3, natural images (about 16 minutes).**
   - Download Imagenette-320 once (342 MB).
   - Composite arcs onto photo backgrounds and rerun H1.
   - For curve-present photos, use the rotated photo as the source (H5).
   - Mean-ablate the family over whole maps and count changes in the network's top-1 prediction.
6. **Hero.** Re-renders the selection and confirmatory splits and draws the README example figure and GIF for the
   pre-chosen cell (θ = 0° base, radius 45, width 3, dark-on-light).
7. **Examples (about 3 minutes on an RTX 5070 Laptop; log `20261024_024626`).**
   - Re-render the two splits, recompute the H1 family swap for all 4608 pairs and assert that the IIA equals
     `results/e2_synthetic.json`.
   - Draw 6 random successes and 3 random failures (`stats.random_rows`, seed 0), plus the hero pair.
   - Capture their full `mixed3b` maps, swap the family's 11×11 window, re-run `mixed4a` on the full maps, and
     assert that this equals the scored readout.
   - Render feature visualizations of the 24 curve units with Lucent (512 steps, redirected ReLU).
   - Draw one flow figure per pair, an overview, and the neuron grid, and write `reports/generated/examples.md`.
8. **Report.** Markdown tables (`reports/generated/results.md`), LaTeX macros, and the E2/E3 and summary figures
   are produced from the JSON files only.

## Design choices

- **Cropped-window recomputation.** Thousands of null sets × 4608 pairs would mean millions of full forward passes.
  We re-run only the block between the patched layer and the readout, on the receptive-field window.
  `tests/test_model.py` checks that this equals the full network in float64: it asserts ≤ 1e-9, and a re-check
  measured a difference of 0.
- **One generic experiment function.** `interchange_experiment` serves every hypothesis. Only the splits, pairs,
  family, readout and labels change, which keeps H1, H2, H4, H5 and the robustness runs comparable.
- **No hand-copied numbers.** Reports quote `reports/generated/results.md`, which `report_tables.py` writes from
  the JSONs.
