---
license: cc-by-4.0
pretty_name: InceptionV1 curve-detector concept swaps
tags:
- interpretability
- mechanistic-interpretability
- causal-intervention
- activation-patching
- inceptionv1
- circuits
size_categories:
- 1K<n<10K
configs:
- config_name: pairs_h1
  data_files: data/pairs_h1_confirmatory.csv
- config_name: stimuli
  data_files: data/stimuli.csv
---

# InceptionV1 curve-detector concept swaps

This dataset holds the stimuli and per-pair outcomes from
[counterfactually-causal-analysis-of-convolutional-neural-networks-with-concept-swaps](https://github.com/BurnyCoder/counterfactually-causal-analysis-of-convolutional-neural-networks-with-concept-swaps).
That repository tests causally whether the 10 InceptionV1 `mixed3b` curve detectors
([Cammarata et al. 2020](https://distill.pub/2020/circuits/curve-detectors/)) mediate curve orientation for the next
layer, using interchange interventions ([Huang et al. 2023](https://arxiv.org/abs/2309.10312)).

## Contents

| File | Description |
|---|---|
| `data/stimuli.csv` | Synthetic arcs for the selection split (jitter seed 1) and the confirmatory split (seed 2). Columns: `split`, `idx` (row in the split), `kind`, `theta` (degrees; direction from the arc midpoint to its centre of curvature), `radius`, `width`, `polarity` (0 = dark on light), `cell` (nuisance cell), `dy`/`dx` (jitter in pixels). |
| `images/<split>/<idx>.png` | The rendered 224×224 stimuli, with zero-padded file names (`0000.png`, `0001.png`, …). The arc midpoint is at pixel (111,111), the receptive-field centre of `mixed4a` unit (6,6), shifted by the cell's jitter (`dy`, `dx`). |
| `data/pairs_h1_confirmatory.csv` | The 4608 confirmatory (base, source) minimal pairs. Columns: `base` and `source` (the `idx` of the confirmatory stimuli); `dtheta`; clean decodes of base and source; `eligible`; then, for `family` (the 10 curve units), `complement` (the other 470 channels) and `all`, the decoded orientation after the swap, `*_success` (within 10° of the clean source decode) and `*_effect_numerator` (⟨z_p−z_b, z_s−z_b⟩). `effect_denominator` is ‖z_s−z_b‖². |
| `results/*.json` | All aggregate results: E1 replication, E2 synthetic, E3 natural images, plus the post-hoc (not pre-registered) robustness checks in `posthoc_robustness.json` and the post-hoc audit checks in `posthoc_audit.json` and `posthoc_audit2.json`. Per-set null distributions are included for the pre-registered tests (`e2_synthetic.json`, `e3_natural.json`); the post-hoc JSONs store null summaries (mean, q95, p) only. The `verdict` field in each test applies the H1 three-way rule to every test. The pre-registered verdicts of the secondary tests are in `results/results.md`. |
| `results/results.md` | The generated results tables. |
| `results/preregistration.md` | The pre-registered hypotheses and decision rules. |

The headline per-pair statistic can be recomputed from `pairs_h1_confirmatory.csv`: IIA = mean of `family_success`
over `eligible` rows. The dataset is reproducible with `uv run python main.py`, then `--phase posthoc`,
`--phase audit`, `--phase audit2` and `--phase export` (each as `uv run python main.py --phase <name>`).

## Not included
- **Natural-image pixels.** Imagenette is an ImageNet subset, so it is not redistributed. `results/e3_natural.json`
  lists only the relative file names of the confirmatory images (`image_ids_test`; Windows path separators in the
  reported run).
- **Model weights.** These are the Lucid InceptionV1 weights via
  [ProGamerGov's conversion](https://github.com/ProGamerGov/pytorch-old-tensorflow-models).
