# All experiments

This page covers the overall story, the scientific-method loop, and decisions that apply across experiments.
Details live in the per-experiment reports, which this page does not repeat:
[E1 replication](e1_replication.md) · [E2 synthetic interventions](e2_synthetic.md) · [E3 natural images](e3_natural.md)
· [pre-registration](preregistration.md) · [generated tables](generated/results.md) · [paper](paper/paper.pdf).

## Question → hypothesis → test → result

| Step | Content |
|---|---|
| Domain | Mechanistic interpretability of CNNs at the neuron level (Circuits thread). |
| Question | Is the Circuits explanation "the `mixed3b` curve units detect curve orientation" also a *causal* explanation? Do the units mediate orientation for downstream computation ([Huang et al. 2023](https://arxiv.org/abs/2309.10312) intervention mode)? |
| Prior knowledge | Observational and weight-based evidence from [Cammarata et al. 2020](https://distill.pub/2020/circuits/curve-detectors/) and [2021](https://distill.pub/2020/circuits/curve-circuits/). Equivariance ([Olah et al. 2020](https://distill.pub/2020/circuits/equivariance/)). SAE "missing curve detectors" ([Gorton 2024](https://arxiv.org/abs/2406.03662)). See `docs/literature_review.md`. |
| Hypothesis H1 (strong) | Swapping the 10 units makes the next layer decode the source orientation (IIA ≥ 0.5, beating matched nulls). |
| Falsifier (pre-registered) | IIA < 0.2, or a margin over the null that is confidently below 0.05. |
| E1 (gate) | The published tuning replicates: 3b r = 0.969, 4a r = 0.847, with the orientation convention fitted on 3a only. |
| E2 (test) | **H1 falsified**: IIA 0.177 [0.156, 0.199]. The family is still far above every matched null (IIA null q95 ≤ 0.001; effect 0.410 vs 0.098). |
| E3 (generalisation) | The same pattern holds on natural backgrounds (IIA 0.190, effect 0.415). Under a natural rotation counterfactual the effect is 0.287 vs a null q95 of 0.075. Ablation changes 17.2% of top-1 predictions vs 9.3% for matched ablations. |
| Refined hypothesis | **The curve family is a specific, partial mediator of a distributed, redundant orientation code.** Its effect is roughly additive (about 0.04 of the readout displacement per unit). It carries curvature evidence that 4a curve units use (H2: line → arc flips 81%), but the other `mixed3b` channels also carry orientation (complement IIA 0.40; about 80 channels needed for IIA ≈ 0.8). |

## Overview of all pre-registered tests

| Test | IIA family | Effect family | Δ-null q95 (effect) | Verdict |
|---|---|---|---|---|
| H1 primary (3b → 4a curve units) | 0.177 | 0.410 | 0.098 | falsified (pre-registered rule) |
| H1-ridge (3b → all 4a) | 0.105 | 0.248 | 0.106 | supported (secondary criterion) |
| H2 curvature swap | 0.405 | 0.393 | 0.141 | supported |
| H4 3a → 3b curve units | 0.058 | 0.324 | 0.169 | supported |
| Composited (natural backgrounds) | 0.190 | 0.415 | 0.104 | supported |
| H5 natural rotation | 0.072 | 0.287 | 0.075 | supported |
| H1, jitter seeds 3 / 4 | 0.172 / 0.171 | 0.410 / 0.407 | 0.094 / 0.104 | falsified (same as H1) |
| H1, Lucent default LRN | 0.172 | 0.406 | 0.096 | falsified (same as H1) |

"Supported" for a secondary hypothesis means the effect margin over the activation-change-matched null q95 has a
95% CI above 0 and a Holm-adjusted p < 0.05. All Holm-adjusted p ≤ 0.004.

**Post-hoc robustness** (added after an independent code review; [details](posthoc_robustness.md)):
- A stricter rank-matched null and the strongest competitor set (top-10 non-family channels by activation change)
  leave every conclusion unchanged.
- The H1 effect is 0.410, against a rank-null q95 of 0.121; the competitor set reaches only 0.125.
- The ridge readout's decoded orientation moves 57% of the way to the source.
- Blur-free H5 rotations give a *larger* effect: 0.336 vs 0.287.

![summary of effects](../figures/summary_effect.png)

## Learnings: what we tried, what went wrong, and why we chose what we did

### Design decisions
- **Explanation.** We chose the curve family over high-low frequency or the 4e:55 "V-shape" claim, for three
  reasons: it is the best-characterised family, orientation is an exactly swappable variable, and the authors'
  tuning data allow a replication gate. Decided with the user.
- **Readout.** We use the next layer's published curve units rather than logits. A lone arc's class logits are
  off-distribution and outside the explanation's claims. The circularity this introduces is handled by the ridge
  readout over all of 4a, by Δ-matched nulls, and by natural-image tests. All three were suggested by an
  independent design critique before any data.
- **Nulls.** A random-channel null would be beaten trivially: most channels barely change with orientation, and the
  family sits in the top 4% by activation change. Matching on activation change Δ is the honest bar.
- **Statistics.** We used the pigeonhole bootstrap because pairs share images, and we recompute the null q95 inside
  every replicate. We dropped the plan's "power analysis with seeds": the model is deterministic, so seeds only
  change jitter. Instead we did a precision analysis (target CI half-width ≤ 0.03; achieved about 0.021) and three
  jitter seeds as a stability check.
- **Hyperparameters.**
  - 72 orientations at 5° steps; decoder tolerance 10°.
  - Radii {30, 45, 60, 80}, chosen on the pilot split where 3b and 4a matched the published tuning best.
  - Widths {3, 5}; arc span 120° (under 180°, so an arc never reads as a full circle).
  - 4 sources per base with |Δθ| ≥ 30°, so that base and source decodes are clearly distinct.
  - 2000 null sets, for p ≈ 5·10⁻⁴ resolution.
  - Effect denominator filter: the bottom 5% of pairs.

### Problems and fixes, in chronological order
1. **Lucent's LRN constants are not TensorFlow's.**
   - Fixed with a namespace shim.
   - The TF-exact constants (window 11 = 2·depth_radius + 1) come from the discussion of
     [lucent#51](https://github.com/greentfrapp/lucent/pull/51). The PR itself proposes size 10.
   - Sensitivity: the conclusions are unchanged.
   - Erratum: the frozen pre-registration says "per lucent#51". More precisely, the constants are per a comment in
     that PR's discussion.
2. **`torch-lucent` 0.1.8 on PyPI pins `kornia<=0.4.1`, and Lucent's `setup.py` breaks on Windows with cp1252.**
   Installed from git, with a uv `extra-build-variables` fix for `PYTHONUTF8`.
3. **PIL draws arc strokes inward.** The first renderer was off by half a stroke width. A unit test caught it, and
   we fixed the bounding radius.
4. **Cropped vs full forward differed by up to about 3e-3 in fp32,** about 1e-5 of the value range. We traced this
   to cuDNN choosing different algorithms for different input sizes. In fp64 the crop is exact: a re-check measured
   a difference of 0, and the test asserts ≤ 1e-9. The test now checks fp64.
5. **The first pilot ran for over 15 minutes, CPU-bound.** py-spy showed the process blocked on GPU
   synchronisation, because the 8 GB WDDM GPU was paging. Fixes: gather only the patched channels, call
   `empty_cache()`, and use batch 64. Afterwards: 0.15 s per null set.
6. **Quick-mode seeds equalled the confirmatory seeds.** We offset them by 100 before any interventional smoke test
   touched confirmatory data, and disclosed the E1 smoke run.
7. **The pilot already showed IIA ≈ 0.16.** We did not lower the IIA threshold. The primary result is therefore
   reported as "falsified", with the specificity and redundancy findings as pre-registered secondaries.
8. **A dose-response plot used a categorical x-axis for K = 1, 2, 3, 5, 7, 10,** which distorts the linear trend.
   Replaced with a numeric axis; all E2/E3 figures are now drawn from JSON in the report phase.

9. **The independent code review found no result-changing bug, but did find weak spots.** The decile-matched null
   is soft at the extreme tail; the H1-ridge effect lives in z-space; H5's ±45° rotations add blur; the bootstrap
   is somewhat narrow; and absolute local paths had leaked into the results.
   - The first three were checked with post-hoc analyses, and the conclusions are unchanged.
   - The bootstrap issue is documented as a limitation.
   - The paths were scrubbed, and the code now writes repo-relative paths.
   - Downloads are now checksum-verified.
   - Commits on GitHub from before the scrub still contain the old absolute paths: a local username, no
     credentials.

### What did not work, or remains open
- **Strict sufficiency of the 10 units.** No configuration reached IIA ≥ 0.5 short of patching more than 40
  channels.
- **The family-rotation activation swap** shifts the decoded orientation in the right direction, but by only 21–39%
  of the predicted angle (k = 5 aside). This is a channel reassignment within the family, inspired by the
  concept-swap tests of Gurnee et al. 2026.
- **Open:** whether SAE features ([Gorton 2024](https://arxiv.org/abs/2406.03662)) or a learned subspace
  (distributed alignment search, [Geiger et al. 2024](https://arxiv.org/abs/2303.02536)) would form a
  *sufficient* orientation mediator with few dimensions.
