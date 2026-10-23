# Post-hoc robustness checks (not pre-registered)

These analyses were added **after** the confirmatory run. A single-pass independent code review raised three ways
the pre-registered comparisons could flatter the curve family. Nothing here changes a pre-registered verdict.

Numbers come from `results/posthoc_robustness.json` (run log `logs/20261023_184009.log`). They are tabulated in
[`generated/results.md`](generated/results.md#post-hoc-robustness-not-pre-registered-added-after-code-review).
Run it with `uv run python main.py --phase posthoc`.

## Why

1. **The decile-matched null is a soft baseline.**
   - All 10 curve units sit at the 96.5th–99.4th percentile of activation change Δ, inside the top decile.
   - A decile-matched null set draws 10 of the other 38 top-decile channels, and most of those change *less* than
     the family.
   - The new **rank-matched null** matches each curve unit to one of its 10 nearest non-family channels by Δ (1000
     sets).
   - We also report the **single strongest competitor**: the 10 non-family channels with the largest Δ.
2. **The pre-registered H1-ridge "effect" was measured over all 508 z-scored `mixed4a` channels.** That says the
   whole layer moves toward the source, not that the *decoded orientation* does. We now also measure the effect in
   the ridge decoder's (cos θ, sin θ) output space.
3. **H5 confounded rotation with resampling blur.**
   - The 0° base is never resampled, while ±45° rotations, and ±90° on non-square images, go through bilinear
     interpolation.
   - We repeat H5 on square crops rotated by −90/90/180°, which PIL performs as exact transposes. This is verified
     against `np.rot90` in `tests/test_natural.py`.

## Results

| Test | Family effect [95% CI] | Rank-null q95 | Decile-matched null q95 | Top-Δ non-family set: IIA / effect |
|---|---|---|---|---|
| H1 (3b → 4a curve units) | 0.410 [0.400, 0.419] | 0.121 | 0.098 | 0.000 / 0.125 |
| H1-ridge, z-space (as pre-registered) | 0.248 | 0.135 | 0.106 | 0.000 / 0.155 |
| **H1-ridge, decoded (cos, sin) space** | **0.573** [0.564, 0.583] | 0.126 | 0.097 | — |
| H2 curvature swap | 0.393 | 0.100 | 0.141 | 0.068 / 0.172 |
| H4 3a → 3b | 0.324 | 0.146 | 0.169 | 0.000 / 0.116 |
| Composited arcs on photos | 0.415 | 0.106 | 0.104 | 0.001 / 0.123 |
| H5 natural rotation (as pre-registered) | 0.287 | 0.073 | 0.075 | 0.000 / 0.083 |
| **H5, exact rotations only (no blur)** | **0.336** [0.310, 0.358] | 0.077 | 0.079 | 0.000 / 0.066 |

- The decile-matched column is the pre-registered null type. Its values come from the pre-registered runs, except
  in the two bold rows, where the decile-matched null was run post hoc.
- Every margin over the rank-matched q95 has a 95% pigeonhole CI above 0.
- Every permutation p is at its floor: 0.001 with 1000 sets, 0.002 with 500.
- IIA against the rank-matched null: the null q95 is at most 0.003 for every test except H2 (0.017).
- **H5 exact-rotation IIA** is 0.108 [0.080, 0.141], against 0.072 for the original rotations.

## Interpretation
- **The stricter null changes little.**
  - The rank-matched q95 rises above the decile-matched one for H1 (0.121 vs 0.098) and H1-ridge (0.135 vs 0.106).
    It is similar or lower for the other tests: H2 0.100 vs 0.141; H4 0.146 vs 0.169; composited 0.106 vs 0.104;
    H5 0.073 vs 0.075.
  - Even the strongest 10 non-family channels move the 4a curve code only 12.5% of the way, against 41% for the
    family. They transfer the decoded orientation in only 1 of 4574 pairs.
  - "The family is special among channels that change as much" survives.
- **On the decoded orientation itself, the family's effect is larger than the z-space number suggested.**
  - The ridge readout's predicted orientation vector moves 57% of the way to the source's, yet lands within 10° of
    it in only 10.5% of pairs.
  - So the swap rotates the decoded orientation part of the way, consistent with partial mediation.
- **The blur confound did not create the H5 effect.** With blur-free rotations the effect is *larger* (0.336 vs
  0.287), and IIA rises from 0.072 to 0.108.
- **The pre-registered conclusion stands:** a specific but partial mediator. H1's strict sufficiency claim remains
  falsified under every analysis.

## Other review findings, handled without new analyses
- **H5 clean decoding accuracy is 1.0 by construction,** because the templates are the photo's own clean versions.
  It is not evidence that the decoder works; success still requires the patched base to match the rotated version
  better than the unpatched base.
- **The composited test reuses the same 32 photo backgrounds in both splits.** It generalises across jitter, not
  backgrounds.
- **The pigeonhole bootstrap treats base and source as separate factors** although images play both roles, and it
  ignores jitter-cell clustering, so the CIs are somewhat narrow. The seed-to-seed spread (IIA 0.171–0.177) and
  the size of every margin make this immaterial for the verdicts.
- **The effect's 5% denominator filter used all pairs,** not only eligible ones as pre-registered. At most 1.7% of
  pairs are ineligible.
- **Reliability and privacy:**
  - Result files and logs now store repo-relative paths.
  - The Imagenette archive (SHA-256) and the InceptionV1 weights (SHA-256) are verified on use.
  - The archive downloads atomically, and the validation image count is checked.

## Second round: independent audit (not pre-registered)

A second, independent audit re-checked the finished study from scratch. It covered the pre-registration, every
module, the run logs, the git history and every number in the README and paper. It ran in phase `audit`:
`uv run python main.py --phase audit` writes `results/posthoc_audit.json` (run log `logs/20261023_210800.log`).

### What the audit verified, with no change needed
- **Pre-registration integrity.**
  - The pre-registration was committed at 17:13:04, and the only confirmatory run started at 17:13:49.
  - Between the registered code (`de6f002`) and the results commit, the code changed only in comments and in where
    figures are drawn. A whitespace- and comment-insensitive diff shows this.
  - Smoke runs used seeds 101–104, except one disclosed E1-only run on seed 1.
  - The hero-figure rule was committed at 17:20:08, before H1's first result was logged at 17:26:04.
- **Intervention exactness.** The crop indices were re-derived by hand: 4a(6,6) ← `maxpool4` cells 4–8 ← 3b rows
  8–18. The "all channels" set gives IIA = effect = 1.000.
- **Statistics.** Holm-adjusted p-values were recomputed. Every README and paper number traced back to the JSONs.
- **Relevance null.** We suspected that the variance-matched null might under-match the family, as the decile null
  did for patching. It does not (table below).

### Claims that lacked their own test, and the outcome that would overturn each (written before running)
| Claim | Check | Overturning outcome |
|---|---|---|
| The family is special (beats a "data-driven" set) | Rank all 480 channels by single-channel patching effect on the **selection** split; score the top-10 and the top-10 non-family on the confirmatory pairs | Non-family top-10 effect ≥ family effect |
| The family carries curvature (H2: 81% line → arc) | Each swap direction against its own decile-matched null (1000 sets) | Line → arc null q95 ≥ 0.807 |
| Ablating the family disrupts orientation far more than matched channels | Mean ablation against the rank-matched null (200 sets) | Rank-null retention ≤ the family's 0.806 |
| The rest of the code is "redundant" | Family vs complement in the ridge decoder's (cos, sin) output space | Wording only |

Two more checks have no overturning rule:
- **Cross-nuisance sham.** The source has the same orientation as the base, but another radius, width or polarity.
- **Relevance null size check.**

### Results
| Check | Result | Overturned? |
|---|---|---|
| Causal ranking | The top-10 by single-channel effect is 9 family units plus channel 380. Unit 379 ranks 12th; every family unit ranks within the top 12. Scored on the confirmatory pairs, this causal top-10 reaches effect 0.420 (IIA 0.168), against the family's 0.410 (0.177). | — |
| Best 10 non-family channels by effect | Effect 0.201, IIA 0.001. The family's margin over them is CI [0.197, 0.221]. | no |
| H2 line → arc | Family IIA 0.807. Null mean 0.030, q95 0.114 (p = 0.001). Effect 0.418 vs null q95 0.154. | no |
| H2 arc → line | IIA 0.003, but the effect is 0.367 vs null q95 0.130 (p = 0.001). | — |
| Mean ablation | Base orientation retained: family 0.806; rank-matched null 0.982 (q05 0.963). | no |
| Ridge readout | Family vs complement: (cos, sin) effect 0.573 vs 0.427; decoded-angle IIA 0.105 vs 0.019. | wording corrected |
| Cross-nuisance sham | Base orientation retained by the family: 0.992 (radius), 0.999 (width), 0.998 (polarity). Complement: 0.927–0.975. All channels: 0.905–0.969. | — |
| Relevance null size | Family mean variance / matched-null mean variance = 0.96. The family units sit at the 74th–94th variance percentiles. | — |

Top-K channels by single-channel effect give IIA 0.485 / 0.693 / 0.809 for K = 20 / 40 / 80. By activation change
the same K give 0.359 / 0.442 / 0.792.

### Interpretation
- **Specificity is stronger than the reports had argued.** The earlier "data-driven" comparison was the
  observational Fisher top-10. It looks only at the window centre, so it was a weak competitor (IIA 0.028).
  - A causal ranking nearly reproduces the published family.
  - The best non-family set moves the curve code only half as far as the family.
  - Channel 380, which is not one of the 10 published units, edges out unit 379. It is worth inspecting as one of
    the additional curve features that [Gorton 2024](https://arxiv.org/abs/2406.03662) reports. We have not checked
    this.
- **"Specific" also holds on the concept side.** Copying the family from a same-orientation stimulus with a
  different radius, width or polarity almost never changes the decoded orientation. The family does not inject
  nuisance information that the readout mistakes for orientation.
  - The "all channels" row is a different check. It measures how well the per-cell templates generalise across
    cells (0.905–0.969).
- **The code is shared, not redundant.**
  - Under the 14-unit template readout, the other 470 channels transfer the orientation in 40% of pairs.
  - Under the whole-layer ridge readout they do so in only 1.9%, and the family carries 57% of the decoded-orientation
    shift.
  - "Redundant" (the rest can substitute for the family) is therefore readout-dependent. The README, the reports and
    the paper now say "shared".
- **Additivity is largely architectural.** The family and complement effects sum to 1.000 for both readouts:
  0.410 + 0.590, and 0.573 + 0.427. Max-pooling acts per channel, 6 of the 14 curve readout units are linear in the
  pooled channels, and the ridge decoder is linear. The abstract and README no longer present additivity as a
  finding about the code.
- **H2's curvature claim survives its own null.** The binary "arc" decision is a lower bar than H1's 72-way decode.
  The continuous effect shows that the family's curvature evidence counts in both directions.

### Wording fixes with no new analysis
- "About 80 channels are needed" now reads as "about 80 under greedy orderings", which is an upper bound on the
  smallest sufficient set.
- The H5 curve-absent contrast had no null, so "the family matters more when driven" was removed.
- E1's 3a row is labelled in-sample, because the orientation convention was fitted there.
- `families.py` no longer calls Fisher selection an upper bound.

### Learnings
- **Readout dependence.** A claim about "the rest of the network" must name its readout. The same complement swap
  reads as "redundant" through 14 hand-picked units and as "mostly dependent on the family" through a whole-layer
  decoder.
- **Strength of the comparison set.** A data-driven baseline is only as strong as its selection criterion.
  Selecting by the quantity under test (causal effect) is the competitor a specificity claim has to beat.
- **How the checks ran.** Each check was first run as a read-only diagnostic on the selection split. The
  overturning outcomes were then fixed in the audit plan before the confirmatory run. Selection and confirmatory
  results agree closely, for example the complement's ridge-space effect: 0.429 vs 0.427.
- **Reproducibility.** The audit phase uses its own fresh RNG stream. The pre-registered results and the first
  post-hoc round stay byte-identical.
