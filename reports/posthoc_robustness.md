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

| Test | Family effect [95% CI] | Rank-null q95 | Pre-registered decile-null q95 | Top-Δ non-family set: IIA / effect |
|---|---|---|---|---|
| H1 (3b → 4a curve units) | 0.410 [0.400, 0.419] | 0.121 | 0.098 | 0.000 / 0.125 |
| H1-ridge, z-space (as pre-registered) | 0.248 | 0.135 | 0.106 | 0.000 / 0.155 |
| **H1-ridge, decoded (cos, sin) space** | **0.573** [0.564, 0.583] | 0.126 | 0.097 | — |
| H2 curvature swap | 0.393 | 0.100 | 0.141 | 0.068 / 0.172 |
| H4 3a → 3b | 0.324 | 0.146 | 0.169 | 0.000 / 0.116 |
| Composited arcs on photos | 0.415 | 0.106 | 0.104 | 0.001 / 0.123 |
| H5 natural rotation (as pre-registered) | 0.287 | 0.073 | 0.075 | 0.000 / 0.083 |
| **H5, exact rotations only (no blur)** | **0.336** [0.310, 0.358] | 0.077 | 0.079 | 0.000 / 0.066 |

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
    family, and never transfer the decoded orientation.
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
