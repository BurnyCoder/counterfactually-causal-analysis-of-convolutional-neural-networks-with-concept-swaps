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
   - We also report the **most-changing competitor**: the 10 non-family channels with the largest Δ. (The second
     round below found a stronger one, ranked by single-channel effect.)
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
- **H5 exact-rotation IIA** is 0.108 [0.080, 0.141], against 0.072 for the original rotations. These runs use
  different angle sets (see the third round).

## Interpretation
- **The stricter null changes little.**
  - The rank-matched q95 rises above the decile-matched one for H1 (0.121 vs 0.098) and H1-ridge (0.135 vs 0.106).
    It is similar or lower for the other tests: H2 0.100 vs 0.141; H4 0.146 vs 0.169; composited 0.106 vs 0.104;
    H5 0.073 vs 0.075.
  - Even the 10 most-changing non-family channels move the 4a curve code only 12.5% of the way, against 41% for
    the family. They transfer the decoded orientation in only 1 of 4574 pairs.
  - "The family is special among channels that change as much" survives.
- **On the decoded orientation itself, the family's effect is larger than the z-space number suggested.**
  - The ridge readout's predicted orientation vector moves 57% of the way to the source's, yet lands within 10° of
    it in only 10.5% of pairs.
  - So the swap rotates the decoded orientation part of the way, consistent with partial mediation.
- **The blur confound did not create the H5 effect.** With blur-free rotations the effect is 0.336, far above its
  null.
  - *Corrected in the third round.* The earlier "larger than 0.287" comparison mixed angle sets: the exact run has
    no ±45°.
  - On matched angles, blur changes neither the effect nor IIA measurably.
- **The pre-registered conclusion stands:** a specific but partial mediator. H1's strict sufficiency claim remains
  falsified under every analysis.

## Other review findings, handled without new analyses
- **H5 clean decoding accuracy is 1.0 by construction,** because the templates are the photo's own clean versions.
  It is not evidence that the decoder works; success still requires the patched base to match the rotated version
  better than the unpatched base.
- **The composited test reuses the same 32 photo backgrounds in both splits.** It generalises across jitter, not
  backgrounds.
- **The pigeonhole bootstrap treats base and source as separate factors** although images play both roles, and it
  ignores jitter-cell clustering, so the CIs are somewhat narrow. This cannot change the verdicts:
  - H1's falsification uses the point estimate, IIA < 0.2. The estimate is 0.177, and the seeds give 0.171–0.172.
    Its CI upper bound (0.199) is, however, close to 0.2.
  - Every secondary margin's CI lower bound is at least 0.136, far from 0.
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

### Claims that lacked their own test, and the outcome that would overturn each
The overturning outcomes were set in the audit plan before the phase ran. They were first committed together with
the results (commit 69368e2), so git does not timestamp them.

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
  - Channel 380, which is not one of the 10 published units, edges out unit 379. It lies in the 5×5 branch, which
    holds all 30 curve-related `mixed3b` features
    ([Voss et al. 2021](https://distill.pub/2020/circuits/branch-specialization/)). It is worth inspecting as a
    candidate curve-related channel; [Gorton 2024](https://arxiv.org/abs/2406.03662) found extra curve features
    with sparse autoencoders, not as single channels. We have not checked this.
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
- **Additivity is largely architectural** (*corrected in the third round*: architectural only for the 1×1 and pool
  branches; empirical for the 8 5×5-branch readout units). The family and complement effects sum to 1.000 for both readouts:
  0.410 + 0.590, and 0.573 + 0.427. Max-pooling acts per channel, 6 of the 14 curve readout units are linear in the
  pooled channels, and the ridge decoder is linear. The abstract and README no longer present additivity as a
  finding about the code.
- **H2's curvature claim survives its own null.** The binary "arc" decision is a lower bar than H1's 72-way decode.
  The continuous effect shows that the family's curvature evidence counts in both directions.

### Wording fixes with no new analysis
- "About 80 channels are needed" now reads as "about 80 under fixed per-channel rankings", which is an upper bound
  on the smallest sufficient set.
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
  overturning outcomes were then set in the audit plan before the audit phase scored the confirmatory pairs. The
  selection-split diagnostics were not saved, so the repository has no record of them.
- **Reproducibility.** The audit phase uses its own fresh RNG stream. The pre-registered results and the first
  post-hoc round stay byte-identical.

## Third round: independent audit (not pre-registered)

A third audit re-derived the study from the code, the result JSONs, the logs and the pilot files, without relying on
the earlier rounds. It found **no error that invalidates a pre-registered finding**. It re-checked:
- the crop geometry, the hooks, the stimulus geometry and the eligibility filter;
- the bootstrap, the permutation p-values, the Holm correction and the secondary-verdict code;
- fidelity to the pre-registration;
- the extreme null values (best Δ-matched null IIA 0.0013).

It did find one confounded post-hoc comparison and several interpretive claims that go beyond their evidence.

### Findings
1. **H5 blur comparison is confounded with the angle set.**
   - The exact-rotation run uses −90/90/180°. The pre-registered H5 also includes ±45°, which have the lowest IIA (0.030).
   - On the same three angles, the original IIA is (0.077 + 0.081 + 0.140) / 3 = 0.0995, against 0.108 for the exact
     run.
   - PIL's `Image.rotate` already uses an exact transpose for 180° on every image. Only ±45°, and ±90° on non-square
     images, were bilinear.
   - So "blur-free rotations give a larger effect" was not shown. Per-angle effects were not stored.
2. **"Additivity is largely architectural" overstates the mechanism.**
   - Only the 6 readout units in the 1×1 branch, and 4a's pool branch, are additive by construction.
   - The other 8 readout units are 5×5-branch units behind a ReLU bottleneck. The 3×3 and 5×5 branches hold 252 of
     the 508 `mixed4a` channels.
   - A linear ridge decoder on top does not make the sum structural.
3. **The causal ranking (9 of 10 family units in the top 10) was run only for the 14-unit curve readout.**
   - Curve Circuits describes that readout as built from 3b curves, so the ranking partly restates the weights.
   - It was never run for the whole-layer ridge readout.
4. **The H2 asymmetry is a decoder property.**
   - Line → arc flips 81% of pairs and arc → line 0.3%, but the continuous effects are nearly symmetric (0.418 vs 0.367).
   - The asymmetry therefore reflects where the correlation decoder puts its line/arc boundary.
5. **The family-rotation circular mean is a mixture, not a uniform partial shift.**
   - 34–74% of pairs are unchanged, and 10–50% land within 30° of the predicted shift.
6. **E1 has no pre-registered pass threshold, and its radii were chosen on the pilot.**
   - The radii were chosen to maximise agreement with the same published curves, so the r values are optimistic.
   - The conclusion is robust: on the pilot, 3b r ≥ 0.929 at every radius from 20 to 80 px.
7. **Minor issues.**
   - "Out of sample" in the causal ranking means only another ±2 px jitter. This favours the competitor.
   - "No 10-channel set reached IIA ≥ 0.5" means no *tested* set.
   - The effect-based secondary criterion was specified after the pilot had shown H1 IIA ≈ 0.16. The pilot did not
     compute the effect.

### Overturning rules, committed before the checks run
The checks run in phase `audit2` (`results/posthoc_audit2.json`, its own RNG stream). This section was committed before
that phase first touched confirmatory data, so git timestamps the rules.

| Claim | Check | Outcome that overturns or changes it |
|---|---|---|
| (g) The family is special under the whole-layer readout too | Rank all 480 `mixed3b` channels by single-channel effect in the ridge decoder's (cos, sin) output on the selection split; score the top-10 non-family set on the confirmatory pairs | The top-10 non-family (cos, sin) effect ≥ the family's |
| (h) Removing interpolation blur increases the H5 effect | Family effect and IIA per angle, for the original and the exact rotations, on the same curve-present photos and the same angles (−90/90/180°); photo-cluster bootstrap CI of the difference | The CI of (exact − original) effect does not lie above 0. The claim is then replaced by the matched-angle numbers. Sanity check: the 180° effect is nearly equal in both modes |
| (i) Additivity is largely structural | Relative non-additivity Σ‖Δ_F + Δ_C − Δ_all‖² / Σ‖Δ_all‖², per `mixed4a` branch and for the 6 1×1 and 8 5×5 readout units | Below 0.05 for the 3×3/5×5 branches or the 8 5×5 readout units. The wording then becomes "exact for the 1×1 and pool branches by construction, near-additive elsewhere empirically" |

### Results
Run log: `logs/20261023_230809.log`. Tables: [`generated/results.md`](generated/results.md#third-audit-checks-not-pre-registered-post-hoc-exploratory).

| Check | Result | Outcome under the rule |
|---|---|---|
| (g) Ridge-readout causal ranking | 8 of the top 10 channels are family units; every family unit ranks within the top 24. The family's (cos, sin) effect is 0.573 (IIA 0.105). The best 10 non-family channels reach 0.211 (IIA 0.003), and the family's margin over them is CI [0.350, 0.374]. The causal top-10 (8 family units plus channels 359 and 357) reaches 0.596 | not overturned |
| (h) H5 blur on matched angles | Effect 0.335 with the original rotations vs 0.336 with exact transposes. The difference is 0.000, CI [−0.001, 0.001]. IIA is 0.108 in both. Per angle (original / exact effect): −90° 0.322 / 0.322, 90° 0.324 / 0.324, 180° 0.356 / 0.357 | claim withdrawn |
| (i) Additivity | Non-additivity is 2·10⁻¹³ (1×1 branch) and 3·10⁻¹³ (pool branch), i.e. exact to fp32. It is 0.095 (3×3 branch) and 0.070 (5×5 branch). For the 14 curve readout units: 0 for the 6 1×1 units, 0.018 for the 8 5×5 units, and 0.011 overall | wording changed |

### Interpretation
- **(g) Specificity does not hinge on the circular readout.**
  - Ranked by their effect on a decoder over the whole of `mixed4a`, 8 of the 10 published units are still among
    the 10 most effective `mixed3b` channels.
  - The best non-family set moves the decoded orientation only 21% of the way, against 57% for the family.
  - A set chosen by this ranking slightly beats the family (0.596 vs 0.573). It was selected on stimuli that differ
    from the confirmatory ones only in jitter, so this is close to an in-sample comparison.
- **(h) Interpolation blur is irrelevant for H5.**
  - With the same photos, angles and templates, exact transposes and bilinear rotations give the same effect and IIA.
  - The earlier comparison, 0.336 vs 0.287 (IIA 0.108 vs 0.072), compared different angle sets. The pre-registered
    H5 includes ±45°, where the family transfers least (IIA 0.030), and has six templates per photo instead of four.
  - Blur cannot be tested at ±45°, because no rotation by 45° is exact.
  - The higher IIA at 180° remains in the exact run, so it is an angle effect, not a blur effect.
- **(i) Additivity is part architecture, part empirical regime.**
  - It is exact only in the 1×1 and pool branches, where max-pooling and a linear 1×1 convolution act per channel.
  - For the 8 readout units in the 5×5 branch, swapping the family and the complement adds up to within 1.8%
    (squared-norm ratio). That is a property of these stimuli, not of the architecture.
  - The unit test `test_readout_additivity_is_structural_only_for_1x1_and_pool_branches` confirms the architectural
    part: with sparse random inputs, the 3×3 and 5×5 branches are not additive.
  - The projected effects still sum to 1.000 in every branch: the interaction is almost orthogonal to the
    base-to-source direction.
