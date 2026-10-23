# Pre-registration: counterfactual concept swaps on the InceptionV1 curve family

- **Registered:** 2026-10-23, before any confirmatory run. The git commit that adds this file is the time stamp.
  The code at registration is `master@de6f002`.
- **Authors:** Libor Burian (idea). Claude Code (implementation).
- **Status of data at registration:**
  - Confirmatory stimuli (jitter seed 2, stability seeds 3 and 4) and the confirmatory Imagenette half have never
    been passed through an intervention.
  - Exactly what was seen beforehand is listed under [Disclosures](#disclosures-what-was-seen-before-registration).

## 1. Question and explanation under test

**Explanation (observational, from prior work).** Cammarata et al. 2020,
[Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/), say that the InceptionV1 `mixed3b` units
`[379, 406, 385, 343, 342, 388, 324, 340, 330, 349]` are curve detectors. Each is tuned to a curve orientation, and
together they tile all 360°.
- The same work lists curve families in `mixed3a` (11 units) and `mixed4a` (14 units).
- Their published 720-point tuning curves are stored in `data/circuits/`.

**Causal question.** Do these units *mediate* curve orientation for downstream computation? This is the
"intervention mode" of Huang et al. 2023, [arXiv 2309.10312](https://arxiv.org/abs/2309.10312).
- Take two images that differ only in curve orientation.
- Copy the family's activations from the source image into the base image.
- Ask whether the network's downstream curve units then represent the source orientation.

Huang et al.'s criterion: τ(M_{a←z}(b)) = τ(M(s)), where z holds the family's values on source s.

## 2. Model, stimuli and splits (frozen)

**Model.** Lucid InceptionV1 via Lucent ([greentfrapp/lucent](https://github.com/greentfrapp/lucent) @ `2ccc434`).
- Weights: `inception5h.pth`.
- LRN: TensorFlow-exact (size 11, α = 11·9.9999997e-5, β = 0.5, k = 2), per
  [lucent#51](https://github.com/greentfrapp/lucent/pull/51).
- Input: `x*255-117`.
- fp32 convolutions (TF32 off).

**Readout units.**
- `mixed4a` unit (6,6) and `mixed3b` unit (13,13).
- Their receptive-field centre is pixel (111,111). This is unit-tested via the input-gradient centre of mass.

**Synthetic stimuli** (`src/concept_swap/stimuli.py`):
- Anti-aliased arcs, rendered with PIL at 4× supersampling.
- Arc span 120°. The arc midpoint sits at the RF centre.
- θ is the direction from the midpoint to the centre of curvature.
- Grid: 72 θ (5° steps) × radii {30, 45, 60, 80} px × widths {3, 5} px × 2 polarities = 1152 images per split.
- One random jitter (±2 px) per nuisance cell, shared by all θ in that cell.

**Splits** (by jitter seed):

| Split | Seed | Used for |
|---|---|---|
| Selection | 1 | z-statistics, decoder templates, activation-change Δ for null matching, Fisher selection, preferred orientations, E1 |
| Confirmatory | 2 | all tests |
| Stability | 3, 4 | robustness only |

**Minimal pairs.**
- Each base gets 4 sources from the same nuisance cell, with |Δθ| ≥ 30°.
- Sampling is seeded, giving 4608 pairs.

**Natural images.** The Imagenette-320 validation set (3925 images), split into seeded halves (selection / confirmatory).

## 3. Interventions, readouts, metrics (frozen)

**Interchange.**
- The patched channels' whole receptive-field window (`mixed3b` 11×11 crop that feeds 4a unit (6,6)) is copied
  from source to base.
- Only the readout block is recomputed. This equals the full forward pass, as shown by the fp64 unit test.

**Primary readout τ.**
- The 14 published `mixed4a` curve units, read pre-ReLU at (6,6) and z-scored with selection statistics.
- They are decoded by a nearest-template decoder: Pearson correlation against the 72 clean selection-split
  templates of the pair's nuisance cell.

**Primary metric: IIA.**
- IIA is the fraction of eligible pairs where |decode(patched base) − decode(clean source)| ≤ 10°.
- Eligible pairs are those whose clean base and clean source both decode within 10° of their true θ. The same filter
  applies to every neuron set.

**Continuous metric: normalized effect.**
- Computed as Σ⟨z_p − z_b, z_s − z_b⟩ / Σ‖z_s − z_b‖² over eligible pairs.
- Pairs whose ‖z_s − z_b‖² is in the lowest 5% are dropped.
- 0 means no change; 1 means the readout became the source's.

**Null neuron sets**, each the same size as the family:
- **Δ-matched (primary null, 2000 sets).** Non-family channels sampled to match the family's histogram over deciles
  of Δ, where Δ is the mean squared base→source change in the window on selection pairs. A patch can only act
  through channels that change, so this removes the family's trivial advantage.
- **Branch-matched (co-primary, 2000 sets).** Random 10 from the rest of the 5×5 branch (channels 320–415). This
  branch also holds circle and double-curve units.
- **Random-10 from the other 470 channels (500 sets).** Descriptive only.

**Reference sets** (descriptive):
- Fisher top-10 for θ on selection (Bäumel et al. 2025, [arXiv 2508.02513](https://arxiv.org/abs/2508.02513)).
- The complement (470 channels).
- All 480 channels (must give IIA = 1).

**Statistics.**
- Pigeonhole bootstrap (Owen 2007) over base and source images, with B = 2000.
- The null q95 is recomputed inside every replicate.
- Permutation p = (1 + #{null ≥ observed}) / (1 + M) (Phipson & Smyth 2010).

## 4. Hypotheses and decision rules

### H1 (primary): the 3b curve family mediates orientation for the 4a curve family
Let D = IIA_family − q95(Δ-matched null IIA). The rule is unchanged from the plan approved before the pilot.

| Verdict | Condition |
|---|---|
| **Supported** | IIA_family ≥ 0.5, D ≥ 0.10, the 95% CI lower bound of D > 0, and the same CI condition holds against the branch-matched null |
| **Falsified** | the CI upper bound of D < 0.05, **or** IIA_family < 0.2 |
| **Inconclusive** | anything else |

H1 is a strong *sufficiency* claim: swapping the 10 units alone should carry the orientation the next layer reads.

### Secondary hypotheses (Holm–Bonferroni across the five, α = 0.05)
**Criterion for each:**
- The normalized-effect margin over the Δ-matched null q95 has a 95% pigeonhole CI lower bound > 0, **and**
- the Holm-adjusted permutation p (effect vs the Δ-matched null) < 0.05.

IIA is reported alongside.

1. **H1-ridge (non-circular readout).** Same swap. τ = a ridge (cos θ, sin θ) decoder over all 508 `mixed4a`
   channels, fitted on selection. Null sets: 500 per null.
2. **H2 (curvature swap: curve detector vs oriented-line detector).**
   - Base = straight line; source = arc with the same midpoint tangent, width, polarity and jitter (and the reverse).
   - τ = nearest of the two clean templates (line, arc) within the same (cell, θ).
   - Null sets: 1000 per null. Lines vs arcs is also reported as a Fisher reference.
3. **H4 (Curve Circuits claim, 3a → 3b).**
   - Patch the 11 `mixed3a` curve units, using their 5×5 window feeding 3b unit (13,13).
   - Readout = the 10 `mixed3b` curve units, with the same decoder and metrics.
   - The branch null is the rest of the 3a 3×3 branch. Null sets: 2000 per null.
4. **Composited (H1 on natural backgrounds).**
   - Arcs alpha-composited onto Imagenette selection-half crops, one background per nuisance cell.
   - 2 replicates × 16 cells = 32 cells. 2 sources per base. Null sets: 1000 per null.
5. **H5 (natural rotation counterfactual).**
   - Base = a 224 crop centred on the RF centre. Source = the same photo rotated by φ ∈ {−90, −45, 45, 90, 180}° about
     that centre.
   - Only curve-present crops are used: family max activation at the RF centre above the selection-half 75th
     percentile.
   - τ = which of the photo's 6 clean versions the patched readout correlates with best (z-statistics from the
     selection half). IIA = decodes as the rotated version.
   - Null sets: 1000 per null.

### Exploratory (labelled as such, no inferential claims)
- IIA by |Δθ| bin and by rotation angle.
- Dose-response over random K-subsets of the family.
- Top-K channels by Δ, with and without the family.
- Mean ablation (fraction of the base orientation retained).
- Same-θ sham with a different jitter.
- Family-rotation activation-space swap (each unit receives the map of the unit k steps around the orientation
  circle).
- "Closer to source than base" rate.
- A curve-absent contrast for H5 (bottom quartile).
- Natural-image relevance: whole-map mean ablation vs 100 Δ-matched (variance-matched) ablations, with top-1 flip
  rate and KL on the confirmatory half.
- Stability seeds 3 and 4, and the Lucent-default-LRN sensitivity run (500 null sets each).

### Interpretation map, fixed in advance
- **H1 falsified but the effect margin positive.** The family is a *specific but partial* mediator: no other
  equally-changing set does what it does, yet it is not sufficient alone.
- **High complement IIA.** The orientation code is *redundant*: other 3b channels carry it too. This does not refute
  the family's causal role.
- **H2 effect not above null.** Supports the "oriented-line detector" alternative over "curve detector" for the
  information the 4a curve units use.

## 5. Sample size and precision
The pilot used jitter seeds 0 and 100 and is excluded from all analyses. With 4608 pairs, the family-IIA CI
half-width was 0.022. Pre-registered target: ≤ 0.03, so 4608 pairs suffice. The smallest effect of interest is
D = 0.10.

2000 null sets make the smallest attainable p ≈ 5·10⁻⁴. With Holm over 5 tests, the smallest adjusted p is about
2.5·10⁻³, so every secondary test can reach significance.

## 6. Deviations from the approved plan (all decided before confirmatory data)
- **Plan H3 (locality).** "RF-window patch = full patch" and "outside-RF patch = 0" are architectural identities for
  a single readout unit, so they became unit tests (`tests/test_model.py`) rather than hypotheses.
- **Secondary decision criterion.** Not fully specified in the plan; defined in §4.
- **Stimulus radii {30, 45, 60, 80}.** Chosen on the pilot split as the radii where both 3b and 4a match the
  published tuning best, as the plan required.
- **H5 decoder.** Natural photos have no orientation label, so τ is "nearest clean version of the same photo"
  (templates = that photo's clean versions). This keeps Huang's τ(patched) = τ(source) form.

## 7. Disclosures: what was seen before registration
- **Pilot** (jitter seeds 0/100, discarded): clean decoding accuracy 1.00.
  - Family IIA 0.162, 95% CI [0.141, 0.185]. Δ-matched null mean 0.000, q95 0.001. Branch null q95 0.000.
  - Complement IIA 0.421. All-480 IIA 1.000. Fisher top-10 IIA 0.027 (Jaccard with the family 0.25).
  - Family units sat in the 96th–99th percentile of Δ.
  - IIA was highest for |Δθ| in 90–150°.
- **Thresholds were not changed after the pilot.** Under the frozen H1 rule this pilot pattern would read
  "falsified" (IIA < 0.2). We keep the rule unchanged and say so here.
- **Smoke tests** (`main.py --quick`) ran on seeds 101–104 with a 24-θ grid and 20 null sets; numbers are only in the
  gitignored `results/quick/`. Before the seed offset was added, one E1 smoke run used a subset of the selection
  split (seed 1), where 3b mean r = 0.964. The selection split is not used for any interventional test.
