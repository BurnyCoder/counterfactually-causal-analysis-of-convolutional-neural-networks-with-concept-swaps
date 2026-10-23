# Methodology: why each design choice

The exact frozen protocol (thresholds, null sizes, seeds) is in `reports/preregistration.md`. This page explains
*why* the protocol looks the way it does. It does not restate the protocol.

## Why the curve family
- **Best characterised.** Curve detectors are "likely the best characterised neurons in InceptionV1"
  ([Gorton 2024](https://arxiv.org/abs/2406.03662)), studied in detail by
  [Cammarata et al. 2020](https://distill.pub/2020/circuits/curve-detectors/). The explanation is precise: one
  orientation per unit, tiling 360°.
- **A clean concept variable.** Orientation can be swapped with everything else held fixed, which minimal-pair
  interchange interventions require. Huang et al. needed templates such as "The year after Y is"; arcs give us
  exact counterfactuals for free.
- **Ground truth for our setup.** The authors published their tuning data, so we can check our model port,
  renderer and orientation convention before any intervention (E1). Without this, a negative intervention result
  could just mean a broken setup.

## Why a downstream curve-family readout (and not the class logits)
- A CNN has no output that names orientation; its 1008 class logits on a lone arc are off-distribution and not
  something the explanation makes claims about.
- Huang et al.'s τ maps a model state to an output behaviour that depends on the explained concept. The nearest such
  behaviour the network has is the next layer's curve units. The same authors published tuning data for 14 `mixed4a` curve units;
  the article's figure labels 12.
- **This choice is partly circular.** The 4a curve units read `mixed3b` directly and plausibly use its curves.
  [Curve Circuits](https://distill.pub/2020/circuits/curve-circuits/) shows 3b curves feeding 4a shape detectors
  (circles, spirals, evolutes) and notes that 4a builds its own curve detectors, "mostly in the 5x5 branch". We
  address that three ways:
  1. A **non-circular** ridge decoder over all 508 `mixed4a` channels (H1-ridge).
  2. **Δ-matched nulls**, so any set that changes as much as the family gets the same chance.
  3. Natural-image tests (H5), plus an end-to-end prediction check (relevance).

## Why activation-change-matched (Δ-matched) nulls
- Patching a channel whose value is the same in base and source does nothing. Random channels mostly barely change
  with orientation, so beating a random null only proves "channels that change, matter".
- Matching the family's decile histogram of Δ (mean squared base→source change, measured on the selection split)
  asks the sharper question: is the family special *among channels that change as much*?
- The branch-matched null (rest of the 5×5 branch) guards against "any 5×5-branch unit would do" (Voss et al.,
  [Branch Specialization](https://distill.pub/2020/circuits/branch-specialization/)).

## Why both IIA and a continuous effect
- **IIA (binary)** is Huang et al.'s metric and answers "does the network now *decode* the source orientation?".
  It is strict: the 4a readout must move far enough that the source orientation's template (within 10°) becomes the
  best-correlated one.
- Metric choice can change patching conclusions ([Zhang & Nanda 2024](https://arxiv.org/abs/2309.16042)).
  [Heimersheim & Nanda 2024](https://arxiv.org/abs/2404.15255) advise having at least one continuous, roughly linear
  metric. The **normalized projection effect** (0 to 1) measures
  partial movement toward the source. It is a ratio of sums, so pairs with tiny base/source differences cannot blow
  it up, and the bottom 5% of such pairs are dropped.

## Why the cropped-window recomputation
- The readout depends only on an 11×11 `mixed3b` window, which follows from stride and kernel arithmetic; see
  `model.crop_window`. Re-running only `maxpool4 → mixed4a` on that window is exact (fp64 test) and much cheaper
  than a full forward pass: about 0.15 s per neuron set for 4608 pairs, decoding included, on the RTX 5070 Laptop.
  The pilot log shows 200 sets in 31 s.
- This makes 2000-set nulls on 4608 pairs feasible on a laptop GPU, which in turn makes p-values below 10⁻³
  attainable.

## Why the pigeonhole bootstrap
- Each base image appears in 4 pairs and each source in about 4 pairs, so pairs are not independent. Treating them
  as i.i.d. would understate the uncertainty.
- Owen's pigeonhole bootstrap ([2007](https://doi.org/10.1214/07-AOAS122)) resamples the two crossed factors (base
  images, source images) independently.
- We recompute the null's q95 inside each replicate, so the margin's CI includes uncertainty in the null bar too.

## Why these splits and seeds
- **Selection split (seed 1):** fits everything learned from data: z-statistics, decoder templates, Δ for null
  matching, Fisher selection, and preferred orientations.
- **Confirmatory split (seed 2):** used only for the tests. It differs from selection by jitter (translation)
  within each nuisance cell, so clean decoding accuracy is also a small generalisation check.
- **Pilot seeds (0, 100):** used to choose radii and estimate precision, then discarded and disclosed in the
  pre-registration.
- **`--quick` smoke tests:** use seeds 101–104, so development runs never see confirmatory stimuli.

## Why rotation for natural images
- Natural photos have no orientation label. Rotating a photo about the readout's RF centre changes the orientation
  of every curve the unit sees and nothing else about the content.
- Olah et al.'s [equivariance](https://distill.pub/2020/circuits/equivariance/) account predicts the curve family's
  code rotates accordingly.
- The test asks whether the family's code from the rotated photo moves the 4a curve readout to the rotated version.
- The comparison is restricted to crops where the family actually fires at the RF centre (top quartile). The
  bottom quartile serves as an exploratory contrast.
