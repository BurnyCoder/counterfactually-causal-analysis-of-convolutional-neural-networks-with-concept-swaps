# Circuits curve-family tuning data (third-party)

These three JSON files are copied verbatim from the Distill article repository of
Cammarata et al. 2020, "Curve Detectors" (https://distill.pub/2020/circuits/curve-detectors/),
file path `pages/data/curves-mixed{3a,3b,4a}.json` in
https://github.com/distillpub/post--circuits-curve-detectors (commit on `master`, retrieved 2026-10-23).
Distill content is licensed CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Format: `{"layer": str, "neurons": [int, ...], "activations": {neuron: [720 floats]}}`.
The 720 values are the authors' radial tuning curve sampled every 0.5 degrees of curve orientation
(index / 2 = degrees; see `pages/radial/radial.js`, `getOrientation`, in the same repository).

We use them as (1) the pre-specified neuron families under test (no data-driven selection by us) and
(2) a ground-truth target for replicating the tuning curves with our own stimulus renderer.
