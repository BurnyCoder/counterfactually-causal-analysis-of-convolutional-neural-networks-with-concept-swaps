# Circuits curve-family tuning data (third-party)

These three JSON files are copied verbatim from the Distill article repository of
Cammarata et al. 2020, "Curve Detectors" (https://distill.pub/2020/circuits/curve-detectors/),
file path `pages/data/curves-mixed{3a,3b,4a}.json` in
https://github.com/distillpub/post--circuits-curve-detectors at commit `fa89dbf594a9dd09e9d4dfd65293ba8c199cb63f`
(the `master` head, 2021-02-19; retrieved 2026-10-23; the files are byte-identical to it).
The article footer licenses Distill's "diagrams and text" under CC-BY 4.0 "unless noted otherwise"
(https://creativecommons.org/licenses/by/4.0/). The source repository has no separate LICENSE file, so we treat these
source data files as covered by that licence.

Format: `{"layer": str, "neurons": [int, ...], "activations": {neuron: [720 floats]}}`.
The 720 values are the authors' radial tuning curve sampled every 0.5 degrees of curve orientation
(index / 2 = degrees; see `pages/radial/radial.js`, `getOrientation`, in the same repository).

We use them as (1) the pre-specified neuron families under test (no data-driven selection by us) and
(2) a ground-truth target for replicating the tuning curves with our own stimulus renderer.
