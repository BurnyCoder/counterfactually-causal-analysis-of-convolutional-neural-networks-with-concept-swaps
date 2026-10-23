# Summary of all experiments (plain-language)

<!-- RESULTS-SUMMARY -->

## Key terms

| Term | Plain-language meaning | Precise meaning in this repo |
|---|---|---|
| Neuron / unit / channel | One feature detector in a CNN layer. It is copied across every image position, giving a "map". | One output channel of an Inception block, e.g. `mixed3b:379` is channel 379 of the 480-channel `mixed3b` output. |
| Curve detector | A unit that fires for a curve bent in one particular direction. | The 10 `mixed3b` units listed by [Cammarata et al. 2020](https://distill.pub/2020/circuits/curve-detectors/). |
| Orientation θ | The direction the curve bends towards. | The direction from the arc's midpoint to its centre of curvature, from 0 to 360°. |
| Receptive field (RF) | The patch of the image a unit can "see". | For `mixed4a` unit (6,6), centred on pixel (111,111). Only the 11×11 `mixed3b` window feeding it matters. |
| Explanation | A claim of what a unit does. | "These 10 units detect curves of one orientation each." |
| Observational test | Check that the unit fires on what the explanation says. | E1: tuning curves vs the authors' published curves, and AUROC. |
| Interchange intervention / concept swap / activation patching | Run image A, but copy some units' activations from image B, then see what the rest of the network does. | Copy the family's channels in the RF window from the *source* image into the *base* image, then recompute the next block. |
| Base / source | The image being edited / the image the activations come from. | Same stimulus except orientation (synthetic), or the same photo rotated (H5). |
| Readout τ | How we read "which orientation the network now represents" downstream. | A nearest-template decoder over the 14 published `mixed4a` curve units (primary), or a ridge decoder over all 508 `mixed4a` channels (secondary). |
| IIA (interchange intervention accuracy) | How often the swap makes the network act as if it saw the source. | The fraction of eligible pairs where decode(patched base) is within 10° of decode(clean source) ([Huang et al. 2023](https://arxiv.org/abs/2309.10312)). |
| Normalized effect | How far the readout moved toward the source, from 0 (no change) to 1 (fully the source). | Σ⟨z_p−z_b, z_s−z_b⟩ / Σ‖z_s−z_b‖² over pairs. |
| Null neuron set | A control set of non-explanation units of the same size. | Δ-matched: its activation changes as much as the family's. Branch-matched: same 5×5 branch. Random. |
| q95 of the null | The value that 95% of control sets do not reach. | Used as the bar the explanation must clear. |
| Pigeonhole bootstrap | A way to get error bars when pairs share images. | Resample base images and source images separately ([Owen 2007](https://doi.org/10.1214/07-AOAS122)). |
| Holm correction | Adjusts p-values when testing several hypotheses. | Applied across the 5 secondary hypotheses. |
| Pre-registration | Writing down hypotheses and pass/fail rules before seeing the data. | `reports/preregistration.md`, committed before the confirmatory runs. |
| Sufficiency / necessity / redundancy | The units alone carry the signal / removing them breaks it / other units also carry it. | IIA of the family / mean-ablation retention / complement IIA. |

## Networks used

No network was trained or fine-tuned in this project. The table describes the single pretrained model studied.

| Network | Architecture | Training algorithm and objective | Training data | Accuracy | Domain | Role here |
|---|---|---|---|---|---|---|
| InceptionV1 / GoogLeNet ("inception5h", [Lucid model zoo](https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py)) | 22-layer CNN built from Inception blocks: parallel 1×1, 3×3, 5×5 and pool branches, concatenated. Uses LRN layers. 1008-way softmax output. | Supervised image classification with a softmax output, GoogLeNet architecture ([Szegedy et al. 2015](https://arxiv.org/abs/1409.4842)). Lucid's docstring says the weights "were trained at Google and released in an early TensorFlow tutorial"; the exact recipe is not documented. | ImageNet (per Lucid's metadata) | Not measured here; we compare predictions only to the model's own clean predictions | Natural-image classification | The model whose neurons are explained (Circuits thread) and intervened on |

There are no training curricula, loss curves or reward curves, because nothing is trained. The figures show
tuning replication, null distributions, effect sizes and dose-response instead.
