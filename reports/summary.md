# Summary of all experiments (plain-language)

**What we asked.** InceptionV1, an image classifier, has 10 neurons in layer `mixed3b` that researchers identified
as "curve detectors". Each fires for curves bent in one direction. We asked whether the rest of the network
*uses* these neurons to know which way a curve bends, or whether they merely light up when curves are present.

**How.** We showed the network two images that are identical except for the curve's direction. Then we transplanted
only those 10 neurons' activity from the second image into the first, and checked what the next layer's curve
neurons "see". As a control we transplanted thousands of other 10-neuron groups chosen to change just as much
between the images. All pass/fail rules were written down and published before running the experiment.

**What we found.**
1. **The observation replicates.** Our setup reproduces the original authors' tuning curves closely: mean
   correlation 0.97 for the 10 curve neurons and 0.85 for the next layer's 14 curve neurons.
2. **The 10 neurons are special.** Transplanting them moves the next layer's curve code about 41% of the way to the
   other image. 95% of the control groups of 10 neurons stay below 10%. Ranking all 480 neurons by how much each one
   moves the next layer on its own (a check added later) picks 9 of the 10 published neurons as the top 10 when we read
   the next layer's 14 curve neurons. It picks 8 of the 10 when we read the whole next layer.
3. **They are not enough on their own.** The next layer actually switches to the transplanted direction in only
   about 18% of cases, so our pre-registered "strong" hypothesis (at least 50%) is **falsified**. The rest of the
   direction information is spread over many more neurons: picking the most useful ones first, it takes about 80 for reliable transfer. How
   well the other 470 neurons can replace the 10 depends on how we read the next layer. Read from its 14 curve
   neurons, they transfer the direction in 40% of cases. Read from the whole layer, they transfer it in only 2%,
   and the 10 curve neurons carry 57% of the shift.
4. **They carry "curviness".** Transplanting them from a curve into a straight line makes the next layer's curve
   neurons call it a curve 81% of the time. Control groups do so 3% of the time on average. The reverse transplant
   almost never makes a curve read as a line. That lopsidedness comes from where the readout draws its line/curve
   boundary: the 10 neurons move the next layer about equally far in both directions.
5. **This holds on real photos, more weakly.** On rotated photos the effect is 0.29, while 95% of control groups stay at
   or below 0.075.
   Replacing the 10 neurons' activity with its average changes the network's final answer on 17% of photos, versus
   9% for control groups.

**Bottom line.** "Curve detector" is a good *observational* description, and the units are *causally specific*, but
they are one part of a shared orientation code rather than its single carrier. Details:
[all experiments](all_experiments.md).

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
| Sufficiency / necessity / redundancy | The units alone carry the signal / removing them breaks it / other units can do the job without them. | IIA of the family / mean-ablation retention / complement IIA (which depends on the readout). |

## Networks used

No network was trained or fine-tuned in this project. The table describes the single pretrained model studied.

| Network | Architecture | Training algorithm and objective | Training data | Accuracy | Domain | Role here |
|---|---|---|---|---|---|---|
| InceptionV1 / GoogLeNet ("inception5h", [Lucid model zoo](https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py)) | 22-layer CNN built from Inception blocks: parallel 1×1, 3×3, 5×5 and pool branches, concatenated. Uses LRN layers. 1008-way softmax output. | Supervised image classification with a softmax output, GoogLeNet architecture ([Szegedy et al. 2015](https://arxiv.org/abs/1409.4842)). Lucid's docstring says the weights "were trained at Google and released in an early TensorFlow tutorial"; the exact recipe is not documented. | ImageNet (per Lucid's metadata) | Not measured here; we compare predictions only to the model's own clean predictions | Natural-image classification | The model whose neurons are explained (Circuits thread) and intervened on |

There are no training curricula, loss curves or reward curves, because nothing is trained. The figures show
tuning replication, null distributions, effect sizes and dose-response instead.
