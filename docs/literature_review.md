# Literature review

This note covers what existed before this repo, what we reuse from it, and where this project sits.
Links were checked on 2026-10-23.

## 1. The explanation under test: Circuits curve detectors in InceptionV1

| Source | What it says | What we take |
|---|---|---|
| Olah et al. 2020, [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) | Neurons in vision models implement meaningful features, and those features connect through weights into circuits. | The framing: neuron-level explanations are claims that can be tested. |
| Olah et al. 2020, [An Overview of Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/) | A catalogue of neuron families in `conv2d0`–`mixed3b`, including curves, high-low frequency, color center-surround and others. | The families and the layer naming. |
| Cammarata et al. 2020, [Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/) | The 10 `mixed3b` units respond to curves of a preferred orientation. Each has a "gaussian-like bump", and together they span all orientations. The paper studies them with feature visualization, dataset examples, synthetic curves, radial and joint tuning curves, and weights. It argues causality from feature visualization and attribution. It reports **no ablation or activation-intervention experiments**, although its conclusion mentions circuit editing as a technique of the broader project. | The explanation under test. The synthetic-curve method. The authors' **published tuning data**: [`pages/data/curves-mixed{3a,3b,4a}.json`](https://github.com/distillpub/post--circuits-curve-detectors/tree/master/pages/data), CC-BY 4.0, copied into `data/circuits/`. |
| Cammarata et al. 2021, [Curve Circuits](https://distill.pub/2020/circuits/curve-circuits/) | 3b curves are built from 3a curves and earlier features. A hand-written "artificial artificial neural network" approximates them. Some claims are validated with **circuit (weight) editing**, e.g. removing negative weights makes "echoes" appear. | Hypothesis H4 (3a → 3b). |
| Olah et al. 2020, [Naturally Occurring Equivariance](https://distill.pub/2020/circuits/equivariance/) | Curve detectors are rotated copies. Rotating the input permutes which units fire. | Motivation for the natural rotation counterfactual (H5) and the "family rotation" swap. |
| Voss et al. 2021, [Branch Specialization](https://distill.pub/2020/circuits/branch-specialization/) | All 30 curve-related features of `mixed3b` are in its 5×5 branch. | The channel-order check (all 10 indices fall in 320–415) and the branch-matched null. |
| Gorton 2024, [The Missing Curve Detectors of InceptionV1](https://arxiv.org/abs/2406.03662) | SAE features on `mixed3b` reveal extra curve detectors that fill the orientation gaps of the 10 neurons. The paper notes that Lucent's LRN differs from the original and links a fix, [lucent#51](https://github.com/greentfrapp/lucent/pull/51), which is still unmerged; see §4 for the TF-exact values. | A prior hint that curve information extends beyond the 10 units. This is relevant to interpreting partial sufficiency. |
| Cammarata 2026, [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) (Goodfire essay) | Argues that superposition was the wrong explanation for InceptionV1. Apparent polysemanticity is attributed to recursive generalization plus several ordinary, mostly fixable issues: tools, explanations, architecture and data. Revisits layer 3b: SAEs recover clean curves on the original layer but not on a "squeezed" 3b cut from 480 to 10 neurons. Uses steering and negative/positive-pole analyses. | Context: the same research programme now calls for *causal* tests of neuron explanations. |

## 2. Counterfactual / interventional evaluation of explanations

| Source | Method | Use here |
|---|---|---|
| Huang et al. 2023, [Rigorously Assessing Natural Language Explanations of Neurons](https://arxiv.org/abs/2309.10312) | Two modes. *Observational*: precision and recall of an explanation. *Intervention*: an interchange intervention, scored by IIA. The explanation holds causally if τ(M_{a←z}(b)) = τ(M(s)). Baselines are random neurons and token-activation correlation. | Our primary metric (IIA) and our baselines. |
| Bäumel et al. 2025, [Modular Arithmetic: Language Models Solve Math Digit by Digit](https://arxiv.org/abs/2508.02513) ([code](https://github.com/tbaeumel/transformer-digit-arithmetic)) | Fisher-score selection of digit-position neurons. Activations are swapped from source to base prompts (via pyvene). Reports flip rate, target-variant probability shift, and off-target controls. | The Fisher-top-k reference set and the source→base swap design. |
| Geiger et al. 2021, [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997); Geiger et al. 2023, [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://arxiv.org/abs/2301.04709) | Interchange interventions test whether a high-level causal variable is realised by a set of units. | The formal meaning of "the family mediates orientation". |
| Zhang & Nanda 2024 (ICLR), [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042); Heimersheim & Nanda 2024, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) | Zhang & Nanda: metric choice changes conclusions; prefer logit difference over probability. Heimersheim & Nanda: have at least one continuous, roughly linear metric. | Our normalized (projection) effect, reported next to the binary IIA. |
| Gurnee, Sofroniew, …, Lindsey 2026, [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) (Anthropic, Transformer Circuits Thread; secondary summary: [Zvi Mowshowitz](https://thezvi.substack.com/p/no-space-like-j-space)) | The Jacobian lens ("J-lens") and **concept-swap** tests, e.g. swapping an internal "spider" representation for "ant" changes the answer from 8 legs to 6. | Inspiration for the activation-space "family rotation" swap. |
| Ameisen et al. 2025, [Circuit Tracing / attribution graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); Bricken et al. 2023, [Towards Monosemanticity (SAEs)](https://transformer-circuits.pub/2023/monosemantic-features) | Feature-level circuits validated causally by interventions such as feature ablation, steering (pinned features) and suppression. They do not use swaps. | Context for interventional validation of explanations. |
| Chan et al. 2022, [Causal Scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) | A hypothesis is tested by resampling activations the hypothesis says are irrelevant. | The complement and null-set logic. |

## 3. Neuron-level analysis of vision models: evaluation and critiques

| Source | Point | Use here |
|---|---|---|
| Bau et al. 2017, [Network Dissection](https://arxiv.org/abs/1704.05796); Bau et al. 2020, [Understanding the role of individual units](https://doi.org/10.1073/pnas.1907375117) (PNAS) | Unit labels, plus causal unit ablation and insertion in classifiers and GANs. | The precedent for unit interventions. Ours swaps, not ablates. |
| Morcos et al. 2018, [On the importance of single directions for generalization](https://arxiv.org/abs/1803.06959); Leavitt & Morcos 2020, [Selectivity considered harmful](https://arxiv.org/abs/2003.01262) | Class selectivity does not predict the effect of ablation. | Why selectivity (E1) is not taken as causal evidence on its own. |
| Geirhos et al. 2023, [Don't trust your eyes: on the (un)reliability of feature visualizations](https://arxiv.org/abs/2306.04719) | Feature visualizations can be manipulated and may not reflect natural-input behaviour. | Motivation for interventional tests. |
| Oikarinen, Yan & Weng 2025, [Evaluating Neuron Explanations: A Unified Framework with Sanity Checks](https://arxiv.org/abs/2506.05774) (ICML) | Most neuron-explanation metrics fail simple sanity checks. Only Pearson correlation, cosine, AUPRC, IoU and F1 pass; AUC (AUROC) fails the extra-labels test. | The E1 gate uses Pearson r. AUROC is reported only as a descriptive selectivity summary. |
| Kopf et al. 2024, [CoSy: Evaluating Textual Explanations of Neurons](https://arxiv.org/abs/2405.20331) (NeurIPS) | Generates synthetic images from the explanation text and compares activations with controls. | Our synthetic arcs play the same role, rendered parametrically. |
| Aguilar Canto, Calvo & Menchaca-Mendez 2026, [A Critical Analysis of Color Neurons in Vision Models](https://icml.cc/virtual/2026/79523) (ICML 2026 Workshop on Mechanistic Interpretability; [OpenReview](https://openreview.net/forum?id=GWpe8AdMk3)) | Unit interpretations should meet completeness, falsifiability and relevance, tested with ablation, synthetic stimuli and input interventions. | The relevance experiment (E3) and our falsifiable decision rules. |
| Luo et al. 2016, [Understanding the Effective Receptive Field](https://arxiv.org/abs/1701.04128) | Effective RFs are gradient-based and Gaussian-like. | How we locate the readout unit's RF centre. |

## 4. Tools, model and data

- **Model:** InceptionV1 / GoogLeNet ([Szegedy et al. 2015](https://arxiv.org/abs/1409.4842)), in Lucid's `inception5h` version, loaded with
  [Lucent](https://github.com/greentfrapp/lucent).
  - Weights are ProGamerGov's conversion:
    [pytorch-old-tensorflow-models](https://github.com/ProGamerGov/pytorch-old-tensorflow-models).
  - Lucent's LRN constants (size 9, k = 1) differ from the TF graph.
    - [lucent#51](https://github.com/greentfrapp/lucent/pull/51) (open, unmerged) proposes size 10 and k = 2.
    - A comment in its discussion derives the TF-exact window 11 = 2·depth_radius + 1 from
      [TF's LRN formula](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization). That
      comment also reports a maximum error of about 6e-6 against the original graph.
    - We use the size-11 values.
- **Natural images:** [Imagenette](https://github.com/fastai/imagenette) (fast.ai), a 10-class ImageNet subset. We use
  the 320 px validation split.
- **Statistics:**
  - Owen 2007, [The pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122): crossed random effects, here base
    and source images.
  - Phipson & Smyth 2010, [Permutation p-values should never be zero](https://doi.org/10.2202/1544-6115.1585).
  - Holm-Bonferroni via [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
  - Fisher score as defined in Gu, Li & Han 2011, [Generalized Fisher Score](https://arxiv.org/abs/1202.3725).

## 5. Gap this project fills
We found no prior work that runs **interchange interventions (concept swaps)** on the Circuits curve family, or on
any InceptionV1 neuron family, with matched null neuron sets. The curve-detector explanation was validated
observationally, by weight inspection and by circuit (weight) editing, but not by counterfactual activation swaps. This is a search result, not
proof of absence, so the paper states the novelty claim cautiously.
