"""Feature visualizations: the image each neuron responds to most, found by optimising the input.

Local: a thin wrapper over Lucent's `render_vis` that renders several neurons in one batched optimisation
(objective = sum of one `objectives.neuron` term per batch element; https://github.com/greentfrapp/lucent).
Global context: the Circuits papers show every neuron by its feature visualization, "each neuron's ideal curve"
(Cammarata et al. 2020, https://distill.pub/2020/circuits/curve-detectors/; method: Olah et al. 2017,
https://distill.pub/2017/feature-visualization/). The example figures use these images as neuron icons.
"""

import numpy as np
import torch
from lucent.optvis import objectives, param, render  # https://github.com/greentfrapp/lucent/tree/dev/lucent/optvis

from concept_swap.model import READOUT_POS


def render_units(model, units: list[tuple[str, int]], steps: int = 512, seed: int = 0, size: int = 224) -> np.ndarray:
    """(len(units), size, size, 3) images in [0, 1]; image i maximises unit i = (layer, channel) at the readout
    position (the single neuron the experiments read, so the result is one curve, not a tiled texture).

    Local: Lucent defaults (FFT-parameterised, colour-decorrelated image; standard jitter/scale/rotate transforms;
    Adam, lr 0.05). Lucent's transforms draw from numpy's global RNG, so both RNGs are seeded for reproducibility.
    """
    np.random.seed(seed), torch.manual_seed(seed)  # Lucent: torch.randn init, np.random.choice transforms
    dev = next(model.parameters()).device
    obj = objectives.Objective.sum([objectives.neuron(l, n, x=READOUT_POS[l], y=READOUT_POS[l], batch=i)
                                    for i, (l, n) in enumerate(units)])
    img = render.render_vis(model, obj, lambda: param.image(size, batch=len(units), device=dev), thresholds=(steps,),
                            show_image=False, progress=False)
    return img[-1]  # the image after the last step, NHWC float in [0, 1]
