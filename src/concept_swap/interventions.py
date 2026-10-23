"""Interchange interventions (concept swaps) and ablations on cached receptive-field windows.

Local: `collect` runs the network once per image and caches the previous layer's window that the readout unit
sees; `patched_readouts` overwrites chosen channels of that window (with a source image's values, a mean, or
another channel's values) and recomputes only the readout block, for thousands of channel sets.
Global context: this is the interchange intervention M_{a<-z}(b) of Geiger et al. (https://arxiv.org/abs/2301.04709)
and Huang et al. (https://arxiv.org/abs/2309.10312), applied to whole channel maps of a CNN layer.
"""

from dataclasses import dataclass

import numpy as np
import torch

from concept_swap import model as M

PREV = {"mixed3b": "mixed3a", "mixed4a": "mixed3b"}  # layer whose window is patched -> readout layer


@dataclass
class Collected:
    """Cached windows of the patched layer (N, C_prev, w, w) and clean pre-ReLU readouts (N, C_out), on CPU."""

    windows: torch.Tensor
    readout: torch.Tensor


@torch.no_grad()
def collect(model, X: torch.Tensor, layer: str, device: str, batch: int = 256) -> Collected:
    """Forward images X (N, 3, 224, 224) in [0, 1]; keep the window of PREV[layer] that feeds `layer`'s readout unit.

    The clean readout is recomputed from the window with the same code path as the patched readouts, so a patch of
    zero channels reproduces it exactly (up to cuDNN batch-size effects).
    """
    wins, outs = [], []
    for i in range(0, len(X), batch):
        with M.capture(model, [PREV[layer]]) as a:
            model(M.preprocess(X[i:i + batch].to(device)))
        w = M.crop_window(a[PREV[layer]], layer)
        wins.append(w.cpu())
        outs.append(M.readout_from_window(model, layer, w).cpu())
    return Collected(torch.cat(wins), torch.cat(outs))


@torch.no_grad()
def patched_readouts(model, layer: str, windows: torch.Tensor, base: np.ndarray, src: np.ndarray, sets: list,
                     out_idx: np.ndarray, reduce, batch: int = 2048, mode: str = "swap",
                     mean: torch.Tensor | None = None, src_sets: list | None = None) -> np.ndarray:
    """For every channel set, patch it into each base window and return `reduce(readouts)` (stacked over sets).

    mode "swap": x[:, set] = windows[src][:, src_set] (src_set defaults to set; a different src_set gives the
    channel-reassignment "family rotation" swap). mode "mean": x[:, set] = mean[set] (mean ablation).
    readouts: (n_pairs, len(out_idx)) pre-ReLU values of the readout unit; `reduce` maps them to per-pair outputs.
    """
    dev = next(model.parameters()).device
    W = windows.to(dev)  # whole split's windows on the GPU (a few hundred MB)
    mean = None if mean is None else mean.to(dev)
    out_idx = torch.as_tensor(out_idx, device=dev)
    results = []
    for k, ch in enumerate(sets):
        ch = torch.as_tensor(ch, dtype=torch.long, device=dev)
        sch = ch if src_sets is None else torch.as_tensor(src_sets[k], dtype=torch.long, device=dev)
        R = []
        for i in range(0, len(base), batch):
            x = W[torch.as_tensor(base[i:i + batch], device=dev)].clone()  # base windows
            if len(ch):
                x[:, ch] = W[torch.as_tensor(src[i:i + batch], device=dev)][:, sch] if mode == "swap" else mean[ch]
            R.append(M.readout_from_window(model, layer, x)[:, out_idx])
        results.append(reduce(torch.cat(R).cpu().numpy()))
    return np.stack(results)
