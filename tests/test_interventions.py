"""Tests for `concept_swap.interventions`: cached windows and batched interchange / ablation over channel sets."""

import numpy as np
import torch

from concept_swap import interventions as I


def test_swap_invariants(model, device, images):
    """Patching no channels = clean base; patching all 480 = clean source; partial swaps lie in between."""
    X = torch.cat([images, images.flip(-1)]).cpu()  # 4 images
    c = I.collect(model, X, "mixed4a", device, batch=4)
    base, src = np.array([0, 1, 2, 3]), np.array([2, 3, 0, 1])
    ident = lambda R: R
    sets = [np.array([], int), np.arange(480), np.arange(320, 416)]
    out = I.patched_readouts(model, "mixed4a", c.windows, base, src, sets, np.arange(508), ident, batch=3)
    scale = c.readout.abs().max().item()
    np.testing.assert_allclose(out[0], c.readout[base].numpy(), atol=1e-5 * scale)
    np.testing.assert_allclose(out[1], c.readout[src].numpy(), atol=1e-5 * scale)
    assert not np.allclose(out[2], out[0]) and not np.allclose(out[2], out[1])


def test_swap_readouts_bracket_base_and_source(model, device, images):
    """`swap_readouts` (used by the hero and example figures): swapping no channel keeps the base readout and
    decode; swapping all 480 channels gives the source's."""
    import pandas as pd

    from concept_swap.phases.common import Readout, Split, swap_readouts

    X = torch.cat([images, images.flip(-1)]).cpu()  # 4 images, one nuisance cell, labels 0/90/180/270
    sp = Split(pd.DataFrame(dict(theta=[0.0, 90.0, 180.0, 270.0], cell=0)), I.collect(model, X, "mixed4a", device, batch=4))
    rd = Readout.fit(sp, np.arange(14), sp.grid.theta.values)
    base, src = np.array([0, 1]), np.array([2, 3])
    none, full = (swap_readouts(model, sp, rd, base, src, ch) for ch in (np.array([], int), np.arange(480)))
    np.testing.assert_allclose(none["z_p"], none["z_b"], atol=1e-3)
    np.testing.assert_allclose(full["z_p"], full["z_s"], atol=1e-3)
    np.testing.assert_array_equal(none["dec_p"], none["dec_b"]), np.testing.assert_array_equal(full["dec_p"], full["dec_s"])


def test_mean_ablation_and_self_permutation(model, device, images):
    """Mean-ablating with a window equal to the base itself is a no-op; swapping a channel with itself is a no-op."""
    c = I.collect(model, images.cpu(), "mixed4a", device, batch=2)
    base = np.array([0])
    out = I.patched_readouts(model, "mixed4a", c.windows, base, base, [np.arange(10)], np.arange(508), lambda R: R,
                             mode="mean", mean=c.windows[0])
    np.testing.assert_allclose(out[0], c.readout[:1].numpy(), atol=1e-5 * c.readout.abs().max().item())
