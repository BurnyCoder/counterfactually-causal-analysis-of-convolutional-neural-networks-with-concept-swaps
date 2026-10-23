"""Tests for `concept_swap.model`: channel bookkeeping, LRN shim, block re-implementation, hooks.

Global context: every intervention in the experiments goes through these functions, so they are checked
against the full (unmodified) Lucent forward pass and against the TensorFlow LRN formula.
"""

import copy

import numpy as np
import torch

from concept_swap import model as M


def test_branch_slices_match_lucent_concat_order():
    """mixed3b = 1x1:128 | 3x3:192 | 5x5:96 | pool:64 and mixed4a = 192|204|48|64 (Lucent InceptionV1.__init__)."""
    s3b, s4a = M.branch_slices("mixed3b"), M.branch_slices("mixed4a")
    assert s3b["5x5"] == slice(320, 416) and s3b["pool_reduce"].stop == 480
    assert s4a["1x1"] == slice(0, 192) and s4a["5x5"] == slice(396, 444) and s4a["pool_reduce"].stop == 508
    assert M.branch_of("mixed3b", 379) == "5x5" and M.branch_of("mixed4a", 26) == "1x1"


def test_tf_lrn_matches_tensorflow_formula():
    """TF: out = x / (bias + alpha * sum_{|j-c|<=r} x_j^2)^beta with r=5, bias=2, alpha=1e-4, beta=0.5
    (https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)."""
    x = np.random.default_rng(0).normal(0, 30, size=(1, 20, 3, 3)).astype(np.float32)  # channels-first
    sq = np.pad(x ** 2, ((0, 0), (5, 5), (0, 0), (0, 0)))  # zero-pad the channel axis by depth_radius
    sqr_sum = np.stack([sq[:, c:c + 11].sum(1) for c in range(20)], 1)  # window of 2*5+1 channels
    expected = x / (2.0 + 9.999999747378752e-05 * sqr_sum) ** 0.5
    got = M.lrn(torch.from_numpy(x), M.TF_LRN).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_lrn_mode_changes_network_output(device, images):
    """The shim is really active: TF-exact and Lucent-default LRN give different logits."""
    tf_model, lu_model = M.load_inceptionv1(device, "tf"), M.load_inceptionv1(device, "lucent")
    with torch.no_grad():
        a, b = tf_model(M.preprocess(images)), lu_model(M.preprocess(images))
    assert not torch.allclose(a, b, atol=1e-3)


def test_run_block_matches_full_forward(model, images):
    """Our generic Inception block equals Lucent's monolithic forward for mixed3b and mixed4a (pre and post ReLU)."""
    with M.capture(model, ["mixed3a", "mixed3b", "mixed4a"]) as acts, M.capture_pre_relu(model, ["mixed3b", "mixed4a"]) as pre:
        model(M.preprocess(images))
    for layer, prev in (("mixed3b", "mixed3a"), ("mixed4a", "mixed3b")):
        torch.testing.assert_close(M.run_block(model, layer, acts[prev]), acts[layer], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(M.run_block(model, layer, acts[prev], pre_relu=True), pre[layer], atol=1e-4, rtol=1e-4)


def test_cropped_readout_equals_full_map(model, images):
    """Running a block on the cropped input window reproduces the full-map value at the readout position.

    Checked in float64 (exact up to rounding, ~1e-14) because in float32 cuDNN may pick a different convolution
    algorithm for the 28x28 map than for the 11x11 crop (observed max relative gap ~1.3e-5 of the value range).
    """
    m64 = copy.deepcopy(model).double()
    with M.capture(m64, ["mixed3a", "mixed3b"]) as acts:
        m64(M.preprocess(images.double()))
    for layer, prev in (("mixed3b", "mixed3a"), ("mixed4a", "mixed3b")):
        pos = M.READOUT_POS[layer]
        full = M.run_block(m64, layer, acts[prev], pre_relu=True)[:, :, pos, pos]
        got = M.readout_from_window(m64, layer, M.crop_window(acts[prev], layer))
        torch.testing.assert_close(got, full, atol=1e-9, rtol=1e-9)


def test_patch_hook_identity_and_full_swap(model, images):
    """Patching nothing changes nothing; patching all 480 mixed3b channels from a source gives the source's 4a."""
    x = M.preprocess(images)
    with torch.no_grad(), M.capture(model, ["mixed3b", "mixed4a"]) as clean:
        model(x)
    src = clean["mixed3b"].flip(0)  # image 1 is the source for image 0 and vice versa
    with torch.no_grad(), M.capture(model, ["mixed4a"]) as same, M.patch(model, "mixed3b", lambda out: out.clone()):
        model(x)
    torch.testing.assert_close(same["mixed4a"], clean["mixed4a"])
    with torch.no_grad(), M.capture(model, ["mixed4a"]) as swapped, M.patch(model, "mixed3b", lambda out: src):
        model(x)
    torch.testing.assert_close(swapped["mixed4a"], clean["mixed4a"].flip(0))


def test_readout_rf_center_is_near_image_center(model, device):
    """The input-gradient centre of mass of the 4a readout unit lies at pixel ~111 (8*13+7 by stride arithmetic)."""
    cy, cx = M.rf_center(model, "mixed4a", device)
    assert abs(cy - 111) <= 3 and abs(cx - 111) <= 3
