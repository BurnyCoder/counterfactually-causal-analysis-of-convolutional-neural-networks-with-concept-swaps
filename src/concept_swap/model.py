"""InceptionV1 loading, TensorFlow-exact LRN, activation capture/patch hooks and a re-implemented Inception block.

Local: wraps Lucent's PyTorch port of the Lucid InceptionV1 ("inception5h") so that (a) its LRN matches the
original TensorFlow graph, (b) any layer can be read (post- or pre-ReLU) or overwritten with forward hooks,
and (c) a single Inception block can be re-run on a cropped input window (fast patching).
Global context: the Circuits neuron indices (e.g. mixed3b:379) refer to this exact network
(https://distill.pub/2020/circuits/curve-detectors/); every intervention in `interventions.py` uses these helpers.
"""

import importlib  # to reach Lucent's module object (its package re-exports a class of the same name)
from contextlib import contextmanager  # https://docs.python.org/3/library/contextlib.html
from types import SimpleNamespace

import torch
import torch.nn.functional as F  # https://pytorch.org/docs/stable/nn.functional.html
from lucent.modelzoo import inceptionv1  # https://github.com/greentfrapp/lucent

# Output channels of the four branches (1x1, 3x3, 5x5, pool_reduce) of each block, read off the conv
# definitions in Lucent's InceptionV1.__init__; the concat order is 1x1|3x3|5x5|pool_reduce in its forward().
BRANCHES = ("1x1", "3x3", "5x5", "pool_reduce")
BRANCH_WIDTHS = {"mixed3a": (64, 128, 32, 32), "mixed3b": (128, 192, 96, 64), "mixed4a": (192, 204, 48, 64)}

# TF graph LRN attrs are depth_radius=5, bias=2, alpha=9.999999747378752e-05, beta=0.5. PyTorch divides alpha by
# the window size n=2*5+1, hence alpha*11 (lucent PR #51 discussion: https://github.com/greentfrapp/lucent/pull/51).
TF_LRN = dict(size=11, alpha=11 * 9.999999747378752e-05, beta=0.5, k=2.0)
LUCENT_LRN = dict(size=9, alpha=9.99999974738e-05, beta=0.5, k=1)  # values hardcoded in Lucent's forward()
_ACTIVE_LRN = {"params": TF_LRN}  # set by each model's forward pre-hook, read by the shim below

# Readout positions: 4a unit (6,6) and the 3b unit (13,13) it is centred on. By stride arithmetic both have their
# receptive-field centre at input pixel 8*13+7 = 111 (checked empirically by `rf_center` in the tests).
READOUT_POS = {"mixed3b": 13, "mixed4a": 6}


def lrn(x: torch.Tensor, params: dict) -> torch.Tensor:
    """Local response normalisation with explicit constants (https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html)."""
    return F.local_response_norm(x, **params)


def _shim_lrn(x, *args, **kwargs):
    """Drop-in for `F.local_response_norm` inside Lucent: ignores Lucent's hardcoded arguments and applies the
    constants selected by the model currently running (see `load_inceptionv1`)."""
    return lrn(x, _ACTIVE_LRN["params"])


# Lucent calls `F.local_response_norm(...)` through its module-level name `F`; we swap that name for a namespace
# that is identical to torch.nn.functional except for LRN. This changes Lucent's behaviour without copying its code.
_lucent_module = importlib.import_module("lucent.modelzoo.inceptionv1.InceptionV1")
_lucent_module.F = SimpleNamespace(**{**vars(F), "local_response_norm": _shim_lrn})


def load_inceptionv1(device: str = "cuda", lrn_mode: str = "tf") -> torch.nn.Module:
    """Load pretrained InceptionV1 (eval mode, plain ReLU) with the requested LRN constants.

    Local: weights come from ProGamerGov's conversion of the Lucid graph (downloaded and cached by torch.hub).
    `redirected_ReLU=False` because the redirected gradient only matters for feature visualisation.
    Global context: all experiments use lrn_mode="tf"; "lucent" exists for the LRN sensitivity analysis.
    """
    torch.backends.cudnn.allow_tf32 = False  # full-fp32 convolutions (TF32 would add ~1e-3 relative error)
    params = {"tf": TF_LRN, "lucent": LUCENT_LRN}[lrn_mode]  # KeyError on typos is intended
    net = inceptionv1(pretrained=True, redirected_ReLU=False).eval().to(device)
    net.register_forward_pre_hook(lambda m, a: _ACTIVE_LRN.update(params=params))  # select LRN per model
    for p in net.parameters():  # inference only: no gradients w.r.t. weights
        p.requires_grad_(False)
    return net


def preprocess(x01: torch.Tensor) -> torch.Tensor:
    """Map RGB images in [0, 1] to the Lucid InceptionV1 input range [-117, 138] (Lucent `preprocess_inceptionv1`)."""
    return x01 * 255 - 117


def branch_slices(layer: str) -> dict[str, slice]:
    """Channel range of each branch inside the concatenated `layer` output, e.g. mixed3b 5x5 -> slice(320, 416)."""
    out, start = {}, 0
    for name, width in zip(BRANCHES, BRANCH_WIDTHS[layer]):
        out[name], start = slice(start, start + width), start + width
    return out


def branch_of(layer: str, channel: int) -> str:
    """Name of the branch that produces `channel` of `layer`."""
    return next(b for b, s in branch_slices(layer).items() if s.start <= channel < s.stop)


@contextmanager
def capture(model: torch.nn.Module, names: list[str]):
    """Record the (post-ReLU) outputs of named modules during forward passes run inside the `with` block.

    Uses forward hooks (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
    """
    acts: dict[str, torch.Tensor] = {}
    hooks = [getattr(model, n).register_forward_hook(lambda m, a, o, n=n: acts.__setitem__(n, o.detach())) for n in names]
    try:
        yield acts
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def capture_pre_relu(model: torch.nn.Module, layers: list[str]):
    """Record pre-ReLU outputs of whole Inception layers (concatenation of the four `*_pre_relu_conv` outputs).

    Local: each branch conv is hooked; the layer's CatLayer hook (which fires after all branches) concatenates them.
    Global context: the Circuits authors analyse pre-activations because "ReLU just truncates" (Curve Detectors).
    """
    acts: dict[str, torch.Tensor] = {}
    parts: dict[str, dict[str, torch.Tensor]] = {l: {} for l in layers}
    hooks = []
    for l in layers:
        for b in BRANCHES:
            conv = getattr(model, f"{l}_{b}_pre_relu_conv")
            hooks.append(conv.register_forward_hook(lambda m, a, o, l=l, b=b: parts[l].__setitem__(b, o.detach())))
        hooks.append(getattr(model, l).register_forward_hook(
            lambda m, a, o, l=l: acts.__setitem__(l, torch.cat([parts[l][b] for b in BRANCHES], 1))))
    try:
        yield acts
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def patch(model: torch.nn.Module, name: str, fn):
    """Replace the output of module `name` by `fn(output)` (a forward hook that returns a value overrides the output)."""
    h = getattr(model, name).register_forward_hook(lambda m, a, o: fn(o))
    try:
        yield
    finally:
        h.remove()


def run_block(model: torch.nn.Module, layer: str, x: torch.Tensor, pre_relu: bool = False) -> torch.Tensor:
    """Re-run one Inception block on the previous layer's output `x` (mirrors Lucent's forward line by line).

    Local: mixed4a first applies `maxpool4` (3x3, stride 2, right/bottom -inf pad), as Lucent's forward does.
    Global context: lets interventions patch mixed3a/mixed3b and recompute only the next block instead of the
    whole network (see `readout_from_window`); verified against the full forward in tests/test_model.py.
    """
    conv = lambda name: getattr(model, f"{layer}_{name}_pre_relu_conv")  # Lucent naming convention
    if layer == "mixed4a":  # maxpool4 between mixed3b and mixed4a
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), 3, stride=2)
    b1 = conv("1x1")(x)
    b3 = conv("3x3")(F.pad(F.relu(conv("3x3_bottleneck")(x)), (1, 1, 1, 1)))  # 3x3 conv, zero "same" padding
    b5 = conv("5x5")(F.pad(F.relu(conv("5x5_bottleneck")(x)), (2, 2, 2, 2)))  # 5x5 conv, zero "same" padding
    bp = conv("pool_reduce")(F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float("-inf")), 3, stride=1))
    out = torch.cat([b1, b3, b5, bp], 1)  # same concat order as Lucent's CatLayer call
    return out if pre_relu else F.relu(out)


def crop_window(prev: torch.Tensor, layer: str) -> torch.Tensor:
    """Crop the part of the previous layer's map that the readout unit of `layer` can see.

    mixed3b (from 3a): 5x5 conv radius 2 -> rows p-2..p+2. mixed4a (from 3b): the 5x5 conv sees maxpool4 cells
    p-2..p+2, and cell q pools 3b rows 2q..2q+2, so 3b rows 2p-4..2p+6 (11 rows). The readout ends up at index 2.
    """
    p = READOUT_POS[layer]
    lo, hi = (2 * p - 4, 2 * p + 7) if layer == "mixed4a" else (p - 2, p + 3)
    return prev[..., lo:hi, lo:hi]


def readout_from_window(model: torch.nn.Module, layer: str, window: torch.Tensor) -> torch.Tensor:
    """Pre-ReLU vector (all channels) of the readout unit of `layer`, computed from a cropped input window."""
    return run_block(model, layer, window, pre_relu=True)[:, :, 2, 2]


def rf_center(model: torch.nn.Module, layer: str, device: str, n: int = 16) -> tuple[float, float]:
    """Empirical receptive-field centre (row, col) of the readout unit: centre of mass of |d unit / d input|.

    Local: averages gradient magnitude over `n` random images (gradient-based effective RF, Luo et al. 2016,
    https://arxiv.org/abs/1701.04128). Global context: stimuli are centred on this pixel.
    """
    x = torch.rand(n, 3, 224, 224, device=device, generator=torch.Generator(device).manual_seed(0)).requires_grad_(True)
    p, acts = READOUT_POS[layer], {}
    h = getattr(model, f"{layer}_1x1_pre_relu_conv").register_forward_hook(lambda m, a, o: acts.__setitem__("o", o))
    model(preprocess(x))  # forward with autograd on (hook keeps the non-detached 1x1-branch output)
    h.remove()
    acts["o"][:, :, p, p].sum().backward()  # sum over images and 1x1-branch channels of the readout unit
    g = x.grad.abs().sum((0, 1))  # (224, 224) gradient magnitude
    ys, xs = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing="ij")
    return float((g * ys).sum() / g.sum()), float((g * xs).sum() / g.sum())
