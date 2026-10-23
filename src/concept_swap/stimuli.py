"""Synthetic curve (arc) and straight-line stimuli, stimulus grids and counterfactual minimal pairs.

Local: renders anti-aliased strokes with PIL (4x supersampling + box downsampling), centred on the receptive-field
centre of the readout unit, and builds (base, source) pairs that differ only in curve orientation.
Global context: follows the "synthetic curve" methodology of Cammarata et al. 2020 (Curve Detectors,
https://distill.pub/2020/circuits/curve-detectors/): curves parametrised by orientation and radius. Their original
renderer is not public, so `phases/e1_replication.py` validates ours against the authors' published tuning curves.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw  # https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html

from concept_swap.config import Config

SIZE = 224  # InceptionV1 input resolution
CENTER = 111  # receptive-field centre (pixel) of the readout units, see model.READOUT_POS / model.rf_center
SS = 4  # supersampling factor for anti-aliasing


@dataclass(frozen=True)
class Stim:
    """One stimulus. theta: direction (deg, counter-clockwise, y up) from the stroke midpoint towards the arc's centre
    of curvature (the arc's tangent at the midpoint is perpendicular to theta). polarity 0: dark stroke on light."""

    kind: str  # "arc" or "line"
    theta: float
    radius: float
    width: int
    polarity: int
    dy: int  # jitter (rows)
    dx: int  # jitter (cols)


def circ_diff(a, b):
    """Signed circular difference a - b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def stroke_mask(s: Stim, span_deg: float) -> np.ndarray:
    """Anti-aliased stroke coverage in [0, 1] (SIZE x SIZE) for one arc or line.

    Local: geometry in supersampled pixel coordinates (x right, y down). u = (cos t, -sin t) is the theta direction
    in image coordinates. Arc: circle centre C = M + R u, drawn over the `span_deg` of the circle closest to M.
    PIL measures arc angles clockwise from 3 o'clock, i.e. image angle = -(math angle).
    Line: same midpoint, tangent t = (sin, cos) perpendicular to u, same length as the arc (R * span in radians).
    """
    img = Image.new("L", (SIZE * SS, SIZE * SS), 0)  # black canvas, stroke drawn in white
    draw = ImageDraw.Draw(img)  # https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
    mx, my = (CENTER + s.dx + 0.5) * SS, (CENTER + s.dy + 0.5) * SS  # midpoint (pixel centres at +0.5)
    t = np.radians(s.theta)
    u = np.array([np.cos(t), -np.sin(t)])  # theta direction in image coordinates (y axis points down)
    r, w = s.radius * SS, int(round(s.width * SS))  # radius and stroke width at supersampled resolution
    if s.kind == "arc":
        cx, cy = mx + r * u[0], my + r * u[1]
        mid = -(s.theta + 180)  # PIL angle of the direction from C back to M
        rb = r + w / 2  # PIL draws the outline inwards from the box, so this centres the stroke on radius r
        draw.arc([cx - rb, cy - rb, cx + rb, cy + rb], start=mid - span_deg / 2, end=mid + span_deg / 2, fill=255, width=w)
    else:
        half = r * np.radians(span_deg) / 2  # half the arc length
        tx, ty = np.sin(t), np.cos(t)  # tangent direction, perpendicular to u
        draw.line([mx - half * tx, my - half * ty, mx + half * tx, my + half * ty], fill=255, width=w)
    small = img.resize((SIZE, SIZE), resample=Image.Resampling.BOX)  # area-average = anti-aliasing
    return np.asarray(small, dtype=np.float32) / 255.0


def compose(mask: np.ndarray, polarity: int, background: np.ndarray | None = None) -> np.ndarray:
    """Blend a stroke onto a background (H x W x 3 in [0, 1]); default background is the opposite polarity."""
    bg = np.full((SIZE, SIZE, 3), 1.0 - polarity, np.float32) if background is None else background
    return bg * (1 - mask[..., None]) + float(polarity) * mask[..., None]  # alpha compositing


def render(df: pd.DataFrame, span_deg: float, backgrounds: np.ndarray | None = None) -> torch.Tensor:
    """Render every row of a stimulus table into an N x 3 x 224 x 224 float tensor in [0, 1]."""
    out = np.empty((len(df), SIZE, SIZE, 3), np.float32)  # preallocated HWC batch
    for i, row in enumerate(df.itertuples(index=False)):  # one stimulus per table row
        s = Stim(row.kind, row.theta, row.radius, row.width, row.polarity, row.dy, row.dx)
        out[i] = compose(stroke_mask(s, span_deg), s.polarity, None if backgrounds is None else backgrounds[i])
    return torch.from_numpy(out).permute(0, 3, 1, 2)  # NHWC -> NCHW


def grid(cfg: Config, seed: int, kind: str = "arc") -> pd.DataFrame:
    """Full factorial stimulus table: n_theta orientations x radii x widths x 2 polarities.

    Local: each nuisance cell (radius, width, polarity) gets one random jitter (dy, dx) shared by all its
    orientations, so stimuli within a cell differ only in theta. Different seeds = different splits.
    """
    rng = np.random.default_rng(seed)  # the split's seed determines all jitters
    rows = []
    for cell, (r, w, pol) in enumerate(product(cfg.radii, cfg.widths, (0, 1))):  # every nuisance combination
        dy, dx = rng.integers(-cfg.jitter_px, cfg.jitter_px + 1, size=2)  # one translation per cell, in [-J, J]
        rows += [dict(kind=kind, theta=k * 360 / cfg.n_theta, radius=r, width=w, polarity=pol, cell=cell,
                      dy=int(dy), dx=int(dx)) for k in range(cfg.n_theta)]
    return pd.DataFrame(rows)


def minimal_pairs(g: pd.DataFrame, cfg: Config, seed: int) -> pd.DataFrame:
    """For every base stimulus, sample `sources_per_base` sources from the same cell with |dtheta| >= min_dtheta.

    Local: sampling without replacement within the base's cell. Global context: these are the counterfactual
    (base, source) pairs of the interchange intervention (Huang et al. 2023, https://arxiv.org/abs/2309.10312).
    """
    rng = np.random.default_rng(seed + 10_000)  # decorrelated from the grid's jitter stream
    rows = []
    for _, cell in g.groupby("cell"):  # pairs never cross nuisance cells
        for b in cell.index:
            d = circ_diff(cell.theta.values, g.theta[b])  # orientation difference to every candidate source
            cand = cell.index[np.abs(d) >= cfg.min_dtheta]  # sources far enough from the base orientation
            for s in rng.choice(cand, cfg.sources_per_base, replace=False):  # distinct sources per base
                rows.append(dict(base=b, source=s, dtheta=float(circ_diff(g.theta[s], g.theta[b]))))
    return pd.DataFrame(rows)


def nuisance_swap_pairs(g: pd.DataFrame, factor: str) -> np.ndarray:
    """Row index of each stimulus's partner with the same kind and theta whose `factor` ("radius", "width" or
    "polarity") takes the next level in cyclic order; other factors are equal (the cell's jitter differs).

    Local: a permutation of the rows. Global context: source for the post-hoc cross-nuisance sham, which asks whether
    swapping the curve family from a same-orientation stimulus leaves the decoded orientation intact.
    """
    levels = np.sort(g[factor].unique())  # the factor's levels present in the grid
    nxt = dict(zip(levels, np.roll(levels, -1)))  # level -> next level (cyclic, so every row has a partner)
    keys = ["kind", "theta", "radius", "width", "polarity"]  # identify a stimulus up to its cell's jitter
    pos, rows = keys.index(factor), list(g[keys].itertuples(index=False, name=None))
    row = {k: i for i, k in enumerate(rows)}  # stimulus key -> row position
    return np.array([row[k[:pos] + (nxt[k[pos]],) + k[pos + 1:]] for k in rows])  # replace only `factor`
