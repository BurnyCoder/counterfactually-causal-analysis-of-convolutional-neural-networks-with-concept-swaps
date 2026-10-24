"""Readouts that turn downstream activations into the concept value (curve orientation) or an effect size.

Local: z-scoring, a nearest-template orientation decoder over a neuron family, a ridge (cos, sin) decoder over a
whole layer, and the normalized patching effect.
Global context: these implement Huang et al.'s behaviour map tau (https://arxiv.org/abs/2309.10312): the
intervention succeeds when tau(patched base) = tau(source). For a CNN without an orientation-valued output, tau is
read from the next layer's curve family (primary) or from the whole next layer (secondary, non-circular).
"""

import numpy as np
from sklearn.linear_model import RidgeCV  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from concept_swap.stimuli import circ_diff as circ_dist  # signed circular difference in degrees


class ZScore:
    """Per-feature standardisation with statistics frozen on the selection split (no test-set leakage)."""

    def fit(self, X: np.ndarray) -> "ZScore":
        """Store per-feature mean and std of the selection-split responses X (n, d)."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8  # epsilon guards never-active units
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Standardise X with the frozen selection statistics."""
        return (X - self.mu) / self.sd


class TemplateDecoder:
    """Nearest-template orientation decoder: for a response vector, return the orientation whose clean template
    (from the same nuisance cell) has the highest Pearson correlation with it.

    Local: templates are the selection-split responses, one per (cell, theta). Global context: primary tau.
    """

    def fit(self, Z: np.ndarray, theta: np.ndarray, cell: np.ndarray) -> "TemplateDecoder":
        """Store each cell's clean response vectors (rows of Z) as templates labelled by theta."""
        # one (templates, labels) table per nuisance cell; templates are pre-normalised for correlation
        self.t = {c: (self._center(Z[cell == c]), theta[cell == c]) for c in np.unique(cell)}
        return self

    @staticmethod
    def _center(X: np.ndarray) -> np.ndarray:
        """Row-wise centring and unit-norm scaling, so a dot product equals the Pearson correlation."""
        X = X - X.mean(1, keepdims=True)  # remove each vector's mean across units
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # unit length (epsilon avoids 0/0)

    def correlations(self, Z: np.ndarray, c) -> tuple[np.ndarray, np.ndarray]:
        """Template labels of cell `c` and the (rows x templates) Pearson correlations of the rows of Z with them.

        Local: what `predict` maximises. Global context: the example figures plot it as the decoder's output curve.
        """
        T, th = self.t[c]
        return th, self._center(Z) @ T.T  # unit-norm centred vectors: dot product = Pearson r

    def predict(self, Z: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Label of the best-correlated template from each row's own cell."""
        out = np.full(len(Z), np.nan)  # decoded label per row (NaN until decoded)
        for c in self.t:  # decode each cell's rows against that cell's templates only
            m = cell == c
            th, r = self.correlations(Z[m], c)
            out[m] = th[np.argmax(r, axis=1)]  # best-correlated template's label
        assert not np.isnan(out).any(), "a row belongs to a cell without templates"
        return out


class RidgeDecoder:
    """Linear decoder of (cos theta, sin theta) from standardised features; returns atan2 angle in degrees.

    Local: RidgeCV picks the L2 penalty by efficient leave-one-out CV on the selection split.
    Global context: secondary, non-circular readout over all 508 mixed4a channels.
    """

    def fit(self, X: np.ndarray, theta: np.ndarray) -> "RidgeDecoder":
        """Fit standardisation + RidgeCV mapping features X to (cos theta, sin theta)."""
        t = np.radians(theta)  # circular target encoded as (cos, sin) so 355 deg and 5 deg are neighbours
        self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(X, np.c_[np.cos(t), np.sin(t)])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decoded orientation in degrees [0, 360) for every row of X."""
        cs = self.m.predict(X)  # predicted (cos, sin) per row
        return np.degrees(np.arctan2(cs[:, 1], cs[:, 0])) % 360  # back to an angle in [0, 360)


def denominator_mask(rb: np.ndarray, rs: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Keep pairs whose clean base/source difference |rs - rb|^2 is above its q-quantile (avoids 0/0 blow-ups)."""
    d = ((rs - rb) ** 2).sum(-1)  # squared length of the clean base -> source displacement per pair
    return d > np.quantile(d, q)  # True for pairs above the q-quantile


def normalized_effect(rp: np.ndarray, rb: np.ndarray, rs: np.ndarray, w: np.ndarray | None = None) -> float:
    """Ratio-of-sums patching effect sum<rp-rb, rs-rb> / sum|rs-rb|^2: 0 = patch did nothing, 1 = output became the
    source's (projection onto the base->source direction; Heimersheim & Nanda 2024, https://arxiv.org/abs/2404.15255,
    recommend such continuous, roughly linear metrics). `w` are optional per-pair (bootstrap) weights."""
    w = np.ones(len(rb)) if w is None else w  # unweighted by default
    # numerator: how far the patched readout moved along the base->source direction; denominator: that direction's length^2
    return float((w * ((rp - rb) * (rs - rb)).sum(-1)).sum() / (w * ((rs - rb) ** 2).sum(-1)).sum())
