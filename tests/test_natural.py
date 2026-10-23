"""Tests for `concept_swap.natural`: centred crops and blur-free (exact) rotations on square crops."""

import numpy as np
from PIL import Image

from concept_swap import natural as N


def _photo(tmp_path, w=500, h=380):
    """A random-noise RGB image saved as PNG (lossless, so pixel values are exact)."""
    p = tmp_path / "img.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
    return p


def test_square_option_keeps_the_unrotated_crop(tmp_path):
    p = _photo(tmp_path)
    assert np.array_equal(N.crop(p, 0.0, square=True), N.crop(p, 0.0, square=False))


def test_square_rotations_by_90_and_180_are_exact_transposes(tmp_path):
    """With square=True, rotating by 90/180/270 equals np.rot90 of the square (pure pixel permutation, no
    interpolation); without the square crop, PIL's 90-degree rotation of a non-square image is resampled."""
    p = _photo(tmp_path)
    im = Image.open(p).convert("RGB")
    s = N.SHORT / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.BILINEAR)
    l, t = (im.width - N.SHORT) // 2, (im.height - N.SHORT) // 2
    sq = np.asarray(im.crop((l, t, l + N.SHORT, t + N.SHORT)), dtype=np.float32) / 255  # independent re-derivation
    c = N.SHORT // 2
    box = (slice(c - N.CENTER, c - N.CENTER + N.SIZE),) * 2
    for angle, k in ((90, 1), (180, 2), (-90, 3)):  # PIL rotates counter-clockwise, like np.rot90 with positive k
        assert np.array_equal(N.crop(p, angle, square=True), np.rot90(sq, k)[box])
    assert not np.array_equal(N.crop(p, 90, square=False), np.rot90(sq, 1)[box])
