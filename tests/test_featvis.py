"""Tests for `concept_swap.featvis`: Lucent feature visualizations of single channels."""

from concept_swap import featvis as V


def test_render_units_returns_one_rgb_image_per_unit(model):
    """One optimised 224-px RGB image in [0, 1] per (layer, channel), from one batched optimisation."""
    imgs = V.render_units(model, [("mixed3b", 379), ("mixed4a", 26)], steps=8, seed=0)
    assert imgs.shape == (2, 224, 224, 3) and imgs.min() >= 0 and imgs.max() <= 1
