"""Shared pytest fixtures.

Local: loads InceptionV1 once per test session (weights are cached by torch.hub after the first download).
Global context: model-level invariants (block re-implementation, cropped readout, patch hooks) are tested
against the real network, because the causal claims in the paper rest on these invariants.
"""

import pytest
import torch

from concept_swap.model import load_inceptionv1


@pytest.fixture(scope="session")
def device() -> str:
    """CUDA when available, otherwise CPU (tests must also pass on CPU-only machines)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model(device):
    """InceptionV1 with TensorFlow-exact LRN in eval mode (the configuration used by every experiment)."""
    return load_inceptionv1(device, lrn_mode="tf")


@pytest.fixture(scope="session")
def images(device) -> torch.Tensor:
    """Two random RGB images in [0, 1] (shape 2x3x224x224) as generic test inputs."""
    return torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0)).to(device)
