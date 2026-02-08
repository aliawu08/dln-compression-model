import numpy as np
import pytest

from dln_cycle.metrics import compute_entropy


def test_entropy_uniform_is_one() -> None:
    probs = np.ones(4) / 4
    assert compute_entropy(probs) == pytest.approx(1.0, rel=1e-6)


def test_entropy_delta_is_zero() -> None:
    probs = np.array([1.0, 0.0, 0.0])
    assert compute_entropy(probs) == pytest.approx(0.0, rel=1e-6)


def test_entropy_in_range() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random(5)
    probs /= probs.sum()
    entropy = compute_entropy(probs)
    assert 0.0 <= entropy <= 1.0
