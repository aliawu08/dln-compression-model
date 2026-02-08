"""Metrics and utility functions for DLN simulations."""

from __future__ import annotations

import numpy as np


def compute_entropy(probs: np.ndarray) -> float:
    """Compute normalized Shannon entropy in [0, 1]."""
    probs = np.clip(probs, 1e-12, 1.0)
    raw = -np.sum(probs * np.log(probs))
    if raw < 1e-8:
        return 0.0
    normalizer = np.log(len(probs)) if len(probs) > 1 else 1.0
    return float(raw / normalizer)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    """Compute softmax with temperature."""
    if temperature <= 1e-12:
        result = np.zeros_like(values)
        result[np.argmax(values)] = 1.0
        return result
    scaled = (values - np.max(values)) / temperature
    exp_vals = np.exp(scaled)
    return exp_vals / np.sum(exp_vals)
