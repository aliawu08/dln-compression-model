"""Cycle hypothesis-test-update utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CycleConfig:
    error_window: int = 10
    mismatch_threshold: float = 0.12
    test_enabled: bool = True
    update_enabled: bool = True
    reset_to_zero: bool = False
    allow_repeat_resets: bool = False
    reset_prior_precision: float | None = None


class CycleMonitor:
    def __init__(self, config: CycleConfig) -> None:
        self.config = config
        self.errors: List[float] = []
        self.triggered = False

    def record_error(self, error: float) -> bool:
        self.errors.append(error)
        if len(self.errors) > self.config.error_window:
            self.errors.pop(0)
        if not self.config.test_enabled:
            return False
        if len(self.errors) < self.config.error_window:
            return False
        mean_error = float(np.mean(self.errors))
        if mean_error >= self.config.mismatch_threshold:
            if self.config.allow_repeat_resets:
                return True
            if not self.triggered:
                self.triggered = True
                return True
        return False
