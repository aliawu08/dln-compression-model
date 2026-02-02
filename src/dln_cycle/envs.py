"""Environment definitions for DLN simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EnvState:
    arm_means: np.ndarray
    t: int = 0


class BanditEnvironment:
    """Multi-armed bandit with configurable distributions."""

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        reward_std: float = 0.1,
        arm_means: Optional[np.ndarray] = None,
    ) -> None:
        self.n_arms = n_arms
        self.rng = rng
        self.reward_std = reward_std
        if arm_means is None:
            arm_means = self.rng.uniform(0.2, 0.8, n_arms)
        self.state = EnvState(arm_means=arm_means.copy())
        self._initial_means = arm_means.copy()

    def reset(self) -> None:
        self.state = EnvState(arm_means=self._initial_means.copy(), t=0)

    def pull(self, arm: int) -> float:
        reward = self.rng.normal(self.state.arm_means[arm], self.reward_std)
        self.state.t += 1
        return float(reward)

    def optimal_arm(self) -> int:
        return int(np.argmax(self.state.arm_means))

    def optimal_reward(self) -> float:
        return float(self.state.arm_means[self.optimal_arm()])


class StructuredBandit(BanditEnvironment):
    """Bandit with optional shared low-rank structure."""

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        rng: np.random.Generator,
        reward_std: float = 0.1,
        structured: bool = True,
        weight_scale: float = 0.4,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.features = rng.normal(0.0, 1.0, size=(n_arms, n_features))
        self.structured = structured
        if structured:
            if weights is None:
                weights = rng.normal(0.0, weight_scale, size=n_features)
            arm_means = 0.5 + self.features @ weights
        else:
            arm_means = rng.uniform(0.2, 0.8, n_arms)
        arm_means = np.clip(arm_means, 0.05, 0.95)
        super().__init__(n_arms, rng=rng, reward_std=reward_std, arm_means=arm_means)
        self.weights = weights if structured else None


class ShiftedStructuredBandit(StructuredBandit):
    """Bandit that changes its structure mid-episode."""

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        rng: np.random.Generator,
        shift_time: int = 50,
        reward_std: float = 0.1,
        structured: bool = True,
        weight_scale: float = 0.4,
    ) -> None:
        self.shift_time = shift_time
        super().__init__(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            reward_std=reward_std,
            structured=structured,
            weight_scale=weight_scale,
        )
        self._initial_weights = None if self.weights is None else self.weights.copy()

    def reset(self) -> None:
        super().reset()
        if self._initial_weights is not None:
            self.weights = self._initial_weights.copy()

    def pull(self, arm: int) -> float:
        if self.state.t == self.shift_time and self.structured:
            new_weights = self.rng.normal(0.0, 0.4, size=self.features.shape[1])
            self.weights = new_weights
            shifted_means = 0.5 + self.features @ new_weights
            self.state.arm_means = np.clip(shifted_means, 0.05, 0.95)
        return super().pull(arm)
