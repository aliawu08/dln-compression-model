"""Agent implementations for DLN simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .metrics import compute_entropy, softmax


@dataclass
class AgentState:
    q_values: np.ndarray
    counts: np.ndarray
    total_reward: float
    rewards_history: List[float]
    entropy_history: List[float]

    @classmethod
    def create(cls, n_arms: int) -> "AgentState":
        return cls(
            q_values=np.zeros(n_arms) + 0.5,
            counts=np.zeros(n_arms),
            total_reward=0.0,
            rewards_history=[],
            entropy_history=[],
        )

    def update(self, arm: int, reward: float, entropy: float) -> None:
        self.counts[arm] += 1
        alpha = 1.0 / self.counts[arm]
        self.q_values[arm] += alpha * (reward - self.q_values[arm])
        self.total_reward += reward
        self.rewards_history.append(reward)
        self.entropy_history.append(entropy)


class Agent:
    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        self.n_arms = n_arms
        self.rng = rng
        self.name = "Base"

    def select_action(self, state: AgentState) -> Tuple[int, float]:
        raise NotImplementedError

    def run_episode(self, env, n_steps: int) -> AgentState:
        state = AgentState.create(self.n_arms)
        env.reset()
        for _ in range(n_steps):
            action, entropy = self.select_action(state)
            reward = env.pull(action)
            state.update(action, reward, entropy)
        return state


class RandomAgent(Agent):
    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        super().__init__(n_arms, rng)
        self.name = "Random"

    def select_action(self, state: AgentState) -> Tuple[int, float]:
        probs = np.ones(self.n_arms) / self.n_arms
        action = self.rng.choice(self.n_arms)
        return int(action), compute_entropy(probs)


class GreedyAgent(Agent):
    def __init__(self, n_arms: int, rng: np.random.Generator, temperature: float = 0.1) -> None:
        super().__init__(n_arms, rng)
        self.name = "Greedy"
        self.temperature = temperature

    def select_action(self, state: AgentState) -> Tuple[int, float]:
        probs = softmax(state.q_values, self.temperature)
        action = self.rng.choice(self.n_arms, p=probs)
        return int(action), compute_entropy(probs)


class StructuredAgent(Agent):
    """Linear model agent that exploits shared structure."""

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        rng: np.random.Generator,
        prior_mean: np.ndarray,
        prior_precision: float = 1.0,
        noise_precision: float = 25.0,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(n_arms, rng)
        self.name = "Structured"
        self.n_features = n_features
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision
        self.temperature = temperature
        self.reset_model()

    def reset_model(
        self,
        prior_mean: np.ndarray | None = None,
        prior_precision: float | None = None,
    ) -> None:
        if prior_mean is None:
            prior_mean = self.prior_mean
        if prior_precision is None:
            prior_precision = self.prior_precision
        self.a_matrix = prior_precision * np.eye(self.n_features)
        self.b_vector = prior_precision * prior_mean.copy()
        self.weights = np.linalg.solve(self.a_matrix, self.b_vector)

    def update_model(self, feature: np.ndarray, reward: float) -> None:
        self.a_matrix += self.noise_precision * np.outer(feature, feature)
        self.b_vector += self.noise_precision * feature * reward
        self.weights = np.linalg.solve(self.a_matrix, self.b_vector)

    def select_action(self, state: AgentState, features: np.ndarray) -> Tuple[int, float]:
        preds = features @ self.weights
        probs = softmax(preds, temperature=self.temperature)
        action = self.rng.choice(self.n_arms, p=probs)
        return int(action), compute_entropy(probs)
