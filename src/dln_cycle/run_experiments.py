"""Experiment runner for DLN cycle simulations."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .agents import Agent, GreedyAgent, StructuredAgent
from .cycle import CycleConfig, CycleMonitor
from .envs import BanditEnvironment, ShiftedStructuredBandit, StructuredBandit


@dataclass
class ExperimentResult:
    label: str
    mean_reward: float
    std_reward: float
    mean_entropy: float
    std_entropy: float
    mean_regret: float
    std_regret: float
    mean_cost_adjusted: float
    std_cost_adjusted: float
    n_seeds: int


def spawn_rngs(seed: int) -> Tuple[np.random.Generator, np.random.Generator]:
    sequence = np.random.SeedSequence(seed)
    env_ss, agent_ss = sequence.spawn(2)
    return np.random.default_rng(env_ss), np.random.default_rng(agent_ss)


def run_episode(
    agent: Agent,
    env: BanditEnvironment,
    n_steps: int,
    agent_state,
    cycle_config: CycleConfig | None = None,
) -> Tuple[List[float], List[float]]:
    rewards: List[float] = []
    entropies: List[float] = []
    monitor = CycleMonitor(cycle_config) if cycle_config else None

    env.reset()
    if isinstance(agent, StructuredAgent) and monitor:
        agent.reset_model()

    for _ in range(n_steps):
        if isinstance(agent, StructuredAgent):
            features = env.features
            action, entropy = agent.select_action(agent_state, features)
        else:
            action, entropy = agent.select_action(agent_state)
        reward = env.pull(action)
        agent_state.update(action, reward, entropy)
        rewards.append(reward)
        entropies.append(entropy)

        if isinstance(agent, StructuredAgent):
            prediction = float(env.features[action] @ agent.weights)
            if not monitor or monitor.config.update_enabled:
                agent.update_model(env.features[action], reward)
            if monitor:
                mismatch = monitor.record_error(abs(reward - prediction))
                if mismatch and monitor.config.update_enabled:
                    reset_mean = (
                        np.zeros(agent.n_features)
                        if monitor.config.reset_to_zero
                        else None
                    )
                    agent.reset_model(
                        prior_mean=reset_mean,
                        prior_precision=monitor.config.reset_prior_precision,
                    )

    return rewards, entropies


def agent_state_init(agent: Agent):
    from .agents import AgentState

    return AgentState.create(agent.n_arms)


def run_condition(
    label: str,
    agent_factory: Callable[[np.random.Generator], Agent],
    env_factory: Callable[[np.random.Generator], BanditEnvironment],
    seeds: List[int],
    n_steps: int,
    lambda_cog: float = 0.0,
    cycle_config: CycleConfig | None = None,
) -> ExperimentResult:
    rewards = []
    entropies = []
    regrets = []
    cost_adjusted = []

    for seed in seeds:
        env_rng, agent_rng = spawn_rngs(seed)
        env = env_factory(env_rng)
        agent = agent_factory(agent_rng)
        agent_state = agent_state_init(agent)
        episode_rewards, episode_entropies = run_episode(
            agent, env, n_steps, agent_state, cycle_config=cycle_config
        )
        mean_reward = float(np.mean(episode_rewards))
        rewards.append(mean_reward)
        mean_entropy = float(np.mean(episode_entropies))
        entropies.append(mean_entropy)
        regrets.append(float(env.optimal_reward() - mean_reward))
        cost_adjusted.append(float(mean_reward - lambda_cog * (1.0 - mean_entropy)))

    return ExperimentResult(
        label=label,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_entropy=float(np.mean(entropies)),
        std_entropy=float(np.std(entropies)),
        mean_regret=float(np.mean(regrets)),
        std_regret=float(np.std(regrets)),
        mean_cost_adjusted=float(np.mean(cost_adjusted)),
        std_cost_adjusted=float(np.std(cost_adjusted)),
        n_seeds=len(seeds),
    )


def run_ablation_suite(
    seeds: List[int], n_steps: int, lambda_cog: float
) -> Tuple[Dict[str, ExperimentResult], Dict[str, float | int | List[int] | None]]:
    results: Dict[str, ExperimentResult] = {}
    n_arms = 6
    n_features = 3

    def structured_env(
        rng: np.random.Generator,
        structured: bool,
        weights: np.ndarray | None = None,
    ) -> StructuredBandit:
        return StructuredBandit(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            structured=structured,
            weights=weights,
        )

    def structured_agent(
        rng: np.random.Generator,
        prior_mean: np.ndarray,
        temperature: float = 0.2,
    ) -> StructuredAgent:
        return StructuredAgent(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            prior_mean=prior_mean,
            prior_precision=2.0,
            temperature=temperature,
        )

    def greedy_agent(rng: np.random.Generator) -> GreedyAgent:
        return GreedyAgent(n_arms=n_arms, rng=rng, temperature=0.2)

    # Ablation Group 1: Structure present vs absent
    results["structured_agent_structured_env"] = run_condition(
        "structured_agent_structured_env",
        agent_factory=lambda rng: structured_agent(rng, prior_mean=np.zeros(n_features)),
        env_factory=lambda rng: structured_env(rng, structured=True),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )
    results["greedy_agent_structured_env"] = run_condition(
        "greedy_agent_structured_env",
        agent_factory=greedy_agent,
        env_factory=lambda rng: structured_env(rng, structured=True),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )
    results["structured_agent_unstructured_env"] = run_condition(
        "structured_agent_unstructured_env",
        agent_factory=lambda rng: structured_agent(rng, prior_mean=np.zeros(n_features)),
        env_factory=lambda rng: structured_env(rng, structured=False),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )
    results["greedy_agent_unstructured_env"] = run_condition(
        "greedy_agent_unstructured_env",
        agent_factory=greedy_agent,
        env_factory=lambda rng: structured_env(rng, structured=False),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )

    # Ablation Group 2: Prior correctness
    true_weights = np.ones(n_features) * 0.2

    def correct_prior_agent(rng: np.random.Generator) -> StructuredAgent:
        return structured_agent(rng, prior_mean=true_weights)

    def wrong_prior_agent(
        rng: np.random.Generator, prior_precision: float = 2.0
    ) -> StructuredAgent:
        wrong_weights = np.ones(n_features) * -0.6
        return StructuredAgent(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            prior_mean=wrong_weights,
            prior_precision=prior_precision,
        )

    results["correct_prior"] = run_condition(
        "correct_prior",
        agent_factory=correct_prior_agent,
        env_factory=lambda rng: structured_env(rng, structured=True, weights=true_weights),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )
    results["wrong_prior"] = run_condition(
        "wrong_prior",
        agent_factory=lambda rng: wrong_prior_agent(rng, prior_precision=2.0),
        env_factory=lambda rng: structured_env(rng, structured=True, weights=true_weights),
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )

    # Ablation Group 3: Cycle necessity
    cycle_env_factory = lambda rng: ShiftedStructuredBandit(
        n_arms=n_arms,
        n_features=n_features,
        rng=rng,
        shift_time=n_steps // 2,
        structured=True,
    )

    base_cycle = CycleConfig(error_window=6, mismatch_threshold=0.1)
    wrong_prior_cycle = CycleConfig(
        error_window=4,
        mismatch_threshold=1.0,
    )
    results["wrong_prior_with_update"] = run_condition(
        "wrong_prior_with_update",
        agent_factory=lambda rng: wrong_prior_agent(rng, prior_precision=0.2),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=wrong_prior_cycle,
    )
    results["wrong_prior_no_update"] = run_condition(
        "wrong_prior_no_update",
        agent_factory=lambda rng: wrong_prior_agent(rng, prior_precision=20.0),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=CycleConfig(
            error_window=wrong_prior_cycle.error_window,
            mismatch_threshold=wrong_prior_cycle.mismatch_threshold,
            test_enabled=True,
            update_enabled=False,
        ),
    )
    results["full_cycle"] = run_condition(
        "full_cycle",
        agent_factory=lambda rng: structured_agent(rng, prior_mean=np.zeros(n_features)),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=base_cycle,
    )
    results["no_test"] = run_condition(
        "no_test",
        agent_factory=lambda rng: structured_agent(rng, prior_mean=np.zeros(n_features)),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=CycleConfig(
            error_window=base_cycle.error_window,
            mismatch_threshold=base_cycle.mismatch_threshold,
            test_enabled=False,
            update_enabled=True,
        ),
    )
    results["no_update"] = run_condition(
        "no_update",
        agent_factory=lambda rng: structured_agent(rng, prior_mean=np.zeros(n_features)),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=CycleConfig(
            error_window=base_cycle.error_window,
            mismatch_threshold=base_cycle.mismatch_threshold,
            test_enabled=True,
            update_enabled=False,
        ),
    )

    results["structured_cycle_cost"] = run_condition(
        "structured_cycle_cost",
        agent_factory=lambda rng: structured_agent(
            rng, prior_mean=np.zeros(n_features), temperature=0.6
        ),
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
        cycle_config=base_cycle,
    )
    results["greedy_cost"] = run_condition(
        "greedy_cost",
        agent_factory=greedy_agent,
        env_factory=cycle_env_factory,
        seeds=seeds,
        n_steps=n_steps,
        lambda_cog=lambda_cog,
    )

    config = {
        "N": n_arms,
        "F": n_features,
        "T": n_steps,
        "drift_params": {"type": "shift", "shift_time": n_steps // 2},
        "lambda_cog": lambda_cog,
        "seeds": seeds,
    }
    return results, config


def run_smoke(
    seed_count: int = 4, n_steps: int = 60
) -> Tuple[Dict[str, ExperimentResult], Dict[str, float | int | List[int] | None]]:
    seeds = list(range(seed_count))
    return run_ablation_suite(seeds, n_steps, lambda_cog=0.8)


def run_ablate_small(
    seed_count: int = 8, n_steps: int = 80
) -> Tuple[Dict[str, ExperimentResult], Dict[str, float | int | List[int] | None]]:
    seeds = list(range(seed_count))
    return run_ablation_suite(seeds, n_steps, lambda_cog=0.8)


def run_paper(
    seed_count: int = 50, n_steps: int = 200
) -> Tuple[Dict[str, ExperimentResult], Dict[str, float | int | List[int] | None]]:
    seeds = list(range(seed_count))
    return run_ablation_suite(seeds, n_steps, lambda_cog=0.8)


def run_scaling_suite(
    n_values: List[int],
    seed_count: int = 4,
    n_steps: int = 50,
    lambda_cog: float = 0.8,
) -> Dict[str, float]:
    slopes: Dict[str, float] = {}
    n_features = 3
    seeds = list(range(seed_count))

    def structured_env_factory(rng: np.random.Generator, n_arms: int) -> StructuredBandit:
        return StructuredBandit(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            structured=True,
        )

    def structured_agent_factory(
        rng: np.random.Generator,
        n_arms: int,
        temperature: float = 0.2,
    ) -> StructuredAgent:
        return StructuredAgent(
            n_arms=n_arms,
            n_features=n_features,
            rng=rng,
            prior_mean=np.zeros(n_features),
            prior_precision=2.0,
            temperature=temperature,
        )

    base_cycle = CycleConfig(error_window=6, mismatch_threshold=0.1)
    greedy_regrets = []
    structured_regrets = []

    for n_arms in n_values:
        greedy_result = run_condition(
            f"greedy_scaling_{n_arms}",
            agent_factory=lambda rng, n_arms=n_arms: GreedyAgent(
                n_arms=n_arms, rng=rng, temperature=0.2
            ),
            env_factory=lambda rng, n_arms=n_arms: structured_env_factory(rng, n_arms),
            seeds=seeds,
            n_steps=n_steps,
            lambda_cog=lambda_cog,
        )
        structured_result = run_condition(
            f"structured_scaling_{n_arms}",
            agent_factory=lambda rng, n_arms=n_arms: structured_agent_factory(
                rng, n_arms
            ),
            env_factory=lambda rng, n_arms=n_arms: structured_env_factory(rng, n_arms),
            seeds=seeds,
            n_steps=n_steps,
            lambda_cog=lambda_cog,
            cycle_config=base_cycle,
        )
        greedy_regrets.append(greedy_result.mean_regret)
        structured_regrets.append(structured_result.mean_regret)

    slopes["greedy"] = float(np.polyfit(n_values, greedy_regrets, 1)[0])
    slopes["structured"] = float(np.polyfit(n_values, structured_regrets, 1)[0])
    return slopes


def compute_fingerprint(results: Dict[str, ExperimentResult]) -> Dict[str, Dict[str, float]]:
    fingerprint: Dict[str, Dict[str, float]] = {}
    for key in sorted(results.keys()):
        result = results[key]
        fingerprint[key] = {
            "mean_reward": round(result.mean_reward, 6),
            "mean_entropy": round(result.mean_entropy, 6),
            "mean_regret": round(result.mean_regret, 6),
            "mean_cost_adjusted": round(result.mean_cost_adjusted, 6),
        }
    return fingerprint


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def gather_versions(packages: List[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def save_results(results: Dict[str, ExperimentResult], preset: str) -> Path:
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{preset}_results.json"
    payload = {key: asdict(value) for key, value in results.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_manifest(
    output_dir: Path,
    preset: str,
    config: Dict[str, float | int | List[int] | None],
) -> Path:
    manifest = {
        "git_sha": get_git_sha(),
        "preset": preset,
        "config": config,
        "packages": gather_versions(["numpy", "pytest"]),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DLN cycle experiments.")
    parser.add_argument(
        "--preset", choices=["smoke", "ablate-small", "paper"], default="smoke"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "paper":
        results, config = run_paper()
    elif args.preset == "ablate-small":
        results, config = run_ablate_small()
    else:
        results, config = run_smoke()

    output_path = save_results(results, args.preset)
    output_dir = output_path.parent
    save_manifest(output_dir, args.preset, config)
    if args.preset == "smoke":
        fingerprint_path = output_dir / "smoke_fingerprint.json"
        fingerprint_path.write_text(json.dumps(compute_fingerprint(results), indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
