from dln_cycle.run_experiments import run_smoke


def test_ablation_directions() -> None:
    results, _ = run_smoke(seed_count=4, n_steps=50)

    structured_advantage = (
        results["structured_agent_structured_env"].mean_reward
        - results["greedy_agent_structured_env"].mean_reward
    )
    unstructured_gap = (
        results["structured_agent_unstructured_env"].mean_reward
        - results["greedy_agent_unstructured_env"].mean_reward
    )
    assert structured_advantage > -0.02
    assert abs(unstructured_gap) < 0.05

    assert (
        results["correct_prior"].mean_reward
        >= results["wrong_prior"].mean_reward
    )

    assert (
        results["full_cycle"].mean_reward
        >= results["no_update"].mean_reward
    )

    assert (
        results["wrong_prior_with_update"].mean_reward
        >= results["wrong_prior_no_update"].mean_reward
    )

    assert (
        results["structured_cycle_cost"].mean_cost_adjusted
        >= results["greedy_cost"].mean_cost_adjusted
    )
