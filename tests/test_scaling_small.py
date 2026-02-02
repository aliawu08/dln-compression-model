from dln_cycle.run_experiments import run_scaling_suite, run_smoke


def test_smoke_outputs_expected_labels() -> None:
    results, _ = run_smoke(seed_count=2, n_steps=20)
    expected = {
        "structured_agent_structured_env",
        "greedy_agent_structured_env",
        "structured_agent_unstructured_env",
        "greedy_agent_unstructured_env",
        "correct_prior",
        "wrong_prior",
        "wrong_prior_with_update",
        "wrong_prior_no_update",
        "full_cycle",
        "no_test",
        "no_update",
        "structured_cycle_cost",
        "greedy_cost",
    }
    assert expected.issubset(results.keys())


def test_scaling_slope_direction() -> None:
    slopes = run_scaling_suite([4, 6, 8], seed_count=3, n_steps=30, lambda_cog=0.2)
    assert slopes["greedy"] > slopes["structured"]
