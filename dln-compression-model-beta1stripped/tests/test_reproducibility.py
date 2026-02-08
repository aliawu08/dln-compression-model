from dln_cycle.run_experiments import compute_fingerprint, run_smoke, spawn_rngs


def test_rng_streams_are_independent() -> None:
    env_rng, agent_rng = spawn_rngs(123)
    assert env_rng.random() != agent_rng.random()


def test_smoke_is_deterministic() -> None:
    first, _ = run_smoke(seed_count=3, n_steps=30)
    second, _ = run_smoke(seed_count=3, n_steps=30)
    assert compute_fingerprint(first) == compute_fingerprint(second)
