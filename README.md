# DLN Computational Framework

A computational instantiation of three DLN cognitive stages (Dot → Linear → Network) testing a **compression-efficiency** thesis: Network cognition is more efficient because it represents shared structure once, rather than redundantly across options.

This repository accompanies the paper *"Compression Efficiency and Structural Learning as a Computational Model of DLN Cognitive Stages"* (Wu, 2026).

**Preprint:** [ResearchGate](https://www.researchgate.net/publication/400341092_Compression_Efficiency_and_Structural_Learning_as_a_Computational_Model_of_DLN_Cognitive_Stages) | DOI: [10.13140/RG.2.2.11937.26728](https://doi.org/10.13140/RG.2.2.11937.26728)

**Contact:** Alia Wu — wut08@nyu.edu
**ORCID:** [0009-0005-4424-102X](https://orcid.org/0009-0005-4424-102X)

---

## Key Claims

| Compression Target | Description |
|--------------------|-------------|
| **Core** (options) | Network represents shared option structure once (O(F) factors) instead of independently (O(K) options) |
| **Variable** (stakes) | Network learns factor-level exposure structure and tracks cumulative exposure to hedge common risk drivers once |
| **Dual-purpose** | Actions that are simultaneously good for expected reward AND reduce marginal exposure penalty |

---

## Repository Structure

```
├── src/
│   ├── dln_core_variable_cycle.py   # Main simulation (paper results)
│   └── dln_cycle/                   # Supporting module library
│       ├── agents.py                # Reusable agent implementations
│       ├── cycle.py                 # Hypothesis test-update cycle logic
│       ├── envs.py                  # Bandit environment definitions
│       ├── metrics.py               # Entropy and utility computations
│       └── run_experiments.py       # Ablation suite runner
├── paper/
│   ├── main.tex                     # LaTeX source
│   ├── claims.yaml                  # Formal claims verification spec
│   └── figures/*.png                # Publication figures
├── outputs/paper/
│   ├── results/
│   │   ├── episode_metrics.csv      # Per-seed, per-condition metrics
│   │   └── manifest.json            # Run configuration
│   └── artifacts/
│       ├── tables/agg_summary.csv   # Aggregated results
│       └── figures/*.png            # Generated figures
├── tests/                           # pytest suite
├── requirements.txt
└── pyproject.toml
```

---

## Agents

| Agent | DLN Stage | Description | Cognitive Cost |
|-------|-----------|-------------|----------------|
| **DotRandom** | Dot | Uniform random action selection; no learning | O(1) |
| **LinearTabular** | Linear | Learns Q-values per option; ignores cumulative exposure cross-terms | O(K) |
| **NetworkCycle** | Network | Factor-level learning with structural hypothesis → predictive test → update cycle; tracks cumulative exposure | O(F) |

NetworkCycle variants for ablation:
- `Network-Full`: test + update enabled
- `Network-NoTest`: updates without testing
- `Network-NoUpdate`: tests without updating

---

## Quickstart

**Requirements:** Python ≥ 3.8

```bash
# Install
pip install -r requirements.txt

# Run smoke test (fast, ~15 seeds)
python src/dln_core_variable_cycle.py --preset smoke --out outputs/smoke

# Run paper suite (100 seeds, K ∈ {20, 50, 100, 200})
python src/dln_core_variable_cycle.py --preset paper --out outputs/paper

# Run tests
pytest
```

---

## Experiment Pipelines

This repository provides two experiment runners for different use cases:

| Script | Purpose | Presets |
|--------|---------|---------|
| `src/dln_core_variable_cycle.py` | **Paper results** — generates publication figures and claim verification data | `smoke`, `paper`, `boundary-kf`, `stakes-sweep` |
| `src/dln_cycle/run_experiments.py` | **Ablation suite** — modular experiments with reproducibility fingerprints | `smoke`, `ablate-small`, `paper` |

**When to use each:**

- Use `dln_core_variable_cycle.py` to reproduce the paper's main results and figures. This is the primary entry point for verifying claims.
- Use `dln_cycle/run_experiments.py` for ablation studies, scaling analysis, and experiments requiring the modular `dln_cycle` components (e.g., `ShiftedStructuredBandit` for distribution shifts).

```bash
# Paper results (main simulation)
python src/dln_core_variable_cycle.py --preset paper --out outputs/paper

# Ablation suite (modular experiments)
python src/dln_cycle/run_experiments.py --preset ablate-small --out outputs/ablate-small
```

---

## Reproducing Paper Results

The `--preset paper` run generates:

| Output | Description |
|--------|-------------|
| `results/episode_metrics.csv` | Per-episode metrics for all seeds and conditions |
| `results/manifest.json` | Full configuration (seeds, K values, cost weights, etc.) |
| `artifacts/tables/agg_summary.csv` | Aggregated mean ± std across seeds |
| `artifacts/figures/*.png` | Publication figures |

Pre-computed outputs are committed in `outputs/paper/` for reproducibility verification.

---

## Citation

```bibtex
@misc{wu_dln_compression_2026,
  title  = {Compression Efficiency and Structural Learning
            as a Computational Model of DLN Cognitive Stages},
  author = {Wu, Alia},
  year   = {2026},
  doi    = {10.13140/RG.2.2.11937.26728}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
