#!/usr/bin/env python3
"""Test Proposition 1(i): empirical crossover matches analytic K*.

Computes the empirical crossover point (where Network CAU exceeds Linear CAU)
from simulation data and compares to the analytic formula K* = F + c_meta/c_param.

Key finding: the empirical crossover is much earlier than the analytic prediction,
because the compression advantage is primarily *statistical* (better predictions
from factor pooling / shrinkage) rather than purely *computational* (fewer
parameters to store).  The analytic K* only captures the cost component.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Load paper preset results
data_path = Path(__file__).parent.parent / "outputs" / "paper" / "artifacts" / "tables" / "agg_summary.csv"
if not data_path.exists():
    print(f"Error: {data_path} not found. Run the paper preset first:")
    print("  python src/dln_core_variable_cycle.py --preset paper --out outputs/paper")
    sys.exit(1)

df = pd.read_csv(data_path)

# ─────────────────────────────────────────────────────────────────
# 1. Empirical advantage at each K (structured, stakes=0)
# ─────────────────────────────────────────────────────────────────
s0 = df[(df["structure"] == "structured") & (df["stakes"] == 0.0)]

print("=" * 60)
print("EMPIRICAL CROSSOVER ANALYSIS")
print("=" * 60)
print()
print("Network CAU - Linear CAU (structured, stakes=0):")
print(f"{'K':>6s}  {'Network':>10s}  {'Linear':>10s}  {'Δ':>10s}")
print("-" * 42)

K_values = sorted(s0["K"].unique())
advantages = []
for K in K_values:
    net = s0[(s0["agent"] == "Network-Full") & (s0["K"] == K)]["cau_mean"].values
    lin = s0[(s0["agent"] == "Linear") & (s0["K"] == K)]["cau_mean"].values
    if len(net) == 0 or len(lin) == 0:
        continue
    net_val, lin_val = float(net[0]), float(lin[0])
    delta = net_val - lin_val
    advantages.append((K, delta))
    sign = "+" if delta > 0 else ""
    print(f"{K:6d}  {net_val:10.2f}  {lin_val:10.2f}  {sign}{delta:9.2f}")

print()

# ─────────────────────────────────────────────────────────────────
# 2. Empirical crossover via linear interpolation
# ─────────────────────────────────────────────────────────────────
K_cross_empirical = None
for i in range(len(advantages) - 1):
    K1, d1 = advantages[i]
    K2, d2 = advantages[i + 1]
    if d1 <= 0 and d2 > 0:
        # Linear interpolation: find K where delta crosses zero
        K_cross_empirical = K1 + (0 - d1) * (K2 - K1) / (d2 - d1)
        break

if K_cross_empirical is not None:
    print(f"Empirical crossover K* (interpolated): {K_cross_empirical:.1f}")
else:
    # Check if advantage is positive at all K values
    if all(d > 0 for _, d in advantages):
        print(f"Empirical crossover K* < {advantages[0][0]} (Network wins at all tested K)")
        K_cross_empirical = advantages[0][0]
    elif all(d <= 0 for _, d in advantages):
        print(f"Empirical crossover K* > {advantages[-1][0]} (Network never wins at tested K)")
        K_cross_empirical = float("inf")

# ─────────────────────────────────────────────────────────────────
# 3. Analytic crossover: K* = F + c_meta / c_param
# ─────────────────────────────────────────────────────────────────
# From CogCostWeights in the paper preset:
F = 5
c_param = 0.01   # mem cost per option-level parameter
c_meta = 25.0    # switch cost (expansion overhead)
K_cross_analytic = F + c_meta / c_param

print(f"Analytic crossover K* = F + c_meta/c_param = {F} + {c_meta}/{c_param} = {K_cross_analytic:.0f}")
print()

# ─────────────────────────────────────────────────────────────────
# 4. Gap analysis
# ─────────────────────────────────────────────────────────────────
if K_cross_empirical is not None and K_cross_empirical < float("inf"):
    ratio = K_cross_analytic / K_cross_empirical
    print("=" * 60)
    print("GAP ANALYSIS")
    print("=" * 60)
    print()
    print(f"  Analytic K*:  {K_cross_analytic:,.0f}")
    print(f"  Empirical K*: {K_cross_empirical:,.1f}")
    print(f"  Ratio:        {ratio:,.1f}x")
    print()
    print("Interpretation:")
    print("  The analytic formula only accounts for COST savings")
    print("  (O(F) memory vs O(K) memory). The empirical crossover")
    print(f"  occurs ~{ratio:.0f}x earlier because factor pooling also")
    print("  improves PREDICTION QUALITY — a statistical shrinkage")
    print("  effect analogous to James-Stein estimation.")
    print()
    print("  The compression advantage has two sources:")
    print("    1. Computational: fewer parameters → lower cognitive cost")
    print("    2. Statistical:   factor pooling → better reward estimates")
    print("  The analytic K* captures only (1). The simulation reveals")
    print("  that (2) dominates, making the advantage emerge much earlier.")
    print()

    # ─────────────────────────────────────────────────────────────
    # 5. Decompose: utility vs cost contributions to the advantage
    # ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ADVANTAGE DECOMPOSITION (utility vs cost)")
    print("=" * 60)
    print()
    print(f"{'K':>6s}  {'Δ_utility':>12s}  {'Δ_cost':>12s}  {'Δ_CAU':>12s}  {'%_from_util':>12s}")
    print("-" * 60)
    for K in K_values:
        net = s0[(s0["agent"] == "Network-Full") & (s0["K"] == K)]
        lin = s0[(s0["agent"] == "Linear") & (s0["K"] == K)]
        if len(net) == 0 or len(lin) == 0:
            continue
        # CAU = utility - cog_cost, so Δ_CAU = Δ_utility - Δ_cost
        # where Δ = Network - Linear (positive means Network is better)
        du = float(net["utility_mean"].values[0]) - float(lin["utility_mean"].values[0])
        dc = float(net["cog_mean"].values[0]) - float(lin["cog_mean"].values[0])
        dcau = du - dc  # same as net_cau - lin_cau
        pct_util = 100.0 * du / dcau if abs(dcau) > 0.01 else float("nan")
        print(f"{K:6d}  {du:12.2f}  {dc:12.2f}  {dcau:12.2f}  {pct_util:11.1f}%")

    print()
    print("Δ_utility > 0 means Network gets better REWARDS (statistical advantage).")
    print("Δ_cost < 0 means Network has lower COGNITIVE COST (computational advantage).")
    print("When %_from_util >> 100%, the utility gain exceeds the total CAU advantage,")
    print("meaning the cost savings are actually negative (Network pays more due to")
    print("the learning cycle overhead) but the reward improvement more than compensates.")

print()
print("Done.")
