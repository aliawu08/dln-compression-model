#!/usr/bin/env python3
"""
Verify paper claims against computed artifacts.

This script checks that all claims in paper/claims.yaml match the actual
values in outputs/paper/artifacts/tables/agg_summary.csv within tolerance.

Usage:
    python scripts/verify_paper_claims.py [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class ClaimResult:
    name: str
    description: str
    passed: bool
    expected: float | None
    actual: float | None
    tolerance: float | None
    message: str


def load_claims(claims_path: Path) -> dict:
    """Load claims from YAML file."""
    with open(claims_path) as f:
        return yaml.safe_load(f)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load aggregated summary data."""
    return pd.read_csv(data_path)


def filter_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to dataframe."""
    mask = pd.Series([True] * len(df))
    for col, val in filters.items():
        mask &= df[col] == val
    return df[mask]


def verify_value_claim(df: pd.DataFrame, claim: dict, name: str) -> ClaimResult:
    """Verify a claim about a specific value."""
    filtered = filter_data(df, claim['filters'])

    if len(filtered) == 0:
        return ClaimResult(
            name=name,
            description=claim.get('description', ''),
            passed=False,
            expected=claim.get('expected'),
            actual=None,
            tolerance=claim.get('tolerance'),
            message="No data matches filters"
        )

    actual = filtered[claim['metric']].values[0]

    # Check direction constraint
    if 'direction' in claim:
        threshold = claim.get('threshold', 0)
        if claim['direction'] == 'gt':
            passed = actual > threshold
            message = f"actual={actual:.2f} {'>' if passed else '<='} threshold={threshold}"
        elif claim['direction'] == 'lt':
            passed = actual < threshold
            message = f"actual={actual:.2f} {'<' if passed else '>='} threshold={threshold}"
        else:
            passed = False
            message = f"Unknown direction: {claim['direction']}"

        return ClaimResult(
            name=name,
            description=claim.get('description', ''),
            passed=passed,
            expected=threshold,
            actual=actual,
            tolerance=None,
            message=message
        )

    # Check expected value within tolerance
    expected = claim['expected']
    tolerance = claim.get('tolerance', 0.0)
    diff = abs(actual - expected)
    passed = diff <= tolerance

    return ClaimResult(
        name=name,
        description=claim.get('description', ''),
        passed=passed,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        message=f"actual={actual:.2f}, expected={expected:.2f}, diff={diff:.2f}, tol={tolerance}"
    )


def verify_comparison_claim(df: pd.DataFrame, claim: dict, name: str) -> ClaimResult:
    """Verify a comparison between two agents."""
    filters = claim['filters'].copy()

    # Get agent A value
    filters_a = {**filters, 'agent': claim['agent_a']}
    filtered_a = filter_data(df, filters_a)
    if len(filtered_a) == 0:
        return ClaimResult(name=name, description=claim.get('description', ''),
                          passed=False, expected=None, actual=None, tolerance=None,
                          message=f"No data for agent {claim['agent_a']}")
    val_a = filtered_a[claim['metric']].values[0]

    # Get agent B value
    filters_b = {**filters, 'agent': claim['agent_b']}
    filtered_b = filter_data(df, filters_b)
    if len(filtered_b) == 0:
        return ClaimResult(name=name, description=claim.get('description', ''),
                          passed=False, expected=None, actual=None, tolerance=None,
                          message=f"No data for agent {claim['agent_b']}")
    val_b = filtered_b[claim['metric']].values[0]

    diff = val_a - val_b
    min_diff = claim.get('min_difference', 0)

    if claim['direction'] == 'gt':
        passed = diff >= min_diff
        message = f"{claim['agent_a']}={val_a:.2f}, {claim['agent_b']}={val_b:.2f}, Δ={diff:.2f} (min={min_diff})"
    else:
        passed = False
        message = f"Unknown direction: {claim['direction']}"

    return ClaimResult(
        name=name,
        description=claim.get('description', ''),
        passed=passed,
        expected=min_diff,
        actual=diff,
        tolerance=None,
        message=message
    )


def verify_ratio_claim(df: pd.DataFrame, claim: dict, name: str) -> ClaimResult:
    """Verify a ratio between two K values."""
    filters = claim['filters'].copy()

    # Get numerator value
    filters_num = {**filters, 'K': claim['K_numerator']}
    filtered_num = filter_data(df, filters_num)
    if len(filtered_num) == 0:
        return ClaimResult(name=name, description=claim.get('description', ''),
                          passed=False, expected=None, actual=None, tolerance=None,
                          message=f"No data for K={claim['K_numerator']}")
    val_num = filtered_num[claim['metric']].values[0]

    # Get denominator value
    filters_den = {**filters, 'K': claim['K_denominator']}
    filtered_den = filter_data(df, filters_den)
    if len(filtered_den) == 0:
        return ClaimResult(name=name, description=claim.get('description', ''),
                          passed=False, expected=None, actual=None, tolerance=None,
                          message=f"No data for K={claim['K_denominator']}")
    val_den = filtered_den[claim['metric']].values[0]

    if val_den == 0:
        return ClaimResult(name=name, description=claim.get('description', ''),
                          passed=False, expected=None, actual=None, tolerance=None,
                          message="Denominator is zero")

    ratio = val_num / val_den
    expected = claim['expected_ratio']
    tolerance = claim.get('tolerance_ratio', 0.5)

    passed = abs(ratio - expected) <= tolerance

    return ClaimResult(
        name=name,
        description=claim.get('description', ''),
        passed=passed,
        expected=expected,
        actual=ratio,
        tolerance=tolerance,
        message=f"ratio={ratio:.2f}, expected={expected:.2f}, tol={tolerance}"
    )


def verify_all_claims(claims: dict, df: pd.DataFrame, verbose: bool = False) -> list[ClaimResult]:
    """Verify all claims."""
    results = []

    for name, claim in claims.get('claims', {}).items():
        claim_type = claim.get('type', 'value')

        if claim_type == 'comparison':
            result = verify_comparison_claim(df, claim, name)
        elif claim_type == 'ratio':
            result = verify_ratio_claim(df, claim, name)
        else:
            result = verify_value_claim(df, claim, name)

        results.append(result)

        if verbose:
            status = "✓" if result.passed else "✗"
            print(f"{status} {name}")
            print(f"   {result.description}")
            print(f"   {result.message}")
            print()

    return results


def compute_bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    filters: dict,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric."""
    filtered = filter_data(df, filters)
    if len(filtered) == 0:
        return (np.nan, np.nan, np.nan)

    # Get the raw episode data if available, otherwise use point estimate
    mean_val = filtered[metric].values[0]
    std_val = filtered.get(metric.replace('_mean', '_std'), pd.Series([0])).values[0]

    # Simulate bootstrap from normal approximation
    np.random.seed(42)
    n_seeds = 100  # from manifest
    se = std_val / np.sqrt(n_seeds)
    bootstrap_means = np.random.normal(mean_val, se, n_bootstrap)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (mean_val, lower, upper)


def compute_effect_size(
    df: pd.DataFrame,
    metric: str,
    filters: dict,
    agent_a: str,
    agent_b: str
) -> float:
    """Compute Cohen's d effect size between two agents."""
    filters_a = {**filters, 'agent': agent_a}
    filters_b = {**filters, 'agent': agent_b}

    filtered_a = filter_data(df, filters_a)
    filtered_b = filter_data(df, filters_b)

    if len(filtered_a) == 0 or len(filtered_b) == 0:
        return np.nan

    mean_a = filtered_a[metric].values[0]
    mean_b = filtered_b[metric].values[0]
    std_a = filtered_a[metric.replace('_mean', '_std')].values[0]
    std_b = filtered_b[metric.replace('_mean', '_std')].values[0]

    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    if pooled_std == 0:
        return np.nan

    return (mean_a - mean_b) / pooled_std


def print_summary(results: list[ClaimResult]) -> None:
    """Print summary of verification results."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("=" * 60)
    print(f"CLAIMS VERIFICATION SUMMARY: {passed}/{total} passed")
    print("=" * 60)

    if passed < total:
        print("\nFailed claims:")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.message}")


def print_statistical_summary(df: pd.DataFrame, verbose: bool = False) -> None:
    """Print statistical summary with CIs and effect sizes."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY (Key Claims)")
    print("=" * 60)

    # Claim 1: Network vs Linear at K=200
    filters = {'structure': 'structured', 'stakes': 0.0, 'K': 200}

    net_mean, net_lo, net_hi = compute_bootstrap_ci(df, 'cau_mean', {**filters, 'agent': 'Network-Full'})
    lin_mean, lin_lo, lin_hi = compute_bootstrap_ci(df, 'cau_mean', {**filters, 'agent': 'Linear'})
    cohens_d = compute_effect_size(df, 'cau_mean', filters, 'Network-Full', 'Linear')

    print(f"\n1. Core Compression (K=200, structured, stakes=0)")
    print(f"   Network CAU: {net_mean:.2f} [95% CI: {net_lo:.2f}, {net_hi:.2f}]")
    print(f"   Linear CAU:  {lin_mean:.2f} [95% CI: {lin_lo:.2f}, {lin_hi:.2f}]")
    print(f"   Δ = {net_mean - lin_mean:.2f}")
    print(f"   Cohen's d = {cohens_d:.2f}")

    # Claim 2: Stakes comparison
    filters_stakes = {'structure': 'structured', 'stakes': 1.0, 'K': 200}

    net_util, net_lo, net_hi = compute_bootstrap_ci(df, 'utility_mean', {**filters_stakes, 'agent': 'Network-Full'})
    lin_util, lin_lo, lin_hi = compute_bootstrap_ci(df, 'utility_mean', {**filters_stakes, 'agent': 'Linear'})

    print(f"\n2. Variable Compression (K=200, structured, stakes=1)")
    print(f"   Network utility: {net_util:.2f} [95% CI: {net_lo:.2f}, {net_hi:.2f}]")
    print(f"   Linear utility:  {lin_util:.2f} [95% CI: {lin_lo:.2f}, {lin_hi:.2f}]")
    print(f"   Δ = {net_util - lin_util:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Verify paper claims")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--claims', default='paper/claims.yaml', help='Path to claims file')
    parser.add_argument('--data', default='outputs/paper/artifacts/tables/agg_summary.csv',
                       help='Path to data file')
    parser.add_argument('--stats', action='store_true', help='Print statistical summary')
    args = parser.parse_args()

    claims_path = Path(args.claims)
    data_path = Path(args.data)

    if not claims_path.exists():
        print(f"Error: Claims file not found: {claims_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    claims = load_claims(claims_path)
    df = load_data(data_path)

    print("=" * 60)
    print("PAPER CLAIMS VERIFICATION")
    print("=" * 60)
    print(f"Claims file: {claims_path}")
    print(f"Data file: {data_path}")
    print()

    results = verify_all_claims(claims, df, verbose=args.verbose)
    print_summary(results)

    if args.stats:
        print_statistical_summary(df, verbose=args.verbose)

    # Exit with error if any claims failed
    failed = sum(1 for r in results if not r.passed)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
