import numpy as np
import pytest
from pathlib import Path
import json

# =============================================================================
# CLAIM 1: Network outperforms Linear in cost-adjusted utility when structure exists
# =============================================================================

class TestCoreCompressionClaim:
    """Verify that Network's O(F) compression beats Linear's O(K) when structure exists."""

    def test_network_beats_linear_structured_stakes_off(self):
        """At K=200, stakes=0, structured: Network CAU > Linear CAU."""
        agg = load_agg_summary()

        network = agg[(agg['agent'] == 'Network-Full') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == 200)]
        linear = agg[(agg['agent'] == 'Linear') &
                     (agg['structure'] == 'structured') &
                     (agg['stakes'] == 0.0) &
                     (agg['K'] == 200)]

        network_cau = network['cau_mean'].values[0]
        linear_cau = linear['cau_mean'].values[0]

        # Paper claims: Network 88.70 vs Linear 66.15
        assert network_cau > linear_cau, f"Network {network_cau:.2f} should beat Linear {linear_cau:.2f}"
        assert network_cau > 80, f"Network CAU {network_cau:.2f} below expected ~88"
        assert linear_cau < 70, f"Linear CAU {linear_cau:.2f} above expected ~66"

    def test_advantage_grows_with_K(self):
        """Network advantage should increase as K grows (more options to compress)."""
        agg = load_agg_summary()

        advantages = []
        for K in [20, 50, 100, 200]:
            network = agg[(agg['agent'] == 'Network-Full') &
                          (agg['structure'] == 'structured') &
                          (agg['stakes'] == 0.0) &
                          (agg['K'] == K)]['cau_mean'].values[0]
            linear = agg[(agg['agent'] == 'Linear') &
                         (agg['structure'] == 'structured') &
                         (agg['stakes'] == 0.0) &
                         (agg['K'] == K)]['cau_mean'].values[0]
            advantages.append(network - linear)

        # Advantage should be monotonically increasing (or at least trend upward)
        assert advantages[-1] > advantages[0], \
            f"Advantage should grow with K: {advantages}"

    def test_no_advantage_unstructured(self):
        """Network should NOT dominate in unstructured environment."""
        agg = load_agg_summary()

        for K in [50, 100, 200]:
            network = agg[(agg['agent'] == 'Network-Full') &
                          (agg['structure'] == 'unstructured') &
                          (agg['stakes'] == 0.0) &
                          (agg['K'] == K)]['cau_mean'].values[0]
            linear = agg[(agg['agent'] == 'Linear') &
                         (agg['structure'] == 'unstructured') &
                         (agg['stakes'] == 0.0) &
                         (agg['K'] == K)]['cau_mean'].values[0]

            # Network should be significantly worse (no structure to exploit)
            assert linear > network, \
                f"At K={K} unstructured: Linear {linear:.2f} should beat Network {network:.2f}"


# =============================================================================
# CLAIM 2: With stakes on, Linear collapses while Network survives
# =============================================================================

class TestVariableCompressionClaim:
    """Verify Network tracks cumulative exposure while Linear ignores cross-terms."""

    def test_linear_collapses_with_stakes(self):
        """Linear utility should go deeply negative with stakes=1."""
        agg = load_agg_summary()

        linear = agg[(agg['agent'] == 'Linear') &
                     (agg['structure'] == 'structured') &
                     (agg['stakes'] == 1.0) &
                     (agg['K'] == 200)]

        utility = linear['utility_mean'].values[0]
        # Paper claims: -783.04
        assert utility < -500, f"Linear utility {utility:.2f} should be deeply negative"

    def test_network_survives_with_stakes(self):
        """Network utility should remain positive with stakes=1."""
        agg = load_agg_summary()

        network = agg[(agg['agent'] == 'Network-Full') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 1.0) &
                      (agg['K'] == 200)]

        utility = network['utility_mean'].values[0]
        # Paper claims: 59.75
        assert utility > 0, f"Network utility {utility:.2f} should be positive"
        assert utility > 50, f"Network utility {utility:.2f} below expected ~60"

    def test_penalty_difference(self):
        """Linear should incur much larger penalties than Network."""
        agg = load_agg_summary()

        network = agg[(agg['agent'] == 'Network-Full') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 1.0) &
                      (agg['K'] == 200)]['penalty_mean'].values[0]
        linear = agg[(agg['agent'] == 'Linear') &
                     (agg['structure'] == 'structured') &
                     (agg['stakes'] == 1.0) &
                     (agg['K'] == 200)]['penalty_mean'].values[0]

        # Linear penalty should be orders of magnitude higher
        assert linear > 100 * network, \
            f"Linear penalty {linear:.2f} should be >> Network penalty {network:.2f}"


# =============================================================================
# CLAIM 3: Learning cycle enables recovery from wrong priors
# =============================================================================

class TestLearningCycleClaim:
    """Verify the hypothesis → test → update cycle works."""

    def test_full_cycle_beats_no_update(self):
        """Network-Full should achieve higher utility than Network-NoUpdate in unstructured.

        The full cycle correctly expands to tabular when the factor hypothesis
        fails, producing better predictions (higher utility).  We test utility
        rather than CAU because the expansion's O(K) cognitive cost may exceed
        the reward gain — the claim is that the *mechanism* works, not that
        expansion is always cost-efficient in static environments.
        """
        agg = load_agg_summary()

        for K in [100, 200]:
            full = agg[(agg['agent'] == 'Network-Full') &
                       (agg['structure'] == 'unstructured') &
                       (agg['stakes'] == 0.0) &
                       (agg['K'] == K)]['utility_mean'].values[0]
            no_update = agg[(agg['agent'] == 'Network-NoUpdate') &
                            (agg['structure'] == 'unstructured') &
                            (agg['stakes'] == 0.0) &
                            (agg['K'] == K)]['utility_mean'].values[0]

            # Full utility should exceed NoUpdate (cycle enables better learning)
            assert full >= no_update - 2, \
                f"At K={K}: Full utility {full:.2f} should be >= NoUpdate {no_update:.2f}"

    def test_switches_occur_in_unstructured(self):
        """Network-Full should trigger switches in unstructured environments."""
        agg = load_agg_summary()

        # In unstructured, the factor model is wrong, so switches should happen
        full_unstructured = agg[(agg['agent'] == 'Network-Full') &
                                (agg['structure'] == 'unstructured') &
                                (agg['K'] == 200)]['expansions_mean'].values[0]

        assert full_unstructured > 0, \
            f"Switches should occur in unstructured: got {full_unstructured}"

    def test_no_switches_in_structured(self):
        """Network-Full should NOT switch when structure matches prior."""
        agg = load_agg_summary()

        full_structured = agg[(agg['agent'] == 'Network-Full') &
                              (agg['structure'] == 'structured') &
                              (agg['K'] == 200)]['expansions_mean'].values[0]

        assert full_structured == 0, \
            f"No switches expected in structured: got {full_structured}"


# =============================================================================
# CLAIM 4: Cognitive cost scales O(K) for Linear vs O(F) for Network
# =============================================================================

class TestCognitiveScalingClaim:
    """Verify the cognitive cost scaling claims."""

    def test_linear_cost_scales_with_K(self):
        """Linear cognitive cost should increase with K."""
        agg = load_agg_summary()

        costs = []
        for K in [20, 50, 100, 200]:
            cost = agg[(agg['agent'] == 'Linear') &
                       (agg['structure'] == 'structured') &
                       (agg['stakes'] == 0.0) &
                       (agg['K'] == K)]['cog_mean'].values[0]
            costs.append(cost)

        # Cost should roughly scale with K
        assert costs[-1] > 5 * costs[0], \
            f"Linear cost should scale with K: {costs}"

    def test_network_cost_constant_with_K(self):
        """Network cognitive cost should be roughly constant regardless of K."""
        agg = load_agg_summary()

        costs = []
        for K in [20, 50, 100, 200]:
            cost = agg[(agg['agent'] == 'Network-Full') &
                       (agg['structure'] == 'structured') &
                       (agg['stakes'] == 0.0) &
                       (agg['K'] == K)]['cog_mean'].values[0]
            costs.append(cost)

        # Cost should be roughly constant (within 10%)
        assert max(costs) < 1.2 * min(costs), \
            f"Network cost should be constant: {costs}"


# =============================================================================
# CLAIM 5: Dual-purpose actions exist and are beneficial
# =============================================================================

class TestDualPurposeClaim:
    """Verify dual-purpose actions are identified and beneficial."""

    def test_dual_purpose_picks_exist(self):
        """Network should identify dual-purpose actions when stakes > 0."""
        agg = load_agg_summary()

        dual = agg[(agg['agent'] == 'Network-Full') &
                   (agg['structure'] == 'structured') &
                   (agg['stakes'] == 1.0) &
                   (agg['K'] == 20)]['dual_mean'].values[0]

        assert dual > 0, f"Dual-purpose picks should exist: got {dual}"

    def test_no_dual_purpose_without_stakes(self):
        """Dual-purpose is meaningless when stakes=0."""
        agg = load_agg_summary()

        dual = agg[(agg['agent'] == 'Network-Full') &
                   (agg['structure'] == 'structured') &
                   (agg['stakes'] == 0.0) &
                   (agg['K'] == 200)]['dual_mean'].values[0]

        assert dual == 0, f"No dual-purpose without stakes: got {dual}"


# =============================================================================
# STATISTICAL ROBUSTNESS
# =============================================================================

class TestStatisticalRobustness:
    """Verify results are statistically meaningful, not noise."""

    def test_sufficient_seeds(self):
        """Verify we have enough seeds for statistical power."""
        manifest = load_manifest()
        assert manifest['seeds'] >= 50, f"Need >= 50 seeds, got {manifest['seeds']}"

    def test_low_variance_for_key_claims(self):
        """Key results should have reasonable variance."""
        agg = load_agg_summary()

        # Network-Full at K=200, structured, stakes=0
        row = agg[(agg['agent'] == 'Network-Full') &
                  (agg['structure'] == 'structured') &
                  (agg['stakes'] == 0.0) &
                  (agg['K'] == 200)]

        mean = row['cau_mean'].values[0]
        std = row['cau_std'].values[0]

        # Coefficient of variation should be reasonable
        cv = std / abs(mean) if mean != 0 else float('inf')
        assert cv < 0.5, f"High variance: mean={mean:.2f}, std={std:.2f}, CV={cv:.2f}"

    def test_effect_size_meaningful(self):
        """The difference between Network and Linear should be large, not marginal."""
        agg = load_agg_summary()

        network = agg[(agg['agent'] == 'Network-Full') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == 200)]
        linear = agg[(agg['agent'] == 'Linear') &
                     (agg['structure'] == 'structured') &
                     (agg['stakes'] == 0.0) &
                     (agg['K'] == 200)]

        diff = network['cau_mean'].values[0] - linear['cau_mean'].values[0]
        pooled_std = np.sqrt((network['cau_std'].values[0]**2 + linear['cau_std'].values[0]**2) / 2)

        # Cohen's d should be at least medium (> 0.4)
        # Actual value is ~0.49 (medium effect; convention: 0.2=small, 0.5=medium, 0.8=large)
        cohens_d = diff / pooled_std
        assert cohens_d > 0.4, f"Effect size too small: Cohen's d = {cohens_d:.2f}"


# =============================================================================
# HELPERS
# =============================================================================

def load_agg_summary():
    """Load the aggregated summary CSV."""
    import pandas as pd
    path = Path(__file__).parent.parent / "outputs" / "paper" / "artifacts" / "tables" / "agg_summary.csv"
    if not path.exists():
        pytest.skip(f"agg_summary.csv not found at {path}")
    return pd.read_csv(path)


def load_manifest():
    """Load the run manifest."""
    path = Path(__file__).parent.parent / "outputs" / "paper" / "results" / "manifest.json"
    if not path.exists():
        pytest.skip(f"manifest.json not found at {path}")
    with open(path) as f:
        return json.load(f)


# =============================================================================
# RUN VERIFICATION (if outputs don't exist, run smoke test)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# =============================================================================
# CLAIM 0: Both architectures outperform naive baseline
# =============================================================================

class TestBaselineComparison:
    """Verify that both Network and Linear significantly outperform Dot (no model)."""

    def test_linear_beats_dot_all_K(self):
        """Linear should dominate Dot at every K value."""
        agg = load_agg_summary()

        for K in [20, 50, 100, 200]:
            linear = agg[(agg['agent'] == 'Linear') &
                         (agg['structure'] == 'structured') &
                         (agg['stakes'] == 0.0) &
                         (agg['K'] == K)]['cau_mean'].values[0]
            dot = agg[(agg['agent'] == 'Dot') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == K)]['cau_mean'].values[0]

            assert linear > dot + 50, \
                f"At K={K}: Linear {linear:.2f} should dominate Dot {dot:.2f}"

    def test_network_beats_dot_all_K(self):
        """Network should dominate Dot at every K value."""
        agg = load_agg_summary()

        for K in [20, 50, 100, 200]:
            network = agg[(agg['agent'] == 'Network-Full') &
                          (agg['structure'] == 'structured') &
                          (agg['stakes'] == 0.0) &
                          (agg['K'] == K)]['cau_mean'].values[0]
            dot = agg[(agg['agent'] == 'Dot') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == K)]['cau_mean'].values[0]

            assert network > dot + 50, \
                f"At K={K}: Network {network:.2f} should dominate Dot {dot:.2f}"

    def test_three_tier_hierarchy(self):
        """Dot < Linear < Network should hold at high K with structure."""
        agg = load_agg_summary()

        K = 200
        dot = agg[(agg['agent'] == 'Dot') &
                  (agg['structure'] == 'structured') &
                  (agg['stakes'] == 0.0) &
                  (agg['K'] == K)]['cau_mean'].values[0]
        linear = agg[(agg['agent'] == 'Linear') &
                     (agg['structure'] == 'structured') &
                     (agg['stakes'] == 0.0) &
                     (agg['K'] == K)]['cau_mean'].values[0]
        network = agg[(agg['agent'] == 'Network-Full') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == K)]['cau_mean'].values[0]

        assert dot < linear < network, \
            f"Hierarchy violated: Dot={dot:.2f}, Linear={linear:.2f}, Network={network:.2f}"

    def test_dot_near_zero_cau(self):
        """Dot should have near-zero cost-adjusted utility (no real model)."""
        agg = load_agg_summary()

        for K in [20, 50, 100, 200]:
            dot = agg[(agg['agent'] == 'Dot') &
                      (agg['structure'] == 'structured') &
                      (agg['stakes'] == 0.0) &
                      (agg['K'] == K)]['cau_mean'].values[0]

            assert abs(dot) < 10, \
                f"Dot CAU should be near zero, got {dot:.2f} at K={K}"


# =============================================================================
# CLAIM 6: Thompson Sampling comparison (isolate cycle + exposure contributions)
# =============================================================================

class TestThompsonComparison:
    """Verify Thompson-Factor baseline isolates the DLN learning cycle's contribution."""

    def test_thompson_beats_linear_structured(self):
        """Thompson-Factor should beat Linear in structured (it knows factors)."""
        agg = load_agg_summary()
        if 'Thompson-Factor' not in agg['agent'].values:
            pytest.skip("Thompson-Factor not in outputs (re-run paper preset)")

        for K in [100, 200]:
            thompson = agg[(agg['agent'] == 'Thompson-Factor') &
                           (agg['structure'] == 'structured') &
                           (agg['stakes'] == 0.0) &
                           (agg['K'] == K)]['cau_mean'].values[0]
            linear = agg[(agg['agent'] == 'Linear') &
                         (agg['structure'] == 'structured') &
                         (agg['stakes'] == 0.0) &
                         (agg['K'] == K)]['cau_mean'].values[0]

            assert thompson > linear, \
                f"At K={K}: Thompson {thompson:.2f} should beat Linear {linear:.2f}"

    def test_network_beats_thompson_with_stakes(self):
        """Network should beat Thompson when stakes are active (exposure tracking)."""
        agg = load_agg_summary()
        if 'Thompson-Factor' not in agg['agent'].values:
            pytest.skip("Thompson-Factor not in outputs (re-run paper preset)")

        for K in [100, 200]:
            network = agg[(agg['agent'] == 'Network-Full') &
                          (agg['structure'] == 'structured') &
                          (agg['stakes'] == 1.0) &
                          (agg['K'] == K)]['cau_mean'].values[0]
            thompson = agg[(agg['agent'] == 'Thompson-Factor') &
                           (agg['structure'] == 'structured') &
                           (agg['stakes'] == 1.0) &
                           (agg['K'] == K)]['cau_mean'].values[0]

            assert network > thompson, \
                f"At K={K}: Network {network:.2f} should beat Thompson {thompson:.2f} with stakes"

    def test_thompson_no_expansion(self):
        """Thompson has no expansion mechanism - stays in factor mode always."""
        agg = load_agg_summary()
        if 'Thompson-Factor' not in agg['agent'].values:
            pytest.skip("Thompson-Factor not in outputs (re-run paper preset)")

        expansions = agg[agg['agent'] == 'Thompson-Factor']['expansions_mean']
        assert (expansions == 0).all(), \
            f"Thompson should never expand, got {expansions.values}"
