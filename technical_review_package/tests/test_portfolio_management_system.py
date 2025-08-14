"""
Portfolio Management System Test

Test Fractional Kelly sizing, correlation management, and
stress testing for correlation shocks with drawdown limits.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.portfolio.kelly_sizing import (
    FractionalKellySizer,
    KellyParameters,
    KellySizingResult,
    KellyMode,
)
from cryptosmarttrader.portfolio.correlation_manager import (
    CorrelationManager,
    CorrelationShockScenario,
    ClusterType,
    CorrelationLevel,
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PortfolioManagementTester:
    """
    Comprehensive test suite for portfolio management systems
    """

    def __init__(self):
        self.test_results = {}
        self.kelly_sizer = FractionalKellySizer()
        self.correlation_manager = CorrelationManager()

    def test_fractional_kelly_sizing(self) -> bool:
        """Test Fractional Kelly sizing calculations"""

        logger.info("Testing Fractional Kelly sizing system...")

        try:
            # Test case 1: Standard Kelly parameters
            base_params = KellyParameters(
                win_rate=0.6,
                avg_win=0.025,
                avg_loss=0.015,
                confidence=0.75,
                volatility=0.02,
                kelly_fraction=0.25,
                target_volatility=0.15,
                current_portfolio_vol=0.12,
            )

            # Calculate Kelly sizing
            result = self.kelly_sizer.calculate_fractional_kelly(base_params)

            # Validate results
            if not result.is_valid:
                logger.error("Kelly sizing result is invalid")
                return False

            logger.info(f"Kelly Results:")
            logger.info(f"  Full Kelly: {result.full_kelly_size:.3f}")
            logger.info(f"  Fractional Kelly: {result.fractional_kelly_size:.3f}")
            logger.info(f"  Final Size: {result.final_size:.3f}")
            logger.info(f"  Confidence Adj: {result.confidence_adjustment:.2f}")
            logger.info(f"  Vol Ratio: {result.volatility_ratio:.2f}")
            logger.info(f"  Max DD Estimate: {result.max_dd_estimate:.1%}")

            # Test case 2: Low confidence scenario
            low_conf_params = KellyParameters(
                win_rate=0.55,
                avg_win=0.02,
                avg_loss=0.018,
                confidence=0.45,  # Low confidence
                volatility=0.03,
                kelly_fraction=0.25,
            )

            low_conf_result = self.kelly_sizer.calculate_fractional_kelly(low_conf_params)

            # Should apply low confidence penalty
            if low_conf_result.confidence_adjustment >= 1.0:
                logger.error("Low confidence penalty not applied")
                return False

            # Test case 3: High volatility scenario
            high_vol_params = KellyParameters(
                win_rate=0.65,
                avg_win=0.03,
                avg_loss=0.02,
                confidence=0.8,
                volatility=0.05,  # High volatility
                kelly_fraction=0.25,
                target_volatility=0.15,
            )

            high_vol_result = self.kelly_sizer.calculate_fractional_kelly(high_vol_params)

            # Test expectancy calculation
            expectancy = base_params.expectancy
            if expectancy <= 0:
                logger.error("Expectancy should be positive for profitable strategy")
                return False

            logger.info(f"Strategy expectancy: {expectancy:.3f}")
            logger.info(f"Profit factor: {base_params.profit_factor:.2f}")

            # Test constraints
            constraints_applied = (
                len(result.applied_caps) > 0 or len(high_vol_result.applied_caps) > 0
            )

            self.test_results["kelly_sizing"] = {
                "success": True,
                "full_kelly": result.full_kelly_size,
                "final_size": result.final_size,
                "expectancy": expectancy,
                "profit_factor": base_params.profit_factor,
                "constraints_applied": constraints_applied,
                "low_conf_penalty": low_conf_result.confidence_adjustment < 1.0,
                "volatility_adjustment": high_vol_result.volatility_ratio != 1.0,
            }

            return True

        except Exception as e:
            logger.error(f"Kelly sizing test failed: {e}")
            return False

    def test_volatility_targeting(self) -> bool:
        """Test volatility targeting functionality"""

        logger.info("Testing volatility targeting...")

        try:
            # Different volatility scenarios
            scenarios = [
                ("Low Vol Asset", 0.01, 0.12, 0.15),  # asset_vol, portfolio_vol, target_vol
                ("Medium Vol Asset", 0.02, 0.12, 0.15),
                ("High Vol Asset", 0.04, 0.12, 0.15),
                ("Very High Vol", 0.08, 0.12, 0.15),
            ]

            results = []

            for scenario_name, asset_vol, portfolio_vol, target_vol in scenarios:
                # Test volatility adjustment
                base_size = 0.03
                adjusted_size, vol_ratio = self.kelly_sizer.apply_volatility_targeting(
                    base_size, asset_vol, portfolio_vol
                )

                vol_target_ratio = target_vol / portfolio_vol
                asset_vol_ratio = target_vol / asset_vol
                expected_adjustment = vol_target_ratio * asset_vol_ratio

                results.append(
                    {
                        "scenario": scenario_name,
                        "asset_vol": asset_vol,
                        "base_size": base_size,
                        "adjusted_size": adjusted_size,
                        "vol_ratio": vol_ratio,
                        "expected_adjustment": expected_adjustment,
                    }
                )

                logger.info(
                    f"{scenario_name}: {base_size:.3f} → {adjusted_size:.3f} (ratio: {vol_ratio:.2f})"
                )

            # Validate that high volatility reduces position size
            low_vol_size = results[0]["adjusted_size"]
            high_vol_size = results[2]["adjusted_size"]

            vol_targeting_works = high_vol_size < low_vol_size

            if not vol_targeting_works:
                logger.error("Volatility targeting not working: high vol should have smaller size")
                return False

            self.test_results["volatility_targeting"] = {
                "success": True,
                "scenarios": results,
                "vol_targeting_works": vol_targeting_works,
            }

            return True

        except Exception as e:
            logger.error(f"Volatility targeting test failed: {e}")
            return False

    def test_correlation_management(self) -> bool:
        """Test correlation matrix calculation and cluster management"""

        logger.info("Testing correlation management...")

        try:
            # Generate synthetic price data
            np.random.seed(42)
            dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")

            # Create correlated price series
            base_returns = np.random.normal(0, 0.02, 100)

            price_data = {}
            correlations = {
                "BTC/USD": 1.0,
                "ETH/USD": 0.8,  # High correlation with BTC
                "SOL/USD": 0.6,  # Moderate correlation
                "AVAX/USD": 0.7,  # High correlation
                "DOGE/USD": 0.3,  # Low correlation
                "LINK/USD": 0.4,  # Low-moderate correlation
            }

            for symbol, target_corr in correlations.items():
                # Generate correlated returns
                noise = np.random.normal(0, 0.015, 100)
                returns = target_corr * base_returns + np.sqrt(1 - target_corr**2) * noise

                # Convert to price series
                prices = [50000.0]  # Starting price
                for ret in returns:
                    prices.append(prices[-1] * (1 + ret))

                # Create DataFrame
                df = pd.DataFrame(
                    {
                        "date": dates,
                        "close": prices[1:],  # Remove first price as it's the starting point
                        "volume": np.random.lognormal(10, 0.5, 100),
                    }
                )

                price_data[symbol] = df

            # Calculate correlation matrix
            correlation_matrix = self.correlation_manager.calculate_correlation_matrix(price_data)

            if correlation_matrix.empty:
                logger.error("Correlation matrix calculation failed")
                return False

            logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
            logger.info("Sample correlations:")
            for i in range(min(3, len(correlation_matrix))):
                for j in range(i + 1, min(3, len(correlation_matrix))):
                    symbol1 = correlation_matrix.index[i]
                    symbol2 = correlation_matrix.index[j]
                    corr = correlation_matrix.iloc[i, j]
                    logger.info(f"  {symbol1} - {symbol2}: {corr:.3f}")

            # Test cluster identification
            clusters = self.correlation_manager.identify_correlation_clusters(
                correlation_matrix, cluster_threshold=0.6
            )

            logger.info(f"Identified {len(clusters)} correlation clusters:")
            for cluster_id, assets in clusters.items():
                logger.info(f"  Cluster {cluster_id}: {assets}")

            # Setup asset clusters
            asset_cluster_mapping = {
                "BTC/USD": "large_cap",
                "ETH/USD": "large_cap",
                "SOL/USD": "layer1",
                "AVAX/USD": "layer1",
                "DOGE/USD": "meme",
                "LINK/USD": "mid_cap",
            }

            self.correlation_manager.update_asset_clusters(asset_cluster_mapping)

            self.test_results["correlation_management"] = {
                "success": True,
                "correlation_matrix_size": correlation_matrix.shape[0],
                "clusters_identified": len(clusters),
                "asset_mapping_updated": len(self.correlation_manager.asset_cluster_mapping) > 0,
            }

            return True

        except Exception as e:
            logger.error(f"Correlation management test failed: {e}")
            return False

    def test_cluster_limits(self) -> bool:
        """Test cluster limit checking and enforcement"""

        logger.info("Testing cluster limit enforcement...")

        try:
            # Setup test portfolio
            test_positions = {
                "BTC/USD": 0.08,  # Large cap
                "ETH/USD": 0.12,  # Large cap - should trigger violation (0.08 + 0.12 = 0.20 > 0.15)
                "SOL/USD": 0.05,  # Layer 1
                "AVAX/USD": 0.06,  # Layer 1
                "DOGE/USD": 0.04,  # Meme - should be OK (under 0.03 limit but warning)
                "LINK/USD": 0.03,  # Mid cap
            }

            # Check cluster limits
            cluster_check = self.correlation_manager.check_cluster_limits(test_positions)

            if cluster_check["status"] != "checked":
                logger.error("Cluster limit check failed to run")
                return False

            violations = cluster_check["violations"]
            warnings = cluster_check["warnings"]
            cluster_summary = cluster_check["cluster_summary"]

            logger.info("Cluster limit check results:")
            logger.info(f"  Violations: {len(violations)}")
            logger.info(f"  Warnings: {len(warnings)}")

            # Print cluster summary
            for cluster, info in cluster_summary.items():
                logger.info(
                    f"  {cluster}: {info['weight']:.3f}/{info['max_weight']:.3f} weight, "
                    f"{info['positions']}/{info['max_positions']} positions"
                )

            # Verify large cap violation (should exceed 15% limit)
            large_cap_weight = cluster_summary["large_cap"]["weight"]
            large_cap_violation = any(
                v["type"] == "cluster_weight_violation" and v["cluster"] == "large_cap"
                for v in violations
            )

            if large_cap_weight > 0.15 and not large_cap_violation:
                logger.error("Large cap limit violation not detected")
                return False

            # Verify meme coin warning (should exceed recommended allocation)
            meme_weight = cluster_summary["meme"]["weight"]
            meme_over_limit = meme_weight > 0.03

            logger.info(f"Large cap exposure: {large_cap_weight:.1%} (limit: 15%)")
            logger.info(f"Meme exposure: {meme_weight:.1%} (limit: 3%)")

            self.test_results["cluster_limits"] = {
                "success": True,
                "violations_detected": len(violations) > 0,
                "large_cap_violation": large_cap_violation,
                "meme_over_limit": meme_over_limit,
                "cluster_summary": cluster_summary,
            }

            return True

        except Exception as e:
            logger.error(f"Cluster limits test failed: {e}")
            return False

    def test_correlation_shock_stress(self) -> bool:
        """Test correlation shock stress testing"""

        logger.info("Testing correlation shock stress scenarios...")

        try:
            # First ensure we have correlation data
            if self.correlation_manager.correlation_matrix is None:
                logger.error("No correlation matrix available for stress testing")
                return False

            # Test portfolio
            test_positions = {
                "BTC/USD": 0.05,
                "ETH/USD": 0.04,
                "SOL/USD": 0.03,
                "AVAX/USD": 0.03,
                "DOGE/USD": 0.02,
                "LINK/USD": 0.02,
            }

            # Historical volatility estimates
            historical_volatility = {
                "BTC/USD": 0.04,
                "ETH/USD": 0.05,
                "SOL/USD": 0.06,
                "AVAX/USD": 0.07,
                "DOGE/USD": 0.08,
                "LINK/USD": 0.05,
            }

            # Define stress scenarios
            stress_scenarios = [
                CorrelationShockScenario(
                    name="Moderate Correlation Shock",
                    correlation_increase=0.2,
                    volatility_spike=1.5,
                    duration_days=3,
                ),
                CorrelationShockScenario(
                    name="Severe Correlation Shock",
                    correlation_increase=0.4,
                    volatility_spike=2.5,
                    duration_days=7,
                ),
                CorrelationShockScenario(
                    name="Extreme Market Crisis",
                    correlation_increase=0.6,
                    volatility_spike=3.0,
                    duration_days=14,
                ),
            ]

            # Run stress tests
            stress_results = self.correlation_manager.stress_test_correlation_shock(
                test_positions, stress_scenarios, historical_volatility
            )

            if not stress_results:
                logger.error("Stress testing failed to produce results")
                return False

            logger.info("Correlation shock stress test results:")

            max_drawdowns = []
            policy_breaches = 0

            for scenario_name, result in stress_results.items():
                logger.info(f"\n{scenario_name}:")
                logger.info(f"  Max Drawdown: {result.max_drawdown:.1%}")
                logger.info(f"  Portfolio VaR: {result.portfolio_var:.1%}")
                logger.info(
                    f"  Avg Correlation: {result.correlation_metrics['average_correlation']:.3f}"
                )
                logger.info(f"  Risk Breaches: {len(result.risk_limit_breaches)}")

                max_drawdowns.append(result.max_drawdown)

                # Check policy breach (drawdown > 10%)
                if result.max_drawdown > 0.10:
                    policy_breaches += 1

                # Print recommendations
                if result.recommended_actions:
                    logger.info(f"  Recommendations: {result.recommended_actions[0]}")

            # Validate stress test effectiveness
            avg_max_drawdown = np.mean(max_drawdowns)
            stress_escalation = max_drawdowns[-1] > max_drawdowns[0]  # Extreme > Moderate

            logger.info(f"\nStress Test Summary:")
            logger.info(f"  Average Max Drawdown: {avg_max_drawdown:.1%}")
            logger.info(f"  Policy Breaches (>10% DD): {policy_breaches}/{len(stress_scenarios)}")
            logger.info(f"  Stress Escalation: {stress_escalation}")

            # Test passes if:
            # 1. At least one scenario shows policy breach (demonstrates sensitivity)
            # 2. Extreme scenario has higher drawdown than moderate
            # 3. Average drawdown is reasonable (not unrealistically high/low)

            test_success = (
                0 < policy_breaches <= len(stress_scenarios)
                and stress_escalation
                and 0.05 <= avg_max_drawdown <= 0.30
            )

            self.test_results["correlation_stress"] = {
                "success": test_success,
                "scenarios_tested": len(stress_scenarios),
                "avg_max_drawdown": avg_max_drawdown,
                "policy_breaches": policy_breaches,
                "stress_escalation": stress_escalation,
                "max_drawdowns": max_drawdowns,
            }

            return test_success

        except Exception as e:
            logger.error(f"Correlation shock stress test failed: {e}")
            return False

    def test_integrated_portfolio_system(self) -> bool:
        """Test integrated portfolio management workflow"""

        logger.info("Testing integrated portfolio management system...")

        try:
            # Scenario: New position sizing decision
            kelly_params = KellyParameters(
                win_rate=0.62,
                avg_win=0.028,
                avg_loss=0.017,
                confidence=0.8,
                volatility=0.025,
                kelly_fraction=0.3,
            )

            # Calculate Kelly size
            kelly_result = self.kelly_sizer.calculate_fractional_kelly(kelly_params)

            # Current portfolio state
            current_positions = {
                "BTC/USD": 0.04,
                "ETH/USD": 0.03,
                "SOL/USD": 0.025,
                "LINK/USD": 0.02,
            }

            new_symbol = "AVAX/USD"
            new_position_size = kelly_result.final_size

            # Check if new position would violate correlation limits
            updated_positions = current_positions.copy()
            updated_positions[new_symbol] = new_position_size

            # Check cluster limits
            cluster_check = self.correlation_manager.check_cluster_limits(updated_positions)

            # Check correlation limits (if correlation matrix available)
            correlation_check = {"status": "no_data"}
            if self.correlation_manager.correlation_matrix is not None:
                correlation_check = self.correlation_manager.check_correlation_limits(
                    updated_positions, self.correlation_manager.correlation_matrix
                )

            # Portfolio risk metrics
            total_crypto_exposure = sum(updated_positions.values())

            # Risk assessment
            risk_flags = []

            if kelly_result.max_dd_estimate > 0.08:
                risk_flags.append("High estimated drawdown")

            if total_crypto_exposure > 0.25:
                risk_flags.append("Total crypto exposure exceeds 25%")

            if len(cluster_check.get("violations", [])) > 0:
                risk_flags.append("Cluster limit violations")

            if len(correlation_check.get("violations", [])) > 0:
                risk_flags.append("Correlation limit violations")

            # Decision logic
            position_approved = (
                kelly_result.is_valid and new_position_size > 0.001 and len(risk_flags) == 0
            )

            logger.info("Integrated Portfolio Management Decision:")
            logger.info(f"  Proposed Position: {new_symbol} at {new_position_size:.3f}")
            logger.info(f"  Kelly Size: {kelly_result.final_size:.3f}")
            logger.info(f"  Total Crypto Exposure: {total_crypto_exposure:.1%}")
            logger.info(f"  Risk Flags: {len(risk_flags)}")
            logger.info(f"  Position Approved: {position_approved}")

            if risk_flags:
                for flag in risk_flags:
                    logger.info(f"    - {flag}")

            self.test_results["integrated_system"] = {
                "success": True,
                "kelly_size": kelly_result.final_size,
                "total_exposure": total_crypto_exposure,
                "risk_flags": len(risk_flags),
                "position_approved": position_approved,
                "cluster_violations": len(cluster_check.get("violations", [])),
                "correlation_violations": len(correlation_check.get("violations", [])),
            }

            return True

        except Exception as e:
            logger.error(f"Integrated system test failed: {e}")
            return False

    def run_comprehensive_tests(self):
        """Run all portfolio management tests"""

        logger.info("=" * 60)
        logger.info("🧪 PORTFOLIO MANAGEMENT SYSTEM TESTS")
        logger.info("=" * 60)

        tests = [
            ("Fractional Kelly Sizing", self.test_fractional_kelly_sizing),
            ("Volatility Targeting", self.test_volatility_targeting),
            ("Correlation Management", self.test_correlation_management),
            ("Cluster Limits", self.test_cluster_limits),
            ("Correlation Shock Stress", self.test_correlation_shock_stress),
            ("Integrated Portfolio System", self.test_integrated_portfolio_system),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n📋 {test_name}")
            try:
                success = test_func()
                if success:
                    logger.info(f"✅ {test_name} - PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} failed with exception: {e}")

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("🏁 PORTFOLIO MANAGEMENT TEST RESULTS")
        logger.info("=" * 60)

        for test_name, _ in tests:
            test_key = test_name.lower().replace(" ", "_")
            result = (
                "✅ PASSED"
                if self.test_results.get(test_key, {}).get("success", False)
                else "❌ FAILED"
            )
            logger.info(f"{test_name:<35} {result}")

        logger.info("=" * 60)
        logger.info(
            f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests / total_tests * 100:.1f}%)"
        )

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - PORTFOLIO SYSTEM READY")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

        # Key metrics summary
        logger.info("\n📊 KEY PORTFOLIO METRICS:")

        if "kelly_sizing" in self.test_results:
            kelly = self.test_results["kelly_sizing"]
            logger.info(f"• Kelly Full Size: {kelly.get('full_kelly', 0):.3f}")
            logger.info(f"• Kelly Final Size: {kelly.get('final_size', 0):.3f}")
            logger.info(f"• Strategy Expectancy: {kelly.get('expectancy', 0):.3f}")
            logger.info(f"• Profit Factor: {kelly.get('profit_factor', 0):.2f}")

        if "correlation_stress" in self.test_results:
            stress = self.test_results["correlation_stress"]
            logger.info(f"• Avg Max Drawdown: {stress.get('avg_max_drawdown', 0):.1%}")
            logger.info(
                f"• Policy Breaches: {stress.get('policy_breaches', 0)}/{stress.get('scenarios_tested', 0)}"
            )

        if "cluster_limits" in self.test_results:
            cluster = self.test_results["cluster_limits"]
            logger.info(
                f"• Cluster Violations Detected: {cluster.get('violations_detected', False)}"
            )

        return passed_tests == total_tests


def main():
    """Run portfolio management tests"""

    tester = PortfolioManagementTester()
    success = tester.run_comprehensive_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
