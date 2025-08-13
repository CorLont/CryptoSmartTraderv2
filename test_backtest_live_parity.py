"""
Backtest-Live Parity System Test

Test execution simulator, tracking error monitoring,
and A/B comparison of paper vs live fills over 7 days.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.execution.execution_simulator import (
    ExecutionSimulator,
    OrderSide,
    OrderType,
    FillType,
    ExecutionQuality,
)
from cryptosmarttrader.execution.tracking_error_monitor import (
    TrackingErrorMonitor,
    TrackingErrorComponent,
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BacktestLiveParityTester:
    """
    Comprehensive test suite for backtest-live parity systems
    """

    def __init__(self):
        self.test_results = {}
        self.execution_simulator = ExecutionSimulator(exchange="kraken")
        self.tracking_error_monitor = TrackingErrorMonitor(max_tracking_error_bps=20.0)

    def test_execution_simulator_components(self) -> bool:
        """Test all execution simulator components"""

        logger.info("Testing execution simulator components...")

        try:
            # Test 1: Market order simulation
            market_result = self.execution_simulator.simulate_order_execution(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                size=0.5,
                order_type=OrderType.MARKET,
                volume_24h=50000000,  # $50M daily volume
                market_cap_tier="large_cap",
            )

            market_success = (
                market_result.filled_size > 0
                and len(market_result.fills) > 0
                and market_result.total_fees > 0
                and market_result.slippage_bps >= 0
            )

            logger.info(f"Market order test:")
            logger.info(f"  Filled: {market_result.filled_size}/{market_result.requested_size}")
            logger.info(f"  Avg price: ${market_result.average_price:.2f}")
            logger.info(f"  Slippage: {market_result.slippage_bps:.1f} bps")
            logger.info(f"  Fees: ${market_result.total_fees:.2f}")
            logger.info(f"  Execution time: {market_result.execution_time_ms:.1f}ms")
            logger.info(f"  Quality: {market_result.execution_quality.value}")

            # Test 2: Limit order simulation
            limit_result = self.execution_simulator.simulate_order_execution(
                symbol="ETH/USD",
                side=OrderSide.SELL,
                size=2.0,
                order_type=OrderType.LIMIT,
                limit_price=3000.0,  # Assume current price around 3000
                volume_24h=30000000,
                market_cap_tier="large_cap",
            )

            limit_success = (
                limit_result.order_type == OrderType.LIMIT and limit_result.slippage_bps >= 0
            )

            logger.info(f"Limit order test:")
            logger.info(f"  Filled: {limit_result.filled_size}/{limit_result.requested_size}")
            logger.info(f"  Fill rate: {limit_result.fill_rate:.1%}")
            logger.info(f"  Partial fill: {limit_result.partial_fill}")

            # Test 3: Fee calculation accuracy
            fee_test_notional = 10000  # $10k trade
            maker_fee = self.execution_simulator.fee_calculator.calculate_fee(
                fee_test_notional, "kraken", FillType.MAKER, 0
            )
            taker_fee = self.execution_simulator.fee_calculator.calculate_fee(
                fee_test_notional, "kraken", FillType.TAKER, 0
            )

            # Kraken tier 1: maker 16bps, taker 26bps
            expected_maker_fee = fee_test_notional * 0.0016
            expected_taker_fee = fee_test_notional * 0.0026

            fee_accuracy = (
                abs(maker_fee - expected_maker_fee) < 1.0
                and abs(taker_fee - expected_taker_fee) < 1.0
            )

            logger.info(f"Fee calculation test:")
            logger.info(f"  Maker fee: ${maker_fee:.2f} (expected ${expected_maker_fee:.2f})")
            logger.info(f"  Taker fee: ${taker_fee:.2f} (expected ${expected_taker_fee:.2f})")

            # Test 4: Market impact modeling
            large_order_result = self.execution_simulator.simulate_order_execution(
                symbol="SOL/USD",
                side=OrderSide.BUY,
                size=100.0,  # Large order
                order_type=OrderType.MARKET,
                volume_24h=5000000,  # Lower liquidity
                market_cap_tier="mid_cap",
            )

            small_order_result = self.execution_simulator.simulate_order_execution(
                symbol="SOL/USD",
                side=OrderSide.BUY,
                size=1.0,  # Small order
                order_type=OrderType.MARKET,
                volume_24h=5000000,
                market_cap_tier="mid_cap",
            )

            impact_scaling = (
                large_order_result.market_impact_bps > small_order_result.market_impact_bps
            )

            logger.info(f"Market impact test:")
            logger.info(f"  Large order impact: {large_order_result.market_impact_bps:.1f} bps")
            logger.info(f"  Small order impact: {small_order_result.market_impact_bps:.1f} bps")
            logger.info(f"  Impact scales with size: {impact_scaling}")

            overall_success = market_success and limit_success and fee_accuracy and impact_scaling

            self.test_results["execution_simulator"] = {
                "success": overall_success,
                "market_order_success": market_success,
                "limit_order_success": limit_success,
                "fee_accuracy": fee_accuracy,
                "impact_scaling": impact_scaling,
                "market_execution_quality": market_result.execution_quality.value,
                "avg_slippage_bps": (market_result.slippage_bps + large_order_result.slippage_bps)
                / 2,
            }

            return overall_success

        except Exception as e:
            logger.error(f"Execution simulator test failed: {e}")
            return False

    def test_tracking_error_monitoring(self) -> bool:
        """Test tracking error monitoring system"""

        logger.info("Testing tracking error monitoring...")

        try:
            # Simulate 50 trade pairs (paper vs live)
            test_trades = []

            for i in range(50):
                trade_id = f"test_trade_{i + 1}"
                symbol = random.choice(["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"])
                side = random.choice([OrderSide.BUY, OrderSide.SELL])
                size = random.uniform(0.1, 5.0)
                base_price = random.uniform(20000, 60000)

                # Record paper trade (perfect execution)
                self.tracking_error_monitor.record_paper_trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    side=side,
                    size=size,
                    execution_price=base_price,
                    fees=base_price * size * 0.001,  # 10bps fee
                    timestamp=datetime.now()
                    - timedelta(minutes=random.randint(1, 10080)),  # Within 7 days
                )

                # Simulate live execution (with realistic deviations)
                live_result = self.execution_simulator.simulate_order_execution(
                    symbol=symbol,
                    side=side,
                    size=size,
                    order_type=OrderType.MARKET,
                    volume_24h=random.uniform(1000000, 100000000),
                    market_cap_tier=random.choice(["large_cap", "mid_cap", "small_cap"]),
                )

                # Record live execution
                self.tracking_error_monitor.record_live_execution(trade_id, live_result)

                test_trades.append(
                    {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "paper_price": base_price,
                        "live_price": live_result.average_price,
                    }
                )

            # Wait brief moment for processing
            time.sleep(0.1)

            # Analyze tracking error statistics
            stats_7d = self.tracking_error_monitor.calculate_tracking_error_statistics(days_back=7)

            stats_success = (
                stats_7d.get("total_trades", 0) >= 40  # Most trades recorded
                and stats_7d.get("mean_abs_tracking_error_bps", 0) > 0
                and stats_7d.get("std_tracking_error_bps", 0) >= 0
            )

            logger.info(f"Tracking error statistics (7 days):")
            logger.info(f"  Total trades: {stats_7d.get('total_trades', 0)}")
            logger.info(
                f"  Mean tracking error: {stats_7d.get('mean_tracking_error_bps', 0):.2f} bps"
            )
            logger.info(
                f"  Mean absolute TE: {stats_7d.get('mean_abs_tracking_error_bps', 0):.2f} bps"
            )
            logger.info(f"  Std deviation: {stats_7d.get('std_tracking_error_bps', 0):.2f} bps")
            logger.info(
                f"  Max absolute TE: {stats_7d.get('max_abs_tracking_error_bps', 0):.2f} bps"
            )
            logger.info(
                f"  P95 absolute TE: {stats_7d.get('p95_abs_tracking_error_bps', 0):.2f} bps"
            )

            # Test component attribution
            attribution = self.tracking_error_monitor.get_component_attribution(days_back=7)

            attribution_success = (
                attribution.get("status") != "no_data"
                and len(attribution.get("component_breakdown", {})) > 0
            )

            if attribution_success:
                logger.info("Component attribution:")
                for component, stats in attribution["component_breakdown"].items():
                    logger.info(
                        f"  {component}: {stats['mean_bps']:.1f} bps "
                        f"({stats['contribution_percentage']:.1f}%)"
                    )

            # Test comprehensive report generation
            report = self.tracking_error_monitor.generate_tracking_error_report(days_back=7)

            report_success = (
                report.total_trades > 0
                and report.mean_tracking_error_bps is not None
                and report.threshold_compliance_rate >= 0
            )

            logger.info(f"Comprehensive report:")
            logger.info(f"  Compliance rate: {report.threshold_compliance_rate:.1%}")
            logger.info(f"  PnL difference: ${report.pnl_difference:.2f}")
            logger.info(f"  Statistical significance: p={report.tracking_error_significance:.3f}")

            overall_success = stats_success and attribution_success and report_success

            self.test_results["tracking_error_monitoring"] = {
                "success": overall_success,
                "stats_success": stats_success,
                "attribution_success": attribution_success,
                "report_success": report_success,
                "total_trades_tracked": stats_7d.get("total_trades", 0),
                "mean_tracking_error_bps": stats_7d.get("mean_abs_tracking_error_bps", 0),
                "compliance_rate": report.threshold_compliance_rate,
            }

            return overall_success

        except Exception as e:
            logger.error(f"Tracking error monitoring test failed: {e}")
            return False

    def test_ab_paper_vs_live_comparison(self) -> bool:
        """Test A/B comparison between paper and live executions"""

        logger.info("Testing A/B paper vs live comparison...")

        try:
            # Generate realistic trading scenario over 7 days
            test_scenarios = [
                {"symbol": "BTC/USD", "base_price": 45000, "volatility": 0.02, "volume": 100000000},
                {"symbol": "ETH/USD", "base_price": 3000, "volatility": 0.03, "volume": 50000000},
                {"symbol": "SOL/USD", "base_price": 80, "volatility": 0.05, "volume": 10000000},
                {"symbol": "AVAX/USD", "base_price": 30, "volatility": 0.04, "volume": 5000000},
            ]

            ab_comparisons = []
            daily_tracking_errors = []

            # Simulate 7 days of trading
            for day in range(7):
                day_date = datetime.now() - timedelta(days=6 - day)
                day_trades = []

                # 10-20 trades per day
                num_trades = random.randint(10, 20)

                for trade in range(num_trades):
                    scenario = random.choice(test_scenarios)

                    # Price movement within day
                    price_drift = (
                        scenario["base_price"] * scenario["volatility"] * random.gauss(0, 1)
                    )
                    current_price = scenario["base_price"] + price_drift

                    trade_id = f"ab_test_d{day}_t{trade}"
                    side = random.choice([OrderSide.BUY, OrderSide.SELL])
                    size = random.uniform(0.1, 2.0)

                    # Paper execution (theoretical perfect)
                    paper_price = current_price
                    paper_fees = current_price * size * 0.001  # 10bps

                    self.tracking_error_monitor.record_paper_trade(
                        trade_id=trade_id,
                        symbol=scenario["symbol"],
                        side=side,
                        size=size,
                        execution_price=paper_price,
                        fees=paper_fees,
                        timestamp=day_date + timedelta(hours=random.randint(9, 16)),
                    )

                    # Live execution (realistic with costs)
                    live_result = self.execution_simulator.simulate_order_execution(
                        symbol=scenario["symbol"],
                        side=side,
                        size=size,
                        order_type=OrderType.MARKET,
                        volume_24h=scenario["volume"],
                        market_cap_tier="large_cap" if scenario["volume"] > 20000000 else "mid_cap",
                    )

                    self.tracking_error_monitor.record_live_execution(trade_id, live_result)

                    # Calculate immediate tracking error
                    if live_result.filled_size > 0:
                        paper_total = paper_price * size + paper_fees
                        live_total = (
                            live_result.average_price * live_result.filled_size
                            + live_result.total_fees
                        )
                        tracking_error_bps = ((live_total - paper_total) / paper_total) * 10000

                        ab_comparisons.append(
                            {
                                "day": day,
                                "trade_id": trade_id,
                                "symbol": scenario["symbol"],
                                "tracking_error_bps": tracking_error_bps,
                                "paper_price": paper_price,
                                "live_price": live_result.average_price,
                                "slippage_bps": live_result.slippage_bps,
                            }
                        )

                        day_trades.append(abs(tracking_error_bps))

                # Calculate daily average tracking error
                if day_trades:
                    daily_avg_te = np.mean(day_trades)
                    daily_tracking_errors.append(daily_avg_te)
                    logger.info(
                        f"Day {day + 1} average tracking error: {daily_avg_te:.1f} bps ({len(day_trades)} trades)"
                    )

            # Overall A/B test analysis
            if ab_comparisons:
                all_tracking_errors = [abs(comp["tracking_error_bps"]) for comp in ab_comparisons]

                # Statistical analysis
                mean_te = np.mean(all_tracking_errors)
                std_te = np.std(all_tracking_errors)
                p95_te = np.percentile(all_tracking_errors, 95)
                max_te = np.max(all_tracking_errors)

                # Check against target threshold (20 bps)
                target_threshold = 20.0
                trades_within_threshold = sum(
                    1 for te in all_tracking_errors if te <= target_threshold
                )
                compliance_rate = trades_within_threshold / len(all_tracking_errors)

                # Daily consistency check
                daily_te_std = (
                    np.std(daily_tracking_errors) if len(daily_tracking_errors) > 1 else 0
                )

                logger.info(f"A/B Test Results (7 days):")
                logger.info(f"  Total comparisons: {len(ab_comparisons)}")
                logger.info(f"  Mean tracking error: {mean_te:.1f} ± {std_te:.1f} bps")
                logger.info(f"  P95 tracking error: {p95_te:.1f} bps")
                logger.info(f"  Max tracking error: {max_te:.1f} bps")
                logger.info(f"  Compliance rate (<{target_threshold} bps): {compliance_rate:.1%}")
                logger.info(f"  Daily consistency (std): {daily_te_std:.1f} bps")

                # Success criteria:
                # 1. Mean tracking error < 15 bps
                # 2. P95 tracking error < 30 bps
                # 3. Compliance rate > 80%
                # 4. Daily consistency < 10 bps std

                success_criteria = {
                    "mean_te_acceptable": mean_te < 15.0,
                    "p95_te_acceptable": p95_te < 30.0,
                    "compliance_rate_acceptable": compliance_rate > 0.8,
                    "daily_consistency_acceptable": daily_te_std < 10.0,
                }

                overall_success = all(success_criteria.values())

                self.test_results["ab_comparison"] = {
                    "success": overall_success,
                    "total_comparisons": len(ab_comparisons),
                    "mean_tracking_error_bps": mean_te,
                    "std_tracking_error_bps": std_te,
                    "p95_tracking_error_bps": p95_te,
                    "max_tracking_error_bps": max_te,
                    "compliance_rate": compliance_rate,
                    "daily_consistency_std": daily_te_std,
                    "success_criteria": success_criteria,
                }

                # Log success criteria details
                for criterion, met in success_criteria.items():
                    status = "✅ PASS" if met else "❌ FAIL"
                    logger.info(f"  {criterion}: {status}")

                return overall_success
            else:
                logger.error("No A/B comparisons generated")
                return False

        except Exception as e:
            logger.error(f"A/B comparison test failed: {e}")
            return False

    def test_execution_quality_assessment(self) -> bool:
        """Test execution quality assessment and reporting"""

        logger.info("Testing execution quality assessment...")

        try:
            # Test different execution scenarios
            test_scenarios = [
                {
                    "name": "excellent_execution",
                    "size": 0.5,
                    "volume_24h": 100000000,
                    "market_cap_tier": "large_cap",
                    "expected_quality": ExecutionQuality.EXCELLENT,
                },
                {
                    "name": "good_execution",
                    "size": 1.0,
                    "volume_24h": 50000000,
                    "market_cap_tier": "mid_cap",
                    "expected_quality": ExecutionQuality.GOOD,
                },
                {
                    "name": "fair_execution",
                    "size": 5.0,
                    "volume_24h": 10000000,
                    "market_cap_tier": "small_cap",
                    "expected_quality": ExecutionQuality.FAIR,
                },
            ]

            quality_results = {}

            for scenario in test_scenarios:
                results = []

                # Run multiple executions for statistical significance
                for _ in range(10):
                    result = self.execution_simulator.simulate_order_execution(
                        symbol="TEST/USD",
                        side=OrderSide.BUY,
                        size=scenario["size"],
                        order_type=OrderType.MARKET,
                        volume_24h=scenario["volume_24h"],
                        market_cap_tier=scenario["market_cap_tier"],
                    )
                    results.append(result)

                # Analyze quality distribution
                quality_counts = {}
                for result in results:
                    quality = result.execution_quality.value
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1

                avg_slippage = np.mean([r.slippage_bps for r in results])
                avg_execution_time = np.mean([r.execution_time_ms for r in results])
                avg_fill_rate = np.mean([r.fill_rate for r in results])

                quality_results[scenario["name"]] = {
                    "quality_distribution": quality_counts,
                    "avg_slippage_bps": avg_slippage,
                    "avg_execution_time_ms": avg_execution_time,
                    "avg_fill_rate": avg_fill_rate,
                }

                logger.info(f"{scenario['name']}:")
                logger.info(f"  Quality distribution: {quality_counts}")
                logger.info(f"  Avg slippage: {avg_slippage:.1f} bps")
                logger.info(f"  Avg execution time: {avg_execution_time:.1f} ms")
                logger.info(f"  Avg fill rate: {avg_fill_rate:.1%}")

            # Verify quality differentiation
            excellent_slippage = quality_results["excellent_execution"]["avg_slippage_bps"]
            fair_slippage = quality_results["fair_execution"]["avg_slippage_bps"]

            quality_differentiation = fair_slippage > excellent_slippage

            # Get execution report
            execution_report = self.execution_simulator.get_execution_report(days_back=1)
            report_generated = execution_report.get("total_orders", 0) > 0

            overall_success = quality_differentiation and report_generated

            self.test_results["execution_quality"] = {
                "success": overall_success,
                "quality_differentiation": quality_differentiation,
                "report_generated": report_generated,
                "quality_results": quality_results,
                "execution_report": execution_report,
            }

            return overall_success

        except Exception as e:
            logger.error(f"Execution quality test failed: {e}")
            return False

    def run_comprehensive_tests(self):
        """Run all backtest-live parity tests"""

        logger.info("=" * 60)
        logger.info("🧪 BACKTEST-LIVE PARITY SYSTEM TESTS")
        logger.info("=" * 60)

        tests = [
            ("Execution Simulator Components", self.test_execution_simulator_components),
            ("Tracking Error Monitoring", self.test_tracking_error_monitoring),
            ("A/B Paper vs Live Comparison", self.test_ab_paper_vs_live_comparison),
            ("Execution Quality Assessment", self.test_execution_quality_assessment),
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
        logger.info("🏁 BACKTEST-LIVE PARITY TEST RESULTS")
        logger.info("=" * 60)

        for test_name, _ in tests:
            test_key = test_name.lower().replace(" ", "_").replace("-", "_")
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
            logger.info("🎉 ALL TESTS PASSED - BACKTEST-LIVE PARITY READY")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

        # Key metrics summary
        logger.info("\n📊 KEY PARITY METRICS:")

        if "ab_comparison" in self.test_results:
            ab = self.test_results["ab_comparison"]
            logger.info(f"• Mean Tracking Error: {ab.get('mean_tracking_error_bps', 0):.1f} bps")
            logger.info(f"• P95 Tracking Error: {ab.get('p95_tracking_error_bps', 0):.1f} bps")
            logger.info(f"• Compliance Rate: {ab.get('compliance_rate', 0):.1%}")

        if "execution_simulator" in self.test_results:
            exec_sim = self.test_results["execution_simulator"]
            logger.info(f"• Avg Slippage: {exec_sim.get('avg_slippage_bps', 0):.1f} bps")
            logger.info(
                f"• Execution Quality: {exec_sim.get('market_execution_quality', 'unknown')}"
            )

        if "tracking_error_monitoring" in self.test_results:
            tracking = self.test_results["tracking_error_monitoring"]
            logger.info(f"• Trades Tracked: {tracking.get('total_trades_tracked', 0)}")

        return passed_tests == total_tests


def main():
    """Run backtest-live parity tests"""

    tester = BacktestLiveParityTester()
    success = tester.run_comprehensive_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
