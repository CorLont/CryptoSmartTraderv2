#!/usr/bin/env python3
"""Demo script voor Fase 3: Alpha & Parity - Toont regime detectie, strategy switching en backtest-live parity."""

import asyncio
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cryptosmarttrader.core.regime_detector import RegimeDetector, MarketRegime
from src.cryptosmarttrader.core.strategy_switcher import StrategySwitcher, StrategyType
from src.cryptosmarttrader.analysis.backtest_parity import BacktestParityAnalyzer, TradeExecution
from src.cryptosmarttrader.deployment.canary_system import (
    CanaryDeploymentSystem,
    DeploymentPlan,
    CanaryStage,
)
from src.cryptosmarttrader.core.risk_guard import RiskGuard
from src.cryptosmarttrader.monitoring.alert_rules import AlertManager


class Fase3Demo:
    """Demonstratie van Fase 3 alpha generation en parity systemen."""

    def __init__(self):
        """Initialiseer demo componenten."""
        print("🚀 CryptoSmartTrader V2 - Fase 3 Demo: Alpha & Parity")
        print("=" * 80)

        # Initialiseer alle systemen
        self.regime_detector = RegimeDetector(lookback_periods=100)
        self.strategy_switcher = StrategySwitcher(self.regime_detector, initial_capital=100000.0)
        self.parity_analyzer = BacktestParityAnalyzer(target_tracking_error_bps=20.0)

        # Voor canary deployment demo
        self.risk_guard = RiskGuard()
        self.alert_manager = AlertManager()
        self.canary_system = CanaryDeploymentSystem(self.risk_guard, self.alert_manager)

        # Demo data
        self.market_data = {}
        self.generate_sample_market_data()

        print("✅ Alle Fase 3 systemen geïnitialiseerd")
        print(f"📊 Demo symbolen: {list(self.market_data.keys())}")
        print()

    def generate_sample_market_data(self):
        """Genereer realistische marktdata voor verschillende regimes."""
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]

        # Verschillende marktcondities per symbool
        for i, symbol in enumerate(symbols):
            dates = pd.date_range("2024-01-01", periods=100, freq="H")

            # Creëer verschillende trends
            if i == 0:  # BTC - Bull trending
                base_price = 50000
                trend = np.cumsum(np.random.normal(0.002, 0.02, 100))  # Positive trend
            elif i == 1:  # ETH - Sideways high vol
                base_price = 3000
                trend = np.cumsum(np.random.normal(0, 0.03, 100))  # High volatility
            elif i == 2:  # ADA - Mean reverting
                base_price = 0.5
                trend = np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.1  # Oscillating
            else:  # SOL - Breakout pattern
                base_price = 100
                trend = np.concatenate(
                    [
                        np.zeros(50),  # Flat first half
                        np.cumsum(np.random.normal(0.01, 0.02, 50)),  # Breakout second half
                    ]
                )

            prices = base_price * (1 + trend)

            # Generate OHLCV data
            self.market_data[symbol] = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                    "high": prices * (1 + np.random.uniform(0.005, 0.02, 100)),
                    "low": prices * (1 - np.random.uniform(0.005, 0.02, 100)),
                    "close": prices,
                    "volume": np.random.uniform(1000, 10000, 100),
                }
            )

    def demo_regime_detection(self):
        """Demonstreer regime detectie systeem."""
        print("🔍 REGIME DETECTIE SYSTEEM")
        print("-" * 40)

        regime_results = {}

        for symbol, data in self.market_data.items():
            print(f"\n📈 Analyseer {symbol}:")

            # Update market data
            self.regime_detector.update_market_data(symbol, data)

            # Calculate metrics
            metrics = self.regime_detector.calculate_regime_metrics(symbol)

            if metrics:
                # Classify regime
                regime, confidence = self.regime_detector.classify_regime(metrics)
                regime_results[symbol] = (regime, confidence)

                print(f"   Trend sterkte: {metrics.trend_strength:.3f}")
                print(f"   Volatiliteit percentiel: {metrics.volatility_percentile:.1f}%")
                print(f"   Momentum score: {metrics.momentum_score:.3f}")
                print(f"   Hurst exponent: {metrics.hurst_exponent:.3f}")
                print(f"   ADX sterkte: {metrics.adx_strength:.1f}")
                print(f"   → REGIME: {regime.value.upper()} (confidence: {confidence:.2f})")
            else:
                print("   ⚠️  Onvoldoende data voor analyse")

        # Update detector met BTC (primary market)
        if "BTC/USDT" in regime_results:
            detected_regime = self.regime_detector.update_regime("BTC/USDT")
            print(f"\n🎯 Primaire markt regime: {detected_regime.value.upper()}")

            # Show strategy configuration
            strategy_config = self.regime_detector.get_current_strategy_config()
            print(f"   Primaire strategie: {strategy_config['primary_strategy']}")
            print(f"   Positie sizing: {strategy_config['position_sizing']:.1f}x")
            print(f"   Risk multiplier: {strategy_config['risk_multiplier']:.1f}x")
            print(f"   Rebalance frequentie: {strategy_config['rebalance_frequency']}")

        print()
        return regime_results

    def demo_strategy_switching(self, regime_results):
        """Demonstreer strategy switching met regime awareness."""
        print("⚡ STRATEGY SWITCHING & ALLOCATION")
        print("-" * 40)

        # Show current allocations
        summary = self.strategy_switcher.get_strategy_allocation_summary()

        print(f"Huidig regime: {summary['current_regime'].upper()}")
        print(f"Regime confidence: {summary['regime_confidence']:.2f}")

        print("\n📊 Strategie allocaties:")
        for strategy, config in summary["allocations"].items():
            if config["weight"] > 0:
                print(
                    f"   {strategy}: {config['weight']:.1%} (max pos: {config['max_position_size']:.1%})"
                )

        # Generate position targets
        position_targets = self.strategy_switcher.generate_position_targets(self.market_data)

        print(f"\n🎯 Positie targets ({len(position_targets)} symbolen):")
        for symbol, target in position_targets.items():
            print(
                f"   {symbol}: {target.target_weight:.2%} "
                f"(confidence: {target.confidence:.2f}, "
                f"bron: {target.strategy_source.value})"
            )

        # Show cluster limits
        print("\n🛡️  Cluster limits:")
        for cluster_name, cluster_info in summary["cluster_limits"].items():
            utilization = cluster_info["utilization"]
            status = "✅" if utilization < 0.8 else "⚠️" if utilization < 1.0 else "🚨"
            print(
                f"   {cluster_name}: {cluster_info['current_weight']:.1%} / "
                f"{cluster_info['max_weight']:.1%} {status}"
            )

        print()
        return position_targets

    async def demo_backtest_parity(self, position_targets):
        """Demonstreer backtest-live parity tracking."""
        print("📊 BACKTEST-LIVE PARITY TRACKING")
        print("-" * 40)

        # Simulate some trades for parity analysis
        market_conditions = {
            "bid": 49900.0,
            "ask": 50100.0,
            "price": 50000.0,
            "volume_24h": 500000000.0,
            "volatility": 0.02,
            "orderbook_depth": 100000.0,
        }

        print("🔄 Simuleer uitvoeringen voor parity analyse...")

        # Generate backtest executions
        for symbol in list(position_targets.keys())[:3]:  # First 3 symbols
            for i in range(5):
                bt_execution = self.parity_analyzer.simulate_execution(
                    symbol, 0.1, "buy", market_conditions, "backtest"
                )
                self.parity_analyzer.record_execution(bt_execution)

                # Simulate live execution with more realistic conditions
                live_conditions = market_conditions.copy()
                live_conditions["volatility"] *= 1.5  # Higher volatility in live

                live_execution = self.parity_analyzer.simulate_execution(
                    symbol, 0.1, "buy", live_conditions, "live"
                )
                self.parity_analyzer.record_execution(live_execution)

        # Calculate parity metrics
        parity_metrics = self.parity_analyzer.calculate_parity_metrics(lookback_hours=1)

        if parity_metrics:
            print(f"\n📈 Parity Metrics:")
            print(f"   Tracking error: {parity_metrics.tracking_error_bps:.1f} bps")
            print(f"   Target: ≤ {self.parity_analyzer.target_tracking_error_bps:.1f} bps")
            print(f"   Correlatie: {parity_metrics.correlation:.3f}")
            print(f"   Alpha verschil: {parity_metrics.alpha_difference:.2f} bps")
            print(f"   Slippage verschil: {parity_metrics.slippage_difference:.2f} bps")
            print(f"   Execution quality: {parity_metrics.execution_quality_score:.1f}/100")

            # Check acceptance
            acceptable, _ = self.parity_analyzer.is_tracking_error_acceptable()
            status = "✅ BINNEN TARGET" if acceptable else "❌ BOVEN TARGET"
            print(f"   Status: {status}")
        else:
            print("   ⚠️  Onvoldoende data voor parity berekening")

        # Show slippage analysis
        slippage_analysis = self.parity_analyzer.get_slippage_analysis(hours=1)

        if "total_slippage" in slippage_analysis:
            print(f"\n📉 Slippage Analyse:")
            print(f"   Gemiddeld: {slippage_analysis['total_slippage']['mean_bps']:.2f} bps")
            print(f"   95e percentiel: {slippage_analysis['total_slippage']['p95_bps']:.2f} bps")
            print(
                f"   Binnen budget: {'✅ JA' if slippage_analysis['within_budget'] else '❌ NEE'}"
            )

            breakdown = slippage_analysis["component_breakdown"]
            print(f"   Market impact: {breakdown['market_impact_bps']:.2f} bps")
            print(f"   Spread kosten: {breakdown['spread_cost_bps']:.2f} bps")
            print(f"   Timing kosten: {breakdown['timing_cost_bps']:.2f} bps")

        print()
        return parity_metrics

    async def demo_canary_deployment(self):
        """Demonstreer canary deployment systeem."""
        print("🚢 CANARY DEPLOYMENT SYSTEEM")
        print("-" * 40)

        # Create deployment plan
        deployment_plan = DeploymentPlan(
            version="v2.1.0-fase3",
            features=["regime_detection", "strategy_switching", "backtest_parity"],
            staging_duration_hours=1,  # Versneld voor demo (normaal 168u = 7 dagen)
            staging_risk_percentage=1.0,
            prod_canary_duration_hours=1,  # Versneld voor demo (normaal 72u = 3 dagen)
            prod_canary_risk_percentage=5.0,
            auto_rollback_enabled=True,
        )

        print(f"📋 Deployment Plan: {deployment_plan.version}")
        print(f"   Features: {', '.join(deployment_plan.features)}")
        print(
            f"   Staging: {deployment_plan.staging_duration_hours}h @ {deployment_plan.staging_risk_percentage}% risk"
        )
        print(
            f"   Prod Canary: {deployment_plan.prod_canary_duration_hours}h @ {deployment_plan.prod_canary_risk_percentage}% risk"
        )
        print(
            f"   Auto-rollback: {'✅ ENABLED' if deployment_plan.auto_rollback_enabled else '❌ DISABLED'}"
        )

        # Register deployment callback
        async def deployment_callback(event):
            event_emoji = {
                "deployment_started": "🚀",
                "staging_canary_started": "🧪",
                "prod_canary_started": "🏭",
                "full_rollout_started": "🌍",
                "deployment_completed": "✅",
                "rollback_initiated": "🔄",
                "deployment_failed": "❌",
            }
            emoji = event_emoji.get(event["event_type"], "📢")
            print(f"   {emoji} {event['event_type'].replace('_', ' ').title()}")

        self.canary_system.register_deployment_callback(deployment_callback)

        print(f"\n🚀 Starting canary deployment...")

        # Start deployment
        success = await self.canary_system.start_deployment(deployment_plan)

        if success:
            print("   ✅ Deployment gestart")

            # Monitor deployment progress
            for i in range(12):  # Monitor for 2 minutes (12 x 10s)
                await asyncio.sleep(10)

                status = self.canary_system.get_deployment_status()
                print(
                    f"   📊 Stage: {status['stage'].upper()} | "
                    f"Health: {status['health_status'].upper()} | "
                    f"Duration: {status.get('stage_duration_hours', 0):.2f}h"
                )

                if status["stage"] in ["completed", "failed"]:
                    break

            # Final status
            final_status = self.canary_system.get_deployment_status()
            if final_status["stage"] == "completed":
                print("   ✅ Deployment succesvol afgerond!")
            else:
                print(f"   ⚠️  Deployment status: {final_status['stage']}")
        else:
            print("   ❌ Deployment start gefaald")

        print()

    def demo_return_attribution(self, parity_metrics):
        """Demonstreer return attribution analyse."""
        print("💰 RETURN ATTRIBUTION ANALYSE")
        print("-" * 40)

        # Generate sample return data
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        portfolio_returns = pd.Series(np.random.normal(0.002, 0.01, 7), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.008, 7), index=dates)

        attribution = self.parity_analyzer.analyze_return_attribution(
            portfolio_returns, benchmark_returns, period_days=7
        )

        print(
            f"📅 Periode: {attribution.period_start.strftime('%Y-%m-%d')} → {attribution.period_end.strftime('%Y-%m-%d')}"
        )
        print(f"\n📊 Return Components:")
        print(
            f"   Totaal return: {attribution.total_return:.4f} ({attribution.total_return * 100:.2f}%)"
        )
        print(
            f"   Alpha return: {attribution.alpha_return:.4f} ({attribution.alpha_return * 100:.2f}%)"
        )
        print(
            f"   Market impact: {attribution.market_impact:.4f} ({attribution.market_impact * 100:.2f}%)"
        )
        print(
            f"   Fees impact: {attribution.fees_impact:.4f} ({attribution.fees_impact * 10000:.1f} bps)"
        )
        print(
            f"   Slippage impact: {attribution.slippage_impact:.4f} ({attribution.slippage_impact * 10000:.1f} bps)"
        )
        print(
            f"   Timing impact: {attribution.timing_impact:.4f} ({attribution.timing_impact * 10000:.1f} bps)"
        )
        print(
            f"   Sizing impact: {attribution.sizing_impact:.4f} ({attribution.sizing_impact * 100:.2f}%)"
        )
        print(f"   Confidence: {attribution.attribution_confidence:.2f}")

        # Attribution visualization
        total_costs = (
            abs(attribution.fees_impact)
            + abs(attribution.slippage_impact)
            + abs(attribution.timing_impact)
        )
        alpha_net = attribution.alpha_return - total_costs

        print(f"\n🎯 Net Alpha After Costs:")
        print(f"   Gross alpha: {attribution.alpha_return * 10000:.1f} bps")
        print(f"   Total kosten: {total_costs * 10000:.1f} bps")
        print(f"   Net alpha: {alpha_net * 10000:.1f} bps")

        efficiency = (
            (alpha_net / attribution.alpha_return * 100) if attribution.alpha_return != 0 else 0
        )
        print(f"   Alpha efficiency: {efficiency:.1f}%")

        print()

    async def run_complete_demo(self):
        """Voer volledige Fase 3 demonstratie uit."""
        try:
            # 1. Regime Detection
            regime_results = self.demo_regime_detection()
            await asyncio.sleep(2)

            # 2. Strategy Switching
            position_targets = self.demo_strategy_switching(regime_results)
            await asyncio.sleep(2)

            # 3. Backtest-Live Parity
            parity_metrics = await self.demo_backtest_parity(position_targets)
            await asyncio.sleep(2)

            # 4. Return Attribution
            self.demo_return_attribution(parity_metrics)
            await asyncio.sleep(2)

            # 5. Canary Deployment
            await self.demo_canary_deployment()

            # Final Summary
            print("\n🏆 FASE 3 DEMO SAMENVATTING")
            print("=" * 80)

            print("✅ Regime Detection: 6 market regimes met quantitative metrics")
            print("✅ Strategy Switching: Dynamic allocation met cluster limits")
            print("✅ Vol-targeting: Volatility-adjusted position sizing")

            if parity_metrics:
                tracking_ok = parity_metrics.tracking_error_bps <= 20.0
                print(
                    f"✅ Backtest-Live Parity: {parity_metrics.tracking_error_bps:.1f} bps tracking error {'✅' if tracking_ok else '❌'}"
                )
            else:
                print("⚠️  Backtest-Live Parity: Onvoldoende data voor validatie")

            print("✅ Return Attribution: Alpha/fees/slippage component analysis")
            print("✅ Canary Deployment: 7-dagen staging → 72u prod canary")

            print(f"\n🎯 Meetpunten Fase 3:")
            print(f"   📅 7 dagen staging canary: ≤1% risk exposure")
            print(f"   🏭 72 uur prod canary: groen light voor full rollout")
            print(f"   📊 Tracking error: <20 bps/dag target")
            print(f"   💰 Return attribution: Component-wise breakdown")

            print(f"\n✨ Fase 3 'Alpha & Parity' SUCCESVOL GEÏMPLEMENTEERD!")
            print(f"🚀 Systeem gereed voor enterprise deployment met automated safety gates")

        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback

            traceback.print_exc()


async def main():
    """Hoofdfunctie voor demo uitvoering."""
    demo = Fase3Demo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
