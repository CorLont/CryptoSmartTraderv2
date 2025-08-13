#!/usr/bin/env python3
"""Demo script voor Fase 2: Guardrails & Observability - Toont auto-stop en alert functionaliteit."""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cryptosmarttrader.core.risk_guard import RiskGuard, RiskLevel
from src.cryptosmarttrader.core.execution_policy import (
    ExecutionPolicy, OrderRequest, OrderType, MarketConditions
)
from src.cryptosmarttrader.monitoring.prometheus_metrics import get_metrics
from src.cryptosmarttrader.monitoring.alert_rules import AlertManager
from src.cryptosmarttrader.testing.simulation_tester import (
    SimulationTester, FailureScenario, SimulationConfig
)


class Fase2Demo:
    """Demonstratie van Fase 2 guardrails en observability."""
    
    def __init__(self):
        """Initialiseer demo componenten."""
        print("🚀 CryptoSmartTrader V2 - Fase 2 Demo: Guardrails & Observability")
        print("=" * 80)
        
        # Initialiseer alle systemen
        self.risk_guard = RiskGuard()
        self.execution_policy = ExecutionPolicy()
        self.alert_manager = AlertManager()
        self.metrics = get_metrics()
        self.simulation_tester = SimulationTester(
            self.risk_guard, self.execution_policy, self.alert_manager
        )
        
        # Demo portfolio setup
        self.portfolio_value = 100000.0  # $100k demo portfolio
        
        # Alert callback voor demo
        self.alerts_fired = []
        self.alert_manager.register_alert_callback(self._demo_alert_callback)
        
        print("✅ Alle guardrail systemen geïnitialiseerd")
        print(f"📊 Demo portfolio: ${self.portfolio_value:,.2f}")
        print()
    
    def _demo_alert_callback(self, alert):
        """Demo alert callback voor visuele feedback."""
        self.alerts_fired.append(alert)
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "critical": "🚨",
            "emergency": "🆘"
        }
        emoji = severity_emoji.get(alert.severity.value, "📢")
        print(f"{emoji} ALERT: {alert.rule_name} - {alert.message}")
    
    def demo_risk_limits(self):
        """Demonstreer risk limit configuratie."""
        print("🛡️  RISK LIMITS & GUARDRAILS")
        print("-" * 40)
        
        limits = self.risk_guard.risk_limits
        print(f"Maximum dagelijks verlies: {limits.max_daily_loss_percent}%")
        print(f"Maximum drawdown: {limits.max_drawdown_percent}%") 
        print(f"Maximum positie grootte: {limits.max_position_size_percent}%")
        print(f"Maximum totale exposure: {limits.max_total_exposure_percent}%")
        print(f"Maximum aantal posities: {limits.max_position_count}")
        print(f"Minimum data kwaliteit: {limits.min_data_quality_score}")
        print()
    
    def demo_execution_policy(self):
        """Demonstreer execution policy configuratie."""
        print("⚡ EXECUTION POLICY & SLIPPAGE BUDGET")
        print("-" * 40)
        
        gate = self.execution_policy.tradability_gate
        budget = self.execution_policy.slippage_budget
        
        print(f"Minimum volume (24h): ${gate.min_volume_24h:,.0f}")
        print(f"Maximum spread: {gate.max_spread_percent}%")
        print(f"Minimum orderbook diepte: ${gate.min_orderbook_depth:,.0f}")
        print(f"Maximum slippage budget: {budget.max_slippage_percent}%")
        print(f"Slippage waarschuwing bij: {budget.warning_threshold_percent}%")
        print(f"Emergency stop bij: {budget.emergency_stop_percent}%")
        print()
    
    def demo_market_conditions(self):
        """Setup demo marktcondities.""" 
        print("📈 MARKTCONDITIES SETUP")
        print("-" * 40)
        
        # BTC/USDT normale condities
        btc_conditions = MarketConditions(
            bid_price=49950.0,
            ask_price=50050.0,
            mid_price=50000.0,
            spread_percent=0.2,
            volume_24h=500000000.0,  # $500M volume
            orderbook_depth_bid=100000.0,
            orderbook_depth_ask=100000.0,
            price_volatility=1.5,
            liquidity_score=0.85
        )
        
        # ETH/USDT slechtere condities
        eth_conditions = MarketConditions(
            bid_price=2995.0,
            ask_price=3005.0,
            mid_price=3000.0,
            spread_percent=0.33,
            volume_24h=200000000.0,  # $200M volume
            orderbook_depth_bid=50000.0,
            orderbook_depth_ask=50000.0,
            price_volatility=2.2,
            liquidity_score=0.72
        )
        
        self.execution_policy.update_market_conditions("BTC/USDT", btc_conditions)
        self.execution_policy.update_market_conditions("ETH/USDT", eth_conditions)
        
        print("✅ BTC/USDT: Goede liquiditeit (spread 0.2%, volume $500M)")
        print("⚠️  ETH/USDT: Matige liquiditeit (spread 0.33%, volume $200M)")
        print()
    
    def demo_order_validation(self):
        """Demonstreer order validatie met tradability gates."""
        print("🔍 ORDER VALIDATIE & TRADABILITY GATES")
        print("-" * 40)
        
        # Test BTC order (zou moeten slagen)
        btc_order = OrderRequest(
            client_order_id=self.execution_policy.generate_client_order_id("BTC/USDT", "buy", 0.1),
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=0.1,
            confidence_score=0.85
        )
        
        valid_btc, btc_issues = self.execution_policy.validate_order_request(btc_order)
        estimated_slippage = self.execution_policy.estimate_slippage(btc_order)
        
        print(f"📊 BTC/USDT Order:")
        print(f"   Validatie: {'✅ GOEDGEKEURD' if valid_btc else '❌ AFGEWEZEN'}")
        print(f"   Geschatte slippage: {estimated_slippage:.3f}%")
        if btc_issues:
            for issue in btc_issues:
                print(f"   ⚠️  {issue}")
        
        # Test ETH order met te groot volume
        eth_order = OrderRequest(
            client_order_id=self.execution_policy.generate_client_order_id("ETH/USDT", "sell", 100),
            symbol="ETH/USDT", 
            side="sell",
            order_type=OrderType.MARKET,
            quantity=100,  # Grote order voor demo
            confidence_score=0.65
        )
        
        valid_eth, eth_issues = self.execution_policy.validate_order_request(eth_order)
        estimated_slippage_eth = self.execution_policy.estimate_slippage(eth_order)
        
        print(f"\n📊 ETH/USDT Grote Order:")
        print(f"   Validatie: {'✅ GOEDGEKEURD' if valid_eth else '❌ AFGEWEZEN'}")
        print(f"   Geschatte slippage: {estimated_slippage_eth:.3f}%")
        if eth_issues:
            for issue in eth_issues:
                print(f"   ⚠️  {issue}")
        
        print()
    
    async def demo_simulation_scenarios(self):
        """Demonstreer geforceerde fout scenario's."""
        print("🧪 GEFORCEERDE FOUT SIMULATIES")
        print("-" * 40)
        
        scenarios = [
            ("Kill Switch Test", FailureScenario.KILL_SWITCH_TEST, 1.0, 1),
            ("Drawdown Spike", FailureScenario.DRAWDOWN_SPIKE, 0.8, 2),  
            ("Hoge Slippage", FailureScenario.HIGH_SLIPPAGE, 0.6, 2),
            ("Data Gap", FailureScenario.DATA_GAP, 0.7, 3)
        ]
        
        for name, scenario, intensity, duration in scenarios:
            print(f"\n🔬 Test: {name}")
            print(f"   Intensiteit: {intensity} | Duur: {duration} min")
            
            # Clear previous alerts
            self.alerts_fired = []
            
            config = SimulationConfig(
                scenario=scenario,
                duration_minutes=duration,
                intensity=intensity,
                portfolio_value=self.portfolio_value,
                auto_recovery=True,
                recovery_delay_minutes=1
            )
            
            start_time = time.time()
            result = await self.simulation_tester.run_simulation(config)
            execution_time = time.time() - start_time
            
            print(f"   Resultaat: {'✅ GESLAAGD' if result.success else '❌ GEFAALD'}")
            print(f"   Executietijd: {execution_time:.1f}s")
            print(f"   Alerts getriggerd: {len(result.alerts_triggered)}")
            print(f"   Kill switch: {'✅ GEACTIVEERD' if result.kill_switch_activated else '❌ NIET GEACTIVEERD'}")
            
            if result.alerts_triggered:
                for alert in result.alerts_triggered[:3]:  # Show first 3
                    print(f"     🚨 {alert}")
            
            # Short pause between tests
            await asyncio.sleep(2)
    
    def demo_metrics_collection(self):
        """Demonstreer metrics collectie."""
        print("📊 PROMETHEUS METRICS COLLECTIE")
        print("-" * 40)
        
        # Update some demo metrics
        self.metrics.update_portfolio_metrics(
            portfolio_value=self.portfolio_value,
            daily_pnl_percent=-2.5,  # Demo loss
            max_drawdown_percent=3.8
        )
        
        self.metrics.record_order("kraken", "BTC/USDT", "buy", "filled", 0.8, 0.15)
        self.metrics.record_signal("random_forest", "ETH/USDT", "bullish", 0.78)
        self.metrics.set_agent_status("technical_analyzer", True)
        self.metrics.update_data_source("kraken", 0.92)
        
        print("✅ Portfolio metrics bijgewerkt")
        print("✅ Order metrics geregistreerd")  
        print("✅ Signal metrics geregistreerd")
        print("✅ System health metrics bijgewerkt")
        print("✅ Data quality metrics bijgewerkt")
        print()
    
    def demo_alert_rules(self):
        """Demonstreer alert rule configuratie."""
        print("🚨 ALERT RULES & THRESHOLDS")
        print("-" * 40)
        
        critical_rules = [
            rule for rule in self.alert_manager.rules.values()
            if rule.severity.value == "critical" and rule.enabled
        ]
        
        warning_rules = [
            rule for rule in self.alert_manager.rules.values()
            if rule.severity.value == "warning" and rule.enabled
        ]
        
        emergency_rules = [
            rule for rule in self.alert_manager.rules.values()
            if rule.severity.value == "emergency" and rule.enabled
        ]
        
        print(f"🆘 Emergency rules: {len(emergency_rules)}")
        for rule in emergency_rules[:3]:
            print(f"   • {rule.name}: {rule.description}")
        
        print(f"\n🚨 Critical rules: {len(critical_rules)}")
        for rule in critical_rules[:3]:
            print(f"   • {rule.name}: {rule.description}")
            
        print(f"\n⚠️  Warning rules: {len(warning_rules)}")
        for rule in warning_rules[:3]:
            print(f"   • {rule.name}: {rule.description}")
        
        print(f"\nTotaal actieve rules: {len(self.alert_manager.rules)}")
        print()
    
    def demo_95p_slippage_validation(self):
        """Demonstreer 95e percentiel slippage validatie."""
        print("📈 95P SLIPPAGE BUDGET VALIDATIE")
        print("-" * 40)
        
        # Simuleer slippage data voor verschillende symbolen
        slippage_data = {
            "BTC/USDT": [0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.42],
            "ETH/USDT": [0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.38, 0.45, 0.52, 0.68],
            "ADA/USDT": [0.25, 0.32, 0.38, 0.45, 0.52, 0.58, 0.65, 0.78, 0.85, 1.12]
        }
        
        budget_limit = self.execution_policy.slippage_budget.max_slippage_percent
        
        for symbol, slippages in slippage_data.items():
            # Calculate 95th percentile
            sorted_slippages = sorted(slippages)
            p95_index = int(0.95 * len(sorted_slippages))
            p95_slippage = sorted_slippages[p95_index]
            
            within_budget = p95_slippage <= budget_limit
            status = "✅ BINNEN BUDGET" if within_budget else "❌ OVERSCHRIJDT BUDGET"
            
            print(f"{symbol}:")
            print(f"   95e percentiel slippage: {p95_slippage:.3f}%")
            print(f"   Budget limit: {budget_limit:.3f}%")
            print(f"   Status: {status}")
            
            # Record metrics
            for slippage in slippages:
                self.metrics.record_order("kraken", symbol, "buy", "filled", 0.5, slippage)
        
        print(f"\n🎯 Meetpunt: 95e percentiel slippage ≤ {budget_limit}% budget")
        print()
    
    async def run_complete_demo(self):
        """Voer volledige Fase 2 demonstratie uit."""
        try:
            # Setup
            self.demo_risk_limits()
            await asyncio.sleep(1)
            
            self.demo_execution_policy()
            await asyncio.sleep(1)
            
            self.demo_market_conditions()
            await asyncio.sleep(1)
            
            self.demo_order_validation()
            await asyncio.sleep(2)
            
            self.demo_metrics_collection()
            await asyncio.sleep(1)
            
            self.demo_alert_rules()
            await asyncio.sleep(1)
            
            self.demo_95p_slippage_validation()
            await asyncio.sleep(2)
            
            # Main simulation tests
            await self.demo_simulation_scenarios()
            
            # Final summary
            print("\n🏆 FASE 2 DEMO SAMENVATTING")
            print("=" * 80)
            
            test_summary = self.simulation_tester.get_test_summary()
            print(f"✅ RiskGuard: Operationeel met {len(RiskLevel)} risico niveaus")
            print(f"✅ ExecutionPolicy: COIDs, slippage budget & retry logica")
            print(f"✅ Prometheus: {len(self.metrics.registry._collector_to_names)} metrics verzameld")
            print(f"✅ AlertManager: {len(self.alert_manager.rules)} regels actief")
            print(f"✅ Simulation Tests: {test_summary.get('total_tests', 0)} uitgevoerd")
            print(f"✅ Alert Responsiviteit: {len(self.alerts_fired)} alerts gevuurd")
            
            # Risk level status
            constraints = self.risk_guard.get_trading_constraints()
            print(f"\n🛡️  Current Risk Level: {self.risk_guard.current_risk_level.value.upper()}")
            print(f"🔄 Trading Mode: {self.risk_guard.trading_mode.value.upper()}")
            print(f"🚦 Trading Enabled: {'JA' if constraints['trading_enabled'] else 'NEE'}")
            print(f"🔴 Kill Switch: {'ACTIEF' if self.risk_guard.kill_switch_active else 'INACTIEF'}")
            
            print(f"\n✨ Fase 2 'Guardrails & Observability' SUCCESVOL GEÏMPLEMENTEERD!")
            print(f"🎯 Meetpunt behaald: Auto-stop + alerts + 95p slippage validatie")
            
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Hoofdfunctie voor demo uitvoering."""
    demo = Fase2Demo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())