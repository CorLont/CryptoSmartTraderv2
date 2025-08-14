#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Black Swan Simulation & Stress Testing Engine
Simulates extreme market events and tests system robustness against unseen scenarios
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum
import random
import math


class BlackSwanType(Enum):
    """Types of black swan events"""

    FLASH_CRASH = "flash_crash"
    EXCHANGE_HACK = "exchange_hack"
    REGULATORY_BAN = "regulatory_ban"
    CHAIN_FORK = "chain_fork"
    WHALE_DUMP = "whale_dump"
    STABLECOIN_DEPEG = "stablecoin_depeg"
    EXCHANGE_OUTAGE = "exchange_outage"
    MARKET_MANIPULATION = "market_manipulation"
    TECHNICAL_EXPLOIT = "technical_exploit"
    GEOPOLITICAL_SHOCK = "geopolitical_shock"


class MarketCondition(Enum):
    """Market conditions for simulation"""

    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass
class BlackSwanEvent:
    """Black swan event specification"""

    event_type: BlackSwanType
    severity: float  # 0.0 to 1.0
    duration_hours: float
    affected_coins: List[str]
    price_impact: Dict[str, float]  # coin -> impact percentage
    volume_impact: Dict[str, float]  # coin -> volume multiplier
    liquidity_impact: float  # liquidity reduction factor
    description: str
    probability: float  # Historical probability estimate


@dataclass
class StressTestResult:
    """Stress test execution result"""

    event_type: BlackSwanType
    severity: float
    market_condition: MarketCondition
    portfolio_impact: float
    max_drawdown: float
    recovery_time_hours: float
    model_accuracy_degradation: float
    system_failures: List[str]
    risk_mitigation_effectiveness: float
    lessons_learned: List[str]


class BlackSwanEventGenerator:
    """Generate realistic black swan events for simulation"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.historical_events = self._initialize_historical_events()

    def _initialize_historical_events(self) -> Dict[BlackSwanType, List[Dict[str, Any]]]:
        """Initialize database of historical black swan events"""
        return {
            BlackSwanType.FLASH_CRASH: [
                {
                    "date": "2021-05-19",
                    "description": "Bitcoin flash crash from $43k to $30k",
                    "impact": -30.0,
                    "duration_hours": 4,
                    "volume_spike": 5.0,
                },
                {
                    "date": "2022-05-12",
                    "description": "LUNA/UST collapse crash",
                    "impact": -80.0,
                    "duration_hours": 72,
                    "volume_spike": 10.0,
                },
            ],
            BlackSwanType.EXCHANGE_HACK: [
                {
                    "date": "2022-08-02",
                    "description": "Nomad Bridge hack $190M",
                    "impact": -15.0,
                    "duration_hours": 12,
                    "affected_protocols": ["cross_chain"],
                },
                {
                    "date": "2022-03-23",
                    "description": "Ronin Bridge hack $625M",
                    "impact": -25.0,
                    "duration_hours": 24,
                    "affected_protocols": ["gaming", "defi"],
                },
            ],
            BlackSwanType.REGULATORY_BAN: [
                {
                    "date": "2021-09-24",
                    "description": "China bans cryptocurrency transactions",
                    "impact": -20.0,
                    "duration_hours": 168,  # 1 week
                    "affected_regions": ["china", "asia"],
                }
            ],
            BlackSwanType.EXCHANGE_OUTAGE: [
                {
                    "date": "2021-04-18",
                    "description": "Coinbase outage during high volume",
                    "impact": -5.0,
                    "duration_hours": 6,
                    "volume_impact": 0.3,
                }
            ],
            BlackSwanType.STABLECOIN_DEPEG: [
                {
                    "date": "2022-05-09",
                    "description": "UST depeg and collapse",
                    "impact": -50.0,
                    "duration_hours": 120,
                    "contagion_effect": True,
                }
            ],
        }

    def generate_flash_crash_event(self, severity: float = 0.5) -> BlackSwanEvent:
        """Generate flash crash scenario"""
        try:
            # Scale impact based on severity
            base_impact = -15.0 * (0.5 + severity)  # -7.5% to -30%
            duration = 2 + severity * 6  # 2-8 hours

            # Affected coins (major coins get hit hardest)
            major_coins = ["BTC/USD", "ETH/USD", "BNB/USD", "ADA/USD"]
            alt_coins = ["LINK/USD", "DOT/USD", "MATIC/USD", "AVAX/USD"]

            price_impact = {}
            volume_impact = {}

            # Major coins impact
            for coin in major_coins:
                price_impact[coin] = base_impact * (0.8 + random.uniform(-0.2, 0.2))
                volume_impact[coin] = 3.0 + severity * 5.0  # 3x to 8x volume

            # Alt coins get hit harder
            for coin in alt_coins:
                price_impact[coin] = base_impact * (1.2 + random.uniform(-0.3, 0.3))
                volume_impact[coin] = 4.0 + severity * 6.0  # 4x to 10x volume

            return BlackSwanEvent(
                event_type=BlackSwanType.FLASH_CRASH,
                severity=severity,
                duration_hours=duration,
                affected_coins=major_coins + alt_coins,
                price_impact=price_impact,
                volume_impact=volume_impact,
                liquidity_impact=0.3 + severity * 0.4,  # 30-70% liquidity reduction
                description=f"Flash crash scenario with {severity * 100:.0f}% severity",
                probability=0.02,  # 2% annual probability
            )

        except Exception as e:
            self.logger.error(f"Flash crash generation failed: {e}")
            return self._create_minimal_event()

    def generate_exchange_hack_event(self, exchange_type: str = "major") -> BlackSwanEvent:
        """Generate exchange hack scenario"""
        try:
            if exchange_type == "major":
                severity = random.uniform(0.6, 1.0)
                base_impact = -10.0 * severity
                duration = 12 + severity * 36  # 12-48 hours
                probability = 0.05  # 5% annual
            else:
                severity = random.uniform(0.3, 0.7)
                base_impact = -5.0 * severity
                duration = 6 + severity * 18  # 6-24 hours
                probability = 0.1  # 10% annual

            affected_coins = ["BTC/USD", "ETH/USD", "USDT/USD", "BNB/USD"]

            price_impact = {
                coin: base_impact * (0.7 + random.uniform(-0.3, 0.3)) for coin in affected_coins
            }
            volume_impact = {coin: 2.0 + severity * 3.0 for coin in affected_coins}

            return BlackSwanEvent(
                event_type=BlackSwanType.EXCHANGE_HACK,
                severity=severity,
                duration_hours=duration,
                affected_coins=affected_coins,
                price_impact=price_impact,
                volume_impact=volume_impact,
                liquidity_impact=0.2 + severity * 0.3,
                description=f"{exchange_type.capitalize()} exchange hack scenario",
                probability=probability,
            )

        except Exception as e:
            self.logger.error(f"Exchange hack generation failed: {e}")
            return self._create_minimal_event()

    def generate_regulatory_ban_event(self, region: str = "major_country") -> BlackSwanEvent:
        """Generate regulatory ban scenario"""
        try:
            if region == "major_country":
                severity = random.uniform(0.7, 1.0)
                base_impact = -25.0 * severity
                duration = 72 + severity * 168  # 3-10 days
                probability = 0.03  # 3% annual
            else:
                severity = random.uniform(0.4, 0.7)
                base_impact = -10.0 * severity
                duration = 24 + severity * 72  # 1-4 days
                probability = 0.08  # 8% annual

            affected_coins = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

            price_impact = {
                coin: base_impact * (0.8 + random.uniform(-0.2, 0.2)) for coin in affected_coins
            }
            volume_impact = {coin: 1.5 + severity * 2.0 for coin in affected_coins}

            return BlackSwanEvent(
                event_type=BlackSwanType.REGULATORY_BAN,
                severity=severity,
                duration_hours=duration,
                affected_coins=affected_coins,
                price_impact=price_impact,
                volume_impact=volume_impact,
                liquidity_impact=0.4 + severity * 0.4,
                description=f"Regulatory ban in {region}",
                probability=probability,
            )

        except Exception as e:
            self.logger.error(f"Regulatory ban generation failed: {e}")
            return self._create_minimal_event()

    def generate_whale_dump_event(self, coin: str = "BTC/USD") -> BlackSwanEvent:
        """Generate whale dump scenario"""
        try:
            severity = random.uniform(0.4, 0.8)
            base_impact = -12.0 * severity
            duration = 1 + severity * 4  # 1-5 hours

            affected_coins = [coin]
            if coin in ["BTC/USD", "ETH/USD"]:
                # Major coins affect others
                affected_coins.extend(["BNB/USD", "ADA/USD", "LINK/USD"])

            price_impact = {}
            volume_impact = {}

            # Primary coin takes biggest hit
            price_impact[coin] = base_impact
            volume_impact[coin] = 5.0 + severity * 10.0

            # Secondary coins get smaller impact
            for other_coin in affected_coins[1:]:
                price_impact[other_coin] = base_impact * 0.3
                volume_impact[other_coin] = 2.0 + severity * 3.0

            return BlackSwanEvent(
                event_type=BlackSwanType.WHALE_DUMP,
                severity=severity,
                duration_hours=duration,
                affected_coins=affected_coins,
                price_impact=price_impact,
                volume_impact=volume_impact,
                liquidity_impact=0.6 + severity * 0.3,
                description=f"Large whale dump of {coin}",
                probability=0.1,  # 10% annual for any major coin
            )

        except Exception as e:
            self.logger.error(f"Whale dump generation failed: {e}")
            return self._create_minimal_event()

    def generate_chain_fork_event(self, chain: str = "ethereum") -> BlackSwanEvent:
        """Generate blockchain fork scenario"""
        try:
            severity = random.uniform(0.5, 0.9)
            base_impact = -20.0 * severity
            duration = 24 + severity * 72  # 1-4 days

            if chain == "ethereum":
                affected_coins = ["ETH/USD", "LINK/USD", "UNI/USD", "MATIC/USD"]
            elif chain == "bitcoin":
                affected_coins = ["BTC/USD"]
            else:
                affected_coins = ["BNB/USD", "ADA/USD", "DOT/USD"]

            price_impact = {
                coin: base_impact * (0.7 + random.uniform(-0.2, 0.3)) for coin in affected_coins
            }
            volume_impact = {coin: 3.0 + severity * 4.0 for coin in affected_coins}

            return BlackSwanEvent(
                event_type=BlackSwanType.CHAIN_FORK,
                severity=severity,
                duration_hours=duration,
                affected_coins=affected_coins,
                price_impact=price_impact,
                volume_impact=volume_impact,
                liquidity_impact=0.5 + severity * 0.3,
                description=f"{chain.capitalize()} blockchain fork",
                probability=0.01,  # 1% annual per major chain
            )

        except Exception as e:
            self.logger.error(f"Chain fork generation failed: {e}")
            return self._create_minimal_event()

    def _create_minimal_event(self) -> BlackSwanEvent:
        """Create minimal black swan event as fallback"""
        return BlackSwanEvent(
            event_type=BlackSwanType.FLASH_CRASH,
            severity=0.3,
            duration_hours=2.0,
            affected_coins=["BTC/USD"],
            price_impact={"BTC/USD": -5.0},
            volume_impact={"BTC/USD": 2.0},
            liquidity_impact=0.2,
            description="Minimal flash crash event",
            probability=0.05,
        )


class StressTestSimulator:
    """Simulate black swan events and test system responses"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.event_generator = BlackSwanEventGenerator(config_manager)

    def simulate_event_impact(
        self,
        event: BlackSwanEvent,
        initial_portfolio: Dict[str, float],
        market_condition: MarketCondition = MarketCondition.BULL_MARKET,
    ) -> StressTestResult:
        """Simulate the impact of a black swan event"""
        try:
            # Calculate portfolio impact
            portfolio_impact = self._calculate_portfolio_impact(event, initial_portfolio)

            # Calculate maximum drawdown
            max_drawdown = self._simulate_drawdown(event, market_condition)

            # Estimate recovery time
            recovery_time = self._estimate_recovery_time(event, market_condition)

            # Simulate model performance degradation
            model_degradation = self._simulate_model_degradation(event)

            # Identify system failures
            system_failures = self._identify_system_failures(event)

            # Assess risk mitigation effectiveness
            risk_mitigation = self._assess_risk_mitigation(event, portfolio_impact)

            # Generate lessons learned
            lessons_learned = self._generate_lessons_learned(event, portfolio_impact)

            return StressTestResult(
                event_type=event.event_type,
                severity=event.severity,
                market_condition=market_condition,
                portfolio_impact=portfolio_impact,
                max_drawdown=max_drawdown,
                recovery_time_hours=recovery_time,
                model_accuracy_degradation=model_degradation,
                system_failures=system_failures,
                risk_mitigation_effectiveness=risk_mitigation,
                lessons_learned=lessons_learned,
            )

        except Exception as e:
            self.logger.error(f"Event simulation failed: {e}")
            return self._create_minimal_result(event)

    def _calculate_portfolio_impact(
        self, event: BlackSwanEvent, portfolio: Dict[str, float]
    ) -> float:
        """Calculate total portfolio impact from event"""
        try:
            total_impact = 0.0
            total_value = sum(portfolio.values())

            if total_value == 0:
                return 0.0

            for coin, position_value in portfolio.items():
                if coin in event.price_impact:
                    coin_impact = event.price_impact[coin] / 100.0  # Convert percentage
                    weighted_impact = (position_value / total_value) * coin_impact
                    total_impact += weighted_impact

            return total_impact

        except Exception as e:
            self.logger.warning(f"Portfolio impact calculation failed: {e}")
            return -0.1  # Conservative 10% loss estimate

    def _simulate_drawdown(self, event: BlackSwanEvent, market_condition: MarketCondition) -> float:
        """Simulate maximum drawdown during event"""
        try:
            base_drawdown = abs(sum(event.price_impact.values()) / len(event.price_impact))

            # Adjust for market conditions
            condition_multipliers = {
                MarketCondition.BULL_MARKET: 0.8,
                MarketCondition.BEAR_MARKET: 1.5,
                MarketCondition.SIDEWAYS_MARKET: 1.0,
                MarketCondition.HIGH_VOLATILITY: 1.3,
                MarketCondition.LOW_LIQUIDITY: 1.8,
            }

            multiplier = condition_multipliers.get(market_condition, 1.0)
            max_drawdown = base_drawdown * multiplier * (1 + event.severity * 0.5)

            return min(100.0, max_drawdown)  # Cap at 100%

        except Exception:
            return 20.0  # Conservative 20% drawdown

    def _estimate_recovery_time(
        self, event: BlackSwanEvent, market_condition: MarketCondition
    ) -> float:
        """Estimate time to recover from event"""
        try:
            base_recovery = event.duration_hours * 2  # 2x event duration

            # Adjust for event severity
            severity_multiplier = 1 + event.severity * 2

            # Adjust for market conditions
            condition_multipliers = {
                MarketCondition.BULL_MARKET: 0.5,
                MarketCondition.BEAR_MARKET: 3.0,
                MarketCondition.SIDEWAYS_MARKET: 1.5,
                MarketCondition.HIGH_VOLATILITY: 2.0,
                MarketCondition.LOW_LIQUIDITY: 4.0,
            }

            condition_multiplier = condition_multipliers.get(market_condition, 1.5)

            recovery_time = base_recovery * severity_multiplier * condition_multiplier

            # Event-specific adjustments
            if event.event_type == BlackSwanType.REGULATORY_BAN:
                recovery_time *= 5.0  # Regulatory issues take longer
            elif event.event_type == BlackSwanType.FLASH_CRASH:
                recovery_time *= 0.3  # Flash crashes recover faster

            return max(event.duration_hours, recovery_time)

        except Exception:
            return 72.0  # Conservative 3-day recovery

    def _simulate_model_degradation(self, event: BlackSwanEvent) -> float:
        """Simulate model accuracy degradation during event"""
        try:
            # Base degradation depends on event type
            base_degradations = {
                BlackSwanType.FLASH_CRASH: 0.3,
                BlackSwanType.EXCHANGE_HACK: 0.2,
                BlackSwanType.REGULATORY_BAN: 0.4,
                BlackSwanType.CHAIN_FORK: 0.35,
                BlackSwanType.WHALE_DUMP: 0.25,
                BlackSwanType.STABLECOIN_DEPEG: 0.5,
                BlackSwanType.EXCHANGE_OUTAGE: 0.15,
            }

            base_degradation = base_degradations.get(event.event_type, 0.25)
            severity_factor = 1 + event.severity

            degradation = base_degradation * severity_factor

            return min(0.8, degradation)  # Cap at 80% degradation

        except Exception:
            return 0.3  # Conservative 30% degradation

    def _identify_system_failures(self, event: BlackSwanEvent) -> List[str]:
        """Identify potential system failures during event"""
        try:
            failures = []

            # High volume events may cause API failures
            if any(vol > 5.0 for vol in event.volume_impact.values()):
                failures.append("Exchange API rate limiting")
                failures.append("Data feed latency spikes")

            # High severity events may cause model failures
            if event.severity > 0.7:
                failures.append("Model prediction confidence drop")
                failures.append("Risk management system alerts")

            # Specific event types have known failure modes
            if event.event_type == BlackSwanType.EXCHANGE_OUTAGE:
                failures.extend(["Trading execution failures", "Portfolio synchronization issues"])
            elif event.event_type == BlackSwanType.FLASH_CRASH:
                failures.extend(["Stop-loss order slippage", "Liquidity provider disconnections"])
            elif event.event_type == BlackSwanType.CHAIN_FORK:
                failures.extend(["Transaction confirmation delays", "Wallet balance discrepancies"])

            # Liquidity issues
            if event.liquidity_impact > 0.5:
                failures.append("Order execution delays")
                failures.append("Bid-ask spread widening")

            return failures

        except Exception:
            return ["Unknown system stress"]

    def _assess_risk_mitigation(self, event: BlackSwanEvent, portfolio_impact: float) -> float:
        """Assess effectiveness of risk mitigation"""
        try:
            # Calculate theoretical impact without mitigation
            max_theoretical_impact = abs(max(event.price_impact.values())) / 100.0

            # Actual impact vs theoretical impact
            if max_theoretical_impact > 0:
                mitigation_effectiveness = 1.0 - (abs(portfolio_impact) / max_theoretical_impact)
                return max(0.0, min(1.0, mitigation_effectiveness))
            else:
                return 0.5  # Neutral

        except Exception:
            return 0.3  # Conservative mitigation effectiveness

    def _generate_lessons_learned(
        self, event: BlackSwanEvent, portfolio_impact: float
    ) -> List[str]:
        """Generate lessons learned from simulation"""
        try:
            lessons = []

            # Impact-based lessons
            if abs(portfolio_impact) > 0.2:
                lessons.append("Consider increasing portfolio diversification")
                lessons.append("Review position sizing limits")

            if abs(portfolio_impact) > 0.1:
                lessons.append("Implement additional hedging strategies")

            # Event-specific lessons
            if event.event_type == BlackSwanType.FLASH_CRASH:
                lessons.extend(
                    ["Review stop-loss order placement", "Consider implementing circuit breakers"]
                )
            elif event.event_type == BlackSwanType.EXCHANGE_HACK:
                lessons.extend(
                    ["Diversify across multiple exchanges", "Implement cold storage protocols"]
                )
            elif event.event_type == BlackSwanType.REGULATORY_BAN:
                lessons.extend(
                    ["Monitor regulatory developments closely", "Prepare compliance procedures"]
                )

            # Severity-based lessons
            if event.severity > 0.7:
                lessons.append("Develop crisis management procedures")
                lessons.append("Stress test models more frequently")

            return lessons

        except Exception:
            return ["Review and improve risk management procedures"]

    def _create_minimal_result(self, event: BlackSwanEvent) -> StressTestResult:
        """Create minimal stress test result as fallback"""
        return StressTestResult(
            event_type=event.event_type,
            severity=event.severity,
            market_condition=MarketCondition.SIDEWAYS_MARKET,
            portfolio_impact=-0.1,
            max_drawdown=15.0,
            recovery_time_hours=24.0,
            model_accuracy_degradation=0.2,
            system_failures=["Simulation error"],
            risk_mitigation_effectiveness=0.5,
            lessons_learned=["Review simulation system"],
        )


class BlackSwanCoordinator:
    """Main coordinator for black swan simulation and stress testing"""

    def __init__(self, config_manager=None, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.event_generator = BlackSwanEventGenerator(config_manager)
        self.stress_simulator = StressTestSimulator(config_manager)

        # Simulation history
        self.simulation_history = []

        self.logger.info("Black Swan Simulation Coordinator initialized")

    def run_comprehensive_stress_test(
        self, portfolio: Dict[str, float], scenarios: Optional[List[BlackSwanType]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive stress test across multiple scenarios"""
        try:
            if scenarios is None:
                scenarios = list(BlackSwanType)

            stress_test_results = {
                "portfolio": portfolio,
                "total_scenarios": len(scenarios),
                "scenario_results": [],
                "summary_statistics": {},
                "risk_recommendations": [],
                "timestamp": datetime.now().isoformat(),
            }

            results = []

            for scenario_type in scenarios:
                # Generate event
                if scenario_type == BlackSwanType.FLASH_CRASH:
                    event = self.event_generator.generate_flash_crash_event()
                elif scenario_type == BlackSwanType.EXCHANGE_HACK:
                    event = self.event_generator.generate_exchange_hack_event()
                elif scenario_type == BlackSwanType.REGULATORY_BAN:
                    event = self.event_generator.generate_regulatory_ban_event()
                elif scenario_type == BlackSwanType.WHALE_DUMP:
                    event = self.event_generator.generate_whale_dump_event()
                elif scenario_type == BlackSwanType.CHAIN_FORK:
                    event = self.event_generator.generate_chain_fork_event()
                else:
                    # Default to flash crash for other types
                    event = self.event_generator.generate_flash_crash_event()

                # Simulate across different market conditions
                for market_condition in [MarketCondition.BULL_MARKET, MarketCondition.BEAR_MARKET]:
                    result = self.stress_simulator.simulate_event_impact(
                        event, portfolio, market_condition
                    )

                    scenario_result = {
                        "event_type": scenario_type.value,
                        "market_condition": market_condition.value,
                        "event_details": asdict(event),
                        "stress_result": asdict(result),
                    }

                    stress_test_results["scenario_results"].append(scenario_result)
                    results.append(result)

            # Calculate summary statistics
            stress_test_results["summary_statistics"] = self._calculate_summary_statistics(results)

            # Generate risk recommendations
            stress_test_results["risk_recommendations"] = self._generate_risk_recommendations(
                results
            )

            # Store in history
            self.simulation_history.append(stress_test_results)

            return stress_test_results

        except Exception as e:
            self.logger.error(f"Comprehensive stress test failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def simulate_custom_scenario(
        self, event_params: Dict[str, Any], portfolio: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate custom black swan scenario"""
        try:
            # Create custom event
            event_type = BlackSwanType(event_params.get("event_type", "flash_crash"))
            severity = event_params.get("severity", 0.5)

            if event_type == BlackSwanType.FLASH_CRASH:
                event = self.event_generator.generate_flash_crash_event(severity)
            elif event_type == BlackSwanType.EXCHANGE_HACK:
                exchange_type = event_params.get("exchange_type", "major")
                event = self.event_generator.generate_exchange_hack_event(exchange_type)
            elif event_type == BlackSwanType.WHALE_DUMP:
                coin = event_params.get("coin", "BTC/USD")
                event = self.event_generator.generate_whale_dump_event(coin)
            else:
                event = self.event_generator.generate_flash_crash_event(severity)

            # Simulate impact
            market_condition = MarketCondition(event_params.get("market_condition", "bull_market"))
            result = self.stress_simulator.simulate_event_impact(event, portfolio, market_condition)

            return {
                "custom_scenario": True,
                "event_details": asdict(event),
                "simulation_result": asdict(result),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Custom scenario simulation failed: {e}")
            return {"error": str(e)}

    def _calculate_summary_statistics(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all stress test results"""
        try:
            if not results:
                return {}

            portfolio_impacts = [r.portfolio_impact for r in results]
            max_drawdowns = [r.max_drawdown for r in results]
            recovery_times = [r.recovery_time_hours for r in results]
            model_degradations = [r.model_accuracy_degradation for r in results]

            return {
                "portfolio_impact": {
                    "mean": float(np.mean(portfolio_impacts)),
                    "std": float(np.std(portfolio_impacts)),
                    "min": float(np.min(portfolio_impacts)),
                    "max": float(np.max(portfolio_impacts)),
                    "percentile_95": float(np.percentile(portfolio_impacts, 95)),
                },
                "max_drawdown": {
                    "mean": float(np.mean(max_drawdowns)),
                    "std": float(np.std(max_drawdowns)),
                    "max": float(np.max(max_drawdowns)),
                },
                "recovery_time_hours": {
                    "mean": float(np.mean(recovery_times)),
                    "median": float(np.median(recovery_times)),
                    "max": float(np.max(recovery_times)),
                },
                "model_degradation": {
                    "mean": float(np.mean(model_degradations)),
                    "max": float(np.max(model_degradations)),
                },
                "worst_case_scenario": self._identify_worst_case(results),
                "total_simulations": len(results),
            }

        except Exception as e:
            self.logger.warning(f"Summary statistics calculation failed: {e}")
            return {"error": str(e)}

    def _identify_worst_case(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """Identify worst-case scenario from results"""
        try:
            if not results:
                return {}

            # Find worst by portfolio impact
            worst_by_impact = min(results, key=lambda r: r.portfolio_impact)

            # Find worst by drawdown
            worst_by_drawdown = max(results, key=lambda r: r.max_drawdown)

            return {
                "worst_portfolio_impact": {
                    "event_type": worst_by_impact.event_type.value,
                    "impact": worst_by_impact.portfolio_impact,
                    "market_condition": worst_by_impact.market_condition.value,
                },
                "worst_drawdown": {
                    "event_type": worst_by_drawdown.event_type.value,
                    "drawdown": worst_by_drawdown.max_drawdown,
                    "market_condition": worst_by_drawdown.market_condition.value,
                },
            }

        except Exception:
            return {}

    def _generate_risk_recommendations(self, results: List[StressTestResult]) -> List[str]:
        """Generate risk management recommendations based on stress test results"""
        try:
            recommendations = []

            if not results:
                return ["No simulation results available"]

            # Analyze worst-case impacts
            worst_impact = min(r.portfolio_impact for r in results)
            worst_drawdown = max(r.max_drawdown for r in results)

            if worst_impact < -0.3:
                recommendations.append(
                    "Portfolio at risk of >30% loss in extreme scenarios - consider position limits"
                )

            if worst_drawdown > 50:
                recommendations.append(
                    "Maximum drawdown exceeds 50% - implement stop-loss mechanisms"
                )

            # Analyze recovery times
            max_recovery = max(r.recovery_time_hours for r in results)
            if max_recovery > 168:  # 1 week
                recommendations.append("Recovery times exceed 1 week - prepare liquidity reserves")

            # Analyze model degradation
            max_degradation = max(r.model_accuracy_degradation for r in results)
            if max_degradation > 0.5:
                recommendations.append(
                    "Model accuracy degrades >50% in stress - implement fallback systems"
                )

            # Event-specific recommendations
            event_types = [r.event_type for r in results]
            if BlackSwanType.EXCHANGE_HACK in event_types:
                recommendations.append(
                    "Diversify across multiple exchanges and implement cold storage"
                )

            if BlackSwanType.REGULATORY_BAN in event_types:
                recommendations.append(
                    "Monitor regulatory landscape and prepare compliance procedures"
                )

            # System failure analysis
            all_failures = []
            for result in results:
                all_failures.extend(result.system_failures)

            if "Exchange API rate limiting" in all_failures:
                recommendations.append("Implement backup data sources and API rate limiting")

            if not recommendations:
                recommendations.append(
                    "Current risk profile appears acceptable under stress scenarios"
                )

            return recommendations

        except Exception as e:
            return [f"Error generating recommendations: {e}"]

    def get_simulation_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        try:
            return {
                "total_simulations": len(self.simulation_history),
                "recent_simulations": self.simulation_history[-5:]
                if self.simulation_history
                else [],
                "available_scenarios": [scenario.value for scenario in BlackSwanType],
                "system_status": {
                    "event_generator_active": self.event_generator is not None,
                    "stress_simulator_active": self.stress_simulator is not None,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Convenience function
def get_black_swan_coordinator(config_manager=None, cache_manager=None) -> BlackSwanCoordinator:
    """Get configured black swan coordinator"""
    return BlackSwanCoordinator(config_manager, cache_manager)


if __name__ == "__main__":
    # Test the black swan simulation engine
    coordinator = get_black_swan_coordinator()

    print("Testing Black Swan Simulation Engine...")

    # Test portfolio
    test_portfolio = {
        "BTC/USD": 50000.0,
        "ETH/USD": 30000.0,
        "BNB/USD": 10000.0,
        "ADA/USD": 5000.0,
        "LINK/USD": 5000.0,
    }

    # Run comprehensive stress test
    stress_results = coordinator.run_comprehensive_stress_test(
        test_portfolio,
        [BlackSwanType.FLASH_CRASH, BlackSwanType.EXCHANGE_HACK, BlackSwanType.WHALE_DUMP],
    )

    print(f"\nStress Test Results:")
    print(f"  Total Scenarios: {stress_results['total_scenarios']}")
    print(f"  Scenario Results: {len(stress_results['scenario_results'])}")

    if "summary_statistics" in stress_results:
        stats = stress_results["summary_statistics"]
        if "portfolio_impact" in stats:
            print(f"  Worst Portfolio Impact: {stats['portfolio_impact']['min']:.2%}")
            print(f"  Average Impact: {stats['portfolio_impact']['mean']:.2%}")

        if "max_drawdown" in stats:
            print(f"  Maximum Drawdown: {stats['max_drawdown']['max']:.1f}%")

    print(f"  Risk Recommendations: {len(stress_results.get('risk_recommendations', []))}")

    # Test custom scenario
    custom_result = coordinator.simulate_custom_scenario(
        {"event_type": "flash_crash", "severity": 0.8, "market_condition": "bear_market"},
        test_portfolio,
    )

    print(f"\nCustom Scenario Result:")
    if "simulation_result" in custom_result:
        result = custom_result["simulation_result"]
        print(f"  Portfolio Impact: {result['portfolio_impact']:.2%}")
        print(f"  Max Drawdown: {result['max_drawdown']:.1f}%")
        print(f"  Recovery Time: {result['recovery_time_hours']:.1f} hours")

    print("Black swan simulation test completed")
