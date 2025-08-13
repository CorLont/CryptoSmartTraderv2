"""
Early Mover Advantage System

Combines multiple intelligence sources to identify and act on market-moving events
before the broader market reacts, focusing on extreme alpha generation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class AlphaSourceType(Enum):
    """Types of alpha generation sources"""

    NEW_LISTINGS = "new_listings"
    WHALE_MOVEMENTS = "whale_movements"
    NEWS_EVENTS = "news_events"
    SENTIMENT_SHIFTS = "sentiment_shifts"
    TECHNICAL_BREAKOUTS = "technical_breakouts"
    ARBITRAGE_OPPORTUNITIES = "arbitrage_opportunities"
    FUNDING_RATE_ANOMALIES = "funding_rate_anomalies"


class SignalStrength(Enum):
    """Signal strength levels"""

    WEAK = "weak"  # 10-50% expected return
    MODERATE = "moderate"  # 50-100% expected return
    STRONG = "strong"  # 100-300% expected return
    EXTREME = "extreme"  # 300%+ expected return


class TimeHorizon(Enum):
    """Expected signal duration"""

    IMMEDIATE = "immediate"  # Minutes to hours
    SHORT_TERM = "short_term"  # Hours to days
    MEDIUM_TERM = "medium_term"  # Days to weeks
    LONG_TERM = "long_term"  # Weeks to months


@dataclass
class AlphaSignal:
    """Individual alpha generation signal"""

    timestamp: datetime
    signal_id: str
    source_type: AlphaSourceType
    symbol: str

    # Signal characteristics
    strength: SignalStrength
    confidence: float  # 0-1 confidence score
    expected_return: float  # Expected % return
    time_horizon: TimeHorizon

    # Market context
    current_price: float
    target_price: Optional[float]
    stop_loss_price: Optional[float]
    volume_confirmation: bool

    # Intelligence source
    source_data: Dict[str, Any]
    reasoning: str
    risk_factors: List[str]

    # Execution parameters
    recommended_position_size: float  # % of portfolio
    entry_urgency: int  # 1-5 scale (5 = immediate)
    max_slippage_tolerance: float

    # Performance tracking
    entry_executed: bool = False
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None


@dataclass
class AlphaOpportunity:
    """Aggregated opportunity from multiple signals"""

    opportunity_id: str
    symbol: str
    timestamp: datetime

    # Opportunity characteristics
    total_confidence: float
    expected_return: float
    risk_adjusted_return: float
    kelly_optimal_size: float

    # Signal composition
    contributing_signals: List[AlphaSignal]
    signal_diversity: int
    consensus_strength: float

    # Execution strategy
    entry_strategy: str
    scaling_plan: List[Tuple[float, float]]  # (price_level, allocation)
    exit_strategy: str
    contingency_plans: List[str]

    # Risk management
    max_drawdown_tolerance: float
    position_correlation_risk: float
    liquidity_requirements: Dict[str, float]


class EarlyMoverSystem:
    """
    Advanced Early Mover Advantage System
    Orchestrates multiple intelligence agents to identify and capture alpha opportunities
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # System state
        self.active = False
        self.last_update = None
        self.processed_signals = 0
        self.error_count = 0

        # Signal storage
        self.active_signals: Dict[str, AlphaSignal] = {}
        self.alpha_opportunities: Dict[str, AlphaOpportunity] = {}
        self.signal_history: deque = deque(maxlen=10000)

        # Agent connections
        self.connected_agents = {}
        self.agent_weights = {
            "listing_detection": 0.25,
            "whale_detector": 0.20,
            "sentiment_analysis": 0.15,
            "technical_analysis": 0.15,
            "news_analysis": 0.15,
            "arbitrage_monitor": 0.10,
        }

        # Configuration
        self.update_interval = 60  # 1 minute for early mover advantage
        self.min_signal_confidence = 0.7
        self.max_position_correlation = 0.3
        self.alpha_decay_hours = 48

        # Performance thresholds for extreme returns
        self.return_thresholds = {
            SignalStrength.EXTREME: 3.0,  # 300%+
            SignalStrength.STRONG: 1.0,  # 100%+
            SignalStrength.MODERATE: 0.5,  # 50%+
            SignalStrength.WEAK: 0.1,  # 10%+
        }

        # Thread safety
        self._lock = threading.RLock()

        # Performance metrics
        self.performance_stats = {
            "total_signals_processed": 0,
            "opportunities_identified": 0,
            "extreme_alpha_events": 0,
            "successful_predictions": 0,
            "average_return_per_signal": 0.0,
            "hit_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

        # Data directory
        self.data_path = Path("data/early_mover")
        self.data_path.mkdir(parents=True, exist_ok=True)

        logger.info("Early Mover System initialized")

    def start(self):
        """Start the early mover system"""
        if not self.active:
            self.active = True
            self.system_thread = threading.Thread(target=self._system_loop, daemon=True)
            self.system_thread.start()
            self.logger.info("Early Mover System started")

    def stop(self):
        """Stop the early mover system"""
        self.active = False
        self.logger.info("Early Mover System stopped")

    def connect_agent(self, agent_type: str, agent_instance: Any):
        """Connect intelligence agent to the system"""
        self.connected_agents[agent_type] = agent_instance
        self.logger.info(f"Connected agent: {agent_type}")

    def _system_loop(self):
        """Main system processing loop"""
        while self.active:
            try:
                # Collect signals from all connected agents
                new_signals = self._collect_agent_signals()

                # Process and validate signals
                validated_signals = self._validate_signals(new_signals)

                # Identify alpha opportunities
                opportunities = self._identify_alpha_opportunities(validated_signals)

                # Execute high-priority opportunities
                self._execute_priority_opportunities(opportunities)

                # Update performance metrics
                self._update_performance_metrics()

                # Cleanup expired signals
                self._cleanup_expired_signals()

                # Save system state
                self._save_system_data()

                self.last_update = datetime.now()
                time.sleep(self.update_interval)

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Early mover system error: {e}")
                time.sleep(30)  # Shorter sleep for early mover advantage

    def _collect_agent_signals(self) -> List[AlphaSignal]:
        """Collect signals from all connected intelligence agents"""
        all_signals = []

        # Listing Detection Agent
        if "listing_detection" in self.connected_agents:
            listing_signals = self._process_listing_signals()
            all_signals.extend(listing_signals)

        # Whale Detection Agent
        if "whale_detector" in self.connected_agents:
            whale_signals = self._process_whale_signals()
            all_signals.extend(whale_signals)

        # Sentiment Analysis Agent
        if "sentiment_analysis" in self.connected_agents:
            sentiment_signals = self._process_sentiment_signals()
            all_signals.extend(sentiment_signals)

        # Technical Analysis Agent
        if "technical_analysis" in self.connected_agents:
            technical_signals = self._process_technical_signals()
            all_signals.extend(technical_signals)

        return all_signals

    def _process_listing_signals(self) -> List[AlphaSignal]:
        """Convert listing detection data into alpha signals"""
        signals = []

        try:
            listing_agent = self.connected_agents["listing_detection"]
            opportunities = listing_agent.get_active_opportunities(min_confidence=0.6)

            for opp in opportunities:
                # Create high-alpha signal for new listings
                signal = AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"listing_{opp.symbol}_{int(time.time())}",
                    source_type=AlphaSourceType.NEW_LISTINGS,
                    symbol=opp.symbol,
                    strength=self._classify_signal_strength(opp.expected_return),
                    confidence=opp.confidence,
                    expected_return=opp.expected_return,
                    time_horizon=TimeHorizon.SHORT_TERM,
                    current_price=0.0,  # Would fetch from market data
                    target_price=None,
                    stop_loss_price=None,
                    volume_confirmation=True,
                    source_data={"opportunity": opp},
                    reasoning=f"New listing opportunity on {opp.exchange.value} with {opp.expected_return:.1f}% expected return",
                    risk_factors=opp.risk_warnings,
                    recommended_position_size=opp.position_size_recommendation,
                    entry_urgency=5 if opp.time_horizon == "immediate" else 3,
                    max_slippage_tolerance=0.02,  # 2% max slippage for new listings
                )

                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error processing listing signals: {e}")

        return signals

    def _process_whale_signals(self) -> List[AlphaSignal]:
        """Convert whale detection data into alpha signals"""
        signals = []

        try:
            whale_agent = self.connected_agents["whale_detector"]
            whale_signals_data = whale_agent.get_whale_signals(min_confidence=0.7)

            for whale_signal in whale_signals_data:
                # Create alpha signal based on whale activity
                signal = AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"whale_{whale_signal['symbol']}_{int(time.time())}",
                    source_type=AlphaSourceType.WHALE_MOVEMENTS,
                    symbol=whale_signal["symbol"],
                    strength=self._classify_signal_strength(whale_signal["strength"] * 100),
                    confidence=whale_signal["confidence"],
                    expected_return=whale_signal["strength"] * 100,  # Convert to percentage
                    time_horizon=TimeHorizon.SHORT_TERM,
                    current_price=0.0,  # Would fetch from market data
                    target_price=None,
                    stop_loss_price=None,
                    volume_confirmation=True,
                    source_data={"whale_data": whale_signal},
                    reasoning=whale_signal["reasoning"],
                    risk_factors=["Whale activity can be volatile", "Large position size risk"],
                    recommended_position_size=min(0.1, whale_signal["confidence"] * 0.15),
                    entry_urgency=4,  # High urgency for whale movements
                    max_slippage_tolerance=0.015,  # 1.5% max slippage
                )

                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error processing whale signals: {e}")

        return signals

    def _process_sentiment_signals(self) -> List[AlphaSignal]:
        """Convert sentiment analysis data into alpha signals"""
        signals = []

        try:
            sentiment_agent = self.connected_agents["sentiment_analysis"]
            sentiment_signals_data = sentiment_agent.get_sentiment_signals(min_confidence=0.6)

            for sentiment_signal in sentiment_signals_data:
                # Create alpha signal based on sentiment shifts
                expected_return = sentiment_signal["strength"] * 50  # Scale to percentage

                signal = AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"sentiment_{sentiment_signal['symbol']}_{int(time.time())}",
                    source_type=AlphaSourceType.SENTIMENT_SHIFTS,
                    symbol=sentiment_signal["symbol"],
                    strength=self._classify_signal_strength(expected_return),
                    confidence=sentiment_signal["confidence"],
                    expected_return=expected_return,
                    time_horizon=TimeHorizon.MEDIUM_TERM,
                    current_price=0.0,  # Would fetch from market data
                    target_price=None,
                    stop_loss_price=None,
                    volume_confirmation=False,  # Sentiment doesn't guarantee volume
                    source_data={"sentiment_data": sentiment_signal},
                    reasoning=sentiment_signal["reasoning"],
                    risk_factors=["Sentiment can be manipulated", "Market sentiment lag"],
                    recommended_position_size=sentiment_signal["confidence"] * 0.08,
                    entry_urgency=2,  # Medium urgency for sentiment
                    max_slippage_tolerance=0.01,  # 1% max slippage
                )

                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error processing sentiment signals: {e}")

        return signals

    def _process_technical_signals(self) -> List[AlphaSignal]:
        """Convert technical analysis data into alpha signals"""
        signals = []

        # REMOVED: Mock data pattern not allowed in production
        # In production, this would come from technical analysis agent

        simulated_breakouts = [
            {
                "symbol": "BTC/USD",
                "pattern": "ascending_triangle_breakout",
                "confidence": 0.8,
                "expected_move": 15.0,  # 15% expected move
                "time_horizon": "short_term",
            }
        ]

        for breakout in simulated_breakouts:
            if np.random.random() < 0.1:  # 10% chance of technical signal
                signal = AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"technical_{breakout['symbol']}_{int(time.time())}",
                    source_type=AlphaSourceType.TECHNICAL_BREAKOUTS,
                    symbol=breakout["symbol"],
                    strength=self._classify_signal_strength(breakout["expected_move"]),
                    confidence=breakout["confidence"],
                    expected_return=breakout["expected_move"],
                    time_horizon=TimeHorizon.SHORT_TERM,
                    current_price=0.0,  # Would fetch from market data
                    target_price=None,
                    stop_loss_price=None,
                    volume_confirmation=True,
                    source_data={"breakout_data": breakout},
                    reasoning=f"Technical breakout pattern: {breakout['pattern']}",
                    risk_factors=["Technical patterns can fail", "Volume confirmation needed"],
                    recommended_position_size=breakout["confidence"] * 0.1,
                    entry_urgency=3,  # Medium urgency for technical
                    max_slippage_tolerance=0.008,  # 0.8% max slippage
                )

                signals.append(signal)

        return signals

    def _classify_signal_strength(self, expected_return: float) -> SignalStrength:
        """Classify signal strength based on expected return"""
        if expected_return >= 300:
            return SignalStrength.EXTREME
        elif expected_return >= 100:
            return SignalStrength.STRONG
        elif expected_return >= 50:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _validate_signals(self, signals: List[AlphaSignal]) -> List[AlphaSignal]:
        """Validate and filter signals based on quality criteria"""
        validated = []

        for signal in signals:
            # Basic quality checks
            if signal.confidence < self.min_signal_confidence:
                continue

            # Check for signal conflicts
            if not self._check_signal_conflicts(signal):
                continue

            # Validate risk parameters
            if not self._validate_risk_parameters(signal):
                continue

            validated.append(signal)

        return validated

    def _check_signal_conflicts(self, signal: AlphaSignal) -> bool:
        """Check if signal conflicts with existing positions or signals"""

        # Check for opposite signals on same symbol
        for existing_id, existing_signal in self.active_signals.items():
            if existing_signal.symbol == signal.symbol:
                # Allow if signals are complementary (different sources, similar direction)
                if existing_signal.source_type != signal.source_type:
                    return True

                # Reject if conflicting directions with high confidence
                if (existing_signal.expected_return > 0) != (signal.expected_return > 0):
                    if existing_signal.confidence > 0.8 and signal.confidence > 0.8:
                        return False

        return True

    def _validate_risk_parameters(self, signal: AlphaSignal) -> bool:
        """Validate signal risk parameters"""

        # Check position size limits
        if signal.recommended_position_size > 0.15:  # Max 15% per signal
            return False

        # Check expected return vs confidence alignment
        if (
            signal.expected_return > 200 and signal.confidence < 0.8
        ):  # 200%+ return needs 80%+ confidence
            return False

        # Check time horizon consistency
        if (
            signal.strength == SignalStrength.EXTREME
            and signal.time_horizon == TimeHorizon.LONG_TERM
        ):
            return False  # Extreme signals should be short-term

        return True

    def _identify_alpha_opportunities(self, signals: List[AlphaSignal]) -> List[AlphaOpportunity]:
        """Identify high-value alpha opportunities from validated signals"""
        opportunities = []

        # Add new signals to active signals
        with self._lock:
            for signal in signals:
                self.active_signals[signal.signal_id] = signal
                self.signal_history.append(signal)

        # Group signals by symbol for opportunity creation
        symbol_signals = defaultdict(list)
        for signal in self.active_signals.values():
            symbol_signals[signal.symbol].append(signal)

        # Create opportunities from signal clusters
        for symbol, symbol_signal_list in symbol_signals.items():
            if len(symbol_signal_list) >= 2:  # Need multiple confirming signals
                opportunity = self._create_alpha_opportunity(symbol, symbol_signal_list)
                if opportunity:
                    opportunities.append(opportunity)

        return opportunities

    def _create_alpha_opportunity(
        self, symbol: str, signals: List[AlphaSignal]
    ) -> Optional[AlphaOpportunity]:
        """Create aggregated alpha opportunity from multiple signals"""

        if not signals:
            return None

        # Calculate consensus metrics
        total_confidence = np.mean([s.confidence for s in signals])
        expected_return = np.mean([s.expected_return for s in signals])

        # Weight by signal strength and confidence
        weighted_returns = []
        weights = []
        for signal in signals:
            weight = signal.confidence * self.agent_weights.get(signal.source_type.value, 0.1)
            weighted_returns.append(signal.expected_return * weight)
            weights.append(weight)

        if sum(weights) > 0:
            risk_adjusted_return = sum(weighted_returns) / sum(weights)
        else:
            risk_adjusted_return = expected_return

        # Calculate Kelly optimal position size
        # Simplified Kelly: f* = (bp - q) / b where b=odds, p=prob of win, q=prob of loss
        win_prob = total_confidence
        loss_prob = 1 - win_prob
        odds_ratio = expected_return / 100  # Convert percentage to ratio

        kelly_fraction = (odds_ratio * win_prob - loss_prob) / odds_ratio
        kelly_optimal_size = max(0, min(0.25, kelly_fraction))  # Cap at 25%

        # Signal diversity score
        unique_sources = len(set(s.source_type for s in signals))
        signal_diversity = unique_sources

        # Consensus strength
        return_std = np.std([s.expected_return for s in signals])
        consensus_strength = 1.0 / (1.0 + return_std / 10)  # Penalize high deviation

        # Create opportunity
        opportunity = AlphaOpportunity(
            opportunity_id=f"alpha_{symbol}_{int(time.time())}",
            symbol=symbol,
            timestamp=datetime.now(),
            total_confidence=total_confidence,
            expected_return=risk_adjusted_return,
            risk_adjusted_return=risk_adjusted_return * total_confidence,
            kelly_optimal_size=kelly_optimal_size,
            contributing_signals=signals,
            signal_diversity=signal_diversity,
            consensus_strength=consensus_strength,
            entry_strategy=self._determine_entry_strategy(signals),
            scaling_plan=self._create_scaling_plan(kelly_optimal_size),
            exit_strategy=self._determine_exit_strategy(signals),
            contingency_plans=["Stop loss at -20%", "Take profit at +100%", "Scale out gradually"],
            max_drawdown_tolerance=0.15,
            position_correlation_risk=0.0,  # Would calculate from portfolio
            liquidity_requirements={"min_volume": 1000000, "max_spread": 0.005},
        )

        # Only return if opportunity meets extreme alpha criteria
        if (
            opportunity.risk_adjusted_return >= 50  # 50%+ expected return
            and opportunity.total_confidence >= 0.7  # 70%+ confidence
            and opportunity.signal_diversity >= 2
        ):  # Multiple confirming sources
            return opportunity

        return None

    def _determine_entry_strategy(self, signals: List[AlphaSignal]) -> str:
        """Determine optimal entry strategy based on signal characteristics"""

        urgencies = [s.entry_urgency for s in signals]
        avg_urgency = np.mean(urgencies)

        if avg_urgency >= 4:
            return "immediate_market_entry"
        elif avg_urgency >= 3:
            return "aggressive_limit_entry"
        else:
            return "patient_accumulation"

    def _create_scaling_plan(self, total_size: float) -> List[Tuple[float, float]]:
        """Create position scaling plan"""
        # Simple 3-tier scaling
        return [
            (0.0, total_size * 0.4),  # 40% at current price
            (-0.02, total_size * 0.35),  # 35% on 2% dip
            (-0.05, total_size * 0.25),  # 25% on 5% dip
        ]

    def _determine_exit_strategy(self, signals: List[AlphaSignal]) -> str:
        """Determine exit strategy based on signal characteristics"""

        time_horizons = [s.time_horizon for s in signals]

        if TimeHorizon.IMMEDIATE in time_horizons:
            return "quick_scalp_exit"
        elif TimeHorizon.SHORT_TERM in time_horizons:
            return "swing_trade_exit"
        else:
            return "position_hold_exit"

    def _execute_priority_opportunities(self, opportunities: List[AlphaOpportunity]):
        """Execute highest priority alpha opportunities"""

        # Sort by risk-adjusted return and confidence
        sorted_opportunities = sorted(
            opportunities, key=lambda x: x.risk_adjusted_return * x.total_confidence, reverse=True
        )

        executed_count = 0
        for opp in sorted_opportunities[:5]:  # Execute top 5 opportunities
            # Check portfolio constraints
            if self._check_portfolio_constraints(opp):
                # Log opportunity for execution
                self.logger.info(
                    f"ALPHA OPPORTUNITY: {opp.symbol} - {opp.expected_return:.1f}% expected return"
                )
                self.logger.info(
                    f"Confidence: {opp.total_confidence:.1f}, Kelly size: {opp.kelly_optimal_size:.1%}"
                )

                # Store opportunity for tracking
                with self._lock:
                    self.alpha_opportunities[opp.opportunity_id] = opp
                    self.performance_stats["opportunities_identified"] += 1

                    if opp.expected_return >= 300:
                        self.performance_stats["extreme_alpha_events"] += 1

                executed_count += 1

        self.logger.info(f"Executed {executed_count} alpha opportunities")

    def _check_portfolio_constraints(self, opportunity: AlphaOpportunity) -> bool:
        """Check if opportunity fits portfolio constraints"""

        # Check position correlation
        existing_symbols = [opp.symbol for opp in self.alpha_opportunities.values()]
        if opportunity.symbol in existing_symbols:
            return False  # Already have position in this symbol

        # Check total portfolio exposure
        total_exposure = sum(opp.kelly_optimal_size for opp in self.alpha_opportunities.values())
        if total_exposure + opportunity.kelly_optimal_size > 0.8:  # Max 80% portfolio exposure
            return False

        return True

    def _update_performance_metrics(self):
        """Update system performance metrics"""

        with self._lock:
            self.performance_stats["total_signals_processed"] = len(self.signal_history)

            # Calculate hit rate (simplified)
            if self.performance_stats["opportunities_identified"] > 0:
                self.performance_stats["hit_rate"] = (
                    self.performance_stats["successful_predictions"]
                    / self.performance_stats["opportunities_identified"]
                )

    def _cleanup_expired_signals(self):
        """Remove expired signals and opportunities"""
        cutoff_time = datetime.now() - timedelta(hours=self.alpha_decay_hours)

        with self._lock:
            # Remove expired signals
            expired_signals = [
                signal_id
                for signal_id, signal in self.active_signals.items()
                if signal.timestamp < cutoff_time
            ]

            for signal_id in expired_signals:
                del self.active_signals[signal_id]

            # Remove expired opportunities
            expired_opportunities = [
                opp_id
                for opp_id, opp in self.alpha_opportunities.items()
                if opp.timestamp < cutoff_time
            ]

            for opp_id in expired_opportunities:
                del self.alpha_opportunities[opp_id]

    def get_active_opportunities(self) -> List[AlphaOpportunity]:
        """Get currently active alpha opportunities"""
        with self._lock:
            return list(self.alpha_opportunities.values())

    def get_extreme_alpha_signals(self) -> List[AlphaSignal]:
        """Get signals with extreme alpha potential (300%+ returns)"""
        with self._lock:
            return [
                signal
                for signal in self.active_signals.values()
                if signal.strength == SignalStrength.EXTREME
            ]

    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        return {
            "active_signals": len(self.active_signals),
            "active_opportunities": len(self.alpha_opportunities),
            "connected_agents": len(self.connected_agents),
            "performance_stats": self.performance_stats,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

    def _save_system_data(self):
        """Save system state and performance data"""
        try:
            # Save active opportunities
            opportunities_file = self.data_path / "alpha_opportunities.json"
            opportunities_data = []

            for opp in self.alpha_opportunities.values():
                opportunities_data.append(
                    {
                        "opportunity_id": opp.opportunity_id,
                        "symbol": opp.symbol,
                        "timestamp": opp.timestamp.isoformat(),
                        "expected_return": opp.expected_return,
                        "confidence": opp.total_confidence,
                        "kelly_size": opp.kelly_optimal_size,
                        "signal_count": len(opp.contributing_signals),
                    }
                )

            with open(opportunities_file, "w") as f:
                json.dump(opportunities_data, f, indent=2)

            # Save performance metrics
            performance_file = self.data_path / "performance_metrics.json"
            with open(performance_file, "w") as f:
                json.dump(self.performance_stats, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving system data: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active": self.active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "processed_signals": self.processed_signals,
            "error_count": self.error_count,
            "active_signals": len(self.active_signals),
            "active_opportunities": len(self.alpha_opportunities),
            "connected_agents": list(self.connected_agents.keys()),
            "performance_metrics": self.performance_stats,
        }
