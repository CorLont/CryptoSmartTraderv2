"""
Open Interest Tracker

Tracks open interest changes and divergences across exchanges
to identify institutional positioning and trend strength.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OITrend(Enum):
    """Open interest trend direction"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class OIDivergenceType(Enum):
    """Types of OI-price divergences"""

    BULLISH_DIVERGENCE = "bullish_divergence"  # Price down, OI up
    BEARISH_DIVERGENCE = "bearish_divergence"  # Price up, OI down
    CONFIRMING_TREND = "confirming_trend"  # Price and OI same direction
    NEUTRAL = "neutral"  # No clear divergence


class OIRegime(Enum):
    """Open interest regime classification"""

    ACCUMULATION = "accumulation"  # Rising OI, stable/rising price
    DISTRIBUTION = "distribution"  # Falling OI, stable/falling price
    BREAKOUT = "breakout"  # Rising OI, rising price
    BREAKDOWN = "breakdown"  # Rising OI, falling price
    CONSOLIDATION = "consolidation"  # Stable OI, stable price


@dataclass
class OIEvent:
    """Open interest event"""

    timestamp: datetime
    pair: str
    exchange: str

    # OI data
    current_oi: float
    previous_oi: float
    oi_change: float
    oi_change_pct: float

    # Price context
    price_at_event: float
    price_change_pct: float

    # Volume context
    volume_24h: float
    volume_oi_ratio: float  # Volume to OI ratio

    # Classification
    oi_trend: OITrend
    divergence_type: OIDivergenceType
    regime: OIRegime

    # Derived metrics
    oi_momentum: float = 0.0  # Rate of OI change acceleration
    institutional_flow: float = 0.0  # Estimated institutional positioning

    @property
    def is_significant_change(self) -> bool:
        """Check if OI change is significant"""
        return abs(self.oi_change_pct) > 0.05  # 5% threshold

    @property
    def strength_score(self) -> float:
        """Calculate event strength score (0-1)"""
        return min(1.0, abs(self.oi_change_pct) / 0.2)  # Max at 20% change


@dataclass
class OIDivergence:
    """OI-Price divergence event"""

    divergence_id: str
    timestamp: datetime
    pair: str
    exchange: str

    # Divergence details
    divergence_type: OIDivergenceType
    strength: float  # Divergence strength (0-1)
    duration_hours: float  # How long divergence has persisted

    # Data points
    price_change_pct: float
    oi_change_pct: float
    correlation: float  # Price-OI correlation over period

    # Context
    volume_during_divergence: float
    volatility_during_divergence: float

    # Predictions
    expected_resolution_direction: str  # "up", "down", "neutral"
    confidence: float
    time_to_resolution_hours: float

    @property
    def divergence_score(self) -> float:
        """Calculate overall divergence score"""
        return self.strength * (1 - abs(self.correlation)) * min(1.0, self.duration_hours / 24)


class OpenInterestTracker:
    """
    Advanced open interest tracking and analysis
    """

    def __init__(self):
        self.oi_history = {}  # exchange -> pair -> List[OIEvent]
        self.divergence_history = []

        # Configuration
        self.significant_change_threshold = 0.05  # 5%
        self.divergence_strength_threshold = 0.3  # 30%
        self.regime_window_hours = 24  # Hours for regime analysis

        # Pattern tracking
        self.regime_transitions = {}
        self.divergence_patterns = {}

        # Performance tracking
        self.signal_performance = {}

    def analyze_oi_data(
        self, exchange: str, pair: str, oi_data: List[Dict[str, Any]]
    ) -> List[OIEvent]:
        """Analyze open interest data and detect events"""
        try:
            events = []

            if not oi_data:
                return events

            # Sort by timestamp
            oi_data = sorted(oi_data, key=lambda x: x.get("timestamp", datetime.min))

            # Process each OI period
            for i, current_data in enumerate(oi_data):
                if i == 0:
                    continue  # Skip first record

                previous_data = oi_data[i - 1]

                # Create OI event
                event = self._create_oi_event(exchange, pair, current_data, previous_data)

                if event:
                    events.append(event)

                    # Store in history
                    self._store_oi_event(exchange, pair, event)

                    # Check for divergences
                    divergence = self._detect_oi_divergence(exchange, pair, event)
                    if divergence:
                        self.divergence_history.append(divergence)
                        logger.info(
                            f"OI divergence detected: {divergence.divergence_type.value} for {pair} on {exchange}"
                        )

            return events

        except Exception as e:
            logger.error(f"OI analysis failed for {exchange} {pair}: {e}")
            return []

    def get_current_oi_signals(
        self, exchange: str, pair: str, current_price: float
    ) -> Dict[str, Any]:
        """Get current OI-based trading signals"""
        try:
            # Get recent OI events
            recent_events = self._get_recent_oi_events(exchange, pair, hours_back=48)

            if not recent_events:
                return {"status": "no_data"}

            latest_event = recent_events[-1]

            # Generate signals based on OI analysis
            signals = self._generate_oi_signals(latest_event, recent_events, current_price)

            return signals

        except Exception as e:
            logger.error(f"OI signal generation failed: {e}")
            return {"status": "error", "error": str(e)}

    def detect_oi_flow_patterns(
        self, exchange: str, pair: str, lookback_hours: int = 168
    ) -> Dict[str, Any]:
        """Detect institutional flow patterns from OI data"""
        try:
            events = self._get_recent_oi_events(exchange, pair, hours_back=lookback_hours)

            if len(events) < 10:
                return {"status": "insufficient_data"}

            flow_analysis = {
                "pair": pair,
                "exchange": exchange,
                "analysis_period_hours": lookback_hours,
                "patterns": [],
            }

            # Detect accumulation/distribution patterns
            accumulation_periods = self._detect_accumulation_periods(events)
            distribution_periods = self._detect_distribution_periods(events)

            flow_analysis["patterns"].extend(accumulation_periods)
            flow_analysis["patterns"].extend(distribution_periods)

            # Calculate net institutional flow
            net_flow = self._calculate_net_institutional_flow(events)
            flow_analysis["net_institutional_flow"] = net_flow

            # Detect flow reversals
            flow_reversals = self._detect_flow_reversals(events)
            flow_analysis["flow_reversals"] = flow_reversals

            return flow_analysis

        except Exception as e:
            logger.error(f"OI flow pattern detection failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_cross_exchange_oi_analysis(self, pair: str, exchanges: List[str]) -> Dict[str, Any]:
        """Analyze OI patterns across exchanges"""
        try:
            cross_analysis = {
                "pair": pair,
                "timestamp": datetime.now(),
                "exchanges_analyzed": exchanges,
                "oi_data": {},
                "flow_divergences": [],
            }

            # Get latest OI for each exchange
            latest_oi = {}
            for exchange in exchanges:
                recent_events = self._get_recent_oi_events(exchange, pair, hours_back=1)
                if recent_events:
                    latest_event = recent_events[-1]
                    latest_oi[exchange] = {
                        "oi": latest_event.current_oi,
                        "oi_change_pct": latest_event.oi_change_pct,
                        "regime": latest_event.regime.value,
                        "timestamp": latest_event.timestamp,
                    }
                    cross_analysis["oi_data"][exchange] = latest_oi[exchange]

            if len(latest_oi) < 2:
                return cross_analysis

            # Detect flow divergences between exchanges
            oi_changes = [(ex, data["oi_change_pct"]) for ex, data in latest_oi.items()]

            # Find exchanges with opposing flows
            positive_flow_exchanges = [ex for ex, change in oi_changes if change > 0.02]
            negative_flow_exchanges = [ex for ex, change in oi_changes if change < -0.02]

            if positive_flow_exchanges and negative_flow_exchanges:
                divergence = {
                    "type": "cross_exchange_flow_divergence",
                    "positive_flow_exchanges": positive_flow_exchanges,
                    "negative_flow_exchanges": negative_flow_exchanges,
                    "divergence_strength": abs(
                        np.mean([change for _, change in oi_changes if change > 0])
                        - np.mean([change for _, change in oi_changes if change < 0]),
                    "interpretation": "Institutional disagreement on direction",
                }
                cross_analysis["flow_divergences"].append(divergence)

            # Calculate market-wide OI momentum
            if len(oi_changes) > 1:
                all_changes = [change for _, change in oi_changes]
                cross_analysis["market_oi_momentum"] = {
                    "mean_change": np.mean(all_changes),
                    "consistency": 1 - np.std(all_changes) / (abs(np.mean(all_changes)) + 0.01),
                    "direction": "increasing" if np.mean(all_changes) > 0 else "decreasing",
                }

            return cross_analysis

        except Exception as e:
            logger.error(f"Cross-exchange OI analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_oi_analytics(
        self, exchange: Optional[str] = None, pair: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive OI analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter events
            all_events = []
            for ex in self.oi_history:
                if exchange and ex != exchange:
                    continue
                for p in self.oi_history[ex]:
                    if pair and p != pair:
                        continue
                    events = [
                        event for event in self.oi_history[ex][p] if event.timestamp >= cutoff_time
                    ]
                    all_events.extend(events)

            if not all_events:
                return {"status": "no_data"}

            analytics = self._calculate_oi_analytics(all_events)

            # Add divergence analysis
            recent_divergences = [
                div
                for div in self.divergence_history
                if div.timestamp >= cutoff_time
                and (exchange is None or div.exchange == exchange)
                and (pair is None or div.pair == pair)
            ]

            analytics["divergence_analysis"] = self._analyze_oi_divergences(recent_divergences)

            return analytics

        except Exception as e:
            logger.error(f"OI analytics failed: {e}")
            return {"status": "error", "error": str(e)}

    def _create_oi_event(
        self, exchange: str, pair: str, current_data: Dict[str, Any], previous_data: Dict[str, Any]
    ) -> Optional[OIEvent]:
        """Create OI event from data"""
        try:
            current_oi = current_data.get("open_interest", 0.0)
            previous_oi = previous_data.get("open_interest", 0.0)

            if previous_oi == 0:
                return None

            # Calculate changes
            oi_change = current_oi - previous_oi
            oi_change_pct = oi_change / previous_oi

            # Price context
            current_price = current_data.get("price", 0.0)
            previous_price = previous_data.get("price", 0.0)
            price_change_pct = (
                (current_price - previous_price) / previous_price if previous_price > 0 else 0
            )

            # Volume context
            volume_24h = current_data.get("volume_24h", 0.0)
            volume_oi_ratio = volume_24h / current_oi if current_oi > 0 else 0

            # Classifications
            oi_trend = self._classify_oi_trend(oi_change_pct)
            divergence_type = self._classify_divergence_type(oi_change_pct, price_change_pct)
            regime = self._classify_oi_regime(oi_change_pct, price_change_pct)

            # Create event
            event = OIEvent(
                timestamp=current_data.get("timestamp", datetime.now()),
                pair=pair,
                exchange=exchange,
                current_oi=current_oi,
                previous_oi=previous_oi,
                oi_change=oi_change,
                oi_change_pct=oi_change_pct,
                price_at_event=current_price,
                price_change_pct=price_change_pct,
                volume_24h=volume_24h,
                volume_oi_ratio=volume_oi_ratio,
                oi_trend=oi_trend,
                divergence_type=divergence_type,
                regime=regime,
            )

            # Calculate derived metrics
            event.oi_momentum = self._calculate_oi_momentum(exchange, pair, event)
            event.institutional_flow = self._estimate_institutional_flow(event)

            return event

        except Exception as e:
            logger.error(f"OI event creation failed: {e}")
            return None

    def _classify_oi_trend(self, oi_change_pct: float) -> OITrend:
        """Classify OI trend"""
        if oi_change_pct > 0.02:  # 2% increase
            return OITrend.INCREASING
        elif oi_change_pct < -0.02:  # 2% decrease
            return OITrend.DECREASING
        else:
            return OITrend.STABLE

    def _classify_divergence_type(
        self, oi_change_pct: float, price_change_pct: float
    ) -> OIDivergenceType:
        """Classify OI-price divergence"""
        # Threshold for significant moves
        oi_threshold = 0.02  # 2%
        price_threshold = 0.01  # 1%

        oi_up = oi_change_pct > oi_threshold
        oi_down = oi_change_pct < -oi_threshold
        price_up = price_change_pct > price_threshold
        price_down = price_change_pct < -price_threshold

        if price_down and oi_up:
            return OIDivergenceType.BULLISH_DIVERGENCE
        elif price_up and oi_down:
            return OIDivergenceType.BEARISH_DIVERGENCE
        elif (price_up and oi_up) or (price_down and oi_down):
            return OIDivergenceType.CONFIRMING_TREND
        else:
            return OIDivergenceType.NEUTRAL

    def _classify_oi_regime(self, oi_change_pct: float, price_change_pct: float) -> OIRegime:
        """Classify OI regime"""
        oi_threshold = 0.02
        price_threshold = 0.01

        oi_up = oi_change_pct > oi_threshold
        oi_down = oi_change_pct < -oi_threshold
        price_up = price_change_pct > price_threshold
        price_down = price_change_pct < -price_threshold

        if oi_up and price_up:
            return OIRegime.BREAKOUT
        elif oi_up and price_down:
            return OIRegime.BREAKDOWN
        elif oi_down and price_up:
            return OIRegime.DISTRIBUTION
        elif oi_down and price_down:
            return OIRegime.DISTRIBUTION
        elif oi_up and not (price_up or price_down):
            return OIRegime.ACCUMULATION
        else:
            return OIRegime.CONSOLIDATION

    def _detect_oi_divergence(
        self, exchange: str, pair: str, current_event: OIEvent
    ) -> Optional[OIDivergence]:
        """Detect significant OI divergences"""
        try:
            if current_event.divergence_type in [
                OIDivergenceType.BULLISH_DIVERGENCE,
                OIDivergenceType.BEARISH_DIVERGENCE,
            ]:
                # Calculate divergence strength
                strength = min(
                    1.0, abs(current_event.oi_change_pct) + abs(current_event.price_change_pct)

                if strength > self.divergence_strength_threshold:
                    divergence_id = f"{exchange}_{pair}_{current_event.timestamp.timestamp()}"

                    # Calculate correlation over recent period
                    recent_events = self._get_recent_oi_events(exchange, pair, hours_back=24)
                    correlation = self._calculate_oi_price_correlation(recent_events)

                    # Predict resolution
                    expected_direction, confidence = self._predict_divergence_resolution(
                        current_event, recent_events
                    )

                    divergence = OIDivergence(
                        divergence_id=divergence_id,
                        timestamp=current_event.timestamp,
                        pair=pair,
                        exchange=exchange,
                        divergence_type=current_event.divergence_type,
                        strength=strength,
                        duration_hours=1.0,  # Initial duration
                        price_change_pct=current_event.price_change_pct,
                        oi_change_pct=current_event.oi_change_pct,
                        correlation=correlation,
                        volume_during_divergence=current_event.volume_24h,
                        volatility_during_divergence=abs(current_event.price_change_pct),
                        expected_resolution_direction=expected_direction,
                        confidence=confidence,
                        time_to_resolution_hours=24.0,  # Estimated resolution time
                    )

                    return divergence

            return None

        except Exception as e:
            logger.error(f"OI divergence detection failed: {e}")
            return None

    def _generate_oi_signals(
        self, latest_event: OIEvent, recent_events: List[OIEvent], current_price: float
    ) -> Dict[str, Any]:
        """Generate trading signals based on OI analysis"""
        try:
            signals = {
                "timestamp": datetime.now(),
                "pair": latest_event.pair,
                "exchange": latest_event.exchange,
                "current_oi": latest_event.current_oi,
                "oi_regime": latest_event.regime.value,
                "signals": [],
            }

            # Divergence-based signals
            if latest_event.divergence_type == OIDivergenceType.BULLISH_DIVERGENCE:
                signal = {
                    "type": "oi_bullish_divergence",
                    "direction": "buy",
                    "confidence": min(0.8, latest_event.strength_score),
                    "rationale": f"Bullish OI divergence: OI +{latest_event.oi_change_pct:.1%}, Price {latest_event.price_change_pct:.1%}",
                    "target_hold_hours": 48,
                    "stop_loss_bp": 150,
                    "take_profit_bp": 300,
                }
                signals["signals"].append(signal)

            elif latest_event.divergence_type == OIDivergenceType.BEARISH_DIVERGENCE:
                signal = {
                    "type": "oi_bearish_divergence",
                    "direction": "sell",
                    "confidence": min(0.8, latest_event.strength_score),
                    "rationale": f"Bearish OI divergence: OI {latest_event.oi_change_pct:.1%}, Price +{latest_event.price_change_pct:.1%}",
                    "target_hold_hours": 48,
                    "stop_loss_bp": 150,
                    "take_profit_bp": 300,
                }
                signals["signals"].append(signal)

            # Trend confirmation signals
            if latest_event.regime == OIRegime.BREAKOUT and latest_event.is_significant_change:
                signal = {
                    "type": "oi_breakout_confirmation",
                    "direction": "buy",
                    "confidence": 0.75,
                    "rationale": f"OI breakout regime with {latest_event.oi_change_pct:.1%} OI increase",
                    "target_hold_hours": 72,
                    "stop_loss_bp": 200,
                    "take_profit_bp": 400,
                }
                signals["signals"].append(signal)

            elif latest_event.regime == OIRegime.BREAKDOWN and latest_event.is_significant_change:
                signal = {
                    "type": "oi_breakdown_confirmation",
                    "direction": "sell",
                    "confidence": 0.75,
                    "rationale": f"OI breakdown regime with {latest_event.oi_change_pct:.1%} OI increase",
                    "target_hold_hours": 72,
                    "stop_loss_bp": 200,
                    "take_profit_bp": 400,
                }
                signals["signals"].append(signal)

            # Flow reversal signals
            if len(recent_events) >= 5:
                flow_reversal = self._detect_recent_flow_reversal(recent_events)
                if flow_reversal:
                    direction = (
                        "buy" if flow_reversal["new_direction"] == "accumulation" else "sell"
                    )

                    signal = {
                        "type": "oi_flow_reversal",
                        "direction": direction,
                        "confidence": flow_reversal["confidence"],
                        "rationale": f"OI flow reversal detected: {flow_reversal['description']}",
                        "target_hold_hours": 96,
                        "stop_loss_bp": 250,
                        "take_profit_bp": 500,
                    }
                    signals["signals"].append(signal)

            return signals

        except Exception as e:
            logger.error(f"OI signal generation failed: {e}")
            return {"status": "error"}

    def _get_recent_oi_events(self, exchange: str, pair: str, hours_back: int) -> List[OIEvent]:
        """Get recent OI events"""
        try:
            if exchange not in self.oi_history or pair not in self.oi_history[exchange]:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            events = self.oi_history[exchange][pair]

            return [event for event in events if event.timestamp >= cutoff_time]

        except Exception as e:
            logger.error(f"Failed to get recent OI events: {e}")
            return []

    def _store_oi_event(self, exchange: str, pair: str, event: OIEvent) -> None:
        """Store OI event in history"""
        try:
            if exchange not in self.oi_history:
                self.oi_history[exchange] = {}

            if pair not in self.oi_history[exchange]:
                self.oi_history[exchange][pair] = []

            self.oi_history[exchange][pair].append(event)

            # Keep only recent history (30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.oi_history[exchange][pair] = [
                e for e in self.oi_history[exchange][pair] if e.timestamp >= cutoff_time
            ]

        except Exception as e:
            logger.error(f"Failed to store OI event: {e}")

    def _calculate_oi_momentum(self, exchange: str, pair: str, current_event: OIEvent) -> float:
        """Calculate OI momentum (acceleration)"""
        try:
            recent_events = self._get_recent_oi_events(exchange, pair, hours_back=24)

            if len(recent_events) < 3:
                return 0.0

            # Calculate second derivative (acceleration)
            oi_changes = [event.oi_change_pct for event in recent_events[-3:]]

            if len(oi_changes) >= 3:
                acceleration = (oi_changes[-1] - 2 * oi_changes[-2] + oi_changes[-3]) / 2
                return acceleration

            return 0.0

        except Exception as e:
            logger.error(f"OI momentum calculation failed: {e}")
            return 0.0

    def _estimate_institutional_flow(self, event: OIEvent) -> float:
        """Estimate institutional flow direction and strength"""
        try:
            # Large OI changes with low volume suggest institutional activity
            if event.volume_oi_ratio < 0.1 and abs(event.oi_change_pct) > 0.05:
                # Low volume, high OI change = likely institutional
                flow_strength = min(1.0, abs(event.oi_change_pct) / 0.2)
                return flow_strength if event.oi_change > 0 else -flow_strength

            # High volume with proportional OI change suggests retail activity
            return 0.0

        except Exception as e:
            logger.error(f"Institutional flow estimation failed: {e}")
            return 0.0

    def _calculate_oi_price_correlation(self, events: List[OIEvent]) -> float:
        """Calculate OI-price correlation"""
        try:
            if len(events) < 5:
                return 0.0

            oi_changes = [event.oi_change_pct for event in events]
            price_changes = [event.price_change_pct for event in events]

            correlation = np.corrcoef(oi_changes, price_changes)[0, 1]

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.error(f"OI-price correlation calculation failed: {e}")
            return 0.0

    def _predict_divergence_resolution(
        self, event: OIEvent, recent_events: List[OIEvent]
    ) -> Tuple[str, float]:
        """Predict how divergence will resolve"""
        try:
            if event.divergence_type == OIDivergenceType.BULLISH_DIVERGENCE:
                # Price down, OI up - expect price to catch up
                return "up", 0.7
            elif event.divergence_type == OIDivergenceType.BEARISH_DIVERGENCE:
                # Price up, OI down - expect price to correct down
                return "down", 0.7
            else:
                return "neutral", 0.5

        except Exception as e:
            logger.error(f"Divergence resolution prediction failed: {e}")
            return "neutral", 0.5

    def _detect_accumulation_periods(self, events: List[OIEvent]) -> List[Dict[str, Any]]:
        """Detect accumulation periods from OI data"""
        try:
            accumulation_periods = []

            # Look for sustained OI increases with stable/rising prices
            window_size = 6  # 6 period window

            for i in range(window_size, len(events)):
                window_events = events[i - window_size : i]

                # Check for accumulation pattern
                oi_increases = sum(1 for event in window_events if event.oi_change_pct > 0.01)
                price_stability = sum(
                    1 for event in window_events if abs(event.price_change_pct) < 0.02
                )

                if oi_increases >= window_size * 0.7 and price_stability >= window_size * 0.5:
                    period = {
                        "type": "accumulation",
                        "start_time": window_events[0].timestamp,
                        "end_time": window_events[-1].timestamp,
                        "oi_increase_pct": sum(event.oi_change_pct for event in window_events),
                        "avg_price_change": np.mean(
                            [event.price_change_pct for event in window_events]
                        ),
                        "strength": min(1.0, oi_increases / window_size),
                    }
                    accumulation_periods.append(period)

            return accumulation_periods

        except Exception as e:
            logger.error(f"Accumulation period detection failed: {e}")
            return []

    def _detect_distribution_periods(self, events: List[OIEvent]) -> List[Dict[str, Any]]:
        """Detect distribution periods from OI data"""
        try:
            distribution_periods = []

            # Look for sustained OI decreases
            window_size = 6

            for i in range(window_size, len(events)):
                window_events = events[i - window_size : i]

                # Check for distribution pattern
                oi_decreases = sum(1 for event in window_events if event.oi_change_pct < -0.01)

                if oi_decreases >= window_size * 0.7:
                    period = {
                        "type": "distribution",
                        "start_time": window_events[0].timestamp,
                        "end_time": window_events[-1].timestamp,
                        "oi_decrease_pct": sum(event.oi_change_pct for event in window_events),
                        "avg_price_change": np.mean(
                            [event.price_change_pct for event in window_events]
                        ),
                        "strength": min(1.0, oi_decreases / window_size),
                    }
                    distribution_periods.append(period)

            return distribution_periods

        except Exception as e:
            logger.error(f"Distribution period detection failed: {e}")
            return []

    def _calculate_net_institutional_flow(self, events: List[OIEvent]) -> Dict[str, Any]:
        """Calculate net institutional flow"""
        try:
            institutional_events = [
                event for event in events if abs(event.institutional_flow) > 0.3
            ]

            if not institutional_events:
                return {"net_flow": 0.0, "confidence": 0.0}

            net_flow = sum(event.institutional_flow for event in institutional_events)
            avg_confidence = np.mean(
                [abs(event.institutional_flow) for event in institutional_events]
            )

            return {
                "net_flow": net_flow,
                "confidence": avg_confidence,
                "institutional_periods": len(institutional_events),
                "flow_direction": "accumulation" if net_flow > 0 else "distribution",
            }

        except Exception as e:
            logger.error(f"Net institutional flow calculation failed: {e}")
            return {"net_flow": 0.0, "confidence": 0.0}

    def _detect_flow_reversals(self, events: List[OIEvent]) -> List[Dict[str, Any]]:
        """Detect institutional flow reversals"""
        try:
            flow_reversals = []

            if len(events) < 10:
                return flow_reversals

            # Look for significant changes in flow direction
            window_size = 5

            for i in range(window_size, len(events) - window_size):
                before_window = events[i - window_size : i]
                after_window = events[i : i + window_size]

                before_flow = np.mean([event.institutional_flow for event in before_window])
                after_flow = np.mean([event.institutional_flow for event in after_window])

                # Check for flow reversal
                if abs(before_flow - after_flow) > 0.5 and before_flow * after_flow < 0:
                    reversal = {
                        "timestamp": events[i].timestamp,
                        "from_flow": before_flow,
                        "to_flow": after_flow,
                        "reversal_strength": abs(before_flow - after_flow),
                        "type": "accumulation_to_distribution"
                        if before_flow > 0
                        else "distribution_to_accumulation",
                    }
                    flow_reversals.append(reversal)

            return flow_reversals

        except Exception as e:
            logger.error(f"Flow reversal detection failed: {e}")
            return []

    def _detect_recent_flow_reversal(
        self, recent_events: List[OIEvent]
    ) -> Optional[Dict[str, Any]]:
        """Detect recent flow reversal in last few periods"""
        try:
            if len(recent_events) < 6:
                return None

            # Split into before/after
            mid_point = len(recent_events) // 2
            before_events = recent_events[:mid_point]
            after_events = recent_events[mid_point:]

            before_flow = np.mean([event.institutional_flow for event in before_events])
            after_flow = np.mean([event.institutional_flow for event in after_events])

            # Check for significant reversal
            if abs(before_flow - after_flow) > 0.4 and before_flow * after_flow < 0:
                new_direction = "accumulation" if after_flow > 0 else "distribution"

                return {
                    "new_direction": new_direction,
                    "confidence": min(0.9, abs(before_flow - after_flow)),
                    "description": f"Flow reversed from {before_flow:.2f} to {after_flow:.2f}",
                }

            return None

        except Exception as e:
            logger.error(f"Recent flow reversal detection failed: {e}")
            return None

    def _calculate_oi_analytics(self, events: List[OIEvent]) -> Dict[str, Any]:
        """Calculate comprehensive OI analytics"""
        try:
            if not events:
                return {"status": "no_events"}

            oi_changes = [event.oi_change_pct for event in events]
            price_changes = [event.price_change_pct for event in events]

            analytics = {
                "total_events": len(events),
                "oi_statistics": {
                    "mean_change_pct": np.mean(oi_changes),
                    "std_change_pct": np.std(oi_changes),
                    "max_increase_pct": np.max(oi_changes),
                    "max_decrease_pct": np.min(oi_changes),
                },
                "regime_distribution": {},
                "divergence_frequency": {},
                "correlation_with_price": np.corrcoef(oi_changes, price_changes)[0, 1]
                if len(oi_changes) > 1
                else 0,
                "significant_changes": sum(1 for event in events if event.is_significant_change),
            }

            # Regime distribution
            for regime in OIRegime:
                count = sum(1 for event in events if event.regime == regime)
                analytics["regime_distribution"][regime.value] = {
                    "count": count,
                    "percentage": count / len(events),
                }

            # Divergence frequency
            for div_type in OIDivergenceType:
                count = sum(1 for event in events if event.divergence_type == div_type)
                analytics["divergence_frequency"][div_type.value] = {
                    "count": count,
                    "percentage": count / len(events),
                }

            return analytics

        except Exception as e:
            logger.error(f"OI analytics calculation failed: {e}")
            return {"status": "error"}

    def _analyze_oi_divergences(self, divergences: List[OIDivergence]) -> Dict[str, Any]:
        """Analyze OI divergence patterns"""
        try:
            if not divergences:
                return {"total_divergences": 0}

            divergence_analysis = {
                "total_divergences": len(divergences),
                "divergence_types": {},
                "average_strength": np.mean([div.strength for div in divergences]),
                "average_duration_hours": np.mean([div.duration_hours for div in divergences]),
                "resolution_accuracy": 0.0,  # Would track actual vs predicted
            }

            # Divergence type distribution
            for div_type in OIDivergenceType:
                count = sum(1 for div in divergences if div.divergence_type == div_type)
                divergence_analysis["divergence_types"][div_type.value] = count

            return divergence_analysis

        except Exception as e:
            logger.error(f"Divergence analysis failed: {e}")
            return {"total_divergences": 0}
