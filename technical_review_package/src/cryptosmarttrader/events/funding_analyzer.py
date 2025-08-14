"""
Funding Rate Analyzer

Analyzes funding rate patterns, flips, and anomalies across exchanges
to identify mean reversion and trend continuation opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FundingDirection(Enum):
    """Funding rate direction"""

    POSITIVE = "positive"  # Longs pay shorts
    NEGATIVE = "negative"  # Shorts pay longs
    NEUTRAL = "neutral"  # Near zero funding


class FundingRegime(Enum):
    """Funding rate regime classification"""

    EXTREME_LONG = "extreme_long"  # Very high positive funding
    HIGH_LONG = "high_long"  # High positive funding
    NORMAL_LONG = "normal_long"  # Normal positive funding
    NEUTRAL = "neutral"  # Balanced funding
    NORMAL_SHORT = "normal_short"  # Normal negative funding
    HIGH_SHORT = "high_short"  # High negative funding
    EXTREME_SHORT = "extreme_short"  # Very high negative funding


class FlipType(Enum):
    """Types of funding flips"""

    POSITIVE_TO_NEGATIVE = "pos_to_neg"
    NEGATIVE_TO_POSITIVE = "neg_to_pos"
    EXTREME_TO_NORMAL = "extreme_to_normal"
    NORMAL_TO_EXTREME = "normal_to_extreme"


@dataclass
class FundingEvent:
    """Individual funding rate event"""

    timestamp: datetime
    pair: str
    exchange: str

    # Funding data
    current_rate: float  # Current funding rate (8hr annualized)
    previous_rate: float  # Previous funding rate
    rate_change: float  # Absolute change in funding
    rate_change_pct: float  # Percentage change

    # Classification
    regime: FundingRegime
    direction: FundingDirection

    # Context
    price_at_event: float
    volume_24h: float
    open_interest: Optional[float] = None

    # Derived metrics
    funding_velocity: float = 0.0  # Rate of change acceleration
    cross_exchange_basis: float = 0.0  # Basis vs other exchanges

    @property
    def is_extreme(self) -> bool:
        """Check if funding rate is extreme"""
        return abs(self.current_rate) > 0.5  # 50%+ annualized

    @property
    def funding_pressure(self) -> float:
        """Calculate funding pressure intensity (0-1)"""
        return min(1.0, abs(self.current_rate) / 2.0)  # Max at 200% annualized


@dataclass
class FundingFlip:
    """Funding rate flip event"""

    flip_id: str
    timestamp: datetime
    pair: str
    exchange: str

    # Flip details
    flip_type: FlipType
    from_regime: FundingRegime
    to_regime: FundingRegime

    # Rates
    pre_flip_rate: float
    post_flip_rate: float
    flip_magnitude: float  # Absolute change

    # Market context
    price_before: float
    price_after: float
    volume_spike: float  # Volume increase during flip

    # Timing
    flip_duration_hours: float  # How long the flip took
    time_since_last_flip: Optional[float] = None  # Hours since last flip

    @property
    def price_impact_bp(self) -> float:
        """Price impact in basis points during flip"""
        if self.price_before > 0:
            return (self.price_after - self.price_before) / self.price_before * 10000
        return 0.0


class FundingAnalyzer:
    """
    Advanced funding rate analysis and pattern detection
    """

    def __init__(self):
        self.funding_history = {}  # exchange -> pair -> List[FundingEvent]
        self.flip_history = []

        # Configuration
        self.extreme_funding_threshold = 0.5  # 50% annualized
        self.high_funding_threshold = 0.2  # 20% annualized
        self.neutral_threshold = 0.05  # 5% annualized

        # Pattern tracking
        self.regime_transitions = {}
        self.flip_patterns = {}

        # Performance tracking
        self.signal_performance = {}

    def analyze_funding_data(
        self, exchange: str, pair: str, funding_data: List[Dict[str, Any]]
    ) -> List[FundingEvent]:
        """Analyze funding rate data and detect events"""
        try:
            events = []

            if not funding_data:
                return events

            # Sort by timestamp
            funding_data = sorted(funding_data, key=lambda x: x.get("timestamp", datetime.min))

            # Process each funding period
            for i, current_data in enumerate(funding_data):
                if i == 0:
                    continue  # Skip first record (no previous to compare)

                previous_data = funding_data[i - 1]

                # Create funding event
                event = self._create_funding_event(exchange, pair, current_data, previous_data)

                if event:
                    events.append(event)

                    # Store in history
                    self._store_funding_event(exchange, pair, event)

                    # Check for flips
                    flip = self._detect_funding_flip(exchange, pair, event)
                    if flip:
                        self.flip_history.append(flip)
                        logger.info(
                            f"Funding flip detected: {flip.flip_type.value} for {pair} on {exchange}"
                        )

            return events

        except Exception as e:
            logger.error(f"Funding analysis failed for {exchange} {pair}: {e}")
            return []

    def get_current_funding_signals(
        self, exchange: str, pair: str, current_price: float
    ) -> Dict[str, Any]:
        """Get current funding-based trading signals"""
        try:
            # Get recent funding events
            recent_events = self._get_recent_events(exchange, pair, hours_back=24)

            if not recent_events:
                return {"status": "no_data"}

            latest_event = recent_events[-1]

            # Generate signals based on current regime
            signals = self._generate_funding_signals(latest_event, recent_events, current_price)

            return signals

        except Exception as e:
            logger.error(f"Funding signal generation failed: {e}")
            return {"status": "error", "error": str(e)}

    def detect_funding_anomalies(
        self, exchange: str, pair: str, lookback_hours: int = 168
    ) -> List[Dict[str, Any]]:
        """Detect funding rate anomalies and patterns"""
        try:
            events = self._get_recent_events(exchange, pair, hours_back=lookback_hours)

            if len(events) < 10:
                return []

            anomalies = []

            # Statistical anomaly detection
            rates = [event.current_rate for event in events]
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)

            for event in events[-24:]:  # Last 24 periods
                z_score = (event.current_rate - mean_rate) / std_rate if std_rate > 0 else 0

                if abs(z_score) > 2.5:  # 2.5 sigma threshold
                    anomaly = {
                        "timestamp": event.timestamp,
                        "type": "statistical_outlier",
                        "funding_rate": event.current_rate,
                        "z_score": z_score,
                        "regime": event.regime.value,
                        "severity": min(1.0, abs(z_score) / 5.0),  # Max severity at 5 sigma
                    }
                    anomalies.append(anomaly)

            # Pattern-based anomalies
            pattern_anomalies = self._detect_pattern_anomalies(events)
            anomalies.extend(pattern_anomalies)

            return sorted(anomalies, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Funding anomaly detection failed: {e}")
            return []

    def get_cross_exchange_analysis(self, pair: str, exchanges: List[str]) -> Dict[str, Any]:
        """Analyze funding rates across exchanges for arbitrage opportunities"""
        try:
            cross_analysis = {
                "pair": pair,
                "timestamp": datetime.now(),
                "exchanges_analyzed": exchanges,
                "funding_rates": {},
                "arbitrage_opportunities": [],
            }

            # Get latest funding for each exchange
            latest_rates = {}
            for exchange in exchanges:
                recent_events = self._get_recent_events(exchange, pair, hours_back=1)
                if recent_events:
                    latest_event = recent_events[-1]
                    latest_rates[exchange] = {
                        "rate": latest_event.current_rate,
                        "regime": latest_event.regime.value,
                        "timestamp": latest_event.timestamp,
                    }
                    cross_analysis["funding_rates"][exchange] = latest_rates[exchange]

            if len(latest_rates) < 2:
                return cross_analysis

            # Find arbitrage opportunities
            rates_list = [(ex, data["rate"]) for ex, data in latest_rates.items()]
            rates_list.sort(key=lambda x: x[1])  # Sort by rate

            lowest_exchange, lowest_rate = rates_list[0]
            highest_exchange, highest_rate = rates_list[-1]

            rate_spread = highest_rate - lowest_rate

            if abs(rate_spread) > 0.1:  # 10% spread threshold
                opportunity = {
                    "type": "funding_arbitrage",
                    "long_exchange": lowest_exchange,
                    "short_exchange": highest_exchange,
                    "rate_spread": rate_spread,
                    "profit_potential_bp": rate_spread * 10000,
                    "confidence": min(1.0, abs(rate_spread) / 0.5),
                }
                cross_analysis["arbitrage_opportunities"].append(opportunity)

            # Calculate cross-exchange basis
            if len(latest_rates) > 1:
                all_rates = [data["rate"] for data in latest_rates.values()]
                cross_analysis["rate_statistics"] = {
                    "mean": np.mean(all_rates),
                    "std": np.std(all_rates),
                    "range": max(all_rates) - min(all_rates),
                    "coefficient_of_variation": np.std(all_rates) / abs(np.mean(all_rates))
                    if np.mean(all_rates) != 0
                    else 0,
                }

            return cross_analysis

        except Exception as e:
            logger.error(f"Cross-exchange funding analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_funding_analytics(
        self, exchange: Optional[str] = None, pair: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive funding analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter events
            all_events = []
            for ex in self.funding_history:
                if exchange and ex != exchange:
                    continue
                for p in self.funding_history[ex]:
                    if pair and p != pair:
                        continue
                    events = [
                        event
                        for event in self.funding_history[ex][p]
                        if event.timestamp >= cutoff_time
                    ]
                    all_events.extend(events)

            if not all_events:
                return {"status": "no_data"}

            analytics = self._calculate_funding_analytics(all_events)

            # Add flip analysis
            recent_flips = [
                flip
                for flip in self.flip_history
                if flip.timestamp >= cutoff_time
                and (exchange is None or flip.exchange == exchange)
                and (pair is None or flip.pair == pair)
            ]

            analytics["flip_analysis"] = self._analyze_funding_flips(recent_flips)

            return analytics

        except Exception as e:
            logger.error(f"Funding analytics failed: {e}")
            return {"status": "error", "error": str(e)}

    def _create_funding_event(
        self, exchange: str, pair: str, current_data: Dict[str, Any], previous_data: Dict[str, Any]
    ) -> Optional[FundingEvent]:
        """Create funding event from data"""
        try:
            current_rate = current_data.get("funding_rate", 0.0)
            previous_rate = previous_data.get("funding_rate", 0.0)

            # Calculate changes
            rate_change = current_rate - previous_rate
            rate_change_pct = (rate_change / abs(previous_rate)) if previous_rate != 0 else 0

            # Classify regime
            regime = self._classify_funding_regime(current_rate)
            direction = self._get_funding_direction(current_rate)

            # Create event
            event = FundingEvent(
                timestamp=current_data.get("timestamp", datetime.now()),
                pair=pair,
                exchange=exchange,
                current_rate=current_rate,
                previous_rate=previous_rate,
                rate_change=rate_change,
                rate_change_pct=rate_change_pct,
                regime=regime,
                direction=direction,
                price_at_event=current_data.get("price", 0.0),
                volume_24h=current_data.get("volume_24h", 0.0),
                open_interest=current_data.get("open_interest"),
            )

            # Calculate derived metrics
            event.funding_velocity = self._calculate_funding_velocity(exchange, pair, event)

            return event

        except Exception as e:
            logger.error(f"Funding event creation failed: {e}")
            return None

    def _classify_funding_regime(self, funding_rate: float) -> FundingRegime:
        """Classify funding rate regime"""
        abs_rate = abs(funding_rate)

        if abs_rate >= self.extreme_funding_threshold:
            return FundingRegime.EXTREME_LONG if funding_rate > 0 else FundingRegime.EXTREME_SHORT
        elif abs_rate >= self.high_funding_threshold:
            return FundingRegime.HIGH_LONG if funding_rate > 0 else FundingRegime.HIGH_SHORT
        elif abs_rate >= self.neutral_threshold:
            return FundingRegime.NORMAL_LONG if funding_rate > 0 else FundingRegime.NORMAL_SHORT
        else:
            return FundingRegime.NEUTRAL

    def _get_funding_direction(self, funding_rate: float) -> FundingDirection:
        """Get funding direction"""
        if funding_rate > self.neutral_threshold:
            return FundingDirection.POSITIVE
        elif funding_rate < -self.neutral_threshold:
            return FundingDirection.NEGATIVE
        else:
            return FundingDirection.NEUTRAL

    def _detect_funding_flip(
        self, exchange: str, pair: str, current_event: FundingEvent
    ) -> Optional[FundingFlip]:
        """Detect funding rate flips"""
        try:
            recent_events = self._get_recent_events(exchange, pair, hours_back=48)

            if len(recent_events) < 2:
                return None

            previous_event = recent_events[-2]

            # Check for regime flip
            if current_event.regime != previous_event.regime:
                flip_type = self._determine_flip_type(previous_event.regime, current_event.regime)

                if flip_type:
                    flip_id = f"{exchange}_{pair}_{current_event.timestamp.timestamp()}"

                    flip = FundingFlip(
                        flip_id=flip_id,
                        timestamp=current_event.timestamp,
                        pair=pair,
                        exchange=exchange,
                        flip_type=flip_type,
                        from_regime=previous_event.regime,
                        to_regime=current_event.regime,
                        pre_flip_rate=previous_event.current_rate,
                        post_flip_rate=current_event.current_rate,
                        flip_magnitude=abs(
                            current_event.current_rate - previous_event.current_rate
                        ),
                        price_before=previous_event.price_at_event,
                        price_after=current_event.price_at_event,
                        volume_spike=current_event.volume_24h / max(previous_event.volume_24h, 1),
                        flip_duration_hours=8.0,  # Standard funding period
                    )

                    # Calculate time since last flip
                    last_flip = self._get_last_flip(exchange, pair)
                    if last_flip:
                        flip.time_since_last_flip = (
                            current_event.timestamp - last_flip.timestamp
                        ).total_seconds() / 3600

                    return flip

            return None

        except Exception as e:
            logger.error(f"Funding flip detection failed: {e}")
            return None

    def _determine_flip_type(
        self, from_regime: FundingRegime, to_regime: FundingRegime
    ) -> Optional[FlipType]:
        """Determine type of funding flip"""
        # Positive to negative transitions
        if from_regime in [
            FundingRegime.NORMAL_LONG,
            FundingRegime.HIGH_LONG,
            FundingRegime.EXTREME_LONG,
        ] and to_regime in [
            FundingRegime.NORMAL_SHORT,
            FundingRegime.HIGH_SHORT,
            FundingRegime.EXTREME_SHORT,
        ]:
            return FlipType.POSITIVE_TO_NEGATIVE

        # Negative to positive transitions
        if from_regime in [
            FundingRegime.NORMAL_SHORT,
            FundingRegime.HIGH_SHORT,
            FundingRegime.EXTREME_SHORT,
        ] and to_regime in [
            FundingRegime.NORMAL_LONG,
            FundingRegime.HIGH_LONG,
            FundingRegime.EXTREME_LONG,
        ]:
            return FlipType.NEGATIVE_TO_POSITIVE

        # Extreme to normal transitions
        if from_regime in [
            FundingRegime.EXTREME_LONG,
            FundingRegime.EXTREME_SHORT,
        ] and to_regime in [
            FundingRegime.NORMAL_LONG,
            FundingRegime.NORMAL_SHORT,
            FundingRegime.NEUTRAL,
        ]:
            return FlipType.EXTREME_TO_NORMAL

        # Normal to extreme transitions
        if from_regime in [
            FundingRegime.NORMAL_LONG,
            FundingRegime.NORMAL_SHORT,
            FundingRegime.NEUTRAL,
        ] and to_regime in [FundingRegime.EXTREME_LONG, FundingRegime.EXTREME_SHORT]:
            return FlipType.NORMAL_TO_EXTREME

        return None

    def _generate_funding_signals(
        self, latest_event: FundingEvent, recent_events: List[FundingEvent], current_price: float
    ) -> Dict[str, Any]:
        """Generate trading signals based on funding analysis"""
        try:
            signals = {
                "timestamp": datetime.now(),
                "pair": latest_event.pair,
                "exchange": latest_event.exchange,
                "current_funding_rate": latest_event.current_rate,
                "funding_regime": latest_event.regime.value,
                "signals": [],
            }

            # Mean reversion signals on extreme funding
            if latest_event.regime in [FundingRegime.EXTREME_LONG, FundingRegime.EXTREME_SHORT]:
                direction = "sell" if latest_event.regime == FundingRegime.EXTREME_LONG else "buy"

                signal = {
                    "type": "funding_mean_reversion",
                    "direction": direction,
                    "confidence": min(0.9, latest_event.funding_pressure),
                    "rationale": f"Extreme funding ({latest_event.current_rate:.2%}) suggests mean reversion",
                    "target_hold_hours": 24,  # Hold until next funding period
                    "stop_loss_bp": 200,
                    "take_profit_bp": 100,
                }
                signals["signals"].append(signal)

            # Trend continuation on funding flips
            recent_flip = self._get_last_flip(latest_event.exchange, latest_event.pair)
            if recent_flip and recent_flip.timestamp >= datetime.now() - timedelta(hours=16):
                if recent_flip.flip_type == FlipType.POSITIVE_TO_NEGATIVE:
                    # Funding turned negative - potential short continuation
                    signal = {
                        "type": "funding_trend_continuation",
                        "direction": "sell",
                        "confidence": 0.7,
                        "rationale": f"Recent funding flip to negative suggests continued selling pressure",
                        "target_hold_hours": 48,
                        "stop_loss_bp": 150,
                        "take_profit_bp": 300,
                    }
                    signals["signals"].append(signal)
                elif recent_flip.flip_type == FlipType.NEGATIVE_TO_POSITIVE:
                    # Funding turned positive - potential long continuation
                    signal = {
                        "type": "funding_trend_continuation",
                        "direction": "buy",
                        "confidence": 0.7,
                        "rationale": f"Recent funding flip to positive suggests continued buying pressure",
                        "target_hold_hours": 48,
                        "stop_loss_bp": 150,
                        "take_profit_bp": 300,
                    }
                    signals["signals"].append(signal)

            # Funding velocity signals
            if len(recent_events) >= 3:
                velocity_trend = self._analyze_funding_velocity_trend(recent_events[-3:])
                if abs(velocity_trend) > 0.1:  # 10% acceleration threshold
                    direction = "sell" if velocity_trend > 0 else "buy"  # Contrarian

                    signal = {
                        "type": "funding_velocity_contrarian",
                        "direction": direction,
                        "confidence": min(0.8, abs(velocity_trend)),
                        "rationale": f"Funding velocity acceleration ({velocity_trend:.2%}) suggests reversal",
                        "target_hold_hours": 16,
                        "stop_loss_bp": 100,
                        "take_profit_bp": 150,
                    }
                    signals["signals"].append(signal)

            return signals

        except Exception as e:
            logger.error(f"Funding signal generation failed: {e}")
            return {"status": "error"}

    def _get_recent_events(self, exchange: str, pair: str, hours_back: int) -> List[FundingEvent]:
        """Get recent funding events"""
        try:
            if exchange not in self.funding_history or pair not in self.funding_history[exchange]:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            events = self.funding_history[exchange][pair]

            return [event for event in events if event.timestamp >= cutoff_time]

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    def _store_funding_event(self, exchange: str, pair: str, event: FundingEvent) -> None:
        """Store funding event in history"""
        try:
            if exchange not in self.funding_history:
                self.funding_history[exchange] = {}

            if pair not in self.funding_history[exchange]:
                self.funding_history[exchange][pair] = []

            self.funding_history[exchange][pair].append(event)

            # Keep only recent history (30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.funding_history[exchange][pair] = [
                e for e in self.funding_history[exchange][pair] if e.timestamp >= cutoff_time
            ]

        except Exception as e:
            logger.error(f"Failed to store funding event: {e}")

    def _calculate_funding_velocity(
        self, exchange: str, pair: str, current_event: FundingEvent
    ) -> float:
        """Calculate funding rate velocity (acceleration)"""
        try:
            recent_events = self._get_recent_events(exchange, pair, hours_back=24)

            if len(recent_events) < 3:
                return 0.0

            # Calculate second derivative (acceleration)
            rates = [event.current_rate for event in recent_events[-3:]]

            if len(rates) >= 3:
                # Simple finite difference approximation
                acceleration = (rates[-1] - 2 * rates[-2] + rates[-3]) / 2
                return acceleration

            return 0.0

        except Exception as e:
            logger.error(f"Funding velocity calculation failed: {e}")
            return 0.0

    def _get_last_flip(self, exchange: str, pair: str) -> Optional[FundingFlip]:
        """Get last funding flip for pair"""
        try:
            matching_flips = [
                flip
                for flip in self.flip_history
                if flip.exchange == exchange and flip.pair == pair
            ]

            if matching_flips:
                return max(matching_flips, key=lambda x: x.timestamp)

            return None

        except Exception as e:
            logger.error(f"Failed to get last flip: {e}")
            return None

    def _analyze_funding_velocity_trend(self, events: List[FundingEvent]) -> float:
        """Analyze funding velocity trend"""
        try:
            if len(events) < 3:
                return 0.0

            velocities = []
            for i in range(1, len(events)):
                velocity = events[i].current_rate - events[i - 1].current_rate
                velocities.append(velocity)

            if len(velocities) >= 2:
                # Trend in velocity (acceleration)
                trend = velocities[-1] - velocities[-2]
                return trend

            return 0.0

        except Exception as e:
            logger.error(f"Velocity trend analysis failed: {e}")
            return 0.0

    def _detect_pattern_anomalies(self, events: List[FundingEvent]) -> List[Dict[str, Any]]:
        """Detect pattern-based funding anomalies"""
        try:
            anomalies = []

            if len(events) < 10:
                return anomalies

            # Detect rapid oscillations
            regime_changes = 0
            for i in range(1, len(events)):
                if events[i].regime != events[i - 1].regime:
                    regime_changes += 1

            if regime_changes > len(events) * 0.3:  # More than 30% regime changes
                anomaly = {
                    "timestamp": events[-1].timestamp,
                    "type": "excessive_oscillation",
                    "regime_change_rate": regime_changes / len(events),
                    "severity": min(1.0, regime_changes / len(events) / 0.5),
                }
                anomalies.append(anomaly)

            # Detect sustained extreme funding
            extreme_count = sum(1 for event in events[-10:] if event.is_extreme)
            if extreme_count >= 5:  # 5 or more extreme periods in last 10
                anomaly = {
                    "timestamp": events[-1].timestamp,
                    "type": "sustained_extreme_funding",
                    "extreme_periods": extreme_count,
                    "severity": min(1.0, extreme_count / 10),
                }
                anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
            return []

    def _calculate_funding_analytics(self, events: List[FundingEvent]) -> Dict[str, Any]:
        """Calculate comprehensive funding analytics"""
        try:
            if not events:
                return {"status": "no_events"}

            rates = [event.current_rate for event in events]

            analytics = {
                "total_events": len(events),
                "rate_statistics": {
                    "mean": np.mean(rates),
                    "median": np.median(rates),
                    "std": np.std(rates),
                    "min": np.min(rates),
                    "max": np.max(rates),
                    "skewness": self._calculate_skewness(rates),
                    "kurtosis": self._calculate_kurtosis(rates),
                },
                "regime_distribution": {},
                "extreme_periods": sum(1 for event in events if event.is_extreme),
                "average_funding_pressure": np.mean([event.funding_pressure for event in events]),
            }

            # Regime distribution
            for regime in FundingRegime:
                count = sum(1 for event in events if event.regime == regime)
                analytics["regime_distribution"][regime.value] = {
                    "count": count,
                    "percentage": count / len(events),
                }

            return analytics

        except Exception as e:
            logger.error(f"Funding analytics calculation failed: {e}")
            return {"status": "error"}

    def _analyze_funding_flips(self, flips: List[FundingFlip]) -> Dict[str, Any]:
        """Analyze funding flip patterns"""
        try:
            if not flips:
                return {"total_flips": 0}

            flip_analysis = {
                "total_flips": len(flips),
                "flip_types": {},
                "average_magnitude": np.mean([flip.flip_magnitude for flip in flips]),
                "average_price_impact_bp": np.mean([abs(flip.price_impact_bp) for flip in flips]),
                "flip_frequency_hours": 0,
            }

            # Flip type distribution
            for flip_type in FlipType:
                count = sum(1 for flip in flips if flip.flip_type == flip_type)
                flip_analysis["flip_types"][flip_type.value] = count

            # Calculate flip frequency
            if len(flips) > 1:
                time_span = (flips[-1].timestamp - flips[0].timestamp).total_seconds() / 3600
                flip_analysis["flip_frequency_hours"] = time_span / len(flips)

            return flip_analysis

        except Exception as e:
            logger.error(f"Flip analysis failed: {e}")
            return {"total_flips": 0}

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data"""
        try:
            if len(data) < 3:
                return 0.0

            mean = np.mean(data)
            std = np.std(data)

            if std == 0:
                return 0.0

            skewness = np.mean([((x - mean) / std) ** 3 for x in data])
            return skewness

        except Exception:
            return 0.0

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data"""
        try:
            if len(data) < 4:
                return 0.0

            mean = np.mean(data)
            std = np.std(data)

            if std == 0:
                return 0.0

            kurtosis = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
            return kurtosis

        except Exception:
            return 0.0
