"""
Basis Analyzer

Analyzes perpetual-spot basis patterns and z-scores to identify
mean reversion and trend continuation opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BasisRegime(Enum):
    """Basis regime classification"""

    EXTREME_CONTANGO = "extreme_contango"  # Very high positive basis
    HIGH_CONTANGO = "high_contango"  # High positive basis
    NORMAL_CONTANGO = "normal_contango"  # Normal positive basis
    NEUTRAL = "neutral"  # Near zero basis
    NORMAL_BACKWARDATION = "normal_backwardation"  # Normal negative basis
    HIGH_BACKWARDATION = "high_backwardation"  # High negative basis
    EXTREME_BACKWARDATION = "extreme_backwardation"  # Very high negative basis


class BasisSignalType(Enum):
    """Types of basis signals"""

    MEAN_REVERSION = "mean_reversion"  # Extreme basis reverting
    TREND_CONTINUATION = "trend_continuation"  # Low basis trending
    ARBITRAGE = "arbitrage"  # Cross-exchange arbitrage
    MOMENTUM = "momentum"  # Basis momentum signal


@dataclass
class BasisZScore:
    """Basis z-score analysis"""

    timestamp: datetime
    pair: str
    exchange: str

    # Raw basis data
    perp_price: float
    spot_price: float
    basis_bp: float  # Basis in basis points

    # Statistical analysis
    z_score: float  # Statistical z-score
    lookback_mean: float  # Historical mean basis
    lookback_std: float  # Historical standard deviation
    lookback_periods: int  # Number of periods in analysis

    # Classification
    regime: BasisRegime
    is_extreme: bool  # |z-score| > 2
    is_very_extreme: bool  # |z-score| > 3

    # Context
    volume_ratio: float  # Perp/spot volume ratio
    funding_rate: Optional[float] = None
    time_to_expiry: Optional[float] = None  # For futures

    @property
    def basis_annual_pct(self) -> float:
        """Annualized basis percentage"""
        if self.time_to_expiry and self.time_to_expiry > 0:
            # For futures: annualize based on time to expiry
            return (self.basis_bp / 10000) * (365 * 24 / self.time_to_expiry)
        else:
            # For perpetuals: assume instant
            return self.basis_bp / 10000

    @property
    def mean_reversion_signal_strength(self) -> float:
        """Signal strength for mean reversion (0-1)"""
        return min(1.0, abs(self.z_score) / 4.0)  # Max at 4 sigma


@dataclass
class BasisSignal:
    """Basis-based trading signal"""

    signal_id: str
    timestamp: datetime
    pair: str
    exchange: str

    # Signal details
    signal_type: BasisSignalType
    direction: str  # "long", "short", "neutral"
    confidence: float  # Signal confidence (0-1)

    # Basis context
    current_basis_bp: float
    z_score: float
    regime: BasisRegime

    # Trade parameters
    entry_rationale: str
    target_hold_hours: float
    stop_loss_bp: int
    take_profit_bp: int

    # Risk management
    max_position_size_pct: float
    correlation_with_spot: float

    # Expected outcomes
    expected_pnl_bp: float
    win_probability: float
    risk_reward_ratio: float

    @property
    def kelly_fraction(self) -> float:
        """Optimal Kelly fraction for position sizing"""
        if self.risk_reward_ratio > 0:
            return (
                self.win_probability * (1 + self.risk_reward_ratio) - 1
            ) / self.risk_reward_ratio
        return 0.0


class BasisAnalyzer:
    """
    Advanced perpetual-spot basis analysis
    """

    def __init__(self):
        self.basis_history = {}  # exchange -> pair -> List[BasisZScore]
        self.signal_history = []

        # Configuration
        self.z_score_window = 168  # 7 days of hourly data
        self.extreme_threshold = 2.0  # 2 sigma
        self.very_extreme_threshold = 3.0  # 3 sigma

        # Basis thresholds (basis points)
        self.contango_thresholds = {
            "normal": 10,  # 10bp
            "high": 50,  # 50bp
            "extreme": 200,  # 200bp
        }

        self.backwardation_thresholds = {
            "normal": -10,  # -10bp
            "high": -50,  # -50bp
            "extreme": -200,  # -200bp
        }

        # Performance tracking
        self.signal_performance = {}

    def calculate_basis_z_scores(
        self, exchange: str, pair: str, basis_data: List[Dict[str, Any]]
    ) -> List[BasisZScore]:
        """Calculate basis z-scores and regime classification"""
        try:
            z_scores = []

            if len(basis_data) < self.z_score_window:
                logger.warning(
                    f"Insufficient data for z-score calculation: {len(basis_data)} < {self.z_score_window}"
                )
                return z_scores

            # Sort by timestamp
            basis_data = sorted(basis_data, key=lambda x: x.get("timestamp", datetime.min))

            # Calculate rolling z-scores
            for i in range(self.z_score_window, len(basis_data)):
                current_data = basis_data[i]
                historical_window = basis_data[i - self.z_score_window : i]

                # Calculate basis
                perp_price = current_data.get("perp_price", 0.0)
                spot_price = current_data.get("spot_price", 0.0)

                if spot_price <= 0:
                    continue

                basis_bp = (perp_price - spot_price) / spot_price * 10000

                # Calculate historical statistics
                historical_basis = []
                for hist_data in historical_window:
                    hist_perp = hist_data.get("perp_price", 0.0)
                    hist_spot = hist_data.get("spot_price", 0.0)
                    if hist_spot > 0:
                        hist_basis = (hist_perp - hist_spot) / hist_spot * 10000
                        historical_basis.append(hist_basis)

                if len(historical_basis) < 10:
                    continue

                # Statistical analysis
                mean_basis = np.mean(historical_basis)
                std_basis = np.std(historical_basis)

                if std_basis <= 0:
                    z_score = 0.0
                else:
                    z_score = (basis_bp - mean_basis) / std_basis

                # Classify regime
                regime = self._classify_basis_regime(basis_bp)

                # Create z-score object
                z_score_obj = BasisZScore(
                    timestamp=current_data.get("timestamp", datetime.now()),
                    pair=pair,
                    exchange=exchange,
                    perp_price=perp_price,
                    spot_price=spot_price,
                    basis_bp=basis_bp,
                    z_score=z_score,
                    lookback_mean=mean_basis,
                    lookback_std=std_basis,
                    lookback_periods=len(historical_basis),
                    regime=regime,
                    is_extreme=abs(z_score) > self.extreme_threshold,
                    is_very_extreme=abs(z_score) > self.very_extreme_threshold,
                    volume_ratio=current_data.get("volume_ratio", 1.0),
                    funding_rate=current_data.get("funding_rate"),
                    time_to_expiry=current_data.get("time_to_expiry"),
                )

                z_scores.append(z_score_obj)

                # Store in history
                self._store_basis_z_score(exchange, pair, z_score_obj)

            return z_scores

        except Exception as e:
            logger.error(f"Basis z-score calculation failed for {exchange} {pair}: {e}")
            return []

    def generate_basis_signals(
        self, exchange: str, pair: str, current_price: float
    ) -> List[BasisSignal]:
        """Generate basis-based trading signals"""
        try:
            # Get recent z-scores
            recent_z_scores = self._get_recent_z_scores(exchange, pair, hours_back=24)

            if not recent_z_scores:
                return []

            latest_z_score = recent_z_scores[-1]
            signals = []

            # Mean reversion signals on extreme basis
            if latest_z_score.is_extreme:
                signal = self._generate_mean_reversion_signal(latest_z_score, recent_z_scores)
                if signal:
                    signals.append(signal)

            # Trend continuation on low basis
            if abs(latest_z_score.z_score) < 0.5:  # Low basis
                signal = self._generate_trend_continuation_signal(latest_z_score, recent_z_scores)
                if signal:
                    signals.append(signal)

            # Momentum signals
            if len(recent_z_scores) >= 5:
                signal = self._generate_momentum_signal(latest_z_score, recent_z_scores)
                if signal:
                    signals.append(signal)

            # Arbitrage signals
            arbitrage_signal = self._generate_arbitrage_signal(latest_z_score)
            if arbitrage_signal:
                signals.append(arbitrage_signal)

            return signals

        except Exception as e:
            logger.error(f"Basis signal generation failed: {e}")
            return []

    def detect_basis_anomalies(
        self, exchange: str, pair: str, lookback_hours: int = 168
    ) -> List[Dict[str, Any]]:
        """Detect basis anomalies and patterns"""
        try:
            z_scores = self._get_recent_z_scores(exchange, pair, hours_back=lookback_hours)

            if len(z_scores) < 20:
                return []

            anomalies = []

            # Extreme z-score anomalies
            for z_score in z_scores[-24:]:  # Last 24 hours
                if z_score.is_very_extreme:
                    anomaly = {
                        "timestamp": z_score.timestamp,
                        "type": "extreme_basis_z_score",
                        "z_score": z_score.z_score,
                        "basis_bp": z_score.basis_bp,
                        "regime": z_score.regime.value,
                        "severity": min(1.0, abs(z_score.z_score) / 5.0),
                    }
                    anomalies.append(anomaly)

            # Persistent extreme basis
            extreme_streak = 0
            for z_score in reversed(z_scores[-10:]):
                if z_score.is_extreme:
                    extreme_streak += 1
                else:
                    break

            if extreme_streak >= 5:
                anomaly = {
                    "timestamp": z_scores[-1].timestamp,
                    "type": "persistent_extreme_basis",
                    "streak_length": extreme_streak,
                    "avg_z_score": np.mean([abs(z.z_score) for z in z_scores[-extreme_streak:]]),
                    "severity": min(1.0, extreme_streak / 10),
                }
                anomalies.append(anomaly)

            # Basis oscillation
            z_score_values = [z.z_score for z in z_scores[-20:]]
            oscillations = sum(
                1
                for i in range(1, len(z_score_values))
                if z_score_values[i] * z_score_values[i - 1] < 0
            )

            if oscillations > len(z_score_values) * 0.4:  # More than 40% sign changes
                anomaly = {
                    "timestamp": z_scores[-1].timestamp,
                    "type": "excessive_basis_oscillation",
                    "oscillation_rate": oscillations / len(z_score_values),
                    "severity": min(1.0, oscillations / len(z_score_values) / 0.5),
                }
                anomalies.append(anomaly)

            return sorted(anomalies, key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Basis anomaly detection failed: {e}")
            return []

    def get_cross_exchange_basis_analysis(self, pair: str, exchanges: List[str]) -> Dict[str, Any]:
        """Analyze basis patterns across exchanges"""
        try:
            cross_analysis = {
                "pair": pair,
                "timestamp": datetime.now(),
                "exchanges_analyzed": exchanges,
                "basis_data": {},
                "arbitrage_opportunities": [],
            }

            # Get latest basis for each exchange
            latest_basis = {}
            for exchange in exchanges:
                recent_z_scores = self._get_recent_z_scores(exchange, pair, hours_back=1)
                if recent_z_scores:
                    latest_z_score = recent_z_scores[-1]
                    latest_basis[exchange] = {
                        "basis_bp": latest_z_score.basis_bp,
                        "z_score": latest_z_score.z_score,
                        "regime": latest_z_score.regime.value,
                        "perp_price": latest_z_score.perp_price,
                        "spot_price": latest_z_score.spot_price,
                        "timestamp": latest_z_score.timestamp,
                    }
                    cross_analysis["basis_data"][exchange] = latest_basis[exchange]

            if len(latest_basis) < 2:
                return cross_analysis

            # Find arbitrage opportunities
            basis_list = [(ex, data["basis_bp"]) for ex, data in latest_basis.items()]
            basis_list.sort(key=lambda x: x[1])  # Sort by basis

            lowest_exchange, lowest_basis = basis_list[0]
            highest_exchange, highest_basis = basis_list[-1]

            basis_spread = highest_basis - lowest_basis

            if abs(basis_spread) > 20:  # 20bp spread threshold
                opportunity = {
                    "type": "cross_exchange_basis_arbitrage",
                    "long_exchange": lowest_exchange if basis_spread > 0 else highest_exchange,
                    "short_exchange": highest_exchange if basis_spread > 0 else lowest_exchange,
                    "basis_spread": abs(basis_spread),
                    "profit_potential_bp": abs(basis_spread) * 0.8,  # 80% capture efficiency
                    "confidence": min(1.0, abs(basis_spread) / 100),
                }
                cross_analysis["arbitrage_opportunities"].append(opportunity)

            # Calculate cross-exchange basis statistics
            if len(latest_basis) > 1:
                all_basis = [data["basis_bp"] for data in latest_basis.values()]
                all_z_scores = [data["z_score"] for data in latest_basis.values()]

                cross_analysis["basis_statistics"] = {
                    "mean_basis_bp": np.mean(all_basis),
                    "std_basis_bp": np.std(all_basis),
                    "range_bp": max(all_basis) - min(all_basis),
                    "mean_z_score": np.mean(all_z_scores),
                    "agreement_score": 1 - np.std(all_z_scores) / (abs(np.mean(all_z_scores)) + 1),
                }

            return cross_analysis

        except Exception as e:
            logger.error(f"Cross-exchange basis analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_basis_analytics(
        self, exchange: Optional[str] = None, pair: Optional[str] = None, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive basis analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter z-scores
            all_z_scores = []
            for ex in self.basis_history:
                if exchange and ex != exchange:
                    continue
                for p in self.basis_history[ex]:
                    if pair and p != pair:
                        continue
                    z_scores = [
                        z_score
                        for z_score in self.basis_history[ex][p]
                        if z_score.timestamp >= cutoff_time
                    ]
                    all_z_scores.extend(z_scores)

            if not all_z_scores:
                return {"status": "no_data"}

            analytics = self._calculate_basis_analytics(all_z_scores)

            # Add signal analysis
            recent_signals = [
                signal
                for signal in self.signal_history
                if signal.timestamp >= cutoff_time
                and (exchange is None or signal.exchange == exchange)
                and (pair is None or signal.pair == pair)
            ]

            analytics["signal_analysis"] = self._analyze_basis_signals(recent_signals)

            return analytics

        except Exception as e:
            logger.error(f"Basis analytics failed: {e}")
            return {"status": "error", "error": str(e)}

    def _classify_basis_regime(self, basis_bp: float) -> BasisRegime:
        """Classify basis regime"""
        if basis_bp >= self.contango_thresholds["extreme"]:
            return BasisRegime.EXTREME_CONTANGO
        elif basis_bp >= self.contango_thresholds["high"]:
            return BasisRegime.HIGH_CONTANGO
        elif basis_bp >= self.contango_thresholds["normal"]:
            return BasisRegime.NORMAL_CONTANGO
        elif basis_bp <= self.backwardation_thresholds["extreme"]:
            return BasisRegime.EXTREME_BACKWARDATION
        elif basis_bp <= self.backwardation_thresholds["high"]:
            return BasisRegime.HIGH_BACKWARDATION
        elif basis_bp <= self.backwardation_thresholds["normal"]:
            return BasisRegime.NORMAL_BACKWARDATION
        else:
            return BasisRegime.NEUTRAL

    def _generate_mean_reversion_signal(
        self, latest_z_score: BasisZScore, recent_z_scores: List[BasisZScore]
    ) -> Optional[BasisSignal]:
        """Generate mean reversion signal for extreme basis"""
        try:
            if not latest_z_score.is_extreme:
                return None

            # Direction opposite to extreme
            direction = "short" if latest_z_score.z_score > 0 else "long"

            # Calculate confidence based on extremeness
            confidence = min(0.9, latest_z_score.mean_reversion_signal_strength)

            # Risk/reward based on z-score
            expected_reversion_bp = abs(latest_z_score.basis_bp - latest_z_score.lookback_mean)
            stop_loss_bp = int(expected_reversion_bp * 0.5)  # 50% of expected move
            take_profit_bp = int(expected_reversion_bp * 0.8)  # 80% of expected move

            signal_id = f"basis_mr_{latest_z_score.exchange}_{latest_z_score.pair}_{latest_z_score.timestamp.timestamp()}"

            signal = BasisSignal(
                signal_id=signal_id,
                timestamp=latest_z_score.timestamp,
                pair=latest_z_score.pair,
                exchange=latest_z_score.exchange,
                signal_type=BasisSignalType.MEAN_REVERSION,
                direction=direction,
                confidence=confidence,
                current_basis_bp=latest_z_score.basis_bp,
                z_score=latest_z_score.z_score,
                regime=latest_z_score.regime,
                entry_rationale=f"Extreme basis z-score {latest_z_score.z_score:.2f} suggests mean reversion",
                target_hold_hours=24.0,
                stop_loss_bp=stop_loss_bp,
                take_profit_bp=take_profit_bp,
                max_position_size_pct=0.05,  # 5% max position
                correlation_with_spot=0.8,  # High correlation assumption
                expected_pnl_bp=take_profit_bp * 0.7,  # 70% probability weighted
                win_probability=0.7,
                risk_reward_ratio=take_profit_bp / max(stop_loss_bp, 1),
            )

            return signal

        except Exception as e:
            logger.error(f"Mean reversion signal generation failed: {e}")
            return None

    def _generate_trend_continuation_signal(
        self, latest_z_score: BasisZScore, recent_z_scores: List[BasisZScore]
    ) -> Optional[BasisSignal]:
        """Generate trend continuation signal for low basis"""
        try:
            if abs(latest_z_score.z_score) > 0.5:  # Not low enough
                return None

            # Check for price momentum
            if len(recent_z_scores) < 5:
                return None

            # Look at recent price changes
            recent_price_changes = []
            for i in range(1, min(5, len(recent_z_scores))):
                curr = recent_z_scores[-i]
                prev = recent_z_scores[-i - 1]
                price_change = (curr.perp_price - prev.perp_price) / prev.perp_price
                recent_price_changes.append(price_change)

            avg_momentum = np.mean(recent_price_changes)

            if abs(avg_momentum) < 0.002:  # Less than 0.2% momentum
                return None

            # Direction follows momentum
            direction = "long" if avg_momentum > 0 else "short"

            # Confidence based on momentum consistency and low basis
            momentum_consistency = 1 - np.std(recent_price_changes) / (abs(avg_momentum) + 0.001)
            low_basis_strength = max(0, 1 - abs(latest_z_score.z_score) / 2)
            confidence = min(0.8, momentum_consistency * low_basis_strength)

            signal_id = f"basis_tc_{latest_z_score.exchange}_{latest_z_score.pair}_{latest_z_score.timestamp.timestamp()}"

            signal = BasisSignal(
                signal_id=signal_id,
                timestamp=latest_z_score.timestamp,
                pair=latest_z_score.pair,
                exchange=latest_z_score.exchange,
                signal_type=BasisSignalType.TREND_CONTINUATION,
                direction=direction,
                confidence=confidence,
                current_basis_bp=latest_z_score.basis_bp,
                z_score=latest_z_score.z_score,
                regime=latest_z_score.regime,
                entry_rationale=f"Low basis ({latest_z_score.z_score:.2f}) with {avg_momentum:.1%} momentum suggests continuation",
                target_hold_hours=48.0,
                stop_loss_bp=100,
                take_profit_bp=200,
                max_position_size_pct=0.03,
                correlation_with_spot=0.9,
                expected_pnl_bp=140,  # 70% win rate * 200bp
                win_probability=0.65,
                risk_reward_ratio=2.0,
            )

            return signal

        except Exception as e:
            logger.error(f"Trend continuation signal generation failed: {e}")
            return None

    def _generate_momentum_signal(
        self, latest_z_score: BasisZScore, recent_z_scores: List[BasisZScore]
    ) -> Optional[BasisSignal]:
        """Generate momentum signal based on basis acceleration"""
        try:
            if len(recent_z_scores) < 5:
                return None

            # Calculate basis momentum (acceleration)
            z_score_values = [z.z_score for z in recent_z_scores[-5:]]

            # First derivative (velocity)
            velocities = np.diff(z_score_values)

            # Second derivative (acceleration)
            if len(velocities) < 2:
                return None

            acceleration = np.diff(velocities)[-1]  # Latest acceleration

            if abs(acceleration) < 0.1:  # Not significant acceleration
                return None

            # Direction based on acceleration (contrarian)
            direction = "short" if acceleration > 0 else "long"

            # Confidence based on acceleration magnitude
            confidence = min(0.75, abs(acceleration) / 0.5)

            signal_id = f"basis_mom_{latest_z_score.exchange}_{latest_z_score.pair}_{latest_z_score.timestamp.timestamp()}"

            signal = BasisSignal(
                signal_id=signal_id,
                timestamp=latest_z_score.timestamp,
                pair=latest_z_score.pair,
                exchange=latest_z_score.exchange,
                signal_type=BasisSignalType.MOMENTUM,
                direction=direction,
                confidence=confidence,
                current_basis_bp=latest_z_score.basis_bp,
                z_score=latest_z_score.z_score,
                regime=latest_z_score.regime,
                entry_rationale=f"Basis acceleration {acceleration:.2f} suggests contrarian opportunity",
                target_hold_hours=16.0,
                stop_loss_bp=75,
                take_profit_bp=150,
                max_position_size_pct=0.02,
                correlation_with_spot=0.7,
                expected_pnl_bp=105,  # 70% win rate * 150bp
                win_probability=0.65,
                risk_reward_ratio=2.0,
            )

            return signal

        except Exception as e:
            logger.error(f"Momentum signal generation failed: {e}")
            return None

    def _generate_arbitrage_signal(self, z_score: BasisZScore) -> Optional[BasisSignal]:
        """Generate arbitrage signal for significant basis"""
        try:
            # Only generate for significant basis
            if abs(z_score.basis_bp) < 30:  # Less than 30bp
                return None

            # Direction based on basis sign
            direction = "short" if z_score.basis_bp > 0 else "long"

            # Confidence based on basis magnitude
            confidence = min(0.8, abs(z_score.basis_bp) / 100)

            # Conservative risk/reward for arbitrage
            expected_capture = abs(z_score.basis_bp) * 0.6  # 60% capture rate
            stop_loss_bp = int(abs(z_score.basis_bp) * 0.3)  # 30% adverse move
            take_profit_bp = int(expected_capture)

            signal_id = (
                f"basis_arb_{z_score.exchange}_{z_score.pair}_{z_score.timestamp.timestamp()}"
            )

            signal = BasisSignal(
                signal_id=signal_id,
                timestamp=z_score.timestamp,
                pair=z_score.pair,
                exchange=z_score.exchange,
                signal_type=BasisSignalType.ARBITRAGE,
                direction=direction,
                confidence=confidence,
                current_basis_bp=z_score.basis_bp,
                z_score=z_score.z_score,
                regime=z_score.regime,
                entry_rationale=f"Basis arbitrage opportunity: {z_score.basis_bp:.1f}bp basis",
                target_hold_hours=8.0,  # Short holding period for arbitrage
                stop_loss_bp=stop_loss_bp,
                take_profit_bp=take_profit_bp,
                max_position_size_pct=0.1,  # Higher size for arbitrage
                correlation_with_spot=0.95,  # Very high correlation for arbitrage
                expected_pnl_bp=expected_capture * 0.8,  # 80% success rate
                win_probability=0.8,
                risk_reward_ratio=take_profit_bp / max(stop_loss_bp, 1),
            )

            return signal

        except Exception as e:
            logger.error(f"Arbitrage signal generation failed: {e}")
            return None

    def _get_recent_z_scores(self, exchange: str, pair: str, hours_back: int) -> List[BasisZScore]:
        """Get recent basis z-scores"""
        try:
            if exchange not in self.basis_history or pair not in self.basis_history[exchange]:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            z_scores = self.basis_history[exchange][pair]

            return [z_score for z_score in z_scores if z_score.timestamp >= cutoff_time]

        except Exception as e:
            logger.error(f"Failed to get recent z-scores: {e}")
            return []

    def _store_basis_z_score(self, exchange: str, pair: str, z_score: BasisZScore) -> None:
        """Store basis z-score in history"""
        try:
            if exchange not in self.basis_history:
                self.basis_history[exchange] = {}

            if pair not in self.basis_history[exchange]:
                self.basis_history[exchange][pair] = []

            self.basis_history[exchange][pair].append(z_score)

            # Keep only recent history (30 days)
            cutoff_time = datetime.now() - timedelta(days=30)
            self.basis_history[exchange][pair] = [
                z for z in self.basis_history[exchange][pair] if z.timestamp >= cutoff_time
            ]

        except Exception as e:
            logger.error(f"Failed to store basis z-score: {e}")

    def _calculate_basis_analytics(self, z_scores: List[BasisZScore]) -> Dict[str, Any]:
        """Calculate comprehensive basis analytics"""
        try:
            if not z_scores:
                return {"status": "no_z_scores"}

            basis_values = [z.basis_bp for z in z_scores]
            z_score_values = [z.z_score for z in z_scores]

            analytics = {
                "total_periods": len(z_scores),
                "basis_statistics": {
                    "mean_basis_bp": np.mean(basis_values),
                    "std_basis_bp": np.std(basis_values),
                    "min_basis_bp": np.min(basis_values),
                    "max_basis_bp": np.max(basis_values),
                    "skewness": self._calculate_skewness(basis_values),
                    "kurtosis": self._calculate_kurtosis(basis_values),
                },
                "z_score_statistics": {
                    "mean_z_score": np.mean(z_score_values),
                    "std_z_score": np.std(z_score_values),
                    "extreme_periods": sum(1 for z in z_scores if z.is_extreme),
                    "very_extreme_periods": sum(1 for z in z_scores if z.is_very_extreme),
                    "max_abs_z_score": max(abs(z) for z in z_score_values),
                },
                "regime_distribution": {},
            }

            # Regime distribution
            for regime in BasisRegime:
                count = sum(1 for z in z_scores if z.regime == regime)
                analytics["regime_distribution"][regime.value] = {
                    "count": count,
                    "percentage": count / len(z_scores),
                }

            # Mean reversion efficiency
            extreme_z_scores = [z for z in z_scores if z.is_extreme]
            if extreme_z_scores:
                reversion_times = []
                for i, extreme_z in enumerate(extreme_z_scores):
                    # Find when it reverted (absolute z-score < 1)
                    start_idx = z_scores.index(extreme_z)
                    for j in range(start_idx + 1, len(z_scores)):
                        if abs(z_scores[j].z_score) < 1.0:
                            reversion_time = (
                                z_scores[j].timestamp - extreme_z.timestamp
                            ).total_seconds() / 3600
                            reversion_times.append(reversion_time)
                            break

                if reversion_times:
                    analytics["mean_reversion_stats"] = {
                        "avg_reversion_time_hours": np.mean(reversion_times),
                        "median_reversion_time_hours": np.median(reversion_times),
                        "reversion_success_rate": len(reversion_times) / len(extreme_z_scores),
                    }

            return analytics

        except Exception as e:
            logger.error(f"Basis analytics calculation failed: {e}")
            return {"status": "error"}

    def _analyze_basis_signals(self, signals: List[BasisSignal]) -> Dict[str, Any]:
        """Analyze basis signal patterns"""
        try:
            if not signals:
                return {"total_signals": 0}

            signal_analysis = {
                "total_signals": len(signals),
                "signal_types": {},
                "average_confidence": np.mean([s.confidence for s in signals]),
                "average_expected_pnl_bp": np.mean([s.expected_pnl_bp for s in signals]),
                "total_expected_pnl_bp": sum(s.expected_pnl_bp for s in signals),
            }

            # Signal type distribution
            for signal_type in BasisSignalType:
                count = sum(1 for s in signals if s.signal_type == signal_type)
                signal_analysis["signal_types"][signal_type.value] = {
                    "count": count,
                    "avg_confidence": np.mean(
                        [s.confidence for s in signals if s.signal_type == signal_type]
                    )
                    if count > 0
                    else 0,
                }

            # Direction distribution
            long_signals = sum(1 for s in signals if s.direction == "long")
            short_signals = sum(1 for s in signals if s.direction == "short")

            signal_analysis["direction_distribution"] = {
                "long": long_signals,
                "short": short_signals,
                "net_bias": (long_signals - short_signals) / len(signals),
            }

            return signal_analysis

        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            return {"total_signals": 0}

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
