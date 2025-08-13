"""
Signal Decay Management System

Implements time-to-live (TTL) and decay mechanisms for trading signals
to prevent stale signals from affecting trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from .base_models import ModelPrediction

logger = logging.getLogger(__name__)


class DecayFunction(Enum):
    """Signal decay function types"""

    LINEAR = "linear"  # Linear decay
    EXPONENTIAL = "exponential"  # Exponential decay
    STEP = "step"  # Step function (valid/invalid)
    SIGMOID = "sigmoid"  # Sigmoid decay


@dataclass
class DecayConfig:
    """Configuration for signal decay"""

    default_ttl_hours: float = 4.0  # Default signal TTL
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL

    # Model-specific TTL settings
    model_ttl_overrides: Dict[str, float] = None

    # Decay parameters
    half_life_factor: float = 0.5  # At what fraction of TTL is signal half strength
    min_signal_strength: float = 0.1  # Minimum signal strength before removal

    # Update frequency
    cleanup_frequency_minutes: int = 15  # How often to clean up expired signals


@dataclass
class DecayedSignal:
    """Signal with decay applied"""

    original_prediction: ModelPrediction
    current_strength: float  # Current signal strength (0-1)
    time_remaining_hours: float  # Hours until complete decay
    decay_factor: float  # Current decay multiplier
    is_expired: bool  # Whether signal is considered expired


class SignalDecayManager:
    """
    Manages signal decay and TTL for trading signals
    """

    def __init__(self, config: DecayConfig):
        self.config = config
        if self.config.model_ttl_overrides is None:
            self.config.model_ttl_overrides = {}

        # Active signals storage
        self.active_signals = {}  # symbol -> list of DecayedSignal
        self.last_cleanup_time = datetime.now()

    def add_signal(self, prediction: ModelPrediction) -> DecayedSignal:
        """
        Add a new signal with decay tracking

        Args:
            prediction: Model prediction to track

        Returns:
            DecayedSignal object with initial state
        """
        try:
            # Get TTL for this model type
            ttl_hours = self._get_model_ttl(prediction.model_name)

            # Create decayed signal
            decayed_signal = DecayedSignal(
                original_prediction=prediction,
                current_strength=1.0,  # Full strength initially
                time_remaining_hours=ttl_hours,
                decay_factor=1.0,
                is_expired=False,
            )

            # Store signal
            symbol = prediction.symbol
            if symbol not in self.active_signals:
                self.active_signals[symbol] = []

            self.active_signals[symbol].append(decayed_signal)

            logger.debug(
                f"Added signal for {symbol} from {prediction.model_name}, TTL: {ttl_hours}h"
            )

            return decayed_signal

        except Exception as e:
            logger.error(f"Failed to add signal: {e}")
            raise

    def get_active_signals(self, symbol: str, update_decay: bool = True) -> List[DecayedSignal]:
        """
        Get active signals for a symbol with current decay applied

        Args:
            symbol: Symbol to get signals for
            update_decay: Whether to update decay factors

        Returns:
            List of active (non-expired) signals with current decay
        """
        try:
            if symbol not in self.active_signals:
                return []

            current_time = datetime.now()
            active_signals = []

            for signal in self.active_signals[symbol]:
                if update_decay:
                    self._update_signal_decay(signal, current_time)

                if not signal.is_expired:
                    active_signals.append(signal)

            return active_signals

        except Exception as e:
            logger.error(f"Failed to get active signals for {symbol}: {e}")
            return []

    def get_weighted_prediction(
        self, symbol: str, combine_method: str = "weighted_average"
    ) -> Optional[ModelPrediction]:
        """
        Get combined prediction from all active signals for a symbol

        Args:
            symbol: Symbol to get prediction for
            combine_method: How to combine signals ('weighted_average', 'max_confidence', 'consensus')

        Returns:
            Combined prediction or None if no active signals
        """
        try:
            active_signals = self.get_active_signals(symbol, update_decay=True)

            if not active_signals:
                return None

            if len(active_signals) == 1:
                # Single signal - apply decay factor
                signal = active_signals[0]
                return self._apply_decay_to_prediction(signal)

            # Multiple signals - combine using specified method
            if combine_method == "weighted_average":
                return self._weighted_average_combine(active_signals)
            elif combine_method == "max_confidence":
                return self._max_confidence_combine(active_signals)
            elif combine_method == "consensus":
                return self._consensus_combine(active_signals)
            else:
                logger.warning(f"Unknown combine method: {combine_method}, using weighted_average")
                return self._weighted_average_combine(active_signals)

        except Exception as e:
            logger.error(f"Failed to get weighted prediction for {symbol}: {e}")
            return None

    def cleanup_expired_signals(self, force: bool = False) -> Dict[str, int]:
        """
        Remove expired signals from storage

        Args:
            force: Force cleanup regardless of timing

        Returns:
            Dict with cleanup statistics
        """
        try:
            current_time = datetime.now()

            # Check if cleanup is due
            if not force:
                time_since_cleanup = current_time - self.last_cleanup_time
                minutes_since = time_since_cleanup.total_seconds() / 60

                if minutes_since < self.config.cleanup_frequency_minutes:
                    return {"status": "cleanup_not_due"}

            cleanup_stats = {"symbols_processed": 0, "signals_removed": 0, "signals_remaining": 0}

            for symbol in list(self.active_signals.keys()):
                signals = self.active_signals[symbol]
                cleanup_stats["symbols_processed"] += 1

                # Update decay and filter expired signals
                active_signals = []
                for signal in signals:
                    self._update_signal_decay(signal, current_time)

                    if signal.is_expired:
                        cleanup_stats["signals_removed"] += 1
                    else:
                        active_signals.append(signal)

                # Update storage
                if active_signals:
                    self.active_signals[symbol] = active_signals
                    cleanup_stats["signals_remaining"] += len(active_signals)
                else:
                    # Remove symbol if no active signals
                    del self.active_signals[symbol]

            self.last_cleanup_time = current_time

            logger.info(f"Signal cleanup completed: {cleanup_stats}")

            return cleanup_stats

        except Exception as e:
            logger.error(f"Signal cleanup failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_decay_analytics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics about signal decay and TTL

        Args:
            symbol: Specific symbol to analyze (None for all)

        Returns:
            Comprehensive decay analytics
        """
        try:
            analytics = {
                "total_symbols": len(self.active_signals),
                "total_active_signals": sum(
                    len(signals) for signals in self.active_signals.values(),
                "config": {
                    "default_ttl_hours": self.config.default_ttl_hours,
                    "decay_function": self.config.decay_function.value,
                    "min_signal_strength": self.config.min_signal_strength,
                },
            }

            # Per-model statistics
            model_stats = {}
            signal_ages = []
            signal_strengths = []

            symbols_to_analyze = [symbol] if symbol else list(self.active_signals.keys())

            for sym in symbols_to_analyze:
                if sym not in self.active_signals:
                    continue

                for signal in self.active_signals[sym]:
                    model_name = signal.original_prediction.model_name

                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            "count": 0,
                            "avg_strength": 0.0,
                            "avg_age_hours": 0.0,
                            "expired_count": 0,
                        }

                    stats = model_stats[model_name]
                    stats["count"] += 1

                    # Calculate signal age
                    age_hours = self._calculate_signal_age(signal)
                    signal_ages.append(age_hours)
                    signal_strengths.append(signal.current_strength)

                    stats["avg_strength"] += signal.current_strength
                    stats["avg_age_hours"] += age_hours

                    if signal.is_expired:
                        stats["expired_count"] += 1

            # Finalize model statistics
            for model_name, stats in model_stats.items():
                if stats["count"] > 0:
                    stats["avg_strength"] /= stats["count"]
                    stats["avg_age_hours"] /= stats["count"]

            analytics["model_statistics"] = model_stats

            # Overall statistics
            if signal_ages:
                analytics["overall_statistics"] = {
                    "avg_signal_age_hours": np.mean(signal_ages),
                    "median_signal_age_hours": np.median(signal_ages),
                    "avg_signal_strength": np.mean(signal_strengths),
                    "median_signal_strength": np.median(signal_strengths),
                    "strength_std": np.std(signal_strengths),
                }

            # Symbol-specific analytics if requested
            if symbol and symbol in self.active_signals:
                symbol_signals = self.active_signals[symbol]
                analytics["symbol_details"] = {
                    "active_signals": len(symbol_signals),
                    "model_breakdown": {},
                    "age_distribution": {},
                }

                for signal in symbol_signals:
                    model_name = signal.original_prediction.model_name
                    analytics["symbol_details"]["model_breakdown"][model_name] = (
                        analytics["symbol_details"]["model_breakdown"].get(model_name, 0) + 1
                    )

            return analytics

        except Exception as e:
            logger.error(f"Decay analytics generation failed: {e}")
            return {"status": "Error", "error": str(e)}

    def _get_model_ttl(self, model_name: str) -> float:
        """Get TTL for specific model"""
        return self.config.model_ttl_overrides.get(model_name, self.config.default_ttl_hours)

    def _update_signal_decay(self, signal: DecayedSignal, current_time: datetime) -> None:
        """Update decay factor for a signal"""
        try:
            # Calculate age of signal
            age = current_time - signal.original_prediction.timestamp
            age_hours = age.total_seconds() / 3600

            # Get TTL for this signal
            ttl_hours = signal.original_prediction.ttl_hours

            # Update time remaining
            signal.time_remaining_hours = max(0, ttl_hours - age_hours)

            # Calculate decay factor based on decay function
            if age_hours >= ttl_hours:
                # Signal has expired
                signal.current_strength = 0.0
                signal.decay_factor = 0.0
                signal.is_expired = True
            else:
                # Calculate decay factor
                time_fraction = age_hours / ttl_hours

                if self.config.decay_function == DecayFunction.LINEAR:
                    decay_factor = 1.0 - time_fraction
                elif self.config.decay_function == DecayFunction.EXPONENTIAL:
                    # Exponential decay: strength = e^(-t/tau)
                    tau = ttl_hours * self.config.half_life_factor
                    decay_factor = np.exp(-age_hours / tau)
                elif self.config.decay_function == DecayFunction.STEP:
                    # Step function - full strength until expiry
                    decay_factor = 1.0
                elif self.config.decay_function == DecayFunction.SIGMOID:
                    # Sigmoid decay - slow at first, then rapid
                    steepness = 5.0  # Controls steepness of decay
                    midpoint = ttl_hours * 0.7  # Decay accelerates at 70% of TTL
                    decay_factor = 1.0 / (
                        1.0 + np.exp(steepness * (age_hours - midpoint) / ttl_hours)
                else:
                    decay_factor = 1.0 - time_fraction  # Default to linear

                # Apply model-specific decay factor
                model_decay = signal.original_prediction.decay_factor
                decay_factor *= model_decay

                signal.decay_factor = decay_factor
                signal.current_strength = decay_factor

                # Check if signal is below minimum strength
                if signal.current_strength < self.config.min_signal_strength:
                    signal.is_expired = True

        except Exception as e:
            logger.error(f"Signal decay update failed: {e}")
            signal.is_expired = True

    def _calculate_signal_age(self, signal: DecayedSignal) -> float:
        """Calculate age of signal in hours"""
        try:
            age = datetime.now() - signal.original_prediction.timestamp
            return age.total_seconds() / 3600
        except Exception:
            return 0.0

    def _apply_decay_to_prediction(self, signal: DecayedSignal) -> ModelPrediction:
        """Apply decay factor to a prediction"""
        try:
            original = signal.original_prediction

            # Create new prediction with decay applied
            decayed_prediction = ModelPrediction(
                model_name=original.model_name,
                symbol=original.symbol,
                timestamp=original.timestamp,
                probability=original.probability,  # Keep original probability
                confidence=original.confidence * signal.decay_factor,  # Apply decay to confidence
                direction=original.direction,
                feature_importance=original.feature_importance,
                explanation=f"{original.explanation} (decay: {signal.decay_factor:.2f})",
                model_version=original.model_version,
                training_data_end=original.training_data_end,
                ttl_hours=original.ttl_hours,
                decay_factor=signal.decay_factor,
            )

            return decayed_prediction

        except Exception as e:
            logger.error(f"Failed to apply decay to prediction: {e}")
            return signal.original_prediction

    def _weighted_average_combine(self, signals: List[DecayedSignal]) -> ModelPrediction:
        """Combine signals using weighted average"""
        try:
            if not signals:
                return None

            # Use decay factors as weights
            weights = [signal.current_strength for signal in signals]
            total_weight = sum(weights)

            if total_weight == 0:
                return None

            # Normalize weights
            weights = [w / total_weight for w in weights]

            # Weighted average of probabilities
            weighted_prob = sum(
                w * signal.original_prediction.probability for w, signal in zip(weights, signals)

            # Weighted average of confidences (already decayed)
            weighted_confidence = sum(
                w * signal.original_prediction.confidence * signal.decay_factor
                for w, signal in zip(weights, signals)

            # Determine direction from weighted probability
            direction = "up" if weighted_prob > 0.5 else "down"

            # Combine feature importance
            combined_importance = {}
            for signal in signals:
                for feature, importance in signal.original_prediction.feature_importance.items():
                    combined_importance[feature] = combined_importance.get(feature, 0) + importance

            # Create combined prediction
            first_signal = signals[0].original_prediction
            combined_prediction = ModelPrediction(
                model_name="ensemble_decayed",
                symbol=first_signal.symbol,
                timestamp=datetime.now(),
                probability=weighted_prob,
                confidence=weighted_confidence,
                direction=direction,
                feature_importance=combined_importance,
                explanation=f"Combined from {len(signals)} decayed signals",
                model_version="1.0.0",
                training_data_end=None,
                ttl_hours=min(signal.time_remaining_hours for signal in signals),
                decay_factor=np.mean([signal.decay_factor for signal in signals]),
            )

            return combined_prediction

        except Exception as e:
            logger.error(f"Weighted average combination failed: {e}")
            return None

    def _max_confidence_combine(self, signals: List[DecayedSignal]) -> ModelPrediction:
        """Use signal with highest decayed confidence"""
        try:
            if not signals:
                return None

            # Find signal with highest decayed confidence
            best_signal = max(
                signals, key=lambda s: s.original_prediction.confidence * s.decay_factor
            )

            return self._apply_decay_to_prediction(best_signal)

        except Exception as e:
            logger.error(f"Max confidence combination failed: {e}")
            return None

    def _consensus_combine(self, signals: List[DecayedSignal]) -> ModelPrediction:
        """Combine signals using consensus approach"""
        try:
            if not signals:
                return None

            # Count votes weighted by decay factors
            up_weight = sum(
                signal.decay_factor
                for signal in signals
                if signal.original_prediction.direction == "up"
            )

            down_weight = sum(
                signal.decay_factor
                for signal in signals
                if signal.original_prediction.direction == "down"
            )

            total_weight = up_weight + down_weight

            if total_weight == 0:
                return None

            # Consensus probability
            consensus_prob = up_weight / total_weight

            # Average confidence weighted by decay
            avg_confidence = sum(
                signal.original_prediction.confidence * signal.decay_factor for signal in signals
            ) / len(signals)

            # Direction from consensus
            direction = "up" if consensus_prob > 0.5 else "down"

            # Create consensus prediction
            first_signal = signals[0].original_prediction
            consensus_prediction = ModelPrediction(
                model_name="consensus_decayed",
                symbol=first_signal.symbol,
                timestamp=datetime.now(),
                probability=consensus_prob,
                confidence=avg_confidence,
                direction=direction,
                feature_importance={},
                explanation=f"Consensus from {len(signals)} decayed signals",
                model_version="1.0.0",
                training_data_end=None,
                ttl_hours=min(signal.time_remaining_hours for signal in signals),
                decay_factor=np.mean([signal.decay_factor for signal in signals]),
            )

            return consensus_prediction

        except Exception as e:
            logger.error(f"Consensus combination failed: {e}")
            return None
