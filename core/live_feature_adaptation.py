#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Live Feature Adaptation System
Real-time feature set adjustment based on market regime changes and performance feedback
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
from collections import deque, defaultdict
import warnings

warnings.filterwarnings("ignore")

from .automated_feature_engineering import get_automated_feature_engineer, FeatureEngineeringConfig
from .feature_discovery_engine import get_feature_discovery_engine, DiscoveryConfig
from .shap_regime_analyzer import get_shap_regime_analyzer, SHAPRegimeConfig, MarketRegime


class AdaptationTrigger(Enum):
    REGIME_CHANGE = "regime_change"
    PERFORMANCE_DECLINE = "performance_decline"
    NEW_FEATURE_DISCOVERY = "new_feature_discovery"
    SCHEDULE_BASED = "schedule_based"
    USER_REQUESTED = "user_requested"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_ANOMALY = "volume_anomaly"


@dataclass
class AdaptationConfig:
    """Configuration for live feature adaptation"""

    # Adaptation triggers
    adaptation_frequency_minutes: int = 15
    regime_change_threshold: float = 0.3
    performance_decline_threshold: float = 0.1
    volatility_spike_threshold: float = 2.0

    # Feature selection
    min_feature_performance: float = 0.05
    max_features_per_regime: int = 100
    feature_stability_weight: float = 0.3
    feature_performance_weight: float = 0.7

    # Adaptation aggressiveness
    conservative_adaptation_ratio: float = 0.1  # 10% feature change
    moderate_adaptation_ratio: float = 0.3  # 30% feature change
    aggressive_adaptation_ratio: float = 0.5  # 50% feature change

    # Performance tracking
    performance_history_size: int = 100
    regime_stability_periods: int = 10
    feature_evaluation_window: int = 50

    # Safety mechanisms
    min_adaptation_interval_minutes: int = 5
    max_daily_adaptations: int = 20
    rollback_enabled: bool = True


@dataclass
class AdaptationEvent:
    """Record of a feature adaptation event"""

    timestamp: datetime
    trigger: AdaptationTrigger
    regime_before: Optional[MarketRegime]
    regime_after: Optional[MarketRegime]
    features_before: List[str]
    features_after: List[str]
    performance_before: float
    performance_after: Optional[float] = None
    rollback_performed: bool = False


@dataclass
class FeaturePerformanceMetrics:
    """Performance metrics for feature evaluation"""

    accuracy_score: float
    stability_score: float
    regime_consistency: float
    computational_cost: float
    prediction_confidence: float
    overall_score: float


class LiveFeatureAdaptationEngine:
    """Main engine for live feature adaptation based on market conditions"""

    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()
        self.logger = logging.getLogger(f"{__name__}.LiveFeatureAdaptationEngine")

        # Core components
        self.feature_engineer = get_automated_feature_engineer()
        self.discovery_engine = get_feature_discovery_engine()
        self.shap_analyzer = get_shap_regime_analyzer()

        # Current state
        self.current_features: List[str] = []
        self.current_regime: Optional[MarketRegime] = None
        self.current_performance: float = 0.0

        # Adaptation tracking
        self.adaptation_history: deque = deque(maxlen=self.config.performance_history_size)
        self.feature_performance_cache: Dict[str, FeaturePerformanceMetrics] = {}
        self.regime_feature_cache: Dict[MarketRegime, List[str]] = {}

        # Performance monitoring
        self.performance_monitor = deque(maxlen=self.config.performance_history_size)
        self.regime_stability_counter = 0
        self.last_adaptation_time: Optional[datetime] = None
        self.daily_adaptation_count = 0

        # Safety and rollback
        self.feature_snapshots: deque = deque(maxlen=10)  # Keep last 10 snapshots
        self.rollback_candidate: Optional[List[str]] = None

        self._lock = threading.RLock()

        # Start monitoring threads
        self._start_monitoring_threads()

        self.logger.info("Live Feature Adaptation Engine initialized")

    def adapt_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        trigger: AdaptationTrigger,
        force_adaptation: bool = False,
    ) -> List[str]:
        """Main feature adaptation method"""
        with self._lock:
            try:
                self.logger.info(f"Starting feature adaptation - trigger: {trigger.value}")

                # Safety checks
                if not self._can_adapt(force_adaptation):
                    self.logger.info("Adaptation skipped due to safety constraints")
                    return self.current_features

                # Save current state for potential rollback
                self._save_feature_snapshot()

                # Detect current regime
                current_regime = self._detect_current_regime(data, target_column)
                regime_changed = self.current_regime != current_regime

                # Determine adaptation strategy
                adaptation_strategy = self._determine_adaptation_strategy(trigger, regime_changed)

                # Generate new feature set
                new_features = self._generate_adapted_features(
                    data, target_column, current_regime, adaptation_strategy
                )

                # Evaluate new feature set
                performance_metrics = self._evaluate_feature_set(data, target_column, new_features)

                # Decide whether to accept adaptation
                if self._should_accept_adaptation(performance_metrics):
                    # Create adaptation event
                    adaptation_event = AdaptationEvent(
                        timestamp=datetime.now(),
                        trigger=trigger,
                        regime_before=self.current_regime,
                        regime_after=current_regime,
                        features_before=self.current_features.copy(),
                        features_after=new_features.copy(),
                        performance_before=self.current_performance,
                    )

                    # Apply adaptation
                    self.current_features = new_features
                    self.current_regime = current_regime
                    self.current_performance = performance_metrics.overall_score

                    # Update tracking
                    self.adaptation_history.append(adaptation_event)
                    self.last_adaptation_time = datetime.now()
                    self.daily_adaptation_count += 1

                    # Update feature performance cache
                    self._update_feature_performance_cache(new_features, performance_metrics)

                    self.logger.info(
                        f"Feature adaptation successful: {len(new_features)} features selected"
                    )
                    self.logger.info(f"Performance improvement: {self.current_performance:.4f}")

                    return new_features
                else:
                    self.logger.info("Adaptation rejected due to poor performance")
                    return self.current_features

            except Exception as e:
                self.logger.error(f"Feature adaptation failed: {e}")
                return self.current_features

    def _can_adapt(self, force_adaptation: bool) -> bool:
        """Check if adaptation is allowed based on safety constraints"""
        try:
            # Force adaptation bypasses most checks
            if force_adaptation:
                return True

            # Check minimum interval
            if self.last_adaptation_time:
                time_since_last = datetime.now() - self.last_adaptation_time
                min_interval = timedelta(minutes=self.config.min_adaptation_interval_minutes)

                if time_since_last < min_interval:
                    return False

            # Check daily limit
            today = datetime.now().date()
            adaptations_today = sum(
                1 for event in self.adaptation_history if event.timestamp.date() == today
            )

            if adaptations_today >= self.config.max_daily_adaptations:
                return False

            return True

        except Exception:
            return False

    def _detect_current_regime(self, data: pd.DataFrame, target_column: str) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Use SHAP analyzer for regime detection
            regime_results = self.shap_analyzer.analyze_regime_specific_importance(
                data, target_column
            )

            if regime_results:
                # Get most confident regime
                best_regime = max(
                    regime_results.keys(), key=lambda r: regime_results[r].confidence_score
                )
                return best_regime

            # Fallback regime detection
            if len(data) < 20:
                return MarketRegime.SIDEWAYS

            recent_data = data.tail(20)
            if target_column not in recent_data.columns:
                return MarketRegime.SIDEWAYS

            prices = recent_data[target_column].fillna(method="ffill")
            returns = prices.pct_change().fillna(0)
            volatility = returns.std()

            # Simple regime classification
            if volatility > 0.05:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:
                return MarketRegime.LOW_VOLATILITY
            elif returns.mean() > 0.02:
                return MarketRegime.TRENDING_UP
            elif returns.mean() < -0.02:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return MarketRegime.SIDEWAYS

    def _determine_adaptation_strategy(
        self, trigger: AdaptationTrigger, regime_changed: bool
    ) -> str:
        """Determine how aggressive the adaptation should be"""
        try:
            if trigger == AdaptationTrigger.REGIME_CHANGE and regime_changed:
                return "aggressive"
            elif trigger == AdaptationTrigger.PERFORMANCE_DECLINE:
                return "moderate"
            elif trigger == AdaptationTrigger.VOLATILITY_SPIKE:
                return "aggressive"
            elif trigger == AdaptationTrigger.NEW_FEATURE_DISCOVERY:
                return "conservative"
            else:
                return "conservative"

        except Exception:
            return "conservative"

    def _generate_adapted_features(
        self, data: pd.DataFrame, target_column: str, regime: MarketRegime, strategy: str
    ) -> List[str]:
        """Generate new feature set based on adaptation strategy"""
        try:
            # Get regime-optimized features from SHAP analyzer
            regime_features = self.shap_analyzer.get_regime_optimized_features(regime)

            # Get newly discovered features
            discovered_features = self.discovery_engine.discover_features(data, target_column)

            # Combine feature sources
            candidate_features = set(regime_features)

            # Add discovered features
            for feature_candidate in discovered_features:
                if feature_candidate.performance_score > self.config.min_feature_performance:
                    candidate_features.add(feature_candidate.name)

            # Add some existing high-performing features for stability
            if self.current_features:
                top_current = self._get_top_performing_current_features(5)
                candidate_features.update(top_current)

            # Convert to list and limit
            candidate_list = list(candidate_features)

            if len(candidate_list) > self.config.max_features_per_regime:
                # Score and select top features
                scored_features = self._score_feature_candidates(candidate_list, regime)
                candidate_list = [
                    f[0] for f in scored_features[: self.config.max_features_per_regime]
                ]

            # Apply adaptation strategy
            if strategy == "conservative":
                adaptation_ratio = self.config.conservative_adaptation_ratio
            elif strategy == "moderate":
                adaptation_ratio = self.config.moderate_adaptation_ratio
            else:  # aggressive
                adaptation_ratio = self.config.aggressive_adaptation_ratio

            # Blend with current features based on strategy
            if self.current_features and strategy != "aggressive":
                keep_count = int(len(self.current_features) * (1 - adaptation_ratio))
                new_count = len(candidate_list) - keep_count

                # Keep top performing current features
                kept_features = self._get_top_performing_current_features(keep_count)

                # Add new features
                new_features = [f for f in candidate_list if f not in kept_features][:new_count]

                final_features = list(set(kept_features + new_features))
            else:
                final_features = candidate_list

            return final_features[: self.config.max_features_per_regime]

        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            return self.current_features if self.current_features else []

    def _score_feature_candidates(
        self, features: List[str], regime: MarketRegime
    ) -> List[Tuple[str, float]]:
        """Score feature candidates for selection"""
        try:
            scored_features = []

            for feature in features:
                score = 0.0

                # Get cached performance if available
                if feature in self.feature_performance_cache:
                    metrics = self.feature_performance_cache[feature]
                    score = metrics.overall_score
                else:
                    # Estimate score based on heuristics
                    score = 0.5  # Default score

                    # Boost score for regime-specific features
                    if regime in self.regime_feature_cache:
                        if feature in self.regime_feature_cache[regime]:
                            score += 0.2

                    # Boost score for current high-performing features
                    if feature in self.current_features:
                        top_current = self._get_top_performing_current_features(10)
                        if feature in top_current:
                            score += 0.3

                scored_features.append((feature, score))

            # Sort by score descending
            scored_features.sort(key=lambda x: x[1], reverse=True)
            return scored_features

        except Exception as e:
            self.logger.error(f"Feature scoring failed: {e}")
            return [(f, 0.5) for f in features]

    def _evaluate_feature_set(
        self, data: pd.DataFrame, target_column: str, features: List[str]
    ) -> FeaturePerformanceMetrics:
        """Evaluate performance of a feature set"""
        try:
            if not features:
                return FeaturePerformanceMetrics(0, 0, 0, 0, 0, 0)

            # Basic evaluation using correlation and variance
            available_features = [f for f in features if f in data.columns]

            if not available_features:
                return FeaturePerformanceMetrics(0, 0, 0, 0, 0, 0)

            feature_data = data[available_features].fillna(0)
            target_data = data[target_column].fillna(0)

            # Calculate metrics
            correlations = feature_data.corrwith(target_data).abs()
            accuracy_score = correlations.mean()

            # Stability score (inverse of variance)
            variances = feature_data.var()
            stability_score = 1.0 / (1.0 + variances.mean())

            # Regime consistency (placeholder)
            regime_consistency = 0.7

            # Computational cost (based on feature count)
            computational_cost = min(1.0, len(available_features) / 100)

            # Prediction confidence (placeholder)
            prediction_confidence = accuracy_score * 0.8

            # Overall score
            overall_score = (
                accuracy_score * self.config.feature_performance_weight
                + stability_score * self.config.feature_stability_weight
            )

            return FeaturePerformanceMetrics(
                accuracy_score=accuracy_score,
                stability_score=stability_score,
                regime_consistency=regime_consistency,
                computational_cost=computational_cost,
                prediction_confidence=prediction_confidence,
                overall_score=overall_score,
            )

        except Exception as e:
            self.logger.error(f"Feature evaluation failed: {e}")
            return FeaturePerformanceMetrics(0, 0, 0, 0, 0, 0)

    def _should_accept_adaptation(self, new_metrics: FeaturePerformanceMetrics) -> bool:
        """Decide whether to accept the new feature adaptation"""
        try:
            # Always accept if no current performance
            if self.current_performance == 0:
                return True

            # Accept if performance improves
            improvement = new_metrics.overall_score - self.current_performance

            if improvement > self.config.performance_decline_threshold:
                return True

            # Accept small decline if stability improves significantly
            if (
                improvement > -self.config.performance_decline_threshold * 0.5
                and new_metrics.stability_score > 0.8
            ):
                return True

            return False

        except Exception:
            return False

    def _get_top_performing_current_features(self, n: int) -> List[str]:
        """Get top N performing features from current set"""
        try:
            if not self.current_features:
                return []

            # Score current features based on cached performance
            scored_features = []

            for feature in self.current_features:
                if feature in self.feature_performance_cache:
                    score = self.feature_performance_cache[feature].overall_score
                else:
                    score = 0.5  # Default score

                scored_features.append((feature, score))

            # Sort and return top N
            scored_features.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in scored_features[:n]]

        except Exception:
            return self.current_features[:n] if self.current_features else []

    def _update_feature_performance_cache(
        self, features: List[str], metrics: FeaturePerformanceMetrics
    ):
        """Update feature performance cache"""
        try:
            # Update individual feature metrics (simplified)
            for feature in features:
                self.feature_performance_cache[feature] = FeaturePerformanceMetrics(
                    accuracy_score=metrics.accuracy_score,
                    stability_score=metrics.stability_score,
                    regime_consistency=metrics.regime_consistency,
                    computational_cost=metrics.computational_cost / len(features),
                    prediction_confidence=metrics.prediction_confidence,
                    overall_score=metrics.overall_score,
                )

            # Update regime cache
            if self.current_regime:
                self.regime_feature_cache[self.current_regime] = features.copy()

        except Exception as e:
            self.logger.error(f"Performance cache update failed: {e}")

    def _save_feature_snapshot(self):
        """Save current feature state for potential rollback"""
        try:
            if self.current_features:
                snapshot = {
                    "features": self.current_features.copy(),
                    "regime": self.current_regime,
                    "performance": self.current_performance,
                    "timestamp": datetime.now(),
                }
                self.feature_snapshots.append(snapshot)

        except Exception as e:
            self.logger.error(f"Feature snapshot failed: {e}")

    def rollback_adaptation(self) -> bool:
        """Rollback to previous feature set"""
        with self._lock:
            try:
                if not self.config.rollback_enabled:
                    self.logger.warning("Rollback disabled in configuration")
                    return False

                if not self.feature_snapshots:
                    self.logger.warning("No snapshots available for rollback")
                    return False

                # Get most recent snapshot
                snapshot = self.feature_snapshots[-1]

                # Restore state
                self.current_features = snapshot["features"]
                self.current_regime = snapshot["regime"]
                self.current_performance = snapshot["performance"]

                # Mark last adaptation as rolled back
                if self.adaptation_history:
                    self.adaptation_history[-1].rollback_performed = True

                self.logger.info(
                    f"Rollback successful: restored {len(self.current_features)} features"
                )
                return True

            except Exception as e:
                self.logger.error(f"Rollback failed: {e}")
                return False

    def _start_monitoring_threads(self):
        """Start background monitoring threads"""

        def adaptation_monitor():
            """Monitor for adaptation triggers"""
            while True:
                try:
                    time.sleep(self.config.adaptation_frequency_minutes * 60)

                    # Reset daily counter at midnight
                    if datetime.now().hour == 0 and datetime.now().minute < 15:
                        self.daily_adaptation_count = 0

                    # Scheduled adaptation check would go here
                    # This is a simplified version

                except Exception as e:
                    self.logger.error(f"Adaptation monitor error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        def performance_monitor():
            """Monitor performance and trigger adaptations if needed"""
            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes

                    # Performance monitoring logic would go here
                    # This is a placeholder

                except Exception as e:
                    self.logger.error(f"Performance monitor error: {e}")
                    time.sleep(300)

        # Start threads
        adaptation_thread = threading.Thread(target=adaptation_monitor, daemon=True)
        performance_thread = threading.Thread(target=performance_monitor, daemon=True)

        adaptation_thread.start()
        performance_thread.start()

        self.logger.info("Background monitoring threads started")

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary"""
        with self._lock:
            return {
                "current_features_count": len(self.current_features),
                "current_regime": self.current_regime.value if self.current_regime else None,
                "current_performance": self.current_performance,
                "adaptations_today": sum(
                    1
                    for event in self.adaptation_history
                    if event.timestamp.date() == datetime.now().date(),
                "total_adaptations": len(self.adaptation_history),
                "last_adaptation": self.last_adaptation_time.isoformat()
                if self.last_adaptation_time
                else None,
                "adaptation_triggers": {
                    trigger.value: sum(
                        1 for event in self.adaptation_history if event.trigger == trigger
                    )
                    for trigger in AdaptationTrigger
                },
                "regime_stability": self.regime_stability_counter,
                "snapshots_available": len(self.feature_snapshots),
                "rollback_enabled": self.config.rollback_enabled,
                "feature_performance_cache_size": len(self.feature_performance_cache),
                "regime_feature_cache": {
                    regime.value: len(features)
                    for regime, features in self.regime_feature_cache.items()
                },
            }

    def force_adaptation(
        self,
        data: pd.DataFrame,
        target_column: str,
        trigger: AdaptationTrigger = AdaptationTrigger.USER_REQUESTED,
    ) -> List[str]:
        """Force feature adaptation regardless of safety constraints"""
        return self.adapt_features(data, target_column, trigger, force_adaptation=True)


# Singleton live adaptation engine
_live_adaptation_engine = None
_lae_lock = threading.Lock()


def get_live_adaptation_engine(
    config: Optional[AdaptationConfig] = None,
) -> LiveFeatureAdaptationEngine:
    """Get the singleton live adaptation engine"""
    global _live_adaptation_engine

    with _lae_lock:
        if _live_adaptation_engine is None:
            _live_adaptation_engine = LiveFeatureAdaptationEngine(config)
        return _live_adaptation_engine


def adapt_features_for_regime(
    data: pd.DataFrame,
    target_column: str,
    trigger: AdaptationTrigger = AdaptationTrigger.REGIME_CHANGE,
) -> List[str]:
    """Convenient function to adapt features for current conditions"""
    engine = get_live_adaptation_engine()
    return engine.adapt_features(data, target_column, trigger)
