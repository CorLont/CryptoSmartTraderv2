#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - SHAP Regime Analyzer
Advanced SHAP analysis with live regime-specific feature adaptation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"

@dataclass
class SHAPRegimeConfig:
    """Configuration for SHAP regime analyzer"""
    # Analysis parameters
    min_samples_per_regime: int = 50
    max_features_to_analyze: int = 100
    shap_sample_size: int = 500

    # Regime detection
    regime_lookback_periods: int = 50
    volatility_threshold_high: float = 0.05
    volatility_threshold_low: float = 0.01
    trend_threshold: float = 0.02

    # Feature adaptation
    feature_importance_threshold: float = 0.01
    regime_feature_overlap_threshold: float = 0.3
    adaptive_feature_ratio: float = 0.7

    # Analysis frequency
    analysis_frequency_minutes: int = 30
    regime_stability_periods: int = 5

@dataclass
class SHAPAnalysisResult:
    """Result of SHAP analysis for a specific regime"""
    regime: MarketRegime
    feature_importance: Dict[str, float]
    shap_values: np.ndarray
    base_value: float
    model_performance: Dict[str, float]
    analysis_timestamp: datetime
    sample_count: int
    confidence_score: float

@dataclass
class RegimeFeatureSet:
    """Feature set optimized for specific market regime"""
    regime: MarketRegime
    selected_features: List[str]
    feature_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    stability_score: float

class SHAPRegimeAnalyzer:
    """Advanced SHAP analyzer with market regime specialization"""

    def __init__(self, config: Optional[SHAPRegimeConfig] = None):
        self.config = config or SHAPRegimeConfig()
        self.logger = logging.getLogger(f"{__name__}.SHAPRegimeAnalyzer")

        if not HAS_SHAP:
            self.logger.error("SHAP not available - analyzer disabled")
            # Still initialize basic structure

        if not HAS_SKLEARN:
            self.logger.error("Scikit-learn not available - analyzer disabled")
            # Still initialize basic structure

        # SHAP explainers for different regimes
        self.regime_explainers: Dict[MarketRegime, Any] = {}
        self.regime_models: Dict[MarketRegime, Any] = {}

        # Analysis results
        self.regime_analysis_results: Dict[MarketRegime, SHAPAnalysisResult] = {}
        self.global_shap_values: Optional[np.ndarray] = None
        self.global_feature_importance: Dict[str, float] = {}

        # Regime-specific feature sets
        self.regime_feature_sets: Dict[MarketRegime, RegimeFeatureSet] = {}
        self.current_regime: Optional[MarketRegime] = None

        # Analysis history
        self.analysis_history: List[SHAPAnalysisResult] = []
        self.regime_transitions: List[Tuple[datetime, MarketRegime, MarketRegime]] = []

        # Live adaptation
        self.live_adaptation_enabled = True
        self.feature_adaptation_cache: Dict[str, Any] = {}

        self._lock = threading.RLock()

        # Start background analysis
        self._start_background_analysis()

        self.logger.info("SHAP Regime Analyzer initialized")

    def analyze_regime_specific_importance(self, data: pd.DataFrame, target_column: str,
                                         regime_column: Optional[str] = None) -> Dict[MarketRegime, SHAPAnalysisResult]:
        """Comprehensive SHAP analysis for all market regimes"""
        with self._lock:
            try:
                results = {}

                if not HAS_SHAP or not HAS_SKLEARN:
                    self.logger.error("Required libraries not available")
                    return results

                # Detect current regime if not provided
                if regime_column is None:
                    current_regime = self._detect_market_regime(data, target_column)
                    regime_data = pd.Series([current_regime] * len(data), index=data.index)
                else:
                    regime_data = data[regime_column]
                    current_regime = regime_data.iloc[-1] if not regime_data.empty else MarketRegime.SIDEWAYS

                self.current_regime = current_regime

                # Prepare features
                feature_columns = [col for col in data.columns if col != target_column and col != regime_column]
                X = data[feature_columns].fillna(0)
                y = data[target_column].fillna(0)

                # Analyze each regime
                for regime in MarketRegime:
                    regime_mask = regime_data == regime.value

                    if not isinstance(regime_mask, pd.Series):
                        # Handle case where regime_data is not a Series
                        continue

                    regime_X = X[regime_mask]
                    regime_y = y[regime_mask]

                    if len(regime_X) < self.config.min_samples_per_regime:
                        self.logger.debug(f"Insufficient samples for regime {regime.value}: {len(regime_X)}")
                        continue

                    # Perform SHAP analysis for this regime
                    analysis_result = self._analyze_regime_shap(regime, regime_X, regime_y)

                    if analysis_result:
                        results[regime] = analysis_result
                        self.regime_analysis_results[regime] = analysis_result

                        # Create regime-specific feature set
                        feature_set = self._create_regime_feature_set(regime, analysis_result)
                        self.regime_feature_sets[regime] = feature_set

                # Update global importance
                self._update_global_importance(results)

                # Live adaptation
                if self.live_adaptation_enabled:
                    self._adapt_features_to_regime(current_regime)

                self.logger.info(f"SHAP analysis completed for {len(results)} regimes")
                return results

            except Exception as e:
                self.logger.error(f"Regime-specific SHAP analysis failed: {e}")
                return {}

    def _analyze_regime_shap(self, regime: MarketRegime, X: pd.DataFrame, y: pd.Series) -> Optional[SHAPAnalysisResult]:
        """Perform SHAP analysis for specific regime"""
        try:
            # Limit features if too many
            if len(X.columns) > self.config.max_features_to_analyze:
                # Select top correlated features
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(self.config.max_features_to_analyze).index
                X = X[selected_features]

            # Train regime-specific model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X, y)

            # Calculate model performance
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            performance_metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': np.mean(np.abs(y - y_pred))
            }

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Sample data for SHAP if too large
            sample_size = min(self.config.shap_sample_size, len(X))
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[sample_indices]

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Calculate feature importance
            feature_importance = {}
            for i, feature in enumerate(X.columns):
                importance = np.abs(shap_values[:, i]).mean()
                feature_importance[feature] = importance

            # Calculate confidence score based on model performance and sample size
            confidence_score = min(1.0, r2 * (sample_size / 100))

            # Store explainer and model
            self.regime_explainers[regime] = explainer
            self.regime_models[regime] = model

            result = SHAPAnalysisResult(
                regime=regime,
                feature_importance=feature_importance,
                shap_values=shap_values,
                base_value=explainer.expected_value,
                model_performance=performance_metrics,
                analysis_timestamp=datetime.now(),
                sample_count=len(X),
                confidence_score=confidence_score
            )

            return result

        except Exception as e:
            self.logger.error(f"SHAP analysis failed for regime {regime.value}: {e}")
            return None

    def _detect_market_regime(self, data: pd.DataFrame, target_column: str) -> MarketRegime:
        """Detect current market regime based on price and volatility patterns"""
        try:
            if len(data) < self.config.regime_lookback_periods:
                return MarketRegime.SIDEWAYS

            # Use recent data for regime detection
            recent_data = data.tail(self.config.regime_lookback_periods)

            if target_column not in recent_data.columns:
                return MarketRegime.SIDEWAYS

            prices = recent_data[target_column].fillna(method='ffill')

            if len(prices) < 10:
                return MarketRegime.SIDEWAYS

            # Calculate regime indicators
            returns = prices.pct_change().fillna(0)
            volatility = returns.std()

            # Trend calculation
            x = np.arange(len(prices))
            trend_coef = np.polyfit(x, prices, 1)[0]
            normalized_trend = trend_coef / prices.mean() if prices.mean() != 0 else 0

            # Regime classification logic
            if volatility > self.config.volatility_threshold_high:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < self.config.volatility_threshold_low:
                return MarketRegime.LOW_VOLATILITY
            elif normalized_trend > self.config.trend_threshold:
                return MarketRegime.TRENDING_UP
            elif normalized_trend < -self.config.trend_threshold:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return MarketRegime.SIDEWAYS

    def _create_regime_feature_set(self, regime: MarketRegime, analysis_result: SHAPAnalysisResult) -> RegimeFeatureSet:
        """Create optimized feature set for specific regime"""
        try:
            # Sort features by importance
            sorted_features = sorted(
                analysis_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Select top features above threshold
            selected_features = []
            feature_weights = {}

            for feature, importance in sorted_features:
                if importance > self.config.feature_importance_threshold:
                    selected_features.append(feature)
                    feature_weights[feature] = importance

                    # Limit number of features
                    if len(selected_features) >= int(self.config.max_features_to_analyze * self.config.adaptive_feature_ratio):
                        break

            # Calculate stability score based on feature consistency
            stability_score = self._calculate_regime_stability_score(regime, selected_features)

            regime_feature_set = RegimeFeatureSet(
                regime=regime,
                selected_features=selected_features,
                feature_weights=feature_weights,
                performance_metrics=analysis_result.model_performance.copy(),
                last_updated=datetime.now(),
                stability_score=stability_score
            )

            return regime_feature_set

        except Exception as e:
            self.logger.error(f"Regime feature set creation failed: {e}")
            return RegimeFeatureSet(
                regime=regime,
                selected_features=[],
                feature_weights={},
                performance_metrics={},
                last_updated=datetime.now(),
                stability_score=0.0
            )

    def _calculate_regime_stability_score(self, regime: MarketRegime, current_features: List[str]) -> float:
        """Calculate stability score for regime features"""
        try:
            # Get historical feature sets for this regime
            historical_features = []

            for result in self.analysis_history[-10:]:  # Last 10 analyses
                if result.regime == regime:
                    top_features = sorted(
                        result.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:len(current_features)]

                    historical_features.append([f[0] for f in top_features])

            if not historical_features:
                return 0.5  # Neutral score for new regimes

            # Calculate overlap consistency
            overlaps = []
            for historical_set in historical_features:
                overlap = len(set(current_features) & set(historical_set)) / max(len(current_features), 1)
                overlaps.append(overlap)

            stability_score = np.mean(overlaps) if overlaps else 0.5
            return min(1.0, max(0.0, stability_score))

        except Exception:
            return 0.5

    def _update_global_importance(self, regime_results: Dict[MarketRegime, SHAPAnalysisResult]):
        """Update global feature importance across all regimes"""
        try:
            global_importance = {}
            total_samples = 0

            # Weighted average by sample count
            for regime, result in regime_results.items():
                weight = result.sample_count * result.confidence_score
                total_samples += weight

                for feature, importance in result.feature_importance.items():
                    if feature not in global_importance:
                        global_importance[feature] = 0.0

                    global_importance[feature] += importance * weight

            # Normalize by total samples
            if total_samples > 0:
                for feature in global_importance:
                    global_importance[feature] /= total_samples

            self.global_feature_importance = global_importance

        except Exception as e:
            self.logger.error(f"Global importance update failed: {e}")

    def _adapt_features_to_regime(self, current_regime: MarketRegime):
        """Adapt feature selection to current market regime"""
        try:
            if current_regime not in self.regime_feature_sets:
                self.logger.debug(f"No feature set available for regime {current_regime.value}")
                return

            feature_set = self.regime_feature_sets[current_regime]

            # Check if regime change occurred
            if hasattr(self, '_last_adapted_regime') and self._last_adapted_regime != current_regime:
                self.logger.info(f"Regime change detected: {self._last_adapted_regime} -> {current_regime}")

                # Record regime transition
                self.regime_transitions.append((
                    datetime.now(),
                    self._last_adapted_regime,
                    current_regime
                ))

            # Update adaptation cache
            self.feature_adaptation_cache['current_regime'] = current_regime.value
            self.feature_adaptation_cache['selected_features'] = feature_set.selected_features
            self.feature_adaptation_cache['feature_weights'] = feature_set.feature_weights
            self.feature_adaptation_cache['last_adaptation'] = datetime.now()

            self._last_adapted_regime = current_regime

        except Exception as e:
            self.logger.error(f"Feature adaptation failed: {e}")

    def get_regime_optimized_features(self, regime: Optional[MarketRegime] = None) -> List[str]:
        """Get features optimized for specific regime"""
        try:
            target_regime = regime or self.current_regime

            if target_regime and target_regime in self.regime_feature_sets:
                return self.regime_feature_sets[target_regime].selected_features

            # Fallback to global top features
            sorted_global = sorted(
                self.global_feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return [feature for feature, _ in sorted_global[:50]]

        except Exception as e:
            self.logger.error(f"Feature retrieval failed: {e}")
            return []

    def explain_prediction(self, features: pd.DataFrame, target_value: float,
                          regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """Explain a specific prediction using SHAP values"""
        try:
            target_regime = regime or self.current_regime

            if target_regime not in self.regime_explainers:
                self.logger.warning(f"No explainer available for regime {target_regime}")
                return {}

            explainer = self.regime_explainers[target_regime]
            model = self.regime_models[target_regime]

            # Ensure features match model training features
            model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else features.columns
            common_features = list(set(features.columns) & set(model_features))

            if not common_features:
                return {}

            feature_subset = features[common_features].fillna(0)

            # Get SHAP values for this prediction
            shap_values = explainer.shap_values(feature_subset)

            # Get model prediction
            prediction = model.predict(feature_subset)[0]

            # Create explanation
            explanation = {
                'prediction': prediction,
                'actual_value': target_value,
                'prediction_error': abs(prediction - target_value),
                'base_value': explainer.expected_value,
                'regime': target_regime.value,
                'feature_contributions': {}
            }

            # Feature contributions
            for i, feature in enumerate(common_features):
                contribution = shap_values[0, i] if shap_values.ndim > 1 else shap_values[i]
                explanation['feature_contributions'][feature] = {
                    'shap_value': contribution,
                    'feature_value': feature_subset.iloc[0, i],
                    'contribution_magnitude': abs(contribution)
                }

            # Sort by contribution magnitude
            explanation['top_contributors'] = sorted(
                explanation['feature_contributions'].items(),
                key=lambda x: x[1]['contribution_magnitude'],
                reverse=True
            )[:10]

            return explanation

        except Exception as e:
            self.logger.error(f"Prediction explanation failed: {e}")
            return {}

    def get_regime_feature_dominance(self) -> Dict[str, Dict[str, float]]:
        """Get feature dominance across different regimes"""
        try:
            dominance_map = {}

            for regime, feature_set in self.regime_feature_sets.items():
                regime_name = regime.value
                dominance_map[regime_name] = {}

                # Normalize feature weights
                total_weight = sum(feature_set.feature_weights.values())

                if total_weight > 0:
                    for feature, weight in feature_set.feature_weights.items():
                        normalized_weight = weight / total_weight
                        dominance_map[regime_name][feature] = normalized_weight

            return dominance_map

        except Exception as e:
            self.logger.error(f"Feature dominance calculation failed: {e}")
            return {}

    def _start_background_analysis(self):
        """Start background analysis thread"""
        def analysis_loop():
            while True:
                try:
                    import time
                    time.sleep(self.config.analysis_frequency_minutes * 60)

                    # Cleanup old analysis results
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.analysis_history = [
                        result for result in self.analysis_history
                        if result.analysis_timestamp > cutoff_time
                    ]

                except Exception as e:
                    self.logger.error(f"Background analysis error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        analysis_thread.start()
        self.logger.info("Background SHAP analysis started")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        with self._lock:
            return {
                'current_regime': self.current_regime.value if self.current_regime else None,
                'analyzed_regimes': list(self.regime_analysis_results.keys()),
                'regime_feature_counts': {
                    regime.value: len(feature_set.selected_features)
                    for regime, feature_set in self.regime_feature_sets.items()
                },
                'global_top_features': dict(list(sorted(
                    self.global_feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))[:10]),
                'regime_transitions_today': len([
                    t for t in self.regime_transitions
                    if t[0] > datetime.now() - timedelta(days=1)
                ]),
                'analysis_history_count': len(self.analysis_history),
                'live_adaptation_enabled': self.live_adaptation_enabled,
                'last_adaptation': self.feature_adaptation_cache.get('last_adaptation'),
                'regime_stability_scores': {
                    regime.value: feature_set.stability_score
                    for regime, feature_set in self.regime_feature_sets.items()
                }
            }


# Singleton SHAP regime analyzer
_shap_regime_analyzer = None
_sra_lock = threading.Lock()

def get_shap_regime_analyzer(config: Optional[SHAPRegimeConfig] = None) -> SHAPRegimeAnalyzer:
    """Get the singleton SHAP regime analyzer"""
    global _shap_regime_analyzer

    with _sra_lock:
        if _shap_regime_analyzer is None:
            _shap_regime_analyzer = SHAPRegimeAnalyzer(config)
        return _shap_regime_analyzer

def analyze_features_for_regime(data: pd.DataFrame, target_column: str,
                               regime_column: Optional[str] = None) -> Dict[MarketRegime, SHAPAnalysisResult]:
    """Convenient function to analyze features for market regimes"""
    analyzer = get_shap_regime_analyzer()
    return analyzer.analyze_regime_specific_importance(data, target_column, regime_column)
