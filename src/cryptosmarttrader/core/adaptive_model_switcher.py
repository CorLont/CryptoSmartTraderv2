#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Adaptive Model Switcher
Dynamic model, feature set and threshold switching based on detected market regimes
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .market_regime_detector import MarketRegime, get_market_regime_detector, RegimeDetectionResult
from .automated_feature_engineering import get_automated_feature_engineer
from .deep_learning_engine import get_deep_learning_engine


class ModelType(Enum):
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class AdaptationStrategy(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    DYNAMIC = "dynamic"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    model_type: ModelType
    parameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeModelMapping:
    """Mapping between market regimes and optimal models"""

    regime: MarketRegime
    primary_model: ModelConfig
    backup_models: List[ModelConfig] = field(default_factory=list)
    feature_set: List[str] = field(default_factory=list)
    trading_thresholds: Dict[str, float] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    switch_count: int = 0
    last_performance_check: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptiveSwitcherConfig:
    """Configuration for adaptive model switcher"""

    # Model switching
    performance_evaluation_window: int = 50
    min_performance_improvement: float = 0.05
    model_switch_threshold: float = 0.1
    max_switches_per_day: int = 5

    # Regime adaptation
    regime_stability_periods: int = 10
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE
    enable_feature_adaptation: bool = True
    enable_threshold_adaptation: bool = True

    # Model training
    retrain_frequency_hours: int = 6
    min_training_samples: int = 100
    validation_split: float = 0.2

    # Performance tracking
    performance_metrics: List[str] = field(
        default_factory=lambda: ["mse", "rmse", "mae", "r2", "directional_accuracy"]
    )

    # Thresholds
    buy_threshold_range: Tuple[float, float] = (0.02, 0.10)
    sell_threshold_range: Tuple[float, float] = (-0.10, -0.02)
    volatility_adjustment_factor: float = 0.5

    # Model persistence
    save_models: bool = True
    model_cache_dir: str = "models/adaptive_switcher"


class AdaptiveModelSwitcher:
    """Main adaptive model switcher with regime-based optimization"""

    def __init__(self, config: Optional[AdaptiveSwitcherConfig] = None):
        self.config = config or AdaptiveSwitcherConfig()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveModelSwitcher")

        # Core components
        self.regime_detector = get_market_regime_detector()
        self.feature_engineer = get_automated_feature_engineer()
        self.deep_learning_engine = get_deep_learning_engine()

        # Model mappings
        self.regime_models: Dict[MarketRegime, RegimeModelMapping] = {}
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.current_model: Optional[Any] = None
        self.current_features: List[str] = []
        self.current_thresholds: Dict[str, float] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.model_switch_history: List[Dict[str, Any]] = []
        self.regime_performance: Dict[MarketRegime, List[float]] = {}

        # Adaptation state
        self.last_adaptation_time: Optional[datetime] = None
        self.daily_switch_count: int = 0
        self.last_switch_date: Optional[datetime] = None

        # Model cache
        self.model_cache_dir = Path(self.config.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Initialize default regime mappings
        self._initialize_regime_mappings()

        self.logger.info("Adaptive Model Switcher initialized")

    def _initialize_regime_mappings(self):
        """Initialize default model mappings for each regime"""
        try:
            # Default model configurations for each regime
            regime_defaults = {
                MarketRegime.BULL_MARKET: {
                    "model_type": ModelType.GRADIENT_BOOSTING,
                    "features": ["price_momentum", "volume_surge", "rsi", "macd"],
                    "thresholds": {"buy": 0.03, "sell": -0.05},
                },
                MarketRegime.BEAR_MARKET: {
                    "model_type": ModelType.RANDOM_FOREST,
                    "features": ["volatility", "support_levels", "rsi", "volume_profile"],
                    "thresholds": {"buy": 0.05, "sell": -0.03},
                },
                MarketRegime.SIDEWAYS: {
                    "model_type": ModelType.LINEAR,
                    "features": ["support_resistance", "oscillator_signals", "volume_patterns"],
                    "thresholds": {"buy": 0.02, "sell": -0.02},
                },
                MarketRegime.HIGH_VOLATILITY: {
                    "model_type": ModelType.NEURAL_NETWORK,
                    "features": ["volatility_indicators", "breakout_signals", "volume_spikes"],
                    "thresholds": {"buy": 0.08, "sell": -0.08},
                },
                MarketRegime.LOW_VOLATILITY: {
                    "model_type": ModelType.LINEAR,
                    "features": ["mean_reversion", "gentle_trends", "volume_consistency"],
                    "thresholds": {"buy": 0.01, "sell": -0.01},
                },
                MarketRegime.TRENDING_UP: {
                    "model_type": ModelType.LSTM,
                    "features": ["trend_strength", "momentum_indicators", "breakout_patterns"],
                    "thresholds": {"buy": 0.04, "sell": -0.06},
                },
                MarketRegime.TRENDING_DOWN: {
                    "model_type": ModelType.LSTM,
                    "features": ["downtrend_signals", "support_breaks", "volume_confirmation"],
                    "thresholds": {"buy": 0.06, "sell": -0.04},
                },
                MarketRegime.CONSOLIDATION: {
                    "model_type": ModelType.RANDOM_FOREST,
                    "features": ["range_trading", "oscillators", "volume_balance"],
                    "thresholds": {"buy": 0.015, "sell": -0.015},
                },
                MarketRegime.BREAKOUT: {
                    "model_type": ModelType.XGBOOST,
                    "features": ["breakout_strength", "volume_explosion", "momentum_acceleration"],
                    "thresholds": {"buy": 0.10, "sell": -0.10},
                },
                MarketRegime.CRASH: {
                    "model_type": ModelType.ENSEMBLE,
                    "features": ["crash_indicators", "panic_signals", "recovery_signals"],
                    "thresholds": {"buy": 0.15, "sell": -0.02},
                },
                MarketRegime.RECOVERY: {
                    "model_type": ModelType.GRADIENT_BOOSTING,
                    "features": ["recovery_signals", "sentiment_improvement", "volume_return"],
                    "thresholds": {"buy": 0.05, "sell": -0.08},
                },
            }

            # Create regime mappings
            for regime, defaults in regime_defaults.items():
                primary_model = ModelConfig(
                    model_type=defaults["model_type"],
                    parameters=self._get_default_parameters(defaults["model_type"]),
                    features=defaults["features"],
                    thresholds=defaults["thresholds"],
                )

                # Create backup models
                backup_models = []
                backup_types = [
                    ModelType.RANDOM_FOREST,
                    ModelType.LINEAR,
                    ModelType.GRADIENT_BOOSTING,
                ]
                for backup_type in backup_types:
                    if backup_type != defaults["model_type"]:
                        backup_model = ModelConfig(
                            model_type=backup_type,
                            parameters=self._get_default_parameters(backup_type),
                            features=defaults["features"],
                            thresholds=defaults["thresholds"],
                        )
                        backup_models.append(backup_model)

                self.regime_models[regime] = RegimeModelMapping(
                    regime=regime,
                    primary_model=primary_model,
                    backup_models=backup_models,
                    feature_set=defaults["features"],
                    trading_thresholds=defaults["thresholds"],
                )

            self.logger.info(f"Initialized {len(self.regime_models)} regime model mappings")

        except Exception as e:
            self.logger.error(f"Regime mapping initialization failed: {e}")

    def _get_default_parameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default parameters for model type"""
        parameter_sets = {
            ModelType.LINEAR: {"fit_intercept": True},
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            ModelType.GRADIENT_BOOSTING: {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_samples_split": 5,
                "random_state": 42,
            },
            ModelType.XGBOOST: {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            ModelType.NEURAL_NETWORK: {
                "hidden_layers": [64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "learning_rate": 0.001,
                "epochs": 100,
            },
            ModelType.LSTM: {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 20,
                "learning_rate": 0.001,
                "epochs": 100,
            },
            ModelType.GRU: {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "sequence_length": 20,
                "learning_rate": 0.001,
                "epochs": 100,
            },
            ModelType.TRANSFORMER: {
                "hidden_size": 64,
                "num_heads": 8,
                "num_layers": 4,
                "dropout": 0.2,
                "sequence_length": 20,
                "learning_rate": 0.001,
                "epochs": 100,
            },
        }

        return parameter_sets.get(model_type, {})

    def adapt_to_regime(self, data: pd.DataFrame, target_column: str = "close") -> Dict[str, Any]:
        """Adapt model and configuration to detected market regime"""
        with self._lock:
            try:
                self.logger.info("Starting regime-based adaptation")

                # Detect current regime
                regime_result = self.regime_detector.detect_regime(data, target_column)
                detected_regime = regime_result.regime

                # Check if regime has changed
                regime_changed = detected_regime != self.current_regime

                if regime_changed:
                    self.logger.info(
                        f"Regime change detected: {self.current_regime.value} -> {detected_regime.value}"
                    )
                    self.current_regime = detected_regime

                # Get regime-specific configuration
                if detected_regime in self.regime_models:
                    regime_mapping = self.regime_models[detected_regime]

                    # Adapt features
                    if self.config.enable_feature_adaptation:
                        adapted_features = self._adapt_features(data, regime_mapping, target_column)
                        self.current_features = adapted_features

                    # Adapt model
                    adapted_model = self._adapt_model(data, regime_mapping, target_column)
                    self.current_model = adapted_model

                    # Adapt thresholds
                    if self.config.enable_threshold_adaptation:
                        adapted_thresholds = self._adapt_thresholds(
                            data, regime_mapping, regime_result
                        )
                        self.current_thresholds = adapted_thresholds

                    # Record adaptation
                    adaptation_record = {
                        "timestamp": datetime.now(),
                        "regime": detected_regime.value,
                        "regime_confidence": regime_result.confidence,
                        "model_type": regime_mapping.primary_model.model_type.value,
                        "feature_count": len(self.current_features),
                        "thresholds": self.current_thresholds.copy(),
                        "regime_changed": regime_changed,
                    }

                    self.performance_history.append(adaptation_record)

                    # Update switch count
                    if regime_changed:
                        self._update_switch_count()

                    return {
                        "success": True,
                        "regime": detected_regime.value,
                        "model_type": regime_mapping.primary_model.model_type.value,
                        "features": self.current_features,
                        "thresholds": self.current_thresholds,
                        "confidence": regime_result.confidence,
                        "regime_changed": regime_changed,
                    }

                else:
                    self.logger.warning(
                        f"No model mapping found for regime: {detected_regime.value}"
                    )
                    return {
                        "success": False,
                        "error": f"No mapping for regime {detected_regime.value}",
                        "regime": detected_regime.value,
                    }

            except Exception as e:
                self.logger.error(f"Regime adaptation failed: {e}")
                return {"success": False, "error": str(e), "regime": self.current_regime.value}

    def _adapt_features(
        self, data: pd.DataFrame, regime_mapping: RegimeModelMapping, target_column: str
    ) -> List[str]:
        """Adapt feature set for the current regime"""
        try:
            # Get regime-optimized features from feature engineer
            regime_features = regime_mapping.feature_set.copy()

            # Get additional features from automated feature engineering
            engineered_data = self.feature_engineer.transform(
                data, target_column, regime_mapping.regime.value
            )
            available_features = [col for col in engineered_data.columns if col != target_column]

            # Combine regime-specific and engineered features
            adapted_features = list(set(regime_features + available_features))

            # Limit features based on regime characteristics
            max_features = self._get_max_features_for_regime(regime_mapping.regime)
            if len(adapted_features) > max_features:
                # Prioritize regime-specific features
                priority_features = regime_features[: max_features // 2]
                other_features = [f for f in adapted_features if f not in priority_features][
                    : max_features - len(priority_features)
                ]
                adapted_features = priority_features + other_features

            self.logger.info(
                f"Adapted features for {regime_mapping.regime.value}: {len(adapted_features)} features"
            )
            return adapted_features

        except Exception as e:
            self.logger.error(f"Feature adaptation failed: {e}")
            return regime_mapping.feature_set

    def _adapt_model(
        self, data: pd.DataFrame, regime_mapping: RegimeModelMapping, target_column: str
    ) -> Any:
        """Adapt model for the current regime"""
        try:
            # Check if current model needs switching
            switch_needed = self._should_switch_model(regime_mapping)

            if switch_needed or self.current_model is None:
                # Select best model for regime
                best_model_config = self._select_best_model(data, regime_mapping, target_column)

                # Create and train model
                model = self._create_model(best_model_config)
                trained_model = self._train_model(model, data, best_model_config, target_column)

                # Update regime mapping performance
                regime_mapping.primary_model = best_model_config
                regime_mapping.switch_count += 1

                self.logger.info(
                    f"Switched to {best_model_config.model_type.value} for {regime_mapping.regime.value}"
                )
                return trained_model

            else:
                # Keep current model but retrain if needed
                if self._should_retrain_model():
                    model = self._create_model(regime_mapping.primary_model)
                    trained_model = self._train_model(
                        model, data, regime_mapping.primary_model, target_column
                    )
                    return trained_model

                return self.current_model

        except Exception as e:
            self.logger.error(f"Model adaptation failed: {e}")
            return self.current_model

    def _adapt_thresholds(
        self,
        data: pd.DataFrame,
        regime_mapping: RegimeModelMapping,
        regime_result: RegimeDetectionResult,
    ) -> Dict[str, float]:
        """Adapt trading thresholds for the current regime"""
        try:
            base_thresholds = regime_mapping.trading_thresholds.copy()

            # Adjust thresholds based on regime confidence
            confidence_factor = regime_result.confidence

            # Adjust thresholds based on market volatility
            if "volatility" in regime_result.supporting_evidence:
                volatility = regime_result.supporting_evidence["volatility"]
                volatility_adjustment = volatility * self.config.volatility_adjustment_factor
            else:
                # Calculate volatility from data
                if "close" in data.columns:
                    returns = data["close"].pct_change().fillna(0)
                    volatility_adjustment = returns.std() * self.config.volatility_adjustment_factor
                else:
                    volatility_adjustment = 0

            # Apply adaptations
            adapted_thresholds = {}

            for threshold_name, base_value in base_thresholds.items():
                # Adjust for confidence (higher confidence = more aggressive)
                confidence_adjusted = base_value * (0.8 + 0.4 * confidence_factor)

                # Adjust for volatility
                if threshold_name == "buy":
                    # Higher volatility = higher buy threshold
                    adapted_value = confidence_adjusted + volatility_adjustment
                    adapted_value = max(
                        self.config.buy_threshold_range[0],
                        min(self.config.buy_threshold_range[1], adapted_value),
                    )

                elif threshold_name == "sell":
                    # Higher volatility = lower (more negative) sell threshold
                    adapted_value = confidence_adjusted - volatility_adjustment
                    adapted_value = max(
                        self.config.sell_threshold_range[0],
                        min(self.config.sell_threshold_range[1], adapted_value),
                    )

                else:
                    adapted_value = confidence_adjusted

                adapted_thresholds[threshold_name] = adapted_value

            self.logger.info(
                f"Adapted thresholds for {regime_mapping.regime.value}: {adapted_thresholds}"
            )
            return adapted_thresholds

        except Exception as e:
            self.logger.error(f"Threshold adaptation failed: {e}")
            return regime_mapping.trading_thresholds

    def _should_switch_model(self, regime_mapping: RegimeModelMapping) -> bool:
        """Determine if model should be switched"""
        try:
            # Always switch on regime change
            if self.current_model is None:
                return True

            # Check performance degradation
            if len(regime_mapping.performance_history) >= self.config.performance_evaluation_window:
                recent_performance = np.mean(regime_mapping.performance_history[-10:])
                historical_performance = np.mean(regime_mapping.performance_history[:-10])

                performance_decline = historical_performance - recent_performance

                if performance_decline > self.config.model_switch_threshold:
                    return True

            # Check switch frequency limits
            if self.daily_switch_count >= self.config.max_switches_per_day:
                return False

            return False

        except Exception:
            return False

    def _should_retrain_model(self) -> bool:
        """Determine if current model should be retrained"""
        try:
            if self.last_adaptation_time is None:
                return True

            time_since_last = datetime.now() - self.last_adaptation_time
            retrain_interval = timedelta(hours=self.config.retrain_frequency_hours)

            return time_since_last >= retrain_interval

        except Exception:
            return True

    def _select_best_model(
        self, data: pd.DataFrame, regime_mapping: RegimeModelMapping, target_column: str
    ) -> ModelConfig:
        """Select best model configuration for the regime"""
        try:
            # Prepare candidates (primary + backups)
            model_candidates = [regime_mapping.primary_model] + regime_mapping.backup_models

            if len(data) < self.config.min_training_samples:
                # Not enough data for full evaluation, use primary
                return regime_mapping.primary_model

            best_model = regime_mapping.primary_model
            best_score = float("-inf")

            # Evaluate each candidate
            for candidate in model_candidates:
                try:
                    score = self._evaluate_model_candidate(data, candidate, target_column)

                    if score > best_score:
                        best_score = score
                        best_model = candidate

                except Exception as e:
                    self.logger.warning(f"Model candidate evaluation failed: {e}")
                    continue

            # Update performance metrics
            best_model.performance_metrics["selection_score"] = best_score
            best_model.last_updated = datetime.now()

            return best_model

        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return regime_mapping.primary_model

    def _evaluate_model_candidate(
        self, data: pd.DataFrame, model_config: ModelConfig, target_column: str
    ) -> float:
        """Evaluate a model candidate using cross-validation"""
        try:
            # Create and train model
            model = self._create_model(model_config)

            # Prepare features
            available_features = [
                col for col in data.columns if col != target_column and col in model_config.features
            ]

            if not available_features:
                return 0.0

            X = data[available_features].fillna(0)
            y = data[target_column].fillna(0)

            # Simple train-test split
            split_idx = int(len(X) * (1 - self.config.validation_split))

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            if len(X_train) < 10 or len(X_test) < 5:
                return 0.0

            # Train model
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)

                # Calculate R² score
                r2 = r2_score(y_test, y_pred)

                # Calculate directional accuracy
                actual_direction = (y_test.diff() > 0).astype(int)
                pred_direction = (pd.Series(y_pred, index=y_test.index).diff() > 0).astype(int)
                directional_accuracy = (actual_direction == pred_direction).mean()

                # Combined score
                score = 0.7 * r2 + 0.3 * directional_accuracy

                return max(0.0, score)

            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 0.0

    def _create_model(self, model_config: ModelConfig) -> Any:
        """Create model instance based on configuration"""
        try:
            model_type = model_config.model_type
            params = model_config.parameters

            if not HAS_SKLEARN and model_type in [
                ModelType.LINEAR,
                ModelType.RANDOM_FOREST,
                ModelType.GRADIENT_BOOSTING,
            ]:
                raise ImportError("Scikit-learn required for traditional ML models")

            if model_type == ModelType.LINEAR:
                return LinearRegression(**params)

            elif model_type == ModelType.RANDOM_FOREST:
                return RandomForestRegressor(**params)

            elif model_type == ModelType.GRADIENT_BOOSTING:
                return GradientBoostingRegressor(**params)

            elif model_type == ModelType.XGBOOST:
                if not HAS_XGBOOST:
                    # Fallback to gradient boosting
                    return GradientBoostingRegressor(n_estimators=100, random_state=42)
                return xgb.XGBRegressor(**params)

            elif model_type in [
                ModelType.NEURAL_NETWORK,
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.TRANSFORMER,
            ]:
                # Use deep learning engine
                return self.deep_learning_engine.create_model(model_type.value, params)

            elif model_type == ModelType.ENSEMBLE:
                # Create ensemble of multiple models
                return self._create_ensemble_model(params)

            else:
                # Default fallback
                return LinearRegression()

        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return LinearRegression()  # Safe fallback

    def _create_ensemble_model(self, params: Dict[str, Any]) -> Any:
        """Create ensemble model combining multiple base models"""
        try:
            from sklearn.ensemble import VotingRegressor

            # Base models for ensemble
            base_models = [
                ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
                ("gb", GradientBoostingRegressor(n_estimators=50, random_state=42)),
                ("lr", LinearRegression()),
            ]

            if HAS_XGBOOST:
                base_models.append(("xgb", xgb.XGBRegressor(n_estimators=50, random_state=42)))

            return VotingRegressor(estimators=base_models)

        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def _train_model(
        self, model: Any, data: pd.DataFrame, model_config: ModelConfig, target_column: str
    ) -> Any:
        """Train model with data"""
        try:
            # Prepare features
            available_features = [
                col for col in data.columns if col != target_column and col in model_config.features
            ]

            if not available_features:
                self.logger.warning("No features available for training")
                return model

            X = data[available_features].fillna(0)
            y = data[target_column].fillna(0)

            # For deep learning models, use the deep learning engine
            if model_config.model_type in [
                ModelType.NEURAL_NETWORK,
                ModelType.LSTM,
                ModelType.GRU,
                ModelType.TRANSFORMER,
            ]:
                trained_model = self.deep_learning_engine.train_model(
                    model, X, y, model_config.parameters
                )

            else:
                # Traditional ML models
                if hasattr(model, "fit"):
                    trained_model = model.fit(X, y)
                else:
                    trained_model = model

            # Update model config with training info
            model_config.performance_metrics["training_samples"] = len(X)
            model_config.performance_metrics["feature_count"] = len(available_features)
            model_config.last_updated = datetime.now()

            self.last_adaptation_time = datetime.now()

            return trained_model

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return model

    def _get_max_features_for_regime(self, regime: MarketRegime) -> int:
        """Get maximum number of features for a regime"""
        regime_limits = {
            MarketRegime.HIGH_VOLATILITY: 15,  # Fewer features for volatile markets
            MarketRegime.LOW_VOLATILITY: 30,  # More features for stable markets
            MarketRegime.CRASH: 10,  # Very few features during crashes
            MarketRegime.BREAKOUT: 20,  # Moderate features for breakouts
            MarketRegime.BULL_MARKET: 25,  # Standard feature set
            MarketRegime.BEAR_MARKET: 25,  # Standard feature set
            MarketRegime.SIDEWAYS: 30,  # More features for range trading
            MarketRegime.TRENDING_UP: 20,  # Focus on trend features
            MarketRegime.TRENDING_DOWN: 20,  # Focus on trend features
            MarketRegime.CONSOLIDATION: 25,  # Standard feature set
            MarketRegime.RECOVERY: 20,  # Focus on recovery signals
        }

        return regime_limits.get(regime, 25)  # Default to 25

    def _update_switch_count(self):
        """Update daily switch count"""
        try:
            current_date = datetime.now().date()

            if self.last_switch_date != current_date:
                self.daily_switch_count = 1
                self.last_switch_date = current_date
            else:
                self.daily_switch_count += 1

        except Exception as e:
            self.logger.error(f"Switch count update failed: {e}")

    def evaluate_current_performance(
        self, data: pd.DataFrame, target_column: str
    ) -> Dict[str, float]:
        """Evaluate current model performance"""
        try:
            if self.current_model is None or not self.current_features:
                return {}

            # Prepare data
            available_features = [f for f in self.current_features if f in data.columns]

            if not available_features:
                return {}

            X = data[available_features].fillna(0)
            y = data[target_column].fillna(0)

            if len(X) < 10:
                return {}

            # Make predictions
            if hasattr(self.current_model, "predict"):
                y_pred = self.current_model.predict(X)

                # Calculate metrics
                metrics = {
                    "mse": mean_squared_error(y, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                    "mae": mean_absolute_error(y, y_pred),
                    "r2": r2_score(y, y_pred),
                }

                # Directional accuracy
                actual_direction = (y.diff() > 0).astype(int)
                pred_direction = (pd.Series(y_pred, index=y.index).diff() > 0).astype(int)
                metrics["directional_accuracy"] = (actual_direction == pred_direction).mean()

                # Update regime performance
                if self.current_regime not in self.regime_performance:
                    self.regime_performance[self.current_regime] = []

                self.regime_performance[self.current_regime].append(metrics["r2"])

                # Keep only recent performance
                if len(self.regime_performance[self.current_regime]) > 100:
                    self.regime_performance[self.current_regime] = self.regime_performance[
                        self.current_regime
                    ][-100:]

                return metrics

            else:
                return {}

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return {}

    def get_current_signals(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Get trading signals based on current model and thresholds"""
        try:
            if (
                self.current_model is None
                or not self.current_features
                or not self.current_thresholds
            ):
                return {"signal": "HOLD", "confidence": 0.0, "reason": "No model or configuration"}

            # Prepare features
            available_features = [f for f in self.current_features if f in data.columns]

            if not available_features:
                return {"signal": "HOLD", "confidence": 0.0, "reason": "No features available"}

            # Get latest data point
            latest_data = data[available_features].iloc[-1:].fillna(0)

            # Make prediction
            if hasattr(self.current_model, "predict"):
                prediction = self.current_model.predict(latest_data)[0]

                # Current price
                current_price = data[target_column].iloc[-1]
                predicted_return = (prediction - current_price) / current_price

                # Generate signal based on thresholds
                buy_threshold = self.current_thresholds.get("buy", 0.02)
                sell_threshold = self.current_thresholds.get("sell", -0.02)

                if predicted_return >= buy_threshold:
                    signal = "BUY"
                    confidence = min(1.0, predicted_return / buy_threshold)
                elif predicted_return <= sell_threshold:
                    signal = "SELL"
                    confidence = min(1.0, abs(predicted_return) / abs(sell_threshold))
                else:
                    signal = "HOLD"
                    confidence = 1.0 - (
                        abs(predicted_return) / max(buy_threshold, abs(sell_threshold))
                    )

                return {
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_return": predicted_return,
                    "current_price": current_price,
                    "predicted_price": prediction,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "regime": self.current_regime.value,
                    "model_type": self.regime_models[
                        self.current_regime
                    ].primary_model.model_type.value
                    if self.current_regime in self.regime_models
                    else "unknown",
                }

            else:
                return {"signal": "HOLD", "confidence": 0.0, "reason": "Model cannot predict"}

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "reason": f"Error: {e}"}

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary"""
        with self._lock:
            return {
                "current_regime": self.current_regime.value,
                "current_model_type": self.regime_models[
                    self.current_regime
                ].primary_model.model_type.value
                if self.current_regime in self.regime_models
                else "unknown",
                "current_features_count": len(self.current_features),
                "current_thresholds": self.current_thresholds.copy(),
                "daily_switch_count": self.daily_switch_count,
                "total_adaptations": len(self.performance_history),
                "regime_mappings": {
                    regime.value: {
                        "model_type": mapping.primary_model.model_type.value,
                        "feature_count": len(mapping.feature_set),
                        "switch_count": mapping.switch_count,
                        "avg_performance": np.mean(self.regime_performance.get(regime, [0])),
                    }
                    for regime, mapping in self.regime_models.items()
                },
                "recent_performance": dict(list(self.performance_history[-5:]))
                if self.performance_history
                else {},
                "last_adaptation_time": self.last_adaptation_time.isoformat()
                if self.last_adaptation_time
                else None,
                "adaptation_strategy": self.config.adaptation_strategy.value,
            }


# Singleton adaptive model switcher
_adaptive_model_switcher = None
_ams_lock = threading.Lock()


def get_adaptive_model_switcher(
    config: Optional[AdaptiveSwitcherConfig] = None,
) -> AdaptiveModelSwitcher:
    """Get the singleton adaptive model switcher"""
    global _adaptive_model_switcher

    with _ams_lock:
        if _adaptive_model_switcher is None:
            _adaptive_model_switcher = AdaptiveModelSwitcher(config)
        return _adaptive_model_switcher


def adapt_to_current_regime(data: pd.DataFrame, target_column: str = "close") -> Dict[str, Any]:
    """Convenient function to adapt to current market regime"""
    switcher = get_adaptive_model_switcher()
    return switcher.adapt_to_regime(data, target_column)
