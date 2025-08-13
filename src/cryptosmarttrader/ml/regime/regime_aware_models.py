#!/usr/bin/env python3
"""
Regime-Aware Models
Model routing and adaptation based on market regime detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from ml.regime.market_regime_detector import MarketRegime, RegimeState, MarketRegimeDetector

@dataclass
class RegimeModelConfig:
    """Configuration for regime-specific model"""
    regime: MarketRegime
    model_type: str
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    performance_threshold: float = 0.7

@dataclass
class RegimeModelPerformance:
    """Performance metrics for regime-specific model"""
    regime: MarketRegime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_predictions: int
    regime_coverage: float  # % of regime periods this model was active

class RegimeAwareModel(ABC):
    """Abstract base class for regime-aware models"""

    def __init__(self, regime: MarketRegime):
        self.regime = regime
        self.is_trained = False
        self.performance_metrics = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, regime_mask: np.ndarray):
        """Train model on regime-specific data"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pass

class BullMarketModel(RegimeAwareModel):
    """Specialized model for bull market conditions"""

    def __init__(self):
        super().__init__(MarketRegime.BULL_MARKET)
        self.model = None

        # Bull market specific features
        self.feature_weights = {
            'momentum_features': 1.2,      # Higher weight on momentum
            'trend_features': 1.1,         # Trend following important
            'volume_features': 1.0,        # Standard volume analysis
            'volatility_features': 0.8     # Less focus on volatility
        }

    def fit(self, X: np.ndarray, y: np.ndarray, regime_mask: np.ndarray):
        """Train bull market model"""

        # Filter to bull market periods
        X_bull = X[regime_mask]
        y_bull = y[regime_mask]

        if len(X_bull) < 50:
            raise ValueError(f"Insufficient bull market data: {len(X_bull)} samples")

        # Use ensemble for bull markets (momentum + trend)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        # Momentum-focused model
        self.momentum_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

        # Trend-following model
        self.trend_model = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000
        )

        # Train both models
        self.momentum_model.fit(X_bull, y_bull)
        self.trend_model.fit(X_bull, y_bull)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using bull market ensemble"""

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensemble prediction (weighted)
        momentum_pred = self.momentum_model.predict_proba(X)[:, 1]
        trend_pred = self.trend_model.predict_proba(X)[:, 1]

        # Weight combination for bull markets
        ensemble_pred = (momentum_pred * 0.6 + trend_pred * 0.4)

        return (ensemble_pred > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""

        momentum_proba = self.momentum_model.predict_proba(X)
        trend_proba = self.trend_model.predict_proba(X)

        # Weighted ensemble
        ensemble_proba = momentum_proba * 0.6 + trend_proba * 0.4

        return ensemble_proba

class BearMarketModel(RegimeAwareModel):
    """Specialized model for bear market conditions"""

    def __init__(self):
        super().__init__(MarketRegime.BEAR_MARKET)

        # Bear market specific features
        self.feature_weights = {
            'volatility_features': 1.3,    # Higher focus on volatility
            'risk_features': 1.2,          # Risk indicators important
            'momentum_features': 0.9,      # Less reliable momentum
            'sentiment_features': 1.1      # Sentiment more important
        }

    def fit(self, X: np.ndarray, y: np.ndarray, regime_mask: np.ndarray):
        """Train bear market model"""

        X_bear = X[regime_mask]
        y_bear = y[regime_mask]

        if len(X_bear) < 50:
            raise ValueError(f"Insufficient bear market data: {len(X_bear)} samples")

        # Use conservative models for bear markets
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        # Risk-focused model
        self.risk_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,       # Shallower trees for stability
            min_samples_leaf=5,
            random_state=42
        )

        # Volatility-aware model
        self.volatility_model = SVC(
            C=0.5,             # Lower C for regularization
            kernel='rbf',
            probability=True,
            random_state=42
        )

        self.risk_model.fit(X_bear, y_bear)
        self.volatility_model.fit(X_bear, y_bear)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using bear market ensemble"""

        risk_pred = self.risk_model.predict_proba(X)[:, 1]
        vol_pred = self.volatility_model.predict_proba(X)[:, 1]

        # Conservative combination for bear markets
        ensemble_pred = (risk_pred * 0.7 + vol_pred * 0.3)

        # Higher threshold for bear markets (more conservative)
        return (ensemble_pred > 0.6).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""

        risk_proba = self.risk_model.predict_proba(X)
        vol_proba = self.volatility_model.predict_proba(X)

        return risk_proba * 0.7 + vol_proba * 0.3

class HighVolatilityModel(RegimeAwareModel):
    """Specialized model for high volatility periods"""

    def __init__(self):
        super().__init__(MarketRegime.HIGH_VOLATILITY)

        self.feature_weights = {
            'volatility_features': 1.5,    # Maximum focus on volatility
            'range_features': 1.3,         # High-low ranges important
            'momentum_features': 0.7,      # Momentum less reliable
            'mean_reversion_features': 1.2 # Mean reversion opportunities
        }

    def fit(self, X: np.ndarray, y: np.ndarray, regime_mask: np.ndarray):
        """Train high volatility model"""

        X_hvol = X[regime_mask]
        y_hvol = y[regime_mask]

        if len(X_hvol) < 30:
            raise ValueError(f"Insufficient high volatility data: {len(X_hvol)} samples")

        # Use models that handle volatility well
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.naive_bayes import GaussianNB

        # Volatility-robust model
        self.volatility_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            bootstrap=True      # Bootstrap for robustness
        )

        # Mean reversion model
        self.reversion_model = GaussianNB()

        self.volatility_model.fit(X_hvol, y_hvol)
        self.reversion_model.fit(X_hvol, y_hvol)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using high volatility ensemble"""

        vol_pred = self.volatility_model.predict_proba(X)[:, 1]
        rev_pred = self.reversion_model.predict_proba(X)[:, 1]

        # Balanced combination for high volatility
        ensemble_pred = (vol_pred * 0.6 + rev_pred * 0.4)

        return (ensemble_pred > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""

        vol_proba = self.volatility_model.predict_proba(X)
        rev_proba = self.reversion_model.predict_proba(X)

        return vol_proba * 0.6 + rev_proba * 0.4

class ConsolidationModel(RegimeAwareModel):
    """Specialized model for consolidation periods"""

    def __init__(self):
        super().__init__(MarketRegime.CONSOLIDATION)

        self.feature_weights = {
            'range_bound_features': 1.4,   # Range-bound behavior
            'mean_reversion_features': 1.3, # Mean reversion strong
            'breakout_features': 1.2,      # Breakout detection
            'trend_features': 0.8          # Weak trends
        }

    def fit(self, X: np.ndarray, y: np.ndarray, regime_mask: np.ndarray):
        """Train consolidation model"""

        X_consol = X[regime_mask]
        y_consol = y[regime_mask]

        if len(X_consol) < 50:
            raise ValueError(f"Insufficient consolidation data: {len(X_consol)} samples")

        # Use models good for sideways markets
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import RidgeClassifier

        # Pattern-based model
        self.pattern_model = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance'
        )

        # Linear model for ranges
        self.range_model = RidgeClassifier(
            alpha=1.0,
            random_state=42
        )

        self.pattern_model.fit(X_consol, y_consol)
        self.range_model.fit(X_consol, y_consol)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using consolidation ensemble"""

        pattern_pred = self.pattern_model.predict_proba(X)[:, 1]

        # Ridge doesn't have predict_proba, use decision function
        range_scores = self.range_model.decision_function(X)
        range_pred = 1 / (1 + np.exp(-range_scores))  # Sigmoid transformation

        ensemble_pred = (pattern_pred * 0.6 + range_pred * 0.4)

        return (ensemble_pred > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""

        pattern_proba = self.pattern_model.predict_proba(X)

        range_scores = self.range_model.decision_function(X)
        range_proba = np.column_stack([
            1 - 1 / (1 + np.exp(-range_scores)),
            1 / (1 + np.exp(-range_scores))
        ])

        return pattern_proba * 0.6 + range_proba * 0.4

class RegimeRouter:
    """Routes predictions to appropriate regime-specific models"""

    def __init__(self, regime_detector: MarketRegimeDetector):
        self.regime_detector = regime_detector
        self.regime_models = {}
        self.model_performance = {}
        self.routing_history = []

        # Initialize regime-specific models
        self._initialize_regime_models()

        self.logger = logging.getLogger(__name__)

    def _initialize_regime_models(self):
        """Initialize all regime-specific models"""

        self.regime_models = {
            MarketRegime.BULL_MARKET: BullMarketModel(),
            MarketRegime.BEAR_MARKET: BearMarketModel(),
            MarketRegime.HIGH_VOLATILITY: HighVolatilityModel(),
            MarketRegime.CONSOLIDATION: ConsolidationModel(),
            # Default model for other regimes
            MarketRegime.LOW_VOLATILITY: ConsolidationModel(),  # Use consolidation model
            MarketRegime.TREND_REVERSAL: HighVolatilityModel(), # Use high vol model
        }

    def train_regime_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: List[MarketRegime]
    ):
        """Train all regime-specific models"""

        if len(regime_labels) != len(X):
            raise ValueError("Regime labels must match number of samples")

        # Convert regime labels to arrays for each regime
        regime_masks = {}
        for regime in self.regime_models.keys():
            regime_masks[regime] = np.array([r == regime for r in regime_labels])

        # Train each model on its regime data
        for regime, model in self.regime_models.items():
            mask = regime_masks[regime]

            if np.sum(mask) > 30:  # Minimum samples required
                try:
                    model.fit(X, y, mask)
                    self.logger.info(f"Trained {regime.value} model on {np.sum(mask)} samples")
                except Exception as e:
                    self.logger.error(f"Failed to train {regime.value} model: {e}")
            else:
                self.logger.warning(f"Insufficient data for {regime.value} model: {np.sum(mask)} samples")

    def predict_with_routing(
        self,
        X: np.ndarray,
        market_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[MarketRegime]]:
        """Make predictions using regime-aware routing"""

        # Detect current regime
        current_regime_state = self.regime_detector.detect_current_regime(market_data)
        current_regime = current_regime_state.regime

        # Route to appropriate model
        if current_regime in self.regime_models:
            model = self.regime_models[current_regime]

            if model.is_trained:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                # Record routing decision
                self.routing_history.append({
                    'timestamp': current_regime_state.timestamp,
                    'regime': current_regime,
                    'confidence': current_regime_state.confidence,
                    'model_used': type(model).__name__,
                    'n_predictions': len(predictions)
                })

                regimes_used = [current_regime] * len(predictions)

                return predictions, probabilities, regimes_used

            else:
                self.logger.warning(f"Model for {current_regime.value} not trained, using fallback")

        # Fallback to ensemble of available models
        return self._fallback_prediction(X, current_regime)

    def _fallback_prediction(
        self,
        X: np.ndarray,
        detected_regime: MarketRegime
    ) -> Tuple[np.ndarray, np.ndarray, List[MarketRegime]]:
        """Fallback prediction using ensemble of trained models"""

        # Find trained models
        trained_models = [(regime, model) for regime, model in self.regime_models.items()
                         if model.is_trained]

        if not trained_models:
            raise ValueError("No trained models available")

        # Ensemble prediction
        all_predictions = []
        all_probabilities = []

        for regime, model in trained_models:
            pred = model.predict(X)
            proba = model.predict_proba(X)

            all_predictions.append(pred)
            all_probabilities.append(proba)

        # Average predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)
        ensemble_probabilities = np.mean(all_probabilities, axis=0)

        final_predictions = (ensemble_predictions > 0.5).astype(int)

        regimes_used = [detected_regime] * len(final_predictions)

        return final_predictions, ensemble_probabilities, regimes_used

    def evaluate_regime_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regime_labels_test: List[MarketRegime]
    ) -> Dict[MarketRegime, RegimeModelPerformance]:
        """Evaluate performance of each regime model"""

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        performance_results = {}

        for regime in self.regime_models.keys():
            # Filter test data for this regime
            regime_mask = np.array([r == regime for r in regime_labels_test])

            if np.sum(regime_mask) < 10:  # Need minimum samples
                continue

            X_regime = X_test[regime_mask]
            y_regime = y_test[regime_mask]

            model = self.regime_models[regime]

            if not model.is_trained:
                continue

            try:
                # Make predictions
                y_pred = model.predict(X_regime)

                # Calculate metrics
                accuracy = accuracy_score(y_regime, y_pred)
                precision = precision_score(y_regime, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_regime, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_regime, y_pred, average='weighted', zero_division=0)

                # Create performance object
                performance = RegimeModelPerformance(
                    regime=regime,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    sharpe_ratio=0.0,  # Would need returns data
                    max_drawdown=0.0,  # Would need returns data
                    total_predictions=len(y_pred),
                    regime_coverage=np.sum(regime_mask) / len(regime_labels_test)
                )

                performance_results[regime] = performance

                self.logger.info(f"{regime.value} model - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

            except Exception as e:
                self.logger.error(f"Evaluation failed for {regime.value} model: {e}")

        self.model_performance = performance_results
        return performance_results

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""

        if not self.routing_history:
            return {'error': 'No routing history available'}

        # Routing frequency by regime
        regime_usage = {}
        for record in self.routing_history:
            regime = record['regime']
            regime_usage[regime.value] = regime_usage.get(regime.value, 0) + 1

        # Average confidence by regime
        regime_confidence = {}
        for regime_value in regime_usage.keys():
            confidences = [r['confidence'] for r in self.routing_history
                          if r['regime'].value == regime_value]
            regime_confidence[regime_value] = np.mean(confidences) if confidences else 0.0

        # Model performance summary
        performance_summary = {}
        for regime, perf in self.model_performance.items():
            performance_summary[regime.value] = {
                'accuracy': perf.accuracy,
                'f1_score': perf.f1_score,
                'coverage': perf.regime_coverage
            }

        return {
            'total_routing_decisions': len(self.routing_history),
            'regime_usage_distribution': regime_usage,
            'average_confidence_by_regime': regime_confidence,
            'model_performance_summary': performance_summary,
            'most_used_regime': max(regime_usage, key=regime_usage.get) if regime_usage else None
        }

def create_regime_aware_system(
    regime_detector: MarketRegimeDetector = None
) -> RegimeRouter:
    """Create complete regime-aware modeling system"""

    if regime_detector is None:
        from ml.regime.market_regime_detector import create_regime_detector
        regime_detector = create_regime_detector()

    return RegimeRouter(regime_detector)

def train_regime_aware_models(
    X: np.ndarray,
    y: np.ndarray,
    market_data: pd.DataFrame,
    regime_detector: MarketRegimeDetector = None
) -> RegimeRouter:
    """High-level function to train regime-aware models"""

    # Create regime router
    router = create_regime_aware_system(regime_detector)

    # Fit regime detector on historical data
    router.regime_detector.fit_historical_regimes(market_data)

    # Detect regimes for training data
    regime_labels = []
    for i in range(len(market_data)):
        subset_data = market_data.iloc[:i+1]
        if len(subset_data) > 30:  # Need minimum data for regime detection
            regime_state = router.regime_detector.detect_current_regime(subset_data)
            regime_labels.append(regime_state.regime)
        else:
            regime_labels.append(MarketRegime.UNKNOWN)

    # Filter out unknown regimes
    known_mask = np.array([r != MarketRegime.UNKNOWN for r in regime_labels])
    X_filtered = X[known_mask]
    y_filtered = y[known_mask]
    regime_labels_filtered = [r for r in regime_labels if r != MarketRegime.UNKNOWN]

    # Train regime-specific models
    router.train_regime_models(X_filtered, y_filtered, regime_labels_filtered)

    return router
