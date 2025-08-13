"""
Regime Classification Models

Machine learning models for market regime classification:
- Hidden Markov Models for regime transitions
- Ensemble classification for robust regime detection
- Online learning for adaptive regime recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import joblib  # type: ignore
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TREND_UP = "trend_up"           # Strong uptrend with momentum
    TREND_DOWN = "trend_down"       # Strong downtrend with momentum
    MEAN_REVERSION = "mean_reversion"  # Range-bound, mean-reverting
    HIGH_VOL_CHOP = "high_vol_chop"    # High volatility, no clear trend
    LOW_VOL_DRIFT = "low_vol_drift"    # Low volatility, weak trend
    RISK_OFF = "risk_off"              # Flight to safety, correlation breakdown

    @classmethod
    def get_trading_regimes(cls) -> List['MarketRegime']:
        """Get regimes suitable for active trading"""
        return [cls.TREND_UP, cls.TREND_DOWN, cls.MEAN_REVERSION]

    @classmethod
    def get_no_trade_regimes(cls) -> List['MarketRegime']:
        """Get regimes where trading should be avoided"""
        return [cls.HIGH_VOL_CHOP, cls.RISK_OFF]

@dataclass
class RegimeClassification:
    """Container for regime classification results"""
    primary_regime: MarketRegime
    confidence: float
    probabilities: Dict[MarketRegime, float]
    feature_importance: Dict[str, float]
    timestamp: pd.Timestamp
    should_trade: bool


class RegimeClassifier:
    """
    Multi-model ensemble for market regime classification
    """

    def __init__(self, model_path: str = "models/regime_classifier.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.regime_history = []

        # Model parameters
        self.n_estimators = 100
        self.lookback_window = 50
        self.min_confidence_threshold = 0.6

    def _prepare_features(self, feature_set) -> np.ndarray:
        """Convert RegimeFeatureSet to ML feature vector"""
        features = [
            feature_set.hurst_exponent,
            feature_set.adx,
            feature_set.realized_vol,
            feature_set.atr_normalized,
            feature_set.btc_dominance,
            feature_set.alt_breadth,
            feature_set.funding_impulse,
            feature_set.oi_impulse,
        ]

        # Add derived features
        features.extend([
            # Trend vs mean reversion signal
            (feature_set.hurst_exponent - 0.5) * feature_set.adx,

            # Volatility momentum
            feature_set.realized_vol * (1 if feature_set.volatility_regime == 'high' else -1),

            # Market structure health
            feature_set.btc_dominance * feature_set.alt_breadth / 100,

            # Derivatives sentiment
            feature_set.funding_impulse + feature_set.oi_impulse,
        ])

        return np.array(features).reshape(1, -1)

    def _create_labels_from_returns(self, returns: pd.Series,
                                   features_df: pd.DataFrame) -> List[MarketRegime]:
        """
        Create regime labels based on market behavior patterns
        """
        labels = []

        for i in range(len(returns)):
            if i < 20:  # Need history for classification
                labels.append(MarketRegime.LOW_VOL_DRIFT)
                continue

            # Look at recent window
            window_returns = returns.iloc[i-20:i]
            window_features = features_df.iloc[i] if i < len(features_df) else None

            # Calculate metrics
            total_return = (1 + window_returns).prod() - 1
            volatility = window_returns.std() * np.sqrt(len(window_returns))
            sharpe = total_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(window_returns)

            # Get features for this period
            hurst = window_features.hurst_exponent if window_features is not None else 0.5
            adx = window_features.adx if window_features is not None else 20

            # Classification logic
            if volatility > 0.15:  # High volatility
                if abs(total_return) < 0.05 and max_drawdown > 0.1:
                    labels.append(MarketRegime.HIGH_VOL_CHOP)
                elif total_return < -0.1:
                    labels.append(MarketRegime.RISK_OFF)
                else:
                    labels.append(MarketRegime.TREND_UP if total_return > 0 else MarketRegime.TREND_DOWN)

            elif hurst > 0.55 and adx > 25:  # Strong trend
                labels.append(MarketRegime.TREND_UP if total_return > 0 else MarketRegime.TREND_DOWN)

            elif hurst < 0.45:  # Mean reverting
                labels.append(MarketRegime.MEAN_REVERSION)

            else:  # Low volatility drift
                labels.append(MarketRegime.LOW_VOL_DRIFT)

        return labels

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return abs(drawdown.min())

    def train(self, historical_features: List, returns_data: pd.Series) -> Dict[str, Any]:
        """
        Train the regime classification model

        Args:
            historical_features: List of RegimeFeatureSet objects
            returns_data: Corresponding returns data for labeling

        Returns:
            Training metrics and model performance
        """
        try:
            logger.info("Starting regime classifier training...")

            # Prepare feature matrix
            X = []
            feature_objects = []

            for feature_set in historical_features:
                features = self._prepare_features(feature_set)
                X.append(features[0])
                feature_objects.append(feature_set)

            X = np.array(X)

            # Create feature DataFrame for labeling
            features_df = pd.DataFrame([{
                'hurst_exponent': f.hurst_exponent,
                'adx': f.adx,
                'realized_vol': f.realized_vol,
                'atr_normalized': f.atr_normalized,
                'btc_dominance': f.btc_dominance,
                'alt_breadth': f.alt_breadth,
                'funding_impulse': f.funding_impulse,
                'oi_impulse': f.oi_impulse,
            } for f in feature_objects])

            # Generate labels based on market behavior
            y = self._create_labels_from_returns(returns_data, features_df)
            y_encoded = [regime.value for regime in y]

            # Ensure we have enough samples for each class
            unique_labels, counts = np.unique(y_encoded, return_counts=True)
            logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")

            if len(unique_labels) < 3:
                logger.warning("Insufficient regime diversity for robust training")
                return {"error": "Insufficient regime diversity"}

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Train Random Forest with time series validation
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )

            # Cross-validation scores
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = [y_encoded[i] for i in train_idx]
                y_val = [y_encoded[i] for i in val_idx]

                self.model.fit(X_train, y_train)
                score = self.model.score(X_val, y_val)
                cv_scores.append(score)

            # Final training on all data
            self.model.fit(X_scaled, y_encoded)

            # Feature importance
            feature_names = [
                'hurst_exponent', 'adx', 'realized_vol', 'atr_normalized',
                'btc_dominance', 'alt_breadth', 'funding_impulse', 'oi_impulse',
                'trend_strength', 'volatility_momentum', 'market_health', 'derivatives_sentiment'
            ]

            self.feature_names = feature_names
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))

            # Save model
            self.save_model()
            self.is_trained = True

            training_metrics = {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'feature_importance': feature_importance,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'regime_distribution': dict(zip(unique_labels, counts))
            }

            logger.info(f"Training completed. Mean CV score: {np.mean(cv_scores):.3f}")
            return training_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    def predict_regime(self, feature_set) -> RegimeClassification:
        """
        Predict market regime from features

        Args:
            feature_set: RegimeFeatureSet object

        Returns:
            RegimeClassification with prediction and confidence
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained, loading from disk...")
                if not self.load_model():
                    return self._default_classification()

            # Prepare features
            X = self._prepare_features(feature_set)
            X_scaled = self.scaler.transform(X)

            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]

            # Map to regime enum
            regime_classes = self.model.classes_
            regime_probs = {}

            for i, regime_str in enumerate(regime_classes):
                try:
                    regime_enum = MarketRegime(regime_str)
                    regime_probs[regime_enum] = probabilities[i]
                except ValueError:
                    continue

            primary_regime = MarketRegime(prediction)
            confidence = max(probabilities)

            # Feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

            # Trading decision
            should_trade = (
                primary_regime in MarketRegime.get_trading_regimes() and
                confidence > self.min_confidence_threshold
            )

            classification = RegimeClassification(
                primary_regime=primary_regime,
                confidence=confidence,
                probabilities=regime_probs,
                feature_importance=feature_importance,
                timestamp=pd.Timestamp.now(),
                should_trade=should_trade
            )

            # Update history
            self.regime_history.append(classification)
            if len(self.regime_history) > self.lookback_window:
                self.regime_history = self.regime_history[-self.lookback_window:]

            return classification

        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return self._default_classification()

    def get_regime_stability(self) -> Dict[str, float]:
        """
        Analyze regime stability from recent history

        Returns:
            Metrics about regime persistence and transitions
        """
        if len(self.regime_history) < 5:
            return {"stability": 0.5, "transition_rate": 0.5, "confidence_trend": 0.0}

        recent = self.regime_history[-10:]

        # Regime persistence
        regimes = [r.primary_regime for r in recent]
        unique_regimes = set(regimes)
        stability = 1.0 - (len(unique_regimes) - 1) / len(recent)

        # Transition rate
        transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        transition_rate = transitions / (len(regimes) - 1)

        # Confidence trend
        confidences = [r.confidence for r in recent]
        confidence_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]

        return {
            "stability": stability,
            "transition_rate": transition_rate,
            "confidence_trend": confidence_trend,
            "current_regime": regimes[-1].value,
            "avg_confidence": np.mean(confidences)
        }

    def save_model(self) -> bool:
        """Save trained model to disk"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }

            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            model_data = joblib.load(self.model_path)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', [])
            self.is_trained = model_data.get('is_trained', False)

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _default_classification(self) -> RegimeClassification:
        """Return default classification when prediction fails"""
        return RegimeClassification(
            primary_regime=MarketRegime.LOW_VOL_DRIFT,
            confidence=0.5,
            probabilities={MarketRegime.LOW_VOL_DRIFT: 0.5},
            feature_importance={},
            timestamp=pd.Timestamp.now(),
            should_trade=False
        )
