"""
Meta-Learner for Ensemble Predictions

Implements sophisticated meta-learning to optimally combine base model predictions
using logistic regression, XGBoost, or neural networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import cross_val_score, TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using alternatives")

from .base_models import ModelPrediction

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble meta-learner"""
    meta_model_type: str = "logistic"          # 'logistic', 'xgboost', 'rf'
    calibration_method: str = "isotonic"       # 'platt', 'isotonic'
    cv_folds: int = 5

    # Feature engineering
    include_interactions: bool = True           # Include model interaction terms
    include_confidence_weights: bool = True     # Weight by model confidence
    include_temporal_features: bool = True      # Add time-based features

    # Training parameters
    min_training_samples: int = 100
    retrain_frequency_hours: int = 24
    validation_split: float = 0.2


@dataclass
class EnsemblePrediction:
    """Final ensemble prediction output"""
    symbol: str
    timestamp: datetime

    # Final prediction
    probability: float
    confidence: float
    direction: str

    # Component analysis
    base_predictions: Dict[str, ModelPrediction]
    model_weights: Dict[str, float]
    consensus_score: float              # How much models agree

    # Meta-learner details
    meta_model_confidence: float
    feature_vector: np.ndarray
    calibrated_probability: float

    # Performance metrics
    expected_auc: float
    expected_precision: float
    turnover_reduction: float


class MetaLearner:
    """
    Meta-learner that combines base model predictions optimally
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config

        # Meta-model
        self.meta_model = None
        self.calibrator = None
        self.is_trained = False

        # Feature engineering
        self.feature_names = []
        self.model_names = []

        # Performance tracking
        self.training_history = []
        self.prediction_history = []

        # Last training time
        self.last_training_time = None

    def train(self,
             training_data: List[Dict[str, Any]],
             outcomes: List[bool]) -> Dict[str, Any]:
        """
        Train meta-learner on historical base model predictions and outcomes

        Args:
            training_data: List of dicts containing base model predictions
            outcomes: List of actual outcomes (True=up, False=down)

        Returns:
            Training metrics and performance
        """
        try:
            if len(training_data) < self.config.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(training_data)} < {self.config.min_training_samples}")

            logger.info(f"Training meta-learner on {len(training_data)} samples")

            # Extract features from base model predictions
            X, feature_names = self._extract_features(training_data)
            y = np.array(outcomes, dtype=int)

            self.feature_names = feature_names
            self.model_names = list(set(
                pred.model_name
                for data in training_data
                for pred in data['base_predictions'].values()
            ))

            # Split data for validation
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train meta-model
            self.meta_model = self._create_meta_model()

            # Use time series cross-validation for training
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            cv_scores = cross_val_score(self.meta_model, X_train, y_train, cv=tscv, scoring='roc_auc')

            # Fit on full training set
            self.meta_model.fit(X_train, y_train)

            # Calibrate probabilities
            self.calibrator = CalibratedClassifierCV(
                self.meta_model,
                method=self.config.calibration_method,
                cv=3
            )
            self.calibrator.fit(X_train, y_train)

            # Evaluate on validation set
            val_predictions = self.calibrator.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_predictions)

            # Calculate precision-recall AUC
            precision, recall, _ = precision_recall_curve(y_val, val_predictions)
            pr_auc = auc(recall, precision)

            # Feature importance
            feature_importance = self._get_feature_importance()

            self.is_trained = True
            self.last_training_time = datetime.now()

            training_metrics = {
                'training_samples': len(training_data),
                'cv_auc_mean': np.mean(cv_scores),
                'cv_auc_std': np.std(cv_scores),
                'validation_auc': val_auc,
                'validation_pr_auc': pr_auc,
                'feature_importance': feature_importance,
                'model_type': self.config.meta_model_type,
                'training_time': datetime.now().isoformat()
            }

            self.training_history.append(training_metrics)

            logger.info(f"Meta-learner training completed. Validation AUC: {val_auc:.3f}")

            return training_metrics

        except Exception as e:
            logger.error(f"Meta-learner training failed: {e}")
            raise

    def predict(self, base_predictions: Dict[str, ModelPrediction]) -> EnsemblePrediction:
        """
        Generate ensemble prediction from base model predictions

        Args:
            base_predictions: Dict mapping model_name -> ModelPrediction

        Returns:
            Final ensemble prediction
        """
        try:
            if not self.is_trained:
                raise ValueError("Meta-learner not trained")

            if not base_predictions:
                raise ValueError("No base predictions provided")

            # Extract features for this prediction
            feature_data = {'base_predictions': base_predictions}
            X, _ = self._extract_features([feature_data])

            if len(X) == 0:
                raise ValueError("Failed to extract features")

            # Get meta-model prediction
            raw_probability = self.meta_model.predict_proba(X)[0, 1]
            calibrated_probability = self.calibrator.predict_proba(X)[0, 1]

            # Calculate ensemble confidence
            meta_confidence = self._calculate_meta_confidence(X[0], base_predictions)

            # Calculate model consensus
            consensus_score = self._calculate_consensus(base_predictions)

            # Calculate model weights (feature importance based)
            model_weights = self._calculate_model_weights(base_predictions)

            # Determine final direction
            direction = 'up' if calibrated_probability > 0.5 else 'down'

            # Estimate performance metrics
            expected_auc = self._estimate_expected_auc(base_predictions)
            expected_precision = self._estimate_expected_precision(calibrated_probability)
            turnover_reduction = self._estimate_turnover_reduction(base_predictions)

            ensemble_prediction = EnsemblePrediction(
                symbol=list(base_predictions.values())[0].symbol,
                timestamp=datetime.now(),
                probability=calibrated_probability,
                confidence=meta_confidence,
                direction=direction,
                base_predictions=base_predictions,
                model_weights=model_weights,
                consensus_score=consensus_score,
                meta_model_confidence=meta_confidence,
                feature_vector=X[0],
                calibrated_probability=calibrated_probability,
                expected_auc=expected_auc,
                expected_precision=expected_precision,
                turnover_reduction=turnover_reduction
            )

            # Store prediction for analysis
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'symbol': ensemble_prediction.symbol,
                'probability': calibrated_probability,
                'confidence': meta_confidence,
                'consensus': consensus_score
            })

            # Keep only recent predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return ensemble_prediction

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise

    def needs_retraining(self) -> bool:
        """Check if meta-learner needs retraining"""
        if not self.is_trained or self.last_training_time is None:
            return True

        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() / 3600 > self.config.retrain_frequency_hours

    def _create_meta_model(self):
        """Create the meta-learning model"""
        if self.config.meta_model_type == "logistic":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.config.meta_model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        elif self.config.meta_model_type == "rf":
            return RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced'
            )
        else:
            logger.warning(f"Unknown meta model type: {self.config.meta_model_type}, using logistic regression")
            return LogisticRegression(random_state=42, max_iter=1000)

    def _extract_features(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Extract features from base model predictions"""
        try:
            features_list = []
            feature_names = []

            for data in training_data:
                base_preds = data['base_predictions']

                if not base_preds:
                    continue

                feature_vector = []

                # Basic model predictions (probabilities and confidences)
                for model_name in sorted(base_preds.keys()):
                    pred = base_preds[model_name]

                    # Core features
                    feature_vector.extend([
                        pred.probability,
                        pred.confidence,
                        1.0 if pred.direction == 'up' else 0.0
                    ])

                    if not feature_names:  # Only set feature names once
                        feature_names.extend([
                            f"{model_name}_probability",
                            f"{model_name}_confidence",
                            f"{model_name}_direction"
                        ])

                # Interaction features
                if self.config.include_interactions and len(base_preds) > 1:
                    model_probs = [pred.probability for pred in base_preds.values()]

                    # Probability variance (disagreement measure)
                    feature_vector.append(np.var(model_probs))
                    if len(feature_names) == len(feature_vector) - 1:
                        feature_names.append("probability_variance")

                    # Pairwise differences
                    prob_list = list(model_probs)
                    for i in range(len(prob_list)):
                        for j in range(i + 1, len(prob_list)):
                            feature_vector.append(abs(prob_list[i] - prob_list[j]))
                            if len(feature_names) == len(feature_vector) - 1:
                                feature_names.append(f"prob_diff_{i}_{j}")

                # Confidence-weighted features
                if self.config.include_confidence_weights:
                    confidences = [pred.confidence for pred in base_preds.values()]
                    probabilities = [pred.probability for pred in base_preds.values()]

                    # Confidence-weighted probability
                    if sum(confidences) > 0:
                        weighted_prob = sum(p * c for p, c in zip(probabilities, confidences)) / sum(confidences)
                        feature_vector.append(weighted_prob)
                        if len(feature_names) == len(feature_vector) - 1:
                            feature_names.append("confidence_weighted_probability")

                # Temporal features
                if self.config.include_temporal_features:
                    # Hour of day
                    hour = datetime.now().hour
                    feature_vector.extend([
                        np.sin(2 * np.pi * hour / 24),  # Cyclic encoding
                        np.cos(2 * np.pi * hour / 24)
                    ])

                    if len(feature_names) == len(feature_vector) - 2:
                        feature_names.extend(["hour_sin", "hour_cos"])

                    # Day of week
                    day = datetime.now().weekday()
                    feature_vector.extend([
                        np.sin(2 * np.pi * day / 7),
                        np.cos(2 * np.pi * day / 7)
                    ])

                    if len(feature_names) == len(feature_vector) - 2:
                        feature_names.extend(["day_sin", "day_cos"])

                features_list.append(feature_vector)

            if not features_list:
                return np.array([]), []

            # Ensure all feature vectors have the same length
            max_length = max(len(f) for f in features_list)
            normalized_features = []

            for feature_vector in features_list:
                if len(feature_vector) < max_length:
                    # Pad with zeros
                    feature_vector.extend([0.0] * (max_length - len(feature_vector)))
                normalized_features.append(feature_vector)

            return np.array(normalized_features), feature_names[:max_length]

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.array([]), []

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(self.meta_model, 'feature_importances_'):
                # Tree-based models (RF, XGBoost)
                importances = self.meta_model.feature_importances_
            elif hasattr(self.meta_model, 'coef_'):
                # Linear models (Logistic Regression)
                importances = np.abs(self.meta_model.coef_[0])
            else:
                return {}

            if len(importances) != len(self.feature_names):
                logger.warning("Feature importance length mismatch")
                return {}

            importance_dict = {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }

            return importance_dict

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}

    def _calculate_meta_confidence(self, feature_vector: np.ndarray,
                                  base_predictions: Dict[str, ModelPrediction]) -> float:
        """Calculate meta-learner confidence"""
        try:
            # Base confidence from model prediction uncertainty
            pred_proba = self.meta_model.predict_proba(feature_vector.reshape(1, -1))[0]
            entropy = -sum(p * np.log(p + 1e-8) for p in pred_proba)
            uncertainty_confidence = 1 - entropy / np.log(2)  # Normalize by max entropy

            # Base model confidence consensus
            confidences = [pred.confidence for pred in base_predictions.values()]
            consensus_confidence = np.mean(confidences) if confidences else 0.5

            # Combined confidence
            meta_confidence = 0.6 * uncertainty_confidence + 0.4 * consensus_confidence

            return max(0.0, min(1.0, meta_confidence))

        except Exception as e:
            logger.error(f"Meta-confidence calculation failed: {e}")
            return 0.5

    def _calculate_consensus(self, base_predictions: Dict[str, ModelPrediction]) -> float:
        """Calculate how much base models agree"""
        try:
            if len(base_predictions) < 2:
                return 1.0

            probabilities = [pred.probability for pred in base_predictions.values()]
            directions = [1 if pred.direction == 'up' else 0 for pred in base_predictions.values()]

            # Probability consensus (lower variance = higher consensus)
            prob_variance = np.var(probabilities)
            prob_consensus = max(0, 1 - prob_variance * 4)  # Scale factor

            # Direction consensus
            direction_consensus = 1.0 if len(set(directions)) == 1 else 0.0

            # Combined consensus
            consensus = 0.7 * prob_consensus + 0.3 * direction_consensus

            return max(0.0, min(1.0, consensus))

        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            return 0.5

    def _calculate_model_weights(self, base_predictions: Dict[str, ModelPrediction]) -> Dict[str, float]:
        """Calculate model weights based on feature importance"""
        try:
            weights = {}
            feature_importance = self._get_feature_importance()

            for model_name in base_predictions.keys():
                # Sum importance of features belonging to this model
                model_importance = sum(
                    importance for feature_name, importance in feature_importance.items()
                    if model_name in feature_name
                )
                weights[model_name] = model_importance

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # Equal weights if no importance available
                equal_weight = 1.0 / len(base_predictions)
                weights = {k: equal_weight for k in base_predictions.keys()}

            return weights

        except Exception as e:
            logger.error(f"Model weight calculation failed: {e}")
            # Return equal weights
            equal_weight = 1.0 / len(base_predictions)
            return {k: equal_weight for k in base_predictions.keys()}

    def _estimate_expected_auc(self, base_predictions: Dict[str, ModelPrediction]) -> float:
        """Estimate expected AUC based on historical performance"""
        try:
            if not self.training_history:
                return 0.7  # Default assumption

            # Get latest training metrics
            latest_metrics = self.training_history[-1]
            base_auc = latest_metrics.get('validation_auc', 0.7)

            # Adjust based on current prediction strength
            confidences = [pred.confidence for pred in base_predictions.values()]
            avg_confidence = np.mean(confidences) if confidences else 0.5

            # Higher confidence typically correlates with higher AUC
            adjusted_auc = base_auc * (0.8 + 0.4 * avg_confidence)

            return max(0.5, min(1.0, adjusted_auc))

        except Exception as e:
            logger.error(f"Expected AUC estimation failed: {e}")
            return 0.7

    def _estimate_expected_precision(self, probability: float) -> float:
        """Estimate expected precision based on probability"""
        try:
            # Higher probability predictions tend to have higher precision
            # This is a simplified heuristic
            base_precision = 0.6  # Baseline
            prob_boost = (probability - 0.5) * 0.8  # Boost for high confidence

            expected_precision = base_precision + prob_boost

            return max(0.2, min(1.0, expected_precision))

        except Exception as e:
            logger.error(f"Expected precision estimation failed: {e}")
            return 0.6

    def _estimate_turnover_reduction(self, base_predictions: Dict[str, ModelPrediction]) -> float:
        """Estimate turnover reduction compared to individual models"""
        try:
            # Ensemble typically reduces turnover by filtering out low-confidence predictions
            # and reducing whipsaws through consensus

            # Base models might have different signal frequencies
            model_types = [pred.model_type for pred in base_predictions.values()]

            # Technical analysis tends to generate more signals
            ta_weight = sum(1 for mt in model_types if mt == 'technical') / len(model_types)

            # More TA = more potential for turnover reduction
            base_reduction = 0.1 + ta_weight * 0.2

            # Consensus reduces whipsaws
            consensus = self._calculate_consensus(base_predictions)
            consensus_reduction = consensus * 0.15

            total_reduction = base_reduction + consensus_reduction

            return max(0.0, min(0.5, total_reduction))  # Max 50% reduction

        except Exception as e:
            logger.error(f"Turnover reduction estimation failed: {e}")
            return 0.2

    def save_model(self, path: str) -> bool:
        """Save trained meta-learner to disk"""
        try:
            if not self.is_trained:
                logger.warning("Cannot save untrained meta-learner")
                return False

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'meta_model': self.meta_model,
                'calibrator': self.calibrator,
                'config': self.config,
                'feature_names': self.feature_names,
                'model_names': self.model_names,
                'training_history': self.training_history,
                'last_training_time': self.last_training_time,
                'is_trained': self.is_trained
            }

            joblib.dump(model_data, path)
            logger.info(f"Meta-learner saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save meta-learner: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load trained meta-learner from disk"""
        try:
            if not Path(path).exists():
                logger.warning(f"Meta-learner file not found: {path}")
                return False

            model_data = joblib.load(path)

            self.meta_model = model_data['meta_model']
            self.calibrator = model_data['calibrator']
            self.config = model_data.get('config', self.config)
            self.feature_names = model_data['feature_names']
            self.model_names = model_data['model_names']
            self.training_history = model_data.get('training_history', [])
            self.last_training_time = model_data.get('last_training_time')
            self.is_trained = model_data.get('is_trained', False)

            logger.info(f"Meta-learner loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load meta-learner: {e}")
            return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of meta-learner"""
        try:
            if not self.training_history:
                return {"status": "No training history"}

            latest_training = self.training_history[-1]

            summary = {
                "model_status": "trained" if self.is_trained else "not_trained",
                "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
                "needs_retraining": self.needs_retraining(),
                "latest_performance": {
                    "validation_auc": latest_training.get('validation_auc', 0),
                    "validation_pr_auc": latest_training.get('validation_pr_auc', 0),
                    "cv_auc_mean": latest_training.get('cv_auc_mean', 0),
                    "training_samples": latest_training.get('training_samples', 0)
                },
                "feature_count": len(self.feature_names),
                "model_count": len(self.model_names),
                "recent_predictions": len(self.prediction_history),
                "meta_model_type": self.config.meta_model_type
            }

            # Recent prediction statistics
            if self.prediction_history:
                recent_preds = self.prediction_history[-100:]  # Last 100 predictions
                summary["recent_stats"] = {
                    "avg_confidence": np.mean([p['confidence'] for p in recent_preds]),
                    "avg_consensus": np.mean([p.get('consensus', 0.5) for p in recent_preds]),
                    "prediction_rate": len(recent_preds)
                }

            return summary

        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"status": "Error", "error": str(e)}
