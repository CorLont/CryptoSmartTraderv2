#!/usr/bin/env python3
"""
Meta-Learning Ensemble System with Stacking and Online Weight Adjustment
Combines predictions from base models using regime-aware meta-learners
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, mean_absolute_error, accuracy_score

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.structured_logger import get_structured_logger

class OnlineWeightTracker:
    """Track model performance and adjust weights in sliding window"""
    
    def __init__(self, window_size: int = 100, min_samples: int = 20):
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Performance tracking per model
        self.performance_history = {}  # {model_name: deque of (prediction, actual, error)}
        self.current_weights = {}      # {model_name: weight}
        self.model_scores = {}         # {model_name: current_score}
        
        self.logger = get_structured_logger("OnlineWeightTracker")
    
    def update_performance(self, model_predictions: Dict[str, float], 
                          actual_value: float, 
                          timestamp: Optional[datetime] = None):
        """Update performance tracking with new prediction results"""
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            for model_name, prediction in model_predictions.items():
                # Initialize tracking for new models
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = deque(maxlen=self.window_size)
                    self.current_weights[model_name] = 1.0 / len(model_predictions)
                
                # Calculate error
                error = abs(prediction - actual_value)
                
                # Store performance record
                record = {
                    'prediction': prediction,
                    'actual': actual_value,
                    'error': error,
                    'timestamp': timestamp
                }
                
                self.performance_history[model_name].append(record)
            
            # Update weights based on recent performance
            self._update_weights()
            
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")
    
    def _update_weights(self):
        """Update model weights based on sliding window performance"""
        
        try:
            model_performances = {}
            
            for model_name, history in self.performance_history.items():
                if len(history) < self.min_samples:
                    # Use equal weight for models with insufficient history
                    model_performances[model_name] = 1.0
                    continue
                
                # Calculate recent performance metrics
                recent_errors = [record['error'] for record in history]
                recent_mae = np.mean(recent_errors)
                recent_std = np.std(recent_errors)
                
                # Performance score (lower MAE = higher score)
                # Add stability bonus (lower std = higher score)
                performance_score = 1.0 / (recent_mae + 0.001) + 0.1 / (recent_std + 0.001)
                model_performances[model_name] = performance_score
                self.model_scores[model_name] = {
                    'mae': recent_mae,
                    'std': recent_std,
                    'score': performance_score,
                    'samples': len(history)
                }
            
            # Normalize weights (softmax-like with temperature)
            temperature = 2.0  # Controls weight distribution sharpness
            max_score = max(model_performances.values()) if model_performances else 1.0
            
            exp_scores = {}
            for model_name, score in model_performances.items():
                exp_scores[model_name] = np.exp((score - max_score) / temperature)
            
            total_exp = sum(exp_scores.values())
            
            # Update weights
            for model_name in model_performances.keys():
                if total_exp > 0:
                    self.current_weights[model_name] = exp_scores[model_name] / total_exp
                else:
                    self.current_weights[model_name] = 1.0 / len(model_performances)
            
            self.logger.debug(f"Updated weights: {self.current_weights}")
            
        except Exception as e:
            self.logger.error(f"Weight update failed: {e}")
    
    def get_weighted_prediction(self, model_predictions: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Get weighted ensemble prediction"""
        
        try:
            if not model_predictions:
                return 0.0, {}
            
            # Ensure all models have weights
            for model_name in model_predictions.keys():
                if model_name not in self.current_weights:
                    self.current_weights[model_name] = 1.0 / len(model_predictions)
            
            # Calculate weighted prediction
            weighted_sum = 0.0
            total_weight = 0.0
            
            used_weights = {}
            
            for model_name, prediction in model_predictions.items():
                weight = self.current_weights.get(model_name, 0.0)
                weighted_sum += prediction * weight
                total_weight += weight
                used_weights[model_name] = weight
            
            if total_weight > 0:
                final_prediction = weighted_sum / total_weight
            else:
                final_prediction = np.mean(list(model_predictions.values()))
            
            return final_prediction, used_weights
            
        except Exception as e:
            self.logger.error(f"Weighted prediction failed: {e}")
            # Fallback to simple average
            return np.mean(list(model_predictions.values())), {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'total_models': len(self.current_weights),
            'window_size': self.window_size
        }
        
        for model_name in self.current_weights.keys():
            model_info = {
                'weight': self.current_weights.get(model_name, 0.0),
                'samples': len(self.performance_history.get(model_name, [])),
                'performance': self.model_scores.get(model_name, {})
            }
            summary['models'][model_name] = model_info
        
        return summary

class MetaLearnerStacker:
    """Meta-learner using stacking with base model predictions and regime features"""
    
    def __init__(self, meta_model_type: str = "logistic", use_regime_features: bool = True):
        self.meta_model_type = meta_model_type
        self.use_regime_features = use_regime_features
        
        self.meta_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        self.feature_importance = {}
        self.performance_metrics = {}
        
        self.logger = get_structured_logger("MetaLearnerStacker")
    
    def _prepare_meta_features(self, base_predictions: Dict[str, np.ndarray],
                              uncertainties: Dict[str, np.ndarray],
                              regime_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Prepare features for meta-learner"""
        
        try:
            meta_features = []
            
            # Base model predictions
            for model_name, preds in base_predictions.items():
                meta_features.append(preds.reshape(-1, 1))
            
            # Base model uncertainties
            for model_name, uncerts in uncertainties.items():
                meta_features.append(uncerts.reshape(-1, 1))
            
            # Cross-model features
            if len(base_predictions) > 1:
                pred_values = list(base_predictions.values())
                
                # Prediction variance across models
                pred_variance = np.var(pred_values, axis=0).reshape(-1, 1)
                meta_features.append(pred_variance)
                
                # Prediction range (max - min)
                pred_range = (np.max(pred_values, axis=0) - np.min(pred_values, axis=0)).reshape(-1, 1)
                meta_features.append(pred_range)
                
                # Agreement score (1 - normalized std)
                pred_std = np.std(pred_values, axis=0)
                pred_mean = np.mean(pred_values, axis=0)
                agreement = 1.0 - (pred_std / (np.abs(pred_mean) + 0.001))
                meta_features.append(agreement.reshape(-1, 1))
            
            # Regime features if available
            if self.use_regime_features and regime_features is not None:
                # Select numeric regime features
                numeric_cols = regime_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    regime_values = regime_features[numeric_cols].fillna(0).values
                    meta_features.append(regime_values)
                
                # Encode categorical regime features
                categorical_cols = regime_features.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if col in regime_features.columns:
                        # Simple one-hot encoding
                        unique_values = regime_features[col].unique()
                        for value in unique_values[:5]:  # Limit to top 5 categories
                            encoded = (regime_features[col] == value).astype(int).values.reshape(-1, 1)
                            meta_features.append(encoded)
            
            # Concatenate all features
            if meta_features:
                X_meta = np.hstack(meta_features)
            else:
                # Fallback: just use base predictions
                X_meta = np.hstack(list(base_predictions.values())).reshape(len(list(base_predictions.values())[0]), -1)
            
            return X_meta
            
        except Exception as e:
            self.logger.error(f"Meta feature preparation failed: {e}")
            # Fallback to simple concatenation
            return np.hstack(list(base_predictions.values())).reshape(len(list(base_predictions.values())[0]), -1)
    
    def fit(self, base_predictions: Dict[str, np.ndarray],
           uncertainties: Dict[str, np.ndarray],
           targets: np.ndarray,
           regime_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fit meta-learner on base model outputs"""
        
        try:
            self.logger.info(f"Training meta-learner with {len(base_predictions)} base models")
            
            # Prepare meta features
            X_meta = self._prepare_meta_features(base_predictions, uncertainties, regime_features)
            
            if len(X_meta) == 0:
                raise ValueError("No meta features prepared")
            
            # Scale features
            X_meta_scaled = self.feature_scaler.fit_transform(X_meta)
            
            # Convert regression targets to classification for precision@K
            # Use top/bottom 20% as positive/negative classes
            target_threshold_high = np.percentile(targets, 80)
            target_threshold_low = np.percentile(targets, 20)
            
            y_class = np.zeros(len(targets))
            y_class[targets >= target_threshold_high] = 1  # Positive class (top performers)
            y_class[targets <= target_threshold_low] = -1  # Negative class (poor performers)
            # Middle 60% remains 0 (neutral)
            
            # Train meta-learner
            if self.meta_model_type == "logistic":
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
                self.meta_model.fit(X_meta_scaled, y_class)
                
            elif self.meta_model_type == "gbm" and XGB_AVAILABLE:
                # Use XGBoost for gradient boosting
                self.meta_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                self.meta_model.fit(X_meta_scaled, y_class)
                
            else:
                # Fallback to Random Forest
                from sklearn.ensemble import RandomForestClassifier
                self.meta_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                self.meta_model.fit(X_meta_scaled, y_class)
            
            self.is_fitted = True
            
            # Evaluate meta-learner performance
            y_pred = self.meta_model.predict(X_meta_scaled)
            y_pred_proba = self.meta_model.predict_proba(X_meta_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_class, y_pred)
            
            # Precision@5 for top predictions
            if y_pred_proba.shape[1] > 1:  # Multi-class
                top_5_indices = np.argsort(y_pred_proba[:, 1])[-5:]  # Top 5 positive predictions
                if len(top_5_indices) > 0:
                    precision_at_5 = np.mean(y_class[top_5_indices] == 1)
                else:
                    precision_at_5 = 0.0
            else:
                precision_at_5 = 0.0
            
            # Feature importance
            if hasattr(self.meta_model, 'feature_importances_'):
                self.feature_importance = {
                    f'feature_{i}': importance 
                    for i, importance in enumerate(self.meta_model.feature_importances_)
                }
            elif hasattr(self.meta_model, 'coef_'):
                self.feature_importance = {
                    f'feature_{i}': abs(coef)
                    for i, coef in enumerate(self.meta_model.coef_[0])
                }
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'precision_at_5': precision_at_5,
                'training_samples': len(X_meta),
                'features': X_meta.shape[1],
                'positive_class_ratio': np.mean(y_class == 1),
                'negative_class_ratio': np.mean(y_class == -1)
            }
            
            result = {
                'success': True,
                'meta_model_type': self.meta_model_type,
                'performance': self.performance_metrics,
                'feature_importance': self.feature_importance
            }
            
            self.logger.info(f"Meta-learner trained: Accuracy={accuracy:.3f}, Precision@5={precision_at_5:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Meta-learner training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, base_predictions: Dict[str, np.ndarray],
               uncertainties: Dict[str, np.ndarray],
               regime_features: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make meta-learner predictions"""
        
        if not self.is_fitted or self.meta_model is None:
            self.logger.warning("Meta-learner not fitted, using simple average")
            # Fallback to simple average
            pred_values = list(base_predictions.values())
            avg_predictions = np.mean(pred_values, axis=0)
            avg_confidence = 1.0 - np.std(pred_values, axis=0) / (np.abs(avg_predictions) + 0.001)
            return avg_predictions, np.clip(avg_confidence, 0.1, 0.99)
        
        try:
            # Prepare meta features
            X_meta = self._prepare_meta_features(base_predictions, uncertainties, regime_features)
            X_meta_scaled = self.feature_scaler.transform(X_meta)
            
            # Get predictions and probabilities
            predictions = self.meta_model.predict(X_meta_scaled)
            probabilities = self.meta_model.predict_proba(X_meta_scaled)
            
            # Convert back to regression-like outputs
            # Use probability of positive class as confidence
            if probabilities.shape[1] > 1:
                confidence = np.max(probabilities, axis=1)
                # Scale predictions based on class probabilities
                ensemble_preds = np.mean(list(base_predictions.values()), axis=0)
                # Adjust predictions based on meta-learner confidence
                adjusted_preds = ensemble_preds * confidence
            else:
                confidence = np.full(len(predictions), 0.5)
                adjusted_preds = np.mean(list(base_predictions.values()), axis=0)
            
            return adjusted_preds, confidence
            
        except Exception as e:
            self.logger.error(f"Meta-learner prediction failed: {e}")
            # Fallback to simple average
            pred_values = list(base_predictions.values())
            avg_predictions = np.mean(pred_values, axis=0)
            avg_confidence = np.full(len(avg_predictions), 0.5)
            return avg_predictions, avg_confidence

class EnsembleMetaLearner:
    """Main ensemble system with meta-learning and online weight adjustment"""
    
    def __init__(self, 
                 meta_model_type: str = "logistic",
                 online_window_size: int = 100,
                 enable_failsafe: bool = True):
        
        self.logger = get_structured_logger("EnsembleMetaLearner")
        
        # Components
        self.meta_learner = MetaLearnerStacker(meta_model_type)
        self.weight_tracker = OnlineWeightTracker(online_window_size)
        
        # Configuration
        self.enable_failsafe = enable_failsafe
        self.failsafe_models = {}  # Best model per regime
        
        # Performance tracking
        self.ensemble_performance = deque(maxlen=1000)
        self.comparison_metrics = {}
        
        # Model persistence
        self.model_dir = Path("models/ensemble")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, base_predictions: Dict[str, np.ndarray],
           uncertainties: Dict[str, np.ndarray],
           targets: np.ndarray,
           regime_features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the ensemble meta-learner"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Training ensemble meta-learner with {len(base_predictions)} base models")
            
            # Train meta-learner
            meta_result = self.meta_learner.fit(base_predictions, uncertainties, targets, regime_features)
            
            if not meta_result['success']:
                raise ValueError("Meta-learner training failed")
            
            # Initialize failsafe models if enabled
            if self.enable_failsafe:
                self._initialize_failsafe_models(base_predictions, targets, regime_features)
            
            # Calculate ensemble baseline performance
            ensemble_preds, ensemble_conf = self.meta_learner.predict(
                base_predictions, uncertainties, regime_features
            )
            
            ensemble_mae = mean_absolute_error(targets, ensemble_preds)
            
            # Compare with best single model
            best_single_mae = float('inf')
            best_single_model = None
            
            for model_name, preds in base_predictions.items():
                single_mae = mean_absolute_error(targets, preds)
                if single_mae < best_single_mae:
                    best_single_mae = single_mae
                    best_single_model = model_name
            
            # Calculate improvement
            mae_improvement = (best_single_mae - ensemble_mae) / best_single_mae if best_single_mae > 0 else 0
            
            training_time = time.time() - start_time
            
            result = {
                'success': True,
                'training_time': training_time,
                'meta_learner_result': meta_result,
                'ensemble_mae': ensemble_mae,
                'best_single_mae': best_single_mae,
                'best_single_model': best_single_model,
                'mae_improvement': mae_improvement,
                'ensemble_beats_single': ensemble_mae < best_single_mae,
                'failsafe_models': list(self.failsafe_models.keys()) if self.enable_failsafe else []
            }
            
            self.comparison_metrics = result
            
            # Save models
            self.save_models()
            
            self.logger.info(f"Ensemble training completed: MAE={ensemble_mae:.4f}, Best Single={best_single_mae:.4f}, Improvement={mae_improvement:.1%}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Ensemble training failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'training_time': training_time
            }
    
    def _initialize_failsafe_models(self, base_predictions: Dict[str, np.ndarray],
                                   targets: np.ndarray,
                                   regime_features: Optional[pd.DataFrame] = None):
        """Initialize failsafe models per regime"""
        
        try:
            if regime_features is None or 'trend_regime' not in regime_features.columns:
                # Global failsafe (best overall model)
                best_mae = float('inf')
                best_model = None
                
                for model_name, preds in base_predictions.items():
                    mae = mean_absolute_error(targets, preds)
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_name
                
                self.failsafe_models['global'] = {
                    'model_name': best_model,
                    'mae': best_mae
                }
                
                self.logger.info(f"Global failsafe model: {best_model} (MAE: {best_mae:.4f})")
                return
            
            # Per-regime failsafe
            regimes = regime_features['trend_regime'].unique()
            
            for regime in regimes:
                regime_mask = regime_features['trend_regime'] == regime
                
                if np.sum(regime_mask) < 10:  # Need minimum samples
                    continue
                
                regime_targets = targets[regime_mask]
                best_mae = float('inf')
                best_model = None
                
                for model_name, preds in base_predictions.items():
                    regime_preds = preds[regime_mask]
                    mae = mean_absolute_error(regime_targets, regime_preds)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_name
                
                self.failsafe_models[regime] = {
                    'model_name': best_model,
                    'mae': best_mae,
                    'samples': np.sum(regime_mask)
                }
                
                self.logger.info(f"Failsafe for {regime}: {best_model} (MAE: {best_mae:.4f})")
            
        except Exception as e:
            self.logger.error(f"Failsafe initialization failed: {e}")
    
    def predict(self, base_predictions: Dict[str, np.ndarray],
               uncertainties: Dict[str, np.ndarray],
               regime_features: Optional[pd.DataFrame] = None,
               use_online_weights: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Make ensemble predictions with failsafe"""
        
        try:
            prediction_info = {
                'method': 'unknown',
                'confidence': 0.5,
                'failsafe_used': False,
                'models_used': list(base_predictions.keys())
            }
            
            # Try meta-learner first
            try:
                ensemble_preds, ensemble_conf = self.meta_learner.predict(
                    base_predictions, uncertainties, regime_features
                )
                
                # Apply online weights if enabled
                if use_online_weights:
                    # Convert predictions to dict for weight tracker
                    pred_dict = {f'meta_{i}': pred for i, pred in enumerate(ensemble_preds)}
                    weighted_pred, weights = self.weight_tracker.get_weighted_prediction(pred_dict)
                    
                    # Apply weights to ensemble prediction
                    weight_factor = np.mean(list(weights.values())) if weights else 1.0
                    ensemble_preds = ensemble_preds * weight_factor
                    
                    prediction_info['method'] = 'meta_learner_weighted'
                    prediction_info['weights'] = weights
                else:
                    prediction_info['method'] = 'meta_learner'
                
                prediction_info['confidence'] = np.mean(ensemble_conf)
                
                return ensemble_preds, ensemble_conf, prediction_info
                
            except Exception as e:
                self.logger.warning(f"Meta-learner prediction failed: {e}")
                
                if not self.enable_failsafe:
                    raise e
                
                # Failsafe: use best expert per regime
                prediction_info['failsafe_used'] = True
                return self._failsafe_predict(base_predictions, regime_features, prediction_info)
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction completely failed: {e}")
            
            # Ultimate fallback: simple average
            pred_values = list(base_predictions.values())
            avg_preds = np.mean(pred_values, axis=0)
            avg_conf = np.full(len(avg_preds), 0.3)  # Low confidence
            
            prediction_info.update({
                'method': 'simple_average_fallback',
                'confidence': 0.3,
                'failsafe_used': True,
                'error': str(e)
            })
            
            return avg_preds, avg_conf, prediction_info
    
    def _failsafe_predict(self, base_predictions: Dict[str, np.ndarray],
                         regime_features: Optional[pd.DataFrame] = None,
                         prediction_info: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Failsafe prediction using best expert per regime"""
        
        try:
            n_samples = len(list(base_predictions.values())[0])
            failsafe_preds = np.zeros(n_samples)
            failsafe_conf = np.full(n_samples, 0.6)  # Medium confidence
            
            if prediction_info is None:
                prediction_info = {}
            
            prediction_info['method'] = 'failsafe'
            prediction_info['failsafe_details'] = {}
            
            # Use regime-specific failsafe if available
            if (regime_features is not None and 
                'trend_regime' in regime_features.columns and 
                len(self.failsafe_models) > 1):
                
                regimes = regime_features['trend_regime'].values
                
                for regime in np.unique(regimes):
                    regime_mask = regimes == regime
                    
                    if regime in self.failsafe_models:
                        failsafe_info = self.failsafe_models[regime]
                        best_model = failsafe_info['model_name']
                        
                        if best_model in base_predictions:
                            failsafe_preds[regime_mask] = base_predictions[best_model][regime_mask]
                            prediction_info['failsafe_details'][regime] = best_model
                        else:
                            # Use global failsafe
                            global_info = self.failsafe_models.get('global', {})
                            global_model = global_info.get('model_name')
                            if global_model and global_model in base_predictions:
                                failsafe_preds[regime_mask] = base_predictions[global_model][regime_mask]
            
            else:
                # Use global failsafe
                global_info = self.failsafe_models.get('global', {})
                best_model = global_info.get('model_name')
                
                if best_model and best_model in base_predictions:
                    failsafe_preds = base_predictions[best_model]
                    prediction_info['failsafe_details']['global'] = best_model
                else:
                    # Ultimate fallback
                    failsafe_preds = np.mean(list(base_predictions.values()), axis=0)
                    prediction_info['failsafe_details']['global'] = 'simple_average'
            
            return failsafe_preds, failsafe_conf, prediction_info
            
        except Exception as e:
            self.logger.error(f"Failsafe prediction failed: {e}")
            
            # Ultimate fallback
            avg_preds = np.mean(list(base_predictions.values()), axis=0)
            avg_conf = np.full(len(avg_preds), 0.2)
            
            if prediction_info is None:
                prediction_info = {}
            
            prediction_info.update({
                'method': 'ultimate_fallback',
                'error': str(e)
            })
            
            return avg_preds, avg_conf, prediction_info
    
    def update_online_performance(self, predictions: np.ndarray, 
                                 actuals: np.ndarray,
                                 model_info: Dict[str, Any],
                                 timestamp: Optional[datetime] = None):
        """Update online performance tracking"""
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Update ensemble performance
            for pred, actual in zip(predictions, actuals):
                error = abs(pred - actual)
                record = {
                    'prediction': pred,
                    'actual': actual,
                    'error': error,
                    'timestamp': timestamp,
                    'method': model_info.get('method', 'unknown')
                }
                self.ensemble_performance.append(record)
            
            # Update weight tracker if meta-learner was used
            if 'weights' in model_info:
                model_preds = {}
                for i, pred in enumerate(predictions):
                    model_preds[f'ensemble_{i}'] = pred
                
                for actual in actuals:
                    self.weight_tracker.update_performance(model_preds, actual, timestamp)
            
            self.logger.debug(f"Updated online performance: {len(predictions)} samples")
            
        except Exception as e:
            self.logger.error(f"Online performance update failed: {e}")
    
    def evaluate_vs_single_models(self, ensemble_predictions: np.ndarray,
                                 base_predictions: Dict[str, np.ndarray],
                                 targets: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble vs single model performance"""
        
        try:
            results = {
                'ensemble_mae': mean_absolute_error(targets, ensemble_predictions),
                'single_model_performance': {},
                'precision_at_5': 0.0,
                'ensemble_wins': False
            }
            
            # Calculate single model performance
            best_single_mae = float('inf')
            best_single_model = None
            
            for model_name, preds in base_predictions.items():
                mae = mean_absolute_error(targets, preds)
                results['single_model_performance'][model_name] = {'mae': mae}
                
                if mae < best_single_mae:
                    best_single_mae = mae
                    best_single_model = model_name
            
            results['best_single_mae'] = best_single_mae
            results['best_single_model'] = best_single_model
            results['ensemble_wins'] = results['ensemble_mae'] < best_single_mae
            results['improvement'] = (best_single_mae - results['ensemble_mae']) / best_single_mae
            
            # Calculate Precision@5
            if len(targets) >= 5:
                top_5_indices = np.argsort(ensemble_predictions)[-5:]
                top_5_targets = targets[top_5_indices]
                target_threshold = np.percentile(targets, 80)  # Top 20%
                precision_at_5 = np.mean(top_5_targets >= target_threshold)
                results['precision_at_5'] = precision_at_5
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def save_models(self):
        """Save ensemble models"""
        
        if not JOBLIB_AVAILABLE:
            self.logger.warning("Cannot save models - joblib not available")
            return
        
        try:
            # Save meta-learner
            if self.meta_learner.is_fitted:
                meta_model_path = self.model_dir / "meta_learner_model.pkl"
                meta_scaler_path = self.model_dir / "meta_learner_scaler.pkl"
                
                joblib.dump(self.meta_learner.meta_model, meta_model_path)
                joblib.dump(self.meta_learner.feature_scaler, meta_scaler_path)
            
            # Save metadata
            meta_path = self.model_dir / "ensemble_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'meta_model_type': self.meta_learner.meta_model_type,
                    'enable_failsafe': self.enable_failsafe,
                    'failsafe_models': self.failsafe_models,
                    'comparison_metrics': self.comparison_metrics,
                    'save_time': datetime.now().isoformat()
                }, f, default=str)
            
            self.logger.info("Ensemble models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load saved ensemble models"""
        
        if not JOBLIB_AVAILABLE:
            return
        
        try:
            meta_path = self.model_dir / "ensemble_meta.json"
            if not meta_path.exists():
                return
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            self.enable_failsafe = meta.get('enable_failsafe', True)
            self.failsafe_models = meta.get('failsafe_models', {})
            self.comparison_metrics = meta.get('comparison_metrics', {})
            
            # Load meta-learner
            meta_model_path = self.model_dir / "meta_learner_model.pkl"
            meta_scaler_path = self.model_dir / "meta_learner_scaler.pkl"
            
            if meta_model_path.exists() and meta_scaler_path.exists():
                self.meta_learner.meta_model = joblib.load(meta_model_path)
                self.meta_learner.feature_scaler = joblib.load(meta_scaler_path)
                self.meta_learner.is_fitted = True
                self.meta_learner.meta_model_type = meta.get('meta_model_type', 'logistic')
            
            self.logger.info("Ensemble models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")

# Global ensemble instance
_ensemble_meta_learner: Optional[EnsembleMetaLearner] = None

def get_ensemble_meta_learner(meta_model_type: str = "logistic") -> EnsembleMetaLearner:
    """Get global ensemble meta-learner instance"""
    global _ensemble_meta_learner
    
    if _ensemble_meta_learner is None:
        _ensemble_meta_learner = EnsembleMetaLearner(meta_model_type=meta_model_type)
        _ensemble_meta_learner.load_models()
    
    return _ensemble_meta_learner

def train_ensemble_meta_learner(base_predictions: Dict[str, np.ndarray],
                               uncertainties: Dict[str, np.ndarray],
                               targets: np.ndarray,
                               regime_features: Optional[pd.DataFrame] = None,
                               meta_model_type: str = "logistic") -> Dict[str, Any]:
    """Train ensemble meta-learner"""
    ensemble = get_ensemble_meta_learner(meta_model_type)
    return ensemble.fit(base_predictions, uncertainties, targets, regime_features)

def predict_with_ensemble(base_predictions: Dict[str, np.ndarray],
                         uncertainties: Dict[str, np.ndarray],
                         regime_features: Optional[pd.DataFrame] = None,
                         use_online_weights: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Make ensemble predictions"""
    ensemble = get_ensemble_meta_learner()
    return ensemble.predict(base_predictions, uncertainties, regime_features, use_online_weights)

if __name__ == "__main__":
    # Test ensemble meta-learner
    print("Testing Ensemble Meta-Learner System")
    
    # Create mock base model predictions
    n_samples = 200
    base_preds = {
        'model_1': np.# REMOVED: Mock data pattern not allowed in production(0.05, 0.02, n_samples),
        'model_2': np.# REMOVED: Mock data pattern not allowed in production(0.04, 0.025, n_samples),
        'model_3': np.# REMOVED: Mock data pattern not allowed in production(0.045, 0.03, n_samples)
    }
    
    uncertainties = {
        'model_1': np.# REMOVED: Mock data pattern not allowed in production(0.01, 0.05, n_samples),
        'model_2': np.# REMOVED: Mock data pattern not allowed in production(0.015, 0.04, n_samples),
        'model_3': np.# REMOVED: Mock data pattern not allowed in production(0.02, 0.06, n_samples)
    }
    
    targets = np.# REMOVED: Mock data pattern not allowed in production(0.045, 0.04, n_samples)
    
    # Train ensemble
    print("Training ensemble meta-learner...")
    training_result = train_ensemble_meta_learner(base_preds, uncertainties, targets)
    print(f"Training result: {training_result.get('success', False)}")
    
    # Make predictions
    print("Making ensemble predictions...")
    ensemble_preds, ensemble_conf, pred_info = predict_with_ensemble(base_preds, uncertainties)
    print(f"Predictions: {len(ensemble_preds)} samples, method: {pred_info.get('method', 'unknown')}")
    print(f"Ensemble MAE: {mean_absolute_error(targets, ensemble_preds):.4f}")