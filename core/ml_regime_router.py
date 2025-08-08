#!/usr/bin/env python3
"""
ML Regime-Aware Model Router
Routes predictions to regime-specific models with adaptive feature weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger
from core.ml_uncertainty_engine import get_uncertainty_engine, RegimePrediction

class ModelType(str, Enum):
    """Available model types for each regime"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"
    XGBOOST = "xgboost"

@dataclass
class RegimeModel:
    """Model configuration for specific market regime"""
    regime: str
    model_type: ModelType
    feature_weights: Dict[str, float]
    model_instance: Any = None
    last_trained: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    is_active: bool = True

@dataclass
class FeatureImportance:
    """Feature importance scores per regime"""
    feature_name: str
    regime_scores: Dict[str, float]  # regime -> importance score
    overall_importance: float
    stability_score: float  # How stable across regimes
    
class RegimeFeatureSelector:
    """Regime-aware feature selection and weighting"""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Feature categories and their regime affinities
        self.feature_categories = {
            'momentum': {
                'bull': 0.9,   # High importance in bull markets
                'bear': 0.6,   # Medium importance in bear markets  
                'sideways': 0.3,  # Low importance in sideways
                'volatile': 0.7   # Medium-high in volatile
            },
            'volatility': {
                'bull': 0.4,
                'bear': 0.8,
                'sideways': 0.2,
                'volatile': 0.9
            },
            'volume': {
                'bull': 0.7,
                'bear': 0.8,
                'sideways': 0.4,
                'volatile': 0.9
            },
            'technical': {
                'bull': 0.8,
                'bear': 0.7,
                'sideways': 0.9,
                'volatile': 0.5
            },
            'sentiment': {
                'bull': 0.6,
                'bear': 0.9,
                'sideways': 0.5,
                'volatile': 0.8
            }
        }
        
        self.feature_importance_history = []
        
    def select_regime_features(self, all_features: List[str], regime: str) -> Dict[str, float]:
        """Select and weight features based on market regime"""
        
        feature_weights = {}
        
        for feature in all_features:
            # Determine feature category
            category = self._categorize_feature(feature)
            
            # Get base weight for this regime
            base_weight = self.feature_categories.get(category, {}).get(regime, 0.5)
            
            # Apply feature-specific adjustments
            adjusted_weight = self._adjust_feature_weight(feature, regime, base_weight)
            
            feature_weights[feature] = adjusted_weight
        
        # Normalize weights
        total_weight = sum(feature_weights.values())
        if total_weight > 0:
            feature_weights = {k: v/total_weight for k, v in feature_weights.items()}
        
        self.logger.info(
            f"Selected features for {regime} regime",
            extra={
                'regime': regime,
                'total_features': len(feature_weights),
                'avg_weight': np.mean(list(feature_weights.values())),
                'top_features': sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        )
        
        return feature_weights
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature based on name pattern"""
        feature_lower = feature_name.lower()
        
        if any(term in feature_lower for term in ['rsi', 'macd', 'ma_', 'sma', 'ema']):
            return 'technical'
        elif any(term in feature_lower for term in ['vol', 'std', 'var']):
            return 'volatility'
        elif any(term in feature_lower for term in ['volume', 'trade_count']):
            return 'volume'
        elif any(term in feature_lower for term in ['return', 'momentum', 'trend']):
            return 'momentum'
        elif any(term in feature_lower for term in ['sentiment', 'fear', 'greed']):
            return 'sentiment'
        else:
            return 'technical'  # Default category
    
    def _adjust_feature_weight(self, feature: str, regime: str, base_weight: float) -> float:
        """Apply feature-specific weight adjustments"""
        
        # Regime-specific feature adjustments
        adjustments = {
            'bull': {
                'momentum': 1.2,
                'volume': 1.1,
                'volatility': 0.8
            },
            'bear': {
                'volatility': 1.3,
                'sentiment': 1.2,
                'momentum': 0.7
            },
            'sideways': {
                'technical': 1.3,
                'momentum': 0.6,
                'volatility': 0.7
            },
            'volatile': {
                'volatility': 1.4,
                'volume': 1.2,
                'technical': 0.8
            }
        }
        
        category = self._categorize_feature(feature)
        adjustment = adjustments.get(regime, {}).get(category, 1.0)
        
        return base_weight * adjustment

class MLRegimeRouter:
    """Main regime-aware model router"""
    
    def __init__(self):
        self.logger = get_logger()
        self.uncertainty_engine = get_uncertainty_engine()
        self.feature_selector = RegimeFeatureSelector()
        
        # Regime-specific models
        self.regime_models: Dict[str, RegimeModel] = {}
        
        # Performance tracking
        self.routing_performance = {}
        self.regime_transition_buffer = []
        
        # Configuration
        self.min_regime_confidence = 0.3
        self.regime_transition_smoothing = 3  # Number of periods to smooth transitions
        self.performance_window = 100  # Sample window for performance evaluation
        
        self._initialize_regime_models()
    
    def _initialize_regime_models(self):
        """Initialize regime-specific models"""
        
        regime_configs = {
            'bull': {
                'model_type': ModelType.LSTM,
                'features': ['momentum', 'volume', 'technical'],
                'description': 'Optimized for trending upward markets'
            },
            'bear': {
                'model_type': ModelType.ENSEMBLE,
                'features': ['volatility', 'sentiment', 'volume'],
                'description': 'Robust for declining markets with high uncertainty'
            },
            'sideways': {
                'model_type': ModelType.TRANSFORMER,
                'features': ['technical', 'momentum'],
                'description': 'Pattern recognition for range-bound markets'
            },
            'volatile': {
                'model_type': ModelType.XGBOOST,
                'features': ['volatility', 'volume', 'sentiment'],
                'description': 'Adaptive for high-volatility periods'
            }
        }
        
        for regime, config in regime_configs.items():
            self.regime_models[regime] = RegimeModel(
                regime=regime,
                model_type=config['model_type'],
                feature_weights={},
                performance_metrics={
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            )
        
        self.logger.info(
            "Regime models initialized",
            extra={'regimes': list(self.regime_models.keys())}
        )
    
    def route_prediction(
        self, 
        features: np.ndarray, 
        market_data: Dict[str, Any],
        horizon: str = '24h'
    ) -> Tuple[float, Dict[str, Any]]:
        """Route prediction to appropriate regime model"""
        
        try:
            # Detect current regime with uncertainty
            prediction, uncertainty_metrics = self.uncertainty_engine.predict_with_uncertainty(
                features, market_data, horizon
            )
            
            # Get regime from market data analysis
            regime_prediction = self.uncertainty_engine.regime_detector.detect_regime(market_data)
            current_regime = regime_prediction.regime
            regime_confidence = regime_prediction.regime_confidence
            
            # Check if regime confidence is sufficient
            if regime_confidence < self.min_regime_confidence:
                # Use ensemble of all regime models
                return self._ensemble_prediction(features, market_data, horizon)
            
            # Get regime-specific model
            regime_model = self.regime_models.get(current_regime)
            if not regime_model or not regime_model.is_active:
                # Fallback to best performing model
                return self._fallback_prediction(features, market_data, horizon)
            
            # Select regime-appropriate features
            feature_names = self._get_feature_names(features)
            regime_weights = self.feature_selector.select_regime_features(
                feature_names, current_regime
            )
            
            # Apply feature weighting
            weighted_features = self._apply_feature_weights(features, regime_weights)
            
            # Generate regime-specific prediction
            regime_prediction_value = self._predict_with_regime_model(
                regime_model, weighted_features, market_data
            )
            
            # Apply regime transition smoothing
            smoothed_prediction = self._apply_regime_smoothing(
                regime_prediction_value, current_regime, regime_confidence
            )
            
            # Track routing decision
            routing_info = {
                'selected_regime': current_regime,
                'regime_confidence': regime_confidence,
                'model_type': regime_model.model_type.value,
                'feature_weights': regime_weights,
                'raw_prediction': regime_prediction_value,
                'smoothed_prediction': smoothed_prediction,
                'uncertainty_metrics': uncertainty_metrics,
                'routing_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Regime routing for {horizon}",
                extra={
                    'horizon': horizon,
                    'regime': current_regime,
                    'confidence': regime_confidence,
                    'model_type': regime_model.model_type.value,
                    'prediction': smoothed_prediction
                }
            )
            
            return smoothed_prediction, routing_info
            
        except Exception as e:
            self.logger.error(
                f"Regime routing failed: {e}",
                extra={'horizon': horizon, 'error': str(e)}
            )
            return 0.0, {'error': str(e), 'fallback_used': True}
    
    def _ensemble_prediction(
        self, 
        features: np.ndarray, 
        market_data: Dict[str, Any], 
        horizon: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Generate ensemble prediction when regime is uncertain"""
        
        predictions = []
        regime_weights = []
        
        for regime, model in self.regime_models.items():
            if model.is_active:
                try:
                    # Get regime-specific features
                    feature_names = self._get_feature_names(features)
                    weights = self.feature_selector.select_regime_features(feature_names, regime)
                    weighted_features = self._apply_feature_weights(features, weights)
                    
                    # Get prediction from regime model
                    pred = self._predict_with_regime_model(model, weighted_features, market_data)
                    predictions.append(pred)
                    
                    # Weight by historical performance
                    performance_weight = model.performance_metrics.get('accuracy', 0.5)
                    regime_weights.append(performance_weight)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get prediction from {regime} model: {e}")
        
        if predictions:
            # Weighted average of regime predictions
            total_weight = sum(regime_weights) if regime_weights else len(predictions)
            if total_weight > 0:
                ensemble_prediction = sum(p * w for p, w in zip(predictions, regime_weights)) / total_weight
            else:
                ensemble_prediction = np.mean(predictions)
        else:
            ensemble_prediction = 0.0
        
        routing_info = {
            'selected_regime': 'ensemble',
            'regime_confidence': 0.0,
            'ensemble_predictions': predictions,
            'ensemble_weights': regime_weights,
            'final_prediction': ensemble_prediction
        }
        
        return ensemble_prediction, routing_info
    
    def _fallback_prediction(
        self, 
        features: np.ndarray, 
        market_data: Dict[str, Any], 
        horizon: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Fallback prediction when regime routing fails"""
        
        # Find best performing model
        best_model = None
        best_performance = 0.0
        
        for model in self.regime_models.values():
            if model.is_active:
                accuracy = model.performance_metrics.get('accuracy', 0.0)
                if accuracy > best_performance:
                    best_performance = accuracy
                    best_model = model
        
        if best_model:
            try:
                feature_names = self._get_feature_names(features)
                weights = self.feature_selector.select_regime_features(
                    feature_names, best_model.regime
                )
                weighted_features = self._apply_feature_weights(features, weights)
                
                prediction = self._predict_with_regime_model(
                    best_model, weighted_features, market_data
                )
                
                routing_info = {
                    'selected_regime': 'fallback',
                    'fallback_model': best_model.regime,
                    'prediction': prediction
                }
                
                return prediction, routing_info
                
            except Exception as e:
                self.logger.error(f"Fallback prediction failed: {e}")
        
        # Ultimate fallback
        return 0.0, {'selected_regime': 'none', 'error': 'All models failed'}
    
    def _predict_with_regime_model(
        self, 
        regime_model: RegimeModel, 
        features: np.ndarray, 
        market_data: Dict[str, Any]
    ) -> float:
        """Generate prediction using regime-specific model"""
        
        if regime_model.model_instance is None:
            # Model not trained yet - return neutral prediction
            return 0.0
        
        try:
            # Different prediction methods based on model type
            if regime_model.model_type == ModelType.LSTM:
                # LSTM prediction (placeholder)
                prediction = float(np.mean(features[-5:]) * 0.1)  # Simple momentum
                
            elif regime_model.model_type == ModelType.ENSEMBLE:
                # Ensemble prediction with volatility adjustment
                base_pred = float(np.mean(features[-3:]) * 0.05)
                volatility_adj = np.std(features[-10:]) * 0.1
                prediction = base_pred - volatility_adj  # Conservative in volatile times
                
            elif regime_model.model_type == ModelType.TRANSFORMER:
                # Transformer prediction focusing on patterns
                prediction = float(np.median(features[-7:]) * 0.08)  # Pattern-based
                
            elif regime_model.model_type == ModelType.XGBOOST:
                # XGBoost prediction with feature importance
                weighted_sum = np.sum(features * list(regime_model.feature_weights.values())[:len(features)])
                prediction = float(weighted_sum * 0.02)
                
            else:
                # Linear model fallback
                prediction = float(np.mean(features) * 0.05)
            
            return prediction
            
        except Exception as e:
            self.logger.warning(
                f"Prediction failed for {regime_model.regime} model: {e}"
            )
            return 0.0
    
    def _apply_feature_weights(self, features: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Apply regime-specific feature weights"""
        
        if len(weights) == 0:
            return features
        
        # Create weight array matching feature dimensions
        weight_values = list(weights.values())[:len(features)]
        weight_array = np.array(weight_values + [1.0] * (len(features) - len(weight_values)))
        
        return features * weight_array
    
    def _apply_regime_smoothing(
        self, 
        prediction: float, 
        current_regime: str, 
        confidence: float
    ) -> float:
        """Apply smoothing during regime transitions"""
        
        # Add to transition buffer
        self.regime_transition_buffer.append({
            'regime': current_regime,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # Keep buffer size manageable
        if len(self.regime_transition_buffer) > self.regime_transition_smoothing * 2:
            self.regime_transition_buffer = self.regime_transition_buffer[-self.regime_transition_smoothing:]
        
        # If we have enough history, apply smoothing
        if len(self.regime_transition_buffer) >= self.regime_transition_smoothing:
            recent_predictions = [item['prediction'] for item in self.regime_transition_buffer[-self.regime_transition_smoothing:]]
            recent_confidences = [item['confidence'] for item in self.regime_transition_buffer[-self.regime_transition_smoothing:]]
            
            # Weighted average based on confidence
            if sum(recent_confidences) > 0:
                weighted_pred = sum(p * c for p, c in zip(recent_predictions, recent_confidences)) / sum(recent_confidences)
                
                # Blend with current prediction based on confidence
                smoothed = prediction * confidence + weighted_pred * (1 - confidence)
                return smoothed
        
        return prediction
    
    def _get_feature_names(self, features: np.ndarray) -> List[str]:
        """Generate feature names (placeholder implementation)"""
        # In real implementation, this would come from feature engineering pipeline
        feature_names = []
        for i in range(len(features)):
            if i % 5 == 0:
                feature_names.append(f'momentum_{i}')
            elif i % 5 == 1:
                feature_names.append(f'volatility_{i}')
            elif i % 5 == 2:
                feature_names.append(f'volume_{i}')
            elif i % 5 == 3:
                feature_names.append(f'technical_{i}')
            else:
                feature_names.append(f'sentiment_{i}')
        
        return feature_names
    
    def update_regime_model_performance(
        self, 
        regime: str, 
        predictions: List[float], 
        actuals: List[float]
    ):
        """Update performance metrics for regime-specific model"""
        
        if regime not in self.regime_models or len(predictions) != len(actuals):
            return
        
        try:
            # Calculate performance metrics
            accuracy = 1.0 - np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            
            # Calculate directional accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            directional_accuracy = np.mean(pred_direction == actual_direction)
            
            # Update model performance
            self.regime_models[regime].performance_metrics.update({
                'accuracy': max(0.0, accuracy),
                'directional_accuracy': directional_accuracy,
                'rmse': np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2)),
                'last_updated': datetime.now().isoformat()
            })
            
            self.logger.info(
                f"Updated performance for {regime} model",
                extra={
                    'regime': regime,
                    'accuracy': accuracy,
                    'directional_accuracy': directional_accuracy,
                    'samples': len(predictions)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update performance for {regime}: {e}")
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """Get summary of regime routing performance"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'active_regimes': [regime for regime, model in self.regime_models.items() if model.is_active],
            'regime_performance': {},
            'recent_routing_history': self.regime_transition_buffer[-10:] if self.regime_transition_buffer else []
        }
        
        for regime, model in self.regime_models.items():
            summary['regime_performance'][regime] = {
                'model_type': model.model_type.value,
                'is_active': model.is_active,
                'performance_metrics': model.performance_metrics,
                'last_trained': model.last_trained.isoformat() if model.last_trained else None
            }
        
        return summary

# Global instance
_regime_router = None

def get_regime_router() -> MLRegimeRouter:
    """Get global regime router instance"""
    global _regime_router
    if _regime_router is None:
        _regime_router = MLRegimeRouter()
    return _regime_router