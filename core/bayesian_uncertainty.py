"""
Bayesian Uncertainty Modeling
Advanced uncertainty quantification for better probability filtering
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

class BayesianUncertaintyModel:
    """
    Bayesian uncertainty modeling for cryptocurrency predictions
    Provides confidence intervals and uncertainty quantification
    """
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Bayesian models for different horizons
        self.uncertainty_models = {
            '1h': None,
            '24h': None,
            '7d': None,
            '30d': None
        }
        
        # Uncertainty parameters
        self.config = {
            'confidence_levels': [0.68, 0.95, 0.99],  # 1σ, 2σ, 3σ
            'ensemble_size': 50,
            'prior_alpha': 1.0,
            'prior_beta': 1.0,
            'uncertainty_threshold': 0.2,
            'min_samples_for_training': 100
        }
        
        # Historical uncertainty tracking
        self.uncertainty_history = []
        self.calibration_data = {}
        
        self.logger.info("Bayesian Uncertainty Model initialized")
    
    def initialize_uncertainty_models(self) -> bool:
        """Initialize Bayesian uncertainty models"""
        
        try:
            if GP_AVAILABLE:
                # Gaussian Process models for uncertainty
                for horizon in self.uncertainty_models.keys():
                    kernel = (
                        1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
                    )
                    
                    self.uncertainty_models[horizon] = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        normalize_y=True,
                        n_restarts_optimizer=10,
                        random_state=42
                    )
                
                self.logger.info("Gaussian Process uncertainty models initialized")
            else:
                # Fallback to ensemble-based uncertainty
                for horizon in self.uncertainty_models.keys():
                    self.uncertainty_models[horizon] = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        bootstrap=True
                    )
                
                self.logger.warning("Using ensemble-based uncertainty (GP not available)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Uncertainty model initialization failed: {e}")
            return False
    
    def estimate_prediction_uncertainty(self, 
                                      features: np.ndarray,
                                      predictions: Dict[str, float],
                                      coin: str) -> Dict[str, Any]:
        """Estimate uncertainty for multi-horizon predictions"""
        
        try:
            uncertainty_results = {
                'coin': coin,
                'timestamp': datetime.now().isoformat(),
                'horizons': {}
            }
            
            for horizon, prediction in predictions.items():
                if horizon in self.uncertainty_models and self.uncertainty_models[horizon] is not None:
                    # Get uncertainty estimate
                    uncertainty = self._estimate_single_horizon_uncertainty(
                        features, prediction, horizon
                    )
                    
                    # Calculate confidence intervals
                    confidence_intervals = self._calculate_confidence_intervals(
                        prediction, uncertainty
                    )
                    
                    # Bayesian probability assessment
                    bayesian_confidence = self._calculate_bayesian_confidence(
                        prediction, uncertainty, horizon
                    )
                    
                    uncertainty_results['horizons'][horizon] = {
                        'prediction': prediction,
                        'uncertainty': uncertainty,
                        'confidence_intervals': confidence_intervals,
                        'bayesian_confidence': bayesian_confidence,
                        'reliability_score': self._calculate_reliability_score(uncertainty),
                        'uncertainty_category': self._categorize_uncertainty(uncertainty)
                    }
            
            # Calculate overall uncertainty metrics
            overall_uncertainty = self._calculate_overall_uncertainty(uncertainty_results)
            uncertainty_results['overall_uncertainty'] = overall_uncertainty
            
            # Store for calibration
            self.uncertainty_history.append(uncertainty_results)
            
            return uncertainty_results
            
        except Exception as e:
            self.logger.error(f"Uncertainty estimation failed for {coin}: {e}")
            return {}
    
    def _estimate_single_horizon_uncertainty(self, 
                                           features: np.ndarray,
                                           prediction: float,
                                           horizon: str) -> float:
        """Estimate uncertainty for single horizon"""
        
        try:
            model = self.uncertainty_models[horizon]
            
            if GP_AVAILABLE and hasattr(model, 'predict'):
                # Gaussian Process uncertainty
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                try:
                    # Get mean and standard deviation
                    mean, std = model.predict(features, return_std=True)
                    
                    # Return standard deviation as uncertainty
                    return float(std[0])
                    
                except Exception:
                    # Model not trained yet, use heuristic
                    return self._heuristic_uncertainty(prediction, horizon)
            
            else:
                # Ensemble-based uncertainty using bootstrap
                return self._bootstrap_uncertainty(features, prediction, horizon)
        
        except Exception as e:
            self.logger.error(f"Single horizon uncertainty estimation failed: {e}")
            return self._heuristic_uncertainty(prediction, horizon)
    
    def _heuristic_uncertainty(self, prediction: float, horizon: str) -> float:
        """Heuristic uncertainty based on prediction magnitude and horizon"""
        
        # Base uncertainty increases with prediction magnitude and time horizon
        base_uncertainty = abs(prediction) * 0.1  # 10% of prediction magnitude
        
        # Horizon multipliers
        horizon_multipliers = {
            '1h': 1.0,
            '24h': 1.5,
            '7d': 2.0,
            '30d': 3.0
        }
        
        horizon_multiplier = horizon_multipliers.get(horizon, 2.0)
        
        # Market volatility factor (can be improved with actual market data)
        volatility_factor = 1.2
        
        uncertainty = base_uncertainty * horizon_multiplier * volatility_factor
        
        # Cap uncertainty at reasonable levels
        return min(uncertainty, abs(prediction) * 0.5)
    
    def _bootstrap_uncertainty(self, 
                             features: np.ndarray,
                             prediction: float,
                             horizon: str) -> float:
        """Bootstrap-based uncertainty estimation"""
        
        try:
            model = self.uncertainty_models[horizon]
            
            if hasattr(model, 'estimators_'):
                # Use individual trees/estimators for uncertainty
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                # Get predictions from individual estimators
                individual_predictions = []
                
                for estimator in model.estimators_[:min(20, len(model.estimators_))]:
                    try:
                        pred = estimator.predict(features)[0]
                        individual_predictions.append(pred)
                    except Exception:
                        continue
                
                if individual_predictions:
                    # Calculate standard deviation across predictions
                    uncertainty = float(np.std(individual_predictions))
                    return uncertainty
            
            # Fallback to heuristic
            return self._heuristic_uncertainty(prediction, horizon)
            
        except Exception as e:
            self.logger.error(f"Bootstrap uncertainty estimation failed: {e}")
            return self._heuristic_uncertainty(prediction, horizon)
    
    def _calculate_confidence_intervals(self, 
                                      prediction: float,
                                      uncertainty: float) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals at different levels"""
        
        confidence_intervals = {}
        
        for confidence_level in self.config['confidence_levels']:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf(0.5 + confidence_level / 2)
            
            # Calculate interval bounds
            lower_bound = prediction - z_score * uncertainty
            upper_bound = prediction + z_score * uncertainty
            
            confidence_intervals[f'{int(confidence_level * 100)}%'] = {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'width': float(upper_bound - lower_bound)
            }
        
        return confidence_intervals
    
    def _calculate_bayesian_confidence(self, 
                                     prediction: float,
                                     uncertainty: float,
                                     horizon: str) -> float:
        """Calculate Bayesian confidence score"""
        
        try:
            # Get historical performance for this horizon
            historical_accuracy = self._get_historical_accuracy(horizon)
            
            # Prior belief (based on historical performance)
            prior_confidence = historical_accuracy
            
            # Likelihood based on uncertainty
            # Lower uncertainty = higher likelihood of accuracy
            likelihood = 1.0 / (1.0 + uncertainty)
            
            # Bayesian update
            posterior_confidence = self._bayesian_update(
                prior_confidence, likelihood, uncertainty
            )
            
            # Normalize to [0, 1]
            bayesian_confidence = max(0.0, min(1.0, posterior_confidence))
            
            return float(bayesian_confidence)
            
        except Exception as e:
            self.logger.error(f"Bayesian confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _bayesian_update(self, 
                        prior: float,
                        likelihood: float,
                        uncertainty: float) -> float:
        """Perform Bayesian update of confidence"""
        
        # Beta distribution parameters
        alpha = self.config['prior_alpha']
        beta = self.config['prior_beta']
        
        # Update parameters based on evidence
        # High likelihood and low uncertainty increase alpha
        evidence_strength = likelihood / (1.0 + uncertainty)
        
        updated_alpha = alpha + evidence_strength * prior
        updated_beta = beta + evidence_strength * (1 - prior)
        
        # Calculate posterior mean
        posterior_mean = updated_alpha / (updated_alpha + updated_beta)
        
        return posterior_mean
    
    def _get_historical_accuracy(self, horizon: str) -> float:
        """Get historical accuracy for specific horizon"""
        
        # Get from calibration data if available
        if horizon in self.calibration_data:
            return self.calibration_data[horizon].get('accuracy', 0.7)
        
        # Default based on horizon (shorter horizons more accurate)
        default_accuracy = {
            '1h': 0.75,
            '24h': 0.70,
            '7d': 0.65,
            '30d': 0.60
        }
        
        return default_accuracy.get(horizon, 0.65)
    
    def _calculate_reliability_score(self, uncertainty: float) -> float:
        """Calculate reliability score based on uncertainty"""
        
        # Higher uncertainty = lower reliability
        # Use inverse sigmoid function
        reliability = 1.0 / (1.0 + np.exp(5 * (uncertainty - self.config['uncertainty_threshold'])))
        
        return float(reliability)
    
    def _categorize_uncertainty(self, uncertainty: float) -> str:
        """Categorize uncertainty level"""
        
        if uncertainty < 0.05:
            return 'very_low'
        elif uncertainty < 0.1:
            return 'low'
        elif uncertainty < 0.2:
            return 'moderate'
        elif uncertainty < 0.4:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_overall_uncertainty(self, uncertainty_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall uncertainty metrics across horizons"""
        
        try:
            horizons_data = uncertainty_results.get('horizons', {})
            
            if not horizons_data:
                return {'error': 'No horizon data available'}
            
            # Collect uncertainty values
            uncertainties = [
                data['uncertainty'] for data in horizons_data.values()
                if 'uncertainty' in data
            ]
            
            # Collect confidence scores
            confidences = [
                data['bayesian_confidence'] for data in horizons_data.values()
                if 'bayesian_confidence' in data
            ]
            
            if not uncertainties or not confidences:
                return {'error': 'Insufficient data for overall calculation'}
            
            overall_metrics = {
                'mean_uncertainty': float(np.mean(uncertainties)),
                'max_uncertainty': float(np.max(uncertainties)),
                'min_uncertainty': float(np.min(uncertainties)),
                'uncertainty_variance': float(np.var(uncertainties)),
                'mean_confidence': float(np.mean(confidences)),
                'confidence_consistency': float(1.0 - np.std(confidences)),
                'prediction_quality': self._assess_prediction_quality(uncertainties, confidences),
                'recommendation': self._generate_uncertainty_recommendation(uncertainties, confidences)
            }
            
            return overall_metrics
            
        except Exception as e:
            self.logger.error(f"Overall uncertainty calculation failed: {e}")
            return {'error': str(e)}
    
    def _assess_prediction_quality(self, 
                                 uncertainties: List[float],
                                 confidences: List[float]) -> str:
        """Assess overall prediction quality"""
        
        mean_uncertainty = np.mean(uncertainties)
        mean_confidence = np.mean(confidences)
        
        if mean_uncertainty < 0.1 and mean_confidence > 0.8:
            return 'excellent'
        elif mean_uncertainty < 0.2 and mean_confidence > 0.7:
            return 'good'
        elif mean_uncertainty < 0.3 and mean_confidence > 0.6:
            return 'moderate'
        elif mean_uncertainty < 0.5 and mean_confidence > 0.4:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_uncertainty_recommendation(self, 
                                          uncertainties: List[float],
                                          confidences: List[float]) -> str:
        """Generate recommendation based on uncertainty analysis"""
        
        mean_uncertainty = np.mean(uncertainties)
        mean_confidence = np.mean(confidences)
        uncertainty_variance = np.var(uncertainties)
        
        if mean_confidence > 0.8 and mean_uncertainty < 0.15:
            return 'high_confidence_trade'
        elif mean_confidence > 0.7 and mean_uncertainty < 0.25:
            return 'moderate_confidence_trade'
        elif uncertainty_variance > 0.1:
            return 'inconsistent_predictions_avoid'
        elif mean_uncertainty > 0.4:
            return 'high_uncertainty_avoid'
        else:
            return 'low_confidence_monitor'
    
    def train_uncertainty_model(self, 
                              training_data: List[Dict[str, Any]],
                              horizon: str) -> Dict[str, Any]:
        """Train uncertainty model for specific horizon"""
        
        try:
            if len(training_data) < self.config['min_samples_for_training']:
                return {
                    'success': False,
                    'reason': f'Insufficient training data: {len(training_data)} < {self.config["min_samples_for_training"]}'
                }
            
            # Prepare training data
            X, y = self._prepare_uncertainty_training_data(training_data, horizon)
            
            if len(X) == 0:
                return {'success': False, 'reason': 'No valid training samples'}
            
            # Train model
            model = self.uncertainty_models[horizon]
            
            if model is None:
                return {'success': False, 'reason': 'Model not initialized'}
            
            # Fit model
            model.fit(X, y)
            
            # Evaluate model performance
            performance = self._evaluate_uncertainty_model(model, X, y, horizon)
            
            self.logger.info(f"Uncertainty model trained for {horizon}: {performance}")
            
            return {
                'success': True,
                'horizon': horizon,
                'training_samples': len(X),
                'performance': performance,
                'trained_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Uncertainty model training failed for {horizon}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_uncertainty_training_data(self, 
                                         training_data: List[Dict[str, Any]],
                                         horizon: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for uncertainty model"""
        
        X_list = []
        y_list = []
        
        for item in training_data:
            # Extract features
            features = self._extract_uncertainty_features(item)
            
            # Extract target (actual vs predicted error)
            if horizon in item.get('prediction_errors', {}):
                prediction_error = item['prediction_errors'][horizon]
                
                X_list.append(features)
                y_list.append(abs(prediction_error))  # Use absolute error as uncertainty target
        
        if X_list:
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Remove any invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y
        
        return np.array([]), np.array([])
    
    def _extract_uncertainty_features(self, item: Dict[str, Any]) -> List[float]:
        """Extract features for uncertainty modeling"""
        
        features = []
        
        # Price volatility features
        price_history = item.get('price_history', [])
        if len(price_history) >= 10:
            prices = [p.get('close', 0) for p in price_history[-10:]]
            returns = np.diff(prices) / prices[:-1]
            
            features.extend([
                float(np.std(returns)),  # Volatility
                float(np.mean(np.abs(returns))),  # Mean absolute return
                float(np.max(returns) - np.min(returns))  # Return range
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Volume features
        if price_history:
            volumes = [p.get('volume', 0) for p in price_history[-5:]]
            volume_changes = np.diff(volumes) / np.maximum(volumes[:-1], 1)
            
            features.extend([
                float(np.std(volume_changes)),  # Volume volatility
                float(np.mean(volumes))  # Mean volume
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Market features
        market_data = item.get('market_features', {})
        features.extend([
            market_data.get('market_momentum', 0.0),
            market_data.get('market_volatility', 0.0),
            market_data.get('correlation_with_market', 0.0)
        ])
        
        # Ensure fixed feature count
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def _evaluate_uncertainty_model(self, 
                                  model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  horizon: str) -> Dict[str, float]:
        """Evaluate uncertainty model performance"""
        
        try:
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_.tolist()
            
            performance = {
                'cv_rmse': float(cv_rmse),
                'cv_std': float(cv_scores.std()),
                'training_samples': len(X),
                'feature_count': X.shape[1]
            }
            
            if feature_importance:
                performance['feature_importance'] = feature_importance
            
            # Store calibration data
            self.calibration_data[horizon] = {
                'accuracy': max(0.1, 1.0 - cv_rmse),  # Convert RMSE to accuracy
                'last_updated': datetime.now(),
                'training_samples': len(X)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def filter_predictions_by_uncertainty(self, 
                                        predictions: List[Dict[str, Any]],
                                        max_uncertainty: float = 0.3,
                                        min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Filter predictions based on uncertainty thresholds"""
        
        filtered_predictions = []
        
        for prediction in predictions:
            uncertainty_data = prediction.get('uncertainty', {})
            
            if not uncertainty_data:
                continue
            
            # Check overall uncertainty
            overall = uncertainty_data.get('overall_uncertainty', {})
            mean_uncertainty = overall.get('mean_uncertainty', 1.0)
            mean_confidence = overall.get('mean_confidence', 0.0)
            
            # Apply filters
            if (mean_uncertainty <= max_uncertainty and 
                mean_confidence >= min_confidence):
                
                # Add quality score
                quality_score = (1.0 - mean_uncertainty) * mean_confidence
                prediction['quality_score'] = quality_score
                prediction['passed_uncertainty_filter'] = True
                
                filtered_predictions.append(prediction)
        
        # Sort by quality score
        filtered_predictions.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        self.logger.info(
            f"Uncertainty filtering: {len(filtered_predictions)}/{len(predictions)} predictions passed"
        )
        
        return filtered_predictions
    
    def get_uncertainty_status(self) -> Dict[str, Any]:
        """Get current uncertainty modeling status"""
        
        return {
            'models_initialized': sum(1 for model in self.uncertainty_models.values() if model is not None),
            'total_models': len(self.uncertainty_models),
            'gp_available': GP_AVAILABLE,
            'uncertainty_history_size': len(self.uncertainty_history),
            'calibration_data': {
                horizon: {
                    'accuracy': data.get('accuracy', 0),
                    'last_updated': data.get('last_updated', '').isoformat() if isinstance(data.get('last_updated'), datetime) else str(data.get('last_updated', '')),
                    'training_samples': data.get('training_samples', 0)
                }
                for horizon, data in self.calibration_data.items()
            },
            'config': self.config,
            'last_updated': datetime.now().isoformat()
        }