import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
import threading
import time

# Machine Learning libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class MLModelManager:
    """Machine Learning model management and ensemble system"""
    
    def __init__(self, config_manager, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.pipelines = {}
        self.model_metadata = {}
        self._lock = threading.Lock()
        
        # Model paths
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)
        
        # Available model types
        self.available_models = self._get_available_models()
        
        # Ensemble configuration
        self.ensemble_weights = self.config_manager.get("ensemble_weights", {
            "xgboost": 0.4,
            "lightgbm": 0.4,
            "sklearn": 0.2
        })
        
        # Performance tracking
        self.model_performance = {}
        self.training_history = {}
        
        self.logger.info(f"MLModelManager initialized with {len(self.available_models)} available model types")
    
    def _get_available_models(self) -> Dict[str, bool]:
        """Get available ML model types"""
        return {
            'sklearn': SKLEARN_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'linear_regression': SKLEARN_AVAILABLE,
            'random_forest': SKLEARN_AVAILABLE,
            'gradient_boosting': SKLEARN_AVAILABLE,
            'support_vector': SKLEARN_AVAILABLE
        }
    
    def create_feature_pipeline(self, feature_config: Dict[str, Any]) -> Pipeline:
        """Create feature preprocessing pipeline"""
        try:
            steps = []
            
            # Scaling step
            scaler_type = feature_config.get('scaler', 'standard')
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            steps.append(('scaler', scaler))
            
            # Additional preprocessing steps can be added here
            
            return Pipeline(steps)
            
        except Exception as e:
            self.logger.error(f"Error creating feature pipeline: {str(e)}")
            return Pipeline([('scaler', StandardScaler())])
    
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   hyperparams: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a single ML model"""
        try:
            model_key = f"{model_type}_{int(time.time())}"
            
            # Get model instance
            model = self._create_model(model_type, hyperparams)
            if model is None:
                return None
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            train_pred = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred)
            
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred)
            
            # Store model and metadata
            model_info = {
                'model': model,
                'model_type': model_type,
                'hyperparams': hyperparams or {},
                'training_time': training_time,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'trained_at': datetime.now().isoformat(),
                'feature_count': X_train.shape[1],
                'training_samples': X_train.shape[0]
            }
            
            with self._lock:
                self.models[model_key] = model_info
                self.model_metadata[model_key] = {
                    'model_type': model_type,
                    'performance': val_metrics if val_metrics else train_metrics,
                    'trained_at': model_info['trained_at']
                }
            
            self.logger.info(f"Model {model_type} trained successfully in {training_time:.2f}s")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {str(e)}")
            return None
    
    def _create_model(self, model_type: str, hyperparams: Dict[str, Any] = None):
        """Create model instance based on type"""
        hyperparams = hyperparams or {}
        
        try:
            if model_type == 'xgboost' and XGBOOST_AVAILABLE:
                return xgb.XGBRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 6),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    subsample=hyperparams.get('subsample', 0.8),
                    colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                return lgb.LGBMRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 6),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    subsample=hyperparams.get('subsample', 0.8),
                    colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            
            elif model_type == 'random_forest' and SKLEARN_AVAILABLE:
                return RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', None),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == 'gradient_boosting' and SKLEARN_AVAILABLE:
                return GradientBoostingRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 3),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    subsample=hyperparams.get('subsample', 0.8),
                    random_state=42
                )
            
            elif model_type == 'linear_regression' and SKLEARN_AVAILABLE:
                return LinearRegression(n_jobs=-1)
            
            elif model_type == 'ridge' and SKLEARN_AVAILABLE:
                return Ridge(
                    alpha=hyperparams.get('alpha', 1.0),
                    random_state=42
                )
            
            elif model_type == 'support_vector' and SKLEARN_AVAILABLE:
                return SVR(
                    kernel=hyperparams.get('kernel', 'rbf'),
                    C=hyperparams.get('C', 1.0),
                    gamma=hyperparams.get('gamma', 'scale')
            
            else:
                # Fallback to linear regression
                if SKLEARN_AVAILABLE:
                    return LinearRegression()
                else:
                    self.logger.error("No ML libraries available")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error creating {model_type} model: {str(e)}")
            return None
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Root Mean Squared Error
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Mean Absolute Percentage Error
            y_true_nonzero = y_true[y_true != 0]
            y_pred_nonzero = y_pred[y_true != 0]
            if len(y_true_nonzero) > 0:
                metrics['mape'] = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
            else:
                metrics['mape'] = 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'), 'rmse': float('inf'), 'mape': 100}
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None,
                      model_types: List[str] = None) -> Dict[str, Any]:
        """Train ensemble of models"""
        try:
            model_types = model_types or ['xgboost', 'lightgbm', 'random_forest']
            ensemble_models = {}
            ensemble_predictions = {}
            
            # Train individual models
            for model_type in model_types:
                if not self.available_models.get(model_type, False):
                    self.logger.warning(f"Model type {model_type} not available, skipping")
                    continue
                
                model_info = self.train_model(model_type, X_train, y_train, X_val, y_val)
                if model_info:
                    model_key = f"ensemble_{model_type}_{int(time.time())}"
                    ensemble_models[model_key] = model_info
                    
                    # Get predictions for ensemble weighting
                    if X_val is not None:
                        val_pred = model_info['model'].predict(X_val)
                        ensemble_predictions[model_key] = val_pred
            
            if not ensemble_models:
                self.logger.error("No models successfully trained for ensemble")
                return None
            
            # Calculate ensemble weights based on performance
            ensemble_weights = self._calculate_ensemble_weights(ensemble_models)
            
            # Create ensemble prediction
            if X_val is not None and y_val is not None:
                ensemble_pred = self._make_ensemble_prediction(ensemble_predictions, ensemble_weights)
                ensemble_metrics = self._calculate_metrics(y_val, ensemble_pred)
            else:
                ensemble_metrics = {}
            
            ensemble_info = {
                'models': ensemble_models,
                'weights': ensemble_weights,
                'ensemble_metrics': ensemble_metrics,
                'trained_at': datetime.now().isoformat(),
                'model_count': len(ensemble_models)
            }
            
            # Store ensemble
            ensemble_key = f"ensemble_{int(time.time())}"
            with self._lock:
                self.models[ensemble_key] = ensemble_info
            
            self.logger.info(f"Ensemble trained with {len(ensemble_models)} models")
            return ensemble_info
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            return None
    
    def _calculate_ensemble_weights(self, ensemble_models: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        try:
            model_scores = {}
            
            for model_key, model_info in ensemble_models.items():
                # Use validation R2 score if available, otherwise training R2
                metrics = model_info.get('val_metrics') or model_info.get('train_metrics', {})
                r2_score = metrics.get('r2', 0)
                
                # Convert R2 to positive weight (higher R2 = higher weight)
                model_scores[model_key] = max(0, r2_score)
            
            # Normalize weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                weights = {key: score / total_score for key, score in model_scores.items()}
            else:
                # Equal weights if all models perform poorly
                weights = {key: 1.0 / len(model_scores) for key in model_scores.keys()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble weights: {str(e)}")
            return {}
    
    def _make_ensemble_prediction(self, predictions: Dict[str, np.ndarray], 
                                weights: Dict[str, float]) -> np.ndarray:
        """Make weighted ensemble prediction"""
        try:
            if not predictions or not weights:
                return np.array([])
            
            # Get first prediction to determine array size
            first_pred = list(predictions.values())[0]
            ensemble_pred = np.zeros_like(first_pred)
            
            # Weighted combination
            for model_key, pred in predictions.items():
                weight = weights.get(model_key, 0)
                ensemble_pred += weight * pred
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {str(e)}")
            return np.array([])
    
    def predict(self, model_key: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction with a specific model"""
        try:
            with self._lock:
                if model_key not in self.models:
                    self.logger.error(f"Model {model_key} not found")
                    return None
                
                model_info = self.models[model_key]
                
                # Handle ensemble models
                if 'models' in model_info:  # This is an ensemble
                    return self._predict_ensemble(model_info, X)
                else:  # Single model
                    model = model_info['model']
                    return model.predict(X)
                    
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def _predict_ensemble(self, ensemble_info: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        try:
            models = ensemble_info['models']
            weights = ensemble_info['weights']
            
            predictions = {}
            for model_key, model_info in models.items():
                pred = model_info['model'].predict(X)
                predictions[model_key] = pred
            
            return self._make_ensemble_prediction(predictions, weights)
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {str(e)}")
            return np.array([])
    
    def hyperparameter_optimization(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                                  param_grid: Dict[str, List] = None, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.error("Scikit-learn not available for hyperparameter optimization")
                return None
            
            # Default parameter grids
            default_grids = {
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }
            
            param_grid = param_grid or default_grids.get(model_type, {})
            if not param_grid:
                self.logger.warning(f"No parameter grid for {model_type}, using default parameters")
                return self.train_model(model_type, X_train, y_train)
            
            # Create base model
            base_model = self._create_model(model_type)
            if base_model is None:
                return None
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            optimization_time = time.time() - start_time
            
            # Train final model with best parameters
            best_model_info = self.train_model(model_type, X_train, y_train, 
                                             hyperparams=grid_search.best_params_)
            
            if best_model_info:
                best_model_info['optimization_time'] = optimization_time
                best_model_info['best_params'] = grid_search.best_params_
                best_model_info['best_score'] = grid_search.best_score_
                best_model_info['cv_results'] = grid_search.cv_results_
            
            self.logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f}s")
            return best_model_info
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None
    
    def save_model(self, model_key: str, file_path: str = None) -> bool:
        """Save model to disk"""
        try:
            with self._lock:
                if model_key not in self.models:
                    self.logger.error(f"Model {model_key} not found")
                    return False
                
                model_info = self.models[model_key]
                
                # Generate file path if not provided
                if file_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_path = self.models_path / f"{model_key}_{timestamp}.pkl"
                else:
                    file_path = Path(file_path)
                
                # Prepare data for saving (exclude non-serializable objects if needed)
                save_data = {
                    'model': model_info['model'],
                    'model_type': model_info['model_type'],
                    'hyperparams': model_info['hyperparams'],
                    'train_metrics': model_info['train_metrics'],
                    'val_metrics': model_info['val_metrics'],
                    'trained_at': model_info['trained_at'],
                    'feature_count': model_info['feature_count'],
                    'training_samples': model_info['training_samples']
                }
                
                # Save model
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                self.logger.info(f"Model {model_key} saved to {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, file_path: str, model_key: str = None) -> Optional[str]:
        """Load model from disk"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"Model file {file_path} not found")
                return None
            
            # Load model data
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Generate model key if not provided
            if model_key is None:
                model_key = f"loaded_{file_path.stem}_{int(time.time())}"
            
            # Store loaded model
            with self._lock:
                self.models[model_key] = model_data
                self.model_metadata[model_key] = {
                    'model_type': model_data['model_type'],
                    'performance': model_data['val_metrics'] or model_data['train_metrics'],
                    'trained_at': model_data['trained_at'],
                    'loaded_from': str(file_path)
                }
            
            self.logger.info(f"Model loaded from {file_path} as {model_key}")
            return model_key
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        with self._lock:
            if model_key in self.models:
                model_info = self.models[model_key].copy()
                # Remove the actual model object for the return (too large)
                if 'model' in model_info:
                    model_info.pop('model')
                return model_info
            return None
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        with self._lock:
            return self.model_metadata.copy()
    
    def delete_model(self, model_key: str) -> bool:
        """Delete a model from memory"""
        try:
            with self._lock:
                if model_key in self.models:
                    del self.models[model_key]
                if model_key in self.model_metadata:
                    del self.model_metadata[model_key]
            
            self.logger.info(f"Model {model_key} deleted")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {str(e)}")
            return False
    
    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison of all models"""
        with self._lock:
            comparison = {
                'models': {},
                'best_model': None,
                'best_score': float('-inf')
            }
            
            for model_key, metadata in self.model_metadata.items():
                performance = metadata.get('performance', {})
                r2_score = performance.get('r2', float('-inf'))
                
                comparison['models'][model_key] = {
                    'model_type': metadata['model_type'],
                    'r2_score': r2_score,
                    'mse': performance.get('mse', float('inf')),
                    'mae': performance.get('mae', float('inf')),
                    'trained_at': metadata['trained_at']
                }
                
                if r2_score > comparison['best_score']:
                    comparison['best_score'] = r2_score
                    comparison['best_model'] = model_key
            
            return comparison
    
    def cleanup_old_models(self, max_age_hours: int = 168):  # 1 week default
        """Clean up old models from memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            models_to_delete = []
            
            with self._lock:
                for model_key, metadata in self.model_metadata.items():
                    try:
                        trained_at = datetime.fromisoformat(metadata['trained_at'])
                        if trained_at < cutoff_time:
                            models_to_delete.append(model_key)
                    except:
                        # If can't parse date, consider for deletion
                        models_to_delete.append(model_key)
            
            # Delete old models
            deleted_count = 0
            for model_key in models_to_delete:
                if self.delete_model(model_key):
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old models")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up models: {str(e)}")
            return 0
    
    def get_feature_importance(self, model_key: str, feature_names: List[str] = None) -> Optional[Dict[str, float]]:
        """Get feature importance from a model"""
        try:
            with self._lock:
                if model_key not in self.models:
                    return None
                
                model_info = self.models[model_key]
                model = model_info['model']
                
                # Check if model has feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    self.logger.warning(f"Model {model_key} does not support feature importance")
                    return None
                
                # Create feature importance dictionary
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                importance_dict = dict(zip(feature_names, importances))
                
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(), 
                                              key=lambda x: x[1], reverse=True))
                
                return sorted_importance
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get ML system status"""
        with self._lock:
            return {
                'available_libraries': self.available_models,
                'total_models': len(self.models),
                'model_types': list(set(metadata['model_type'] for metadata in self.model_metadata.values())),
                'memory_usage_mb': self._estimate_memory_usage(),
                'ensemble_weights': self.ensemble_weights,
                'models_path': str(self.models_path)
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of stored models"""
        try:
            import sys
            total_size = 0
            
            for model_info in self.models.values():
                total_size += sys.getsizeof(model_info)
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0

# Fast computation functions with Numba if available
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
        """Fast moving average calculation with Numba"""
        n = len(prices)
        ma = np.empty(n)
        ma[:window-1] = np.nan
        
        for i in range(window-1, n):
            ma[i] = np.mean(prices[i-window+1:i+1])
        
        return ma
    
    @jit(nopython=True)
    def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Fast RSI calculation with Numba"""
        n = len(prices)
        rsi = np.empty(n)
        rsi[:window] = np.nan
        
        deltas = np.diff(prices)
        
        for i in range(window, n):
            gains = deltas[i-window:i][deltas[i-window:i] > 0]
            losses = -deltas[i-window:i][deltas[i-window:i] < 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
else:
    # Fallback implementations without Numba
    def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
        """Moving average calculation"""
        return pd.Series(prices).rolling(window=window).mean().values
    
    def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """RSI calculation"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=window).mean()
        avg_losses = pd.Series(losses).rolling(window=window).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[np.nan], rsi.values])

class FeatureEngineer:
    """Feature engineering for cryptocurrency price prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        try:
            df = df.copy()
            
            # Ensure required columns
            if 'price' not in df.columns:
                if 'close' in df.columns:
                    df['price'] = df['close']
                else:
                    self.logger.error("No price column found")
                    return df
            
            prices = df['price'].values
            
            # Moving averages
            for window in [5, 10, 20, 50, 100]:
                df[f'ma_{window}'] = fast_moving_average(prices, window)
                df[f'price_ma_ratio_{window}'] = df['price'] / df[f'ma_{window}']
            
            # Price-based features
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            
            # Volatility features
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
                df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].shift(1)
            
            # Technical indicators
            df['rsi'] = fast_rsi(prices)
            
            # Bollinger Bands
            ma20 = fast_moving_average(prices, 20)
            std20 = pd.Series(prices).rolling(window=20).std().values
            df['bb_upper'] = ma20 + (2 * std20)
            df['bb_lower'] = ma20 - (2 * std20)
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum
            for lag in [1, 3, 5, 10]:
                df[f'price_momentum_{lag}'] = df['price'] / df['price'].shift(lag) - 1
            
            # Volume features if available
            if 'volume' in df.columns:
                for window in [5, 10, 20]:
                    df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                    df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['day_of_month'] = df['timestamp'].dt.day
                df['month'] = df['timestamp'].dt.month
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating technical features: {str(e)}")
            return df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create lagged features"""
        try:
            df = df.copy()
            lags = lags or [1, 2, 3, 5, 10]
            
            # Lagged price features
            for lag in lags:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                
                if 'volume' in df.columns:
                    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'price_mean_{window}'] = df['price'].rolling(window=window).mean()
                df[f'price_std_{window}'] = df['price'].rolling(window=window).std()
                df[f'price_min_{window}'] = df['price'].rolling(window=window).min()
                df[f'price_max_{window}'] = df['price'].rolling(window=window).max()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating lagged features: {str(e)}")
            return df
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[str] = None) -> pd.DataFrame:
        """Create target variables for different prediction horizons"""
        try:
            df = df.copy()
            horizons = horizons or ['1h', '4h', '1d', '7d']
            
            for horizon in horizons:
                # Convert horizon to periods
                if horizon == '1h':
                    periods = 1
                elif horizon == '4h':
                    periods = 4
                elif horizon == '1d':
                    periods = 24
                elif horizon == '7d':
                    periods = 168
                else:
                    periods = 1
                
                # Price target
                df[f'target_price_{horizon}'] = df['price'].shift(-periods)
                
                # Return target
                df[f'target_return_{horizon}'] = (df[f'target_price_{horizon}'] / df['price'] - 1)
                
                # Direction target (binary)
                df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating target variables: {str(e)}")
            return df
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = 'correlation', top_k: int = 50) -> List[str]:
        """Select top features based on different methods"""
        try:
            # Get feature columns (exclude target and non-feature columns)
            exclude_cols = ['timestamp', 'symbol', target_column] + \
                          [col for col in df.columns if col.startswith('target_')]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if method == 'correlation':
                # Select features by correlation with target
                correlations = df[feature_cols + [target_column]].corr()[target_column].abs()
                correlations = correlations.drop(target_column).sort_values(ascending=False)
                selected_features = correlations.head(top_k).index.tolist()
            
            else:
                # Default to all features if method not recognized
                selected_features = feature_cols[:top_k]
            
            self.logger.info(f"Selected {len(selected_features)} features using {method} method")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return []
    
    def prepare_model_data(self, df: pd.DataFrame, target_column: str, 
                          feature_columns: List[str] = None, 
                          test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        try:
            # Clean data
            df_clean = df.dropna()
            
            if len(df_clean) == 0:
                self.logger.error("No clean data available after dropping NaN values")
                return None, None, None, None
            
            # Select features
            if feature_columns is None:
                feature_columns = self.select_features(df_clean, target_column)
            
            # Prepare features and target
            X = df_clean[feature_columns].values
            y = df_clean[target_column].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            self.logger.info(f"Prepared data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing model data: {str(e)}")
            return None, None, None, None
