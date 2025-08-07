"""
CryptoSmartTrader V2 - AutoML Engine
Automated machine learning met model selection en hyperparameter tuning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some ML libraries not available: {e}")
    ML_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - AutoML optimization disabled")

# Disable AutoML if required dependencies are missing
if ML_AVAILABLE and not OPTUNA_AVAILABLE:
    ML_AVAILABLE = False

@dataclass 
class ModelCandidate:
    """Model candidate for AutoML"""
    name: str
    model_class: Any
    param_space: Dict[str, Any]
    gpu_enabled: bool = False
    default_params: Dict[str, Any] = None

@dataclass
class AutoMLResult:
    """AutoML experiment result"""
    best_model: Any
    best_params: Dict[str, Any]
    best_score: float
    model_name: str
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    training_time: float
    timestamp: datetime

class AutoMLEngine:
    """Automated machine learning engine met model selection en hyperparameter tuning"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
        # GPU detection
        self.gpu_available = self._check_gpu_availability()
        self.logger.info(f"AutoML Engine initialized. GPU available: {self.gpu_available}")
        
        # Model candidates
        self.model_candidates = self._initialize_model_candidates()
        
        # AutoML state
        self.current_experiments = {}
        self.experiment_history = {}
        self.best_models = {}
        
        # Feature selection methods
        self.feature_selectors = {
            'k_best': SelectKBest(f_regression),
            'rfe_rf': RFE(RandomForestRegressor(n_estimators=50, random_state=42)),
            'none': None
        }
        
        # Scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'none': None
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check GPU availability for ML"""
        try:
            # Check for CUDA/GPU support in LightGBM and XGBoost
            import subprocess
            
            # Test LightGBM GPU
            try:
                import lightgbm as lgb
                # Try to create a GPU dataset
                lgb_data = lgb.Dataset(np.random.random((100, 10)), label=np.random.random(100))
                lgb.train({'device': 'gpu', 'objective': 'regression'}, lgb_data, num_boost_round=1, verbose=-1)
                return True
            except:
                pass
            
            # Test XGBoost GPU
            try:
                import xgboost as xgb
                dtrain = xgb.DMatrix(np.random.random((100, 10)), label=np.random.random(100))
                xgb.train({'tree_method': 'gpu_hist', 'objective': 'reg:squarederror'}, dtrain, num_boost_round=1, verbose_eval=False)
                return True
            except:
                pass
            
            return False
            
        except Exception as e:
            self.logger.warning(f"GPU availability check failed: {e}")
            return False
    
    def _initialize_model_candidates(self) -> List[ModelCandidate]:
        """Initialize model candidates for AutoML"""
        candidates = []
        
        if not ML_AVAILABLE:
            return candidates
        
        # LightGBM
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42
        }
        
        if self.gpu_available:
            lgb_params['device'] = 'gpu'
            lgb_params['gpu_platform_id'] = 0
            lgb_params['gpu_device_id'] = 0
        
        candidates.append(ModelCandidate(
            name='lightgbm',
            model_class=lgb.LGBMRegressor,
            param_space={
                'n_estimators': [100, 200, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10, -1],
                'num_leaves': [31, 63, 127, 255],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            },
            gpu_enabled=self.gpu_available,
            default_params=lgb_params
        ))
        
        # XGBoost
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'verbosity': 0
        }
        
        if self.gpu_available:
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
        
        candidates.append(ModelCandidate(
            name='xgboost',
            model_class=xgb.XGBRegressor,
            param_space={
                'n_estimators': [100, 200, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            },
            gpu_enabled=self.gpu_available,
            default_params=xgb_params
        ))
        
        # Random Forest
        candidates.append(ModelCandidate(
            name='random_forest',
            model_class=RandomForestRegressor,
            param_space={
                'n_estimators': [100, 200, 500],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            default_params={'random_state': 42, 'n_jobs': -1}
        ))
        
        # Gradient Boosting
        candidates.append(ModelCandidate(
            name='gradient_boosting',
            model_class=GradientBoostingRegressor,
            param_space={
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            default_params={'random_state': 42}
        ))
        
        # Neural Network
        candidates.append(ModelCandidate(
            name='mlp',
            model_class=MLPRegressor,
            param_space={
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01],
                'activation': ['relu', 'tanh']
            },
            default_params={'random_state': 42, 'max_iter': 500}
        ))
        
        # Support Vector Regression
        candidates.append(ModelCandidate(
            name='svr',
            model_class=SVR,
            param_space={
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        ))
        
        # Linear models
        candidates.extend([
            ModelCandidate(
                name='ridge',
                model_class=Ridge,
                param_space={'alpha': [0.1, 1, 10, 100, 1000]},
                default_params={'random_state': 42}
            ),
            ModelCandidate(
                name='elastic_net',
                model_class=ElasticNet,
                param_space={
                    'alpha': [0.1, 1, 10],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                default_params={'random_state': 42}
            )
        ])
        
        return candidates
    
    def run_automl_experiment(self, coin: str, training_data: pd.DataFrame, 
                            target_column: str = 'target', 
                            n_trials: int = 50,
                            cv_folds: int = 5) -> Optional[AutoMLResult]:
        """Run complete AutoML experiment"""
        try:
            if not ML_AVAILABLE:
                self.logger.error("AutoML not available - required dependencies missing")
                return None
                
            self.logger.info(f"Starting AutoML experiment for {coin}")
            start_time = time.time()
            
            # Prepare data
            X, y = self._prepare_automl_data(training_data, target_column)
            
            if X is None or len(X) < 100:
                raise ValueError("Insufficient training data")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            best_score = float('-inf')
            best_result = None
            
            # Test each model candidate
            for candidate in self.model_candidates:
                try:
                    self.logger.info(f"Testing {candidate.name} for {coin}")
                    
                    # Hyperparameter optimization with Optuna
                    if OPTUNA_AVAILABLE:
                        study = optuna.create_study(
                            direction='maximize',
                            study_name=f"{coin}_{candidate.name}",
                            sampler=optuna.samplers.TPESampler(seed=42)
                        )
                    else:
                        # Fallback to simple grid search
                        best_params = {}
                        best_cv_score = -1.0
                    
                    def objective(trial):
                        return self._objective_function(trial, candidate, X, y, tscv)
                    
                    if OPTUNA_AVAILABLE:
                        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, 
                                     callbacks=[lambda study, trial: None])  # Silent optimization
                        
                        # Get best parameters and score
                        best_params = study.best_params
                        best_cv_score = study.best_value
                    else:
                        # Simple default parameters
                        best_params = candidate.default_params.copy() if candidate.default_params else {}
                        model = candidate.model_class(**best_params)
                        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                        best_cv_score = np.mean(scores)
                    
                    # Train final model with best parameters
                    final_model = self._train_final_model(candidate, best_params, X, y)
                    
                    # Feature importance
                    feature_importance = self._get_feature_importance(final_model, X.columns)
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(final_model, X, y, cv=tscv, scoring='r2')
                    
                    result = AutoMLResult(
                        best_model=final_model,
                        best_params=best_params,
                        best_score=best_cv_score,
                        model_name=candidate.name,
                        feature_importance=feature_importance,
                        cross_val_scores=cv_scores.tolist(),
                        training_time=time.time() - start_time,
                        timestamp=datetime.now()
                    )
                    
                    if best_cv_score > best_score:
                        best_score = best_cv_score
                        best_result = result
                    
                    self.logger.info(f"{candidate.name}: CV score = {best_cv_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to test {candidate.name}: {e}")
                    continue
            
            if best_result is None:
                raise ValueError("No models could be trained successfully")
            
            # Store result
            experiment_key = f"{coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.experiment_history[experiment_key] = best_result
            self.best_models[coin] = best_result
            
            total_time = time.time() - start_time
            self.logger.info(f"AutoML experiment completed for {coin}. Best model: {best_result.model_name} "
                           f"(Score: {best_result.best_score:.4f}, Time: {total_time:.1f}s)")
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"AutoML experiment failed for {coin}: {e}")
            raise
    
    def _prepare_automl_data(self, data: pd.DataFrame, target_column: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for AutoML"""
        try:
            # Drop rows with missing target
            data = data.dropna(subset=[target_column])
            
            # Separate features and target
            y = data[target_column]
            X = data.drop(columns=[target_column])
            
            # Remove non-numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_columns]
            
            # Remove constant columns
            constant_columns = X.columns[X.nunique() <= 1]
            X = X.drop(columns=constant_columns)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            y = y.replace([np.inf, -np.inf], np.nan)
            
            # Drop rows with any NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                self.logger.warning("Insufficient data after cleaning")
                return None, None
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None, None
    
    def _objective_function(self, trial, candidate: ModelCandidate, X: pd.DataFrame, 
                          y: pd.Series, cv) -> float:
        """Optuna objective function"""
        try:
            # Sample hyperparameters
            params = candidate.default_params.copy() if candidate.default_params else {}
            
            for param_name, param_values in candidate.param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, np.integer)) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(v, (float, np.floating)) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    low, high = param_values
                    if isinstance(low, (int, np.integer)):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)
            
            # Create and evaluate model
            model = candidate.model_class(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return np.mean(scores)
            
        except Exception as e:
            # Return very poor score on failure
            return -1000
    
    def _train_final_model(self, candidate: ModelCandidate, best_params: Dict, 
                          X: pd.DataFrame, y: pd.Series):
        """Train final model with best parameters"""
        params = candidate.default_params.copy() if candidate.default_params else {}
        params.update(best_params)
        
        model = candidate.model_class(**params)
        model.fit(X, y)
        
        return model
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            importance_dict = {}
            
            # Different ways to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # Default uniform importance
                importances = np.ones(len(feature_names))
            
            # Normalize importances
            if len(importances) == len(feature_names):
                importances = importances / (np.sum(importances) + 1e-8)
                
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    def predict_with_automl(self, coin: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using AutoML model"""
        try:
            if coin not in self.best_models:
                return {'success': False, 'error': 'No AutoML model available for coin'}
            
            result = self.best_models[coin]
            model = result.best_model
            
            # Prepare input data (same preprocessing as training)
            X_pred = input_data.select_dtypes(include=[np.number])
            
            # Remove constant columns that might have been removed during training
            if hasattr(model, 'feature_names_in_'):
                # Use only features that were used during training
                available_features = [col for col in model.feature_names_in_ if col in X_pred.columns]
                X_pred = X_pred[available_features]
            
            # Handle missing values
            X_pred = X_pred.fillna(X_pred.mean())
            
            # Make prediction
            prediction = model.predict(X_pred)
            
            # Prediction confidence (simplified)
            confidence = min(0.95, max(0.1, result.best_score)) if result.best_score > 0 else 0.5
            
            return {
                'success': True,
                'prediction': float(prediction[-1]) if len(prediction) > 0 else 0.0,
                'predictions': prediction.tolist(),
                'confidence': confidence,
                'model_name': result.model_name,
                'feature_importance': result.feature_importance,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"AutoML prediction failed for {coin}: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_automl_training(self, coins: List[str] = None, n_trials: int = 30):
        """Start AutoML training for multiple coins"""
        def training_worker():
            try:
                if coins is None:
                    # Get coins from cache
                    discovered_coins = self.cache_manager.get("discovered_coins")
                    train_coins = list(discovered_coins.keys())[:10] if discovered_coins else ['BTC', 'ETH']
                else:
                    train_coins = coins
                
                for coin in train_coins:
                    try:
                        # Get training data
                        training_data = self._get_training_data(coin)
                        
                        if training_data is not None and len(training_data) > 100:
                            # Run AutoML experiment
                            result = self.run_automl_experiment(coin, training_data, n_trials=n_trials)
                            
                            self.logger.info(f"AutoML completed for {coin}: {result.model_name} "
                                           f"(Score: {result.best_score:.4f})")
                        else:
                            self.logger.warning(f"Insufficient training data for {coin}")
                            
                    except Exception as e:
                        self.logger.error(f"AutoML training failed for {coin}: {e}")
                    
                    time.sleep(2)  # Brief pause between coins
                
                self.logger.info("AutoML batch training completed")
                
            except Exception as e:
                self.logger.error(f"AutoML training worker failed: {e}")
        
        # Start training in background thread
        training_thread = threading.Thread(target=training_worker, daemon=True)
        training_thread.start()
        
        self.logger.info("Started AutoML batch training")
    
    def _get_training_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Get training data for coin"""
        try:
            # Get cached price data
            price_data = self.cache_manager.get(f"validated_price_data_{coin}")
            if not price_data:
                return None
            
            df = pd.DataFrame(price_data)
            
            # Feature engineering
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
            
            # Price ratios
            df['high_low_ratio'] = df['high'] / (df['low'] + 1e-8)
            df['close_open_ratio'] = df['close'] / (df['open'] + 1e-8)
            
            # Lags
            df['returns_lag1'] = df['returns'].shift(1)
            df['returns_lag2'] = df['returns'].shift(2)
            df['volume_ratio_lag1'] = df['volume_ratio'].shift(1)
            
            # Target: next day return
            df['target'] = df['returns'].shift(-1)
            
            # Clean data
            result_df = df.dropna()
            
            return result_df if len(result_df) > 50 else None
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed for {coin}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_automl_status(self) -> Dict[str, Any]:
        """Get AutoML engine status"""
        return {
            'gpu_available': self.gpu_available,
            'model_candidates': len(self.model_candidates),
            'trained_models': len(self.best_models),
            'experiment_history': len(self.experiment_history),
            'available_models': [candidate.name for candidate in self.model_candidates],
            'best_models': {coin: result.model_name for coin, result in self.best_models.items()},
            'timestamp': datetime.now()
        }
    
    def save_automl_models(self, filepath: str = "models/automl_models.pkl"):
        """Save AutoML models"""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            save_data = {
                'best_models': self.best_models,
                'experiment_history': self.experiment_history,
                'timestamp': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"AutoML models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save AutoML models: {e}")
    
    def load_automl_models(self, filepath: str = "models/automl_models.pkl"):
        """Load AutoML models"""
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"AutoML model file not found: {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.best_models = save_data.get('best_models', {})
            self.experiment_history = save_data.get('experiment_history', {})
            
            self.logger.info(f"Loaded {len(self.best_models)} AutoML models from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load AutoML models: {e}")