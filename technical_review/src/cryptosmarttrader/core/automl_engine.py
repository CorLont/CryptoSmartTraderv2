"""
AutoML Engine - Automated Machine Learning Pipeline
Self-optimizing ML with hyperparameter tuning and meta-learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import optuna
    from optuna import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Define dummy classes to prevent import errors
    class Trial:
        def suggest_int(self, name, low, high):
            return (low + high) // 2
        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2
        def suggest_categorical(self, name, choices):
            return choices[0]

    class optuna:
        @staticmethod
        def create_study(*args, **kwargs):
            return None
        class samplers:
            class TPESampler:
                def __init__(self, *args, **kwargs):
                    pass
        class pruners:
            class MedianPruner:
                def __init__(self, *args, **kwargs):
                    pass

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

class AutoMLEngine:
    """
    Automated Machine Learning engine with hyperparameter optimization
    and meta-learning for cryptocurrency prediction
    """

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)

        # AutoML configuration
        self.config = {
            'optimization_trials': 100,
            'cv_folds': 5,
            'optimization_timeout': 3600,  # 1 hour
            'meta_learning_memory': 50,
            'ensemble_size': 5,
            'feature_selection_k': 30
        }

        # Model registry
        self.model_templates = {
            'xgboost': self._create_xgboost_objective,
            'random_forest': self._create_rf_objective,
            'gradient_boost': self._create_gb_objective,
            'ridge': self._create_ridge_objective,
            'elastic_net': self._create_elastic_objective,
            'svr': self._create_svr_objective
        }

        # Optimized models storage
        self.optimized_models = {}

        # Meta-learning database
        self.meta_learning_db = {
            'model_performances': [],
            'hyperparameter_history': [],
            'dataset_characteristics': [],
            'optimization_history': []
        }

        # Best configurations cache
        self.best_configs = {}

        # Optimization studies
        self.studies = {}

        self.logger.info("AutoML Engine initialized")

    def initialize_automl(self) -> bool:
        """Initialize AutoML components"""

        try:
            if not OPTUNA_AVAILABLE:
                self.logger.warning("Optuna not available - using grid search fallback")
                return self._initialize_fallback()

            # Create optimization studies for each horizon
            for horizon in ['1h', '24h', '7d', '30d']:
                study_name = f'crypto_prediction_{horizon}'

                self.studies[horizon] = optuna.create_study(
                    study_name=study_name,
                    direction='minimize',  # Minimize prediction error
                    sampler=optuna.samplers.TPESampler(
                        n_startup_trials=10,
                        n_ei_candidates=24
                    ),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=5,
                        n_warmup_steps=10,
                        interval_steps=1
                    )
                )

            self.logger.critical("AUTOML ENGINE INITIALIZED - Automated optimization active")
            return True

        except Exception as e:
            self.logger.error(f"AutoML initialization failed: {e}")
            return False

    def _initialize_fallback(self) -> bool:
        """Initialize fallback AutoML without Optuna"""

        # Simple grid search configurations
        self.fallback_configs = {
            'xgboost': [
                {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
                {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
                {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.01}
            ],
            'random_forest': [
                {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
                {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
                {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2}
            ]
        }

        return True

    async def optimize_model_for_coin(self,
                                    coin: str,
                                    training_data: pd.DataFrame,
                                    target_column: str,
                                    horizon: str) -> Dict[str, Any]:
        """Optimize ML model for specific coin and horizon"""

        try:
            if training_data.empty or len(training_data) < 50:
                return {
                    'success': False,
                    'reason': 'Insufficient training data',
                    'samples': len(training_data)
                }

            # Prepare data
            X, y = self._prepare_optimization_data(training_data, target_column)

            if len(X) == 0:
                return {'success': False, 'reason': 'No valid features extracted'}

            # Characterize dataset for meta-learning
            dataset_chars = self._characterize_dataset(X, y, coin, horizon)

            # Get meta-learning recommendations
            recommended_models = self._get_meta_recommendations(dataset_chars)

            # Optimize each recommended model
            optimization_results = {}

            for model_name in recommended_models:
                self.logger.info(f"Optimizing {model_name} for {coin} {horizon}")

                if OPTUNA_AVAILABLE:
                    result = await self._optimize_with_optuna(
                        model_name, X, y, coin, horizon
                    )
                else:
                    result = await self._optimize_with_grid_search(
                        model_name, X, y, coin, horizon
                    )

                optimization_results[model_name] = result

            # Select best model
            best_model_info = self._select_best_model(optimization_results)

            # Store optimization results
            self._store_optimization_results(
                coin, horizon, best_model_info, dataset_chars
            )

            return {
                'success': True,
                'coin': coin,
                'horizon': horizon,
                'best_model': best_model_info,
                'all_results': optimization_results,
                'dataset_characteristics': dataset_chars,
                'optimization_time': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Model optimization failed for {coin} {horizon}: {e}")
            return {'success': False, 'error': str(e)}

    def _prepare_optimization_data(self,
                                 training_data: pd.DataFrame,
                                 target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization"""

        try:
            # Remove target column from features
            feature_columns = [col for col in training_data.columns if col != target_column]

            X = training_data[feature_columns].values
            y = training_data[target_column].values

            # Remove any rows with NaN
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            # Feature selection
            if X.shape[1] > self.config['feature_selection_k']:
                selector = SelectKBest(score_func=f_regression, k=self.config['feature_selection_k'])
                X = selector.fit_transform(X, y)

            return X, y

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return np.array([]), np.array([])

    def _characterize_dataset(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            coin: str,
                            horizon: str) -> Dict[str, Any]:
        """Characterize dataset for meta-learning"""

        try:
            characteristics = {
                'coin': coin,
                'horizon': horizon,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'target_mean': float(np.mean(y)),
                'target_std': float(np.std(y)),
                'target_skew': float(self._calculate_skewness(y)),
                'target_range': float(np.max(y) - np.min(y)),
                'feature_means': np.mean(X, axis=0).tolist(),
                'feature_stds': np.std(X, axis=0).tolist(),
                'correlation_with_target': [
                    float(np.corrcoef(X[:, i], y)[0, 1]) if not np.isnan(np.corrcoef(X[:, i], y)[0, 1]) else 0.0
                    for i in range(X.shape[1])
                ],
                'timestamp': datetime.now().isoformat()
            }

            return characteristics

        except Exception as e:
            self.logger.error(f"Dataset characterization failed: {e}")
            return {}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""

        try:
            mean = np.mean(data)
            std = np.std(data)

            if std == 0:
                return 0.0

            skew = np.mean(((data - mean) / std) ** 3)
            return skew

        except:
            return 0.0

    def _get_meta_recommendations(self, dataset_chars: Dict[str, Any]) -> List[str]:
        """Get model recommendations based on meta-learning"""

        try:
            # Default recommendation order
            default_order = ['xgboost', 'random_forest', 'gradient_boost', 'ridge', 'elastic_net']

            if not dataset_chars or not self.meta_learning_db['model_performances']:
                return default_order

            # Find similar datasets in meta-learning database
            similar_datasets = self._find_similar_datasets(dataset_chars)

            if not similar_datasets:
                return default_order

            # Rank models based on performance on similar datasets
            model_scores = {}

            for dataset in similar_datasets:
                for perf in self.meta_learning_db['model_performances']:
                    if (perf['coin'] == dataset['coin'] and
                        perf['horizon'] == dataset['horizon']):

                        model_name = perf['model_name']
                        score = perf['cv_score']

                        if model_name not in model_scores:
                            model_scores[model_name] = []
                        model_scores[model_name].append(score)

            # Average scores and sort
            avg_scores = {
                model: np.mean(scores)
                for model, scores in model_scores.items()
                if scores
            }

            if avg_scores:
                recommended = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)

                # Include remaining models
                for model in default_order:
                    if model not in recommended:
                        recommended.append(model)

                return recommended

            return default_order

        except Exception as e:
            self.logger.error(f"Meta-learning recommendations failed: {e}")
            return ['xgboost', 'random_forest', 'gradient_boost']

    def _find_similar_datasets(self, target_chars: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar datasets in meta-learning database"""

        try:
            similar_datasets = []

            for dataset in self.meta_learning_db['dataset_characteristics']:
                # Calculate similarity score
                similarity = self._calculate_dataset_similarity(target_chars, dataset)

                if similarity > 0.7:  # Similarity threshold
                    similar_datasets.append(dataset)

            # Sort by similarity (most similar first)
            similar_datasets.sort(
                key=lambda x: self._calculate_dataset_similarity(target_chars, x),
                reverse=True
            )

            return similar_datasets[:10]  # Top 10 most similar

        except Exception as e:
            self.logger.error(f"Similar dataset search failed: {e}")
            return []

    def _calculate_dataset_similarity(self,
                                   dataset1: Dict[str, Any],
                                   dataset2: Dict[str, Any]) -> float:
        """Calculate similarity between two datasets"""

        try:
            # Features to compare
            numeric_features = [
                'n_samples', 'n_features', 'target_mean', 'target_std', 'target_range'
            ]

            similarities = []

            for feature in numeric_features:
                if feature in dataset1 and feature in dataset2:
                    val1 = dataset1[feature]
                    val2 = dataset2[feature]

                    if val1 == 0 and val2 == 0:
                        sim = 1.0
                    elif val1 == 0 or val2 == 0:
                        sim = 0.0
                    else:
                        # Normalized similarity
                        sim = 1.0 / (1.0 + abs(val1 - val2) / max(abs(val1), abs(val2)))

                    similarities.append(sim)

            # Same horizon bonus
            if (dataset1.get('horizon') == dataset2.get('horizon')):
                similarities.append(1.5)  # Bonus for same horizon

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    async def _optimize_with_optuna(self,
                                  model_name: str,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  coin: str,
                                  horizon: str) -> Dict[str, Any]:
        """Optimize model using Optuna"""

        try:
            if horizon not in self.studies:
                return {'error': f'No study available for horizon {horizon}'}

            study = self.studies[horizon]

            # Create objective function
            objective_func = self.model_templates[model_name](X, y)

            # Optimize
            study.optimize(
                objective_func,
                n_trials=self.config['optimization_trials'],
                timeout=self.config['optimization_timeout'],
                show_progress_bar=False
            )

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            # Train final model with best parameters
            final_model = self._train_final_model(model_name, X, y, best_params)

            return {
                'model_name': model_name,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': len(study.trials),
                'model': final_model,
                'optimization_method': 'optuna'
            }

        except Exception as e:
            self.logger.error(f"Optuna optimization failed for {model_name}: {e}")
            return {'error': str(e)}

    async def _optimize_with_grid_search(self,
                                       model_name: str,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       coin: str,
                                       horizon: str) -> Dict[str, Any]:
        """Optimize model using grid search (fallback)"""

        try:
            if model_name not in self.fallback_configs:
                return {'error': f'No fallback config for {model_name}'}

            best_score = float('inf')
            best_params = None
            best_model = None

            configs = self.fallback_configs[model_name]

            for config in configs:
                # Train model with this config
                model = self._create_model(model_name, config)

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                    scoring='neg_mean_squared_error'
                )

                score = -cv_scores.mean()

                if score < best_score:
                    best_score = score
                    best_params = config
                    best_model = model

            # Train final model
            if best_model:
                best_model.fit(X, y)

            return {
                'model_name': model_name,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': len(configs),
                'model': best_model,
                'optimization_method': 'grid_search'
            }

        except Exception as e:
            self.logger.error(f"Grid search optimization failed for {model_name}: {e}")
            return {'error': str(e)}

    def _create_xgboost_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create XGBoost optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'random_state': 42
            }

            model = xgb.XGBRegressor(**params)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()  # Minimize error

        return objective

    def _create_rf_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create Random Forest optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'random_state': 42
            }

            model = RandomForestRegressor(**params)

            cv_scores = cross_val_score(
                model, X, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()

        return objective

    def _create_gb_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create Gradient Boosting optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }

            model = GradientBoostingRegressor(**params)

            cv_scores = cross_val_score(
                model, X, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()

        return objective

    def _create_ridge_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create Ridge optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True),
                'random_state': 42
            }

            # Scale features for Ridge
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(**params)

            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()

        return objective

    def _create_elastic_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create Elastic Net optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': 42
            }

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = ElasticNet(**params)

            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()

        return objective

    def _create_svr_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Create SVR optimization objective"""

        def objective(trial: Trial) -> float:
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }

            # Scale features for SVR
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = SVR(**params)

            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=TimeSeriesSplit(n_splits=self.config['cv_folds']),
                scoring='neg_mean_squared_error'
            )

            return -cv_scores.mean()

        return objective

    def _create_model(self, model_name: str, params: Dict[str, Any]):
        """Create model instance with parameters"""

        if model_name == 'xgboost':
            return xgb.XGBRegressor(**params)
        elif model_name == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_name == 'gradient_boost':
            return GradientBoostingRegressor(**params)
        elif model_name == 'ridge':
            return Ridge(**params)
        elif model_name == 'elastic_net':
            return ElasticNet(**params)
        elif model_name == 'svr':
            return SVR(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _train_final_model(self,
                         model_name: str,
                         X: np.ndarray,
                         y: np.ndarray,
                         params: Dict[str, Any]):
        """Train final model with optimized parameters"""

        try:
            model = self._create_model(model_name, params)

            # Scale data if needed
            if model_name in ['ridge', 'elastic_net', 'svr']:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Store scaler with model
                model.scaler = scaler

            model.fit(X, y)
            return model

        except Exception as e:
            self.logger.error(f"Final model training failed: {e}")
            return None

    def _select_best_model(self, optimization_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Select best model from optimization results"""

        valid_results = {
            name: result for name, result in optimization_results.items()
            if 'error' not in result and 'best_score' in result
        }

        if not valid_results:
            return {'error': 'No valid optimization results'}

        # Find model with lowest error
        best_model_name = min(valid_results.keys(), key=lambda x: valid_results[x]['best_score'])
        best_result = valid_results[best_model_name]

        return {
            'model_name': best_model_name,
            'model': best_result['model'],
            'best_params': best_result['best_params'],
            'cv_score': best_result['best_score'],
            'optimization_method': best_result.get('optimization_method', 'unknown'),
            'n_trials': best_result.get('n_trials', 0)
        }

    def _store_optimization_results(self,
                                  coin: str,
                                  horizon: str,
                                  best_model_info: Dict[str, Any],
                                  dataset_chars: Dict[str, Any]):
        """Store optimization results for meta-learning"""

        try:
            # Store model performance
            performance_record = {
                'coin': coin,
                'horizon': horizon,
                'model_name': best_model_info['model_name'],
                'cv_score': best_model_info['cv_score'],
                'optimization_method': best_model_info.get('optimization_method'),
                'timestamp': datetime.now().isoformat()
            }

            self.meta_learning_db['model_performances'].append(performance_record)

            # Store dataset characteristics
            self.meta_learning_db['dataset_characteristics'].append(dataset_chars)

            # Store hyperparameters
            hyperparams_record = {
                'coin': coin,
                'horizon': horizon,
                'model_name': best_model_info['model_name'],
                'best_params': best_model_info['best_params'],
                'timestamp': datetime.now().isoformat()
            }

            self.meta_learning_db['hyperparameter_history'].append(hyperparams_record)

            # Limit memory usage
            max_records = self.config['meta_learning_memory']

            for key in self.meta_learning_db:
                if len(self.meta_learning_db[key]) > max_records:
                    self.meta_learning_db[key] = self.meta_learning_db[key][-max_records:]

            # Store optimized model
            model_key = f"{coin}_{horizon}"
            self.optimized_models[model_key] = best_model_info

            self.logger.info(f"Optimization results stored for {coin} {horizon}")

        except Exception as e:
            self.logger.error(f"Failed to store optimization results: {e}")

    def get_automl_status(self) -> Dict[str, Any]:
        """Get AutoML engine status"""

        return {
            'optuna_available': OPTUNA_AVAILABLE,
            'studies_initialized': len(self.studies),
            'optimized_models': len(self.optimized_models),
            'meta_learning_records': {
                key: len(records) for key, records in self.meta_learning_db.items()
            },
            'config': self.config,
            'last_updated': datetime.now().isoformat()
        }

    def get_best_model_for_coin(self, coin: str, horizon: str) -> Optional[Any]:
        """Get best optimized model for coin and horizon"""

        model_key = f"{coin}_{horizon}"

        if model_key in self.optimized_models:
            return self.optimized_models[model_key]['model']

        return None
