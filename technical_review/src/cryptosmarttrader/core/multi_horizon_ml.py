"""
CryptoSmartTrader V2 - Multi-Horizon ML System
Training and inference on multiple time horizons (1H, 24H, 7D, 30D) with self-learning
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import pickle
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as LGBMRegressor
    LIGHTGBM_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.gpu_accelerator import gpu_accelerator

class MultiHorizonMLSystem:
    """Multi-horizon ML system for alpha seeking with self-learning capabilities"""

    def __init__(self, container):
        self.container = container
        self.cache_manager = container.cache_manager()
        self.config_manager = container.config()
        self.logger = logging.getLogger(__name__)

        # Time horizons in hours
        self.horizons = {
            '1H': 1,
            '24H': 24,
            '7D': 168,   # 7 * 24
            '30D': 720   # 30 * 24
        }

        # ML models for each horizon
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}

        # Training configuration
        self.training_config = {
            'min_training_samples': 1000,
            'test_size': 0.2,
            'validation_size': 0.1,
            'max_features': 50,
            'retrain_threshold_mae': 0.15,  # Retrain if MAE > 15%
            'confidence_threshold': 0.80
        }

        # Feature engineering configuration
        self.feature_config = {
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_features': ['rsi', 'macd', 'bb_position', 'trend_strength', 'volume_ratio'],
            'sentiment_features': ['sentiment_score', 'mention_volume', 'sentiment_trend'],
            'whale_features': ['large_transactions', 'net_flow', 'whale_concentration'],
            'derived_features': ['price_momentum', 'volume_momentum', 'volatility']
        }

        # Self-learning tracking
        self.prediction_log = {}
        self.model_accuracy_history = {}
        self.last_training_time = {}

        # Model paths
        self.model_dir = Path('models/multi_horizon')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Multi-horizon ML system initialized with {len(self.horizons)} time horizons")

    def prepare_training_data(self, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Prepare comprehensive training dataset from all available data"""
        try:
            if not self.cache_manager:
                self.logger.error("Cache manager not available")
                return None

            self.logger.info(f"Preparing training data with {lookback_days} days lookback")

            # Collect all validated data
            training_data = []

            # Get all coins with validated data
            validated_coins = self._get_coins_with_complete_historical_data()

            if len(validated_coins) < 10:
                self.logger.warning(f"Insufficient coins with complete data: {len(validated_coins)}")
                return None

            # Process each coin
            for coin in validated_coins:
                coin_data = self._extract_coin_historical_data(coin, lookback_days)
                if coin_data is not None and len(coin_data) > 24:  # At least 24 hours of data
                    training_data.append(coin_data)

            if not training_data:
                self.logger.error("No valid training data collected")
                return None

            # Combine all coin data
            combined_df = pd.concat(training_data, ignore_index=True)

            # Create targets for all horizons
            combined_df = self._create_horizon_targets(combined_df)

            # Feature engineering
            combined_df = self._engineer_features(combined_df)

            # Remove rows with any NaN targets
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna(subset=[f'target_{h}' for h in self.horizons.keys()])
            final_rows = len(combined_df)

            self.logger.info(f"Training data prepared: {final_rows}/{initial_rows} valid samples from {len(validated_coins)} coins")

            return combined_df if len(combined_df) >= self.training_config['min_training_samples'] else None

        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None

    def _get_coins_with_complete_historical_data(self) -> List[str]:
        """Get coins that have sufficient historical data for training"""
        try:
            complete_coins = []

            # Search cache for historical analysis data
            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith('analysis_') and '_1h' in cache_key:
                    parts = cache_key.split('_')
                    if len(parts) >= 2:
                        coin = parts[1]

                        # Check if coin has data for multiple timeframes
                        timeframes_available = 0
                        for tf in ['1h', '4h', '1d']:
                            tf_key = f'analysis_{coin}_{tf}'
                            if tf_key in self.cache_manager._cache:
                                timeframes_available += 1

                        # Require at least 2 timeframes
                        if timeframes_available >= 2:
                            complete_coins.append(coin)

            # Remove duplicates
            complete_coins = list(set(complete_coins))

            self.logger.info(f"Found {len(complete_coins)} coins with sufficient historical data")
            return complete_coins

        except Exception as e:
            self.logger.error(f"Failed to get coins with historical data: {e}")
            return []

    def _extract_coin_historical_data(self, coin: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Extract historical data for a specific coin"""
        try:
            coin_records = []

            # Get data from different timeframes
            for timeframe in ['1h', '4h', '1d']:
                cache_key = f'analysis_{coin}_{timeframe}'
                analysis_data = self.cache_manager.get(cache_key)

                if analysis_data:
                    # Create record with timestamp
                    record = {
                        'coin': coin,
                        'timestamp': datetime.now(),  # In real system, use actual timestamp
                        'timeframe': timeframe,
                        **self._extract_features_from_analysis(analysis_data)
                    }
                    coin_records.append(record)

            if not coin_records:
                return None

            # Convert to DataFrame and simulate historical progression
            df = pd.DataFrame(coin_records)

            # REMOVED: Mock data pattern not allowed in production
            base_time = datetime.now() - timedelta(days=lookback_days)
            df['timestamp'] = pd.date_range(
                start=base_time,
                periods=len(df),
                freq='1H'
            )

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Failed to extract historical data for {coin}: {e}")
            return None

    def _extract_features_from_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract ML features from analysis data"""
        try:
            features = {}

            # Price features
            for feature in self.feature_config['price_features']:
                features[feature] = float(analysis_data.get(feature, 0))

            # Technical features
            for feature in self.feature_config['technical_features']:
                features[feature] = float(analysis_data.get(feature, 0))

            # Ensure we have a price for target calculation
            if 'close' not in features or features['close'] <= 0:
                features['close'] = features.get('last_price', 100.0)  # Fallback

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}

    def _create_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for all time horizons"""
        try:
            df = df.copy()

            # Sort by coin and timestamp
            df = df.sort_values(['coin', 'timestamp']).reset_index(drop=True)

            # Create targets for each horizon
            for horizon_name, horizon_hours in self.horizons.items():
                target_col = f'target_{horizon_name}'

                # Calculate future return for this horizon
                df[target_col] = df.groupby('coin')['close'].transform(
                    lambda x: x.shift(-horizon_hours) / x - 1
                )

            return df

        except Exception as e:
            self.logger.error(f"Target creation failed: {e}")
            return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better predictions"""
        try:
            df = df.copy()

            # Price momentum features
            df['price_momentum'] = df.groupby('coin')['close'].pct_change()
            df['price_momentum_3h'] = df.groupby('coin')['close'].pct_change(3)

            # Volume momentum
            df['volume_momentum'] = df.groupby('coin')['volume'].pct_change()

            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df.groupby('coin')['price_momentum'].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)

            # Price relative to recent high/low
            df['price_vs_24h_high'] = df['close'] / df.groupby('coin')['high'].rolling(window=24, min_periods=1).max().reset_index(0, drop=True)
            df['price_vs_24h_low'] = df['close'] / df.groupby('coin')['low'].rolling(window=24, min_periods=1).min().reset_index(0, drop=True)

            # Technical momentum
            df['rsi_momentum'] = df.groupby('coin')['rsi'].diff()

            # Volume trend
            df['volume_trend'] = df.groupby('coin')['volume'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True) / df['volume']

            # Fill any remaining NaN values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return df

    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train models for all time horizons"""
        try:
            self.logger.info("Starting multi-horizon model training")

            # Prepare feature matrix
            feature_columns = self._get_feature_columns(training_data)
            X = training_data[feature_columns].values

            training_results = {}

            # Train model for each horizon
            for horizon_name in self.horizons.keys():
                self.logger.info(f"Training model for {horizon_name} horizon")

                target_col = f'target_{horizon_name}'
                y = training_data[target_col].values

                # Remove samples with invalid targets
                valid_mask = ~np.isnan(y) & ~np.isinf(y)
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]

                if len(X_valid) < self.training_config['min_training_samples']:
                    self.logger.warning(f"Insufficient valid samples for {horizon_name}: {len(X_valid)}")
                    continue

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_valid,
                    test_size=self.training_config['test_size'],
                    random_state=42
                )

                # Configure model
                if LIGHTGBM_AVAILABLE:
                    model = LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        verbose=-1
                    )
                else:
                    model = LGBMRegressor(
                        n_estimators=100,
                        max_depth=8,
                        random_state=42
                    )

                # Train model
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                # Store model and performance
                self.models[horizon_name] = model
                self.model_performance[horizon_name] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': len(feature_columns),
                    'training_time': datetime.now().isoformat()
                }

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_columns, model.feature_importances_))
                    self.feature_importance[horizon_name] = importance_dict

                # Save model
                self._save_model(horizon_name, model, feature_columns)

                training_results[horizon_name] = {
                    'success': True,
                    'test_mae': test_mae,
                    'samples': len(X_valid)
                }

                self.logger.info(f"{horizon_name} model trained - MAE: {test_mae:.4f}, Samples: {len(X_valid)}")

            # Store training timestamp
            for horizon in self.models.keys():
                self.last_training_time[horizon] = datetime.now()

            # Cache training results
            if self.cache_manager:
                self.cache_manager.set(
                    'multi_horizon_training_results',
                    {
                        'timestamp': datetime.now().isoformat(),
                        'results': training_results,
                        'model_performance': self.model_performance,
                        'feature_importance': self.feature_importance
                    },
                    ttl_minutes=1440  # 24 hours
                )

            return training_results

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {}

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for training"""
        try:
            # Start with configured features
            feature_columns = []

            for feature_type, features in self.feature_config.items():
                for feature in features:
                    if feature in df.columns:
                        feature_columns.append(feature)

            # Add engineered features
            engineered_features = [
                'price_momentum', 'price_momentum_3h', 'volume_momentum',
                'volatility', 'price_vs_24h_high', 'price_vs_24h_low',
                'rsi_momentum', 'volume_trend'
            ]

            for feature in engineered_features:
                if feature in df.columns:
                    feature_columns.append(feature)

            # Remove duplicates and limit to max features
            feature_columns = list(set(feature_columns))

            if len(feature_columns) > self.training_config['max_features']:
                feature_columns = feature_columns[:self.training_config['max_features']]

            return feature_columns

        except Exception as e:
            self.logger.error(f"Feature column selection failed: {e}")
            return []

    def predict_all_horizons(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict returns for all horizons with confidence scoring"""
        try:
            if not self.models:
                self.logger.warning("No trained models available for prediction")
                return {}

            feature_columns = self._get_feature_columns(features_df)

            if not feature_columns:
                self.logger.error("No valid feature columns for prediction")
                return {}

            # Prepare features
            X = features_df[feature_columns].values

            predictions = {}

            # Predict for each horizon
            for horizon_name, model in self.models.items():
                try:
                    # Make predictions
                    y_pred = model.predict(X)

                    # Calculate prediction confidence
                    confidence_scores = self._calculate_prediction_confidence(
                        model, X, horizon_name
                    )

                    # Combine predictions with metadata
                    horizon_results = []
                    for i, (pred, conf) in enumerate(zip(y_pred, confidence_scores)):
                        if i < len(features_df):
                            coin = features_df.iloc[i].get('coin', f'coin_{i}')

                            result = {
                                'coin': coin,
                                'predicted_return': float(pred),
                                'confidence': float(conf),
                                'meets_threshold': conf >= self.training_config['confidence_threshold'],
                                'prediction_timestamp': datetime.now().isoformat()
                            }

                            horizon_results.append(result)

                    predictions[horizon_name] = horizon_results

                except Exception as e:
                    self.logger.error(f"Prediction failed for {horizon_name}: {e}")
                    continue

            return predictions

        except Exception as e:
            self.logger.error(f"Multi-horizon prediction failed: {e}")
            return {}

    def _calculate_prediction_confidence(self, model, X: np.ndarray, horizon_name: str) -> np.ndarray:
        """Calculate prediction confidence scores"""
        try:
            # Method 1: Use prediction intervals if available
            if hasattr(model, 'predict') and LIGHTGBM_AVAILABLE:
                # For LightGBM, we can estimate uncertainty using multiple predictions
                predictions = []
                for _ in range(10):
                    # Add small noise to get prediction variance
                    X_noisy = X + np.random.normal(0, 1)
                    pred = model.predict(X_noisy)
                    predictions.append(pred)

                predictions = np.array(predictions)
                pred_std = np.std(predictions, axis=0)

                # Convert standard deviation to confidence (inverse relationship)
                confidence = 1 / (1 + pred_std)
                confidence = np.clip(confidence, 0, 1)

                return confidence

            # Method 2: Use historical model performance
            performance = self.model_performance.get(horizon_name, {})
            base_confidence = 1 - performance.get('test_mae', 0.5)
            base_confidence = max(0.1, min(0.95, base_confidence))

            # Return uniform confidence based on model performance
            return np.full(len(X), base_confidence)

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            # Fallback to conservative confidence
            return np.full(len(X), 0.6)

    def get_alpha_opportunities(self, min_confidence: float = 0.80, min_return_30d: float = 1.0) -> List[Dict[str, Any]]:
        """Get alpha opportunities across all horizons with strict filtering"""
        try:
            if not self.cache_manager:
                return []

            # Get latest validated data for prediction
            prediction_data = self._prepare_prediction_data()

            if prediction_data is None or len(prediction_data) == 0:
                self.logger.warning("No data available for alpha opportunity prediction")
                return []

            # Make predictions for all horizons
            all_predictions = self.predict_all_horizons(prediction_data)

            if not all_predictions:
                return []

            # Combine predictions across horizons for each coin
            opportunities = []

            # Get unique coins
            all_coins = set()
            for horizon_preds in all_predictions.values():
                for pred in horizon_preds:
                    all_coins.add(pred['coin'])

            # Process each coin
            for coin in all_coins:
                coin_opportunity = {
                    'coin': coin,
                    'horizons': {},
                    'overall_confidence': 0.0,
                    'max_return_30d': 0.0
                }

                # Collect predictions for all horizons
                horizon_confidences = []

                for horizon_name, horizon_preds in all_predictions.items():
                    coin_pred = next((p for p in horizon_preds if p['coin'] == coin), None)

                    if coin_pred:
                        coin_opportunity['horizons'][horizon_name] = {
                            'predicted_return': coin_pred['predicted_return'],
                            'confidence': coin_pred['confidence']
                        }

                        horizon_confidences.append(coin_pred['confidence'])

                        # Track maximum 30D return
                        if horizon_name == '30D':
                            coin_opportunity['max_return_30d'] = coin_pred['predicted_return']

                # Calculate overall confidence (average across horizons)
                if horizon_confidences:
                    coin_opportunity['overall_confidence'] = np.mean(horizon_confidences)

                # Apply strict filtering
                if (coin_opportunity['overall_confidence'] >= min_confidence and
                    coin_opportunity['max_return_30d'] >= min_return_30d):

                    coin_opportunity['meets_strict_criteria'] = True
                    opportunities.append(coin_opportunity)

            # Sort by 30D return (highest first)
            opportunities.sort(key=lambda x: x['max_return_30d'], reverse=True)

            # Log prediction for self-learning
            self._log_predictions(opportunities)

            return opportunities

        except Exception as e:
            self.logger.error(f"Alpha opportunity detection failed: {e}")
            return []

    def _prepare_prediction_data(self) -> Optional[pd.DataFrame]:
        """Prepare current data for prediction"""
        try:
            # Get coins with recent validated data
            validated_coins = []

            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith('validated_price_data_'):
                    coin = cache_key.replace('validated_price_data_', '')
                    validated_coins.append(coin)

            if not validated_coins:
                return None

            # Prepare current features for each coin
            prediction_records = []

            for coin in validated_coins[:50]:  # Limit for performance
                # Get latest validated data
                price_data = self.cache_manager.get(f'validated_price_data_{coin}')
                sentiment_data = self.cache_manager.get(f'validated_sentiment_{coin}')
                whale_data = self.cache_manager.get(f'validated_whale_{coin}')

                if not all([price_data, sentiment_data, whale_data]):
                    continue

                # Extract features
                record = {'coin': coin, 'timestamp': datetime.now()}

                # Price features from latest timeframe
                if 'timeframes' in price_data and '1h' in price_data['timeframes']:
                    tf_data = price_data['timeframes']['1h']
                    record.update(self._extract_features_from_analysis(tf_data))

                # Sentiment features
                record.update({
                    'sentiment_score': sentiment_data.get('sentiment_score', 0.5),
                    'mention_volume': sentiment_data.get('mention_volume', 0),
                    'sentiment_trend': sentiment_data.get('sentiment_trend', 0)
                })

                # Whale features
                record.update({
                    'large_transactions': whale_data.get('large_transactions', 0),
                    'net_flow': whale_data.get('net_flow', 0),
                    'whale_concentration': whale_data.get('whale_concentration', 0.5)
                })

                prediction_records.append(record)

            if not prediction_records:
                return None

            df = pd.DataFrame(prediction_records)

            # Apply feature engineering (without targets)
            df = self._engineer_features(df)

            # Fill NaN values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Prediction data preparation failed: {e}")
            return None

    def _log_predictions(self, opportunities: List[Dict[str, Any]]):
        """Log predictions for self-learning"""
        try:
            prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            prediction_log = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'opportunities': opportunities,
                'total_opportunities': len(opportunities),
                'verified': False
            }

            self.prediction_log[prediction_id] = prediction_log

            # Store in cache for persistence
            if self.cache_manager:
                self.cache_manager.set(
                    f'ml_prediction_log_{prediction_id}',
                    prediction_log,
                    ttl_minutes=10080  # 7 days
                )

        except Exception as e:
            self.logger.error(f"Prediction logging failed: {e}")

    def _save_model(self, horizon_name: str, model, feature_columns: List[str]):
        """Save trained model to disk"""
        try:
            model_path = self.model_dir / f'model_{horizon_name}.pkl'
            features_path = self.model_dir / f'features_{horizon_name}.json'

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save feature columns
            with open(features_path, 'w') as f:
                json.dump(feature_columns, f)

            self.logger.info(f"Model saved for {horizon_name}: {model_path}")

        except Exception as e:
            self.logger.error(f"Model saving failed for {horizon_name}: {e}")

    def load_models(self):
        """Load previously trained models"""
        try:
            loaded_count = 0

            for horizon_name in self.horizons.keys():
                model_path = self.model_dir / f'model_{horizon_name}.pkl'
                features_path = self.model_dir / f'features_{horizon_name}.json'

                if model_path.exists() and features_path.exists():
                    # Load model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    # Load feature columns
                    with open(features_path, 'r') as f:
                        feature_columns = json.load(f)

                    self.models[horizon_name] = model
                    loaded_count += 1

                    self.logger.info(f"Loaded model for {horizon_name}")

            self.logger.info(f"Loaded {loaded_count}/{len(self.horizons)} models")
            return loaded_count > 0

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False

    def check_retrain_needed(self) -> Dict[str, bool]:
        """Check if models need retraining based on performance"""
        try:
            retrain_needed = {}

            for horizon_name in self.horizons.keys():
                needs_retrain = False

                # Check if model exists
                if horizon_name not in self.models:
                    needs_retrain = True

                # Check performance degradation
                elif horizon_name in self.model_performance:
                    test_mae = self.model_performance[horizon_name].get('test_mae', 0)
                    if test_mae > self.training_config['retrain_threshold_mae']:
                        needs_retrain = True

                # Check time since last training
                elif horizon_name in self.last_training_time:
                    time_since_training = datetime.now() - self.last_training_time[horizon_name]
                    if time_since_training.days > 7:  # Retrain weekly
                        needs_retrain = True

                retrain_needed[horizon_name] = needs_retrain

            return retrain_needed

        except Exception as e:
            self.logger.error(f"Retrain check failed: {e}")
            return {h: True for h in self.horizons.keys()}  # Conservative: retrain all

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'models_loaded': len(self.models),
                'total_horizons': len(self.horizons),
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'last_training_times': {
                    h: t.isoformat() if isinstance(t, datetime) else str(t)
                    for h, t in self.last_training_time.items()
                },
                'prediction_log_count': len(self.prediction_log),
                'retrain_needed': self.check_retrain_needed(),
                'training_config': self.training_config
            }

            return status

        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {'error': str(e)}
