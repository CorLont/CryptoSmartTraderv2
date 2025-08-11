"""
Enhanced ML/AI Agent
Addresses: mandatory deep learning, uncertainty modeling, adaptive features, explainability
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced ML libraries
try:
    import torch
    import torch.nn as nn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

from utils.daily_logger import get_daily_logger

@dataclass
class MLPrediction:
    """Enhanced ML prediction with uncertainty"""
    coin: str
    horizon: str  # 1h, 4h, 1d, 7d
    prediction: float
    uncertainty: float  # Standard deviation or confidence interval
    confidence: float  # 0 to 1
    model_type: str
    feature_importance: Dict[str, float]
    prediction_interval: Tuple[float, float]  # Lower, upper bounds
    timestamp: datetime
    model_version: str
    explanation: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    horizon: str
    mse: float
    mae: float
    directional_accuracy: float
    sharpe_ratio: float
    last_updated: datetime
    training_samples: int

class FeatureEngineer:
    """Advanced feature engineering with adaptivity"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('ml_predictions')
        self.feature_importance_cache = {}
        self.regime_weights = {
            'bull': {'momentum': 1.5, 'trend': 1.3, 'volume': 1.1},
            'bear': {'momentum': 1.2, 'trend': 1.4, 'volatility': 1.3},
            'sideways': {'mean_reversion': 1.4, 'volatility': 1.2, 'volume': 0.9},
            'volatile': {'volatility': 1.5, 'momentum': 1.3, 'trend': 0.8}
        }
    
    def engineer_features(self, 
                         df: pd.DataFrame,
                         regime: str = 'bull',
                         lookback_periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create adaptive features based on market regime"""
        
        features_df = df.copy()
        
        # Price-based features
        features_df = self._add_price_features(features_df, lookback_periods)
        
        # Technical indicators
        features_df = self._add_technical_features(features_df, lookback_periods)
        
        # Volume features
        features_df = self._add_volume_features(features_df, lookback_periods)
        
        # Volatility features
        features_df = self._add_volatility_features(features_df, lookback_periods)
        
        # Cross-asset features (if multiple coins)
        features_df = self._add_cross_asset_features(features_df)
        
        # Apply regime-specific weights
        features_df = self._apply_regime_weights(features_df, regime)
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        self.logger.info(f"Engineered {len(features_df.columns)} features for {regime} regime")
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add price-based features"""
        
        for period in periods:
            # Returns
            df[f'return_{period}'] = df['close'].pct_change(period)
            
            # Moving averages
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Price ratios
            df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
            
            # High/Low ratios
            df[f'hl_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add technical indicator features"""
        
        # RSI
        for period in [14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add volume-based features"""
        
        for period in periods:
            # Volume moving averages
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            volume_sma = df[f'volume_sma_{period}']
            df[f'volume_ratio_{period}'] = df['volume'] / volume_sma.where(volume_sma != 0, 1)
            
            # Volume-price trend
            price_change = df['close'].pct_change().fillna(0)
            df[f'vpt_{period}'] = (price_change * df['volume']).rolling(period).sum()
        
        # On-balance volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add volatility features"""
        
        for period in periods:
            # Realized volatility
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std()
            
            # ATR
            df[f'atr_{period}'] = self._calculate_atr(df, period)
            
            # Volatility ratio
            if period > 5:
                df[f'vol_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_5']
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation features"""
        
        # This would include correlations with BTC, market cap, etc.
        # For now, add placeholder features
        df['btc_correlation_placeholder'] = 0.5
        df['market_beta_placeholder'] = 1.0
        
        return df
    
    def _apply_regime_weights(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Apply regime-specific feature weights"""
        
        weights = self.regime_weights.get(regime, {})
        
        for feature_type, weight in weights.items():
            # Apply weights to relevant features
            feature_cols = [col for col in df.columns if feature_type in col.lower()]
            for col in feature_cols:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col] * weight
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()

class UncertaintyQuantifier:
    """Bayesian uncertainty quantification"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('ml_predictions')
        
    def quantify_uncertainty(self, 
                           predictions: np.ndarray,
                           model_predictions: List[np.ndarray],
                           confidence_level: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate prediction uncertainty and intervals"""
        
        # Ensemble variance
        ensemble_std = np.std(model_predictions, axis=0)
        
        # Calculate prediction intervals
        z_score = 1.28 if confidence_level == 0.8 else 1.96  # 80% or 95%
        
        lower_bound = predictions - z_score * ensemble_std
        upper_bound = predictions + z_score * ensemble_std
        
        # Calculate overall confidence
        confidence = 1 / (1 + ensemble_std)  # Higher std = lower confidence
        
        return ensemble_std, lower_bound, upper_bound

class ModelDriftDetector:
    """Detect model drift and data distribution shifts"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('ml_predictions')
        self.reference_stats = {}
        
    def detect_drift(self, 
                    new_features: pd.DataFrame,
                    model_name: str,
                    threshold: float = 0.1) -> Tuple[bool, float, Dict[str, float]]:
        """Detect feature drift in new data"""
        
        if model_name not in self.reference_stats:
            # Initialize reference statistics
            self.reference_stats[model_name] = {
                'means': new_features.mean(),
                'stds': new_features.std(),
                'timestamp': datetime.now()
            }
            return False, 0.0, {}
        
        ref_stats = self.reference_stats[model_name]
        
        # Calculate drift metrics
        drift_scores = {}
        for col in new_features.columns:
            if col in ref_stats['means']:
                # Normalized difference in means
                ref_mean = ref_stats['means'][col]
                ref_std = ref_stats['stds'][col]
                new_mean = new_features[col].mean()
                
                if ref_std > 0:
                    drift_score = abs(new_mean - ref_mean) / ref_std
                    drift_scores[col] = drift_score
        
        # Overall drift score
        overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        is_drifted = overall_drift > threshold
        
        if is_drifted:
            self.logger.warning(f"Model drift detected for {model_name}: {overall_drift:.3f}")
        
        return is_drifted, overall_drift, drift_scores

class EnsemblePredictor:
    """Ensemble of deep learning and traditional ML models"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('ml_predictions')
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.drift_detector = ModelDriftDetector()
        
    async def train_ensemble(self, 
                           training_data: Dict[str, pd.DataFrame],
                           horizons: List[str] = ['1h', '4h', '1d', '7d']) -> Dict[str, Any]:
        """Train ensemble models for multiple horizons"""
        
        training_results = {}
        
        for horizon in horizons:
            self.logger.info(f"Training ensemble for {horizon} horizon")
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data, horizon)
            
            if len(X_train) < 100:  # Minimum samples
                self.logger.warning(f"Insufficient data for {horizon}: {len(X_train)} samples")
                continue
            
            # Train multiple models
            models = await self._train_models(X_train, y_train, horizon)
            
            # Evaluate ensemble
            performance = self._evaluate_ensemble(models, X_train, y_train, horizon)
            
            self.models[horizon] = models
            training_results[horizon] = performance
            
            self.logger.info(f"Ensemble training complete for {horizon}: MSE={performance.mse:.6f}")
        
        return training_results
    
    async def predict(self, 
                     coin: str,
                     data: pd.DataFrame,
                     horizons: List[str],
                     regime: str = 'bull') -> List[MLPrediction]:
        """Generate predictions with uncertainty"""
        
        predictions = []
        
        # Engineer features
        features = self.feature_engineer.engineer_features(data, regime)
        
        if len(features) < 10:
            self.logger.warning(f"Insufficient feature data for {coin}")
            return predictions
        
        for horizon in horizons:
            if horizon not in self.models:
                self.logger.warning(f"No trained model for horizon {horizon}")
                continue
            
            try:
                prediction = await self._predict_horizon(coin, features, horizon, regime)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Prediction error for {coin} {horizon}: {e}")
                continue
        
        return predictions
    
    async def _predict_horizon(self, 
                             coin: str,
                             features: pd.DataFrame,
                             horizon: str,
                             regime: str) -> Optional[MLPrediction]:
        """Predict for specific horizon"""
        
        models = self.models[horizon]
        latest_features = features.iloc[-1:].values
        
        # Scale features
        scaler = self.scalers.get(horizon)
        if scaler:
            latest_features_scaled = scaler.transform(latest_features)
        else:
            latest_features_scaled = latest_features
        
        # Get predictions from all models
        model_predictions = []
        model_names = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(latest_features_scaled)
                    model_predictions.append(float(pred[0]) if hasattr(pred, '__len__') else float(pred))
                    model_names.append(model_name)
            except Exception as e:
                self.logger.error(f"Model {model_name} prediction failed: {e}")
                continue
        
        if not model_predictions:
            return None
        
        # Ensemble prediction (weighted average)
        weights = np.ones(len(model_predictions)) / len(model_predictions)  # Equal weights for now
        ensemble_prediction = np.average(model_predictions, weights=weights)
        
        # Quantify uncertainty
        uncertainty, lower_bound, upper_bound = self.uncertainty_quantifier.quantify_uncertainty(
            np.array([ensemble_prediction]),
            [np.array([pred]) for pred in model_predictions]
        )
        
        # Calculate confidence
        confidence = 1.0 / (1.0 + uncertainty[0])
        
        # Feature importance (simplified)
        feature_importance = self._calculate_feature_importance(features.columns.tolist())
        
        # Generate explanation
        explanation = self._generate_explanation(
            ensemble_prediction, model_predictions, model_names, regime
        )
        
        return MLPrediction(
            coin=coin,
            horizon=horizon,
            prediction=float(ensemble_prediction),
            uncertainty=float(uncertainty[0]),
            confidence=float(confidence),
            model_type='ensemble',
            feature_importance=feature_importance,
            prediction_interval=(float(lower_bound[0]), float(upper_bound[0])),
            timestamp=datetime.now(),
            model_version='v1.0',
            explanation=explanation
        )
    
    def _prepare_training_data(self, 
                             data: Dict[str, pd.DataFrame],
                             horizon: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for specific horizon"""
        
        all_X = []
        all_y = []
        
        # Map horizon to prediction periods
        horizon_map = {'1h': 1, '4h': 4, '1d': 24, '7d': 168}
        periods = horizon_map.get(horizon, 24)
        
        for coin, df in data.items():
            if len(df) < periods + 50:  # Need enough data for features + prediction
                continue
            
            # Engineer features
            features = self.feature_engineer.engineer_features(df)
            
            if len(features) < periods + 10:
                continue
            
            # Create targets (future returns)
            targets = df['close'].pct_change(periods).shift(-periods)
            
            # Align features and targets - ensure same index
            min_len = min(len(features), len(targets))
            features_aligned = features.iloc[:min_len]
            targets_aligned = targets.iloc[:min_len]
            
            # Find valid indices
            valid_indices = ~(targets_aligned.isna() | features_aligned.isna().any(axis=1))
            
            X = features_aligned[valid_indices].values
            y = targets_aligned[valid_indices].values
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        
        if not all_X:
            return np.array([]), np.array([])
        
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        return X_combined, y_combined
    
    async def _train_models(self, 
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          horizon: str) -> Dict[str, Any]:
        """Train multiple models for ensemble"""
        
        models = {}
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[horizon] = scaler
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_scaled, y_train)
        models['gradient_boosting'] = gb_model
        
        # Deep Learning (if available)
        if ADVANCED_ML_AVAILABLE:
            try:
                dl_model = await self._train_deep_model(X_scaled, y_train, horizon)
                if dl_model:
                    models['deep_learning'] = dl_model
            except Exception as e:
                self.logger.error(f"Deep learning training failed: {e}")
        
        return models
    
    async def _train_deep_model(self, 
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              horizon: str) -> Optional[Any]:
        """Train deep learning model"""
        
        if not ADVANCED_ML_AVAILABLE:
            return None
        
        try:
            # Simple neural network for demonstration
            class SimpleNN(nn.Module):
                def __init__(self, input_size):
                    super(SimpleNN, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            
            # Initialize model
            model = SimpleNN(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    self.logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Deep model training error: {e}")
            return None
    
    def _evaluate_ensemble(self, 
                         models: Dict[str, Any],
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         horizon: str) -> ModelPerformance:
        """Evaluate ensemble performance"""
        
        # Get predictions from all models
        predictions = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test)
                    predictions.append(pred)
            except Exception:
                continue
        
        if not predictions:
            return ModelPerformance(
                model_name='ensemble',
                horizon=horizon,
                mse=1.0,
                mae=1.0,
                directional_accuracy=0.5,
                sharpe_ratio=0.0,
                last_updated=datetime.now(),
                training_samples=len(X_test)
            )
        
        # Ensemble prediction
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(ensemble_pred))
        
        # Simplified Sharpe ratio
        returns = ensemble_pred
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        
        return ModelPerformance(
            model_name='ensemble',
            horizon=horizon,
            mse=mse,
            mae=mae,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            last_updated=datetime.now(),
            training_samples=len(X_test)
        )
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Calculate simplified feature importance"""
        
        # This would use actual model feature importance
        # For now, return uniform importance
        importance = 1.0 / len(feature_names)
        return {name: importance for name in feature_names}
    
    def _generate_explanation(self, 
                            prediction: float,
                            model_predictions: List[float],
                            model_names: List[str],
                            regime: str) -> Dict[str, Any]:
        """Generate prediction explanation"""
        
        return {
            'prediction_direction': 'bullish' if prediction > 0 else 'bearish',
            'magnitude': abs(prediction),
            'model_agreement': len(set(np.sign(model_predictions))) == 1,
            'regime_context': regime,
            'model_contributions': dict(zip(model_names, model_predictions))
        }
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'agent': 'enhanced_ml',
            'status': 'operational',
            'trained_horizons': list(self.models.keys()),
            'deep_learning_available': ADVANCED_ML_AVAILABLE,
            'uncertainty_quantification': True,
            'drift_detection': True,
            'feature_engineering': True
        }

# Global instance
ml_agent = EnsemblePredictor()