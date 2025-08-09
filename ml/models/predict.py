#!/usr/bin/env python3
"""
Multi-Horizon ML Prediction Engine
Train and predict with XGBoost/LightGBM + LSTM across 1h, 24h, 7d, 30d horizons
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    TREE_MODELS_AVAILABLE = True
except ImportError:
    TREE_MODELS_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.structured_logger import get_structured_logger

class CryptoLSTM(nn.Module):
    """LSTM model with MC Dropout for uncertainty estimation"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(CryptoLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for uncertainty
        self.mc_dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x, training=False):
        """Forward pass with optional MC dropout"""
        
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take last output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply MC dropout during inference if training=True
        if training or self.training:
            lstm_out = self.mc_dropout(lstm_out)
        
        # Fully connected layers
        out = F.relu(self.fc1(lstm_out))
        
        # Batch norm (only if batch size > 1)
        if out.size(0) > 1:
            out = self.batch_norm(out)
        
        # Apply dropout again
        if training or self.training:
            out = self.mc_dropout(out)
        
        out = self.fc2(out)
        
        return out

class CryptoDataset(Dataset):
    """Dataset for LSTM training"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 24):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

class MultiHorizonPredictor:
    """Multi-horizon prediction engine with uncertainty quantification"""
    
    def __init__(self, horizons: List[int] = [1, 24, 168, 720]):  # 1h, 24h, 7d, 30d
        self.logger = get_structured_logger("MultiHorizonPredictor")
        
        # Prediction horizons (in hours)
        self.horizons = horizons
        self.horizon_names = {1: "1h", 24: "24h", 168: "7d", 720: "30d"}
        
        # Models storage
        self.tree_models = {}  # {horizon: {'xgb': model, 'lgb': model}}
        self.lstm_models = {}  # {horizon: model}
        self.scalers = {}      # {horizon: scaler}
        
        # Model parameters
        self.sequence_length = 24  # 24 hours lookback for LSTM
        self.mc_samples = 30      # Monte Carlo samples for uncertainty
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.model_dir = Path("models/multi_horizon")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare features for training"""
        
        try:
            # Sort by symbol and timestamp
            df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            
            # Feature columns (exclude target-related columns)
            feature_columns = [col for col in df.columns if not any(
                keyword in col.lower() for keyword in ['target', 'future', 'return']
            )]
            
            # Remove non-feature columns
            exclude_columns = ['symbol', 'timestamp']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]
            
            # Handle missing values
            feature_df = df[feature_columns].fillna(method='ffill').fillna(0)
            
            self.logger.info(f"Prepared {len(feature_columns)} features for {len(df)} samples")
            
            return {
                'features': feature_df.values,
                'feature_names': feature_columns,
                'symbols': df['symbol'].values,
                'timestamps': df['timestamp'].values,
                'original_df': df
            }
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return {}
    
    def create_targets(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Create price return targets for given horizon"""
        
        try:
            targets = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
                
                # Calculate future returns
                if 'price' in symbol_df.columns:
                    prices = symbol_df['price'].values
                    
                    # Forward returns
                    future_prices = np.roll(prices, -horizon)
                    future_prices[-horizon:] = np.nan  # Can't predict beyond available data
                    
                    # Calculate returns
                    returns = (future_prices - prices) / prices
                    returns = np.nan_to_num(returns, 0)  # Replace NaN with 0
                    
                    targets.extend(returns)
                else:
                    # Fallback: use random targets for development
                    targets.extend(np.random.normal(0, 0.1, len(symbol_df)))
            
            targets = np.array(targets)
            self.logger.info(f"Created targets for {horizon}h horizon: {len(targets)} samples")
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Target creation failed for horizon {horizon}: {e}")
            return np.array([])
    
    def train_tree_models(self, features: np.ndarray, targets: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Train XGBoost and LightGBM models"""
        
        if not TREE_MODELS_AVAILABLE:
            self.logger.warning("Tree models not available (XGBoost/LightGBM not installed)")
            return {}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            models = {}
            
            # Train XGBoost
            self.logger.info(f"Training XGBoost for {horizon}h horizon")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            
            # Evaluate XGBoost
            xgb_pred = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            
            models['xgb'] = {
                'model': xgb_model,
                'mae': xgb_mae,
                'predictions': xgb_pred
            }
            
            # Train LightGBM
            self.logger.info(f"Training LightGBM for {horizon}h horizon")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            
            # Evaluate LightGBM
            lgb_pred = lgb_model.predict(X_test)
            lgb_mae = mean_absolute_error(y_test, lgb_pred)
            
            models['lgb'] = {
                'model': lgb_model,
                'mae': lgb_mae,
                'predictions': lgb_pred
            }
            
            self.logger.info(f"Tree models trained for {horizon}h - XGB MAE: {xgb_mae:.4f}, LGB MAE: {lgb_mae:.4f}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Tree model training failed for horizon {horizon}: {e}")
            return {}
    
    def train_lstm_model(self, features: np.ndarray, targets: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Train LSTM model with MC Dropout"""
        
        try:
            # Prepare sequential data
            dataset = CryptoDataset(features, targets, self.sequence_length)
            
            if len(dataset) < 100:
                self.logger.warning(f"Insufficient data for LSTM training: {len(dataset)} samples")
                return {}
            
            # Split data
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size]
            )
            
            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = features.shape[1]
            model = CryptoLSTM(input_size=input_size).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            self.logger.info(f"Training LSTM for {horizon}h horizon")
            
            model.train()
            for epoch in range(50):  # Reduced epochs for faster training
                epoch_loss = 0
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_features, batch_targets in test_loader:
                            batch_features = batch_features.to(self.device)
                            batch_targets = batch_targets.to(self.device)
                            outputs = model(batch_features)
                            val_loss += criterion(outputs, batch_targets).item()
                    
                    scheduler.step(val_loss)
                    model.train()
                    
                    self.logger.info(f"Epoch {epoch}: Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}")
            
            # Evaluate model
            model.eval()
            test_predictions = []
            test_targets_list = []
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(self.device)
                    outputs = model(batch_features)
                    test_predictions.extend(outputs.cpu().numpy().flatten())
                    test_targets_list.extend(batch_targets.cpu().numpy().flatten())
            
            mae = mean_absolute_error(test_targets_list, test_predictions)
            
            self.logger.info(f"LSTM trained for {horizon}h - MAE: {mae:.4f}")
            
            return {
                'model': model,
                'mae': mae,
                'input_size': input_size
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed for horizon {horizon}: {e}")
            return {}
    
    def train_all_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models for all horizons"""
        
        start_time = time.time()
        
        try:
            # Prepare features
            feature_data = self.prepare_features(features_df)
            if not feature_data:
                raise ValueError("Feature preparation failed")
            
            features = feature_data['features']
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            training_results = {}
            
            # Train for each horizon
            for horizon in self.horizons:
                self.logger.info(f"Training models for {horizon}h horizon")
                
                # Create targets
                targets = self.create_targets(features_df, horizon)
                if len(targets) == 0:
                    continue
                
                horizon_results = {}
                
                # Train tree models
                tree_results = self.train_tree_models(features_scaled, targets, horizon)
                if tree_results:
                    self.tree_models[horizon] = tree_results
                    horizon_results['tree_models'] = {
                        model_type: {'mae': results['mae']} 
                        for model_type, results in tree_results.items()
                    }
                
                # Train LSTM
                lstm_results = self.train_lstm_model(features_scaled, targets, horizon)
                if lstm_results:
                    self.lstm_models[horizon] = lstm_results
                    horizon_results['lstm'] = {'mae': lstm_results['mae']}
                
                # Store scaler
                self.scalers[horizon] = scaler
                
                training_results[f"{horizon}h"] = horizon_results
            
            training_time = time.time() - start_time
            
            # Save models
            await self.save_models()
            
            summary = {
                'success': True,
                'training_time': training_time,
                'horizons_trained': list(self.horizons),
                'models_per_horizon': {
                    f"{h}h": {
                        'tree_models': len(self.tree_models.get(h, {})),
                        'lstm': 1 if h in self.lstm_models else 0
                    }
                    for h in self.horizons
                },
                'detailed_results': training_results
            }
            
            self.logger.info(f"Model training completed in {training_time:.2f}s")
            
            return summary
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': training_time
            }
    
    def predict_with_uncertainty(self, features: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty estimation"""
        
        try:
            predictions = []
            uncertainties = []
            
            # Scale features
            scaler = self.scalers.get(horizon)
            if scaler is None:
                raise ValueError(f"No scaler found for horizon {horizon}")
            
            features_scaled = scaler.transform(features)
            
            # Tree model predictions (ensemble variance)
            tree_preds = []
            tree_models = self.tree_models.get(horizon, {})
            
            for model_type, model_data in tree_models.items():
                model = model_data['model']
                pred = model.predict(features_scaled)
                tree_preds.append(pred)
            
            # LSTM predictions (MC Dropout)
            lstm_model_data = self.lstm_models.get(horizon)
            lstm_preds = []
            
            if lstm_model_data and len(features_scaled) >= self.sequence_length:
                model = lstm_model_data['model']
                model.eval()
                
                # Prepare sequential data
                dataset = CryptoDataset(features_scaled, np.zeros(len(features_scaled)), self.sequence_length)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
                
                # MC Dropout sampling
                for _ in range(self.mc_samples):
                    batch_preds = []
                    with torch.no_grad():
                        for batch_features, _ in dataloader:
                            batch_features = batch_features.to(self.device)
                            # Enable training mode for MC dropout
                            outputs = model(batch_features, training=True)
                            batch_preds.extend(outputs.cpu().numpy().flatten())
                    
                    if len(batch_preds) > 0:
                        lstm_preds.append(batch_preds)
                
                # Convert to proper shape
                if lstm_preds:
                    lstm_preds = np.array(lstm_preds).T  # Shape: (n_samples, mc_samples)
            
            # Combine predictions
            for i in range(len(features_scaled)):
                sample_preds = []
                
                # Add tree predictions
                for tree_pred in tree_preds:
                    if i < len(tree_pred):
                        sample_preds.append(tree_pred[i])
                
                # Add LSTM predictions if available
                if len(lstm_preds) > 0 and i < len(lstm_preds):
                    sample_preds.extend(lstm_preds[i])
                
                if sample_preds:
                    # Mean prediction
                    pred_mean = np.mean(sample_preds)
                    # Uncertainty as standard deviation
                    pred_std = np.std(sample_preds) if len(sample_preds) > 1 else 0.1
                    
                    predictions.append(pred_mean)
                    uncertainties.append(pred_std)
                else:
                    # Fallback
                    predictions.append(0.0)
                    uncertainties.append(0.1)
            
            return np.array(predictions), np.array(uncertainties)
            
        except Exception as e:
            self.logger.error(f"Prediction failed for horizon {horizon}: {e}")
            # Return zeros as fallback
            return np.zeros(len(features)), np.full(len(features), 0.1)
    
    def predict_all(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Main prediction interface - predict all horizons for all coins"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting prediction for {len(features_df)} samples")
            
            # Load models if not already loaded
            if not self.tree_models and not self.lstm_models:
                self.load_models()
            
            # Prepare features
            feature_data = self.prepare_features(features_df)
            if not feature_data:
                raise ValueError("Feature preparation failed")
            
            features = feature_data['features']
            symbols = feature_data['symbols']
            timestamps = feature_data['timestamps']
            
            # Initialize result dataframe
            result_df = pd.DataFrame({
                'coin': symbols,
                'timestamp': timestamps
            })
            
            # Predict for each horizon
            for horizon in self.horizons:
                horizon_name = self.horizon_names.get(horizon, f"{horizon}h")
                
                self.logger.info(f"Predicting for {horizon_name} horizon")
                
                try:
                    predictions, uncertainties = self.predict_with_uncertainty(features, horizon)
                    
                    # Convert uncertainty to confidence (0-1 scale)
                    # Higher uncertainty = lower confidence
                    max_uncertainty = max(np.max(uncertainties), 0.1)
                    confidences = 1.0 - (uncertainties / max_uncertainty)
                    confidences = np.clip(confidences, 0.1, 0.99)  # Keep in reasonable range
                    
                    # Add to result dataframe
                    result_df[f'pred_{horizon}'] = predictions
                    result_df[f'conf_{horizon}'] = confidences
                    
                    self.logger.info(f"Completed {horizon_name}: mean pred={np.mean(predictions):.4f}, mean conf={np.mean(confidences):.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to predict for horizon {horizon}: {e}")
                    # Add fallback values
                    result_df[f'pred_{horizon}'] = 0.0
                    result_df[f'conf_{horizon}'] = 0.1
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Prediction completed in {processing_time:.2f}s for {len(result_df)} samples")
            
            return result_df
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Prediction failed: {e}")
            
            # Return empty dataframe with proper columns
            columns = ['coin', 'timestamp']
            for horizon in self.horizons:
                columns.extend([f'pred_{horizon}', f'conf_{horizon}'])
            
            return pd.DataFrame(columns=columns)
    
    async def save_models(self):
        """Save trained models to disk"""
        
        try:
            # Save tree models
            for horizon, models in self.tree_models.items():
                for model_type, model_data in models.items():
                    model_path = self.model_dir / f"{model_type}_{horizon}h.pkl"
                    joblib.dump(model_data['model'], model_path)
            
            # Save LSTM models
            for horizon, model_data in self.lstm_models.items():
                model_path = self.model_dir / f"lstm_{horizon}h.pth"
                torch.save(model_data['model'].state_dict(), model_path)
                
                # Save metadata
                meta_path = self.model_dir / f"lstm_{horizon}h_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'input_size': model_data['input_size'],
                        'mae': model_data['mae']
                    }, f)
            
            # Save scalers
            for horizon, scaler in self.scalers.items():
                scaler_path = self.model_dir / f"scaler_{horizon}h.pkl"
                joblib.dump(scaler, scaler_path)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        
        try:
            # Load tree models
            for horizon in self.horizons:
                tree_models = {}
                
                for model_type in ['xgb', 'lgb']:
                    model_path = self.model_dir / f"{model_type}_{horizon}h.pkl"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        tree_models[model_type] = {'model': model}
                
                if tree_models:
                    self.tree_models[horizon] = tree_models
                
                # Load LSTM model
                model_path = self.model_dir / f"lstm_{horizon}h.pth"
                meta_path = self.model_dir / f"lstm_{horizon}h_meta.json"
                
                if model_path.exists() and meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    model = CryptoLSTM(input_size=meta['input_size']).to(self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    
                    self.lstm_models[horizon] = {
                        'model': model,
                        'input_size': meta['input_size'],
                        'mae': meta['mae']
                    }
                
                # Load scaler
                scaler_path = self.model_dir / f"scaler_{horizon}h.pkl"
                if scaler_path.exists():
                    self.scalers[horizon] = joblib.load(scaler_path)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")

# Global predictor instance
_predictor: Optional[MultiHorizonPredictor] = None

def get_predictor() -> MultiHorizonPredictor:
    """Get global predictor instance"""
    global _predictor
    
    if _predictor is None:
        _predictor = MultiHorizonPredictor()
    
    return _predictor

def predict_all(features_df: pd.DataFrame) -> pd.DataFrame:
    """Main prediction interface - predict all horizons for all coins"""
    predictor = get_predictor()
    return predictor.predict_all(features_df)

def train_models(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Train all models for all horizons"""
    predictor = get_predictor()
    return predictor.train_all_models(features_df)

# Convenience functions for testing
def create_mock_features(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create mock feature data for testing"""
    
    symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'] * (n_samples // 5)
    timestamps = pd.date_range('2025-01-01', periods=n_samples, freq='H')
    
    # Generate features
    data = {
        'symbol': symbols[:n_samples],
        'timestamp': timestamps[:n_samples],
        'price': np.random.uniform(50, 150, n_samples)
    }
    
    # Add technical features
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test prediction system
    import asyncio
    
    def test_prediction():
        print("Testing Multi-Horizon Prediction System")
        
        # Create mock data
        features_df = create_mock_features(500, 15)
        print(f"Created mock data: {len(features_df)} samples, {len(features_df.columns)} features")
        
        # Train models
        print("Training models...")
        training_results = train_models(features_df)
        print(f"Training results: {training_results.get('success', False)}")
        
        # Make predictions
        print("Making predictions...")
        predictions = predict_all(features_df)
        print(f"Predictions: {len(predictions)} samples, {len(predictions.columns)} columns")
        print(predictions.head())
        
        return predictions
    
    test_prediction()