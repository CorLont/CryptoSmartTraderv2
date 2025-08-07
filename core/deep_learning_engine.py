"""
CryptoSmartTrader V2 - Deep Learning Engine
State-of-the-art deep learning voor time series forecasting met GPU-versnelling
"""

import logging
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - deep learning features disabled")
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import sys
import json
from dataclasses import dataclass
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class ModelConfig:
    """Deep learning model configuration"""
    model_type: str  # lstm, gru, transformer, nbeats
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    sequence_length: int
    prediction_horizon: int
    learning_rate: float
    batch_size: int
    epochs: int
    use_gpu: bool

if TORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        """LSTM-based cryptocurrency forecaster"""
        
        def __init__(self, config: ModelConfig):
            super(LSTMForecaster, self).__init__()
            self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon),
            nn.Softplus()  # Ensure positive values for uncertainty
        )
    
    def forward(self, x):
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step for prediction
        last_output = attn_out[:, -1, :]
        
        # Predictions and uncertainty
        predictions = self.prediction_head(last_output)
        uncertainties = self.uncertainty_head(last_output)
        
        return predictions, uncertainties

    class TransformerForecaster(nn.Module):
        """Transformer-based cryptocurrency forecaster"""
        
        def __init__(self, config: ModelConfig):
            super(TransformerForecaster, self).__init__()
            self.config = config
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.input_size, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_size,
            nhead=8,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.prediction_horizon)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.prediction_horizon),
            nn.Softplus()
        )
    
    def forward(self, x):
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use last time step
        last_output = transformer_out[:, -1, :]
        
        # Predictions and uncertainty
        predictions = self.prediction_head(last_output)
        uncertainties = self.uncertainty_head(last_output)
        
        return predictions, uncertainties

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer"""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

    class NBEATSForecaster(nn.Module):
        """N-BEATS neural forecasting model"""
        
        def __init__(self, config: ModelConfig):
            super(NBEATSForecaster, self).__init__()
            self.config = config
        
        # Stack configuration
        self.trend_stack = NBEATSStack(
            config.input_size, 
            config.hidden_size, 
            config.sequence_length,
            config.prediction_horizon,
            'trend'
        )
        
        self.seasonality_stack = NBEATSStack(
            config.input_size,
            config.hidden_size,
            config.sequence_length, 
            config.prediction_horizon,
            'seasonality'
        )
        
        self.generic_stack = NBEATSStack(
            config.input_size,
            config.hidden_size,
            config.sequence_length,
            config.prediction_horizon,
            'generic'
        )
    
    def forward(self, x):
        # Process through stacks
        residual = x
        forecast = torch.zeros(x.size(0), self.config.prediction_horizon).to(x.device)
        
        # Trend stack
        backcast_trend, forecast_trend = self.trend_stack(residual)
        residual = residual - backcast_trend
        forecast = forecast + forecast_trend
        
        # Seasonality stack
        backcast_season, forecast_season = self.seasonality_stack(residual)
        residual = residual - backcast_season
        forecast = forecast + forecast_season
        
        # Generic stack
        backcast_generic, forecast_generic = self.generic_stack(residual)
        forecast = forecast + forecast_generic
        
        # Simple uncertainty estimation (could be improved)
        uncertainties = torch.abs(forecast) * 0.1
        
        return forecast, uncertainties

    class NBEATSStack(nn.Module):
        """N-BEATS stack component"""
        
        def __init__(self, input_size, hidden_size, backcast_length, forecast_length, stack_type):
            super().__init__()
            self.backcast_length = backcast_length
            self.forecast_length = forecast_length
            self.stack_type = stack_type
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Output layers
        if stack_type == 'trend':
            self.theta_b = nn.Linear(hidden_size, 2)  # Linear trend
            self.theta_f = nn.Linear(hidden_size, 2)
        elif stack_type == 'seasonality':
            self.theta_b = nn.Linear(hidden_size, backcast_length)
            self.theta_f = nn.Linear(hidden_size, forecast_length)
        else:  # generic
            self.theta_b = nn.Linear(hidden_size, backcast_length)
            self.theta_f = nn.Linear(hidden_size, forecast_length)
    
    def forward(self, x):
        # Forward pass through FC layers
        h = torch.relu(self.fc1(x.mean(dim=1)))  # Pool sequence dimension
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        
        # Generate theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        
        if self.stack_type == 'trend':
            # Linear trend basis
            t_b = torch.linspace(0, 1, self.backcast_length).to(x.device)
            t_f = torch.linspace(0, 1, self.forecast_length).to(x.device)
            
            backcast = theta_b[:, 0:1] + theta_b[:, 1:2] * t_b
            forecast = theta_f[:, 0:1] + theta_f[:, 1:2] * t_f
        else:
            # Direct output
            backcast = theta_b
            forecast = theta_f
        
        return backcast, forecast
else:
    # Placeholder classes when PyTorch is not available
    class LSTMForecaster:
        def __init__(self, config): pass
    class TransformerForecaster:
        def __init__(self, config): pass
    class NBEATSForecaster:
        def __init__(self, config): pass
    class PositionalEncoding:
        def __init__(self, *args, **kwargs): pass
    class NBEATSStack:
        def __init__(self, *args, **kwargs): pass

class DeepLearningEngine:
    """Main deep learning engine for cryptocurrency forecasting"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - deep learning features disabled")
            self.device = "cpu"
            self.torch_available = False
            return
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_available = True
        self.logger.info(f"Deep learning engine initialized on device: {self.device}")
        
        # Model configurations (only create if PyTorch available)
        if not self.torch_available:
            self.model_configs = {}
            self.trained_models = {}
            self.model_performance = {}
            self.is_training = False
            self.training_thread = None
            return
            
        self.model_configs = {
            'lstm': ModelConfig(
                model_type='lstm',
                input_size=20,
                hidden_size=128,
                num_layers=3,
                dropout=0.2,
                sequence_length=60,
                prediction_horizon=24,
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                use_gpu=TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
            ),
            'transformer': ModelConfig(
                model_type='transformer',
                input_size=20,
                hidden_size=256,
                num_layers=4,
                dropout=0.1,
                sequence_length=60,
                prediction_horizon=24,
                learning_rate=0.0001,
                batch_size=16,
                epochs=50,
                use_gpu=TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
            ),
            'nbeats': ModelConfig(
                model_type='nbeats',
                input_size=20,
                hidden_size=512,
                num_layers=2,
                dropout=0.1,
                sequence_length=60,
                prediction_horizon=24,
                learning_rate=0.001,
                batch_size=32,
                epochs=200,
                use_gpu=TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
            )
        }
        
        # Trained models storage
        self.trained_models = {}
        self.model_performance = {}
        
        # Training state
        self.is_training = False
        self.training_thread = None
    
    def create_model(self, model_type: str, config: ModelConfig):
        """Create deep learning model"""
        if not self.torch_available:
            raise ValueError("PyTorch not available - cannot create deep learning models")
        try:
            if model_type == 'lstm':
                model = LSTMForecaster(config)
            elif model_type == 'transformer':
                model = TransformerForecaster(config)
            elif model_type == 'nbeats':
                model = NBEATSForecaster(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Move to device
            model = model.to(self.device)
            
            self.logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model creation failed for {model_type}: {e}")
            raise
    
    def prepare_sequences(self, data: pd.DataFrame, config: ModelConfig):
        """Prepare sequences for training"""
        try:
            sequences = []
            targets = []
            
            # Feature columns (excluding target)
            feature_cols = [col for col in data.columns if col != 'target']
            
            for i in range(len(data) - config.sequence_length - config.prediction_horizon + 1):
                # Input sequence
                seq = data[feature_cols].iloc[i:i + config.sequence_length].values
                
                # Target sequence
                target = data['target'].iloc[i + config.sequence_length:i + config.sequence_length + config.prediction_horizon].values
                
                sequences.append(seq)
                targets.append(target)
            
            # Convert to tensors
            if TORCH_AVAILABLE:
                X = torch.FloatTensor(np.array(sequences))
                y = torch.FloatTensor(np.array(targets))
            else:
                X = np.array(sequences)
                y = np.array(targets)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Sequence preparation failed: {e}")
            raise
    
    def train_model(self, coin: str, model_type: str, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train deep learning model"""
        try:
            if not self.torch_available:
                return {'success': False, 'error': 'PyTorch not available'}
                
            self.logger.info(f"Training {model_type} model for {coin}")
            
            config = self.model_configs[model_type]
            
            # Create model
            model = self.create_model(model_type, config)
            
            # Prepare data
            X, y = self.prepare_sequences(training_data, config)
            
            # Split train/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Move to device
            X_train, X_val = X_train.to(self.device), X_val.to(self.device)
            y_train, y_val = y_train.to(self.device), y_val.to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Optimizer and loss
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    predictions, uncertainties = model(batch_X)
                    
                    # Uncertainty-weighted loss
                    mse_loss = nn.MSELoss()(predictions, batch_y)
                    uncertainty_loss = torch.mean(uncertainties)
                    
                    loss = mse_loss + 0.1 * uncertainty_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        predictions, uncertainties = model(batch_X)
                        loss = nn.MSELoss()(predictions, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step(val_loss)
                
                training_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    model_key = f"{coin}_{model_type}"
                    self.trained_models[model_key] = {
                        'model': model.state_dict(),
                        'config': config,
                        'performance': {
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'epoch': epoch
                        }
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= 20:  # Early stopping
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Store performance
            model_key = f"{coin}_{model_type}"
            self.model_performance[model_key] = {
                'best_val_loss': best_val_loss,
                'training_history': training_history,
                'total_epochs': epoch + 1,
                'model_type': model_type,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Training completed for {coin} {model_type}: best_val_loss={best_val_loss:.6f}")
            
            return {
                'success': True,
                'model_key': model_key,
                'best_val_loss': best_val_loss,
                'total_epochs': epoch + 1
            }
            
        except Exception as e:
            self.logger.error(f"Training failed for {coin} {model_type}: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, coin: str, model_type: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with trained model"""
        try:
            if not self.torch_available:
                return {'success': False, 'error': 'PyTorch not available'}
                
            model_key = f"{coin}_{model_type}"
            
            if model_key not in self.trained_models:
                return {'success': False, 'error': 'Model not trained'}
            
            # Load model
            model_info = self.trained_models[model_key]
            config = model_info['config']
            
            model = self.create_model(model_type, config)
            model.load_state_dict(model_info['model'])
            model.eval()
            
            # Prepare input
            X, _ = self.prepare_sequences(input_data, config)
            X = X[-1:].to(self.device)  # Last sequence only
            
            # Predict
            with torch.no_grad():
                predictions, uncertainties = model(X)
                
                predictions = predictions.cpu().numpy().flatten()
                uncertainties = uncertainties.cpu().numpy().flatten()
            
            # Calculate confidence intervals
            confidence_intervals = []
            for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
                lower = pred - 1.96 * unc
                upper = pred + 1.96 * unc
                confidence_intervals.append({'lower': lower, 'upper': upper, 'std': unc})
            
            return {
                'success': True,
                'predictions': predictions.tolist(),
                'uncertainties': uncertainties.tolist(),
                'confidence_intervals': confidence_intervals,
                'model_type': model_type,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {coin} {model_type}: {e}")
            return {'success': False, 'error': str(e)}
    
    def ensemble_predict(self, coin: str, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Ensemble prediction using multiple models"""
        try:
            predictions = {}
            weights = {'lstm': 0.4, 'transformer': 0.35, 'nbeats': 0.25}
            
            total_weight = 0
            ensemble_pred = None
            ensemble_unc = None
            
            for model_type, weight in weights.items():
                result = self.predict(coin, model_type, input_data)
                
                if result['success']:
                    pred = np.array(result['predictions'])
                    unc = np.array(result['uncertainties'])
                    
                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                        ensemble_unc = weight * unc
                    else:
                        ensemble_pred += weight * pred
                        ensemble_unc += weight * unc
                    
                    total_weight += weight
                    predictions[model_type] = result
            
            if total_weight > 0:
                ensemble_pred /= total_weight
                ensemble_unc /= total_weight
                
                return {
                    'success': True,
                    'ensemble_predictions': ensemble_pred.tolist(),
                    'ensemble_uncertainties': ensemble_unc.tolist(),
                    'individual_predictions': predictions,
                    'total_weight': total_weight,
                    'timestamp': datetime.now()
                }
            else:
                return {'success': False, 'error': 'No models available for ensemble'}
                
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed for {coin}: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_batch_training(self, coins: List[str] = None):
        """Start batch training for multiple coins"""
        if self.is_training:
            self.logger.warning("Training already in progress")
            return
        
        self.is_training = True
        self.training_thread = threading.Thread(target=self._batch_training_loop, args=(coins,), daemon=True)
        self.training_thread.start()
        
        self.logger.info("Started batch training process")
    
    def _batch_training_loop(self, coins: List[str] = None):
        """Batch training loop"""
        try:
            if coins is None:
                # Get coins from cache
                discovered_coins = self.cache_manager.get("discovered_coins")
                coins = list(discovered_coins.keys())[:20] if discovered_coins else ['BTC', 'ETH']
            
            for coin in coins:
                if not self.is_training:
                    break
                
                try:
                    # Get training data
                    training_data = self._prepare_training_data(coin)
                    
                    if training_data is not None and len(training_data) > 200:
                        # Train all model types
                        for model_type in ['lstm', 'transformer', 'nbeats']:
                            if not self.is_training:
                                break
                            
                            result = self.train_model(coin, model_type, training_data)
                            
                            if result['success']:
                                self.logger.info(f"Successfully trained {model_type} for {coin}")
                            else:
                                self.logger.error(f"Failed to train {model_type} for {coin}: {result.get('error')}")
                            
                            time.sleep(1)  # Brief pause between models
                    else:
                        self.logger.warning(f"Insufficient training data for {coin}")
                        
                except Exception as e:
                    self.logger.error(f"Training failed for {coin}: {e}")
                
                time.sleep(5)  # Pause between coins
            
        except Exception as e:
            self.logger.error(f"Batch training loop failed: {e}")
        finally:
            self.is_training = False
            self.logger.info("Batch training completed")
    
    def _prepare_training_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Prepare training data for a coin"""
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
            
            # Technical indicators (simplified)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price features
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Target: next day return
            df['target'] = df['returns'].shift(-1)
            
            # Select features
            feature_columns = [
                'returns', 'log_returns', 'volatility', 'sma_20', 'sma_50',
                'rsi', 'volume_ratio', 'high_low_ratio', 'close_open_ratio'
            ]
            
            # Normalize features
            for col in feature_columns:
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Add target column
            feature_columns.append('target')
            
            # Clean data
            result_df = df[feature_columns].dropna()
            
            return result_df if len(result_df) > 100 else None
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed for {coin}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'trained_models': len(self.trained_models),
            'model_performance': self.model_performance,
            'device': str(self.device),
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'torch_available': self.torch_available if hasattr(self, 'torch_available') else False,
            'timestamp': datetime.now()
        }
    
    def stop_training(self):
        """Stop training process"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        
        self.logger.info("Training process stopped")
    
    def save_models(self, filepath: str = "models/deep_learning_models.pkl"):
        """Save trained models to disk"""
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            save_data = {
                'trained_models': self.trained_models,
                'model_performance': self.model_performance,
                'model_configs': self.model_configs,
                'timestamp': datetime.now()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self, filepath: str = "models/deep_learning_models.pkl"):
        """Load trained models from disk"""
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"Model file not found: {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.trained_models = save_data.get('trained_models', {})
            self.model_performance = save_data.get('model_performance', {})
            
            self.logger.info(f"Loaded {len(self.trained_models)} models from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")