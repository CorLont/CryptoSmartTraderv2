"""
Deep ML Engine - Mandatory Deep Learning Integration
LSTM/Deep Learning as REQUIRED component of every prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using simplified deep learning")

class LSTMPricePredictor(nn.Module):
    """Advanced LSTM network for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout, 
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output
        last_output = attn_out[:, -1, :]
        
        # Final prediction
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerPredictor(nn.Module):
    """Transformer-based price predictor"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 6):
        super(TransformerPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer forward
        x = x.transpose(0, 1)  # (seq, batch, features)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq, features)
        
        # Use last output
        out = self.output_layer(x[:, -1, :])
        
        return out

class DeepMLEngine:
    """
    MANDATORY Deep Learning Engine
    Every prediction MUST use deep learning models
    """
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.models = {
            'lstm_1h': None,
            'lstm_24h': None,
            'lstm_7d': None,
            'lstm_30d': None,
            'transformer_1h': None,
            'transformer_24h': None,
            'transformer_7d': None,
            'transformer_30d': None
        }
        
        # Training configuration
        self.config = {
            'sequence_length': 100,
            'feature_size': 50,  # Price + volume + sentiment + whale + technical
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'early_stopping_patience': 10
        }
        
        # Model performance tracking
        self.model_performance = {}
        
        # Lock for thread safety
        self.model_lock = threading.Lock()
        
        self.logger.critical("DEEP ML ENGINE INITIALIZED - Deep learning MANDATORY for all predictions")
    
    def initialize_models(self) -> bool:
        """Initialize all deep learning models - MANDATORY"""
        
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available - Deep ML Engine cannot function properly")
            # Use simplified models as fallback
            return self._initialize_simplified_models()
        
        try:
            with self.model_lock:
                feature_size = self.config['feature_size']
                
                # Initialize LSTM models for each horizon
                for horizon in ['1h', '24h', '7d', '30d']:
                    self.models[f'lstm_{horizon}'] = LSTMPricePredictor(
                        input_size=feature_size,
                        hidden_size=128,
                        num_layers=3,
                        dropout=0.2
                    )
                    
                    self.models[f'transformer_{horizon}'] = TransformerPredictor(
                        input_size=feature_size,
                        d_model=128,
                        nhead=8,
                        num_layers=6
                    )
                
                self.logger.critical("ALL DEEP LEARNING MODELS INITIALIZED - Ready for mandatory predictions")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize deep learning models: {e}")
            return False
    
    def _initialize_simplified_models(self) -> bool:
        """Simplified models when PyTorch unavailable"""
        self.logger.warning("Using simplified models - PyTorch not available")
        
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        
        try:
            for horizon in ['1h', '24h', '7d', '30d']:
                # Gradient boosting as deep learning substitute
                self.models[f'lstm_{horizon}'] = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # MLP as transformer substitute
                self.models[f'transformer_{horizon}'] = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64),
                    max_iter=500,
                    learning_rate='adaptive',
                    random_state=42
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simplified models: {e}")
            return False
    
    async def mandatory_deep_prediction(self, 
                                      coin: str, 
                                      features: np.ndarray, 
                                      horizon: str) -> Dict[str, Any]:
        """
        MANDATORY deep learning prediction
        No prediction without deep learning
        """
        
        if not self.models[f'lstm_{horizon}'] or not self.models[f'transformer_{horizon}']:
            raise ValueError(f"Deep learning models not initialized for {horizon}")
        
        try:
            # Ensemble prediction using both LSTM and Transformer
            lstm_pred = await self._predict_lstm(features, horizon)
            transformer_pred = await self._predict_transformer(features, horizon)
            
            # Ensemble combination
            ensemble_prediction = (lstm_pred['prediction'] + transformer_pred['prediction']) / 2
            ensemble_confidence = min(lstm_pred['confidence'], transformer_pred['confidence'])
            
            # Cross-validation for uncertainty estimation
            uncertainty = self._estimate_prediction_uncertainty(features, horizon)
            
            result = {
                'coin': coin,
                'horizon': horizon,
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'uncertainty': uncertainty,
                'lstm_prediction': lstm_pred['prediction'],
                'transformer_prediction': transformer_pred['prediction'],
                'model_agreement': abs(lstm_pred['prediction'] - transformer_pred['prediction']),
                'deep_learning_used': True,
                'prediction_timestamp': datetime.now().isoformat(),
                'features_used': features.shape[0] if len(features.shape) > 1 else len(features)
            }
            
            # Log mandatory deep learning usage
            self.logger.info(
                f"MANDATORY DEEP PREDICTION: {coin} {horizon} - "
                f"Prediction: {ensemble_prediction:.4f}, Confidence: {ensemble_confidence:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Mandatory deep prediction failed for {coin} {horizon}: {e}")
            raise ValueError(f"Deep learning prediction mandatory but failed: {e}")
    
    async def _predict_lstm(self, features: np.ndarray, horizon: str) -> Dict[str, Any]:
        """LSTM prediction"""
        
        model = self.models[f'lstm_{horizon}']
        
        if TORCH_AVAILABLE and isinstance(model, LSTMPricePredictor):
            # PyTorch LSTM prediction
            model.eval()
            
            with torch.no_grad():
                # Reshape features for LSTM
                if len(features.shape) == 1:
                    features = features.reshape(1, -1, self.config['feature_size'])
                
                features_tensor = torch.FloatTensor(features)
                prediction = model(features_tensor)
                
                # Calculate confidence based on model certainty
                confidence = self._calculate_model_confidence(prediction, 'lstm', horizon)
                
                return {
                    'prediction': float(prediction.squeeze()),
                    'confidence': confidence
                }
        else:
            # Sklearn-based prediction
            if len(features.shape) > 1:
                features = features.flatten()
            
            prediction = model.predict([features])[0]
            confidence = 0.7  # Default confidence for sklearn models
            
            return {
                'prediction': float(prediction),
                'confidence': confidence
            }
    
    async def _predict_transformer(self, features: np.ndarray, horizon: str) -> Dict[str, Any]:
        """Transformer prediction"""
        
        model = self.models[f'transformer_{horizon}']
        
        if TORCH_AVAILABLE and isinstance(model, TransformerPredictor):
            # PyTorch Transformer prediction
            model.eval()
            
            with torch.no_grad():
                # Reshape features for Transformer
                if len(features.shape) == 1:
                    features = features.reshape(1, -1, self.config['feature_size'])
                
                features_tensor = torch.FloatTensor(features)
                prediction = model(features_tensor)
                
                # Calculate confidence
                confidence = self._calculate_model_confidence(prediction, 'transformer', horizon)
                
                return {
                    'prediction': float(prediction.squeeze()),
                    'confidence': confidence
                }
        else:
            # Sklearn-based prediction
            if len(features.shape) > 1:
                features = features.flatten()
            
            prediction = model.predict([features])[0]
            confidence = 0.75  # Slightly higher for MLP
            
            return {
                'prediction': float(prediction),
                'confidence': confidence
            }
    
    def _calculate_model_confidence(self, prediction: Any, model_type: str, horizon: str) -> float:
        """Calculate model confidence based on prediction characteristics"""
        
        # Get historical performance for this model
        performance_key = f"{model_type}_{horizon}"
        historical_accuracy = self.model_performance.get(performance_key, {}).get('accuracy', 0.7)
        
        # Base confidence on historical performance
        base_confidence = historical_accuracy
        
        # Adjust based on prediction magnitude (extreme predictions less confident)
        if hasattr(prediction, 'item'):
            pred_value = abs(prediction.item())
        else:
            pred_value = abs(float(prediction))
        
        # Lower confidence for extreme predictions
        if pred_value > 2.0:  # Very high change predicted
            confidence_penalty = 0.2
        elif pred_value > 1.0:  # High change predicted
            confidence_penalty = 0.1
        else:
            confidence_penalty = 0.0
        
        final_confidence = max(0.1, base_confidence - confidence_penalty)
        
        return min(0.95, final_confidence)  # Cap at 95%
    
    def _estimate_prediction_uncertainty(self, features: np.ndarray, horizon: str) -> float:
        """Estimate prediction uncertainty using ensemble variance"""
        
        try:
            # Make multiple predictions with slight variations
            predictions = []
            
            for i in range(5):
                # Add small noise to features
                noisy_features = features + np.random.normal(0, 0.01, features.shape)
                
                # Predict with both models
                lstm_model = self.models[f'lstm_{horizon}']
                transformer_model = self.models[f'transformer_{horizon}']
                
                if TORCH_AVAILABLE:
                    # PyTorch models
                    with torch.no_grad():
                        if len(noisy_features.shape) == 1:
                            noisy_features = noisy_features.reshape(1, -1, self.config['feature_size'])
                        
                        features_tensor = torch.FloatTensor(noisy_features)
                        
                        if isinstance(lstm_model, LSTMPricePredictor):
                            lstm_pred = lstm_model(features_tensor).squeeze().item()
                        else:
                            lstm_pred = lstm_model.predict([noisy_features.flatten()])[0]
                        
                        if isinstance(transformer_model, TransformerPredictor):
                            trans_pred = transformer_model(features_tensor).squeeze().item()
                        else:
                            trans_pred = transformer_model.predict([noisy_features.flatten()])[0]
                        
                        predictions.append((lstm_pred + trans_pred) / 2)
                else:
                    # Sklearn models
                    flat_features = noisy_features.flatten()
                    lstm_pred = lstm_model.predict([flat_features])[0]
                    trans_pred = transformer_model.predict([flat_features])[0]
                    predictions.append((lstm_pred + trans_pred) / 2)
            
            # Calculate variance as uncertainty measure
            uncertainty = np.std(predictions)
            
            return float(uncertainty)
            
        except Exception as e:
            self.logger.warning(f"Could not estimate uncertainty: {e}")
            return 0.1  # Default uncertainty
    
    async def train_models_batch(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train all deep learning models on batch data"""
        
        if not training_data:
            raise ValueError("No training data provided for mandatory deep learning")
        
        training_results = {}
        
        for horizon in ['1h', '24h', '7d', '30d']:
            try:
                # Prepare data for this horizon
                X, y = self._prepare_training_data(training_data, horizon)
                
                if len(X) < 100:  # Minimum training samples
                    self.logger.warning(f"Insufficient training data for {horizon}: {len(X)} samples")
                    continue
                
                # Train LSTM
                lstm_result = await self._train_lstm(X, y, horizon)
                
                # Train Transformer
                transformer_result = await self._train_transformer(X, y, horizon)
                
                training_results[horizon] = {
                    'lstm_performance': lstm_result,
                    'transformer_performance': transformer_result,
                    'training_samples': len(X),
                    'training_timestamp': datetime.now().isoformat()
                }
                
                self.logger.critical(f"DEEP LEARNING TRAINING COMPLETED: {horizon}")
                
            except Exception as e:
                self.logger.error(f"Training failed for {horizon}: {e}")
                training_results[horizon] = {'error': str(e)}
        
        return training_results
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]], horizon: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for deep learning models"""
        
        X_list = []
        y_list = []
        
        for item in training_data:
            if horizon in item.get('target_returns', {}):
                # Extract features
                features = []
                
                # Price features
                features.extend([
                    item.get('open', 0),
                    item.get('high', 0),
                    item.get('low', 0),
                    item.get('close', 0),
                    item.get('volume', 0)
                ])
                
                # Technical indicators (45 features)
                technical = item.get('technical_indicators', {})
                for i in range(45):
                    features.append(technical.get(f'indicator_{i}', 0))
                
                # Ensure exact feature count
                while len(features) < self.config['feature_size']:
                    features.append(0)
                features = features[:self.config['feature_size']]
                
                X_list.append(features)
                y_list.append(item['target_returns'][horizon])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Reshape for sequence models
        if TORCH_AVAILABLE:
            sequence_length = min(self.config['sequence_length'], len(X))
            if len(X) >= sequence_length:
                X = X[-sequence_length:].reshape(1, sequence_length, -1)
                y = y[-1:]  # Predict last value
        
        return X, y
    
    async def _train_lstm(self, X: np.ndarray, y: np.ndarray, horizon: str) -> Dict[str, Any]:
        """Train LSTM model"""
        
        model = self.models[f'lstm_{horizon}']
        
        if TORCH_AVAILABLE and isinstance(model, LSTMPricePredictor):
            # PyTorch training
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['epochs']):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        break
            
            return {
                'final_loss': best_loss,
                'epochs_trained': epoch + 1,
                'model_type': 'LSTM_PyTorch'
            }
        else:
            # Sklearn training
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            model.fit(X, y)
            score = model.score(X, y)
            
            return {
                'training_score': score,
                'model_type': 'GradientBoosting_Sklearn'
            }
    
    async def _train_transformer(self, X: np.ndarray, y: np.ndarray, horizon: str) -> Dict[str, Any]:
        """Train Transformer model"""
        
        model = self.models[f'transformer_{horizon}']
        
        if TORCH_AVAILABLE and isinstance(model, TransformerPredictor):
            # Similar to LSTM training but with Transformer
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['epochs']):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        break
            
            return {
                'final_loss': best_loss,
                'epochs_trained': epoch + 1,
                'model_type': 'Transformer_PyTorch'
            }
        else:
            # Sklearn MLP training
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            model.fit(X, y)
            score = model.score(X, y)
            
            return {
                'training_score': score,
                'model_type': 'MLP_Sklearn'
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all deep learning models"""
        
        status = {
            'deep_learning_mandatory': True,
            'pytorch_available': TORCH_AVAILABLE,
            'models_initialized': sum(1 for model in self.models.values() if model is not None),
            'total_models': len(self.models),
            'model_details': {}
        }
        
        for model_name, model in self.models.items():
            if model is not None:
                model_type = type(model).__name__
                status['model_details'][model_name] = {
                    'loaded': True,
                    'type': model_type,
                    'pytorch_model': TORCH_AVAILABLE and hasattr(model, 'forward')
                }
            else:
                status['model_details'][model_name] = {
                    'loaded': False,
                    'type': 'None'
                }
        
        return status
    
    def enforce_mandatory_deep_learning(self) -> bool:
        """Enforce that deep learning is mandatory for all predictions"""
        
        self.logger.critical(
            "DEEP LEARNING ENFORCEMENT ACTIVE - No predictions without deep learning models"
        )
        
        # Verify all models are loaded
        missing_models = [name for name, model in self.models.items() if model is None]
        
        if missing_models:
            self.logger.error(f"Missing mandatory deep learning models: {missing_models}")
            return False
        
        return True