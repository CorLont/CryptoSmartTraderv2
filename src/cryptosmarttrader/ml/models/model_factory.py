#!/usr/bin/env python3
"""
Model Factory
Central factory for creating ML models with integrated OpenAI intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# OpenAI integration
try:
    from ml.intelligence.openai_simple_analyzer import create_simple_ai_analyzer, AISentimentResult
    OPENAI_INTELLIGENCE_AVAILABLE = True
except ImportError:
    OPENAI_INTELLIGENCE_AVAILABLE = False

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    parameters: Dict[str, Any]
    use_ai_features: bool = True
    enable_openai_insights: bool = True

class BaseModel:
    """Base model class with OpenAI integration"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.logger = logging.getLogger(__name__)

        # OpenAI integration
        self.ai_analyzer = None
        if config.enable_openai_insights and OPENAI_INTELLIGENCE_AVAILABLE:
            try:
                self.ai_analyzer = create_simple_ai_analyzer()
                self.logger.info("OpenAI intelligence integrated")
            except Exception as e:
                self.logger.warning(f"OpenAI integration failed: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            return np.column_stack([1-predictions, predictions])

    def get_ai_insights(self, market_data: pd.DataFrame, news_data: List[str] = None) -> Dict[str, Any]:
        """Get AI-powered market insights"""

        if not self.ai_analyzer:
            return {"error": "OpenAI intelligence not available"}

        try:
            # Get sentiment analysis
            sentiment = self.ai_analyzer.analyze_market_sentiment(market_data, news_data)

            # Get trading insights
            insights = self.ai_analyzer.generate_trading_insights(market_data)

            # Get anomaly detection
            anomalies = self.ai_analyzer.detect_market_anomalies(market_data)

            return {
                "sentiment": {
                    "overall": sentiment.overall_sentiment,
                    "score": sentiment.sentiment_score,
                    "confidence": sentiment.confidence,
                    "themes": sentiment.key_themes,
                    "drivers": sentiment.market_drivers
                },
                "insights": [
                    {
                        "type": insight.insight_type,
                        "content": insight.content,
                        "confidence": insight.confidence
                    }
                    for insight in insights
                ],
                "anomalies": [
                    {
                        "type": anomaly.insight_type,
                        "content": anomaly.content,
                        "confidence": anomaly.confidence
                    }
                    for anomaly in anomalies
                ]
            }

        except Exception as e:
            self.logger.error(f"AI insights failed: {e}")
            return {"error": f"AI analysis failed: {e}"}

class LSTMModel(BaseModel):
    """LSTM model with PyTorch"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for LSTM model")

        # LSTM parameters
        self.input_size = config.parameters.get('input_size', 10)
        self.hidden_size = config.parameters.get('hidden_size', 64)
        self.num_layers = config.parameters.get('num_layers', 2)
        self.dropout = config.parameters.get('dropout', 0.2)

        # Create LSTM network
        self.model = nn.Sequential(
            nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                   dropout=self.dropout, batch_first=True),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs):
        """Train LSTM model"""

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Setup training
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # LSTM expects 3D input: (batch, sequence, features)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension

            # Forward pass
            lstm_out, _ = self.model[0](X_tensor)
            output = self.model[1](lstm_out[:, -1, :])  # Use last output
            output = self.model[2](output)  # Apply sigmoid

            # Compute loss
            loss = criterion(output, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                self.logger.debug(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        self.is_trained = True
        self.logger.info("LSTM model training completed")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions with LSTM"""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)

            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)

            lstm_out, _ = self.model[0](X_tensor)
            output = self.model[1](lstm_out[:, -1, :])
            output = self.model[2](output)

            return output.numpy().flatten()

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple algorithms"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.models = {}
        self.weights = config.parameters.get('weights', {})

        # Initialize sub-models
        if SKLEARN_AVAILABLE:
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.models['lr'] = LogisticRegression(random_state=42)

        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )

        # Default equal weights
        if not self.weights:
            self.weights = {name: 1.0 for name in self.models.keys()}

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train ensemble models"""

        for name, model in self.models.items():
            try:
                self.logger.debug(f"Training {name} model...")
                model.fit(X, y)
                self.logger.debug(f"{name} model trained successfully")
            except Exception as e:
                self.logger.error(f"Failed to train {name} model: {e}")
                # Remove failed model
                del self.models[name]
                if name in self.weights:
                    del self.weights[name]

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: w/total_weight for name, w in self.weights.items()}

        self.is_trained = True
        self.logger.info(f"Ensemble model trained with {len(self.models)} sub-models")

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Ensemble prediction with probabilities"""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not self.models:
            raise ValueError("No trained models available in ensemble")

        # Get predictions from all models
        predictions = []
        weights = []

        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    # Convert binary predictions to probabilities
                    binary_pred = model.predict(X)
                    pred = np.column_stack([1-binary_pred, binary_pred])

                predictions.append(pred)
                weights.append(self.weights.get(name, 1.0))

            except Exception as e:
                self.logger.warning(f"Prediction failed for {name}: {e}")
                continue

        if not predictions:
            raise ValueError("All ensemble models failed to make predictions")

        # Weighted average of predictions
        weighted_pred = np.zeros_like(predictions[0])
        total_weight = 0

        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
            total_weight += weight

        if total_weight > 0:
            weighted_pred /= total_weight

        return weighted_pred

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Binary predictions from ensemble"""

        proba = self.predict_proba(X, **kwargs)
        return (proba[:, 1] > 0.5).astype(int)

class ModelFactory:
    """Factory for creating ML models with OpenAI integration"""

    @staticmethod
    def create_model(model_type: str, parameters: Dict[str, Any] = None, **kwargs) -> BaseModel:
        """Create model instance"""

        if parameters is None:
            parameters = {}

        config = ModelConfig(
            model_type=model_type,
            parameters=parameters,
            use_ai_features=kwargs.get('use_ai_features', True),
            enable_openai_insights=kwargs.get('enable_openai_insights', True)
        )

        if model_type.lower() == 'lstm':
            return LSTMModel(config)
        elif model_type.lower() == 'ensemble':
            return EnsembleModel(config)
        elif model_type.lower() == 'random_forest' and SKLEARN_AVAILABLE:
            # Wrap sklearn model in BaseModel
            model = BaseModel(config)
            model.model = RandomForestClassifier(**parameters)
            return model
        elif model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
            model = BaseModel(config)
            model.model = xgb.XGBClassifier(**parameters)
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types"""

        models = ['ensemble']  # Always available

        if TORCH_AVAILABLE:
            models.append('lstm')

        if SKLEARN_AVAILABLE:
            models.extend(['random_forest', 'gradient_boosting', 'logistic_regression'])

        if XGBOOST_AVAILABLE:
            models.append('xgboost')

        return models

    @staticmethod
    def get_ai_capabilities() -> Dict[str, bool]:
        """Get AI integration capabilities"""

        return {
            "openai_available": OPENAI_INTELLIGENCE_AVAILABLE,
            "pytorch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "xgboost_available": XGBOOST_AVAILABLE,
            "sentiment_analysis": OPENAI_INTELLIGENCE_AVAILABLE,
            "news_impact_assessment": OPENAI_INTELLIGENCE_AVAILABLE,
            "trading_insights": OPENAI_INTELLIGENCE_AVAILABLE,
            "anomaly_detection": OPENAI_INTELLIGENCE_AVAILABLE
        }

# Convenience functions
def create_lstm_model(input_size: int = 10, hidden_size: int = 64, **kwargs) -> LSTMModel:
    """Create LSTM model with OpenAI integration"""
    parameters = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': kwargs.get('num_layers', 2),
        'dropout': kwargs.get('dropout', 0.2)
    }
    return ModelFactory.create_model('lstm', parameters, **kwargs)

def create_ensemble_model(weights: Dict[str, float] = None, **kwargs) -> EnsembleModel:
    """Create ensemble model with OpenAI integration"""
    parameters = {'weights': weights} if weights else {}
    return ModelFactory.create_model('ensemble', parameters, **kwargs)

def create_ai_enhanced_model(model_type: str = 'ensemble', **kwargs) -> BaseModel:
    """Create AI-enhanced model with full OpenAI integration"""
    kwargs['enable_openai_insights'] = True
    kwargs['use_ai_features'] = True
    return ModelFactory.create_model(model_type, **kwargs)
