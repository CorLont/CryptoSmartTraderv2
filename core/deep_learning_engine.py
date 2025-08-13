#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Deep Learning Engine
LSTM, GRU, Transformer, N-BEATS as core components for multimodal sequence modeling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available - deep learning features disabled")

try:
    from transformers import AutoTokenizer, AutoModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ModelType(Enum):
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    NBEATS = "nbeats"
    MULTIMODAL_FUSION = "multimodal_fusion"


class DataModality(Enum):
    PRICE_TIME_SERIES = "price_timeseries"
    SENTIMENT_TEXT = "sentiment_text"
    WHALE_TRANSACTIONS = "whale_transactions"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_REGIME = "market_regime"
    ORDER_BOOK = "order_book"


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models"""

    sequence_length: int = 100
    prediction_horizons: List[int] = field(
        default_factory=lambda: [1, 6, 24, 168]
    )  # 1h, 6h, 1d, 1w
    hidden_dim: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0

    # Model-specific configs
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024
    nbeats_stack_types: List[str] = field(
        default_factory=lambda: ["trend", "seasonality", "generic"]
    )
    nbeats_num_blocks: int = 3
    nbeats_num_layers: int = 4

    # Multimodal fusion
    fusion_method: str = "attention"  # attention, concat, cross_attention
    text_embedding_dim: int = 768
    enable_cross_modal_attention: bool = True


class MultiModalDataset(Dataset):
    """Dataset for multimodal deep learning"""

    def __init__(
        self, data_dict: Dict[str, np.ndarray], targets: np.ndarray, sequence_length: int = 100
    ):
        self.data_dict = data_dict
        self.targets = targets
        self.sequence_length = sequence_length

        # Ensure all data has same length
        self.length = min(len(v) for v in data_dict.values())
        self.length = min(self.length, len(targets))

        # Adjust sequence length if needed
        self.sequence_length = min(sequence_length, self.length - 1)

    def __len__(self):
        return max(0, self.length - self.sequence_length)

    def __getitem__(self, idx):
        sample = {}

        for modality, data in self.data_dict.items():
            if len(data.shape) == 1:
                sample[modality] = torch.FloatTensor(data[idx : idx + self.sequence_length])
            else:
                sample[modality] = torch.FloatTensor(data[idx : idx + self.sequence_length])

        target = torch.FloatTensor(self.targets[idx + self.sequence_length])
        return sample, target


class LSTMModel(nn.Module):
    """Advanced LSTM model for sequence prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = self.layer_norm(attended_out + lstm_out)

        # Global max pooling and average pooling
        max_pool = torch.max(attended_out, dim=1)[0]
        avg_pool = torch.mean(attended_out, dim=1)

        # Combine pooled features
        combined = max_pool + avg_pool
        combined = self.dropout(combined)

        # Final prediction
        output = self.output_projection(combined)
        return output


class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Transformer encoding
        encoded = self.transformer(x)

        # Global average pooling
        pooled = torch.mean(encoded, dim=1)

        # Final prediction
        output = self.output_projection(pooled)
        return output


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
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class NBeatsBlock(nn.Module):
    """N-BEATS block implementation"""

    def __init__(
        self,
        input_dim: int,
        theta_dim: int,
        basis_function: str,
        layers: int = 4,
        layer_width: int = 256,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layer_width))

        for _ in range(layers - 1):
            self.layers.append(nn.Linear(layer_width, layer_width))

        self.basis_function = basis_function
        self.theta_dim = theta_dim

        # Basis parameters
        self.basis_parameters = nn.Linear(layer_width, theta_dim)
        self.backcast_linear = nn.Linear(theta_dim, input_dim)
        self.forecast_linear = nn.Linear(theta_dim, 1)  # Single step forecast

    def forward(self, x):
        residual = x

        # Forward through layers
        for layer in self.layers:
            residual = F.relu(layer(residual))

        # Generate basis parameters
        theta = self.basis_parameters(residual)

        # Generate backcast and forecast
        backcast = self.backcast_linear(theta)
        forecast = self.forecast_linear(theta)

        return backcast, forecast


class NBeatsModel(nn.Module):
    """N-BEATS model for time series forecasting"""

    def __init__(
        self,
        input_dim: int,
        stack_types: List[str],
        num_blocks: int,
        num_layers: int,
        layer_width: int = 256,
    ):
        super().__init__()

        self.stacks = nn.ModuleList()

        for stack_type in stack_types:
            stack = nn.ModuleList()
            for _ in range(num_blocks):
                if stack_type == "trend":
                    theta_dim = 3  # Linear trend parameters
                elif stack_type == "seasonality":
                    theta_dim = 10  # Seasonal fourier components
                else:  # generic
                    theta_dim = 8

                block = NBeatsBlock(input_dim, theta_dim, stack_type, num_layers, layer_width)
                stack.append(block)

            self.stacks.append(stack)

    def forward(self, x):
        residual = x
        forecast = 0

        for stack in self.stacks:
            stack_forecast = 0

            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                stack_forecast = stack_forecast + block_forecast

            forecast = forecast + stack_forecast

        return forecast


class MultiModalFusionModel(nn.Module):
    """Multimodal fusion model combining different data types"""

    def __init__(self, config: DeepLearningConfig, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()

        self.config = config
        self.modality_dims = modality_dims
        self.hidden_dim = config.hidden_dim

        # Modality-specific encoders
        self.encoders = nn.ModuleDict()

        for modality, input_dim in modality_dims.items():
            if "text" in modality.lower() or "sentiment" in modality.lower():
                # Text encoder
                self.encoders[modality] = self._create_text_encoder(input_dim)
            elif "timeseries" in modality.lower() or "price" in modality.lower():
                # Time series encoder
                self.encoders[modality] = self._create_timeseries_encoder(input_dim)
            else:
                # Generic encoder
                self.encoders[modality] = self._create_generic_encoder(input_dim)

        # Cross-modal attention
        if config.enable_cross_modal_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True,
            )

        # Fusion layer
        total_dim = len(modality_dims) * self.hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def _create_text_encoder(self, input_dim: int):
        """Create text encoder"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _create_timeseries_encoder(self, input_dim: int):
        """Create time series encoder with LSTM"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True),
            nn.Dropout(self.config.dropout_rate),
        )

    def _create_generic_encoder(self, input_dim: int):
        """Create generic encoder"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        encoded_modalities = []

        # Encode each modality
        for modality, data in inputs.items():
            if modality in self.encoders:
                encoder = self.encoders[modality]

                if isinstance(encoder[1], nn.LSTM):
                    # Handle LSTM encoder
                    x = encoder[0](data)
                    lstm_out, _ = encoder[1](x)
                    encoded = torch.mean(lstm_out, dim=1)  # Global average pooling
                    encoded = encoder[2](encoded)  # Dropout
                else:
                    # Handle other encoders
                    encoded = encoder(data.mean(dim=1) if len(data.shape) > 2 else data)

                encoded_modalities.append(encoded)

        if not encoded_modalities:
            raise ValueError("No valid modalities found in input")

        # Cross-modal attention
        if self.config.enable_cross_modal_attention and len(encoded_modalities) > 1:
            # Stack encoded modalities
            stacked = torch.stack(encoded_modalities, dim=1)

            # Apply cross-attention
            attended, _ = self.cross_attention(stacked, stacked, stacked)

            # Flatten attended representations
            fused_representation = attended.flatten(start_dim=1)
        else:
            # Simple concatenation
            fused_representation = torch.cat(encoded_modalities, dim=1)

        # Final fusion and prediction
        output = self.fusion_layer(fused_representation)
        return output


class DeepLearningEngine:
    """Core deep learning engine for CryptoSmartTrader"""

    def __init__(self, config: Optional[DeepLearningConfig] = None):
        self.config = config or DeepLearningConfig()
        self.logger = logging.getLogger(f"{__name__}.DeepLearningEngine")

        if not HAS_TORCH:
            self.logger.error("PyTorch not available - deep learning disabled")
            return

        # Model registry
        self.models: Dict[str, Dict] = {}
        self.active_models: Dict[str, nn.Module] = {}

        # Training state
        self.training_history: Dict[str, List] = {}
        self.model_performance: Dict[str, Dict] = {}

        # Data preprocessing
        self.scalers: Dict[str, Any] = {}
        self.feature_extractors: Dict[str, Any] = {}

        # Model storage
        self.model_dir = Path("models/deep_learning")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Deep Learning Engine initialized on {self.device}")

    def create_model(
        self,
        model_type: ModelType,
        model_id: str,
        input_dims: Union[int, Dict[str, int]],
        output_dim: int,
        **kwargs,
    ) -> bool:
        """
        Create a new deep learning model

        Args:
            model_type: Type of model to create
            model_id: Unique identifier for the model
            input_dims: Input dimensions (int for single modality, dict for multimodal)
            output_dim: Output dimension
            **kwargs: Additional model-specific parameters

        Returns:
            True if model created successfully
        """
        with self._lock:
            try:
                if not HAS_TORCH:
                    self.logger.error("PyTorch not available")
                    return False

                # Create model based on type
                if model_type == ModelType.LSTM:
                    model = LSTMModel(
                        input_dims if isinstance(input_dims, int) else sum(input_dims.values()),
                        self.config.hidden_dim,
                        self.config.num_layers,
                        output_dim,
                        self.config.dropout_rate,
                    )

                elif model_type == ModelType.TRANSFORMER:
                    model = TransformerModel(
                        input_dims if isinstance(input_dims, int) else sum(input_dims.values()),
                        self.config.hidden_dim,
                        self.config.num_layers,
                        self.config.transformer_heads,
                        output_dim,
                        self.config.dropout_rate,
                    )

                elif model_type == ModelType.NBEATS:
                    model = NBeatsModel(
                        input_dims if isinstance(input_dims, int) else sum(input_dims.values()),
                        self.config.nbeats_stack_types,
                        self.config.nbeats_num_blocks,
                        self.config.nbeats_num_layers,
                    )

                elif model_type == ModelType.MULTIMODAL_FUSION:
                    if not isinstance(input_dims, dict):
                        raise ValueError("Multimodal fusion requires dict input_dims")
                    model = MultiModalFusionModel(self.config, input_dims, output_dim)

                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                # Move to device
                model = model.to(self.device)

                # Store model
                self.models[model_id] = {
                    "model": model,
                    "type": model_type,
                    "input_dims": input_dims,
                    "output_dim": output_dim,
                    "created_at": datetime.now(),
                    "trained": False,
                    "performance": {},
                }

                self.active_models[model_id] = model

                self.logger.info(
                    f"Created {model_type.value} model '{model_id}' with {sum(p.numel() for p in model.parameters())} parameters"
                )
                return True

            except Exception as e:
                self.logger.error(f"Failed to create model {model_id}: {e}")
                return False

    def train_model(
        self,
        model_id: str,
        train_data: Dict[str, np.ndarray],
        targets: np.ndarray,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        val_targets: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train a deep learning model

        Args:
            model_id: Model identifier
            train_data: Training data (dict for multimodal, single array for unimodal)
            targets: Target values
            val_data: Validation data (optional)
            val_targets: Validation targets (optional)

        Returns:
            Training results and metrics
        """
        with self._lock:
            try:
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found")

                model_info = self.models[model_id]
                model = model_info["model"]
                model_type = model_info["type"]

                # Prepare data loaders
                train_dataset = self._prepare_dataset(train_data, targets, model_type)
                train_loader = DataLoader(
                    train_dataset, batch_size=self.config.batch_size, shuffle=True
                )

                val_loader = None
                if val_data is not None and val_targets is not None:
                    val_dataset = self._prepare_dataset(val_data, val_targets, model_type)
                    val_loader = DataLoader(
                        val_dataset, batch_size=self.config.batch_size, shuffle=False
                    )

                # Setup training components
                criterion = nn.MSELoss()
                optimizer = AdamW(
                    model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5
                )
                scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

                # Training history
                history = {"train_loss": [], "val_loss": [], "learning_rates": []}

                best_val_loss = float("inf")
                patience_counter = 0

                # Training loop
                for epoch in range(self.config.max_epochs):
                    # Training phase
                    model.train()
                    train_loss = 0.0

                    for batch_idx, (batch_data, batch_targets) in enumerate(train_loader):
                        optimizer.zero_grad()

                        # Forward pass
                        if isinstance(batch_data, dict):
                            # Multimodal data
                            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                        else:
                            # Single modality
                            batch_data = batch_data.to(self.device)

                        batch_targets = batch_targets.to(self.device)

                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_targets)

                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.gradient_clip_norm
                        )
                        optimizer.step()

                        train_loss += loss.item()

                    avg_train_loss = train_loss / len(train_loader)
                    history["train_loss"].append(avg_train_loss)
                    history["learning_rates"].append(optimizer.param_groups[0]["lr"])

                    # Validation phase
                    if val_loader is not None:
                        model.eval()
                        val_loss = 0.0

                        with torch.no_grad():
                            for batch_data, batch_targets in val_loader:
                                if isinstance(batch_data, dict):
                                    batch_data = {
                                        k: v.to(self.device) for k, v in batch_data.items()
                                    }
                                else:
                                    batch_data = batch_data.to(self.device)

                                batch_targets = batch_targets.to(self.device)
                                outputs = model(batch_data)
                                loss = criterion(outputs, batch_targets)
                                val_loss += loss.item()

                        avg_val_loss = val_loss / len(val_loader)
                        history["val_loss"].append(avg_val_loss)

                        # Learning rate scheduling
                        scheduler.step(avg_val_loss)

                        # Early stopping
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            # Save best model
                            self._save_model_checkpoint(model_id, model, epoch, avg_val_loss)
                        else:
                            patience_counter += 1

                        if patience_counter >= self.config.early_stopping_patience:
                            self.logger.info(f"Early stopping at epoch {epoch}")
                            break

                        if epoch % 10 == 0:
                            self.logger.info(
                                f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"
                            )
                    else:
                        if epoch % 10 == 0:
                            self.logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}")

                # Update model info
                model_info["trained"] = True
                model_info["training_history"] = history
                self.training_history[model_id] = history

                # Calculate final performance metrics
                performance = self._calculate_model_performance(
                    model, val_loader if val_loader else train_loader
                )
                model_info["performance"] = performance
                self.model_performance[model_id] = performance

                self.logger.info(f"Training completed for model {model_id}")
                return {
                    "success": True,
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                    "epochs_trained": len(history["train_loss"]),
                    "performance": performance,
                }

            except Exception as e:
                self.logger.error(f"Training failed for model {model_id}: {e}")
                return {"success": False, "error": str(e)}

    def _prepare_dataset(
        self,
        data: Union[Dict[str, np.ndarray], np.ndarray],
        targets: np.ndarray,
        model_type: ModelType,
    ) -> Dataset:
        """Prepare dataset for training"""
        if model_type == ModelType.MULTIMODAL_FUSION:
            if not isinstance(data, dict):
                raise ValueError("Multimodal fusion requires dict data")
            return MultiModalDataset(data, targets, self.config.sequence_length)
        else:
            # Convert single modality data to format expected by MultiModalDataset
            if isinstance(data, dict):
                # Use first modality for single-modal models
                data_array = next(iter(data.values()))
            else:
                data_array = data

            return MultiModalDataset({"main": data_array}, targets, self.config.sequence_length)

    def predict(
        self, model_id: str, input_data: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Make predictions with a trained model

        Args:
            model_id: Model identifier
            input_data: Input data for prediction

        Returns:
            Predictions or None if failed
        """
        try:
            if model_id not in self.active_models:
                return None

            model = self.active_models[model_id]
            model.eval()

            with torch.no_grad():
                # Prepare input
                if isinstance(input_data, dict):
                    batch_data = {
                        k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                        for k, v in input_data.items()
                    }
                else:
                    batch_data = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

                # Predict
                output = model(batch_data)
                predictions = output.cpu().numpy()

                return predictions.squeeze(0)

        except Exception as e:
            self.logger.error(f"Prediction failed for model {model_id}: {e}")
            return None

    def _calculate_model_performance(
        self, model: nn.Module, data_loader: DataLoader
    ) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        model.eval()
        total_loss = 0.0
        predictions = []
        actuals = []

        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch_data, batch_targets in data_loader:
                if isinstance(batch_data, dict):
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                else:
                    batch_data = batch_data.to(self.device)

                batch_targets = batch_targets.to(self.device)
                outputs = model(batch_data)

                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Directional accuracy
        actual_direction = np.sign(actuals[1:] - actuals[:-1])
        pred_direction = np.sign(predictions[1:] - predictions[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "directional_accuracy": float(directional_accuracy),
            "avg_loss": total_loss / len(data_loader),
        }

    def _save_model_checkpoint(self, model_id: str, model: nn.Module, epoch: int, val_loss: float):
        """Save model checkpoint"""
        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": self.config.__dict__,
                "timestamp": datetime.now().isoformat(),
            }

            checkpoint_path = self.model_dir / f"{model_id}_best.pt"
            torch.save(checkpoint, checkpoint_path)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for {model_id}: {e}")

    def load_model(self, model_id: str, checkpoint_path: Optional[str] = None) -> bool:
        """Load a saved model"""
        try:
            if checkpoint_path is None:
                checkpoint_path = self.model_dir / f"{model_id}_best.pt"

            if not Path(checkpoint_path).exists():
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if model_id in self.active_models:
                self.active_models[model_id].load_state_dict(checkpoint["model_state_dict"])
                self.logger.info(f"Loaded model {model_id} from checkpoint")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models"""
        with self._lock:
            summary = {
                "total_models": len(self.models),
                "trained_models": len([m for m in self.models.values() if m.get("trained", False)]),
                "device": str(self.device),
                "pytorch_available": HAS_TORCH,
                "models": {},
            }

            for model_id, model_info in self.models.items():
                summary["models"][model_id] = {
                    "type": model_info["type"].value,
                    "trained": model_info.get("trained", False),
                    "created_at": model_info["created_at"].isoformat(),
                    "performance": model_info.get("performance", {}),
                    "parameters": sum(p.numel() for p in model_info["model"].parameters())
                    if HAS_TORCH
                    else 0,
                }

            return summary

    def get_training_progress(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get training progress for a model"""
        if model_id in self.training_history:
            history = self.training_history[model_id]
            return {
                "epochs_completed": len(history["train_loss"]),
                "current_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
                "current_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
                "learning_rate": history["learning_rates"][-1] if history["learning_rates"] else 0,
                "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            }
        return None


# Singleton deep learning engine
_deep_learning_engine = None
_dl_lock = threading.Lock()


def get_deep_learning_engine(config: Optional[DeepLearningConfig] = None) -> DeepLearningEngine:
    """Get the singleton deep learning engine"""
    global _deep_learning_engine

    with _dl_lock:
        if _deep_learning_engine is None:
            _deep_learning_engine = DeepLearningEngine(config)
        return _deep_learning_engine


def create_multimodal_model(model_id: str, modality_dims: Dict[str, int], output_dim: int) -> bool:
    """Convenient function to create multimodal model"""
    engine = get_deep_learning_engine()
    return engine.create_model(ModelType.MULTIMODAL_FUSION, model_id, modality_dims, output_dim)
