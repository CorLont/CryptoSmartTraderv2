#!/usr/bin/env python3
"""
Advanced Transformer Models
Temporal Fusion Transformer (TFT) and N-BEATS implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear transformation
        output = self.w_o(context)

        return output, attention_weights

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon forecasting
    """

    def __init__(self,
                 input_size: int,
                 static_size: int,
                 output_size: int,
                 hidden_size: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()

        self.input_size = input_size
        self.static_size = static_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.quantiles = quantiles

        # Variable selection networks
        self.dynamic_selection = nn.Linear(input_size, input_size)
        self.static_selection = nn.Linear(static_size, static_size)

        # Encoder-decoder architecture
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True)

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, n_heads, dropout)

        # Static context vectors
        self.static_encoder = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gating mechanisms
        self.encoder_gate = nn.Linear(hidden_size, hidden_size)
        self.decoder_gate = nn.Linear(hidden_size, hidden_size)

        # Output layers for multiple quantiles
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in quantiles
        ])

        # Position encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)

    def forward(self,
                dynamic_inputs: torch.Tensor,
                static_inputs: torch.Tensor,
                decoder_length: int) -> Dict[str, torch.Tensor]:

        batch_size, seq_len, _ = dynamic_inputs.shape

        # Variable selection
        selected_dynamic = torch.sigmoid(self.dynamic_selection(dynamic_inputs))
        selected_static = torch.sigmoid(self.static_selection(static_inputs))

        # Apply variable selection
        dynamic_inputs = dynamic_inputs * selected_dynamic
        static_context = self.static_encoder(static_inputs * selected_static)

        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder_lstm(dynamic_inputs)

        # Add static context to encoder outputs
        static_context_expanded = static_context.unsqueeze(1).expand(
            batch_size, seq_len, -1
        )
        encoder_outputs = encoder_outputs + static_context_expanded

        # Positional encoding
        encoder_outputs = self.pos_encoding(encoder_outputs)

        # Self-attention
        attended_outputs, attention_weights = self.attention(
            encoder_outputs, encoder_outputs, encoder_outputs
        )

        # Gating mechanism
        gate = torch.sigmoid(self.encoder_gate(attended_outputs))
        encoder_outputs = gate * attended_outputs + (1 - gate) * encoder_outputs

        # Decoder
        decoder_input = torch.zeros(batch_size, 1, self.input_size).to(dynamic_inputs.device)
        decoder_outputs = []

        for _ in range(decoder_length):
            # Combine decoder input with context
            decoder_input_with_context = torch.cat([
                decoder_input,
                hidden.transpose(0, 1)[:, -1:, :]
            ], dim=-1)

            decoder_output, (hidden, cell) = self.decoder_lstm(
                decoder_input_with_context, (hidden, cell)
            )

            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output[:, :, :self.input_size]

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # Generate predictions for each quantile
        quantile_predictions = {}
        for i, quantile in enumerate(self.quantiles):
            pred = self.output_layers[i](decoder_outputs)
            quantile_predictions[f'quantile_{quantile}'] = pred

        return {
            'predictions': quantile_predictions,
            'attention_weights': attention_weights,
            'variable_selection': {
                'dynamic': selected_dynamic,
                'static': selected_static
            }
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class NBEATSBlock(nn.Module):
    """Single N-BEATS block"""

    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 basis_function: str = 'generic',
                 layers: int = 4,
                 layer_size: int = 256):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function

        # Fully connected stack
        layers_list = []
        for i in range(layers):
            if i == 0:
                layers_list.append(nn.Linear(input_size, layer_size))
            else:
                layers_list.append(nn.Linear(layer_size, layer_size))
            layers_list.append(nn.ReLU())

        self.layers = nn.Sequential(*layers_list)

        # Theta layers (for basis function parameters)
        self.theta_b_fc = nn.Linear(layer_size, theta_size)  # Backcast
        self.theta_f_fc = nn.Linear(layer_size, theta_size)  # Forecast

    def forward(self, x):
        # Forward through FC stack
        out = self.layers(x)

        # Generate theta parameters
        theta_b = self.theta_b_fc(out)
        theta_f = self.theta_f_fc(out)

        # Generate basis functions (simplified)
        if self.basis_function == 'trend':
            backcast = self._trend_basis(theta_b, self.input_size)
            forecast = self._trend_basis(theta_f, self.input_size)
        elif self.basis_function == 'seasonality':
            backcast = self._seasonality_basis(theta_b, self.input_size)
            forecast = self._seasonality_basis(theta_f, self.input_size)
        else:  # generic
            backcast = self._generic_basis(theta_b, self.input_size)
            forecast = self._generic_basis(theta_f, self.input_size)

        return backcast, forecast

    def _generic_basis(self, theta, size):
        """Generic basis function"""
        return theta[:, :size]

    def _trend_basis(self, theta, size):
        """Trend basis function (polynomial)"""
        time = torch.arange(size, dtype=torch.float).unsqueeze(0).to(theta.device)
        basis = torch.zeros(theta.shape[0], size).to(theta.device)

        for i in range(min(theta.shape[1], 4)):  # Up to degree 3 polynomial
            basis += theta[:, i:i+1] * (time ** i)

        return basis

    def _seasonality_basis(self, theta, size):
        """Seasonality basis function (Fourier)"""
        time = torch.arange(size, dtype=torch.float).unsqueeze(0).to(theta.device)
        basis = torch.zeros(theta.shape[0], size).to(theta.device)

        # Fourier terms
        for i in range(0, min(theta.shape[1], 20), 2):
            if i + 1 < theta.shape[1]:
                freq = (i // 2 + 1) / size * 2 * math.pi
                basis += theta[:, i:i+1] * torch.cos(freq * time)
                basis += theta[:, i+1:i+2] * torch.sin(freq * time)

        return basis

class NBEATS(nn.Module):
    """
    N-BEATS model for time series forecasting
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 stacks: List[Dict[str, Any]] = None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        if stacks is None:
            stacks = [
                {'type': 'trend', 'blocks': 3, 'layers': 4, 'layer_size': 256, 'theta_size': 8},
                {'type': 'seasonality', 'blocks': 3, 'layers': 4, 'layer_size': 256, 'theta_size': 20},
                {'type': 'generic', 'blocks': 3, 'layers': 4, 'layer_size': 256, 'theta_size': input_size}
            ]

        self.stacks = nn.ModuleList()

        for stack_config in stacks:
            stack = nn.ModuleList()
            for _ in range(stack_config['blocks']):
                block = NBEATSBlock(
                    input_size=input_size,
                    theta_size=stack_config['theta_size'],
                    basis_function=stack_config['type'],
                    layers=stack_config['layers'],
                    layer_size=stack_config['layer_size']
                )
                stack.append(block)
            self.stacks.append(stack)

    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize residuals and forecasts
        residual = x
        total_forecast = torch.zeros(batch_size, self.output_size).to(x.device)

        # Process each stack
        for stack in self.stacks:
            stack_forecast = torch.zeros(batch_size, self.output_size).to(x.device)

            for block in stack:
                backcast, forecast = block(residual)
                residual = residual - backcast
                stack_forecast = stack_forecast + forecast[:, :self.output_size]

            total_forecast = total_forecast + stack_forecast

        return total_forecast

class AdvancedTransformerPredictor:
    """
    Advanced transformer-based predictor combining TFT and N-BEATS
    """

    def __init__(self,
                 input_size: int,
                 static_size: int = 10,
                 output_horizons: List[int] = [1, 24, 168, 720]):

        self.input_size = input_size
        self.static_size = static_size
        self.output_horizons = output_horizons

        # Initialize models
        self.tft_model = TemporalFusionTransformer(
            input_size=input_size,
            static_size=static_size,
            output_size=len(output_horizons),
            hidden_size=128,
            n_heads=8,
            n_layers=4
        )

        self.nbeats_model = NBEATS(
            input_size=input_size,
            output_size=max(output_horizons)
        )

        self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # TFT, N-BEATS

    def train_models(self,
                    train_data: Dict[str, torch.Tensor],
                    val_data: Dict[str, torch.Tensor],
                    epochs: int = 100) -> Dict[str, List[float]]:
        """Train both models"""

        # Training setup
        tft_optimizer = torch.optim.Adam(self.tft_model.parameters(), lr=0.001)
        nbeats_optimizer = torch.optim.Adam(self.nbeats_model.parameters(), lr=0.001)

        criterion = nn.MSELoss()

        train_losses = {'tft': [], 'nbeats': [], 'ensemble': []}
        val_losses = {'tft': [], 'nbeats': [], 'ensemble': []}

        for epoch in range(epochs):
            # Train TFT
            self.tft_model.train()
            tft_optimizer.zero_grad()

            tft_outputs = self.tft_model(
                train_data['dynamic'],
                train_data['static'],
                len(self.output_horizons)
            )

            tft_loss = criterion(
                tft_outputs['predictions']['quantile_0.5'],
                train_data['targets']
            )

            tft_loss.backward()
            tft_optimizer.step()

            # Train N-BEATS
            self.nbeats_model.train()
            nbeats_optimizer.zero_grad()

            nbeats_outputs = self.nbeats_model(train_data['dynamic'])
            nbeats_loss = criterion(
                nbeats_outputs[:, :len(self.output_horizons)],
                train_data['targets']
            )

            nbeats_loss.backward()
            nbeats_optimizer.step()

            # Ensemble loss
            ensemble_loss = (
                F.softmax(self.ensemble_weights, dim=0)[0] * tft_loss +
                F.softmax(self.ensemble_weights, dim=0)[1] * nbeats_loss
            )

            train_losses['tft'].append(tft_loss.item())
            train_losses['nbeats'].append(nbeats_loss.item())
            train_losses['ensemble'].append(ensemble_loss.item())

            # Validation
            if epoch % 10 == 0:
                val_metrics = self._validate(val_data)
                val_losses['tft'].append(val_metrics['tft_loss'])
                val_losses['nbeats'].append(val_metrics['nbeats_loss'])
                val_losses['ensemble'].append(val_metrics['ensemble_loss'])

                print(f"Epoch {epoch}: TFT: {tft_loss.item():.4f}, "
                      f"N-BEATS: {nbeats_loss.item():.4f}, "
                      f"Ensemble: {ensemble_loss.item():.4f}")

        return {'train': train_losses, 'val': val_losses}

    def _validate(self, val_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validate models"""

        self.tft_model.eval()
        self.nbeats_model.eval()

        with torch.no_grad():
            # TFT validation
            tft_outputs = self.tft_model(
                val_data['dynamic'],
                val_data['static'],
                len(self.output_horizons)
            )

            tft_loss = F.mse_loss(
                tft_outputs['predictions']['quantile_0.5'],
                val_data['targets']
            )

            # N-BEATS validation
            nbeats_outputs = self.nbeats_model(val_data['dynamic'])
            nbeats_loss = F.mse_loss(
                nbeats_outputs[:, :len(self.output_horizons)],
                val_data['targets']
            )

            # Ensemble
            ensemble_loss = (
                F.softmax(self.ensemble_weights, dim=0)[0] * tft_loss +
                F.softmax(self.ensemble_weights, dim=0)[1] * nbeats_loss
            )

        return {
            'tft_loss': tft_loss.item(),
            'nbeats_loss': nbeats_loss.item(),
            'ensemble_loss': ensemble_loss.item()
        }

    def predict(self,
               dynamic_inputs: torch.Tensor,
               static_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions"""

        self.tft_model.eval()
        self.nbeats_model.eval()

        with torch.no_grad():
            # TFT predictions
            tft_outputs = self.tft_model(
                dynamic_inputs,
                static_inputs,
                len(self.output_horizons)
            )

            tft_pred = tft_outputs['predictions']['quantile_0.5']

            # N-BEATS predictions
            nbeats_pred = self.nbeats_model(dynamic_inputs)
            nbeats_pred = nbeats_pred[:, :len(self.output_horizons)]

            # Ensemble predictions
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_pred = weights[0] * tft_pred + weights[1] * nbeats_pred

            # Uncertainty from TFT quantiles
            uncertainty = (
                tft_outputs['predictions']['quantile_0.9'] -
                tft_outputs['predictions']['quantile_0.1']
            ) / 2

        return {
            'predictions': ensemble_pred,
            'uncertainty': uncertainty,
            'tft_predictions': tft_pred,
            'nbeats_predictions': nbeats_pred,
            'attention_weights': tft_outputs['attention_weights'],
            'variable_importance': tft_outputs['variable_selection']
        }

if __name__ == "__main__":
    print("🤖 TESTING ADVANCED TRANSFORMER MODELS")
    print("=" * 45)

    # Test parameters
    batch_size = 32
    seq_length = 168  # 1 week
    input_size = 20
    static_size = 10
    output_horizons = [1, 24, 168, 720]

    # Create sample data
    dynamic_data = torch.randn(batch_size, seq_length, input_size)
    static_data = torch.randn(batch_size, static_size)
    targets = torch.randn(batch_size, len(output_horizons))

    # Initialize predictor
    predictor = AdvancedTransformerPredictor(
        input_size=input_size,
        static_size=static_size,
        output_horizons=output_horizons
    )

    print(f"Model initialized:")
    print(f"   Input size: {input_size}")
    print(f"   Static size: {static_size}")
    print(f"   Output horizons: {output_horizons}")

    # Test prediction
    predictions = predictor.predict(dynamic_data, static_data)

    print(f"\nPrediction results:")
    print(f"   Predictions shape: {predictions['predictions'].shape}")
    print(f"   Uncertainty shape: {predictions['uncertainty'].shape}")
    print(f"   Attention weights shape: {predictions['attention_weights'].shape}")

    # Test training (short run)
    train_data = {
        'dynamic': dynamic_data,
        'static': static_data,
        'targets': targets
    }

    val_data = {
        'dynamic': dynamic_data[:16],
        'static': static_data[:16],
        'targets': targets[:16]
    }

    losses = predictor.train_models(train_data, val_data, epochs=10)

    print(f"\nTraining completed:")
    print(f"   Final TFT loss: {losses['train']['tft'][-1]:.4f}")
    print(f"   Final N-BEATS loss: {losses['train']['nbeats'][-1]:.4f}")
    print(f"   Final ensemble loss: {losses['train']['ensemble'][-1]:.4f}")

    print("✅ Advanced transformer models testing completed")
