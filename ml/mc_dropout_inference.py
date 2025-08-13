#!/usr/bin/env python3
"""
Monte Carlo Dropout Inference (PyTorch)
Bayesian uncertainty quantification for neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings("ignore")


def mc_dropout_predict(
    model: nn.Module, x: torch.Tensor, passes: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout inference for uncertainty quantification

    Args:
        model: PyTorch model with dropout layers
        x: Input tensor
        passes: Number of MC forward passes

    Returns:
        mu: Mean predictions
        sigma: Standard deviation (uncertainty)
    """

    model.train()  # Enable dropout during inference
    preds = []

    with torch.no_grad():
        for _ in range(passes):
            pred = model(x)
            preds.append(pred.cpu().numpy())

    # Calculate statistics
    preds = np.array(preds)
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)

    return mu, sigma


def mc_dropout_predict_with_confidence(
    model: nn.Module, x: torch.Tensor, passes: int = 30, confidence_level: float = 0.95
) -> dict:
    """
    MC Dropout with confidence intervals
    """

    model.train()
    preds = []

    with torch.no_grad():
        for _ in range(passes):
            pred = model(x)
            preds.append(pred.cpu().numpy())

    preds = np.array(preds)

    # Calculate statistics
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)

    # Confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(preds, lower_percentile, axis=0)
    upper_bound = np.percentile(preds, upper_percentile, axis=0)

    # Confidence score (inverse of relative uncertainty)
    confidence_score = 1.0 / (1.0 + sigma / (np.abs(mu) + 1e-8))

    return {
        "mean": mu,
        "std": sigma,
        "confidence": confidence_score,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_passes": passes,
    }


class MCDropoutLSTM(nn.Module):
    """
    LSTM with Monte Carlo Dropout for uncertainty quantification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        output_size: int = 1,
    ):
        super(MCDropoutLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        last_output = lstm_out[:, -1, :]

        # Apply dropout and fully connected layers
        out = self.dropout1(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

    def predict_with_uncertainty(self, x: torch.Tensor, passes: int = 30) -> dict:
        """
        Predict with uncertainty quantification
        """
        return mc_dropout_predict_with_confidence(self, x, passes)


class BayesianCryptoPredictor:
    """
    Bayesian crypto predictor with MC Dropout uncertainty
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 128,
        dropout_rate: float = 0.2,
        mc_passes: int = 30,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.mc_passes = mc_passes

        # Create model
        self.model = MCDropoutLSTM(
            input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate
        )

        self.is_trained = False

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, lr: float = 0.001
    ) -> dict:
        """
        Train the Bayesian model
        """

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.model.train()
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

        self.is_trained = True

        return {"final_loss": losses[-1], "training_losses": losses, "epochs": epochs}

    def predict_with_uncertainty(self, X: np.ndarray, confidence_threshold: float = 0.8) -> dict:
        """
        Make predictions with uncertainty quantification
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_tensor = torch.FloatTensor(X)

        # MC Dropout prediction
        results = self.model.predict_with_uncertainty(X_tensor, self.mc_passes)

        # Filter by confidence
        high_confidence_mask = results["confidence"].flatten() >= confidence_threshold

        predictions = {
            "predictions": results["mean"].flatten(),
            "uncertainty": results["std"].flatten(),
            "confidence": results["confidence"].flatten(),
            "lower_bound": results["lower_bound"].flatten(),
            "upper_bound": results["upper_bound"].flatten(),
            "high_confidence_mask": high_confidence_mask,
            "high_confidence_count": np.sum(high_confidence_mask),
            "confidence_threshold": confidence_threshold,
            "mc_passes": self.mc_passes,
        }

        return predictions


if __name__ == "__main__":
    print("🧠 TESTING MC DROPOUT INFERENCE")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    sequence_length = 30
    input_size = 20

    # Generate synthetic time series data
    X = np.random.randn(n_samples, sequence_length, input_size)
    y = np.sum(X[:, -1, :5], axis=1) + 0.1 * np.random.randn(n_samples)  # Simple target

    print(f"📊 Dataset: {X.shape} -> {y.shape}")

    # Create Bayesian predictor
    predictor = BayesianCryptoPredictor(
        input_size=input_size, hidden_size=64, dropout_rate=0.3, mc_passes=50
    )

    # Train model
    print("\n🔄 Training Bayesian model...")

    train_results = predictor.train_model(X[:800], y[:800], epochs=50, lr=0.001)

    print(f"   Final training loss: {train_results['final_loss']:.6f}")

    # Test predictions with uncertainty
    print("\n🎯 Testing MC Dropout predictions...")

    test_results = predictor.predict_with_uncertainty(X[800:850], confidence_threshold=0.8)

    print(f"   Test predictions: {len(test_results['predictions'])}")
    print(f"   Mean uncertainty: {np.mean(test_results['uncertainty']):.4f}")
    print(f"   Mean confidence: {np.mean(test_results['confidence']):.4f}")
    print(f"   High confidence predictions: {test_results['high_confidence_count']}/50")

    # Show sample predictions with uncertainty
    print(f"\n📈 Sample predictions with uncertainty:")
    for i in range(5):
        pred = test_results["predictions"][i]
        unc = test_results["uncertainty"][i]
        conf = test_results["confidence"][i]
        actual = y[800 + i]

        print(f"   Sample {i + 1}: Pred={pred:.4f}±{unc:.4f}, Actual={actual:.4f}, Conf={conf:.3f}")

    print("\n✅ MC Dropout test completed")
