#!/usr/bin/env python3
"""
Regime-Aware Mixture of Experts System
Different specialized models for bull/bear/sideways markets with intelligent routing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
from sklearn.cluster import KMeans
from hmmlearn import hmm

from ..core.structured_logger import get_logger


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERY = "recovery"


class RegimeDetector:
    """Market regime detection using multiple methods"""

    def __init__(self, lookback_window: int = 50):
        self.logger = get_logger("RegimeDetector")
        self.lookback_window = lookback_window
        self.hmm_model = None
        self.regime_thresholds = {
            "bull_volatility": 0.02,  # Low volatility for bull markets
            "bear_volatility": 0.05,  # High volatility for bear markets
            "sideways_trend": 0.001,  # Low trend for sideways
            "volatile_threshold": 0.04,  # High volatility threshold
        }

    def detect_regime_hmm(self, price_data: pd.DataFrame) -> np.ndarray:
        """Detect market regime using Hidden Markov Model"""

        try:
            self.logger.info("Detecting market regime using HMM")

            # Calculate features for regime detection
            features = self._calculate_regime_features(price_data)

            # Fit HMM model if not already fitted
            if self.hmm_model is None:
                self.hmm_model = hmm.GaussianHMM(
                    n_components=4, covariance_type="full", random_state=42
                )
                self.hmm_model.fit(features)

            # Predict regimes
            regime_probs = self.hmm_model.predict_proba(features)
            regime_states = self.hmm_model.predict(features)

            # Map HMM states to market regimes
            regime_mapping = self._map_hmm_states_to_regimes(features, regime_states)
            mapped_regimes = [regime_mapping[state] for state in regime_states]

            return np.array(mapped_regimes)

        except Exception as e:
            self.logger.error(f"HMM regime detection failed: {e}")
            # Fallback to rule-based detection
            return self.detect_regime_rules(price_data)

    def detect_regime_rules(self, price_data: pd.DataFrame) -> np.ndarray:
        """Detect market regime using rule-based approach"""

        try:
            self.logger.info("Detecting market regime using rules")

            # Calculate market indicators
            returns = price_data["close"].pct_change().fillna(0)
            volatility = returns.rolling(self.lookback_window).std()
            trend = returns.rolling(self.lookback_window).mean()

            regimes = []

            for i in range(len(price_data)):
                vol = volatility.iloc[i] if i < len(volatility) else 0.02
                tr = trend.iloc[i] if i < len(trend) else 0.0

                # Rule-based regime classification
                if vol > self.regime_thresholds["volatile_threshold"]:
                    regime = MarketRegime.VOLATILE
                elif tr > 0.002:  # Strong positive trend
                    if vol < self.regime_thresholds["bull_volatility"]:
                        regime = MarketRegime.BULL
                    else:
                        regime = MarketRegime.RECOVERY
                elif tr < -0.002:  # Strong negative trend
                    regime = MarketRegime.BEAR
                else:  # Low trend
                    regime = MarketRegime.SIDEWAYS

                regimes.append(regime)

            return np.array(regimes)

        except Exception as e:
            self.logger.error(f"Rule-based regime detection failed: {e}")
            # Return default regime
            return np.array([MarketRegime.SIDEWAYS] * len(price_data))

    def _calculate_regime_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Calculate features for regime detection"""

        # Price-based features
        returns = price_data["close"].pct_change().fillna(0)
        log_returns = np.log(price_data["close"] / price_data["close"].shift(1)).fillna(0)

        # Volatility features
        volatility = returns.rolling(20).std().fillna(0.02)

        # Trend features
        short_ma = price_data["close"].rolling(10).mean()
        long_ma = price_data["close"].rolling(30).mean()
        trend_strength = ((short_ma - long_ma) / long_ma).fillna(0)

        # Volume features
        volume_ma = price_data["volume"].rolling(20).mean()
        volume_ratio = (price_data["volume"] / volume_ma).fillna(1.0)

        # Combine features
        features = np.column_stack(
            [
                returns.values,
                volatility.values,
                trend_strength.values,
                np.log(volume_ratio.values + 1e-8),
            ]
        )

        return features

    def _map_hmm_states_to_regimes(
        self, features: np.ndarray, states: np.ndarray
    ) -> Dict[int, MarketRegime]:
        """Map HMM states to interpretable market regimes"""

        mapping = {}

        for state in np.unique(states):
            state_mask = states == state
            state_features = features[state_mask]

            # Analyze characteristics of this state
            avg_volatility = np.mean(state_features[:, 1])
            avg_trend = np.mean(state_features[:, 2])
            avg_returns = np.mean(state_features[:, 0])

            # Map to regime based on characteristics
            if avg_volatility > 0.04:
                mapping[state] = MarketRegime.VOLATILE
            elif avg_trend > 0.01 and avg_returns > 0:
                mapping[state] = MarketRegime.BULL
            elif avg_trend < -0.01 and avg_returns < 0:
                mapping[state] = MarketRegime.BEAR
            elif abs(avg_trend) < 0.005:
                mapping[state] = MarketRegime.SIDEWAYS
            else:
                mapping[state] = MarketRegime.RECOVERY

        return mapping


class ExpertModel(nn.Module):
    """Individual expert model for specific market regime"""

    def __init__(
        self, input_size: int, hidden_size: int = 64, regime: MarketRegime = MarketRegime.BULL
    ):
        super(ExpertModel, self).__init__()
        self.regime = regime
        self.input_size = input_size

        # Architecture adapted for specific regime
        if regime == MarketRegime.BULL:
            # Bull market expert: focus on momentum and growth patterns
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
        elif regime == MarketRegime.BEAR:
            # Bear market expert: focus on risk and downside protection
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_size // 2, 1),
            )
        elif regime == MarketRegime.SIDEWAYS:
            # Sideways expert: focus on mean reversion
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )
        else:  # VOLATILE or RECOVERY
            # Volatile market expert: robust to noise
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x):
        return self.layers(x)


class GatingNetwork(nn.Module):
    """Gating network to weight expert predictions"""

    def __init__(self, input_size: int, n_experts: int = 5):
        super(GatingNetwork, self).__init__()
        self.n_experts = n_experts

        self.gating_layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.gating_layers(x)


class MixtureOfExpertsRegimeRouter:
    """Complete Mixture of Experts system with regime-aware routing"""

    def __init__(self, input_size: int, device: str = "cpu"):
        self.logger = get_logger("MoE_RegimeRouter")
        self.device = device
        self.input_size = input_size

        # Initialize regime detector
        self.regime_detector = RegimeDetector()

        # Initialize expert models for each regime
        self.experts = {
            MarketRegime.BULL: ExpertModel(input_size, regime=MarketRegime.BULL).to(device),
            MarketRegime.BEAR: ExpertModel(input_size, regime=MarketRegime.BEAR).to(device),
            MarketRegime.SIDEWAYS: ExpertModel(input_size, regime=MarketRegime.SIDEWAYS).to(device),
            MarketRegime.VOLATILE: ExpertModel(input_size, regime=MarketRegime.VOLATILE).to(device),
            MarketRegime.RECOVERY: ExpertModel(input_size, regime=MarketRegime.RECOVERY).to(device),
        }

        # Initialize gating network
        self.gating_network = GatingNetwork(input_size, len(self.experts)).to(device)

        self.is_trained = False

    def train_experts(
        self, X_train: np.ndarray, y_train: np.ndarray, price_data: pd.DataFrame, epochs: int = 100
    ) -> Dict[str, Any]:
        """Train all expert models and gating network"""

        self.logger.info("Training mixture of experts system")

        try:
            # Detect regimes for training data
            regimes = self.regime_detector.detect_regime_hmm(price_data)

            training_results = {}

            # Train each expert on regime-specific data
            for regime, expert_model in self.experts.items():
                regime_mask = np.array([r == regime for r in regimes])

                if np.sum(regime_mask) < 10:  # Not enough data for this regime
                    self.logger.warning(f"Insufficient data for {regime.value} regime")
                    continue

                X_regime = X_train[regime_mask]
                y_regime = y_train[regime_mask]

                # Train expert
                expert_loss = self._train_expert(expert_model, X_regime, y_regime, epochs)
                training_results[regime.value] = {
                    "samples": len(X_regime),
                    "final_loss": expert_loss,
                    "trained": True,
                }

                self.logger.info(f"Trained {regime.value} expert on {len(X_regime)} samples")

            # Train gating network
            gating_loss = self._train_gating_network(X_train, regimes, epochs)
            training_results["gating_network"] = {"final_loss": gating_loss, "trained": True}

            self.is_trained = True

            self.logger.info("Mixture of experts training completed")
            return training_results

        except Exception as e:
            self.logger.error(f"MoE training failed: {e}")
            return {}

    def _train_expert(
        self, expert_model: ExpertModel, X: np.ndarray, y: np.ndarray, epochs: int
    ) -> float:
        """Train individual expert model"""

        expert_model.train()
        optimizer = torch.optim.Adam(expert_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = expert_model(X_tensor)
            loss = criterion(predictions.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

        return loss.item()

    def _train_gating_network(self, X: np.ndarray, regimes: np.ndarray, epochs: int) -> float:
        """Train gating network to predict regime weights"""

        self.gating_network.train()
        optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Convert regimes to one-hot encoding
        regime_to_idx = {regime: idx for idx, regime in enumerate(MarketRegime)}
        regime_indices = [regime_to_idx[regime] for regime in regimes]

        X_tensor = torch.FloatTensor(X).to(self.device)
        regime_tensor = torch.LongTensor(regime_indices).to(self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            gating_weights = self.gating_network(X_tensor)
            loss = criterion(gating_weights, regime_tensor)
            loss.backward()
            optimizer.step()

        return loss.item()

    def predict_with_routing(
        self, X: np.ndarray, price_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Generate predictions using mixture of experts with regime routing"""

        if not self.is_trained:
            raise ValueError("MoE system must be trained first")

        try:
            self.logger.info(f"Generating MoE predictions for {len(X)} samples")

            # Detect current regime
            if price_data is not None:
                current_regime = self.regime_detector.detect_regime_rules(price_data)[-1]
            else:
                current_regime = MarketRegime.SIDEWAYS  # Default

            X_tensor = torch.FloatTensor(X).to(self.device)

            # Get gating weights
            self.gating_network.eval()
            with torch.no_grad():
                gating_weights = self.gating_network(X_tensor)

            # Get predictions from all experts
            expert_predictions = {}
            for regime, expert_model in self.experts.items():
                expert_model.eval()
                with torch.no_grad():
                    pred = expert_model(X_tensor)
                    expert_predictions[regime] = pred.cpu().numpy().flatten()

            # Weighted combination based on gating network
            gating_weights_np = gating_weights.cpu().numpy()
            regime_list = list(MarketRegime)

            final_predictions = np.zeros(len(X))
            for i, regime in enumerate(regime_list):
                if regime in expert_predictions:
                    final_predictions += gating_weights_np[:, i] * expert_predictions[regime]

            # Calculate prediction confidence based on expert agreement
            expert_values = np.array(list(expert_predictions.values()))
            expert_std = np.std(expert_values, axis=0)
            max_std = np.max(expert_std) + 1e-8
            expert_agreement = 1.0 - (expert_std / max_std)

            # Boost confidence for regime-specific predictions
            regime_boost = np.zeros(len(X))
            regime_idx = regime_list.index(current_regime)
            regime_boost = gating_weights_np[:, regime_idx] * 0.2  # Up to 20% boost

            final_confidence = np.clip(expert_agreement + regime_boost, 0.1, 0.99)

            self.logger.info(f"MoE routing complete - detected regime: {current_regime.value}")
            self.logger.info(f"Mean expert agreement: {np.mean(expert_agreement):.3f}")

            return {
                "predictions": final_predictions,
                "confidence_scores": final_confidence,
                "detected_regime": current_regime.value,
                "expert_predictions": expert_predictions,
                "gating_weights": gating_weights_np,
                "expert_agreement": expert_agreement,
            }

        except Exception as e:
            self.logger.error(f"MoE prediction failed: {e}")
            raise
