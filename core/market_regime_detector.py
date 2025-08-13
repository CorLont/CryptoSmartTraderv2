#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Market Regime Detection
Automatic regime detection using unsupervised learning and adaptive model switching
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.cluster import KMeans, DBSCAN, GaussianMixture
    from sklearn.decomposition import PCA, ICA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.ensemble import IsolationForest
    from sklearn.manifold import TSNE

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    CRASH = "crash"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class DetectionMethod(Enum):
    AUTOENCODER = "autoencoder"
    CLUSTERING = "clustering"
    PCA_BASED = "pca_based"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


@dataclass
class RegimeDetectionConfig:
    """Configuration for market regime detection"""

    # Detection methods
    primary_method: DetectionMethod = DetectionMethod.ENSEMBLE
    enable_autoencoder: bool = True
    enable_clustering: bool = True
    enable_pca_analysis: bool = True

    # Data parameters
    lookback_periods: int = 100
    min_regime_duration: int = 5
    regime_change_threshold: float = 0.3

    # Feature engineering
    technical_indicators: List[str] = field(
        default_factory=lambda: [
            "sma_short",
            "sma_long",
            "ema_short",
            "ema_long",
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "volatility",
            "volume_sma",
            "atr",
        ]
    )

    # Clustering parameters
    n_clusters: int = 6
    clustering_method: str = "kmeans"  # 'kmeans', 'gmm', 'dbscan'

    # Autoencoder parameters
    autoencoder_latent_dim: int = 8
    autoencoder_epochs: int = 100
    reconstruction_threshold: float = 0.1

    # Regime validation
    min_confidence_score: float = 0.7
    ensemble_voting_threshold: float = 0.6

    # Model persistence
    save_models: bool = True
    model_cache_dir: str = "models/regime_detection"


@dataclass
class RegimeDetectionResult:
    """Result of market regime detection"""

    regime: MarketRegime
    confidence: float
    method: DetectionMethod
    features_used: List[str]
    detection_timestamp: datetime
    regime_probability: Dict[MarketRegime, float] = field(default_factory=dict)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeTransition:
    """Market regime transition record"""

    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    trigger_features: List[str]
    transition_strength: float


class AutoencoderRegimeDetector(nn.Module):
    """Autoencoder for unsupervised regime detection"""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, latent_dim),
            nn.Tanh(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim),
        )

        # Regime classifier (trained on latent representations)
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 2, len(MarketRegime)),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """Forward pass"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        regime_probs = self.regime_classifier(latent)

        return reconstructed, latent, regime_probs

    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)

    def get_reconstruction_error(self, x):
        """Calculate reconstruction error"""
        with torch.no_grad():
            reconstructed, _, _ = self.forward(x)
            mse_loss = F.mse_loss(reconstructed, x, reduction="none")
            return mse_loss.mean(dim=1)


class MarketRegimeDetector:
    """Advanced market regime detector using multiple unsupervised methods"""

    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        self.config = config or RegimeDetectionConfig()
        self.logger = logging.getLogger(f"{__name__}.MarketRegimeDetector")

        # Models and components
        self.autoencoder: Optional[AutoencoderRegimeDetector] = None
        self.clustering_model = None
        self.pca_model = None

        # Initialize scaler based on availability
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # Detection state
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.regime_confidence: float = 0.0
        self.regime_history: List[RegimeDetectionResult] = []
        self.regime_transitions: List[RegimeTransition] = []

        # Feature cache
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.regime_features: Dict[MarketRegime, Dict[str, float]] = {}

        # Model persistence
        self.model_cache_dir = Path(self.config.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Initialize device
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.logger.info(f"Market Regime Detector initialized on {self.device}")

    def fit(self, data: pd.DataFrame, target_column: str = "close") -> "MarketRegimeDetector":
        """Fit the regime detection models"""
        with self._lock:
            try:
                self.logger.info(f"Training regime detection models on {len(data)} samples")

                # Extract and engineer features
                features = self._engineer_regime_features(data, target_column)

                if len(features) < self.config.lookback_periods:
                    self.logger.warning(
                        f"Insufficient data for training: {len(features)} < {self.config.lookback_periods}"
                    )
                    return self

                # Scale features
                if self.scaler is not None:
                    scaled_features = self.scaler.fit_transform(features.values)
                else:
                    # Simple normalization fallback
                    feature_values = features.values
                    scaled_features = (feature_values - feature_values.mean(axis=0)) / (
                        feature_values.std(axis=0) + 1e-8
                    )

                # Train autoencoder
                if self.config.enable_autoencoder and HAS_TORCH:
                    self._train_autoencoder(scaled_features)

                # Train clustering models
                if self.config.enable_clustering and HAS_SKLEARN:
                    self._train_clustering_models(scaled_features)

                # Train PCA model
                if self.config.enable_pca_analysis and HAS_SKLEARN:
                    self._train_pca_model(scaled_features)

                # Extract regime characteristics
                self._extract_regime_characteristics(scaled_features, features.index)

                # Save models
                if self.config.save_models:
                    self._save_models()

                self.logger.info("Regime detection models trained successfully")
                return self

            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                return self

    def detect_regime(
        self, data: pd.DataFrame, target_column: str = "close"
    ) -> RegimeDetectionResult:
        """Detect current market regime"""
        with self._lock:
            try:
                # Extract features for current period
                features = self._engineer_regime_features(data, target_column)

                if len(features) < self.config.min_regime_duration:
                    return RegimeDetectionResult(
                        regime=MarketRegime.UNKNOWN,
                        confidence=0.0,
                        method=DetectionMethod.STATISTICAL,
                        features_used=[],
                        detection_timestamp=datetime.now(),
                    )

                # Use recent data for detection
                recent_features = features.tail(self.config.lookback_periods)

                if self.scaler is not None:
                    scaled_features = self.scaler.transform(recent_features.values)
                else:
                    # Simple normalization fallback
                    feature_values = recent_features.values
                    scaled_features = (feature_values - feature_values.mean(axis=0)) / (
                        feature_values.std(axis=0) + 1e-8
                    )

                # Ensemble detection
                if self.config.primary_method == DetectionMethod.ENSEMBLE:
                    result = self._ensemble_detection(scaled_features, recent_features.index)

                elif self.config.primary_method == DetectionMethod.AUTOENCODER:
                    result = self._autoencoder_detection(scaled_features, recent_features.index)

                elif self.config.primary_method == DetectionMethod.CLUSTERING:
                    result = self._clustering_detection(scaled_features, recent_features.index)

                elif self.config.primary_method == DetectionMethod.PCA_BASED:
                    result = self._pca_detection(scaled_features, recent_features.index)

                else:  # STATISTICAL
                    result = self._statistical_detection(recent_features)

                # Update regime history
                self._update_regime_history(result)

                # Check for regime transitions
                self._check_regime_transition(result)

                return result

            except Exception as e:
                self.logger.error(f"Regime detection failed: {e}")
                return RegimeDetectionResult(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    method=DetectionMethod.STATISTICAL,
                    features_used=[],
                    detection_timestamp=datetime.now(),
                )

    def _engineer_regime_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Engineer features for regime detection"""
        try:
            features = pd.DataFrame(index=data.index)

            # Price-based features
            if target_column in data.columns:
                price = (
                    data[target_column].fillna(method="ffill")
                    if hasattr(data[target_column], "fillna")
                    else data[target_column]
                )

                # Moving averages
                features["sma_short"] = price.rolling(10).mean()
                features["sma_long"] = price.rolling(50).mean()
                features["ema_short"] = price.ewm(span=12).mean()
                features["ema_long"] = price.ewm(span=26).mean()

                # Price ratios
                features["price_sma_ratio"] = price / features["sma_long"]
                features["sma_cross"] = features["sma_short"] / features["sma_long"]

                # Returns and volatility
                returns = price.pct_change().fillna(0)
                features["returns"] = returns
                features["returns_abs"] = returns.abs()
                features["volatility"] = returns.rolling(20).std()
                features["volatility_short"] = returns.rolling(5).std()

                # Momentum indicators
                features["rsi"] = self._calculate_rsi(price)
                features["macd"], features["macd_signal"] = self._calculate_macd(price)
                features["macd_histogram"] = features["macd"] - features["macd_signal"]

                # Bollinger bands
                bb_upper, bb_lower = self._calculate_bollinger_bands(price)
                features["bollinger_upper"] = bb_upper
                features["bollinger_lower"] = bb_lower
                features["bollinger_position"] = (price - bb_lower) / (bb_upper - bb_lower)

                # ATR (Average True Range)
                if all(col in data.columns for col in ["high", "low"]):
                    features["atr"] = self._calculate_atr(data["high"], data["low"], price)

                # Trend strength
                features["trend_strength"] = price.rolling(20).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                )

            # Volume-based features
            if "volume" in data.columns:
                volume = data["volume"].fillna(0)
                features["volume"] = volume
                features["volume_sma"] = volume.rolling(20).mean()
                features["volume_ratio"] = volume / features["volume_sma"]
                features["volume_volatility"] = volume.rolling(10).std()

                # Price-volume features
                if target_column in data.columns:
                    features["price_volume_corr"] = price.rolling(20).corr(volume)

            # Regime-specific features
            features["regime_momentum"] = features["returns"].rolling(10).mean()
            features["regime_volatility_regime"] = features["volatility"].rolling(20).mean()
            features["regime_trend_consistency"] = features["trend_strength"].rolling(10).std()

            # Higher-order moments
            features["skewness"] = returns.rolling(20).skew()
            features["kurtosis"] = returns.rolling(20).kurt()

            # Fill NaN values
            try:
                features = features.fillna(method="ffill").fillna(0)
            except Exception:
                features = features.fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)

    def _train_autoencoder(self, features: np.ndarray):
        """Train autoencoder for regime detection"""
        try:
            if not HAS_TORCH:
                self.logger.warning("PyTorch not available, skipping autoencoder training")
                return

            input_dim = features.shape[1]
            self.autoencoder = AutoencoderRegimeDetector(
                input_dim=input_dim, latent_dim=self.config.autoencoder_latent_dim
            ).to(self.device)

            # Prepare data
            X_tensor = torch.FloatTensor(features).to(self.device)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Training setup
            optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
            reconstruction_criterion = nn.MSELoss()

            # Training loop
            self.autoencoder.train()
            for epoch in range(self.config.autoencoder_epochs):
                total_loss = 0.0

                for batch in dataloader:
                    batch_data = batch[0]

                    optimizer.zero_grad()

                    reconstructed, latent, regime_probs = self.autoencoder(batch_data)

                    # Reconstruction loss
                    recon_loss = reconstruction_criterion(reconstructed, batch_data)

                    # Total loss (can add regularization terms here)
                    total_loss_batch = recon_loss

                    total_loss_batch.backward()
                    optimizer.step()

                    total_loss += total_loss_batch.item()

                if epoch % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.logger.debug(f"Autoencoder epoch {epoch}, loss: {avg_loss:.4f}")

            self.logger.info("Autoencoder training completed")

        except Exception as e:
            self.logger.error(f"Autoencoder training failed: {e}")

    def _train_clustering_models(self, features: np.ndarray):
        """Train clustering models for regime detection"""
        try:
            if not HAS_SKLEARN:
                self.logger.warning("Scikit-learn not available, skipping clustering")
                return

            if self.config.clustering_method == "kmeans":
                self.clustering_model = KMeans(
                    n_clusters=self.config.n_clusters, random_state=42, n_init=10
                )

            elif self.config.clustering_method == "gmm":
                self.clustering_model = GaussianMixture(
                    n_components=self.config.n_clusters, random_state=42
                )

            elif self.config.clustering_method == "dbscan":
                self.clustering_model = DBSCAN(eps=0.5, min_samples=5)

            # Fit clustering model
            cluster_labels = self.clustering_model.fit_predict(features)

            # Evaluate clustering quality
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(features, cluster_labels)
                calinski = calinski_harabasz_score(features, cluster_labels)

                self.logger.info(
                    f"Clustering quality - Silhouette: {silhouette:.3f}, Calinski: {calinski:.1f}"
                )

            self.logger.info(
                f"Clustering model trained with {len(np.unique(cluster_labels))} clusters"
            )

        except Exception as e:
            self.logger.error(f"Clustering training failed: {e}")

    def _train_pca_model(self, features: np.ndarray):
        """Train PCA model for dimensionality reduction and regime detection"""
        try:
            if not HAS_SKLEARN:
                self.logger.warning("Scikit-learn not available, skipping PCA")
                return

            # Fit PCA
            n_components = min(10, features.shape[1])
            self.pca_model = PCA(n_components=n_components)

            pca_features = self.pca_model.fit_transform(features)

            # Calculate explained variance
            explained_variance = self.pca_model.explained_variance_ratio_.sum()

            self.logger.info(f"PCA model trained, explained variance: {explained_variance:.3f}")

        except Exception as e:
            self.logger.error(f"PCA training failed: {e}")

    def _extract_regime_characteristics(self, features: np.ndarray, index: pd.Index):
        """Extract characteristics for each detected regime"""
        try:
            # This is a simplified implementation
            # In practice, you would use the clustering results to define regime characteristics

            for regime in MarketRegime:
                if regime != MarketRegime.UNKNOWN:
                    # For now, create default characteristics
                    self.regime_features[regime] = {
                        "volatility_threshold": 0.02,
                        "trend_threshold": 0.01,
                        "volume_threshold": 1.5,
                    }

        except Exception as e:
            self.logger.error(f"Regime characteristic extraction failed: {e}")

    def _ensemble_detection(self, features: np.ndarray, index: pd.Index) -> RegimeDetectionResult:
        """Ensemble-based regime detection"""
        try:
            detection_methods = []
            regime_votes = {}

            # Autoencoder detection
            if self.autoencoder is not None:
                autoencoder_result = self._autoencoder_detection(features, index)
                detection_methods.append(autoencoder_result)
                regime_votes[autoencoder_result.regime] = (
                    regime_votes.get(autoencoder_result.regime, 0) + autoencoder_result.confidence
                )

            # Clustering detection
            if self.clustering_model is not None:
                clustering_result = self._clustering_detection(features, index)
                detection_methods.append(clustering_result)
                regime_votes[clustering_result.regime] = (
                    regime_votes.get(clustering_result.regime, 0) + clustering_result.confidence
                )

            # PCA detection
            if self.pca_model is not None:
                pca_result = self._pca_detection(features, index)
                detection_methods.append(pca_result)
                regime_votes[pca_result.regime] = (
                    regime_votes.get(pca_result.regime, 0) + pca_result.confidence
                )

            # Statistical detection (always available)
            features_df = pd.DataFrame(features, index=index)
            statistical_result = self._statistical_detection(features_df)
            detection_methods.append(statistical_result)
            regime_votes[statistical_result.regime] = (
                regime_votes.get(statistical_result.regime, 0) + statistical_result.confidence
            )

            # Ensemble voting
            if regime_votes:
                best_regime = max(regime_votes.items(), key=lambda x: x[1])
                ensemble_confidence = best_regime[1] / len(detection_methods)

                # Normalize confidence
                ensemble_confidence = min(1.0, ensemble_confidence)

                return RegimeDetectionResult(
                    regime=best_regime[0],
                    confidence=ensemble_confidence,
                    method=DetectionMethod.ENSEMBLE,
                    features_used=list(range(features.shape[1])),
                    detection_timestamp=datetime.now(),
                    regime_probability={
                        regime: score / len(detection_methods)
                        for regime, score in regime_votes.items()
                    },
                    supporting_evidence={
                        "method_results": [
                            {
                                "method": result.method.value,
                                "regime": result.regime.value,
                                "confidence": result.confidence,
                            }
                            for result in detection_methods
                        ]
                    },
                )

            else:
                return RegimeDetectionResult(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    method=DetectionMethod.ENSEMBLE,
                    features_used=[],
                    detection_timestamp=datetime.now(),
                )

        except Exception as e:
            self.logger.error(f"Ensemble detection failed: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                method=DetectionMethod.ENSEMBLE,
                features_used=[],
                detection_timestamp=datetime.now(),
            )

    def _autoencoder_detection(
        self, features: np.ndarray, index: pd.Index
    ) -> RegimeDetectionResult:
        """Autoencoder-based regime detection"""
        try:
            if self.autoencoder is None or not HAS_TORCH:
                return self._statistical_detection(pd.DataFrame(features, index=index))

            self.autoencoder.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(features).to(self.device)

                # Get reconstruction error and regime probabilities
                reconstructed, latent, regime_probs = self.autoencoder(X_tensor)
                reconstruction_errors = self.autoencoder.get_reconstruction_error(X_tensor)

                # Average regime probabilities over recent period
                avg_regime_probs = regime_probs.mean(dim=0).cpu().numpy()

                # Map probabilities to regimes
                regime_list = list(MarketRegime)
                max_prob_idx = np.argmax(avg_regime_probs)
                detected_regime = (
                    regime_list[max_prob_idx]
                    if max_prob_idx < len(regime_list)
                    else MarketRegime.UNKNOWN
                )

                confidence = float(avg_regime_probs[max_prob_idx])

                # Check for anomalies (high reconstruction error might indicate regime change)
                avg_reconstruction_error = reconstruction_errors.mean().item()
                if avg_reconstruction_error > self.config.reconstruction_threshold:
                    confidence *= 0.7  # Reduce confidence for anomalous periods

                regime_probability = {
                    regime_list[i]: float(avg_regime_probs[i])
                    for i in range(min(len(regime_list), len(avg_regime_probs)))
                }

                return RegimeDetectionResult(
                    regime=detected_regime,
                    confidence=confidence,
                    method=DetectionMethod.AUTOENCODER,
                    features_used=list(range(features.shape[1])),
                    detection_timestamp=datetime.now(),
                    regime_probability=regime_probability,
                    supporting_evidence={
                        "reconstruction_error": avg_reconstruction_error,
                        "latent_representation": latent.mean(dim=0).cpu().numpy().tolist(),
                    },
                )

        except Exception as e:
            self.logger.error(f"Autoencoder detection failed: {e}")
            return self._statistical_detection(pd.DataFrame(features, index=index))

    def _clustering_detection(self, features: np.ndarray, index: pd.Index) -> RegimeDetectionResult:
        """Clustering-based regime detection"""
        try:
            if self.clustering_model is None or not HAS_SKLEARN:
                return self._statistical_detection(pd.DataFrame(features, index=index))

            # Predict cluster for recent data
            recent_features = features[-min(10, len(features)) :]  # Last 10 periods
            cluster_predictions = self.clustering_model.predict(recent_features)

            # Most common cluster
            most_common_cluster = np.bincount(cluster_predictions).argmax()

            # Map cluster to regime (simplified mapping)
            cluster_to_regime = {
                0: MarketRegime.BULL_MARKET,
                1: MarketRegime.BEAR_MARKET,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.HIGH_VOLATILITY,
                4: MarketRegime.TRENDING_UP,
                5: MarketRegime.TRENDING_DOWN,
            }

            detected_regime = cluster_to_regime.get(most_common_cluster, MarketRegime.UNKNOWN)

            # Calculate confidence based on cluster consistency
            cluster_consistency = np.mean(cluster_predictions == most_common_cluster)

            return RegimeDetectionResult(
                regime=detected_regime,
                confidence=cluster_consistency,
                method=DetectionMethod.CLUSTERING,
                features_used=list(range(features.shape[1])),
                detection_timestamp=datetime.now(),
                supporting_evidence={
                    "cluster_id": int(most_common_cluster),
                    "cluster_consistency": float(cluster_consistency),
                    "cluster_predictions": cluster_predictions.tolist(),
                },
            )

        except Exception as e:
            self.logger.error(f"Clustering detection failed: {e}")
            return self._statistical_detection(pd.DataFrame(features, index=index))

    def _pca_detection(self, features: np.ndarray, index: pd.Index) -> RegimeDetectionResult:
        """PCA-based regime detection"""
        try:
            if self.pca_model is None or not HAS_SKLEARN:
                return self._statistical_detection(pd.DataFrame(features, index=index))

            # Transform to PCA space
            pca_features = self.pca_model.transform(features)

            # Analyze PCA components for regime characteristics
            recent_pca = pca_features[-min(20, len(pca_features)) :]

            # Simple heuristic based on PCA components
            pc1_mean = np.mean(recent_pca[:, 0])
            pc2_mean = np.mean(recent_pca[:, 1]) if recent_pca.shape[1] > 1 else 0

            # Map PCA space to regimes (simplified)
            if pc1_mean > 0.5:
                if pc2_mean > 0:
                    detected_regime = MarketRegime.BULL_MARKET
                else:
                    detected_regime = MarketRegime.TRENDING_UP
            elif pc1_mean < -0.5:
                if pc2_mean > 0:
                    detected_regime = MarketRegime.BEAR_MARKET
                else:
                    detected_regime = MarketRegime.TRENDING_DOWN
            else:
                if abs(pc2_mean) > 0.5:
                    detected_regime = MarketRegime.HIGH_VOLATILITY
                else:
                    detected_regime = MarketRegime.SIDEWAYS

            # Confidence based on how clear the signal is
            confidence = min(1.0, (abs(pc1_mean) + abs(pc2_mean)) / 2)

            return RegimeDetectionResult(
                regime=detected_regime,
                confidence=confidence,
                method=DetectionMethod.PCA_BASED,
                features_used=list(range(features.shape[1])),
                detection_timestamp=datetime.now(),
                supporting_evidence={
                    "pc1_mean": float(pc1_mean),
                    "pc2_mean": float(pc2_mean),
                    "explained_variance": self.pca_model.explained_variance_ratio_.tolist(),
                },
            )

        except Exception as e:
            self.logger.error(f"PCA detection failed: {e}")
            return self._statistical_detection(pd.DataFrame(features, index=index))

    def _statistical_detection(self, features: pd.DataFrame) -> RegimeDetectionResult:
        """Statistical rule-based regime detection (fallback method)"""
        try:
            # Simple statistical regime detection based on returns and volatility
            if "returns" in features.columns and "volatility" in features.columns:
                recent_returns = features["returns"].tail(20).mean()
                recent_volatility = features["volatility"].tail(20).mean()

                # Regime classification rules
                if recent_volatility > 0.05:  # High volatility threshold
                    if abs(recent_returns) > 0.03:
                        detected_regime = (
                            MarketRegime.CRASH if recent_returns < 0 else MarketRegime.BREAKOUT
                        )
                    else:
                        detected_regime = MarketRegime.HIGH_VOLATILITY

                elif recent_volatility < 0.01:  # Low volatility threshold
                    detected_regime = MarketRegime.LOW_VOLATILITY

                elif recent_returns > 0.02:  # Bull market threshold
                    detected_regime = MarketRegime.BULL_MARKET

                elif recent_returns < -0.02:  # Bear market threshold
                    detected_regime = MarketRegime.BEAR_MARKET

                elif abs(recent_returns) < 0.005:  # Sideways market
                    detected_regime = MarketRegime.SIDEWAYS

                else:
                    detected_regime = (
                        MarketRegime.TRENDING_UP
                        if recent_returns > 0
                        else MarketRegime.TRENDING_DOWN
                    )

                # Confidence based on how clear the signals are
                volatility_signal = min(1.0, recent_volatility / 0.05)
                returns_signal = min(1.0, abs(recent_returns) / 0.03)
                confidence = (volatility_signal + returns_signal) / 2

            else:
                detected_regime = MarketRegime.UNKNOWN
                confidence = 0.0
                recent_returns = 0.0
                recent_volatility = 0.0

            return RegimeDetectionResult(
                regime=detected_regime,
                confidence=confidence,
                method=DetectionMethod.STATISTICAL,
                features_used=["returns", "volatility"],
                detection_timestamp=datetime.now(),
                supporting_evidence={
                    "recent_returns": float(recent_returns),
                    "recent_volatility": float(recent_volatility),
                },
            )

        except Exception as e:
            self.logger.error(f"Statistical detection failed: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                method=DetectionMethod.STATISTICAL,
                features_used=[],
                detection_timestamp=datetime.now(),
            )

    def _update_regime_history(self, result: RegimeDetectionResult):
        """Update regime detection history"""
        try:
            self.regime_history.append(result)

            # Keep only recent history
            max_history = 1000
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

            # Update current regime if confidence is high enough
            if result.confidence >= self.config.min_confidence_score:
                self.current_regime = result.regime
                self.regime_confidence = result.confidence

        except Exception as e:
            self.logger.error(f"Regime history update failed: {e}")

    def _check_regime_transition(self, result: RegimeDetectionResult):
        """Check for regime transitions"""
        try:
            if (
                self.current_regime != MarketRegime.UNKNOWN
                and result.regime != self.current_regime
                and result.confidence >= self.config.regime_change_threshold
            ):
                # Create transition record
                transition = RegimeTransition(
                    timestamp=datetime.now(),
                    from_regime=self.current_regime,
                    to_regime=result.regime,
                    confidence=result.confidence,
                    trigger_features=result.features_used,
                    transition_strength=abs(result.confidence - self.regime_confidence),
                )

                self.regime_transitions.append(transition)

                # Keep only recent transitions
                max_transitions = 100
                if len(self.regime_transitions) > max_transitions:
                    self.regime_transitions = self.regime_transitions[-max_transitions:]

                self.logger.info(
                    f"Regime transition detected: {self.current_regime.value} -> {result.regime.value}"
                )

        except Exception as e:
            self.logger.error(f"Transition check failed: {e}")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save autoencoder
            if self.autoencoder is not None and HAS_TORCH:
                torch.save(self.autoencoder.state_dict(), self.model_cache_dir / "autoencoder.pt")

            # Save clustering model
            if self.clustering_model is not None:
                with open(self.model_cache_dir / "clustering_model.pkl", "wb") as f:
                    pickle.dump(self.clustering_model, f)

            # Save PCA model
            if self.pca_model is not None:
                with open(self.model_cache_dir / "pca_model.pkl", "wb") as f:
                    pickle.dump(self.pca_model, f)

            # Save scaler
            with open(self.model_cache_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            self.logger.info("Models saved successfully")

        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load autoencoder
            autoencoder_path = self.model_cache_dir / "autoencoder.pt"
            if autoencoder_path.exists() and HAS_TORCH:
                # Need to know input dimensions to recreate model
                # This is a simplified approach
                self.autoencoder = AutoencoderRegimeDetector(
                    input_dim=20,  # Default, should be stored separately
                    latent_dim=self.config.autoencoder_latent_dim,
                ).to(self.device)
                self.autoencoder.load_state_dict(
                    torch.load(autoencoder_path, map_location=self.device)
                )

            # Load clustering model
            clustering_path = self.model_cache_dir / "clustering_model.pkl"
            if clustering_path.exists():
                with open(clustering_path, "rb") as f:
                    self.clustering_model = pickle.load(f)

            # Load PCA model
            pca_path = self.model_cache_dir / "pca_model.pkl"
            if pca_path.exists():
                with open(pca_path, "rb") as f:
                    self.pca_model = pickle.load(f)

            # Load scaler
            scaler_path = self.model_cache_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime detection summary"""
        with self._lock:
            return {
                "current_regime": self.current_regime.value,
                "regime_confidence": self.regime_confidence,
                "detection_method": self.config.primary_method.value,
                "regime_history_length": len(self.regime_history),
                "recent_transitions": [
                    {
                        "timestamp": t.timestamp.isoformat(),
                        "from_regime": t.from_regime.value,
                        "to_regime": t.to_regime.value,
                        "confidence": t.confidence,
                    }
                    for t in self.regime_transitions[-5:]  # Last 5 transitions
                ],
                "regime_distribution": self._get_regime_distribution(),
                "model_status": {
                    "autoencoder_available": self.autoencoder is not None,
                    "clustering_available": self.clustering_model is not None,
                    "pca_available": self.pca_model is not None,
                    "scaler_fitted": hasattr(self.scaler, "mean_"),
                },
                "avg_detection_confidence": np.mean(
                    [r.confidence for r in self.regime_history[-50:]]
                )
                if self.regime_history
                else 0.0,
            }

    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of regimes in recent history"""
        try:
            if not self.regime_history:
                return {}

            recent_regimes = [r.regime for r in self.regime_history[-100:]]  # Last 100 detections

            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

            total = len(recent_regimes)
            return {regime: count / total for regime, count in regime_counts.items()}

        except Exception:
            return {}

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        except Exception:
            return pd.Series(50, index=prices.index)

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd, macd_signal
        except Exception:
            return pd.Series(0, index=prices.index), pd.Series(0, index=prices.index)

    def _calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, lower_band
        except Exception:
            return pd.Series(prices), pd.Series(prices)

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=window).mean()
        except Exception:
            return pd.Series(0, index=high.index)


# Singleton market regime detector
_market_regime_detector = None
_mrd_lock = threading.Lock()


def get_market_regime_detector(
    config: Optional[RegimeDetectionConfig] = None,
) -> MarketRegimeDetector:
    """Get the singleton market regime detector"""
    global _market_regime_detector

    with _mrd_lock:
        if _market_regime_detector is None:
            _market_regime_detector = MarketRegimeDetector(config)
        return _market_regime_detector


def detect_current_regime(
    data: pd.DataFrame, target_column: str = "close"
) -> RegimeDetectionResult:
    """Convenient function to detect current market regime"""
    detector = get_market_regime_detector()
    return detector.detect_regime(data, target_column)
