#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Advanced Feature Fusion Engine
Real feature fusion & cross-feature interactions for enhanced ML performance
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion engine"""

    enable_attention_mechanism: bool = True
    enable_cross_feature_interactions: bool = True
    enable_temporal_fusion: bool = True
    enable_multi_modal_fusion: bool = True
    attention_heads: int = 8
    fusion_hidden_dim: int = 256
    dropout_rate: float = 0.1
    max_sequence_length: int = 100
    feature_selection_k: int = 50
    pca_components: int = 20
    interaction_degree: int = 2


class AttentionFusionModule(nn.Module):
    """Advanced attention mechanism for feature fusion"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        if not HAS_TORCH:
            raise ImportError("PyTorch required for attention fusion")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Multi-head attention layers
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = features.shape

        # Generate queries, keys, values
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)

        # Concatenate heads
        attended_values = (
            attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )

        # Output projection
        output = self.output_proj(attended_values)

        # Residual connection and layer norm
        output = self.layer_norm(output + features)

        # Feed-forward network with residual connection
        ff_output = self.ff_network(output)
        output = self.layer_norm(output + ff_output)

        return output


class CrossFeatureInteractionEngine:
    """Engine for generating cross-feature interactions"""

    def __init__(self, config: FeatureFusionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CrossFeatureInteractionEngine")
        self.interaction_cache = {}
        self._lock = threading.RLock()

    def generate_polynomial_interactions(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate polynomial feature interactions"""
        from sklearn.preprocessing import PolynomialFeatures

        with self._lock:
            poly = PolynomialFeatures(
                degree=self.config.interaction_degree, interaction_only=True, include_bias=False
            )

            interaction_features = poly.fit_transform(features)
            interaction_names = poly.get_feature_names_out(feature_names)

            self.logger.debug(f"Generated {interaction_features.shape[1]} polynomial interactions")
            return interaction_features, list(interaction_names)

    def generate_ratio_interactions(
        self, features: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate ratio-based feature interactions"""
        n_features = features.shape[1]
        ratio_features = []
        ratio_names = []

        # Generate ratios between all feature pairs
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Avoid division by zero
                denominator = features[:, j]
                denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)

                ratio = features[:, i] / denominator
                ratio_features.append(ratio)
                ratio_names.append(f"{feature_names[i]}_div_{feature_names[j]}")

                # Reverse ratio
                numerator = features[:, i]
                numerator = np.where(np.abs(numerator) < 1e-8, 1e-8, numerator)

                reverse_ratio = features[:, j] / numerator
                ratio_features.append(reverse_ratio)
                ratio_names.append(f"{feature_names[j]}_div_{feature_names[i]}")

        if ratio_features:
            ratio_array = np.column_stack(ratio_features)
            self.logger.debug(f"Generated {len(ratio_names)} ratio interactions")
            return ratio_array, ratio_names
        else:
            return np.array([]).reshape(features.shape[0], 0), []

    def generate_statistical_interactions(
        self, features: np.ndarray, feature_names: List[str], window_size: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate statistical interactions (rolling statistics)"""
        stat_features = []
        stat_names = []

        if len(features) < window_size:
            return np.array([]).reshape(features.shape[0], 0), []

        # Rolling statistics for each feature
        for i, name in enumerate(feature_names):
            feature_series = pd.Series(features[:, i])

            # Rolling mean
            rolling_mean = feature_series.rolling(window=window_size, min_periods=1).mean().values
            stat_features.append(rolling_mean)
            stat_names.append(f"{name}_rolling_mean_{window_size}")

            # Rolling std
            rolling_std = (
                feature_series.rolling(window=window_size, min_periods=1).std().fillna(0).values
            )
            stat_features.append(rolling_std)
            stat_names.append(f"{name}_rolling_std_{window_size}")

            # Rolling correlation with price (if price feature exists)
            price_cols = [j for j, fname in enumerate(feature_names) if "price" in fname.lower()]
            if price_cols and i not in price_cols:
                price_series = pd.Series(features[:, price_cols[0]])
                rolling_corr = (
                    feature_series.rolling(window=window_size, min_periods=1)
                    .corr(price_series)
                    .fillna(0)
                    .values
                )
                stat_features.append(rolling_corr)
                stat_names.append(f"{name}_price_corr_{window_size}")

        if stat_features:
            stat_array = np.column_stack(stat_features)
            self.logger.debug(f"Generated {len(stat_names)} statistical interactions")
            return stat_array, stat_names
        else:
            return np.array([]).reshape(features.shape[0], 0), []


class FeatureFusionEngine:
    """Advanced feature fusion engine with multi-modal capabilities"""

    def __init__(self, config: Optional[FeatureFusionConfig] = None):
        self.config = config or FeatureFusionConfig()
        self.logger = logging.getLogger(f"{__name__}.FeatureFusionEngine")

        # Components
        self.cross_interaction_engine = CrossFeatureInteractionEngine(self.config)
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_models = {}

        # Attention module (if PyTorch available)
        self.attention_module = None
        if HAS_TORCH and self.config.enable_attention_mechanism:
            self._initialize_attention_module()

        # Feature metadata
        self.feature_metadata = {
            "price_features": [],
            "volume_features": [],
            "technical_features": [],
            "sentiment_features": [],
            "whale_features": [],
            "macro_features": [],
            "interaction_features": [],
        }

        self._lock = threading.RLock()
        self.logger.info("Feature Fusion Engine initialized with advanced capabilities")

    def _initialize_attention_module(self):
        """Initialize attention module for feature fusion"""
        try:
            # Will be dynamically sized based on input features
            self.attention_module = None
            self.logger.info("Attention module will be initialized on first use")
        except Exception as e:
            self.logger.warning(f"Failed to initialize attention module: {e}")
            self.attention_module = None

    def fuse_multi_modal_features(
        self, feature_dict: Dict[str, np.ndarray], target: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fuse features from multiple modalities with advanced interactions

        Args:
            feature_dict: Dictionary of features by modality
            target: Optional target values for supervised feature selection

        Returns:
            Dictionary containing fused features and metadata
        """
        with self._lock:
            start_time = datetime.now()

            try:
                # Validate inputs
                if not feature_dict:
                    raise ValueError("No features provided for fusion")

                # Align all feature arrays to same length
                feature_dict = self._align_feature_arrays(feature_dict)

                # Stage 1: Individual modality processing
                processed_modalities = {}
                for modality, features in feature_dict.items():
                    processed_modalities[modality] = self._process_single_modality(
                        features, modality, target
                    )

                # Stage 2: Cross-modality interactions
                if self.config.enable_cross_feature_interactions:
                    interaction_features = self._generate_cross_modality_interactions(
                        processed_modalities
                    )
                    processed_modalities["interactions"] = interaction_features

                # Stage 3: Temporal fusion (if enabled)
                if self.config.enable_temporal_fusion:
                    temporal_features = self._apply_temporal_fusion(processed_modalities)
                    processed_modalities["temporal"] = temporal_features

                # Stage 4: Attention-based fusion
                fused_features = self._apply_attention_fusion(processed_modalities)

                # Stage 5: Final feature selection and dimensionality reduction
                final_features, selected_indices = self._final_feature_selection(
                    fused_features, target
                )

                # Generate metadata
                fusion_metadata = {
                    "input_modalities": list(feature_dict.keys()),
                    "input_feature_counts": {k: v.shape[1] for k, v in feature_dict.items()},
                    "final_feature_count": final_features.shape[1],
                    "selected_feature_indices": selected_indices,
                    "fusion_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "fusion_timestamp": start_time.isoformat(),
                }

                self.logger.info(
                    f"Feature fusion completed: {sum(fusion_metadata['input_feature_counts'].values())} -> {fusion_metadata['final_feature_count']} features"
                )

                return {
                    "fused_features": final_features,
                    "metadata": fusion_metadata,
                    "feature_importance": self._calculate_feature_importance(
                        final_features, target
                    ),
                    "modality_contributions": self._calculate_modality_contributions(
                        processed_modalities
                    ),
                }

            except Exception as e:
                self.logger.error(f"Feature fusion failed: {e}")
                raise

    def _align_feature_arrays(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align all feature arrays to the same length"""
        if not feature_dict:
            return feature_dict

        # Find minimum length
        min_length = min(arr.shape[0] for arr in feature_dict.values())

        # Truncate all arrays to minimum length
        aligned_dict = {}
        for modality, features in feature_dict.items():
            aligned_dict[modality] = features[:min_length]

        return aligned_dict

    def _process_single_modality(
        self, features: np.ndarray, modality: str, target: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process features from a single modality"""
        if features.size == 0:
            return {"features": features, "feature_names": [], "scaler": None}

        # Generate feature names
        feature_names = [f"{modality}_feat_{i}" for i in range(features.shape[1])]

        # Scaling
        scaler_key = f"{modality}_scaler"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()

        scaled_features = self.scalers[scaler_key].fit_transform(features)

        # Generate interactions within modality
        if self.config.enable_cross_feature_interactions and features.shape[1] > 1:
            poly_features, poly_names = (
                self.cross_interaction_engine.generate_polynomial_interactions(
                    scaled_features, feature_names
                )
            )

            ratio_features, ratio_names = self.cross_interaction_engine.generate_ratio_interactions(
                scaled_features, feature_names
            )

            stat_features, stat_names = (
                self.cross_interaction_engine.generate_statistical_interactions(
                    scaled_features, feature_names
                )
            )

            # Combine all features
            all_features = [scaled_features]
            all_names = feature_names.copy()

            if poly_features.size > 0:
                all_features.append(poly_features)
                all_names.extend(poly_names)

            if ratio_features.size > 0:
                all_features.append(ratio_features)
                all_names.extend(ratio_names)

            if stat_features.size > 0:
                all_features.append(stat_features)
                all_names.extend(stat_names)

            combined_features = np.column_stack(all_features)
        else:
            combined_features = scaled_features
            all_names = feature_names

        # Feature selection within modality
        if target is not None and combined_features.shape[1] > self.config.feature_selection_k:
            selector_key = f"{modality}_selector"
            if selector_key not in self.feature_selectors:
                self.feature_selectors[selector_key] = SelectKBest(
                    score_func=mutual_info_regression,
                    k=min(self.config.feature_selection_k, combined_features.shape[1]),
                )

            selected_features = self.feature_selectors[selector_key].fit_transform(
                combined_features, target
            )
            selected_names = [
                all_names[i] for i in self.feature_selectors[selector_key].get_support(indices=True)
            ]
        else:
            selected_features = combined_features
            selected_names = all_names

        return {
            "features": selected_features,
            "feature_names": selected_names,
            "scaler": self.scalers[scaler_key],
            "original_shape": features.shape,
        }

    def _generate_cross_modality_interactions(
        self, processed_modalities: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Generate interactions between different modalities"""
        interaction_features = []
        interaction_names = []

        modality_names = list(processed_modalities.keys())

        # Cross-modality correlations
        for i, mod1 in enumerate(modality_names):
            for j, mod2 in enumerate(modality_names[i + 1 :], i + 1):
                feat1 = processed_modalities[mod1]["features"]
                feat2 = processed_modalities[mod2]["features"]

                if feat1.size == 0 or feat2.size == 0:
                    continue

                # Calculate cross-correlations
                min_cols = min(feat1.shape[1], feat2.shape[1], 5)  # Limit to prevent explosion

                for k in range(min_cols):
                    if k < feat1.shape[1] and k < feat2.shape[1]:
                        # Element-wise product
                        interaction = feat1[:, k] * feat2[:, k]
                        interaction_features.append(interaction)
                        interaction_names.append(f"{mod1}_{k}_x_{mod2}_{k}")

                        # Difference
                        diff = feat1[:, k] - feat2[:, k]
                        interaction_features.append(diff)
                        interaction_names.append(f"{mod1}_{k}_minus_{mod2}_{k}")

        if interaction_features:
            interaction_array = np.column_stack(interaction_features)

            return {
                "features": interaction_array,
                "feature_names": interaction_names,
                "scaler": None,
                "original_shape": interaction_array.shape,
            }
        else:
            return {
                "features": np.array([]).reshape(0, 0),
                "feature_names": [],
                "scaler": None,
                "original_shape": (0, 0),
            }

    def _apply_temporal_fusion(self, processed_modalities: Dict[str, Dict]) -> Dict[str, Any]:
        """Apply temporal fusion across modalities"""
        temporal_features = []
        temporal_names = []

        window_sizes = [5, 10, 20]

        for modality, data in processed_modalities.items():
            features = data["features"]
            if features.size == 0:
                continue

            feature_names = data["feature_names"]

            for window in window_sizes:
                if len(features) < window:
                    continue

                # Rolling statistics
                for i, name in enumerate(
                    feature_names[: min(10, len(feature_names))]
                ):  # Limit features
                    if i >= features.shape[1]:
                        continue

                    series = pd.Series(features[:, i])

                    # Rolling mean
                    rolling_mean = series.rolling(window=window, min_periods=1).mean().values
                    temporal_features.append(rolling_mean)
                    temporal_names.append(f"{name}_rolling_mean_{window}")

                    # Rolling volatility
                    rolling_std = (
                        series.rolling(window=window, min_periods=1).std().fillna(0).values
                    )
                    temporal_features.append(rolling_std)
                    temporal_names.append(f"{name}_rolling_std_{window}")

        if temporal_features:
            temporal_array = np.column_stack(temporal_features)

            return {
                "features": temporal_array,
                "feature_names": temporal_names,
                "scaler": None,
                "original_shape": temporal_array.shape,
            }
        else:
            return {
                "features": np.array([]).reshape(0, 0),
                "feature_names": [],
                "scaler": None,
                "original_shape": (0, 0),
            }

    def _apply_attention_fusion(self, processed_modalities: Dict[str, Dict]) -> np.ndarray:
        """Apply attention-based fusion to combine all modalities"""
        # Collect all features
        all_features = []

        for modality, data in processed_modalities.items():
            features = data["features"]
            if features.size > 0:
                all_features.append(features)

        if not all_features:
            return np.array([]).reshape(0, 0)

        # Concatenate all features
        concatenated_features = np.column_stack(all_features)

        # Apply attention if available
        if HAS_TORCH and self.config.enable_attention_mechanism and self.attention_module is None:
            try:
                # Initialize attention module with correct dimensions
                input_dim = concatenated_features.shape[1]
                self.attention_module = AttentionFusionModule(
                    input_dim=input_dim,
                    hidden_dim=min(self.config.fusion_hidden_dim, input_dim),
                    num_heads=min(self.config.attention_heads, input_dim // 8),
                )
                self.attention_module.eval()
            except Exception as e:
                self.logger.warning(f"Failed to initialize attention module: {e}")
                self.attention_module = None

        if self.attention_module is not None:
            try:
                # Convert to torch tensor
                features_tensor = torch.FloatTensor(concatenated_features).unsqueeze(
                    0
                )  # Add batch dimension

                with torch.no_grad():
                    attended_features = self.attention_module(features_tensor)
                    attended_features = attended_features.squeeze(
                        0
                    ).numpy()  # Remove batch dimension

                return attended_features
            except Exception as e:
                self.logger.warning(f"Attention fusion failed, falling back to concatenation: {e}")
                return concatenated_features
        else:
            return concatenated_features

    def _final_feature_selection(
        self, features: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Apply final feature selection and dimensionality reduction"""
        if features.size == 0:
            return features, []

        # Remove constant features
        feature_var = np.var(features, axis=0)
        non_constant_mask = feature_var > 1e-8
        features = features[:, non_constant_mask]
        non_constant_indices = np.where(non_constant_mask)[0].tolist()

        if features.size == 0:
            return features, []

        # Remove highly correlated features
        if features.shape[1] > 1:
            correlation_matrix = np.corrcoef(features.T)
            correlation_matrix = np.nan_to_num(correlation_matrix)

            # Find highly correlated pairs
            high_corr_mask = np.ones(features.shape[1], dtype=bool)
            for i in range(features.shape[1]):
                for j in range(i + 1, features.shape[1]):
                    if abs(correlation_matrix[i, j]) > 0.95:
                        high_corr_mask[j] = False

            features = features[:, high_corr_mask]
            selected_indices = [
                non_constant_indices[i] for i, mask in enumerate(high_corr_mask) if mask
            ]
        else:
            selected_indices = non_constant_indices

        # Apply PCA if too many features
        if features.shape[1] > self.config.pca_components:
            pca_key = "final_pca"
            if pca_key not in self.pca_models:
                self.pca_models[pca_key] = PCA(n_components=self.config.pca_components)

            features = self.pca_models[pca_key].fit_transform(features)
            # PCA creates new feature indices
            selected_indices = list(range(features.shape[1]))

        return features, selected_indices

    def _calculate_feature_importance(
        self, features: np.ndarray, target: Optional[np.ndarray] = None
    ) -> List[float]:
        """Calculate feature importance scores"""
        if target is None or features.size == 0:
            return [1.0] * features.shape[1]

        try:
            # Use mutual information for feature importance
            scores = mutual_info_regression(features, target)
            # Normalize scores
            if np.sum(scores) > 0:
                scores = scores / np.sum(scores)
            else:
                scores = np.ones_like(scores) / len(scores)

            return scores.tolist()
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return [1.0 / features.shape[1]] * features.shape[1]

    def _calculate_modality_contributions(
        self, processed_modalities: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate contribution of each modality to final features"""
        contributions = {}
        total_features = sum(
            data["features"].shape[1]
            for data in processed_modalities.values()
            if data["features"].size > 0
        )

        if total_features == 0:
            return contributions

        for modality, data in processed_modalities.items():
            feature_count = data["features"].shape[1] if data["features"].size > 0 else 0
            contributions[modality] = feature_count / total_features

        return contributions

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get feature fusion statistics"""
        with self._lock:
            stats = {
                "scalers_trained": len(self.scalers),
                "feature_selectors_trained": len(self.feature_selectors),
                "pca_models_trained": len(self.pca_models),
                "attention_module_available": self.attention_module is not None,
                "torch_available": HAS_TORCH,
                "config": {
                    "attention_enabled": self.config.enable_attention_mechanism,
                    "cross_interactions_enabled": self.config.enable_cross_feature_interactions,
                    "temporal_fusion_enabled": self.config.enable_temporal_fusion,
                    "feature_selection_k": self.config.feature_selection_k,
                },
            }

            return stats


# Singleton fusion engine
_fusion_engine = None
_fusion_lock = threading.Lock()


def get_feature_fusion_engine(config: Optional[FeatureFusionConfig] = None) -> FeatureFusionEngine:
    """Get the singleton feature fusion engine"""
    global _fusion_engine

    with _fusion_lock:
        if _fusion_engine is None:
            _fusion_engine = FeatureFusionEngine(config)
        return _fusion_engine


def fuse_multi_modal_features(
    feature_dict: Dict[str, np.ndarray], target: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Convenient function to fuse multi-modal features"""
    engine = get_feature_fusion_engine()
    return engine.fuse_multi_modal_features(feature_dict, target)
