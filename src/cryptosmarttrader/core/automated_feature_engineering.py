#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Automated Feature Engineering & Discovery
Auto-featuretools, deep feature synthesis, auto-crosses, attention-based feature pruning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import itertools
from pathlib import Path
import pickle

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
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None


class FeatureType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    PRICE = "price"
    CROSS_FEATURE = "cross_feature"
    SYNTHETIC = "synthetic"


class TransformationType(Enum):
    IDENTITY = "identity"
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    RECIPROCAL = "reciprocal"
    DIFF = "diff"
    PCT_CHANGE = "pct_change"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_STD = "rolling_std"
    ROLLING_MIN = "rolling_min"
    ROLLING_MAX = "rolling_max"
    EMA = "ema"
    RSI = "rsi"
    BOLLINGER = "bollinger"
    MACD = "macd"
    STOCHASTIC = "stochastic"


class FeatureImportanceMethod(Enum):
    SHAP = "shap"
    PERMUTATION = "permutation"
    TREE_IMPORTANCE = "tree_importance"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    ATTENTION_WEIGHTS = "attention_weights"


@dataclass
class FeatureSpec:
    """Specification for a generated feature"""

    name: str
    feature_type: FeatureType
    source_columns: List[str]
    transformation: TransformationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    performance_impact: float = 0.0


@dataclass
class FeatureEngineeringConfig:
    """Configuration for automated feature engineering"""

    # Feature generation
    max_features_per_iteration: int = 50
    max_total_features: int = 1000
    feature_selection_threshold: float = 0.01
    cross_feature_max_depth: int = 3

    # Temporal features
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    ema_spans: List[int] = field(default_factory=lambda: [12, 26, 50])

    # Feature importance
    importance_methods: List[FeatureImportanceMethod] = field(
        default_factory=lambda: [
            FeatureImportanceMethod.SHAP,
            FeatureImportanceMethod.TREE_IMPORTANCE,
            FeatureImportanceMethod.MUTUAL_INFO,
        ]
    )

    # Feature pruning
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    importance_threshold: float = 0.001

    # Regime-specific adaptation
    regime_adaptation_enabled: bool = True
    min_regime_samples: int = 100
    regime_feature_ratio: float = 0.3

    # Performance optimization
    feature_cache_size: int = 10000
    parallel_processing: bool = True
    gpu_acceleration: bool = True


@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis"""

    feature_name: str
    importance_score: float
    method: FeatureImportanceMethod
    confidence: float
    regime_specific: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AttentionBasedFeaturePruner:
    """Neural attention mechanism for feature importance and pruning"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if not HAS_TORCH:
            self.logger = logging.getLogger(f"{__name__}.AttentionBasedFeaturePruner")
            self.logger.warning("PyTorch not available - using simplified attention")
            return

        # Feature attention layers
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Softmax(dim=-1),
        )

        # Feature interaction attention
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x) -> Tuple[Any, Any]:
        """Forward pass with attention weights"""
        batch_size = x.shape[0]

        # Feature-level attention
        feature_weights = self.feature_attention(x)

        # Apply feature attention
        attended_features = x * feature_weights

        # Feature interaction attention
        if len(x.shape) == 2:
            x_expanded = x.unsqueeze(1)  # Add sequence dimension
        else:
            x_expanded = x

        interaction_output, interaction_weights = self.interaction_attention(
            x_expanded, x_expanded, x_expanded
        )

        # Combine attended features
        if len(interaction_output.shape) == 3:
            interaction_output = interaction_output.squeeze(1)

        combined_features = attended_features + interaction_output

        # Final prediction
        output = self.output_layer(combined_features)

        return output, feature_weights

    def get_feature_importance(self, x) -> Any:
        """Get feature importance scores"""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x)
            return attention_weights.mean(dim=0)


class DeepFeatureSynthesizer:
    """Deep feature synthesis for automatic feature generation"""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DeepFeatureSynthesizer")

        # Feature specifications
        self.feature_specs: Dict[str, FeatureSpec] = {}
        self.generated_features: Dict[str, pd.Series] = {}

        # Transformation functions
        self.transformations = self._initialize_transformations()

    def _initialize_transformations(self) -> Dict[TransformationType, Callable]:
        """Initialize transformation functions"""
        return {
            TransformationType.IDENTITY: lambda x, **kwargs: x,
            TransformationType.LOG: lambda x, **kwargs: np.log1p(np.abs(x)),
            TransformationType.SQRT: lambda x, **kwargs: np.sqrt(np.abs(x)),
            TransformationType.SQUARE: lambda x, **kwargs: x**2,
            TransformationType.RECIPROCAL: lambda x, **kwargs: 1 / (x + 1e-8),
            TransformationType.DIFF: lambda x, **kwargs: x.diff().fillna(0),
            TransformationType.PCT_CHANGE: lambda x, **kwargs: x.pct_change().fillna(0),
        }

    def generate_temporal_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Generate temporal features using rolling windows and technical indicators"""
        try:
            features_df = data.copy()

            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            for col in numeric_cols:
                series = data[col].fillna(method="ffill").fillna(0)

                # Rolling statistics
                for window in self.config.rolling_windows:
                    if len(series) >= window:
                        # Basic rolling features
                        features_df[f"{col}_rolling_mean_{window}"] = series.rolling(window).mean()
                        features_df[f"{col}_rolling_std_{window}"] = series.rolling(window).std()
                        features_df[f"{col}_rolling_min_{window}"] = series.rolling(window).min()
                        features_df[f"{col}_rolling_max_{window}"] = series.rolling(window).max()
                        features_df[f"{col}_rolling_skew_{window}"] = series.rolling(window).skew()
                        features_df[f"{col}_rolling_kurt_{window}"] = series.rolling(window).kurt()

                        # Technical indicators
                        if "price" in col.lower() or "close" in col.lower():
                            features_df[f"{col}_rsi_{window}"] = self._calculate_rsi(series, window)
                            bb_upper, bb_lower = self._calculate_bollinger_bands(series, window)
                            features_df[f"{col}_bollinger_upper_{window}"] = bb_upper
                            features_df[f"{col}_bollinger_lower_{window}"] = bb_lower

                        # Register feature specs
                        for suffix in ["mean", "std", "min", "max", "skew", "kurt"]:
                            feature_name = f"{col}_rolling_{suffix}_{window}"
                            self.feature_specs[feature_name] = FeatureSpec(
                                name=feature_name,
                                feature_type=FeatureType.TECHNICAL,
                                source_columns=[col],
                                transformation=TransformationType.ROLLING_MEAN,
                                parameters={"window": window, "suffix": suffix},
                            )

                # Exponential moving averages
                for span in self.config.ema_spans:
                    ema_col = f"{col}_ema_{span}"
                    features_df[ema_col] = series.ewm(span=span).mean()

                    self.feature_specs[ema_col] = FeatureSpec(
                        name=ema_col,
                        feature_type=FeatureType.TECHNICAL,
                        source_columns=[col],
                        transformation=TransformationType.EMA,
                        parameters={"span": span},
                    )

                # Lag features
                for lag in [1, 2, 3, 5, 10]:
                    lag_col = f"{col}_lag_{lag}"
                    features_df[lag_col] = series.shift(lag)

                    self.feature_specs[lag_col] = FeatureSpec(
                        name=lag_col,
                        feature_type=FeatureType.TEMPORAL,
                        source_columns=[col],
                        transformation=TransformationType.IDENTITY,
                        parameters={"lag": lag},
                    )

            return features_df.fillna(0)

        except Exception as e:
            self.logger.error(f"Temporal feature generation failed: {e}")
            return data

    def generate_cross_features(
        self, data: pd.DataFrame, target_col: str, max_depth: int = 2
    ) -> pd.DataFrame:
        """Generate cross features using feature interactions"""
        try:
            features_df = data.copy()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            # Limit to most important columns to avoid explosion
            if len(numeric_cols) > 20:
                # Select top correlated features with target
                correlations = (
                    data[numeric_cols].corrwith(data[target_col]).abs().sort_values(ascending=False)
                )
                numeric_cols = correlations.head(20).index.tolist()

            generated_count = 0
            max_features = self.config.max_features_per_iteration

            # Two-way interactions
            for col1, col2 in itertools.combinations(numeric_cols, 2):
                if generated_count >= max_features:
                    break

                series1 = data[col1].fillna(0)
                series2 = data[col2].fillna(0)

                # Multiplication
                mult_col = f"{col1}_x_{col2}"
                features_df[mult_col] = series1 * series2

                # Division (with protection)
                div_col = f"{col1}_div_{col2}"
                features_df[div_col] = series1 / (series2 + 1e-8)

                # Ratio
                ratio_col = f"{col1}_ratio_{col2}"
                features_df[ratio_col] = series1 / (series1 + series2 + 1e-8)

                # Difference
                diff_col = f"{col1}_diff_{col2}"
                features_df[diff_col] = series1 - series2

                # Register cross feature specs
                for op, op_name in [
                    (mult_col, "multiply"),
                    (div_col, "divide"),
                    (ratio_col, "ratio"),
                    (diff_col, "difference"),
                ]:
                    self.feature_specs[op] = FeatureSpec(
                        name=op,
                        feature_type=FeatureType.CROSS_FEATURE,
                        source_columns=[str(col1), str(col2)],
                        transformation=TransformationType.IDENTITY,
                        parameters={"operation": op_name},
                    )

                generated_count += 4

            # Three-way interactions (limited)
            if max_depth >= 3 and generated_count < max_features:
                top_cols = numeric_cols[:10]  # Limit to top 10 features

                for col1, col2, col3 in itertools.combinations(top_cols, 3):
                    if generated_count >= max_features:
                        break

                    series1 = data[col1].fillna(0)
                    series2 = data[col2].fillna(0)
                    series3 = data[col3].fillna(0)

                    # Three-way multiplication
                    triple_mult = f"{col1}_x_{col2}_x_{col3}"
                    features_df[triple_mult] = series1 * series2 * series3

                    self.feature_specs[triple_mult] = FeatureSpec(
                        name=triple_mult,
                        feature_type=FeatureType.CROSS_FEATURE,
                        source_columns=[str(col1), str(col2), str(col3)],
                        transformation=TransformationType.IDENTITY,
                        parameters={"operation": "triple_multiply"},
                    )

                    generated_count += 1

            return features_df.fillna(0)

        except Exception as e:
            self.logger.error(f"Cross feature generation failed: {e}")
            return data

    def generate_polynomial_features(
        self, data: pd.DataFrame, target_col: str, degree: int = 2
    ) -> pd.DataFrame:
        """Generate polynomial features"""
        try:
            features_df = data.copy()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            # Limit to most important features
            if len(numeric_cols) > 15:
                correlations = (
                    data[numeric_cols].corrwith(data[target_col]).abs().sort_values(ascending=False)
                )
                numeric_cols = correlations.head(15).index.tolist()

            for col in numeric_cols:
                series = data[col].fillna(0)

                # Polynomial degrees
                for deg in range(2, degree + 1):
                    poly_col = f"{col}_poly_{deg}"
                    features_df[poly_col] = series**deg

                    self.feature_specs[poly_col] = FeatureSpec(
                        name=poly_col,
                        feature_type=FeatureType.SYNTHETIC,
                        source_columns=[col],
                        transformation=TransformationType.SQUARE
                        if deg == 2
                        else TransformationType.IDENTITY,
                        parameters={"degree": deg},
                    )

                # Non-linear transformations
                transformations = [
                    (TransformationType.LOG, np.log1p(np.abs(series))),
                    (TransformationType.SQRT, np.sqrt(np.abs(series))),
                    (TransformationType.RECIPROCAL, 1 / (series + 1e-8)),
                ]

                for transform_type, transformed_data in transformations:
                    transform_col = f"{col}_{transform_type.value}"
                    if hasattr(transformed_data, "fillna"):
                        features_df[transform_col] = transformed_data.fillna(0)
                    else:
                        features_df[transform_col] = pd.Series(
                            transformed_data, index=features_df.index
                        ).fillna(0)

                    self.feature_specs[transform_col] = FeatureSpec(
                        name=transform_col,
                        feature_type=FeatureType.SYNTHETIC,
                        source_columns=[col],
                        transformation=transform_type,
                    )

            return features_df.fillna(0)

        except Exception as e:
            self.logger.error(f"Polynomial feature generation failed: {e}")
            return data

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception:
            return pd.Series(50, index=series.index)

    def _calculate_bollinger_bands(
        self, series: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band.fillna(series), lower_band.fillna(series)
        except Exception:
            return pd.Series(series), pd.Series(series)


class SHAPFeatureAnalyzer:
    """SHAP-based feature importance analyzer with regime-specific analysis"""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SHAPFeatureAnalyzer")

        if not HAS_SHAP:
            self.logger.warning("SHAP not available - using fallback methods")

        # SHAP explainers
        self.explainers: Dict[str, Any] = {}
        self.shap_values: Dict[str, np.ndarray] = {}

        # Regime-specific analysis
        self.regime_importance: Dict[str, Dict[str, float]] = {}
        self.global_importance: Dict[str, float] = {}

    def analyze_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model: Optional[Any] = None,
        regime_column: Optional[str] = None,
    ) -> List[FeatureImportanceResult]:
        """Comprehensive feature importance analysis using multiple methods"""
        try:
            results = []

            if not HAS_SHAP or not HAS_SKLEARN:
                self.logger.warning(
                    "Required libraries not available for feature importance analysis"
                )
                return results

            # Prepare data
            X = features.fillna(0).select_dtypes(include=[np.number])
            y = target.fillna(0)

            if len(X) == 0 or len(y) == 0:
                return results

            # Align indices
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if len(X) < 10:  # Minimum samples required
                return results

            # Train model if not provided
            if model is None:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

            # Method 1: SHAP analysis
            if HAS_SHAP:
                shap_results = self._analyze_with_shap(X, y, model)
                results.extend(shap_results)

            # Method 2: Tree-based importance
            tree_results = self._analyze_tree_importance(X, y, model)
            results.extend(tree_results)

            # Method 3: Mutual information
            mutual_info_results = self._analyze_mutual_information(X, y)
            results.extend(mutual_info_results)

            # Method 4: Permutation importance
            perm_results = self._analyze_permutation_importance(X, y, model)
            results.extend(perm_results)

            # Regime-specific analysis
            if regime_column and regime_column in features.columns:
                regime_series = features[regime_column]
                if isinstance(regime_series, pd.Series):
                    regime_results = self._analyze_regime_specific_importance(
                        X, y, model, regime_series
                    )
                    results.extend(regime_results)

            # Update global importance scores
            self._update_global_importance(results)

            return sorted(results, key=lambda x: x.importance_score, reverse=True)

        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return []

    def _analyze_with_shap(
        self, X: pd.DataFrame, y: pd.Series, model
    ) -> List[FeatureImportanceResult]:
        """SHAP-based feature importance"""
        try:
            results = []

            # Create SHAP explainer
            if HAS_SHAP and shap is not None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                self.logger.warning("SHAP not available, using fallback")
                return []

            # Calculate feature importance
            importance_scores = np.abs(shap_values).mean(axis=0)

            for i, feature in enumerate(X.columns):
                result = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=importance_scores[i],
                    method=FeatureImportanceMethod.SHAP,
                    confidence=0.9,  # SHAP typically has high confidence
                )
                results.append(result)

            # Store for later use
            self.shap_values["global"] = shap_values
            self.explainers["global"] = explainer

            return results

        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")
            return []

    def _analyze_tree_importance(
        self, X: pd.DataFrame, y: pd.Series, model
    ) -> List[FeatureImportanceResult]:
        """Tree-based feature importance"""
        try:
            results = []

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                for i, feature in enumerate(X.columns):
                    result = FeatureImportanceResult(
                        feature_name=feature,
                        importance_score=importances[i],
                        method=FeatureImportanceMethod.TREE_IMPORTANCE,
                        confidence=0.7,
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Tree importance analysis failed: {e}")
            return []

    def _analyze_mutual_information(
        self, X: pd.DataFrame, y: pd.Series
    ) -> List[FeatureImportanceResult]:
        """Mutual information feature importance"""
        try:
            results = []

            # Calculate mutual information
            from sklearn.feature_selection import mutual_info_regression

            mi_scores = mutual_info_regression(X, y, random_state=42)

            for i, feature in enumerate(X.columns):
                result = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=mi_scores[i],
                    method=FeatureImportanceMethod.MUTUAL_INFO,
                    confidence=0.6,
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Mutual information analysis failed: {e}")
            return []

    def _analyze_permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, model
    ) -> List[FeatureImportanceResult]:
        """Permutation-based feature importance"""
        try:
            from sklearn.inspection import permutation_importance

            results = []

            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)

            for i, feature in enumerate(X.columns):
                mean_importance = (
                    perm_importance.importances_mean[i]
                    if hasattr(perm_importance, "importances_mean")
                    else 0.0
                )
                std_importance = (
                    perm_importance.importances_std[i]
                    if hasattr(perm_importance, "importances_std")
                    else 0.0
                )

                result = FeatureImportanceResult(
                    feature_name=feature,
                    importance_score=mean_importance,
                    method=FeatureImportanceMethod.PERMUTATION,
                    confidence=1.0 - std_importance / (mean_importance + 1e-8),
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Permutation importance analysis failed: {e}")
            return []

    def _analyze_regime_specific_importance(
        self, X: pd.DataFrame, y: pd.Series, model, regime_series: pd.Series
    ) -> List[FeatureImportanceResult]:
        """Analyze feature importance for specific market regimes"""
        try:
            results = []

            # Get unique regimes
            regimes = regime_series.unique()

            for regime in regimes:
                regime_mask = regime_series == regime
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]

                if len(X_regime) < self.config.min_regime_samples:
                    continue

                # Train regime-specific model
                regime_model = RandomForestRegressor(n_estimators=50, random_state=42)
                regime_model.fit(X_regime, y_regime)

                # Calculate regime-specific importance
                if hasattr(regime_model, "feature_importances_"):
                    importances = regime_model.feature_importances_

                    for i, feature in enumerate(X.columns):
                        result = FeatureImportanceResult(
                            feature_name=f"{feature}_regime_{regime}",
                            importance_score=importances[i],
                            method=FeatureImportanceMethod.TREE_IMPORTANCE,
                            confidence=0.7,
                            regime_specific={str(regime): importances[i]},
                        )
                        results.append(result)

                # Store regime-specific importance
                if str(regime) not in self.regime_importance:
                    self.regime_importance[str(regime)] = {}

                for i, feature in enumerate(X.columns):
                    self.regime_importance[str(regime)][feature] = importances[i]

            return results

        except Exception as e:
            self.logger.error(f"Regime-specific importance analysis failed: {e}")
            return []

    def _update_global_importance(self, results: List[FeatureImportanceResult]):
        """Update global feature importance scores"""
        try:
            # Aggregate importance scores across methods
            feature_scores = {}

            for result in results:
                base_feature = result.feature_name.split("_regime_")[0]  # Remove regime suffix

                if base_feature not in feature_scores:
                    feature_scores[base_feature] = []

                feature_scores[base_feature].append(result.importance_score)

            # Calculate weighted average
            for feature, scores in feature_scores.items():
                if scores:
                    self.global_importance[feature] = np.mean(scores)

        except Exception as e:
            self.logger.error(f"Global importance update failed: {e}")

    def get_top_features(self, n_features: int = 50, regime: Optional[str] = None) -> List[str]:
        """Get top N features based on importance scores"""
        try:
            if regime and regime in self.regime_importance:
                importance_dict = self.regime_importance[regime]
            else:
                importance_dict = self.global_importance

            if not importance_dict:
                return []

            # Sort by importance score
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            return [feature for feature, _ in sorted_features[:n_features]]

        except Exception as e:
            self.logger.error(f"Top features retrieval failed: {e}")
            return []


class AutomatedFeatureEngineer:
    """Main automated feature engineering system"""

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.logger = logging.getLogger(f"{__name__}.AutomatedFeatureEngineer")

        # Core components
        self.synthesizer = DeepFeatureSynthesizer(self.config)
        self.shap_analyzer = SHAPFeatureAnalyzer(self.config)

        # Attention-based pruner
        self.attention_pruner: Optional[AttentionBasedFeaturePruner] = None

        # Feature management
        self.feature_pipeline: List[Callable] = []
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.feature_performance: Dict[str, float] = {}

        # Regime-specific feature sets
        self.regime_feature_sets: Dict[str, List[str]] = {}
        self.current_regime: Optional[str] = None

        if HAS_TORCH:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.gpu_acceleration else "cpu"
            )
        else:
            self.device = "cpu"
        self._lock = threading.RLock()

        self.logger.info(f"Automated Feature Engineer initialized on {self.device}")

    def fit(
        self, data: pd.DataFrame, target_column: str, regime_column: Optional[str] = None
    ) -> "AutomatedFeatureEngineer":
        """Fit the feature engineering pipeline"""
        with self._lock:
            try:
                self.logger.info(f"Fitting feature engineering pipeline on {len(data)} samples")

                # Generate comprehensive feature set
                engineered_features = self._engineer_features(data, target_column)

                # Analyze feature importance
                if target_column in engineered_features.columns:
                    target = engineered_features[target_column]
                    features = engineered_features.drop(columns=[target_column])
                else:
                    self.logger.error(f"Target column {target_column} not found")
                    return self

                importance_results = self.shap_analyzer.analyze_feature_importance(
                    features, target, regime_column=regime_column
                )

                # Update feature specs with importance scores
                for result in importance_results:
                    if result.feature_name in self.synthesizer.feature_specs:
                        self.synthesizer.feature_specs[
                            result.feature_name
                        ].importance_score = result.importance_score

                # Initialize attention-based pruner
                if HAS_TORCH and len(features.columns) > 0:
                    try:
                        self.attention_pruner = AttentionBasedFeaturePruner(
                            input_dim=len(features.columns)
                        )

                        if hasattr(self.attention_pruner, "to"):
                            self.attention_pruner = self.attention_pruner.to(self.device)

                        self._train_attention_pruner(features, target)
                    except Exception as e:
                        self.logger.warning(f"Attention pruner initialization failed: {e}")
                        self.attention_pruner = None

                # Create regime-specific feature sets
                if regime_column:
                    self._create_regime_feature_sets(
                        engineered_features, target_column, regime_column
                    )

                self.logger.info(
                    f"Feature engineering pipeline fitted with {len(features.columns)} features"
                )
                return self

            except Exception as e:
                self.logger.error(f"Feature engineering fit failed: {e}")
                return self

    def transform(
        self, data: pd.DataFrame, target_column: str, regime: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform data using fitted feature engineering pipeline"""
        with self._lock:
            try:
                # Generate features
                engineered_features = self._engineer_features(data, target_column)

                # Select features based on regime
                if regime and regime in self.regime_feature_sets:
                    selected_features = self.regime_feature_sets[regime]
                    # Include target column
                    selected_features = [
                        f for f in selected_features if f in engineered_features.columns
                    ]
                    if target_column not in selected_features:
                        selected_features.append(target_column)

                    engineered_features = engineered_features[selected_features]
                else:
                    # Use top features from global importance
                    top_features = self.shap_analyzer.get_top_features(
                        n_features=min(
                            self.config.max_total_features, len(engineered_features.columns) - 1
                        )
                    )

                    # Include available top features plus target
                    available_features = [
                        f for f in top_features if f in engineered_features.columns
                    ]
                    if target_column not in available_features:
                        available_features.append(target_column)

                    if available_features:
                        engineered_features = engineered_features[available_features]

                return engineered_features

            except Exception as e:
                self.logger.error(f"Feature engineering transform failed: {e}")
                return data

    def _engineer_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        try:
            # Start with original data
            features = data.copy()

            # Generate temporal features
            features = self.synthesizer.generate_temporal_features(features, target_column)

            # Generate cross features
            features = self.synthesizer.generate_cross_features(
                features, target_column, self.config.cross_feature_max_depth
            )

            # Generate polynomial features
            features = self.synthesizer.generate_polynomial_features(features, target_column)

            # Remove highly correlated features
            features = self._remove_correlated_features(features, target_column)

            # Remove low variance features
            features = self._remove_low_variance_features(features, target_column)

            return features

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return data

    def _remove_correlated_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)

            if len(numeric_cols) <= 1:
                return data

            # Calculate correlation matrix
            corr_matrix = data[numeric_cols].corr().abs()

            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to remove
            to_remove = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]

            # Keep features that are important
            important_features = self.shap_analyzer.get_top_features(n_features=50)
            to_remove = [f for f in to_remove if f not in important_features[:20]]  # Keep top 20

            if to_remove:
                self.logger.info(f"Removing {len(to_remove)} highly correlated features")
                data = data.drop(columns=to_remove)

            return data

        except Exception as e:
            self.logger.error(f"Correlation removal failed: {e}")
            return data

    def _remove_low_variance_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Remove low variance features"""
        try:
            from sklearn.feature_selection import VarianceThreshold

            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)

            if len(numeric_cols) == 0:
                return data

            # Apply variance threshold
            selector = VarianceThreshold(threshold=self.config.variance_threshold)

            # Fit and transform
            selected_data = selector.fit_transform(data[numeric_cols])
            selected_features = [
                numeric_cols[i]
                for i in range(len(numeric_cols))
                if selector.variances_[i] > self.config.variance_threshold
            ]

            # Reconstruct dataframe
            result_data = data[[target_column]].copy()

            for i, feature in enumerate(selected_features):
                result_data[feature] = selected_data[:, i]

            # Add back non-numeric columns
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
            if target_column in non_numeric_cols:
                non_numeric_cols.remove(target_column)

            for col in non_numeric_cols:
                result_data[col] = data[col]

            removed_count = len(numeric_cols) - len(selected_features)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} low variance features")

            return result_data

        except Exception as e:
            self.logger.error(f"Low variance removal failed: {e}")
            return data

    def _train_attention_pruner(self, features: pd.DataFrame, target: pd.Series):
        """Train attention-based feature pruner"""
        try:
            if not HAS_TORCH or self.attention_pruner is None:
                return

            # Prepare data
            X = features.fillna(0).values.astype(np.float32)
            y = target.fillna(0).values.astype(np.float32)

            # Create tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Create dataset and dataloader
            from torch.utils.data import TensorDataset, DataLoader

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Training setup
            self.attention_pruner.train()
            optimizer = torch.optim.Adam(self.attention_pruner.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Training loop
            for epoch in range(50):  # Quick training
                total_loss = 0.0

                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()

                    predictions, attention_weights = self.attention_pruner(batch_x)
                    loss = criterion(predictions.squeeze(), batch_y)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    self.logger.debug(
                        f"Attention pruner training epoch {epoch}, loss: {avg_loss:.4f}"
                    )

            self.logger.info("Attention-based feature pruner trained successfully")

        except Exception as e:
            self.logger.error(f"Attention pruner training failed: {e}")

    def _create_regime_feature_sets(
        self, data: pd.DataFrame, target_column: str, regime_column: str
    ):
        """Create regime-specific feature sets"""
        try:
            regimes = data[regime_column].unique()

            for regime in regimes:
                regime_mask = data[regime_column] == regime
                regime_data = data[regime_mask]

                if len(regime_data) < self.config.min_regime_samples:
                    continue

                # Get regime-specific top features
                regime_features = self.shap_analyzer.get_top_features(
                    n_features=int(
                        self.config.max_total_features * self.config.regime_feature_ratio
                    ),
                    regime=str(regime),
                )

                # Ensure we have some features
                if not regime_features:
                    # Fallback to global top features
                    regime_features = self.shap_analyzer.get_top_features(
                        n_features=int(
                            self.config.max_total_features * self.config.regime_feature_ratio
                        )
                    )

                self.regime_feature_sets[str(regime)] = regime_features
                self.logger.info(
                    f"Created feature set for regime {regime}: {len(regime_features)} features"
                )

        except Exception as e:
            self.logger.error(f"Regime feature set creation failed: {e}")

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature importance summary"""
        with self._lock:
            return {
                "total_features_generated": len(self.synthesizer.feature_specs),
                "global_importance_scores": dict(
                    list(self.shap_analyzer.global_importance.items())[:20]
                ),
                "regime_specific_sets": {
                    regime: len(features) for regime, features in self.regime_feature_sets.items()
                },
                "feature_types_distribution": self._get_feature_type_distribution(),
                "attention_pruner_available": self.attention_pruner is not None,
                "top_features_by_type": self._get_top_features_by_type(),
            }

    def _get_feature_type_distribution(self) -> Dict[str, int]:
        """Get distribution of feature types"""
        type_counts = {}
        for spec in self.synthesizer.feature_specs.values():
            feature_type = spec.feature_type.value
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        return type_counts

    def _get_top_features_by_type(self) -> Dict[str, List[str]]:
        """Get top features grouped by type"""
        features_by_type = {}

        for feature_name, importance in self.shap_analyzer.global_importance.items():
            if feature_name in self.synthesizer.feature_specs:
                feature_type = self.synthesizer.feature_specs[feature_name].feature_type.value

                if feature_type not in features_by_type:
                    features_by_type[feature_type] = []

                features_by_type[feature_type].append((feature_name, importance))

        # Sort and limit each type
        for feature_type in features_by_type:
            features_by_type[feature_type].sort(key=lambda x: x[1], reverse=True)
            features_by_type[feature_type] = [
                name for name, _ in features_by_type[feature_type][:5]
            ]

        return features_by_type


# Singleton automated feature engineer
_automated_feature_engineer = None
_afe_lock = threading.Lock()


def get_automated_feature_engineer(
    config: Optional[FeatureEngineeringConfig] = None,
) -> AutomatedFeatureEngineer:
    """Get the singleton automated feature engineer"""
    global _automated_feature_engineer

    with _afe_lock:
        if _automated_feature_engineer is None:
            _automated_feature_engineer = AutomatedFeatureEngineer(config)
        return _automated_feature_engineer


def engineer_features_for_coin(
    data: pd.DataFrame, target_column: str, regime_column: Optional[str] = None
) -> pd.DataFrame:
    """Convenient function to engineer features for cryptocurrency data"""
    engineer = get_automated_feature_engineer()
    return engineer.fit(data, target_column, regime_column).transform(data, target_column)
