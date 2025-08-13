#!/usr/bin/env python3
"""
Regime Features and Classification
Add regime features and measure MAE improvement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


class MarketRegimeClassifier:
    """
    Classify market regimes and create regime features
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_names = ["Bull_Trend", "Bear_Trend", "Sideways", "High_Volatility"]
        self.is_fitted = False

    def prepare_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime classification"""

        regime_df = df.copy()

        # Price momentum features
        regime_df["price_momentum_1h"] = df["close"].pct_change(1)
        regime_df["price_momentum_24h"] = df["close"].pct_change(24)
        regime_df["price_momentum_7d"] = df["close"].pct_change(168)

        # Volatility features
        regime_df["volatility_1h"] = df["close"].rolling(24).std() / df["close"].rolling(24).mean()
        regime_df["volatility_24h"] = (
            df["close"].rolling(168).std() / df["close"].rolling(168).mean()

        # Volume features
        regime_df["volume_momentum"] = df["volume_24h"].pct_change(1)
        regime_df["volume_ma_ratio"] = df["volume_24h"] / df["volume_24h"].rolling(24).mean()

        # Trend strength
        regime_df["trend_strength"] = abs(regime_df["price_momentum_24h"])

        # Market cap momentum (if available)
        if "market_cap" in df.columns:
            regime_df["market_cap_momentum"] = df["market_cap"].pct_change(24)
        else:
            regime_df["market_cap_momentum"] = 0

        # RSI regime indicator
        if "technical_rsi" in df.columns:
            regime_df["rsi_regime"] = pd.cut(
                df["technical_rsi"],
                bins=[0, 30, 70, 100],
                labels=["oversold", "neutral", "overbought"],
            )
            regime_df["rsi_extreme"] = (
                (df["technical_rsi"] < 20) | (df["technical_rsi"] > 80).astype(int)
        else:
            regime_df["rsi_regime"] = "neutral"
            regime_df["rsi_extreme"] = 0

        return regime_df

    def fit_regime_classifier(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit regime classification model"""

        # Prepare features
        regime_df = self.prepare_regime_features(df)

        # Select numerical features for clustering
        feature_cols = [
            "price_momentum_1h",
            "price_momentum_24h",
            "price_momentum_7d",
            "volatility_1h",
            "volatility_24h",
            "volume_momentum",
            "volume_ma_ratio",
            "trend_strength",
            "market_cap_momentum",
            "rsi_extreme",
        ]

        # Filter and clean features
        features_df = regime_df[feature_cols].fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)

        # Fit Gaussian Mixture Model for regime classification
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full", random_state=42
        )

        regime_labels = self.regime_model.fit_predict(scaled_features)

        self.is_fitted = True

        # Analyze regimes
        regime_analysis = self._analyze_regimes(regime_df, regime_labels, feature_cols)

        return regime_analysis

    def predict_regime(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime for new data"""

        if not self.is_fitted:
            raise ValueError("Regime classifier must be fitted first")

        # Prepare features
        regime_df = self.prepare_regime_features(df)

        feature_cols = [
            "price_momentum_1h",
            "price_momentum_24h",
            "price_momentum_7d",
            "volatility_1h",
            "volatility_24h",
            "volume_momentum",
            "volume_ma_ratio",
            "trend_strength",
            "market_cap_momentum",
            "rsi_extreme",
        ]

        features_df = regime_df[feature_cols].fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Scale and predict
        scaled_features = self.scaler.transform(features_df)
        regime_labels = self.regime_model.predict(scaled_features)

        return regime_labels

    def _analyze_regimes(
        self, df: pd.DataFrame, regime_labels: np.ndarray, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Analyze characteristics of each regime"""

        analysis = {"regime_counts": {}, "regime_characteristics": {}, "regime_performance": {}}

        # Add regime labels to dataframe
        df_with_regimes = df.copy()
        df_with_regimes["regime"] = regime_labels

        for regime_id in range(self.n_regimes):
            regime_data = df_with_regimes[df_with_regimes["regime"] == regime_id]

            if len(regime_data) == 0:
                continue

            regime_name = (
                self.regime_names[regime_id]
                if regime_id < len(self.regime_names)
                else f"Regime_{regime_id}"
            )

            # Count and percentage
            analysis["regime_counts"][regime_name] = {
                "count": len(regime_data),
                "percentage": len(regime_data) / len(df_with_regimes) * 100,
            }

            # Characteristics
            characteristics = {}
            for col in feature_cols:
                if col in regime_data.columns:
                    characteristics[col] = {
                        "mean": regime_data[col].mean(),
                        "std": regime_data[col].std(),
                        "median": regime_data[col].median(),
                    }

            analysis["regime_characteristics"][regime_name] = characteristics

            # Performance (if target available)
            if "target_24h" in regime_data.columns:
                analysis["regime_performance"][regime_name] = {
                    "avg_return": regime_data["target_24h"].mean(),
                    "volatility": regime_data["target_24h"].std(),
                    "sharpe": regime_data["target_24h"].mean()
                    / max(regime_data["target_24h"].std(), 0.001),
                }

        return analysis


class RegimeAwarePredictor:
    """
    Regime-aware prediction system to measure MAE improvement
    """

    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_models = {}
        self.baseline_model = None
        self.is_trained = False

    def train_baseline_model(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Train baseline model without regime awareness"""

        from sklearn.ensemble import RandomForestRegressor

        # Simple baseline model
        self.baseline_model = RandomForestRegressor(n_estimators=50, random_state=42)

        # Prepare features (exclude regime columns)
        feature_cols = [col for col in X.columns if not col.startswith("regime")]
        X_baseline = X[feature_cols].fillna(0)

        self.baseline_model.fit(X_baseline, y)

        # Calculate baseline MAE
        baseline_pred = self.baseline_model.predict(X_baseline)
        baseline_mae = mean_absolute_error(y, baseline_pred)

        return baseline_mae

    def train_regime_aware_models(
        self, X: pd.DataFrame, y: pd.Series, regime_labels: np.ndarray
    ) -> Dict[str, float]:
        """Train separate models for each regime"""

        regime_maes = {}

        # Prepare base features
        feature_cols = [col for col in X.columns if not col.startswith("regime")]
        X_features = X[feature_cols].fillna(0)

        # Train model for each regime
        for regime_id in np.unique(regime_labels):
            regime_mask = regime_labels == regime_id

            if regime_mask.sum() < 10:  # Skip regimes with too few samples
                continue

            regime_name = (
                self.regime_classifier.regime_names[regime_id]
                if regime_id < len(self.regime_classifier.regime_names)
                else f"Regime_{regime_id}"
            )

            # Train regime-specific model
            regime_model = RandomForestRegressor(n_estimators=50, random_state=42)
            regime_model.fit(X_features[regime_mask], y[regime_mask])

            self.regime_models[regime_id] = regime_model

            # Calculate regime MAE
            regime_pred = regime_model.predict(X_features[regime_mask])
            regime_mae = mean_absolute_error(y[regime_mask], regime_pred)
            regime_maes[regime_name] = regime_mae

        self.is_trained = True
        return regime_maes

    def predict_regime_aware(self, X: pd.DataFrame, regime_labels: np.ndarray) -> np.ndarray:
        """Make regime-aware predictions"""

        if not self.is_trained:
            raise ValueError("Regime-aware models must be trained first")

        feature_cols = [col for col in X.columns if not col.startswith("regime")]
        X_features = X[feature_cols].fillna(0)

        predictions = np.zeros(len(X))

        for regime_id, model in self.regime_models.items():
            regime_mask = regime_labels == regime_id

            if regime_mask.sum() > 0:
                regime_pred = model.predict(X_features[regime_mask])
                predictions[regime_mask] = regime_pred

        return predictions

    def evaluate_regime_improvement(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate improvement from regime awareness"""

        # Fit regime classifier
        regime_analysis = self.regime_classifier.fit_regime_classifier(X)
        regime_labels = self.regime_classifier.predict_regime(X)

        # Train baseline model
        baseline_mae = self.train_baseline_model(X, y)

        # Train regime-aware models
        regime_maes = self.train_regime_aware_models(X, y, regime_labels)

        # Make regime-aware predictions
        regime_predictions = self.predict_regime_aware(X, regime_labels)
        regime_aware_mae = mean_absolute_error(y, regime_predictions)

        # Calculate improvement
        mae_improvement = baseline_mae - regime_aware_mae
        improvement_percentage = (mae_improvement / baseline_mae) * 100

        evaluation_results = {
            "baseline_mae": baseline_mae,
            "regime_aware_mae": regime_aware_mae,
            "mae_improvement": mae_improvement,
            "improvement_percentage": improvement_percentage,
            "regime_specific_maes": regime_maes,
            "regime_analysis": regime_analysis,
            "regime_distribution": pd.Series(regime_labels).value_counts().to_dict(),
        }

        return evaluation_results


def create_regime_feature_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create complete regime feature pipeline"""

    # Initialize regime classifier
    regime_classifier = MarketRegimeClassifier()

    # Fit regime classifier
    regime_analysis = regime_classifier.fit_regime_classifier(df)

    # Add regime features to dataframe
    regime_labels = regime_classifier.predict_regime(df)
    df_with_regimes = df.copy()
    df_with_regimes["regime"] = regime_labels
    df_with_regimes["regime_name"] = [
        regime_classifier.regime_names[label]
        if label < len(regime_classifier.regime_names)
        else f"Regime_{label}"
        for label in regime_labels
    ]

    # Add regime-specific features
    for regime_id in range(regime_classifier.n_regimes):
        regime_name = (
            regime_classifier.regime_names[regime_id]
            if regime_id < len(regime_classifier.regime_names)
            else f"Regime_{regime_id}"
        )
        df_with_regimes[f"is_{regime_name.lower()}"] = (regime_labels == regime_id).astype(int)

    return df_with_regimes, regime_analysis


if __name__ == "__main__":
    print("🌊 TESTING REGIME FEATURES AND CLASSIFICATION")
    print("=" * 50)

    # Create sample market data
    np.random.seed(42)
    n_samples = 1000

    # Generate different market regimes
    regime_periods = [250, 250, 250, 250]  # 4 regimes
    sample_data = []

    for i, period_length in enumerate(regime_periods):
        if i == 0:  # Bull trend
            prices = np.cumsum(np.random.normal(0, 1)) + 100
            volumes = np.random.normal(0, 1)
        elif i == 1:  # Bear trend
            prices = np.cumsum(np.random.normal(0, 1)) + 100
            volumes = np.random.normal(0, 1)
        elif i == 2:  # Sideways
            prices = np.cumsum(np.random.normal(0, 1)) + 100
            volumes = np.random.normal(0, 1)
        else:  # High volatility
            prices = np.cumsum(np.random.normal(0, 1)) + 100
            volumes = np.random.normal(0, 1)

        regime_data = pd.DataFrame(
            {
                "close": prices,
                "volume_24h": volumes,
                "market_cap": prices * 1e6,
                "technical_rsi": np.random.normal(0, 1),
            }
        )

        sample_data.append(regime_data)

    # Combine all regimes
    market_data = pd.concat(sample_data, ignore_index=True)

    # Add synthetic target
    market_data["target_24h"] = market_data["close"].pct_change(1).shift(-1)

    print(f"Generated {len(market_data)} samples across 4 market regimes")

    # Test regime classification
    regime_df, regime_analysis = create_regime_feature_pipeline(market_data)

    print(f"\nRegime Analysis:")
    for regime_name, counts in regime_analysis["regime_counts"].items():
        print(f"   {regime_name}: {counts['count']} samples ({counts['percentage']:.1f}%)")

    # Test regime-aware prediction
    if "target_24h" in market_data.columns:
        predictor = RegimeAwarePredictor()

        # Prepare features (simple example)
        feature_cols = ["close", "volume_24h", "market_cap", "technical_rsi"]
        X = market_data[feature_cols].dropna()
        y = market_data["target_24h"].dropna()

        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

        evaluation = predictor.evaluate_regime_improvement(X, y)

        print(f"\nMAE Improvement Analysis:")
        print(f"   Baseline MAE: {evaluation['baseline_mae']:.6f}")
        print(f"   Regime-aware MAE: {evaluation['regime_aware_mae']:.6f}")
        print(f"   Improvement: {evaluation['improvement_percentage']:.2f}%")

    print("✅ Regime features and classification testing completed")
