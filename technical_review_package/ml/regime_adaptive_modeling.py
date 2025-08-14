#!/usr/bin/env python3
"""
Market Regime Detection & Adaptive Modeling
Prevents regime-blind predictions
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class MarketRegimeDetector:
    """Detect market regimes for adaptive modeling"""

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_names = {
            0: "Low_Volatility_Bull",
            1: "High_Volatility_Bull",
            2: "Low_Volatility_Bear",
            3: "High_Volatility_Bear",
        }

    def fit(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit regime detector on market data"""

        # Create regime features
        features = self._create_regime_features(market_data)

        if features.empty:
            return {"success": False, "error": "No valid features for regime detection"}

        # Fit scaler and GMM
        features_scaled = self.scaler.fit_transform(features)
        self.gmm.fit(features_scaled)
        self.is_fitted = True

        # Predict regimes
        regimes = self.gmm.predict(features_scaled)

        # Analyze regime characteristics
        regime_analysis = self._analyze_regimes(market_data, regimes)

        return {
            "success": True,
            "regimes_detected": len(np.unique(regimes)),
            "regime_distribution": {
                self.regime_names.get(i, f"Regime_{i}"): (regimes == i).sum()
                for i in range(self.n_regimes)
            },
            "regime_analysis": regime_analysis,
        }

    def predict_regime(self, market_data: pd.DataFrame) -> np.ndarray:
        """Predict regime for new market data"""

        if not self.is_fitted:
            raise ValueError("Regime detector must be fitted first")

        features = self._create_regime_features(market_data)
        features_scaled = self.scaler.transform(features)

        return self.gmm.predict(features_scaled)

    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection"""

        features_list = []

        # Price momentum features
        if "price" in data.columns:
            data["price_return_1d"] = data["price"].pct_change()
            data["price_return_7d"] = data["price"].pct_change(7)
            features_list.extend(["price_return_1d", "price_return_7d"])

        # Volatility features
        if "price" in data.columns:
            data["volatility_10d"] = data["price"].pct_change().rolling(10).std()
            data["volatility_30d"] = data["price"].pct_change().rolling(30).std()
            features_list.extend(["volatility_10d", "volatility_30d"])

        # Volume features
        if "volume_24h" in data.columns:
            data["volume_ratio"] = data["volume_24h"] / data["volume_24h"].rolling(30).mean()
            features_list.append("volume_ratio")

        # Market stress indicators
        if "change_24h" in data.columns:
            data["market_stress"] = data["change_24h"].rolling(7).std()
            features_list.append("market_stress")

        # Select valid features
        valid_features = [f for f in features_list if f in data.columns]

        if not valid_features:
            return pd.DataFrame()

        # Return clean features
        features_df = data[valid_features].dropna()
        return features_df

    def _analyze_regimes(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of detected regimes"""

        analysis = {}

        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id
            regime_data = data[regime_mask]

            if len(regime_data) == 0:
                continue

            regime_stats = {}

            # Price statistics
            if "price" in regime_data.columns:
                returns = regime_data["price"].pct_change().dropna()
                regime_stats.update(
                    {
                        "avg_return": returns.mean(),
                        "volatility": returns.std(),
                        "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0,
                    }
                )

            # Volume statistics
            if "volume_24h" in regime_data.columns:
                regime_stats["avg_volume"] = regime_data["volume_24h"].mean()

            analysis[self.regime_names.get(regime_id, f"Regime_{regime_id}")] = regime_stats

        return analysis


class RegimeAdaptiveModel:
    """Model that adapts predictions based on market regime"""

    def __init__(self, base_models: Dict[str, Any] = None):
        self.regime_detector = MarketRegimeDetector()
        self.regime_models = base_models or {}
        self.current_regime = None
        self.performance_by_regime = {}

    def train_regime_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Train separate models for each regime"""

        # Detect regimes
        regime_fit_result = self.regime_detector.fit(features)

        if not regime_fit_result.get("success", False):
            return regime_fit_result

        regimes = self.regime_detector.predict_regime(features)

        # Train models per regime
        training_results = {}

        for regime_id in range(self.regime_detector.n_regimes):
            regime_mask = regimes == regime_id
            regime_features = features[regime_mask]
            regime_targets = targets[regime_mask]

            if len(regime_features) < 20:  # Minimum samples
                continue

            # Simple linear model for each regime (can be replaced with sophisticated models)
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            try:
                model.fit(
                    regime_features.select_dtypes(include=[np.number]),
                    regime_targets.iloc[:, 0]
                    if isinstance(regime_targets, pd.DataFrame)
                    else regime_targets,
                )

                regime_name = self.regime_detector.regime_names.get(
                    regime_id, f"Regime_{regime_id}"
                )
                self.regime_models[regime_name] = model

                training_results[regime_name] = {"samples": len(regime_features), "success": True}

            except Exception as e:
                training_results[f"Regime_{regime_id}"] = {"success": False, "error": str(e)}

        return {
            "success": True,
            "models_trained": len(self.regime_models),
            "training_results": training_results,
            "regime_fit": regime_fit_result,
        }

    def predict_adaptive(self, features: pd.DataFrame) -> np.ndarray:
        """Make regime-adaptive predictions"""

        if not self.regime_detector.is_fitted:
            raise ValueError("Regime detector not fitted")

        # Detect current regime
        regimes = self.regime_detector.predict_regime(features)
        predictions = np.zeros(len(features))

        for regime_id in range(self.regime_detector.n_regimes):
            regime_mask = regimes == regime_id

            if not regime_mask.any():
                continue

            regime_name = self.regime_detector.regime_names.get(regime_id, f"Regime_{regime_id}")

            if regime_name in self.regime_models:
                model = self.regime_models[regime_name]
                regime_features = features[regime_mask].select_dtypes(include=[np.number])

                try:
                    regime_predictions = model.predict(regime_features)
                    predictions[regime_mask] = regime_predictions
                except Exception:
                    # Fallback to zero predictions
                    predictions[regime_mask] = 0

        return predictions
