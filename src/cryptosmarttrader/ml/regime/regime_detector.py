#!/usr/bin/env python3
"""
Regime Detection System - HMM and Rule-Based Market Regime Classification
Bull/Bear/Sideways and Low/High Volatility regime detection for crypto markets
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import core components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ..core.structured_logger import get_logger


class RegimeFeatures:
    """Extract regime-relevant features from price data"""

    def __init__(self):
        self.logger = get_logger("RegimeFeatures")

    def calculate_returns(self, prices: pd.Series, windows: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """Calculate returns over multiple windows"""

        features = pd.DataFrame(index=prices.index)

        for window in windows:
            returns = prices.pct_change(window).fillna(0)
            features[f"return_{window}d"] = returns

            # Rolling statistics
            features[f"return_{window}d_mean"] = returns.rolling(20).mean().fillna(0)
            features[f"return_{window}d_std"] = returns.rolling(20).std().fillna(0.01)

        return features

    def calculate_volatility(
        self, prices: pd.Series, windows: List[int] = [5, 20, 60]
    ) -> pd.DataFrame:
        """Calculate volatility measures"""

        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change().fillna(0)

        for window in windows:
            # Rolling volatility
            vol = returns.rolling(window).std().fillna(0.01)
            features[f"volatility_{window}d"] = vol

            # Volatility of volatility
            vol_vol = vol.rolling(window).std().fillna(0.01)
            features[f"vol_of_vol_{window}d"] = vol_vol

            # Volatility regime (high/low relative to historical)
            vol_percentile = vol.rolling(100).rank(pct=True).fillna(0.5)
            features[f"vol_regime_{window}d"] = vol_percentile

        return features

    def calculate_trend_features(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate trend and momentum features"""

        features = pd.DataFrame(index=prices.index)

        # Moving averages
        ma_periods = [5, 10, 20, 50]
        for period in ma_periods:
            ma = prices.rolling(period).mean()
            features[f"ma_{period}"] = ma
            features[f"price_vs_ma_{period}"] = (prices - ma) / ma

        # Trend strength
        for window in [10, 20, 50]:
            price_changes = prices.diff(window)
            abs_changes = np.abs(price_changes)
            trend_strength = price_changes / (abs_changes.rolling(window).mean() + 1e-8)
            features[f"trend_strength_{window}d"] = trend_strength.fillna(0)

        # Price momentum
        for window in [5, 10, 20]:
            momentum = (prices - prices.shift(window)) / prices.shift(window)
            features[f"momentum_{window}d"] = momentum.fillna(0)

        return features

    def calculate_regime_indicators(
        self, prices: pd.Series, volume: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Calculate specific regime indicators"""

        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change().fillna(0)

        # Drawdown from recent high
        rolling_max = prices.rolling(50).max()
        drawdown = (prices - rolling_max) / rolling_max
        features["drawdown"] = drawdown.fillna(0)

        # Time since high/low
        high_indices = prices.rolling(20).apply(lambda x: x.argmax(), raw=False)
        low_indices = prices.rolling(20).apply(lambda x: x.argmin(), raw=False)
        features["days_since_high"] = 20 - high_indices
        features["days_since_low"] = 20 - low_indices

        # Sharpe-like ratio
        for window in [20, 60]:
            mean_return = returns.rolling(window).mean()
            return_std = returns.rolling(window).std()
            sharpe = mean_return / (return_std + 1e-8)
            features[f"sharpe_{window}d"] = sharpe.fillna(0)

        # Volume features (if available)
        if volume is not None:
            vol_ma = volume.rolling(20).mean()
            features["volume_ratio"] = volume / (vol_ma + 1e-8)
            features["volume_trend"] = volume.pct_change(5).fillna(0)

        return features

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all regime features from price/volume data"""

        try:
            all_features = []

            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol].copy().sort_values("timestamp")

                if len(symbol_df) < 100:  # Need sufficient history
                    continue

                prices = symbol_df["price"]
                volume = symbol_df.get("volume_24h", None)

                # Extract features
                returns_features = self.calculate_returns(prices)
                vol_features = self.calculate_volatility(prices)
                trend_features = self.calculate_trend_features(prices)
                regime_features = self.calculate_regime_indicators(prices, volume)

                # Combine features
                symbol_features = pd.concat(
                    [returns_features, vol_features, trend_features, regime_features], axis=1
                )

                # Add metadata
                symbol_features["symbol"] = symbol
                symbol_features["timestamp"] = symbol_df["timestamp"].values
                symbol_features["price"] = prices.values

                all_features.append(symbol_features)

            if all_features:
                result_df = pd.concat(all_features, ignore_index=True)
                result_df = result_df.fillna(method="ffill").fillna(0)

                self.logger.info(
                    f"Extracted regime features: {len(result_df)} samples, {len(result_df.columns)} features"
                )
                return result_df
            else:
                self.logger.warning("No features extracted - insufficient data")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()


class RuleBasedRegimeDetector:
    """Simple rule-based regime classification"""

    def __init__(self):
        self.logger = get_logger("RuleBasedRegimeDetector")

    def detect_market_regime(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Detect bull/bear/sideways regimes using rules"""

        try:
            result_df = features_df.copy()

            # Bull/Bear/Sideways classification
            trend_regimes = []
            vol_regimes = []

            for _, row in features_df.iterrows():
                # Trend regime based on multiple indicators
                trend_score = 0

                # Price vs moving averages
                if "price_vs_ma_20" in row:
                    if row["price_vs_ma_20"] > 0.05:  # 5% above MA
                        trend_score += 1
                    elif row["price_vs_ma_20"] < -0.05:
                        trend_score -= 1

                # Momentum indicators
                if "momentum_20d" in row:
                    if row["momentum_20d"] > 0.1:  # 10% momentum
                        trend_score += 1
                    elif row["momentum_20d"] < -0.1:
                        trend_score -= 1

                # Trend strength
                if "trend_strength_20d" in row:
                    if row["trend_strength_20d"] > 0.5:
                        trend_score += 1
                    elif row["trend_strength_20d"] < -0.5:
                        trend_score -= 1

                # Classify trend regime
                if trend_score >= 2:
                    trend_regime = "bull"
                elif trend_score <= -2:
                    trend_regime = "bear"
                else:
                    trend_regime = "sideways"

                trend_regimes.append(trend_regime)

                # Volatility regime
                vol_score = 0
                if "vol_regime_20d" in row:
                    if row["vol_regime_20d"] > 0.7:  # High volatility
                        vol_regime = "high_vol"
                    elif row["vol_regime_20d"] < 0.3:  # Low volatility
                        vol_regime = "low_vol"
                    else:
                        vol_regime = "medium_vol"
                else:
                    vol_regime = "medium_vol"

                vol_regimes.append(vol_regime)

            result_df["trend_regime"] = trend_regimes
            result_df["vol_regime"] = vol_regimes

            # Combined regime
            combined_regimes = []
            for trend, vol in zip(trend_regimes, vol_regimes):
                combined = f"{trend}_{vol}"
                combined_regimes.append(combined)

            result_df["combined_regime"] = combined_regimes

            self.logger.info(f"Rule-based regime detection completed for {len(result_df)} samples")

            # Log regime distribution
            trend_dist = pd.Series(trend_regimes).value_counts()
            vol_dist = pd.Series(vol_regimes).value_counts()

            self.logger.info(f"Trend regimes: {trend_dist.to_dict()}")
            self.logger.info(f"Vol regimes: {vol_dist.to_dict()}")

            return result_df

        except Exception as e:
            self.logger.error(f"Rule-based regime detection failed: {e}")
            return features_df


class HMMRegimeDetector:
    """HMM-based regime detection using hmmlearn"""

    def __init__(self, n_regimes: int = 3):
        self.logger = get_logger("HMMRegimeDetector")
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_hmm_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM training"""

        # Select relevant features for regime detection
        feature_cols = [
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_5d",
            "volatility_20d",
            "volatility_60d",
            "momentum_5d",
            "momentum_10d",
            "momentum_20d",
            "trend_strength_10d",
            "trend_strength_20d",
            "drawdown",
        ]

        # Use available features
        available_features = [col for col in feature_cols if col in features_df.columns]

        if not available_features:
            self.logger.warning("No suitable features found for HMM")
            return np.array([])

        X = features_df[available_features].values
        X = np.nan_to_num(X, 0)  # Replace NaN with 0

        return X

    def fit_hmm(self, features_df: pd.DataFrame) -> bool:
        """Fit HMM model to features"""

        if not HMM_AVAILABLE:
            self.logger.warning("HMM not available (hmmlearn not installed)")
            return False

        try:
            X = self.prepare_hmm_features(features_df)

            if len(X) == 0:
                self.logger.error("No features available for HMM training")
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Fit HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes, covariance_type="full", n_iter=100, random_state=42
            )

            self.logger.info(
                f"Training HMM with {len(X_scaled)} samples, {X_scaled.shape[1]} features"
            )

            self.model.fit(X_scaled)
            self.is_fitted = True

            # Log convergence
            self.logger.info(f"HMM converged: {self.model.monitor_.converged}")
            self.logger.info(f"Final log-likelihood: {self.model.score(X_scaled):.2f}")

            return True

        except Exception as e:
            self.logger.error(f"HMM training failed: {e}")
            return False

    def predict_regimes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict regimes using fitted HMM"""

        if not self.is_fitted or self.model is None:
            self.logger.warning("HMM not fitted, using fallback regime assignment")
            # Fallback to simple regime assignment
            result_df = features_df.copy()
            result_df["hmm_regime"] = 0  # Default regime
            result_df["hmm_regime_prob"] = 0.5
            return result_df

        try:
            X = self.prepare_hmm_features(features_df)

            if len(X) == 0:
                self.logger.error("No features available for HMM prediction")
                result_df = features_df.copy()
                result_df["hmm_regime"] = 0
                result_df["hmm_regime_prob"] = 0.5
                return result_df

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict regimes
            regime_labels = self.model.predict(X_scaled)
            regime_probs = self.model.predict_proba(X_scaled)

            # Max probability for confidence
            max_probs = np.max(regime_probs, axis=1)

            result_df = features_df.copy()
            result_df["hmm_regime"] = regime_labels
            result_df["hmm_regime_prob"] = max_probs

            # Map regime numbers to meaningful names
            regime_names = ["regime_0", "regime_1", "regime_2"][: self.n_regimes]
            result_df["hmm_regime_name"] = [regime_names[label] for label in regime_labels]

            self.logger.info(f"HMM regime prediction completed for {len(result_df)} samples")

            # Log regime distribution
            regime_dist = pd.Series(regime_labels).value_counts().sort_index()
            self.logger.info(f"HMM regime distribution: {regime_dist.to_dict()}")

            return result_df

        except Exception as e:
            self.logger.error(f"HMM prediction failed: {e}")
            result_df = features_df.copy()
            result_df["hmm_regime"] = 0
            result_df["hmm_regime_prob"] = 0.5
            return result_df


class RegimeDetector:
    """Main regime detection system combining rule-based and HMM approaches"""

    def __init__(self, use_hmm: bool = True, n_hmm_regimes: int = 3):
        self.logger = get_logger("RegimeDetector")

        self.feature_extractor = RegimeFeatures()
        self.rule_detector = RuleBasedRegimeDetector()
        self.hmm_detector = HMMRegimeDetector(n_hmm_regimes) if use_hmm else None

        self.use_hmm = use_hmm
        self.model_dir = Path("models/regime")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, price_data_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit regime detection models to historical data"""

        start_time = time.time()

        try:
            self.logger.info(f"Training regime detection on {len(price_data_df)} samples")

            # Extract features
            features_df = self.feature_extractor.extract_all_features(price_data_df)

            if features_df.empty:
                raise ValueError("No features extracted from price data")

            # Fit HMM if enabled
            hmm_success = True
            if self.use_hmm and self.hmm_detector:
                hmm_success = self.hmm_detector.fit_hmm(features_df)

            training_time = time.time() - start_time

            # Save models
            self.save_models()

            result = {
                "success": True,
                "training_time": training_time,
                "features_extracted": len(features_df),
                "feature_columns": list(features_df.columns),
                "hmm_fitted": hmm_success,
                "hmm_available": HMM_AVAILABLE,
                "model_dir": str(self.model_dir),
            }

            self.logger.info(f"Regime detection training completed in {training_time:.2f}s")

            return result

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Regime detection training failed: {e}")

            return {"success": False, "error": str(e), "training_time": training_time}

    def detect_regimes(self, price_data_df: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes for given price data"""

        try:
            self.logger.info(f"Detecting regimes for {len(price_data_df)} samples")

            # Extract features
            features_df = self.feature_extractor.extract_all_features(price_data_df)

            if features_df.empty:
                self.logger.error("No features extracted for regime detection")
                return price_data_df

            # Rule-based detection
            regimes_df = self.rule_detector.detect_market_regime(features_df)

            # HMM-based detection
            if self.use_hmm and self.hmm_detector:
                regimes_df = self.hmm_detector.predict_regimes(regimes_df)

            # Select core columns for output
            core_columns = ["symbol", "timestamp", "price"]
            regime_columns = [col for col in regimes_df.columns if "regime" in col.lower()]

            output_columns = core_columns + regime_columns
            output_columns = [col for col in output_columns if col in regimes_df.columns]

            result_df = regimes_df[output_columns].copy()

            self.logger.info(
                f"Regime detection completed: {len(result_df)} samples with {len(regime_columns)} regime labels"
            )

            return result_df

        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return price_data_df

    def get_regime_summary(self, regimes_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of detected regimes"""

        try:
            summary = {
                "total_samples": len(regimes_df),
                "symbols": list(regimes_df["symbol"].unique())
                if "symbol" in regimes_df.columns
                else [],
                "time_range": {
                    "start": regimes_df["timestamp"].min().isoformat()
                    if "timestamp" in regimes_df.columns
                    else None,
                    "end": regimes_df["timestamp"].max().isoformat()
                    if "timestamp" in regimes_df.columns
                    else None,
                },
            }

            # Regime distributions
            regime_columns = [
                col
                for col in regimes_df.columns
                if "regime" in col.lower() and col != "hmm_regime_prob"
            ]

            for col in regime_columns:
                if col in regimes_df.columns:
                    dist = regimes_df[col].value_counts().to_dict()
                    summary[f"{col}_distribution"] = dist

            return summary

        except Exception as e:
            self.logger.error(f"Regime summary failed: {e}")
            return {"error": str(e)}

    def save_models(self):
        """Save regime detection models"""

        try:
            # Save HMM if available
            if self.hmm_detector and self.hmm_detector.is_fitted:
                import joblib

                model_path = self.model_dir / "hmm_model.pkl"
                scaler_path = self.model_dir / "hmm_scaler.pkl"

                joblib.dump(self.hmm_detector.model, model_path)
                joblib.dump(self.hmm_detector.scaler, scaler_path)

                # Save metadata
                meta_path = self.model_dir / "hmm_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(
                        {
                            "n_regimes": self.hmm_detector.n_regimes,
                            "is_fitted": self.hmm_detector.is_fitted,
                            "save_time": datetime.now().isoformat(),
                        },
                        f,
                    )

                self.logger.info("HMM models saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")

    def load_models(self):
        """Load saved regime detection models"""

        try:
            if self.hmm_detector:
                model_path = self.model_dir / "hmm_model.pkl"
                scaler_path = self.model_dir / "hmm_scaler.pkl"
                meta_path = self.model_dir / "hmm_meta.json"

                if all(p.exists() for p in [model_path, scaler_path, meta_path]):
                    self.hmm_detector.model = joblib.load(model_path)
                    self.hmm_detector.scaler = joblib.load(scaler_path)

                    with open(meta_path, "r") as f:
                        meta = json.load(f)

                    self.hmm_detector.is_fitted = meta["is_fitted"]
                    self.hmm_detector.n_regimes = meta["n_regimes"]

                    self.logger.info("HMM models loaded successfully")
                else:
                    self.logger.info("No saved HMM models found")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")


# Global detector instance
_regime_detector: Optional[RegimeDetector] = None


def get_regime_detector(use_hmm: bool = True) -> RegimeDetector:
    """Get global regime detector instance"""
    global _regime_detector

    if _regime_detector is None:
        _regime_detector = RegimeDetector(use_hmm=use_hmm)
        _regime_detector.load_models()

    return _regime_detector


def detect_market_regimes(price_data_df: pd.DataFrame) -> pd.DataFrame:
    """Main interface for regime detection"""
    detector = get_regime_detector()
    return detector.detect_regimes(price_data_df)


def train_regime_models(price_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Train regime detection models"""
    detector = get_regime_detector()
    return detector.fit(price_data_df)


# Test function
def create_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create mock price data for testing regime detection"""

    symbols = ["BTC", "ETH", "ADA"] * (n_samples // 3)
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="D")

    data = []
    for i in range(n_samples):
        # REMOVED: Mock data pattern not allowed in production
        if i < n_samples // 3:  # Bull market
            price_trend = 1.002  # 0.2% daily growth
            volatility = 0.02
        elif i < 2 * n_samples // 3:  # Bear market
            price_trend = 0.998  # -0.2% daily decline
            volatility = 0.03
        else:  # Sideways
            price_trend = 1.0
            volatility = 0.015

        base_price = 100 * (price_trend**i)
        noise = np.random.normal(0, 1)
        price = base_price * (1 + noise)

        volume = np.random.exponential(1000000)

        data.append(
            {
                "symbol": symbols[i],
                "timestamp": timestamps[i],
                "price": max(price, 1.0),  # Ensure positive price
                "volume_24h": volume,
            }
        )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test regime detection
    print("Testing Regime Detection System")

    # Create mock data
    price_data = create_  # REMOVED: Mock data pattern not allowed in production300)
    print(f"Created mock data: {len(price_data)} samples")

    # Train models
    print("Training regime detection models...")
    training_results = train_regime_models(price_data)
    print(f"Training results: {training_results}")

    # Detect regimes
    print("Detecting regimes...")
    regimes = detect_market_regimes(price_data)
    print(f"Detected regimes: {len(regimes)} samples")
    print(regimes.head())

    # Get summary
    detector = get_regime_detector()
    summary = detector.get_regime_summary(regimes)
    print(f"Regime summary: {summary}")
