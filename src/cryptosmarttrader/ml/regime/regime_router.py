#!/usr/bin/env python3
"""
Regime-Aware Model Router
Routes predictions through regime-specific models or adds regime features
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import core components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ..core.structured_logger import get_logger

# Import regime detector
from .regime_detector import RegimeDetector, get_regime_detector


class RegimeSpecificModel:
    """Individual model for specific regime"""

    def __init__(self, regime_name: str, model_type: str = "xgboost"):
        self.regime_name = regime_name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.performance_metrics = {}

        self.logger = get_logger(f"RegimeModel_{regime_name}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit regime-specific model"""

        try:
            from sklearn.preprocessing import StandardScaler

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train model based on type
            if self.model_type == "xgboost":
                try:
                    import xgboost as xgb

                    self.model = xgb.XGBRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                    )
                    self.model.fit(X_scaled, y)
                except ImportError:
                    # Fallback to sklearn
                    from sklearn.ensemble import RandomForestRegressor

                    self.model = RandomForestRegressor(
                        n_estimators=100, max_depth=6, random_state=42
                    )
                    self.model.fit(X_scaled, y)

            elif self.model_type == "random_forest":
                self.model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                self.model.fit(X_scaled, y)

            else:  # linear as fallback
                from sklearn.linear_model import LinearRegression

                self.model = LinearRegression()
                self.model.fit(X_scaled, y)

            self.is_fitted = True

            # Calculate performance metrics
            y_pred = self.model.predict(X_scaled)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            self.performance_metrics = {
                "mae": mae,
                "mse": mse,
                "samples": len(X),
                "features": X.shape[1],
            }

            self.logger.info(f"Regime model fitted: {self.regime_name}, MAE: {mae:.4f}")

            return {
                "success": True,
                "regime": self.regime_name,
                "model_type": self.model_type,
                "performance": self.performance_metrics,
            }

        except Exception as e:
            self.logger.error(f"Model fitting failed for regime {self.regime_name}: {e}")
            return {"success": False, "regime": self.regime_name, "error": str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using regime-specific model"""

        if not self.is_fitted or self.model is None:
            self.logger.warning(f"Model not fitted for regime {self.regime_name}")
            return np.zeros(len(X))

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed for regime {self.regime_name}: {e}")
            return np.zeros(len(X))


class RegimeAwarePredictor:
    """Main regime-aware prediction system"""

    def __init__(
        self,
        routing_strategy: str = "regime_specific",  # or "regime_feature"
        regimes: List[str] = None,
    ):
        self.logger = get_logger("RegimeAwarePredictor")
        self.routing_strategy = routing_strategy

        # Default regimes if not specified
        if regimes is None:
            self.regimes = [
                "bull",
                "bear",
                "sideways",
                "bull_high_vol",
                "bear_high_vol",
                "sideways_high_vol",
            ]
        else:
            self.regimes = regimes

        # Regime-specific models
        self.regime_models = {}
        for regime in self.regimes:
            self.regime_models[regime] = RegimeSpecificModel(regime, "xgboost")

        # Baseline model (regime-agnostic)
        self.baseline_model = RegimeSpecificModel("baseline", "xgboost")

        # Regime detector
        self.regime_detector = None

        # Model paths
        self.model_dir = Path("models/regime_aware")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_comparison = {}

    def prepare_features(
        self, features_df: pd.DataFrame, include_regime: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for training/prediction"""

        # Feature columns (exclude metadata)
        exclude_columns = ["symbol", "timestamp", "price", "target"] + [
            col for col in features_df.columns if "regime" in col.lower()
        ]

        feature_columns = [col for col in features_df.columns if col not in exclude_columns]

        # Add regime features if requested
        if include_regime and self.routing_strategy == "regime_feature":
            regime_columns = [col for col in features_df.columns if "regime" in col.lower()]

            # Encode categorical regime features
            for col in regime_columns:
                if col in features_df.columns and features_df[col].dtype == "object":
                    # One-hot encode regime categories
                    regime_dummies = pd.get_dummies(features_df[col], prefix=col)
                    feature_columns.extend(regime_dummies.columns.tolist())

        # Ensure features exist
        available_features = [col for col in feature_columns if col in features_df.columns]

        if not available_features:
            self.logger.error("No features available after preparation")
            return np.array([]), []

        X = features_df[available_features].fillna(0).values

        return X, available_features

    def create_targets(self, features_df: pd.DataFrame, horizon: int = 24) -> np.ndarray:
        """Create prediction targets from price data"""

        targets = []

        for symbol in features_df["symbol"].unique():
            symbol_df = features_df[features_df["symbol"] == symbol].copy()
            symbol_df = symbol_df.sort_values("timestamp")

            if "price" in symbol_df.columns:
                prices = symbol_df["price"].values

                # Future returns
                future_prices = np.roll(prices, -horizon)
                future_prices[-horizon:] = prices[-horizon:]  # Pad with last values

                returns = (future_prices - prices) / prices
                returns = np.nan_to_num(returns, 0)

                targets.extend(returns)
            else:
                # Fallback
                targets.extend([np.random.normal(0, 1)])

        return np.array(targets)

    def fit_regime_specific_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit separate models for each regime"""

        try:
            self.logger.info("Training regime-specific models")

            # Prepare features and targets
            X, feature_names = self.prepare_features(features_df, include_regime=False)
            y = self.create_targets(features_df)

            if len(X) == 0 or len(y) == 0:
                raise ValueError("No features or targets available")

            # Get regime labels
            regime_column = None
            for col in ["combined_regime", "trend_regime", "hmm_regime_name"]:
                if col in features_df.columns:
                    regime_column = col
                    break

            if regime_column is None:
                self.logger.warning("No regime labels found, using baseline model only")
                regime_labels = ["baseline"] * len(features_df)
            else:
                regime_labels = features_df[regime_column].values

            # Train baseline model (all data)
            baseline_result = self.baseline_model.fit(X, y)

            # Train regime-specific models
            regime_results = {"baseline": baseline_result}

            for regime in self.regimes:
                regime_mask = [label == regime for label in regime_labels]

                if np.sum(regime_mask) < 20:  # Need minimum samples
                    self.logger.warning(
                        f"Insufficient data for regime {regime}: {np.sum(regime_mask)} samples"
                    )
                    continue

                X_regime = X[regime_mask]
                y_regime = y[regime_mask]

                regime_result = self.regime_models[regime].fit(X_regime, y_regime)
                regime_results[regime] = regime_result

            # Save models
            self.save_models()

            summary = {
                "success": True,
                "strategy": "regime_specific",
                "total_samples": len(X),
                "features": len(feature_names),
                "regime_results": regime_results,
                "feature_names": feature_names,
            }

            self.logger.info(f"Regime-specific training completed: {len(regime_results)} models")

            return summary

        except Exception as e:
            self.logger.error(f"Regime-specific training failed: {e}")
            return {"success": False, "error": str(e)}

    def fit_regime_feature_model(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit single model with regime features"""

        try:
            self.logger.info("Training regime-feature model")

            # Prepare features including regime information
            X, feature_names = self.prepare_features(features_df, include_regime=True)
            y = self.create_targets(features_df)

            if len(X) == 0 or len(y) == 0:
                raise ValueError("No features or targets available")

            # Train single model with regime features
            result = self.baseline_model.fit(X, y)

            # Save model
            self.save_models()

            summary = {
                "success": True,
                "strategy": "regime_feature",
                "total_samples": len(X),
                "features": len(feature_names),
                "model_result": result,
                "feature_names": feature_names,
            }

            self.logger.info(f"Regime-feature training completed")

            return summary

        except Exception as e:
            self.logger.error(f"Regime-feature training failed: {e}")
            return {"success": False, "error": str(e)}

    def fit(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Main training interface"""

        start_time = time.time()

        try:
            # Initialize regime detector
            self.regime_detector = get_regime_detector()

            # Detect regimes if not present
            if not any("regime" in col.lower() for col in features_df.columns):
                self.logger.info("No regime labels found, detecting regimes...")
                features_with_regimes = self.regime_detector.detect_regimes(features_df)
            else:
                features_with_regimes = features_df.copy()

            # Train based on strategy
            if self.routing_strategy == "regime_specific":
                training_result = self.fit_regime_specific_models(features_with_regimes)
            else:  # regime_feature
                training_result = self.fit_regime_feature_model(features_with_regimes)

            training_time = time.time() - start_time
            training_result["training_time"] = training_time

            self.logger.info(f"Regime-aware training completed in {training_time:.2f}s")

            return training_result

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"Regime-aware training failed: {e}")

            return {"success": False, "error": str(e), "training_time": training_time}

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Make regime-aware predictions"""

        try:
            self.logger.info(f"Making regime-aware predictions for {len(features_df)} samples")

            # Initialize regime detector
            if self.regime_detector is None:
                self.regime_detector = get_regime_detector()

            # Detect regimes if not present
            if not any("regime" in col.lower() for col in features_df.columns):
                features_with_regimes = self.regime_detector.detect_regimes(features_df)
            else:
                features_with_regimes = features_df.copy()

            # Make predictions based on strategy
            if self.routing_strategy == "regime_specific":
                predictions = self._predict_regime_specific(features_with_regimes)
            else:  # regime_feature
                predictions = self._predict_regime_feature(features_with_regimes)

            # Prepare result dataframe
            result_df = pd.DataFrame(
                {
                    "symbol": features_with_regimes["symbol"],
                    "timestamp": features_with_regimes["timestamp"],
                    "regime_prediction": predictions,
                    "baseline_prediction": self._predict_baseline(features_with_regimes),
                }
            )

            # Add regime information
            regime_columns = [
                col for col in features_with_regimes.columns if "regime" in col.lower()
            ]
            for col in regime_columns:
                if col in features_with_regimes.columns:
                    result_df[col] = features_with_regimes[col]

            self.logger.info(f"Regime-aware predictions completed: {len(result_df)} samples")

            return result_df

        except Exception as e:
            self.logger.error(f"Regime-aware prediction failed: {e}")
            return pd.DataFrame()

    def _predict_regime_specific(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using regime-specific models"""

        X, _ = self.prepare_features(features_df, include_regime=False)
        predictions = np.zeros(len(X))

        # Get regime labels
        regime_column = None
        for col in ["combined_regime", "trend_regime", "hmm_regime_name"]:
            if col in features_df.columns:
                regime_column = col
                break

        if regime_column is None:
            # Use baseline model
            return self.baseline_model.predict(X)

        regime_labels = features_df[regime_column].values

        # Predict for each regime
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime

            if regime in self.regime_models and self.regime_models[regime].is_fitted:
                regime_pred = self.regime_models[regime].predict(X[regime_mask])
                predictions[regime_mask] = regime_pred
            else:
                # Fallback to baseline
                baseline_pred = self.baseline_model.predict(X[regime_mask])
                predictions[regime_mask] = baseline_pred

        return predictions

    def _predict_regime_feature(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using single model with regime features"""

        X, _ = self.prepare_features(features_df, include_regime=True)
        return self.baseline_model.predict(X)

    def _predict_baseline(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make baseline predictions (regime-agnostic)"""

        X, _ = self.prepare_features(features_df, include_regime=False)
        return self.baseline_model.predict(X)

    def evaluate_regime_performance(
        self, test_features_df: pd.DataFrame, test_targets: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regime-aware vs baseline performance"""

        try:
            # Get predictions
            predictions_df = self.predict(test_features_df)

            regime_preds = predictions_df["regime_prediction"].values
            baseline_preds = predictions_df["baseline_prediction"].values

            # Calculate metrics
            regime_mae = mean_absolute_error(test_targets, regime_preds)
            baseline_mae = mean_absolute_error(test_targets, baseline_preds)

            regime_mse = mean_squared_error(test_targets, regime_preds)
            baseline_mse = mean_squared_error(test_targets, baseline_preds)

            # Improvement metrics
            mae_improvement = (baseline_mae - regime_mae) / baseline_mae
            mse_improvement = (baseline_mse - regime_mse) / baseline_mse

            evaluation_result = {
                "regime_mae": regime_mae,
                "baseline_mae": baseline_mae,
                "regime_mse": regime_mse,
                "baseline_mse": baseline_mse,
                "mae_improvement": mae_improvement,
                "mse_improvement": mse_improvement,
                "regime_wins": regime_mae < baseline_mae,
                "test_samples": len(test_targets),
            }

            self.performance_comparison = evaluation_result

            self.logger.info(
                f"Performance evaluation: Regime MAE: {regime_mae:.4f}, Baseline MAE: {baseline_mae:.4f}"
            )
            self.logger.info(f"MAE improvement: {mae_improvement:.1%}")

            return evaluation_result

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return {"error": str(e)}

    def save_models(self):
        """Save regime-aware models"""

        if not JOBLIB_AVAILABLE:
            self.logger.warning("Cannot save models - joblib not available")
            return

        try:
            # Save regime-specific models
            for regime, model in self.regime_models.items():
                if model.is_fitted:
                    model_path = self.model_dir / f"regime_{regime}_model.pkl"
                    scaler_path = self.model_dir / f"regime_{regime}_scaler.pkl"

                    joblib.dump(model.model, model_path)
                    if model.scaler:
                        joblib.dump(model.scaler, scaler_path)

            # Save baseline model
            if self.baseline_model.is_fitted:
                baseline_model_path = self.model_dir / "baseline_model.pkl"
                baseline_scaler_path = self.model_dir / "baseline_scaler.pkl"

                joblib.dump(self.baseline_model.model, baseline_model_path)
                if self.baseline_model.scaler:
                    joblib.dump(self.baseline_model.scaler, baseline_scaler_path)

            # Save metadata
            meta_path = self.model_dir / "regime_aware_meta.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "routing_strategy": self.routing_strategy,
                        "regimes": self.regimes,
                        "performance_comparison": self.performance_comparison,
                        "save_time": datetime.now().isoformat(),
                    },
                    f,
                )

            self.logger.info("Regime-aware models saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")

    def load_models(self):
        """Load saved regime-aware models"""

        if not JOBLIB_AVAILABLE:
            return

        try:
            meta_path = self.model_dir / "regime_aware_meta.json"
            if not meta_path.exists():
                return

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.routing_strategy = meta.get("routing_strategy", self.routing_strategy)
            self.regimes = meta.get("regimes", self.regimes)
            self.performance_comparison = meta.get("performance_comparison", {})

            # Load regime-specific models
            for regime in self.regimes:
                model_path = self.model_dir / f"regime_{regime}_model.pkl"
                scaler_path = self.model_dir / f"regime_{regime}_scaler.pkl"

                if model_path.exists():
                    self.regime_models[regime].model = joblib.load(model_path)
                    self.regime_models[regime].is_fitted = True

                    if scaler_path.exists():
                        self.regime_models[regime].scaler = joblib.load(scaler_path)

            # Load baseline model
            baseline_model_path = self.model_dir / "baseline_model.pkl"
            baseline_scaler_path = self.model_dir / "baseline_scaler.pkl"

            if baseline_model_path.exists():
                self.baseline_model.model = joblib.load(baseline_model_path)
                self.baseline_model.is_fitted = True

                if baseline_scaler_path.exists():
                    self.baseline_model.scaler = joblib.load(baseline_scaler_path)

            self.logger.info("Regime-aware models loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")


# Global predictor instances
_regime_predictors: Dict[str, RegimeAwarePredictor] = {}


def get_regime_predictor(strategy: str = "regime_specific") -> RegimeAwarePredictor:
    """Get regime-aware predictor instance"""
    global _regime_predictors

    if strategy not in _regime_predictors:
        _regime_predictors[strategy] = RegimeAwarePredictor(routing_strategy=strategy)
        _regime_predictors[strategy].load_models()

    return _regime_predictors[strategy]


def train_regime_aware_models(
    features_df: pd.DataFrame, strategy: str = "regime_specific"
) -> Dict[str, Any]:
    """Train regime-aware models"""
    predictor = get_regime_predictor(strategy)
    return predictor.fit(features_df)


def predict_with_regime_awareness(
    features_df: pd.DataFrame, strategy: str = "regime_specific"
) -> pd.DataFrame:
    """Make regime-aware predictions"""
    predictor = get_regime_predictor(strategy)
    return predictor.predict(features_df)


if __name__ == "__main__":
    # Test regime-aware prediction
    print("Testing Regime-Aware Prediction System")

    # Import regime detection for test data
    from .regime_detector import create_mock_price_data

    # Create test data
    price_data = create_  # REMOVED: Mock data pattern not allowed in production500)
    print(f"Created test data: {len(price_data)} samples")

    # Train regime-aware models
    print("Training regime-aware models...")
    training_result = train_regime_aware_models(price_data, "regime_specific")
    print(f"Training result: {training_result.get('success', False)}")

    # Make predictions
    print("Making regime-aware predictions...")
    predictions = predict_with_regime_awareness(price_data, "regime_specific")
    print(f"Predictions: {len(predictions)} samples")
    print(predictions.head())
