import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import json  # SECURITY: Replaced pickle with json
from pathlib import Path

# Machine Learning libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class MLPredictorAgent:
    """Machine Learning Price Prediction Agent with ensemble models"""

    def __init__(self, config_manager, data_manager, cache_manager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0

        # Model storage
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.model_performance = {}
        self._lock = threading.Lock()

        # Model paths
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)

        # Prediction horizons
        self.horizons = self.config_manager.get("prediction_horizons", ["1d", "7d"])

        # Load existing models
        self._load_models()

        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("ml_predictor", {}).get("enabled", True):
            self.start()

    def start(self):
        """Start the ML prediction agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._prediction_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("ML Predictor Agent started")

    def stop(self):
        """Stop the ML prediction agent"""
        self.active = False
        self.logger.info("ML Predictor Agent stopped")

    def _prediction_loop(self):
        """Main prediction loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = (
                    self.config_manager.get("agents", {})
                    .get("ml_predictor", {})
                    .get("update_interval", 900)
                )

                # Train models and make predictions
                self._update_predictions()

                # Update last update time
                self.last_update = datetime.now()

                # Sleep until next update
                time.sleep(interval)

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"ML prediction error: {str(e)}")
                time.sleep(300)  # Sleep 5 minutes on error

    def _update_predictions(self):
        """Update predictions for all symbols"""
        try:
            symbols = self.data_manager.get_supported_symbols()

            for symbol in symbols[:50]:  # Limit for efficiency
                try:
                    self._process_symbol(symbol)
                    self.processed_count += 1

                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Prediction update error: {str(e)}")

    def _process_symbol(self, symbol: str):
        """Process a single symbol for predictions"""
        # Get historical data for training
        historical_data = self._prepare_training_data(symbol)

        if historical_data is None or len(historical_data) < 100:
            return

        # Train models for each horizon
        symbol_predictions = {}

        for horizon in self.horizons:
            try:
                # Train model
                model_key = f"{symbol}_{horizon}"
                model_info = self._train_model(historical_data, symbol, horizon)

                if model_info:
                    # Make prediction
                    prediction = self._make_prediction(historical_data, model_info, horizon)

                    if prediction:
                        symbol_predictions[horizon] = prediction

            except Exception as e:
                self.logger.error(f"Error training model for {symbol} {horizon}: {str(e)}")
                continue

        # Store predictions
        if symbol_predictions:
            self._store_predictions(symbol, symbol_predictions)

    def _prepare_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare data for ML training"""
        # Get historical data
        historical_data = self.data_manager.get_historical_data(symbol, days=90)

        if historical_data is None or len(historical_data) < 50:
            # Placeholder removed
            return self._generate_  # Placeholder removed

        return historical_data

    def _get_authentic_training_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get authentic historical data for model training - NO MOCK DATA"""

        try:
            # Get data from real exchange API
            if hasattr(self.data_manager, "get_historical_data"):
                historical_data = self.data_manager.get_historical_data(symbol, limit=200)
                if historical_data is not None and not historical_data.empty:
                    self.logger.info(
                        f"✅ Retrieved {len(historical_data)} authentic data points for {symbol}"
                    )
                    return historical_data

            # Try direct CCXT call
            import ccxt

            exchange = ccxt.kraken({"enableRateLimit": True})
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=200)

            if ohlcv and len(ohlcv) > 100:
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["symbol"] = symbol
                df["price"] = df["close"]

                self.logger.info(f"✅ Retrieved {len(df)} authentic OHLCV data points for {symbol}")
                return df
            else:
                self.logger.error(f"❌ Insufficient authentic data for {symbol}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to get authentic training data for {symbol}: {e}")
            return None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        df = df.copy()
        df = df.sort_values("timestamp")

        # Price-based features
        df["returns"] = df["price"].pct_change()
        df["log_returns"] = np.log(df["price"] / df["price"].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"ma_{window}"] = df["price"].rolling(window=window).mean()
            df[f"price_ma_ratio_{window}"] = df["price"] / df[f"ma_{window}"]

        # Volatility features
        df["volatility_5"] = df["returns"].rolling(window=5).std()
        df["volatility_20"] = df["returns"].rolling(window=20).std()

        # Volume features
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"]

        # Technical indicators
        df["rsi"] = self._get_technical_analyzer().calculate_indicator("RSI", df["price"], 14).values
        df["bb_position"] = self._calculate_bollinger_position(df["price"], 20)

        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f"price_lag_{lag}"] = df["price"].shift(lag)
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day

        return df

    def _get_technical_analyzer().calculate_indicator("RSI", self, prices: pd.Series, window: int = 14).values -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position

    def _train_model(
        self, data: pd.DataFrame, symbol: str, horizon: str
    ) -> Optional[Dict[str, Any]]:
        """Train ML models for prediction"""
        try:
            # Create features
            df_features = self._create_features(data)

            # Create target variable based on horizon
            horizon_hours = self._horizon_to_hours(horizon)
            df_features[f"target_{horizon}"] = df_features["price"].shift(-horizon_hours)

            # Remove NaN values
            df_clean = df_features.dropna()

            if len(df_clean) < 50:
                return None

            # Select feature columns
            feature_columns = [
                col
                for col in df_clean.columns
                if col not in ["timestamp", "symbol", "price", "open", "high", "low", "volume"]
                and not col.startswith("target_")
            ]

            X = df_clean[feature_columns]
            y = df_clean[f"target_{horizon}"]

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble models
            models = {}

            # Placeholder removed
            if SKLEARN_AVAILABLE:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                models["random_forest"] = rf_model

            # XGBoost
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
                models["xgboost"] = xgb_model

            # LightGBM
            if LIGHTGBM_AVAILABLE:
                lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
                lgb_model.fit(X_train_scaled, y_train)
                models["lightgbm"] = lgb_model

            if not models:
                # Fallback to simple linear model
                from sklearn.linear_model import LinearRegression

                lr_model = LinearRegression()
                lr_model.fit(X_train_scaled, y_train)
                models["linear"] = lr_model

            # Evaluate models
            model_scores = {}
            predictions = {}

            for model_name, model in models.items():
                try:
                    pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, pred)
                    r2 = r2_score(y_test, pred)

                    model_scores[model_name] = {"mse": mse, "r2": r2}
                    predictions[model_name] = pred

                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name}: {str(e)}")

            # Save best model
            if model_scores:
                best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x]["mse"])
                best_model = models[best_model_name]

                model_info = {
                    "model": best_model,
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "model_name": best_model_name,
                    "performance": model_scores[best_model_name],
                    "trained_at": datetime.now().isoformat(),
                }

                # Store model
                model_key = f"{symbol}_{horizon}"
                self.models[model_key] = model_info
                self.scalers[model_key] = scaler

                # Save to disk
                self._save_model(model_key, model_info)

                return model_info

            return None

        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            return None

    def _horizon_to_hours(self, horizon: str) -> int:
        """Convert horizon string to hours"""
        if horizon == "1h":
            return 1
        elif horizon == "4h":
            return 4
        elif horizon == "1d":
            return 24
        elif horizon == "3d":
            return 72
        elif horizon == "7d":
            return 168
        elif horizon == "30d":
            return 720
        else:
            return 24  # Default to 1 day

    def _make_prediction(
        self, data: pd.DataFrame, model_info: Dict[str, Any], horizon: str
    ) -> Optional[Dict[str, Any]]:
        """Make price prediction using trained model"""
        try:
            # Create features for the latest data point
            df_features = self._create_features(data)
            df_clean = df_features.dropna()

            if df_clean.empty:
                return None

            # Get latest features
            feature_columns = model_info["feature_columns"]
            latest_features = df_clean[feature_columns].iloc[-1:].values

            # Scale features
            scaler = model_info["scaler"]
            latest_features_scaled = scaler.transform(latest_features)

            # Make prediction
            model = model_info["model"]
            prediction = model.predict(latest_features_scaled)[0]

            # Calculate prediction metrics
            current_price = data["price"].iloc[-1]
            predicted_change = (prediction - current_price) / current_price

            # Estimate confidence based on model performance
            r2_score = model_info["performance"]["r2"]
            confidence = max(0.1, min(0.95, r2_score))

            return {
                "timestamp": datetime.now().isoformat(),
                "horizon": horizon,
                "current_price": current_price,
                "predicted_price": prediction,
                "predicted_change_percent": predicted_change * 100,
                "confidence": confidence,
                "model_name": model_info["model_name"],
                "model_performance": model_info["performance"],
            }

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return None

    def _store_predictions(self, symbol: str, predictions: Dict[str, Any]):
        """Store predictions for a symbol"""
        with self._lock:
            if symbol not in self.predictions:
                self.predictions[symbol] = []

            prediction_entry = {"timestamp": datetime.now().isoformat(), "predictions": predictions}

            self.predictions[symbol].append(prediction_entry)

            # Keep only last 48 hours of predictions
            cutoff_time = datetime.now() - timedelta(hours=48)
            self.predictions[symbol] = [
                pred
                for pred in self.predictions[symbol]
                if datetime.fromisoformat(pred["timestamp"]) > cutoff_time
            ]

    def _save_model(self, model_key: str, model_info: Dict[str, Any]):
        """Save model to disk"""
        try:
            model_file = self.models_path / f"{model_key.replace('/', '_')}.pkl"

            # Save model data (exclude the actual model object for disk storage)
            save_data = {
                "feature_columns": model_info["feature_columns"],
                "model_name": model_info["model_name"],
                "performance": model_info["performance"],
                "trained_at": model_info["trained_at"],
            }

            with open(model_file, "wb") as f:
                json.dump(save_data, f)

        except Exception as e:
            self.logger.error(f"Error saving model {model_key}: {str(e)}")

    def _load_models(self):
        """Load existing models from disk"""
        try:
            for model_file in self.models_path.glob("*.pkl"):
                try:
                    with open(model_file, "rb") as f:
                        model_data = json.load(f)

                    model_key = model_file.stem.replace("_", "/")
                    self.model_performance[model_key] = model_data

                except Exception as e:
                    self.logger.error(f"Error loading model {model_file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")

    def get_prediction(self, symbol: str, horizon: str = None) -> Optional[Dict[str, Any]]:
        """Get latest prediction for a symbol"""
        with self._lock:
            if symbol not in self.predictions or not self.predictions[symbol]:
                return None

            latest_predictions = self.predictions[symbol][-1]["predictions"]

            if horizon:
                return latest_predictions.get(horizon)
            else:
                return latest_predictions

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions"""
        with self._lock:
            if not self.predictions:
                return {
                    "total_symbols": 0,
                    "total_predictions": 0,
                    "bullish_predictions": 0,
                    "bearish_predictions": 0,
                    "avg_confidence": 0,
                }

            total_predictions = 0
            bullish_count = 0
            bearish_count = 0
            confidence_sum = 0

            for symbol, prediction_list in self.predictions.items():
                if prediction_list:
                    latest = prediction_list[-1]["predictions"]

                    for horizon, pred in latest.items():
                        total_predictions += 1
                        if pred["predicted_change_percent"] > 0:
                            bullish_count += 1
                        else:
                            bearish_count += 1
                        confidence_sum += pred["confidence"]

            avg_confidence = confidence_sum / total_predictions if total_predictions > 0 else 0

            return {
                "total_symbols": len(self.predictions),
                "total_predictions": total_predictions,
                "bullish_predictions": bullish_count,
                "bearish_predictions": bearish_count,
                "avg_confidence": avg_confidence,
                "prediction_timestamp": datetime.now().isoformat(),
            }

    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        with self._lock:
            return {
                "active_models": len(self.models),
                "model_performance": self.model_performance.copy(),
                "available_libraries": {
                    "sklearn": SKLEARN_AVAILABLE,
                    "xgboost": XGBOOST_AVAILABLE,
                    "lightgbm": LIGHTGBM_AVAILABLE,
                },
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "active": self.active,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "trained_models": len(self.models),
            "predicted_symbols": len(self.predictions),
            "horizons": self.horizons,
        }
