#!/usr/bin/env python3
"""
Regime Detector - Market regime identification and routing
Provides regime features and model routing for better predictions
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import json

from core.structured_logger import get_structured_logger

class MarketRegimeDetector:
    """Detect market regimes and route to appropriate models"""
    
    def __init__(self, n_regimes: int = 4):
        self.logger = get_structured_logger("MarketRegimeDetector")
        self.n_regimes = n_regimes
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_names = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
        self.is_fitted = False
        
        # Regime-specific model performance tracking
        self.regime_performance = {}
        
    def engineer_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for regime detection"""
        
        if data.empty:
            return data
        
        # Ensure we have required columns
        required_cols = ["coin", "timestamp"]
        feature_cols = [col for col in data.columns if col.startswith("feat_")]
        
        if not all(col in data.columns for col in required_cols):
            self.logger.error("Missing required columns for regime features")
            return data
        
        if len(feature_cols) == 0:
            self.logger.error("No feature columns found")
            return data
        
        # Sort by coin and timestamp for rolling calculations
        data_sorted = data.sort_values(["coin", "timestamp"]).copy()
        
        # Group by coin for rolling features
        regime_features = []
        
        for coin, coin_data in data_sorted.groupby("coin"):
            coin_features = coin_data.copy()
            
            # Market momentum features
            if "feat_momentum" in coin_data.columns:
                coin_features["regime_momentum_ma5"] = coin_data["feat_momentum"].rolling(5, min_periods=1).mean()
                coin_features["regime_momentum_std5"] = coin_data["feat_momentum"].rolling(5, min_periods=1).std().fillna(0)
            
            # Volatility regime
            if "feat_volatility" in coin_data.columns:
                coin_features["regime_vol_ma10"] = coin_data["feat_volatility"].rolling(10, min_periods=1).mean()
                coin_features["regime_vol_ratio"] = (coin_data["feat_volatility"] / 
                                                   coin_features["regime_vol_ma10"]).fillna(1)
            
            # Volume regime
            if "feat_volume" in coin_data.columns:
                coin_features["regime_vol_ma7"] = coin_data["feat_volume"].rolling(7, min_periods=1).mean()
                coin_features["regime_vol_spike"] = (coin_data["feat_volume"] > 
                                                   coin_features["regime_vol_ma7"] * 1.5).astype(int)
            
            # Technical regime
            if "feat_rsi" in coin_data.columns:
                coin_features["regime_rsi_extreme"] = ((coin_data["feat_rsi"] < 30) | 
                                                     (coin_data["feat_rsi"] > 70)).astype(int)
                coin_features["regime_rsi_neutral"] = ((coin_data["feat_rsi"] >= 40) & 
                                                     (coin_data["feat_rsi"] <= 60)).astype(int)
            
            # Cross-asset correlation proxy (simplified)
            if len(feature_cols) >= 3:
                # Use feature correlation as market coherence measure
                feature_matrix = coin_data[feature_cols[:3]].values
                if len(feature_matrix) > 5:
                    correlation_matrix = np.corrcoef(feature_matrix.T)
                    avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
                    coin_features["regime_market_coherence"] = avg_correlation
                else:
                    coin_features["regime_market_coherence"] = 0.5
            
            regime_features.append(coin_features)
        
        # Combine all coin data
        if regime_features:
            result = pd.concat(regime_features, ignore_index=True)
            
            # Add aggregate market features
            result = self._add_market_level_features(result)
            
            self.logger.info(f"Engineered regime features: {len(result)} samples")
            return result
        else:
            return data
    
    def _add_market_level_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market-level regime features"""
        
        # Group by timestamp to get market-wide measures
        market_features = []
        
        for timestamp, timestamp_data in data.groupby("timestamp"):
            timestamp_features = timestamp_data.copy()
            
            # Market-wide volatility
            if "feat_volatility" in timestamp_data.columns:
                market_vol = timestamp_data["feat_volatility"].mean()
                timestamp_features["regime_market_vol"] = market_vol
            
            # Market breadth (how many coins are moving in same direction)
            if "feat_momentum" in timestamp_data.columns:
                positive_momentum = (timestamp_data["feat_momentum"] > 0).sum()
                total_coins = len(timestamp_data)
                breadth = positive_momentum / max(total_coins, 1)
                timestamp_features["regime_market_breadth"] = breadth
            
            # Market stress (dispersion of returns)
            if "feat_momentum" in timestamp_data.columns:
                momentum_std = timestamp_data["feat_momentum"].std()
                timestamp_features["regime_market_stress"] = momentum_std
            
            market_features.append(timestamp_features)
        
        if market_features:
            return pd.concat(market_features, ignore_index=True)
        else:
            return data
    
    def fit_regime_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit Gaussian Mixture Model for regime detection"""
        
        # Get regime feature columns
        regime_cols = [col for col in data.columns if col.startswith("regime_")]
        
        if len(regime_cols) == 0:
            raise ValueError("No regime features found - run engineer_regime_features first")
        
        # Prepare data
        regime_data = data[regime_cols].fillna(0)
        
        if len(regime_data) < 100:
            self.logger.warning(f"Limited data for regime fitting: {len(regime_data)} samples")
        
        # Scale features
        regime_scaled = self.scaler.fit_transform(regime_data)
        
        # Fit Gaussian Mixture Model
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
            max_iter=200
        )
        
        self.regime_model.fit(regime_scaled)
        self.is_fitted = True
        
        # Predict regimes for training data
        regime_labels = self.regime_model.predict(regime_scaled)
        regime_probs = self.regime_model.predict_proba(regime_scaled)
        
        # Analyze regime characteristics
        regime_analysis = self._analyze_regimes(data, regime_labels, regime_cols)
        
        self.logger.info(f"Regime model fitted with {self.n_regimes} regimes")
        
        return {
            "success": True,
            "n_regimes": self.n_regimes,
            "regime_features": regime_cols,
            "samples_used": len(regime_data),
            "regime_analysis": regime_analysis
        }
    
    def _analyze_regimes(self, data: pd.DataFrame, regime_labels: np.ndarray, regime_cols: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of detected regimes"""
        
        analysis = {}
        
        for regime_id in range(self.n_regimes):
            regime_mask = regime_labels == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate regime characteristics
            regime_stats = {}
            
            for col in regime_cols:
                if col in regime_data.columns:
                    regime_stats[col] = {
                        "mean": regime_data[col].mean(),
                        "std": regime_data[col].std(),
                        "median": regime_data[col].median()
                    }
            
            # Calculate typical market conditions
            conditions = self._classify_regime_conditions(regime_data, regime_cols)
            
            analysis[f"regime_{regime_id}"] = {
                "samples": len(regime_data),
                "percentage": len(regime_data) / len(data) * 100,
                "statistics": regime_stats,
                "conditions": conditions,
                "name": self._name_regime(conditions)
            }
        
        return analysis
    
    def _classify_regime_conditions(self, regime_data: pd.DataFrame, regime_cols: List[str]) -> Dict[str, str]:
        """Classify regime based on feature patterns"""
        
        conditions = {}
        
        # Volatility condition
        if "regime_vol_ratio" in regime_data.columns:
            vol_ratio = regime_data["regime_vol_ratio"].mean()
            if vol_ratio > 1.2:
                conditions["volatility"] = "HIGH"
            elif vol_ratio < 0.8:
                conditions["volatility"] = "LOW"
            else:
                conditions["volatility"] = "NORMAL"
        
        # Momentum condition
        if "regime_momentum_ma5" in regime_data.columns:
            momentum = regime_data["regime_momentum_ma5"].mean()
            if momentum > 0.01:
                conditions["momentum"] = "BULLISH"
            elif momentum < -0.01:
                conditions["momentum"] = "BEARISH"
            else:
                conditions["momentum"] = "NEUTRAL"
        
        # Market breadth
        if "regime_market_breadth" in regime_data.columns:
            breadth = regime_data["regime_market_breadth"].mean()
            if breadth > 0.6:
                conditions["breadth"] = "BROAD"
            elif breadth < 0.4:
                conditions["breadth"] = "NARROW"
            else:
                conditions["breadth"] = "MIXED"
        
        # Volume activity
        if "regime_vol_spike" in regime_data.columns:
            vol_spikes = regime_data["regime_vol_spike"].mean()
            if vol_spikes > 0.3:
                conditions["volume"] = "HIGH_ACTIVITY"
            else:
                conditions["volume"] = "NORMAL_ACTIVITY"
        
        return conditions
    
    def _name_regime(self, conditions: Dict[str, str]) -> str:
        """Name regime based on conditions"""
        
        momentum = conditions.get("momentum", "NEUTRAL")
        volatility = conditions.get("volatility", "NORMAL")
        
        if momentum == "BULLISH" and volatility in ["LOW", "NORMAL"]:
            return "BULL"
        elif momentum == "BEARISH" and volatility in ["LOW", "NORMAL"]:
            return "BEAR"
        elif volatility == "HIGH":
            return "VOLATILE"
        else:
            return "SIDEWAYS"
    
    def predict_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for new data"""
        
        if not self.is_fitted:
            self.logger.warning("Regime model not fitted - cannot predict")
            return data
        
        # Get regime features
        regime_cols = [col for col in data.columns if col.startswith("regime_")]
        
        if len(regime_cols) == 0:
            self.logger.warning("No regime features found for prediction")
            return data
        
        # Prepare and scale data
        regime_data = data[regime_cols].fillna(0)
        regime_scaled = self.scaler.transform(regime_data)
        
        # Predict regimes
        regime_labels = self.regime_model.predict(regime_scaled)
        regime_probs = self.regime_model.predict_proba(regime_scaled)
        
        # Add predictions to data
        result = data.copy()
        result["regime_id"] = regime_labels
        result["regime_confidence"] = np.max(regime_probs, axis=1)
        
        # Add regime names
        regime_names_mapped = [self.regime_names[min(label, len(self.regime_names)-1)] for label in regime_labels]
        result["regime_name"] = regime_names_mapped
        
        return result
    
    def evaluate_regime_routing(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate if regime-based routing improves performance"""
        
        if "regime_id" not in predictions.columns:
            return {"error": "No regime predictions found"}
        
        horizons = ["1h", "24h", "168h", "720h"]
        evaluation_results = {}
        
        for horizon in horizons:
            pred_col = f"pred_{horizon}"
            actual_col = f"actual_{horizon}"
            
            if pred_col not in predictions.columns or actual_col not in actuals.columns:
                continue
            
            # Merge predictions and actuals
            merged = pd.merge(predictions, actuals, on=["coin", "timestamp"], how="inner")
            
            if len(merged) == 0:
                continue
            
            # Calculate overall MAE
            overall_mae = mean_absolute_error(merged[actual_col], merged[pred_col])
            
            # Calculate regime-specific MAE
            regime_maes = {}
            regime_counts = {}
            
            for regime_id in range(self.n_regimes):
                regime_data = merged[merged["regime_id"] == regime_id]
                
                if len(regime_data) > 10:  # Minimum samples for meaningful MAE
                    regime_mae = mean_absolute_error(regime_data[actual_col], regime_data[pred_col])
                    regime_maes[regime_id] = regime_mae
                    regime_counts[regime_id] = len(regime_data)
            
            # Calculate weighted MAE improvement
            if regime_maes:
                weighted_regime_mae = sum(
                    mae * regime_counts[regime_id] 
                    for regime_id, mae in regime_maes.items()
                ) / sum(regime_counts.values())
                
                mae_improvement = (overall_mae - weighted_regime_mae) / overall_mae
            else:
                weighted_regime_mae = overall_mae
                mae_improvement = 0
            
            evaluation_results[horizon] = {
                "overall_mae": overall_mae,
                "weighted_regime_mae": weighted_regime_mae,
                "mae_improvement": mae_improvement,
                "regime_maes": regime_maes,
                "regime_counts": regime_counts
            }
        
        self.regime_performance = evaluation_results
        
        return {
            "evaluation_results": evaluation_results,
            "overall_improvement": np.mean([
                result["mae_improvement"] 
                for result in evaluation_results.values()
            ])
        }
    
    def save_regime_model(self, filepath: str):
        """Save fitted regime model"""
        
        if not self.is_fitted:
            raise ValueError("Regime model not fitted - cannot save")
        
        model_data = {
            "regime_model": self.regime_model,
            "scaler": self.scaler,
            "n_regimes": self.n_regimes,
            "regime_names": self.regime_names,
            "is_fitted": self.is_fitted,
            "regime_performance": self.regime_performance
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Regime model saved to {filepath}")
    
    def load_regime_model(self, filepath: str):
        """Load fitted regime model"""
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Regime model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.regime_model = model_data["regime_model"]
        self.scaler = model_data["scaler"]
        self.n_regimes = model_data["n_regimes"]
        self.regime_names = model_data["regime_names"]
        self.is_fitted = model_data["is_fitted"]
        self.regime_performance = model_data.get("regime_performance", {})
        
        self.logger.info(f"Regime model loaded from {filepath}")

def create_test_regime_data() -> pd.DataFrame:
    """Create test data with different regime patterns"""
    
    np.random.seed(42)
    n_samples = 1000
    n_coins = 20
    
    data = []
    
    for i in range(n_samples):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        
        # Create regime-dependent patterns
        regime_cycle = i // 200  # Change regime every 200 samples
        
        for coin_idx in range(n_coins):
            coin = f"COIN_{coin_idx:03d}"
            
            if regime_cycle % 4 == 0:  # Bull regime
                momentum = np.random.normal(0.02, 0.01)
                volatility = np.random.uniform(0.01, 0.03)
                volume = np.random.lognormal(10, 0.5)
            elif regime_cycle % 4 == 1:  # Bear regime
                momentum = np.random.normal(-0.02, 0.01)
                volatility = np.random.uniform(0.01, 0.03)
                volume = np.random.lognormal(9.5, 0.7)
            elif regime_cycle % 4 == 2:  # Volatile regime
                momentum = np.random.normal(0, 0.03)
                volatility = np.random.uniform(0.05, 0.15)
                volume = np.random.lognormal(10.5, 1.0)
            else:  # Sideways regime
                momentum = np.random.normal(0, 0.005)
                volatility = np.random.uniform(0.005, 0.015)
                volume = np.random.lognormal(9.8, 0.3)
            
            data.append({
                "coin": coin,
                "timestamp": timestamp,
                "feat_momentum": momentum,
                "feat_volatility": volatility,
                "feat_volume": volume,
                "feat_rsi": np.random.uniform(30, 70),
                "feat_macd": np.random.normal(0, 0.01)
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Testing Market Regime Detector...")
    
    # Create test data
    test_data = create_test_regime_data()
    print(f"Created test data: {len(test_data)} samples")
    
    # Initialize detector
    detector = MarketRegimeDetector(n_regimes=4)
    
    # Engineer regime features
    data_with_regimes = detector.engineer_regime_features(test_data)
    regime_cols = [col for col in data_with_regimes.columns if col.startswith("regime_")]
    print(f"Engineered {len(regime_cols)} regime features")
    
    # Fit regime model
    fit_results = detector.fit_regime_model(data_with_regimes)
    print(f"Regime model fit: {fit_results['success']}")
    
    # Predict regimes
    predictions = detector.predict_regime(data_with_regimes)
    print(f"Regime predictions: {predictions['regime_name'].value_counts().to_dict()}")
    
    # Show regime distribution
    regime_distribution = predictions['regime_name'].value_counts()
    print(f"\nRegime Distribution:")
    for regime, count in regime_distribution.items():
        percentage = count / len(predictions) * 100
        print(f"  {regime}: {count} samples ({percentage:.1f}%)")
    
    print("\n✅ Market Regime Detector test completed!")