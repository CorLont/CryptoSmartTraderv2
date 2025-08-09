#!/usr/bin/env python3
"""
Baseline Trainer - RF ensemble + uncertainty voor alle horizons
Train minimaal 4 modellen (1h, 24h, 168h, 720h) met ensemble uncertainty
"""

import pandas as pd
import joblib
import os
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime

# Horizon mappings
HORIZONS = {
    "1h": "target_1h",
    "24h": "target_24h", 
    "168h": "target_168h",
    "720h": "target_720h"
}

def create_synthetic_features(n_coins=20, n_timestamps=1000):
    """Create synthetic features for testing when real data unavailable"""
    
    np.random.seed(42)
    
    data = []
    
    for i in range(n_timestamps):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        
        for coin_idx in range(n_coins):
            coin = f"COIN_{coin_idx:03d}"
            
            # Generate correlated features and targets
            base_momentum = np.random.normal(0, 0.02)
            volatility = np.random.lognormal(-3, 0.5)  # ~0.05 typical
            volume = np.random.lognormal(10, 0.8)
            
            # Technical indicators
            rsi = np.clip(np.random.normal(50, 15), 10, 90)
            macd = np.random.normal(0, 0.01)
            
            # Sentiment/whale scores (0-1)
            sent_score = np.clip(np.random.beta(2, 2), 0.1, 0.9)
            whale_score = np.clip(np.random.beta(1.5, 3), 0.05, 0.95)
            
            # Targets with realistic correlation to features
            target_1h = base_momentum * 0.1 + np.random.normal(0, volatility * 0.5)
            target_24h = base_momentum * 0.5 + np.random.normal(0, volatility * 1.0)
            target_168h = base_momentum * 2.0 + np.random.normal(0, volatility * 2.0)
            target_720h = base_momentum * 5.0 + np.random.normal(0, volatility * 3.0)
            
            data.append({
                "coin": coin,
                "timestamp": timestamp,
                "feat_momentum": base_momentum,
                "feat_volatility": volatility,
                "feat_volume": volume,
                "feat_rsi_14": rsi,
                "feat_macd": macd,
                "feat_sent_score": sent_score,
                "feat_whale_score": whale_score,
                "target_1h": target_1h,
                "target_24h": target_24h,
                "target_168h": target_168h,
                "target_720h": target_720h
            })
    
    df = pd.DataFrame(data)
    
    # Add UTC timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    
    return df

def train_one(df, target_col, feature_cols, n_models=5):
    """Train ensemble of RandomForest models for one horizon"""
    
    print(f"Training {target_col} with {len(feature_cols)} features on {len(df)} samples")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Use TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    
    for seed in range(n_models):
        rf = RandomForestRegressor(
            n_estimators=200,  # Reduced for speed
            random_state=seed,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Train on all but last fold (most recent data for validation)
        train_indices, _ = list(tscv.split(X))[-1]
        X_train, y_train = X[train_indices], y[train_indices]
        
        rf.fit(X_train, y_train)
        models.append(rf)
        
        print(f"  Model {seed+1}/{n_models} trained")
    
    return models

def validate_data(df):
    """Validate data meets training requirements"""
    
    print("Validating training data...")
    
    # Check timestamps are UTC
    if df['timestamp'].dt.tz is None:
        print("  WARNING: Timestamps have no timezone, assuming UTC")
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Check for required columns
    required_features = ["feat_sent_score", "feat_whale_score", "feat_rsi_14"]
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"  WARNING: Missing features {missing_features}, using synthetic data")
    
    # Check target scales
    for horizon, target_col in HORIZONS.items():
        if target_col in df.columns:
            q99 = df[target_col].abs().quantile(0.99)
            if q99 > 3.0:
                print(f"  WARNING: {horizon} target scale suspicious (q99={q99:.3f})")
    
    # Check for NaN values
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.any():
        print(f"  WARNING: NaN values in features: {nan_counts[nan_counts > 0].to_dict()}")
    
    print("  Data validation complete")
    return df

def train_all_models():
    """Train models for all horizons"""
    
    print("=" * 60)
    print("TRAINING BASELINE MODELS")
    print("=" * 60)
    
    # Load or create features
    features_file = Path("exports/features.parquet")
    
    if features_file.exists():
        print(f"Loading features from {features_file}")
        df = pd.read_parquet(features_file)
    else:
        print("No features file found, creating synthetic data")
        df = create_synthetic_features(n_coins=20, n_timestamps=1000)
        
        # Save synthetic features
        features_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(features_file)
        print(f"Saved synthetic features to {features_file}")
    
    # Validate data
    df = validate_data(df)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Create models directory
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train each horizon
    training_results = {}
    
    for horizon, target_col in HORIZONS.items():
        print(f"\n🎯 Training {horizon} models...")
        
        if target_col not in df.columns:
            print(f"  ❌ Target {target_col} not found, skipping")
            continue
        
        # Filter complete cases
        training_data = df.dropna(subset=feature_cols + [target_col])
        
        if len(training_data) < 100:
            print(f"  ❌ Insufficient data for {horizon}: {len(training_data)} samples")
            continue
        
        # Train ensemble
        ensemble = train_one(training_data, target_col, feature_cols, n_models=3)
        
        # Save models
        model_path = models_dir / f"rf_{horizon}.pkl"
        joblib.dump(ensemble, model_path)
        
        # Calculate training stats
        X_test = training_data[feature_cols].values
        y_test = training_data[target_col].values
        
        # Ensemble predictions
        preds = np.column_stack([m.predict(X_test) for m in ensemble])
        mu = preds.mean(axis=1)
        sigma = preds.std(axis=1)
        
        mae = np.mean(np.abs(mu - y_test))
        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        mean_uncertainty = sigma.mean()
        
        training_results[horizon] = {
            "model_path": str(model_path),
            "ensemble_size": len(ensemble),
            "training_samples": len(training_data),
            "mae": mae,
            "rmse": rmse,
            "mean_uncertainty": mean_uncertainty
        }
        
        print(f"  ✅ {horizon} models saved: MAE={mae:.4f}, RMSE={rmse:.4f}, σ={mean_uncertainty:.4f}")
    
    # Save training summary
    summary = {
        "training_timestamp": datetime.now().isoformat(),
        "total_features": len(feature_cols),
        "feature_columns": feature_cols,
        "horizons_trained": list(training_results.keys()),
        "results": training_results
    }
    
    summary_path = models_dir / "training_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ All models trained successfully!")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Training summary: {summary_path}")
    print(f"🎯 Horizons trained: {list(training_results.keys())}")
    
    return training_results

if __name__ == "__main__":
    results = train_all_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    for horizon, stats in results.items():
        print(f"{horizon:>6s}: {stats['ensemble_size']} models, "
              f"MAE={stats['mae']:.4f}, "
              f"σ={stats['mean_uncertainty']:.4f}")
    
    print(f"\nNext steps:")
    print(f"1. Run: python ml/models/predict.py (test predictions)")
    print(f"2. Run: python generate_final_predictions.py (create predictions.csv)")
    print(f"3. Check dashboard for live predictions")