# ml/train_baseline.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

HORIZONS = {"1h": "target_1h", "24h":"target_24h", "168h":"target_168h", "720h":"target_720h"}

def train_one(df: pd.DataFrame, target_col: str, features: list[str], n_models=5):
    """Train ensemble of RandomForest models for one horizon"""
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    X, y = df[features].values, df[target_col].values
    
    for seed in range(n_models):
        m = RandomForestRegressor(
            n_estimators=400, 
            max_depth=None, 
            random_state=seed, 
            n_jobs=-1, 
            oob_score=False
        )
        # Use laatste split als validatie, de rest voor training
        train_idx, test_idx = list(tscv.split(X))[-1]
        m.fit(X[train_idx], y[train_idx])
        models.append(m)
    
    return models

def create_synthetic_features(n_coins=100, n_timestamps=1000):
    """Create synthetic feature data for testing when no real data exists"""
    print("Creating synthetic feature data for baseline training...")
    
    # Generate synthetic data
    np.random.seed(42)
    data = []
    
    coins = [f"COIN_{i:03d}" for i in range(n_coins)]
    
    for i in range(n_timestamps):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        
        for coin in coins:
            # Generate correlated features and targets
            base_trend = np.sin(i / 100) * 0.1  # Market cycle
            coin_factor = np.random.normal(0, 0.05)  # Coin-specific noise
            
            # Technical features
            feat_rsi = np.random.uniform(20, 80)
            feat_macd = np.random.normal(0, 0.02)
            feat_volume = np.random.lognormal(10, 1)
            feat_volatility = np.random.uniform(0.01, 0.1)
            feat_momentum = np.random.normal(base_trend, 0.03)
            
            # Additional features
            feat_sma_ratio = np.random.uniform(0.95, 1.05)
            feat_bb_position = np.random.uniform(0, 1)
            feat_obv = np.random.normal(0, 1)
            
            # Targets based on features (with noise)
            target_1h = feat_momentum * 0.3 + np.random.normal(0, 0.01)
            target_24h = (feat_momentum + feat_rsi/100 - 0.5) * 0.1 + np.random.normal(0, 0.02)
            target_168h = base_trend * 0.2 + coin_factor + np.random.normal(0, 0.03)
            target_720h = base_trend * 0.5 + coin_factor * 2 + np.random.normal(0, 0.05)
            
            data.append({
                'coin': coin,
                'timestamp': timestamp,
                'feat_rsi': feat_rsi,
                'feat_macd': feat_macd,
                'feat_volume': feat_volume,
                'feat_volatility': feat_volatility,
                'feat_momentum': feat_momentum,
                'feat_sma_ratio': feat_sma_ratio,
                'feat_bb_position': feat_bb_position,
                'feat_obv': feat_obv,
                'target_1h': target_1h,
                'target_24h': target_24h,
                'target_168h': target_168h,
                'target_720h': target_720h
            })
    
    df = pd.DataFrame(data)
    
    # Ensure exports directory exists
    Path("exports").mkdir(exist_ok=True)
    df.to_parquet("exports/features.parquet")
    print(f"Created synthetic features.parquet with {len(df)} rows")
    
    return df

if __name__ == "__main__":
    # Check if features exist, create synthetic if not
    features_file = Path("exports/features.parquet")
    
    if features_file.exists():
        print("Loading existing features.parquet...")
        feats = pd.read_parquet("exports/features.parquet")
    else:
        print("No features.parquet found - creating synthetic data...")
        feats = create_synthetic_features()
    
    # Get feature columns
    features = [c for c in feats.columns if c.startswith("feat_")]
    print(f"Found {len(features)} features: {features}")
    
    # Create models directory
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models for each horizon
    for h, tgt in HORIZONS.items():
        print(f"\nTraining models for horizon {h} (target: {tgt})")
        
        # Filter data with valid target
        train_data = feats.dropna(subset=features + [tgt])
        
        if len(train_data) < 100:
            print(f"Warning: Only {len(train_data)} samples for {h}, might be insufficient")
            continue
        
        # Train ensemble
        models = train_one(train_data, tgt, features)
        
        # Save models
        model_path = models_dir / f"rf_{h}.pkl"
        joblib.dump(models, model_path)
        
        print(f"✅ Saved {model_path} ({len(models)} ensemble members)")
        
        # Quick validation
        from ml.models.predict import predict_all
        test_sample = train_data.tail(10)
        try:
            predictions = predict_all(test_sample)
            pred_col = f"pred_{h}"
            conf_col = f"conf_{h}"
            
            if pred_col in predictions.columns and conf_col in predictions.columns:
                mean_pred = predictions[pred_col].mean()
                mean_conf = predictions[conf_col].mean()
                print(f"   Test predictions: mean={mean_pred:.4f}, conf={mean_conf:.3f}")
            else:
                print(f"   Warning: Prediction columns not found")
                
        except Exception as e:
            print(f"   Validation failed: {e}")
    
    print(f"\n🎯 BASELINE MODEL TRAINING COMPLETE!")
    print(f"   Models saved in: {models_dir}")
    print(f"   Features file: {features_file}")