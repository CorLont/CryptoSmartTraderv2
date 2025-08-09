# ml/models/predict.py
import pandas as pd
import numpy as np
import joblib
import glob
from pathlib import Path

HORIZONS = ["1h","24h","168h","720h"]

def load_models():
    """Load trained RandomForest ensemble models for all horizons"""
    models = {}
    for h in HORIZONS:
        model_path = f"models/saved/rf_{h}.pkl"
        if Path(model_path).exists():
            try:
                models[h] = joblib.load(model_path)
                print(f"Loaded {len(models[h])} models for horizon {h}")
            except Exception as e:
                print(f"Failed to load models for {h}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models

def predict_all(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for all horizons with uncertainty estimates
    
    Returns DataFrame with columns:
    - coin, timestamp (from input)
    - pred_{horizon}: mean prediction across ensemble
    - conf_{horizon}: confidence = 1/(1+std), higher = more certain
    """
    if features_df.empty:
        return pd.DataFrame()
    
    # Load models
    models = load_models()
    
    if not models:
        print("Warning: No models loaded - returning empty predictions")
        return pd.DataFrame()
    
    # Start with coin and timestamp
    out = features_df[["coin","timestamp"]].copy()
    
    # Get feature columns
    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
    
    if not feature_cols:
        print("Warning: No feature columns found (expected feat_*)")
        return out
    
    print(f"Making predictions using {len(feature_cols)} features for {len(features_df)} samples")
    
    # Prepare feature matrix
    X = features_df[feature_cols].values
    
    # Handle missing values
    if np.isnan(X).any():
        print("Warning: NaN values found in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Generate predictions for each horizon
    for h, ensemble in models.items():
        try:
            # Get predictions from all ensemble members
            ensemble_preds = []
            for model in ensemble:
                pred = model.predict(X)
                ensemble_preds.append(pred)
            
            # Stack predictions (n_samples x n_models)
            preds = np.column_stack(ensemble_preds)
            
            # Calculate ensemble statistics
            mu = preds.mean(axis=1)  # Mean prediction
            sigma = preds.std(axis=1) + 1e-9  # Standard deviation + small epsilon
            
            # Store predictions and confidence
            out[f"pred_{h}"] = mu
            out[f"conf_{h}"] = 1.0 / (1.0 + sigma)  # 0..1, higher = more certain
            
            print(f"Generated predictions for {h}: mean={mu.mean():.4f}, conf={out[f'conf_{h}'].mean():.3f}")
            
        except Exception as e:
            print(f"Failed to generate predictions for horizon {h}: {e}")
            # Fill with default values
            out[f"pred_{h}"] = 0.0
            out[f"conf_{h}"] = 0.5
    
    return out

def predict_single_horizon(features_df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Generate predictions for a single horizon"""
    
    if horizon not in HORIZONS:
        raise ValueError(f"Invalid horizon {horizon}, must be one of {HORIZONS}")
    
    model_path = f"models/saved/rf_{horizon}.pkl"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return pd.DataFrame()
    
    try:
        ensemble = joblib.load(model_path)
        
        # Prepare features
        feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Ensemble predictions
        preds = np.column_stack([model.predict(X) for model in ensemble])
        mu = preds.mean(axis=1)
        sigma = preds.std(axis=1) + 1e-9
        
        # Create result
        result = features_df[["coin", "timestamp"]].copy()
        result[f"pred_{horizon}"] = mu
        result[f"conf_{horizon}"] = 1.0 / (1.0 + sigma)
        
        return result
        
    except Exception as e:
        print(f"Prediction failed for {horizon}: {e}")
        return pd.DataFrame()

def get_model_status():
    """Get status of all trained models"""
    status = {}
    
    for h in HORIZONS:
        model_path = Path(f"models/saved/rf_{h}.pkl")
        
        if model_path.exists():
            try:
                # Get file info
                stat = model_path.stat()
                
                # Try to load model info
                models = joblib.load(model_path)
                
                status[h] = {
                    "exists": True,
                    "path": str(model_path),
                    "size_mb": stat.st_size / 1024 / 1024,
                    "ensemble_size": len(models) if isinstance(models, list) else 1,
                    "modified": stat.st_mtime
                }
            except Exception as e:
                status[h] = {
                    "exists": True,
                    "error": str(e)
                }
        else:
            status[h] = {
                "exists": False,
                "path": str(model_path)
            }
    
    return status

if __name__ == "__main__":
    # Test predictions with sample data
    print("Testing prediction system...")
    
    # Check model status
    status = get_model_status()
    print("\nModel Status:")
    for h, info in status.items():
        if info.get("exists"):
            if "error" in info:
                print(f"  {h}: ERROR - {info['error']}")
            else:
                print(f"  {h}: OK - {info['ensemble_size']} models, {info['size_mb']:.1f}MB")
        else:
            print(f"  {h}: MISSING - {info['path']}")
    
    # Test with sample features if available
    features_file = Path("exports/features.parquet")
    if features_file.exists():
        print(f"\nTesting with features from {features_file}")
        
        # Load sample data
        features = pd.read_parquet(features_file)
        sample = features.head(10)
        
        print(f"Sample features shape: {sample.shape}")
        print(f"Feature columns: {[c for c in sample.columns if c.startswith('feat_')]}")
        
        # Generate predictions
        predictions = predict_all(sample)
        
        print(f"\nPrediction results:")
        print(f"Shape: {predictions.shape}")
        print(f"Columns: {list(predictions.columns)}")
        
        # Show sample predictions
        pred_cols = [c for c in predictions.columns if c.startswith(('pred_', 'conf_'))]
        if pred_cols:
            print(f"\nSample predictions:")
            print(predictions[['coin'] + pred_cols].head())
    else:
        print(f"\nNo features file found at {features_file}")
        print("Run ml/train_baseline.py first to create models and test data")