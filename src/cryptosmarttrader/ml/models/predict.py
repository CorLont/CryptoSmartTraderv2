#!/usr/bin/env python3
"""
Predictor + Confidence - Load ensembles and generate predictions with uncertainty
"""

import pandas as pd
import numpy as np
import joblib
import glob
from pathlib import Path
import time
from typing import Dict, List, Optional

HORIZONS = ["1h", "24h", "168h", "720h"]

def get_model_status() -> Dict[str, Dict]:
    """Get status of all trained models"""
    
    status = {}
    
    for horizon in HORIZONS:
        model_path = Path(f"models/saved/rf_{horizon}.pkl")
        
        if model_path.exists():
            stat = model_path.stat()
            status[horizon] = {
                "exists": True,
                "path": str(model_path),
                "size_mb": stat.st_size / 1024 / 1024,
                "modified": stat.st_mtime,
                "age_hours": (time.time() - stat.st_mtime) / 3600
            }
        else:
            status[horizon] = {
                "exists": False,
                "path": str(model_path),
                "size_mb": 0,
                "modified": 0,
                "age_hours": float('inf')
            }
    
    return status

def predict_all(features: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for all horizons using trained ensemble models"""
    # Start with basic info
    out = features[["coin", "timestamp"]].copy() if "timestamp" in features.columns else features[["coin"]].copy()
    
    # Get feature columns
    feature_cols = [c for c in features.columns if c.startswith("feat_")]
    
    if not feature_cols:
        return out
    
    X = features[feature_cols].values
    
    for horizon in HORIZONS:
        model_path = f"models/saved/rf_{horizon}.pkl"
        
        if not Path(model_path).exists():
            continue
        
        try:
            # Load ensemble models
            ensemble = joblib.load(model_path)
            
            # Get predictions from all models in ensemble
            predictions = np.column_stack([model.predict(X) for model in ensemble])
            
            # Calculate mean and uncertainty
            mu = predictions.mean(axis=1)
            sigma = predictions.std(axis=1) + 1e-9
            
            # Convert uncertainty to confidence
            confidence = 1.0 / (1.0 + sigma)
            
            # Store results
            out[f"pred_{horizon}"] = mu
            out[f"conf_{horizon}"] = confidence
            
        except Exception as e:
            continue
    
    return out

def load_models() -> Dict[str, List]:
    """Load all available trained models"""
    
    models = {}
    
    for horizon in HORIZONS:
        model_path = f"models/saved/rf_{horizon}.pkl"
        
        if Path(model_path).exists():
            try:
                ensemble = joblib.load(model_path)
                models[horizon] = ensemble
                print(f"✅ Loaded {horizon} ensemble: {len(ensemble)} models")
            except Exception as e:
                print(f"❌ Failed to load {horizon} models: {e}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    return models

def predict_all(features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions with confidence for all horizons"""
    
    if features_df.empty:
        print("❌ Empty features DataFrame")
        return pd.DataFrame()
    
    print(f"🔮 Generating predictions for {len(features_df)} samples")
    
    # Load models
    models = load_models()
    
    if not models:
        print("❌ No trained models available")
        return pd.DataFrame()
    
    # Start with base columns
    result = features_df[["coin", "timestamp"]].copy()
    
    # Get feature columns
    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
    
    if not feature_cols:
        print("❌ No feature columns found")
        return pd.DataFrame()
    
    # Prepare feature matrix
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)  # Replace NaN with 0
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Generate predictions for each horizon
    for horizon, ensemble in models.items():
        try:
            print(f"  Predicting {horizon}...")
            
            # Get predictions from all ensemble members
            ensemble_predictions = []
            for i, model in enumerate(ensemble):
                pred = model.predict(X)
                ensemble_predictions.append(pred)
            
            # Stack predictions (samples x models)
            preds = np.column_stack(ensemble_predictions)
            
            # Calculate ensemble statistics
            mu = preds.mean(axis=1)  # Mean prediction
            sigma = preds.std(axis=1) + 1e-9  # Standard deviation + small epsilon
            
            # Convert uncertainty to confidence (0-1, higher = more confident)
            confidence = 1.0 / (1.0 + sigma)
            
            # Store results
            result[f"pred_{horizon}"] = mu
            result[f"conf_{horizon}"] = confidence
            
            print(f"    {horizon}: μ={mu.mean():.4f}, σ={sigma.mean():.4f}, conf={confidence.mean():.3f}")
            
        except Exception as e:
            print(f"❌ Failed to predict {horizon}: {e}")
    
    print(f"✅ Generated predictions for {len(result)} samples")
    return result

def test_predictions():
    """Test prediction pipeline with sample data"""
    
    print("🧪 Testing prediction pipeline...")
    
    # Load features
    features_file = Path("exports/features.parquet")
    
    if not features_file.exists():
        print("❌ No features file found for testing")
        return False
    
    features_df = pd.read_parquet(features_file)
    print(f"Loaded {len(features_df)} feature samples")
    
    # Take sample for testing
    sample_size = min(50, len(features_df))
    sample_features = features_df.tail(sample_size).copy()
    
    # Generate predictions
    predictions_df = predict_all(sample_features)
    
    if predictions_df.empty:
        print("❌ No predictions generated")
        return False
    
    # Analyze predictions
    print("\n📊 Prediction Analysis:")
    
    pred_cols = [c for c in predictions_df.columns if c.startswith("pred_")]
    conf_cols = [c for c in predictions_df.columns if c.startswith("conf_")]
    
    for pred_col, conf_col in zip(pred_cols, conf_cols):
        horizon = pred_col.split("_")[1]
        
        pred_values = predictions_df[pred_col]
        conf_values = predictions_df[conf_col]
        
        print(f"  {horizon:>6s}: pred μ={pred_values.mean():+.4f} ± {pred_values.std():.4f}, "
              f"conf μ={conf_values.mean():.3f} [{conf_values.min():.3f}, {conf_values.max():.3f}]")
    
    # Check high confidence predictions
    print(f"\n🎯 High Confidence Predictions (>80%):")
    
    high_conf_count = 0
    for _, row in predictions_df.iterrows():
        for horizon in HORIZONS:
            conf_col = f"conf_{horizon}"
            pred_col = f"pred_{horizon}"
            
            if conf_col in row and pred_col in row:
                if row[conf_col] >= 0.80:
                    expected_return = row[pred_col] * 100
                    print(f"    {row['coin']} ({horizon}): {expected_return:+.2f}% (conf: {row[conf_col]:.1%})")
                    high_conf_count += 1
    
    print(f"Total high-confidence predictions: {high_conf_count}")
    
    # Save test predictions
    test_output = Path("exports/test_predictions.csv")
    predictions_df.to_csv(test_output, index=False)
    print(f"✅ Test predictions saved to: {test_output}")
    
    return True

def predict_latest():
    """Generate predictions for latest features and save"""
    
    print("🎯 Generating latest predictions...")
    
    features_file = Path("exports/features.parquet")
    
    if not features_file.exists():
        print("❌ No features file found")
        return False
    
    # Load features
    features_df = pd.read_parquet(features_file)
    
    # Take latest samples
    latest_features = features_df.tail(200).copy()
    print(f"Using latest {len(latest_features)} samples")
    
    # Generate predictions
    predictions_df = predict_all(latest_features)
    
    if predictions_df.empty:
        return False
    
    # Save predictions
    output_dir = Path("exports/production")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "latest_predictions.csv"
    predictions_df.to_csv(output_file, index=False)
    
    print(f"✅ Latest predictions saved to: {output_file}")
    return True

if __name__ == "__main__":
    print("ML Predictor - Testing and Validation")
    print("=" * 50)
    
    # Check model status
    print("1️⃣ Checking model status...")
    status = get_model_status()
    
    available_models = [h for h, s in status.items() if s["exists"]]
    missing_models = [h for h, s in status.items() if not s["exists"]]
    
    print(f"   Available models: {available_models}")
    print(f"   Missing models: {missing_models}")
    
    if not available_models:
        print("❌ No trained models found. Run: python ml/train_baseline.py")
        exit(1)
    
    # Test predictions
    print("\n2️⃣ Testing predictions...")
    if test_predictions():
        print("✅ Prediction test successful")
    else:
        print("❌ Prediction test failed")
        exit(1)
    
    # Generate latest predictions
    print("\n3️⃣ Generating latest predictions...")
    if predict_latest():
        print("✅ Latest predictions generated")
    else:
        print("❌ Latest prediction generation failed")
    
    print("\n🎉 Prediction pipeline test complete!")
    print("\nNext steps:")
    print("• Run: python generate_final_predictions.py")
    print("• Check dashboard for live predictions")
    print("• Verify predictions.csv in exports/production/")