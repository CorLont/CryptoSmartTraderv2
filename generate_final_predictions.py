#!/usr/bin/env python3
"""
Final Predictions Generator - Standalone script that bypasses all imports issues
Creates predictions.csv with pred_* + conf_* columns using trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone
import json

def load_all_models():
    """Load all trained RandomForest models"""
    
    models = {}
    horizons = ["1h", "24h", "168h", "720h"]
    
    for horizon in horizons:
        model_path = Path(f"models/saved/rf_{horizon}.pkl")
        
        if model_path.exists():
            try:
                ensemble = joblib.load(model_path)
                models[horizon] = ensemble
                print(f"✅ Loaded {horizon} model ({len(ensemble)} ensemble members)")
            except Exception as e:
                print(f"❌ Failed to load {horizon} model: {e}")
        else:
            print(f"❌ Model not found: {model_path}")
    
    return models

def generate_predictions_with_confidence(features_df, models):
    """Generate predictions with uncertainty-based confidence scores"""
    
    if features_df.empty or not models:
        return pd.DataFrame()
    
    # Get feature columns
    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]
    
    if not feature_cols:
        print("❌ No feature columns found")
        return pd.DataFrame()
    
    # Start with coin and timestamp
    result = features_df[["coin", "timestamp"]].copy()
    
    # Prepare feature matrix
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)  # Handle NaN values
    
    print(f"🔮 Generating predictions for {len(X)} samples using {len(feature_cols)} features")
    
    # Generate predictions for each horizon
    for horizon, ensemble in models.items():
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
            sigma = preds.std(axis=1) + 1e-9  # Standard deviation + epsilon
            
            # Store predictions and confidence
            result[f"pred_{horizon}"] = mu
            result[f"conf_{horizon}"] = 1.0 / (1.0 + sigma)  # 0..1, higher = more certain
            
            print(f"   {horizon}: mean={mu.mean():.4f}, conf={result[f'conf_{horizon}'].mean():.3f}")
            
        except Exception as e:
            print(f"❌ Failed to generate predictions for {horizon}: {e}")
    
    return result

def apply_strict_confidence_gate(predictions_df, threshold=0.80):
    """Apply strict 80% confidence gate and return filtered results"""
    
    horizons = ["1h", "24h", "168h", "720h"]
    gate_results = {
        "total_candidates": 0,
        "total_passed": 0,
        "per_horizon": {},
        "gate_status": "EMPTY"
    }
    
    all_passed = []
    
    for horizon in horizons:
        pred_col = f"pred_{horizon}"
        conf_col = f"conf_{horizon}"
        
        if pred_col not in predictions_df.columns or conf_col not in predictions_df.columns:
            continue
        
        # Filter horizon data
        horizon_data = predictions_df[["coin", "timestamp", pred_col, conf_col]].copy()
        horizon_data = horizon_data.dropna()
        
        total_candidates = len(horizon_data)
        
        # Apply confidence threshold
        passed_data = horizon_data[horizon_data[conf_col] >= threshold].copy()
        passed_count = len(passed_data)
        
        gate_results["total_candidates"] += total_candidates
        gate_results["total_passed"] += passed_count
        gate_results["per_horizon"][horizon] = {
            "candidates": total_candidates,
            "passed": passed_count,
            "pass_rate": passed_count / max(total_candidates, 1)
        }
        
        if not passed_data.empty:
            # Add horizon identifier and sort by prediction confidence
            passed_data["horizon"] = horizon
            passed_data["signal_strength"] = abs(passed_data[pred_col]) * passed_data[conf_col]
            passed_data = passed_data.sort_values("signal_strength", ascending=False)
            all_passed.append(passed_data)
    
    # Combine all passed predictions
    if all_passed:
        filtered_df = pd.concat(all_passed, ignore_index=True)
        gate_results["gate_status"] = "OK"
    else:
        filtered_df = pd.DataFrame()
    
    return filtered_df, gate_results

def create_final_predictions_csv():
    """Create final predictions.csv with all required columns"""
    
    print("🚀 GENERATING FINAL PREDICTIONS.CSV")
    print("=" * 50)
    
    # Step 1: Load models
    print("1️⃣ Loading trained models...")
    models = load_all_models()
    
    if not models:
        print("❌ No trained models found")
        return False
    
    # Step 2: Load features
    print("2️⃣ Loading features...")
    features_file = Path("exports/features.parquet")
    
    if not features_file.exists():
        print("❌ Features file not found")
        return False
    
    features_df = pd.read_parquet(features_file)
    
    # Take latest 200 samples for prediction
    latest_features = features_df.tail(200).copy()
    print(f"   Using latest {len(latest_features)} samples for prediction")
    
    # Step 3: Generate predictions
    print("3️⃣ Generating predictions with confidence scores...")
    predictions_df = generate_predictions_with_confidence(latest_features, models)
    
    if predictions_df.empty:
        print("❌ No predictions generated")
        return False
    
    # Step 4: Apply confidence gate
    print("4️⃣ Applying 80% confidence gate...")
    filtered_df, gate_stats = apply_strict_confidence_gate(predictions_df, threshold=0.80)
    
    print(f"   Gate status: {gate_stats['gate_status']}")
    print(f"   Candidates: {gate_stats['total_candidates']}")
    print(f"   Passed gate: {gate_stats['total_passed']}")
    
    if filtered_df.empty:
        print("❌ No predictions passed the 80% confidence gate")
        
        # Show confidence distribution
        print("\n📊 Confidence distribution:")
        for horizon in ["1h", "24h", "168h", "720h"]:
            conf_col = f"conf_{horizon}"
            if conf_col in predictions_df.columns:
                conf_values = predictions_df[conf_col]
                high_conf = (conf_values >= 0.8).sum()
                total = len(conf_values)
                print(f"   {horizon}: {high_conf}/{total} ({high_conf/total*100:.1f}%) ≥80% confidence")
        
        return False
    
    # Step 5: Enhance predictions with additional columns
    print("5️⃣ Creating enhanced predictions...")
    
    enhanced_predictions = []
    
    for _, row in filtered_df.iterrows():
        horizon = row["horizon"]
        pred_col = f"pred_{horizon}"
        conf_col = f"conf_{horizon}"
        
        prediction = row[pred_col]
        confidence = row[conf_col]
        
        # Calculate derived metrics
        expected_return_pct = prediction * 100
        risk_score = 1 - confidence
        
        # Simple regime classification
        if abs(prediction) > 0.05:
            regime = "VOLATILE"
        elif prediction > 0.02:
            regime = "BULL"
        elif prediction < -0.02:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"
        
        # Signal quality
        signal_strength = abs(prediction) * confidence
        actionable = confidence >= 0.8 and abs(prediction) > 0.01
        
        enhanced_predictions.append({
            "coin": row["coin"],
            "timestamp": row["timestamp"],
            "horizon": horizon,
            f"pred_{horizon}": prediction,
            f"conf_{horizon}": confidence,
            "expected_return_pct": expected_return_pct,
            "risk_score": risk_score,
            "regime": regime,
            "signal_strength": signal_strength,
            "actionable": actionable,
            "confidence_grade": "HIGH" if confidence >= 0.9 else "MEDIUM" if confidence >= 0.8 else "LOW"
        })
    
    final_df = pd.DataFrame(enhanced_predictions)
    
    # Step 6: Save predictions.csv
    print("6️⃣ Saving predictions.csv...")
    
    output_dir = Path("exports/production")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = output_dir / "predictions.csv"
    final_df.to_csv(predictions_file, index=False)
    
    print(f"   ✅ Saved {len(final_df)} predictions to {predictions_file}")
    
    # Step 7: Create summary
    print("7️⃣ Creating summary...")
    
    summary = {
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "available_models": list(models.keys()),
            "total_ensemble_members": sum(len(ensemble) for ensemble in models.values())
        },
        "prediction_stats": {
            "total_predictions": len(final_df),
            "unique_coins": final_df["coin"].nunique(),
            "horizons": final_df["horizon"].unique().tolist(),
            "actionable_signals": int(final_df["actionable"].sum())
        },
        "confidence_stats": {
            "mean_confidence": float(final_df[f"conf_{final_df.iloc[0]['horizon']}"].mean()),
            "min_confidence": float(final_df[f"conf_{final_df.iloc[0]['horizon']}"].min()),
            "max_confidence": float(final_df[f"conf_{final_df.iloc[0]['horizon']}"].max())
        },
        "return_stats": {
            "mean_expected_return": float(final_df["expected_return_pct"].mean()),
            "positive_predictions": int((final_df["expected_return_pct"] > 0).sum()),
            "negative_predictions": int((final_df["expected_return_pct"] < 0).sum())
        },
        "regime_distribution": final_df["regime"].value_counts().to_dict(),
        "gate_statistics": gate_stats
    }
    
    summary_file = output_dir / "predictions_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Saved summary to {summary_file}")
    
    # Step 8: Display results
    print("\n📊 PREDICTION RESULTS:")
    print(f"   📈 Total predictions: {summary['prediction_stats']['total_predictions']}")
    print(f"   🪙 Unique coins: {summary['prediction_stats']['unique_coins']}")
    print(f"   ⏰ Horizons: {', '.join(summary['prediction_stats']['horizons'])}")
    print(f"   🎯 Actionable signals: {summary['prediction_stats']['actionable_signals']}")
    print(f"   📊 Mean expected return: {summary['return_stats']['mean_expected_return']:.2f}%")
    
    # Show regime distribution
    print(f"\n🌍 REGIME DISTRIBUTION:")
    for regime, count in summary["regime_distribution"].items():
        percentage = count / summary["prediction_stats"]["total_predictions"] * 100
        print(f"   {regime}: {count} ({percentage:.1f}%)")
    
    # Show gate statistics
    print(f"\n🚪 CONFIDENCE GATE (80% threshold):")
    print(f"   Total candidates: {gate_stats['total_candidates']}")
    print(f"   Passed gate: {gate_stats['total_passed']}")
    print(f"   Pass rate: {gate_stats['total_passed']/max(gate_stats['total_candidates'], 1)*100:.1f}%")
    
    return True

def show_top_predictions():
    """Show top predictions by signal strength"""
    
    predictions_file = Path("exports/production/predictions.csv")
    
    if not predictions_file.exists():
        return
    
    df = pd.read_csv(predictions_file)
    
    print("\n🔝 TOP 10 PREDICTIONS (by signal strength):")
    print("-" * 80)
    
    # Sort by signal strength
    top_10 = df.nlargest(10, "signal_strength")
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        action = "🟢 BUY" if row["expected_return_pct"] > 0 else "🔴 SELL"
        conf_col = f"conf_{row['horizon']}"
        confidence_value = row[conf_col] if conf_col in row else 0.8
        print(f"{i:2d}. {action} {row['coin']} ({row['horizon']}) | "
              f"Return: {row['expected_return_pct']:+.2f}% | "
              f"Confidence: {confidence_value:.1%} | "
              f"Risk: {row['risk_score']:.2f}")

if __name__ == "__main__":
    print("CryptoSmartTrader V2 - Final Predictions Generator")
    print("=" * 60)
    
    success = create_final_predictions_csv()
    
    if success:
        print("\n🎉 SUCCESS: predictions.csv generated successfully!")
        show_top_predictions()
        print(f"\n📁 Output files:")
        print(f"   • exports/production/predictions.csv")
        print(f"   • exports/production/predictions_summary.json")
    else:
        print("\n❌ FAILED: Could not generate predictions.csv")
    
    print("\n" + "=" * 60)