#!/usr/bin/env python3
"""
Direct Predictions Generator - Bypass orchestrator and generate predictions.csv directly
Implements strict confidence gates and ensures meaningful outputs
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.models.predict import predict_all, get_model_status
from orchestration.strict_gate import apply_strict_gate_orchestration

def generate_production_predictions():
    """Generate production-ready predictions with 80% confidence gate"""
    
    print("🚀 GENERATING PRODUCTION PREDICTIONS")
    print("=" * 50)
    
    # Step 1: Validate models
    print("1️⃣ Validating models...")
    model_status = get_model_status()
    
    available_models = [h for h, status in model_status.items() if status.get("exists", False)]
    print(f"   Available models: {available_models}")
    
    if len(available_models) < 4:
        print(f"❌ Only {len(available_models)}/4 models available")
        return False
    
    # Step 2: Load features
    print("2️⃣ Loading features...")
    features_file = Path("exports/features.parquet")
    
    if not features_file.exists():
        print("❌ No features file found")
        return False
    
    features_df = pd.read_parquet(features_file)
    print(f"   Loaded {len(features_df)} feature samples")
    
    # Take latest 200 samples for prediction
    latest_features = features_df.tail(200).copy()
    
    # Step 3: Generate predictions
    print("3️⃣ Generating predictions...")
    predictions_df = predict_all(latest_features)
    
    if predictions_df.empty:
        print("❌ No predictions generated")
        return False
    
    print(f"   Generated predictions for {len(predictions_df)} samples")
    
    # Step 4: Apply strict 80% confidence gate
    print("4️⃣ Applying strict confidence gates...")
    
    # Organize predictions by horizon
    horizons = ["1h", "24h", "168h", "720h"]
    predictions_by_horizon = {}
    
    for horizon in horizons:
        pred_col = f"pred_{horizon}"
        conf_col = f"conf_{horizon}"
        
        if pred_col in predictions_df.columns and conf_col in predictions_df.columns:
            horizon_df = predictions_df[["coin", "timestamp", pred_col, conf_col]].copy()
            predictions_by_horizon[horizon] = horizon_df
    
    # Apply strict gates
    gate_results = apply_strict_gate_orchestration(predictions_by_horizon, thr=0.80)
    
    print(f"   Gate status: {gate_results['gate_status']}")
    print(f"   Total candidates: {gate_results['total_candidates']}")
    print(f"   Passed 80% gate: {gate_results['total_passed']}")
    print(f"   Pass rate: {gate_results['total_passed']/max(gate_results['total_candidates'], 1)*100:.1f}%")
    
    # Step 5: Create final predictions DataFrame
    print("5️⃣ Creating final predictions...")
    
    final_predictions = []
    
    for horizon, passed_df in gate_results["per_horizon"].items():
        if not passed_df.empty:
            pred_col = f"pred_{horizon}"
            conf_col = f"conf_{horizon}"
            
            for _, row in passed_df.iterrows():
                prediction = row[pred_col]
                confidence = row[conf_col]
                
                # Calculate expected return percentage
                expected_return = prediction * 100
                
                # Calculate risk score
                risk_score = 1 - confidence
                
                # Add regime classification (simplified)
                if abs(prediction) > 0.05:
                    regime = "VOLATILE"
                elif prediction > 0.02:
                    regime = "BULL"
                elif prediction < -0.02:
                    regime = "BEAR"
                else:
                    regime = "SIDEWAYS"
                
                final_predictions.append({
                    "coin": row["coin"],
                    "timestamp": row["timestamp"],
                    "horizon": horizon,
                    "prediction": prediction,
                    "confidence": confidence,
                    "expected_return_pct": expected_return,
                    "risk_score": risk_score,
                    "regime": regime,
                    "signal_strength": abs(prediction) * confidence,
                    "actionable": confidence >= 0.8 and abs(prediction) > 0.01
                })
    
    if not final_predictions:
        print("❌ No predictions passed the 80% confidence gate")
        
        # Show what was filtered out
        print("\n📊 Confidence distribution:")
        for horizon in horizons:
            conf_col = f"conf_{horizon}"
            if conf_col in predictions_df.columns:
                conf_values = predictions_df[conf_col]
                high_conf = (conf_values >= 0.8).sum()
                total = len(conf_values)
                print(f"   {horizon}: {high_conf}/{total} ({high_conf/total*100:.1f}%) ≥80% confidence")
        
        return False
    
    final_df = pd.DataFrame(final_predictions)
    
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
        "total_predictions": len(final_df),
        "unique_coins": final_df["coin"].nunique(),
        "horizons": final_df["horizon"].unique().tolist(),
        "confidence_stats": {
            "mean_confidence": final_df["confidence"].mean(),
            "min_confidence": final_df["confidence"].min(),
            "max_confidence": final_df["confidence"].max()
        },
        "return_stats": {
            "mean_expected_return": final_df["expected_return_pct"].mean(),
            "positive_predictions": (final_df["expected_return_pct"] > 0).sum(),
            "negative_predictions": (final_df["expected_return_pct"] < 0).sum()
        },
        "regime_distribution": final_df["regime"].value_counts().to_dict(),
        "actionable_signals": final_df["actionable"].sum(),
        "gate_statistics": {
            "total_candidates": gate_results["total_candidates"],
            "passed_gate": gate_results["total_passed"],
            "pass_rate_pct": gate_results["total_passed"]/max(gate_results["total_candidates"], 1)*100
        }
    }
    
    summary_file = output_dir / "predictions_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   ✅ Saved summary to {summary_file}")
    
    # Step 8: Display results
    print("\n📊 PREDICTION RESULTS:")
    print(f"   Total predictions: {summary['total_predictions']}")
    print(f"   Unique coins: {summary['unique_coins']}")
    print(f"   Horizons: {', '.join(summary['horizons'])}")
    print(f"   Mean confidence: {summary['confidence_stats']['mean_confidence']:.3f}")
    print(f"   Mean expected return: {summary['return_stats']['mean_expected_return']:.2f}%")
    print(f"   Actionable signals: {summary['actionable_signals']}")
    
    print(f"\n📈 REGIME DISTRIBUTION:")
    for regime, count in summary["regime_distribution"].items():
        percentage = count / summary["total_predictions"] * 100
        print(f"   {regime}: {count} ({percentage:.1f}%)")
    
    print(f"\n🚪 CONFIDENCE GATE STATISTICS:")
    print(f"   Candidates: {summary['gate_statistics']['total_candidates']}")
    print(f"   Passed 80% gate: {summary['gate_statistics']['passed_gate']}")
    print(f"   Pass rate: {summary['gate_statistics']['pass_rate_pct']:.1f}%")
    
    return True

def show_sample_predictions():
    """Show sample of generated predictions"""
    
    predictions_file = Path("exports/production/predictions.csv")
    
    if not predictions_file.exists():
        print("No predictions.csv file found")
        return
    
    df = pd.read_csv(predictions_file)
    
    print("\n📋 SAMPLE PREDICTIONS (Top 10 by signal strength):")
    print("-" * 80)
    
    # Sort by signal strength and show top 10
    top_predictions = df.nlargest(10, "signal_strength")
    
    for _, row in top_predictions.iterrows():
        action = "🟢 BUY" if row["expected_return_pct"] > 0 else "🔴 SELL"
        print(f"{action} {row['coin']} ({row['horizon']}) | "
              f"Return: {row['expected_return_pct']:+.2f}% | "
              f"Confidence: {row['confidence']:.1%} | "
              f"Risk: {row['risk_score']:.2f} | "
              f"Regime: {row['regime']}")

if __name__ == "__main__":
    print("CryptoSmartTrader V2 - Direct Predictions Generator")
    print("=" * 60)
    
    success = generate_production_predictions()
    
    if success:
        print("\n🎉 SUCCESS: Predictions generated successfully!")
        show_sample_predictions()
        print(f"\n📁 Check exports/production/ for:")
        print(f"   • predictions.csv (main output)")
        print(f"   • predictions_summary.json (statistics)")
    else:
        print("\n❌ FAILED: Could not generate predictions")
        print("Check model training and data availability")
    
    print("\n" + "=" * 60)