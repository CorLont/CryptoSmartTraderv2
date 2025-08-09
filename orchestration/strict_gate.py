#!/usr/bin/env python3
"""
Strict Backend Gate - Hard enforcement van 80% confidence threshold
Geen UI-cosmetic filtering, alleen echte backend gates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

def apply_strict_gate_orchestration(predictions_by_horizon: Dict[str, pd.DataFrame], thr: float = 0.80) -> Dict[str, Any]:
    """
    Apply strict confidence gate across all horizons
    
    Args:
        predictions_by_horizon: {horizon: df with coin, timestamp, pred_{horizon}, conf_{horizon}}
        thr: confidence threshold (default 0.80)
    
    Returns:
        Gate results with filtered predictions per horizon
    """
    
    print(f"🚪 Applying strict {thr:.0%} confidence gate...")
    
    gate_results = {
        "gate_status": "EMPTY",
        "total_candidates": 0,
        "total_passed": 0,
        "per_horizon": {},
        "threshold": thr,
        "applied_at": datetime.now().isoformat()
    }
    
    for horizon, df in predictions_by_horizon.items():
        if df.empty:
            gate_results["per_horizon"][horizon] = pd.DataFrame()
            continue
        
        # Expected column names
        pred_col = f"pred_{horizon}"
        conf_col = f"conf_{horizon}"
        
        # Validate required columns
        required_cols = ["coin", "timestamp", pred_col, conf_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ {horizon}: Missing columns {missing_cols}")
            gate_results["per_horizon"][horizon] = pd.DataFrame()
            continue
        
        # Clean data (remove NaN)
        clean_df = df[required_cols].dropna()
        candidates = len(clean_df)
        
        if candidates == 0:
            print(f"  ❌ {horizon}: No valid candidates after cleaning")
            gate_results["per_horizon"][horizon] = pd.DataFrame()
            continue
        
        # Apply strict confidence filter
        passed_df = clean_df[clean_df[conf_col] >= thr].copy()
        passed_count = len(passed_df)
        
        # Sort by confidence descending
        if not passed_df.empty:
            passed_df = passed_df.sort_values(conf_col, ascending=False)
        
        gate_results["total_candidates"] += candidates
        gate_results["total_passed"] += passed_count
        gate_results["per_horizon"][horizon] = passed_df
        
        pass_rate = passed_count / candidates if candidates > 0 else 0
        print(f"  {horizon}: {passed_count}/{candidates} passed ({pass_rate:.1%})")
    
    # Overall gate status
    if gate_results["total_passed"] > 0:
        gate_results["gate_status"] = "OK"
    else:
        gate_results["gate_status"] = "BLOCKED"
    
    overall_pass_rate = (gate_results["total_passed"] / max(gate_results["total_candidates"], 1)) * 100
    print(f"🚪 Gate result: {gate_results['gate_status']} - {gate_results['total_passed']}/{gate_results['total_candidates']} passed ({overall_pass_rate:.1f}%)")
    
    return gate_results

def strict_filter_single(pred_df: pd.DataFrame, pred_col: str = "pred_720h", conf_col: str = "conf_720h", thr: float = 0.80) -> pd.DataFrame:
    """
    Single horizon strict filter (compatibility function)
    
    Args:
        pred_df: DataFrame with predictions
        pred_col: prediction column name
        conf_col: confidence column name  
        thr: confidence threshold
    
    Returns:
        Filtered DataFrame
    """
    
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    
    # Clean and filter
    clean_df = pred_df.dropna(subset=[pred_col, conf_col])
    filtered_df = clean_df[clean_df[conf_col] >= thr].copy()
    
    # Sort by prediction value (highest first)
    if not filtered_df.empty:
        filtered_df = filtered_df.sort_values(pred_col, ascending=False).reset_index(drop=True)
    
    return filtered_df

def validate_confidence_calibration(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that confidence scores are properly calibrated
    
    Args:
        predictions_df: DataFrame with conf_{horizon} columns
        actuals_df: DataFrame with realized_{horizon} (binary: 1=correct, 0=incorrect)
    
    Returns:
        Calibration analysis
    """
    
    calibration_results = {}
    horizons = ["1h", "24h", "168h", "720h"]
    
    for horizon in horizons:
        conf_col = f"conf_{horizon}"
        real_col = f"realized_{horizon}"
        
        if conf_col not in predictions_df.columns or real_col not in actuals_df.columns:
            continue
        
        # Merge data
        merged = pd.merge(predictions_df, actuals_df, on=["coin", "timestamp"], how="inner")
        
        if len(merged) < 50:  # Need sufficient data
            continue
        
        # Bin confidence scores
        merged["conf_bin"] = pd.cut(merged[conf_col], bins=[0.0, 0.8, 0.9, 0.95, 1.0], 
                                  labels=["<80%", "80-90%", "90-95%", "95%+"])
        
        # Calculate hit rates per bin
        hit_rates = merged.groupby("conf_bin")[real_col].agg(['mean', 'count']).round(3)
        
        # Expected: 80-90% bin should have ~70-85% hit rate
        # 90-95% bin should have ~85-95% hit rate, etc.
        
        calibration_results[horizon] = {
            "sample_size": len(merged),
            "hit_rates": hit_rates.to_dict(),
            "overall_accuracy": merged[real_col].mean()
        }
    
    return calibration_results

def create_empty_state_message(gate_report: Dict[str, Any]) -> Dict[str, Any]:
    """Create informative empty state message for UI"""
    
    total_candidates = gate_report.get("total_candidates", 0)
    threshold = gate_report.get("threshold", 0.80)
    
    message = {
        "title": f"Geen signalen ≥{threshold:.0%} confidence",
        "subtitle": f"Van {total_candidates} kandidaten voldeed geen enkele aan de strikte 80% gate",
        "recommendations": [
            "Wacht op nieuwe marktdata (update elke 4 uur)",
            "Controleer of modellen recent getraind zijn",
            "Overweeg lagere confidence drempel voor testing (niet aanbevolen voor live trading)"
        ],
        "technical_details": {
            "total_evaluated": total_candidates,
            "confidence_threshold": threshold,
            "gate_status": gate_report.get("gate_status", "UNKNOWN"),
            "per_horizon_stats": {}
        }
    }
    
    # Add per-horizon breakdown
    for horizon, horizon_data in gate_report.get("per_horizon", {}).items():
        if isinstance(horizon_data, pd.DataFrame):
            message["technical_details"]["per_horizon_stats"][horizon] = {
                "candidates": len(horizon_data) if hasattr(horizon_data, '__len__') else 0,
                "passed": 0
            }
    
    return message

if __name__ == "__main__":
    print("Testing Strict Confidence Gate")
    print("=" * 40)
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    
    test_data = {
        "1h": pd.DataFrame({
            "coin": [f"COIN_{i:03d}" for i in range(n_samples)],
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
            "pred_1h": np.random.normal(0, 0.02, n_samples),
            "conf_1h": np.random.beta(8, 2, n_samples)  # Skewed toward high confidence
        }),
        "24h": pd.DataFrame({
            "coin": [f"COIN_{i:03d}" for i in range(n_samples)],
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
            "pred_24h": np.random.normal(0, 0.05, n_samples),
            "conf_24h": np.random.beta(5, 3, n_samples)  # More spread
        })
    }
    
    print("Test data created:")
    for horizon, df in test_data.items():
        high_conf = (df[f"conf_{horizon}"] >= 0.80).sum()
        print(f"  {horizon}: {len(df)} samples, {high_conf} with ≥80% confidence")
    
    # Test strict gate
    print("\nApplying strict 80% gate...")
    gate_results = apply_strict_gate_orchestration(test_data, thr=0.80)
    
    print(f"\nGate Results:")
    print(f"  Status: {gate_results['gate_status']}")
    print(f"  Total passed: {gate_results['total_passed']}/{gate_results['total_candidates']}")
    
    for horizon, passed_df in gate_results["per_horizon"].items():
        if not passed_df.empty:
            mean_conf = passed_df[f"conf_{horizon}"].mean()
            mean_pred = passed_df[f"pred_{horizon}"].mean()
            print(f"  {horizon}: {len(passed_df)} passed, μ_conf={mean_conf:.3f}, μ_pred={mean_pred:+.4f}")
    
    # Test empty state message
    if gate_results["gate_status"] == "BLOCKED":
        empty_msg = create_empty_state_message(gate_results)
        print(f"\nEmpty State Message:")
        print(f"  Title: {empty_msg['title']}")
        print(f"  Subtitle: {empty_msg['subtitle']}")
    
    print("\n✅ Strict gate test complete!")