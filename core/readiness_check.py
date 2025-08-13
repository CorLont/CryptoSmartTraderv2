#!/usr/bin/env python3
"""
Readiness Check - Hard system readiness based on models, features, health, calibration
Replaces cosmetic "System Online" with real production readiness
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime, timezone

def is_fresh(filepath: str, max_age_seconds: int = 24 * 3600) -> bool:
    """Check if file exists and is fresh (within max_age)"""
    
    path = Path(filepath)
    if not path.exists():
        return False
    
    file_age = time.time() - path.stat().st_mtime
    return file_age < max_age_seconds

def check_models_ready() -> Dict[str, Any]:
    """Check if all required models are trained and available"""
    
    horizons = ["1h", "24h", "168h", "720h"]
    models_status = {}
    
    for horizon in horizons:
        model_path = Path(f"models/saved/rf_{horizon}.pkl")
        
        models_status[horizon] = {
            "exists": model_path.exists(),
            "path": str(model_path),
            "fresh": is_fresh(str(model_path), max_age_seconds=7 * 24 * 3600)  # 7 days
        }
        
        if model_path.exists():
            file_size = model_path.stat().st_size / 1024 / 1024  # MB
            models_status[horizon]["size_mb"] = round(file_size, 2)
            models_status[horizon]["modified"] = datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat()
    
    all_exist = all(status["exists"] for status in models_status.values())
    all_fresh = all(status.get("fresh", False) for status in models_status.values())
    
    return {
        "ready": all_exist and all_fresh,
        "all_exist": all_exist,
        "all_fresh": all_fresh,
        "models": models_status,
        "missing_models": [h for h, s in models_status.items() if not s["exists"]],
        "stale_models": [h for h, s in models_status.items() if s["exists"] and not s.get("fresh", False)]
    }

def check_features_ready() -> Dict[str, Any]:
    """Check if features are available and fresh"""
    
    features_path = Path("exports/features.parquet")
    
    if not features_path.exists():
        return {
            "ready": False,
            "exists": False,
            "reason": "Features file not found"
        }
    
    # Check freshness (features should be updated daily)
    fresh = is_fresh(str(features_path), max_age_seconds=24 * 3600)
    
    file_info = {
        "size_mb": round(features_path.stat().st_size / 1024 / 1024, 2),
        "modified": datetime.fromtimestamp(features_path.stat().st_mtime).isoformat(),
        "age_hours": round((time.time() - features_path.stat().st_mtime) / 3600, 1)
    }
    
    return {
        "ready": fresh,
        "exists": True,
        "fresh": fresh,
        "file_info": file_info,
        "reason": "OK" if fresh else f"Features stale ({file_info['age_hours']}h old)"
    }

def check_health_score() -> Dict[str, Any]:
    """Check latest health score from daily logs"""
    
    # Try multiple potential health score sources
    health_sources = [
        "logs/daily/latest.json",
        "logs/daily/health_summary.json",
        "exports/production/predictions_summary.json"
    ]
    
    for source_path in health_sources:
        path = Path(source_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Extract health score (different formats possible)
                health_score = None
                
                if "health_score" in data:
                    health_score = data["health_score"]
                elif "system_health" in data:
                    health_score = data["system_health"].get("score", 0)
                elif "gate_statistics" in data:
                    # Derive health from prediction success
                    gate_stats = data["gate_statistics"]
                    total_candidates = gate_stats.get("total_candidates", 0)
                    total_passed = gate_stats.get("total_passed", 0)
                    
                    if total_candidates > 0:
                        pass_rate = total_passed / total_candidates
                        health_score = min(100, pass_rate * 100 + 50)  # 50-100 scale
                
                if health_score is not None:
                    return {
                        "ready": health_score >= 85,
                        "score": health_score,
                        "source": source_path,
                        "threshold": 85,
                        "reason": "OK" if health_score >= 85 else f"Health score {health_score:.1f} < 85"
                    }
                    
            except Exception as e:
                continue
    
    # No health score available
    return {
        "ready": False,
        "score": 0,
        "source": None,
        "reason": "No health score available"
    }

def check_calibration_ready() -> Dict[str, Any]:
    """Check if probability calibration is available"""
    
    calibration_sources = [
        "models/probability_calibrator.pkl",
        "models/calibration_report.json",
        "logs/daily/calibration_summary.json"
    ]
    
    for source_path in calibration_sources:
        path = Path(source_path)
        if path.exists():
            fresh = is_fresh(str(path), max_age_seconds=7 * 24 * 3600)  # 7 days
            
            return {
                "ready": fresh,
                "exists": True,
                "fresh": fresh,
                "source": source_path,
                "reason": "OK" if fresh else "Calibration stale"
            }
    
    return {
        "ready": False,
        "exists": False,
        "source": None,
        "reason": "No calibration found"
    }

def check_predictions_ready() -> Dict[str, Any]:
    """Check if recent predictions are available"""
    
    predictions_path = Path("exports/production/predictions.csv")
    
    if not predictions_path.exists():
        return {
            "ready": False,
            "exists": False,
            "reason": "No predictions.csv found"
        }
    
    # Check freshness (predictions should be recent)
    fresh = is_fresh(str(predictions_path), max_age_seconds=6 * 3600)  # 6 hours
    
    # Try to get prediction count
    prediction_count = 0
    try:
        import pandas as pd
        pred_df = pd.read_csv(predictions_path)
        prediction_count = len(pred_df)
    except Exception:
        pass
    
    return {
        "ready": fresh and prediction_count > 0,
        "exists": True,
        "fresh": fresh,
        "prediction_count": prediction_count,
        "reason": "OK" if fresh and prediction_count > 0 else f"Predictions stale or empty ({prediction_count})"
    }

def system_readiness_check() -> Tuple[bool, Dict[str, Any], float]:
    """
    Comprehensive system readiness check
    
    Returns:
        (is_ready, status_details, overall_score)
    """
    
    print("🔍 SYSTEM READINESS CHECK")
    print("=" * 40)
    
    # Run all checks
    models_check = check_models_ready()
    features_check = check_features_ready()
    health_check = check_health_score()
    calibration_check = check_calibration_ready()
    predictions_check = check_predictions_ready()
    
    # Component status
    components = {
        "models": models_check,
        "features": features_check,
        "health": health_check,
        "calibration": calibration_check,
        "predictions": predictions_check
    }
    
    # Calculate readiness scores
    component_scores = {}
    for component, check in components.items():
        if check["ready"]:
            component_scores[component] = 100
        elif component == "health" and "score" in check:
            component_scores[component] = check["score"]
        else:
            component_scores[component] = 0
    
    # Weighted overall score
    weights = {
        "models": 0.3,      # Critical
        "features": 0.25,   # Critical  
        "health": 0.2,      # Important
        "calibration": 0.15, # Important
        "predictions": 0.1   # Nice to have
    }
    
    overall_score = sum(component_scores[comp] * weights[comp] for comp in component_scores.keys())
    
    # System is ready if score >= 85
    is_ready = overall_score >= 85.0
    
    # Create detailed status
    status_details = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_ready": is_ready,
        "overall_score": round(overall_score, 1),
        "readiness_threshold": 85.0,
        "components": components,
        "component_scores": component_scores,
        "blocking_issues": []
    }
    
    # Identify blocking issues
    for comp_name, comp_check in components.items():
        if not comp_check["ready"]:
            status_details["blocking_issues"].append({
                "component": comp_name,
                "reason": comp_check.get("reason", "Unknown issue"),
                "critical": comp_name in ["models", "features"]
            })
    
    # Print summary
    print(f"Overall Readiness: {'✅ READY' if is_ready else '❌ NOT READY'} ({overall_score:.1f}/100)")
    
    for comp_name, comp_check in components.items():
        status = "✅" if comp_check["ready"] else "❌"
        score = component_scores[comp_name]
        reason = comp_check.get("reason", "OK")
        print(f"  {comp_name:>12s}: {status} {score:3.0f}/100 - {reason}")
    
    if status_details["blocking_issues"]:
        print("\n🚫 Blocking Issues:")
        for issue in status_details["blocking_issues"]:
            critical = "CRITICAL" if issue["critical"] else "Warning"
            print(f"  • {issue['component']}: {issue['reason']} ({critical})")
    
    return is_ready, status_details, overall_score

def get_readiness_badge() -> Dict[str, str]:
    """Get readiness badge for UI display"""
    
    is_ready, status, score = system_readiness_check()
    
    if is_ready:
        return {
            "status": "READY",
            "color": "#22c55e",  # Green
            "text": f"System Ready ({score:.0f}/100)",
            "icon": "✅"
        }
    elif score >= 60:
        return {
            "status": "PARTIAL",
            "color": "#f59e0b",  # Amber
            "text": f"Partial Ready ({score:.0f}/100)",
            "icon": "⚠️"
        }
    else:
        return {
            "status": "NOT_READY",
            "color": "#ef4444",  # Red
            "text": f"Not Ready ({score:.0f}/100)",
            "icon": "❌"
        }

if __name__ == "__main__":
    # Run readiness check
    is_ready, status, score = system_readiness_check()
    
    print(f"\n{'='*40}")
    print("READINESS SUMMARY")
    print(f"{'='*40}")
    
    if is_ready:
        print("🟢 SYSTEM IS PRODUCTION READY")
        print("   • All critical components operational")
        print("   • Models trained and calibrated")
        print("   • Fresh features and health monitoring")
    else:
        print("🔴 SYSTEM NOT READY FOR PRODUCTION")
        print("   • Fix blocking issues before live trading")
        
        critical_issues = [i for i in status["blocking_issues"] if i["critical"]]
        if critical_issues:
            print("\n   Critical fixes needed:")
            for issue in critical_issues:
                print(f"     - {issue['component']}: {issue['reason']}")
    
    # Get UI badge
    badge = get_readiness_badge()
    print(f"\nUI Badge: {badge['icon']} {badge['text']}")
    
    # Save status for dashboard
    output_dir = Path("logs/daily")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    status_file = output_dir / "readiness_status.json"
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n💾 Readiness status saved to: {status_file}")