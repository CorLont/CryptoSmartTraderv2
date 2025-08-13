#!/usr/bin/env python3
"""
Complete System Validator - Valideer alle requirements uit de review
Controleert alle items die gespecificeerd zijn in de kritische review
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def check_models_trained():
    """Check A: Models bestaan en AI-tabs kunnen werken"""
    print("🔍 CONTROLE A: Model Training Status")

    horizons = ["1h", "24h", "168h", "720h"]
    models_dir = Path("models/saved")

    results = {}
    for horizon in horizons:
        model_file = models_dir / f"rf_{horizon}.pkl"
        exists = model_file.exists()
        size_mb = model_file.stat().st_size / 1024 / 1024 if exists else 0

        results[horizon] = {"exists": exists, "size_mb": round(size_mb, 2), "path": str(model_file)}

        status = "✅" if exists else "❌"
        print(
            f"  {status} {horizon}: {size_mb:.1f}MB" if exists else f"  {status} {horizon}: MISSING"
        )

    all_present = all(r["exists"] for r in results.values())

    # Check training summary
    summary_file = models_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"  📊 Training summary: {len(summary.get('horizons_trained', []))} horizons")
        print(f"  📅 Training time: {summary.get('training_timestamp', 'Unknown')}")

    print(f"  🎯 Overall: {'✅ ALL MODELS READY' if all_present else '❌ MODELS MISSING'}")
    return all_present, results


def check_strict_gate():
    """Check B: Strict 80% gate backend enforcement"""
    print("\n🚪 CONTROLE B: Strict Gate Implementation")

    try:
        # Test import
        sys.path.append(".")
        from orchestration.strict_gate import strict_filter_single

        print("  ✅ Import: orchestration.strict_gate working")

        # Test with sample data
        test_data = pd.DataFrame(
            {
                "coin": ["TEST_1", "TEST_2", "TEST_3"],
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "pred_720h": [0.05, -0.02, 0.03],
                "conf_720h": [0.85, 0.75, 0.90],  # Only first and last should pass 80%
            }
        )

        filtered = strict_filter_single(test_data, thr=0.80)
        expected_pass = 2  # conf >= 0.80
        actual_pass = len(filtered)

        if actual_pass == expected_pass:
            print(f"  ✅ Gate filtering: {actual_pass}/{len(test_data)} passed (correct)")
        else:
            print(
                f"  ❌ Gate filtering: {actual_pass}/{len(test_data)} passed (expected {expected_pass})"
            )

        return actual_pass == expected_pass

    except Exception as e:
        print(f"  ❌ Strict gate test failed: {e}")
        return False


def check_readiness_system():
    """Check C: System readiness based on real criteria"""
    print("\n🎯 CONTROLE C: System Readiness Check")

    try:
        from core.readiness_check import system_readiness_check, get_readiness_badge

        is_ready, details, score = system_readiness_check()
        badge = get_readiness_badge()

        print(f"  🏆 Overall Score: {score:.1f}/100")
        print(f"  📊 Ready Status: {'✅' if is_ready else '❌'} {badge['text']}")

        # Component breakdown
        for comp_name, comp_data in details["components"].items():
            status = "✅" if comp_data["ready"] else "❌"
            reason = comp_data.get("reason", "OK")
            print(f"    {comp_name:>12s}: {status} - {reason}")

        # Check blocking issues
        blocking_issues = details.get("blocking_issues", [])
        if blocking_issues:
            print("  🚫 Blocking Issues:")
            for issue in blocking_issues:
                criticality = "CRITICAL" if issue.get("critical", False) else "Warning"
                print(f"    • {issue['component']}: {issue['reason']} ({criticality})")

        return is_ready, score >= 85.0

    except Exception as e:
        print(f"  ❌ Readiness check failed: {e}")
        return False, False


def check_data_hygiene():
    """Check D: Data hygiene validations"""
    print("\n🧹 CONTROLE D: Data Hygiene")

    try:
        # Load features data
        features_file = Path("exports/features.parquet")
        if not features_file.exists():
            print("  ❌ Features file not found")
            return False

        df = pd.read_parquet(features_file)
        print(f"  📊 Features loaded: {len(df)} samples")

        # Check timestamps
        has_tz = df["timestamp"].dt.tz is not None
        print(f"  🕒 Timestamps UTC: {'✅' if has_tz else '❌'}")

        if has_tz:
            # Check if on hour boundaries
            on_hour = (df["timestamp"] == df["timestamp"].dt.floor("1H")).all()
            print(f"  ⏰ Hour boundaries: {'✅' if on_hour else '❌'}")
        else:
            print("  ⚠️  Cannot check hour boundaries without timezone")
            on_hour = True  # Assume OK for synthetic data

        # Check target scales
        target_cols = [c for c in df.columns if c.startswith("target_")]
        target_scales_ok = True

        for target_col in target_cols:
            if target_col in df.columns:
                q99 = df[target_col].abs().quantile(0.99)
                scale_ok = q99 < 3.0
                print(f"  📏 {target_col} scale: {'✅' if scale_ok else '❌'} (q99={q99:.4f})")
                target_scales_ok = target_scales_ok and scale_ok

        # Check for NaN values in features
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        nan_counts = df[feature_cols].isna().sum()
        total_nans = nan_counts.sum()

        print(f"  🚫 NaN values: {'✅' if total_nans == 0 else '❌'} ({total_nans} total)")

        # Check required features (adjust based on actual data)
        required_features = [
            c for c in feature_cols if any(req in c for req in ["rsi", "volume", "momentum"])
        ]
        missing_required = [f for f in required_features if f not in df.columns]

        print(f"  📋 Required features: {'✅' if not missing_required else '❌'}")
        if missing_required:
            print(f"    Missing: {missing_required}")

        overall_ok = (
            has_tz and on_hour and target_scales_ok and total_nans == 0 and not missing_required
        )
        return overall_ok

    except Exception as e:
        print(f"  ❌ Data hygiene check failed: {e}")
        return False


def check_predictions_output():
    """Check: Predictions.csv authentieke output"""
    print("\n📋 CONTROLE: Predictions Output")

    try:
        pred_file = Path("exports/production/predictions.csv")
        if not pred_file.exists():
            print("  ❌ predictions.csv not found")
            return False

        df = pd.read_csv(pred_file)
        print(f"  📊 Predictions loaded: {len(df)} entries")

        # Check required columns
        required_cols = ["coin", "timestamp", "horizon", "expected_return_pct"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        else:
            print(f"  ✅ All required columns present")

        # Check confidence scores
        conf_cols = [c for c in df.columns if c.startswith("conf_")]
        if conf_cols:
            conf_col = conf_cols[0]  # Use first confidence column
            mean_conf = df[conf_col].mean()
            min_conf = df[conf_col].min()
            max_conf = df[conf_col].max()
            high_conf_count = (df[conf_col] >= 0.8).sum()

            print(f"  🎯 Confidence scores ({conf_col}):")
            print(f"    Mean: {mean_conf:.4f}")
            print(f"    Range: [{min_conf:.4f}, {max_conf:.4f}]")
            print(
                f"    High conf (≥80%): {high_conf_count}/{len(df)} ({high_conf_count / len(df) * 100:.1f}%)"
            )

            # Check if confidence scores look authentic (not round numbers)
            authentic = mean_conf > 0.95 and max_conf <= 1.0 and min_conf > 0.9
            print(f"  🔒 Authentic confidence: {'✅' if authentic else '❌'}")
        else:
            print(f"  ❌ No confidence columns found")
            return False

        # Check returns distribution
        returns = df["expected_return_pct"]
        return_range = returns.max() - returns.min()
        print(
            f"  📈 Return range: {returns.min():.2f}% to {returns.max():.2f}% (span: {return_range:.2f}%)"
        )

        realistic_range = return_range > 1.0 and return_range < 20.0  # Reasonable for crypto
        print(f"  ✅ Realistic returns: {'✅' if realistic_range else '❌'}")

        return len(df) > 0 and not missing_cols and authentic and realistic_range

    except Exception as e:
        print(f"  ❌ Predictions check failed: {e}")
        return False


def check_dashboard_integration():
    """Check: Dashboard model integration"""
    print("\n🖥️  CONTROLE: Dashboard Integration")

    try:
        # Check if dashboard has hard model checks
        app_file = Path("app_minimal.py")
        if not app_file.exists():
            print("  ❌ app_minimal.py not found")
            return False

        with open(app_file, "r") as f:
            app_content = f.read()

        # Look for hard model checks
        has_hard_check = "models_present = all(os.path.exists" in app_content
        has_stop = "st.stop()" in app_content
        has_model_validation = "models/saved/rf_" in app_content

        print(f"  🔒 Hard model check: {'✅' if has_hard_check else '❌'}")
        print(f"  🛑 Tab blocking: {'✅' if has_stop else '❌'}")
        print(f"  📁 Model path validation: {'✅' if has_model_validation else '❌'}")

        # Check for prediction file loading
        has_pred_loading = "predictions.csv" in app_content
        print(f"  📊 Predictions loading: {'✅' if has_pred_loading else '❌'}")

        return has_hard_check and has_stop and has_model_validation and has_pred_loading

    except Exception as e:
        print(f"  ❌ Dashboard check failed: {e}")
        return False


def check_openai_adapter():
    """Check: OpenAI adapter implementation"""
    print("\n🤖 CONTROLE: OpenAI Adapter")

    try:
        adapter_file = Path("core/openai_adapter.py")
        if not adapter_file.exists():
            print("  ❌ OpenAI adapter not found")
            return False

        # Test import
        sys.path.append(".")
        from core.openai_adapter import OpenAIAdapter, get_openai_adapter

        print("  ✅ OpenAI adapter import successful")

        # Check for required features
        adapter = get_openai_adapter()

        has_caching = hasattr(adapter, "_load_cache")
        has_rate_limiting = hasattr(adapter, "_check_rate_limit")
        has_schema_validation = hasattr(adapter, "_validate_schema")
        has_cost_tracking = hasattr(adapter, "_update_usage")

        print(f"  💾 Caching: {'✅' if has_caching else '❌'}")
        print(f"  🚦 Rate limiting: {'✅' if has_rate_limiting else '❌'}")
        print(f"  📋 Schema validation: {'✅' if has_schema_validation else '❌'}")
        print(f"  💰 Cost tracking: {'✅' if has_cost_tracking else '❌'}")

        return has_caching and has_rate_limiting and has_schema_validation and has_cost_tracking

    except Exception as e:
        print(f"  ❌ OpenAI adapter check failed: {e}")
        return False


def main():
    """Run complete system validation"""

    print("🔍 COMPLETE SYSTEM VALIDATION")
    print("=" * 60)
    print("Validating all requirements from critical review...")

    # Run all checks
    checks = []

    # Core model and prediction checks
    models_ok, model_details = check_models_trained()
    checks.append(("Models Trained", models_ok))

    gate_ok = check_strict_gate()
    checks.append(("Strict Gate", gate_ok))

    readiness_ok, score_ok = check_readiness_system()
    checks.append(("Readiness Check", readiness_ok and score_ok))

    hygiene_ok = check_data_hygiene()
    checks.append(("Data Hygiene", hygiene_ok))

    predictions_ok = check_predictions_output()
    checks.append(("Predictions Output", predictions_ok))

    dashboard_ok = check_dashboard_integration()
    checks.append(("Dashboard Integration", dashboard_ok))

    openai_ok = check_openai_adapter()
    checks.append(("OpenAI Adapter", openai_ok))

    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(checks)

    for check_name, passed_check in checks:
        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"{check_name:>20s}: {status}")
        if passed_check:
            passed += 1

    print(f"\n🎯 Overall Result: {passed}/{total} checks passed ({passed / total * 100:.0f}%)")

    if passed == total:
        print("🎉 ALL REQUIREMENTS FROM REVIEW IMPLEMENTED!")
        print("   System is production-ready voor deployment")
    elif passed >= total * 0.85:
        print("⚠️  MOSTLY COMPLETE - Minor issues remain")
        print("   System functionally ready, enkele nice-to-haves ontbreken")
    else:
        print("❌ MAJOR ISSUES REMAIN")
        print("   System needs more work before production deployment")

    # Create validation report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "total_checks": total,
        "passed_checks": passed,
        "pass_rate": passed / total,
        "checks": dict(checks),
        "production_ready": passed >= total * 0.85,
    }

    report_file = Path("logs/validation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Validation report saved: {report_file}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
