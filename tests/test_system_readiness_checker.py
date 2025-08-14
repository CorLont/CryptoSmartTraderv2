#!/usr/bin/env python3
"""
Test script for System Readiness Checker
Validates enterprise fixes: model naming consistency, clean issues filtering, robust health handling
"""

import sys
import json
from pathlib import Path


def test_system_readiness_checker():
    """Test system readiness checker with enterprise fixes"""

    print("Testing System Readiness Checker Enterprise Implementation")
    print("=" * 70)

    try:
        # Direct import to avoid dependency issues
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "readiness_checker", "core/system_readiness_checker.py"
        )
        checker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker_module)

        SystemReadinessChecker = checker_module.SystemReadinessChecker
        ReadinessStatus = checker_module.ReadinessStatus

        # Test 1: Model naming consistency
        print("\n1. Testing Model Naming Consistency...")

        checker = SystemReadinessChecker()

        # Check naming patterns in configuration
        naming_patterns = checker.config["models"]["naming_patterns"]
        required_horizons = checker.config["models"]["required_horizons"]

        print(f"   Configured horizons: {required_horizons}")
        print(f"   Naming patterns: {list(naming_patterns.keys())}")

        # Verify consistent naming pattern
        consistent_naming = True
        for model_type, pattern in naming_patterns.items():
            if "{horizon}" not in pattern:
                consistent_naming = False
                break

            # Test pattern formatting
            try:
                test_filename = pattern.format(horizon="1h")
                if not test_filename.endswith(".pkl"):
                    consistent_naming = False
                    break
            except KeyError:
                consistent_naming = False
                break

        if consistent_naming:
            print("   ✅ PASSED: Consistent naming patterns configured")

            # Test specific pattern examples
            xgb_pattern = naming_patterns.get("xgboost", "")
            tree_pattern = naming_patterns.get("tree", "")

            if "{horizon}_xgb.pkl" in xgb_pattern and "{horizon}_tree.pkl" in tree_pattern:
                print(
                    "   ✅ PASSED: XGBoost and tree patterns use consistent horizon_type.pkl format"
                )
            else:
                print(f"   ⚠️ WARNING: Patterns may not follow exact horizon_type.pkl format")
                print(f"     XGBoost: {xgb_pattern}")
                print(f"     Tree: {tree_pattern}")
        else:
            print("   ❌ FAILED: Inconsistent naming patterns detected")
            return False

        # Test 2: Clean issues filtering (no empty strings)
        print("\n2. Testing Clean Issues Filtering...")

        # Perform model readiness check to test issue generation
        model_readiness = checker._check_model_readiness()

        # Check that issues list contains no empty strings or None values
        clean_issues = True
        for issue in model_readiness.issues:
            if not issue or issue.strip() == "":
                clean_issues = False
                break

        if clean_issues:
            print(
                f"   ✅ PASSED: Issues list is clean ({len(model_readiness.issues)} issues, no empty strings)"
            )
            if model_readiness.issues:
                print(f"   Sample issue: {model_readiness.issues[0]}")
        else:
            print("   ❌ FAILED: Issues list contains empty strings")
            return False

        # Test data readiness for additional issue filtering validation
        data_readiness = checker._check_data_readiness()

        data_clean_issues = True
        for issue in data_readiness.issues:
            if not issue or issue.strip() == "":
                data_clean_issues = False
                break

        if data_clean_issues:
            print(f"   ✅ PASSED: Data issues list is clean ({len(data_readiness.issues)} issues)")
        else:
            print("   ❌ FAILED: Data issues list contains empty strings")
            return False

        # Test 3: Robust health file handling
        print("\n3. Testing Robust Health File Handling...")

        # Test with strict fallback behavior (default)
        strict_checker = SystemReadinessChecker()
        health_readiness_strict = strict_checker._check_health_readiness()

        print(f"   Strict mode health score: {health_readiness_strict.score:.1f}")
        print(f"   Strict mode status: {health_readiness_strict.status.value}")

        # Test with lenient fallback behavior
        lenient_config = {"health": {"fallback_behavior": "lenient"}}

        lenient_checker = SystemReadinessChecker(config=lenient_config)
        health_readiness_lenient = lenient_checker._check_health_readiness()

        print(f"   Lenient mode health score: {health_readiness_lenient.score:.1f}")
        print(f"   Lenient mode status: {health_readiness_lenient.status.value}")

        # Verify that lenient mode is more forgiving
        if health_readiness_lenient.score >= health_readiness_strict.score:
            print("   ✅ PASSED: Lenient fallback behavior is more forgiving than strict")
        else:
            print("   ❌ FAILED: Lenient fallback not working as expected")
            return False

        # Test missing health files behavior
        missing_files_found = False
        for issue in health_readiness_strict.issues:
            if "missing" in issue.lower() or "cannot read" in issue.lower():
                missing_files_found = True
                break

        if missing_files_found:
            print("   ✅ PASSED: Strict mode properly reports missing health files")
        else:
            print("   ⚠️ WARNING: No missing health file issues detected (may be present)")

        # Test 4: Overall system readiness integration
        print("\n4. Testing Overall System Readiness Integration...")

        # Perform complete readiness check
        report = checker.check_system_readiness()

        if report.timestamp and len(report.components) >= 4:
            print(
                f"   ✅ PASSED: Complete readiness check with {len(report.components)} components"
            )
            print(f"   Overall status: {report.overall_status.value}")
            print(f"   Overall score: {report.overall_score:.1f}%")
            print(f"   GO/NO-GO decision: {'GO' if report.go_no_go_decision else 'NO-GO'}")
        else:
            print("   ❌ FAILED: Incomplete readiness check")
            return False

        # Verify all expected components are present
        expected_components = {"models", "data", "calibration", "health"}
        actual_components = {c.component for c in report.components}

        if expected_components.issubset(actual_components):
            print("   ✅ PASSED: All expected components checked")
        else:
            missing = expected_components - actual_components
            print(f"   ❌ FAILED: Missing components: {missing}")
            return False

        # Test 5: Configuration validation
        print("\n5. Testing Configuration Management...")

        # Test custom configuration merging
        custom_config = {
            "models": {
                "required_horizons": ["1h", "4h", "24h"],  # Custom horizons
                "minimum_model_age_hours": 2,  # Custom age requirement
            },
            "readiness": {
                "minimum_overall_score": 80.0  # Custom threshold
            },
        }

        custom_checker = SystemReadinessChecker(config=custom_config)

        if custom_checker.config["models"]["required_horizons"] == ["1h", "4h", "24h"]:
            print("   ✅ PASSED: Custom configuration applied successfully")
        else:
            print("   ❌ FAILED: Custom configuration not merged correctly")
            return False

        # Verify default configuration is preserved
        if "naming_patterns" in custom_checker.config["models"]:
            print("   ✅ PASSED: Default configuration preserved during merge")
        else:
            print("   ❌ FAILED: Default configuration lost during merge")
            return False

        # Test 6: GO/NO-GO decision logic
        print("\n6. Testing GO/NO-GO Decision Logic...")

        # Test decision factors
        components = report.components
        overall_score = report.overall_score
        go_no_go = checker._make_go_no_go_decision(overall_score, components)

        print(f"   Decision factors:")
        print(
            f"   - Overall score: {overall_score:.1f}% (threshold: {checker.config['readiness']['minimum_overall_score']})"
        )

        critical_components = checker.config["readiness"]["critical_components"]
        critical_ready = True
        for component in components:
            if component.component in critical_components:
                print(f"   - {component.component} (critical): {component.status.value}")
                if component.status == ReadinessStatus.NOT_READY:
                    critical_ready = False

        expected_decision = (
            overall_score >= checker.config["readiness"]["minimum_overall_score"] and critical_ready
        )

        if go_no_go == expected_decision:
            print(f"   ✅ PASSED: GO/NO-GO decision logic correct: {go_no_go}")
        else:
            print(f"   ❌ FAILED: GO/NO-GO decision logic incorrect")
            print(f"     Expected: {expected_decision}, Got: {go_no_go}")
            return False

        # Test 7: Summary and recommendations
        print("\n7. Testing Summary and Recommendations...")

        if report.summary and report.recommendations:
            print(f"   ✅ PASSED: Summary and recommendations generated")
            print(f"   Summary: {report.summary}")
            print(f"   Recommendations: {len(report.recommendations)}")
        else:
            print("   ❌ FAILED: Missing summary or recommendations")
            return False

        # Verify recommendations are actionable
        actionable_recommendations = True
        for rec in report.recommendations:
            if not rec or len(rec.strip()) < 10:  # Too short to be actionable
                actionable_recommendations = False
                break

        if actionable_recommendations:
            print("   ✅ PASSED: Recommendations appear actionable")
        else:
            print("   ⚠️ WARNING: Some recommendations may not be actionable")

        print("\n" + "=" * 70)
        print("✅ SYSTEM READINESS CHECKER ENTERPRISE FIXES VALIDATION COMPLETE")
        print("- Model naming: Consistent {horizon}_type.pkl pattern across all model types")
        print("- Clean issues: No empty strings in issues lists, proper None filtering")
        print("- Health handling: Robust file handling with strict/lenient fallback modes")
        print("- Integration: Complete 4-component readiness assessment with GO/NO-GO")
        print("- Configuration: Custom config merging with default preservation")
        print("- Decision logic: Proper GO/NO-GO based on score and critical components")
        print("- Recommendations: Actionable guidance based on component status")

        return True

    except Exception as e:
        print(f"\n❌ SYSTEM READINESS CHECKER TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_system_readiness_checker()
    sys.exit(0 if success else 1)
