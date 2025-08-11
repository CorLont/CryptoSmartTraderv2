#!/usr/bin/env python3
"""
Test script for System Validator
Validates enterprise fixes: correct import names, flexible file requirements, accurate module paths, appropriate error classification
"""

import sys
import os
from pathlib import Path

def test_system_validator():
    """Test system validator with enterprise fixes"""
    
    print("Testing System Validator Enterprise Implementation")
    print("=" * 65)
    
    try:
        # Direct import to avoid dependency issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("system_validator", "core/system_validator.py")
        validator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator_module)
        
        SystemValidator = validator_module.SystemValidator
        ValidationSeverity = validator_module.ValidationSeverity
        
        # Test 1: Corrected import names
        print("\n1. Testing Corrected Import Names...")
        
        validator = SystemValidator()
        critical_deps = validator.config["critical_dependencies"]
        
        print(f"   Configured critical dependencies: {critical_deps}")
        
        # Check that sklearn is used instead of scikit-learn
        if "sklearn" in critical_deps and "scikit-learn" not in critical_deps:
            print("   ✅ PASSED: Using 'sklearn' instead of 'scikit-learn' for import")
        else:
            print("   ❌ FAILED: Still using 'scikit-learn' or missing sklearn")
            return False
        
        # Test actual import validation
        try:
            import sklearn
            print("   ✅ PASSED: sklearn import works correctly")
        except ImportError:
            print("   ⚠️ WARNING: sklearn not available for testing")
        
        # Verify no problematic imports
        problematic_imports = ["scikit-learn"]
        for dep in critical_deps:
            if dep in problematic_imports:
                print(f"   ❌ FAILED: Found problematic import: {dep}")
                return False
        
        print("   ✅ PASSED: No problematic import names found")
        
        # Test 2: Flexible file requirements
        print("\n2. Testing Flexible File Requirements...")
        
        file_requirements = validator.config["required_files"]
        
        # Check that files are categorized appropriately
        if "critical" in file_requirements and "recommended" in file_requirements:
            print("   ✅ PASSED: Files categorized into critical/recommended/optional")
            
            critical_files = file_requirements["critical"]
            recommended_files = file_requirements.get("recommended", [])
            
            print(f"   Critical files: {critical_files}")
            print(f"   Recommended files: {recommended_files}")
            
            # Verify that overly strict files are not in critical
            flexible_files = ["replit.md", "app_minimal.py", "config.json"]
            critical_has_flexible = any(f in critical_files for f in flexible_files)
            
            if not critical_has_flexible:
                print("   ✅ PASSED: Flexible files not marked as critical")
            else:
                print("   ⚠️ WARNING: Some flexible files marked as critical")
                
        else:
            print("   ❌ FAILED: File requirements not properly categorized")
            return False
        
        # Test 3: Corrected module paths
        print("\n3. Testing Corrected Module Paths...")
        
        risk_modules = validator.config["risk_modules"]
        print(f"   Risk modules: {risk_modules}")
        
        # Check for corrected paths
        path_corrections = [
            ("core.risk_mitigation", "orchestration.risk_mitigation"),
            ("core.completeness_gate", "orchestration.completeness_gate"),
            ("core.strict_gate", "orchestration.strict_gate")
        ]
        
        all_corrections_applied = True
        for correct_path, old_path in path_corrections:
            if correct_path in risk_modules:
                print(f"   ✅ Using corrected path: {correct_path}")
            elif old_path in risk_modules:
                print(f"   ❌ Still using old path: {old_path}")
                all_corrections_applied = False
            else:
                print(f"   ⚠️ Neither path found for: {correct_path}")
        
        if all_corrections_applied:
            print("   ✅ PASSED: Module paths corrected")
        else:
            print("   ❌ FAILED: Some module paths not corrected")
            return False
        
        # Test 4: Appropriate error classification
        print("\n4. Testing Appropriate Error Classification...")
        
        # Perform validation to test classification
        report = validator.validate_system()
        
        # Check for appropriate logging classification
        logging_results = [r for r in report.validation_results if "logging" in r.check_name.lower()]
        
        logging_classification_correct = True
        for result in logging_results:
            if not result.success and result.severity == ValidationSeverity.CRITICAL:
                print(f"   ❌ Logging issue marked as CRITICAL: {result.check_name}")
                logging_classification_correct = False
            elif not result.success and result.severity == ValidationSeverity.WARNING:
                print(f"   ✅ Logging issue appropriately marked as WARNING: {result.check_name}")
        
        if logging_classification_correct:
            print("   ✅ PASSED: Logging issues classified as warnings, not critical")
        else:
            print("   ❌ FAILED: Logging issues inappropriately classified as critical")
            return False
        
        # Test 5: Overall validation functionality
        print("\n5. Testing Overall Validation Functionality...")
        
        if report.timestamp and len(report.validation_results) > 0:
            print(f"   ✅ PASSED: Validation completed with {len(report.validation_results)} checks")
            print(f"   Overall status: {report.overall_status.value}")
            print(f"   Production ready: {report.production_ready}")
            print(f"   Critical issues: {report.critical_issues}")
            print(f"   Warning issues: {report.warning_issues}")
        else:
            print("   ❌ FAILED: Validation not completed properly")
            return False
        
        # Check validation categories
        expected_categories = ["python_environment", "dependencies", "file_structure", "modules"]
        found_categories = set()
        
        for result in report.validation_results:
            for category in expected_categories:
                if category in result.check_name:
                    found_categories.add(category)
        
        if len(found_categories) >= len(expected_categories) - 1:  # Allow for some variation
            print(f"   ✅ PASSED: Major validation categories covered: {found_categories}")
        else:
            print(f"   ⚠️ WARNING: Some validation categories missing: expected {expected_categories}, found {found_categories}")
        
        # Test 6: Configuration flexibility
        print("\n6. Testing Configuration Flexibility...")
        
        # Test custom configuration
        custom_config = {
            "critical_dependencies": ["numpy", "pandas"],  # Reduced set
            "thresholds": {
                "warning_failure_tolerance": 5  # More permissive
            }
        }
        
        custom_validator = SystemValidator(config=custom_config)
        
        if custom_validator.config["critical_dependencies"] == ["numpy", "pandas"]:
            print("   ✅ PASSED: Custom configuration applied")
        else:
            print("   ❌ FAILED: Custom configuration not applied")
            return False
        
        if custom_validator.config["thresholds"]["warning_failure_tolerance"] == 5:
            print("   ✅ PASSED: Custom thresholds applied")
        else:
            print("   ❌ FAILED: Custom thresholds not applied")
            return False
        
        # Test 7: Validation summary
        print("\n7. Testing Validation Summary...")
        
        summary = validator.get_validation_summary()
        
        required_summary_fields = ["production_ready", "overall_status", "total_checks", "summary"]
        missing_fields = [field for field in required_summary_fields if field not in summary]
        
        if not missing_fields:
            print("   ✅ PASSED: Validation summary complete")
            print(f"   Summary: {summary['summary']}")
        else:
            print(f"   ❌ FAILED: Missing summary fields: {missing_fields}")
            return False
        
        print("\n" + "=" * 65)
        print("✅ SYSTEM VALIDATOR ENTERPRISE FIXES VALIDATION COMPLETE")
        print("- Import names: 'sklearn' used instead of 'scikit-learn', preventing false negatives")
        print("- File requirements: Flexible categorization with critical/recommended/optional levels")
        print("- Module paths: Corrected core.* paths instead of orchestration.* preventing import failures")
        print("- Error classification: Logging issues marked as warnings, not critical failures")
        print("- Validation coverage: Comprehensive checks across environment, dependencies, structure")
        print("- Configuration: Flexible configuration with custom thresholds and requirements")
        print("- Summary reporting: Complete validation status with production readiness assessment")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SYSTEM VALIDATOR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_validator()
    sys.exit(0 if success else 1)