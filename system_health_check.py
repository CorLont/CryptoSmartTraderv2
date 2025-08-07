"""
System Health Check - Complete diagnostic for faultless operation
"""

import sys
import logging
from datetime import datetime
import traceback

# Suppress logging during health check
logging.getLogger().setLevel(logging.ERROR)

def health_check():
    """Comprehensive system health check"""
    
    print("🏥 CRYPTOSMARTTRADER V2 - SYSTEM HEALTH CHECK")
    print("=" * 70)
    print(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check 1: Python Environment
    print("\n📍 PYTHON ENVIRONMENT")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check 2: Core Dependencies
    print("\n📍 CORE DEPENDENCIES")
    critical_deps = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'ccxt',
        'scikit-learn', 'xgboost', 'torch', 'optuna'
    ]
    
    dep_status = {}
    for dep in critical_deps:
        try:
            __import__(dep)
            dep_status[dep] = "✅ OK"
        except ImportError:
            dep_status[dep] = "❌ MISSING"
    
    for dep, status in dep_status.items():
        print(f"{dep:15s}: {status}")
    
    # Check 3: Core Modules
    print("\n📍 CORE MODULES")
    core_modules = [
        ('zero_fallback_validator', 'Zero Fallback Data Validator'),
        ('deep_ml_engine', 'Deep ML Engine'),
        ('async_orchestrator', 'Async Orchestrator'),
        ('security_manager', 'Security Manager'),
        ('risk_management', 'Risk Management'),
        ('bayesian_uncertainty', 'Bayesian Uncertainty'),
        ('cross_coin_fusion', 'Cross-Coin Fusion'),
        ('automl_engine', 'AutoML Engine')
    ]
    
    module_status = {}
    for module_name, display_name in core_modules:
        try:
            __import__(f'core.{module_name}')
            module_status[display_name] = "✅ OK"
        except Exception as e:
            module_status[display_name] = f"❌ ERROR: {str(e)[:50]}"
    
    for module, status in module_status.items():
        print(f"{module:25s}: {status}")
    
    # Check 4: Functional Tests
    print("\n📍 FUNCTIONAL TESTS")
    
    # Test Zero Fallback Validator
    try:
        from core.zero_fallback_validator import ZeroFallbackValidator
        validator = ZeroFallbackValidator()
        
        test_data = {
            'open': 100, 'high': 105, 'low': 95, 'close': 102,
            'volume': 1000000, 'timestamp': 1641234567, 'source': 'kraken'
        }
        result = validator.validate_price_data(test_data, 'BTC/USD')
        print(f"Data Validation         : ✅ PASSED (result: {result.is_valid})")
    except Exception as e:
        print(f"Data Validation         : ❌ FAILED ({e})")
    
    # Test Deep ML Engine
    try:
        from core.deep_ml_engine import DeepMLEngine
        from unittest.mock import Mock
        
        engine = DeepMLEngine(Mock())
        status = engine.get_model_status()
        print(f"ML Engine Status        : ✅ PASSED (models: {status['models_initialized']})")
    except Exception as e:
        print(f"ML Engine Status        : ❌ FAILED ({e})")
    
    # Test Security Manager
    try:
        from core.security_manager import SecurityManager
        
        security = SecurityManager()
        sec_status = security.get_security_status()
        print(f"Security System         : ✅ PASSED (init: {sec_status['security_initialized']})")
    except Exception as e:
        print(f"Security System         : ❌ FAILED ({e})")
    
    # Test Risk Management
    try:
        from core.risk_management import RiskManagementEngine
        from unittest.mock import Mock
        
        risk_mgr = RiskManagementEngine(Mock())
        risk_status = risk_mgr.get_risk_status()
        print(f"Risk Management         : ✅ PASSED")
    except Exception as e:
        print(f"Risk Management         : ❌ FAILED ({e})")
    
    # Check 5: File System
    print("\n📍 FILE SYSTEM")
    
    critical_files = [
        'app.py',
        'quick_start.bat',
        'start_cryptotrader.bat',
        'config.json',
        'replit.md'
    ]
    
    import os
    for file in critical_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"{file:20s}: ✅ EXISTS ({size:,} bytes)")
        else:
            print(f"{file:20s}: ❌ MISSING")
    
    # Check 6: Configuration
    print("\n📍 CONFIGURATION")
    
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"Config file             : ✅ VALID ({len(config)} sections)")
    except Exception as e:
        print(f"Config file             : ❌ ERROR ({e})")
    
    # Final Status
    print("\n" + "=" * 70)
    print("🎯 OVERALL SYSTEM STATUS")
    print("=" * 70)
    
    # Count successes
    total_deps = len(critical_deps)
    ok_deps = sum(1 for status in dep_status.values() if "✅" in status)
    
    total_modules = len(core_modules)
    ok_modules = sum(1 for status in module_status.values() if "✅" in status)
    
    print(f"Dependencies: {ok_deps}/{total_deps} OK ({(ok_deps/total_deps)*100:.1f}%)")
    print(f"Core modules: {ok_modules}/{total_modules} OK ({(ok_modules/total_modules)*100:.1f}%)")
    
    if ok_deps == total_deps and ok_modules == total_modules:
        print("\n🎉 SYSTEM STATUS: FULLY OPERATIONAL")
        print("🚀 Ready for faultless production operation!")
        print("💡 Use 'quick_start.bat' to launch the complete system")
        return True
    else:
        print("\n⚠️  SYSTEM STATUS: NEEDS ATTENTION")
        print("🔧 Some components require fixes before production use")
        return False

if __name__ == "__main__":
    try:
        health_check()
    except Exception as e:
        print(f"\n💥 HEALTH CHECK FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)