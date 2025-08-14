#!/usr/bin/env python3
"""
Quick Workstation Test - Simulates .bat file dependency checks
Tests all critical dependencies that Windows .bat files need
"""

import sys
import importlib
from pathlib import Path
import subprocess

def test_dependency(package_name, display_name=None):
    """Test if a dependency can be imported"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {display_name}: NOT INSTALLED")
        return False

def test_mandatory_dependencies():
    """Test dependencies that .bat files require"""
    print("=== MANDATORY DEPENDENCIES TEST ===")
    
    critical_deps = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'), 
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('requests', 'Requests'),
        ('aiohttp', 'aiohttp'),
        ('pydantic', 'Pydantic'),
        ('prometheus_client', 'Prometheus Client')
    ]
    
    failed = []
    for package, display in critical_deps:
        if not test_dependency(package, display):
            failed.append(display)
    
    if failed:
        print(f"\n❌ CRITICAL: {len(failed)} mandatory dependencies missing:")
        for dep in failed:
            print(f"   - {dep}")
        return False
    else:
        print(f"\n✅ All {len(critical_deps)} mandatory dependencies OK")
        return True

def test_ai_ml_dependencies():
    """Test AI/ML dependencies"""
    print("\n=== AI/ML DEPENDENCIES TEST ===")
    
    ai_deps = [
        ('openai', 'OpenAI'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers')
    ]
    
    optional_count = 0
    for package, display in ai_deps:
        if test_dependency(package, display):
            optional_count += 1
    
    print(f"\n✓ {optional_count}/{len(ai_deps)} AI/ML dependencies available")
    return True

def test_trading_dependencies():
    """Test trading-specific dependencies"""
    print("\n=== TRADING DEPENDENCIES TEST ===")
    
    trading_deps = [
        ('ccxt', 'CCXT (Exchange Integration)'),
    ]
    
    for package, display in trading_deps:
        test_dependency(package, display)
    
    return True

def test_project_structure():
    """Test project structure that .bat files expect"""
    print("\n=== PROJECT STRUCTURE TEST ===")
    
    critical_paths = [
        'src/cryptosmarttrader',
        'src/cryptosmarttrader/core/mandatory_execution_gateway.py',
        'src/cryptosmarttrader/risk/central_risk_guard.py',
        'src/cryptosmarttrader/execution/execution_discipline.py'
    ]
    
    missing = []
    for path in critical_paths:
        if Path(path).exists():
            print(f"✓ {path}")
        else:
            print(f"❌ {path}: MISSING")
            missing.append(path)
    
    if missing:
        print(f"\n❌ CRITICAL: {len(missing)} essential files missing")
        return False
    else:
        print(f"\n✅ All {len(critical_paths)} critical paths exist")
        return True

def test_main_applications():
    """Test main applications that .bat files try to start"""
    print("\n=== MAIN APPLICATIONS TEST ===")
    
    apps = [
        'app_trading_analysis_dashboard.py',
        'app_fixed_all_issues.py'
    ]
    
    found_apps = []
    for app in apps:
        if Path(app).exists():
            print(f"✓ {app}")
            found_apps.append(app)
        else:
            print(f"❌ {app}: NOT FOUND")
    
    if found_apps:
        print(f"\n✅ {len(found_apps)} application(s) available")
        return True, found_apps[0]  # Return first available app
    else:
        print(f"\n❌ CRITICAL: No main applications found")
        return False, None

def test_gateway_functionality():
    """Test the mandatory execution gateway"""
    print("\n=== MANDATORY EXECUTION GATEWAY TEST ===")
    
    try:
        sys.path.insert(0, '.')
        sys.path.insert(0, 'src')
        
        # Try multiple import paths
        try:
            from src.cryptosmarttrader.core.mandatory_execution_gateway import MANDATORY_GATEWAY, UniversalOrderRequest
        except ImportError:
            # Fallback path
            try:
                from cryptosmarttrader.core.mandatory_execution_gateway import MANDATORY_GATEWAY, UniversalOrderRequest
            except ImportError:
                print("⚠️ Gateway import failed - this is expected in some environments")
                raise ImportError("Gateway module not accessible")
        
        print("✓ Gateway import successful")
        
        # Test order creation
        test_order = UniversalOrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=0.01,
            source_module='workstation_test',
            source_function='validation'
        )
        print("✓ Order request creation successful")
        
        print("✅ Mandatory Execution Gateway operational")
        return True
        
    except Exception as e:
        print(f"❌ Gateway test failed: {str(e)}")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("\n=== ENVIRONMENT CONFIGURATION TEST ===")
    
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env file found")
        
        # Check for critical keys without exposing values
        content = env_file.read_text()
        if 'KRAKEN_API_KEY' in content:
            print("✓ Kraken API key configured")
        else:
            print("⚠️ Kraken API key missing")
            
        if 'OPENAI_API_KEY' in content:
            print("✓ OpenAI API key configured")
        else:
            print("⚠️ OpenAI API key missing")
    else:
        print("⚠️ .env file not found")
    
    # Check pyproject.toml
    if Path('pyproject.toml').exists():
        print("✓ pyproject.toml found")
    else:
        print("⚠️ pyproject.toml missing")
    
    return True

def main():
    """Run complete workstation test"""
    print("CryptoSmartTrader V2 - Workstation Dependency Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    tests = [
        ("Mandatory Dependencies", test_mandatory_dependencies),
        ("AI/ML Dependencies", test_ai_ml_dependencies), 
        ("Trading Dependencies", test_trading_dependencies),
        ("Project Structure", test_project_structure),
        ("Main Applications", test_main_applications),
        ("Mandatory Gateway", test_gateway_functionality),
        ("Environment Config", test_environment_config)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                results[test_name] = result[0]
            else:
                results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("WORKSTATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 WORKSTATION READY FOR CRYPTOSMARTTRADER V2!")
        print("Next steps:")
        print("  1. Run: 2_start_background_services.bat")
        print("  2. Run: 3_start_dashboard.bat")
    elif passed >= total - 2:
        print("\n✅ WORKSTATION MOSTLY READY")
        print("Minor issues detected - system should work")
    else:
        print("\n❌ WORKSTATION NEEDS ATTENTION")
        print("Run: 1_install_all_dependencies.bat")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)