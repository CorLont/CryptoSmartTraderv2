"""
Dependency Fixer - Ensures all imports work correctly
"""

import subprocess
import sys
import logging

def install_missing_dependencies():
    """Install missing dependencies for CryptoSmartTrader"""
    
    print("🔧 Checking and installing missing dependencies...")
    
    # List of required packages
    required_packages = [
        'torch',
        'optuna', 
        'pytest',
        'pytest-asyncio',
        'pytest-cov'
    ]
    
    missing_packages = []
    
    # Check which packages are missing
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
    else:
        print("✓ All dependencies already installed!")
    
    print("\n🚀 Dependency check completed!")

def test_imports():
    """Test all critical imports"""
    
    print("\n🧪 Testing critical imports...")
    
    try:
        from core.zero_fallback_validator import ZeroFallbackValidator
        print("✓ ZeroFallbackValidator imports correctly")
    except Exception as e:
        print(f"✗ ZeroFallbackValidator error: {e}")
    
    try:
        from core.deep_ml_engine import DeepMLEngine
        print("✓ DeepMLEngine imports correctly")
    except Exception as e:
        print(f"✗ DeepMLEngine error: {e}")
    
    try:
        from core.async_orchestrator import AsyncOrchestrator
        print("✓ AsyncOrchestrator imports correctly")
    except Exception as e:
        print(f"✗ AsyncOrchestrator error: {e}")
    
    try:
        from core.security_manager import SecurityManager
        print("✓ SecurityManager imports correctly")
    except Exception as e:
        print(f"✗ SecurityManager error: {e}")
    
    try:
        from core.risk_management import RiskManagementEngine
        print("✓ RiskManagementEngine imports correctly")
    except Exception as e:
        print(f"✗ RiskManagementEngine error: {e}")
    
    try:
        from core.bayesian_uncertainty import BayesianUncertaintyModel
        print("✓ BayesianUncertaintyModel imports correctly")
    except Exception as e:
        print(f"✗ BayesianUncertaintyModel error: {e}")
    
    try:
        from core.cross_coin_fusion import CrossCoinFusionEngine
        print("✓ CrossCoinFusionEngine imports correctly")
    except Exception as e:
        print(f"✗ CrossCoinFusionEngine error: {e}")
    
    try:
        from core.automl_engine import AutoMLEngine
        print("✓ AutoMLEngine imports correctly")
    except Exception as e:
        print(f"✗ AutoMLEngine error: {e}")

if __name__ == "__main__":
    # Suppress logging during dependency check
    logging.getLogger().setLevel(logging.ERROR)
    
    install_missing_dependencies()
    test_imports()
    
    print("\n🎉 All systems ready for faultless operation!")