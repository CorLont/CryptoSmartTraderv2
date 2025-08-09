#!/usr/bin/env python3
"""
Sober and fault-tolerant dependency checker
"""
import subprocess
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return f"✅ Python {version.major}.{version.minor}"
    else:
        return f"⚠️  Python {version.major}.{version.minor} (recommend 3.8+)"

def check_core_packages():
    """Check core packages only"""
    core_packages = [
        "streamlit",
        "pandas",
        "numpy", 
        "plotly",
        "ccxt",
        "openai"
    ]
    
    results = {}
    for package in core_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results[package] = f"✅ {version}"
        except ImportError:
            results[package] = "❌ Missing"
    
    return results

def main():
    """Sober dependency check"""
    print("🔧 SOBER DEPENDENCY CHECKER")
    print("-" * 40)
    
    # Python version
    print(f"Python: {check_python_version()}")
    print()
    
    # Core packages
    print("📦 Core Packages:")
    package_results = check_core_packages()
    for package, status in package_results.items():
        print(f"  {package}: {status}")
    
    print()
    
    # Summary
    missing = sum(1 for status in package_results.values() if "❌" in status)
    
    if missing == 0:
        print("✅ All core dependencies available")
        return 0
    else:
        print(f"⚠️  {missing} missing dependencies")
        print("Install with: pip install streamlit pandas numpy plotly ccxt openai")
        return 1

if __name__ == "__main__":
    exit(main())