#!/usr/bin/env python3
"""
Install Python dependencies for production readiness
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install core dependencies needed for production"""
    
    print("🔧 Installing Python dependencies...")
    
    # Core dependencies for production
    dependencies = [
        "pydantic>=2.0.0",
        "numpy>=1.24.0", 
        "pandas>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.0",
        "prometheus-client>=0.19.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                dep, "--quiet", "--disable-pip-version-check"
            ], check=True, capture_output=True)
            print(f"✅ {dep} installed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    # Test imports
    print("\n🔍 Testing imports...")
    
    test_imports = [
        "pydantic",
        "numpy", 
        "pandas",
        "fastapi",
        "streamlit",
        "plotly",
        "requests"
    ]
    
    for module in test_imports:
        try:
            import importlib
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    print("\n✅ All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)