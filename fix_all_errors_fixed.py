#!/usr/bin/env python3
"""
Fixed version - alle kritische issues opgelost
"""
import importlib
import sys
from pathlib import Path
import logging
import subprocess
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_essential_imports():
    """Check alleen essentiële imports - geen blind pip install"""
    essential_modules = [
        "streamlit",
        "pandas", 
        "numpy",
        "ccxt",
        "plotly",
        "openai"
    ]
    
    results = {}
    for module in essential_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ Available"
        except ImportError as e:
            results[module] = f"❌ Missing: {str(e)}"
    
    return results

def check_port_availability(port=5000):
    """Check of poort beschikbaar is"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # True als poort vrij is
    except Exception:
        return False

def check_project_structure():
    """Check essentiële project structuur"""
    essential_dirs = [
        "models/saved",
        "exports/production", 
        "logs",
        ".streamlit"
    ]
    
    essential_files = [
        ".streamlit/config.toml",
        "config.json"
    ]
    
    results = {}
    
    # Check directories
    for dir_path in essential_dirs:
        path = Path(dir_path)
        if path.exists():
            results[dir_path] = "✅ Exists"
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                results[dir_path] = "✅ Created"
            except Exception as e:
                results[dir_path] = f"❌ Failed: {e}"
    
    # Check files
    for file_path in essential_files:
        path = Path(file_path)
        if path.exists():
            results[file_path] = "✅ Exists"
        else:
            results[file_path] = "❌ Missing"
    
    return results

def test_system_functionality_safe():
    """Safe system test zonder misleidende success claims"""
    test_results = {}
    
    # Test core imports
    core_modules = ["streamlit", "pandas", "numpy", "plotly"]
    for module in core_modules:
        try:
            importlib.import_module(module)
            test_results[f"Import {module}"] = "✅ Pass"
        except ImportError as e:
            test_results[f"Import {module}"] = f"❌ Fail: {e}"
    
    # Test configuration
    config_file = Path("config.json")
    if config_file.exists():
        test_results["Configuration"] = "✅ Pass"
    else:
        test_results["Configuration"] = "❌ Fail: No config.json"
    
    # Test Streamlit config
    st_config = Path(".streamlit/config.toml")
    if st_config.exists():
        test_results["Streamlit Config"] = "✅ Pass"
    else:
        test_results["Streamlit Config"] = "❌ Fail: No .streamlit/config.toml"
    
    return test_results

def main():
    """Fixed main function met honest reporting"""
    print("🔧 FIXED ERROR CHECKER - NO FALSE POSITIVES")
    print("-" * 50)
    
    # Check imports
    print("📦 Essential Imports:")
    import_results = check_essential_imports()
    for module, status in import_results.items():
        print(f"  {module}: {status}")
    
    print()
    
    # Check structure
    print("📁 Project Structure:")
    structure_results = check_project_structure()
    for path, status in structure_results.items():
        print(f"  {path}: {status}")
    
    print()
    
    # Check port
    print("🌐 Network:")
    port_free = check_port_availability(5000)
    print(f"  Port 5000: {'✅ Free' if port_free else '❌ In Use'}")
    
    print()
    
    # System tests
    print("⚙️ System Tests:")
    test_results = test_system_functionality_safe()
    for test, result in test_results.items():
        print(f"  {test}: {result}")
    
    print()
    
    # Honest summary
    import_failures = sum(1 for status in import_results.values() if "❌" in status)
    structure_failures = sum(1 for status in structure_results.values() if "❌" in status)
    test_failures = sum(1 for result in test_results.values() if "❌" in result)
    
    total_issues = import_failures + structure_failures + test_failures
    
    if total_issues == 0:
        print("✅ All checks passed - system ready")
        return 0
    else:
        print(f"⚠️ {total_issues} issues found:")
        print(f"  - Import issues: {import_failures}")
        print(f"  - Structure issues: {structure_failures}")
        print(f"  - System test failures: {test_failures}")
        print("\nResolve these issues before deployment.")
        return 1

if __name__ == "__main__":
    exit(main())