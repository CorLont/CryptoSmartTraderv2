#!/usr/bin/env python3
"""
Sober and fault-tolerant error fixer
"""
import importlib
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_essential_imports():
    """Check only essential imports - fault tolerant"""
    essential_modules = [
        "streamlit",
        "pandas", 
        "numpy",
        "ccxt",
        "plotly"
    ]
    
    results = {}
    for module in essential_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ Available"
        except ImportError:
            results[module] = "❌ Missing"
    
    return results

def check_project_structure():
    """Check essential project structure"""
    essential_dirs = [
        "models/saved",
        "exports/production", 
        "logs",
        ".streamlit"
    ]
    
    results = {}
    for dir_path in essential_dirs:
        path = Path(dir_path)
        if path.exists():
            results[dir_path] = "✅ Exists"
        else:
            results[dir_path] = "❌ Missing"
            # Create if missing
            try:
                path.mkdir(parents=True, exist_ok=True)
                results[dir_path] = "✅ Created"
            except Exception as e:
                results[dir_path] = f"❌ Failed: {e}"
    
    return results

def main():
    """Sober error checking"""
    print("🔧 SOBER ERROR CHECKER")
    print("-" * 40)
    
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
    
    # Summary
    import_failures = sum(1 for status in import_results.values() if "❌" in status)
    structure_failures = sum(1 for status in structure_results.values() if "❌" in status)
    
    if import_failures == 0 and structure_failures == 0:
        print("✅ All essential components available")
        return 0
    else:
        print(f"⚠️  {import_failures} import issues, {structure_failures} structure issues")
        return 1

if __name__ == "__main__":
    exit(main())