#!/usr/bin/env python3
"""
Fixed dependency checker - implements audit point H
"""
import sys
import importlib
from pathlib import Path

def check_dependencies():
    """Fixed: beperk ambitie & faal vriendelijk"""
    print("🔍 Checking essential dependencies...")
    
    # Fixed: houd klein & echt
    modules_to_test = ["app_minimal", "ccxt"]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

def main():
    """Fixed main function"""
    print("=== FIXED DEPENDENCY CHECKER ===")
    check_dependencies()
    print("✅ Dependencies checked")

if __name__ == "__main__":
    main()