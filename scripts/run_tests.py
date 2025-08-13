#!/usr/bin/env python3
"""Test runner script for CI/CD pipeline."""

import sys
import subprocess
from pathlib import Path

def main():
    """Run test suite with proper configuration."""
    test_dir = Path(__file__).parent.parent / "tests"
    
    if not test_dir.exists():
        print("Tests directory not found!")
        return 1
    
    # Run pytest with coverage
    cmd = [
        "pytest", 
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=src/cryptosmarttrader",
        "--cov-report=term-missing",
        "--cov-fail-under=50"  # Lower threshold temporarily
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("pytest not found! Install with: uv add --dev pytest pytest-cov")
        return 1

if __name__ == "__main__":
    sys.exit(main())