#!/usr/bin/env python3
"""
Test Runner Script - Smart test execution based on markers
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run command and return result"""
    if description:
        print(f"\n🔄 {description}")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Failed (exit code: {result.returncode})")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Smart test runner for CryptoSmartTrader")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--property", action="store_true", help="Run property-based tests")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--api-key", action="store_true", help="Include tests requiring API keys")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--regression", action="store_true", help="Run regression tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel (number of workers)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("pytest.ini").exists():
        print("❌ pytest.ini not found. Please run from project root.")
        return 1
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=core", "--cov=ml", "--cov=dashboards", "--cov-report=html", "--cov-report=term"])
    
    # Add specific file
    if args.file:
        cmd.append(args.file)
    
    # Add pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Build marker expression
    markers = []
    
    if args.unit:
        markers.append("unit")
    
    if args.integration:
        markers.append("integration")
    
    if args.property:
        markers.append("property")
    
    if args.smoke:
        markers.append("smoke")
    
    if args.regression:
        markers.append("regression")
    
    # Exclude slow tests by default
    if not args.slow:
        markers.append("not slow")
    
    # Exclude API key tests by default
    if not args.api_key:
        markers.append("not api_key")
    
    # If no specific markers selected, run default set
    if not any([args.unit, args.integration, args.property, args.smoke, args.regression]):
        # Default: unit tests and fast integration tests
        markers = ["(unit or integration) and not slow and not api_key"]
    
    # Add marker expression to command
    if markers:
        marker_expr = " and ".join(markers)
        cmd.extend(["-m", marker_expr])
    
    # Default test options
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print("🧪 CryptoSmartTrader Test Runner")
    print("=" * 50)
    
    # Show what we're going to run
    print(f"Command: {' '.join(cmd)}")
    
    if markers:
        print(f"Marker expression: {marker_expr}")
    
    # Check for required API keys if needed
    if args.api_key:
        required_keys = ["KRAKEN_API_KEY", "BINANCE_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
            print("Some tests may be skipped.")
    
    # Run tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - Terminal output above")
    else:
        print("\n💥 Some tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())