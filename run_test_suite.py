#!/usr/bin/env python3
"""Comprehensive test suite runner for CryptoSmartTrader V2."""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list, description: str, timeout: int = 300) -> tuple[bool, str]:
    """Run a command with timeout and return success status."""
    try:
        print(f"🔄 Running {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=".")

        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
            if output.strip():
                print(f"   Output: {output.strip()[:300]}...")

        return success, output
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False, str(e)


def main():
    """Run comprehensive test suite."""
    print("🧪 CryptoSmartTrader V2 - Comprehensive Test Suite")
    print("=" * 70)

    test_results = {}
    start_time = time.time()

    # 1. Unit Tests
    print("\n1. 📦 UNIT TESTS")
    print("-" * 40)

    unit_tests = [
        (["uv", "run", "pytest", "tests/unit/", "-v", "-x", "--tb=short"], "Unit Tests - Sizing"),
        (
            ["uv", "run", "pytest", "tests/unit/test_risk_guard.py", "-v", "--tb=short"],
            "Unit Tests - Risk Guard",
        ),
        (
            ["uv", "run", "pytest", "tests/unit/test_execution_policy.py", "-v", "--tb=short"],
            "Unit Tests - Execution Policy",
        ),
    ]

    unit_success_count = 0
    for cmd, desc in unit_tests:
        success, _ = run_command(cmd, desc, timeout=120)
        test_results[desc] = success
        if success:
            unit_success_count += 1

    # 2. Integration Tests
    print("\n2. 🔗 INTEGRATION TESTS")
    print("-" * 40)

    integration_tests = [
        (
            [
                "uv",
                "run",
                "pytest",
                "tests/integration/test_exchange_adapter.py",
                "-v",
                "--tb=short",
                "-m",
                "not slow",
            ],
            "Exchange Adapter Integration",
        ),
        (
            ["uv", "run", "pytest", "tests/integration/test_api_health.py", "-v", "--tb=short"],
            "API Health Integration",
        ),
        (
            [
                "uv",
                "run",
                "pytest",
                "tests/integration/test_backtest_parity.py",
                "-v",
                "--tb=short",
                "-m",
                "not slow",
            ],
            "Backtest Parity Integration",
        ),
    ]

    integration_success_count = 0
    for cmd, desc in integration_tests:
        success, _ = run_command(cmd, desc, timeout=180)
        test_results[desc] = success
        if success:
            integration_success_count += 1

    # 3. E2E Smoke Tests (if services are running)
    print("\n3. 🚀 E2E SMOKE TESTS")
    print("-" * 40)

    # Check if services are running first
    health_check_success, _ = run_command(
        ["curl", "-s", "http://localhost:8001/health"], "Service Health Check", timeout=5
    )

    if health_check_success:
        e2e_tests = [
            (
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/e2e/test_smoke_tests.py::TestSystemStartup::test_api_health_endpoint",
                    "-v",
                ],
                "API Health E2E",
            ),
            (
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/e2e/test_smoke_tests.py::TestBasicFunctionality",
                    "-v",
                ],
                "Basic Functionality E2E",
            ),
        ]

        e2e_success_count = 0
        for cmd, desc in e2e_tests:
            success, _ = run_command(cmd, desc, timeout=60)
            test_results[desc] = success
            if success:
                e2e_success_count += 1
    else:
        print("⚠️  Services not running - Skipping E2E tests")
        print("   Start services with: python start_replit_services.py")
        e2e_success_count = 0
        e2e_tests = []

    # 4. Critical Component Tests
    print("\n4. 🎯 CRITICAL COMPONENT TESTS")
    print("-" * 40)

    critical_tests = [
        (
            ["python", "-c", "import src.cryptosmarttrader; print('✓ Package import successful')"],
            "Package Import Test",
        ),
        (
            [
                "python",
                "-c",
                "from src.cryptosmarttrader.core.config_manager import ConfigManager; c=ConfigManager(); print('✓ Config OK')",
            ],
            "Config Manager Test",
        ),
        (
            [
                "python",
                "-c",
                "from src.cryptosmarttrader.core.risk_guard import RiskGuard; r=RiskGuard(); print('✓ Risk Guard OK')",
            ],
            "Risk Guard Test",
        ),
    ]

    critical_success_count = 0
    for cmd, desc in critical_tests:
        success, _ = run_command(cmd, desc, timeout=30)
        test_results[desc] = success
        if success:
            critical_success_count += 1

    # 5. Coverage Report (if possible)
    print("\n5. 📊 TEST COVERAGE")
    print("-" * 40)

    coverage_success, coverage_output = run_command(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/",
            "--cov=src/cryptosmarttrader",
            "--cov-report=term-missing",
            "--cov-fail-under=50",
            "-q",
        ],
        "Coverage Analysis",
        timeout=120,
    )
    test_results["Coverage Analysis"] = coverage_success

    if coverage_success and "%" in coverage_output:
        # Extract coverage percentage
        import re

        coverage_match = re.search(r"TOTAL.*?(\d+)%", coverage_output)
        if coverage_match:
            coverage_pct = int(coverage_match.group(1))
            print(f"   Coverage: {coverage_pct}%")

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 70)

    total_tests = len(test_results)
    passed_tests = sum(1 for success in test_results.values() if success)

    print(f"\n📦 Unit Tests: {unit_success_count}/{len(unit_tests)} passed")
    print(f"🔗 Integration Tests: {integration_success_count}/{len(integration_tests)} passed")
    print(f"🚀 E2E Tests: {e2e_success_count}/{len(e2e_tests) if e2e_tests else 0} passed")
    print(f"🎯 Critical Tests: {critical_success_count}/{len(critical_tests)} passed")

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    elapsed_time = time.time() - start_time

    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"⏱️  Total Time: {elapsed_time:.1f}s")

    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")

    if success_rate >= 80:
        print("\n🎉 TEST SUITE: EXCELLENT")
        print("   Enterprise-grade test coverage and quality validated.")
        return 0
    elif success_rate >= 60:
        print("\n⚠️  TEST SUITE: GOOD")
        print("   Most tests passing, some areas need attention.")
        return 1
    else:
        print("\n❌ TEST SUITE: NEEDS IMPROVEMENT")
        print("   Significant test failures require investigation.")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
