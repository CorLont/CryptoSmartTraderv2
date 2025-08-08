#!/usr/bin/env python3
"""
Test runner script for CryptoSmartTrader V2
Provides comprehensive testing with coverage gates and CI/CD integration
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
import json
from typing import List, Dict, Any
import time

class TestRunner:
    """Comprehensive test runner with coverage enforcement"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.coverage_threshold = 80.0
        self.results = {
            'unit_tests': None,
            'integration_tests': None,
            'contract_tests': None,
            'smoke_tests': None,
            'coverage': None,
            'total_duration': 0
        }
    
    def run_command(self, cmd: List[str], env: Dict[str, str] = None) -> subprocess.CompletedProcess:
        """Run command with proper error handling"""
        print(f"Running: {' '.join(cmd)}")
        
        # Set up environment
        test_env = os.environ.copy()
        if env:
            test_env.update(env)
        
        # Ensure we're in the project root
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env=test_env
        )
        
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage"""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-m", "unit",
            "--cov=core",
            "--cov=agents", 
            "--cov=config",
            "--cov-report=term-missing",
            "--cov-report=json:unit_coverage.json",
            "--junit-xml=unit_test_results.xml",
            "-v"
        ]
        
        result = self.run_command(cmd)
        self.results['unit_tests'] = result.returncode == 0
        
        return result.returncode == 0
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--junit-xml=integration_test_results.xml",
            "-v"
        ]
        
        result = self.run_command(cmd)
        self.results['integration_tests'] = result.returncode == 0
        
        return result.returncode == 0
    
    def run_contract_tests(self) -> bool:
        """Run contract tests for external APIs"""
        print("\n" + "="*60)
        print("RUNNING CONTRACT TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/contract/",
            "-m", "contract",
            "--junit-xml=contract_test_results.xml",
            "-v"
        ]
        
        result = self.run_command(cmd)
        self.results['contract_tests'] = result.returncode == 0
        
        return result.returncode == 0
    
    def run_smoke_tests(self) -> bool:
        """Run smoke tests for dashboard"""
        print("\n" + "="*60)
        print("RUNNING SMOKE TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/smoke/",
            "-m", "smoke",
            "--junit-xml=smoke_test_results.xml",
            "-v"
        ]
        
        result = self.run_command(cmd)
        self.results['smoke_tests'] = result.returncode == 0
        
        return result.returncode == 0
    
    def check_coverage_gate(self) -> bool:
        """Check if coverage meets minimum threshold"""
        print("\n" + "="*60)
        print("CHECKING COVERAGE GATE")
        print("="*60)
        
        coverage_file = self.project_root / "unit_coverage.json"
        
        if not coverage_file.exists():
            print("ERROR: Coverage report not found!")
            return False
        
        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data['totals']['percent_covered']
            
            print(f"Current coverage: {total_coverage:.2f}%")
            print(f"Required coverage: {self.coverage_threshold}%")
            
            if total_coverage >= self.coverage_threshold:
                print("✅ Coverage gate PASSED")
                self.results['coverage'] = True
                return True
            else:
                print("❌ Coverage gate FAILED")
                print(f"Need {self.coverage_threshold - total_coverage:.2f}% more coverage")
                
                # Show files with low coverage
                files = coverage_data['files']
                low_coverage_files = [
                    (filename, data['summary']['percent_covered'])
                    for filename, data in files.items()
                    if data['summary']['percent_covered'] < self.coverage_threshold
                ]
                
                if low_coverage_files:
                    print("\nFiles with low coverage:")
                    for filename, coverage in sorted(low_coverage_files, key=lambda x: x[1]):
                        print(f"  {filename}: {coverage:.2f}%")
                
                self.results['coverage'] = False
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to parse coverage report: {e}")
            return False
    
    def run_all_tests(self, test_types: List[str] = None) -> bool:
        """Run all specified test types"""
        start_time = time.time()
        
        if test_types is None:
            test_types = ['unit', 'integration', 'contract', 'smoke']
        
        print("🧪 CRYPTOSMARTTRADER V2 TEST SUITE")
        print("="*60)
        print(f"Running test types: {', '.join(test_types)}")
        print(f"Coverage threshold: {self.coverage_threshold}%")
        print("="*60)
        
        success = True
        
        # Run each test type
        if 'unit' in test_types:
            if not self.run_unit_tests():
                success = False
                print("❌ Unit tests FAILED")
            else:
                print("✅ Unit tests PASSED")
        
        if 'integration' in test_types:
            if not self.run_integration_tests():
                success = False
                print("❌ Integration tests FAILED")
            else:
                print("✅ Integration tests PASSED")
        
        if 'contract' in test_types:
            if not self.run_contract_tests():
                success = False
                print("❌ Contract tests FAILED")
            else:
                print("✅ Contract tests PASSED")
        
        if 'smoke' in test_types:
            if not self.run_smoke_tests():
                success = False
                print("❌ Smoke tests FAILED")
            else:
                print("✅ Smoke tests PASSED")
        
        # Check coverage gate (only if unit tests ran)
        if 'unit' in test_types:
            if not self.check_coverage_gate():
                success = False
        
        # Record total duration
        self.results['total_duration'] = time.time() - start_time
        
        # Print final summary
        self.print_summary()
        
        return success
    
    def print_summary(self):
        """Print test execution summary"""
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        for test_type, result in self.results.items():
            if test_type == 'total_duration':
                continue
            
            if result is None:
                status = "SKIPPED"
                icon = "⏭️"
            elif result:
                status = "PASSED"
                icon = "✅"
            else:
                status = "FAILED"
                icon = "❌"
            
            print(f"{icon} {test_type.replace('_', ' ').title()}: {status}")
        
        print(f"\n⏱️ Total Duration: {self.results['total_duration']:.2f}s")
        
        # Overall status
        if all(r for r in self.results.values() if r is not None):
            print("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT")
            return True
        else:
            print("\n💥 SOME TESTS FAILED - FIX BEFORE DEPLOYMENT")
            return False
    
    def run_ci_tests(self) -> bool:
        """Run tests suitable for CI/CD pipeline"""
        print("🤖 RUNNING CI/CD TEST SUITE")
        
        # Set CI-specific environment variables
        ci_env = {
            'CI': 'true',
            'COVERAGE_PROCESS_START': '.coveragerc'
        }
        
        # Run comprehensive test suite
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=core",
            "--cov=agents",
            "--cov=config",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "--cov-report=html:htmlcov",
            f"--cov-fail-under={self.coverage_threshold}",
            "--junit-xml=test_results.xml",
            "--durations=10",
            "-v",
            "-x"  # Stop on first failure in CI
        ]
        
        result = self.run_command(cmd, env=ci_env)
        
        if result.returncode == 0:
            print("✅ CI/CD Tests PASSED")
        else:
            print("❌ CI/CD Tests FAILED")
        
        return result.returncode == 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CryptoSmartTrader V2 Test Runner")
    
    parser.add_argument(
        '--type', 
        choices=['unit', 'integration', 'contract', 'smoke', 'all'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=80.0,
        help='Minimum coverage percentage required'
    )
    
    parser.add_argument(
        '--ci',
        action='store_true',
        help='Run in CI/CD mode'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    runner.coverage_threshold = args.coverage_threshold
    
    if args.ci:
        success = runner.run_ci_tests()
    else:
        if args.type == 'all':
            test_types = ['unit', 'integration', 'contract', 'smoke']
        else:
            test_types = [args.type]
        
        success = runner.run_all_tests(test_types)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()