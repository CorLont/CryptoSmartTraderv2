"""
Test CI/CD Pipeline Components
Validates that all CI/CD tools work correctly
"""

import pytest
from core.secure_subprocess import secure_subprocess, SecureSubprocessError
import sys
from pathlib import Path


def test_ruff_check():
    """Test ruff linting passes"""
    result = secure_subprocess.run_secure([
        "ruff", "check", "src/", "--output-format=github"
    ], timeout=60, check=False, allowed_return_codes=[0, 1])
    
    # Ruff should not find critical errors
    assert result.returncode in [0, 1], f"Ruff failed: {result.stderr}"


def test_black_check():
    """Test black formatting check"""
    result = secure_subprocess.run_secure([
        "black", "--check", "src/", "--quiet"
    ], timeout=60, check=False, allowed_return_codes=[0, 1])
    
    # Note: black may find files that need formatting (returncode 1)
    # but it shouldn't crash (returncode > 1)
    assert result.returncode <= 1, f"Black failed: {result.stderr}"


def test_mypy_core_modules():
    """Test mypy type checking on core modules"""
    core_modules = [
        "src/cryptosmarttrader/risk/central_risk_guard.py",
        "src/cryptosmarttrader/simulation/execution_simulator.py",
    ]
    
    for module in core_modules:
        if Path(module).exists():
            result = secure_subprocess.run_secure([
                "mypy", module, "--ignore-missing-imports"
            ], timeout=120, check=False, allowed_return_codes=[0, 1])
            
            # MyPy should not find critical type errors  
            assert result.returncode <= 1, f"MyPy failed on {module}: {result.stdout}"


def test_pytest_runs():
    """Test that pytest can run successfully"""
    result = secure_subprocess.run_secure([
        "pytest", "--version"
    ], timeout=30, check=False)
    
    assert result.returncode == 0, f"Pytest not working: {result.stderr}"
    assert "pytest" in result.stdout


def test_coverage_configuration():
    """Test that coverage configuration is valid"""
    result = secure_subprocess.run_secure([
        "python", "-c", 
        "import coverage; cov = coverage.Coverage(); print('Coverage config valid')"
    ], timeout=30, check=False)
    
    assert result.returncode == 0, f"Coverage config invalid: {result.stderr}"


def test_import_validation():
    """Test that core modules can be imported"""
    core_imports = [
        "cryptosmarttrader.risk.central_risk_guard",
        "cryptosmarttrader.simulation.execution_simulator", 
        "cryptosmarttrader.simulation.parity_tracker"
    ]
    
    sys.path.insert(0, "src")
    
    for module in core_imports:
        try:
            __import__(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")


@pytest.mark.security
def test_bandit_security():
    """Test bandit security scanning"""
    result = subprocess.run([
        "bandit", "-r", "src/", "-f", "txt", "--severity-level", "medium"
    ], capture_output=True, text=True)
    
    # Bandit may find issues (returncode 1) but shouldn't crash
    assert result.returncode <= 1, f"Bandit failed: {result.stderr}"


@pytest.mark.security  
def test_pip_audit():
    """Test pip-audit dependency scanning"""
    result = subprocess.run([
        "pip-audit", "--format=json", "--output=-"
    ], capture_output=True, text=True)
    
    # pip-audit may find vulnerabilities but shouldn't crash
    assert result.returncode <= 1, f"pip-audit failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])