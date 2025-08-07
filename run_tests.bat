@echo off
REM CryptoSmartTrader V2 - Test Suite Runner
REM Comprehensive testing for robustness and reliability

echo ================================
echo CryptoSmartTrader V2 Test Suite
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Installing test dependencies...
pip install pytest pytest-asyncio pytest-cov

echo.
echo [2/4] Running unit tests...
python -m pytest tests/test_production_systems.py -v --tb=short

echo.
echo [3/4] Running integration tests...
python -m pytest tests/test_production_systems.py::TestIntegrationScenarios -v

echo.
echo [4/4] Running performance tests...
python -m pytest tests/test_production_systems.py::TestPerformance -v

echo.
echo ================================
echo Test Results Summary
echo ================================
echo.

REM Generate coverage report if pytest-cov is available
echo Generating coverage report...
python -m pytest tests/test_production_systems.py --cov=core --cov-report=html --cov-report=term

echo.
echo Coverage report generated in htmlcov/index.html
echo.
echo Test suite completed!
pause