@echo off
echo ========================================
echo CryptoSmartTrader V2 - Quick Health Check
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.9+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Running comprehensive system health check...
echo.

REM Run the Python health check script
python system_health_check.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ========================================
    echo HEALTH CHECK FAILED
    echo ========================================
    echo Critical issues found that need to be resolved
    echo before installing CryptoSmartTrader V2.
    echo.
    echo Common solutions:
    echo 1. Install missing Python packages: pip install psutil
    echo 2. Install TA-Lib binary from GitHub releases
    echo 3. Install NVIDIA drivers for GPU acceleration
    echo 4. Ensure sufficient RAM (16GB+) and disk space (20GB+)
    echo.
    echo Run this script again after resolving issues.
) else (
    echo.
    echo ========================================
    echo HEALTH CHECK PASSED
    echo ========================================
    echo Your system is ready for CryptoSmartTrader V2!
    echo.
    echo Next steps:
    echo 1. Run: setup_windows_environment.bat (complete setup)
    echo 2. Or run: install_dependencies.bat (dependencies only)
    echo 3. Then start: start_cryptotrader.bat
)

echo.
pause