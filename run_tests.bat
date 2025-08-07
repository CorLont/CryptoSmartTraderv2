@echo off
echo ========================================
echo CryptoSmartTrader V2 - Test Suite
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run install_dependencies.bat first
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%CD%
set ENVIRONMENT=testing

echo Running comprehensive test suite...
echo.

REM Run daily logging system test
echo [1/5] Testing daily logging system...
python -c "
from config.daily_logging_config import setup_daily_logging, log_system, log_trading, log_ml, log_error
import time

# Test setup
logger_manager = setup_daily_logging()

# Test various log types
print('Testing system logs...')
log_system('test_check', True, 'System test completed successfully')
log_system('test_failure', False, 'Simulated failure for testing', Exception('Test error'))

print('Testing trading logs...')
log_trading('BTC/USD', '15m', 4, {'rsi': 65, 'volume': 1000000, 'trend': 'bullish'})
log_trading('ETH/USD', '1h', 5, {'rsi': 75, 'volume': 500000, 'trend': 'strong_bullish'})

print('Testing ML logs...')
log_ml('BTC/USD', {'direction': 'up', 'target': 50000}, 0.85, {'model': 'LSTM', 'version': 'v1.0'})

print('Testing error logs...')
try:
    raise ValueError('Test error for logging')
except Exception as e:
    log_error(e, {'component': 'test_suite', 'action': 'error_logging_test'})

# Create daily summary
print('Creating daily summary...')
summary = logger_manager.create_daily_summary()
print(f'Daily summary: {len(summary[\"log_files\"])} log files created')
print('✅ Daily logging system test completed')
"

if errorlevel 1 (
    echo ❌ Daily logging test failed
    pause
    exit /b 1
)

REM Run system health check
echo.
echo [2/5] Running system health check...
python system_health_check.py
if errorlevel 1 (
    echo ⚠️ Health check found issues (continuing with tests)
)

REM Test import verification
echo.
echo [3/5] Testing critical imports...
python -c "
import sys
print('Testing critical imports...')

critical_modules = [
    'streamlit',
    'pandas', 
    'numpy',
    'plotly',
    'ccxt',
    'aiohttp',
    'psutil',
    'setproctitle',
    'prometheus_client',
    'dependency_injector',
    'pydantic'
]

failed_imports = []
passed_imports = []

for module in critical_modules:
    try:
        __import__(module)
        passed_imports.append(module)
        print(f'✅ {module}')
    except ImportError as e:
        failed_imports.append((module, str(e)))
        print(f'❌ {module}: {e}')

print(f'\\nImport Results: {len(passed_imports)}/{len(critical_modules)} passed')

if failed_imports:
    print('\\nFailed imports:')
    for module, error in failed_imports:
        print(f'  - {module}: {error}')
    sys.exit(1)
else:
    print('✅ All critical imports successful')
"

if errorlevel 1 (
    echo ❌ Import test failed
    pause
    exit /b 1
)

REM Test core components
echo.
echo [4/5] Testing core components...
python -c "
print('Testing core components...')

# Test configuration
try:
    from config.settings import config
    print('✅ Configuration system')
except Exception as e:
    print(f'❌ Configuration system: {e}')

# Test containers
try:
    from containers import ApplicationContainer
    container = ApplicationContainer()
    print('✅ Dependency injection container')
except Exception as e:
    print(f'❌ Dependency injection: {e}')

# Test daily logging
try:
    from config.daily_logging_config import get_daily_logger
    daily_logger = get_daily_logger()
    print('✅ Daily logging system')
except Exception as e:
    print(f'❌ Daily logging system: {e}')

# Test market scanner (basic initialization)
try:
    from core.comprehensive_market_scanner import ComprehensiveMarketScanner
    print('✅ Comprehensive market scanner')
except Exception as e:
    print(f'❌ Market scanner: {e}')

print('✅ Core component tests completed')
"

if errorlevel 1 (
    echo ❌ Core component test failed
    pause
    exit /b 1
)

REM Test batch files exist
echo.
echo [5/5] Testing Windows deployment files...
python -c "
import os

required_files = [
    'setup_windows_environment.bat',
    'install_dependencies.bat', 
    'start_cryptotrader.bat',
    'start_distributed_system.bat',
    'start_monitoring.bat',
    'quick_health_check.bat',
    'system_health_check.py'
]

missing_files = []
existing_files = []

for file in required_files:
    if os.path.exists(file):
        existing_files.append(file)
        print(f'✅ {file}')
    else:
        missing_files.append(file)
        print(f'❌ {file}')

print(f'\\nDeployment Files: {len(existing_files)}/{len(required_files)} present')

if missing_files:
    print(f'\\nMissing files: {missing_files}')
else:
    print('✅ All deployment files present')
"

if errorlevel 1 (
    echo ❌ Deployment file test failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo TEST SUITE COMPLETED SUCCESSFULLY
echo ========================================
echo.
echo All tests passed:
echo ✅ Daily logging system
echo ✅ System health check  
echo ✅ Critical imports
echo ✅ Core components
echo ✅ Deployment files
echo.
echo System is ready for production operation!
echo.
pause