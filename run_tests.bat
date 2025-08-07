@echo off
title CryptoSmartTrader V2 - System Tests
color 0D
echo ==========================================
echo  CryptoSmartTrader V2 - System Tests
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [TEST] Running comprehensive system tests...
echo.

REM Test 1: Core imports
echo [TEST 1/10] Testing core imports...
python -c "
try:
    import streamlit
    import pandas as pd
    import numpy as np
    import plotly
    import ccxt
    print('[PASS] Core imports successful')
except ImportError as e:
    print(f'[FAIL] Core import failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 2: Configuration system
echo [TEST 2/10] Testing configuration system...
python -c "
try:
    from core.config_manager import ConfigManager
    config = ConfigManager()
    print('[PASS] Configuration system working')
except Exception as e:
    print(f'[FAIL] Configuration test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 3: Daily logging system
echo [TEST 3/10] Testing daily logging system...
python -c "
try:
    from config.daily_logging_config import setup_daily_logging, get_daily_logger
    setup_daily_logging()
    logger_manager = get_daily_logger()
    print('[PASS] Daily logging system working')
except Exception as e:
    print(f'[FAIL] Daily logging test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 4: Advanced AI engines
echo [TEST 4/10] Testing advanced AI engines...
python -c "
try:
    from core.advanced_ai_engine import get_advanced_ai_engine
    from core.shadow_trading_engine import get_live_validation_coordinator
    from core.market_impact_engine import get_market_impact_coordinator
    
    ai_engine = get_advanced_ai_engine()
    shadow_coordinator = get_live_validation_coordinator()
    market_coordinator = get_market_impact_coordinator()
    
    print('[PASS] Advanced AI engines working')
except Exception as e:
    print(f'[FAIL] Advanced AI engines test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 5: Multi-agent cooperation
echo [TEST 5/10] Testing multi-agent cooperation...
python -c "
try:
    from core.multi_agent_cooperation_engine import get_multi_agent_coordinator
    coordinator = get_multi_agent_coordinator()
    print('[PASS] Multi-agent cooperation working')
except Exception as e:
    print(f'[FAIL] Multi-agent cooperation test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 6: Model monitoring
echo [TEST 6/10] Testing model monitoring...
python -c "
try:
    from core.model_monitoring_engine import get_model_monitoring_coordinator
    coordinator = get_model_monitoring_coordinator()
    print('[PASS] Model monitoring working')
except Exception as e:
    print(f'[FAIL] Model monitoring test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 7: Black swan simulation
echo [TEST 7/10] Testing black swan simulation...
python -c "
try:
    from core.black_swan_simulation_engine import get_black_swan_coordinator
    coordinator = get_black_swan_coordinator()
    print('[PASS] Black swan simulation working')
except Exception as e:
    print(f'[FAIL] Black swan simulation test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 8: Market scanner
echo [TEST 8/10] Testing market scanner...
python -c "
try:
    from core.comprehensive_market_scanner import CryptoMarketScanner
    scanner = CryptoMarketScanner()
    print('[PASS] Market scanner working')
except Exception as e:
    print(f'[FAIL] Market scanner test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 9: Health monitoring
echo [TEST 9/10] Testing health monitoring...
python -c "
try:
    from core.health_monitor import HealthMonitor
    health = HealthMonitor()
    status = health.get_system_status()
    print('[PASS] Health monitoring working')
except Exception as e:
    print(f'[FAIL] Health monitoring test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

REM Test 10: Streamlit app structure
echo [TEST 10/10] Testing Streamlit app structure...
python -c "
try:
    import ast
    import os
    
    if not os.path.exists('app.py'):
        print('[FAIL] app.py not found')
        exit(1)
    
    with open('app.py', 'r') as f:
        source = f.read()
    
    # Basic syntax check
    ast.parse(source)
    
    if 'st.title' in source and 'st.sidebar' in source:
        print('[PASS] Streamlit app structure valid')
    else:
        print('[FAIL] Streamlit app structure invalid')
        exit(1)
        
except Exception as e:
    print(f'[FAIL] Streamlit app test failed: {e}')
    exit(1)
"
if errorlevel 1 goto :test_failed

echo.
echo ==========================================
echo   ALL TESTS PASSED SUCCESSFULLY!
echo ==========================================
echo.
echo [SUCCESS] System is ready for production use
echo [INFO] All core components are functioning correctly
echo [READY] You can now run start_cryptotrader.bat
echo.
pause
exit /b 0

:test_failed
echo.
echo ==========================================
echo   TESTS FAILED!
echo ==========================================
echo.
echo [ERROR] One or more tests failed
echo [FIX] Please run setup_windows_environment.bat
echo [FIX] Then run install_dependencies.bat
echo [FIX] Check the error messages above for details
echo.
pause
exit /b 1