@echo off
echo ===================================================
echo CryptoSmartTrader V2 - Starting Background Services
echo ===================================================

REM Activate virtual environment
echo [1/6] Activating virtual environment...
call .venv\Scripts\activate 2>nul
if errorlevel 1 (
    echo ⚠️ Virtual environment not found. Run 1_install_all_dependencies.bat first
    pause
    exit /b 1
)

REM Stop existing services first
echo.
echo [2/6] Stopping existing background services...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Metrics Server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Health Monitor*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq MLflow Server*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Production Orchestrator*" 2>nul
timeout /t 2 /nobreak >nul

REM Set environment variables
echo.
echo [3/6] Setting environment variables...
set PYTHONPATH=%CD%
set CRYPTOSMARTTRADER_ENV=production

REM Start Prometheus metrics server
echo.
echo [4/6] Starting Prometheus metrics server on port 8090...
start "Metrics Server" /MIN python -c "from prometheus_client import start_http_server; import time; start_http_server(8090); print('Prometheus metrics server started on port 8090'); time.sleep(86400)"

REM Start health monitoring system
echo.
echo [5/6] Starting health monitoring system...
if exist src/cryptosmarttrader/core/system_health_monitor.py (
    start "Health Monitor" /MIN python -m src.cryptosmarttrader.core.system_health_monitor
) else if exist core/system_health_monitor.py (
    start "Health Monitor" /MIN python core/system_health_monitor.py
) else (
    echo ⚠️ Health monitor not found, skipping
)

REM Start centralized monitoring
echo.
echo [6/6] Starting centralized monitoring...
if exist src/cryptosmarttrader/observability/centralized_prometheus.py (
    start "Centralized Monitoring" /MIN python -c "from src.cryptosmarttrader.observability.centralized_prometheus import PrometheusMetrics; p = PrometheusMetrics(); p.start_server(8091); import time; time.sleep(86400)"
) else (
    echo ⚠️ Centralized monitoring not found, skipping
)

REM Verify services
echo.
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ===================================================
echo ✅ BACKGROUND SERVICES STARTED!
echo ===================================================
echo.
echo Active Services:
echo   • Prometheus Metrics      - http://localhost:8090
echo   • Health Monitor          - Running in background
echo   • Production Orchestrator - Running in background
echo.
echo Service Status:
netstat -an | findstr :8090 >nul && echo "  ✓ Metrics server (port 8090)" || echo "  ⚠️ Metrics server not responding"
echo.
echo Log files available in: logs\
echo.
echo To stop services, run:
echo   taskkill /F /IM python.exe
echo.
echo Next step: Run 3_start_dashboard.bat
pause
