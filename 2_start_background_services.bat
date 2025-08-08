@echo off
echo CryptoSmartTrader V2 - Starting Background Services
echo ================================================

echo Starting Prometheus metrics server...
start /B python -c "from prometheus_client import start_http_server; start_http_server(8090); import time; time.sleep(3600)"

echo Starting health monitoring...
start /B python core/daily_health_dashboard.py

echo Starting MLflow tracking server...
start /B mlflow server --host 0.0.0.0 --port 5555 --backend-store-uri sqlite:///mlflow.db

echo Background services started!
echo Check ports: 8090 (metrics), 5555 (MLflow)
pause
