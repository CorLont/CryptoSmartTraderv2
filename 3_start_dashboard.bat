@echo off
echo CryptoSmartTrader V2 - Starting Dashboard
echo ======================================

echo Configuring high-performance power plan...
powershell -Command "powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

echo Starting main dashboard on port 5000...
streamlit run app_minimal.py --server.port 5000 --server.address 0.0.0.0

pause
