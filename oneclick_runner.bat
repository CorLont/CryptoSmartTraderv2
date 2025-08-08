@echo off
echo CryptoSmartTrader V2 - One-Click Complete Pipeline
echo ===============================================

echo Running complete system validation...
python core/system_validator.py

echo Running workstation optimization...
python core/workstation_optimizer.py

echo Generating daily health report...
python core/daily_health_dashboard.py

echo Starting complete pipeline...
echo 1. Data collection and validation
echo 2. ML prediction generation
echo 3. Strict confidence filtering
echo 4. Risk assessment
echo 5. Trading opportunity export

python -c "
import sys, os
sys.path.append('.')
from orchestration.strict_gate import run_strict_orchestration
result = run_strict_orchestration()
print(f'Pipeline completed: {result}')
"

echo Pipeline execution completed!
echo Check logs/daily/%date:~-4,4%%date:~-10,2%%date:~-7,2%/ for results
pause
