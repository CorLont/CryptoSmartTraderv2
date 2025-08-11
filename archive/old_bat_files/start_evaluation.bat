@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Running Evaluation
echo ==========================================
echo.

echo 🔄 Starting complete evaluation pipeline...
echo.

REM Set environment variables
set PYTHONPATH=%cd%
set CRYPTOSMARTTRADER_ENV=production

REM Run evaluation pipeline
echo 📊 Step 1: Data scraping and feature engineering...
python scripts/run_evaluation_pipeline.py --step scrape
if %errorlevel% neq 0 (
    echo ❌ Data scraping failed
    pause
    exit /b 1
)

echo 🤖 Step 2: ML prediction pipeline...
python scripts/run_evaluation_pipeline.py --step predict
if %errorlevel% neq 0 (
    echo ❌ Prediction pipeline failed
    pause
    exit /b 1
)

echo 🔒 Step 3: Strict confidence gate...
python scripts/run_evaluation_pipeline.py --step gate
if %errorlevel% neq 0 (
    echo ❌ Confidence gate failed
    pause
    exit /b 1
)

echo 📤 Step 4: Export results...
python scripts/run_evaluation_pipeline.py --step export
if %errorlevel% neq 0 (
    echo ❌ Export failed
    pause
    exit /b 1
)

echo 📈 Step 5: Evaluation and logging...
python scripts/run_evaluation_pipeline.py --step eval
if %errorlevel% neq 0 (
    echo ❌ Evaluation failed
    pause
    exit /b 1
)

echo.
echo ✅ Complete evaluation pipeline finished!
echo.
echo 📋 Check exports/ directory for results
echo 📊 Check logs/ directory for detailed logs
echo.
pause
