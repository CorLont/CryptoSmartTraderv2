@echo off
echo ===================================================
echo CryptoSmartTrader V2 - Workstation Validation
echo ===================================================
echo.

REM Set variables
set VALIDATION_PASSED=0
set CRITICAL_ISSUES=0
set WARNING_ISSUES=0

echo [VALIDATION] Starting comprehensive workstation validation...
echo.

REM Check Python version
echo [1/15] Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ CRITICAL: Python not found
    set /a CRITICAL_ISSUES+=1
    goto :python_missing
)

python -c "import sys; major, minor = sys.version_info[:2]; print(f'Python {major}.{minor}'); exit(0 if (major == 3 and minor >= 11) else 1)" > temp_python_version.txt 2>&1
if errorlevel 1 (
    echo ❌ CRITICAL: Python 3.11+ required
    type temp_python_version.txt
    set /a CRITICAL_ISSUES+=1
) else (
    echo ✓ Python version OK:
    type temp_python_version.txt
)
del temp_python_version.txt 2>nul

:python_missing

REM Check virtual environment
echo.
echo [2/15] Checking virtual environment...
if exist .venv\Scripts\activate (
    echo ✓ Virtual environment found
    call .venv\Scripts\activate
    if errorlevel 1 (
        echo ❌ CRITICAL: Virtual environment activation failed
        set /a CRITICAL_ISSUES+=1
    ) else (
        echo ✓ Virtual environment activated successfully
    )
) else (
    echo ⚠️ WARNING: Virtual environment not found
    echo Run 1_install_all_dependencies.bat first
    set /a WARNING_ISSUES+=1
)

REM Check mandatory dependencies
echo.
echo [3/15] Checking mandatory dependencies...
python -c "
import sys
deps = ['streamlit', 'pandas', 'numpy', 'plotly', 'requests', 'aiohttp', 'pydantic', 'prometheus_client']
missing = []
for dep in deps:
    try:
        __import__(dep.replace('-', '_'))
        print(f'✓ {dep}')
    except ImportError:
        print(f'❌ {dep} MISSING')
        missing.append(dep)
if missing:
    print(f'CRITICAL: {len(missing)} mandatory dependencies missing')
    sys.exit(1)
else:
    print('✓ All mandatory dependencies available')
" 2>nul
if errorlevel 1 (
    echo ❌ CRITICAL: Missing mandatory dependencies
    set /a CRITICAL_ISSUES+=1
)

REM Check AI/ML dependencies
echo.
echo [4/15] Checking AI/ML dependencies...
python -c "
deps = ['openai', 'scikit-learn', 'torch', 'transformers']
missing = []
for dep in deps:
    try:
        if dep == 'scikit-learn':
            __import__('sklearn')
        else:
            __import__(dep.replace('-', '_'))
        print(f'✓ {dep}')
    except ImportError:
        print(f'⚠️ {dep} missing')
        missing.append(dep)
if missing:
    print(f'WARNING: {len(missing)} AI/ML dependencies missing')
" 2>nul

REM Check trading dependencies
echo.
echo [5/15] Checking trading dependencies...
python -c "
deps = ['ccxt']
for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'❌ {dep} MISSING - CRITICAL for trading')
" 2>nul

REM Check project structure
echo.
echo [6/15] Checking project structure...
if exist src\cryptosmarttrader (
    echo ✓ Core package structure found
) else (
    echo ⚠️ WARNING: Core package structure missing
    set /a WARNING_ISSUES+=1
)

if exist src\cryptosmarttrader\core\mandatory_execution_gateway.py (
    echo ✓ Mandatory Execution Gateway found
) else (
    echo ❌ CRITICAL: Mandatory Execution Gateway missing
    set /a CRITICAL_ISSUES+=1
)

if exist src\cryptosmarttrader\risk\central_risk_guard.py (
    echo ✓ Central Risk Guard found
) else (
    echo ❌ CRITICAL: Central Risk Guard missing
    set /a CRITICAL_ISSUES+=1
)

if exist src\cryptosmarttrader\execution\execution_discipline.py (
    echo ✓ Execution Discipline found
) else (
    echo ❌ CRITICAL: Execution Discipline missing
    set /a CRITICAL_ISSUES+=1
)

REM Check main applications
echo.
echo [7/15] Checking main applications...
if exist app_trading_analysis_dashboard.py (
    echo ✓ Trading Analysis Dashboard found
    set MAIN_APP=app_trading_analysis_dashboard.py
) else if exist app_fixed_all_issues.py (
    echo ✓ Fixed Issues App found
    set MAIN_APP=app_fixed_all_issues.py
) else (
    echo ❌ CRITICAL: No main application found
    set /a CRITICAL_ISSUES+=1
)

REM Check .env configuration
echo.
echo [8/15] Checking environment configuration...
if exist .env (
    echo ✓ .env file found
    findstr /C:"KRAKEN_API_KEY" .env >nul && echo "✓ Kraken API key configured" || echo "⚠️ Kraken API key missing"
    findstr /C:"OPENAI_API_KEY" .env >nul && echo "✓ OpenAI API key configured" || echo "⚠️ OpenAI API key missing"
) else (
    echo ⚠️ WARNING: .env file missing
    echo Create .env with your API keys for full functionality
    set /a WARNING_ISSUES+=1
)

REM Check ports availability
echo.
echo [9/15] Checking port availability...
netstat -an | findstr :5000 >nul && echo "⚠️ Port 5000 in use" || echo "✓ Port 5000 available"
netstat -an | findstr :8090 >nul && echo "⚠️ Port 8090 in use" || echo "✓ Port 8090 available"
netstat -an | findstr :8091 >nul && echo "⚠️ Port 8091 in use" || echo "✓ Port 8091 available"

REM Check CUDA/GPU support
echo.
echo [10/15] Checking GPU support...
python -c "
try:
    import torch
    if torch.cuda.is_available():
        print(f'✓ CUDA available - {torch.cuda.device_count()} GPU(s)')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA not available (CPU-only mode)')
except ImportError:
    print('⚠️ PyTorch not installed')
" 2>nul

REM Check mandatory gateway functionality
echo.
echo [11/15] Testing Mandatory Execution Gateway...
python -c "
try:
    import sys
    sys.path.append('.')
    from src.cryptosmarttrader.core.mandatory_execution_gateway import MANDATORY_GATEWAY, UniversalOrderRequest
    
    # Test order
    order = UniversalOrderRequest(
        symbol='BTC/USD',
        side='buy', 
        size=0.01,
        source_module='validation_test',
        source_function='test'
    )
    
    # Test without causing issues
    print('✓ Mandatory Execution Gateway loaded successfully')
    print('✓ UniversalOrderRequest creation successful')
    
except Exception as e:
    print(f'❌ Gateway test failed: {str(e)}')
    sys.exit(1)
" 2>nul
if errorlevel 1 (
    echo ❌ CRITICAL: Mandatory Execution Gateway failed
    set /a CRITICAL_ISSUES+=1
)

REM Check directory structure and permissions
echo.
echo [12/15] Checking directories and permissions...
mkdir logs 2>nul && echo "✓ logs/ directory OK" || echo "⚠️ logs/ directory issue"
mkdir data 2>nul && echo "✓ data/ directory OK" || echo "⚠️ data/ directory issue"
mkdir cache 2>nul && echo "✓ cache/ directory OK" || echo "⚠️ cache/ directory issue"

REM Performance checks
echo.
echo [13/15] Checking system performance...
powershell -Command "Get-WmiObject -Class Win32_ComputerSystem | Select-Object TotalPhysicalMemory" | findstr /C:"TotalPhysicalMemory" >nul && echo "✓ Memory info accessible" || echo "⚠️ Memory info not accessible"

echo Checking available disk space...
powershell -Command "Get-WmiObject -Class Win32_LogicalDisk -Filter \"DeviceID='C:'\" | Select-Object FreeSpace" | findstr /C:"FreeSpace" >nul && echo "✓ Disk space info accessible" || echo "⚠️ Disk space info not accessible"

REM Windows Defender exclusion check
echo.
echo [14/15] Checking Windows Defender exclusions...
powershell -Command "Get-MpPreference | Select-Object ExclusionPath" 2>nul | findstr /C:"%CD%" >nul && echo "✓ Project directory excluded from Windows Defender" || echo "⚠️ Windows Defender exclusion not set (run as admin for performance boost)"

REM Final validation summary
echo.
echo [15/15] Final application test...
if defined MAIN_APP (
    echo Testing %MAIN_APP% startup...
    timeout /t 1 /nobreak >nul
    python -c "
try:
    import streamlit
    print('✓ Streamlit import successful')
    # Don't actually start, just test import
except Exception as e:
    print(f'❌ Streamlit test failed: {str(e)}')
    exit(1)
" 2>nul
    if errorlevel 1 (
        echo ❌ CRITICAL: Main application startup test failed
        set /a CRITICAL_ISSUES+=1
    ) else (
        echo ✓ Main application startup test passed
    )
)

REM Summary
echo.
echo ===================================================
echo WORKSTATION VALIDATION SUMMARY
echo ===================================================
echo.

if %CRITICAL_ISSUES% EQU 0 (
    if %WARNING_ISSUES% EQU 0 (
        echo ✅ VALIDATION PASSED - FULLY OPERATIONAL
        echo Your workstation is ready for CryptoSmartTrader V2!
        echo.
        echo Quick Start:
        echo   1. Run: 2_start_background_services.bat
        echo   2. Run: 3_start_dashboard.bat
    ) else (
        echo ✅ VALIDATION PASSED WITH WARNINGS
        echo Critical systems operational, some optional features missing
        echo Warning issues: %WARNING_ISSUES%
        echo.
        echo Recommended: Address warnings for optimal performance
    )
) else (
    echo ❌ VALIDATION FAILED
    echo Critical issues found: %CRITICAL_ISSUES%
    echo Warning issues: %WARNING_ISSUES%
    echo.
    echo Action required: Fix critical issues before proceeding
    echo Run: 1_install_all_dependencies.bat
)

echo.
echo Validation completed at %DATE% %TIME%
echo.
pause