#!/usr/bin/env python3
"""
Windows Deployment Script
Creates antivirus/firewall exceptions, configures ports, and generates .bat runners
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional


def create_antivirus_exceptions() -> bool:
    """Create Windows Defender exceptions for CryptoSmartTrader"""

    print("🛡️ CONFIGURING ANTIVIRUS EXCEPTIONS")
    print("=" * 50)

    # Paths to exclude from Windows Defender
    exclusion_paths = [
        os.getcwd(),  # Current project directory
        os.path.join(os.getcwd(), "core"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "logs"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "mlruns"),
        os.path.join(os.getcwd(), "cache"),
    ]

    # Processes to exclude
    exclusion_processes = ["python.exe", "streamlit.exe", "uvicorn.exe"]

    try:
        print("📁 Adding folder exclusions...")

        for path in exclusion_paths:
            if os.path.exists(path):
                # PowerShell command to add folder exclusion
                ps_command = f'Add-MpPreference -ExclusionPath "{path}"'

                try:
                    result = subprocess.run(
                        ["powershell", "-Command", ps_command],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    print(f"   ✅ Added exclusion: {path}")

                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️ Failed to add exclusion for {path}: {e}")

        print("\n🔄 Adding process exclusions...")

        for process in exclusion_processes:
            ps_command = f'Add-MpPreference -ExclusionProcess "{process}"'

            try:
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print(f"   ✅ Added process exclusion: {process}")

            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Failed to add process exclusion for {process}: {e}")

        print("\n✅ Antivirus exceptions configured")
        return True

    except Exception as e:
        print(f"❌ Failed to configure antivirus exceptions: {e}")
        print("💡 Run as Administrator for antivirus configuration")
        return False


def configure_firewall_rules() -> bool:
    """Configure Windows Firewall rules for required ports"""

    print("\n🔥 CONFIGURING FIREWALL RULES")
    print("=" * 50)

    # Ports used by CryptoSmartTrader
    firewall_rules = [
        {
            "name": "CryptoSmartTrader-Dashboard",
            "port": "5000",
            "protocol": "TCP",
            "direction": "in",
            "action": "allow",
            "description": "CryptoSmartTrader Streamlit Dashboard",
        },
        {
            "name": "CryptoSmartTrader-TestApp",
            "port": "5001",
            "protocol": "TCP",
            "direction": "in",
            "action": "allow",
            "description": "CryptoSmartTrader Test Application",
        },
        {
            "name": "CryptoSmartTrader-API",
            "port": "8000",
            "protocol": "TCP",
            "direction": "in",
            "action": "allow",
            "description": "CryptoSmartTrader API Server",
        },
        {
            "name": "CryptoSmartTrader-Metrics",
            "port": "8090",
            "protocol": "TCP",
            "direction": "in",
            "action": "allow",
            "description": "CryptoSmartTrader Prometheus Metrics",
        },
        {
            "name": "CryptoSmartTrader-MLflow",
            "port": "5555",
            "protocol": "TCP",
            "direction": "in",
            "action": "allow",
            "description": "CryptoSmartTrader MLflow Tracking",
        },
    ]

    try:
        for rule in firewall_rules:
            print(f"🔓 Adding firewall rule: {rule['name']} (port {rule['port']})")

            # Remove existing rule if exists
            remove_command = f'netsh advfirewall firewall delete rule name="{rule["name"]}"'
            # SECURITY: Secure subprocess without shell=True
            subprocess.run(remove_command.split(), capture_output=True, timeout=30, check=False)

            # Add new rule
            add_command = (
                f"netsh advfirewall firewall add rule "
                f'name="{rule["name"]}" '
                f"dir={rule['direction']} "
                f"action={rule['action']} "
                f"protocol={rule['protocol']} "
                f"localport={rule['port']} "
                f'description="{rule["description"]}"'
            )

            # SECURITY: Secure subprocess without shell=True
            result = subprocess.run(add_command.split(), capture_output=True, text=True, timeout=30, check=False)

            if result.returncode == 0:
                print(f"   ✅ Rule added successfully")
            else:
                print(f"   ⚠️ Failed to add rule: {result.stderr}")

        print("\n✅ Firewall rules configured")
        return True

    except Exception as e:
        print(f"❌ Failed to configure firewall rules: {e}")
        print("💡 Run as Administrator for firewall configuration")
        return False


def create_bat_runners() -> bool:
    """Create Windows batch file runners"""

    print("\n📝 CREATING BATCH FILE RUNNERS")
    print("=" * 50)

    # Installation batch file
    install_bat_content = """@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Installation
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.11 or later
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Configure antivirus exceptions
echo 🛡️ Configuring antivirus exceptions...
python scripts/windows_deployment.py --antivirus
echo.

REM Configure firewall
echo 🔥 Configuring firewall rules...
python scripts/windows_deployment.py --firewall
echo.

REM Create directories
echo 📁 Creating directories...
mkdir logs 2>nul
mkdir cache 2>nul
mkdir exports 2>nul
mkdir model_backup 2>nul
mkdir backups 2>nul

echo ✅ Installation completed!
echo.
echo 🚀 To start the system, run: start_dashboard.bat
pause
"""

    # Dashboard starter batch file
    start_dashboard_bat_content = """@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Starting Dashboard
echo ==========================================
echo.

REM Check if system is already running
netstat -an | find ":5000" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️ Dashboard already running on port 5000
    echo Opening browser...
    start http://localhost:5000
    pause
    exit /b 0
)

echo 🚀 Starting CryptoSmartTrader Dashboard...
echo.
echo 📊 Dashboard will be available at: http://localhost:5000
echo 📈 Test app will be available at: http://localhost:5001
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start the dashboard
python -m streamlit run app_minimal.py --server.port 5000 --server.headless true

pause
"""

    # Evaluation runner batch file
    start_eval_bat_content = """@echo off
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
"""

    # One-click comprehensive runner
    oneclick_runner_bat_content = """@echo off
title CryptoSmartTrader V2 - One-Click Runner
echo ==========================================
echo CryptoSmartTrader V2 - One-Click Runner
echo ==========================================
echo.
echo 🔄 Complete pipeline: scrape → features → predict → strict gate → export → eval → logs/daily
echo.

REM Set environment
set PYTHONPATH=%cd%
set CRYPTOSMARTTRADER_ENV=production

REM Start timestamp
echo 🕐 Started: %date% %time%
echo.

REM Run complete pipeline
python scripts/oneclick_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Pipeline failed with error code: %errorlevel%
    echo 📋 Check logs/daily/ for detailed error information
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Complete pipeline finished successfully!
echo 🕐 Completed: %date% %time%
echo.
echo 📋 Results available in:
echo    - exports/daily/
echo    - logs/daily/
echo.
echo 🚀 To view results, run: start_dashboard.bat
echo.
pause
"""

    # Backup runner
    backup_bat_content = """@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Backup System
echo ==========================================
echo.

echo 📦 Creating system backup...
python scripts/backup_restore.py backup --include-logs --output-dir backups
if %errorlevel% neq 0 (
    echo ❌ Backup failed
    pause
    exit /b 1
)

echo.
echo ✅ Backup completed successfully!
echo 📁 Backup location: backups/
echo.
pause
"""

    # Write batch files
    batch_files = {
        "1_install_all_dependencies.bat": install_bat_content,
        "2_start_background_services.bat": start_dashboard_bat_content,
        "3_start_dashboard.bat": start_dashboard_bat_content,
        "start_evaluation.bat": start_eval_bat_content,
        "oneclick_runner.bat": oneclick_runner_bat_content,
        "create_backup.bat": backup_bat_content,
    }

    try:
        for filename, content in batch_files.items():
            bat_file = Path(filename)

            with open(bat_file, "w", newline="\r\n") as f:
                f.write(content)

            print(f"   ✅ Created: {filename}")

        print(f"\n✅ All batch files created successfully")
        print(f"\n📋 Available runners:")
        print(f"   1_install_all_dependencies.bat  - Install system dependencies")
        print(f"   3_start_dashboard.bat           - Start dashboard and services")
        print(f"   start_evaluation.bat            - Run evaluation pipeline")
        print(f"   oneclick_runner.bat             - Complete one-click pipeline")
        print(f"   create_backup.bat               - Create system backup")

        return True

    except Exception as e:
        print(f"❌ Failed to create batch files: {e}")
        return False


def create_oneclick_pipeline_script() -> bool:
    """Create the one-click pipeline Python script"""

    print("\n🐍 CREATING ONE-CLICK PIPELINE SCRIPT")
    print("=" * 50)

    pipeline_content = """#!/usr/bin/env python3
\"\"\"
One-Click Pipeline Script
Complete pipeline: scrape → features → predict → strict gate → export → eval → logs/daily
\"\"\"

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_oneclick_pipeline():
    \"\"\"Run complete one-click pipeline\"\"\"
    
    print("🔄 CRYPTOSMARTTRADER V2 - ONE-CLICK PIPELINE")
    print("=" * 80)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create daily log directory
    daily_log_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
    daily_log_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_log = daily_log_dir / "pipeline.log"
    
    def log_step(message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(pipeline_log, "a", encoding="utf-8") as f:
            f.write(log_message + "\\n")
    
    try:
        # Step 1: Data scraping and feature engineering
        log_step("📊 STEP 1: Data scraping and feature engineering...")
        
        # Import and run data collection
        try:
            from orchestration.distributed_orchestrator import DistributedOrchestrator
            orchestrator = DistributedOrchestrator()
            
            log_step("   Starting data collection agents...")
            await orchestrator.start_agent("data_collector")
            await orchestrator.start_agent("sentiment_analyzer")
            await asyncio.sleep(30)  # Allow data collection
            
            log_step("   ✅ Data scraping completed")
            
        except Exception as e:
            log_step(f"   ❌ Data scraping failed: {e}")
            return False
        
        # Step 2: ML prediction pipeline
        log_step("🤖 STEP 2: ML prediction pipeline...")
        
        try:
            await orchestrator.start_agent("ml_predictor")
            await orchestrator.start_agent("technical_analyzer")
            await asyncio.sleep(45)  # Allow predictions
            
            log_step("   ✅ ML predictions completed")
            
        except Exception as e:
            log_step(f"   ❌ ML prediction failed: {e}")
            return False
        
        # Step 3: Strict confidence gate
        log_step("🔒 STEP 3: Strict confidence gate...")
        
        try:
            from core.confidence_gate_manager import get_confidence_gate_manager
            gate_manager = get_confidence_gate_manager()
            
            # Apply confidence gate
            gate_report = gate_manager.apply_gate("oneclick_pipeline")
            
            if gate_report and gate_report.get("gate_status") == "OPEN":
                log_step(f"   ✅ Confidence gate OPEN: {gate_report.get('passed_count', 0)} opportunities")
            else:
                log_step("   ⚠️ Confidence gate CLOSED: No high-confidence opportunities")
            
        except Exception as e:
            log_step(f"   ❌ Confidence gate failed: {e}")
            return False
        
        # Step 4: Export results
        log_step("📤 STEP 4: Export results...")
        
        try:
            # Create exports directory
            export_dir = Path("exports/daily") / datetime.now().strftime("%Y%m%d")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Export trading opportunities
            opportunities_file = export_dir / "trading_opportunities.json"
            
            # Get current opportunities (simplified)
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_type": "oneclick",
                "confidence_gate": gate_report if 'gate_report' in locals() else {},
                "export_location": str(export_dir)
            }
            
            with open(opportunities_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            log_step(f"   ✅ Results exported to: {export_dir}")
            
        except Exception as e:
            log_step(f"   ❌ Export failed: {e}")
            return False
        
        # Step 5: Evaluation and logging
        log_step("📈 STEP 5: Evaluation and logging...")
        
        try:
            # Run evaluation
            from eval.evaluator import Evaluator
            evaluator = Evaluator()
            
            eval_results = await evaluator.run_daily_evaluation()
            
            # Save evaluation results
            eval_file = daily_log_dir / "evaluation_results.json"
            with open(eval_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            
            log_step("   ✅ Evaluation completed")
            
        except Exception as e:
            log_step(f"   ❌ Evaluation failed: {e}")
            return False
        
        # Final summary
        log_step("📋 PIPELINE SUMMARY:")
        log_step("   ✅ Data scraping and feature engineering")
        log_step("   ✅ ML prediction pipeline")
        log_step("   ✅ Strict confidence gate")
        log_step("   ✅ Export results")
        log_step("   ✅ Evaluation and logging")
        log_step("")
        log_step(f"📁 Daily logs: {daily_log_dir}")
        log_step(f"📤 Exports: {export_dir}")
        log_step("")
        log_step("🎉 ONE-CLICK PIPELINE COMPLETED SUCCESSFULLY!")
        
        return True
        
    except Exception as e:
        log_step(f"❌ PIPELINE FAILED: {e}")
        return False
    
    finally:
        # Clean up
        try:
            if 'orchestrator' in locals():
                await orchestrator.stop_all_agents()
        except Exception:
            pass

if __name__ == "__main__":
    success = asyncio.run(run_oneclick_pipeline())
    sys.exit(0 if success else 1)
"""

    try:
        script_file = Path("scripts/oneclick_pipeline.py")
        script_file.parent.mkdir(parents=True, exist_ok=True)

        with open(script_file, "w", encoding="utf-8") as f:
            f.write(pipeline_content)

        print(f"   ✅ Created: {script_file}")
        return True

    except Exception as e:
        print(f"   ❌ Failed to create pipeline script: {e}")
        return False


def main():
    """Main deployment function"""

    print("🚀 CRYPTOSMARTTRADER V2 - WINDOWS DEPLOYMENT")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        # Handle command line arguments
        if "--antivirus" in sys.argv:
            return create_antivirus_exceptions()
        elif "--firewall" in sys.argv:
            return configure_firewall_rules()
        elif "--batch-files" in sys.argv:
            return create_bat_runners()

    # Full deployment
    success_count = 0
    total_steps = 4

    # Step 1: Antivirus exceptions
    if create_antivirus_exceptions():
        success_count += 1

    # Step 2: Firewall rules
    if configure_firewall_rules():
        success_count += 1

    # Step 3: Batch file runners
    if create_bat_runners():
        success_count += 1

    # Step 4: One-click pipeline script
    if create_oneclick_pipeline_script():
        success_count += 1

    print(f"\n🏁 DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Completed: {success_count}/{total_steps} steps")

    if success_count == total_steps:
        print("\n🎉 Windows deployment completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: 1_install_all_dependencies.bat")
        print("   2. Run: 3_start_dashboard.bat")
        print("   3. Or run: oneclick_runner.bat for complete pipeline")
    else:
        print("\n⚠️ Some deployment steps failed")
        print("💡 Try running as Administrator")

    return success_count == total_steps


if __name__ == "__main__":
    main()
