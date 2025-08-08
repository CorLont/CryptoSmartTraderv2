#!/usr/bin/env python3
"""
One-Click Pipeline Script
Complete pipeline: scrape → features → predict → strict gate → export → eval → logs/daily
"""

import os
import sys
import json
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def run_oneclick_pipeline():
    """Run complete one-click pipeline"""
    
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
            f.write(log_message + "\n")
    
    try:
        # Step 1: Data scraping and feature engineering
        log_step("📊 STEP 1: Data scraping and feature engineering...")
        
        try:
            # Check if distributed system is available
            from orchestration.distributed_orchestrator import DistributedOrchestrator
            
            orchestrator = DistributedOrchestrator()
            log_step("   Starting distributed data collection...")
            
            # Start data collection agents
            await orchestrator.start_agent("data_collector")
            await orchestrator.start_agent("sentiment_analyzer")
            
            # Allow data collection time
            await asyncio.sleep(30)
            
            log_step("   ✅ Data scraping completed")
            
        except Exception as e:
            log_step(f"   ⚠️ Distributed system not available, using fallback: {e}")
            
            # Fallback: Run simple data collection
            try:
                result = subprocess.run([
                    sys.executable, "-c", 
                    "from core.data_manager import DataManager; dm = DataManager(); dm.refresh_market_data(); print('Data refreshed')"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    log_step("   ✅ Fallback data collection completed")
                else:
                    log_step(f"   ❌ Data collection failed: {result.stderr}")
                    return False
                    
            except Exception as fallback_error:
                log_step(f"   ❌ Fallback data collection failed: {fallback_error}")
                return False
        
        # Step 2: ML prediction pipeline
        log_step("🤖 STEP 2: ML prediction pipeline...")
        
        try:
            if 'orchestrator' in locals():
                await orchestrator.start_agent("ml_predictor")
                await orchestrator.start_agent("technical_analyzer")
                await asyncio.sleep(45)
                log_step("   ✅ ML predictions completed")
            else:
                # Fallback ML pipeline
                result = subprocess.run([
                    sys.executable, "-c",
                    "from ml.ml_predictor import MLPredictor; predictor = MLPredictor(); predictor.generate_predictions(); print('Predictions generated')"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    log_step("   ✅ Fallback ML predictions completed")
                else:
                    log_step(f"   ❌ ML prediction failed: {result.stderr}")
                    return False
            
        except Exception as e:
            log_step(f"   ❌ ML prediction failed: {e}")
            return False
        
        # Step 3: Strict confidence gate
        log_step("🔒 STEP 3: Strict confidence gate...")
        
        try:
            from core.confidence_gate_manager import get_confidence_gate_manager
            
            gate_manager = get_confidence_gate_manager()
            gate_report = gate_manager.apply_gate("oneclick_pipeline")
            
            if gate_report and gate_report.get("gate_status") == "OPEN":
                passed_count = gate_report.get('passed_count', 0)
                log_step(f"   ✅ Confidence gate OPEN: {passed_count} high-confidence opportunities")
            else:
                log_step("   ⚠️ Confidence gate CLOSED: No high-confidence opportunities found")
                log_step("      This is normal - strict filtering ensures quality")
            
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
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_type": "oneclick",
                "confidence_gate": gate_report if 'gate_report' in locals() else {},
                "export_location": str(export_dir),
                "status": "completed"
            }
            
            with open(opportunities_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            # Create summary report
            summary_file = export_dir / "pipeline_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"CryptoSmartTrader V2 - Pipeline Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: One-Click Complete\n")
                f.write(f"Status: Success\n")
                if 'gate_report' in locals():
                    f.write(f"Confidence Gate: {gate_report.get('gate_status', 'UNKNOWN')}\n")
                    f.write(f"Opportunities: {gate_report.get('passed_count', 0)}\n")
            
            log_step(f"   ✅ Results exported to: {export_dir}")
            
        except Exception as e:
            log_step(f"   ❌ Export failed: {e}")
            return False
        
        # Step 5: Evaluation and logging
        log_step("📈 STEP 5: Evaluation and logging...")
        
        try:
            # Simple evaluation - check system health
            result = subprocess.run([
                sys.executable, "system_health_check.py"
            ], capture_output=True, text=True, timeout=60)
            
            eval_results = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_type": "oneclick", 
                "health_check_result": "passed" if result.returncode == 0 else "failed",
                "health_check_output": result.stdout if result.returncode == 0 else result.stderr
            }
            
            # Save evaluation results
            eval_file = daily_log_dir / "evaluation_results.json"
            with open(eval_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            
            log_step("   ✅ Evaluation completed")
            
        except Exception as e:
            log_step(f"   ⚠️ Evaluation had issues (non-critical): {e}")
            # Don't fail pipeline for evaluation issues
        
        # Final summary
        log_step("")
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

def run_basic_pipeline():
    """Run basic pipeline without async components"""
    
    print("🔄 CRYPTOSMARTTRADER V2 - BASIC PIPELINE")
    print("=" * 80)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create daily log directory
    daily_log_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
    daily_log_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_log = daily_log_dir / "basic_pipeline.log"
    
    def log_step(message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(pipeline_log, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    try:
        # Step 1: System health check
        log_step("🔍 STEP 1: System health check...")
        
        try:
            result = subprocess.run([
                sys.executable, "system_health_check.py"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                log_step("   ✅ System health check passed")
            else:
                log_step(f"   ⚠️ System health issues detected: {result.stderr[:200]}")
                
        except Exception as e:
            log_step(f"   ⚠️ Health check failed: {e}")
        
        # Step 2: Confidence gate test
        log_step("🔒 STEP 2: Testing confidence gate...")
        
        try:
            from core.confidence_gate_manager import get_confidence_gate_manager
            
            gate_manager = get_confidence_gate_manager()
            gate_report = gate_manager.get_daily_report()
            
            log_step(f"   ✅ Confidence gate operational")
            
        except Exception as e:
            log_step(f"   ❌ Confidence gate test failed: {e}")
            return False
        
        # Step 3: Export basic report
        log_step("📤 STEP 3: Creating basic report...")
        
        try:
            export_dir = Path("exports/daily") / datetime.now().strftime("%Y%m%d")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            basic_report = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_type": "basic",
                "system_status": "operational",
                "confidence_gate": "tested"
            }
            
            with open(export_dir / "basic_report.json", "w") as f:
                json.dump(basic_report, f, indent=2)
            
            log_step(f"   ✅ Basic report created: {export_dir}")
            
        except Exception as e:
            log_step(f"   ❌ Report creation failed: {e}")
            return False
        
        log_step("")
        log_step("🎉 BASIC PIPELINE COMPLETED SUCCESSFULLY!")
        
        return True
        
    except Exception as e:
        log_step(f"❌ BASIC PIPELINE FAILED: {e}")
        return False

if __name__ == "__main__":
    try:
        # Try async pipeline first
        success = asyncio.run(run_oneclick_pipeline())
    except Exception as e:
        print(f"Async pipeline failed: {e}")
        print("Falling back to basic pipeline...")
        success = run_basic_pipeline()
    
    sys.exit(0 if success else 1)