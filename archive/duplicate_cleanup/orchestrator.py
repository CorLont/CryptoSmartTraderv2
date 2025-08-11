#!/usr/bin/env python3
"""
Production orchestrator with atomic writes and clear exit codes
"""
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionOrchestrator:
    """Sequential production pipeline orchestrator"""
    
    def __init__(self, run_id: str = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("logs/daily") / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main log file
        self.main_log = self.log_dir / "orchestrator.log"
        
        logger.info(f"Production orchestrator initialized - Run ID: {self.run_id}")
        logger.info(f"Logs directory: {self.log_dir}")
    
    def run_step(self, step_name: str, command: list, timeout: int = 3600) -> int:
        """Run a pipeline step with atomic logging"""
        logger.info(f"Starting step: {step_name}")
        
        # Prepare log files
        stdout_log = self.log_dir / f"{step_name}.log"
        stderr_log = self.log_dir / f"{step_name}_error.log"
        
        try:
            # Run command with timeout
            with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
                result = subprocess.run(
                    command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout,
                    cwd=Path.cwd()
                )
            
            if result.returncode == 0:
                logger.info(f"Step {step_name} completed successfully")
                # Remove empty error log
                if stderr_log.exists() and stderr_log.stat().st_size == 0:
                    stderr_log.unlink()
            else:
                logger.error(f"Step {step_name} failed with exit code {result.returncode}")
                
                # Show last few lines of error log
                if stderr_log.exists() and stderr_log.stat().st_size > 0:
                    with open(stderr_log, 'r') as f:
                        error_lines = f.readlines()
                        logger.error(f"Last error lines:\n{''.join(error_lines[-5:])}")
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            logger.error(f"Step {step_name} timed out after {timeout} seconds")
            return 124  # Timeout exit code
            
        except Exception as e:
            logger.error(f"Step {step_name} failed with exception: {e}")
            return 1
    
    def atomic_write_status(self, status: dict):
        """Atomically write pipeline status"""
        status_file = self.log_dir / "pipeline_status.json"
        temp_file = status_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            # Atomic move
            shutil.move(str(temp_file), str(status_file))
            
        except Exception as e:
            logger.error(f"Failed to write status: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def run_production_pipeline(self) -> int:
        """Run complete production pipeline"""
        logger.info("Starting production pipeline")
        
        pipeline_steps = [
            {
                'name': 'scrape',
                'command': [sys.executable, 'scripts/scrape_all.py'],
                'timeout': 1800,  # 30 minutes
                'description': 'Data scraping from Kraken'
            },
            {
                'name': 'train',
                'command': [sys.executable, 'ml/train_baseline.py'],
                'timeout': 3600,  # 60 minutes
                'description': 'Baseline RF-ensemble training'
            },
            {
                'name': 'predict',
                'command': [sys.executable, 'scripts/predict_all.py'],
                'timeout': 1800,  # 30 minutes
                'description': 'Generate predictions with confidence gating'
            },
            {
                'name': 'evaluate',
                'command': [sys.executable, 'scripts/evaluate.py'],
                'timeout': 600,   # 10 minutes
                'description': 'Evaluation and calibration report'
            }
        ]
        
        # Initialize status
        status = {
            'run_id': self.run_id,
            'started_at': datetime.now().isoformat(),
            'steps': {},
            'overall_status': 'running'
        }
        self.atomic_write_status(status)
        
        # Run each step
        for step in pipeline_steps:
            step_name = step['name']
            logger.info(f"Pipeline step: {step['description']}")
            
            # Update status
            status['steps'][step_name] = {
                'started_at': datetime.now().isoformat(),
                'status': 'running'
            }
            self.atomic_write_status(status)
            
            # Run step
            exit_code = self.run_step(step_name, step['command'], step['timeout'])
            
            # Update status
            status['steps'][step_name].update({
                'completed_at': datetime.now().isoformat(),
                'status': 'success' if exit_code == 0 else 'failed',
                'exit_code': exit_code
            })
            self.atomic_write_status(status)
            
            # Stop on failure
            if exit_code != 0:
                logger.error(f"Pipeline failed at step: {step_name}")
                status['overall_status'] = 'failed'
                status['failed_at'] = step_name
                status['completed_at'] = datetime.now().isoformat()
                self.atomic_write_status(status)
                return exit_code
        
        # All steps completed successfully
        logger.info("All pipeline steps completed successfully")
        status['overall_status'] = 'success'
        status['completed_at'] = datetime.now().isoformat()
        self.atomic_write_status(status)
        
        # Create latest symlink
        latest_link = Path("logs/daily/latest")
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(self.run_id, target_is_directory=True)
        
        return 0
    
    def check_readiness(self) -> bool:
        """Check system readiness before starting"""
        logger.info("Checking system readiness...")
        
        checks = []
        
        # Check Python environment
        try:
            import pandas, numpy, sklearn, ccxt, joblib
            checks.append(("Python packages", True))
        except ImportError as e:
            checks.append(("Python packages", False, str(e)))
        
        # Check environment variables
        required_env = ['KRAKEN_API_KEY', 'KRAKEN_SECRET']
        env_status = all(os.getenv(var) for var in required_env)
        checks.append(("Environment variables", env_status))
        
        # Check directories
        required_dirs = ['data', 'logs', 'exports', 'models']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        checks.append(("Directories", True))
        
        # Print readiness report
        logger.info("=== Readiness Report ===")
        all_ready = True
        for check in checks:
            status = "✓" if check[1] else "✗"
            logger.info(f"{status} {check[0]}")
            if not check[1]:
                all_ready = False
                if len(check) > 2:
                    logger.error(f"  Error: {check[2]}")
        
        return all_ready

def main():
    """Main orchestrator entry point"""
    
    # Parse run ID from environment or generate
    run_id = os.getenv('RUN_ID') or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    orchestrator = ProductionOrchestrator(run_id)
    
    # Check readiness
    if not orchestrator.check_readiness():
        logger.error("System readiness check failed")
        return 2
    
    # Run pipeline
    exit_code = orchestrator.run_production_pipeline()
    
    # Final report
    if exit_code == 0:
        logger.info("=== PRODUCTION PIPELINE SUCCESS ===")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Logs: logs/daily/{run_id}/")
        
        # Check outputs
        pred_file = Path("exports/production/predictions.parquet")
        if pred_file.exists():
            logger.info(f"Predictions: {pred_file}")
        
        report_file = Path("logs/daily/latest.json")
        if report_file.exists():
            logger.info(f"Report: {report_file}")
            
    else:
        logger.error("=== PRODUCTION PIPELINE FAILED ===")
        logger.error(f"Exit code: {exit_code}")
        logger.error(f"Check logs: logs/daily/{run_id}/")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())