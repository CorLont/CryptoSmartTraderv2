#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - CI/CD Pipeline
Automated testing, performance benchmarks, security scans, and deployment automation
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

class CICDPipeline:
    """Comprehensive CI/CD pipeline for CryptoSmartTrader"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {
            "pipeline_start": datetime.now().isoformat(),
            "stages": {},
            "overall_status": "running",
            "errors": [],
            "warnings": []
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup pipeline logging"""
        log_dir = self.project_root / "logs" / "cicd"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - CICD - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete CI/CD pipeline"""
        self.logger.info("🚀 Starting CryptoSmartTrader CI/CD Pipeline")
        
        pipeline_stages = [
            ("dependency_check", self.check_dependencies),
            ("code_quality", self.run_code_quality_checks),
            ("security_scan", self.run_security_scans),
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("performance_tests", self.run_performance_tests),
            ("system_health_check", self.run_system_health_check),
            ("documentation_check", self.check_documentation),
            ("deployment_validation", self.validate_deployment_readiness)
        ]
        
        for stage_name, stage_function in pipeline_stages:
            self.logger.info(f"🔄 Running stage: {stage_name}")
            stage_start = time.time()
            
            try:
                stage_result = stage_function()
                stage_duration = time.time() - stage_start
                
                self.results["stages"][stage_name] = {
                    "status": "passed" if stage_result.get("success", False) else "failed",
                    "duration": stage_duration,
                    "details": stage_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not stage_result.get("success", False):
                    self.logger.error(f"❌ Stage {stage_name} failed")
                    if stage_result.get("critical", False):
                        self.logger.error("💥 Critical failure - stopping pipeline")
                        break
                else:
                    self.logger.info(f"✅ Stage {stage_name} completed successfully")
                    
            except Exception as e:
                self.logger.error(f"💥 Stage {stage_name} crashed: {e}")
                self.results["stages"][stage_name] = {
                    "status": "error",
                    "duration": time.time() - stage_start,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                break
        
        # Calculate overall status
        self.calculate_overall_status()
        self.results["pipeline_end"] = datetime.now().isoformat()
        
        # Generate report
        self.generate_pipeline_report()
        
        return self.results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check all required dependencies"""
        result = {"success": True, "details": {}, "issues": []}
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        result["details"]["python_version"] = python_version
        
        if sys.version_info < (3, 8):
            result["success"] = False
            result["issues"].append(f"Python version {python_version} too old, need 3.8+")
        
        # Check required packages
        required_packages = [
            "streamlit", "pandas", "numpy", "plotly", "ccxt", "scikit-learn",
            "xgboost", "openai", "aiohttp", "pydantic", "dependency-injector"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                result["details"][f"package_{package}"] = "available"
            except ImportError:
                missing_packages.append(package)
                result["details"][f"package_{package}"] = "missing"
        
        if missing_packages:
            result["success"] = False
            result["critical"] = True
            result["issues"].append(f"Missing packages: {', '.join(missing_packages)}")
        
        # Check file structure
        required_dirs = ["core", "agents", "dashboards", "tests", "logs"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                result["details"][f"dir_{dir_name}"] = "exists"
            else:
                result["success"] = False
                result["issues"].append(f"Missing directory: {dir_name}")
        
        return result
    
    def run_code_quality_checks(self) -> Dict[str, Any]:
        """Run code quality checks"""
        result = {"success": True, "details": {}, "issues": []}
        
        # Check for basic Python syntax
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
        
        if syntax_errors:
            result["success"] = False
            result["critical"] = True
            result["issues"].extend(syntax_errors)
        
        result["details"]["python_files_checked"] = len(python_files)
        result["details"]["syntax_errors"] = len(syntax_errors)
        
        # Check for common code quality issues
        quality_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for obvious issues
                if "print(" in content and "logger" not in content:
                    quality_issues.append(f"{py_file}: Uses print() instead of logging")
                
                if "TODO" in content or "FIXME" in content:
                    quality_issues.append(f"{py_file}: Contains TODO/FIXME comments")
                
                line_count = len(content.split('\n'))
                if line_count > 1000:
                    quality_issues.append(f"{py_file}: Very long file ({line_count} lines)")
                    
            except Exception:
                continue
        
        result["details"]["quality_issues"] = len(quality_issues)
        if quality_issues:
            result["issues"].extend(quality_issues[:10])  # Limit to first 10
        
        return result
    
    def run_security_scans(self) -> Dict[str, Any]:
        """Run basic security scans"""
        result = {"success": True, "details": {}, "issues": []}
        
        # Check for hardcoded secrets
        security_issues = []
        sensitive_patterns = [
            "password", "api_key", "secret", "token", "private_key"
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for pattern in sensitive_patterns:
                    if f'{pattern} = "' in content or f"{pattern} = '" in content:
                        security_issues.append(f"{py_file}: Potential hardcoded {pattern}")
                        
            except Exception:
                continue
        
        result["details"]["security_issues"] = len(security_issues)
        if security_issues:
            result["issues"].extend(security_issues)
            result["success"] = False
        
        # Check for .env file security
        env_file = self.project_root / ".env"
        if env_file.exists():
            result["details"]["env_file_exists"] = True
            # Check if .env is in .gitignore
            gitignore = self.project_root / ".gitignore"
            if gitignore.exists():
                with open(gitignore, 'r') as f:
                    if ".env" not in f.read():
                        result["issues"].append(".env file not in .gitignore")
        
        return result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        result = {"success": True, "details": {}, "issues": []}
        
        try:
            # Run the comprehensive test suite
            test_file = self.project_root / "tests" / "test_comprehensive_system.py"
            
            if test_file.exists():
                import subprocess
                cmd = [sys.executable, str(test_file)]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=self.project_root
                )
                
                result["details"]["test_output"] = process.stdout
                result["details"]["test_errors"] = process.stderr
                result["details"]["return_code"] = process.returncode
                
                if process.returncode != 0:
                    result["success"] = False
                    result["issues"].append("Unit tests failed")
                
                # Parse test results from output
                if "SUCCESS RATE:" in process.stdout:
                    for line in process.stdout.split('\n'):
                        if "SUCCESS RATE:" in line:
                            success_rate = line.split(':')[1].strip().replace('%', '')
                            result["details"]["success_rate"] = float(success_rate)
                            break
                
            else:
                result["issues"].append("Test file not found")
                result["success"] = False
                
        except subprocess.TimeoutExpired:
            result["success"] = False
            result["issues"].append("Tests timed out")
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Test execution error: {e}")
        
        return result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        result = {"success": True, "details": {}, "issues": []}
        
        try:
            # Test core system integration
            sys.path.insert(0, str(self.project_root))
            
            # Test security manager
            try:
                from core.security_manager import SecurityManager
                security_manager = SecurityManager()
                health = security_manager.validate_secrets_health()
                result["details"]["security_manager"] = "operational"
            except Exception as e:
                result["issues"].append(f"Security manager integration failed: {e}")
                result["success"] = False
            
            # Test async coordinator
            try:
                from core.async_coordinator import AsyncCoordinator
                coordinator = AsyncCoordinator()
                health = coordinator.get_system_health()
                result["details"]["async_coordinator"] = "operational"
            except Exception as e:
                result["issues"].append(f"Async coordinator integration failed: {e}")
                result["success"] = False
            
            # Test exception handler
            try:
                from core.exception_handler import ExceptionHandler
                handler = ExceptionHandler()
                stats = handler.get_error_statistics()
                result["details"]["exception_handler"] = "operational"
            except Exception as e:
                result["issues"].append(f"Exception handler integration failed: {e}")
                result["success"] = False
            
            # Test ML differentiators
            try:
                from core.ml_ai_differentiators import get_ml_differentiators_coordinator
                ml_coordinator = get_ml_differentiators_coordinator()
                status = ml_coordinator.get_system_status()
                result["details"]["ml_differentiators"] = "operational"
            except Exception as e:
                result["issues"].append(f"ML differentiators integration failed: {e}")
                result["success"] = False
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Integration test setup failed: {e}")
        
        return result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        result = {"success": True, "details": {}, "issues": []}
        
        try:
            start_time = time.time()
            
            # Test system startup time
            sys.path.insert(0, str(self.project_root))
            
            startup_start = time.time()
            from core.async_coordinator import get_async_coordinator
            coordinator = get_async_coordinator()
            startup_time = time.time() - startup_start
            
            result["details"]["startup_time"] = startup_time
            
            if startup_time > 10:  # 10 seconds threshold
                result["issues"].append(f"Slow startup time: {startup_time:.2f}s")
            
            # Test concurrent task performance
            task_start = time.time()
            task_ids = []
            
            for i in range(10):
                task_id = coordinator.submit_task(
                    task_name=f"perf_test_{i}",
                    function=lambda x=i: x * 2,
                    timeout=1.0
                )
                task_ids.append(task_id)
            
            # Wait for completion
            time.sleep(2)
            task_time = time.time() - task_start
            
            result["details"]["concurrent_task_time"] = task_time
            result["details"]["tasks_submitted"] = len(task_ids)
            
            if task_time > 5:  # 5 seconds threshold
                result["issues"].append(f"Slow concurrent task execution: {task_time:.2f}s")
            
            # Memory usage check
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            result["details"]["memory_usage_mb"] = memory_mb
            
            if memory_mb > 500:  # 500MB threshold
                result["issues"].append(f"High memory usage: {memory_mb:.1f}MB")
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Performance test error: {e}")
        
        return result
    
    def run_system_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        result = {"success": True, "details": {}, "issues": []}
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Check system health from various components
            health_scores = {}
            
            # Security health
            try:
                from core.security_manager import get_security_manager
                security = get_security_manager()
                health = security.validate_secrets_health()
                health_scores["security"] = 1.0 if health.get("total_secrets_cached", 0) > 0 else 0.5
            except Exception as e:
                health_scores["security"] = 0.0
                result["issues"].append(f"Security health check failed: {e}")
            
            # Async coordinator health
            try:
                from core.async_coordinator import get_async_coordinator
                coordinator = get_async_coordinator()
                health = coordinator.get_system_health()
                health_scores["async"] = 1.0 if health.get("event_loop_running", False) else 0.0
            except Exception as e:
                health_scores["async"] = 0.0
                result["issues"].append(f"Async health check failed: {e}")
            
            # Exception handler health
            try:
                from core.exception_handler import get_exception_handler
                handler = get_exception_handler()
                impact_score = handler.get_health_impact_score()
                health_scores["exceptions"] = impact_score
            except Exception as e:
                health_scores["exceptions"] = 0.5
                result["issues"].append(f"Exception handler health check failed: {e}")
            
            # Calculate overall system health
            overall_health = sum(health_scores.values()) / len(health_scores) if health_scores else 0.0
            result["details"]["component_health"] = health_scores
            result["details"]["overall_health"] = overall_health
            
            # Health thresholds
            if overall_health < 0.7:
                result["success"] = False
                result["issues"].append(f"Low system health score: {overall_health:.2f}")
            elif overall_health < 0.8:
                result["issues"].append(f"Moderate system health score: {overall_health:.2f}")
            
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"System health check error: {e}")
        
        return result
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness"""
        result = {"success": True, "details": {}, "issues": []}
        
        # Check for required documentation files
        required_docs = [
            "README.md", "replit.md", "SETUP_GUIDE.md", 
            "TECHNICAL_REVIEW.md", "API_INTEGRATION_STATUS.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                result["details"][doc] = "exists"
                
                # Check if file has content
                if doc_path.stat().st_size < 100:
                    result["issues"].append(f"{doc} is too short")
            else:
                missing_docs.append(doc)
        
        if missing_docs:
            result["success"] = False
            result["issues"].append(f"Missing documentation: {', '.join(missing_docs)}")
        
        result["details"]["missing_docs"] = len(missing_docs)
        result["details"]["total_docs_checked"] = len(required_docs)
        
        return result
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness"""
        result = {"success": True, "details": {}, "issues": []}
        
        # Check for deployment files
        deployment_files = [
            "app.py", "containers.py", "pyproject.toml",
            "start_cryptotrader.bat", "setup_windows_environment.bat"
        ]
        
        missing_files = []
        for file_name in deployment_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                result["details"][file_name] = "exists"
            else:
                missing_files.append(file_name)
        
        if missing_files:
            result["issues"].append(f"Missing deployment files: {', '.join(missing_files)}")
        
        # Check environment variables
        required_env_vars = ["OPENAI_API_KEY"]
        missing_env = []
        
        for env_var in required_env_vars:
            if env_var in os.environ:
                result["details"][f"env_{env_var}"] = "available"
            else:
                missing_env.append(env_var)
        
        if missing_env:
            result["issues"].append(f"Missing environment variables: {', '.join(missing_env)}")
        
        # Check if Streamlit app can be imported
        try:
            sys.path.insert(0, str(self.project_root))
            import app
            result["details"]["streamlit_app"] = "importable"
        except Exception as e:
            result["success"] = False
            result["issues"].append(f"Cannot import Streamlit app: {e}")
        
        return result
    
    def calculate_overall_status(self):
        """Calculate overall pipeline status"""
        stage_results = self.results["stages"]
        
        if not stage_results:
            self.results["overall_status"] = "failed"
            return
        
        passed_stages = sum(1 for stage in stage_results.values() if stage["status"] == "passed")
        total_stages = len(stage_results)
        
        if passed_stages == total_stages:
            self.results["overall_status"] = "passed"
        elif passed_stages >= total_stages * 0.8:
            self.results["overall_status"] = "passed_with_warnings"
        else:
            self.results["overall_status"] = "failed"
        
        self.results["success_rate"] = passed_stages / total_stages if total_stages > 0 else 0.0
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        report_dir = self.project_root / "logs" / "cicd"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_file = report_dir / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("CryptoSmartTrader V2 - CI/CD Pipeline Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Pipeline Status: {self.results['overall_status'].upper()}\n")
            f.write(f"Success Rate: {self.results.get('success_rate', 0) * 100:.1f}%\n")
            f.write(f"Total Stages: {len(self.results['stages'])}\n\n")
            
            f.write("Stage Results:\n")
            f.write("-" * 20 + "\n")
            
            for stage_name, stage_result in self.results["stages"].items():
                status_emoji = "✅" if stage_result["status"] == "passed" else "❌"
                f.write(f"{status_emoji} {stage_name}: {stage_result['status']} ({stage_result['duration']:.2f}s)\n")
                
                if stage_result.get("details", {}).get("issues"):
                    for issue in stage_result["details"]["issues"][:3]:  # First 3 issues
                        f.write(f"    - {issue}\n")
            
            f.write(f"\nReport generated: {datetime.now().isoformat()}\n")
        
        self.logger.info(f"📊 Pipeline report saved to {report_file}")

def main():
    """Run the CI/CD pipeline"""
    pipeline = CICDPipeline()
    results = pipeline.run_full_pipeline()
    
    print("\n" + "="*60)
    print("CICD PIPELINE RESULTS")
    print("="*60)
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Success Rate: {results.get('success_rate', 0) * 100:.1f}%")
    
    if results['overall_status'] == 'passed':
        print("🎉 All systems go! Ready for deployment.")
        return 0
    elif results['overall_status'] == 'passed_with_warnings':
        print("⚠️ Pipeline passed with warnings. Review issues before deployment.")
        return 1
    else:
        print("❌ Pipeline failed. Fix critical issues before deployment.")
        return 2

if __name__ == "__main__":
    sys.exit(main())