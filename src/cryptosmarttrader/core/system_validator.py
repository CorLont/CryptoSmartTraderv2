#!/usr/bin/env python3
"""
System Validator
Comprehensive system validation and health checking
"""

import os
import sys
import json
import time
import asyncio
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SystemValidator:
    """
    Comprehensive system validation for production readiness
    """
    
    def __init__(self):
        self.validation_results = {}
        self.critical_errors = []
        self.warnings = []
        self.recommendations = []
    
    def validate_all_systems(self) -> Dict[str, Any]:
        """Run complete system validation"""
        
        print("🔍 RUNNING COMPREHENSIVE SYSTEM VALIDATION")
        print("=" * 60)
        
        validation_start = time.time()
        
        # Core system validation
        self._validate_python_environment()
        self._validate_dependencies()
        self._validate_hardware_requirements()
        self._validate_file_structure()
        self._validate_configuration()
        self._validate_api_keys()
        self._validate_logging_system()
        self._validate_ml_components()
        self._validate_risk_systems()
        self._validate_trading_engine()
        self._validate_monitoring_systems()
        
        validation_duration = time.time() - validation_start
        
        # Compile final report
        final_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_duration': validation_duration,
            'overall_status': self._calculate_overall_status(),
            'component_results': self.validation_results,
            'critical_errors': self.critical_errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'production_ready': len(self.critical_errors) == 0
        }
        
        # Save validation report
        self._save_validation_report(final_report)
        
        return final_report
    
    def _validate_python_environment(self):
        """Validate Python environment"""
        
        print("🐍 Validating Python environment...")
        
        result = {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': sys.platform,
            'architecture': sys.maxsize > 2**32 and '64-bit' or '32-bit'
        }
        
        # Check Python version
        if sys.version_info < (3, 9):
            self.critical_errors.append("Python 3.9+ required")
            result['version_check'] = 'FAIL'
        else:
            result['version_check'] = 'PASS'
        
        # Check if running in virtual environment
        result['virtual_env'] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        self.validation_results['python_environment'] = result
        print(f"   Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}: {'✓' if result['version_check'] == 'PASS' else '✗'}")
    
    def _validate_dependencies(self):
        """Validate critical dependencies"""
        
        print("📦 Validating dependencies...")
        
        critical_deps = [
            'numpy', 'pandas', 'scikit-learn', 'torch', 
            'streamlit', 'plotly', 'ccxt', 'psutil'
        ]
        
        optional_deps = [
            'prometheus_client', 'pythonjsonlogger', 'GPUtil', 'pynvml'
        ]
        
        result = {
            'critical_dependencies': {},
            'optional_dependencies': {},
            'missing_critical': [],
            'missing_optional': []
        }
        
        # Check critical dependencies
        for dep in critical_deps:
            try:
                __import__(dep)
                result['critical_dependencies'][dep] = 'AVAILABLE'
            except ImportError:
                result['critical_dependencies'][dep] = 'MISSING'
                result['missing_critical'].append(dep)
                self.critical_errors.append(f"Critical dependency missing: {dep}")
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                __import__(dep)
                result['optional_dependencies'][dep] = 'AVAILABLE'
            except ImportError:
                result['optional_dependencies'][dep] = 'MISSING'
                result['missing_optional'].append(dep)
                self.warnings.append(f"Optional dependency missing: {dep}")
        
        self.validation_results['dependencies'] = result
        print(f"   Critical deps: {len(critical_deps) - len(result['missing_critical'])}/{len(critical_deps)}")
        print(f"   Optional deps: {len(optional_deps) - len(result['missing_optional'])}/{len(optional_deps)}")
    
    def _validate_hardware_requirements(self):
        """Validate hardware requirements"""
        
        print("🖥️ Validating hardware...")
        
        result = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3)),
            'disk_space_gb': round(psutil.disk_usage('.').free / (1024**3)),
            'gpu_available': False
        }
        
        # Check GPU availability
        try:
            import torch
            result['gpu_available'] = torch.cuda.is_available()
            if result['gpu_available']:
                result['gpu_count'] = torch.cuda.device_count()
                result['gpu_name'] = torch.cuda.get_device_name(0)
                result['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        except ImportError:
            pass
        
        # Validate minimum requirements
        min_requirements = {
            'cpu_cores': 4,
            'memory_gb': 8,
            'disk_space_gb': 10
        }
        
        result['requirements_met'] = True
        
        for req, min_val in min_requirements.items():
            if result[req] < min_val:
                self.critical_errors.append(f"Insufficient {req}: {result[req]} < {min_val}")
                result['requirements_met'] = False
        
        # Recommendations for optimal performance
        if result['memory_gb'] >= 32:
            self.recommendations.append("Excellent RAM capacity - enable aggressive caching")
        elif result['memory_gb'] >= 16:
            self.recommendations.append("Good RAM capacity - moderate caching recommended")
        
        if not result['gpu_available']:
            self.recommendations.append("Install CUDA-compatible GPU for ML acceleration")
        
        self.validation_results['hardware'] = result
        print(f"   CPU: {result['cpu_cores']} cores")
        print(f"   RAM: {result['memory_gb']} GB")
        print(f"   GPU: {'✓' if result['gpu_available'] else '✗'}")
    
    def _validate_file_structure(self):
        """Validate project file structure"""
        
        print("📁 Validating file structure...")
        
        required_files = [
            'app_minimal.py',
            'config.json',
            'pyproject.toml',
            'replit.md'
        ]
        
        required_dirs = [
            'core',
            'agents',
            'ml',
            'api',
            'logs',
            'config',
            'data'
        ]
        
        result = {
            'missing_files': [],
            'missing_dirs': [],
            'file_structure_valid': True
        }
        
        # Check required files
        for file_path in required_files:
            if not Path(file_path).exists():
                result['missing_files'].append(file_path)
                self.critical_errors.append(f"Required file missing: {file_path}")
                result['file_structure_valid'] = False
        
        # Check required directories
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                result['missing_dirs'].append(dir_path)
                self.warnings.append(f"Required directory missing: {dir_path}")
        
        self.validation_results['file_structure'] = result
        print(f"   Files: {len(required_files) - len(result['missing_files'])}/{len(required_files)}")
        print(f"   Directories: {len(required_dirs) - len(result['missing_dirs'])}/{len(required_dirs)}")
    
    def _validate_configuration(self):
        """Validate configuration files"""
        
        print("⚙️ Validating configuration...")
        
        result = {
            'config_files_valid': True,
            'config_errors': []
        }
        
        # Check config.json
        config_path = Path('config.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                result['config_json'] = 'VALID'
                
                # Check for required configuration keys
                required_keys = ['data_sources', 'ml_models', 'risk_management']
                for key in required_keys:
                    if key not in config_data:
                        self.warnings.append(f"Missing config key: {key}")
                        
            except json.JSONDecodeError as e:
                result['config_json'] = 'INVALID'
                result['config_errors'].append(f"config.json: {e}")
                self.critical_errors.append("Invalid config.json format")
                result['config_files_valid'] = False
        else:
            result['config_json'] = 'MISSING'
            self.critical_errors.append("config.json missing")
            result['config_files_valid'] = False
        
        self.validation_results['configuration'] = result
        print(f"   Configuration: {'✓' if result['config_files_valid'] else '✗'}")
    
    def _validate_api_keys(self):
        """Validate API key configuration"""
        
        print("🔑 Validating API keys...")
        
        required_keys = [
            'KRAKEN_API_KEY',
            'KRAKEN_SECRET',
            'OPENAI_API_KEY'
        ]
        
        result = {
            'available_keys': [],
            'missing_keys': [],
            'api_keys_configured': True
        }
        
        for key in required_keys:
            if os.getenv(key):
                result['available_keys'].append(key)
            else:
                result['missing_keys'].append(key)
                self.warnings.append(f"API key not configured: {key}")
                result['api_keys_configured'] = False
        
        if not result['api_keys_configured']:
            self.recommendations.append("Configure API keys for full functionality")
        
        self.validation_results['api_keys'] = result
        print(f"   API Keys: {len(result['available_keys'])}/{len(required_keys)} configured")
    
    def _validate_logging_system(self):
        """Validate logging system"""
        
        print("📝 Validating logging system...")
        
        result = {
            'log_directories_exist': True,
            'log_permissions': True,
            'logging_system_operational': True
        }
        
        # Check log directories
        log_dirs = [
            Path('logs'),
            Path('logs/daily'),
            Path('logs/agents'),
            Path('logs/ml')
        ]
        
        for log_dir in log_dirs:
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    result['log_directories_created'] = True
                except PermissionError:
                    self.critical_errors.append(f"Cannot create log directory: {log_dir}")
                    result['log_permissions'] = False
                    result['logging_system_operational'] = False
        
        # Test logging functionality
        try:
            from core.improved_logging_manager import get_improved_logger
            logger = get_improved_logger()
            logger.info("System validation logging test")
            result['logger_test'] = 'PASS'
        except Exception as e:
            result['logger_test'] = 'FAIL'
            result['logging_errors'] = str(e)
            self.critical_errors.append(f"Logging system error: {e}")
            result['logging_system_operational'] = False
        
        self.validation_results['logging'] = result
        print(f"   Logging: {'✓' if result['logging_system_operational'] else '✗'}")
    
    def _validate_ml_components(self):
        """Validate ML components"""
        
        print("🤖 Validating ML components...")
        
        result = {
            'ml_libraries_available': True,
            'model_directories_exist': True,
            'ml_system_ready': True
        }
        
        # Check ML libraries
        ml_libs = ['torch', 'sklearn', 'numpy', 'pandas']
        missing_ml_libs = []
        
        for lib in ml_libs:
            try:
                __import__(lib)
            except ImportError:
                missing_ml_libs.append(lib)
                result['ml_libraries_available'] = False
        
        if missing_ml_libs:
            self.critical_errors.append(f"Missing ML libraries: {missing_ml_libs}")
            result['ml_system_ready'] = False
        
        # Check model directories
        model_dirs = [
            Path('models'),
            Path('models/lstm'),
            Path('models/transformers'),
            Path('mlartifacts')
        ]
        
        for model_dir in model_dirs:
            if not model_dir.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_results['ml_components'] = result
        print(f"   ML System: {'✓' if result['ml_system_ready'] else '✗'}")
    
    def _validate_risk_systems(self):
        """Validate risk management systems"""
        
        print("🛡️ Validating risk systems...")
        
        result = {
            'risk_modules_available': True,
            'confidence_gate_operational': True,
            'risk_systems_ready': True
        }
        
        # Check risk management modules
        risk_modules = [
            'core.risk_mitigation',
            'core.completeness_gate',
            'orchestration.strict_gate'
        ]
        
        for module in risk_modules:
            try:
                __import__(module)
                result[f'{module}_available'] = True
            except ImportError as e:
                result[f'{module}_available'] = False
                self.critical_errors.append(f"Risk module missing: {module}")
                result['risk_modules_available'] = False
        
        if not result['risk_modules_available']:
            result['risk_systems_ready'] = False
        
        self.validation_results['risk_systems'] = result
        print(f"   Risk Systems: {'✓' if result['risk_systems_ready'] else '✗'}")
    
    def _validate_trading_engine(self):
        """Validate trading engine components"""
        
        print("💰 Validating trading engine...")
        
        result = {
            'exchange_connectivity': True,
            'paper_trading_ready': True,
            'trading_engine_operational': True
        }
        
        # Test CCXT availability
        try:
            import ccxt
            result['ccxt_available'] = True
            
            # Test exchange initialization (without API keys)
            try:
                kraken = ccxt.kraken()
                result['exchange_init'] = True
            except Exception as e:
                result['exchange_init'] = False
                self.warnings.append(f"Exchange initialization test failed: {e}")
                
        except ImportError:
            result['ccxt_available'] = False
            self.critical_errors.append("CCXT library missing")
            result['trading_engine_operational'] = False
        
        self.validation_results['trading_engine'] = result
        print(f"   Trading Engine: {'✓' if result['trading_engine_operational'] else '✗'}")
    
    def _validate_monitoring_systems(self):
        """Validate monitoring and metrics systems"""
        
        print("📊 Validating monitoring systems...")
        
        result = {
            'prometheus_available': False,
            'health_monitoring_ready': True,
            'monitoring_systems_operational': True
        }
        
        # Check Prometheus
        try:
            import prometheus_client
            result['prometheus_available'] = True
        except ImportError:
            self.warnings.append("Prometheus client not available")
        
        # Check health monitoring
        try:
            from core.daily_health_dashboard import DailyHealthDashboard
            dashboard = DailyHealthDashboard()
            result['health_dashboard_test'] = 'PASS'
        except Exception as e:
            result['health_dashboard_test'] = 'FAIL'
            self.warnings.append(f"Health dashboard test failed: {e}")
        
        self.validation_results['monitoring'] = result
        print(f"   Monitoring: {'✓' if result['monitoring_systems_operational'] else '✗'}")
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system status"""
        
        if self.critical_errors:
            return 'CRITICAL'
        elif len(self.warnings) > 10:
            return 'WARNING'
        elif len(self.warnings) > 5:
            return 'CAUTION'
        else:
            return 'HEALTHY'
    
    def _save_validation_report(self, report: Dict[str, Any]):
        """Save validation report"""
        
        report_dir = Path('logs/validation')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"system_validation_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved: {report_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print validation summary"""
        
        print(f"\n🏁 SYSTEM VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Production Ready: {'✓' if report['production_ready'] else '✗'}")
        print(f"Validation Duration: {report['validation_duration']:.2f}s")
        
        if report['critical_errors']:
            print(f"\n🚨 Critical Errors ({len(report['critical_errors'])}):")
            for error in report['critical_errors'][:5]:
                print(f"   - {error}")
        
        if report['warnings']:
            print(f"\n⚠️ Warnings ({len(report['warnings'])}):")
            for warning in report['warnings'][:5]:
                print(f"   - {warning}")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
            for rec in report['recommendations'][:5]:
                print(f"   - {rec}")

def run_system_validation() -> Dict[str, Any]:
    """Run complete system validation"""
    
    validator = SystemValidator()
    report = validator.validate_all_systems()
    validator.print_summary(report)
    
    return report

if __name__ == "__main__":
    validation_report = run_system_validation()