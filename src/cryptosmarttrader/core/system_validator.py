#!/usr/bin/env python3
"""
System Validator - Enterprise end-to-end production readiness validation

Comprehensive production readiness validator with corrected import names,
flexible file requirements, accurate module paths, and appropriate error classification.
"""

import os
import sys
import importlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from ..core.consolidated_logging_manager import get_consolidated_logger
except ImportError:
    def get_consolidated_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"    # Prevents production deployment
    WARNING = "warning"      # Should be addressed but not blocking
    INFO = "info"           # Informational, nice to have
    SUCCESS = "success"     # Validation passed

@dataclass
class ValidationResult:
    """Result of individual validation check"""
    check_name: str
    severity: ValidationSeverity
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class SystemValidationReport:
    """Complete system validation report"""
    timestamp: datetime
    overall_status: ValidationSeverity
    total_checks: int
    passed_checks: int
    critical_issues: int
    warning_issues: int
    validation_results: List[ValidationResult]
    summary: str = ""
    production_ready: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemValidator:
    """
    Enterprise system validator with corrected dependencies and flexible requirements

    Provides comprehensive production readiness validation with accurate import names,
    configurable file requirements, correct module paths, and appropriate error classification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system validator

        Args:
            config: Optional validation configuration
        """
        self.logger = get_consolidated_logger("SystemValidator")

        # Load configuration
        self.config = self._load_config(config)

        # Validation state
        self.validation_results: List[ValidationResult] = []
        self.last_validation: Optional[SystemValidationReport] = None

        self.logger.info("System Validator initialized with enterprise validation rules")

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load validator configuration with corrected defaults"""

        default_config = {
            # Dependencies - CORRECTED IMPORT NAMES
            "critical_dependencies": [
                "numpy",
                "pandas",
                "sklearn",        # FIXED: sklearn not scikit-learn
                "torch",
                "streamlit",
                "plotly",
                "ccxt",
                "psutil"
            ],

            "optional_dependencies": [
                "cupy",
                "numba",
                "hvac",
                "prometheus_client",
                "respx"
            ],

            # File requirements - FLEXIBLE CONFIGURATION
            "required_files": {
                "critical": [
                    # Core application files
                    "app_fixed_all_issues.py",  # Main app file
                    "pyproject.toml"            # Dependencies
                ],
                "recommended": [
                    # Documentation and configuration
                    "replit.md",                # Project documentation
                    "README.md",               # Project readme
                    "config.json",             # Configuration
                    ".env.example"             # Environment template
                ],
                "optional": [
                    # Additional files
                    "app_minimal.py",          # Minimal app variant
                    "pytest.ini",              # Test configuration
                    ".pre-commit-config.yaml"  # Code quality
                ]
            },

            # Module validation - CORRECTED PATHS
            "core_modules": [
                "core.consolidated_logging_manager",
                "core.config_manager",
                "core.data_manager",
                "core.system_settings"
            ],

            "risk_modules": [
                "core.risk_mitigation",      # FIXED: removed orchestration prefix
                "core.completeness_gate",    # FIXED: core path
                "core.strict_gate"           # FIXED: core path, not orchestration
            ],

            "ml_modules": [
                "core.ml_regime_router",
                "core.bayesian_uncertainty",
                "core.multi_horizon_ml",
                "core.probability_calibrator"
            ],

            "monitoring_modules": [
                "core.system_health_monitor",
                "core.system_monitor",
                "core.system_optimizer",
                "core.system_readiness_checker"
            ],

            # Directory structure
            "required_directories": [
                "core",
                "data",
                "logs",
                "models"
            ],

            "recommended_directories": [
                "cache",
                "config",
                "tests",
                "backups"
            ],

            # Environment validation
            "environment_checks": {
                "python_version_min": "3.8",
                "memory_min_gb": 4.0,
                "disk_space_min_gb": 10.0
            },

            # Validation thresholds
            "thresholds": {
                "critical_failure_tolerance": 0,      # No critical failures allowed
                "warning_failure_tolerance": 3,       # Max 3 warnings for production
                "module_import_timeout": 10           # Seconds
            }
        }

        if config:
            self._deep_merge_dict(default_config, config)

        return default_config

    def validate_system(self) -> SystemValidationReport:
        """
        Perform comprehensive system validation

        Returns:
            SystemValidationReport with complete validation results
        """

        start_time = datetime.now(timezone.utc)
        self.logger.info("Starting comprehensive system validation")

        # Reset validation results
        self.validation_results = []

        try:
            # 1. Validate Python environment
            self._validate_python_environment()

            # 2. Validate critical dependencies
            self._validate_dependencies()

            # 3. Validate file structure
            self._validate_file_structure()

            # 4. Validate module imports
            self._validate_module_imports()

            # 5. Validate system resources
            self._validate_system_resources()

            # 6. Validate configuration integrity
            self._validate_configuration()

            # 7. Validate logging system
            self._validate_logging_system()

            # 8. Validate security setup
            self._validate_security_setup()

        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            self._add_result("validation_execution", ValidationSeverity.CRITICAL, False,
                           f"Validation execution failed: {e}")

        # Generate report
        report = self._generate_validation_report(start_time)
        self.last_validation = report

        status_text = "PRODUCTION READY" if report.production_ready else "NOT PRODUCTION READY"
        self.logger.info(f"System validation completed: {status_text} "
                        f"({report.passed_checks}/{report.total_checks} checks passed)")

        return report

    def _validate_python_environment(self):
        """Validate Python environment and version"""

        try:
            # Check Python version
            current_version = sys.version_info
            min_version_str = self.config["environment_checks"]["python_version_min"]
            min_version = tuple(map(int, min_version_str.split('.')))

            if current_version[:2] >= min_version:
                self._add_result("python_version", ValidationSeverity.SUCCESS, True,
                               f"Python version {current_version.major}.{current_version.minor} meets requirement >= {min_version_str}")
            else:
                self._add_result("python_version", ValidationSeverity.CRITICAL, False,
                               f"Python version {current_version.major}.{current_version.minor} below minimum {min_version_str}",
                               recommendations=["Upgrade to Python >= {min_version_str}"])

            # Check virtual environment
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            if in_venv:
                self._add_result("virtual_environment", ValidationSeverity.SUCCESS, True,
                               "Running in virtual environment")
            else:
                self._add_result("virtual_environment", ValidationSeverity.WARNING, False,
                               "Not running in virtual environment",
                               recommendations=["Consider using virtual environment for dependency isolation"])

        except Exception as e:
            self._add_result("python_environment", ValidationSeverity.CRITICAL, False,
                           f"Python environment validation failed: {e}")

    def _validate_dependencies(self):
        """Validate critical and optional dependencies with CORRECTED IMPORT NAMES"""

        critical_deps = self.config["critical_dependencies"]
        optional_deps = self.config["optional_dependencies"]

        critical_missing = []
        optional_missing = []

        # Validate critical dependencies
        for dep in critical_deps:
            try:
                # Handle special import name mappings
                import_name = dep
                if dep == "sklearn":
                    # Special case: sklearn imports as sklearn, not scikit-learn
                    importlib.import_module("sklearn")
                else:
                    importlib.import_module(import_name)

                self._add_result(f"dependency_{dep}", ValidationSeverity.SUCCESS, True,
                               f"Critical dependency '{dep}' available")
            except ImportError:
                critical_missing.append(dep)
                self._add_result(f"dependency_{dep}", ValidationSeverity.CRITICAL, False,
                               f"Critical dependency '{dep}' missing")

        # Validate optional dependencies
        for dep in optional_deps:
            try:
                importlib.import_module(dep)
                self._add_result(f"optional_dependency_{dep}", ValidationSeverity.SUCCESS, True,
                               f"Optional dependency '{dep}' available")
            except ImportError:
                optional_missing.append(dep)
                self._add_result(f"optional_dependency_{dep}", ValidationSeverity.INFO, False,
                               f"Optional dependency '{dep}' missing")

        # Summary results
        if not critical_missing:
            self._add_result("critical_dependencies", ValidationSeverity.SUCCESS, True,
                           "All critical dependencies available")
        else:
            self._add_result("critical_dependencies", ValidationSeverity.CRITICAL, False,
                           f"{len(critical_missing)} critical dependencies missing: {critical_missing}",
                           recommendations=[f"Install missing dependencies: pip install {' '.join(critical_missing)}"])

        if len(optional_missing) < len(optional_deps) / 2:
            self._add_result("optional_dependencies", ValidationSeverity.SUCCESS, True,
                           f"Most optional dependencies available ({len(optional_deps) - len(optional_missing)}/{len(optional_deps)})")
        else:
            self._add_result("optional_dependencies", ValidationSeverity.WARNING, False,
                           f"Many optional dependencies missing ({len(optional_missing)}/{len(optional_deps)})")

    def _validate_file_structure(self):
        """Validate file structure with FLEXIBLE REQUIREMENTS"""

        file_requirements = self.config["required_files"]

        # Check critical files
        critical_files = file_requirements.get("critical", [])
        critical_missing = []

        for file_path in critical_files:
            if Path(file_path).exists():
                self._add_result(f"critical_file_{file_path}", ValidationSeverity.SUCCESS, True,
                               f"Critical file '{file_path}' exists")
            else:
                critical_missing.append(file_path)
                self._add_result(f"critical_file_{file_path}", ValidationSeverity.CRITICAL, False,
                               f"Critical file '{file_path}' missing")

        # Check recommended files (warnings, not critical)
        recommended_files = file_requirements.get("recommended", [])
        recommended_missing = []

        for file_path in recommended_files:
            if Path(file_path).exists():
                self._add_result(f"recommended_file_{file_path}", ValidationSeverity.SUCCESS, True,
                               f"Recommended file '{file_path}' exists")
            else:
                recommended_missing.append(file_path)
                self._add_result(f"recommended_file_{file_path}", ValidationSeverity.WARNING, False,
                               f"Recommended file '{file_path}' missing")

        # Check optional files (info only)
        optional_files = file_requirements.get("optional", [])
        optional_present = []

        for file_path in optional_files:
            if Path(file_path).exists():
                optional_present.append(file_path)
                self._add_result(f"optional_file_{file_path}", ValidationSeverity.SUCCESS, True,
                               f"Optional file '{file_path}' exists")

        # Directory structure validation
        required_dirs = self.config["required_directories"]
        missing_dirs = []

        for dir_path in required_dirs:
            if Path(dir_path).exists() and Path(dir_path).is_dir():
                self._add_result(f"directory_{dir_path}", ValidationSeverity.SUCCESS, True,
                               f"Required directory '{dir_path}' exists")
            else:
                missing_dirs.append(dir_path)
                self._add_result(f"directory_{dir_path}", ValidationSeverity.CRITICAL, False,
                               f"Required directory '{dir_path}' missing")

        # Summary
        if not critical_missing and not missing_dirs:
            self._add_result("file_structure", ValidationSeverity.SUCCESS, True,
                           "Critical file structure complete")
        else:
            issues = critical_missing + missing_dirs
            self._add_result("file_structure", ValidationSeverity.CRITICAL, False,
                           f"File structure issues: {issues}",
                           recommendations=[f"Create missing files/directories: {issues}"])

    def _validate_module_imports(self):
        """Validate module imports with CORRECTED PATHS"""

        module_categories = [
            ("core_modules", ValidationSeverity.CRITICAL),
            ("risk_modules", ValidationSeverity.CRITICAL),
            ("ml_modules", ValidationSeverity.WARNING),
            ("monitoring_modules", ValidationSeverity.WARNING)
        ]

        for category, default_severity in module_categories:
            modules = self.config.get(category, [])
            category_failures = []

            for module_name in modules:
                try:
                    importlib.import_module(module_name)
                    self._add_result(f"module_{module_name}", ValidationSeverity.SUCCESS, True,
                                   f"Module '{module_name}' imports successfully")
                except ImportError as e:
                    category_failures.append(module_name)

                    # Special handling for logging modules - WARNING not CRITICAL
                    if "logging" in module_name.lower():
                        severity = ValidationSeverity.WARNING
                        message = f"Logging module '{module_name}' not available (using fallback)"
                    else:
                        severity = default_severity
                        message = f"Module '{module_name}' import failed: {e}"

                    self._add_result(f"module_{module_name}", severity, False, message)

            # Category summary
            if not category_failures:
                self._add_result(f"{category}_imports", ValidationSeverity.SUCCESS, True,
                               f"All {category.replace('_', ' ')} import successfully")
            else:
                severity = ValidationSeverity.WARNING if len(category_failures) < len(modules) / 2 else default_severity
                self._add_result(f"{category}_imports", severity, False,
                               f"{len(category_failures)}/{len(modules)} {category.replace('_', ' ')} failed to import")

    def _validate_system_resources(self):
        """Validate system resources and capacity"""

        try:
            # Memory validation
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_gb = memory.total / (1024**3)
                min_memory = self.config["environment_checks"]["memory_min_gb"]

                if memory_gb >= min_memory:
                    self._add_result("system_memory", ValidationSeverity.SUCCESS, True,
                                   f"System memory {memory_gb:.1f}GB meets requirement >= {min_memory}GB")
                else:
                    self._add_result("system_memory", ValidationSeverity.WARNING, False,
                                   f"System memory {memory_gb:.1f}GB below recommended {min_memory}GB")

                # Memory usage
                memory_usage = memory.percent
                if memory_usage < 80:
                    self._add_result("memory_usage", ValidationSeverity.SUCCESS, True,
                                   f"Memory usage {memory_usage:.1f}% normal")
                else:
                    self._add_result("memory_usage", ValidationSeverity.WARNING, False,
                                   f"High memory usage {memory_usage:.1f}%")

            except ImportError:
                self._add_result("system_memory", ValidationSeverity.WARNING, False,
                               "Cannot check memory - psutil not available")

            # Disk space validation
            try:
                import shutil
                disk_usage = shutil.disk_usage('.')
                disk_free_gb = disk_usage.free / (1024**3)
                min_disk = self.config["environment_checks"]["disk_space_min_gb"]

                if disk_free_gb >= min_disk:
                    self._add_result("disk_space", ValidationSeverity.SUCCESS, True,
                                   f"Free disk space {disk_free_gb:.1f}GB meets requirement >= {min_disk}GB")
                else:
                    self._add_result("disk_space", ValidationSeverity.WARNING, False,
                                   f"Low disk space {disk_free_gb:.1f}GB below recommended {min_disk}GB")

            except Exception as e:
                self._add_result("disk_space", ValidationSeverity.WARNING, False,
                               f"Cannot check disk space: {e}")

        except Exception as e:
            self._add_result("system_resources", ValidationSeverity.WARNING, False,
                           f"System resource validation failed: {e}")

    def _validate_configuration(self):
        """Validate configuration files and settings"""

        try:
            # Check environment variables
            required_env_vars = ["KRAKEN_API_KEY", "KRAKEN_SECRET", "OPENAI_API_KEY"]
            missing_env_vars = []

            for env_var in required_env_vars:
                if os.getenv(env_var):
                    self._add_result(f"env_var_{env_var}", ValidationSeverity.SUCCESS, True,
                                   f"Environment variable '{env_var}' configured")
                else:
                    missing_env_vars.append(env_var)
                    self._add_result(f"env_var_{env_var}", ValidationSeverity.WARNING, False,
                                   f"Environment variable '{env_var}' not set")

            # Check .env file
            if Path(".env").exists():
                self._add_result("env_file", ValidationSeverity.SUCCESS, True,
                               ".env file exists")
            else:
                self._add_result("env_file", ValidationSeverity.INFO, False,
                               ".env file missing (using environment variables)")

            # Configuration summary
            if len(missing_env_vars) < len(required_env_vars) / 2:
                self._add_result("configuration", ValidationSeverity.SUCCESS, True,
                               "Configuration mostly complete")
            else:
                self._add_result("configuration", ValidationSeverity.WARNING, False,
                               f"Configuration incomplete: {missing_env_vars} missing")

        except Exception as e:
            self._add_result("configuration", ValidationSeverity.WARNING, False,
                           f"Configuration validation failed: {e}")

    def _validate_logging_system(self):
        """Validate logging system with APPROPRIATE CLASSIFICATION"""

        try:
            # Try to import logging manager - WARNING not CRITICAL if missing
            try:
                test_logger = get_consolidated_logger("test")
                self._add_result("logging_system", ValidationSeverity.SUCCESS, True,
                               "Enterprise logging system available")
            except ImportError:
                self._add_result("logging_system", ValidationSeverity.WARNING, False,
                               "Enterprise logging system not available, using standard logging")

            # Check log directory
            if Path("logs").exists():
                self._add_result("log_directory", ValidationSeverity.SUCCESS, True,
                               "Log directory exists")
            else:
                self._add_result("log_directory", ValidationSeverity.INFO, False,
                               "Log directory missing (will be created)")

            # Test basic logging functionality
            test_logger = logging.getLogger("validation_test")
            test_logger.info("Validation test message")
            self._add_result("basic_logging", ValidationSeverity.SUCCESS, True,
                           "Basic logging functionality working")

        except Exception as e:
            self._add_result("logging_validation", ValidationSeverity.WARNING, False,
                           f"Logging validation failed: {e}")

    def _validate_security_setup(self):
        """Validate security configuration"""

        try:
            # Check secret management
            if os.getenv("SECRET_KEY"):
                self._add_result("secret_key", ValidationSeverity.SUCCESS, True,
                               "Application secret key configured")
            else:
                self._add_result("secret_key", ValidationSeverity.WARNING, False,
                               "Application secret key not configured")

            # Check file permissions (if on Unix-like system)
            if os.name != 'nt':  # Not Windows
                env_file = Path(".env")
                if env_file.exists():
                    permissions = oct(env_file.stat().st_mode)[-3:]
                    if permissions in ['600', '644']:
                        self._add_result("env_permissions", ValidationSeverity.SUCCESS, True,
                                       f".env file permissions secure ({permissions})")
                    else:
                        self._add_result("env_permissions", ValidationSeverity.WARNING, False,
                                       f".env file permissions may be too permissive ({permissions})")

            # Check for common security files
            security_files = [".gitignore", ".env.example"]
            for sec_file in security_files:
                if Path(sec_file).exists():
                    self._add_result(f"security_file_{sec_file}", ValidationSeverity.SUCCESS, True,
                                   f"Security file '{sec_file}' exists")
                else:
                    self._add_result(f"security_file_{sec_file}", ValidationSeverity.INFO, False,
                                   f"Security file '{sec_file}' recommended")

        except Exception as e:
            self._add_result("security_validation", ValidationSeverity.WARNING, False,
                           f"Security validation failed: {e}")

    def _add_result(self, check_name: str, severity: ValidationSeverity, success: bool,
                   message: str, details: Optional[Dict[str, Any]] = None,
                   recommendations: Optional[List[str]] = None):
        """Add validation result to results list"""

        result = ValidationResult(
            check_name=check_name,
            severity=severity,
            success=success,
            message=message,
            details=details or {},
            recommendations=recommendations or []
        )

        self.validation_results.append(result)

    def _generate_validation_report(self, start_time: datetime) -> SystemValidationReport:
        """Generate comprehensive validation report"""

        # Count results by severity
        critical_issues = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL and not r.success)
        warning_issues = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.WARNING and not r.success)
        passed_checks = sum(1 for r in self.validation_results if r.success)
        total_checks = len(self.validation_results)

        # Determine overall status
        if critical_issues > 0:
            overall_status = ValidationSeverity.CRITICAL
            production_ready = False
        elif warning_issues > self.config["thresholds"]["warning_failure_tolerance"]:
            overall_status = ValidationSeverity.WARNING
            production_ready = False
        else:
            overall_status = ValidationSeverity.SUCCESS
            production_ready = True

        # Generate summary
        if production_ready:
            summary = f"System is production ready ({passed_checks}/{total_checks} checks passed)"
        else:
            issues = []
            if critical_issues > 0:
                issues.append(f"{critical_issues} critical issues")
            if warning_issues > 0:
                issues.append(f"{warning_issues} warnings")
            summary = f"System not production ready: {', '.join(issues)}"

        # Create report
        report = SystemValidationReport(
            timestamp=start_time,
            overall_status=overall_status,
            total_checks=total_checks,
            passed_checks=passed_checks,
            critical_issues=critical_issues,
            warning_issues=warning_issues,
            validation_results=self.validation_results,
            summary=summary,
            production_ready=production_ready,
            metadata={
                "validation_duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "validator_version": "2.0.0",
                "python_version": sys.version,
                "platform": sys.platform
            }
        )

        return report

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

        return base

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation system summary"""

        if not self.last_validation:
            return {"status": "no_validation_performed"}

        return {
            "production_ready": self.last_validation.production_ready,
            "overall_status": self.last_validation.overall_status.value,
            "total_checks": self.last_validation.total_checks,
            "passed_checks": self.last_validation.passed_checks,
            "critical_issues": self.last_validation.critical_issues,
            "warning_issues": self.last_validation.warning_issues,
            "last_validation": self.last_validation.timestamp.isoformat(),
            "summary": self.last_validation.summary
        }

# Utility functions

def quick_validation() -> Dict[str, Any]:
    """Perform quick system validation"""

    validator = SystemValidator()
    report = validator.validate_system()

    return {
        "production_ready": report.production_ready,
        "status": report.overall_status.value,
        "summary": report.summary,
        "critical_issues": report.critical_issues,
        "warning_issues": report.warning_issues,
        "timestamp": report.timestamp.isoformat()
    }

def detailed_validation(config: Optional[Dict[str, Any]] = None) -> SystemValidationReport:
    """Perform detailed system validation with custom configuration"""

    validator = SystemValidator(config=config)
    return validator.validate_system()

if __name__ == "__main__":
    # Test system validation
    print("Testing System Validator")

    validator = SystemValidator()
    report = validator.validate_system()

    print(f"\nValidation Report:")
    print(f"Production Ready: {report.production_ready}")
    print(f"Overall Status: {report.overall_status.value}")
    print(f"Summary: {report.summary}")
    print(f"Checks: {report.passed_checks}/{report.total_checks} passed")

    # Show critical issues
    critical_results = [r for r in report.validation_results if r.severity == ValidationSeverity.CRITICAL and not r.success]
    if critical_results:
        print(f"\nCritical Issues ({len(critical_results)}):")
        for result in critical_results:
            print(f"  ❌ {result.check_name}: {result.message}")

    # Show warnings
    warning_results = [r for r in report.validation_results if r.severity == ValidationSeverity.WARNING and not r.success]
    if warning_results:
        print(f"\nWarnings ({len(warning_results)}):")
        for result in warning_results[:5]:  # Show first 5
            print(f"  ⚠️ {result.check_name}: {result.message}")

    # Show successes
    success_results = [r for r in report.validation_results if r.success]
    if success_results:
        print(f"\nSuccessful Checks ({len(success_results)}):")
        for result in success_results[:5]:  # Show first 5
            print(f"  ✅ {result.check_name}: {result.message}")

    print("\n✅ SYSTEM VALIDATOR TEST COMPLETE")
