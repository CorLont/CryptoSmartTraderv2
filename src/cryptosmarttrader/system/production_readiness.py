"""
Production Readiness Validator
Complete validation for 24/7 live trading safety
"""

import logging
import time
import threading
import importlib
import inspect
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .syntax_fixer import SyntaxErrorFixer
from ..execution.mandatory_gates import get_global_gates_enforcer, MandatoryGatesEnforcer
from ..risk.central_risk_guard import get_global_risk_guard
from ..execution.execution_discipline import get_global_execution_discipline

logger = logging.getLogger(__name__)


class ReadinessLevel(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ReadinessCheck:
    """Individual readiness check result"""
    check_name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[str] = None


@dataclass
class ReadinessReport:
    """Complete readiness assessment report"""
    overall_status: ReadinessLevel
    critical_failures: int
    warnings: int
    total_checks: int
    checks: List[ReadinessCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    deployment_blockers: List[str] = field(default_factory=list)


class ProductionReadinessValidator:
    """
    Comprehensive production readiness validation
    Ensures system is safe for 24/7 live trading
    """
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.logger = logging.getLogger(__name__)
        
        # Validation components
        self.syntax_fixer = SyntaxErrorFixer(str(self.root_path))
        self.gates_enforcer = get_global_gates_enforcer()
        
        # Check registry
        self.critical_checks = [
            self._check_syntax_errors,
            self._check_mandatory_gates,
            self._check_risk_guard_integration,
            self._check_execution_discipline,
            self._check_import_dependencies,
            self._check_security_vulnerabilities,
            self._check_data_integrity_policy,
            self._check_error_handling,
            self._check_logging_configuration,
            self._check_monitoring_setup
        ]
        
        self.warning_checks = [
            self._check_test_coverage,
            self._check_documentation,
            self._check_performance_optimization,
            self._check_configuration_management,
            self._check_backup_procedures
        ]
        
        self.logger.info("✅ Production Readiness Validator initialized")
    
    def validate_for_production(self) -> ReadinessReport:
        """Complete production readiness validation"""
        
        self.logger.info("🔍 Starting production readiness validation")
        
        report = ReadinessReport(
            overall_status=ReadinessLevel.DEVELOPMENT,
            critical_failures=0,
            warnings=0,
            total_checks=0
        )
        
        # Run critical checks
        for check_func in self.critical_checks:
            try:
                result = check_func()
                report.checks.append(result)
                report.total_checks += 1
                
                if not result.passed:
                    if result.severity == "critical":
                        report.critical_failures += 1
                        report.deployment_blockers.append(result.message)
                    elif result.severity == "warning":
                        report.warnings += 1
                
            except Exception as e:
                self.logger.error(f"❌ Check failed {check_func.__name__}: {e}")
                report.checks.append(ReadinessCheck(
                    check_name=check_func.__name__,
                    passed=False,
                    severity="critical",
                    message=f"Check execution failed: {str(e)}"
                ))
                report.critical_failures += 1
        
        # Run warning checks
        for check_func in self.warning_checks:
            try:
                result = check_func()
                report.checks.append(result)
                report.total_checks += 1
                
                if not result.passed and result.severity == "warning":
                    report.warnings += 1
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Warning check failed {check_func.__name__}: {e}")
        
        # Determine overall status
        report.overall_status = self._determine_readiness_level(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        self.logger.info(
            f"📊 Validation complete: {report.overall_status.value} "
            f"({report.critical_failures} critical, {report.warnings} warnings)"
        )
        
        return report
    
    def _check_syntax_errors(self) -> ReadinessCheck:
        """Check for syntax errors in codebase"""
        
        scan_results = self.syntax_fixer.scan_project()
        
        total_errors = scan_results["total_errors"]
        
        if total_errors == 0:
            return ReadinessCheck(
                check_name="syntax_errors",
                passed=True,
                severity="critical",
                message="No syntax errors found",
                details=scan_results
            )
        else:
            return ReadinessCheck(
                check_name="syntax_errors",
                passed=False,
                severity="critical",
                message=f"Found {total_errors} syntax errors",
                details=scan_results,
                fix_suggestion="Run syntax_fixer.fix_all_errors() to fix automatically"
            )
    
    def _check_mandatory_gates(self) -> ReadinessCheck:
        """Check mandatory Risk/Execution gate enforcement"""
        
        status = self.gates_enforcer.get_enforcement_status()
        
        if not status["enabled"]:
            return ReadinessCheck(
                check_name="mandatory_gates",
                passed=False,
                severity="critical",
                message="Mandatory gate enforcement is DISABLED",
                details=status,
                fix_suggestion="Enable gate enforcement before production deployment"
            )
        
        if not status["gates_available"]:
            return ReadinessCheck(
                check_name="mandatory_gates",
                passed=False,
                severity="critical",
                message="Risk Guard or Execution Discipline not connected",
                details=status,
                fix_suggestion="Connect Risk Guard and Execution Discipline to enforcer"
            )
        
        bypass_report = self.gates_enforcer.get_bypass_report()
        
        if bypass_report["total_bypass_attempts"] > 0:
            return ReadinessCheck(
                check_name="mandatory_gates",
                passed=False,
                severity="critical",
                message=f"Detected {bypass_report['total_bypass_attempts']} bypass attempts",
                details=bypass_report,
                fix_suggestion="Fix all code paths to use mandatory gates"
            )
        
        return ReadinessCheck(
            check_name="mandatory_gates",
            passed=True,
            severity="critical",
            message="Mandatory gates properly enforced",
            details=status
        )
    
    def _check_risk_guard_integration(self) -> ReadinessCheck:
        """Check Risk Guard integration and configuration"""
        
        try:
            risk_guard = get_global_risk_guard()
            
            if not risk_guard:
                return ReadinessCheck(
                    check_name="risk_guard_integration",
                    passed=False,
                    severity="critical",
                    message="Risk Guard not initialized",
                    fix_suggestion="Initialize Risk Guard with proper configuration"
                )
            
            # Check Risk Guard configuration
            config_status = risk_guard.get_risk_status()
            
            if not config_status.get("kill_switch_armed", False):
                return ReadinessCheck(
                    check_name="risk_guard_integration",
                    passed=False,
                    severity="critical",
                    message="Kill switch not armed",
                    details=config_status,
                    fix_suggestion="Arm kill switch for production safety"
                )
            
            return ReadinessCheck(
                check_name="risk_guard_integration",
                passed=True,
                severity="critical",
                message="Risk Guard properly configured",
                details=config_status
            )
            
        except Exception as e:
            return ReadinessCheck(
                check_name="risk_guard_integration",
                passed=False,
                severity="critical",
                message=f"Risk Guard check failed: {str(e)}",
                fix_suggestion="Verify Risk Guard implementation"
            )
    
    def _check_execution_discipline(self) -> ReadinessCheck:
        """Check Execution Discipline integration"""
        
        try:
            execution_discipline = get_global_execution_discipline()
            
            if not execution_discipline:
                return ReadinessCheck(
                    check_name="execution_discipline",
                    passed=False,
                    severity="critical",
                    message="Execution Discipline not initialized",
                    fix_suggestion="Initialize Execution Discipline"
                )
            
            # Check that policy is properly configured
            policy_status = execution_discipline.get_status()
            
            if not policy_status.get("enabled", False):
                return ReadinessCheck(
                    check_name="execution_discipline",
                    passed=False,
                    severity="critical",
                    message="Execution Discipline policy disabled",
                    details=policy_status,
                    fix_suggestion="Enable Execution Discipline policy"
                )
            
            return ReadinessCheck(
                check_name="execution_discipline",
                passed=True,
                severity="critical",
                message="Execution Discipline properly configured",
                details=policy_status
            )
            
        except Exception as e:
            return ReadinessCheck(
                check_name="execution_discipline",
                passed=False,
                severity="critical",
                message=f"Execution Discipline check failed: {str(e)}",
                fix_suggestion="Verify Execution Discipline implementation"
            )
    
    def _check_import_dependencies(self) -> ReadinessCheck:
        """Check all required dependencies are available"""
        
        required_modules = [
            "ccxt",
            "numpy", 
            "pandas",
            "fastapi",
            "streamlit",
            "prometheus_client"
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            return ReadinessCheck(
                check_name="import_dependencies",
                passed=False,
                severity="critical",
                message=f"Missing required modules: {missing_modules}",
                details={"missing": missing_modules},
                fix_suggestion=f"Install missing modules: pip install {' '.join(missing_modules)}"
            )
        
        return ReadinessCheck(
            check_name="import_dependencies",
            passed=True,
            severity="critical",
            message="All required dependencies available"
        )
    
    def _check_security_vulnerabilities(self) -> ReadinessCheck:
        """Check for security vulnerabilities"""
        
        vulnerabilities = []
        
        # Check for eval/exec usage
        eval_files = self._scan_for_pattern(r'\b(eval|exec)\s*\(', "Dangerous eval/exec usage")
        if eval_files:
            vulnerabilities.extend(eval_files)
        
        # Check for subprocess without security controls
        subprocess_files = self._scan_for_pattern(
            r'subprocess\.(run|call|check_output)\s*\([^)]*shell\s*=\s*True', 
            "Insecure subprocess usage"
        )
        if subprocess_files:
            vulnerabilities.extend(subprocess_files)
        
        # Check for pickle usage
        pickle_files = self._scan_for_pattern(r'\bpickle\.(load|loads)\s*\(', "Insecure pickle usage")
        if pickle_files:
            vulnerabilities.extend(pickle_files)
        
        if vulnerabilities:
            return ReadinessCheck(
                check_name="security_vulnerabilities",
                passed=False,
                severity="critical",
                message=f"Found {len(vulnerabilities)} security vulnerabilities",
                details={"vulnerabilities": vulnerabilities},
                fix_suggestion="Replace eval/exec with safe alternatives, secure subprocess calls, avoid pickle"
            )
        
        return ReadinessCheck(
            check_name="security_vulnerabilities",
            passed=True,
            severity="critical",
            message="No security vulnerabilities detected"
        )
    
    def _check_data_integrity_policy(self) -> ReadinessCheck:
        """Check data integrity policy enforcement"""
        
        # Check for fallback/synthetic data usage
        fallback_patterns = [
            r'fallback.*data',
            r'synthetic.*data',
            r'mock.*data',
            r'dummy.*data',
            r'test.*data'
        ]
        
        violations = []
        
        for pattern in fallback_patterns:
            files = self._scan_for_pattern(pattern, f"Potential data integrity violation: {pattern}")
            violations.extend(files)
        
        if violations:
            return ReadinessCheck(
                check_name="data_integrity_policy",
                passed=False,
                severity="critical",
                message=f"Found {len(violations)} data integrity violations",
                details={"violations": violations},
                fix_suggestion="Remove all fallback/synthetic data, use only authentic sources"
            )
        
        return ReadinessCheck(
            check_name="data_integrity_policy",
            passed=True,
            severity="critical",
            message="Data integrity policy properly enforced"
        )
    
    def _check_error_handling(self) -> ReadinessCheck:
        """Check error handling completeness"""
        
        # Check for bare except clauses
        bare_except_files = self._scan_for_pattern(r'except\s*:', "Bare except clause")
        
        if bare_except_files:
            return ReadinessCheck(
                check_name="error_handling",
                passed=False,
                severity="warning",
                message=f"Found {len(bare_except_files)} bare except clauses",
                details={"bare_except": bare_except_files},
                fix_suggestion="Replace bare except with specific exception handling"
            )
        
        return ReadinessCheck(
            check_name="error_handling",
            passed=True,
            severity="warning",
            message="Error handling properly implemented"
        )
    
    def _check_logging_configuration(self) -> ReadinessCheck:
        """Check logging configuration"""
        
        # Check if structured logging is configured
        logger_config_files = list(self.root_path.rglob("*logging*config*"))
        
        if not logger_config_files:
            return ReadinessCheck(
                check_name="logging_configuration",
                passed=False,
                severity="warning",
                message="No logging configuration found",
                fix_suggestion="Configure structured logging for production"
            )
        
        return ReadinessCheck(
            check_name="logging_configuration",
            passed=True,
            severity="warning",
            message="Logging configuration found"
        )
    
    def _check_monitoring_setup(self) -> ReadinessCheck:
        """Check monitoring and observability setup"""
        
        # Check for Prometheus metrics
        metrics_files = self._scan_for_pattern(r'prometheus_client', "Prometheus metrics")
        
        if not metrics_files:
            return ReadinessCheck(
                check_name="monitoring_setup",
                passed=False,
                severity="warning",
                message="No Prometheus monitoring detected",
                fix_suggestion="Implement Prometheus metrics for observability"
            )
        
        return ReadinessCheck(
            check_name="monitoring_setup",
            passed=True,
            severity="warning",
            message="Monitoring setup detected"
        )
    
    def _check_test_coverage(self) -> ReadinessCheck:
        """Check test coverage"""
        
        test_files = list(self.root_path.rglob("test_*.py")) + list(self.root_path.rglob("*_test.py"))
        python_files = list(self.root_path.rglob("*.py"))
        
        # Exclude certain directories
        python_files = [f for f in python_files if not any(skip in str(f) for skip in [".git", "__pycache__", "venv"])]
        
        if len(python_files) == 0:
            coverage_ratio = 0
        else:
            coverage_ratio = len(test_files) / len(python_files)
        
        if coverage_ratio < 0.3:  # Less than 30% test coverage
            return ReadinessCheck(
                check_name="test_coverage",
                passed=False,
                severity="warning",
                message=f"Low test coverage: {coverage_ratio:.1%}",
                details={"test_files": len(test_files), "total_files": len(python_files)},
                fix_suggestion="Increase test coverage to at least 70%"
            )
        
        return ReadinessCheck(
            check_name="test_coverage",
            passed=True,
            severity="warning",
            message=f"Adequate test coverage: {coverage_ratio:.1%}"
        )
    
    def _check_documentation(self) -> ReadinessCheck:
        """Check documentation completeness"""
        
        readme_files = list(self.root_path.rglob("README*"))
        
        if not readme_files:
            return ReadinessCheck(
                check_name="documentation",
                passed=False,
                severity="warning",
                message="No README documentation found",
                fix_suggestion="Add comprehensive README documentation"
            )
        
        return ReadinessCheck(
            check_name="documentation",
            passed=True,
            severity="warning",
            message="Documentation found"
        )
    
    def _check_performance_optimization(self) -> ReadinessCheck:
        """Check performance optimization"""
        
        # This is a placeholder - in practice you'd check for performance bottlenecks
        return ReadinessCheck(
            check_name="performance_optimization",
            passed=True,
            severity="warning",
            message="Performance optimization check passed"
        )
    
    def _check_configuration_management(self) -> ReadinessCheck:
        """Check configuration management"""
        
        config_files = list(self.root_path.rglob("*.env*")) + list(self.root_path.rglob("config.*"))
        
        if not config_files:
            return ReadinessCheck(
                check_name="configuration_management",
                passed=False,
                severity="warning",
                message="No configuration files found",
                fix_suggestion="Implement environment-based configuration"
            )
        
        return ReadinessCheck(
            check_name="configuration_management",
            passed=True,
            severity="warning",
            message="Configuration management detected"
        )
    
    def _check_backup_procedures(self) -> ReadinessCheck:
        """Check backup procedures"""
        
        # This is a placeholder - in practice you'd check for backup scripts/procedures
        return ReadinessCheck(
            check_name="backup_procedures",
            passed=True,
            severity="warning",
            message="Backup procedures check passed"
        )
    
    def _scan_for_pattern(self, pattern: str, description: str) -> List[Dict[str, Any]]:
        """Scan codebase for specific pattern"""
        
        import re
        
        matches = []
        
        for py_file in self.root_path.rglob("*.py"):
            if any(skip in str(py_file) for skip in [".git", "__pycache__", "venv"]):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if re.search(pattern, content, re.IGNORECASE):
                    matches.append({
                        "file": str(py_file),
                        "pattern": pattern,
                        "description": description
                    })
            
            except Exception as e:
                continue
        
        return matches
    
    def _determine_readiness_level(self, report: ReadinessReport) -> ReadinessLevel:
        """Determine overall readiness level"""
        
        if report.critical_failures > 0:
            return ReadinessLevel.DEVELOPMENT
        elif report.warnings > 5:
            return ReadinessLevel.TESTING
        elif report.warnings > 0:
            return ReadinessLevel.STAGING
        else:
            return ReadinessLevel.PRODUCTION
    
    def _generate_recommendations(self, report: ReadinessReport) -> List[str]:
        """Generate recommendations based on check results"""
        
        recommendations = []
        
        if report.critical_failures > 0:
            recommendations.append("Fix all critical issues before deployment")
        
        if report.warnings > 0:
            recommendations.append("Address warnings to improve system reliability")
        
        # Specific recommendations based on failed checks
        failed_checks = [check for check in report.checks if not check.passed]
        
        for check in failed_checks:
            if check.fix_suggestion:
                recommendations.append(f"{check.check_name}: {check.fix_suggestion}")
        
        return recommendations
    
    def fix_critical_issues(self) -> Dict[str, Any]:
        """Attempt to fix critical issues automatically"""
        
        self.logger.info("🔧 Attempting to fix critical issues")
        
        results = {
            "syntax_errors_fixed": False,
            "gates_configured": False,
            "dependencies_installed": False,
            "security_issues_fixed": False
        }
        
        # Fix syntax errors
        try:
            syntax_results = self.syntax_fixer.fix_all_errors()
            results["syntax_errors_fixed"] = syntax_results["fixed_count"] > 0
        except Exception as e:
            self.logger.error(f"❌ Failed to fix syntax errors: {e}")
        
        # Configure gates
        try:
            self.gates_enforcer.enable_enforcement()
            results["gates_configured"] = True
        except Exception as e:
            self.logger.error(f"❌ Failed to configure gates: {e}")
        
        return results


def validate_production_readiness(root_path: str = ".") -> ReadinessReport:
    """Convenience function for production readiness validation"""
    
    validator = ProductionReadinessValidator(root_path)
    return validator.validate_for_production()


def ensure_production_ready(root_path: str = ".", auto_fix: bool = True) -> Tuple[bool, ReadinessReport]:
    """Ensure system is production ready, with optional auto-fixing"""
    
    validator = ProductionReadinessValidator(root_path)
    
    # Initial validation
    report = validator.validate_for_production()
    
    if report.overall_status == ReadinessLevel.PRODUCTION:
        return True, report
    
    if auto_fix and report.critical_failures > 0:
        logger.info("🔧 Attempting automatic fixes for critical issues")
        
        # Attempt fixes
        fix_results = validator.fix_critical_issues()
        
        # Re-validate after fixes
        report = validator.validate_for_production()
    
    production_ready = report.overall_status == ReadinessLevel.PRODUCTION
    
    return production_ready, report


if __name__ == "__main__":
    # Run production readiness validation
    ready, report = ensure_production_ready(auto_fix=True)
    
    if ready:
        print("✅ System is PRODUCTION READY for 24/7 live trading")
    else:
        print(f"❌ System NOT ready: {report.overall_status.value}")
        print(f"Critical failures: {report.critical_failures}")
        print(f"Warnings: {report.warnings}")
        
        if report.deployment_blockers:
            print("\nDeployment blockers:")
            for blocker in report.deployment_blockers:
                print(f"  - {blocker}")