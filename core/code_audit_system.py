#!/usr/bin/env python3
"""
Code Audit System
Complete code quality audit based on enterprise checklist
"""

import os
import sys
import json
import time
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CodeAuditSystem:
    """
    Comprehensive code audit system for enterprise quality assurance
    """
    
    def __init__(self):
        self.audit_results = {}
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
    def run_complete_audit(self) -> Dict[str, Any]:
        """Run complete code audit"""
        
        print("🔍 RUNNING COMPLETE CODE AUDIT")
        print("=" * 45)
        
        audit_start = time.time()
        
        # Core audit categories
        self._audit_data_time_labels()
        self._audit_completeness_nans()
        self._audit_splits_evaluation()
        self._audit_concurrency_io()
        self._audit_ml_ai_systems()
        self._audit_backtest_execution()
        self._audit_logging_monitoring()
        self._audit_infrastructure_tests()
        
        audit_duration = time.time() - audit_start
        
        # Compile audit report
        audit_report = {
            'audit_timestamp': datetime.now().isoformat(),
            'audit_duration': audit_duration,
            'total_files_audited': self._count_files_audited(),
            'critical_issues': len(self.critical_issues),
            'warnings': len(self.warnings),
            'recommendations': len(self.recommendations),
            'audit_results': self.audit_results,
            'critical_issues_list': self.critical_issues,
            'warnings_list': self.warnings,
            'recommendations_list': self.recommendations,
            'overall_quality_score': self._calculate_quality_score()
        }
        
        # Save audit report
        self._save_audit_report(audit_report)
        
        return audit_report
    
    def _audit_data_time_labels(self):
        """Audit A: Data tijd & label-bouw"""
        
        print("📅 Auditing data time & label construction...")
        
        issues = []
        
        # Check for label leakage patterns
        ml_files = list(Path('ml').glob('*.py')) if Path('ml').exists() else []
        
        for file_path in ml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for look-ahead bias patterns
                if 'shift(' in content and '-' not in content:
                    issues.append(f"Potential look-ahead bias in {file_path}: shift without negative value")
                
                # Check for timezone handling
                if 'datetime' in content and 'UTC' not in content:
                    self.warnings.append(f"Missing UTC timezone handling in {file_path}")
                
            except Exception as e:
                self.warnings.append(f"Could not read {file_path}: {e}")
        
        # Check for proper timestamp validation
        validation_found = False
        for file_path in Path('.').glob('**/*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'assert' in content and 'timestamp' in content:
                        validation_found = True
                        break
            except:
                continue
        
        if not validation_found:
            self.critical_issues.append("No timestamp validation assertions found")
        
        self.audit_results['data_time_labels'] = {
            'label_leakage_checks': len(issues),
            'timezone_warnings': len([w for w in self.warnings if 'timezone' in w.lower()]),
            'validation_present': validation_found,
            'issues_found': issues
        }
        
        print(f"   Data time audit: {len(issues)} issues found")
    
    def _audit_completeness_nans(self):
        """Audit B: Completeness & NaN's"""
        
        print("🕳️ Auditing completeness & NaN handling...")
        
        nan_issues = []
        
        # Check for forward-fill patterns
        code_files = list(Path('.').glob('**/*.py'))
        
        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for dangerous fillna patterns
                if 'fillna(' in content and 'forward' in content:
                    nan_issues.append(f"Forward-fill detected in {file_path}")
                
                # Check for NaN validation
                if 'notna()' in content and 'assert' in content:
                    self.recommendations.append(f"Good NaN validation in {file_path}")
                
            except Exception:
                continue
        
        # Check completeness gate implementation
        completeness_gate_exists = Path('core/completeness_gate.py').exists()
        
        if not completeness_gate_exists:
            self.critical_issues.append("Completeness gate implementation missing")
        
        self.audit_results['completeness_nans'] = {
            'forward_fill_issues': len([i for i in nan_issues if 'forward' in i]),
            'completeness_gate_exists': completeness_gate_exists,
            'nan_validation_found': len([r for r in self.recommendations if 'NaN validation' in r]),
            'issues_found': nan_issues
        }
        
        print(f"   NaN audit: {len(nan_issues)} issues found")
    
    def _audit_splits_evaluation(self):
        """Audit C: Splits & evaluatie"""
        
        print("📊 Auditing splits & evaluation...")
        
        split_issues = []
        
        # Check for proper time series splits
        ml_files = list(Path('ml').glob('*.py')) if Path('ml').exists() else []
        
        time_series_split_found = False
        
        for file_path in ml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for dangerous random splits
                if 'KFold' in content or 'ShuffleSplit' in content:
                    split_issues.append(f"Random split detected in {file_path}")
                
                # Check for proper time series splits
                if 'TimeSeriesSplit' in content:
                    time_series_split_found = True
                
                # Check for target scaling issues
                if 'target' in content and 'quantile' in content:
                    self.recommendations.append(f"Target validation found in {file_path}")
                
            except Exception:
                continue
        
        if not time_series_split_found:
            self.critical_issues.append("No TimeSeriesSplit implementation found")
        
        self.audit_results['splits_evaluation'] = {
            'random_split_issues': len([i for i in split_issues if 'Random' in i]),
            'time_series_split_found': time_series_split_found,
            'target_validation_found': len([r for r in self.recommendations if 'Target validation' in r]),
            'issues_found': split_issues
        }
        
        print(f"   Splits audit: {len(split_issues)} issues found")
    
    def _audit_concurrency_io(self):
        """Audit D: Concurrency & IO"""
        
        print("⚡ Auditing concurrency & I/O...")
        
        concurrency_issues = []
        
        # Check for async implementation
        async_files = []
        for file_path in Path('.').glob('**/*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for blocking patterns
                if 'requests.get' in content and 'async' not in content:
                    concurrency_issues.append(f"Blocking requests in {file_path}")
                
                # Check for async implementation
                if 'async def' in content:
                    async_files.append(str(file_path))
                
                # Check for atomic file operations
                if 'open(' in content and 'tmp' not in content and 'rename' not in content:
                    if 'w' in content:  # Writing mode
                        self.warnings.append(f"Non-atomic file write in {file_path}")
                
            except Exception:
                continue
        
        async_coverage = len(async_files) / max(1, len(list(Path('.').glob('**/*.py'))))
        
        self.audit_results['concurrency_io'] = {
            'blocking_requests': len([i for i in concurrency_issues if 'Blocking' in i]),
            'async_files_count': len(async_files),
            'async_coverage_percent': round(async_coverage * 100, 1),
            'atomic_write_warnings': len([w for w in self.warnings if 'atomic' in w.lower()]),
            'issues_found': concurrency_issues
        }
        
        print(f"   Concurrency audit: {len(concurrency_issues)} issues, {len(async_files)} async files")
    
    def _audit_ml_ai_systems(self):
        """Audit E: ML/AI"""
        
        print("🤖 Auditing ML/AI systems...")
        
        ml_issues = []
        
        # Check for calibration
        calibration_found = False
        uncertainty_found = False
        regime_awareness_found = False
        
        ml_files = list(Path('ml').glob('*.py')) if Path('ml').exists() else []
        
        for file_path in ml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for calibration
                if 'CalibratedClassifier' in content or 'calibration' in content.lower():
                    calibration_found = True
                
                # Check for uncertainty quantification
                if 'uncertainty' in content.lower() or 'confidence' in content.lower():
                    uncertainty_found = True
                
                # Check for regime awareness
                if 'regime' in content.lower() or 'market_state' in content:
                    regime_awareness_found = True
                
            except Exception:
                continue
        
        if not calibration_found:
            self.critical_issues.append("No probability calibration found in ML systems")
        
        if not uncertainty_found:
            self.critical_issues.append("No uncertainty quantification found")
        
        if not regime_awareness_found:
            self.warnings.append("No regime awareness detected in ML models")
        
        self.audit_results['ml_ai_systems'] = {
            'calibration_implemented': calibration_found,
            'uncertainty_quantification': uncertainty_found,
            'regime_awareness': regime_awareness_found,
            'ml_files_audited': len(ml_files),
            'issues_found': ml_issues
        }
        
        print(f"   ML/AI audit: Calibration: {'✓' if calibration_found else '✗'}, Uncertainty: {'✓' if uncertainty_found else '✗'}")
    
    def _audit_backtest_execution(self):
        """Audit F: Backtest & Execution"""
        
        print("📈 Auditing backtest & execution...")
        
        backtest_issues = []
        
        # Check for slippage/fees implementation
        slippage_found = False
        fees_found = False
        latency_found = False
        
        trading_files = []
        for pattern in ['*trading*', '*backtest*', '*execution*', '*paper*']:
            trading_files.extend(list(Path('.').glob(f'**/{pattern}.py')))
        
        for file_path in trading_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'slippage' in content.lower():
                    slippage_found = True
                    
                if 'fee' in content.lower() or 'commission' in content.lower():
                    fees_found = True
                    
                if 'latency' in content.lower() or 'delay' in content.lower():
                    latency_found = True
                
            except Exception:
                continue
        
        if not slippage_found:
            self.critical_issues.append("No slippage modeling found in trading systems")
        
        if not fees_found:
            self.warnings.append("No fee modeling found in trading systems")
        
        self.audit_results['backtest_execution'] = {
            'slippage_modeling': slippage_found,
            'fee_modeling': fees_found,
            'latency_modeling': latency_found,
            'trading_files_audited': len(trading_files),
            'issues_found': backtest_issues
        }
        
        print(f"   Backtest audit: {len(trading_files)} files, Slippage: {'✓' if slippage_found else '✗'}")
    
    def _audit_logging_monitoring(self):
        """Audit G: Logging/monitoring"""
        
        print("📝 Auditing logging & monitoring...")
        
        logging_issues = []
        
        # Check for secrets in logs
        log_files = list(Path('logs').glob('**/*')) if Path('logs').exists() else []
        
        secrets_in_logs = False
        correlation_id_found = False
        
        for file_path in log_files:
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for secrets
                    if re.search(r'(api_key|token|secret|password)', content, re.IGNORECASE):
                        secrets_in_logs = True
                        logging_issues.append(f"Potential secrets in {file_path}")
                    
                    # Check for correlation IDs
                    if 'correlation_id' in content or 'run_id' in content:
                        correlation_id_found = True
                        
                except Exception:
                    continue
        
        # Check logging configuration
        logging_config_exists = Path('core/improved_logging_manager.py').exists()
        
        if secrets_in_logs:
            self.critical_issues.append("Secrets detected in log files")
        
        if not correlation_id_found and len(log_files) > 0:
            self.warnings.append("No correlation IDs found in logging")
        
        self.audit_results['logging_monitoring'] = {
            'secrets_in_logs': secrets_in_logs,
            'correlation_id_implemented': correlation_id_found,
            'logging_config_exists': logging_config_exists,
            'log_files_audited': len(log_files),
            'issues_found': logging_issues
        }
        
        print(f"   Logging audit: {len(log_files)} files, Secrets: {'✗' if secrets_in_logs else '✓'}")
    
    def _audit_infrastructure_tests(self):
        """Audit H: Infra/tests"""
        
        print("🏗️ Auditing infrastructure & tests...")
        
        infra_issues = []
        
        # Check for CI/CD configuration
        ci_files = [
            '.github/workflows',
            '.gitlab-ci.yml',
            'Jenkinsfile',
            '.pre-commit-config.yaml'
        ]
        
        ci_found = any(Path(ci_file).exists() for ci_file in ci_files)
        
        # Check for test files
        test_files = list(Path('.').glob('**/test_*.py'))
        pytest_config = Path('pytest.ini').exists()
        
        # Check for linting configuration
        lint_configs = [
            'pyproject.toml',
            '.flake8',
            '.mypy.ini',
            'setup.cfg'
        ]
        
        lint_config_found = any(Path(config).exists() for config in lint_configs)
        
        # Calculate test coverage (approximate)
        py_files = list(Path('.').glob('**/*.py'))
        core_py_files = [f for f in py_files if not str(f).startswith('test_')]
        test_coverage = len(test_files) / max(1, len(core_py_files))
        
        if not ci_found:
            self.warnings.append("No CI/CD configuration found")
        
        if test_coverage < 0.3:
            self.critical_issues.append(f"Low test coverage: {test_coverage:.1%}")
        
        if not lint_config_found:
            self.warnings.append("No linting configuration found")
        
        self.audit_results['infrastructure_tests'] = {
            'ci_cd_configured': ci_found,
            'test_files_count': len(test_files),
            'pytest_configured': pytest_config,
            'test_coverage_estimate': round(test_coverage * 100, 1),
            'linting_configured': lint_config_found,
            'issues_found': infra_issues
        }
        
        print(f"   Infrastructure audit: {len(test_files)} tests, {test_coverage:.1%} coverage")
    
    def _count_files_audited(self) -> int:
        """Count total files audited"""
        return len(list(Path('.').glob('**/*.py')))
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score"""
        
        # Base score
        base_score = 100.0
        
        # Deduct for critical issues
        base_score -= len(self.critical_issues) * 15
        
        # Deduct for warnings
        base_score -= len(self.warnings) * 5
        
        # Bonus for good practices
        if self.audit_results.get('ml_ai_systems', {}).get('calibration_implemented'):
            base_score += 5
        
        if self.audit_results.get('concurrency_io', {}).get('async_coverage_percent', 0) > 50:
            base_score += 5
        
        if self.audit_results.get('infrastructure_tests', {}).get('test_coverage_estimate', 0) > 30:
            base_score += 10
        
        return max(0.0, min(100.0, base_score))
    
    def _save_audit_report(self, report: Dict[str, Any]):
        """Save audit report"""
        
        report_dir = Path('logs/audit')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"code_audit_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Audit report saved: {report_path}")
    
    def print_audit_summary(self, report: Dict[str, Any]):
        """Print audit summary"""
        
        print(f"\n🎯 CODE AUDIT SUMMARY")
        print("=" * 40)
        print(f"Files Audited: {report['total_files_audited']}")
        print(f"Quality Score: {report['overall_quality_score']:.1f}/100")
        print(f"Critical Issues: {report['critical_issues']}")
        print(f"Warnings: {report['warnings']}")
        print(f"Audit Duration: {report['audit_duration']:.2f}s")
        
        if report['critical_issues_list']:
            print(f"\n🚨 Critical Issues:")
            for issue in report['critical_issues_list'][:5]:
                print(f"   • {issue}")
        
        if report['warnings_list']:
            print(f"\n⚠️ Warnings:")
            for warning in report['warnings_list'][:5]:
                print(f"   • {warning}")
        
        if report['recommendations_list']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations_list'][:3]:
                print(f"   • {rec}")

def run_code_audit() -> Dict[str, Any]:
    """Run complete code audit"""
    
    auditor = CodeAuditSystem()
    report = auditor.run_complete_audit()
    auditor.print_audit_summary(report)
    
    return report

if __name__ == "__main__":
    audit_report = run_code_audit()