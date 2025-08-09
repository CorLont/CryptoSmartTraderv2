#!/usr/bin/env python3
"""
Daily Logging Configuration - Enterprise daily evaluation system
GO/NO-GO decision making based on system health
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DailyMetrics:
    """Daily system metrics for GO/NO-GO decisions"""
    date: str
    system_health_score: float
    data_integrity_score: float
    model_performance_score: float
    coverage_compliance_score: float
    overall_score: float
    go_nogo_decision: str
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

class DailyLogger:
    """Enterprise daily logging and evaluation system"""
    
    def __init__(self):
        self.log_dir = Path("logs/daily_evaluations")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize daily log file
        today = datetime.utcnow().strftime("%Y%m%d")
        self.daily_log_file = self.log_dir / f"daily_eval_{today}.json"
        
        # Current day metrics
        self.daily_metrics = {
            "date": today,
            "system_checks": [],
            "performance_metrics": [],
            "errors": [],
            "warnings": [],
            "security_events": []
        }
        
        # GO/NO-GO thresholds
        self.go_threshold = 0.8  # 80% overall score required for GO
        self.critical_thresholds = {
            "data_integrity": 0.95,  # 95% minimum for data integrity
            "system_health": 0.7,    # 70% minimum for system health
            "model_performance": 0.6, # 60% minimum for model performance
            "coverage_compliance": 0.99  # 99% minimum for coverage
        }
    
    def log_system_check(self, check_name: str, passed: bool, details: str = ""):
        """Log system check result"""
        
        check_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_name": check_name,
            "passed": passed,
            "details": details,
            "severity": "INFO" if passed else "ERROR"
        }
        
        self.daily_metrics["system_checks"].append(check_result)
        self._update_daily_log()
    
    def log_performance_metric(self, metric_name: str, value: float, target: float = None):
        """Log performance metric"""
        
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "target": target,
            "meets_target": value >= target if target is not None else True
        }
        
        self.daily_metrics["performance_metrics"].append(metric)
        self._update_daily_log()
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with detailed context"""
        
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": "ERROR"
        }
        
        self.daily_metrics["errors"].append(error_entry)
        self._update_daily_log()
    
    def log_warning(self, warning_message: str, context: Dict[str, Any] = None):
        """Log warning message"""
        
        warning_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": warning_message,
            "context": context or {},
            "severity": "WARNING"
        }
        
        self.daily_metrics["warnings"].append(warning_entry)
        self._update_daily_log()
    
    def log_security_event(self, event_type: str, severity: str, description: str):
        """Log security event"""
        
        security_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "description": description
        }
        
        self.daily_metrics["security_events"].append(security_event)
        self._update_daily_log()
    
    def generate_daily_evaluation(self) -> DailyMetrics:
        """Generate comprehensive daily evaluation with GO/NO-GO decision"""
        
        evaluation_time = datetime.utcnow()
        
        # Calculate component scores
        system_health_score = self._calculate_system_health_score()
        data_integrity_score = self._calculate_data_integrity_score()
        model_performance_score = self._calculate_model_performance_score()
        coverage_compliance_score = self._calculate_coverage_compliance_score()
        
        # Calculate overall score
        weights = {
            "system_health": 0.25,
            "data_integrity": 0.35,  # Highest weight for data integrity
            "model_performance": 0.25,
            "coverage_compliance": 0.15
        }
        
        overall_score = (
            system_health_score * weights["system_health"] +
            data_integrity_score * weights["data_integrity"] +
            model_performance_score * weights["model_performance"] +
            coverage_compliance_score * weights["coverage_compliance"]
        )
        
        # Make GO/NO-GO decision
        go_nogo_decision, critical_issues, warnings, recommendations = self._make_go_nogo_decision(
            overall_score, system_health_score, data_integrity_score, 
            model_performance_score, coverage_compliance_score
        )
        
        # Create daily metrics
        daily_eval = DailyMetrics(
            date=evaluation_time.strftime("%Y-%m-%d"),
            system_health_score=system_health_score,
            data_integrity_score=data_integrity_score,
            model_performance_score=model_performance_score,
            coverage_compliance_score=coverage_compliance_score,
            overall_score=overall_score,
            go_nogo_decision=go_nogo_decision,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
        
        # Save evaluation
        self._save_daily_evaluation(daily_eval)
        
        return daily_eval
    
    def _calculate_system_health_score(self) -> float:
        """Calculate system health score"""
        
        system_checks = self.daily_metrics["system_checks"]
        
        if not system_checks:
            return 0.5  # Neutral score if no checks
        
        # Count passed checks
        passed_checks = sum(1 for check in system_checks if check["passed"])
        total_checks = len(system_checks)
        
        return passed_checks / total_checks if total_checks > 0 else 0.5
    
    def _calculate_data_integrity_score(self) -> float:
        """Calculate data integrity score"""
        
        # Check for data integrity violations
        integrity_violations = [
            error for error in self.daily_metrics["errors"]
            if "integrity" in error.get("error_message", "").lower()
        ]
        
        # Perfect score if no violations, severe penalty if violations exist
        if integrity_violations:
            return 0.0  # Zero tolerance for data integrity violations
        
        # Check for data quality checks
        data_checks = [
            check for check in self.daily_metrics["system_checks"]
            if "data" in check["check_name"].lower()
        ]
        
        if data_checks:
            passed_data_checks = sum(1 for check in data_checks if check["passed"])
            return passed_data_checks / len(data_checks)
        
        return 0.8  # Default good score if no specific data checks
    
    def _calculate_model_performance_score(self) -> float:
        """Calculate model performance score"""
        
        performance_metrics = self.daily_metrics["performance_metrics"]
        
        if not performance_metrics:
            return 0.6  # Default score if no metrics
        
        # Calculate average performance
        target_met_count = sum(1 for metric in performance_metrics if metric["meets_target"])
        total_metrics = len(performance_metrics)
        
        return target_met_count / total_metrics if total_metrics > 0 else 0.6
    
    def _calculate_coverage_compliance_score(self) -> float:
        """Calculate coverage compliance score"""
        
        # Check for coverage audits
        coverage_checks = [
            check for check in self.daily_metrics["system_checks"]
            if "coverage" in check["check_name"].lower()
        ]
        
        if coverage_checks:
            # Use latest coverage check
            latest_coverage = coverage_checks[-1]
            return 1.0 if latest_coverage["passed"] else 0.0
        
        return 0.8  # Default score if no coverage checks
    
    def _make_go_nogo_decision(self, overall_score: float, system_health: float, 
                             data_integrity: float, model_performance: float, 
                             coverage_compliance: float) -> tuple:
        """Make GO/NO-GO decision based on scores"""
        
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Check critical thresholds
        if data_integrity < self.critical_thresholds["data_integrity"]:
            critical_issues.append(f"Data integrity below critical threshold: {data_integrity:.1%}")
        
        if system_health < self.critical_thresholds["system_health"]:
            critical_issues.append(f"System health below critical threshold: {system_health:.1%}")
        
        if model_performance < self.critical_thresholds["model_performance"]:
            critical_issues.append(f"Model performance below critical threshold: {model_performance:.1%}")
        
        if coverage_compliance < self.critical_thresholds["coverage_compliance"]:
            critical_issues.append(f"Coverage compliance below critical threshold: {coverage_compliance:.1%}")
        
        # Add warnings for scores below optimal
        if system_health < 0.9:
            warnings.append("System health could be improved")
            recommendations.append("Review system resources and performance")
        
        if model_performance < 0.8:
            warnings.append("Model performance could be improved")
            recommendations.append("Consider model retraining or hyperparameter tuning")
        
        # Make decision
        if critical_issues or overall_score < self.go_threshold:
            decision = "NO-GO"
            if overall_score < self.go_threshold:
                critical_issues.append(f"Overall score below threshold: {overall_score:.1%} < {self.go_threshold:.1%}")
        else:
            decision = "GO"
        
        return decision, critical_issues, warnings, recommendations
    
    def _save_daily_evaluation(self, daily_eval: DailyMetrics):
        """Save daily evaluation to file"""
        
        try:
            # Save detailed evaluation
            eval_file = self.log_dir / f"daily_evaluation_{daily_eval.date}.json"
            with open(eval_file, 'w') as f:
                json.dump(asdict(daily_eval), f, indent=2)
            
            # Update current status
            status_file = self.log_dir / "current_go_nogo_status.json"
            with open(status_file, 'w') as f:
                json.dump({
                    "last_evaluation": daily_eval.date,
                    "decision": daily_eval.go_nogo_decision,
                    "overall_score": daily_eval.overall_score,
                    "timestamp": datetime.utcnow().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save daily evaluation: {e}")
    
    def _update_daily_log(self):
        """Update daily log file"""
        
        try:
            with open(self.daily_log_file, 'w') as f:
                json.dump(self.daily_metrics, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to update daily log: {e}")

# Global daily logger instance
_daily_logger = None

def get_daily_logger() -> DailyLogger:
    """Get global daily logger instance"""
    global _daily_logger
    if _daily_logger is None:
        _daily_logger = DailyLogger()
    return _daily_logger