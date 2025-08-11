#!/usr/bin/env python3
"""
System Readiness Checker - Real system status validation
Validates actual system readiness before enabling UI components
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

from core.structured_logger import get_structured_logger

class SystemReadinessChecker:
    """Comprehensive system readiness validation"""
    
    def __init__(self):
        self.logger = get_structured_logger("SystemReadinessChecker")
        
        # Readiness thresholds
        self.max_model_age_hours = 24  # Models older than 24h = stale
        self.min_health_score = 70     # Minimum health score for GO
        self.max_calibration_age_hours = 12  # Calibration must be recent
        
    def check_complete_system_readiness(self) -> Dict[str, Any]:
        """Complete system readiness check for UI status"""
        
        try:
            self.logger.info("Performing complete system readiness check")
            
            # Check all subsystems
            model_status = self._check_model_readiness()
            data_status = self._check_data_completeness()
            calibration_status = self._check_calibration_readiness()
            health_status = self._check_system_health()
            
            # Overall readiness decision
            is_system_ready = (
                model_status['models_ready'] and
                data_status['data_complete'] and
                calibration_status['calibration_recent'] and
                health_status['health_acceptable']
            )
            
            # Calculate readiness score
            readiness_score = self._calculate_readiness_score(
                model_status, data_status, calibration_status, health_status
            )
            
            # Determine UI component states
            ui_states = self._determine_ui_component_states(
                is_system_ready, model_status, data_status, health_status
            )
            
            readiness_result = {
                'system_ready': is_system_ready,
                'readiness_score': readiness_score,
                'status_message': self._get_status_message(is_system_ready, readiness_score),
                'status_color': 'green' if is_system_ready else 'red',
                'component_status': {
                    'models': model_status,
                    'data': data_status,
                    'calibration': calibration_status,
                    'health': health_status
                },
                'ui_component_states': ui_states,
                'last_check': datetime.utcnow().isoformat(),
                'blocking_issues': self._get_blocking_issues(model_status, data_status, calibration_status, health_status)
            }
            
            self.logger.info(f"System readiness: {'READY' if is_system_ready else 'NOT READY'} (score: {readiness_score})")
            
            return readiness_result
            
        except Exception as e:
            self.logger.error(f"System readiness check failed: {e}")
            return self._get_error_status(str(e))
    
    def _check_model_readiness(self) -> Dict[str, Any]:
        """Check if ML models are trained and available"""
        
        try:
            model_dir = Path("models")
            
            # Check for model files
            model_files = {
                'lstm_models': list(model_dir.glob("lstm_*.pth")) if model_dir.exists() else [],
                'tree_models': list(model_dir.glob("*_tree.pkl")) if model_dir.exists() else [],
                'scaler_files': list(model_dir.glob("scaler_*.pkl")) if model_dir.exists() else []
            }
            
            total_models = sum(len(files) for files in model_files.values())
            
            # Check model freshness
            recent_models = 0
            oldest_model_age = 0
            
            if total_models > 0:
                all_model_files = []
                for files in model_files.values():
                    all_model_files.extend(files)
                
                current_time = datetime.utcnow()
                for model_file in all_model_files:
                    if model_file.exists():
                        model_age_hours = (current_time - datetime.fromtimestamp(model_file.stat().st_mtime)).total_seconds() / 3600
                        oldest_model_age = max(oldest_model_age, model_age_hours)
                        
                        if model_age_hours <= self.max_model_age_hours:
                            recent_models += 1
            
            # Check for horizon coverage
            required_horizons = [1, 24, 168, 720]  # 1h, 24h, 7d, 30d
            horizon_coverage = 0
            
            for horizon in required_horizons:
                lstm_file = model_dir / f"lstm_{horizon}h.pth"
                tree_file = model_dir / f"xgb_{horizon}h.pkl"
                if (lstm_file.exists() or tree_file.exists()):
                    horizon_coverage += 1
            
            horizon_coverage_pct = horizon_coverage / len(required_horizons)
            
            models_ready = (
                total_models >= 4 and  # Minimum number of models
                recent_models > 0 and  # At least one recent model
                horizon_coverage_pct >= 0.5  # At least 50% horizon coverage
            )
            
            return {
                'models_ready': models_ready,
                'total_models': total_models,
                'recent_models': recent_models,
                'oldest_model_age_hours': oldest_model_age,
                'horizon_coverage_percent': horizon_coverage_pct * 100,
                'model_files_found': {k: len(v) for k, v in model_files.items()},
                'readiness_issues': [] if models_ready else [
                    f"Insufficient models: {total_models} (need ≥4)",
                    f"No recent models (age: {oldest_model_age:.1f}h, max: {self.max_model_age_hours}h)",
                    f"Low horizon coverage: {horizon_coverage_pct:.1%} (need ≥50%)"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Model readiness check failed: {e}")
            return {
                'models_ready': False,
                'total_models': 0,
                'recent_models': 0,
                'readiness_issues': [f"Model check error: {str(e)}"]
            }
    
    def _check_data_completeness(self) -> Dict[str, Any]:
        """Check data availability and completeness"""
        
        try:
            data_dir = Path("data")
            
            # Check for recent market data
            market_data_files = list(data_dir.glob("market_data*.json")) if data_dir.exists() else []
            
            recent_data_files = 0
            oldest_data_age = 0
            
            current_time = datetime.utcnow()
            for data_file in market_data_files:
                if data_file.exists():
                    data_age_hours = (current_time - datetime.fromtimestamp(data_file.stat().st_mtime)).total_seconds() / 3600
                    oldest_data_age = max(oldest_data_age, data_age_hours)
                    
                    if data_age_hours <= 24:  # Data within 24 hours
                        recent_data_files += 1
            
            # Check coverage audit results
            coverage_audit_file = Path("logs/coverage_audits/latest_coverage_audit.json")
            coverage_compliance = False
            coverage_percentage = 0.0
            
            if coverage_audit_file.exists():
                try:
                    with open(coverage_audit_file, 'r') as f:
                        coverage_data = json.load(f)
                        coverage_percentage = coverage_data.get('coverage_percentage', 0.0)
                        coverage_compliance = coverage_data.get('compliance_status') == 'COMPLIANT'
                except:
                    pass
            
            # Check data integrity reports
            integrity_reports = list(Path("logs/data_integrity_violations").glob("*.json")) if Path("logs/data_integrity_violations").exists() else []
            recent_violations = len([f for f in integrity_reports if (current_time - datetime.fromtimestamp(f.stat().st_mtime)).total_seconds() < 3600])
            
            data_complete = (
                recent_data_files > 0 and
                coverage_compliance and
                coverage_percentage >= 0.8 and
                recent_violations == 0
            )
            
            return {
                'data_complete': data_complete,
                'recent_data_files': recent_data_files,
                'oldest_data_age_hours': oldest_data_age,
                'coverage_percentage': coverage_percentage,
                'coverage_compliant': coverage_compliance,
                'recent_integrity_violations': recent_violations,
                'readiness_issues': [] if data_complete else [
                    f"No recent data files" if recent_data_files == 0 else "",
                    f"Coverage non-compliant: {coverage_percentage:.1%}" if not coverage_compliance else "",
                    f"Recent integrity violations: {recent_violations}" if recent_violations > 0 else ""
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Data completeness check failed: {e}")
            return {
                'data_complete': False,
                'recent_data_files': 0,
                'readiness_issues': [f"Data check error: {str(e)}"]
            }
    
    def _check_calibration_readiness(self) -> Dict[str, Any]:
        """Check if confidence calibration is recent"""
        
        try:
            # Check for calibration data
            calibration_files = list(Path("models").glob("*calibration*.json")) if Path("models").exists() else []
            
            most_recent_calibration = None
            calibration_age_hours = float('inf')
            
            current_time = datetime.utcnow()
            for cal_file in calibration_files:
                if cal_file.exists():
                    file_age = (current_time - datetime.fromtimestamp(cal_file.stat().st_mtime)).total_seconds() / 3600
                    if file_age < calibration_age_hours:
                        calibration_age_hours = file_age
                        most_recent_calibration = cal_file
            
            calibration_recent = calibration_age_hours <= self.max_calibration_age_hours
            
            return {
                'calibration_recent': calibration_recent,
                'calibration_age_hours': calibration_age_hours if calibration_age_hours != float('inf') else 0,
                'calibration_files_found': len(calibration_files),
                'most_recent_calibration': str(most_recent_calibration) if most_recent_calibration else None,
                'readiness_issues': [] if calibration_recent else [
                    f"Calibration too old: {calibration_age_hours:.1f}h (max: {self.max_calibration_age_hours}h)"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Calibration readiness check failed: {e}")
            return {
                'calibration_recent': False,
                'calibration_age_hours': 0,
                'readiness_issues': [f"Calibration check error: {str(e)}"]
            }
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health score"""
        
        try:
            # Check latest health status
            health_file = Path("health_status.json")
            
            if health_file.exists():
                with open(health_file, 'r') as f:
                    health_data = json.load(f)
                    health_score = health_data.get('score', 0)
                    health_grade = health_data.get('grade', 'F')
            else:
                health_score = 0
                health_grade = 'F'
            
            # Check GO/NO-GO status
            go_nogo_file = Path("logs/current_trading_status.json")
            go_nogo_decision = "NO-GO"
            
            if go_nogo_file.exists():
                try:
                    with open(go_nogo_file, 'r') as f:
                        go_data = json.load(f)
                        go_nogo_decision = go_data.get('decision', 'NO-GO')
                except:
                    pass
            
            health_acceptable = health_score >= self.min_health_score and go_nogo_decision == "GO"
            
            return {
                'health_acceptable': health_acceptable,
                'health_score': health_score,
                'health_grade': health_grade,
                'go_nogo_decision': go_nogo_decision,
                'readiness_issues': [] if health_acceptable else [
                    f"Low health score: {health_score} (need ≥{self.min_health_score})",
                    f"GO/NO-GO status: {go_nogo_decision}"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'health_acceptable': False,
                'health_score': 0,
                'go_nogo_decision': 'NO-GO',
                'readiness_issues': [f"Health check error: {str(e)}"]
            }
    
    def _calculate_readiness_score(self, model_status: Dict, data_status: Dict, 
                                 calibration_status: Dict, health_status: Dict) -> int:
        """Calculate overall readiness score (0-100)"""
        
        weights = {
            'models': 0.4,      # 40% weight
            'data': 0.3,        # 30% weight  
            'calibration': 0.15, # 15% weight
            'health': 0.15      # 15% weight
        }
        
        # Component scores
        model_score = 100 if model_status['models_ready'] else 0
        data_score = 100 if data_status['data_complete'] else 0
        calibration_score = 100 if calibration_status['calibration_recent'] else 0
        health_score = health_status.get('health_score', 0)
        
        # Weighted total
        total_score = (
            model_score * weights['models'] +
            data_score * weights['data'] +
            calibration_score * weights['calibration'] +
            health_score * weights['health']
        )
        
        return int(total_score)
    
    def _determine_ui_component_states(self, system_ready: bool, model_status: Dict, 
                                     data_status: Dict, health_status: Dict) -> Dict[str, str]:
        """Determine which UI components should be enabled/disabled"""
        
        return {
            'ai_predictions_tab': 'enabled' if model_status['models_ready'] else 'disabled',
            'top_opportunities_tab': 'enabled' if (model_status['models_ready'] and system_ready) else 'disabled',
            'confidence_gate_controls': 'enabled' if model_status['models_ready'] else 'disabled',
            'filtering_controls': 'enabled' if model_status['models_ready'] else 'disabled',
            'system_status_indicator': 'green' if system_ready else 'red',
            'trading_signals': 'enabled' if system_ready else 'disabled'
        }
    
    def _get_status_message(self, system_ready: bool, readiness_score: int) -> str:
        """Get appropriate status message"""
        
        if system_ready:
            return "System Ready"
        elif readiness_score >= 70:
            return "System Almost Ready"
        elif readiness_score >= 40:
            return "System Partially Ready"
        else:
            return "System Not Ready"
    
    def _get_blocking_issues(self, model_status: Dict, data_status: Dict,
                           calibration_status: Dict, health_status: Dict) -> List[str]:
        """Get list of issues blocking system readiness"""
        
        issues = []
        
        for status in [model_status, data_status, calibration_status, health_status]:
            issues.extend(status.get('readiness_issues', []))
        
        # Filter out empty issues
        return [issue for issue in issues if issue.strip()]
    
    def _get_error_status(self, error_message: str) -> Dict[str, Any]:
        """Return error status when readiness check fails"""
        
        return {
            'system_ready': False,
            'readiness_score': 0,
            'status_message': 'System Check Failed',
            'status_color': 'red',
            'component_status': {},
            'ui_component_states': {
                'ai_predictions_tab': 'disabled',
                'top_opportunities_tab': 'disabled',
                'confidence_gate_controls': 'disabled',
                'filtering_controls': 'disabled',
                'system_status_indicator': 'red',
                'trading_signals': 'disabled'
            },
            'last_check': datetime.utcnow().isoformat(),
            'blocking_issues': [f"System check error: {error_message}"],
            'error': error_message
        }

# Global instance
system_checker = SystemReadinessChecker()

def get_system_readiness() -> Dict[str, Any]:
    """Get current system readiness status"""
    return system_checker.check_complete_system_readiness()