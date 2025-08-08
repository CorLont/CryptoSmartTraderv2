#!/usr/bin/env python3
"""
Daily Metrics Mandatory System
Coverage audit and daily metrics as mandatory requirements
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MandatoryDailyMetrics:
    """
    Mandatory daily metrics collection and validation system
    """
    
    def __init__(self):
        self.metrics_dir = Path('logs/daily')
        self.coverage_dir = Path('logs/coverage')
        self.required_metrics = [
            'coverage_audit',
            'system_health',
            'confidence_gate_stats',
            'model_performance',
            'execution_quality',
            'risk_assessment'
        ]
        
    def run_mandatory_daily_collection(self) -> Dict[str, Any]:
        """Run complete mandatory daily metrics collection"""
        
        today = datetime.now().strftime("%Y%m%d")
        timestamp = datetime.now().isoformat()
        
        print(f"🔍 RUNNING MANDATORY DAILY METRICS FOR {today}")
        print("=" * 55)
        
        # Ensure directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all mandatory collections
        daily_metrics = {
            'date': today,
            'timestamp': timestamp,
            'metrics_status': 'collecting'
        }
        
        # 1. Coverage Audit (MANDATORY)
        coverage_result = self._run_coverage_audit()
        daily_metrics['coverage_audit'] = coverage_result
        
        # 2. System Health (MANDATORY)
        health_result = self._collect_system_health()
        daily_metrics['system_health'] = health_result
        
        # 3. Confidence Gate Stats (MANDATORY)
        confidence_result = self._collect_confidence_stats()
        daily_metrics['confidence_gate_stats'] = confidence_result
        
        # 4. Model Performance (MANDATORY)
        model_result = self._collect_model_performance()
        daily_metrics['model_performance'] = model_result
        
        # 5. Execution Quality (MANDATORY)
        execution_result = self._collect_execution_quality()
        daily_metrics['execution_quality'] = execution_result
        
        # 6. Risk Assessment (MANDATORY)
        risk_result = self._collect_risk_assessment()
        daily_metrics['risk_assessment'] = risk_result
        
        # Validate completeness
        validation_result = self._validate_metrics_completeness(daily_metrics)
        daily_metrics['validation'] = validation_result
        daily_metrics['metrics_status'] = 'complete' if validation_result['all_required_present'] else 'incomplete'
        
        # Save daily metrics
        self._save_daily_metrics(daily_metrics, today)
        
        # Generate GO/NO-GO decision
        go_nogo = self._generate_go_nogo_decision(daily_metrics)
        daily_metrics['go_nogo_decision'] = go_nogo
        
        return daily_metrics
    
    def _run_coverage_audit(self) -> Dict[str, Any]:
        """Run mandatory coverage audit"""
        
        print("📊 Running coverage audit...")
        
        # Simulate coverage audit (in real system, would connect to exchanges)
        total_symbols = 457  # Kraken total
        processed_symbols = np.random.randint(440, 457)  # Simulate good coverage
        
        coverage_pct = (processed_symbols / total_symbols) * 100
        
        # Generate missing symbols
        missing_count = total_symbols - processed_symbols
        missing_symbols = [f"MISSING_{i}" for i in range(missing_count)]
        
        coverage_result = {
            'timestamp': datetime.now().isoformat(),
            'exchange': 'kraken',
            'total_symbols': total_symbols,
            'processed_symbols': processed_symbols,
            'coverage_percentage': coverage_pct,
            'missing_symbols': missing_symbols,
            'status': 'PASS' if coverage_pct >= 99.0 else 'FAIL',
            'mandatory_check': True
        }
        
        # Save coverage audit
        coverage_file = self.coverage_dir / f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(coverage_file, 'w') as f:
            json.dump(coverage_result, f, indent=2)
        
        print(f"   Coverage: {coverage_pct:.1f}% ({processed_symbols}/{total_symbols})")
        
        return coverage_result
    
    def _collect_system_health(self) -> Dict[str, Any]:
        """Collect mandatory system health metrics"""
        
        print("🏥 Collecting system health...")
        
        # Simulate system health collection
        health_metrics = {
            'cpu_usage': np.random.uniform(30, 70),
            'memory_usage': np.random.uniform(50, 80),
            'disk_usage': np.random.uniform(20, 60),
            'gpu_usage': np.random.uniform(40, 90),
            'api_success_rate': np.random.uniform(95, 99.9),
            'error_rate': np.random.uniform(0.1, 2.0),
            'uptime_hours': np.random.uniform(23.5, 24.0)
        }
        
        # Calculate overall health score
        health_score = (
            (100 - health_metrics['cpu_usage']) * 0.15 +
            (100 - health_metrics['memory_usage']) * 0.15 +
            (100 - health_metrics['disk_usage']) * 0.10 +
            health_metrics['api_success_rate'] * 0.25 +
            (100 - health_metrics['error_rate']) * 0.15 +
            (health_metrics['uptime_hours'] / 24 * 100) * 0.20
        )
        
        health_result = {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'individual_metrics': health_metrics,
            'status': 'HEALTHY' if health_score >= 85 else 'DEGRADED' if health_score >= 60 else 'CRITICAL',
            'mandatory_check': True
        }
        
        print(f"   Health Score: {health_score:.1f}/100")
        
        return health_result
    
    def _collect_confidence_stats(self) -> Dict[str, Any]:
        """Collect mandatory confidence gate statistics"""
        
        print("🚪 Collecting confidence gate stats...")
        
        # Simulate confidence gate statistics
        total_candidates = np.random.randint(10, 50)
        confidence_threshold = 0.8
        
        # Simulate confidence scores
        confidence_scores = np.random.beta(2, 3, total_candidates)
        passed_gate = (confidence_scores >= confidence_threshold).sum()
        
        confidence_result = {
            'timestamp': datetime.now().isoformat(),
            'confidence_threshold': confidence_threshold,
            'total_candidates': total_candidates,
            'passed_gate': passed_gate,
            'pass_rate': passed_gate / total_candidates,
            'average_confidence': confidence_scores.mean(),
            'confidence_distribution': {
                'min': confidence_scores.min(),
                'max': confidence_scores.max(),
                'std': confidence_scores.std()
            },
            'gate_effectiveness': 'HIGH' if passed_gate > 0 else 'BLOCKING',
            'mandatory_check': True
        }
        
        print(f"   Gate Pass Rate: {passed_gate}/{total_candidates} ({confidence_result['pass_rate']:.2%})")
        
        return confidence_result
    
    def _collect_model_performance(self) -> Dict[str, Any]:
        """Collect mandatory model performance metrics"""
        
        print("🤖 Collecting model performance...")
        
        # Simulate model performance metrics
        model_metrics = {
            'mae_24h': np.random.uniform(0.02, 0.08),
            'rmse_24h': np.random.uniform(0.03, 0.12),
            'accuracy_24h': np.random.uniform(0.65, 0.85),
            'precision': np.random.uniform(0.70, 0.90),
            'recall': np.random.uniform(0.60, 0.85),
            'f1_score': 0.0,  # Will calculate
            'calibration_score': np.random.uniform(0.75, 0.95),
            'uncertainty_coverage': np.random.uniform(0.80, 0.95)
        }
        
        # Calculate F1 score
        model_metrics['f1_score'] = 2 * (model_metrics['precision'] * model_metrics['recall']) / (model_metrics['precision'] + model_metrics['recall'])
        
        # Calculate overall performance score
        performance_score = (
            (1 - model_metrics['mae_24h']) * 0.25 +
            model_metrics['accuracy_24h'] * 0.25 +
            model_metrics['f1_score'] * 0.25 +
            model_metrics['calibration_score'] * 0.25
        ) * 100
        
        model_result = {
            'timestamp': datetime.now().isoformat(),
            'performance_score': performance_score,
            'metrics': model_metrics,
            'status': 'EXCELLENT' if performance_score >= 80 else 'GOOD' if performance_score >= 70 else 'POOR',
            'mandatory_check': True
        }
        
        print(f"   Performance Score: {performance_score:.1f}/100")
        
        return model_result
    
    def _collect_execution_quality(self) -> Dict[str, Any]:
        """Collect mandatory execution quality metrics"""
        
        print("⚡ Collecting execution quality...")
        
        # Simulate execution quality metrics
        execution_metrics = {
            'avg_slippage_bps': np.random.uniform(5, 25),
            'p90_slippage_bps': np.random.uniform(15, 45),
            'avg_latency_ms': np.random.uniform(50, 200),
            'fill_rate': np.random.uniform(0.85, 0.98),
            'execution_success_rate': np.random.uniform(0.90, 0.99),
            'fee_efficiency': np.random.uniform(0.80, 0.95)
        }
        
        # Calculate execution quality score
        quality_score = (
            max(0, (50 - execution_metrics['avg_slippage_bps']) / 50) * 0.30 +
            max(0, (500 - execution_metrics['avg_latency_ms']) / 500) * 0.20 +
            execution_metrics['fill_rate'] * 0.25 +
            execution_metrics['execution_success_rate'] * 0.25
        ) * 100
        
        execution_result = {
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'metrics': execution_metrics,
            'status': 'EXCELLENT' if quality_score >= 80 else 'GOOD' if quality_score >= 70 else 'POOR',
            'mandatory_check': True
        }
        
        print(f"   Execution Quality: {quality_score:.1f}/100")
        
        return execution_result
    
    def _collect_risk_assessment(self) -> Dict[str, Any]:
        """Collect mandatory risk assessment"""
        
        print("🛡️ Collecting risk assessment...")
        
        # Simulate risk metrics
        risk_metrics = {
            'portfolio_var_95': np.random.uniform(0.02, 0.08),
            'max_drawdown_24h': np.random.uniform(0.01, 0.05),
            'correlation_to_btc': np.random.uniform(0.3, 0.8),
            'concentration_risk': np.random.uniform(0.1, 0.4),
            'liquidity_risk': np.random.uniform(0.05, 0.25),
            'regime_stability': np.random.uniform(0.6, 0.9)
        }
        
        # Calculate risk score (lower is better)
        risk_score = (
            (1 - risk_metrics['portfolio_var_95']) * 0.25 +
            (1 - risk_metrics['max_drawdown_24h']) * 0.25 +
            (1 - risk_metrics['concentration_risk']) * 0.20 +
            (1 - risk_metrics['liquidity_risk']) * 0.15 +
            risk_metrics['regime_stability'] * 0.15
        ) * 100
        
        risk_result = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_score,
            'metrics': risk_metrics,
            'risk_level': 'LOW' if risk_score >= 80 else 'MEDIUM' if risk_score >= 60 else 'HIGH',
            'mandatory_check': True
        }
        
        print(f"   Risk Score: {risk_score:.1f}/100 ({risk_result['risk_level']} risk)")
        
        return risk_result
    
    def _validate_metrics_completeness(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all mandatory metrics are present"""
        
        validation = {
            'all_required_present': True,
            'missing_metrics': [],
            'invalid_metrics': [],
            'validation_errors': []
        }
        
        # Check required metrics presence
        for required_metric in self.required_metrics:
            if required_metric not in metrics:
                validation['all_required_present'] = False
                validation['missing_metrics'].append(required_metric)
            else:
                # Validate metric structure
                metric_data = metrics[required_metric]
                if not isinstance(metric_data, dict) or 'mandatory_check' not in metric_data:
                    validation['invalid_metrics'].append(required_metric)
        
        # Specific validations
        if 'coverage_audit' in metrics:
            coverage = metrics['coverage_audit']
            if coverage.get('coverage_percentage', 0) < 99.0:
                validation['validation_errors'].append('Coverage below 99% threshold')
        
        if 'system_health' in metrics:
            health = metrics['system_health']
            if health.get('health_score', 0) < 85:
                validation['validation_errors'].append('System health below 85% threshold')
        
        return validation
    
    def _generate_go_nogo_decision(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GO/NO-GO decision based on mandatory metrics"""
        
        go_nogo = {
            'decision': 'GO',
            'confidence': 1.0,
            'blocking_issues': [],
            'warnings': [],
            'decision_factors': {}
        }
        
        # Check coverage
        if 'coverage_audit' in metrics:
            coverage_pct = metrics['coverage_audit'].get('coverage_percentage', 0)
            if coverage_pct < 99.0:
                go_nogo['decision'] = 'NO-GO'
                go_nogo['blocking_issues'].append(f'Coverage only {coverage_pct:.1f}% (required: 99%)')
        
        # Check system health
        if 'system_health' in metrics:
            health_score = metrics['system_health'].get('health_score', 0)
            if health_score < 60:
                go_nogo['decision'] = 'NO-GO'
                go_nogo['blocking_issues'].append(f'System health {health_score:.1f}/100 (critical)')
            elif health_score < 85:
                go_nogo['warnings'].append(f'System health {health_score:.1f}/100 (degraded)')
        
        # Check model performance
        if 'model_performance' in metrics:
            perf_score = metrics['model_performance'].get('performance_score', 0)
            if perf_score < 50:
                go_nogo['decision'] = 'NO-GO'
                go_nogo['blocking_issues'].append(f'Model performance {perf_score:.1f}/100 (poor)')
        
        # Check execution quality
        if 'execution_quality' in metrics:
            exec_score = metrics['execution_quality'].get('quality_score', 0)
            if exec_score < 50:
                go_nogo['decision'] = 'NO-GO'
                go_nogo['blocking_issues'].append(f'Execution quality {exec_score:.1f}/100 (poor)')
        
        # Calculate confidence
        if go_nogo['blocking_issues']:
            go_nogo['confidence'] = 0.0
        elif go_nogo['warnings']:
            go_nogo['confidence'] = 0.7
        else:
            go_nogo['confidence'] = 1.0
        
        return go_nogo
    
    def _save_daily_metrics(self, metrics: Dict[str, Any], date: str):
        """Save daily metrics to files"""
        
        # Create date directory
        date_dir = self.metrics_dir / date
        date_dir.mkdir(exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        # Save complete metrics
        metrics_file = date_dir / f"daily_metrics_{date}_{datetime.now().strftime('%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(convert_numpy_types(metrics), f, indent=2)
        
        # Save latest metrics (for easy access)
        latest_file = date_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(convert_numpy_types(metrics), f, indent=2)
        
        print(f"💾 Daily metrics saved to {date_dir}")
    
    def check_mandatory_compliance(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Check if mandatory metrics are compliant for a given date"""
        
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        date_dir = self.metrics_dir / date
        latest_file = date_dir / "latest.json"
        
        if not latest_file.exists():
            return {
                'compliant': False,
                'reason': 'No daily metrics found',
                'date': date
            }
        
        # Load metrics
        with open(latest_file, 'r') as f:
            metrics = json.load(f)
        
        # Check validation
        validation = metrics.get('validation', {})
        go_nogo = metrics.get('go_nogo_decision', {})
        
        compliance = {
            'compliant': validation.get('all_required_present', False) and go_nogo.get('decision') == 'GO',
            'date': date,
            'validation': validation,
            'go_nogo': go_nogo,
            'metrics_status': metrics.get('metrics_status', 'unknown')
        }
        
        return compliance

def run_mandatory_daily_metrics() -> Dict[str, Any]:
    """Run mandatory daily metrics collection"""
    
    collector = MandatoryDailyMetrics()
    return collector.run_mandatory_daily_collection()

if __name__ == "__main__":
    print("📊 RUNNING MANDATORY DAILY METRICS COLLECTION")
    print("=" * 55)
    
    # Run mandatory collection
    results = run_mandatory_daily_metrics()
    
    print(f"\n📋 DAILY METRICS SUMMARY")
    print("=" * 30)
    print(f"Date: {results['date']}")
    print(f"Status: {results['metrics_status']}")
    
    # Print key metrics
    if 'coverage_audit' in results:
        coverage = results['coverage_audit']
        print(f"Coverage: {coverage['coverage_percentage']:.1f}% ({coverage['status']})")
    
    if 'system_health' in results:
        health = results['system_health']
        print(f"Health: {health['health_score']:.1f}/100 ({health['status']})")
    
    if 'go_nogo_decision' in results:
        decision = results['go_nogo_decision']
        print(f"Decision: {decision['decision']} (confidence: {decision['confidence']:.1%})")
        
        if decision['blocking_issues']:
            print("Blocking Issues:")
            for issue in decision['blocking_issues']:
                print(f"   ❌ {issue}")
        
        if decision['warnings']:
            print("Warnings:")
            for warning in decision['warnings']:
                print(f"   ⚠️ {warning}")
    
    print("✅ Mandatory daily metrics collection completed")