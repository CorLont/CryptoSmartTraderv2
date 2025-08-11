#!/usr/bin/env python3
"""
System Health Monitor with GO/NO-GO Decision System
Implements enterprise health scoring with weighted components
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

class HealthScoreCalculator:
    """Calculate weighted health scores across system components"""
    
    def __init__(self):
        self.logger = get_structured_logger("HealthScoreCalculator")
        
        # Weighted components (must sum to 100%)
        self.component_weights = {
            'validation_accuracy': 0.25,  # 25%
            'sharpe_ratio': 0.20,        # 20%
            'feedback_hit_rate': 0.15,   # 15%
            'error_ratio': 0.15,         # 15%
            'data_completeness': 0.15,   # 15%
            'tuning_freshness': 0.10     # 10%
        }
    
    def calculate_validation_accuracy_score(self, evaluation_metrics: Dict[str, Any]) -> float:
        """Calculate validation accuracy component score (0-100)"""
        
        try:
            # Extract key metrics
            precision_5 = evaluation_metrics.get('precision_at_k', {}).get('precision_at_5', 0)
            hit_rate_30d = evaluation_metrics.get('hit_rates', {}).get('hit_rate_30d', 0)
            mae_normalized = evaluation_metrics.get('mae_metrics', {}).get('mae_normalized_30d', float('inf'))
            
            # Scoring thresholds
            precision_score = min(100, max(0, precision_5 * 100 / 0.60))  # Target: 60%
            hit_rate_score = min(100, max(0, hit_rate_30d * 100 / 0.55))  # Target: 55%
            mae_score = min(100, max(0, (0.25 - min(mae_normalized, 0.25)) / 0.25 * 100))  # Target: ≤0.25
            
            # Weighted average
            accuracy_score = (precision_score * 0.4 + hit_rate_score * 0.3 + mae_score * 0.3)
            
            self.logger.info(f"Validation accuracy score: {accuracy_score:.1f}")
            return accuracy_score
            
        except Exception as e:
            self.logger.error(f"Validation accuracy calculation failed: {e}")
            return 0.0
    
    def calculate_sharpe_ratio_score(self, evaluation_metrics: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio component score (0-100)"""
        
        try:
            sharpe_ratio = evaluation_metrics.get('sharpe_metrics', {}).get('sharpe_ratio', 0)
            
            # Scoring: Target Sharpe ≥ 1.0
            # 0.0 → 0 points, 1.0 → 80 points, 2.0+ → 100 points
            if sharpe_ratio >= 2.0:
                sharpe_score = 100
            elif sharpe_ratio >= 1.0:
                sharpe_score = 80 + (sharpe_ratio - 1.0) * 20
            else:
                sharpe_score = max(0, sharpe_ratio * 80)
            
            self.logger.info(f"Sharpe ratio score: {sharpe_score:.1f}")
            return sharpe_score
            
        except Exception as e:
            self.logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def calculate_feedback_hit_rate_score(self, feedback_data: Dict[str, Any]) -> float:
        """Calculate feedback hit rate component score (0-100)"""
        
        try:
            # Mock feedback data - in production this would come from user feedback
            total_trades = feedback_data.get('total_trades', 100)
            successful_trades = feedback_data.get('successful_trades', 65)
            
            if total_trades > 0:
                hit_rate = successful_trades / total_trades
                # Target: 65% hit rate
                hit_rate_score = min(100, max(0, hit_rate * 100 / 0.65 * 100))
            else:
                hit_rate_score = 50  # Neutral score when no feedback
            
            self.logger.info(f"Feedback hit rate score: {hit_rate_score:.1f}")
            return hit_rate_score
            
        except Exception as e:
            self.logger.error(f"Feedback hit rate calculation failed: {e}")
            return 50.0
    
    def calculate_error_ratio_score(self, system_logs: Dict[str, Any]) -> float:
        """Calculate error ratio component score (0-100)"""
        
        try:
            total_operations = system_logs.get('total_operations', 1000)
            error_count = system_logs.get('error_count', 15)
            
            if total_operations > 0:
                error_ratio = error_count / total_operations
                # Target: <2% error ratio
                if error_ratio <= 0.02:
                    error_score = 100
                elif error_ratio <= 0.05:
                    error_score = 100 - (error_ratio - 0.02) / 0.03 * 30  # 100→70
                else:
                    error_score = max(0, 70 - (error_ratio - 0.05) * 1400)  # Rapid decline
            else:
                error_score = 100  # No operations = no errors
            
            self.logger.info(f"Error ratio score: {error_score:.1f}")
            return error_score
            
        except Exception as e:
            self.logger.error(f"Error ratio calculation failed: {e}")
            return 70.0
    
    def calculate_data_completeness_score(self, coverage_audit: Dict[str, Any]) -> float:
        """Calculate data completeness component score (0-100)"""
        
        try:
            coverage_pct = coverage_audit.get('coverage_analysis', {}).get('coverage_summary', {}).get('coverage_percentage', 90)
            
            # Target: >98% completeness
            if coverage_pct >= 98:
                completeness_score = 100
            elif coverage_pct >= 95:
                completeness_score = 80 + (coverage_pct - 95) / 3 * 20  # 80→100
            else:
                completeness_score = max(0, coverage_pct - 20)  # Linear below 95%
            
            self.logger.info(f"Data completeness score: {completeness_score:.1f}")
            return completeness_score
            
        except Exception as e:
            self.logger.error(f"Data completeness calculation failed: {e}")
            return 80.0
    
    def calculate_tuning_freshness_score(self, model_info: Dict[str, Any]) -> float:
        """Calculate model tuning freshness component score (0-100)"""
        
        try:
            last_training = model_info.get('last_training_timestamp', datetime.now() - timedelta(days=3))
            if isinstance(last_training, str):
                last_training = datetime.fromisoformat(last_training)
            
            days_since_training = (datetime.now() - last_training).days
            
            # Target: <7 days since last training
            if days_since_training <= 7:
                freshness_score = 100
            elif days_since_training <= 14:
                freshness_score = 100 - (days_since_training - 7) / 7 * 30  # 100→70
            else:
                freshness_score = max(0, 70 - (days_since_training - 14) * 5)  # Rapid decline
            
            self.logger.info(f"Tuning freshness score: {freshness_score:.1f}")
            return freshness_score
            
        except Exception as e:
            self.logger.error(f"Tuning freshness calculation failed: {e}")
            return 70.0
    
    def calculate_overall_health_score(self, evaluation_metrics: Dict[str, Any],
                                     feedback_data: Dict[str, Any],
                                     system_logs: Dict[str, Any],
                                     coverage_audit: Dict[str, Any],
                                     model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall weighted health score"""
        
        try:
            # Calculate component scores
            component_scores = {
                'validation_accuracy': self.calculate_validation_accuracy_score(evaluation_metrics),
                'sharpe_ratio': self.calculate_sharpe_ratio_score(evaluation_metrics),
                'feedback_hit_rate': self.calculate_feedback_hit_rate_score(feedback_data),
                'error_ratio': self.calculate_error_ratio_score(system_logs),
                'data_completeness': self.calculate_data_completeness_score(coverage_audit),
                'tuning_freshness': self.calculate_tuning_freshness_score(model_info)
            }
            
            # Calculate weighted overall score
            overall_score = sum(
                score * self.component_weights[component] 
                for component, score in component_scores.items()
            )
            
            # Determine GO/NO-GO status
            if overall_score >= 85:
                status = "GO"
                recommendation = "Live trading authorized"
            elif overall_score >= 60:
                status = "WARNING"
                recommendation = "Paper trading only - improve weak components"
            else:
                status = "NO-GO"
                recommendation = "Trading blocked - critical issues detected"
            
            health_assessment = {
                'overall_score': overall_score,
                'status': status,
                'recommendation': recommendation,
                'component_scores': component_scores,
                'component_weights': self.component_weights,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Overall health score: {overall_score:.1f} ({status})")
            
            return health_assessment
            
        except Exception as e:
            self.logger.error(f"Overall health calculation failed: {e}")
            return {
                'overall_score': 0.0,
                'status': 'NO-GO',
                'recommendation': 'Health calculation failed',
                'error': str(e)
            }

class SystemHealthMonitor:
    """Complete system health monitoring with GO/NO-GO decisions"""
    
    def __init__(self):
        self.logger = get_structured_logger("SystemHealthMonitor")
        self.health_calculator = HealthScoreCalculator()
    
    def run_health_assessment(self, evaluation_metrics: Dict[str, Any] = None,
                            coverage_audit: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive health assessment"""
        
        start_time = time.time()
        
        self.logger.info("Starting comprehensive health assessment")
        
        try:
            # Use provided data or generate mock data
            if evaluation_metrics is None:
                evaluation_metrics = self._get_mock_evaluation_metrics()
            
            if coverage_audit is None:
                coverage_audit = self._get_mock_coverage_audit()
            
            # Get additional data
            feedback_data = self._get_feedback_data()
            system_logs = self._get_system_logs()
            model_info = self._get_model_info()
            
            # Calculate health score
            health_assessment = self.health_calculator.calculate_overall_health_score(
                evaluation_metrics, feedback_data, system_logs, coverage_audit, model_info
            )
            
            # Add assessment metadata
            health_assessment.update({
                'assessment_duration': time.time() - start_time,
                'assessment_timestamp': datetime.now().isoformat(),
                'data_sources': {
                    'evaluation_metrics': bool(evaluation_metrics),
                    'coverage_audit': bool(coverage_audit),
                    'feedback_data': bool(feedback_data),
                    'system_logs': bool(system_logs),
                    'model_info': bool(model_info)
                }
            })
            
            self.logger.info(f"Health assessment completed in {time.time() - start_time:.2f}s")
            
            return health_assessment
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            return {
                'overall_score': 0.0,
                'status': 'NO-GO',
                'recommendation': 'Health assessment failed',
                'error': str(e),
                'assessment_timestamp': datetime.now().isoformat()
            }
    
    def _get_mock_evaluation_metrics(self) -> Dict[str, Any]:
        """Mock evaluation metrics for testing"""
        
        return {
            'precision_at_k': {
                'precision_at_5': 0.65  # Above 60% threshold
            },
            'hit_rates': {
                'hit_rate_30d': 0.58  # Above 55% threshold
            },
            'mae_metrics': {
                'mae_normalized_30d': 0.22  # Below 0.25 threshold
            },
            'sharpe_metrics': {
                'sharpe_ratio': 1.2  # Above 1.0 threshold
            }
        }
    
    def _get_mock_coverage_audit(self) -> Dict[str, Any]:
        """Mock coverage audit for testing"""
        
        return {
            'coverage_analysis': {
                'coverage_summary': {
                    'coverage_percentage': 96.5  # Above 95% threshold
                }
            }
        }
    
    def _get_feedback_data(self) -> Dict[str, Any]:
        """Get user feedback data"""
        
        # Mock feedback data - in production this would come from user feedback
        return {
            'total_trades': 150,
            'successful_trades': 98,
            'feedback_period_days': 30
        }
    
    def _get_system_logs(self) -> Dict[str, Any]:
        """Get system error logs"""
        
        # Mock system logs
        return {
            'total_operations': 2500,
            'error_count': 35,
            'critical_errors': 2,
            'log_period_days': 7
        }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model training information"""
        
        return {
            'last_training_timestamp': datetime.now() - timedelta(days=5),
            'model_version': 'v2.1.3',
            'training_data_size': 50000,
            'validation_score': 0.87
        }

def save_health_assessment(health_assessment: Dict[str, Any]) -> Path:
    """Save health assessment to daily logs"""
    
    try:
        # Create daily log directory
        today_str = datetime.now().strftime("%Y%m%d")
        daily_log_dir = Path("logs/daily") / today_str
        daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save timestamped assessment
        timestamp_str = datetime.now().strftime("%H%M%S")
        assessment_file = daily_log_dir / f"health_assessment_{timestamp_str}.json"
        
        with open(assessment_file, 'w') as f:
            json.dump(health_assessment, f, indent=2)
        
        # Save as latest
        latest_file = daily_log_dir / "health_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(health_assessment, f, indent=2)
        
        return assessment_file
        
    except Exception as e:
        print(f"Failed to save health assessment: {e}")
        return None

if __name__ == "__main__":
    # Test system health monitor
    print("🔍 TESTING SYSTEM HEALTH MONITOR")
    print("=" * 60)
    
    # Run health assessment
    monitor = SystemHealthMonitor()
    health_results = monitor.run_health_assessment()
    
    print("\n🏥 HEALTH ASSESSMENT RESULTS:")
    print(f"Overall Score: {health_results['overall_score']:.1f}/100")
    print(f"Status: {health_results['status']}")
    print(f"Recommendation: {health_results['recommendation']}")
    
    print("\n📊 COMPONENT SCORES:")
    component_scores = health_results.get('component_scores', {})
    for component, score in component_scores.items():
        print(f"• {component}: {score:.1f}")
    
    print("\n🚪 GO/NO-GO DECISION:")
    status = health_results['status']
    if status == "GO":
        print("✅ GO - Live trading authorized")
    elif status == "WARNING":
        print("⚠️ WARNING - Paper trading only")
    else:
        print("❌ NO-GO - Trading blocked")
    
    # Save results
    saved_file = save_health_assessment(health_results)
    if saved_file:
        print(f"\n💾 Results saved: {saved_file}")
    
    print(f"\n⏱️ Assessment duration: {health_results.get('assessment_duration', 0):.2f}s")