#!/usr/bin/env python3
"""
Drift Detection System
Error trending, KS-test on feature distributions, and automated drift alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

try:
    from scipy.stats import ks_2samp, chi2_contingency
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    timestamp: datetime
    drift_type: str  # 'error_trending', 'feature_distribution', 'performance_degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str  # 'ml_model', 'data_pipeline', 'sentiment_analyzer', etc.
    metrics: Dict[str, Any]
    description: str
    recommended_action: str

@dataclass
class FeatureDistribution:
    """Feature distribution statistics"""
    feature_name: str
    timestamp: datetime
    mean: float
    std: float
    median: float
    percentiles: Dict[str, float]  # p5, p25, p75, p95
    min_value: float
    max_value: float
    sample_count: int

class ErrorTrendDetector:
    """Detects trending errors and performance degradation"""
    
    def __init__(self, window_size: int = 100, trend_threshold: float = 0.1):
        self.window_size = window_size
        self.trend_threshold = trend_threshold  # 10% increase triggers alert
        self.logger = get_structured_logger("ErrorTrendDetector")
        
        # Error tracking by component
        self.error_history: Dict[str, deque] = {}
        self.performance_history: Dict[str, deque] = {}
        
    def record_error(self, component: str, error_rate: float, timestamp: Optional[datetime] = None) -> None:
        """Record error rate for component"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        if component not in self.error_history:
            self.error_history[component] = deque(maxlen=self.window_size)
            
        self.error_history[component].append({
            'timestamp': timestamp,
            'error_rate': error_rate
        })
        
        self.logger.debug(f"Recorded error rate {error_rate:.3f} for {component}")
    
    def record_performance(self, component: str, metric: str, value: float, 
                          timestamp: Optional[datetime] = None) -> None:
        """Record performance metric for component"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        key = f"{component}_{metric}"
        if key not in self.performance_history:
            self.performance_history[key] = deque(maxlen=self.window_size)
            
        self.performance_history[key].append({
            'timestamp': timestamp,
            'value': value
        })
        
        self.logger.debug(f"Recorded {metric} {value:.3f} for {component}")
    
    def detect_error_trend(self, component: str) -> Optional[DriftAlert]:
        """Detect error rate trending for component"""
        
        if component not in self.error_history:
            return None
            
        history = list(self.error_history[component])
        if len(history) < 10:  # Need minimum data points
            return None
        
        try:
            # Extract error rates and smooth trend line
            error_rates = [h['error_rate'] for h in history]
            timestamps = [h['timestamp'] for h in history]
            
            if SCIPY_AVAILABLE and len(error_rates) >= 5:
                # Smooth the data using Savitzky-Golay filter
                if len(error_rates) >= 5:
                    smoothed = savgol_filter(error_rates, 
                                           min(5, len(error_rates) if len(error_rates) % 2 == 1 else len(error_rates) - 1), 
                                           2)
                else:
                    smoothed = error_rates
            else:
                # Simple moving average
                window = min(5, len(error_rates))
                smoothed = np.convolve(error_rates, np.ones(window)/window, mode='valid')
            
            # Calculate trend (slope of recent data)
            if len(smoothed) >= 5:
                recent_points = smoothed[-5:]
                x = np.arange(len(recent_points))
                trend_slope = np.polyfit(x, recent_points, 1)[0]
                
                # Check if trend exceeds threshold
                baseline_error = np.mean(error_rates[:max(1, len(error_rates)//2)])
                trend_change = trend_slope * len(recent_points)
                relative_change = trend_change / max(baseline_error, 0.001)
                
                if relative_change > self.trend_threshold:
                    severity = self._determine_severity(relative_change)
                    
                    alert = DriftAlert(
                        alert_id=f"error_trend_{component}_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        drift_type="error_trending",
                        severity=severity,
                        component=component,
                        metrics={
                            'trend_slope': trend_slope,
                            'relative_change': relative_change,
                            'current_error_rate': error_rates[-1],
                            'baseline_error_rate': baseline_error,
                            'data_points': len(history)
                        },
                        description=f"Error rate trending upward for {component}. "
                                  f"Relative increase: {relative_change:.1%}",
                        recommended_action="Investigate recent changes, consider fine-tuning or rollback"
                    )
                    
                    self.logger.warning(f"Error trend detected for {component}: {relative_change:.1%} increase")
                    return alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error trend detection failed for {component}: {e}")
            return None
    
    def detect_performance_degradation(self, component: str, metric: str) -> Optional[DriftAlert]:
        """Detect performance degradation for specific metric"""
        
        key = f"{component}_{metric}"
        if key not in self.performance_history:
            return None
            
        history = list(self.performance_history[key])
        if len(history) < 10:
            return None
        
        try:
            values = [h['value'] for h in history]
            
            # Compare recent performance to baseline
            baseline_period = max(1, len(values) // 3)
            recent_period = max(5, len(values) // 4)
            
            baseline_perf = np.mean(values[:baseline_period])
            recent_perf = np.mean(values[-recent_period:])
            
            # Calculate degradation (assumes lower is worse for most metrics)
            if baseline_perf != 0:
                degradation = (baseline_perf - recent_perf) / abs(baseline_perf)
            else:
                degradation = 0
            
            # For metrics where higher is worse (like error rates), flip the sign
            if metric in ['error_rate', 'latency', 'loss']:
                degradation = -degradation
            
            if degradation > self.trend_threshold:
                severity = self._determine_severity(degradation)
                
                alert = DriftAlert(
                    alert_id=f"perf_degrade_{component}_{metric}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    drift_type="performance_degradation",
                    severity=severity,
                    component=component,
                    metrics={
                        'degradation': degradation,
                        'baseline_performance': baseline_perf,
                        'recent_performance': recent_perf,
                        'metric_name': metric,
                        'data_points': len(history)
                    },
                    description=f"Performance degradation detected for {component} {metric}. "
                              f"Degradation: {degradation:.1%}",
                    recommended_action="Check for data quality issues, consider model retraining"
                )
                
                self.logger.warning(f"Performance degradation detected for {component} {metric}: {degradation:.1%}")
                return alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"Performance degradation detection failed for {component} {metric}: {e}")
            return None
    
    def _determine_severity(self, change_ratio: float) -> str:
        """Determine alert severity based on change ratio"""
        if change_ratio > 0.5:  # 50%+ change
            return "critical"
        elif change_ratio > 0.3:  # 30%+ change
            return "high"
        elif change_ratio > 0.2:  # 20%+ change
            return "medium"
        else:
            return "low"

class FeatureDistributionMonitor:
    """Monitors feature distributions using KS-test"""
    
    def __init__(self, reference_window: int = 1000, test_window: int = 100):
        self.reference_window = reference_window
        self.test_window = test_window
        self.logger = get_structured_logger("FeatureDistributionMonitor")
        
        # Feature distribution history
        self.reference_distributions: Dict[str, List[float]] = {}
        self.recent_distributions: Dict[str, deque] = {}
        
    def record_feature_batch(self, features: Dict[str, List[float]], 
                           timestamp: Optional[datetime] = None) -> None:
        """Record batch of feature values"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        for feature_name, values in features.items():
            # Store reference distribution (first N samples)
            if feature_name not in self.reference_distributions:
                self.reference_distributions[feature_name] = []
                self.recent_distributions[feature_name] = deque(maxlen=self.test_window)
            
            # Build reference distribution
            if len(self.reference_distributions[feature_name]) < self.reference_window:
                remaining_space = self.reference_window - len(self.reference_distributions[feature_name])
                self.reference_distributions[feature_name].extend(values[:remaining_space])
            
            # Always add to recent distribution
            self.recent_distributions[feature_name].extend(values)
            
        self.logger.debug(f"Recorded feature batch for {len(features)} features")
    
    def calculate_distribution_stats(self, feature_name: str, values: List[float], 
                                   timestamp: Optional[datetime] = None) -> FeatureDistribution:
        """Calculate distribution statistics for feature"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        values_array = np.array(values)
        
        stats = FeatureDistribution(
            feature_name=feature_name,
            timestamp=timestamp,
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            median=float(np.median(values_array)),
            percentiles={
                'p5': float(np.percentile(values_array, 5)),
                'p25': float(np.percentile(values_array, 25)),
                'p75': float(np.percentile(values_array, 75)),
                'p95': float(np.percentile(values_array, 95))
            },
            min_value=float(np.min(values_array)),
            max_value=float(np.max(values_array)),
            sample_count=len(values)
        )
        
        return stats
    
    def detect_distribution_drift(self, feature_name: str) -> Optional[DriftAlert]:
        """Detect distribution drift using KS-test"""
        
        if feature_name not in self.reference_distributions:
            return None
            
        if feature_name not in self.recent_distributions:
            return None
            
        reference_data = self.reference_distributions[feature_name]
        recent_data = list(self.recent_distributions[feature_name])
        
        if len(reference_data) < 50 or len(recent_data) < 20:
            return None  # Not enough data
        
        try:
            if SCIPY_AVAILABLE:
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = ks_2samp(reference_data, recent_data)
                
                # Alert if distributions are significantly different
                if p_value < 0.01:  # 99% confidence
                    severity = self._determine_drift_severity(ks_statistic, p_value)
                    
                    # Calculate distribution stats for both periods
                    ref_stats = self.calculate_distribution_stats(f"{feature_name}_reference", reference_data)
                    recent_stats = self.calculate_distribution_stats(f"{feature_name}_recent", recent_data)
                    
                    alert = DriftAlert(
                        alert_id=f"feature_drift_{feature_name}_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        drift_type="feature_distribution",
                        severity=severity,
                        component="data_pipeline",
                        metrics={
                            'feature_name': feature_name,
                            'ks_statistic': ks_statistic,
                            'p_value': p_value,
                            'reference_mean': ref_stats.mean,
                            'recent_mean': recent_stats.mean,
                            'reference_std': ref_stats.std,
                            'recent_std': recent_stats.std,
                            'mean_shift': abs(recent_stats.mean - ref_stats.mean) / max(ref_stats.std, 0.001),
                            'std_change': abs(recent_stats.std - ref_stats.std) / max(ref_stats.std, 0.001)
                        },
                        description=f"Distribution drift detected for feature {feature_name}. "
                                  f"KS statistic: {ks_statistic:.3f}, p-value: {p_value:.3e}",
                        recommended_action="Investigate data source changes, consider feature engineering updates"
                    )
                    
                    self.logger.warning(f"Distribution drift detected for {feature_name}: "
                                      f"KS={ks_statistic:.3f}, p={p_value:.3e}")
                    return alert
            else:
                # Simple statistical comparison without scipy
                ref_mean = np.mean(reference_data)
                ref_std = np.std(reference_data)
                recent_mean = np.mean(recent_data)
                recent_std = np.std(recent_data)
                
                # Calculate simple drift metrics
                mean_shift = abs(recent_mean - ref_mean) / max(ref_std, 0.001)
                std_change = abs(recent_std - ref_std) / max(ref_std, 0.001)
                
                if mean_shift > 2.0 or std_change > 0.5:  # 2 std or 50% std change
                    severity = "medium" if mean_shift > 3.0 or std_change > 1.0 else "low"
                    
                    alert = DriftAlert(
                        alert_id=f"feature_drift_{feature_name}_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        drift_type="feature_distribution",
                        severity=severity,
                        component="data_pipeline",
                        metrics={
                            'feature_name': feature_name,
                            'mean_shift': mean_shift,
                            'std_change': std_change,
                            'reference_mean': ref_mean,
                            'recent_mean': recent_mean,
                            'reference_std': ref_std,
                            'recent_std': recent_std
                        },
                        description=f"Statistical drift detected for feature {feature_name}. "
                                  f"Mean shift: {mean_shift:.2f} std, Std change: {std_change:.1%}",
                        recommended_action="Check data quality and feature generation pipeline"
                    )
                    
                    self.logger.warning(f"Statistical drift detected for {feature_name}: "
                                      f"mean_shift={mean_shift:.2f}, std_change={std_change:.1%}")
                    return alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"Distribution drift detection failed for {feature_name}: {e}")
            return None
    
    def _determine_drift_severity(self, ks_statistic: float, p_value: float) -> str:
        """Determine drift severity based on KS test results"""
        if p_value < 0.001 and ks_statistic > 0.3:
            return "critical"
        elif p_value < 0.001 and ks_statistic > 0.2:
            return "high"
        elif p_value < 0.01 and ks_statistic > 0.1:
            return "medium"
        else:
            return "low"

class DriftDetectionSystem:
    """Complete drift detection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_structured_logger("DriftDetectionSystem")
        
        # Default configuration
        self.config = {
            'error_trend_window': 100,
            'error_trend_threshold': 0.15,  # 15% increase
            'feature_reference_window': 1000,
            'feature_test_window': 100,
            'monitoring_interval': 300,  # 5 minutes
            'alert_cooldown': 3600,  # 1 hour between same alerts
            'enabled_detectors': ['error_trending', 'feature_distribution', 'performance_degradation']
        }
        
        if config:
            self.config.update(config)
        
        # Initialize detectors
        self.error_detector = ErrorTrendDetector(
            window_size=self.config['error_trend_window'],
            trend_threshold=self.config['error_trend_threshold']
        )
        
        self.feature_monitor = FeatureDistributionMonitor(
            reference_window=self.config['feature_reference_window'],
            test_window=self.config['feature_test_window']
        )
        
        # Alert management
        self.recent_alerts: Dict[str, datetime] = {}
        self.all_alerts: List[DriftAlert] = []
        
    def record_error_metrics(self, component: str, error_rate: float, 
                           timestamp: Optional[datetime] = None) -> None:
        """Record error metrics for drift detection"""
        if 'error_trending' in self.config['enabled_detectors']:
            self.error_detector.record_error(component, error_rate, timestamp)
    
    def record_performance_metrics(self, component: str, metric: str, value: float,
                                 timestamp: Optional[datetime] = None) -> None:
        """Record performance metrics for drift detection"""
        if 'performance_degradation' in self.config['enabled_detectors']:
            self.error_detector.record_performance(component, metric, value, timestamp)
    
    def record_feature_data(self, features: Dict[str, List[float]], 
                          timestamp: Optional[datetime] = None) -> None:
        """Record feature data for distribution monitoring"""
        if 'feature_distribution' in self.config['enabled_detectors']:
            self.feature_monitor.record_feature_batch(features, timestamp)
    
    def run_drift_detection(self) -> List[DriftAlert]:
        """Run complete drift detection and return alerts"""
        
        alerts = []
        
        try:
            # Error trend detection
            if 'error_trending' in self.config['enabled_detectors']:
                for component in self.error_detector.error_history.keys():
                    alert = self.error_detector.detect_error_trend(component)
                    if alert and self._should_send_alert(alert):
                        alerts.append(alert)
            
            # Performance degradation detection
            if 'performance_degradation' in self.config['enabled_detectors']:
                for key in self.error_detector.performance_history.keys():
                    component, metric = key.rsplit('_', 1)
                    alert = self.error_detector.detect_performance_degradation(component, metric)
                    if alert and self._should_send_alert(alert):
                        alerts.append(alert)
            
            # Feature distribution drift detection
            if 'feature_distribution' in self.config['enabled_detectors']:
                for feature_name in self.feature_monitor.reference_distributions.keys():
                    alert = self.feature_monitor.detect_distribution_drift(feature_name)
                    if alert and self._should_send_alert(alert):
                        alerts.append(alert)
            
            # Store all alerts
            self.all_alerts.extend(alerts)
            
            if alerts:
                self.logger.info(f"Drift detection completed: {len(alerts)} new alerts")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return []
    
    def _should_send_alert(self, alert: DriftAlert) -> bool:
        """Check if alert should be sent (respects cooldown)"""
        
        alert_key = f"{alert.drift_type}_{alert.component}"
        
        if alert_key in self.recent_alerts:
            time_since_last = datetime.now() - self.recent_alerts[alert_key]
            if time_since_last.total_seconds() < self.config['alert_cooldown']:
                return False
        
        self.recent_alerts[alert_key] = datetime.now()
        return True
    
    def get_recent_alerts(self, hours: int = 24) -> List[DriftAlert]:
        """Get recent alerts within specified hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.all_alerts if alert.timestamp > cutoff_time]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get drift detection system status"""
        
        recent_alerts = self.get_recent_alerts(24)
        
        status = {
            'detectors_enabled': self.config['enabled_detectors'],
            'total_alerts': len(self.all_alerts),
            'alerts_24h': len(recent_alerts),
            'components_monitored': {
                'error_tracking': len(self.error_detector.error_history),
                'performance_tracking': len(self.error_detector.performance_history),
                'features_monitored': len(self.feature_monitor.reference_distributions)
            },
            'alert_breakdown': {
                'critical': len([a for a in recent_alerts if a.severity == 'critical']),
                'high': len([a for a in recent_alerts if a.severity == 'high']),
                'medium': len([a for a in recent_alerts if a.severity == 'medium']),
                'low': len([a for a in recent_alerts if a.severity == 'low'])
            },
            'last_detection_run': datetime.now().isoformat()
        }
        
        return status

if __name__ == "__main__":
    async def test_drift_detection():
        """Test drift detection system"""
        
        print("🔍 TESTING DRIFT DETECTION SYSTEM")
        print("=" * 60)
        
        # Create drift detection system
        drift_system = DriftDetectionSystem()
        
        print("📊 Generating test data with drift...")
        
        # Simulate error rate trending
        for i in range(20):
            # Gradually increasing error rate
            error_rate = 0.01 + (i * 0.005)  # 1% to 10%
            drift_system.record_error_metrics("ml_predictor", error_rate)
        
        # Simulate performance degradation
        for i in range(20):
            # Decreasing accuracy
            accuracy = 0.95 - (i * 0.01)  # 95% to 75%
            drift_system.record_performance_metrics("ml_predictor", "accuracy", accuracy)
        
        # Simulate feature distribution drift
        import random
        
        # Reference distribution (normal)
        for batch in range(10):
            features = {
                'price_volatility': [random.gauss(0.1, 0.02) for _ in range(100)],
                'volume_trend': [random.gauss(1.0, 0.1) for _ in range(100)]
            }
            drift_system.record_feature_data(features)
        
        # Drifted distribution (shifted mean)
        for batch in range(5):
            features = {
                'price_volatility': [random.gauss(0.15, 0.02) for _ in range(100)],  # Mean shifted
                'volume_trend': [random.gauss(1.2, 0.15) for _ in range(100)]  # Mean + std changed
            }
            drift_system.record_feature_data(features)
        
        print("🚨 Running drift detection...")
        alerts = drift_system.run_drift_detection()
        
        print(f"   Found {len(alerts)} drift alerts")
        
        for alert in alerts:
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}[alert.severity]
            print(f"   {severity_emoji} {alert.drift_type}: {alert.component}")
            print(f"      {alert.description}")
            print(f"      Action: {alert.recommended_action}")
        
        print("\n📈 System status:")
        status = drift_system.get_system_status()
        for key, value in status.items():
            if key != 'components_monitored' and key != 'alert_breakdown':
                print(f"   {key}: {value}")
        
        print("\n📊 Components monitored:")
        for key, value in status['components_monitored'].items():
            print(f"   {key}: {value}")
        
        print("\n🚨 Alert breakdown:")
        for severity, count in status['alert_breakdown'].items():
            print(f"   {severity}: {count}")
        
        print("\n✅ DRIFT DETECTION TEST COMPLETED")
        return len(alerts) > 0
    
    # Run test
    import asyncio
    success = asyncio.run(test_drift_detection())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")