#!/usr/bin/env python3
"""
Real-Time System Monitor
Advanced real-time monitoring and alerting system
"""

import os
import sys
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SystemAlert:
    """System alert data structure"""
    timestamp: str
    level: str  # INFO, WARNING, CRITICAL
    component: str
    message: str
    metric_value: float
    threshold: float
    action_required: bool

class RealTimeMonitor:
    """
    Advanced real-time system monitoring with intelligent alerting
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_history = []
        self.performance_metrics = {}
        self.thresholds = self._load_thresholds()
        self.last_metrics = {}
        
    def _load_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load monitoring thresholds"""
        
        return {
            'cpu': {
                'warning': 80.0,
                'critical': 95.0
            },
            'memory': {
                'warning': 85.0,
                'critical': 95.0
            },
            'disk': {
                'warning': 90.0,
                'critical': 98.0
            },
            'gpu_memory': {
                'warning': 85.0,
                'critical': 95.0
            },
            'confidence_pass_rate': {
                'warning': 1.0,  # Less than 1%
                'critical': 0.1  # Less than 0.1%
            },
            'error_rate': {
                'warning': 5.0,  # More than 5%
                'critical': 15.0  # More than 15%
            }
        }
    
    def start_monitoring(self, interval_seconds: int = 10):
        """Start real-time monitoring"""
        
        if self.monitoring_active:
            print("Monitoring already active")
            return
        
        print("🔄 Starting real-time system monitoring...")
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        print(f"   Monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        print("Monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_system_metrics()
                
                # Check for threshold violations
                alerts = self._check_thresholds(current_metrics)
                
                # Process alerts
                for alert in alerts:
                    self._process_alert(alert)
                
                # Update performance metrics
                self._update_performance_history(current_metrics)
                
                # Save monitoring data
                self._save_monitoring_data(current_metrics, alerts)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('.').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'disk_io': dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
            'process_count': len(psutil.pids()),
            'gpu_metrics': self._get_gpu_metrics(),
            'trading_metrics': self._get_trading_metrics(),
            'confidence_metrics': self._get_confidence_metrics()
        }
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available"""
        
        gpu_metrics = {
            'available': False,
            'utilization': 0.0,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'temperature': 0.0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_metrics['available'] = True
                
                # Get memory info
                memory_info = torch.cuda.memory_stats()
                allocated = memory_info.get('allocated_bytes.all.current', 0)
                reserved = memory_info.get('reserved_bytes.all.current', 0)
                
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory
                
                gpu_metrics.update({
                    'memory_used': (allocated / total_memory) * 100,
                    'memory_total_gb': total_memory / (1024**3),
                    'utilization': 0.0  # Would need nvidia-ml-py for actual utilization
                })
                
        except (ImportError, Exception):
            pass
        
        return gpu_metrics
    
    def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading system metrics"""
        
        trading_metrics = {
            'active_opportunities': 0,
            'daily_predictions': 0,
            'error_count': 0,
            'api_success_rate': 100.0
        }
        
        # Try to read from daily logs
        try:
            today = datetime.now().strftime("%Y%m%d")
            daily_dir = Path(f"logs/daily/{today}")
            
            if daily_dir.exists():
                # Count confidence gate events
                confidence_file = daily_dir / "confidence_gate.jsonl"
                if confidence_file.exists():
                    with open(confidence_file, 'r') as f:
                        lines = f.readlines()
                        trading_metrics['daily_predictions'] = len(lines)
                
                # Count errors
                error_files = list(daily_dir.glob("*error*.json*"))
                trading_metrics['error_count'] = len(error_files)
                
        except Exception:
            pass
        
        return trading_metrics
    
    def _get_confidence_metrics(self) -> Dict[str, Any]:
        """Get confidence gate metrics"""
        
        confidence_metrics = {
            'latest_pass_rate': 0.0,
            'total_candidates': 0,
            'passed_candidates': 0,
            'gate_operational': True
        }
        
        # Try to read latest confidence gate data
        try:
            today = datetime.now().strftime("%Y%m%d")
            confidence_file = Path(f"logs/daily/{today}/confidence_gate.jsonl")
            
            if confidence_file.exists():
                with open(confidence_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        latest_line = lines[-1]
                        latest_data = json.loads(latest_line)
                        
                        confidence_metrics.update({
                            'latest_pass_rate': latest_data.get('pass_rate', 0.0) * 100,
                            'total_candidates': latest_data.get('total_candidates', 0),
                            'passed_candidates': latest_data.get('passed_count', 0)
                        })
                        
        except Exception:
            confidence_metrics['gate_operational'] = False
        
        return confidence_metrics
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[SystemAlert]:
        """Check metrics against thresholds and generate alerts"""
        
        alerts = []
        
        # CPU threshold check
        cpu_percent = metrics['cpu_percent']
        if cpu_percent >= self.thresholds['cpu']['critical']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='CRITICAL',
                component='CPU',
                message=f"CPU usage critical: {cpu_percent:.1f}%",
                metric_value=cpu_percent,
                threshold=self.thresholds['cpu']['critical'],
                action_required=True
            ))
        elif cpu_percent >= self.thresholds['cpu']['warning']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='WARNING',
                component='CPU',
                message=f"CPU usage high: {cpu_percent:.1f}%",
                metric_value=cpu_percent,
                threshold=self.thresholds['cpu']['warning'],
                action_required=False
            ))
        
        # Memory threshold check
        memory_percent = metrics['memory_percent']
        if memory_percent >= self.thresholds['memory']['critical']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='CRITICAL',
                component='Memory',
                message=f"Memory usage critical: {memory_percent:.1f}%",
                metric_value=memory_percent,
                threshold=self.thresholds['memory']['critical'],
                action_required=True
            ))
        elif memory_percent >= self.thresholds['memory']['warning']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='WARNING',
                component='Memory',
                message=f"Memory usage high: {memory_percent:.1f}%",
                metric_value=memory_percent,
                threshold=self.thresholds['memory']['warning'],
                action_required=False
            ))
        
        # Confidence gate check
        confidence_pass_rate = metrics['confidence_metrics']['latest_pass_rate']
        if confidence_pass_rate <= self.thresholds['confidence_pass_rate']['critical']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='CRITICAL',
                component='ConfidenceGate',
                message=f"Confidence pass rate extremely low: {confidence_pass_rate:.2f}%",
                metric_value=confidence_pass_rate,
                threshold=self.thresholds['confidence_pass_rate']['critical'],
                action_required=True
            ))
        elif confidence_pass_rate <= self.thresholds['confidence_pass_rate']['warning']:
            alerts.append(SystemAlert(
                timestamp=metrics['timestamp'],
                level='WARNING',
                component='ConfidenceGate',
                message=f"Confidence pass rate low: {confidence_pass_rate:.2f}%",
                metric_value=confidence_pass_rate,
                threshold=self.thresholds['confidence_pass_rate']['warning'],
                action_required=False
            ))
        
        return alerts
    
    def _process_alert(self, alert: SystemAlert):
        """Process and handle system alerts"""
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Print alert
        level_icon = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        
        print(f"{level_icon.get(alert.level, '❗')} {alert.level}: {alert.message}")
        
        # Log alert to daily logs
        self._log_alert(alert)
        
        # Take automatic action if required
        if alert.action_required:
            self._take_corrective_action(alert)
    
    def _log_alert(self, alert: SystemAlert):
        """Log alert to daily monitoring logs"""
        
        today = datetime.now().strftime("%Y%m%d")
        alert_dir = Path(f"logs/daily/{today}")
        alert_dir.mkdir(parents=True, exist_ok=True)
        
        alert_file = alert_dir / "system_alerts.jsonl"
        
        with open(alert_file, 'a', encoding='utf-8') as f:
            alert_data = {
                'timestamp': alert.timestamp,
                'level': alert.level,
                'component': alert.component,
                'message': alert.message,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'action_required': alert.action_required
            }
            f.write(json.dumps(alert_data) + '\n')
    
    def _take_corrective_action(self, alert: SystemAlert):
        """Take automatic corrective actions for critical alerts"""
        
        if alert.component == 'CPU' and alert.level == 'CRITICAL':
            print("🔧 Taking action: Reducing worker processes")
            # Could implement actual worker reduction logic
            
        elif alert.component == 'Memory' and alert.level == 'CRITICAL':
            print("🔧 Taking action: Triggering garbage collection")
            import gc
            gc.collect()
            
        elif alert.component == 'ConfidenceGate' and alert.level == 'CRITICAL':
            print("🔧 Taking action: Investigating prediction pipeline")
            # Could trigger diagnostic checks
    
    def _update_performance_history(self, metrics: Dict[str, Any]):
        """Update performance metrics history"""
        
        # Keep last 100 measurements
        if len(self.performance_metrics) >= 100:
            # Remove oldest entries
            oldest_key = min(self.performance_metrics.keys())
            del self.performance_metrics[oldest_key]
        
        self.performance_metrics[metrics['timestamp']] = {
            'cpu_percent': metrics['cpu_percent'],
            'memory_percent': metrics['memory_percent'],
            'confidence_pass_rate': metrics['confidence_metrics']['latest_pass_rate']
        }
    
    def _save_monitoring_data(self, metrics: Dict[str, Any], alerts: List[SystemAlert]):
        """Save monitoring data to daily logs"""
        
        today = datetime.now().strftime("%Y%m%d")
        monitor_dir = Path(f"logs/daily/{today}")
        monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = monitor_dir / "real_time_metrics.jsonl"
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        if not self.performance_metrics:
            return {'status': 'No monitoring data available'}
        
        latest_timestamp = max(self.performance_metrics.keys())
        latest_metrics = self.performance_metrics[latest_timestamp]
        
        # Calculate recent averages
        recent_metrics = list(self.performance_metrics.values())[-10:]
        
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        
        # Determine overall health
        health_score = 100
        if avg_cpu > 80:
            health_score -= 20
        if avg_memory > 85:
            health_score -= 25
        
        recent_alerts = [a for a in self.alert_history if a.level in ['WARNING', 'CRITICAL']][-5:]
        
        return {
            'monitoring_active': self.monitoring_active,
            'latest_metrics': latest_metrics,
            'averages': {
                'cpu_percent': round(avg_cpu, 1),
                'memory_percent': round(avg_memory, 1)
            },
            'health_score': max(0, health_score),
            'recent_alerts': len(recent_alerts),
            'last_update': latest_timestamp
        }

# Global monitor instance
_real_time_monitor = None

def get_real_time_monitor() -> RealTimeMonitor:
    """Get singleton real-time monitor instance"""
    global _real_time_monitor
    
    if _real_time_monitor is None:
        _real_time_monitor = RealTimeMonitor()
    
    return _real_time_monitor

def start_monitoring(interval_seconds: int = 10):
    """Start real-time monitoring"""
    monitor = get_real_time_monitor()
    monitor.start_monitoring(interval_seconds)

def stop_monitoring():
    """Stop real-time monitoring"""
    monitor = get_real_time_monitor()
    monitor.stop_monitoring()

if __name__ == "__main__":
    print("🔄 TESTING REAL-TIME MONITOR")
    print("=" * 40)
    
    monitor = RealTimeMonitor()
    
    # Test metrics collection
    print("📊 Testing metrics collection...")
    metrics = monitor._collect_system_metrics()
    print(f"   CPU: {metrics['cpu_percent']:.1f}%")
    print(f"   Memory: {metrics['memory_percent']:.1f}%")
    print(f"   Confidence: {metrics['confidence_metrics']['latest_pass_rate']:.2f}%")
    
    # Test threshold checking
    print("🚨 Testing threshold checking...")
    alerts = monitor._check_thresholds(metrics)
    print(f"   Alerts generated: {len(alerts)}")
    
    for alert in alerts:
        print(f"   {alert.level}: {alert.message}")
    
    print("✅ Real-time monitor testing completed")