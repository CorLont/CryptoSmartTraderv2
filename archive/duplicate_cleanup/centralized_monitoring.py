"""
Centralized Monitoring and Alerting System
- Consolidates all logging, metrics, and health monitoring
- Prometheus/Grafana-style metrics endpoint
- Real-time dashboards and alerts
- Performance optimization tracking
"""

import asyncio
import aiohttp
from aiohttp import web
import json
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import psutil
import statistics

from utils.daily_logger import get_daily_logger

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str = ""

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    duration_seconds: int
    severity: str  # "info", "warning", "critical"
    enabled: bool = True

@dataclass
class Alert:
    """Active alert"""
    rule_name: str
    message: str
    severity: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class MetricsCollector:
    """Collects and stores metrics from all system components"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=retention_hours * 360))  # 10-second intervals
        self.logger = get_daily_logger().get_logger('performance_metrics')
        self.lock = threading.RLock()
        
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a metric point"""
        if labels is None:
            labels = {}
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            unit=unit
        )
        
        with self.lock:
            metric_key = f"{name}_{hash(json.dumps(labels, sort_keys=True))}"
            self.metrics[metric_key].append(metric_point)
    
    def get_metric_values(self, name: str, labels: Optional[Dict[str, str]] = None, 
                         start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metric values for specified time range"""
        if labels is None:
            labels = {}
        metric_key = f"{name}_{hash(json.dumps(labels, sort_keys=True))}"
        
        with self.lock:
            if metric_key not in self.metrics:
                return []
            
            points = list(self.metrics[metric_key])
            
            # Filter by time range
            if start_time or end_time:
                filtered_points = []
                for point in points:
                    if start_time and point.timestamp < start_time:
                        continue
                    if end_time and point.timestamp > end_time:
                        continue
                    filtered_points.append(point)
                return filtered_points
            
            return points
    
    def get_latest_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get the latest value for a metric"""
        values = self.get_metric_values(name, labels)
        return values[-1].value if values else None
    
    def calculate_statistics(self, name: str, labels: Optional[Dict[str, str]] = None, 
                           minutes: int = 60) -> Dict[str, float]:
        """Calculate statistics for a metric over time period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        values = self.get_metric_values(name, labels, start_time, end_time)
        if not values:
            return {}
        
        numeric_values = [point.value for point in values]
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values), 
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'stdev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            'latest': numeric_values[-1]
        }

class AlertManager:
    """Manages alert rules and active alerts"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.logger = get_daily_logger().get_logger('security_events')
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default system alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_percent",
                condition="gt",
                threshold=85.0,
                duration_seconds=300,
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage", 
                metric_name="system_memory_percent",
                condition="gt",
                threshold=90.0,
                duration_seconds=180,
                severity="critical"
            ),
            AlertRule(
                name="agent_failure",
                metric_name="agent_error_count",
                condition="gt", 
                threshold=5.0,
                duration_seconds=60,
                severity="critical"
            ),
            AlertRule(
                name="low_trading_opportunities",
                metric_name="trading_opportunities_per_hour",
                condition="lt",
                threshold=10.0,
                duration_seconds=3600,
                severity="warning"
            ),
            AlertRule(
                name="ml_prediction_accuracy_drop",
                metric_name="ml_prediction_accuracy",
                condition="lt",
                threshold=0.6,
                duration_seconds=1800,
                severity="warning"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def add_rule(self, rule: AlertRule):
        """Add or update alert rule"""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def check_alerts(self):
        """Check all alert rules against current metrics"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get recent metric values
                end_time = current_time
                start_time = end_time - timedelta(seconds=rule.duration_seconds)
                
                metric_values = self.metrics_collector.get_metric_values(
                    rule.metric_name, start_time=start_time, end_time=end_time
                )
                
                if not metric_values:
                    continue
                
                # Check condition
                latest_value = metric_values[-1].value
                condition_met = self._evaluate_condition(latest_value, rule.condition, rule.threshold)
                
                # Check if condition has been met for duration
                if condition_met:
                    condition_duration = self._check_condition_duration(metric_values, rule)
                    
                    if condition_duration >= rule.duration_seconds:
                        self._trigger_alert(rule, latest_value, current_time)
                else:
                    # Resolve alert if active
                    if rule_name in self.active_alerts:
                        self._resolve_alert(rule_name, current_time)
                        
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        return False
    
    def _check_condition_duration(self, metric_values: List[MetricPoint], rule: AlertRule) -> int:
        """Check how long condition has been true"""
        if not metric_values:
            return 0
        
        duration = 0
        for point in reversed(metric_values):
            if self._evaluate_condition(point.value, rule.condition, rule.threshold):
                if duration == 0:
                    duration = int((datetime.now() - point.timestamp).total_seconds())
                continue
            else:
                break
        
        return duration
    
    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Trigger an alert"""
        if rule.name in self.active_alerts:
            return  # Alert already active
        
        alert = Alert(
            rule_name=rule.name,
            message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {value:.2f})",
            severity=rule.severity,
            triggered_at=timestamp,
            metadata={'current_value': value, 'threshold': rule.threshold}
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        # Keep only recent history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
    
    def _resolve_alert(self, rule_name: str, timestamp: datetime):
        """Resolve an active alert"""
        if rule_name not in self.active_alerts:
            return
        
        alert = self.active_alerts[rule_name]
        alert.resolved_at = timestamp
        
        del self.active_alerts[rule_name]
        
        self.logger.info(f"ALERT RESOLVED: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at > cutoff_time]

class PerformanceTracker:
    """Track system and agent performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = get_daily_logger().get_logger('performance_metrics')
        
        # Performance baselines
        self.baselines = {
            'sentiment_analysis_time': 5.0,  # seconds
            'technical_analysis_time': 2.0,
            'ml_prediction_time': 10.0,
            'whale_analysis_time': 3.0,
            'trading_opportunities_per_hour': 50,
            'system_throughput_ops_per_sec': 100
        }
    
    def track_operation(self, operation_name: str, duration: float, success: bool = True, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Track an operation's performance"""
        
        # Record duration
        self.metrics_collector.record_metric(
            f"{operation_name}_duration_seconds",
            duration,
            labels={'success': str(success)},
            unit="seconds"
        )
        
        # Record success/failure
        self.metrics_collector.record_metric(
            f"{operation_name}_success_rate",
            1.0 if success else 0.0,
            unit="ratio"
        )
        
        # Check against baseline
        baseline = self.baselines.get(f"{operation_name}_time")
        if baseline and duration > baseline * 2:  # 2x slower than baseline
            self.logger.warning(f"Performance degradation: {operation_name} took {duration:.2f}s (baseline: {baseline:.2f}s)")
        
        # Track metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    self.metrics_collector.record_metric(
                        f"{operation_name}_{key}",
                        float(value),
                        labels={'operation': operation_name}
                    )
    
    def track_throughput(self, component: str, operations_count: int, time_window_seconds: int = 60):
        """Track throughput for a component"""
        ops_per_second = operations_count / time_window_seconds
        
        self.metrics_collector.record_metric(
            f"{component}_throughput_ops_per_sec",
            ops_per_second,
            labels={'component': component},
            unit="ops/sec"
        )
    
    def track_resource_usage(self, component: str, cpu_percent: float, memory_mb: float, 
                           gpu_percent: float = 0.0):
        """Track resource usage for a component"""
        
        self.metrics_collector.record_metric(
            f"{component}_cpu_percent",
            cpu_percent,
            labels={'component': component},
            unit="percent"
        )
        
        self.metrics_collector.record_metric(
            f"{component}_memory_mb", 
            memory_mb,
            labels={'component': component},
            unit="MB"
        )
        
        if gpu_percent > 0:
            self.metrics_collector.record_metric(
                f"{component}_gpu_percent",
                gpu_percent,
                labels={'component': component},
                unit="percent"
            )
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        summary = {}
        
        # Get statistics for key metrics
        key_metrics = [
            'sentiment_analysis_duration_seconds',
            'technical_analysis_duration_seconds', 
            'ml_prediction_duration_seconds',
            'whale_analysis_duration_seconds',
            'trading_opportunities_per_hour',
            'system_cpu_percent',
            'system_memory_percent'
        ]
        
        for metric in key_metrics:
            stats = self.metrics_collector.calculate_statistics(metric, minutes=hours*60)
            if stats:
                summary[metric] = stats
        
        return summary

class MonitoringServer:
    """HTTP server for monitoring dashboard and API"""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager, 
                 performance_tracker: PerformanceTracker, port: int = 8001):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.performance_tracker = performance_tracker
        self.port = port
        self.app = web.Application()
        self.logger = get_daily_logger().get_logger('api_calls')
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/metrics', self.metrics_endpoint)
        self.app.router.add_get('/health', self.health_endpoint)
        self.app.router.add_get('/alerts', self.alerts_endpoint)
        self.app.router.add_get('/performance', self.performance_endpoint)
        self.app.router.add_get('/dashboard', self.dashboard_endpoint)
        
        # Add CORS headers
        self.app.middlewares.append(self._cors_middleware)
    
    async def _cors_middleware(self, request, handler):
        """Add CORS headers"""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    async def metrics_endpoint(self, request):
        """Prometheus-style metrics endpoint"""
        try:
            # Get query parameters
            metric_name = request.query.get('metric')
            hours = int(request.query.get('hours', 1))
            
            if metric_name:
                # Return specific metric
                stats = self.metrics_collector.calculate_statistics(metric_name, minutes=hours*60)
                return web.json_response(stats)
            else:
                # Return all recent metrics
                all_metrics = {}
                for metric_key in self.metrics_collector.metrics.keys():
                    metric_name = metric_key.split('_')[0]
                    if metric_name not in all_metrics:
                        latest_value = self.metrics_collector.get_latest_value(metric_name)
                        if latest_value is not None:
                            all_metrics[metric_name] = latest_value
                
                return web.json_response(all_metrics)
        except Exception as e:
            self.logger.error(f"Metrics endpoint error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def health_endpoint(self, request):
        """System health endpoint"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'uptime': time.time()
                },
                'alerts': {
                    'active_count': len(self.alert_manager.get_active_alerts()),
                    'total_count': len(self.alert_manager.get_alert_history())
                }
            }
            
            # Determine overall health status
            if cpu_percent > 90 or memory.percent > 95:
                health_data['status'] = 'unhealthy'
            elif cpu_percent > 80 or memory.percent > 85:
                health_data['status'] = 'degraded'
            
            return web.json_response(health_data)
        except Exception as e:
            self.logger.error(f"Health endpoint error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def alerts_endpoint(self, request):
        """Alerts endpoint"""
        try:
            hours = int(request.query.get('hours', 24))
            
            active_alerts = self.alert_manager.get_active_alerts()
            alert_history = self.alert_manager.get_alert_history(hours)
            
            alerts_data = {
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'alert_history': [asdict(alert) for alert in alert_history],
                'summary': {
                    'active_count': len(active_alerts),
                    'history_count': len(alert_history),
                    'critical_count': len([a for a in active_alerts if a.severity == 'critical']),
                    'warning_count': len([a for a in active_alerts if a.severity == 'warning'])
                }
            }
            
            return web.json_response(alerts_data)
        except Exception as e:
            self.logger.error(f"Alerts endpoint error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def performance_endpoint(self, request):
        """Performance metrics endpoint"""
        try:
            hours = int(request.query.get('hours', 1))
            
            performance_data = self.performance_tracker.get_performance_summary(hours)
            
            return web.json_response({
                'time_period_hours': hours,
                'performance_metrics': performance_data,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Performance endpoint error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def dashboard_endpoint(self, request):
        """Simple dashboard HTML"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CryptoSmartTrader Monitoring</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-box { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
                .critical { background-color: #ffebee; }
                .warning { background-color: #fff3e0; }
                .healthy { background-color: #e8f5e8; }
            </style>
        </head>
        <body>
            <h1>CryptoSmartTrader Real-time Monitoring</h1>
            <div id="content">Loading...</div>
            
            <script>
                async function loadData() {
                    try {
                        const [health, alerts, metrics] = await Promise.all([
                            fetch('/health').then(r => r.json()),
                            fetch('/alerts').then(r => r.json()),
                            fetch('/metrics').then(r => r.json())
                        ]);
                        
                        document.getElementById('content').innerHTML = `
                            <div class="metric-box ${health.status}">
                                <h2>System Health: ${health.status.toUpperCase()}</h2>
                                <p>CPU: ${health.system.cpu_percent.toFixed(1)}%</p>
                                <p>Memory: ${health.system.memory_percent.toFixed(1)}%</p>
                                <p>Disk: ${health.system.disk_percent.toFixed(1)}%</p>
                            </div>
                            
                            <div class="metric-box">
                                <h2>Active Alerts: ${alerts.summary.active_count}</h2>
                                <p>Critical: ${alerts.summary.critical_count}</p>
                                <p>Warning: ${alerts.summary.warning_count}</p>
                            </div>
                            
                            <div class="metric-box">
                                <h2>Key Metrics</h2>
                                ${Object.entries(metrics).map(([name, value]) => 
                                    `<p>${name}: ${typeof value === 'number' ? value.toFixed(2) : value}</p>`
                                ).join('')}
                            </div>
                        `;
                    } catch (error) {
                        document.getElementById('content').innerHTML = 
                            `<div class="metric-box critical">Error loading data: ${error.message}</div>`;
                    }
                }
                
                loadData();
                setInterval(loadData, 30000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def start_server(self):
        """Start the monitoring server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        self.logger.info(f"Monitoring server started on port {self.port}")

class CentralizedMonitoring:
    """Main centralized monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.monitoring_server = MonitoringServer(
            self.metrics_collector, 
            self.alert_manager, 
            self.performance_tracker
        )
        
        self.logger = get_daily_logger().get_logger('performance_metrics')
        self.running = False
        
        # Auto-collection tasks
        self.system_collection_task = None
        self.alert_check_task = None
    
    async def start(self):
        """Start the centralized monitoring system"""
        self.running = True
        
        # Start monitoring server
        await self.monitoring_server.start_server()
        
        # Start background tasks
        self.system_collection_task = asyncio.create_task(self._system_metrics_loop())
        self.alert_check_task = asyncio.create_task(self._alert_check_loop())
        
        self.logger.info("Centralized monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system"""
        self.running = False
        
        if self.system_collection_task:
            self.system_collection_task.cancel()
        if self.alert_check_task:
            self.alert_check_task.cancel()
        
        self.logger.info("Centralized monitoring system stopped")
    
    async def _system_metrics_loop(self):
        """Collect system metrics periodically"""
        while self.running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Record metrics
                self.metrics_collector.record_metric('system_cpu_percent', cpu_percent, unit='percent')
                self.metrics_collector.record_metric('system_memory_percent', memory.percent, unit='percent')
                self.metrics_collector.record_metric('system_disk_percent', disk.percent, unit='percent')
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_check_loop(self):
        """Check alerts periodically"""
        while self.running:
            try:
                self.alert_manager.check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert check error: {e}")
                await asyncio.sleep(60)
    
    def record_agent_metric(self, agent_name: str, metric_name: str, value: float, unit: str = ""):
        """Record a metric from an agent"""
        self.metrics_collector.record_metric(
            f"agent_{metric_name}",
            value,
            labels={'agent': agent_name},
            unit=unit
        )
    
    def track_operation_performance(self, operation: str, duration: float, success: bool = True):
        """Track operation performance"""
        self.performance_tracker.track_operation(operation, duration, success)

# Global instance
centralized_monitoring = CentralizedMonitoring()