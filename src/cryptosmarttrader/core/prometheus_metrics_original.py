#!/usr/bin/env python3
"""
Prometheus Metrics System
Comprehensive metrics collection for latency, error-ratio, completeness, GPU usage
"""

import time
import asyncio
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..core.structured_logger import get_logger

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum,
        start_http_server, generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class MetricValue:
    """Individual metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class MetricsCollector:
    """Base metrics collector"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"MetricsCollector_{name}")
        self.enabled = True

    def collect(self) -> Dict[str, MetricValue]:
        """Collect metrics - override in subclasses"""
        return {}

    def enable(self) -> None:
        """Enable metrics collection"""
        self.enabled = True

    def disable(self) -> None:
        """Disable metrics collection"""
        self.enabled = False

class LatencyMetricsCollector(MetricsCollector):
    """Latency metrics collector"""

    def __init__(self):
        super().__init__("Latency")
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_operations: Dict[str, float] = {}

    def start_operation(self, operation_name: str, operation_id: str = None) -> str:
        """Start timing an operation"""
        if not self.enabled:
            return ""

        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        self.active_operations[operation_id] = time.time()
        return operation_id

    def end_operation(self, operation_id: str, operation_name: str = None) -> float:
        """End timing an operation"""
        if not self.enabled or operation_id not in self.active_operations:
            return 0.0

        start_time = self.active_operations.pop(operation_id)
        duration = time.time() - start_time

        if operation_name:
            self.operation_times[operation_name].append(duration)

        return duration

    def record_duration(self, operation_name: str, duration: float) -> None:
        """Record operation duration directly"""
        if self.enabled:
            self.operation_times[operation_name].append(duration)

    def collect(self) -> Dict[str, MetricValue]:
        """Collect latency metrics"""
        metrics = {}

        for operation, durations in self.operation_times.items():
            if durations:
                avg_latency = sum(durations) / len(durations)
                max_latency = max(durations)
                min_latency = min(durations)

                # P95 latency
                sorted_durations = sorted(durations)
                p95_index = int(len(sorted_durations) * 0.95)
                p95_latency = sorted_durations[p95_index] if sorted_durations else 0.0

                metrics.update({
                    f"{operation}_latency_avg": MetricValue(
                        avg_latency, datetime.now(), {"operation": operation}
                    ),
                    f"{operation}_latency_max": MetricValue(
                        max_latency, datetime.now(), {"operation": operation}
                    ),
                    f"{operation}_latency_min": MetricValue(
                        min_latency, datetime.now(), {"operation": operation}
                    ),
                    f"{operation}_latency_p95": MetricValue(
                        p95_latency, datetime.now(), {"operation": operation}
                    )
                })

        return metrics

class ErrorRatioMetricsCollector(MetricsCollector):
    """Error ratio metrics collector"""

    def __init__(self, window_size: int = 1000):
        super().__init__("ErrorRatio")
        self.window_size = window_size
        self.operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def record_success(self, operation_name: str) -> None:
        """Record successful operation"""
        if self.enabled:
            self.operations[operation_name].append(True)

    def record_error(self, operation_name: str) -> None:
        """Record failed operation"""
        if self.enabled:
            self.operations[operation_name].append(False)

    def collect(self) -> Dict[str, MetricValue]:
        """Collect error ratio metrics"""
        metrics = {}

        for operation, results in self.operations.items():
            if results:
                total_ops = len(results)
                error_count = sum(1 for r in results if not r)
                success_count = total_ops - error_count

                error_ratio = error_count / total_ops
                success_ratio = success_count / total_ops

                metrics.update({
                    f"{operation}_error_ratio": MetricValue(
                        error_ratio, datetime.now(), {"operation": operation}
                    ),
                    f"{operation}_success_ratio": MetricValue(
                        success_ratio, datetime.now(), {"operation": operation}
                    ),
                    f"{operation}_total_operations": MetricValue(
                        total_ops, datetime.now(), {"operation": operation}
                    )
                })

        return metrics

class CompletenessMetricsCollector(MetricsCollector):
    """Data completeness metrics collector"""

    def __init__(self):
        super().__init__("Completeness")
        self.completeness_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def record_completeness(self, data_source: str, completeness_percentage: float) -> None:
        """Record data completeness percentage"""
        if self.enabled:
            self.completeness_data[data_source].append(completeness_percentage)

    def collect(self) -> Dict[str, MetricValue]:
        """Collect completeness metrics"""
        metrics = {}

        for source, percentages in self.completeness_data.items():
            if percentages:
                avg_completeness = sum(percentages) / len(percentages)
                min_completeness = min(percentages)

                metrics.update({
                    f"{source}_completeness_avg": MetricValue(
                        avg_completeness, datetime.now(), {"source": source}
                    ),
                    f"{source}_completeness_min": MetricValue(
                        min_completeness, datetime.now(), {"source": source}
                    )
                })

        return metrics

class SystemMetricsCollector(MetricsCollector):
    """System resource metrics collector"""

    def __init__(self):
        super().__init__("System")
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            return True
        except ImportError:
            return False

    def collect(self) -> Dict[str, MetricValue]:
        """Collect system metrics"""
        metrics = {}
        now = datetime.now()

        if PSUTIL_AVAILABLE:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics.update({
                "cpu_usage_percent": MetricValue(
                    cpu_percent, now, {"type": "cpu"}
                ),
                "memory_usage_percent": MetricValue(
                    memory.percent, now, {"type": "memory"}
                ),
                "memory_usage_mb": MetricValue(
                    memory.used / 1024 / 1024, now, {"type": "memory"}
                ),
                "disk_usage_percent": MetricValue(
                    disk.percent, now, {"type": "disk"}
                ),
                "disk_free_gb": MetricValue(
                    disk.free / 1024 / 1024 / 1024, now, {"type": "disk"}
                )
            })

        # GPU metrics (if available)
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()

                for i, gpu in enumerate(gpus):
                    gpu_labels = {"gpu_id": str(i), "gpu_name": gpu.name}

                    metrics.update({
                        f"gpu_{i}_usage_percent": MetricValue(
                            gpu.load * 100, now, gpu_labels
                        ),
                        f"gpu_{i}_memory_usage_percent": MetricValue(
                            gpu.memoryUtil * 100, now, gpu_labels
                        ),
                        f"gpu_{i}_memory_used_mb": MetricValue(
                            gpu.memoryUsed, now, gpu_labels
                        ),
                        f"gpu_{i}_temperature_c": MetricValue(
                            gpu.temperature, now, gpu_labels
                        )
                    })
            except Exception as e:
                self.logger.error(f"GPU metrics collection failed: {e}")

        return metrics

class PrometheusMetricsServer:
    """Prometheus metrics server with HTTP endpoint"""

    def __init__(self, port: int = 8090):
        self.port = port
        self.logger = get_logger("PrometheusMetricsServer")
        self.collectors: List[MetricsCollector] = []
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # Create custom registry to avoid conflicts
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self.prom_metrics = self._create_prometheus_metrics()
        else:
            self.registry = None
            self.prom_metrics = {}
            self.logger.warning("Prometheus client not available, using mock metrics")

    def _create_prometheus_metrics(self) -> Dict[str, Any]:
        """Create Prometheus metric objects"""

        metrics = {}

        # Latency metrics
        metrics['operation_duration'] = Histogram(
            'cryptosmarttrader_operation_duration_seconds',
            'Time spent on operations',
            ['operation_name', 'agent'],
            registry=self.registry
        )

        # Error metrics
        metrics['operation_errors'] = Counter(
            'cryptosmarttrader_operation_errors_total',
            'Total operation errors',
            ['operation_name', 'error_type', 'agent'],
            registry=self.registry
        )

        metrics['operation_success'] = Counter(
            'cryptosmarttrader_operation_success_total',
            'Total successful operations',
            ['operation_name', 'agent'],
            registry=self.registry
        )

        # Completeness metrics
        metrics['data_completeness'] = Gauge(
            'cryptosmarttrader_data_completeness_ratio',
            'Data completeness ratio',
            ['data_source', 'agent'],
            registry=self.registry
        )

        # System metrics
        metrics['cpu_usage'] = Gauge(
            'cryptosmarttrader_cpu_usage_percent',
            'CPU usage percentage',
            ['agent'],
            registry=self.registry
        )

        metrics['memory_usage'] = Gauge(
            'cryptosmarttrader_memory_usage_bytes',
            'Memory usage in bytes',
            ['agent'],
            registry=self.registry
        )

        metrics['gpu_usage'] = Gauge(
            'cryptosmarttrader_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id', 'gpu_name', 'agent'],
            registry=self.registry
        )

        metrics['gpu_memory'] = Gauge(
            'cryptosmarttrader_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'gpu_name', 'agent'],
            registry=self.registry
        )

        # Queue metrics
        metrics['queue_size'] = Gauge(
            'cryptosmarttrader_queue_size',
            'Queue size',
            ['queue_name', 'agent'],
            registry=self.registry
        )

        metrics['messages_processed'] = Counter(
            'cryptosmarttrader_messages_processed_total',
            'Total messages processed',
            ['queue_name', 'message_type', 'agent'],
            registry=self.registry
        )

        # ML metrics
        metrics['model_inference_time'] = Histogram(
            'cryptosmarttrader_model_inference_seconds',
            'Model inference time',
            ['model_name', 'horizon', 'agent'],
            registry=self.registry
        )

        metrics['prediction_confidence'] = Histogram(
            'cryptosmarttrader_prediction_confidence',
            'Prediction confidence scores',
            ['model_name', 'coin', 'agent'],
            registry=self.registry
        )

        return metrics

    def add_collector(self, collector: MetricsCollector) -> None:
        """Add metrics collector"""
        self.collectors.append(collector)
        self.logger.info(f"Added metrics collector: {collector.name}")

    def start_server(self) -> bool:
        """Start Prometheus HTTP server"""

        if self.running:
            self.logger.warning("Metrics server already running")
            return True

        try:
            if PROMETHEUS_AVAILABLE:
                start_http_server(self.port, registry=self.registry)
                self.logger.info(f"Prometheus metrics server started on port {self.port}")
            else:
                self.logger.info(f"Mock metrics server started on port {self.port}")

            self.running = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            return False

    def stop_server(self) -> None:
        """Stop metrics server"""
        self.running = False
        self.logger.info("Metrics server stopped")

    def update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics from collectors"""

        if not PROMETHEUS_AVAILABLE:
            return

        try:
            for collector in self.collectors:
                metrics = collector.collect()

                for metric_name, metric_value in metrics.items():
                    # Update appropriate Prometheus metric
                    self._update_prometheus_metric(metric_name, metric_value)

        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metrics: {e}")

    def _update_prometheus_metric(self, metric_name: str, metric_value: MetricValue) -> None:
        """Update specific Prometheus metric"""

        try:
            # Map metric names to Prometheus metrics
            if 'latency' in metric_name or 'duration' in metric_name:
                self.prom_metrics['operation_duration'].labels(
                    operation_name=metric_value.labels.get('operation', 'unknown'),
                    agent=metric_value.labels.get('agent', 'system')
                ).observe(metric_value.value)

            elif 'completeness' in metric_name:
                self.prom_metrics['data_completeness'].labels(
                    data_source=metric_value.labels.get('source', 'unknown'),
                    agent=metric_value.labels.get('agent', 'system')
                ).set(metric_value.value)

            elif 'cpu_usage' in metric_name:
                self.prom_metrics['cpu_usage'].labels(
                    agent=metric_value.labels.get('agent', 'system')
                ).set(metric_value.value)

            elif 'memory_usage' in metric_name:
                self.prom_metrics['memory_usage'].labels(
                    agent=metric_value.labels.get('agent', 'system')
                ).set(metric_value.value * 1024 * 1024)  # Convert MB to bytes

            elif 'gpu' in metric_name and 'usage' in metric_name:
                self.prom_metrics['gpu_usage'].labels(
                    gpu_id=metric_value.labels.get('gpu_id', '0'),
                    gpu_name=metric_value.labels.get('gpu_name', 'unknown'),
                    agent=metric_value.labels.get('agent', 'system')
                ).set(metric_value.value)

            elif 'gpu' in metric_name and 'memory' in metric_name:
                self.prom_metrics['gpu_memory'].labels(
                    gpu_id=metric_value.labels.get('gpu_id', '0'),
                    gpu_name=metric_value.labels.get('gpu_name', 'unknown'),
                    agent=metric_value.labels.get('agent', 'system')
                ).set(metric_value.value * 1024 * 1024)  # Convert MB to bytes

        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metric {metric_name}: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""

        summary = {
            'collectors_active': len(self.collectors),
            'server_running': self.running,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'metrics_by_collector': {}
        }

        for collector in self.collectors:
            metrics = collector.collect()
            summary['metrics_by_collector'][collector.name] = {
                'metric_count': len(metrics),
                'latest_metrics': {k: v.value for k, v in list(metrics.items())[:5]}
            }

        return summary

async def metrics_collection_loop(metrics_server: PrometheusMetricsServer,
                                interval: float = 10.0) -> None:
    """Background metrics collection loop"""

    logger = get_logger("MetricsCollectionLoop")
    logger.info(f"Starting metrics collection loop (interval: {interval}s)")

    while metrics_server.running:
        try:
            metrics_server.update_prometheus_metrics()
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Metrics collection loop error: {e}")
            await asyncio.sleep(interval)

    logger.info("Metrics collection loop stopped")

if __name__ == "__main__":
    async def test_prometheus_metrics():
        """Test Prometheus metrics system"""

        print("🔍 TESTING PROMETHEUS METRICS SYSTEM")
        print("=" * 60)

        # Create metrics server
        metrics_server = PrometheusMetricsServer(port=8091)

        # Create collectors
        latency_collector = LatencyMetricsCollector()
        error_collector = ErrorRatioMetricsCollector()
        completeness_collector = CompletenessMetricsCollector()
        system_collector = SystemMetricsCollector()

        # Add collectors
        metrics_server.add_collector(latency_collector)
        metrics_server.add_collector(error_collector)
        metrics_server.add_collector(completeness_collector)
        metrics_server.add_collector(system_collector)

        print("🚀 Starting metrics server...")
        success = metrics_server.start_server()
        print(f"   Server started: {'✅' if success else '❌'}")

        print("\n📊 Generating test metrics...")

        # Generate test data
        for i in range(10):
            # Latency metrics
            op_id = latency_collector.start_operation("ml_prediction")
            await asyncio.sleep(0.01)  # REMOVED: Mock data pattern not allowed in production
            latency_collector.end_operation(op_id, "ml_prediction")

            # Error metrics
            if i % 4 == 0:
                error_collector.record_error("data_collection")
            else:
                error_collector.record_success("data_collection")

            # Completeness metrics
            completeness_collector.record_completeness("kraken_api", 95.5 + i)

        print("   Test metrics generated")

        # Update Prometheus metrics
        metrics_server.update_prometheus_metrics()

        print("\n📈 Metrics summary:")
        summary = metrics_server.get_metrics_summary()
        for key, value in summary.items():
            if key != 'metrics_by_collector':
                print(f"   {key}: {value}")

        print("\n📋 Collector metrics:")
        for collector_name, data in summary['metrics_by_collector'].items():
            print(f"   {collector_name}: {data['metric_count']} metrics")
            for metric_name, value in data['latest_metrics'].items():
                print(f"     {metric_name}: {value:.3f}")

        print(f"\n🌐 Metrics endpoint: http://localhost:8091/metrics")
        print("   (Access with curl or Grafana)")

        print("\n⏱️  Running collection loop for 15 seconds...")

        # Run collection loop
        collection_task = asyncio.create_task(
            metrics_collection_loop(metrics_server, interval=2.0)
        )

        await asyncio.sleep(15.0)

        # Stop server
        metrics_server.stop_server()
        collection_task.cancel()

        try:
            await collection_task
        except asyncio.CancelledError:
            pass

        print("\n✅ PROMETHEUS METRICS TEST COMPLETED")

    # Run test
    asyncio.run(test_prometheus_metrics())
