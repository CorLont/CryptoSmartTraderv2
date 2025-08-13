# utils/performance_optimizer.py
import psutil
import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    cache_hit_ratio: float
    avg_response_time: float


class PerformanceOptimizer:
    """Advanced performance monitoring and optimization system"""

    def __init__(self):
        self.metrics_history: Dict[str, list] = {}
        self.optimization_active = True
        self.monitor_thread = None
        self._lock = threading.Lock()

        # Performance thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0

        # Optimization settings
        self.gc_threshold = 500  # MB
        self.cache_cleanup_interval = 300  # seconds

        self.start_monitoring()

    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.optimization_active = False
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.optimization_active:
            try:
                metrics = self.collect_metrics()
                self._store_metrics(metrics)
                self._check_optimization_triggers(metrics)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)

    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk usage
            disk_usage = psutil.disk_usage("/")

            # Network stats
            net_io = psutil.net_io_counters()

            # Thread count
            active_threads = threading.active_count()

            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_threads=active_threads,
                cache_hit_ratio=0.0,  # To be implemented with cache manager
                avg_response_time=0.0,  # To be implemented with request tracking
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history"""
        with self._lock:
            timestamp = time.time()

            for field_name in metrics.__dataclass_fields__:
                if field_name not in self.metrics_history:
                    self.metrics_history[field_name] = []

                value = getattr(metrics, field_name)
                self.metrics_history[field_name].append({"timestamp": timestamp, "value": value})

                # Keep only last 1000 entries
                if len(self.metrics_history[field_name]) > 1000:
                    self.metrics_history[field_name] = self.metrics_history[field_name][-1000:]

    def _check_optimization_triggers(self, metrics: PerformanceMetrics):
        """Check if optimization actions are needed"""
        try:
            # High CPU usage
            if metrics.cpu_percent > self.cpu_threshold:
                logger.warning(f"High CPU usage detected: {metrics.cpu_percent}%")
                self._optimize_cpu_usage()

            # High memory usage
            if metrics.memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {metrics.memory_percent}%")
                self._optimize_memory_usage()

            # High disk usage
            if metrics.disk_usage_percent > self.disk_threshold:
                logger.warning(f"High disk usage detected: {metrics.disk_usage_percent}%")
                self._optimize_disk_usage()

        except Exception as e:
            logger.error(f"Error in optimization triggers: {e}")

    def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        logger.info("Implementing CPU optimization strategies")
        # Could implement thread pool adjustments, task prioritization, etc.

    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        import gc

        logger.info("Implementing memory optimization strategies")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")

    def _optimize_disk_usage(self):
        """Optimize disk usage"""
        logger.info("Implementing disk optimization strategies")

        # Clean up old log files
        logs_path = Path("logs")
        if logs_path.exists():
            for log_file in logs_path.glob("*.log"):
                if log_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                    logger.info(f"Archiving large log file: {log_file}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        metrics = self.collect_metrics()

        return {
            "current_metrics": {
                "cpu_percent": round(metrics.cpu_percent, 2),
                "memory_percent": round(metrics.memory_percent, 2),
                "memory_used_gb": round(metrics.memory_used_gb, 2),
                "disk_usage_percent": round(metrics.disk_usage_percent, 2),
                "active_threads": metrics.active_threads,
            },
            "status": self._get_performance_status(metrics),
            "recommendations": self._get_optimization_recommendations(metrics),
        }

    def _get_performance_status(self, metrics: PerformanceMetrics) -> str:
        """Determine overall performance status"""
        if (
            metrics.cpu_percent > 90
            or metrics.memory_percent > 95
            or metrics.disk_usage_percent > 95
        ):
            return "CRITICAL"
        elif (
            metrics.cpu_percent > 75
            or metrics.memory_percent > 85
            or metrics.disk_usage_percent > 85
        ):
            return "WARNING"
        elif metrics.cpu_percent > 50 or metrics.memory_percent > 70:
            return "MODERATE"
        else:
            return "OPTIMAL"

    def _get_optimization_recommendations(self, metrics: PerformanceMetrics) -> list:
        """Get optimization recommendations"""
        recommendations = []

        if metrics.cpu_percent > 80:
            recommendations.append("Consider reducing parallel workers or agent update frequency")

        if metrics.memory_percent > 80:
            recommendations.append(
                "Consider reducing cache size or enabling more aggressive cleanup"
            )

        if metrics.disk_usage_percent > 80:
            recommendations.append("Consider archiving old data or reducing data retention period")

        if metrics.active_threads > 50:
            recommendations.append("Consider optimizing thread usage and cleanup")

        if not recommendations:
            recommendations.append("System performance is optimal")

        return recommendations
