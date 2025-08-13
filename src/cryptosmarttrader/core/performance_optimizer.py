"""
CryptoSmartTrader V2 - Performance Optimizer
Real-time performance monitoring with automatic optimization
"""

import logging
import time
import threading
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import json
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    cache_hit_rate: float
    api_response_times: Dict[str, float]
    ml_inference_time: float
    data_processing_time: float
    error_rate: float
    throughput: float


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""

    category: str  # memory, cpu, cache, api, ml
    priority: str  # high, medium, low
    description: str
    implementation: str
    expected_improvement: float
    estimated_effort: str


class PerformanceMonitor:
    """Real-time system performance monitoring"""

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000  # Keep last 1000 measurements

        # Performance baselines
        self.baselines = {
            "cpu_normal": 20.0,  # Normal CPU usage %
            "memory_normal": 50.0,  # Normal memory usage %
            "api_response_normal": 2.0,  # Normal API response time (seconds)
            "ml_inference_normal": 1.0,  # Normal ML inference time (seconds)
            "cache_hit_normal": 0.8,  # Normal cache hit rate
            "error_rate_normal": 0.01,  # Normal error rate
        }

        # Current performance state
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.performance_alerts: List[Dict] = []

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Performance monitoring already active")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)
                self.current_metrics = metrics

                # Trim history if needed
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history :]

                # Check for performance issues
                self._check_performance_alerts(metrics)

                # Cache metrics for dashboard
                self._cache_metrics(metrics)

                # Wait before next collection
                time.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io_counters = psutil.disk_io_counters()
            disk_io = disk_io_counters._asdict() if disk_io_counters else {}
            network_io_counters = psutil.net_io_counters()
            network_io = network_io_counters._asdict() if network_io_counters else {}

            # Application-specific metrics
            cache_hit_rate = self._calculate_cache_hit_rate()
            api_response_times = self._get_api_response_times()
            ml_inference_time = self._get_ml_inference_time()
            data_processing_time = self._get_data_processing_time()
            error_rate = self._calculate_error_rate()
            throughput = self._calculate_throughput()

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_io=disk_io,
                network_io=network_io,
                cache_hit_rate=cache_hit_rate,
                api_response_times=api_response_times,
                ml_inference_time=ml_inference_time,
                data_processing_time=data_processing_time,
                error_rate=error_rate,
                throughput=throughput,
            )

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return self._get_default_metrics()

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            if (
                cache_stats
                and "total_accesses" in cache_stats
                and cache_stats["total_accesses"] > 0
            ):
                hits = cache_stats.get("cache_hits", 0)
                total = cache_stats["total_accesses"]
                return hits / total
            return 0.8  # Default assumption
        except Exception:
            return 0.8

    def _get_api_response_times(self) -> Dict[str, float]:
        """Get API response times"""
        try:
            # Get cached response times from data manager
            api_times = self.cache_manager.get("api_response_times")
            return api_times or {"kraken": 1.5, "average": 1.5}
        except Exception:
            return {"kraken": 1.5, "average": 1.5}

    def _get_ml_inference_time(self) -> float:
        """Get ML inference time"""
        try:
            inference_times = self.cache_manager.get("ml_inference_times")
            if inference_times:
                return float(np.mean(list(inference_times.values())))
            return 1.0  # Default
        except Exception:
            return 1.0

    def _get_data_processing_time(self) -> float:
        """Get data processing time"""
        try:
            processing_times = self.cache_manager.get("data_processing_times")
            if processing_times:
                return float(np.mean(list(processing_times.values())))
            return 0.5  # Default
        except Exception:
            return 0.5

    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        try:
            # Get error statistics from error handler
            error_handler = getattr(self.container, "error_handler", None)
            if error_handler and hasattr(error_handler, "get_error_statistics"):
                error_stats = error_handler().get_error_statistics()
                recent_errors = sum(error_stats.get("errors_by_hour", {}).values())
                total_operations = error_stats.get("total_operations", 1000)
                return recent_errors / total_operations
            return 0.01  # Default low error rate
        except Exception:
            return 0.01

    def _calculate_throughput(self) -> float:
        """Calculate system throughput (operations per second)"""
        try:
            # Estimate based on cache operations and API calls
            if len(self.metrics_history) >= 2:
                current = self.metrics_history[-1]
                previous = self.metrics_history[-2]
                time_diff = (current.timestamp - previous.timestamp).total_seconds()

                # Rough estimate based on system activity
                estimated_ops = (current.cpu_usage + current.memory_usage) / 10
                return estimated_ops / time_diff if time_diff > 0 else 0

            return 10.0  # Default throughput
        except Exception:
            return 10.0

    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default metrics when collection fails"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_io={},
            network_io={},
            cache_hit_rate=0.0,
            api_response_times={},
            ml_inference_time=0.0,
            data_processing_time=0.0,
            error_rate=0.0,
            throughput=0.0,
        )

    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and generate alerts"""
        try:
            alerts = []

            # CPU usage alert
            if metrics.cpu_usage > 80:
                alerts.append(
                    {
                        "type": "high_cpu",
                        "severity": "high" if metrics.cpu_usage > 90 else "medium",
                        "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                        "timestamp": metrics.timestamp,
                        "value": metrics.cpu_usage,
                    }
                )

            # Memory usage alert
            if metrics.memory_usage > 85:
                alerts.append(
                    {
                        "type": "high_memory",
                        "severity": "high" if metrics.memory_usage > 95 else "medium",
                        "message": f"High memory usage: {metrics.memory_usage:.1f}%",
                        "timestamp": metrics.timestamp,
                        "value": metrics.memory_usage,
                    }
                )

            # Cache hit rate alert
            if metrics.cache_hit_rate < 0.5:
                alerts.append(
                    {
                        "type": "low_cache_hit",
                        "severity": "medium",
                        "message": f"Low cache hit rate: {metrics.cache_hit_rate:.1%}",
                        "timestamp": metrics.timestamp,
                        "value": metrics.cache_hit_rate,
                    }
                )

            # API response time alert
            avg_api_time = (
                np.mean(list(metrics.api_response_times.values()))
                if metrics.api_response_times
                else 0
            )
            if avg_api_time > 5.0:
                alerts.append(
                    {
                        "type": "slow_api",
                        "severity": "medium",
                        "message": f"Slow API responses: {avg_api_time:.1f}s average",
                        "timestamp": metrics.timestamp,
                        "value": avg_api_time,
                    }
                )

            # ML inference time alert
            if metrics.ml_inference_time > 3.0:
                alerts.append(
                    {
                        "type": "slow_ml",
                        "severity": "medium",
                        "message": f"Slow ML inference: {metrics.ml_inference_time:.1f}s",
                        "timestamp": metrics.timestamp,
                        "value": metrics.ml_inference_time,
                    }
                )

            # Error rate alert
            if metrics.error_rate > 0.05:
                alerts.append(
                    {
                        "type": "high_errors",
                        "severity": "high",
                        "message": f"High error rate: {metrics.error_rate:.1%}",
                        "timestamp": metrics.timestamp,
                        "value": metrics.error_rate,
                    }
                )

            # Store alerts
            self.performance_alerts.extend(alerts)

            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_alerts = [
                alert for alert in self.performance_alerts if alert["timestamp"] > cutoff_time
            ]

            # Log high severity alerts
            for alert in alerts:
                if alert["severity"] == "high":
                    self.logger.warning(f"Performance alert: {alert['message']}")

        except Exception as e:
            self.logger.error(f"Performance alert check failed: {e}")

    def _cache_metrics(self, metrics: PerformanceMetrics):
        """Cache metrics for dashboard access"""
        try:
            # Store current metrics
            self.cache_manager.set("current_performance_metrics", asdict(metrics), ttl_minutes=5)

            # Store recent history for charts
            recent_history = self.metrics_history[-100:]  # Last 100 measurements
            self.cache_manager.set(
                "performance_history", [asdict(m) for m in recent_history], ttl_minutes=30
            )

            # Store performance alerts
            self.cache_manager.set("performance_alerts", self.performance_alerts, ttl_minutes=60)

        except Exception as e:
            self.logger.error(f"Metrics caching failed: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.current_metrics:
                return {"status": "monitoring_not_active"}

            metrics = self.current_metrics

            # Calculate performance scores
            cpu_score = max(0, 100 - metrics.cpu_usage)
            memory_score = max(0, 100 - metrics.memory_usage)
            cache_score = metrics.cache_hit_rate * 100
            api_score = (
                max(0, 100 - (np.mean(list(metrics.api_response_times.values())) * 20))
                if metrics.api_response_times
                else 50
            )
            error_score = max(0, 100 - (metrics.error_rate * 1000))

            overall_score = float(
                np.mean([cpu_score, memory_score, cache_score, api_score, error_score])

            return {
                "timestamp": metrics.timestamp.isoformat(),
                "overall_score": overall_score,
                "scores": {
                    "cpu": cpu_score,
                    "memory": memory_score,
                    "cache": cache_score,
                    "api": api_score,
                    "errors": error_score,
                },
                "metrics": asdict(metrics),
                "status": self._get_performance_status(overall_score),
                "active_alerts": len(
                    [a for a in self.performance_alerts if a["severity"] == "high"]
                ),
                "recommendations_count": 3,  # Placeholder since method is in optimizer class
            }

        except Exception as e:
            self.logger.error(f"Performance summary failed: {e}")
            return {"status": "error", "error": str(e)}

    def _get_performance_status(self, score: float) -> str:
        """Get performance status based on score"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"


class PerformanceOptimizer:
    """Automatic performance optimization"""

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        self.monitor = PerformanceMonitor(container)

        # Optimization state
        self.optimizations_applied: List[Dict] = []
        self.optimization_history: List[Dict] = []

    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get optimization recommendations based on current performance"""
        try:
            recommendations = []

            if not self.monitor.current_metrics:
                return recommendations

            metrics = self.monitor.current_metrics

            # Memory optimization
            if metrics.memory_usage > 80:
                recommendations.append(
                    OptimizationRecommendation(
                        category="memory",
                        priority="high",
                        description="High memory usage detected",
                        implementation="Increase cache cleanup frequency, reduce cache size limits",
                        expected_improvement=15.0,
                        estimated_effort="Low",
                    )

            # CPU optimization
            if metrics.cpu_usage > 75:
                recommendations.append(
                    OptimizationRecommendation(
                        category="cpu",
                        priority="high",
                        description="High CPU usage detected",
                        implementation="Optimize ML batch sizes, reduce analysis frequency",
                        expected_improvement=20.0,
                        estimated_effort="Medium",
                    )

            # Cache optimization
            if metrics.cache_hit_rate < 0.6:
                recommendations.append(
                    OptimizationRecommendation(
                        category="cache",
                        priority="medium",
                        description="Low cache hit rate",
                        implementation="Increase cache TTL for stable data, pre-cache popular queries",
                        expected_improvement=25.0,
                        estimated_effort="Low",
                    )

            # API optimization
            avg_api_time = (
                np.mean(list(metrics.api_response_times.values()))
                if metrics.api_response_times
                else 0
            )
            if avg_api_time > 3.0:
                recommendations.append(
                    OptimizationRecommendation(
                        category="api",
                        priority="medium",
                        description="Slow API response times",
                        implementation="Implement request batching, add retry logic with backoff",
                        expected_improvement=30.0,
                        estimated_effort="Medium",
                    )

            # ML optimization
            if metrics.ml_inference_time > 2.0:
                recommendations.append(
                    OptimizationRecommendation(
                        category="ml",
                        priority="medium",
                        description="Slow ML inference",
                        implementation="Batch inference, model quantization, feature selection",
                        expected_improvement=40.0,
                        estimated_effort="High",
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"Optimization recommendations failed: {e}")
            return []

    def apply_automatic_optimizations(self) -> Dict[str, Any]:
        """Apply automatic performance optimizations"""
        try:
            optimizations_applied = []

            if not self.monitor.current_metrics:
                return {"status": "no_metrics", "optimizations": []}

            metrics = self.monitor.current_metrics

            # Memory optimization
            if metrics.memory_usage > 85:
                success = self._optimize_memory()
                if success:
                    optimizations_applied.append(
                        {
                            "type": "memory_cleanup",
                            "description": "Aggressive cache cleanup and memory optimization",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # Cache optimization
            if metrics.cache_hit_rate < 0.5:
                success = self._optimize_cache()
                if success:
                    optimizations_applied.append(
                        {
                            "type": "cache_optimization",
                            "description": "Cache configuration tuning and cleanup",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # API rate limiting
            avg_api_time = (
                np.mean(list(metrics.api_response_times.values()))
                if metrics.api_response_times
                else 0
            )
            if avg_api_time > 4.0:
                success = self._optimize_api_calls()
                if success:
                    optimizations_applied.append(
                        {
                            "type": "api_optimization",
                            "description": "API call rate limiting and batching",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # Store optimization history
            self.optimizations_applied.extend(optimizations_applied)
            self.optimization_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "optimizations": optimizations_applied,
                    "metrics_before": asdict(metrics),
                }
            )

            return {
                "status": "success",
                "optimizations": optimizations_applied,
                "recommendations": [asdict(r) for r in self.get_optimization_recommendations()],
            }

        except Exception as e:
            self.logger.error(f"Automatic optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    def _optimize_memory(self) -> bool:
        """Optimize memory usage"""
        try:
            # Force cache cleanup
            if hasattr(self.cache_manager, "_cleanup_expired"):
                self.cache_manager._cleanup_expired()

            # Reduce cache size limits temporarily
            if hasattr(self.cache_manager, "max_memory_mb"):
                original_limit = self.cache_manager.max_memory_mb
                self.cache_manager.max_memory_mb = original_limit * 0.8

                # Force memory limit enforcement
                if hasattr(self.cache_manager, "_enforce_memory_limit"):
                    self.cache_manager._enforce_memory_limit()

                self.logger.info(
                    f"Reduced cache memory limit from {original_limit}MB to {self.cache_manager.max_memory_mb}MB"
                )

            return True

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return False

    def _optimize_cache(self) -> bool:
        """Optimize cache configuration"""
        try:
            # Clear old cache entries
            if hasattr(self.cache_manager, "_cleanup_expired"):
                self.cache_manager._cleanup_expired()

            # Update cache configuration for better hit rates
            cache_config = {
                "default_ttl_minutes": 30,  # Increase default TTL
                "aggressive_cleanup": True,
                "prefetch_enabled": True,
            }

            self.cache_manager.set("cache_optimization_config", cache_config, ttl_minutes=120)

            self.logger.info("Applied cache optimization configuration")
            return True

        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return False

    def _optimize_api_calls(self) -> bool:
        """Optimize API call patterns"""
        try:
            # Implement more aggressive rate limiting
            api_config = {
                "rate_limit_enabled": True,
                "max_requests_per_minute": 30,  # Reduce from default
                "batch_size": 5,  # Increase batch size
                "retry_backoff_multiplier": 2.0,
            }

            self.cache_manager.set("api_optimization_config", api_config, ttl_minutes=60)

            self.logger.info("Applied API optimization configuration")
            return True

        except Exception as e:
            self.logger.error(f"API optimization failed: {e}")
            return False

    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history"""
        try:
            return self.optimization_history[-50:]  # Last 50 optimizations
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return []

    def reset_optimizations(self) -> bool:
        """Reset all applied optimizations"""
        try:
            # Reset cache limits
            if hasattr(self.cache_manager, "max_memory_mb"):
                self.cache_manager.max_memory_mb = 1000  # Default limit

            # Clear optimization configs
            self.cache_manager.delete("cache_optimization_config")
            self.cache_manager.delete("api_optimization_config")

            # Clear history
            self.optimizations_applied.clear()

            self.logger.info("Reset all performance optimizations")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reset optimizations: {e}")
            return False
