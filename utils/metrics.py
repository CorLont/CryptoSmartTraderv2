# utils/metrics.py
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from typing import Dict, Any, Optional
import threading
import time
import logging
from config.settings import config


logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("crypto_requests_total", "Total number of API requests", ["endpoint", "method"])
HEALTH_GAUGE = Gauge("system_health_score", "Current system health score")
AGENT_PERFORMANCE = Gauge("agent_performance_score", "Agent performance score", ["agent_name"])
DATA_FRESHNESS = Gauge("data_freshness_seconds", "Age of data in seconds", ["data_type"])
PREDICTION_ACCURACY = Gauge("prediction_accuracy", "ML prediction accuracy", ["model_name", "horizon"])
EXCHANGE_LATENCY = Histogram("exchange_latency_seconds", "Exchange API latency", ["exchange"])
CACHE_HITS = Counter("cache_hits_total", "Cache hit count", ["cache_type"])
CACHE_MISSES = Counter("cache_misses_total", "Cache miss count", ["cache_type"])
ERROR_COUNT = Counter("errors_total", "Total error count", ["error_type", "component"])


class MetricsServer:
    """
    Prometheus metrics server for monitoring system performance.
    Implements Dutch requirements for metrics infrastructure.
    """
    
    def __init__(self, config_manager=None, health_monitor=None):
        self.config_manager = config_manager
        self.health_monitor = health_monitor
        self.server_started = False
        self._lock = threading.Lock()
        
        # Start metrics server
        if config.enable_prometheus:
            self.start_metrics_server()
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.metrics_thread.start()
    
    def start_metrics_server(self, port: Optional[int] = None) -> bool:
        """Start Prometheus metrics HTTP server"""
        try:
            with self._lock:
                if not self.server_started:
                    server_port = port or config.metrics_port
                    start_http_server(server_port)
                    self.server_started = True
                    logger.info(f"Metrics server started on port {server_port}")
                    
                    # Set up health gauge callback
                    HEALTH_GAUGE.set_function(self._get_health_score)
                    
                    return True
                else:
                    logger.warning("Metrics server already started")
                    return False
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def _get_health_score(self) -> float:
        """Get current system health score"""
        if self.health_monitor:
            try:
                health_data = self.health_monitor.get_system_health()
                return health_data.get("overall_grade_numeric", 0.0)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return 0.0
        return 0.5  # Default health score
    
    def _collect_metrics_loop(self):
        """Background thread to collect and update metrics"""
        while True:
            try:
                self._update_agent_metrics()
                self._update_data_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_agent_metrics(self):
        """Update agent performance metrics"""
        if self.health_monitor:
            try:
                agent_status = self.health_monitor.get_agent_status()
                for agent_name, status in agent_status.items():
                    performance_score = status.get("performance_score", 0.0)
                    AGENT_PERFORMANCE.labels(agent_name=agent_name).set(performance_score)
            except Exception as e:
                logger.error(f"Error updating agent metrics: {e}")
    
    def _update_data_metrics(self):
        """Update data freshness metrics"""
        try:
            current_time = time.time()
            # This would be updated by actual data sources
            # For now, set placeholder values
            DATA_FRESHNESS.labels(data_type="market_data").set(current_time % 300)
            DATA_FRESHNESS.labels(data_type="sentiment_data").set(current_time % 600)
        except Exception as e:
            logger.error(f"Error updating data metrics: {e}")
    
    # Metric recording methods
    @staticmethod
    def record_request(endpoint: str, method: str = "GET"):
        """Record API request"""
        REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
    
    @staticmethod
    def record_cache_hit(cache_type: str):
        """Record cache hit"""
        CACHE_HITS.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_cache_miss(cache_type: str):
        """Record cache miss"""
        CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_error(error_type: str, component: str):
        """Record error occurrence"""
        ERROR_COUNT.labels(error_type=error_type, component=component).inc()
    
    @staticmethod
    def record_exchange_latency(exchange: str, latency: float):
        """Record exchange API latency"""
        EXCHANGE_LATENCY.labels(exchange=exchange).observe(latency)
    
    @staticmethod
    def set_prediction_accuracy(model_name: str, horizon: str, accuracy: float):
        """Set ML model prediction accuracy"""
        PREDICTION_ACCURACY.labels(model_name=model_name, horizon=horizon).set(accuracy)


# Global metrics instance
metrics_server = MetricsServer()