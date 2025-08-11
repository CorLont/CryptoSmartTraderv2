#!/usr/bin/env python3
"""
Enterprise-Grade Logging & Monitoring System
Structured JSON logging with correlation IDs, metrics, and alerting
"""

import logging
import json
import time
import uuid
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import contextmanager
from pythonjsonlogger import jsonlogger
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, 
    push_to_gateway, start_http_server
)
import sys
import os
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class CorrelationContext:
    """Correlation context for request/operation tracking"""
    correlation_id: str
    operation: str
    agent_name: Optional[str] = None
    exchange: Optional[str] = None
    start_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    """Prometheus metrics collector for application monitoring"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Request/Operation metrics
        self.operation_duration = Histogram(
            'operation_duration_seconds',
            'Time spent on operations',
            ['operation', 'agent', 'status'],
            registry=self.registry
        )
        
        self.operation_total = Counter(
            'operations_total',
            'Total number of operations',
            ['operation', 'agent', 'status'],
            registry=self.registry
        )
        
        # Data collection metrics
        self.data_collection_completeness = Gauge(
            'data_collection_completeness_ratio',
            'Completeness ratio of data collection',
            ['exchange', 'data_type'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['exchange', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['exchange', 'endpoint'],
            registry=self.registry
        )
        
        # System resource metrics
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        # Error tracking
        self.error_total = Counter(
            'errors_total',
            'Total number of errors',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['component'],
            registry=self.registry
        )
        
        # ML model metrics
        self.model_prediction_accuracy = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy',
            ['model_type', 'timeframe'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'model_training_duration_seconds',
            'Model training duration',
            ['model_type'],
            registry=self.registry
        )
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        self.system_memory_usage.set(memory.used)
        self.system_cpu_usage.set(cpu_percent)

class StructuredLogger:
    """Enterprise structured logger with correlation tracking"""
    
    _instance = None
    _lock = threading.Lock()
    _correlation_context: threading.local = threading.local()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        if hasattr(self, '_initialized'):
            return
            
        self.config = config or {}
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics collector
        self.metrics = MetricsCollector()
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate_5min': 0.1,  # 10% error rate
            'memory_usage_percent': 90,
            'cpu_usage_percent': 95,
            'api_latency_p95': 5.0,  # 5 seconds
            'data_completeness': 0.8  # 80% completeness
        }
        
        # Alert state tracking
        self.alert_states: Dict[str, Dict] = {}
        
        self._setup_loggers()
        self._start_metrics_server()
        self._initialized = True
    
    def _setup_loggers(self):
        """Setup structured JSON loggers"""
        
        # Create custom JSON formatter
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(correlation_id)s %(operation)s %(agent_name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Main application logger
        self.app_logger = logging.getLogger('CryptoSmartTrader')
        self.app_logger.setLevel(logging.INFO)
        
        # File handler for JSON logs
        json_handler = logging.FileHandler(
            self.log_dir / 'application.jsonl',
            encoding='utf-8'
        )
        json_handler.setFormatter(json_formatter)
        json_handler.setLevel(logging.INFO)
        
        # Error-specific handler
        error_handler = logging.FileHandler(
            self.log_dir / 'errors.jsonl',
            encoding='utf-8'
        )
        error_handler.setFormatter(json_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.app_logger.addHandler(json_handler)
        self.app_logger.addHandler(error_handler)
        self.app_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.app_logger.propagate = False
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            metrics_port = self.config.get('metrics_port', 8090)
            start_http_server(metrics_port, registry=self.metrics.registry)
            # Use print for startup to avoid correlation_id issues
            print(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            print(f"Failed to start metrics server: {e}")
    
    @contextmanager
    def correlation_context(self, operation: str, agent_name: str = None, **kwargs):
        """Context manager for correlation tracking"""
        correlation_id = str(uuid.uuid4())
        context = CorrelationContext(
            correlation_id=correlation_id,
            operation=operation,
            agent_name=agent_name,
            **kwargs
        )
        
        # Store in thread-local storage
        old_context = getattr(self._correlation_context, 'current', None)
        self._correlation_context.current = context
        
        # Start operation timer
        start_time = time.time()
        
        try:
            self.info(f"Operation started: {operation}", extra=context.to_dict())
            yield context
            
            # Record successful operation
            duration = time.time() - start_time
            self.metrics.operation_duration.labels(
                operation=operation,
                agent=agent_name or 'unknown',
                status='success'
            ).observe(duration)
            
            self.metrics.operation_total.labels(
                operation=operation,
                agent=agent_name or 'unknown',
                status='success'
            ).inc()
            
            self.info(
                f"Operation completed: {operation} (duration: {duration:.3f}s)",
                extra={**context.to_dict(), 'duration': duration}
            )
            
        except Exception as e:
            # Record failed operation
            duration = time.time() - start_time
            self.metrics.operation_duration.labels(
                operation=operation,
                agent=agent_name or 'unknown',
                status='error'
            ).observe(duration)
            
            self.metrics.operation_total.labels(
                operation=operation,
                agent=agent_name or 'unknown',
                status='error'
            ).inc()
            
            self.error(
                f"Operation failed: {operation} - {str(e)}",
                extra={**context.to_dict(), 'duration': duration, 'error': str(e)},
                exc_info=True
            )
            
            # Track error metrics
            self.metrics.error_total.labels(
                component=agent_name or 'unknown',
                error_type=type(e).__name__,
                severity='error'
            ).inc()
            
            raise
        finally:
            # Restore previous context
            self._correlation_context.current = old_context
    
    def _get_log_context(self) -> Dict[str, Any]:
        """Get current correlation context for logging"""
        context = getattr(self._correlation_context, 'current', None)
        if context:
            return context.to_dict()
        
        return {
            'correlation_id': 'no-context',
            'operation': 'unknown',
            'agent_name': None
        }
    
    def _log_with_context(self, level: str, message: str, extra: Dict[str, Any] = None):
        """Log message with correlation context"""
        log_context = self._get_log_context()
        if extra:
            log_context.update(extra)
        
        # Update system metrics periodically
        if hasattr(self, '_last_metrics_update'):
            if time.time() - self._last_metrics_update > 30:  # Every 30 seconds
                self.metrics.update_system_metrics()
                self._last_metrics_update = time.time()
        else:
            self._last_metrics_update = time.time()
            self.metrics.update_system_metrics()
        
        # Check for alerts
        self._check_alert_conditions(level, log_context)
        
        getattr(self.app_logger, level.lower())(message, extra=log_context)
    
    def _check_alert_conditions(self, level: str, context: Dict[str, Any]):
        """Check if alert conditions are met"""
        current_time = time.time()
        
        # Error rate alerting
        if level.upper() in ['ERROR', 'CRITICAL']:
            alert_key = 'error_rate_5min'
            if alert_key not in self.alert_states:
                self.alert_states[alert_key] = {'count': 0, 'window_start': current_time}
            
            alert_state = self.alert_states[alert_key]
            
            # Reset window if 5 minutes passed
            if current_time - alert_state['window_start'] > 300:
                alert_state['count'] = 0
                alert_state['window_start'] = current_time
            
            alert_state['count'] += 1
            
            # Check if error rate exceeds threshold
            total_ops = sum([
                self.metrics.operation_total.labels(op, agent, status)._value.get()
                for op in ['data_collection', 'prediction', 'analysis']
                for agent in ['data_collector', 'ml_predictor', 'technical_analyzer']
                for status in ['success', 'error']
            ])
            
            if total_ops > 10:  # Minimum operations for meaningful rate
                error_rate = alert_state['count'] / total_ops
                if error_rate > self.alert_thresholds['error_rate_5min']:
                    self._trigger_alert(
                        'HIGH_ERROR_RATE',
                        f"Error rate {error_rate:.2%} exceeds threshold {self.alert_thresholds['error_rate_5min']:.2%}",
                        AlertLevel.HIGH,
                        context
                    )
        
        # Memory usage alerting
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        if memory_percent > self.alert_thresholds['memory_usage_percent']:
            self._trigger_alert(
                'HIGH_MEMORY_USAGE',
                f"Memory usage {memory_percent:.1f}% exceeds threshold {self.alert_thresholds['memory_usage_percent']}%",
                AlertLevel.MEDIUM,
                {'memory_percent': memory_percent, **context}
            )
    
    def _trigger_alert(self, alert_type: str, message: str, level: AlertLevel, context: Dict[str, Any]):
        """Trigger an alert with rate limiting"""
        current_time = time.time()
        alert_key = f"{alert_type}_{level.value}"
        
        # Rate limit alerts (max 1 per 5 minutes per type)
        if alert_key in self.alert_states:
            last_alert = self.alert_states[alert_key].get('last_triggered', 0)
            if current_time - last_alert < 300:  # 5 minutes
                return
        
        if alert_key not in self.alert_states:
            self.alert_states[alert_key] = {}
        
        self.alert_states[alert_key]['last_triggered'] = current_time
        
        alert_context = {
            'alert_type': alert_type,
            'alert_level': level.value,
            'alert_time': datetime.now().isoformat(),
            **context
        }
        
        self.app_logger.critical(
            f"ALERT [{level.value}] {alert_type}: {message}",
            extra=alert_context
        )
        
        # Could integrate with external alerting systems here
        # e.g., Slack, PagerDuty, email, etc.
    
    def info(self, message: str, extra: Dict[str, Any] = None):
        """Log info message with context"""
        self._log_with_context('INFO', message, extra)
    
    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Log warning message with context"""
        self._log_with_context('WARNING', message, extra)
    
    def error(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log error message with context"""
        if exc_info:
            self.app_logger.error(message, extra={**self._get_log_context(), **(extra or {})}, exc_info=True)
        else:
            self._log_with_context('ERROR', message, extra)
    
    def critical(self, message: str, extra: Dict[str, Any] = None):
        """Log critical message with context"""
        self._log_with_context('CRITICAL', message, extra)
    
    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Log debug message with context"""
        self._log_with_context('DEBUG', message, extra)
    
    def log_api_request(self, exchange: str, endpoint: str, duration: float, status: str):
        """Log API request with metrics"""
        self.metrics.api_request_duration.labels(
            exchange=exchange,
            endpoint=endpoint
        ).observe(duration)
        
        self.metrics.api_requests_total.labels(
            exchange=exchange,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.info(
            f"API request to {exchange}/{endpoint} completed",
            extra={
                'exchange': exchange,
                'endpoint': endpoint,
                'duration': duration,
                'status': status
            }
        )
    
    def log_data_completeness(self, exchange: str, data_type: str, completeness: float):
        """Log data collection completeness"""
        self.metrics.data_collection_completeness.labels(
            exchange=exchange,
            data_type=data_type
        ).set(completeness)
        
        if completeness < self.alert_thresholds['data_completeness']:
            self._trigger_alert(
                'LOW_DATA_COMPLETENESS',
                f"Data completeness {completeness:.1%} below threshold for {exchange}/{data_type}",
                AlertLevel.MEDIUM,
                {'exchange': exchange, 'data_type': data_type, 'completeness': completeness}
            )
    
    def log_model_performance(self, model_type: str, timeframe: str, accuracy: float, training_duration: float = None):
        """Log ML model performance metrics"""
        self.metrics.model_prediction_accuracy.labels(
            model_type=model_type,
            timeframe=timeframe
        ).set(accuracy)
        
        if training_duration:
            self.metrics.model_training_duration.labels(
                model_type=model_type
            ).observe(training_duration)
        
        self.info(
            f"Model performance recorded for {model_type}/{timeframe}",
            extra={
                'model_type': model_type,
                'timeframe': timeframe,
                'accuracy': accuracy,
                'training_duration': training_duration
            }
        )

# Global logger instance
_logger_instance: Optional[StructuredLogger] = None

def get_logger(config: Dict[str, Any] = None) -> StructuredLogger:
    """Get global structured logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(config)
    return _logger_instance

def configure_logging(config: Dict[str, Any]) -> StructuredLogger:
    """Configure logging with specific settings"""
    global _logger_instance
    _logger_instance = StructuredLogger(config)
    return _logger_instance

def reset_logging():
    """Reset logging instance (for testing)"""
    global _logger_instance
    _logger_instance = None