#!/usr/bin/env python3
"""
Improved Logging Manager
Fixed version without correlation_id issues and proper type annotations
"""

import os
import sys
import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings('ignore')

import re

def mask_sensitive_data(message: str) -> str:
    """Mask sensitive data in log messages"""

    # Patterns to mask
    patterns = [
        (r'(api_key["\']?\s*[:=]\s*["\']?)([^"\'\s]+)', r'\1***MASKED***'),
        (r'(token["']?\s*[:=]\s*["']?)([^"'\s]+)', r'\1***MASKED***'),
        (r'(secret["']?\s*[:=]\s*["']?)([^"'\s]+)', r'\1***MASKED***'),
        (r'(password["']?\s*[:=]\s*["']?)([^"'\s]+)', r'\1***MASKED***'),
        (r'(key["']?\s*[:=]\s*["']?)([A-Za-z0-9+/]{20,})', r'\1***MASKED***')
    ]

    masked_message = message
    for pattern, replacement in patterns:
        masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)

    return masked_message


# Try prometheus imports
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Try JSON logger import
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

@dataclass
class CorrelationContext:
    """Correlation context for tracking operations"""
    correlation_id: str
    operation: str
    agent_name: Optional[str] = None
    start_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class PrometheusMetrics:
    """Prometheus metrics for monitoring"""

    def __init__(self, registry: Optional[object] = None):
        if not PROMETHEUS_AVAILABLE:
            return

        self.registry = registry or CollectorRegistry()

        # Operation metrics
        self.operation_counter = Counter(
            'cryptotrader_operations_total',
            'Total number of operations',
            ['operation', 'agent', 'status'],
            registry=self.registry
        )

        self.operation_duration = Histogram(
            'cryptotrader_operation_duration_seconds',
            'Operation duration in seconds',
            ['operation', 'agent', 'status'],
            registry=self.registry
        )

        # System metrics
        self.active_agents = Gauge(
            'cryptotrader_active_agents',
            'Number of active agents',
            ['agent_type'],
            registry=self.registry
        )

        self.prediction_accuracy = Gauge(
            'cryptotrader_prediction_accuracy',
            'Current prediction accuracy',
            ['model', 'horizon'],
            registry=self.registry
        )

    def record_operation(self, operation: str, agent: str, status: str, duration: float):
        """Record operation metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.operation_counter.labels(
            operation=operation,
            agent=agent,
            status=status
        ).inc()

        self.operation_duration.labels(
            operation=operation,
            agent=agent,
            status=status
        ).observe(duration)

class ImprovedLogger:
    """
    Improved logging manager without correlation_id formatting issues
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'log_level': 'INFO',
            'log_dir': './logs',
            'max_file_size_mb': 100,
            'backup_count': 5,
            'enable_metrics': True,
            'metrics_port': 8090,
            'enable_json_logging': True,
            'correlation_tracking': True
        }

        if config:
            self.config.update(config)

        # Thread-local storage for correlation context
        self._correlation_context = threading.local()

        # Setup logging
        self.log_dir = Path(self.config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        if self.config['enable_metrics'] and PROMETHEUS_AVAILABLE:
            self.metrics = PrometheusMetrics()
            self._start_metrics_server()
        else:
            self.metrics = None

        # Setup loggers
        self._setup_loggers()

    def _setup_loggers(self):
        """Setup application loggers"""

        # Main application logger
        self.app_logger = logging.getLogger('CryptoSmartTrader')
        self.app_logger.setLevel(getattr(logging, self.config['log_level']))

        # Clear existing handlers
        self.app_logger.handlers.clear()

        # File handler for general logs
        log_file = self.log_dir / 'application.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.config['max_file_size_mb'] * 1024 * 1024,
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )

        # Simple formatter without correlation_id
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(simple_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers
        self.app_logger.addHandler(file_handler)
        self.app_logger.addHandler(console_handler)

        # Prevent duplicate logs
        self.app_logger.propagate = False

        # Setup JSON logger if available
        if JSON_LOGGER_AVAILABLE and self.config['enable_json_logging']:
            self._setup_json_logger()

    def _setup_json_logger(self):
        """Setup JSON structured logging"""

        json_file = self.log_dir / 'structured.jsonl'
        json_handler = RotatingFileHandler(
            json_file,
            maxBytes=self.config['max_file_size_mb'] * 1024 * 1024,
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )

        # JSON formatter
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        json_handler.setFormatter(json_formatter)

        # Separate JSON logger
        self.json_logger = logging.getLogger('CryptoSmartTrader.JSON')
        self.json_logger.setLevel(getattr(logging, self.config['log_level']))
        self.json_logger.addHandler(json_handler)
        self.json_logger.propagate = False

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            metrics_port = self.config['metrics_port']
            start_http_server(metrics_port, registry=self.metrics.registry)
            print(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            print(f"Failed to start metrics server: {e}")

    @contextmanager
    def correlation_context(self, operation: str, agent_name: Optional[str] = None, **kwargs):
        """Context manager for correlation tracking"""
        correlation_id = str(uuid.uuid4())
        context = CorrelationContext(
            correlation_id=correlation_id,
            operation=operation,
            agent_name=agent_name,
            start_time=time.time(),
            metadata=kwargs
        )

        # Store in thread-local storage
        old_context = getattr(self._correlation_context, 'current', None)
        self._correlation_context.current = context

        try:
            self.info(f"Operation started: {operation}")
            yield context

            # Record successful operation
            duration = time.time() - (context.start_time or 0)
            if self.metrics:
                self.metrics.record_operation(operation, agent_name or 'unknown', 'success', duration)

            self.info(f"Operation completed: {operation} in {duration:.3f}s")

        except Exception as e:
            # Record failed operation
            duration = time.time() - (context.start_time or 0)
            if self.metrics:
                self.metrics.record_operation(operation, agent_name or 'unknown', 'error', duration)

            self.error(f"Operation failed: {operation} after {duration:.3f}s - {e}")
            raise

        finally:
            # Restore previous context
            self._correlation_context.current = old_context

    def get_current_correlation_id(self) -> str:
        """Get current correlation ID"""
        context = getattr(self._correlation_context, 'current', None)
        return context.correlation_id if context else 'no-context'

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        enhanced_extra = self._enhance_extra(extra)
        self.app_logger.info(message, extra=enhanced_extra)

        if hasattr(self, 'json_logger'):
            self.json_logger.info(message, extra=enhanced_extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        enhanced_extra = self._enhance_extra(extra)
        self.app_logger.warning(message, extra=enhanced_extra)

        if hasattr(self, 'json_logger'):
            self.json_logger.warning(message, extra=enhanced_extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        enhanced_extra = self._enhance_extra(extra)
        self.app_logger.error(message, extra=enhanced_extra)

        if hasattr(self, 'json_logger'):
            self.json_logger.error(message, extra=enhanced_extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        enhanced_extra = self._enhance_extra(extra)
        self.app_logger.debug(message, extra=enhanced_extra)

        if hasattr(self, 'json_logger'):
            self.json_logger.debug(message, extra=enhanced_extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        enhanced_extra = self._enhance_extra(extra)
        self.app_logger.critical(message, extra=enhanced_extra)

        if hasattr(self, 'json_logger'):
            self.json_logger.critical(message, extra=enhanced_extra)

    def _enhance_extra(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance extra data with correlation context"""
        enhanced = extra or {}

        # Add correlation context if available
        context = getattr(self._correlation_context, 'current', None)
        if context:
            enhanced.update({
                'correlation_id': context.correlation_id,
                'operation': context.operation,
                'agent_name': context.agent_name
            })

        return enhanced

    def log_performance_metric(self, metric_name: str, value: float,
                             labels: Optional[Dict[str, str]] = None):
        """Log performance metric"""
        if self.metrics and PROMETHEUS_AVAILABLE:
            # Update Prometheus metrics based on metric type
            if hasattr(self.metrics, metric_name):
                metric = getattr(self.metrics, metric_name)
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)

        # Also log as structured data
        self.info(f"Performance metric: {metric_name} = {value}",
                 extra={'metric_name': metric_name, 'metric_value': value, 'labels': labels})

    def flush_logs(self):
        """Flush all log handlers"""
        for handler in self.app_logger.handlers:
            handler.flush()

        if hasattr(self, 'json_logger'):
            for handler in self.json_logger.handlers:
                handler.flush()

# Global logger instance
_improved_logger_instance: Optional[ImprovedLogger] = None

def get_improved_logger(config: Optional[Dict[str, Any]] = None) -> ImprovedLogger:
    """Get singleton improved logger instance"""
    global _improved_logger_instance

    if _improved_logger_instance is None:
        _improved_logger_instance = ImprovedLogger(config)

    return _improved_logger_instance

class DailyLogManager:
    """
    Simplified daily log manager for workstation health reports
    """

    def __init__(self, logger: Optional[ImprovedLogger] = None):
        self.logger = logger or get_improved_logger()
        self.daily_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def log_health_metric(self, metric_name: str, value: Union[int, float],
                         category: str = 'system'):
        """Log health metric to daily logs"""

        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'value': value,
            'category': category
        }

        # Write to daily health file
        health_file = self.daily_dir / f"{category}_metrics.jsonl"

        with open(health_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metric_data) + '\n')

        # Also log to main logger
        self.logger.info(f"Health metric: {metric_name} = {value} ({category})",
                        extra=metric_data)

    def log_trading_event(self, event_type: str, data: Dict[str, Any]):
        """Log trading event to daily logs"""

        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }

        # Write to daily trading file
        trading_file = self.daily_dir / "trading_events.jsonl"

        with open(trading_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data) + '\n')

        # Also log to main logger
        self.logger.info(f"Trading event: {event_type}", extra=event_data)

    def log_confidence_gate_result(self, total_candidates: int, passed_count: int,
                                  pass_rate: float):
        """Log confidence gate results"""

        gate_data = {
            'timestamp': datetime.now().isoformat(),
            'total_candidates': total_candidates,
            'passed_count': passed_count,
            'pass_rate': pass_rate
        }

        # Write to daily confidence file
        confidence_file = self.daily_dir / "confidence_gate.jsonl"

        with open(confidence_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(gate_data) + '\n')

        # Also log to main logger
        self.logger.info(f"Confidence gate: {passed_count}/{total_candidates} passed ({pass_rate:.1%})",
                        extra=gate_data)

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's logs"""

        summary = {
            'date': datetime.now().strftime("%Y%m%d"),
            'health_metrics_count': 0,
            'trading_events_count': 0,
            'confidence_gate_events': 0,
            'latest_pass_rate': 0.0
        }

        # Count health metrics
        for metrics_file in self.daily_dir.glob("*_metrics.jsonl"):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    summary['health_metrics_count'] += len(f.readlines())
            except Exception:
                continue

        # Count trading events
        trading_file = self.daily_dir / "trading_events.jsonl"
        if trading_file.exists():
            try:
                with open(trading_file, 'r', encoding='utf-8') as f:
                    summary['trading_events_count'] = len(f.readlines())
            except Exception:
                pass

        # Get latest confidence gate result
        confidence_file = self.daily_dir / "confidence_gate.jsonl"
        if confidence_file.exists():
            try:
                with open(confidence_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    summary['confidence_gate_events'] = len(lines)

                    if lines:
                        latest_line = lines[-1]
                        latest_data = json.loads(latest_line)
                        summary['latest_pass_rate'] = latest_data.get('pass_rate', 0.0)
            except Exception:
                pass

        return summary

if __name__ == "__main__":
    print("🔧 TESTING IMPROVED LOGGING MANAGER")
    print("=" * 50)

    # Test improved logger
    logger = get_improved_logger()

    print("📝 Testing basic logging...")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    print("🔄 Testing correlation context...")
    with logger.correlation_context("test_operation", "test_agent"):
        logger.info("Inside correlation context")
        time.sleep(0.1)  # REMOVED: Mock data pattern not allowed in production

    print("📊 Testing daily log manager...")
    daily_manager = DailyLogManager(logger)

    # Log some test metrics
    daily_manager.log_health_metric("cpu_usage", 45.2, "system")
    daily_manager.log_health_metric("memory_usage", 32.1, "system")
    daily_manager.log_confidence_gate_result(100, 5, 0.05)

    # Get summary
    summary = daily_manager.get_daily_summary()
    print(f"   Daily summary: {summary}")

    print("✅ Improved logging manager testing completed")
