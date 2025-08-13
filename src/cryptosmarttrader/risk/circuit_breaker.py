"""
Circuit Breaker System

Automated circuit breakers for data gaps, latency spikes,
model drift, and other system anomalies.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    DATA_GAP = "data_gap"
    LATENCY_SPIKE = "latency_spike"
    MODEL_DRIFT = "model_drift"
    PREDICTION_ANOMALY = "prediction_anomaly"
    EXECUTION_FAILURE = "execution_failure"
    MARKET_ANOMALY = "market_anomaly"
    SYSTEM_RESOURCE = "system_resource"

class BreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Breaker triggered, blocking operations
    HALF_OPEN = "half_open"  # Testing if issue is resolved

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""
    name: str
    breaker_type: CircuitBreakerType
    description: str

    # Thresholds
    failure_threshold: int = 5      # Number of failures to trigger
    timeout_seconds: float = 300    # How long to stay open (5 minutes)
    half_open_max_calls: int = 3    # Max calls in half-open state

    # Detection parameters
    latency_threshold_ms: float = 1000.0
    data_gap_threshold_minutes: float = 5.0
    drift_threshold: float = 0.1
    anomaly_threshold: float = 3.0  # Standard deviations

    # Actions
    block_new_trades: bool = True
    send_alerts: bool = True
    log_violations: bool = True
    alert_severity: AlertSeverity = AlertSeverity.HIGH

@dataclass
class BreakerEvent:
    """Circuit breaker event record"""
    timestamp: datetime
    breaker_name: str
    event_type: str  # "triggered", "reset", "half_open"
    trigger_reason: str

    # Context data
    failure_count: int = 0
    latency_ms: Optional[float] = None
    data_gap_minutes: Optional[float] = None
    drift_score: Optional[float] = None
    anomaly_score: Optional[float] = None

    # System context
    system_load: Optional[float] = None
    memory_usage: Optional[float] = None

class CircuitBreakerManager:
    """
    Comprehensive circuit breaker system for trading protection
    """

    def __init__(self):
        self.breakers: Dict[str, 'CircuitBreaker'] = {}
        self.events: List[BreakerEvent] = []

        # Global state
        self.trading_enabled = True
        self.last_data_timestamp = datetime.now()
        self.system_monitoring_active = False

        # Performance tracking
        self.latency_history: List[Tuple[datetime, float]] = []
        self.prediction_history: List[Tuple[datetime, float]] = []
        self.execution_history: List[Tuple[datetime, bool]] = []

        # Setup default breakers
        self._setup_default_breakers()

        # Start monitoring thread
        self._start_monitoring()

    def _setup_default_breakers(self):
        """Setup default circuit breakers"""

        # Data gap breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="data_gap_breaker",
            breaker_type=CircuitBreakerType.DATA_GAP,
            description="Triggers when market data is stale",
            failure_threshold=1,  # Immediate trigger
            timeout_seconds=300,  # 5 minutes
            data_gap_threshold_minutes=2.0,  # 2 minutes without data
            block_new_trades=True,
            alert_severity=AlertSeverity.CRITICAL
        ))

        # Latency spike breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="latency_spike_breaker",
            breaker_type=CircuitBreakerType.LATENCY_SPIKE,
            description="Triggers on excessive API/execution latency",
            failure_threshold=3,  # 3 high latency events
            timeout_seconds=180,  # 3 minutes
            latency_threshold_ms=2000.0,  # 2 seconds
            block_new_trades=True,
            alert_severity=AlertSeverity.HIGH
        ))

        # Model drift breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="model_drift_breaker",
            breaker_type=CircuitBreakerType.MODEL_DRIFT,
            description="Triggers on significant model drift",
            failure_threshold=5,  # 5 drift events
            timeout_seconds=600,  # 10 minutes
            drift_threshold=0.15,  # 15% drift
            block_new_trades=True,
            alert_severity=AlertSeverity.HIGH
        ))

        # Prediction anomaly breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="prediction_anomaly_breaker",
            breaker_type=CircuitBreakerType.PREDICTION_ANOMALY,
            description="Triggers on extreme prediction values",
            failure_threshold=3,  # 3 anomalous predictions
            timeout_seconds=300,  # 5 minutes
            anomaly_threshold=4.0,  # 4 standard deviations
            block_new_trades=True,
            alert_severity=AlertSeverity.HIGH
        ))

        # Execution failure breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="execution_failure_breaker",
            breaker_type=CircuitBreakerType.EXECUTION_FAILURE,
            description="Triggers on repeated execution failures",
            failure_threshold=5,  # 5 failed executions
            timeout_seconds=600,  # 10 minutes
            block_new_trades=True,
            alert_severity=AlertSeverity.CRITICAL
        ))

        # System resource breaker
        self.add_circuit_breaker(CircuitBreakerConfig(
            name="system_resource_breaker",
            breaker_type=CircuitBreakerType.SYSTEM_RESOURCE,
            description="Triggers on system resource exhaustion",
            failure_threshold=2,  # 2 resource alerts
            timeout_seconds=300,  # 5 minutes
            block_new_trades=True,
            alert_severity=AlertSeverity.CRITICAL
        ))

    def add_circuit_breaker(self, config: CircuitBreakerConfig):
        """Add a circuit breaker to the system"""
        breaker = CircuitBreaker(config, self)
        self.breakers[config.name] = breaker
        logger.info(f"Added circuit breaker: {config.name}")

    def record_data_update(self, timestamp: Optional[datetime] = None):
        """Record that new market data was received"""
        if timestamp is None:
            timestamp = datetime.now()
        self.last_data_timestamp = timestamp

    def record_api_latency(self, latency_ms: float):
        """Record API call latency"""
        timestamp = datetime.now()
        self.latency_history.append((timestamp, latency_ms))

        # Keep only last 100 entries
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]

        # Check latency breaker
        latency_breaker = self.breakers.get("latency_spike_breaker")
        if latency_breaker:
            latency_breaker.check_latency(latency_ms)

    def record_prediction(self, prediction_value: float):
        """Record a model prediction"""
        timestamp = datetime.now()
        self.prediction_history.append((timestamp, prediction_value))

        # Keep only last 100 entries
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]

        # Check anomaly breaker
        anomaly_breaker = self.breakers.get("prediction_anomaly_breaker")
        if anomaly_breaker:
            anomaly_breaker.check_prediction_anomaly(prediction_value)

    def record_execution_result(self, success: bool):
        """Record execution success/failure"""
        timestamp = datetime.now()
        self.execution_history.append((timestamp, success))

        # Keep only last 100 entries
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

        # Check execution breaker
        execution_breaker = self.breakers.get("execution_failure_breaker")
        if execution_breaker:
            execution_breaker.check_execution_failure(not success)

    def record_model_drift(self, drift_score: float):
        """Record model drift score"""
        drift_breaker = self.breakers.get("model_drift_breaker")
        if drift_breaker:
            drift_breaker.check_model_drift(drift_score)

    def check_system_resources(self):
        """Check system resource usage"""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            memory_pct = memory.percent

            # CPU usage
            cpu_pct = psutil.cpu_percent(interval=1)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_pct = (disk.used / disk.total) * 100

            # Check thresholds
            resource_critical = (
                memory_pct > 90 or
                cpu_pct > 90 or
                disk_pct > 95
            )

            if resource_critical:
                resource_breaker = self.breakers.get("system_resource_breaker")
                if resource_breaker:
                    resource_breaker.trigger_failure(
                        f"System resources critical: Memory {memory_pct:.1f}%, "
                        f"CPU {cpu_pct:.1f}%, Disk {disk_pct:.1f}%"
                    )

        except ImportError:
            logger.warning("psutil not available for system resource monitoring")
        except Exception as e:
            logger.error(f"System resource check failed: {e}")

    def _start_monitoring(self):
        """Start background monitoring thread"""

        def monitoring_loop():
            while True:
                try:
                    # Check data gaps
                    self._check_data_gaps()

                    # Check system resources
                    self.check_system_resources()

                    # Update breaker states
                    self._update_breaker_states()

                    # Sleep for 30 seconds
                    time.sleep(30)

                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(60)  # Longer sleep on error

        self.system_monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info("Circuit breaker monitoring started")

    def _check_data_gaps(self):
        """Check for data gaps"""
        if not self.last_data_timestamp:
            return

        gap_minutes = (datetime.now() - self.last_data_timestamp).total_seconds() / 60

        data_gap_breaker = self.breakers.get("data_gap_breaker")
        if data_gap_breaker:
            data_gap_breaker.check_data_gap(gap_minutes)

    def _update_breaker_states(self):
        """Update circuit breaker states (timeouts, half-open, etc.)"""
        for breaker in self.breakers.values():
            breaker.update_state()

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on circuit breaker states"""

        if not self.trading_enabled:
            return False, "Trading globally disabled"

        # Check each breaker
        for breaker_name, breaker in self.breakers.items():
            if breaker.state == BreakerState.OPEN and breaker.config.block_new_trades:
                return False, f"Circuit breaker open: {breaker_name}"

        return True, "Trading allowed"

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        breaker_status = {}
        for name, breaker in self.breakers.items():
            breaker_status[name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                "time_until_half_open": breaker.time_until_half_open(),
                "description": breaker.config.description
            }

        # Recent events
        recent_events = [
            event for event in self.events
            if event.timestamp >= datetime.now() - timedelta(hours=24)
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "trading_enabled": self.trading_enabled,
            "system_monitoring_active": self.system_monitoring_active,
            "last_data_update": self.last_data_timestamp.isoformat(),
            "minutes_since_data": (datetime.now() - self.last_data_timestamp).total_seconds() / 60,
            "breaker_status": breaker_status,
            "recent_events_24h": len(recent_events),
            "open_breakers": [
                name for name, breaker in self.breakers.items()
                if breaker.state == BreakerState.OPEN
            ]
        }

    def manually_trigger_breaker(self, breaker_name: str, reason: str) -> bool:
        """Manually trigger a circuit breaker"""
        breaker = self.breakers.get(breaker_name)
        if breaker:
            breaker.trigger_failure(f"Manual trigger: {reason}")
            return True
        return False

    def manually_reset_breaker(self, breaker_name: str) -> bool:
        """Manually reset a circuit breaker"""
        breaker = self.breakers.get(breaker_name)
        if breaker:
            breaker.reset()
            return True
        return False

    def export_events_report(self, filepath: str, days_back: int = 7):
        """Export circuit breaker events report"""

        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_events = [
                event for event in self.events
                if event.timestamp >= cutoff_date
            ]

            report_data = {
                "report_timestamp": datetime.now().isoformat(),
                "period_days": days_back,
                "system_status": self.get_system_status(),

                "events": [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "breaker_name": event.breaker_name,
                        "event_type": event.event_type,
                        "trigger_reason": event.trigger_reason,
                        "failure_count": event.failure_count,
                        "latency_ms": event.latency_ms,
                        "data_gap_minutes": event.data_gap_minutes,
                        "drift_score": event.drift_score,
                        "anomaly_score": event.anomaly_score
                    }
                    for event in recent_events
                ],

                "performance_metrics": {
                    "avg_latency_ms": np.mean([l for _, l in self.latency_history]) if self.latency_history else 0,
                    "max_latency_ms": np.max([l for _, l in self.latency_history]) if self.latency_history else 0,
                    "execution_success_rate": np.mean([s for _, s in self.execution_history]) if self.execution_history else 0,
                    "recent_predictions": len(self.prediction_history)
                }
            }

            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info(f"Circuit breaker events report exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export events report: {e}")

class CircuitBreaker:
    """
    Individual circuit breaker implementation
    """

    def __init__(self, config: CircuitBreakerConfig, manager: CircuitBreakerManager):
        self.config = config
        self.manager = manager

        # State
        self.state = BreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.now()
        self.half_open_call_count = 0

        # Statistics for anomaly detection
        self.prediction_stats = {"mean": 0.0, "std": 1.0, "count": 0}

    def trigger_failure(self, reason: str):
        """Trigger a failure event"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == BreakerState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self._open_breaker(reason)
        elif self.state == BreakerState.HALF_OPEN:
            self._open_breaker(f"Half-open failure: {reason}")

        # Log event
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            event_type="failure",
            trigger_reason=reason,
            failure_count=self.failure_count
        )
        self.manager.events.append(event)

        if self.config.log_violations:
            logger.warning(f"Circuit breaker failure: {self.config.name} - {reason}")

    def _open_breaker(self, reason: str):
        """Open the circuit breaker"""
        self.state = BreakerState.OPEN
        self.state_change_time = datetime.now()

        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            event_type="triggered",
            trigger_reason=reason,
            failure_count=self.failure_count
        )
        self.manager.events.append(event)

        if self.config.send_alerts:
            logger.critical(f"CIRCUIT BREAKER OPEN: {self.config.name} - {reason}")

    def reset(self):
        """Reset the circuit breaker"""
        self.state = BreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.state_change_time = datetime.now()
        self.half_open_call_count = 0

        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            event_type="reset",
            trigger_reason="Manual reset",
            failure_count=0
        )
        self.manager.events.append(event)

        logger.info(f"Circuit breaker reset: {self.config.name}")

    def update_state(self):
        """Update breaker state based on timeout"""
        if self.state == BreakerState.OPEN:
            time_open = (datetime.now() - self.state_change_time).total_seconds()
            if time_open >= self.config.timeout_seconds:
                self._transition_to_half_open()

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self.state = BreakerState.HALF_OPEN
        self.state_change_time = datetime.now()
        self.half_open_call_count = 0

        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_name=self.config.name,
            event_type="half_open",
            trigger_reason="Timeout elapsed",
            failure_count=self.failure_count
        )
        self.manager.events.append(event)

        logger.info(f"Circuit breaker half-open: {self.config.name}")

    def time_until_half_open(self) -> Optional[float]:
        """Get seconds until breaker transitions to half-open"""
        if self.state == BreakerState.OPEN:
            elapsed = (datetime.now() - self.state_change_time).total_seconds()
            remaining = self.config.timeout_seconds - elapsed
            return max(0, remaining)
        return None

    def check_data_gap(self, gap_minutes: float):
        """Check for data gap violation"""
        if self.config.breaker_type != CircuitBreakerType.DATA_GAP:
            return

        if gap_minutes > self.config.data_gap_threshold_minutes:
            self.trigger_failure(f"Data gap: {gap_minutes:.1f} minutes > {self.config.data_gap_threshold_minutes:.1f}")

    def check_latency(self, latency_ms: float):
        """Check for latency violation"""
        if self.config.breaker_type != CircuitBreakerType.LATENCY_SPIKE:
            return

        if latency_ms > self.config.latency_threshold_ms:
            self.trigger_failure(f"High latency: {latency_ms:.1f}ms > {self.config.latency_threshold_ms:.1f}ms")

    def check_model_drift(self, drift_score: float):
        """Check for model drift violation"""
        if self.config.breaker_type != CircuitBreakerType.MODEL_DRIFT:
            return

        if drift_score > self.config.drift_threshold:
            self.trigger_failure(f"Model drift: {drift_score:.3f} > {self.config.drift_threshold:.3f}")

    def check_prediction_anomaly(self, prediction_value: float):
        """Check for prediction anomaly"""
        if self.config.breaker_type != CircuitBreakerType.PREDICTION_ANOMALY:
            return

        # Update rolling statistics
        self._update_prediction_stats(prediction_value)

        # Check if prediction is anomalous
        if self.prediction_stats["std"] > 0:
            z_score = abs(prediction_value - self.prediction_stats["mean"]) / self.prediction_stats["std"]

            if z_score > self.config.anomaly_threshold:
                self.trigger_failure(f"Prediction anomaly: z-score {z_score:.2f} > {self.config.anomaly_threshold:.2f}")

    def check_execution_failure(self, failed: bool):
        """Check execution failure"""
        if self.config.breaker_type != CircuitBreakerType.EXECUTION_FAILURE:
            return

        if failed:
            self.trigger_failure("Execution failed")
        else:
            # Success resets failure count for execution breaker
            if self.state == BreakerState.HALF_OPEN:
                self.half_open_call_count += 1
                if self.half_open_call_count >= self.config.half_open_max_calls:
                    self.reset()

    def _update_prediction_stats(self, value: float):
        """Update rolling statistics for prediction values"""
        # Simple exponential moving average
        alpha = 0.1  # Smoothing factor

        if self.prediction_stats["count"] == 0:
            self.prediction_stats["mean"] = value
            self.prediction_stats["std"] = 1.0
        else:
            # Update mean
            old_mean = self.prediction_stats["mean"]
            self.prediction_stats["mean"] = alpha * value + (1 - alpha) * old_mean

            # Update std (simplified)
            squared_diff = (value - self.prediction_stats["mean"]) ** 2
            if self.prediction_stats["count"] == 1:
                self.prediction_stats["std"] = max(0.1, squared_diff ** 0.5)
            else:
                old_var = self.prediction_stats["std"] ** 2
                new_var = alpha * squared_diff + (1 - alpha) * old_var
                self.prediction_stats["std"] = max(0.1, new_var ** 0.5)

        self.prediction_stats["count"] += 1
