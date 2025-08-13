#!/usr/bin/env python3
"""
System Monitor - Cross-platform resource and availability monitoring

Provides comprehensive system monitoring with configurable ports, authentic metrics,
and cross-platform compatibility for production environments.
"""

import os
import json
import socket
import time
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging

# Cross-platform imports with fallbacks
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - using limited system monitoring")

try:
    from ..core.consolidated_logging_manager import get_consolidated_logger
except ImportError:

    def get_consolidated_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


@dataclass
class SystemMetrics:
    """System resource metrics"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    load_average: Optional[List[float]] = None
    network_connections: int = 0
    process_count: int = 0
    boot_time: Optional[datetime] = None
    platform_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceStatus:
    """Service availability status"""

    name: str
    host: str
    port: int
    is_accessible: bool
    response_time_ms: Optional[float] = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


@dataclass
class SystemMonitorReport:
    """Complete system monitoring report"""

    timestamp: datetime
    system_metrics: SystemMetrics
    service_statuses: List[ServiceStatus]
    alerts: List[str]
    overall_health: str  # "healthy", "warning", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """
    Enterprise system monitor with cross-platform compatibility

    Provides resource monitoring, service availability checks, and configurable
    alert thresholds with support for multiple dashboard ports and platforms.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize system monitor

        Args:
            config_path: Optional path to monitor configuration
        """
        self.logger = get_consolidated_logger("SystemMonitor")

        # Platform detection - MUST BE BEFORE CONFIG LOADING
        self.platform_name = platform.system().lower()
        self.is_windows = self.platform_name == "windows"
        self.is_linux = self.platform_name == "linux"
        self.is_darwin = self.platform_name == "darwin"

        # Load configuration with proper defaults (needs platform info)
        self.config = self._load_config(config_path)

        # Monitoring state
        self.last_report: Optional[SystemMonitorReport] = None
        self.report_history: List[SystemMonitorReport] = []
        self.alert_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.monitor_count = 0
        self.total_monitor_time = 0.0

        self.logger.info(f"System Monitor initialized for platform: {self.platform_name}")

        # Validate psutil availability
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - limited monitoring capabilities")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load system monitor configuration with proper defaults"""

        default_config = {
            # Alert thresholds - CLEANED UP: removed unused response_time
            "alert_thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0,
                "load_average_per_core": 2.0,
            },
            # Service monitoring - CONFIGURABLE PORTS
            "services": {
                "dashboard": {
                    "name": "Dashboard",
                    "host": "localhost",
                    "ports": [
                        int(os.getenv("DASHBOARD_PORT", "5000")),  # Primary Replit port
                        8501,  # Default Streamlit port
                        3000,  # Common development port
                        8080,  # Alternative port
                    ],
                    "timeout_seconds": 5.0,
                },
                "api": {
                    "name": "API Server",
                    "host": "localhost",
                    "ports": [int(os.getenv("API_PORT", "8000")), 5000, 8080],
                    "timeout_seconds": 3.0,
                },
            },
            # Monitoring intervals
            "intervals": {
                "monitor_seconds": 30,
                "history_retention_hours": 24,
                "alert_cooldown_minutes": 5,
            },
            # Cross-platform settings
            "platform": {
                "enable_load_average": True,  # Will be disabled on Windows
                "enable_network_monitoring": True,
                "enable_process_monitoring": True,
            },
            # Output configuration
            "output": {
                "json_file": "system_monitor_report.json",
                "log_level": "INFO",
                "include_metadata": True,
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                self._deep_merge_dict(default_config, user_config)
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to load config from {config_path}: {e}, using defaults"
                )

        # Platform-specific configuration adjustments
        if self.is_windows:
            default_config["platform"]["enable_load_average"] = False
            self.logger.info("Disabled load average monitoring on Windows platform")

        return default_config

    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics with cross-platform compatibility

        Returns:
            SystemMetrics with current system state
        """

        timestamp = datetime.now(timezone.utc)

        if not PSUTIL_AVAILABLE:
            # Fallback metrics when psutil not available
            return SystemMetrics(
                timestamp=timestamp,
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                platform_info={"system": self.platform_name, "psutil": "unavailable"},
            )

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage(".")
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100.0

            # Load average - CROSS-PLATFORM COMPATIBLE
            load_average = None
            if self.config["platform"]["enable_load_average"] and hasattr(psutil, "getloadavg"):
                try:
                    load_average = list(psutil.getloadavg())
                except (AttributeError, OSError) as e:
                    self.logger.debug(f"Load average not available: {e}")
                    load_average = None

            # Network connections - optional
            network_connections = 0
            if self.config["platform"]["enable_network_monitoring"]:
                try:
                    connections = psutil.net_connections()
                    network_connections = len(connections)
                except (psutil.AccessDenied, OSError):
                    self.logger.debug("Network connections monitoring requires privileges")

            # Process count - optional
            process_count = 0
            if self.config["platform"]["enable_process_monitoring"]:
                try:
                    process_count = len(psutil.pids())
                except (psutil.AccessDenied, OSError):
                    self.logger.debug("Process monitoring limited due to permissions")

            # Boot time
            boot_time = None
            try:
                boot_timestamp = psutil.boot_time()
                boot_time = datetime.fromtimestamp(boot_timestamp, timezone.utc)
            except OSError:
                self.logger.debug("Boot time not available")

            # Platform information
            platform_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "psutil_version": psutil.__version__ if PSUTIL_AVAILABLE else "unavailable",
            }

            return SystemMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                load_average=load_average,
                network_connections=network_connections,
                process_count=process_count,
                boot_time=boot_time,
                platform_info=platform_info,
            )

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=timestamp,
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                platform_info={"error": str(e), "system": self.platform_name},
            )

    def check_service_availability(self) -> List[ServiceStatus]:
        """
        Check availability of configured services with MULTIPLE PORT SUPPORT

        Returns:
            List of ServiceStatus for all configured services
        """

        service_statuses = []

        for service_key, service_config in self.config["services"].items():
            service_name = service_config["name"]
            host = service_config["host"]
            ports = service_config.get("ports", [service_config.get("port", 80)])
            timeout = service_config.get("timeout_seconds", 5.0)

            # Ensure ports is a list
            if not isinstance(ports, list):
                ports = [ports]

            service_accessible = False
            successful_port = None
            best_response_time = None
            error_messages = []

            # Try each port until one succeeds - CONFIGURABLE PORT STRATEGY
            for port in ports:
                try:
                    start_time = time.time()

                    # Create socket with timeout
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)

                    # Test connection
                    result = sock.connect_ex((host, port))

                    response_time_ms = (time.time() - start_time) * 1000
                    sock.close()

                    if result == 0:
                        # Service is accessible on this port
                        service_accessible = True
                        successful_port = port
                        best_response_time = response_time_ms
                        self.logger.debug(f"{service_name} accessible on port {port}")
                        break
                    else:
                        error_messages.append(f"Port {port}: Connection failed (code {result})")

                except Exception as e:
                    error_messages.append(f"Port {port}: {str(e)}")
                    continue

            # Create status report
            status = ServiceStatus(
                name=service_name,
                host=host,
                port=successful_port if successful_port else ports[0],
                is_accessible=service_accessible,
                response_time_ms=best_response_time,
                last_check=datetime.now(timezone.utc),
                error_message="; ".join(error_messages)
                if error_messages and not service_accessible
                else None,
            )

            service_statuses.append(status)

            if not service_accessible:
                self.logger.warning(
                    f"{service_name} not accessible on any configured port: {ports}"
                )
            else:
                self.logger.debug(
                    f"{service_name} accessible on {host}:{successful_port} ({best_response_time:.1f}ms)"
                )

        return service_statuses

    def analyze_alerts(self, metrics: SystemMetrics, services: List[ServiceStatus]) -> List[str]:
        """
        Analyze system state and generate relevant alerts

        Args:
            metrics: Current system metrics
            services: Current service statuses

        Returns:
            List of alert messages
        """

        alerts = []
        thresholds = self.config["alert_thresholds"]

        # CPU alerts
        if metrics.cpu_percent > thresholds["cpu_percent"]:
            alerts.append(
                f"High CPU usage: {metrics.cpu_percent:.1f}% (threshold: {thresholds['cpu_percent']}%)"
            )

        # Memory alerts
        if metrics.memory_percent > thresholds["memory_percent"]:
            alerts.append(
                f"High memory usage: {metrics.memory_percent:.1f}% (threshold: {thresholds['memory_percent']}%)"
            )

        # Disk alerts
        if metrics.disk_percent > thresholds["disk_percent"]:
            alerts.append(
                f"High disk usage: {metrics.disk_percent:.1f}% (threshold: {thresholds['disk_percent']}%)"
            )

        # Load average alerts - CROSS-PLATFORM SAFE
        if metrics.load_average and PSUTIL_AVAILABLE and hasattr(psutil, "cpu_count"):
            try:
                cpu_count = psutil.cpu_count()
                if cpu_count and len(metrics.load_average) > 0:
                    load_per_core = metrics.load_average[0] / cpu_count
                    threshold = thresholds["load_average_per_core"]

                    if load_per_core > threshold:
                        alerts.append(
                            f"High load average: {load_per_core:.2f} per core (threshold: {threshold})"
                        )
            except Exception as e:
                self.logger.debug(f"Load average analysis failed: {e}")

        # Service availability alerts
        for service in services:
            if not service.is_accessible:
                alerts.append(
                    f"Service unavailable: {service.name} ({service.host}:{service.port})"
                )

        # Low resource alerts
        if PSUTIL_AVAILABLE and metrics.memory_total_gb > 0:
            free_memory_gb = metrics.memory_total_gb - metrics.memory_used_gb
            if free_memory_gb < 0.5:  # Less than 500MB free
                alerts.append(f"Low free memory: {free_memory_gb:.1f}GB remaining")

        if metrics.disk_total_gb > 0:
            free_disk_gb = metrics.disk_total_gb - metrics.disk_used_gb
            if free_disk_gb < 1.0:  # Less than 1GB free
                alerts.append(f"Low free disk space: {free_disk_gb:.1f}GB remaining")

        return alerts

    def determine_overall_health(
        self, metrics: SystemMetrics, services: List[ServiceStatus], alerts: List[str]
    ) -> str:
        """
        Determine overall system health status

        Args:
            metrics: System metrics
            services: Service statuses
            alerts: Current alerts

        Returns:
            Health status: "healthy", "warning", or "critical"
        """

        # Critical conditions
        critical_conditions = [
            metrics.cpu_percent > 95,
            metrics.memory_percent > 95,
            metrics.disk_percent > 95,
            any(
                not service.is_accessible
                for service in services
                if "dashboard" in service.name.lower(),
        ]

        if any(critical_conditions):
            return "critical"

        # Warning conditions
        warning_conditions = [
            len(alerts) > 0,
            metrics.cpu_percent > self.config["alert_thresholds"]["cpu_percent"],
            metrics.memory_percent > self.config["alert_thresholds"]["memory_percent"],
            metrics.disk_percent > self.config["alert_thresholds"]["disk_percent"],
        ]

        if any(warning_conditions):
            return "warning"

        return "healthy"

    def generate_monitoring_report(self) -> SystemMonitorReport:
        """
        Generate comprehensive system monitoring report

        Returns:
            SystemMonitorReport with complete system state
        """

        start_time = time.time()

        try:
            # Collect system metrics
            metrics = self.collect_system_metrics()

            # Check service availability
            services = self.check_service_availability()

            # Analyze alerts
            alerts = self.analyze_alerts(metrics, services)

            # Determine overall health
            overall_health = self.determine_overall_health(metrics, services, alerts)

            # Create report
            report = SystemMonitorReport(
                timestamp=datetime.now(timezone.utc),
                system_metrics=metrics,
                service_statuses=services,
                alerts=alerts,
                overall_health=overall_health,
                metadata={
                    "monitor_duration_seconds": time.time() - start_time,
                    "platform": self.platform_name,
                    "psutil_available": PSUTIL_AVAILABLE,
                    "config_version": "2.0.0",
                    "services_checked": len(services),
                    "alerts_generated": len(alerts),
                },
            )

            # Update monitoring state
            self.last_report = report
            self.report_history.append(report)
            self._cleanup_history()

            # Update performance metrics
            self.monitor_count += 1
            self.total_monitor_time += report.metadata["monitor_duration_seconds"]

            self.logger.info(
                f"Monitoring report generated: {overall_health} ({len(alerts)} alerts)"
            )

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate monitoring report: {e}")

            # Return emergency report
            return self._create_emergency_report(str(e))

    def save_report_to_json(
        self, report: SystemMonitorReport, output_path: Optional[str] = None
    ) -> bool:
        """
        Save monitoring report to JSON file

        Args:
            report: SystemMonitorReport to save
            output_path: Optional custom output path

        Returns:
            True if successful, False otherwise
        """

        if output_path is None:
            output_path = self.config["output"]["json_file"]

        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert report to serializable format
            report_data = {
                "timestamp": report.timestamp.isoformat(),
                "overall_health": report.overall_health,
                "system_metrics": {
                    "timestamp": report.system_metrics.timestamp.isoformat(),
                    "cpu_percent": report.system_metrics.cpu_percent,
                    "memory_percent": report.system_metrics.memory_percent,
                    "memory_used_gb": report.system_metrics.memory_used_gb,
                    "memory_total_gb": report.system_metrics.memory_total_gb,
                    "disk_percent": report.system_metrics.disk_percent,
                    "disk_used_gb": report.system_metrics.disk_used_gb,
                    "disk_total_gb": report.system_metrics.disk_total_gb,
                    "load_average": report.system_metrics.load_average,
                    "network_connections": report.system_metrics.network_connections,
                    "process_count": report.system_metrics.process_count,
                    "boot_time": report.system_metrics.boot_time.isoformat()
                    if report.system_metrics.boot_time
                    else None,
                    "platform_info": report.system_metrics.platform_info,
                },
                "service_statuses": [
                    {
                        "name": service.name,
                        "host": service.host,
                        "port": service.port,
                        "is_accessible": service.is_accessible,
                        "response_time_ms": service.response_time_ms,
                        "last_check": service.last_check.isoformat(),
                        "error_message": service.error_message,
                    }
                    for service in report.service_statuses
                ],
                "alerts": report.alerts,
                "metadata": report.metadata,
            }

            with open(output_file, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            self.logger.debug(f"Report saved to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return False

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring system status"""

        return {
            "monitor_status": "active",
            "platform": self.platform_name,
            "psutil_available": PSUTIL_AVAILABLE,
            "last_report": self.last_report.timestamp.isoformat() if self.last_report else None,
            "total_reports": self.monitor_count,
            "average_duration": self.total_monitor_time / max(1, self.monitor_count),
            "history_length": len(self.report_history),
            "services_configured": len(self.config.get("services", {})),
            "alert_thresholds": self.config.get("alert_thresholds", {}),
        }

    def _create_emergency_report(self, error_message: str) -> SystemMonitorReport:
        """Create emergency report when monitoring fails"""

        return SystemMonitorReport(
            timestamp=datetime.now(timezone.utc),
            system_metrics=SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                platform_info={"error": error_message},
            ),
            service_statuses=[],
            alerts=[f"CRITICAL: System monitoring failure - {error_message}"],
            overall_health="critical",
            metadata={"emergency_report": True, "error": error_message},
        )

    def _cleanup_history(self):
        """Clean up old monitoring reports"""

        retention_hours = self.config["intervals"]["history_retention_hours"]
        cutoff_time = datetime.now(timezone.utc).timestamp() - (retention_hours * 3600)

        self.report_history = [
            report for report in self.report_history if report.timestamp.timestamp() > cutoff_time
        ]

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

        return base


# Utility functions for system monitoring


def quick_system_check() -> Dict[str, Any]:
    """Perform quick system resource check"""

    monitor = SystemMonitor()
    report = monitor.generate_monitoring_report()

    return {
        "health": report.overall_health,
        "cpu_percent": report.system_metrics.cpu_percent,
        "memory_percent": report.system_metrics.memory_percent,
        "disk_percent": report.system_metrics.disk_percent,
        "alerts": len(report.alerts),
        "services_up": sum(1 for s in report.service_statuses if s.is_accessible),
        "timestamp": report.timestamp.isoformat(),
    }


def detailed_system_report(output_file: Optional[str] = None) -> SystemMonitorReport:
    """Generate detailed system monitoring report"""

    monitor = SystemMonitor()
    report = monitor.generate_monitoring_report()

    if output_file:
        monitor.save_report_to_json(report, output_file)

    return report


if __name__ == "__main__":
    # Test system monitoring
    print("Testing System Monitor")

    monitor = SystemMonitor()
    report = monitor.generate_monitoring_report()

    print(f"\nSystem Health: {report.overall_health}")
    print(f"CPU: {report.system_metrics.cpu_percent:.1f}%")
    print(f"Memory: {report.system_metrics.memory_percent:.1f}%")
    print(f"Disk: {report.system_metrics.disk_percent:.1f}%")

    print(f"\nServices ({len(report.service_statuses)}):")
    for service in report.service_statuses:
        status_icon = "✅" if service.is_accessible else "❌"
        print(f"  {status_icon} {service.name}: {service.host}:{service.port}")

    if report.alerts:
        print(f"\nAlerts ({len(report.alerts)}):")
        for alert in report.alerts:
            print(f"  ⚠️ {alert}")

    # Save report
    monitor.save_report_to_json(report)

    print(f"\n✅ SYSTEM MONITOR TEST COMPLETE")
    print(f"Platform: {monitor.platform_name}")
    print(f"Report saved to: {monitor.config['output']['json_file']}")
