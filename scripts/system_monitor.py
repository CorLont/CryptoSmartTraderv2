#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - System Monitor
Continuous system health and performance monitoring
"""

import asyncio
import logging
import signal
import sys
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.structured_logging import setup_structured_logging


class SystemMonitor:
    """System health and performance monitoring service"""

    def __init__(self):
        # Setup structured logging
        setup_structured_logging(
            service_name="system_monitor", log_level="INFO", enable_console=True, enable_file=True
        )

        self.logger = logging.getLogger(__name__)
        self.running = False

        # Monitoring configuration
        self.check_interval = 30  # 30 seconds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time": 5.0,
        }

        # Data storage
        self.metrics_dir = Path("data") / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("System Monitor initialized")

    async def start_monitoring(self):
        """Start the system monitoring service"""
        self.running = True
        self.logger.info("Starting System Monitor")

        try:
            while self.running:
                # Collect system metrics
                metrics = self._collect_system_metrics()

                # Check for alerts
                alerts = self._check_alert_conditions(metrics)

                # Log metrics and alerts
                self._log_metrics(metrics, alerts)

                # Save metrics to file
                self._save_metrics(metrics, alerts)

                # Wait for next check
                await asyncio.sleep(self.check_interval)

        except Exception as e:
            self.logger.error(f"System monitoring error: {e}")
        finally:
            self.logger.info("System Monitor stopped")

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage("/")

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process_count = len(psutil.pids())

            # Application-specific metrics
            app_metrics = self._get_application_metrics()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu": {
                        "percent": cpu_percent,
                        "count": cpu_count,
                        "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
                    },
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "used": memory.used,
                        "percent": memory.percent,
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": disk.percent,
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv,
                    },
                    "processes": {"count": process_count},
                },
                "application": app_metrics,
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def _get_application_metrics(self) -> Dict[str, Any]:
        """Get CryptoSmartTrader application-specific metrics"""
        app_metrics = {"services": {}, "data_files": {}, "log_files": {}}

        try:
            # Check for running Python processes (our services)
            python_processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cmdline", "cpu_percent", "memory_percent"]
            ):
                try:
                    if "python" in proc.info["name"].lower():
                        cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                        if "cryptosmarttrader" in cmdline.lower() or "streamlit" in cmdline.lower():
                            python_processes.append(
                                {
                                    "pid": proc.info["pid"],
                                    "name": proc.info["name"],
                                    "cmdline": cmdline,
                                    "cpu_percent": proc.info["cpu_percent"],
                                    "memory_percent": proc.info["memory_percent"],
                                }
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            app_metrics["services"]["python_processes"] = python_processes

            # Check data directories
            data_dir = Path("data")
            if data_dir.exists():
                app_metrics["data_files"] = {
                    "analysis_files": len(list((data_dir / "analysis").glob("*.json")))
                    if (data_dir / "analysis").exists()
                    else 0,
                    "social_files": len(list((data_dir / "social").glob("*.json")))
                    if (data_dir / "social").exists()
                    else 0,
                    "metrics_files": len(list((data_dir / "metrics").glob("*.json")))
                    if (data_dir / "metrics").exists()
                    else 0,
                }

            # Check log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                log_files = {}
                for log_file in logs_dir.glob("*.log"):
                    try:
                        stat = log_file.stat()
                        log_files[log_file.name] = {
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    except Exception:
                        continue

                app_metrics["log_files"] = log_files

            # Check if main dashboard is accessible
            try:
                import socket

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("localhost", 5000))
                app_metrics["services"]["dashboard_accessible"] = result == 0
                sock.close()
            except Exception:
                app_metrics["services"]["dashboard_accessible"] = False

        except Exception as e:
            self.logger.warning(f"Failed to collect application metrics: {e}")
            app_metrics["error"] = str(e)

        return app_metrics

    def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions based on metrics"""
        alerts = []

        try:
            system = metrics.get("system", {})

            # CPU alert
            cpu_percent = system.get("cpu", {}).get("percent", 0)
            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                alerts.append(
                    {
                        "type": "cpu_high",
                        "severity": "warning",
                        "message": f"High CPU usage: {cpu_percent:.1f}%",
                        "threshold": self.alert_thresholds["cpu_percent"],
                        "current_value": cpu_percent,
                    }
                )

            # Memory alert
            memory_percent = system.get("memory", {}).get("percent", 0)
            if memory_percent > self.alert_thresholds["memory_percent"]:
                alerts.append(
                    {
                        "type": "memory_high",
                        "severity": "warning",
                        "message": f"High memory usage: {memory_percent:.1f}%",
                        "threshold": self.alert_thresholds["memory_percent"],
                        "current_value": memory_percent,
                    }
                )

            # Disk alert
            disk_percent = system.get("disk", {}).get("percent", 0)
            if disk_percent > self.alert_thresholds["disk_percent"]:
                alerts.append(
                    {
                        "type": "disk_high",
                        "severity": "critical",
                        "message": f"High disk usage: {disk_percent:.1f}%",
                        "threshold": self.alert_thresholds["disk_percent"],
                        "current_value": disk_percent,
                    }
                )

            # Application alerts
            app_metrics = metrics.get("application", {})
            services = app_metrics.get("services", {})

            # Dashboard accessibility
            if not services.get("dashboard_accessible", False):
                alerts.append(
                    {
                        "type": "dashboard_unavailable",
                        "severity": "critical",
                        "message": "Main dashboard is not accessible on port 5000",
                        "threshold": "accessible",
                        "current_value": "not_accessible",
                    }
                )

            # Check for running services
            python_processes = services.get("python_processes", [])
            service_names = [proc.get("cmdline", "") for proc in python_processes]

            if not any("streamlit" in cmd for cmd in service_names):
                alerts.append(
                    {
                        "type": "streamlit_not_running",
                        "severity": "warning",
                        "message": "Streamlit dashboard process not detected",
                        "threshold": "running",
                        "current_value": "not_running",
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to check alert conditions: {e}")
            alerts.append(
                {
                    "type": "monitoring_error",
                    "severity": "error",
                    "message": f"Monitoring check failed: {e}",
                    "threshold": "no_error",
                    "current_value": "error",
                }
            )

        return alerts

    def _log_metrics(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """Log system metrics and alerts"""
        try:
            system = metrics.get("system", {})

            # Log basic system info
            cpu_percent = system.get("cpu", {}).get("percent", 0)
            memory_percent = system.get("memory", {}).get("percent", 0)
            disk_percent = system.get("disk", {}).get("percent", 0)

            self.logger.info(
                f"System Status - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%"
            )

            # Log application status
            app_metrics = metrics.get("application", {})
            python_processes = app_metrics.get("services", {}).get("python_processes", [])
            dashboard_accessible = app_metrics.get("services", {}).get(
                "dashboard_accessible", False
            )

            self.logger.info(
                f"Application Status - Python Processes: {len(python_processes)}, Dashboard: {'OK' if dashboard_accessible else 'FAIL'}"
            )

            # Log alerts
            if alerts:
                for alert in alerts:
                    if alert["severity"] == "critical":
                        self.logger.critical(alert["message"])
                    elif alert["severity"] == "warning":
                        self.logger.warning(alert["message"])
                    else:
                        self.logger.error(alert["message"])
            else:
                self.logger.debug("No alerts detected")

        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")

    def _save_metrics(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]):
        """Save metrics and alerts to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.metrics_dir / f"system_metrics_{timestamp}.json"

            data = {"metrics": metrics, "alerts": alerts, "alert_count": len(alerts)}

            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Cleanup old metrics files (keep last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            for old_file in self.metrics_dir.glob("system_metrics_*.json"):
                try:
                    file_time = datetime.fromtimestamp(old_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        old_file.unlink()
                except Exception:
                    continue

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.logger.info("Stopping System Monitor")
        self.running = False


# Signal handlers for graceful shutdown
monitor_instance = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    if monitor_instance:
        monitor_instance.stop_monitoring()
    sys.exit(0)


async def main():
    """Main function to run the system monitor"""
    global monitor_instance

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start monitor
    monitor_instance = SystemMonitor()

    try:
        await monitor_instance.start_monitoring()
    except KeyboardInterrupt:
        monitor_instance.stop_monitoring()
    except Exception as e:
        logging.error(f"Monitor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
