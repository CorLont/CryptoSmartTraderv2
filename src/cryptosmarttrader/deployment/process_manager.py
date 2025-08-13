"""
Process Manager for Auto-Restart and Health Monitoring

Enterprise-grade process management with auto-restart, health checks,
and exponential back-off for upstream failures.
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

class ProcessState(Enum):
    """Process states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RECOVERING = "recovering"

class HealthStatus(Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

@dataclass
class ProcessConfig:
    """Configuration for a managed process"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    working_dir: str = "."
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Restart configuration
    auto_restart: bool = True
    max_restarts: int = 10
    restart_delay: float = 5.0
    exponential_backoff: bool = True
    max_backoff_delay: float = 300.0  # 5 minutes max

    # Health check configuration
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    health_check_url: Optional[str] = None
    health_check_command: Optional[str] = None

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None

    # Recovery configuration
    grace_period: float = 30.0  # Time to wait for graceful shutdown
    kill_timeout: float = 10.0  # Time to wait before SIGKILL

@dataclass
class ProcessStatus:
    """Current status of a managed process"""
    name: str
    state: ProcessState
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None

    # Resource usage
    memory_mb: float = 0.0
    cpu_percent: float = 0.0

    # Error tracking
    last_error: Optional[str] = None
    consecutive_failures: int = 0

    @property
    def uptime_seconds(self) -> float:
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    @property
    def is_healthy(self) -> bool:
        return (self.state == ProcessState.RUNNING and
                self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED])

class ProcessManager:
    """
    Enterprise process manager with auto-restart and health monitoring
    """

    def __init__(self, config_dir: str = "./config/processes"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Process tracking
        self.processes: Dict[str, subprocess.Popen] = {}
        self.process_configs: Dict[str, ProcessConfig] = {}
        self.process_status: Dict[str, ProcessStatus] = {}

        # Health monitoring
        self.health_checkers: Dict[str, threading.Thread] = {}
        self.monitoring_active = False

        # Recovery tracking
        self.recovery_start_times: Dict[str, datetime] = {}
        self.restart_delays: Dict[str, float] = {}

        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        self.stop_all_processes()

    def register_process(self, config: ProcessConfig):
        """Register a process for management"""
        try:
            self.process_configs[config.name] = config
            self.process_status[config.name] = ProcessStatus(
                name=config.name,
                state=ProcessState.STOPPED
            )
            self.restart_delays[config.name] = config.restart_delay

            logger.info(f"Registered process: {config.name}")

        except Exception as e:
            logger.error(f"Failed to register process {config.name}: {e}")

    def start_process(self, name: str) -> bool:
        """Start a managed process"""
        try:
            if name not in self.process_configs:
                logger.error(f"Process {name} not registered")
                return False

            config = self.process_configs[name]
            status = self.process_status[name]

            if status.state == ProcessState.RUNNING:
                logger.warning(f"Process {name} already running")
                return True

            # Update state
            status.state = ProcessState.STARTING
            logger.info(f"Starting process: {name}")

            # Prepare environment
            env = os.environ.copy()
            env.update(config.env_vars)

            # Start process
            full_command = [config.command] + config.args
            process = subprocess.Popen(
                full_command,
                cwd=config.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Store process reference
            self.processes[name] = process

            # Update status
            status.pid = process.pid
            status.start_time = datetime.now()
            status.state = ProcessState.RUNNING
            status.consecutive_failures = 0

            # Start health monitoring if configured
            if config.health_check_url or config.health_check_command:
                self._start_health_monitoring(name)

            logger.info(f"Process {name} started with PID {process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start process {name}: {e}")
            self.process_status[name].state = ProcessState.FAILED
            self.process_status[name].last_error = str(e)
            return False

    def stop_process(self, name: str, graceful: bool = True) -> bool:
        """Stop a managed process"""
        try:
            if name not in self.processes:
                logger.warning(f"Process {name} not running")
                return True

            process = self.processes[name]
            config = self.process_configs[name]
            status = self.process_status[name]

            status.state = ProcessState.STOPPING
            logger.info(f"Stopping process: {name}")

            if graceful:
                # Try graceful shutdown first
                process.terminate()

                try:
                    process.wait(timeout=config.grace_period)
                    logger.info(f"Process {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {name} did not stop gracefully, killing...")
                    process.kill()
                    process.wait(timeout=config.kill_timeout)
            else:
                # Force kill
                process.kill()
                process.wait(timeout=config.kill_timeout)

            # Cleanup
            del self.processes[name]
            status.state = ProcessState.STOPPED
            status.pid = None
            status.start_time = None

            # Stop health monitoring
            self._stop_health_monitoring(name)

            logger.info(f"Process {name} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop process {name}: {e}")
            return False

    def restart_process(self, name: str) -> bool:
        """Restart a managed process"""
        try:
            config = self.process_configs[name]
            status = self.process_status[name]

            # Check restart limits
            if status.restart_count >= config.max_restarts:
                logger.error(f"Process {name} has exceeded max restarts ({config.max_restarts})")
                status.state = ProcessState.FAILED
                return False

            # Apply exponential backoff
            if config.exponential_backoff and status.restart_count > 0:
                delay = min(
                    self.restart_delays[name] * (2 ** status.consecutive_failures),
                    config.max_backoff_delay
                )
                logger.info(f"Applying restart delay: {delay:.1f}s for {name}")
                time.sleep(delay)
            else:
                time.sleep(config.restart_delay)

            # Stop process if running
            if name in self.processes:
                self.stop_process(name)

            # Start process
            success = self.start_process(name)

            if success:
                status.restart_count += 1
                status.last_restart = datetime.now()
                status.consecutive_failures = 0
                self.recovery_start_times[name] = datetime.now()
                logger.info(f"Process {name} restarted (count: {status.restart_count})")
            else:
                status.consecutive_failures += 1
                logger.error(f"Failed to restart process {name}")

            return success

        except Exception as e:
            logger.error(f"Failed to restart process {name}: {e}")
            return False

    def _start_health_monitoring(self, name: str):
        """Start health monitoring thread for a process"""
        if name in self.health_checkers:
            return

        def health_monitor():
            config = self.process_configs[name]
            status = self.process_status[name]

            while (not self.shutdown_requested and
                   name in self.processes and
                   status.state == ProcessState.RUNNING):

                try:
                    # Perform health check
                    health_result = self._perform_health_check(name)
                    status.health_status = health_result
                    status.last_health_check = datetime.now()

                    # Update resource usage
                    self._update_resource_usage(name)

                    # Check if restart is needed
                    if health_result == HealthStatus.UNHEALTHY:
                        logger.warning(f"Health check failed for {name}, restarting...")
                        if config.auto_restart:
                            self.restart_process(name)
                            break

                    time.sleep(config.health_check_interval)

                except Exception as e:
                    logger.error(f"Health monitoring error for {name}: {e}")
                    time.sleep(config.health_check_interval)

        thread = threading.Thread(target=health_monitor, daemon=True)
        thread.start()
        self.health_checkers[name] = thread

    def _stop_health_monitoring(self, name: str):
        """Stop health monitoring for a process"""
        if name in self.health_checkers:
            # Thread will stop naturally when process is not running
            del self.health_checkers[name]

    def _perform_health_check(self, name: str) -> HealthStatus:
        """Perform health check for a process"""
        try:
            config = self.process_configs[name]

            # Check if process is still running
            if name not in self.processes:
                return HealthStatus.UNHEALTHY

            process = self.processes[name]
            if process.poll() is not None:
                return HealthStatus.UNHEALTHY

            # URL-based health check
            if config.health_check_url:
                import requests
                try:
                    response = requests.get(
                        config.health_check_url,
                        timeout=config.health_check_timeout
                    )
                    if response.status_code == 200:
                        return HealthStatus.HEALTHY
                    elif 500 <= response.status_code < 600:
                        return HealthStatus.UNHEALTHY
                    else:
                        return HealthStatus.DEGRADED
                except requests.RequestException:
                    return HealthStatus.UNHEALTHY

            # Command-based health check
            if config.health_check_command:
                try:
                    result = subprocess.run(
                        config.health_check_command.split(),
                        timeout=config.health_check_timeout,
                        capture_output=True
                    )
                    return HealthStatus.HEALTHY if result.returncode == 0 else HealthStatus.UNHEALTHY
                except subprocess.TimeoutExpired:
                    return HealthStatus.UNHEALTHY

            # Default: just check if process is running
            return HealthStatus.HEALTHY

        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return HealthStatus.UNKNOWN

    def _update_resource_usage(self, name: str):
        """Update resource usage for a process"""
        try:
            if name not in self.processes:
                return

            process = self.processes[name]
            status = self.process_status[name]

            if process.pid:
                psutil_process = psutil.Process(process.pid)

                # Memory usage
                memory_info = psutil_process.memory_info()
                status.memory_mb = memory_info.rss / 1024 / 1024

                # CPU usage
                status.cpu_percent = psutil_process.cpu_percent()

                # Check resource limits
                config = self.process_configs[name]

                if (config.max_memory_mb and
                    status.memory_mb > config.max_memory_mb):
                    logger.warning(f"Process {name} exceeds memory limit: {status.memory_mb:.1f}MB > {config.max_memory_mb}MB")

                if (config.max_cpu_percent and
                    status.cpu_percent > config.max_cpu_percent):
                    logger.warning(f"Process {name} exceeds CPU limit: {status.cpu_percent:.1f}% > {config.max_cpu_percent}%")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Could not get resource usage for {name}: {e}")
        except Exception as e:
            logger.error(f"Resource usage update failed for {name}: {e}")

    def get_process_status(self, name: str) -> Optional[ProcessStatus]:
        """Get status of a specific process"""
        return self.process_status.get(name)

    def get_all_status(self) -> Dict[str, ProcessStatus]:
        """Get status of all managed processes"""
        return self.process_status.copy()

    def is_healthy(self) -> bool:
        """Check if all processes are healthy"""
        return all(
            status.is_healthy for status in self.process_status.values()
            if self.process_configs[status.name].auto_restart
        )

    def get_recovery_time(self, name: str) -> Optional[float]:
        """Get recovery time for a process (0-to-healthy)"""
        if name not in self.recovery_start_times:
            return None

        status = self.process_status.get(name)
        if not status or not status.is_healthy:
            return None

        recovery_start = self.recovery_start_times[name]
        return (datetime.now() - recovery_start).total_seconds()

    def start_all_processes(self) -> bool:
        """Start all registered processes"""
        success = True
        for name in self.process_configs:
            if not self.start_process(name):
                success = False
        return success

    def stop_all_processes(self):
        """Stop all managed processes"""
        for name in list(self.processes.keys()):
            self.stop_process(name)

    def monitor_processes(self):
        """Main monitoring loop"""
        self.monitoring_active = True
        logger.info("Process monitoring started")

        try:
            while not self.shutdown_requested and self.monitoring_active:
                # Check for dead processes
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        # Process died
                        status = self.process_status[name]
                        config = self.process_configs[name]

                        logger.warning(f"Process {name} died (exit code: {process.returncode})")
                        status.state = ProcessState.FAILED
                        status.consecutive_failures += 1

                        # Auto-restart if enabled
                        if config.auto_restart and not self.shutdown_requested:
                            logger.info(f"Auto-restarting process {name}")
                            self.restart_process(name)

                time.sleep(5.0)  # Check every 5 seconds

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted")
        finally:
            self.monitoring_active = False
            logger.info("Process monitoring stopped")

    def save_status_snapshot(self, filepath: str):
        """Save current status to file for recovery"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'processes': {}
            }

            for name, status in self.process_status.items():
                snapshot['processes'][name] = {
                    'state': status.state.value,
                    'restart_count': status.restart_count,
                    'health_status': status.health_status.value,
                    'uptime_seconds': status.uptime_seconds,
                    'memory_mb': status.memory_mb,
                    'cpu_percent': status.cpu_percent
                }

            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2)

            logger.debug(f"Status snapshot saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save status snapshot: {e}")

    def load_status_snapshot(self, filepath: str) -> bool:
        """Load status from snapshot file"""
        try:
            if not os.path.exists(filepath):
                return False

            with open(filepath, 'r') as f:
                snapshot = json.load(f)

            logger.info(f"Loaded status snapshot from {snapshot['timestamp']}")
            return True

        except Exception as e:
            logger.error(f"Failed to load status snapshot: {e}")
            return False
