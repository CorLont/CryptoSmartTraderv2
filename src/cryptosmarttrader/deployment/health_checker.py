"""
Health Checker with Real Dependency Validation

Comprehensive health checking system that validates real dependencies
including APIs, databases, file systems, and external services.
"""

import os
import time
import requests
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import tempfile
from pathlib import Path
import socket
import subprocess

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies to check"""

    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK_PORT = "network_port"


class HealthLevel(Enum):
    """Health levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class DependencyCheck:
    """Configuration for a dependency health check"""

    name: str
    type: DependencyType
    description: str
    critical: bool = True

    # Check parameters
    url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    command: Optional[str] = None
    expected_response: Optional[str] = None

    # Thresholds
    timeout_seconds: float = 10.0
    retry_count: int = 2
    warning_threshold: float = 5.0  # Response time warning
    critical_threshold: float = 10.0  # Response time critical

    # Resource thresholds
    max_memory_percent: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    max_disk_percent: Optional[float] = None


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    dependency_name: str
    status: HealthLevel
    response_time_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status in [HealthLevel.HEALTHY, HealthLevel.DEGRADED]


class HealthChecker:
    """
    Comprehensive health checker for real dependencies
    """

    def __init__(self):
        self.dependencies: Dict[str, DependencyCheck] = {}
        self.check_history: List[HealthCheckResult] = []
        self.last_check_time: Optional[datetime] = None

        # Health check results cache
        self.cached_results: Dict[str, HealthCheckResult] = {}
        self.cache_ttl: float = 30.0  # Cache results for 30 seconds

        # Setup default system checks
        self._setup_default_checks()

    def _setup_default_checks(self):
        """Setup default system health checks"""

        # System resources
        self.add_dependency_check(
            DependencyCheck(
                name="system_memory",
                type=DependencyType.SYSTEM_RESOURCE,
                description="System memory usage",
                critical=True,
                max_memory_percent=90.0,
            )
        )

        self.add_dependency_check(
            DependencyCheck(
                name="system_cpu",
                type=DependencyType.SYSTEM_RESOURCE,
                description="System CPU usage",
                critical=False,
                max_cpu_percent=80.0,
            )
        )

        self.add_dependency_check(
            DependencyCheck(
                name="system_disk",
                type=DependencyType.SYSTEM_RESOURCE,
                description="System disk usage",
                critical=True,
                max_disk_percent=85.0,
            )
        )

        # File system write access
        self.add_dependency_check(
            DependencyCheck(
                name="write_directory",
                type=DependencyType.FILE_SYSTEM,
                description="Write directory access",
                critical=True,
                path="./data",
            )
        )

    def add_dependency_check(self, check: DependencyCheck):
        """Add a dependency check"""
        self.dependencies[check.name] = check
        logger.info(f"Added dependency check: {check.name} ({check.type.value})")

    def add_api_check(self, name: str, url: str, critical: bool = True, timeout: float = 10.0):
        """Add API endpoint health check"""
        check = DependencyCheck(
            name=name,
            type=DependencyType.API_ENDPOINT,
            description=f"API endpoint: {url}",
            critical=critical,
            url=url,
            timeout_seconds=timeout,
        )
        self.add_dependency_check(check)

    def add_port_check(self, name: str, host: str, port: int, critical: bool = True):
        """Add network port health check"""
        check = DependencyCheck(
            name=name,
            type=DependencyType.NETWORK_PORT,
            description=f"Network port: {host}:{port}",
            critical=critical,
            host=host,
            port=port,
        )
        self.add_dependency_check(check)

    def add_database_check(self, name: str, connection_string: str, critical: bool = True):
        """Add database connectivity check"""
        check = DependencyCheck(
            name=name,
            type=DependencyType.DATABASE,
            description=f"Database: {name}",
            critical=critical,
            url=connection_string,
        )
        self.add_dependency_check(check)

    def check_api_endpoint(self, check: DependencyCheck) -> HealthCheckResult:
        """Check API endpoint health"""
        start_time = time.time()

        try:
            response = requests.get(
                check.url,
                timeout=check.timeout_seconds,
                headers={"User-Agent": "CryptoSmartTrader-HealthCheck/1.0"},
            )

            response_time = (time.time() - start_time) * 1000

            # Determine status based on response
            if response.status_code == 200:
                if response_time > check.critical_threshold * 1000:
                    status = HealthLevel.CRITICAL
                    message = f"API responsive but slow ({response_time:.0f}ms)"
                elif response_time > check.warning_threshold * 1000:
                    status = HealthLevel.DEGRADED
                    message = f"API responsive but degraded ({response_time:.0f}ms)"
                else:
                    status = HealthLevel.HEALTHY
                    message = f"API healthy ({response_time:.0f}ms)"
            elif 500 <= response.status_code < 600:
                status = HealthLevel.UNHEALTHY
                message = f"API server error: {response.status_code}"
                response_time = (time.time() - start_time) * 1000
            else:
                status = HealthLevel.DEGRADED
                message = f"API unexpected status: {response.status_code}"
                response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                dependency_name=check.name,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={
                    "status_code": response.status_code,
                    "url": check.url,
                    "headers": dict(response.headers),
                },
            )

        except requests.Timeout:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"API timeout after {check.timeout_seconds}s",
                details={"url": check.url, "error": "timeout"},
            )
        except requests.ConnectionError as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"API connection failed: {str(e)}",
                details={"url": check.url, "error": "connection_error"},
            )
        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"API check error: {str(e)}",
                details={"url": check.url, "error": str(e)},
            )

    def check_network_port(self, check: DependencyCheck) -> HealthCheckResult:
        """Check network port connectivity"""
        start_time = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout_seconds)

            result = sock.connect_ex((check.host, check.port))
            response_time = (time.time() - start_time) * 1000

            sock.close()

            if result == 0:
                status = HealthLevel.HEALTHY
                message = f"Port {check.port} accessible ({response_time:.0f}ms)"
            else:
                status = HealthLevel.UNHEALTHY
                message = f"Port {check.port} not accessible"

            return HealthCheckResult(
                dependency_name=check.name,
                status=status,
                response_time_ms=response_time,
                message=message,
                details={"host": check.host, "port": check.port},
            )

        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Port check error: {str(e)}",
                details={"host": check.host, "port": check.port, "error": str(e)},
            )

    def check_file_system(self, check: DependencyCheck) -> HealthCheckResult:
        """Check file system access and write capabilities"""
        start_time = time.time()

        try:
            if not check.path:
                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.CRITICAL,
                    response_time_ms=0,
                    message="No path specified for file system check",
                )

            path = Path(check.path)

            # Check if path exists and is accessible
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return HealthCheckResult(
                        dependency_name=check.name,
                        status=HealthLevel.CRITICAL,
                        response_time_ms=(time.time() - start_time) * 1000,
                        message=f"Cannot create directory: {str(e)}",
                        details={"path": str(path)},
                    )

            # Test write access
            test_file = path / f".health_check_{int(time.time())}"
            try:
                with open(test_file, "w") as f:
                    f.write("health_check_test")

                # Test read access
                with open(test_file, "r") as f:
                    content = f.read()

                if content != "health_check_test":
                    raise Exception("Read/write verification failed")

                # Cleanup
                test_file.unlink()

                response_time = (time.time() - start_time) * 1000

                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.HEALTHY,
                    response_time_ms=response_time,
                    message=f"File system accessible ({response_time:.0f}ms)",
                    details={"path": str(path), "writable": True},
                )

            except Exception as e:
                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.CRITICAL,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message=f"File system write test failed: {str(e)}",
                    details={"path": str(path), "error": str(e)},
                )

        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"File system check error: {str(e)}",
                details={"error": str(e)},
            )

    def check_system_resources(self, check: DependencyCheck) -> HealthCheckResult:
        """Check system resource usage"""
        start_time = time.time()

        try:
            details = {}
            messages = []
            overall_status = HealthLevel.HEALTHY

            # Memory check
            if check.max_memory_percent:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                details["memory_percent"] = memory_percent
                details["memory_available_gb"] = memory.available / (1024**3)

                if memory_percent > check.max_memory_percent:
                    overall_status = HealthLevel.CRITICAL
                    messages.append(f"Memory usage critical: {memory_percent:.1f}%")
                elif memory_percent > check.max_memory_percent * 0.8:
                    overall_status = max(overall_status, HealthLevel.DEGRADED)
                    messages.append(f"Memory usage high: {memory_percent:.1f}%")
                else:
                    messages.append(f"Memory usage normal: {memory_percent:.1f}%")

            # CPU check
            if check.max_cpu_percent:
                cpu_percent = psutil.cpu_percent(interval=1)
                details["cpu_percent"] = cpu_percent
                details["cpu_count"] = psutil.cpu_count()

                if cpu_percent > check.max_cpu_percent:
                    overall_status = max(overall_status, HealthLevel.DEGRADED)
                    messages.append(f"CPU usage high: {cpu_percent:.1f}%")
                else:
                    messages.append(f"CPU usage normal: {cpu_percent:.1f}%")

            # Disk check
            if check.max_disk_percent:
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100
                details["disk_percent"] = disk_percent
                details["disk_free_gb"] = disk.free / (1024**3)

                if disk_percent > check.max_disk_percent:
                    overall_status = max(overall_status, HealthLevel.CRITICAL)
                    messages.append(f"Disk usage critical: {disk_percent:.1f}%")
                elif disk_percent > check.max_disk_percent * 0.8:
                    overall_status = max(overall_status, HealthLevel.DEGRADED)
                    messages.append(f"Disk usage high: {disk_percent:.1f}%")
                else:
                    messages.append(f"Disk usage normal: {disk_percent:.1f}%")

            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                dependency_name=check.name,
                status=overall_status,
                response_time_ms=response_time,
                message="; ".join(messages),
                details=details,
            )

        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"System resource check error: {str(e)}",
                details={"error": str(e)},
            )

    def check_database(self, check: DependencyCheck) -> HealthCheckResult:
        """Check database connectivity"""
        start_time = time.time()

        try:
            # This is a simplified database check
            # In practice, you'd use specific database connectors

            if "postgresql" in check.url.lower():
                return self._check_postgresql(check, start_time)
            elif "mysql" in check.url.lower():
                return self._check_mysql(check, start_time)
            else:
                # Generic database check via command
                return self._check_database_generic(check, start_time)

        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Database check error: {str(e)}",
                details={"error": str(e)},
            )

    def _check_postgresql(self, check: DependencyCheck, start_time: float) -> HealthCheckResult:
        """Check PostgreSQL database"""
        try:
            import psycopg2

            conn = psycopg2.connect(check.url, connect_timeout=int(check.timeout_seconds))

            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            conn.close()

            response_time = (time.time() - start_time) * 1000

            if result and result[0] == 1:
                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.HEALTHY,
                    response_time_ms=response_time,
                    message=f"PostgreSQL healthy ({response_time:.0f}ms)",
                    details={"database_type": "postgresql"},
                )
            else:
                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.UNHEALTHY,
                    response_time_ms=response_time,
                    message="PostgreSQL query failed",
                )

        except ImportError:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message="psycopg2 not available for PostgreSQL check",
            )
        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"PostgreSQL connection failed: {str(e)}",
            )

    def _check_database_generic(
        self, check: DependencyCheck, start_time: float
    ) -> HealthCheckResult:
        """Generic database check using command"""
        try:
            if check.command:
                result = subprocess.run(
                    check.command.split(),
                    timeout=check.timeout_seconds,
                    capture_output=True,
                    text=True,
                    check=True
                )

                response_time = (time.time() - start_time) * 1000

                if result.returncode == 0:
                    return HealthCheckResult(
                        dependency_name=check.name,
                        status=HealthLevel.HEALTHY,
                        response_time_ms=response_time,
                        message=f"Database command successful ({response_time:.0f}ms)",
                    )
                else:
                    return HealthCheckResult(
                        dependency_name=check.name,
                        status=HealthLevel.UNHEALTHY,
                        response_time_ms=response_time,
                        message=f"Database command failed: {result.stderr}",
                    )
            else:
                return HealthCheckResult(
                    dependency_name=check.name,
                    status=HealthLevel.CRITICAL,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="No database check command specified",
                )

        except Exception as e:
            return HealthCheckResult(
                dependency_name=check.name,
                status=HealthLevel.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Database command error: {str(e)}",
            )

    def check_dependency(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Check a specific dependency"""
        if name not in self.dependencies:
            logger.error(f"Dependency {name} not found")
            return None

        # Check cache first
        if use_cache and name in self.cached_results:
            cached = self.cached_results[name]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.cache_ttl:
                return cached

        check = self.dependencies[name]

        # Perform appropriate check based on type
        if check.type == DependencyType.API_ENDPOINT:
            result = self.check_api_endpoint(check)
        elif check.type == DependencyType.NETWORK_PORT:
            result = self.check_network_port(check)
        elif check.type == DependencyType.FILE_SYSTEM:
            result = self.check_file_system(check)
        elif check.type == DependencyType.SYSTEM_RESOURCE:
            result = self.check_system_resources(check)
        elif check.type == DependencyType.DATABASE:
            result = self.check_database(check)
        else:
            result = HealthCheckResult(
                dependency_name=name,
                status=HealthLevel.CRITICAL,
                response_time_ms=0,
                message=f"Unknown dependency type: {check.type}",
            )

        # Cache and store result
        self.cached_results[name] = result
        self.check_history.append(result)

        # Limit history size
        if len(self.check_history) > 1000:
            self.check_history = self.check_history[-500:]

        return result

    def check_all_dependencies(
        self, include_non_critical: bool = True
    ) -> Dict[str, HealthCheckResult]:
        """Check all dependencies"""
        results = {}

        for name, check in self.dependencies.items():
            if not include_non_critical and not check.critical:
                continue

            result = self.check_dependency(name, use_cache=False)
            if result:
                results[name] = result

        self.last_check_time = datetime.now()
        return results

    def get_overall_health(self) -> HealthLevel:
        """Get overall system health status"""
        results = self.check_all_dependencies()

        if not results:
            return HealthLevel.UNKNOWN

        # Check critical dependencies first
        critical_unhealthy = any(
            not result.is_healthy
            for name, result in results.items()
            if self.dependencies[name].critical
        )

        if critical_unhealthy:
            return HealthLevel.CRITICAL

        # Check for any unhealthy dependencies
        any_unhealthy = any(not result.is_healthy for result in results.values())
        if any_unhealthy:
            return HealthLevel.DEGRADED

        return HealthLevel.HEALTHY

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        results = self.check_all_dependencies()
        overall_health = self.get_overall_health()

        report = {
            "overall_status": overall_health.value,
            "timestamp": datetime.now().isoformat(),
            "dependencies": {},
            "summary": {
                "total_checks": len(results),
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "critical": 0,
            },
        }

        for name, result in results.items():
            report["dependencies"][name] = {
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "message": result.message,
                "critical": self.dependencies[name].critical,
                "details": result.details,
            }

            # Update summary
            if result.status == HealthLevel.HEALTHY:
                report["summary"]["healthy"] += 1
            elif result.status == HealthLevel.DEGRADED:
                report["summary"]["degraded"] += 1
            elif result.status == HealthLevel.UNHEALTHY:
                report["summary"]["unhealthy"] += 1
            else:
                report["summary"]["critical"] += 1

        return report

    def get_rto_rpo_metrics(self) -> Dict[str, Any]:
        """Get Recovery Time Objective and Recovery Point Objective metrics"""

        # RTO: Maximum tolerable downtime
        # RPO: Maximum tolerable data loss

        return {
            "rto_target_seconds": 30,  # Target: 0-to-healthy < 30s
            "rpo_target_seconds": 60,  # Target: < 1 minute data loss
            "current_uptime_seconds": self._get_current_uptime(),
            "last_recovery_time_seconds": self._get_last_recovery_time(),
            "backup_frequency_seconds": 300,  # 5 minutes
            "checkpoint_frequency_seconds": 60,  # 1 minute
        }

    def _get_current_uptime(self) -> float:
        """Get current system uptime"""
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return 0.0

    def _get_last_recovery_time(self) -> Optional[float]:
        """Get last recovery time from logs"""
        # This would typically read from recovery logs
        # For now, return None
        return None
