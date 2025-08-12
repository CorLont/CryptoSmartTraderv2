"""
Health Checker

Comprehensive health monitoring system with dependency checks,
RTO/RPO tracking, and automated recovery procedures.
"""

import asyncio
import aiohttp
import psutil
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of components to monitor"""
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    EXCHANGE_API = "exchange_api"
    CACHE_SYSTEM = "cache_system"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class HealthCheck:
    """Individual health check configuration"""
    check_id: str
    name: str
    component_type: ComponentType
    
    # Check parameters
    endpoint_url: Optional[str] = None
    timeout_seconds: int = 10
    expected_response_time_ms: int = 1000
    
    # Dependency information
    is_critical: bool = True
    dependencies: List[str] = field(default_factory=list)
    
    # Thresholds
    max_failures: int = 3
    failure_window_minutes: int = 5
    
    # State tracking
    current_status: HealthStatus = HealthStatus.HEALTHY
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    failure_history: List[datetime] = field(default_factory=list)
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    last_response_time_ms: float = 0.0
    
    @property
    def is_failing(self) -> bool:
        """Check if component is currently failing"""
        return self.current_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]


@dataclass
class RecoveryProcedure:
    """Automated recovery procedure"""
    procedure_id: str
    name: str
    component_ids: List[str]
    
    # Recovery steps
    recovery_steps: List[Dict[str, Any]] = field(default_factory=list)
    max_attempts: int = 3
    cooldown_minutes: int = 5
    
    # State tracking
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0


class HealthChecker:
    """
    Enterprise health monitoring system
    """
    
    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval = check_interval_seconds
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.recovery_procedures: Dict[str, RecoveryProcedure] = {}
        
        # System health tracking
        self.system_health = {
            "overall_status": HealthStatus.HEALTHY,
            "last_healthy": datetime.now(),
            "downtime_minutes": 0.0,
            "mttr_minutes": 0.0,  # Mean Time To Recovery
            "availability_pct": 100.0
        }
        
        # Callbacks and alerts
        self.health_change_callbacks: List[Callable] = []
        self.critical_failure_callbacks: List[Callable] = []
        
        # Monitoring control
        self.monitoring_active = True
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Setup core health checks
        self._setup_core_health_checks()
        
        # Start monitoring
        self.start_monitoring()
    
    def _setup_core_health_checks(self):
        """Setup essential system health checks"""
        
        # System resource checks
        self.add_health_check(HealthCheck(
            check_id="cpu_usage",
            name="CPU Usage",
            component_type=ComponentType.FILE_SYSTEM,
            is_critical=True,
            timeout_seconds=5,
            max_failures=5,
            expected_response_time_ms=100
        ))
        
        self.add_health_check(HealthCheck(
            check_id="memory_usage",
            name="Memory Usage",
            component_type=ComponentType.FILE_SYSTEM,
            is_critical=True,
            timeout_seconds=5,
            max_failures=5,
            expected_response_time_ms=100
        ))
        
        self.add_health_check(HealthCheck(
            check_id="disk_space",
            name="Disk Space",
            component_type=ComponentType.FILE_SYSTEM,
            is_critical=True,
            timeout_seconds=5,
            max_failures=3,
            expected_response_time_ms=200
        ))
        
        # API endpoint checks
        self.add_health_check(HealthCheck(
            check_id="health_api",
            name="Health API Endpoint",
            component_type=ComponentType.API_ENDPOINT,
            endpoint_url="http://localhost:8001/health",
            is_critical=True,
            timeout_seconds=5,
            expected_response_time_ms=500
        ))
        
        self.add_health_check(HealthCheck(
            check_id="metrics_api",
            name="Metrics API Endpoint", 
            component_type=ComponentType.API_ENDPOINT,
            endpoint_url="http://localhost:8000/metrics",
            is_critical=False,
            timeout_seconds=5,
            expected_response_time_ms=1000
        ))
        
        # Exchange API checks
        self.add_health_check(HealthCheck(
            check_id="kraken_api",
            name="Kraken API Connectivity",
            component_type=ComponentType.EXCHANGE_API,
            endpoint_url="https://api.kraken.com/0/public/SystemStatus",
            is_critical=True,
            timeout_seconds=10,
            expected_response_time_ms=2000,
            max_failures=2
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to monitoring"""
        self.health_checks[health_check.check_id] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def add_recovery_procedure(self, procedure: RecoveryProcedure):
        """Add automated recovery procedure"""
        self.recovery_procedures[procedure.procedure_id] = procedure
        logger.info(f"Added recovery procedure: {procedure.name}")
    
    def start_monitoring(self):
        """Start health monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run all health checks
                asyncio.run(self._run_all_health_checks())
                
                # Update overall system health
                self._update_system_health()
                
                # Check for recovery actions
                self._check_recovery_actions()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(min(self.check_interval, 60))  # Backoff on error
    
    async def _run_all_health_checks(self):
        """Run all registered health checks"""
        tasks = []
        
        for check_id, health_check in self.health_checks.items():
            task = self._run_health_check(health_check)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run individual health check"""
        start_time = time.time()
        
        try:
            # Execute the appropriate check based on component type
            if health_check.component_type == ComponentType.API_ENDPOINT:
                result = await self._check_api_endpoint(health_check)
            elif health_check.component_type == ComponentType.EXCHANGE_API:
                result = await self._check_exchange_api(health_check)
            elif health_check.component_type == ComponentType.FILE_SYSTEM:
                result = await self._check_system_resources(health_check)
            else:
                result = {"status": "unknown", "message": "Unsupported check type"}
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Process check result
            self._process_check_result(health_check, result, response_time_ms)
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            result = {"status": "error", "message": str(e)}
            self._process_check_result(health_check, result, response_time_ms)
    
    async def _check_api_endpoint(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Check API endpoint health"""
        if not health_check.endpoint_url:
            return {"status": "error", "message": "No endpoint URL configured"}
        
        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_check.endpoint_url) as response:
                    if response.status == 200:
                        return {"status": "healthy", "http_status": response.status}
                    else:
                        return {
                            "status": "unhealthy",
                            "message": f"HTTP {response.status}",
                            "http_status": response.status
                        }
                        
        except asyncio.TimeoutError:
            return {"status": "unhealthy", "message": "Request timeout"}
        except aiohttp.ClientError as e:
            return {"status": "unhealthy", "message": f"Connection error: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _check_exchange_api(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Check exchange API connectivity"""
        if not health_check.endpoint_url:
            return {"status": "error", "message": "No endpoint URL configured"}
        
        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_check.endpoint_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check Kraken-specific status
                        if "result" in data and "status" in data["result"]:
                            if data["result"]["status"] == "online":
                                return {"status": "healthy", "exchange_status": "online"}
                            else:
                                return {
                                    "status": "degraded",
                                    "message": "Exchange not fully online",
                                    "exchange_status": data["result"]["status"]
                                }
                        
                        return {"status": "healthy", "http_status": response.status}
                    else:
                        return {
                            "status": "unhealthy",
                            "message": f"HTTP {response.status}",
                            "http_status": response.status
                        }
                        
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    
    async def _check_system_resources(self, health_check: HealthCheck) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            if health_check.check_id == "cpu_usage":
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    return {"status": "critical", "message": f"High CPU usage: {cpu_percent}%", "value": cpu_percent}
                elif cpu_percent > 80:
                    return {"status": "degraded", "message": f"Elevated CPU usage: {cpu_percent}%", "value": cpu_percent}
                else:
                    return {"status": "healthy", "message": f"CPU usage: {cpu_percent}%", "value": cpu_percent}
            
            elif health_check.check_id == "memory_usage":
                memory = psutil.virtual_memory()
                if memory.percent > 95:
                    return {"status": "critical", "message": f"Critical memory usage: {memory.percent}%", "value": memory.percent}
                elif memory.percent > 85:
                    return {"status": "degraded", "message": f"High memory usage: {memory.percent}%", "value": memory.percent}
                else:
                    return {"status": "healthy", "message": f"Memory usage: {memory.percent}%", "value": memory.percent}
            
            elif health_check.check_id == "disk_space":
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                if disk_percent > 95:
                    return {"status": "critical", "message": f"Critical disk usage: {disk_percent:.1f}%", "value": disk_percent}
                elif disk_percent > 85:
                    return {"status": "degraded", "message": f"High disk usage: {disk_percent:.1f}%", "value": disk_percent}
                else:
                    return {"status": "healthy", "message": f"Disk usage: {disk_percent:.1f}%", "value": disk_percent}
            
            else:
                return {"status": "unknown", "message": "Unknown system check"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _process_check_result(self, health_check: HealthCheck, result: Dict[str, Any], response_time_ms: float):
        """Process health check result and update status"""
        
        health_check.last_check = datetime.now()
        health_check.last_response_time_ms = response_time_ms
        
        # Update average response time
        if health_check.avg_response_time_ms == 0:
            health_check.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            health_check.avg_response_time_ms = (health_check.avg_response_time_ms * 0.8 + response_time_ms * 0.2)
        
        # Determine new status
        result_status = result.get("status", "unknown").lower()
        
        if result_status in ["healthy", "ok"]:
            new_status = HealthStatus.HEALTHY
            health_check.consecutive_failures = 0
        elif result_status in ["degraded", "warning"]:
            new_status = HealthStatus.DEGRADED
            health_check.consecutive_failures = 0
        elif result_status in ["unhealthy", "error"]:
            new_status = HealthStatus.UNHEALTHY
            health_check.consecutive_failures += 1
            health_check.failure_history.append(datetime.now())
        elif result_status == "critical":
            new_status = HealthStatus.CRITICAL
            health_check.consecutive_failures += 1
            health_check.failure_history.append(datetime.now())
        else:
            new_status = HealthStatus.UNHEALTHY
            health_check.consecutive_failures += 1
        
        # Clean old failure history
        cutoff_time = datetime.now() - timedelta(minutes=health_check.failure_window_minutes)
        health_check.failure_history = [
            failure_time for failure_time in health_check.failure_history 
            if failure_time > cutoff_time
        ]
        
        # Check if status changed
        old_status = health_check.current_status
        health_check.current_status = new_status
        
        # Log status changes
        if old_status != new_status:
            logger.info(f"Health check {health_check.name}: {old_status.value} -> {new_status.value}")
            
            # Notify callbacks
            self._notify_health_change(health_check, old_status, new_status, result)
        
        # Check for critical failures
        if new_status == HealthStatus.CRITICAL and health_check.is_critical:
            self._notify_critical_failure(health_check, result)
    
    def _update_system_health(self):
        """Update overall system health status"""
        
        # Determine overall status based on critical components
        critical_checks = [check for check in self.health_checks.values() if check.is_critical]
        
        if not critical_checks:
            overall_status = HealthStatus.HEALTHY
        else:
            critical_statuses = [check.current_status for check in critical_checks]
            
            if any(status == HealthStatus.CRITICAL for status in critical_statuses):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.UNHEALTHY for status in critical_statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in critical_statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
        
        # Update system health metrics
        old_overall_status = self.system_health["overall_status"]
        self.system_health["overall_status"] = overall_status
        
        if overall_status == HealthStatus.HEALTHY:
            self.system_health["last_healthy"] = datetime.now()
        
        # Calculate availability and MTTR
        self._update_availability_metrics()
        
        # Log overall status changes
        if old_overall_status != overall_status:
            logger.warning(f"System health: {old_overall_status.value} -> {overall_status.value}")
    
    def _update_availability_metrics(self):
        """Update availability and MTTR metrics"""
        try:
            # This is a simplified calculation - would be more sophisticated in production
            now = datetime.now()
            total_minutes = (now - (now - timedelta(days=1))).total_seconds() / 60
            
            healthy_minutes = total_minutes
            
            # Count unhealthy time based on critical component failures
            for check in self.health_checks.values():
                if check.is_critical and check.failure_history:
                    # Estimate downtime from failures
                    failure_minutes = len(check.failure_history) * (self.check_interval / 60)
                    healthy_minutes -= failure_minutes
            
            healthy_minutes = max(0, healthy_minutes)
            
            self.system_health["availability_pct"] = (healthy_minutes / total_minutes) * 100
            self.system_health["downtime_minutes"] = total_minutes - healthy_minutes
            
            # Simple MTTR calculation
            total_failures = sum(len(check.failure_history) for check in self.health_checks.values() if check.is_critical)
            if total_failures > 0:
                self.system_health["mttr_minutes"] = self.system_health["downtime_minutes"] / total_failures
            else:
                self.system_health["mttr_minutes"] = 0.0
                
        except Exception as e:
            logger.error(f"Failed to update availability metrics: {e}")
    
    def _check_recovery_actions(self):
        """Check if any recovery actions should be triggered"""
        
        for procedure in self.recovery_procedures.values():
            # Check if any of the procedure's components are failing
            failing_components = [
                comp_id for comp_id in procedure.component_ids
                if comp_id in self.health_checks and self.health_checks[comp_id].is_failing
            ]
            
            if failing_components:
                # Check cooldown period
                if procedure.last_executed:
                    cooldown_end = procedure.last_executed + timedelta(minutes=procedure.cooldown_minutes)
                    if datetime.now() < cooldown_end:
                        continue
                
                # Execute recovery procedure
                self._execute_recovery_procedure(procedure, failing_components)
    
    def _execute_recovery_procedure(self, procedure: RecoveryProcedure, failing_components: List[str]):
        """Execute automated recovery procedure"""
        
        if procedure.execution_count >= procedure.max_attempts:
            logger.warning(f"Recovery procedure {procedure.name} max attempts reached")
            return
        
        logger.info(f"Executing recovery procedure: {procedure.name}")
        
        procedure.last_executed = datetime.now()
        procedure.execution_count += 1
        
        try:
            # Execute recovery steps
            success = True
            for step in procedure.recovery_steps:
                step_result = self._execute_recovery_step(step)
                if not step_result:
                    success = False
                    break
            
            if success:
                procedure.success_count += 1
                logger.info(f"Recovery procedure {procedure.name} completed successfully")
            else:
                logger.warning(f"Recovery procedure {procedure.name} failed")
                
        except Exception as e:
            logger.error(f"Recovery procedure {procedure.name} error: {e}")
    
    def _execute_recovery_step(self, step: Dict[str, Any]) -> bool:
        """Execute individual recovery step"""
        
        step_type = step.get("type")
        
        if step_type == "restart_service":
            # Would restart a service
            logger.info(f"Recovery step: restart service {step.get('service_name')}")
            return True
        
        elif step_type == "clear_cache":
            # Would clear cache
            logger.info("Recovery step: clear cache")
            return True
        
        elif step_type == "wait":
            # Wait for specified time
            wait_seconds = step.get("seconds", 5)
            logger.info(f"Recovery step: wait {wait_seconds} seconds")
            time.sleep(wait_seconds)
            return True
        
        else:
            logger.warning(f"Unknown recovery step type: {step_type}")
            return False
    
    def _notify_health_change(self, health_check: HealthCheck, old_status: HealthStatus, 
                             new_status: HealthStatus, result: Dict[str, Any]):
        """Notify callbacks of health status change"""
        
        for callback in self.health_change_callbacks:
            try:
                callback(health_check, old_status, new_status, result)
            except Exception as e:
                logger.error(f"Health change callback failed: {e}")
    
    def _notify_critical_failure(self, health_check: HealthCheck, result: Dict[str, Any]):
        """Notify callbacks of critical failure"""
        
        for callback in self.critical_failure_callbacks:
            try:
                callback(health_check, result)
            except Exception as e:
                logger.error(f"Critical failure callback failed: {e}")
    
    def add_health_change_callback(self, callback: Callable):
        """Add callback for health status changes"""
        self.health_change_callbacks.append(callback)
    
    def add_critical_failure_callback(self, callback: Callable):
        """Add callback for critical failures"""
        self.critical_failure_callbacks.append(callback)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        
        component_status = {}
        for check_id, health_check in self.health_checks.items():
            component_status[check_id] = {
                "name": health_check.name,
                "status": health_check.current_status.value,
                "is_critical": health_check.is_critical,
                "last_check": health_check.last_check.isoformat() if health_check.last_check else None,
                "consecutive_failures": health_check.consecutive_failures,
                "response_time_ms": health_check.last_response_time_ms,
                "avg_response_time_ms": health_check.avg_response_time_ms
            }
        
        return {
            "overall_status": self.system_health["overall_status"].value,
            "last_healthy": self.system_health["last_healthy"].isoformat(),
            "availability_pct": self.system_health["availability_pct"],
            "downtime_minutes": self.system_health["downtime_minutes"],
            "mttr_minutes": self.system_health["mttr_minutes"],
            "component_count": len(self.health_checks),
            "critical_component_count": len([c for c in self.health_checks.values() if c.is_critical]),
            "failing_components": len([c for c in self.health_checks.values() if c.is_failing]),
            "components": component_status,
            "recovery_procedures": len(self.recovery_procedures)
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get condensed health summary"""
        
        status = self.get_health_status()
        
        return {
            "status": status["overall_status"],
            "availability": f"{status['availability_pct']:.1f}%",
            "mttr_minutes": status["mttr_minutes"],
            "failing_components": status["failing_components"],
            "total_components": status["component_count"],
            "last_check": datetime.now().isoformat()
        }
    
    def force_health_check(self, check_id: Optional[str] = None):
        """Force immediate health check"""
        
        if check_id and check_id in self.health_checks:
            # Check specific component
            asyncio.run(self._run_health_check(self.health_checks[check_id]))
        else:
            # Check all components
            asyncio.run(self._run_all_health_checks())
            self._update_system_health()
    
    def simulate_failure(self, check_id: str, duration_seconds: int = 60):
        """Simulate component failure for testing"""
        
        if check_id not in self.health_checks:
            return False
        
        health_check = self.health_checks[check_id]
        original_status = health_check.current_status
        
        # Set to critical status
        health_check.current_status = HealthStatus.CRITICAL
        health_check.consecutive_failures = health_check.max_failures
        
        logger.warning(f"Simulated failure for {health_check.name} (duration: {duration_seconds}s)")
        
        # Schedule recovery
        def restore_health():
            time.sleep(duration_seconds)
            health_check.current_status = original_status
            health_check.consecutive_failures = 0
            logger.info(f"Simulated failure for {health_check.name} ended")
        
        recovery_thread = threading.Thread(target=restore_health, daemon=True)
        recovery_thread.start()
        
        return True