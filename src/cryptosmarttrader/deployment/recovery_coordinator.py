"""
Recovery Coordinator

Orchestrates system recovery with RTO/RPO compliance,
automated failover, and comprehensive health monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

from .process_manager import ProcessManager, ProcessConfig, ProcessState
from .health_checker import HealthChecker, DependencyCheck, HealthLevel
from .deployment_config import DeploymentConfig

logger = logging.getLogger(__name__)

class RecoveryState(Enum):
    """Recovery states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RecoveryEvent:
    """Recovery event tracking"""
    timestamp: datetime
    event_type: str
    description: str
    alert_level: AlertLevel
    component: str
    recovery_action: Optional[str] = None
    duration_seconds: Optional[float] = None
    success: bool = False

class RecoveryCoordinator:
    """
    Enterprise recovery coordination system
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()
        
        # Recovery state tracking
        self.recovery_state = RecoveryState.HEALTHY
        self.recovery_events: List[RecoveryEvent] = []
        self.last_healthy_time: Optional[datetime] = datetime.now()
        
        # Monitoring threads
        self.monitoring_active = False
        self.recovery_thread: Optional[threading.Thread] = None
        
        # RTO/RPO tracking
        self.rto_violations: List[Dict[str, Any]] = []
        self.rpo_violations: List[Dict[str, Any]] = []
        self.last_backup_time: Optional[datetime] = None
        self.last_checkpoint_time: Optional[datetime] = None
        
        # Setup health dependencies
        self._setup_health_dependencies()
        
        # Recovery statistics
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'rto_violations': 0,
            'rpo_violations': 0
        }
    
    def _setup_health_dependencies(self):
        """Setup health dependencies from configuration"""
        for dep_config in self.config.health_dependencies:
            if dep_config["type"] == "system_resource":
                check = DependencyCheck(
                    name=dep_config["name"],
                    type=DependencyType.SYSTEM_RESOURCE,
                    description=dep_config.get("description", dep_config["name"]),
                    critical=dep_config.get("critical", True),
                    max_memory_percent=dep_config.get("max_memory_percent"),
                    max_cpu_percent=dep_config.get("max_cpu_percent"),
                    max_disk_percent=dep_config.get("max_disk_percent")
                )
            elif dep_config["type"] == "file_system":
                check = DependencyCheck(
                    name=dep_config["name"],
                    type=DependencyType.FILE_SYSTEM,
                    description=dep_config.get("description", dep_config["name"]),
                    critical=dep_config.get("critical", True),
                    path=dep_config.get("path")
                )
            else:
                continue  # Skip unknown types
            
            self.health_checker.add_dependency_check(check)
    
    def start_monitoring(self):
        """Start recovery monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.recovery_thread = threading.Thread(target=self._recovery_loop, daemon=True)
        self.recovery_thread.start()
        
        logger.info("Recovery monitoring started")
    
    def stop_monitoring(self):
        """Stop recovery monitoring"""
        self.monitoring_active = False
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5.0)
        
        logger.info("Recovery monitoring stopped")
    
    def _recovery_loop(self):
        """Main recovery monitoring loop"""
        while self.monitoring_active:
            try:
                # Check overall system health
                overall_health = self.health_checker.get_overall_health()
                
                # Determine recovery actions needed
                if overall_health == HealthLevel.CRITICAL:
                    self._handle_critical_failure()
                elif overall_health == HealthLevel.UNHEALTHY:
                    self._handle_unhealthy_state()
                elif overall_health == HealthLevel.DEGRADED:
                    self._handle_degraded_state()
                else:
                    self._handle_healthy_state()
                
                # Check RTO/RPO compliance
                self._check_rto_compliance()
                self._check_rpo_compliance()
                
                # Perform automatic backups/checkpoints
                self._perform_automatic_backup()
                self._perform_automatic_checkpoint()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                time.sleep(10)  # Longer delay on errors
    
    def _handle_critical_failure(self):
        """Handle critical system failure"""
        if self.recovery_state != RecoveryState.FAILED:
            self.recovery_state = RecoveryState.FAILED
            
            event = RecoveryEvent(
                timestamp=datetime.now(),
                event_type="critical_failure",
                description="Critical system failure detected",
                alert_level=AlertLevel.CRITICAL,
                component="system"
            )
            
            self.recovery_events.append(event)
            logger.critical("Critical system failure - initiating emergency recovery")
            
            # Emergency recovery actions
            self._emergency_recovery()
    
    def _handle_unhealthy_state(self):
        """Handle unhealthy system state"""
        if self.recovery_state not in [RecoveryState.RECOVERING, RecoveryState.FAILED]:
            self.recovery_state = RecoveryState.RECOVERING
            
            event = RecoveryEvent(
                timestamp=datetime.now(),
                event_type="unhealthy_state",
                description="Unhealthy system state detected",
                alert_level=AlertLevel.ERROR,
                component="system",
                recovery_action="automated_recovery"
            )
            
            self.recovery_events.append(event)
            logger.error("System unhealthy - starting recovery")
            
            # Start recovery process
            self._automated_recovery()
    
    def _handle_degraded_state(self):
        """Handle degraded system state"""
        if self.recovery_state == RecoveryState.HEALTHY:
            self.recovery_state = RecoveryState.DEGRADED
            
            event = RecoveryEvent(
                timestamp=datetime.now(),
                event_type="degraded_state",
                description="System performance degraded",
                alert_level=AlertLevel.WARNING,
                component="system"
            )
            
            self.recovery_events.append(event)
            logger.warning("System performance degraded")
    
    def _handle_healthy_state(self):
        """Handle healthy system state"""
        if self.recovery_state != RecoveryState.HEALTHY:
            # Recovery completed
            recovery_time = None
            if self.last_healthy_time:
                recovery_time = (datetime.now() - self.last_healthy_time).total_seconds()
            
            self.recovery_state = RecoveryState.HEALTHY
            self.last_healthy_time = datetime.now()
            
            event = RecoveryEvent(
                timestamp=datetime.now(),
                event_type="recovery_completed",
                description="System recovered to healthy state",
                alert_level=AlertLevel.INFO,
                component="system",
                duration_seconds=recovery_time,
                success=True
            )
            
            self.recovery_events.append(event)
            
            # Update recovery statistics
            if recovery_time:
                self.recovery_stats['total_recoveries'] += 1
                self.recovery_stats['successful_recoveries'] += 1
                
                # Update average recovery time
                total_time = (self.recovery_stats['average_recovery_time'] * 
                            (self.recovery_stats['total_recoveries'] - 1) + recovery_time)
                self.recovery_stats['average_recovery_time'] = total_time / self.recovery_stats['total_recoveries']
                
                logger.info(f"Recovery completed in {recovery_time:.1f} seconds")
    
    def _emergency_recovery(self):
        """Emergency recovery procedures"""
        logger.critical("Executing emergency recovery procedures")
        
        # Stop all processes
        self.process_manager.stop_all_processes()
        
        # Wait briefly
        time.sleep(2)
        
        # Restart critical services only
        critical_services = ["api", "metrics"]
        for service_name in critical_services:
            service_config = self.config.get_service_config(service_name)
            if service_config:
                self._restart_service(service_name, service_config)
    
    def _automated_recovery(self):
        """Automated recovery procedures"""
        logger.info("Executing automated recovery procedures")
        
        # Get health report to identify failing components
        health_report = self.health_checker.get_health_report()
        
        # Restart unhealthy services
        for dep_name, dep_status in health_report['dependencies'].items():
            if dep_status['status'] in ['unhealthy', 'critical']:
                self._recover_dependency(dep_name, dep_status)
    
    def _recover_dependency(self, dep_name: str, dep_status: Dict[str, Any]):
        """Recover a specific dependency"""
        logger.info(f"Recovering dependency: {dep_name}")
        
        # Check if it's a service that can be restarted
        for service_name, service_config in self.config.services.items():
            if dep_name.endswith(service_name) or service_name in dep_name:
                self._restart_service(service_name, service_config)
                return
        
        # For other dependencies, log the issue
        logger.warning(f"Cannot automatically recover dependency: {dep_name}")
    
    def _restart_service(self, service_name: str, service_config: Dict[str, Any]):
        """Restart a specific service"""
        logger.info(f"Restarting service: {service_name}")
        
        # Stop service if running
        self.process_manager.stop_process(service_name)
        
        # Wait briefly
        time.sleep(1)
        
        # Start service
        success = self.process_manager.restart_process(service_name)
        
        if success:
            logger.info(f"Service {service_name} restarted successfully")
        else:
            logger.error(f"Failed to restart service {service_name}")
    
    def _check_rto_compliance(self):
        """Check Recovery Time Objective compliance"""
        if self.recovery_state in [RecoveryState.RECOVERING, RecoveryState.FAILED]:
            if self.last_healthy_time:
                downtime = (datetime.now() - self.last_healthy_time).total_seconds()
                
                if downtime > self.config.rto_target_seconds:
                    # RTO violation
                    violation = {
                        'timestamp': datetime.now().isoformat(),
                        'downtime_seconds': downtime,
                        'rto_target_seconds': self.config.rto_target_seconds,
                        'violation_seconds': downtime - self.config.rto_target_seconds
                    }
                    
                    self.rto_violations.append(violation)
                    self.recovery_stats['rto_violations'] += 1
                    
                    logger.error(f"RTO violation: {downtime:.1f}s > {self.config.rto_target_seconds}s")
    
    def _check_rpo_compliance(self):
        """Check Recovery Point Objective compliance"""
        if self.last_backup_time:
            time_since_backup = (datetime.now() - self.last_backup_time).total_seconds()
            
            if time_since_backup > self.config.rpo_target_seconds:
                # RPO violation
                violation = {
                    'timestamp': datetime.now().isoformat(),
                    'time_since_backup': time_since_backup,
                    'rpo_target_seconds': self.config.rpo_target_seconds,
                    'violation_seconds': time_since_backup - self.config.rpo_target_seconds
                }
                
                self.rpo_violations.append(violation)
                self.recovery_stats['rpo_violations'] += 1
                
                logger.warning(f"RPO violation: {time_since_backup:.1f}s > {self.config.rpo_target_seconds}s")
    
    def _perform_automatic_backup(self):
        """Perform automatic backup if needed"""
        if (not self.last_backup_time or 
            (datetime.now() - self.last_backup_time).total_seconds() >= self.config.backup_interval_seconds):
            
            self._create_backup()
    
    def _perform_automatic_checkpoint(self):
        """Perform automatic checkpoint if needed"""
        if (not self.last_checkpoint_time or 
            (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.config.checkpoint_interval_seconds):
            
            self._create_checkpoint()
    
    def _create_backup(self):
        """Create system backup"""
        try:
            backup_dir = Path(self.config.backup_directory)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.json"
            
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'system_state': {
                    'recovery_state': self.recovery_state.value,
                    'processes': self.process_manager.get_all_status(),
                    'health_status': self.health_checker.get_health_report()
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.last_backup_time = datetime.now()
            logger.debug(f"Backup created: {backup_file}")
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
    
    def _create_checkpoint(self):
        """Create system checkpoint"""
        try:
            # Save process status snapshot
            self.process_manager.save_status_snapshot("./data/process_checkpoint.json")
            
            self.last_checkpoint_time = datetime.now()
            logger.debug("Checkpoint created")
            
        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive recovery report"""
        return {
            'current_state': self.recovery_state.value,
            'last_healthy_time': self.last_healthy_time.isoformat() if self.last_healthy_time else None,
            'recovery_statistics': self.recovery_stats,
            'rto_compliance': {
                'target_seconds': self.config.rto_target_seconds,
                'violations': len(self.rto_violations),
                'last_violation': self.rto_violations[-1] if self.rto_violations else None
            },
            'rpo_compliance': {
                'target_seconds': self.config.rpo_target_seconds,
                'violations': len(self.rpo_violations),
                'last_backup': self.last_backup_time.isoformat() if self.last_backup_time else None,
                'last_checkpoint': self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None
            },
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type,
                    'description': event.description,
                    'alert_level': event.alert_level.value,
                    'component': event.component,
                    'duration': event.duration_seconds,
                    'success': event.success
                }
                for event in self.recovery_events[-10:]  # Last 10 events
            ]
        }
    
    def force_recovery_test(self) -> Dict[str, Any]:
        """Force a recovery test for RTO validation"""
        logger.info("Starting forced recovery test")
        
        test_start = datetime.now()
        
        # Simulate failure by stopping a service
        test_service = "api"
        self.process_manager.stop_process(test_service)
        
        # Wait for detection and recovery
        max_wait = self.config.rto_target_seconds + 10
        recovery_detected = False
        
        for _ in range(int(max_wait)):
            status = self.process_manager.get_process_status(test_service)
            if status and status.state == ProcessState.RUNNING:
                recovery_detected = True
                break
            time.sleep(1)
        
        test_duration = (datetime.now() - test_start).total_seconds()
        
        result = {
            'test_duration_seconds': test_duration,
            'rto_target_seconds': self.config.rto_target_seconds,
            'recovery_detected': recovery_detected,
            'rto_met': recovery_detected and test_duration <= self.config.rto_target_seconds,
            'test_timestamp': test_start.isoformat()
        }
        
        logger.info(f"Recovery test completed: {result}")
        return result