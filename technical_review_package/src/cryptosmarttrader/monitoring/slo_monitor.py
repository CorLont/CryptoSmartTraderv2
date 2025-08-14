"""
SLO Monitor - FASE D
Service Level Objective monitoring voor uptime, alert-to-ack, tracking-error
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from ..core.structured_logger import get_logger


class SLOStatus(Enum):
    """SLO status states."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    BREACHED = "breached"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """Service Level Objective definition."""
    
    name: str
    description: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    measurement_window_hours: int
    unit: str
    higher_is_better: bool = True  # True for uptime, False for latency/errors


@dataclass
class SLOMeasurement:
    """SLO measurement data point."""
    
    timestamp: datetime
    slo_name: str
    value: float
    status: SLOStatus
    breach_duration_minutes: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLOReport:
    """SLO compliance report."""
    
    period_start: datetime
    period_end: datetime
    slo_name: str
    target_value: float
    actual_value: float
    compliance_percentage: float
    status: SLOStatus
    breach_count: int
    total_breach_duration_minutes: float
    measurements: List[SLOMeasurement] = field(default_factory=list)


class SLOMonitor:
    """
    ENTERPRISE SLO MONITORING SYSTEM - FASE D
    
    Monitors key SLOs:
    - Uptime: 99.5% availability target
    - P95 Latency: <1s response time
    - Alert-to-Ack: <15min response time
    - Tracking Error: <20bps deviation
    - Error Rate: <1% failure rate
    """
    
    def __init__(self, data_dir: str = "data/slo"):
        """Initialize SLO monitor."""
        self.logger = get_logger("slo_monitor")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define standard SLOs
        self.slo_definitions = self._create_standard_slos()
        
        # State tracking
        self.measurements: Dict[str, List[SLOMeasurement]] = {}
        self.current_breaches: Dict[str, datetime] = {}  # SLO name -> breach start time
        
        # Load historical data
        self._load_slo_history()
        
        self.logger.info("SLO Monitor initialized with enterprise targets")
    
    def _create_standard_slos(self) -> Dict[str, SLODefinition]:
        """Create standard SLO definitions for Fase D."""
        
        slos = {
            "uptime": SLODefinition(
                name="uptime",
                description="System uptime percentage",
                target_value=99.5,          # 99.5% uptime
                warning_threshold=99.0,     # Warning at 99%
                critical_threshold=98.5,    # Critical at 98.5%
                measurement_window_hours=24,
                unit="percent",
                higher_is_better=True
            ),
            "p95_latency": SLODefinition(
                name="p95_latency",
                description="95th percentile API response time",
                target_value=1000.0,        # 1000ms (1s)
                warning_threshold=1500.0,   # Warning at 1.5s
                critical_threshold=2000.0,  # Critical at 2s
                measurement_window_hours=1,
                unit="milliseconds",
                higher_is_better=False
            ),
            "alert_to_ack": SLODefinition(
                name="alert_to_ack",
                description="Alert acknowledgment response time",
                target_value=15.0,          # 15 minutes
                warning_threshold=20.0,     # Warning at 20min
                critical_threshold=30.0,    # Critical at 30min
                measurement_window_hours=24,
                unit="minutes",
                higher_is_better=False
            ),
            "tracking_error": SLODefinition(
                name="tracking_error",
                description="Backtest-live tracking error",
                target_value=20.0,          # 20 basis points
                warning_threshold=30.0,     # Warning at 30bps
                critical_threshold=50.0,    # Critical at 50bps
                measurement_window_hours=24,
                unit="basis_points",
                higher_is_better=False
            ),
            "error_rate": SLODefinition(
                name="error_rate",
                description="System error rate percentage", 
                target_value=1.0,           # 1% error rate
                warning_threshold=2.0,      # Warning at 2%
                critical_threshold=5.0,     # Critical at 5%
                measurement_window_hours=1,
                unit="percent",
                higher_is_better=False
            )
        }
        
        return slos
    
    def _load_slo_history(self):
        """Load historical SLO data."""
        
        history_file = self.data_dir / "slo_history.json"
        
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                    for slo_name, measurements_data in data.get("measurements", {}).items():
                        measurements = []
                        for measurement_data in measurements_data:
                            measurement = SLOMeasurement(
                                timestamp=datetime.fromisoformat(measurement_data["timestamp"]),
                                slo_name=measurement_data["slo_name"],
                                value=measurement_data["value"],
                                status=SLOStatus(measurement_data["status"]),
                                breach_duration_minutes=measurement_data.get("breach_duration_minutes", 0.0),
                                metadata=measurement_data.get("metadata", {})
                            )
                            measurements.append(measurement)
                        
                        self.measurements[slo_name] = measurements
                
                self.logger.info(f"Loaded SLO history for {len(self.measurements)} SLOs")
                
        except Exception as e:
            self.logger.warning(f"Failed to load SLO history: {e}")
    
    def _save_slo_history(self):
        """Save SLO history to disk."""
        
        history_file = self.data_dir / "slo_history.json"
        
        try:
            # Convert measurements to serializable format
            measurements_data = {}
            for slo_name, measurements in self.measurements.items():
                measurements_data[slo_name] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "slo_name": m.slo_name,
                        "value": m.value,
                        "status": m.status.value,
                        "breach_duration_minutes": m.breach_duration_minutes,
                        "metadata": m.metadata
                    }
                    for m in measurements[-1000:]  # Keep last 1000 measurements per SLO
                ]
            
            data = {
                "measurements": measurements_data,
                "current_breaches": {
                    slo_name: breach_time.isoformat() 
                    for slo_name, breach_time in self.current_breaches.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info("SLO history saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save SLO history: {e}")
    
    def record_measurement(
        self, 
        slo_name: str, 
        value: float, 
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record SLO measurement."""
        
        if slo_name not in self.slo_definitions:
            self.logger.warning(f"Unknown SLO: {slo_name}")
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        slo_def = self.slo_definitions[slo_name]
        
        # Determine status
        status = self._calculate_slo_status(slo_def, value)
        
        # Calculate breach duration if applicable
        breach_duration = 0.0
        if status in [SLOStatus.BREACHED, SLOStatus.CRITICAL]:
            if slo_name in self.current_breaches:
                # Ongoing breach
                breach_start = self.current_breaches[slo_name]
                breach_duration = (timestamp - breach_start).total_seconds() / 60.0
            else:
                # New breach
                self.current_breaches[slo_name] = timestamp
                breach_duration = 0.0
        else:
            # No breach or recovered
            if slo_name in self.current_breaches:
                del self.current_breaches[slo_name]
        
        # Create measurement
        measurement = SLOMeasurement(
            timestamp=timestamp,
            slo_name=slo_name,
            value=value,
            status=status,
            breach_duration_minutes=breach_duration,
            metadata=metadata
        )
        
        # Store measurement
        if slo_name not in self.measurements:
            self.measurements[slo_name] = []
        
        self.measurements[slo_name].append(measurement)
        
        # Log significant status changes
        if status in [SLOStatus.BREACHED, SLOStatus.CRITICAL]:
            self.logger.error(
                f"SLO breach: {slo_name} = {value:.2f} {slo_def.unit} "
                f"(target: {slo_def.target_value:.2f})"
            )
        elif status == SLOStatus.WARNING:
            self.logger.warning(
                f"SLO warning: {slo_name} = {value:.2f} {slo_def.unit} "
                f"(target: {slo_def.target_value:.2f})"
            )
        
        # Clean up old measurements (keep last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.measurements[slo_name] = [
            m for m in self.measurements[slo_name] 
            if m.timestamp > cutoff_time
        ]
        
        # Save periodically
        if len(self.measurements[slo_name]) % 100 == 0:
            self._save_slo_history()
    
    def _calculate_slo_status(self, slo_def: SLODefinition, value: float) -> SLOStatus:
        """Calculate SLO status based on value and thresholds."""
        
        if slo_def.higher_is_better:
            # For metrics like uptime where higher is better
            if value >= slo_def.target_value:
                return SLOStatus.HEALTHY
            elif value >= slo_def.warning_threshold:
                return SLOStatus.WARNING
            elif value >= slo_def.critical_threshold:
                return SLOStatus.BREACHED
            else:
                return SLOStatus.CRITICAL
        else:
            # For metrics like latency where lower is better
            if value <= slo_def.target_value:
                return SLOStatus.HEALTHY
            elif value <= slo_def.warning_threshold:
                return SLOStatus.WARNING
            elif value <= slo_def.critical_threshold:
                return SLOStatus.BREACHED
            else:
                return SLOStatus.CRITICAL
    
    def get_slo_status(self, slo_name: str) -> Optional[SLOStatus]:
        """Get current SLO status."""
        
        if slo_name not in self.measurements or not self.measurements[slo_name]:
            return None
        
        # Return status from most recent measurement
        latest_measurement = self.measurements[slo_name][-1]
        return latest_measurement.status
    
    def calculate_slo_compliance(
        self, 
        slo_name: str, 
        period_hours: int = 24
    ) -> Optional[SLOReport]:
        """Calculate SLO compliance over specified period."""
        
        if slo_name not in self.slo_definitions or slo_name not in self.measurements:
            return None
        
        slo_def = self.slo_definitions[slo_name]
        measurements = self.measurements[slo_name]
        
        if not measurements:
            return None
        
        # Filter measurements to period
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)
        
        period_measurements = [
            m for m in measurements 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not period_measurements:
            return None
        
        # Calculate compliance metrics
        total_measurements = len(period_measurements)
        breach_measurements = [
            m for m in period_measurements 
            if m.status in [SLOStatus.BREACHED, SLOStatus.CRITICAL]
        ]
        
        breach_count = len(breach_measurements)
        compliance_percentage = ((total_measurements - breach_count) / total_measurements) * 100
        
        # Calculate total breach duration
        total_breach_duration = sum(m.breach_duration_minutes for m in breach_measurements)
        
        # Calculate actual value (average for most metrics)
        if slo_name == "uptime":
            # Uptime calculated as percentage of time in healthy state
            healthy_measurements = total_measurements - breach_count
            actual_value = (healthy_measurements / total_measurements) * 100
        else:
            # For other metrics, use average value
            actual_value = sum(m.value for m in period_measurements) / total_measurements
        
        # Determine overall status
        current_status = period_measurements[-1].status
        
        return SLOReport(
            period_start=start_time,
            period_end=end_time,
            slo_name=slo_name,
            target_value=slo_def.target_value,
            actual_value=actual_value,
            compliance_percentage=compliance_percentage,
            status=current_status,
            breach_count=breach_count,
            total_breach_duration_minutes=total_breach_duration,
            measurements=period_measurements
        )
    
    def get_all_slo_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status summary for all SLOs."""
        
        summary = {}
        
        for slo_name, slo_def in self.slo_definitions.items():
            # Get current status
            current_status = self.get_slo_status(slo_name)
            
            # Get 24h compliance
            compliance_report = self.calculate_slo_compliance(slo_name, 24)
            
            # Check if currently breached
            is_breached = slo_name in self.current_breaches
            breach_duration = 0.0
            
            if is_breached:
                breach_start = self.current_breaches[slo_name]
                breach_duration = (datetime.now() - breach_start).total_seconds() / 60.0
            
            summary[slo_name] = {
                "definition": asdict(slo_def),
                "current_status": current_status.value if current_status else "unknown",
                "is_breached": is_breached,
                "breach_duration_minutes": breach_duration,
                "24h_compliance": {
                    "compliance_percentage": compliance_report.compliance_percentage if compliance_report else 0.0,
                    "actual_value": compliance_report.actual_value if compliance_report else 0.0,
                    "breach_count": compliance_report.breach_count if compliance_report else 0,
                    "total_breach_duration_minutes": compliance_report.total_breach_duration_minutes if compliance_report else 0.0
                },
                "measurement_count": len(self.measurements.get(slo_name, []))
            }
        
        return summary
    
    def generate_slo_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for SLO dashboard."""
        
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "overall_health": self._calculate_overall_health(),
            "slo_summary": self.get_all_slo_status(),
            "active_breaches": len(self.current_breaches),
            "critical_slos": [],
            "trending_slos": {}
        }
        
        # Identify critical SLOs
        for slo_name, status_data in dashboard_data["slo_summary"].items():
            if status_data["current_status"] in ["breached", "critical"]:
                dashboard_data["critical_slos"].append({
                    "name": slo_name,
                    "status": status_data["current_status"],
                    "breach_duration_minutes": status_data["breach_duration_minutes"]
                })
        
        # Calculate trends (last 7 days vs previous 7 days)
        for slo_name in self.slo_definitions.keys():
            recent_report = self.calculate_slo_compliance(slo_name, 168)  # 7 days
            if recent_report:
                dashboard_data["trending_slos"][slo_name] = {
                    "compliance_7d": recent_report.compliance_percentage,
                    "breach_count_7d": recent_report.breach_count
                }
        
        return dashboard_data
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health based on all SLOs."""
        
        critical_count = 0
        warning_count = 0
        total_slos = len(self.slo_definitions)
        
        for slo_name in self.slo_definitions.keys():
            status = self.get_slo_status(slo_name)
            
            if status in [SLOStatus.BREACHED, SLOStatus.CRITICAL]:
                critical_count += 1
            elif status == SLOStatus.WARNING:
                warning_count += 1
        
        if critical_count > 0:
            return "CRITICAL"
        elif warning_count > total_slos * 0.5:  # More than 50% in warning
            return "DEGRADED"
        elif warning_count > 0:
            return "WARNING"
        else:
            return "HEALTHY"
    
    def simulate_slo_measurements(self):
        """Simulate SLO measurements for testing."""
        
        import random
        
        # Simulate uptime (usually high)
        uptime = 99.2 + random.uniform(0, 1.0)
        self.record_measurement("uptime", uptime)
        
        # Simulate P95 latency (usually good, occasional spikes)
        if random.random() < 0.05:  # 5% chance of spike
            latency = random.uniform(1500, 3000)
        else:
            latency = random.uniform(200, 800)
        self.record_measurement("p95_latency", latency)
        
        # Simulate alert response time (usually good)
        alert_response = 8 + random.uniform(0, 20)
        self.record_measurement("alert_to_ack", alert_response)
        
        # Simulate tracking error (usually within bounds)
        tracking_error = 15 + random.uniform(-10, 25)
        self.record_measurement("tracking_error", tracking_error)
        
        # Simulate error rate (usually low)
        if random.random() < 0.1:  # 10% chance of elevated errors
            error_rate = random.uniform(2, 8)
        else:
            error_rate = random.uniform(0.1, 1.5)
        self.record_measurement("error_rate", error_rate)


def create_slo_monitor(data_dir: str = "data/slo") -> SLOMonitor:
    """Factory function to create SLO monitor."""
    return SLOMonitor(data_dir)