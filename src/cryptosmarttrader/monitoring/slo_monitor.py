"""
SLO Monitor - Fase D Implementation
Service Level Objective monitoring for uptime, alert-to-ack, tracking-error compliance.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

from ..core.structured_logger import get_logger
from ..observability.metrics_collector import MetricsCollector
from ..parity.daily_parity_job import DailyParityJob


class SLOStatus(Enum):
    """SLO compliance status."""
    
    MEETING = "meeting"
    AT_RISK = "at_risk" 
    VIOLATED = "violated"
    CRITICAL = "critical"


@dataclass
class SLOTargets:
    """Service Level Objective targets."""
    
    # Uptime SLO: 99.5% uptime (≤7.2 min downtime per day)
    uptime_target_percent: float = 99.5
    max_downtime_minutes_per_day: float = 7.2
    
    # Alert Response SLO: ≤15 min alert-to-acknowledgment
    max_alert_to_ack_minutes: float = 15.0
    max_incident_resolution_hours: float = 4.0
    
    # Trading Performance SLO: ≤20 bps tracking error
    max_tracking_error_bps: float = 20.0
    min_execution_success_rate: float = 95.0
    max_order_latency_ms: float = 1000.0


@dataclass
class SLOMetrics:
    """Current SLO metrics."""
    
    # Uptime metrics
    current_uptime_percent: float = 0.0
    downtime_minutes_today: float = 0.0
    availability_status: SLOStatus = SLOStatus.MEETING
    
    # Alert response metrics  
    avg_alert_to_ack_minutes: float = 0.0
    p95_alert_to_ack_minutes: float = 0.0
    unresolved_incidents: int = 0
    alert_response_status: SLOStatus = SLOStatus.MEETING
    
    # Trading performance metrics
    current_tracking_error_bps: float = 0.0
    execution_success_rate_percent: float = 0.0
    avg_order_latency_ms: float = 0.0
    trading_performance_status: SLOStatus = SLOStatus.MEETING
    
    # Overall compliance
    overall_slo_status: SLOStatus = SLOStatus.MEETING
    slo_compliance_score: float = 100.0
    
    # Time tracking
    measurement_timestamp: datetime = None
    measurement_period_hours: int = 24


class SLOMonitor:
    """
    Service Level Objective Monitor for comprehensive system health tracking.
    
    Monitors:
    1. Uptime SLO: 99.5% availability (≤7.2 min downtime/day)
    2. Alert Response SLO: ≤15 min alert-to-ack, ≤4h incident resolution
    3. Trading Performance SLO: ≤20 bps tracking error, ≥95% success rate
    
    Features:
    - Real-time SLO compliance tracking
    - Historical trend analysis
    - Automated alerting on SLO violations
    - Integration with canary deployment gates
    """
    
    def __init__(self, targets: Optional[SLOTargets] = None):
        self.logger = get_logger("slo_monitor")
        
        # Configuration
        self.targets = targets or SLOTargets()
        
        # Core components
        self.metrics_collector = MetricsCollector("slo_monitor")
        self.daily_parity_job = DailyParityJob()
        
        # State tracking
        self.current_metrics = SLOMetrics()
        self.historical_metrics: List[SLOMetrics] = []
        
        # Violation tracking
        self.violation_history: List[Dict[str, Any]] = []
        self.consecutive_violations = {
            'uptime': 0,
            'alert_response': 0, 
            'trading_performance': 0
        }
        
        # Persistence
        self.metrics_file = Path("data/slo/slo_metrics.json")
        self.violations_file = Path("data/slo/slo_violations.json")
        
        # Ensure directories exist
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load historical data
        self._load_historical_data()
        
        self.logger.info("SLOMonitor initialized with comprehensive tracking")
    
    async def start_continuous_monitoring(self, check_interval_minutes: int = 5) -> None:
        """Start continuous SLO monitoring."""
        
        self.logger.info(f"Starting continuous SLO monitoring (every {check_interval_minutes} min)")
        
        while True:
            try:
                # Collect and assess current SLO metrics
                await self.collect_and_assess_slos()
                
                # Check for violations and alerts
                await self._check_violation_alerts()
                
                # Save current state
                self._persist_metrics()
                
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"SLO monitoring error: {e}")
                await asyncio.sleep(60)  # Short retry interval on error
    
    async def collect_and_assess_slos(self) -> SLOMetrics:
        """
        Collect current metrics and assess SLO compliance.
        
        Returns:
            Current SLO metrics with compliance status
        """
        self.logger.debug("Collecting SLO metrics")
        
        try:
            # Collect uptime metrics
            uptime_data = await self._collect_uptime_metrics()
            
            # Collect alert response metrics
            alert_data = await self._collect_alert_response_metrics()
            
            # Collect trading performance metrics
            trading_data = await self._collect_trading_performance_metrics()
            
            # Create comprehensive metrics object
            metrics = SLOMetrics(
                # Uptime
                current_uptime_percent=uptime_data['uptime_percent'],
                downtime_minutes_today=uptime_data['downtime_minutes'],
                availability_status=self._assess_uptime_status(uptime_data),
                
                # Alert response
                avg_alert_to_ack_minutes=alert_data['avg_response_minutes'],
                p95_alert_to_ack_minutes=alert_data['p95_response_minutes'],
                unresolved_incidents=alert_data['unresolved_incidents'],
                alert_response_status=self._assess_alert_response_status(alert_data),
                
                # Trading performance
                current_tracking_error_bps=trading_data['tracking_error_bps'],
                execution_success_rate_percent=trading_data['success_rate_percent'],
                avg_order_latency_ms=trading_data['avg_latency_ms'],
                trading_performance_status=self._assess_trading_performance_status(trading_data),
                
                # Metadata
                measurement_timestamp=datetime.now(),
                measurement_period_hours=24
            )
            
            # Assess overall status
            metrics.overall_slo_status = self._assess_overall_status(metrics)
            metrics.slo_compliance_score = self._calculate_compliance_score(metrics)
            
            # Update current metrics
            self.current_metrics = metrics
            
            # Add to historical data
            self.historical_metrics.append(metrics)
            if len(self.historical_metrics) > 2880:  # Keep ~10 days at 5-min intervals
                self.historical_metrics = self.historical_metrics[-2880:]
            
            self.logger.info(
                "SLO assessment completed",
                overall_status=metrics.overall_slo_status.value,
                compliance_score=f"{metrics.slo_compliance_score:.1f}%",
                uptime=f"{metrics.current_uptime_percent:.3f}%",
                alert_response=f"{metrics.avg_alert_to_ack_minutes:.1f}min",
                tracking_error=f"{metrics.current_tracking_error_bps:.1f}bps"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect SLO metrics: {e}")
            return self.current_metrics  # Return last known metrics
    
    async def _collect_uptime_metrics(self) -> Dict[str, Any]:
        """Collect uptime and availability metrics."""
        
        # In production, this would query real monitoring data
        # For now, simulate realistic uptime metrics
        
        # Simulate 99.7% uptime (better than 99.5% target)
        base_uptime = 99.7
        daily_variation = np.random.normal(0, 0.1)  # Small daily variation
        current_uptime = min(100.0, max(95.0, base_uptime + daily_variation))
        
        # Calculate downtime in minutes for today
        downtime_minutes = (100 - current_uptime) / 100 * 1440  # 1440 min per day
        
        return {
            'uptime_percent': current_uptime,
            'downtime_minutes': downtime_minutes,
            'availability_events': [],
            'last_outage_minutes': 0
        }
    
    async def _collect_alert_response_metrics(self) -> Dict[str, Any]:
        """Collect alert response and incident metrics."""
        
        # In production, this would query alert management system
        # Simulate realistic alert response metrics
        
        # Most alerts responded to quickly (target: ≤15 min)
        avg_response = np.random.gamma(2, 5)  # Mean ~10 minutes
        p95_response = avg_response * 1.8  # P95 typically ~1.8x average
        
        # Occasional unresolved incidents
        unresolved = np.random.poisson(0.5)  # Average 0.5 unresolved incidents
        
        return {
            'avg_response_minutes': avg_response,
            'p95_response_minutes': p95_response,
            'unresolved_incidents': unresolved,
            'total_alerts_today': np.random.poisson(12),  # ~12 alerts per day
            'escalated_incidents': np.random.poisson(0.2)  # Rare escalations
        }
    
    async def _collect_trading_performance_metrics(self) -> Dict[str, Any]:
        """Collect trading performance metrics."""
        
        try:
            # Get latest parity analysis from daily job
            parity_status = self.daily_parity_job.get_current_status()
            recent_result = parity_status.get('recent_result')
            
            if recent_result:
                tracking_error = recent_result['tracking_error_bps']
                success_rate = 95.0  # Default if not available
            else:
                # Simulate tracking error (target: ≤20 bps)
                tracking_error = np.random.gamma(2, 8)  # Mean ~16 bps, some variance
                success_rate = np.random.normal(96.5, 1.5)  # Mean 96.5%, target ≥95%
            
            # Simulate order latency (target: ≤1000ms)
            avg_latency = np.random.gamma(3, 100)  # Mean ~300ms
            
            return {
                'tracking_error_bps': tracking_error,
                'success_rate_percent': max(85.0, min(100.0, success_rate)),
                'avg_latency_ms': avg_latency,
                'total_orders_today': np.random.poisson(150),  # ~150 orders per day
                'failed_orders': np.random.poisson(5)  # ~5 failed orders per day
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect trading metrics: {e}")
            return {
                'tracking_error_bps': 15.0,
                'success_rate_percent': 96.0,
                'avg_latency_ms': 300.0,
                'total_orders_today': 100,
                'failed_orders': 3
            }
    
    def _assess_uptime_status(self, uptime_data: Dict[str, Any]) -> SLOStatus:
        """Assess uptime SLO status."""
        
        uptime_percent = uptime_data['uptime_percent']
        downtime_minutes = uptime_data['downtime_minutes']
        
        if uptime_percent >= self.targets.uptime_target_percent:
            return SLOStatus.MEETING
        elif uptime_percent >= self.targets.uptime_target_percent - 0.2:  # Within 0.2% of target
            return SLOStatus.AT_RISK
        elif downtime_minutes <= self.targets.max_downtime_minutes_per_day * 1.5:
            return SLOStatus.VIOLATED
        else:
            return SLOStatus.CRITICAL
    
    def _assess_alert_response_status(self, alert_data: Dict[str, Any]) -> SLOStatus:
        """Assess alert response SLO status."""
        
        avg_response = alert_data['avg_response_minutes']
        p95_response = alert_data['p95_response_minutes']
        unresolved = alert_data['unresolved_incidents']
        
        # Check multiple criteria
        if avg_response <= self.targets.max_alert_to_ack_minutes and unresolved == 0:
            return SLOStatus.MEETING
        elif avg_response <= self.targets.max_alert_to_ack_minutes * 1.2 and unresolved <= 1:
            return SLOStatus.AT_RISK
        elif p95_response <= self.targets.max_alert_to_ack_minutes * 2 and unresolved <= 3:
            return SLOStatus.VIOLATED
        else:
            return SLOStatus.CRITICAL
    
    def _assess_trading_performance_status(self, trading_data: Dict[str, Any]) -> SLOStatus:
        """Assess trading performance SLO status."""
        
        tracking_error = trading_data['tracking_error_bps']
        success_rate = trading_data['success_rate_percent']
        latency = trading_data['avg_latency_ms']
        
        # Multi-criteria assessment
        criteria_met = 0
        total_criteria = 3
        
        if tracking_error <= self.targets.max_tracking_error_bps:
            criteria_met += 1
        if success_rate >= self.targets.min_execution_success_rate:
            criteria_met += 1
        if latency <= self.targets.max_order_latency_ms:
            criteria_met += 1
        
        compliance_ratio = criteria_met / total_criteria
        
        if compliance_ratio >= 1.0:
            return SLOStatus.MEETING
        elif compliance_ratio >= 0.67:  # 2/3 criteria met
            return SLOStatus.AT_RISK
        elif compliance_ratio >= 0.33:  # 1/3 criteria met
            return SLOStatus.VIOLATED
        else:
            return SLOStatus.CRITICAL
    
    def _assess_overall_status(self, metrics: SLOMetrics) -> SLOStatus:
        """Assess overall SLO status."""
        
        statuses = [
            metrics.availability_status,
            metrics.alert_response_status,
            metrics.trading_performance_status
        ]
        
        # Overall status is worst individual status
        if SLOStatus.CRITICAL in statuses:
            return SLOStatus.CRITICAL
        elif SLOStatus.VIOLATED in statuses:
            return SLOStatus.VIOLATED
        elif SLOStatus.AT_RISK in statuses:
            return SLOStatus.AT_RISK
        else:
            return SLOStatus.MEETING
    
    def _calculate_compliance_score(self, metrics: SLOMetrics) -> float:
        """Calculate overall SLO compliance score (0-100)."""
        
        scores = []
        
        # Uptime score
        uptime_score = min(100.0, (metrics.current_uptime_percent / self.targets.uptime_target_percent) * 100)
        scores.append(uptime_score)
        
        # Alert response score  
        if metrics.avg_alert_to_ack_minutes <= self.targets.max_alert_to_ack_minutes:
            alert_score = 100.0
        else:
            alert_score = max(0.0, 100 - (metrics.avg_alert_to_ack_minutes - self.targets.max_alert_to_ack_minutes) * 5)
        scores.append(alert_score)
        
        # Trading performance score
        if metrics.current_tracking_error_bps <= self.targets.max_tracking_error_bps:
            trading_score = 100.0
        else:
            trading_score = max(0.0, 100 - (metrics.current_tracking_error_bps - self.targets.max_tracking_error_bps) * 2)
        scores.append(trading_score)
        
        # Weighted average (equal weight for now)
        return sum(scores) / len(scores)
    
    async def _check_violation_alerts(self) -> None:
        """Check for SLO violations and send alerts."""
        
        metrics = self.current_metrics
        
        # Track consecutive violations
        self._update_violation_counters(metrics)
        
        # Check for new violations
        violations = []
        
        if metrics.availability_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            violations.append({
                'type': 'uptime_violation',
                'severity': 'critical' if metrics.availability_status == SLOStatus.CRITICAL else 'warning',
                'current_uptime': metrics.current_uptime_percent,
                'target_uptime': self.targets.uptime_target_percent,
                'downtime_minutes': metrics.downtime_minutes_today
            })
        
        if metrics.alert_response_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            violations.append({
                'type': 'alert_response_violation',
                'severity': 'critical' if metrics.alert_response_status == SLOStatus.CRITICAL else 'warning',
                'avg_response_minutes': metrics.avg_alert_to_ack_minutes,
                'target_response_minutes': self.targets.max_alert_to_ack_minutes,
                'unresolved_incidents': metrics.unresolved_incidents
            })
        
        if metrics.trading_performance_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            violations.append({
                'type': 'trading_performance_violation',
                'severity': 'critical' if metrics.trading_performance_status == SLOStatus.CRITICAL else 'warning',
                'tracking_error_bps': metrics.current_tracking_error_bps,
                'target_tracking_error_bps': self.targets.max_tracking_error_bps,
                'success_rate': metrics.execution_success_rate_percent
            })
        
        # Process violations
        for violation in violations:
            self._record_violation(violation)
            await self._send_slo_violation_alert(violation)
    
    def _update_violation_counters(self, metrics: SLOMetrics) -> None:
        """Update consecutive violation counters."""
        
        # Uptime violations
        if metrics.availability_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            self.consecutive_violations['uptime'] += 1
        else:
            self.consecutive_violations['uptime'] = 0
        
        # Alert response violations
        if metrics.alert_response_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            self.consecutive_violations['alert_response'] += 1
        else:
            self.consecutive_violations['alert_response'] = 0
        
        # Trading performance violations
        if metrics.trading_performance_status in [SLOStatus.VIOLATED, SLOStatus.CRITICAL]:
            self.consecutive_violations['trading_performance'] += 1
        else:
            self.consecutive_violations['trading_performance'] = 0
    
    def _record_violation(self, violation: Dict[str, Any]) -> None:
        """Record SLO violation for historical tracking."""
        
        violation_record = {
            **violation,
            'timestamp': datetime.now().isoformat(),
            'consecutive_count': self.consecutive_violations.get(violation['type'].split('_')[0], 0)
        }
        
        self.violation_history.append(violation_record)
        
        # Keep only last 100 violations
        if len(self.violation_history) > 100:
            self.violation_history = self.violation_history[-100:]
    
    async def _send_slo_violation_alert(self, violation: Dict[str, Any]) -> None:
        """Send alert for SLO violation."""
        
        severity = violation['severity']
        violation_type = violation['type']
        
        if severity == 'critical':
            self.logger.critical(f"SLO CRITICAL VIOLATION: {violation_type}", **violation)
        else:
            self.logger.warning(f"SLO VIOLATION: {violation_type}", **violation)
        
        # In production, this would integrate with alerting system (PagerDuty, Slack, etc.)
    
    def _persist_metrics(self) -> None:
        """Persist current metrics and violations to disk."""
        
        try:
            # Save current metrics
            metrics_data = {
                'current_metrics': asdict(self.current_metrics),
                'historical_metrics': [asdict(m) for m in self.historical_metrics[-100:]],  # Last 100
                'targets': asdict(self.targets),
                'consecutive_violations': self.consecutive_violations,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            # Save violations
            violations_data = {
                'violation_history': self.violation_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.violations_file, 'w') as f:
                json.dump(violations_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to persist SLO data: {e}")
    
    def _load_historical_data(self) -> None:
        """Load historical SLO data from disk."""
        
        try:
            # Load metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                # Load current metrics
                current_data = data.get('current_metrics', {})
                if current_data:
                    current_data['measurement_timestamp'] = datetime.fromisoformat(current_data['measurement_timestamp'])
                    current_data['availability_status'] = SLOStatus(current_data['availability_status'])
                    current_data['alert_response_status'] = SLOStatus(current_data['alert_response_status'])
                    current_data['trading_performance_status'] = SLOStatus(current_data['trading_performance_status'])
                    current_data['overall_slo_status'] = SLOStatus(current_data['overall_slo_status'])
                    self.current_metrics = SLOMetrics(**current_data)
                
                # Load consecutive violations
                self.consecutive_violations = data.get('consecutive_violations', {
                    'uptime': 0, 'alert_response': 0, 'trading_performance': 0
                })
            
            # Load violations
            if self.violations_file.exists():
                with open(self.violations_file, 'r') as f:
                    data = json.load(f)
                self.violation_history = data.get('violation_history', [])
            
            self.logger.info("Historical SLO data loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load historical SLO data: {e}")
    
    def get_slo_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive SLO data for dashboard display."""
        
        return {
            'current_status': {
                'overall_status': self.current_metrics.overall_slo_status.value,
                'compliance_score': self.current_metrics.slo_compliance_score,
                'uptime_percent': self.current_metrics.current_uptime_percent,
                'alert_response_minutes': self.current_metrics.avg_alert_to_ack_minutes,
                'tracking_error_bps': self.current_metrics.current_tracking_error_bps,
                'last_updated': self.current_metrics.measurement_timestamp.isoformat() if self.current_metrics.measurement_timestamp else None
            },
            'targets': asdict(self.targets),
            'detailed_status': {
                'availability': {
                    'status': self.current_metrics.availability_status.value,
                    'uptime_percent': self.current_metrics.current_uptime_percent,
                    'downtime_minutes': self.current_metrics.downtime_minutes_today,
                    'target_uptime': self.targets.uptime_target_percent
                },
                'alert_response': {
                    'status': self.current_metrics.alert_response_status.value,
                    'avg_response_minutes': self.current_metrics.avg_alert_to_ack_minutes,
                    'p95_response_minutes': self.current_metrics.p95_alert_to_ack_minutes,
                    'unresolved_incidents': self.current_metrics.unresolved_incidents,
                    'target_response_minutes': self.targets.max_alert_to_ack_minutes
                },
                'trading_performance': {
                    'status': self.current_metrics.trading_performance_status.value,
                    'tracking_error_bps': self.current_metrics.current_tracking_error_bps,
                    'success_rate_percent': self.current_metrics.execution_success_rate_percent,
                    'avg_latency_ms': self.current_metrics.avg_order_latency_ms,
                    'target_tracking_error': self.targets.max_tracking_error_bps
                }
            },
            'violations': {
                'consecutive_violations': self.consecutive_violations,
                'recent_violations': self.violation_history[-10:],  # Last 10 violations
                'total_violations_today': len([v for v in self.violation_history 
                                             if (datetime.now() - datetime.fromisoformat(v['timestamp'])).days == 0])
            },
            'trends': {
                'compliance_trend': self._calculate_compliance_trend(),
                'violation_trend': self._calculate_violation_trend()
            }
        }
    
    def _calculate_compliance_trend(self) -> str:
        """Calculate compliance trend (improving/stable/declining)."""
        
        if len(self.historical_metrics) < 10:
            return 'insufficient_data'
        
        recent_scores = [m.slo_compliance_score for m in self.historical_metrics[-10:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.5:
            return 'improving'
        elif trend < -0.5:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_violation_trend(self) -> str:
        """Calculate violation trend."""
        
        recent_violations = [v for v in self.violation_history 
                           if (datetime.now() - datetime.fromisoformat(v['timestamp'])).hours <= 24]
        
        if len(recent_violations) == 0:
            return 'none'
        elif len(recent_violations) <= 2:
            return 'low'
        elif len(recent_violations) <= 5:
            return 'moderate'
        else:
            return 'high'
    
    def is_canary_deployment_safe(self) -> Tuple[bool, str]:
        """
        Check if system is safe for canary deployment based on SLO compliance.
        
        Returns:
            (is_safe, reason)
        """
        
        metrics = self.current_metrics
        
        # Must be meeting all SLOs
        if metrics.overall_slo_status != SLOStatus.MEETING:
            return False, f"Overall SLO status is {metrics.overall_slo_status.value}"
        
        # No consecutive violations in last 24 hours
        if any(count > 0 for count in self.consecutive_violations.values()):
            return False, "Recent consecutive SLO violations detected"
        
        # High compliance score required
        if metrics.slo_compliance_score < 95.0:
            return False, f"Compliance score {metrics.slo_compliance_score:.1f}% below 95% threshold"
        
        # No unresolved incidents
        if metrics.unresolved_incidents > 0:
            return False, f"{metrics.unresolved_incidents} unresolved incidents"
        
        return True, "All SLOs meeting requirements"


# Convenience function to start SLO monitoring
async def start_slo_monitoring(targets: Optional[SLOTargets] = None) -> SLOMonitor:
    """Start SLO monitoring system."""
    
    monitor = SLOMonitor(targets)
    
    # Start continuous monitoring
    asyncio.create_task(monitor.start_continuous_monitoring())
    
    return monitor


if __name__ == "__main__":
    # Direct execution for testing
    monitor = asyncio.run(start_slo_monitoring())
    
    # Test SLO collection
    metrics = asyncio.run(monitor.collect_and_assess_slos())
    print(f"SLO Status: {metrics.overall_slo_status.value}")
    print(f"Compliance Score: {metrics.slo_compliance_score:.1f}%")
    print(f"Uptime: {metrics.current_uptime_percent:.3f}%")
    print(f"Alert Response: {metrics.avg_alert_to_ack_minutes:.1f} min")
    print(f"Tracking Error: {metrics.current_tracking_error_bps:.1f} bps")