"""
Prometheus Alert Rules Configuration for FASE C
Centralized alert definitions for guardrails and observability
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum
import yaml
import time
from datetime import datetime, timedelta


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    """Alert states"""
    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Prometheus alert rule definition"""
    name: str
    query: str
    duration: str
    severity: AlertSeverity
    description: str
    summary: str
    labels: Dict[str, str]
    annotations: Dict[str, str]


class PrometheusAlerts:
    """
    PROMETHEUS ALERTS CONFIGURATION - FASE C
    
    Required Alerts:
    ✅ HighOrderErrorRate: >10% error rate in 5 minutes
    ✅ DrawdownTooHigh: >3% drawdown from peak
    ✅ NoSignals: No signals received for 30 minutes
    """
    
    def __init__(self):
        self.alert_rules = self._define_alert_rules()
        self.alert_manager_config = self._create_alertmanager_config()
    
    def _define_alert_rules(self) -> List[AlertRule]:
        """Define all alert rules for the system"""
        
        rules = [
            # FASE C REQUIRED ALERTS
            
            # 1. HighOrderErrorRate Alert
            AlertRule(
                name="HighOrderErrorRate",
                query="rate(order_errors_total[5m]) / rate(orders_sent_total[5m]) > 0.10",
                duration="2m",
                severity=AlertSeverity.CRITICAL,
                description="Order error rate is above 10% over the last 5 minutes",
                summary="High order error rate detected: {{ $value | humanizePercentage }}",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "execution",
                    "alert_type": "error_rate"
                },
                annotations={
                    "description": "Order error rate has exceeded 10% for more than 2 minutes. Current rate: {{ $value | humanizePercentage }}",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/high-order-error-rate",
                    "action": "Check exchange connectivity and order validation logic"
                }
            ),
            
            # 2. DrawdownTooHigh Alert
            AlertRule(
                name="DrawdownTooHigh", 
                query="portfolio_drawdown_pct > 3.0",
                duration="1m",
                severity=AlertSeverity.CRITICAL,
                description="Portfolio drawdown exceeds 3% from peak equity",
                summary="Excessive portfolio drawdown: {{ $value }}%",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "risk_management",
                    "alert_type": "drawdown"
                },
                annotations={
                    "description": "Portfolio drawdown has exceeded 3% threshold. Current drawdown: {{ $value }}%",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/drawdown-too-high",
                    "action": "Review risk management settings and consider kill-switch activation"
                }
            ),
            
            # 3. NoSignals Alert
            AlertRule(
                name="NoSignals",
                query="time() - last_signal_timestamp_seconds > 1800",
                duration="0s",
                severity=AlertSeverity.WARNING,
                description="No trading signals received for 30 minutes",
                summary="Signal timeout: No signals for {{ $value | humanizeDuration }}",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "signal_processing",
                    "alert_type": "timeout"
                },
                annotations={
                    "description": "No trading signals have been received for more than 30 minutes",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/no-signals",
                    "action": "Check data pipeline and signal generation agents"
                }
            ),
            
            # ADDITIONAL SYSTEM ALERTS
            
            # 4. KillSwitchTriggered Alert
            AlertRule(
                name="KillSwitchTriggered",
                query="increase(kill_switch_triggers_total[1m]) > 0",
                duration="0s",
                severity=AlertSeverity.EMERGENCY,
                description="Emergency kill switch has been triggered",
                summary="EMERGENCY: Kill switch activated",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "risk_management",
                    "alert_type": "emergency"
                },
                annotations={
                    "description": "The emergency kill switch has been triggered due to critical risk violations",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/kill-switch",
                    "action": "IMMEDIATE ACTION REQUIRED: Review risk violations and manually reset if safe"
                }
            ),
            
            # 5. HighRiskScore Alert
            AlertRule(
                name="HighRiskScore",
                query="portfolio_risk_score > 0.7",
                duration="5m",
                severity=AlertSeverity.WARNING,
                description="Portfolio risk score is elevated",
                summary="High risk score: {{ $value }}",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "risk_management",
                    "alert_type": "risk_score"
                },
                annotations={
                    "description": "Portfolio risk score has been above 0.7 for 5 minutes. Current score: {{ $value }}",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/high-risk-score",
                    "action": "Review risk factors and consider reducing exposure"
                }
            ),
            
            # 6. SlippageBudgetExceeded Alert
            AlertRule(
                name="SlippageBudgetExceeded",
                query="sum(estimated_slippage_bps) > 200",
                duration="1m",
                severity=AlertSeverity.WARNING,
                description="Daily slippage budget has been exceeded",
                summary="Slippage budget exceeded: {{ $value }} bps",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "execution",
                    "alert_type": "slippage"
                },
                annotations={
                    "description": "Daily slippage budget of 200 bps has been exceeded. Current usage: {{ $value }} bps",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/slippage-budget",
                    "action": "Consider halting trading or reducing order sizes"
                }
            ),
            
            # 7. HighExecutionLatency Alert
            AlertRule(
                name="HighExecutionLatency",
                query="histogram_quantile(0.95, execution_latency_ms_bucket) > 100",
                duration="3m",
                severity=AlertSeverity.WARNING,
                description="Execution latency p95 is high",
                summary="High execution latency: {{ $value }}ms p95",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "execution",
                    "alert_type": "latency"
                },
                annotations={
                    "description": "95th percentile execution latency is above 100ms for 3 minutes",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/high-latency",
                    "action": "Check system performance and network connectivity"
                }
            ),
            
            # 8. LowDataQuality Alert
            AlertRule(
                name="LowDataQuality",
                query="min(data_quality_score) < 0.8",
                duration="2m",
                severity=AlertSeverity.WARNING,
                description="Data quality score is below threshold",
                summary="Low data quality detected: {{ $value }}",
                labels={
                    "service": "cryptosmarttrader",
                    "component": "data_pipeline",
                    "alert_type": "data_quality"
                },
                annotations={
                    "description": "Data quality score has dropped below 0.8 threshold",
                    "runbook_url": "https://wiki.cryptosmarttrader.com/alerts/low-data-quality",
                    "action": "Check data sources and validation pipelines"
                }
            )
        ]
        
        return rules
    
    def _create_alertmanager_config(self) -> Dict[str, Any]:
        """Create AlertManager configuration"""
        
        return {
            "global": {
                "smtp_smarthost": "localhost:587",
                "smtp_from": "alerts@cryptosmarttrader.com"
            },
            "route": {
                "group_by": ["alertname", "service"],
                "group_wait": "10s",
                "group_interval": "5m",
                "repeat_interval": "12h",
                "receiver": "default-receiver",
                "routes": [
                    {
                        "match": {"severity": "emergency"},
                        "receiver": "emergency-receiver",
                        "group_wait": "0s",
                        "repeat_interval": "5m"
                    },
                    {
                        "match": {"severity": "critical"},
                        "receiver": "critical-receiver",
                        "group_wait": "30s",
                        "repeat_interval": "1h"
                    },
                    {
                        "match": {"severity": "warning"},
                        "receiver": "warning-receiver",
                        "group_wait": "2m",
                        "repeat_interval": "6h"
                    }
                ]
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "email_configs": [
                        {
                            "to": "team@cryptosmarttrader.com",
                            "subject": "CryptoSmartTrader Alert: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ]
                },
                {
                    "name": "emergency-receiver",
                    "email_configs": [
                        {
                            "to": "emergency@cryptosmarttrader.com",
                            "subject": "🚨 EMERGENCY: {{ .GroupLabels.alertname }}",
                            "body": "IMMEDIATE ACTION REQUIRED:\n{{ range .Alerts }}{{ .Annotations.description }}\nAction: {{ .Annotations.action }}{{ end }}"
                        }
                    ],
                    "webhook_configs": [
                        {
                            "url": "http://localhost:9093/webhook",
                            "title": "Emergency Alert: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ]
                },
                {
                    "name": "critical-receiver",
                    "email_configs": [
                        {
                            "to": "critical@cryptosmarttrader.com",
                            "subject": "🔴 CRITICAL: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}\nAction: {{ .Annotations.action }}{{ end }}"
                        }
                    ]
                },
                {
                    "name": "warning-receiver",
                    "email_configs": [
                        {
                            "to": "monitoring@cryptosmarttrader.com",
                            "subject": "⚠️ WARNING: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ]
                }
            ]
        }
    
    def generate_prometheus_rules_yaml(self) -> str:
        """Generate Prometheus rules YAML configuration"""
        
        rules_dict = {
            "groups": [
                {
                    "name": "cryptosmarttrader.rules",
                    "interval": "30s",
                    "rules": []
                }
            ]
        }
        
        for rule in self.alert_rules:
            rule_dict = {
                "alert": rule.name,
                "expr": rule.query,
                "for": rule.duration,
                "labels": {
                    **rule.labels,
                    "severity": rule.severity.value
                },
                "annotations": {
                    **rule.annotations,
                    "summary": rule.summary,
                    "description": rule.description
                }
            }
            rules_dict["groups"][0]["rules"].append(rule_dict)
        
        return yaml.dump(rules_dict, default_flow_style=False)
    
    def generate_alertmanager_yaml(self) -> str:
        """Generate AlertManager YAML configuration"""
        return yaml.dump(self.alert_manager_config, default_flow_style=False)
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status (would integrate with AlertManager API)"""
        # This would typically query AlertManager API for active alerts
        # For now, return structure for integration
        
        return {
            "active_alerts": [],
            "alert_rules_total": len(self.alert_rules),
            "critical_alerts": 0,
            "warning_alerts": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def save_configurations(self, base_path: str = "./config/monitoring/"):
        """Save Prometheus and AlertManager configurations to files"""
        import os
        
        os.makedirs(base_path, exist_ok=True)
        
        # Save Prometheus rules
        with open(f"{base_path}/prometheus_rules.yml", "w") as f:
            f.write(self.generate_prometheus_rules_yaml())
        
        # Save AlertManager config
        with open(f"{base_path}/alertmanager.yml", "w") as f:
            f.write(self.generate_alertmanager_yaml())
        
        return {
            "prometheus_rules": f"{base_path}/prometheus_rules.yml",
            "alertmanager_config": f"{base_path}/alertmanager.yml",
            "rules_count": len(self.alert_rules)
        }


def create_prometheus_alerts() -> PrometheusAlerts:
    """Factory function to create configured alerts"""
    return PrometheusAlerts()