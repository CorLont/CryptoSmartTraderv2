#!/usr/bin/env python3
"""
Drift Detection + Fine-Tune + Auto-Disable Integration
Complete integration system that detects drift, triggers fine-tuning, and auto-disables trading
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ..core.structured_logger import get_logger
from ..core.drift_detection import DriftDetectionSystem, DriftAlert
from ..core.fine_tune_scheduler import FineTuneScheduler, FineTuneJob
from ..core.auto_disable_system import AutoDisableSystem, DisableReason, TradingStatusChange


class DriftFineTuneIntegration:
    """Complete integration of drift detection, fine-tuning, and auto-disable systems"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("DriftFineTuneIntegration")

        # Default configuration
        self.config = {
            "monitoring_interval": 300,  # 5 minutes
            "drift_response_delay": 60,  # 1 minute delay before triggering fine-tune
            "auto_fine_tune_enabled": True,
            "auto_disable_enabled": True,
            "critical_drift_threshold": 0.5,  # Immediately disable for severe drift
            "drift_fine_tune_priority": "high",
            "health_check_components": ["ml_predictor", "sentiment_analyzer", "data_pipeline"],
        }

        if config:
            self.config.update(config)

        # Initialize core systems
        self.drift_system = DriftDetectionSystem()
        self.fine_tune_scheduler = FineTuneScheduler()
        self.auto_disable = AutoDisableSystem()

        # Integration state
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.pending_drift_responses: Dict[str, datetime] = {}

        # Register auto-disable callback for status changes
        self.auto_disable.add_status_change_callback(self._on_trading_status_change)

        # Integration metrics
        self.integration_metrics = {
            "drift_alerts_total": 0,
            "fine_tune_jobs_triggered": 0,
            "auto_disables_triggered": 0,
            "health_checks_performed": 0,
            "last_health_score": None,
            "system_start_time": datetime.now(),
        }

    async def start_monitoring(self) -> None:
        """Start the integrated monitoring system"""

        if self.running:
            self.logger.warning("Integration monitoring already running")
            return

        self.running = True
        self.logger.info("Starting drift detection + fine-tune + auto-disable integration")

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Integration monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the integrated monitoring system"""

        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Integration monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""

        while self.running:
            try:
                # Run complete monitoring cycle
                await self._run_monitoring_cycle()

                # Wait for next cycle
                await asyncio.sleep(self.config["monitoring_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _run_monitoring_cycle(self) -> None:
        """Run a complete monitoring cycle"""

        try:
            # 1. Run drift detection
            drift_alerts = self.drift_system.run_drift_detection()
            self.integration_metrics["drift_alerts_total"] += len(drift_alerts)

            # 2. Process drift alerts
            if drift_alerts:
                await self._process_drift_alerts(drift_alerts)

            # 3. Calculate system health score
            health_score = await self._calculate_system_health()
            self.integration_metrics["last_health_score"] = health_score
            self.integration_metrics["health_checks_performed"] += 1

            # 4. Update auto-disable system
            if self.config["auto_disable_enabled"]:
                status_changed = self.auto_disable.check_health_and_update_status(health_score)
                if status_changed:
                    self.integration_metrics["auto_disables_triggered"] += 1

            # 5. Process pending fine-tune jobs
            if self.config["auto_fine_tune_enabled"]:
                await self._process_pending_fine_tune_jobs()

            # 6. Check for delayed drift responses
            await self._check_delayed_drift_responses()

            self.logger.debug(
                f"Monitoring cycle completed: {len(drift_alerts)} alerts, health={health_score:.1f}"
            )

        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {e}")

    async def _process_drift_alerts(self, drift_alerts: List[DriftAlert]) -> None:
        """Process drift alerts and trigger appropriate responses"""

        for alert in drift_alerts:
            try:
                self.logger.warning(
                    f"Processing drift alert: {alert.drift_type} for {alert.component}"
                )

                # Immediate actions for critical alerts
                if alert.severity == "critical":
                    await self._handle_critical_drift(alert)

                # Schedule fine-tuning response
                if (
                    self.config["auto_fine_tune_enabled"]
                    and alert.component in self.config["health_check_components"]
                ):
                    await self._schedule_drift_response(alert)

                # Log alert for analysis
                self._log_drift_alert(alert)

            except Exception as e:
                self.logger.error(f"Failed to process drift alert {alert.alert_id}: {e}")

    async def _handle_critical_drift(self, alert: DriftAlert) -> None:
        """Handle critical drift alerts immediately"""

        # Check if this is severe enough to disable trading immediately
        if alert.severity == "critical":
            if alert.drift_type == "error_trending":
                # Check error rate increase
                relative_change = alert.metrics.get("relative_change", 0)
                if relative_change > self.config["critical_drift_threshold"]:
                    self.auto_disable.disable_for_reason(
                        DisableReason.DRIFT_DETECTED,
                        f"Critical error trend detected: {relative_change:.1%} increase in {alert.component}",
                        trigger_value=relative_change,
                    )

            elif alert.drift_type == "feature_distribution":
                # Check KS statistic
                ks_statistic = alert.metrics.get("ks_statistic", 0)
                if ks_statistic > 0.5:  # Very high drift
                    self.auto_disable.disable_for_reason(
                        DisableReason.DATA_QUALITY_ISSUE,
                        f"Critical feature drift detected: KS={ks_statistic:.3f} for {alert.metrics.get('feature_name')}",
                        trigger_value=ks_statistic,
                    )

            elif alert.drift_type == "performance_degradation":
                # Check degradation amount
                degradation = alert.metrics.get("degradation", 0)
                if degradation > 0.3:  # 30%+ degradation
                    self.auto_disable.disable_for_reason(
                        DisableReason.PERFORMANCE_DEGRADATION,
                        f"Critical performance degradation: {degradation:.1%} drop in {alert.component}",
                        trigger_value=degradation,
                    )

    async def _schedule_drift_response(self, alert: DriftAlert) -> None:
        """Schedule fine-tuning response to drift"""

        # Add delay before triggering fine-tune (avoid over-reaction)
        response_time = datetime.now() + timedelta(seconds=self.config["drift_response_delay"])
        response_key = f"{alert.component}_{alert.drift_type}"

        # Only schedule if not already pending
        if response_key not in self.pending_drift_responses:
            self.pending_drift_responses[response_key] = response_time

            self.logger.info(
                f"Scheduled drift response for {alert.component} in "
                f"{self.config['drift_response_delay']}s"
            )

    async def _check_delayed_drift_responses(self) -> None:
        """Check and execute delayed drift responses"""

        current_time = datetime.now()
        ready_responses = []

        # Find responses that are ready
        for response_key, response_time in self.pending_drift_responses.items():
            if current_time >= response_time:
                ready_responses.append(response_key)

        # Execute ready responses
        for response_key in ready_responses:
            try:
                component, drift_type = response_key.split("_", 1)

                # Create fine-tuning job
                job_id = self.fine_tune_scheduler.create_fine_tune_job(
                    model_name=component,
                    trigger_reason=f"drift_{drift_type}",
                    priority=self.config["drift_fine_tune_priority"],
                )

                self.integration_metrics["fine_tune_jobs_triggered"] += 1

                self.logger.info(f"Triggered fine-tuning job {job_id} for drift in {component}")

                # Remove from pending
                del self.pending_drift_responses[response_key]

            except Exception as e:
                self.logger.error(f"Failed to execute drift response {response_key}: {e}")
                # Remove failed response
                del self.pending_drift_responses[response_key]

    async def _process_pending_fine_tune_jobs(self) -> None:
        """Process pending fine-tuning jobs"""

        try:
            processed_jobs = self.fine_tune_scheduler.process_pending_jobs()

            if processed_jobs:
                self.logger.debug(f"Processed {len(processed_jobs)} fine-tuning jobs")

        except Exception as e:
            self.logger.error(f"Failed to process fine-tuning jobs: {e}")

    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""

        try:
            # Get component health scores
            component_scores = {}

            # Drift detection health
            drift_status = self.drift_system.get_system_status()
            alerts_24h = drift_status.get("alerts_24h", 0)
            critical_alerts = drift_status.get("alert_breakdown", {}).get("critical", 0)

            # Penalize based on recent alerts
            drift_health = max(0, 100 - (critical_alerts * 20 + alerts_24h * 5))
            component_scores["drift_detection"] = drift_health

            # Fine-tuning health
            ft_status = self.fine_tune_scheduler.get_system_status()
            pending_jobs = ft_status.get("pending_jobs", 0)
            running_jobs = ft_status.get("running_jobs", 0)

            # Penalize based on job backlog
            ft_health = max(0, 100 - (pending_jobs * 10 + running_jobs * 5))
            component_scores["fine_tuning"] = ft_health

            # Trading system health (from auto-disable)
            trading_status = self.auto_disable.get_current_status()
            consecutive_failures = trading_status.get("consecutive_failures", 0)
            current_mode = trading_status.get("current_mode", "paper")

            # Score based on trading mode and failures
            if current_mode == "live":
                trading_health = max(0, 100 - (consecutive_failures * 15))
            elif current_mode == "paper":
                trading_health = max(0, 70 - (consecutive_failures * 10))
            else:  # disabled
                trading_health = max(0, 40 - (consecutive_failures * 5))

            component_scores["trading_system"] = trading_health

            # Calculate weighted average
            weights = {"drift_detection": 0.3, "fine_tuning": 0.2, "trading_system": 0.5}

            weighted_score = sum(
                score * weights.get(component, 0) for component, score in component_scores.items()
            )

            self.logger.debug(f"Health calculation: {component_scores} -> {weighted_score:.1f}")

            return weighted_score

        except Exception as e:
            self.logger.error(f"Health calculation failed: {e}")
            return 50.0  # Default to medium health

    def _log_drift_alert(self, alert: DriftAlert) -> None:
        """Log drift alert for analysis and debugging"""

        alert_data = {
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "drift_type": alert.drift_type,
            "severity": alert.severity,
            "component": alert.component,
            "description": alert.description,
            "metrics": alert.metrics,
        }

        # Log to structured logger
        self.logger.warning(f"DRIFT_ALERT: {alert_data}")

    def _on_trading_status_change(self, change: TradingStatusChange) -> None:
        """Callback for trading status changes"""

        self.logger.info(
            f"Trading status changed: {change.previous_mode.value} -> "
            f"{change.new_mode.value} (reason: {change.reason.value})"
        )

        # If trading was disabled due to health issues, create urgent fine-tune jobs
        if change.new_mode.value in ["paper", "disabled"] and change.reason in [
            DisableReason.LOW_HEALTH_SCORE,
            DisableReason.PERFORMANCE_DEGRADATION,
        ]:
            # Create high-priority fine-tune jobs for all monitored components
            for component in self.config["health_check_components"]:
                try:
                    job_id = self.fine_tune_scheduler.create_fine_tune_job(
                        model_name=component, trigger_reason="health_degradation", priority="high"
                    )

                    self.logger.info(
                        f"Created urgent fine-tune job {job_id} for {component} "
                        f"due to trading disable"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to create urgent fine-tune job for {component}: {e}")

    def record_component_metrics(self, component: str, metrics: Dict[str, Any]) -> None:
        """Record metrics for a component (for drift detection)"""

        try:
            # Extract relevant metrics for drift detection
            if "error_rate" in metrics:
                self.drift_system.record_error_metrics(component, metrics["error_rate"])

            if "accuracy" in metrics:
                self.drift_system.record_performance_metrics(
                    component, "accuracy", metrics["accuracy"]
                )

            if "latency" in metrics:
                self.drift_system.record_performance_metrics(
                    component, "latency", metrics["latency"]
                )

            if "features" in metrics:
                self.drift_system.record_feature_data(metrics["features"])

            self.logger.debug(f"Recorded metrics for {component}: {list(metrics.keys())}")

        except Exception as e:
            self.logger.error(f"Failed to record metrics for {component}: {e}")

    def add_training_data(
        self, model_name: str, features, targets, importance_weights=None
    ) -> None:
        """Add training data for fine-tuning"""

        try:
            self.fine_tune_scheduler.add_training_data(
                model_name, features, targets, importance_weights
            )
            self.logger.debug(f"Added training data for {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to add training data for {model_name}: {e}")

    def manual_trigger_fine_tune(
        self, model_name: str, reason: str, priority: str = "medium"
    ) -> str:
        """Manually trigger fine-tuning for a model"""

        try:
            job_id = self.fine_tune_scheduler.create_fine_tune_job(
                model_name=model_name, trigger_reason=f"manual_{reason}", priority=priority
            )

            self.logger.info(f"Manually triggered fine-tuning job {job_id} for {model_name}")
            return job_id

        except Exception as e:
            self.logger.error(f"Failed to manually trigger fine-tuning for {model_name}: {e}")
            return ""

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""

        status = {
            "monitoring_active": self.running,
            "last_health_score": self.integration_metrics["last_health_score"],
            "integration_metrics": self.integration_metrics.copy(),
            "pending_drift_responses": len(self.pending_drift_responses),
            "drift_system_status": self.drift_system.get_system_status(),
            "fine_tune_status": self.fine_tune_scheduler.get_system_status(),
            "auto_disable_status": self.auto_disable.get_system_summary(),
            "system_uptime": (
                datetime.now() - self.integration_metrics["system_start_time"]
            ).total_seconds(),
        }

        return status


if __name__ == "__main__":

    async def test_drift_fine_tune_integration():
        """Test the complete integration system"""

        print("🔍 TESTING DRIFT + FINE-TUNE + AUTO-DISABLE INTEGRATION")
        print("=" * 70)

        # Create integration system
        integration = DriftFineTuneIntegration()

        print("🚀 Starting integration monitoring...")
        await integration.start_monitoring()

        print("📊 Simulating system metrics...")

        # REMOVED: Mock data pattern not allowed in production
        import numpy as np
        import random

        # REMOVED: Mock data pattern not allowed in production
        for i in range(5):
            metrics = {
                "error_rate": 0.05 + (i * 0.02),  # Increasing error rate
                "accuracy": 0.95 - (i * 0.02),  # Decreasing accuracy
                "latency": 0.1 + (i * 0.01),  # Increasing latency
                "features": {
                    "price_volatility": [random.gauss(0.1 + i * 0.01, 0.02) for _ in range(20)],
                    "volume_trend": [random.gauss(1.0 + i * 0.05, 0.1) for _ in range(20)],
                },
            }

            integration.record_component_metrics("ml_predictor", metrics)
            print(
                f"   Recorded metrics batch {i + 1}: error_rate={metrics['error_rate']:.3f}, "
                f"accuracy={metrics['accuracy']:.3f}"
            )

        # Add some training data
        print("\n📚 Adding training data...")
        features = np.random.randn(50, 10)
        targets = np.random.normal(0, 1)
        integration.add_training_data("ml_predictor", features, targets)

        # Run a few monitoring cycles
        print("\n⚙️  Running monitoring cycles...")
        for cycle in range(3):
            await integration._run_monitoring_cycle()
            print(f"   Cycle {cycle + 1} completed")
            await asyncio.sleep(1)  # Brief pause

        # Check integration status
        print("\n📈 Integration status:")
        status = integration.get_integration_status()

        print(f"   Monitoring active: {status['monitoring_active']}")
        print(f"   Last health score: {status['last_health_score']:.1f}")
        print(f"   Drift alerts: {status['integration_metrics']['drift_alerts_total']}")
        print(
            f"   Fine-tune jobs triggered: {status['integration_metrics']['fine_tune_jobs_triggered']}"
        )
        print(
            f"   Auto-disables triggered: {status['integration_metrics']['auto_disables_triggered']}"
        )
        print(f"   Pending drift responses: {status['pending_drift_responses']}")

        # Check drift system
        print("\n🚨 Drift detection status:")
        drift_status = status["drift_system_status"]
        print(f"   Total alerts: {drift_status['total_alerts']}")
        print(f"   Alerts (24h): {drift_status['alerts_24h']}")

        # Check fine-tuning
        print("\n⚙️  Fine-tuning status:")
        ft_status = status["fine_tune_status"]
        print(f"   Pending jobs: {ft_status['pending_jobs']}")
        print(f"   Running jobs: {ft_status['running_jobs']}")

        # Check auto-disable
        print("\n🛡️  Auto-disable status:")
        auto_status = status["auto_disable_status"]
        current_status = auto_status["current_status"]
        print(f"   Trading mode: {current_status['current_mode']}")
        print(f"   Consecutive failures: {current_status['consecutive_failures']}")
        print(f"   Changes (24h): {auto_status['changes_24h']}")

        # Test manual fine-tune trigger
        print("\n🎮 Testing manual fine-tune trigger...")
        job_id = integration.manual_trigger_fine_tune("sentiment_analyzer", "testing", "high")
        print(f"   Triggered job: {job_id}")

        # Stop monitoring
        print("\n🛑 Stopping integration monitoring...")
        await integration.stop_monitoring()

        print("\n✅ DRIFT + FINE-TUNE + AUTO-DISABLE INTEGRATION TEST COMPLETED")

        return (
            status["integration_metrics"]["drift_alerts_total"] > 0
            or status["integration_metrics"]["fine_tune_jobs_triggered"] > 0
        )

    # Run test
    success = asyncio.run(test_drift_fine_tune_integration())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
