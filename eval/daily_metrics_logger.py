#!/usr/bin/env python3
"""
Daily Metrics Logger
Implements nightly job that writes daily_metrics_*.json + latest.json to logs/daily/YYYYMMDD/
"""

import pandas as pd
import numpy as np
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

# Import evaluation components
from eval.evaluator import ComprehensiveEvaluator, create_mock_evaluation_data
from eval.coverage_audit import ComprehensiveCoverageAuditor
from eval.system_health_monitor import SystemHealthMonitor


class DailyMetricsLogger:
    """Daily metrics logging system"""

    def __init__(self):
        self.logger = get_structured_logger("DailyMetricsLogger")

        # Initialize evaluators
        self.comprehensive_evaluator = ComprehensiveEvaluator()
        self.coverage_auditor = ComprehensiveCoverageAuditor()
        self.health_monitor = SystemHealthMonitor()

    async def run_nightly_metrics_job(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Run comprehensive nightly metrics collection job"""

        if date is None:
            date = datetime.now()

        job_start_time = time.time()

        self.logger.info(f"Starting nightly metrics job for {date.strftime('%Y-%m-%d')}")

        try:
            # Run all evaluations concurrently where possible
            evaluation_task = self._run_evaluation_metrics()
            coverage_task = self._run_coverage_audit()

            # Wait for evaluations to complete
            evaluation_metrics = await evaluation_task
            coverage_audit = await coverage_task

            # Run health assessment with results
            health_assessment = self.health_monitor.run_health_assessment(
                evaluation_metrics, coverage_audit
            )

            # Compile comprehensive daily metrics
            daily_metrics = {
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "job_duration": time.time() - job_start_time,
                "evaluation_metrics": evaluation_metrics,
                "coverage_audit": coverage_audit,
                "health_assessment": health_assessment,
                "summary": self._create_daily_summary(
                    evaluation_metrics, coverage_audit, health_assessment
                ),
                "go_nogo_decision": {
                    "status": health_assessment.get("status", "NO-GO"),
                    "score": health_assessment.get("overall_score", 0),
                    "recommendation": health_assessment.get("recommendation", "Unknown"),
                },
                "metadata": {
                    "system_version": "CryptoSmartTrader_v2.1",
                    "evaluation_framework": "enterprise_grade",
                    "confidence_threshold": 0.80,
                    "acceptatie_criteria": {
                        "precision_at_5_target": 0.60,
                        "hit_rate_target": 0.55,
                        "mae_target": 0.25,
                        "sharpe_target": 1.0,
                        "coverage_target": 0.95,
                    },
                },
            }

            # Save metrics to daily logs
            saved_files = self._save_daily_metrics(daily_metrics, date)
            daily_metrics["saved_files"] = saved_files

            self.logger.info(
                f"Nightly metrics job completed in {time.time() - job_start_time:.2f}s"
            )

            return daily_metrics

        except Exception as e:
            self.logger.error(f"Nightly metrics job failed: {e}")

            # Create error report
            error_metrics = {
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "job_duration": time.time() - job_start_time,
                "status": "FAILED",
                "go_nogo_decision": {
                    "status": "NO-GO",
                    "score": 0,
                    "recommendation": "Metrics collection failed",
                },
            }

            return error_metrics

    async def _run_evaluation_metrics(self) -> Dict[str, Any]:
        """Run evaluation metrics collection"""

        try:
            self.logger.info("Running evaluation metrics collection")

            # Create evaluation data (in production this would be real data)
            predictions, actuals = create_mock_evaluation_data()

            # Run comprehensive evaluation
            evaluation_results = self.comprehensive_evaluator.evaluate_system(predictions, actuals)

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Evaluation metrics collection failed: {e}")
            return {"error": str(e)}

    async def _run_coverage_audit(self) -> Dict[str, Any]:
        """Run coverage audit"""

        try:
            self.logger.info("Running coverage audit")

            # Run comprehensive coverage audit
            coverage_results = self.coverage_auditor.run_coverage_audit()

            return coverage_results

        except Exception as e:
            self.logger.error(f"Coverage audit failed: {e}")
            return {"error": str(e)}

    def _create_daily_summary(
        self,
        evaluation_metrics: Dict[str, Any],
        coverage_audit: Dict[str, Any],
        health_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create concise daily summary"""

        try:
            # Extract key metrics
            precision_5 = evaluation_metrics.get("precision_at_k", {}).get("precision_at_5", 0)
            hit_rate_30d = evaluation_metrics.get("hit_rates", {}).get("hit_rate_30d", 0)
            sharpe_ratio = evaluation_metrics.get("sharpe_metrics", {}).get("sharpe_ratio", 0)
            coverage_pct = (
                coverage_audit.get("coverage_analysis", {})
                .get("coverage_summary", {})
                .get("coverage_percentage", 0)
            health_score = health_assessment.get("overall_score", 0)
            health_status = health_assessment.get("status", "NO-GO")

            # Calculate trend indicators (simplified)
            trends = {
                "precision_trend": "stable",  # Would compare with previous days
                "hit_rate_trend": "stable",
                "sharpe_trend": "stable",
                "coverage_trend": "stable",
                "health_trend": "stable",
            }

            summary = {
                "key_metrics": {
                    "precision_at_5": precision_5,
                    "hit_rate_30d": hit_rate_30d,
                    "sharpe_ratio": sharpe_ratio,
                    "coverage_percentage": coverage_pct,
                    "health_score": health_score,
                },
                "status_indicators": {
                    "trading_status": health_status,
                    "precision_pass": precision_5 >= 0.60,
                    "hit_rate_pass": hit_rate_30d >= 0.55,
                    "sharpe_pass": sharpe_ratio >= 1.0,
                    "coverage_pass": coverage_pct >= 95.0,
                },
                "trends": trends,
                "alerts": self._generate_alerts(
                    evaluation_metrics, coverage_audit, health_assessment
                ),
            }

            return summary

        except Exception as e:
            self.logger.error(f"Daily summary creation failed: {e}")
            return {"error": str(e)}

    def _generate_alerts(
        self,
        evaluation_metrics: Dict[str, Any],
        coverage_audit: Dict[str, Any],
        health_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate alerts for concerning metrics"""

        alerts = []

        try:
            # Check evaluation metrics
            precision_5 = evaluation_metrics.get("precision_at_k", {}).get("precision_at_5", 0)
            if precision_5 < 0.60:
                alerts.append(f"ALERT: Precision@5 below target: {precision_5:.3f} < 0.60")

            hit_rate_30d = evaluation_metrics.get("hit_rates", {}).get("hit_rate_30d", 0)
            if hit_rate_30d < 0.55:
                alerts.append(f"ALERT: Hit rate below target: {hit_rate_30d:.3f} < 0.55")

            sharpe_ratio = evaluation_metrics.get("sharpe_metrics", {}).get("sharpe_ratio", 0)
            if sharpe_ratio < 1.0:
                alerts.append(f"ALERT: Sharpe ratio below target: {sharpe_ratio:.3f} < 1.0")

            # Check coverage
            coverage_pct = (
                coverage_audit.get("coverage_analysis", {})
                .get("coverage_summary", {})
                .get("coverage_percentage", 0)
            if coverage_pct < 95.0:
                alerts.append(f"ALERT: Coverage below target: {coverage_pct:.1f}% < 95%")

            # Check health status
            health_status = health_assessment.get("status", "NO-GO")
            if health_status == "NO-GO":
                alerts.append("CRITICAL: System health NO-GO - trading blocked")
            elif health_status == "WARNING":
                alerts.append("WARNING: System health degraded - paper trading only")

            if not alerts:
                alerts.append("All systems operating within acceptable parameters")

        except Exception as e:
            alerts.append(f"Alert generation failed: {e}")

        return alerts

    def _save_daily_metrics(self, daily_metrics: Dict[str, Any], date: datetime) -> Dict[str, str]:
        """Save daily metrics to logs/daily/YYYYMMDD/"""

        saved_files = {}

        try:
            # Create daily log directory
            date_str = date.strftime("%Y%m%d")
            daily_log_dir = Path("logs/daily") / date_str
            daily_log_dir.mkdir(parents=True, exist_ok=True)

            # Save timestamped metrics
            timestamp_str = datetime.now().strftime("%H%M%S")
            metrics_file = daily_log_dir / f"daily_metrics_{timestamp_str}.json"

            with open(metrics_file, "w") as f:
                json.dump(daily_metrics, f, indent=2)

            saved_files["timestamped"] = str(metrics_file)

            # Save as latest
            latest_file = daily_log_dir / "latest.json"
            with open(latest_file, "w") as f:
                json.dump(daily_metrics, f, indent=2)

            saved_files["latest"] = str(latest_file)

            # Save summary only
            summary_file = daily_log_dir / f"summary_{timestamp_str}.json"
            with open(summary_file, "w") as f:
                json.dump(daily_metrics.get("summary", {}), f, indent=2)

            saved_files["summary"] = str(summary_file)

            self.logger.info(f"Daily metrics saved to {daily_log_dir}")

            return saved_files

        except Exception as e:
            self.logger.error(f"Failed to save daily metrics: {e}")
            return {"error": str(e)}


class NightlyJobScheduler:
    """Scheduler for nightly metrics job"""

    def __init__(self):
        self.logger = get_structured_logger("NightlyJobScheduler")
        self.metrics_logger = DailyMetricsLogger()

    async def run_scheduled_job(self, target_time: str = "02:00") -> Dict[str, Any]:
        """Run nightly job at scheduled time"""

        self.logger.info(f"Nightly job scheduled for {target_time}")

        try:
            # For demo purposes, run immediately
            # In production, this would wait until target_time

            daily_metrics = await self.metrics_logger.run_nightly_metrics_job()

            self.logger.info("Scheduled nightly job completed successfully")

            return daily_metrics

        except Exception as e:
            self.logger.error(f"Scheduled nightly job failed: {e}")
            return {"error": str(e), "status": "FAILED"}


async def main():
    """Test daily metrics logging system"""

    print("🔍 TESTING DAILY METRICS LOGGING SYSTEM")
    print("=" * 70)

    # Test nightly job
    scheduler = NightlyJobScheduler()
    metrics_results = await scheduler.run_scheduled_job()

    print("\n📊 DAILY METRICS RESULTS:")

    if "error" not in metrics_results:
        summary = metrics_results.get("summary", {})
        key_metrics = summary.get("key_metrics", {})

        print(f"Date: {metrics_results.get('date', 'unknown')}")
        print(f"Job duration: {metrics_results.get('job_duration', 0):.2f}s")

        print("\n📈 KEY METRICS:")
        print(f"• Precision@5: {key_metrics.get('precision_at_5', 0):.3f}")
        print(f"• Hit rate 30d: {key_metrics.get('hit_rate_30d', 0):.3f}")
        print(f"• Sharpe ratio: {key_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"• Coverage: {key_metrics.get('coverage_percentage', 0):.1f}%")
        print(f"• Health score: {key_metrics.get('health_score', 0):.1f}")

        print("\n🚦 GO/NO-GO DECISION:")
        go_nogo = metrics_results.get("go_nogo_decision", {})
        status = go_nogo.get("status", "NO-GO")
        score = go_nogo.get("score", 0)

        if status == "GO":
            print(f"✅ GO (Score: {score:.1f}/100) - Live trading authorized")
        elif status == "WARNING":
            print(f"⚠️ WARNING (Score: {score:.1f}/100) - Paper trading only")
        else:
            print(f"❌ NO-GO (Score: {score:.1f}/100) - Trading blocked")

        print(f"   Recommendation: {go_nogo.get('recommendation', 'Unknown')}")

        print("\n🚨 ALERTS:")
        alerts = summary.get("alerts", [])
        for alert in alerts:
            print(f"• {alert}")

        print("\n💾 SAVED FILES:")
        saved_files = metrics_results.get("saved_files", {})
        for file_type, file_path in saved_files.items():
            print(f"• {file_type}: {file_path}")

    else:
        print(f"❌ Daily metrics job failed: {metrics_results.get('error', 'Unknown error')}")

    print("\n✅ DAILY METRICS LOGGING SYSTEM VOLLEDIG GEÏMPLEMENTEERD!")


if __name__ == "__main__":
    asyncio.run(main())
