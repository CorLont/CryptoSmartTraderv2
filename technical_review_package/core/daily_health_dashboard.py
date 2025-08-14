#!/usr/bin/env python3
"""
Daily Health Dashboard
Centralized daily health logging and reporting for workstation analysis
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DailyHealthDashboard:
    """
    Centralized daily health monitoring and reporting
    """

    def __init__(self, date_str: Optional[str] = None):
        self.date_str = date_str or datetime.now().strftime("%Y%m%d")
        self.daily_dir = Path(f"logs/daily/{self.date_str}")
        self.daily_dir.mkdir(parents=True, exist_ok=True)

        self.health_data = {}
        self.report_path = self.daily_dir / "health_dashboard.json"
        self.summary_path = self.daily_dir / "daily_summary.html"

    def collect_system_health(self) -> Dict[str, Any]:
        """Collect comprehensive system health metrics"""

        import psutil
        from core.workstation_optimizer import WorkstationHealthMonitor

        monitor = WorkstationHealthMonitor()
        current_metrics = monitor.collect_metrics()
        health_score = monitor.get_health_score()

        system_health = {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "cpu_usage": current_metrics["cpu_percent"],
            "memory_usage": current_metrics["memory_percent"],
            "disk_usage": current_metrics["disk_usage_percent"],
            "process_count": current_metrics["process_count"],
            "gpu_available": self._check_gpu_availability(),
            "gpu_memory_usage": current_metrics.get("gpu_memory_percent", 0),
            "gpu_utilization": current_metrics.get("gpu_utilization", 0),
            "uptime_hours": self._get_uptime_hours(),
        }

        return system_health

    def collect_trading_performance(self) -> Dict[str, Any]:
        """Collect trading performance metrics"""

        # Look for trading logs
        trading_files = list(self.daily_dir.glob("*trading*.json"))

        if not trading_files:
            return {"status": "no_trading_data", "message": "No trading data available for today"}

        # Aggregate trading performance
        total_trades = 0
        total_pnl = 0.0
        win_count = 0

        for file_path in trading_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if "metrics" in data:
                    metrics = data["metrics"]
                    total_trades += metrics.get("total_trades", 0)
                    total_pnl += metrics.get("total_pnl", 0.0)
                    win_count += metrics.get("winning_trades", 0)
            except Exception:
                continue

        win_rate = (win_count / total_trades) if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl_per_trade": total_pnl / max(total_trades, 1),
        }

    def collect_confidence_gate_stats(self) -> Dict[str, Any]:
        """Collect confidence gate statistics"""

        gate_files = list(self.daily_dir.glob("*confidence*.json"))

        if not gate_files:
            return {"status": "no_gate_data", "message": "No confidence gate data available"}

        total_candidates = 0
        total_passed = 0
        gate_events = 0

        for file_path in gate_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if "metrics" in data:
                    metrics = data["metrics"]
                    total_candidates += metrics.get("total_candidates", 0)
                    total_passed += metrics.get("passed_count", 0)
                    gate_events += 1

            except Exception:
                continue

        pass_rate = (total_passed / total_candidates) if total_candidates > 0 else 0

        return {
            "gate_events": gate_events,
            "total_candidates": total_candidates,
            "total_passed": total_passed,
            "pass_rate": pass_rate,
            "avg_candidates_per_event": total_candidates / max(gate_events, 1),
        }

    def collect_model_performance(self) -> Dict[str, Any]:
        """Collect ML model performance metrics"""

        model_files = list(self.daily_dir.glob("*model*.json"))
        model_files.extend(list(self.daily_dir.glob("*ml*.json")))

        if not model_files:
            return {"status": "no_model_data", "message": "No model performance data available"}

        prediction_count = 0
        accuracy_sum = 0.0
        confidence_sum = 0.0

        for file_path in model_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                if "metrics" in data:
                    metrics = data["metrics"]
                    prediction_count += metrics.get("prediction_count", 0)
                    accuracy_sum += metrics.get("accuracy", 0)
                    confidence_sum += metrics.get("avg_confidence", 0)

            except Exception:
                continue

        file_count = len(model_files)

        return {
            "model_evaluations": file_count,
            "total_predictions": prediction_count,
            "avg_accuracy": accuracy_sum / max(file_count, 1),
            "avg_confidence": confidence_sum / max(file_count, 1),
        }

    def collect_risk_assessment(self) -> Dict[str, Any]:
        """Collect risk management metrics"""

        # Check for errors and alerts
        error_count = 0
        warning_count = 0
        critical_alerts = 0

        for log_file in self.daily_dir.glob("*.json"):
            try:
                with open(log_file, "r") as f:
                    content = f.read()

                error_count += content.lower().count("error")
                warning_count += content.lower().count("warning")
                critical_alerts += content.lower().count("critical")

            except Exception:
                continue

        # Calculate risk score (lower is better)
        risk_score = min(100, error_count * 2 + warning_count + critical_alerts * 5)

        return {
            "error_count": error_count,
            "warning_count": warning_count,
            "critical_alerts": critical_alerts,
            "risk_score": risk_score,
            "risk_level": self._categorize_risk_level(risk_score),
        }

    def collect_coverage_report(self) -> Dict[str, Any]:
        """Collect data coverage statistics"""

        coverage_files = list(self.daily_dir.glob("*coverage*.json"))

        if not coverage_files:
            return {"status": "no_coverage_data", "message": "No coverage data available"}

        # Use most recent coverage file
        latest_file = max(coverage_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, "r") as f:
                coverage_data = json.load(f)

            return coverage_data.get(
                "metrics", {"total_coins": 0, "covered_coins": 0, "coverage_percentage": 0}
            )

        except Exception:
            return {"status": "coverage_error", "message": "Error reading coverage data"}

    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily health report"""

        print(f"📊 Generating daily health report for {self.date_str}...")

        # Collect all health data
        health_report = {
            "date": self.date_str,
            "generated_at": datetime.now().isoformat(),
            "system_health": self.collect_system_health(),
            "trading_performance": self.collect_trading_performance(),
            "confidence_gate": self.collect_confidence_gate_stats(),
            "model_performance": self.collect_model_performance(),
            "risk_assessment": self.collect_risk_assessment(),
            "coverage_report": self.collect_coverage_report(),
        }

        # Calculate overall health score
        health_report["overall_health"] = self._calculate_overall_health(health_report)

        # Add recommendations
        health_report["recommendations"] = self._generate_recommendations(health_report)

        # Save to JSON
        with open(self.report_path, "w") as f:
            json.dump(health_report, f, indent=2)

        # Generate HTML summary
        self._generate_html_summary(health_report)

        print(f"   ✅ Report saved to: {self.report_path}")
        print(f"   📄 HTML summary: {self.summary_path}")

        return health_report

    def _calculate_overall_health(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score"""

        scores = []
        weights = []

        # System health (30% weight)
        if "health_score" in report["system_health"]:
            scores.append(report["system_health"]["health_score"])
            weights.append(0.3)

        # Risk assessment (25% weight) - inverted because lower risk is better
        if "risk_score" in report["risk_assessment"]:
            risk_health = max(0, 100 - report["risk_assessment"]["risk_score"])
            scores.append(risk_health)
            weights.append(0.25)

        # Model performance (20% weight)
        if "avg_accuracy" in report["model_performance"]:
            accuracy_score = report["model_performance"]["avg_accuracy"] * 100
            scores.append(accuracy_score)
            weights.append(0.2)

        # Confidence gate (15% weight)
        if "pass_rate" in report["confidence_gate"]:
            gate_score = min(100, report["confidence_gate"]["pass_rate"] * 1000)  # Scale up
            scores.append(gate_score)
            weights.append(0.15)

        # Coverage (10% weight)
        if "coverage_percentage" in report["coverage_report"]:
            coverage_score = report["coverage_report"]["coverage_percentage"]
            scores.append(coverage_score)
            weights.append(0.1)

        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_score = 50.0  # Default neutral score

        return {
            "overall_score": round(weighted_score, 2),
            "health_level": self._categorize_health_level(weighted_score),
            "components_evaluated": len(scores),
        }

    def _categorize_health_level(self, score: float) -> str:
        """Categorize health level based on score"""

        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level"""

        if risk_score <= 10:
            return "low"
        elif risk_score <= 25:
            return "medium"
        elif risk_score <= 50:
            return "high"
        else:
            return "critical"

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # System health recommendations
        sys_health = report["system_health"]
        if sys_health["cpu_usage"] > 80:
            recommendations.append(
                "High CPU usage detected - consider optimizing workload distribution"
            )
        if sys_health["memory_usage"] > 85:
            recommendations.append(
                "High memory usage - review cache settings and process allocation"
            )
        if sys_health["gpu_utilization"] < 30 and sys_health["gpu_available"]:
            recommendations.append("Low GPU utilization - check GPU acceleration settings")

        # Risk recommendations
        risk = report["risk_assessment"]
        if risk["critical_alerts"] > 0:
            recommendations.append(
                f"Critical alerts detected ({risk['critical_alerts']}) - immediate attention required"
            )
        if risk["error_count"] > 10:
            recommendations.append("High error count - review system logs and error handling")

        # Performance recommendations
        model_perf = report["model_performance"]
        if "avg_accuracy" in model_perf and model_perf["avg_accuracy"] < 0.6:
            recommendations.append(
                "Low model accuracy - consider retraining or feature engineering"
            )

        # Confidence gate recommendations
        gate = report["confidence_gate"]
        if "pass_rate" in gate and gate["pass_rate"] < 0.05:
            recommendations.append("Very low confidence gate pass rate - review prediction quality")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _generate_html_summary(self, report: Dict[str, Any]):
        """Generate HTML summary for easy viewing"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CryptoSmartTrader V2 - Daily Health Report {self.date_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 5px 0; }}
        .excellent {{ color: #28a745; }}
        .good {{ color: #17a2b8; }}
        .fair {{ color: #ffc107; }}
        .poor {{ color: #fd7e14; }}
        .critical {{ color: #dc3545; }}
        .recommendations {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CryptoSmartTrader V2 - Daily Health Report</h1>
        <p>Date: {self.date_str} | Generated: {report["generated_at"]}</p>
        <h2 class="{report["overall_health"]["health_level"]}">
            Overall Health: {report["overall_health"]["overall_score"]:.1f}% 
            ({report["overall_health"]["health_level"].title()})
        </h2>
    </div>
    
    <div class="section">
        <h3>System Health</h3>
        <div class="metric">Health Score: {report["system_health"]["health_score"]:.1f}%</div>
        <div class="metric">CPU Usage: {report["system_health"]["cpu_usage"]:.1f}%</div>
        <div class="metric">Memory Usage: {report["system_health"]["memory_usage"]:.1f}%</div>
        <div class="metric">GPU Available: {"Yes" if report["system_health"]["gpu_available"] else "No"}</div>
        <div class="metric">Uptime: {report["system_health"]["uptime_hours"]:.1f} hours</div>
    </div>
    
    <div class="section">
        <h3>Trading Performance</h3>
        <div class="metric">Total Trades: {report["trading_performance"].get("total_trades", "N/A")}</div>
        <div class="metric">Total P&L: {report["trading_performance"].get("total_pnl", "N/A")}</div>
        <div class="metric">Win Rate: {report["trading_performance"].get("win_rate", 0):.1%}</div>
    </div>
    
    <div class="section">
        <h3>Confidence Gate</h3>
        <div class="metric">Gate Events: {report["confidence_gate"].get("gate_events", "N/A")}</div>
        <div class="metric">Pass Rate: {report["confidence_gate"].get("pass_rate", 0):.1%}</div>
        <div class="metric">Total Candidates: {report["confidence_gate"].get("total_candidates", "N/A")}</div>
    </div>
    
    <div class="section">
        <h3>Risk Assessment</h3>
        <div class="metric">Risk Level: {report["risk_assessment"]["risk_level"].title()}</div>
        <div class="metric">Error Count: {report["risk_assessment"]["error_count"]}</div>
        <div class="metric">Critical Alerts: {report["risk_assessment"]["critical_alerts"]}</div>
    </div>
    
    <div class="section recommendations">
        <h3>Recommendations</h3>
        {"<br>".join(f"• {rec}" for rec in report["recommendations"])}
    </div>
</body>
</html>
        """

        with open(self.summary_path, "w") as f:
            f.write(html_content)

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_uptime_hours(self) -> float:
        """Get system uptime in hours"""
        try:
            import psutil

            boot_time = psutil.boot_time()
            uptime_seconds = psutil.time.time() - boot_time
            return uptime_seconds / 3600
        except Exception:
            return 0.0


def generate_daily_health_report(date_str: Optional[str] = None) -> Dict[str, Any]:
    """Generate daily health report for specified date"""

    dashboard = DailyHealthDashboard(date_str)
    return dashboard.generate_daily_report()


def get_multi_day_trends(days: int = 7) -> Dict[str, Any]:
    """Get health trends over multiple days"""

    trends = {
        "days_analyzed": 0,
        "health_scores": [],
        "dates": [],
        "trend_direction": "stable",
        "avg_health": 0.0,
    }

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        report_path = Path(f"logs/daily/{date}/health_dashboard.json")

        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    report = json.load(f)

                health_score = report["overall_health"]["overall_score"]
                trends["health_scores"].append(health_score)
                trends["dates"].append(date)
                trends["days_analyzed"] += 1

            except Exception:
                continue

    if trends["health_scores"]:
        trends["avg_health"] = sum(trends["health_scores"]) / len(trends["health_scores"])

        # Calculate trend
        if len(trends["health_scores"]) >= 2:
            recent_avg = sum(trends["health_scores"][:3]) / min(3, len(trends["health_scores"]))
            older_avg = sum(trends["health_scores"][-3:]) / min(3, len(trends["health_scores"]))

            if recent_avg > older_avg + 5:
                trends["trend_direction"] = "improving"
            elif recent_avg < older_avg - 5:
                trends["trend_direction"] = "declining"

    return trends


if __name__ == "__main__":
    print("📊 GENERATING DAILY HEALTH DASHBOARD")
    print("=" * 50)

    # Generate today's report
    report = generate_daily_health_report()

    print(f"\n📋 Daily Health Summary:")
    print(
        f"   Overall Health: {report['overall_health']['overall_score']:.1f}% ({report['overall_health']['health_level']})"
    )
    print(f"   System Health: {report['system_health']['health_score']:.1f}%")
    print(f"   Risk Level: {report['risk_assessment']['risk_level']}")

    if report["recommendations"]:
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"   {i}. {rec}")

    # Get trends
    trends = get_multi_day_trends(7)

    print(f"\n📈 7-Day Trends:")
    print(f"   Days Analyzed: {trends['days_analyzed']}")
    print(f"   Average Health: {trends['avg_health']:.1f}%")
    print(f"   Trend: {trends['trend_direction']}")

    print(f"\n✅ Daily health dashboard completed")
