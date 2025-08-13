#!/usr/bin/env python3
"""
Advanced Analytics Engine
Comprehensive analytics and insights generation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class AnalyticsInsight:
    """Analytics insight data structure"""

    category: str
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_level: str  # LOW, MEDIUM, HIGH
    actionable: bool
    recommended_action: Optional[str] = None


class AdvancedAnalytics:
    """
    Advanced analytics engine for comprehensive system insights
    """

    def __init__(self):
        self.analytics_results = {}
        self.insights_generated = []
        self.trend_analysis = {}

    def generate_comprehensive_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""

        print("📊 GENERATING ADVANCED ANALYTICS")
        print("=" * 45)

        analytics_start = time.time()

        # Core analytics
        self._analyze_system_performance_trends()
        self._analyze_trading_effectiveness()
        self._analyze_confidence_gate_patterns()
        self._analyze_prediction_accuracy()
        self._analyze_resource_utilization()
        self._analyze_error_patterns()
        self._generate_predictive_insights()
        self._calculate_optimization_opportunities()

        analytics_duration = time.time() - analytics_start

        # Compile comprehensive report
        analytics_report = {
            "analytics_timestamp": datetime.now().isoformat(),
            "analytics_duration": analytics_duration,
            "insights_count": len(self.insights_generated),
            "high_impact_insights": len(
                [i for i in self.insights_generated if i.impact_level == "HIGH"]
            ),
            "actionable_insights": len([i for i in self.insights_generated if i.actionable]),
            "analytics_results": self.analytics_results,
            "insights": [self._insight_to_dict(i) for i in self.insights_generated],
            "trend_analysis": self.trend_analysis,
            "recommendations": self._generate_strategic_recommendations(),
        }

        # Save analytics report
        self._save_analytics_report(analytics_report)

        return analytics_report

    def _analyze_system_performance_trends(self):
        """Analyze system performance trends over time"""

        print("📈 Analyzing system performance trends...")

        # Load recent system metrics
        metrics_data = self._load_system_metrics_history()

        if not metrics_data:
            self.analytics_results["performance_trends"] = {"status": "insufficient_data"}
            return

        # Calculate performance trends
        df = pd.DataFrame(metrics_data)

        if len(df) < 10:
            self.analytics_results["performance_trends"] = {"status": "insufficient_data"}
            return

        # Calculate rolling averages
        df["cpu_ma"] = df["cpu_percent"].rolling(window=5).mean()
        df["memory_ma"] = df["memory_percent"].rolling(window=5).mean()

        # Trend analysis
        cpu_trend = np.polyfit(range(len(df)), df["cpu_percent"], 1)[0]
        memory_trend = np.polyfit(range(len(df)), df["memory_percent"], 1)[0]

        performance_analysis = {
            "data_points": len(df),
            "cpu_trend_slope": float(cpu_trend),
            "memory_trend_slope": float(memory_trend),
            "avg_cpu": float(df["cpu_percent"].mean()),
            "avg_memory": float(df["memory_percent"].mean()),
            "cpu_volatility": float(df["cpu_percent"].std()),
            "memory_volatility": float(df["memory_percent"].std()),
        }

        # Generate insights
        if cpu_trend > 0.5:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Performance",
                    insight_type="Trend",
                    title="Increasing CPU Usage Trend",
                    description=f"CPU usage is trending upward by {cpu_trend:.2f}% per measurement",
                    confidence=0.85,
                    impact_level="MEDIUM",
                    actionable=True,
                    recommended_action="Consider process optimization or resource scaling",
                )
            )

        if memory_trend > 0.3:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Performance",
                    insight_type="Trend",
                    title="Memory Usage Growing",
                    description=f"Memory usage is increasing by {memory_trend:.2f}% per measurement",
                    confidence=0.80,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Investigate memory leaks and optimize garbage collection",
                )
            )

        self.analytics_results["performance_trends"] = performance_analysis
        print(f"   Analyzed {len(df)} performance data points")

    def _analyze_trading_effectiveness(self):
        """Analyze trading system effectiveness"""

        print("💰 Analyzing trading effectiveness...")

        # Load trading metrics
        trading_data = self._load_trading_metrics()

        if not trading_data:
            self.analytics_results["trading_effectiveness"] = {"status": "no_trading_data"}
            return

        # Calculate trading metrics
        total_predictions = sum(d.get("daily_predictions", 0) for d in trading_data)
        total_opportunities = sum(d.get("passed_candidates", 0) for d in trading_data)

        if total_predictions > 0:
            overall_pass_rate = (total_opportunities / total_predictions) * 100
        else:
            overall_pass_rate = 0.0

        trading_analysis = {
            "total_predictions": total_predictions,
            "total_opportunities": total_opportunities,
            "overall_pass_rate": overall_pass_rate,
            "active_days": len(trading_data),
            "avg_daily_predictions": total_predictions / max(len(trading_data), 1),
            "predictions_per_day_trend": self._calculate_trend(
                [d.get("daily_predictions", 0) for d in trading_data]
            ),
        }

        # Generate trading insights
        if overall_pass_rate < 2.0:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Trading",
                    insight_type="Performance",
                    title="Very Low Confidence Pass Rate",
                    description=f"Only {overall_pass_rate:.2f}% of predictions pass confidence gate",
                    confidence=0.95,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Review confidence thresholds and model performance",
                )
            )

        if trading_analysis["predictions_per_day_trend"] < -0.5:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Trading",
                    insight_type="Trend",
                    title="Declining Prediction Volume",
                    description="Daily prediction volume is declining over time",
                    confidence=0.75,
                    impact_level="MEDIUM",
                    actionable=True,
                    recommended_action="Investigate data source availability and model health",
                )
            )

        self.analytics_results["trading_effectiveness"] = trading_analysis
        print(f"   Analyzed {len(trading_data)} days of trading data")

    def _analyze_confidence_gate_patterns(self):
        """Analyze confidence gate filtering patterns"""

        print("🚪 Analyzing confidence gate patterns...")

        # Load confidence gate data
        gate_data = self._load_confidence_gate_data()

        if not gate_data:
            self.analytics_results["confidence_patterns"] = {"status": "no_gate_data"}
            return

        # Calculate gate statistics
        pass_rates = [d.get("pass_rate", 0) * 100 for d in gate_data if "pass_rate" in d]
        candidate_counts = [
            d.get("total_candidates", 0) for d in gate_data if "total_candidates" in d
        ]

        if not pass_rates:
            self.analytics_results["confidence_patterns"] = {"status": "no_valid_data"}
            return

        gate_analysis = {
            "measurements": len(pass_rates),
            "avg_pass_rate": np.mean(pass_rates),
            "pass_rate_std": np.std(pass_rates),
            "min_pass_rate": np.min(pass_rates),
            "max_pass_rate": np.max(pass_rates),
            "avg_candidates": np.mean(candidate_counts) if candidate_counts else 0,
            "pass_rate_trend": self._calculate_trend(pass_rates),
            "volatility_index": np.std(pass_rates) / max(np.mean(pass_rates), 0.1),
        }

        # Generate confidence gate insights
        if gate_analysis["avg_pass_rate"] < 1.0:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="ConfidenceGate",
                    insight_type="Performance",
                    title="Ultra-Strict Confidence Filtering",
                    description=f"Average pass rate is {gate_analysis['avg_pass_rate']:.2f}%, indicating very strict filtering",
                    confidence=0.90,
                    impact_level="MEDIUM",
                    actionable=True,
                    recommended_action="Consider if filtering is too restrictive for trading opportunities",
                )
            )

        if gate_analysis["volatility_index"] > 2.0:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="ConfidenceGate",
                    insight_type="Stability",
                    title="High Pass Rate Volatility",
                    description="Confidence gate pass rates are highly volatile",
                    confidence=0.80,
                    impact_level="MEDIUM",
                    actionable=True,
                    recommended_action="Investigate market regime changes or model instability",
                )
            )

        self.analytics_results["confidence_patterns"] = gate_analysis
        print(f"   Analyzed {len(pass_rates)} confidence gate measurements")

    def _analyze_prediction_accuracy(self):
        """Analyze ML prediction accuracy patterns"""

        print("🎯 Analyzing prediction accuracy...")

        # This would normally analyze actual prediction vs outcome data
        # For now, simulate accuracy analysis

        accuracy_analysis = {
            "simulated_data": True,
            "estimated_accuracy": 68.5,
            "confidence_calibration": 0.72,
            "model_stability": 0.85,
            "regime_adaptation": 0.78,
        }

        # Generate prediction insights
        if accuracy_analysis["estimated_accuracy"] < 65.0:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="ML",
                    insight_type="Accuracy",
                    title="Below Target Prediction Accuracy",
                    description=f"Estimated accuracy {accuracy_analysis['estimated_accuracy']:.1f}% is below 65% target",
                    confidence=0.70,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Retrain models with recent data and feature engineering",
                )
            )

        self.analytics_results["prediction_accuracy"] = accuracy_analysis
        print("   Prediction accuracy analysis completed")

    def _analyze_resource_utilization(self):
        """Analyze resource utilization efficiency"""

        print("🖥️ Analyzing resource utilization...")

        # Load recent system metrics
        system_data = self._load_system_metrics_history()

        if not system_data:
            self.analytics_results["resource_utilization"] = {"status": "no_data"}
            return

        # Calculate resource efficiency
        cpu_values = [d.get("cpu_percent", 0) for d in system_data]
        memory_values = [d.get("memory_percent", 0) for d in system_data]

        utilization_analysis = {
            "avg_cpu_utilization": np.mean(cpu_values),
            "avg_memory_utilization": np.mean(memory_values),
            "cpu_efficiency_score": min(100, (np.mean(cpu_values) / 75) * 100),  # 75% target
            "memory_efficiency_score": min(100, (np.mean(memory_values) / 80) * 100),  # 80% target
            "resource_balance_score": 100 - abs(np.mean(cpu_values) - np.mean(memory_values)),
        }

        # Generate resource insights
        if utilization_analysis["avg_cpu_utilization"] < 30:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Resources",
                    insight_type="Efficiency",
                    title="Low CPU Utilization",
                    description=f"CPU utilization averaging {utilization_analysis['avg_cpu_utilization']:.1f}% - potential for increased throughput",
                    confidence=0.85,
                    impact_level="MEDIUM",
                    actionable=True,
                    recommended_action="Consider increasing parallel processing or batch sizes",
                )
            )

        self.analytics_results["resource_utilization"] = utilization_analysis
        print(f"   Resource utilization analysis completed")

    def _analyze_error_patterns(self):
        """Analyze error patterns and failure modes"""

        print("🚨 Analyzing error patterns...")

        # Load error data from logs
        error_data = self._load_error_data()

        error_analysis = {
            "total_errors": len(error_data),
            "error_categories": self._categorize_errors(error_data),
            "error_trend": self._calculate_trend([d.get("daily_errors", 0) for d in error_data]),
            "critical_errors": len([e for e in error_data if e.get("level") == "CRITICAL"]),
        }

        # Generate error insights
        if error_analysis["total_errors"] > 50:
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Reliability",
                    insight_type="Errors",
                    title="High Error Volume",
                    description=f"{error_analysis['total_errors']} errors detected in recent period",
                    confidence=0.95,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Investigate error root causes and implement fixes",
                )
            )

        self.analytics_results["error_patterns"] = error_analysis
        print(f"   Analyzed {len(error_data)} error occurrences")

    def _generate_predictive_insights(self):
        """Generate predictive insights for future performance"""

        print("🔮 Generating predictive insights...")

        # Load historical data for prediction
        historical_data = self._load_system_metrics_history()

        if len(historical_data) < 20:
            self.analytics_results["predictive_insights"] = {"status": "insufficient_data"}
            return

        # Simple trend extrapolation
        cpu_values = [d.get("cpu_percent", 0) for d in historical_data[-20:]]
        memory_values = [d.get("memory_percent", 0) for d in historical_data[-20:]]

        # Predict next 24 hours
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]

        predicted_cpu = cpu_values[-1] + (cpu_trend * 24)  # 24 hours ahead
        predicted_memory = memory_values[-1] + (memory_trend * 24)

        predictive_analysis = {
            "prediction_horizon_hours": 24,
            "predicted_cpu_usage": max(0, min(100, predicted_cpu)),
            "predicted_memory_usage": max(0, min(100, predicted_memory)),
            "cpu_trend_confidence": 0.75,
            "memory_trend_confidence": 0.80,
            "potential_issues": [],
        }

        # Generate predictive insights
        if predicted_cpu > 90:
            predictive_analysis["potential_issues"].append("CPU overload risk")
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Prediction",
                    insight_type="Risk",
                    title="CPU Overload Risk in 24h",
                    description=f"CPU usage predicted to reach {predicted_cpu:.1f}% within 24 hours",
                    confidence=0.75,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Plan resource scaling or workload reduction",
                )
            )

        if predicted_memory > 95:
            predictive_analysis["potential_issues"].append("Memory exhaustion risk")
            self.insights_generated.append(
                AnalyticsInsight(
                    category="Prediction",
                    insight_type="Risk",
                    title="Memory Exhaustion Risk in 24h",
                    description=f"Memory usage predicted to reach {predicted_memory:.1f}% within 24 hours",
                    confidence=0.80,
                    impact_level="HIGH",
                    actionable=True,
                    recommended_action="Implement memory cleanup or restart services",
                )
            )

        self.analytics_results["predictive_insights"] = predictive_analysis
        print("   Predictive insights generated for 24h horizon")

    def _calculate_optimization_opportunities(self):
        """Calculate specific optimization opportunities"""

        print("⚡ Calculating optimization opportunities...")

        opportunities = []

        # Analyze current system state
        if "performance_trends" in self.analytics_results:
            perf = self.analytics_results["performance_trends"]

            if perf.get("avg_cpu", 0) < 40:
                opportunities.append(
                    {
                        "category": "Performance",
                        "opportunity": "Increase parallel processing",
                        "potential_improvement": "30-50% throughput increase",
                        "implementation_effort": "Medium",
                    }
                )

        if "confidence_patterns" in self.analytics_results:
            conf = self.analytics_results["confidence_patterns"]

            if conf.get("avg_pass_rate", 0) < 1:
                opportunities.append(
                    {
                        "category": "Trading",
                        "opportunity": "Optimize confidence thresholds",
                        "potential_improvement": "2-3x more trading opportunities",
                        "implementation_effort": "Low",
                    }
                )

        optimization_analysis = {
            "opportunities_identified": len(opportunities),
            "opportunities": opportunities,
            "total_potential_improvement": "High",
            "priority_order": self._prioritize_opportunities(opportunities),
        }

        self.analytics_results["optimization_opportunities"] = optimization_analysis
        print(f"   Identified {len(opportunities)} optimization opportunities")

    def _load_system_metrics_history(self) -> List[Dict[str, Any]]:
        """Load system metrics history from logs"""

        metrics_data = []

        # Try to load from multiple days
        for days_back in range(7):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            metrics_file = Path(f"logs/daily/{date}/real_time_metrics.jsonl")

            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        for line in f:
                            metrics_data.append(json.loads(line))
                except Exception:
                    continue

        return metrics_data[-100:]  # Last 100 measurements

    def _load_trading_metrics(self) -> List[Dict[str, Any]]:
        """Load trading metrics from daily logs"""

        trading_data = []

        for days_back in range(30):  # 30 days of data
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            confidence_file = Path(f"logs/daily/{date}/confidence_gate.jsonl")

            if confidence_file.exists():
                try:
                    with open(confidence_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            latest_line = lines[-1]
                            daily_data = json.loads(latest_line)
                            daily_data["date"] = date
                            trading_data.append(daily_data)
                except Exception:
                    continue

        return trading_data

    def _load_confidence_gate_data(self) -> List[Dict[str, Any]]:
        """Load confidence gate data"""
        return self._load_trading_metrics()  # Same data source

    def _load_error_data(self) -> List[Dict[str, Any]]:
        """Load error data from logs"""

        error_data = []

        try:
            today = datetime.now().strftime("%Y%m%d")
            alert_file = Path(f"logs/daily/{today}/system_alerts.jsonl")

            if alert_file.exists():
                with open(alert_file, "r") as f:
                    for line in f:
                        error_data.append(json.loads(line))
        except Exception:
            pass

        return error_data

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a series of values"""

        if len(values) < 2:
            return 0.0

        return float(np.polyfit(range(len(values)), values, 1)[0])

    def _categorize_errors(self, error_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize errors by type"""

        categories = {}

        for error in error_data:
            component = error.get("component", "Unknown")
            categories[component] = categories.get(component, 0) + 1

        return categories

    def _prioritize_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Prioritize optimization opportunities"""

        # Simple prioritization by effort/impact ratio
        priority_scores = {}

        for opp in opportunities:
            effort_score = {"Low": 1, "Medium": 2, "High": 3}.get(
                opp.get("implementation_effort", "Medium"), 2
            )
            impact_desc = opp.get("potential_improvement", "")

            # Simple impact scoring
            if "50%" in impact_desc or "3x" in impact_desc:
                impact_score = 3
            elif "30%" in impact_desc or "2x" in impact_desc:
                impact_score = 2
            else:
                impact_score = 1

            priority_scores[opp["opportunity"]] = impact_score / effort_score

        return sorted(priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True)

    def _insight_to_dict(self, insight: AnalyticsInsight) -> Dict[str, Any]:
        """Convert insight to dictionary"""

        return {
            "category": insight.category,
            "insight_type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "confidence": insight.confidence,
            "impact_level": insight.impact_level,
            "actionable": insight.actionable,
            "recommended_action": insight.recommended_action,
        }

    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on analysis"""

        recommendations = []

        # High-impact actionable insights
        high_impact_insights = [
            i for i in self.insights_generated if i.impact_level == "HIGH" and i.actionable
        ]

        for insight in high_impact_insights:
            if insight.recommended_action:
                recommendations.append(insight.recommended_action)

        # General strategic recommendations
        if "confidence_patterns" in self.analytics_results:
            avg_pass_rate = self.analytics_results["confidence_patterns"].get("avg_pass_rate", 0)
            if avg_pass_rate < 2:
                recommendations.append(
                    "Consider implementing ensemble models to improve prediction confidence"
                )

        if "resource_utilization" in self.analytics_results:
            cpu_util = self.analytics_results["resource_utilization"].get("avg_cpu_utilization", 0)
            if cpu_util < 40:
                recommendations.append("Scale up processing to utilize available CPU capacity")

        return recommendations[:10]  # Top 10 recommendations

    def _save_analytics_report(self, report: Dict[str, Any]):
        """Save analytics report"""

        report_dir = Path("logs/analytics")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"advanced_analytics_{timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Analytics report saved: {report_path}")

    def print_analytics_summary(self, report: Dict[str, Any]):
        """Print analytics summary"""

        print(f"\n🎯 ADVANCED ANALYTICS SUMMARY")
        print("=" * 45)
        print(f"Analysis Duration: {report['analytics_duration']:.2f}s")
        print(f"Total Insights: {report['insights_count']}")
        print(f"High Impact Insights: {report['high_impact_insights']}")
        print(f"Actionable Insights: {report['actionable_insights']}")

        # Top insights
        high_impact = [i for i in report["insights"] if i["impact_level"] == "HIGH"]
        if high_impact:
            print(f"\n🚨 High Impact Insights:")
            for insight in high_impact[:3]:
                print(f"   • {insight['title']}: {insight['description']}")

        # Top recommendations
        if report["recommendations"]:
            print(f"\n💡 Strategic Recommendations:")
            for i, rec in enumerate(report["recommendations"][:5], 1):
                print(f"   {i}. {rec}")


def run_advanced_analytics() -> Dict[str, Any]:
    """Run advanced analytics"""

    analytics = AdvancedAnalytics()
    report = analytics.generate_comprehensive_analytics()
    analytics.print_analytics_summary(report)

    return report


if __name__ == "__main__":
    analytics_report = run_advanced_analytics()
