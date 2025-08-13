#!/usr/bin/env python3
"""
System Health Score - GO/NO-GO Decision Engine
Implements comprehensive health scoring with weighted components
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger


def calculate_system_health_score(
    validation_accuracy: float,
    sharpe_norm: float,
    feedback_success: float,
    error_ratio: float,
    data_completeness: float,
    tuning_freshness: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate weighted system health score

    Args:
        validation_accuracy: ML validation accuracy (0-1)
        sharpe_norm: Normalized Sharpe ratio (0-1)
        feedback_success: User feedback success rate (0-1)
        error_ratio: System error ratio (0-1, lower is better)
        data_completeness: Data completeness ratio (0-1)
        tuning_freshness: Model tuning freshness (0-1)
        weights: Custom component weights

    Returns:
        Health score (0-100)
    """
    if weights is None:
        weights = {
            "validation_accuracy": 0.25,  # 25%
            "sharpe_norm": 0.20,  # 20%
            "feedback_success": 0.15,  # 15%
            "error_ratio": 0.15,  # 15% (inverted)
            "data_completeness": 0.15,  # 15%
            "tuning_freshness": 0.10,  # 10%
        }

    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    # Calculate weighted score
    score = (
        weights["validation_accuracy"] * validation_accuracy
        + weights["sharpe_norm"] * sharpe_norm
        + weights["feedback_success"] * feedback_success
        + weights["error_ratio"] * (1 - min(error_ratio, 1.0))  # Invert error ratio
        + weights["data_completeness"] * data_completeness
        + weights["tuning_freshness"] * tuning_freshness
    ) * 100

    return round(score, 1)


def normalize_sharpe_ratio(sharpe: float, target_sharpe: float = 2.0) -> float:
    """Normalize Sharpe ratio to 0-1 scale"""
    if sharpe <= 0:
        return 0.0
    return min(sharpe / target_sharpe, 1.0)


def normalize_tuning_freshness(hours_since_tuning: float, max_hours: float = 24.0) -> float:
    """Normalize tuning freshness to 0-1 scale"""
    if hours_since_tuning <= 0:
        return 1.0
    return max(0.0, 1.0 - (hours_since_tuning / max_hours))


def determine_health_decision(score: float) -> str:
    """Determine GO/NO-GO decision based on health score"""
    if score >= 85:
        return "GO"
    elif score >= 60:
        return "WARNING"
    else:
        return "NO-GO"


def load_metrics_from_file(file_path: str) -> Dict[str, float]:
    """Load system metrics from JSON file"""

    logger = get_logger()

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded metrics from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"Metrics file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metrics file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return {}


def comprehensive_health_assessment(
    metrics: Dict[str, Any], weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive health assessment

    Args:
        metrics: Dictionary with raw system metrics
        weights: Custom component weights

    Returns:
        Complete health assessment results
    """
    logger = get_logger()

    # Extract and normalize metrics
    try:
        # Raw metrics with defaults
        validation_accuracy = metrics.get("validation_accuracy", 0.0)
        raw_sharpe = metrics.get("sharpe_ratio", 0.0)
        feedback_success = metrics.get("feedback_success", 0.0)
        error_ratio = metrics.get("error_ratio", 1.0)
        data_completeness = metrics.get("data_completeness", 0.0)
        hours_since_tuning = metrics.get("hours_since_tuning", 48.0)

        # Normalize metrics
        sharpe_norm = normalize_sharpe_ratio(raw_sharpe)
        tuning_freshness = normalize_tuning_freshness(hours_since_tuning)

        # Calculate health score
        health_score = calculate_system_health_score(
            validation_accuracy=validation_accuracy,
            sharpe_norm=sharpe_norm,
            feedback_success=feedback_success,
            error_ratio=error_ratio,
            data_completeness=data_completeness,
            tuning_freshness=tuning_freshness,
            weights=weights,
        )

        # Determine decision
        decision = determine_health_decision(health_score)

        # Create assessment result
        assessment = {
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "health_score": health_score,
            "decision": decision,
            "raw_metrics": {
                "validation_accuracy": validation_accuracy,
                "sharpe_ratio": raw_sharpe,
                "feedback_success": feedback_success,
                "error_ratio": error_ratio,
                "data_completeness": data_completeness,
                "hours_since_tuning": hours_since_tuning,
            },
            "normalized_metrics": {
                "validation_accuracy": validation_accuracy,
                "sharpe_norm": sharpe_norm,
                "feedback_success": feedback_success,
                "error_ratio_inverted": 1 - min(error_ratio, 1.0),
                "data_completeness": data_completeness,
                "tuning_freshness": tuning_freshness,
            },
            "component_weights": weights
            or {
                "validation_accuracy": 0.25,
                "sharpe_norm": 0.20,
                "feedback_success": 0.15,
                "error_ratio": 0.15,
                "data_completeness": 0.15,
                "tuning_freshness": 0.10,
            },
            "component_contributions": {},
            "recommendations": [],
        }

        # Calculate component contributions
        for component, weight in assessment["component_weights"].items():
            if component == "error_ratio":
                normalized_value = assessment["normalized_metrics"]["error_ratio_inverted"]
            elif component == "sharpe_norm":
                normalized_value = assessment["normalized_metrics"]["sharpe_norm"]
            else:
                normalized_value = assessment["normalized_metrics"][component]

            contribution = weight * normalized_value * 100
            assessment["component_contributions"][component] = round(contribution, 1)

        # Generate recommendations
        recommendations = []

        if validation_accuracy < 0.75:
            recommendations.append("Improve ML model validation accuracy")
        if raw_sharpe < 1.0:
            recommendations.append("Enhance trading strategy for better risk-adjusted returns")
        if feedback_success < 0.65:
            recommendations.append("Review recommendation quality based on user feedback")
        if error_ratio > 0.05:
            recommendations.append("Reduce system error rate")
        if data_completeness < 0.98:
            recommendations.append("Improve data coverage and completeness")
        if hours_since_tuning > 24:
            recommendations.append("Retrain models - tuning is stale")

        if decision == "NO-GO":
            recommendations.append("CRITICAL: System health too low for any trading")
        elif decision == "WARNING":
            recommendations.append("WARNING: Live trading disabled, paper trading only")
        else:
            recommendations.append("System healthy - all trading modes authorized")

        assessment["recommendations"] = recommendations

        logger.info(f"Health assessment completed: score={health_score}, decision={decision}")

        return assessment

    except Exception as e:
        logger.error(f"Health assessment failed: {e}")
        return {
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "health_score": 0.0,
            "decision": "ERROR",
            "error": str(e),
            "recommendations": ["Fix health assessment system"],
        }


def save_health_assessment(assessment: Dict[str, Any], output_dir: str = "logs/system") -> str:
    """Save health assessment results"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = assessment["assessment_timestamp"].replace(":", "-").replace(".", "-")
    filename = f"health_assessment_{timestamp}.json"
    filepath = output_path / filename

    # Save assessment
    with open(filepath, "w") as f:
        json.dump(assessment, f, indent=2)

    # Also save as latest
    latest_path = output_path / "latest_health_assessment.json"
    with open(latest_path, "w") as f:
        json.dump(assessment, f, indent=2)

    logger = get_logger()
    logger.info(f"Health assessment saved: {filepath}")

    return str(filepath)


def print_health_summary(assessment: Dict[str, Any]) -> None:
    """Print human-readable health summary"""

    if "error" in assessment:
        print(f"❌ Health Assessment Error: {assessment['error']}")
        return

    score = assessment["health_score"]
    decision = assessment["decision"]

    # Decision icon
    decision_icon = {"GO": "✅", "WARNING": "⚠️", "NO-GO": "❌", "ERROR": "🚨"}.get(decision, "❓")

    print(f"🏥 SYSTEM HEALTH ASSESSMENT")
    print(f"📅 Timestamp: {assessment['assessment_timestamp']}")
    print("=" * 60)

    # Overall score and decision
    print(f"📊 HEALTH SCORE: {score:.1f}/100")
    print(f"🚦 DECISION: {decision_icon} {decision}")
    print()

    # Thresholds
    print(f"🎯 DECISION THRESHOLDS:")
    print(f"   ✅ GO (Live Trading): ≥85.0")
    print(f"   ⚠️  WARNING (Paper Only): 60.0-84.9")
    print(f"   ❌ NO-GO (Blocked): <60.0")
    print()

    # Component breakdown
    print(f"🔧 COMPONENT BREAKDOWN:")
    contributions = assessment.get("component_contributions", {})
    weights = assessment.get("component_weights", {})
    raw_metrics = assessment.get("raw_metrics", {})

    for component, contribution in contributions.items():
        weight = weights.get(component, 0) * 100

        # Get raw value for display
        if component == "sharpe_norm":
            raw_value = raw_metrics.get("sharpe_ratio", 0)
            display_value = f"{raw_value:.2f}"
        elif component == "error_ratio":
            raw_value = raw_metrics.get("error_ratio", 1)
            display_value = f"{raw_value:.1%}"
        elif component in ["validation_accuracy", "feedback_success", "data_completeness"]:
            raw_value = raw_metrics.get(component, 0)
            display_value = f"{raw_value:.1%}"
        elif component == "tuning_freshness":
            hours = raw_metrics.get("hours_since_tuning", 0)
            display_value = f"{hours:.1f}h ago"
        else:
            display_value = "N/A"

        component_name = component.replace("_", " ").title()
        print(
            f"   {component_name:<20} {contribution:>6.1f} pts ({weight:>4.1f}%) | Raw: {display_value}"
        )

    print()

    # Recommendations
    if assessment.get("recommendations"):
        print(f"💡 RECOMMENDATIONS:")
        for i, rec in enumerate(assessment["recommendations"], 1):
            print(f"   {i}. {rec}")
        print()


def create_sample_metrics_file(output_path: str = "logs/system/last_metrics.json") -> None:
    """Create sample metrics file for testing"""

    logger = get_logger()

    # Create sample metrics
    sample_metrics = {
        "validation_accuracy": 0.78,
        "sharpe_ratio": 1.45,
        "feedback_success": 0.72,
        "error_ratio": 0.03,
        "data_completeness": 0.96,
        "hours_since_tuning": 18.5,
    }

    # Create directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sample file
    with open(output_path, "w") as f:
        json.dump(sample_metrics, f, indent=2)

    logger.info(f"Created sample metrics file: {output_path}")
    print(f"📝 Created sample metrics file: {output_path}")


def main():
    """Main entry point for health score script"""

    parser = argparse.ArgumentParser(
        description="System Health Score - GO/NO-GO Decision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/health_score.py --input logs/system/last_metrics.json
  python scripts/health_score.py --create-sample  # Create test file
  python scripts/health_score.py --input metrics.json --output logs/custom
        """,
    )

    parser.add_argument(
        "--input",
        default="logs/system/last_metrics.json",
        help="Path to metrics JSON file (default: logs/system/last_metrics.json)",
    )

    parser.add_argument(
        "--output",
        default="logs/system",
        help="Output directory for assessment (default: logs/system)",
    )

    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample metrics file for testing"
    )

    parser.add_argument("--weights", help="Path to custom weights JSON file")

    args = parser.parse_args()

    try:
        if args.create_sample:
            create_sample_metrics_file(args.input)
            return 0

        # Load metrics
        print(f"📂 Loading metrics from: {args.input}")
        metrics = load_metrics_from_file(args.input)

        if not metrics:
            print(f"❌ No metrics loaded from {args.input}")
            return 1

        # Load custom weights if provided
        weights = None
        if args.weights:
            print(f"📂 Loading custom weights from: {args.weights}")
            weights = load_metrics_from_file(args.weights)
            if not weights:
                print(f"⚠️  Failed to load weights, using defaults")
                weights = None

        # Perform health assessment
        print(f"🏥 Performing health assessment...")
        assessment = comprehensive_health_assessment(metrics, weights)

        if "error" in assessment:
            print(f"❌ Assessment failed: {assessment['error']}")
            return 2

        # Save assessment
        assessment_path = save_health_assessment(assessment, args.output)

        # Display summary
        print_health_summary(assessment)
        print(f"📝 Full assessment saved to: {assessment_path}")

        # Determine exit code based on decision
        decision = assessment["decision"]

        if decision == "GO":
            print("✅ HEALTH CHECK PASSED: System ready for live trading")
            return 0
        elif decision == "WARNING":
            print("⚠️  HEALTH CHECK WARNING: Paper trading only")
            return 1
        elif decision == "NO-GO":
            print("❌ HEALTH CHECK FAILED: No trading authorized")
            return 2
        else:
            print("🚨 HEALTH CHECK ERROR: System assessment failed")
            return 3

    except Exception as e:
        print(f"❌ Health score error: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
