#!/usr/bin/env python3
"""
Calibration Check - Confidence vs Success Rate Analysis
Implements binning analysis for confidence calibration assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import argparse
import json
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger


def confidence_binning_analysis(
    df: pd.DataFrame, conf_col: str, bins: Tuple = (0.8, 0.9, 1.01) -> pd.DataFrame:
    """
    Perform confidence binning analysis

    Args:
        df: DataFrame with confidence scores and realized returns
        conf_col: Name of confidence column
        bins: Confidence bin edges

    Returns:
        DataFrame with binning results
    """
    logger = get_logger()

    if df.empty:
        logger.warning("Empty DataFrame provided for calibration analysis")
        return pd.DataFrame()

    if conf_col not in df.columns:
        logger.error(f"Confidence column {conf_col} not found in DataFrame")
        return pd.DataFrame()

    if "realized" not in df.columns:
        logger.error("Realized column not found in DataFrame")
        return pd.DataFrame()

    logger.info(f"Performing confidence binning analysis with bins: {bins}")

    # Create bin labels
    labels = [f"{int(a * 100)}–{int(b * 100)}%" for a, b in zip(bins[:-1], bins[1:])]

    # Bin the confidence scores
    try:
        confidence_bins = pd.cut(df[conf_col], bins=bins, labels=labels, include_lowest=True)
    except Exception as e:
        logger.error(f"Failed to create confidence bins: {e}")
        return pd.DataFrame()

    # Calculate success rates per bin
    binning_results = (
        df.groupby(confidence_bins, observed=False)
        .agg(
            {
                "realized": [
                    ("count", "count"),
                    ("success_rate", lambda x: (x > 0).mean()),
                    ("mean_return", "mean"),
                    ("std_return", "std"),
                ]
            }
        )
        .round(4)

    # Flatten column names
    binning_results.columns = ["_".join(col) for col in binning_results.columns]

    # Reset index to get confidence ranges as column
    binning_results = binning_results.reset_index()
    binning_results = binning_results.rename(columns={conf_col: "confidence_range"})

    # Add expected confidence midpoints for comparison
    bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    binning_results["expected_confidence"] = bin_midpoints

    # Calculate calibration error
    binning_results["calibration_error"] = abs(
        binning_results["realized_success_rate"] - binning_results["expected_confidence"]
    )

    logger.info(f"Binning analysis completed: {len(binning_results)} bins created")

    return binning_results


def detailed_calibration_analysis(df: pd.DataFrame, conf_col: str = "conf_720h") -> Dict[str, Any]:
    """
    Perform comprehensive calibration analysis

    Args:
        df: DataFrame with predictions and realized returns
        conf_col: Confidence column name

    Returns:
        Dictionary with calibration metrics
    """
    logger = get_logger()

    if df.empty:
        return {"error": "Empty DataFrame provided"}

    logger.info(f"Performing detailed calibration analysis on {len(df)} samples")

    # Standard confidence bins
    standard_bins = (0.8, 0.9, 1.01)
    standard_results = confidence_binning_analysis(df, conf_col, standard_bins)

    # Fine-grained bins for detailed analysis
    fine_bins = tuple(np.linspace(0.8, 1.0, 6))  # 80%, 84%, 88%, 92%, 96%, 100%
    fine_results = confidence_binning_analysis(df, conf_col, fine_bins)

    # Overall calibration metrics
    overall_metrics = {
        "sample_size": int(len(df)),
        "confidence_range": {
            "min": float(df[conf_col].min()),
            "max": float(df[conf_col].max()),
            "mean": float(df[conf_col].mean()),
            "std": float(df[conf_col].std()),
        },
        "overall_success_rate": float((df["realized"] > 0).mean()),
        "high_confidence_count": int((df[conf_col] >= 0.8).sum()),
        "very_high_confidence_count": int((df[conf_col] >= 0.9).sum()),
    }

    # Calibration quality assessment
    if not standard_results.empty:
        # Expected Calibration Error (ECE)
        ece = (
            standard_results["realized_count"] * standard_results["calibration_error"]
        ).sum() / standard_results["realized_count"].sum()

        # Reliability diagram metrics
        reliability_metrics = {
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(standard_results["calibration_error"].max()),
            "bins_with_data": int((standard_results["realized_count"] > 0).sum()),
            "well_calibrated_bins": int((standard_results["calibration_error"] <= 0.1).sum()),
        }
    else:
        reliability_metrics = {
            "expected_calibration_error": np.nan,
            "max_calibration_error": np.nan,
            "bins_with_data": 0,
            "well_calibrated_bins": 0,
        }

    # Compile results
    calibration_analysis = {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "confidence_column": conf_col,
        "overall_metrics": overall_metrics,
        "reliability_metrics": reliability_metrics,
        "standard_binning": standard_results.to_dict("records")
        if not standard_results.empty
        else [],
        "fine_binning": fine_results.to_dict("records") if not fine_results.empty else [],
    }

    logger.info(
        f"Calibration analysis completed: ECE={reliability_metrics['expected_calibration_error']:.3f}"
    )

    return calibration_analysis


def assess_calibration_quality(results: Dict[str, Any]) -> str:
    """Assess overall calibration quality"""

    if "error" in results:
        return "unknown"

    reliability = results["reliability_metrics"]
    ece = reliability["expected_calibration_error"]

    if np.isnan(ece):
        return "insufficient_data"
    elif ece <= 0.05:
        return "excellent"
    elif ece <= 0.10:
        return "good"
    elif ece <= 0.15:
        return "acceptable"
    else:
        return "poor"


def save_calibration_results(results: Dict[str, Any], output_dir: str = "logs/calibration") -> str:
    """Save calibration analysis results"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = results["analysis_timestamp"].replace(":", "-").replace(".", "-")
    filename = f"calibration_analysis_{timestamp}.json"
    filepath = output_path / filename

    # Save results
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as latest
    latest_path = output_path / "latest_calibration_analysis.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    logger = get_logger()
    logger.info(f"Calibration results saved: {filepath}")

    return str(filepath)


def print_calibration_summary(results: Dict[str, Any]) -> None:
    """Print human-readable calibration summary"""

    if "error" in results:
        print(f"❌ Calibration Error: {results['error']}")
        return

    overall = results["overall_metrics"]
    reliability = results["reliability_metrics"]
    standard_binning = results["standard_binning"]

    print(f"🎯 CONFIDENCE CALIBRATION ANALYSIS")
    print(f"📅 Timestamp: {results['analysis_timestamp']}")
    print("=" * 60)

    # Overall metrics
    print(f"📊 OVERALL METRICS:")
    print(f"   Sample size: {overall['sample_size']:,}")
    print(
        f"   Confidence range: {overall['confidence_range']['min']:.1%} - {overall['confidence_range']['max']:.1%}"
    )
    print(f"   Mean confidence: {overall['confidence_range']['mean']:.1%}")
    print(f"   Overall success rate: {overall['overall_success_rate']:.1%}")
    print(f"   High confidence (≥80%): {overall['high_confidence_count']:,}")
    print(f"   Very high confidence (≥90%): {overall['very_high_confidence_count']:,}")
    print()

    # Calibration quality
    quality = assess_calibration_quality(results)
    quality_icon = {
        "excellent": "✅",
        "good": "🟢",
        "acceptable": "🟡",
        "poor": "🔴",
        "insufficient_data": "❓",
        "unknown": "❓",
    }.get(quality, "❓")

    print(f"🎯 CALIBRATION QUALITY:")
    print(f"   Overall assessment: {quality_icon} {quality.upper()}")

    if not np.isnan(reliability["expected_calibration_error"]):
        print(f"   Expected Calibration Error: {reliability['expected_calibration_error']:.3f}")
        print(f"   Max Calibration Error: {reliability['max_calibration_error']:.3f}")
        print(
            f"   Well-calibrated bins: {reliability['well_calibrated_bins']}/{reliability['bins_with_data']}"
        )
    print()

    # Binning results
    if standard_binning:
        print(f"📈 CONFIDENCE BINNING RESULTS:")
        print(f"{'Range':<12} {'Count':<8} {'Success':<8} {'Expected':<10} {'Error':<8}")
        print("-" * 50)

        for bin_result in standard_binning:
            if bin_result["realized_count"] > 0:
                print(
                    f"{bin_result['confidence_range']:<12} "
                    f"{bin_result['realized_count']:<8.0f} "
                    f"{bin_result['realized_success_rate']:<8.1%} "
                    f"{bin_result['expected_confidence']:<10.1%} "
                    f"{bin_result['calibration_error']:<8.3f}"
                )
        print()

    # Calibration assessment
    if not np.isnan(reliability["expected_calibration_error"]):
        ece = reliability["expected_calibration_error"]

        if ece <= 0.05:
            print("✅ EXCELLENT CALIBRATION: Confidence scores well-calibrated")
        elif ece <= 0.10:
            print("🟢 GOOD CALIBRATION: Minor calibration issues")
        elif ece <= 0.15:
            print("🟡 ACCEPTABLE CALIBRATION: Some calibration drift")
        else:
            print("🔴 POOR CALIBRATION: Significant calibration issues")


def main():
    """Main entry point for calibration script"""

    parser = argparse.ArgumentParser(
        description="Calibration Check - Confidence vs Success Rate Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibration.py --input pred_vs_real_30d.csv
  python scripts/calibration.py --input merged_data.csv --confidence conf_168h
  python scripts/calibration.py --input data.csv --bins 0.8 0.85 0.9 0.95 1.0
        """,
    )

    parser.add_argument(
        "--input", required=True, help="Path to merged predictions+reality CSV file"
    )

    parser.add_argument(
        "--confidence", default="conf_720h", help="Confidence column name (default: conf_720h)"
    )

    parser.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=[0.8, 0.9, 1.01],
        help="Confidence bin edges (default: 0.8 0.9 1.01)",
    )

    parser.add_argument(
        "--output",
        default="logs/calibration",
        help="Output directory for results (default: logs/calibration)",
    )

    args = parser.parse_args()

    try:
        # Load merged data
        print(f"📂 Loading merged data from: {args.input}")
        df = pd.read_csv(args.input, parse_dates=["timestamp"])
        print(f"   Loaded {len(df)} prediction-reality pairs")

        # Validate required columns
        if args.confidence not in df.columns:
            print(f"❌ Error: Confidence column '{args.confidence}' not found")
            return 1

        if "realized" not in df.columns:
            print(f"❌ Error: 'realized' column not found")
            return 1

        # Perform calibration analysis
        print(f"🎯 Performing calibration analysis...")
        results = detailed_calibration_analysis(df, args.confidence)

        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return 2

        # Save results
        results_path = save_calibration_results(results, args.output)

        # Display summary
        print_calibration_summary(results)
        print(f"📝 Full results saved to: {results_path}")

        # Determine exit code based on calibration quality
        quality = assess_calibration_quality(results)

        if quality in ["excellent", "good"]:
            print("✅ CALIBRATION PASSED: Model well-calibrated")
            return 0
        elif quality == "acceptable":
            print("⚠️  CALIBRATION WARNING: Minor calibration issues")
            return 1
        elif quality == "poor":
            print("❌ CALIBRATION FAILED: Significant calibration problems")
            return 2
        else:
            print("❓ CALIBRATION UNKNOWN: Insufficient data for assessment")
            return 3

    except FileNotFoundError:
        print(f"❌ File not found: {args.input}")
        return 4
    except Exception as e:
        print(f"❌ Calibration analysis error: {e}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
