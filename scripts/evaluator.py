#!/usr/bin/env python3
"""
Performance Evaluator - Realized Returns & Metrics
Implements precision@K, hit-rate, MAE, and Sharpe ratio evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse
import json
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger


def realized_forward_return(prices: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    Calculate realized forward returns for given horizon

    Args:
        prices: DataFrame with columns [timestamp, coin, close] sampled hourly
        horizon_hours: Forward-looking horizon in hours

    Returns:
        DataFrame with added realized return column
    """
    logger = get_logger()

    if prices.empty:
        logger.warning("Empty prices DataFrame provided")
        return pd.DataFrame()

    required_cols = ["timestamp", "coin", "close"]
    missing_cols = [col for col in required_cols if col not in prices.columns]

    if missing_cols:
        logger.error(f"Missing required columns in prices: {missing_cols}")
        return pd.DataFrame()

    logger.info(f"Calculating {horizon_hours}h forward returns for {len(prices)} price records")

    # Sort by coin and timestamp
    prices = prices.sort_values(["coin", "timestamp"]).copy()

    # Calculate forward price (price after horizon_hours)
    prices["fwd_price"] = prices.groupby("coin")["close"].shift(-horizon_hours)

    # Calculate realized return
    prices[f"ret_{horizon_hours}h"] = prices["fwd_price"] / prices["close"] - 1

    # Clean up
    prices = prices.drop(columns=["fwd_price"])

    # Count valid returns
    valid_returns = prices[f"ret_{horizon_hours}h"].notna().sum()
    logger.info(f"Generated {valid_returns} valid {horizon_hours}h returns")

    return prices


def merge_predictions_reality(
    pred: pd.DataFrame, real: pd.DataFrame, horizon_hours: int, pred_col: str, conf_col: str
) -> pd.DataFrame:
    """
    Merge predictions with realized returns

    Args:
        pred: Predictions DataFrame
        real: Reality DataFrame with realized returns
        horizon_hours: Horizon in hours
        pred_col: Prediction column name
        conf_col: Confidence column name

    Returns:
        Merged DataFrame aligned on (coin, timestamp)
    """
    logger = get_logger()

    ret_col = f"ret_{horizon_hours}h"

    if ret_col not in real.columns:
        logger.error(f"Realized return column {ret_col} not found in reality DataFrame")
        return pd.DataFrame()

    logger.info(f"Merging predictions with reality for {horizon_hours}h horizon")

    # Merge on coin and timestamp
    df = pred.merge(real[["coin", "timestamp", ret_col]], on=["coin", "timestamp"], how="inner")

    # Rename realized return column
    df = df.rename(columns={ret_col: "realized"})

    # Select relevant columns
    df = df[["coin", "timestamp", pred_col, conf_col, "realized"]]

    # Remove rows with NaN values
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)

    logger.info(f"Merged dataset: {final_count}/{initial_count} complete records")

    return df


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    """
    Calculate Precision@K metric

    Args:
        df: DataFrame with predictions and realized returns
        k: Number of top recommendations to evaluate

    Returns:
        Precision@K score (0-1)
    """
    if df.empty or k <= 0:
        return np.nan

    pred_col = df.columns[2]  # Third column is prediction

    precisions = []

    # Calculate precision for each timestamp (run)
    for timestamp, group in df.groupby("timestamp"):
        # Sort by predicted return (descending)
        group_sorted = group.sort_values(pred_col, ascending=False)

        # Take top-k recommendations
        top_k = group_sorted.head(k)

        # Calculate precision: fraction of positive realized returns
        if len(top_k) == k:  # Only count if we have exactly k recommendations
            precision = (top_k["realized"] > 0).mean()
            precisions.append(precision)

    return float(np.mean(precisions)) if precisions else np.nan


def hit_rate(df: pd.DataFrame, conf_threshold: float) -> float:
    """
    Calculate hit rate for high-confidence predictions

    Args:
        df: DataFrame with predictions and realized returns
        conf_threshold: Minimum confidence threshold

    Returns:
        Hit rate (0-1)
    """
    if df.empty:
        return np.nan

    conf_col = df.columns[3]  # Fourth column is confidence

    # Filter for high-confidence predictions
    high_conf = df[df[conf_col] >= conf_threshold]

    if high_conf.empty:
        return np.nan

    # Calculate hit rate: fraction with positive realized returns
    hit_rate_score = (high_conf["realized"] > 0).mean()

    return float(hit_rate_score)


def mae_calibration(df: pd.DataFrame) -> float:
    """
    Calculate Mean Absolute Error for calibration assessment

    Args:
        df: DataFrame with predictions and realized returns

    Returns:
        MAE value
    """
    if df.empty:
        return np.nan

    pred_col = df.columns[2]  # Third column is prediction

    predicted = df[pred_col].values
    realized = df["realized"].values

    mae = np.mean(np.abs(predicted - realized))

    return float(mae)


def sharpe_strategy(df: pd.DataFrame, conf_threshold: float) -> float:
    """
    Calculate Sharpe ratio for equal-weight strategy

    Args:
        df: DataFrame with predictions and realized returns
        conf_threshold: Minimum confidence for inclusion

    Returns:
        Annualized Sharpe ratio
    """
    if df.empty:
        return np.nan

    conf_col = df.columns[3]  # Fourth column is confidence

    # Filter for high-confidence predictions
    high_conf = df[df[conf_col] >= conf_threshold]

    if high_conf.empty:
        return np.nan

    # Aggregate returns per timestamp (equal-weight portfolio)
    portfolio_returns = high_conf.groupby("timestamp")["realized"].mean()

    if len(portfolio_returns) < 2:
        return np.nan

    # Calculate Sharpe ratio
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()

    if std_return == 0 or np.isnan(std_return):
        return np.nan

    # Annualize (assuming daily returns - adjust based on your data frequency)
    sharpe = np.sqrt(365) * mean_return / std_return

    return float(sharpe)


def comprehensive_evaluation(
    df: pd.DataFrame, conf_threshold: float = 0.80, precision_k: int = 5
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of predictions

    Args:
        df: Merged predictions and reality DataFrame
        conf_threshold: Confidence threshold for hit rate and Sharpe
        precision_k: K value for Precision@K

    Returns:
        Dictionary with all evaluation metrics
    """
    logger = get_logger()

    if df.empty:
        logger.warning("Empty DataFrame provided for evaluation")
        return {"error": "No data to evaluate"}

    logger.info(f"Evaluating {len(df)} prediction-reality pairs")

    # Calculate all metrics
    metrics = {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "sample_size": int(len(df)),
        "unique_coins": int(df["coin"].nunique()),
        "unique_timestamps": int(df["timestamp"].nunique()),
        "confidence_threshold": conf_threshold,
        "precision_k": precision_k,
        "metrics": {
            "precision_at_k": precision_at_k(df, precision_k),
            "hit_rate_conf": hit_rate(df, conf_threshold),
            "mae_calibration": mae_calibration(df),
            "sharpe_ratio": sharpe_strategy(df, conf_threshold),
        },
        "data_quality": {
            "positive_predictions": int((df[df.columns[2]] > 0).sum()),
            "negative_predictions": int((df[df.columns[2]] <= 0).sum()),
            "positive_outcomes": int((df["realized"] > 0).sum()),
            "negative_outcomes": int((df["realized"] <= 0).sum()),
            "high_confidence_predictions": int((df[df.columns[3]] >= conf_threshold).sum()),
        },
    }

    # Add derived metrics
    metrics["derived"] = {
        "prediction_bias": float(df[df.columns[2]].mean()),
        "realized_mean": float(df["realized"].mean()),
        "prediction_volatility": float(df[df.columns[2]].std()),
        "realized_volatility": float(df["realized"].std()),
        "correlation": float(df[df.columns[2]].corr(df["realized"])) if len(df) > 1 else np.nan,
    }

    logger.info(
        f"Evaluation completed: P@{precision_k}={metrics['metrics']['precision_at_k']:.3f}, "
        f"Hit Rate={metrics['metrics']['hit_rate_conf']:.3f}, "
        f"MAE={metrics['metrics']['mae_calibration']:.3f}, "
        f"Sharpe={metrics['metrics']['sharpe_ratio']:.3f}"
    )

    return metrics


def save_evaluation_results(results: Dict[str, Any], output_dir: str = "logs/evaluation") -> str:
    """Save evaluation results to timestamped file"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = results["evaluation_timestamp"].replace(":", "-").replace(".", "-")
    filename = f"evaluation_results_{timestamp}.json"
    filepath = output_path / filename

    # Save results
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as latest
    latest_path = output_path / "latest_evaluation_results.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    logger = get_logger()
    logger.info(f"Evaluation results saved: {filepath}")

    return str(filepath)


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print human-readable evaluation summary"""

    if "error" in results:
        print(f"❌ Evaluation Error: {results['error']}")
        return

    metrics = results["metrics"]
    quality = results["data_quality"]
    derived = results["derived"]

    print(f"📊 PERFORMANCE EVALUATION RESULTS")
    print(f"📅 Timestamp: {results['evaluation_timestamp']}")
    print("=" * 60)

    # Core metrics
    print(f"🎯 CORE PERFORMANCE METRICS:")

    # Precision@K
    p_at_k = metrics["precision_at_k"]
    p_icon = "✅" if not np.isnan(p_at_k) and p_at_k >= 0.60 else "❌"
    print(f"   {p_icon} Precision@{results['precision_k']}: {p_at_k:.1%} (target: ≥60%)")

    # Hit Rate
    hit_rate_val = metrics["hit_rate_conf"]
    hr_icon = "✅" if not np.isnan(hit_rate_val) and hit_rate_val >= 0.55 else "❌"
    print(
        f"   {hr_icon} Hit Rate (≥{results['confidence_threshold']:.0%}): {hit_rate_val:.1%} (target: ≥55%)"
    )

    # MAE
    mae_val = metrics["mae_calibration"]
    mae_median = abs(derived["prediction_bias"]) if derived["prediction_bias"] != 0 else 0.01
    mae_ratio = mae_val / mae_median if mae_median > 0 else float("inf")
    mae_icon = "✅" if not np.isnan(mae_val) and mae_ratio <= 0.25 else "❌"
    print(f"   {mae_icon} MAE: {mae_val:.3f} (ratio: {mae_ratio:.2f}, target: ≤0.25×median)")

    # Sharpe
    sharpe_val = metrics["sharpe_ratio"]
    sharpe_icon = "✅" if not np.isnan(sharpe_val) and sharpe_val >= 1.0 else "❌"
    print(f"   {sharpe_icon} Sharpe Ratio: {sharpe_val:.2f} (target: ≥1.0)")
    print()

    # Data quality
    print(f"📋 DATA QUALITY:")
    print(f"   Sample size: {results['sample_size']:,}")
    print(f"   Unique coins: {quality['unique_coins']:,}")
    print(f"   Unique timestamps: {quality['unique_timestamps']:,}")
    print(f"   High-confidence predictions: {quality['high_confidence_predictions']:,}")
    print(
        f"   Positive predictions: {quality['positive_predictions']:,} ({quality['positive_predictions'] / results['sample_size']:.1%})"
    )
    print(
        f"   Positive outcomes: {quality['positive_outcomes']:,} ({quality['positive_outcomes'] / results['sample_size']:.1%})"
    )
    print()

    # Derived insights
    print(f"📈 DERIVED INSIGHTS:")
    print(f"   Prediction bias: {derived['prediction_bias']:+.3f}")
    print(f"   Prediction-reality correlation: {derived['correlation']:.3f}")
    print(f"   Prediction volatility: {derived['prediction_volatility']:.3f}")
    print(f"   Realized volatility: {derived['realized_volatility']:.3f}")
    print()


def main():
    """Main entry point for evaluation script"""

    parser = argparse.ArgumentParser(
        description="Performance Evaluator - Realized Returns & Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluator.py --predictions pred.csv --prices prices.csv --horizon 720
  python scripts/evaluator.py --predictions pred.csv --prices prices.csv --horizon 168 --confidence 0.85
        """,
    )

    parser.add_argument("--predictions", required=True, help="Path to predictions CSV file")

    parser.add_argument(
        "--prices", required=True, help="Path to prices CSV file (hourly OHLC data)"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Prediction horizon in hours (e.g., 168 for 7d, 720 for 30d)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Confidence threshold for hit rate and Sharpe (default: 0.80)",
    )

    parser.add_argument(
        "--precision-k", type=int, default=5, help="K value for Precision@K metric (default: 5)"
    )

    parser.add_argument(
        "--output",
        default="logs/evaluation",
        help="Output directory for results (default: logs/evaluation)",
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"📂 Loading predictions from: {args.predictions}")
        pred_df = pd.read_csv(args.predictions, parse_dates=["timestamp"])
        print(f"   Loaded {len(pred_df)} prediction records")

        print(f"📂 Loading prices from: {args.prices}")
        prices_df = pd.read_csv(args.prices, parse_dates=["timestamp"])
        print(f"   Loaded {len(prices_df)} price records")

        # Calculate realized returns
        print(f"🔄 Calculating {args.horizon}h forward returns...")
        prices_with_returns = realized_forward_return(prices_df, args.horizon)

        # Determine column names
        pred_col = f"pred_{args.horizon}h"
        conf_col = f"conf_{args.horizon}h"

        if pred_col not in pred_df.columns or conf_col not in pred_df.columns:
            print(f"❌ Error: Missing columns {pred_col} or {conf_col} in predictions file")
            return 1

        # Merge predictions with reality
        print(f"🔗 Merging predictions with realized returns...")
        merged_df = merge_predictions_reality(
            pred_df, prices_with_returns, args.horizon, pred_col, conf_col
        )

        if merged_df.empty:
            print(f"❌ Error: No matching data after merge")
            return 1

        # Evaluate performance
        print(f"📊 Evaluating performance...")
        results = comprehensive_evaluation(
            merged_df, conf_threshold=args.confidence, precision_k=args.precision_k
        )

        # Save results
        results_path = save_evaluation_results(results, args.output)

        # Display summary
        print_evaluation_summary(results)
        print(f"📝 Full results saved to: {results_path}")

        # Determine exit code based on performance
        metrics = results["metrics"]

        # Count passed metrics
        passed_count = 0
        total_count = 4

        if not np.isnan(metrics["precision_at_k"]) and metrics["precision_at_k"] >= 0.60:
            passed_count += 1
        if not np.isnan(metrics["hit_rate_conf"]) and metrics["hit_rate_conf"] >= 0.55:
            passed_count += 1
        if not np.isnan(metrics["sharpe_ratio"]) and metrics["sharpe_ratio"] >= 1.0:
            passed_count += 1

        # MAE check (more complex)
        mae_val = metrics["mae_calibration"]
        pred_bias = results["derived"]["prediction_bias"]
        mae_median = abs(pred_bias) if pred_bias != 0 else 0.01
        if not np.isnan(mae_val) and (mae_val / mae_median) <= 0.25:
            passed_count += 1

        if passed_count == total_count:
            print(f"✅ EVALUATION PASSED: All {total_count} metrics meet targets")
            return 0
        elif passed_count >= total_count // 2:
            print(f"⚠️  EVALUATION WARNING: {passed_count}/{total_count} metrics meet targets")
            return 1
        else:
            print(f"❌ EVALUATION FAILED: Only {passed_count}/{total_count} metrics meet targets")
            return 2

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 3
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
