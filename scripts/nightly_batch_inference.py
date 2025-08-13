#!/usr/bin/env python3
"""
Nightly Batch Inference Runner
Automated script for running multi-horizon batch inference on all coins
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.batch_inference_engine import (
    get_batch_inference_engine,
    BatchInferenceConfig,
    HorizonType,
    InferenceStatus,
)
from core.logging_manager import get_logger


async def run_nightly_batch_inference(
    horizons: list = None, batch_size: int = 100, dry_run: bool = False, target_coins: set = None
):
    """Run nightly batch inference with specified parameters"""

    logger = get_logger()

    print(f"🚀 Starting Nightly Batch Inference")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if dry_run:
        print("🧪 DRY RUN MODE - No actual inference will be performed")
        print()

    try:
        # Configure batch inference
        horizons = horizons or [HorizonType.H1, HorizonType.H24, HorizonType.D7]

        config = BatchInferenceConfig(
            horizons=horizons,
            batch_size=batch_size,
            max_parallel_coins=50,
            model_timeout_seconds=30,
            feature_extraction_timeout=60,
            atomic_write_enabled=True,
            retry_attempts=3,
            completeness_threshold=0.8,
        )

        print(f"📊 BATCH CONFIGURATION:")
        print(f"   Horizons: {[h.value for h in horizons]}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Max Parallel: {config.max_parallel_coins}")
        print(f"   Completeness Threshold: {config.completeness_threshold:.0%}")
        if target_coins:
            print(f"   Target Coins: {len(target_coins)} specified")
        else:
            print(f"   Target Coins: All available coins")
        print()

        if dry_run:
            print("✅ Dry run completed - configuration validated")
            return 0

        # Initialize batch inference engine
        batch_engine = get_batch_inference_engine(config)

        # Run batch inference
        print("🔄 Starting batch inference...")
        result = await batch_engine.run_batch_inference(target_coins)

        # Print results
        print(f"📈 BATCH INFERENCE RESULTS:")
        print(f"   Batch ID: {result.batch_id}")
        print(f"   Status: {result.status.value.upper()}")
        print(f"   Total Coins: {result.total_coins}")
        print(f"   Successful: {result.successful_predictions}")
        print(f"   Failed: {result.failed_predictions}")
        print(
            f"   Success Rate: {result.successful_predictions / result.total_coins * 100:.1f}%"
            if result.total_coins > 0
            else "N/A"
        )
        print(f"   Execution Time: {result.execution_time_seconds:.2f}s")
        print(f"   Average Latency: {result.average_latency_ms:.1f}ms")
        print()

        # Show completeness statistics
        if result.completeness_stats:
            print(f"📊 DATA COMPLETENESS STATS:")
            print(f"   Mean: {result.completeness_stats.get('mean', 0):.2f}")
            print(f"   Median: {result.completeness_stats.get('median', 0):.2f}")
            print(f"   Min: {result.completeness_stats.get('min', 0):.2f}")
            print(f"   Max: {result.completeness_stats.get('max', 0):.2f}")
            print()

        # Show horizon breakdown
        if result.predictions:
            print(f"🎯 PREDICTIONS BY HORIZON:")

            horizon_stats = {}
            for prediction in result.predictions[:10]:  # Show first 10
                for horizon, pred_value in prediction.predictions.items():
                    if horizon not in horizon_stats:
                        horizon_stats[horizon] = []
                    horizon_stats[horizon].append(pred_value)

            for horizon, values in horizon_stats.items():
                print(
                    f"   {horizon.value}: {len(values)} predictions (avg: {sum(values) / len(values):.3f})"
                )

            print()

        # Show top predictions
        if result.predictions:
            print(f"🔝 TOP PREDICTIONS (by 24h horizon):")

            # Sort by 24h prediction value
            sorted_predictions = sorted(
                result.predictions,
                key=lambda p: p.predictions.get(HorizonType.H24, 0),
                reverse=True,
            )

            for i, prediction in enumerate(sorted_predictions[:10], 1):
                h24_pred = prediction.predictions.get(HorizonType.H24, 0)
                h24_conf = prediction.confidence_scores.get(HorizonType.H24, 0)
                print(f"   {i:2d}. {prediction.symbol:<12} {h24_pred:+.3f} (conf: {h24_conf:.2f})")

            print()

        # Show error summary
        if result.error_summary:
            print(f"⚠️  ERROR SUMMARY:")
            for error_type, count in result.error_summary.items():
                print(f"   {error_type}: {count}")
            print()

        # Determine exit code
        if result.status == InferenceStatus.COMPLETED:
            print("✅ Batch inference completed successfully")
            return 0
        elif result.status == InferenceStatus.PARTIAL:
            print("⚠️  Batch inference completed with some failures")
            return 0
        else:
            print("❌ Batch inference failed")
            return 1

    except Exception as e:
        logger.error(f"Nightly batch inference failed: {e}")
        print(f"❌ ERROR: Batch inference failed: {e}")
        return 2


async def run_4hourly_batch_inference():
    """Run 4-hourly batch inference with optimized configuration"""

    print(f"⚡ Starting 4-Hourly Batch Inference")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Optimized config for 4-hourly runs
    config = BatchInferenceConfig(
        horizons=[HorizonType.H1, HorizonType.H4, HorizonType.H24],  # Shorter horizons
        batch_size=150,  # Larger batches for efficiency
        max_parallel_coins=75,  # More parallelism
        model_timeout_seconds=20,  # Faster timeout
        completeness_threshold=0.75,  # Slightly lower threshold
    )

    batch_engine = get_batch_inference_engine(config)
    result = await batch_engine.run_batch_inference()

    print(f"📈 4-HOURLY BATCH RESULTS:")
    print(f"   Status: {result.status.value.upper()}")
    print(f"   Processed: {result.successful_predictions}/{result.total_coins}")
    print(f"   Duration: {result.execution_time_seconds:.1f}s")
    print()

    return 0 if result.status in [InferenceStatus.COMPLETED, InferenceStatus.PARTIAL] else 1


def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Basic nightly run:")
    print("  python scripts/nightly_batch_inference.py")
    print()
    print("Custom horizons:")
    print("  python scripts/nightly_batch_inference.py --horizons 1h 24h 7d")
    print()
    print("Larger batch size:")
    print("  python scripts/nightly_batch_inference.py --batch-size 200")
    print()
    print("Dry run (test configuration):")
    print("  python scripts/nightly_batch_inference.py --dry-run")
    print()
    print("4-hourly optimized run:")
    print("  python scripts/nightly_batch_inference.py --mode 4hourly")
    print()
    print("Specific coins only:")
    print("  python scripts/nightly_batch_inference.py --coins BTC/USD ETH/USD ADA/USD")
    print()
    print("Automated scheduling (cron):")
    print("  # Nightly at 2:00 AM")
    print("  0 2 * * * /usr/bin/python3 /path/to/scripts/nightly_batch_inference.py")
    print("  # Every 4 hours")
    print(
        "  0 */4 * * * /usr/bin/python3 /path/to/scripts/nightly_batch_inference.py --mode 4hourly"
    )
    print()


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Nightly Batch Inference - Multi-horizon ML predictions for all coins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/nightly_batch_inference.py                     # Full nightly run
  python scripts/nightly_batch_inference.py --dry-run           # Test configuration
  python scripts/nightly_batch_inference.py --mode 4hourly      # 4-hourly optimized
  python scripts/nightly_batch_inference.py --batch-size 200    # Larger batches
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["nightly", "4hourly"],
        default="nightly",
        help="Inference mode (default: nightly)",
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        choices=["1h", "4h", "24h", "7d", "30d"],
        default=["1h", "24h", "7d"],
        help="Prediction horizons (default: 1h 24h 7d)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing (default: 100)"
    )

    parser.add_argument("--coins", nargs="+", help="Specific coins to process (default: all coins)")

    parser.add_argument(
        "--dry-run", action="store_true", help="Test configuration without running inference"
    )

    parser.add_argument("--examples", action="store_true", help="Show usage examples")

    args = parser.parse_args()

    if args.examples:
        print_usage_examples()
        return 0

    # Convert string horizons to enum
    horizon_map = {
        "1h": HorizonType.H1,
        "4h": HorizonType.H4,
        "24h": HorizonType.H24,
        "7d": HorizonType.D7,
        "30d": HorizonType.D30,
    }

    horizons = [horizon_map[h] for h in args.horizons]
    target_coins = set(args.coins) if args.coins else None

    # Run appropriate mode
    if args.mode == "4hourly":
        exit_code = await run_4hourly_batch_inference()
    else:
        exit_code = await run_nightly_batch_inference(
            horizons=horizons,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            target_coins=target_coins,
        )

    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Batch inference interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
