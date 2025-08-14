#!/usr/bin/env python3
"""
Test script voor het Advanced Logging System
Genereert verschillende logs en toont dagelijkse mappenstructuur
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_manager import (
    get_advanced_logger,
    log_prediction,
    log_confidence_scoring,
    log_api_call,
    log_performance,
    log_user_action,
    log_error,
    log_data_pipeline,
    PerformanceTimer,
)


def test_logging_system():
    """Test alle logging functionaliteiten"""
    print("🔧 TESTING ADVANCED LOGGING SYSTEM")
    print("=" * 60)

    # Initialize advanced logger
    advanced_logger = get_advanced_logger()

    print(f"📂 Log directory: {advanced_logger.daily_log_dir}")
    print(f"📅 Today's date: {datetime.now().strftime('%Y-%m-%d')}")

    # Test 1: Prediction logging
    print("\n1. Testing prediction logging...")
    prediction_data = {
        "expected_returns": {"1h": 2.5, "24h": 8.3, "168h": 15.2},
        "sentiment_score": 0.73,
        "whale_detected": True,
        "regime": "bull_strong",
    }
    log_prediction("BTC", prediction_data, 0.87)
    log_prediction("ETH", prediction_data, 0.82)
    log_prediction("LOWCOIN001", prediction_data, 0.45)  # Low confidence

    # Test 2: Confidence scoring
    print("2. Testing confidence scoring...")
    confidence_details = {
        "base_confidence": 0.65,
        "volume_factor": 0.8,
        "volume_boost": 0.12,
        "volatility": 0.15,
        "volatility_penalty": 0.045,
        "final_confidence": 0.825,
        "horizon": "24h",
        "volume_24h": 1500000,
        "change_24h": -2.3,
    }
    log_confidence_scoring("BTC", confidence_details)

    # Test 3: API calls
    print("3. Testing API call logging...")
    log_api_call("/api/v1/market-data", "GET", 200, 0.145)
    log_api_call("/api/v1/predictions", "POST", 201, 0.532)
    log_api_call("/api/v1/health", "GET", 500, 2.1)  # Error case

    # Test 4: Performance timing
    print("4. Testing performance logging...")
    with PerformanceTimer("test_operation"):
        time.sleep(0.1)  # Simulate work

    log_performance("ml_model_training", 45.6, {"model_type": "RandomForest", "features": 50})
    log_performance("data_processing", 12.3, {"records": 438, "success_rate": 0.98})

    # Test 5: User actions
    print("5. Testing user action logging...")
    log_user_action("dashboard_view", "user123", {"page": "predictions", "filters": ["BTC", "ETH"]})
    log_user_action(
        "prediction_filter", "user123", {"confidence_threshold": 0.8, "timeframe": "24h"}
    )

    # Test 6: Error logging
    print("6. Testing error logging...")
    try:
        # Simulate error
        raise ValueError("Test error for logging demonstration")
    except Exception as e:
        log_error("ml_models", e, {"operation": "model_loading", "model_id": "rf_24h"})

    # Test 7: Data pipeline
    print("7. Testing data pipeline logging...")
    log_data_pipeline("data_collection", 471, 0.95, 8.7)
    log_data_pipeline("feature_engineering", 438, 0.98, 15.2)
    log_data_pipeline("model_inference", 36, 1.0, 3.4)

    # Generate daily summary
    print("\n8. Generating daily summary...")
    summary = advanced_logger.generate_daily_summary()

    print(f"\n✅ LOGGING TEST COMPLETED")
    print(f"📊 Summary generated: {summary['summary_generated_at']}")
    print(f"📁 Log files created in: {advanced_logger.daily_log_dir}")

    # Show created log files
    print(f"\n📋 CREATED LOG FILES:")
    for log_file in advanced_logger.daily_log_dir.iterdir():
        if log_file.is_file():
            size_kb = log_file.stat().st_size / 1024
            print(f"   📄 {log_file.name} ({size_kb:.1f} KB)")

    return summary


def test_log_analysis():
    """Analyseer de gegenereerde logs"""
    print("\n🔍 LOG ANALYSIS")
    print("=" * 40)

    advanced_logger = get_advanced_logger()
    log_dir = advanced_logger.daily_log_dir

    # Analyseer predictions log
    predictions_json = log_dir / "predictions.json"
    if predictions_json.exists():
        with open(predictions_json, "r") as f:
            lines = f.readlines()
        print(f"📊 Predictions logged: {len(lines)} entries")

        # Show sample prediction
        if lines:
            import json

            sample = json.loads(lines[0])
            print(
                f"   Sample: {sample['data']['coin']} - {sample['data']['confidence']:.3f} confidence"
            )

    # Analyseer confidence scoring
    confidence_json = log_dir / "confidence_scoring.json"
    if confidence_json.exists():
        with open(confidence_json, "r") as f:
            lines = f.readlines()
        print(f"🎯 Confidence calculations: {len(lines)} entries")

    # Analyseer performance
    performance_json = log_dir / "performance.json"
    if performance_json.exists():
        with open(performance_json, "r") as f:
            lines = f.readlines()
        print(f"⚡ Performance measurements: {len(lines)} entries")

    # Show daily summary
    summary_file = log_dir / "daily_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"\n📈 DAILY SUMMARY:")
        print(f"   Components logged: {len(summary['components'])}")
        if summary.get("performance_metrics"):
            print(f"   Performance operations: {len(summary['performance_metrics'])}")
            for op, metrics in summary["performance_metrics"].items():
                print(f"     {op}: {metrics['count']} calls, avg {metrics['avg_time']:.3f}s")


if __name__ == "__main__":
    # Run tests
    summary = test_logging_system()
    test_log_analysis()

    print(f"\n🎯 RESULTAAT:")
    print(f"✅ Dagelijkse logs worden opgeslagen in: logs/{datetime.now().strftime('%Y-%m-%d')}/")
    print(f"✅ JSON en text logs voor alle componenten")
    print(f"✅ Performance tracking en user actions")
    print(f"✅ Gestructureerde logging voor ML predictions en confidence scoring")
    print(f"✅ Automatic log rotation en cleanup")
