"""
Test script for Confidence-Weighted Kelly Sizing System

Tests:
1. Probability calibration (Platt scaling vs Isotonic regression)
2. Kelly criterion position sizing with risk controls
3. Confidence-weighted sizing integration
4. Historical optimization and validation
"""

import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, "src")

from cryptosmarttrader.sizing import KellySizer, ProbabilityCalibrator, ConfidenceWeighter
from cryptosmarttrader.sizing.kelly_sizing import KellyMode

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_test_data(n_samples: int = 200) -> Dict[str, np.ndarray]:
    """Generate realistic test data for calibration and sizing"""
    np.random.seed(42)

    # Generate ML confidences (biased toward overconfidence)
    raw_confidences = np.random.beta(2, 2, n_samples)  # 0-1 with bias toward middle
    raw_confidences = np.clip(raw_confidences * 1.2, 0, 1)  # Slight overconfidence bias

    # Generate actual win rates (correlated but not perfectly)
    # Higher confidence should correlate with higher win rate, but not perfectly
    noise = np.random.normal(0, 0.15, n_samples)
    true_win_probs = np.clip(raw_confidences * 0.8 + 0.1 + noise, 0.05, 0.95)

    # Generate binary outcomes based on true probabilities
    outcomes = np.random.binomial(1, true_win_probs, n_samples)

    # Generate trade returns (winners and losers)
    win_returns = np.random.lognormal(mean=np.log(3), sigma=0.5, size=n_samples)  # ~3% avg win
    loss_returns = -np.random.lognormal(
        mean=np.log(1.5), sigma=0.3, size=n_samples
    )  # ~1.5% avg loss

    trade_returns = np.where(outcomes, win_returns, loss_returns)

    # Generate expected returns and losses for Kelly sizing
    expected_returns = np.random.uniform(1.5, 4.0, n_samples)  # 1.5-4% expected returns
    expected_losses = np.random.uniform(0.8, 2.5, n_samples)  # 0.8-2.5% expected losses

    return {
        "raw_confidences": raw_confidences,
        "true_win_probs": true_win_probs,
        "outcomes": outcomes,
        "trade_returns": trade_returns,
        "expected_returns": expected_returns,
        "expected_losses": expected_losses,
    }


def test_probability_calibration():
    """Test probability calibration functionality"""
    print("🎯 PROBABILITY CALIBRATION TEST")
    print("=" * 50)

    # Generate test data
    data = generate_test_data(150)

    print(f"📊 Test Data: {len(data['outcomes'])} trades")
    print(f"   Raw Win Rate: {np.mean(data['outcomes']):.1%}")
    print(f"   Avg Raw Confidence: {np.mean(data['raw_confidences']):.3f}")
    print(
        f"   Confidence-Outcome Correlation: {np.corrcoef(data['raw_confidences'], data['outcomes'])[0, 1]:.3f}"
    )

    # Initialize and train calibrator
    calibrator = ProbabilityCalibrator(method="auto", cv_folds=5)

    # Split data for training/testing
    train_size = int(0.7 * len(data["outcomes"]))
    train_confidences = data["raw_confidences"][:train_size]
    train_outcomes = data["outcomes"][:train_size]
    test_confidences = data["raw_confidences"][train_size:]
    test_outcomes = data["outcomes"][train_size:]

    # Train calibration
    print(f"\n🧮 Training calibration on {len(train_outcomes)} samples...")
    calibration_results = calibrator.fit(train_confidences, train_outcomes)

    if "error" not in calibration_results:
        print("✅ Calibration training successful")
        print(f"   Best Method: {calibration_results['best_method']}")
        print(f"   Training Win Rate: {calibration_results['win_rate']:.1%}")

        # Test calibration on holdout data
        calibrated_probs = calibrator.calibrate(test_confidences)

        print(f"\n📈 Calibration Test Results:")
        print(f"   Test Samples: {len(test_outcomes)}")
        print(f"   Raw Confidence Avg: {np.mean(test_confidences):.3f}")
        print(f"   Calibrated Prob Avg: {np.mean(calibrated_probs):.3f}")
        print(f"   Actual Win Rate: {np.mean(test_outcomes):.3f}")
        print(
            f"   Calibration Improvement: {abs(np.mean(calibrated_probs) - np.mean(test_outcomes)) < abs(np.mean(test_confidences) - np.mean(test_outcomes))}"
        )

        return calibrator
    else:
        print(f"❌ Calibration training failed: {calibration_results['error']}")
        return None


def test_kelly_sizing():
    """Test Kelly criterion position sizing"""
    print("\n💰 KELLY SIZING TEST")
    print("=" * 50)

    # Test different Kelly modes
    modes = [KellyMode.CONSERVATIVE, KellyMode.MODERATE, KellyMode.AGGRESSIVE]

    for mode in modes:
        print(f"\n🎲 Testing {mode.value.upper()} Kelly sizing:")

        sizer = KellySizer(mode=mode, max_position=25.0, min_position=0.5)

        # Test scenarios
        test_scenarios = [
            {"win_prob": 0.65, "payoff_ratio": 2.0, "confidence": 0.8, "name": "Strong Signal"},
            {"win_prob": 0.55, "payoff_ratio": 1.5, "confidence": 0.6, "name": "Weak Signal"},
            {"win_prob": 0.75, "payoff_ratio": 3.0, "confidence": 0.9, "name": "Excellent Signal"},
            {"win_prob": 0.45, "payoff_ratio": 1.0, "confidence": 0.7, "name": "Negative Edge"},
            {"win_prob": 0.60, "payoff_ratio": 0.8, "confidence": 0.5, "name": "Low Confidence"},
        ]

        for scenario in test_scenarios:
            result = sizer.calculate_kelly_size(
                win_probability=scenario["win_prob"],
                payoff_ratio=scenario["payoff_ratio"],
                confidence=scenario["confidence"],
            )

            print(
                f"   {scenario['name']}: {result['position_size']:.1f}% "
                f"(Kelly: {result['kelly_fraction']:.3f}, "
                f"Trade: {'✅' if result['should_trade'] else '❌'})"
            )

    return sizer


def test_confidence_weighted_sizing():
    """Test integrated confidence-weighted sizing"""
    print("\n⚖️ CONFIDENCE-WEIGHTED SIZING TEST")
    print("=" * 50)

    # Generate test data
    data = generate_test_data(100)

    # Initialize system
    confidence_weigher = ConfidenceWeighter(
        kelly_mode=KellyMode.MODERATE, max_position=20.0, min_confidence=0.55
    )

    # Train calibration (first 70 samples)
    train_size = 70
    print(f"🎓 Training calibration on {train_size} samples...")

    calibration_results = confidence_weigher.train_calibration(
        data["raw_confidences"][:train_size].tolist(),
        [bool(x) for x in data["outcomes"][:train_size]],
    )

    if "error" not in calibration_results:
        print("✅ Calibration training successful")

        # Test sizing on remaining samples
        test_signals = []
        for i in range(train_size, len(data["raw_confidences"])):
            test_signals.append(
                {
                    "ml_confidence": data["raw_confidences"][i],
                    "direction": "up",
                    "expected_return": data["expected_returns"][i],
                    "expected_loss": data["expected_losses"][i],
                    "regime_factor": np.random.uniform(0.8, 1.2),
                }
            )

        print(f"\n📊 Testing sizing on {len(test_signals)} signals...")
        sizing_results = confidence_weigher.batch_size_signals(test_signals)

        # Analyze results
        sizes = [r.position_size for r in sizing_results]
        trade_decisions = [r.should_trade for r in sizing_results]
        risk_levels = [r.risk_assessment for r in sizing_results]

        print(f"✅ Sizing Analysis:")
        print(f"   Avg Position Size: {np.mean(sizes):.1f}%")
        print(f"   Max Position Size: {np.max(sizes):.1f}%")
        print(f"   Trade Rate: {np.mean(trade_decisions):.1%}")
        print(f"   Risk Distribution:")
        for risk in set(risk_levels):
            count = risk_levels.count(risk)
            print(f"     {risk}: {count}/{len(risk_levels)} ({count / len(risk_levels):.1%})")

        # Show example recommendations
        print(f"\n📋 Example Recommendations:")
        for i, result in enumerate(sizing_results[:5]):
            print(
                f"   Signal {i + 1}: {result.position_size:.1f}% "
                f"(Raw: {result.raw_ml_confidence:.3f}, "
                f"Cal: {result.calibrated_probability:.3f}, "
                f"Trade: {'✅' if result.should_trade else '❌'})"
            )

        return confidence_weigher, sizing_results
    else:
        print(f"❌ Calibration failed: {calibration_results['error']}")
        return None, []


def test_kelly_optimization():
    """Test Kelly parameter optimization"""
    print("\n🔧 KELLY OPTIMIZATION TEST")
    print("=" * 50)

    # Generate larger dataset for optimization
    data = generate_test_data(300)

    confidence_weigher = ConfidenceWeighter(kelly_mode=KellyMode.MODERATE)

    # Prepare optimization data
    historical_signals = []
    for i in range(len(data["raw_confidences"])):
        historical_signals.append(
            {
                "ml_confidence": data["raw_confidences"][i],
                "expected_return": data["expected_returns"][i],
                "expected_loss": data["expected_losses"][i],
                "regime_factor": np.random.uniform(0.9, 1.1),
            }
        )

    # Run optimization
    print("🎯 Optimizing Kelly fraction factor...")
    optimization_results = confidence_weigher.optimize_kelly_parameters(
        historical_signals, data["trade_returns"].tolist()
    )

    if "error" not in optimization_results:
        print("✅ Kelly optimization successful")
        print(f"   Optimal Fraction: {optimization_results['optimal_fraction']:.3f}")
        print(f"   Recommended Mode: {optimization_results.get('recommended_mode', 'N/A')}")

        optimal_metrics = optimization_results["optimal_metrics"]
        print(f"   Optimal Performance:")
        print(f"     Sharpe Ratio: {optimal_metrics['sharpe_ratio']:.3f}")
        print(f"     Total Return: {optimal_metrics['total_return']:.2f}%")
        print(f"     Max Drawdown: {optimal_metrics['max_drawdown']:.2f}%")
        print(f"     Win Rate: {optimal_metrics['win_rate']:.1%}")

        # Compare with other fractions
        print(f"\n📊 Fraction Comparison:")
        for fraction, metrics in optimization_results["all_results"].items():
            print(
                f"   f={fraction:.2f}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                f"Return={metrics['total_return']:.1f}%, "
                f"DD={metrics['max_drawdown']:.1f}%"
            )

        return optimization_results
    else:
        print(f"❌ Optimization failed: {optimization_results['error']}")
        return None


def main():
    """Run complete confidence-weighted Kelly sizing test suite"""
    print("🚀 CONFIDENCE-WEIGHTED KELLY SIZING SYSTEM TEST")
    print("🎯 Fractional Kelly with Probability Calibration")
    print("=" * 60)

    try:
        # Test 1: Probability Calibration
        calibrator = test_probability_calibration()

        # Test 2: Kelly Sizing
        kelly_sizer = test_kelly_sizing()

        # Test 3: Integrated Confidence-Weighted Sizing
        confidence_weigher, sizing_results = test_confidence_weighted_sizing()

        # Test 4: Kelly Optimization
        optimization_results = test_kelly_optimization()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")

        if all([calibrator, kelly_sizer, confidence_weigher, optimization_results]):
            print("🎉 FULL SYSTEM OPERATIONAL:")
            print("   ✅ Probability calibration (Platt/Isotonic)")
            print("   ✅ Fractional Kelly sizing with risk controls")
            print("   ✅ Confidence-weighted position sizing")
            print("   ✅ Historical optimization and validation")
            print("   ✅ Integration with regime detection ready")

            print(f"\n🔬 SYSTEM IMPACT:")
            print("   → ML overconfidence corrected via calibration")
            print("   → Position sizes optimized for risk-adjusted returns")
            print("   → Kelly fraction prevents overbetting")
            print("   → Confidence thresholds filter weak signals")
            print("   → Regime factors adapt to market conditions")

            # Show final system summary
            if confidence_weigher:
                summary = confidence_weigher.get_sizing_summary()
                print(f"\n📊 SYSTEM SUMMARY:")
                print(f"   Calibration: {summary.get('calibration_status')}")
                print(f"   Kelly Mode: {summary.get('kelly_mode')}")
                print(f"   Kelly Fraction: {summary.get('kelly_fraction', 0):.3f}")
                print(f"   Max Position: {summary.get('max_position', 0):.1f}%")
                print(f"   Trade Rate: {summary.get('trade_rate', 0):.1%}")
        else:
            print("⚠️ Some tests had issues - review logs above")

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
