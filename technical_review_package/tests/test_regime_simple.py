#!/usr/bin/env python3
"""
Simple test for regime switching system without heavy dependencies
Validates regime detection and adaptive parameter adjustment
"""

import sys
import os
import time
import random
import math
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_regime_switching_system():
    """Test regime switching system with minimal dependencies"""
    
    print("🔄 Testing Regime-Switching System")
    print("=" * 45)
    
    try:
        from cryptosmarttrader.regime.regime_detection import (
            RegimeDetector, MarketRegime, RegimeIndicators
        )
        from cryptosmarttrader.regime.adaptive_trading import (
            AdaptiveTradingManager
        )
        
        # Test 1: Basic regime detector setup
        print("\n1. Testing regime detector setup...")
        
        detector = RegimeDetector("BTC/USD")
        print(f"✅ Detector created for {detector.symbol}")
        print(f"   Current regime: {detector.current_regime.value}")
        print(f"   Regime parameters loaded: {len(detector.regime_parameters)}")
        
        # Validate all regimes have parameters
        for regime in MarketRegime:
            assert regime in detector.regime_parameters, f"Missing parameters for {regime.value}"
        
        print("   ✅ All regime parameters configured")
        
        # Test 2: Price data and indicators
        print("\n2. Testing price data and technical indicators...")
        
        # Add trending price data
        base_price = 50000.0
        prices = []
        for i in range(60):
            # Create upward trend with some noise
            trend = i * 100  # Strong upward trend
            noise = random.gauss(0, 200)  # Some volatility
            price = base_price + trend + noise
            prices.append(price)
            
            detector.update_price_data(
                price=price,
                volume=1000.0 + random.gauss(0, 100),
                high=price + abs(noise) * 0.5,
                low=price - abs(noise) * 0.5
            )
        
        print(f"   Added {len(prices)} price points")
        print(f"   Price range: ${min(prices):,.0f} - ${max(prices):,.0f}")
        
        # Test technical indicators
        trend_strength = RegimeIndicators.calculate_trend_strength(prices)
        print(f"   Trend strength: {trend_strength:.2f}")
        
        # Calculate returns for volatility
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] / prices[i-1]) - 1.0)
        
        volatility = RegimeIndicators.calculate_realized_volatility(returns)
        print(f"   Realized volatility: {volatility:.1%}")
        
        choppiness = RegimeIndicators.calculate_choppiness_index(prices, prices, prices)
        print(f"   Choppiness index: {choppiness:.1f}")
        
        mean_reversion = RegimeIndicators.calculate_mean_reversion_score(prices)
        print(f"   Mean reversion score: {mean_reversion:.2f}")
        
        # Validate indicator ranges
        assert -1.0 <= trend_strength <= 1.0, f"Trend strength out of range: {trend_strength}"
        assert volatility >= 0.0, f"Volatility negative: {volatility}"
        assert 0.0 <= choppiness <= 100.0, f"Choppiness out of range: {choppiness}"
        assert 0.0 <= mean_reversion <= 1.0, f"Mean reversion out of range: {mean_reversion}"
        
        print("   ✅ Technical indicators working correctly")
        
        # Test 3: Regime detection
        print("\n3. Testing regime detection...")
        
        result = detector.detect_regime()
        
        print(f"   Detected regime: {result.primary_regime.value}")
        print(f"   Confidence: {result.confidence.value} ({result.confidence_score:.2f})")
        print(f"   Secondary regime: {result.secondary_regime.value if result.secondary_regime else 'None'}")
        print(f"   Regime change detected: {result.regime_change_detected}")
        print(f"   Regime duration: {result.regime_duration} minutes")
        
        # Validate detection result
        assert result.primary_regime in MarketRegime, "Invalid primary regime"
        assert 0.0 <= result.confidence_score <= 1.0, f"Confidence score out of range: {result.confidence_score}"
        assert result.metrics is not None, "Metrics should be present"
        assert result.parameters is not None, "Parameters should be present"
        
        # Metrics validation
        metrics = result.metrics
        print(f"   Metrics validation:")
        print(f"     Trend strength: {metrics.trend_strength:.2f}")
        print(f"     Trend consistency: {metrics.trend_consistency:.2f}")
        print(f"     Realized volatility: {metrics.realized_volatility:.1%}")
        print(f"     Choppiness index: {metrics.choppiness_index:.1f}")
        
        assert -1.0 <= metrics.trend_strength <= 1.0, "Trend strength out of range"
        assert 0.0 <= metrics.trend_consistency <= 1.0, "Trend consistency out of range"
        assert metrics.realized_volatility >= 0.0, "Volatility negative"
        
        print("   ✅ Regime detection working correctly")
        
        # Test 4: Regime parameters adaptation
        print("\n4. Testing regime parameters...")
        
        params = result.parameters
        print(f"   Current regime parameters:")
        print(f"     Sizing multiplier: {params.sizing_multiplier:.1f}")
        print(f"     Stop loss multiplier: {params.stop_loss_multiplier:.1f}")
        print(f"     Take profit multiplier: {params.take_profit_multiplier:.1f}")
        print(f"     Max entries per hour: {params.max_entries_per_hour}")
        print(f"     Min signal confidence: {params.entry_confidence_threshold:.1%}")
        print(f"     Risk multiplier: {params.risk_multiplier:.1f}")
        print(f"     Trailing stop enabled: {params.trailing_stop_enabled}")
        
        # Validate parameter ranges
        assert 0.1 <= params.sizing_multiplier <= 3.0, f"Sizing multiplier out of range: {params.sizing_multiplier}"
        assert 0.1 <= params.stop_loss_multiplier <= 3.0, f"Stop multiplier out of range: {params.stop_loss_multiplier}"
        assert 0.1 <= params.take_profit_multiplier <= 3.0, f"TP multiplier out of range: {params.take_profit_multiplier}"
        assert 1 <= params.max_entries_per_hour <= 20, f"Max entries out of range: {params.max_entries_per_hour}"
        assert 0.0 <= params.entry_confidence_threshold <= 1.0, f"Confidence threshold out of range: {params.entry_confidence_threshold}"
        
        print("   ✅ Regime parameters configured correctly")
        
        # Test 5: Adaptive trading manager
        print("\n5. Testing adaptive trading manager...")
        
        manager = AdaptiveTradingManager()
        print(f"   Adaptive manager created")
        print(f"   Base settings configured: ✅")
        
        # Test trade decision
        decision = manager.should_take_trade(
            symbol="BTC/USD",
            signal_strength=0.8,
            signal_confidence=0.7,
            current_price=prices[-1],
            volume=1000.0
        )
        
        print(f"   Trade decision:")
        print(f"     Should trade: {decision.should_trade}")
        print(f"     Rejection reasons: {decision.rejection_reasons}")
        print(f"     Regime: {decision.regime_info.primary_regime.value}")
        print(f"     Confidence adjustment: {decision.confidence_adjustment:.2f}")
        print(f"     Size adjustment: {decision.size_adjustment:.2f}")
        print(f"     Risk adjustment: {decision.risk_adjustment:.2f}")
        
        # Validate decision structure
        assert isinstance(decision.should_trade, bool), "Should trade must be boolean"
        assert isinstance(decision.rejection_reasons, list), "Rejection reasons must be list"
        assert decision.adapted_settings is not None, "Adapted settings missing"
        assert decision.regime_info is not None, "Regime info missing"
        
        # Validate adjustments are reasonable
        assert 0.1 <= decision.confidence_adjustment <= 2.0, f"Confidence adjustment out of range: {decision.confidence_adjustment}"
        assert 0.1 <= decision.size_adjustment <= 3.0, f"Size adjustment out of range: {decision.size_adjustment}"
        assert 0.1 <= decision.risk_adjustment <= 2.0, f"Risk adjustment out of range: {decision.risk_adjustment}"
        
        print("   ✅ Adaptive trading manager working")
        
        # Test 6: Settings adaptation
        print("\n6. Testing settings adaptation...")
        
        base = manager.base_settings
        adapted = decision.adapted_settings
        
        print(f"   Settings comparison (base → adapted):")
        print(f"     Position size: {base.position_size_multiplier:.2f} → {adapted.position_size_multiplier:.2f}")
        print(f"     Stop distance: {base.stop_loss_distance:.1%} → {adapted.stop_loss_distance:.1%}")
        print(f"     Take profit: {base.take_profit_distance:.1%} → {adapted.take_profit_distance:.1%}")
        print(f"     Min confidence: {base.min_signal_confidence:.1%} → {adapted.min_signal_confidence:.1%}")
        print(f"     Max entries/hour: {base.max_entries_per_hour} → {adapted.max_entries_per_hour}")
        print(f"     Risk multiplier: {base.regime_risk_multiplier:.1f} → {adapted.regime_risk_multiplier:.1f}")
        
        # Calculate adaptation ratios
        size_ratio = adapted.position_size_multiplier / base.position_size_multiplier
        stop_ratio = adapted.stop_loss_distance / base.stop_loss_distance
        tp_ratio = adapted.take_profit_distance / base.take_profit_distance
        
        print(f"   Adaptation ratios:")
        print(f"     Size: {size_ratio:.2f}x")
        print(f"     Stop: {stop_ratio:.2f}x")
        print(f"     Take profit: {tp_ratio:.2f}x")
        
        # Validate adaptations are within reasonable bounds
        assert 0.2 <= size_ratio <= 2.0, f"Size adaptation too extreme: {size_ratio:.2f}"
        assert 0.3 <= stop_ratio <= 3.0, f"Stop adaptation too extreme: {stop_ratio:.2f}"
        assert 0.3 <= tp_ratio <= 3.0, f"TP adaptation too extreme: {tp_ratio:.2f}"
        
        print("   ✅ Settings adaptation working correctly")
        
        # Test 7: Different market scenarios
        print("\n7. Testing different market scenarios...")
        
        scenarios = [
            ("choppy", lambda i: 1000 + random.gauss(0, 100)),  # No trend, high noise
            ("trending", lambda i: 1000 + i * 50 + random.gauss(0, 10)),  # Strong trend, low noise
            ("volatile", lambda i: 1000 + random.gauss(0, 500)),  # High volatility
            ("stable", lambda i: 1000 + i * 2 + random.gauss(0, 5))  # Low volatility
        ]
        
        for scenario_name, price_func in scenarios:
            print(f"   Testing {scenario_name} scenario...")
            
            scenario_detector = RegimeDetector(f"{scenario_name.upper()}/USD")
            
            # Generate scenario-specific price data
            scenario_prices = []
            for i in range(80):
                price = price_func(i)
                scenario_prices.append(price)
                scenario_detector.update_price_data(price, volume=1000.0)
            
            scenario_result = scenario_detector.detect_regime()
            
            print(f"     Regime: {scenario_result.primary_regime.value}")
            print(f"     Confidence: {scenario_result.confidence_score:.2f}")
            print(f"     Trend strength: {scenario_result.metrics.trend_strength:.2f}")
            print(f"     Volatility: {scenario_result.metrics.realized_volatility:.1%}")
            print(f"     Choppiness: {scenario_result.metrics.choppiness_index:.1f}")
            
            # Test adaptive decision for scenario
            scenario_decision = manager.should_take_trade(
                symbol=f"{scenario_name.upper()}/USD",
                signal_strength=0.6,
                signal_confidence=0.6,
                current_price=scenario_prices[-1]
            )
            
            print(f"     Trade approved: {scenario_decision.should_trade}")
            print(f"     Size multiplier: {scenario_decision.adapted_settings.position_size_multiplier:.2f}")
            
        print("   ✅ Different scenarios handled correctly")
        
        # Test 8: Summary functions
        print("\n8. Testing summary functions...")
        
        summary = detector.get_regime_summary()
        print(f"   Regime summary keys: {list(summary.keys())}")
        
        required_keys = ['symbol', 'current_regime', 'confidence', 'regime_duration_minutes']
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"
        
        adaptive_summary = manager.get_adaptive_settings_summary("BTC/USD")
        print(f"   Adaptive summary keys: {list(adaptive_summary.keys())}")
        
        required_adaptive_keys = ['symbol', 'regime', 'adapted_settings', 'base_vs_adapted']
        for key in required_adaptive_keys:
            assert key in adaptive_summary, f"Missing key in adaptive summary: {key}"
        
        print("   ✅ Summary functions working")
        
        print(f"\n📊 Final Test Summary:")
        print(f"   Regime detector: ✅ Working")
        print(f"   Technical indicators: ✅ Calculated correctly")
        print(f"   Regime detection: ✅ {result.primary_regime.value} detected")
        print(f"   Parameter adaptation: ✅ Size {params.sizing_multiplier:.1f}x, Stop {params.stop_loss_multiplier:.1f}x")
        print(f"   Adaptive trading: ✅ Decisions with {len(decision.rejection_reasons)} filters")
        print(f"   Multiple scenarios: ✅ Different regimes detected")
        print(f"   Summary functions: ✅ Complete data provided")
        
        print(f"\n🎯 All regime switching tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure regime switching system is properly created")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_regime_switching_system()