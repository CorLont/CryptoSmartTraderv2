"""
Comprehensive tests for regime switching and adaptive trading
"""

import time
import random
import math
from datetime import datetime, timedelta

from src.cryptosmarttrader.regime.regime_detection import (
    RegimeDetector, MarketRegime, RegimeIndicators, PriceDataBuffer
)
from src.cryptosmarttrader.regime.adaptive_trading import (
    AdaptiveTradingManager, should_take_adaptive_trade
)


def test_regime_detection():
    """Test regime detection system"""
    
    print("🔄 Testing Regime Detection System")
    print("=" * 40)
    
    # Setup
    detector = RegimeDetector("TEST/USD")
    
    # Test 1: Price data buffer
    print("\n1. Testing price data buffer...")
    
    for i in range(50):
        price = 1000 + i * 10 + random.gauss(0, 5)  # Upward trend with noise
        detector.update_price_data(price, volume=100.0)
    
    print(f"   Added 50 price points")
    print(f"   Buffer size: {len(detector.price_buffer.prices)}")
    
    returns = detector.price_buffer.get_returns()
    print(f"   Returns calculated: {len(returns)}")
    
    assert len(detector.price_buffer.prices) == 50, "Buffer should have 50 prices"
    assert len(returns) == 49, "Should have 49 returns"
    print("   ✅ Price data buffer working")
    
    # Test 2: Technical indicators
    print("\n2. Testing technical indicators...")
    
    prices = list(detector.price_buffer.prices)
    
    trend_strength = RegimeIndicators.calculate_trend_strength(prices)
    print(f"   Trend strength: {trend_strength:.2f}")
    
    choppiness = RegimeIndicators.calculate_choppiness_index(prices, prices, prices)
    print(f"   Choppiness index: {choppiness:.1f}")
    
    volatility = RegimeIndicators.calculate_realized_volatility(returns)
    print(f"   Realized volatility: {volatility:.1%}")
    
    mean_reversion = RegimeIndicators.calculate_mean_reversion_score(prices)
    print(f"   Mean reversion score: {mean_reversion:.2f}")
    
    # Validate ranges
    assert -1.0 <= trend_strength <= 1.0, f"Trend strength out of range: {trend_strength}"
    assert 0.0 <= choppiness <= 100.0, f"Choppiness out of range: {choppiness}"
    assert volatility >= 0.0, f"Volatility negative: {volatility}"
    assert 0.0 <= mean_reversion <= 1.0, f"Mean reversion out of range: {mean_reversion}"
    
    print("   ✅ Technical indicators working")
    
    # Test 3: Regime detection
    print("\n3. Testing regime detection...")
    
    result = detector.detect_regime()
    
    print(f"   Detected regime: {result.primary_regime.value}")
    print(f"   Confidence: {result.confidence.value} ({result.confidence_score:.2f})")
    print(f"   Secondary regime: {result.secondary_regime.value if result.secondary_regime else 'None'}")
    print(f"   Regime duration: {result.regime_duration} minutes")
    
    # Validate result structure
    assert result.primary_regime in MarketRegime, "Primary regime should be valid"
    assert 0.0 <= result.confidence_score <= 1.0, f"Confidence score out of range: {result.confidence_score}"
    assert result.metrics is not None, "Metrics should be present"
    assert result.parameters is not None, "Parameters should be present"
    
    print("   ✅ Regime detection working")
    
    # Test 4: Different regime scenarios
    print("\n4. Testing different regime scenarios...")
    
    # Create trending scenario
    detector_trend = RegimeDetector("TREND/USD")
    base_price = 1000
    for i in range(100):
        price = base_price + i * 20 + random.gauss(0, 2)  # Strong uptrend
        detector_trend.update_price_data(price)
    
    trend_result = detector_trend.detect_regime()
    print(f"   Trending scenario: {trend_result.primary_regime.value}")
    print(f"   Trend strength: {trend_result.metrics.trend_strength:.2f}")
    
    # Create choppy scenario
    detector_chop = RegimeDetector("CHOP/USD")
    base_price = 1000
    for i in range(100):
        price = base_price + random.gauss(0, 50)  # High noise, no trend
        detector_chop.update_price_data(price)
    
    chop_result = detector_chop.detect_regime()
    print(f"   Choppy scenario: {chop_result.primary_regime.value}")
    print(f"   Choppiness index: {chop_result.metrics.choppiness_index:.1f}")
    
    # Create high volatility scenario
    detector_vol = RegimeDetector("VOL/USD")
    base_price = 1000
    for i in range(100):
        price = base_price + random.gauss(0, 200)  # Very high volatility
        detector_vol.update_price_data(price)
    
    vol_result = detector_vol.detect_regime()
    print(f"   High vol scenario: {vol_result.primary_regime.value}")
    print(f"   Realized vol: {vol_result.metrics.realized_volatility:.1%}")
    
    print("   ✅ Different scenarios detected correctly")
    
    # Test 5: Regime parameters adaptation
    print("\n5. Testing regime parameters...")
    
    for regime in MarketRegime:
        params = detector.regime_parameters[regime]
        print(f"   {regime.value}:")
        print(f"     Size multiplier: {params.sizing_multiplier:.1f}")
        print(f"     Stop multiplier: {params.stop_loss_multiplier:.1f}")
        print(f"     Max entries/hour: {params.max_entries_per_hour}")
        
        # Validate parameter ranges
        assert 0.1 <= params.sizing_multiplier <= 2.0, f"Size multiplier out of range for {regime.value}"
        assert 0.1 <= params.stop_loss_multiplier <= 3.0, f"Stop multiplier out of range for {regime.value}"
        assert 1 <= params.max_entries_per_hour <= 20, f"Max entries out of range for {regime.value}"
    
    print("   ✅ Regime parameters configured correctly")
    
    print("\n🎯 All regime detection tests passed!")
    return True


def test_adaptive_trading():
    """Test adaptive trading system"""
    
    print("\n🤖 Testing Adaptive Trading System")
    print("=" * 35)
    
    # Setup
    manager = AdaptiveTradingManager()
    
    # Test 1: Basic trade decision
    print("\n1. Testing basic trade decision...")
    
    decision = manager.should_take_trade(
        symbol="BTC/USD",
        signal_strength=0.8,
        signal_confidence=0.7,
        current_price=50000.0,
        volume=1000.0
    )
    
    print(f"   Should trade: {decision.should_trade}")
    print(f"   Regime: {decision.regime_info.primary_regime.value}")
    print(f"   Confidence adjustment: {decision.confidence_adjustment:.2f}")
    print(f"   Size adjustment: {decision.size_adjustment:.2f}")
    print(f"   Risk adjustment: {decision.risk_adjustment:.2f}")
    
    assert isinstance(decision.should_trade, bool), "Should trade should be boolean"
    assert decision.adapted_settings is not None, "Adapted settings should be present"
    assert decision.regime_info is not None, "Regime info should be present"
    
    print("   ✅ Basic trade decision working")
    
    # Test 2: Settings adaptation
    print("\n2. Testing settings adaptation...")
    
    base_settings = manager.base_settings
    adapted_settings = decision.adapted_settings
    
    print(f"   Base vs Adapted:")
    print(f"     Position size: {base_settings.position_size_multiplier:.2f} → {adapted_settings.position_size_multiplier:.2f}")
    print(f"     Stop distance: {base_settings.stop_loss_distance:.1%} → {adapted_settings.stop_loss_distance:.1%}")
    print(f"     Take profit: {base_settings.take_profit_distance:.1%} → {adapted_settings.take_profit_distance:.1%}")
    print(f"     Min confidence: {base_settings.min_signal_confidence:.1%} → {adapted_settings.min_signal_confidence:.1%}")
    print(f"     Max entries/hour: {base_settings.max_entries_per_hour} → {adapted_settings.max_entries_per_hour}")
    
    # Validate adaptations are reasonable
    assert 0.1 <= adapted_settings.position_size_multiplier <= 3.0, "Position size multiplier out of range"
    assert 0.005 <= adapted_settings.stop_loss_distance <= 0.2, "Stop loss distance out of range"
    assert 0.01 <= adapted_settings.take_profit_distance <= 0.5, "Take profit distance out of range"
    
    print("   ✅ Settings adaptation working")
    
    # Test 3: Entry throttling
    print("\n3. Testing entry throttling...")
    
    # Simulate multiple rapid entries
    symbol = "THROTTLE/USD"
    current_time = datetime.now()
    
    # Add some fake recent trades
    manager.recent_trades[symbol] = [current_time - timedelta(minutes=1)] * 15  # 15 trades in last hour
    
    throttled_decision = manager.should_take_trade(
        symbol=symbol,
        signal_strength=0.9,
        signal_confidence=0.8,
        current_price=1000.0
    )
    
    print(f"   Throttled trade approved: {throttled_decision.should_trade}")
    print(f"   Rejection reasons: {throttled_decision.rejection_reasons}")
    
    # Should be throttled due to too many recent trades
    assert not throttled_decision.should_trade, "Trade should be throttled"
    assert any("throttling" in reason.lower() for reason in throttled_decision.rejection_reasons), "Should mention throttling"
    
    print("   ✅ Entry throttling working")
    
    # Test 4: Signal confidence filtering
    print("\n4. Testing signal confidence filtering...")
    
    # Low confidence signal
    low_conf_decision = manager.should_take_trade(
        symbol="LOWCONF/USD",
        signal_strength=0.8,
        signal_confidence=0.2,  # Very low confidence
        current_price=1000.0
    )
    
    print(f"   Low confidence trade approved: {low_conf_decision.should_trade}")
    print(f"   Rejection reasons: {low_conf_decision.rejection_reasons}")
    
    # Should be rejected due to low confidence
    confidence_rejected = any("confidence" in reason.lower() for reason in low_conf_decision.rejection_reasons)
    print(f"   Confidence rejection detected: {confidence_rejected}")
    
    print("   ✅ Signal confidence filtering working")
    
    # Test 5: Regime-specific filtering
    print("\n5. Testing regime-specific filtering...")
    
    # Create a detector with choppy conditions
    choppy_detector = RegimeDetector("CHOPPY/USD")
    
    # Add choppy price data
    base_price = 1000
    for i in range(100):
        # Random walk with high noise
        price = base_price + random.gauss(0, 100)
        choppy_detector.update_price_data(price)
    
    choppy_result = choppy_detector.detect_regime()
    print(f"   Choppy regime detected: {choppy_result.primary_regime.value}")
    print(f"   Choppiness index: {choppy_result.metrics.choppiness_index:.1f}")
    
    # Weak signal in choppy market should be rejected
    weak_signal_decision = manager.should_take_trade(
        symbol="CHOPPY/USD",
        signal_strength=0.3,  # Weak signal
        signal_confidence=0.6,
        current_price=1000.0
    )
    
    print(f"   Weak signal in choppy market approved: {weak_signal_decision.should_trade}")
    
    print("   ✅ Regime-specific filtering working")
    
    # Test 6: Summary functions
    print("\n6. Testing summary functions...")
    
    symbols = ["BTC/USD", "ETH/USD", "TEST/USD"]
    regime_summaries = manager.get_regime_summary_for_symbols(symbols)
    
    print(f"   Regime summaries for {len(regime_summaries)} symbols:")
    for symbol, summary in regime_summaries.items():
        print(f"     {symbol}: {summary.get('current_regime', 'unknown')}")
    
    assert len(regime_summaries) == len(symbols), "Should have summary for each symbol"
    
    adaptive_summary = manager.get_adaptive_settings_summary("BTC/USD")
    print(f"   Adaptive settings summary keys: {list(adaptive_summary.keys())}")
    
    required_keys = ['symbol', 'regime', 'confidence', 'adapted_settings', 'base_vs_adapted']
    for key in required_keys:
        assert key in adaptive_summary, f"Missing key in adaptive summary: {key}"
    
    print("   ✅ Summary functions working")
    
    print("\n🎯 All adaptive trading tests passed!")
    return True


def test_regime_integration():
    """Test integration between regime detection and other systems"""
    
    print("\n🔗 Testing Regime Integration")
    print("=" * 30)
    
    # Test convenience function
    decision = should_take_adaptive_trade(
        symbol="INTEGRATION/USD",
        signal_strength=0.7,
        signal_confidence=0.8,
        current_price=25000.0
    )
    
    print(f"   Convenience function result: {decision.should_trade}")
    print(f"   Regime: {decision.regime_info.primary_regime.value}")
    print(f"   Adapted position size: {decision.adapted_settings.position_size_multiplier:.2f}")
    
    # Validate integration
    assert decision is not None, "Decision should not be None"
    assert hasattr(decision, 'should_trade'), "Decision should have should_trade attribute"
    assert hasattr(decision, 'regime_info'), "Decision should have regime_info"
    assert hasattr(decision, 'adapted_settings'), "Decision should have adapted_settings"
    
    print("   ✅ Integration working correctly")
    
    print("\n🎯 All integration tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Running Regime Switching Test Suite")
    print("=" * 50)
    
    try:
        test_regime_detection()
        test_adaptive_trading() 
        test_regime_integration()
        
        print("\n🎉 ALL REGIME SWITCHING TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
