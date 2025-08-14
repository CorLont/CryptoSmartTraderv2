#!/usr/bin/env python3
"""
Simple test for Kelly sizing system without heavy dependencies
Validates Kelly criterion, vol-targeting and correlation limits
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_kelly_sizing_system():
    """Test Kelly sizing system with minimal dependencies"""
    
    print("📏 Testing Kelly Vol-Targeting System")
    print("=" * 45)
    
    try:
        from cryptosmarttrader.sizing.kelly_vol_targeting import (
            KellyVolTargetSizer, AssetMetrics, SizingLimits, AssetCluster, SizingMethod
        )
        
        # Setup
        limits = SizingLimits(
            kelly_fraction=0.25,
            max_kelly_position=0.15,
            target_portfolio_vol=0.15,
            max_single_asset=0.20,
            max_cluster_weight=0.40
        )
        
        sizer = KellyVolTargetSizer(limits)
        sizer.update_portfolio_equity(100000.0)
        
        print(f"✅ Kelly sizer initialized")
        print(f"   Portfolio equity: ${sizer.portfolio_equity:,.0f}")
        print(f"   Kelly fraction: {limits.kelly_fraction:.1%}")
        print(f"   Target vol: {limits.target_portfolio_vol:.1%}")
        
        # Test 1: Asset metrics setup
        print("\n1. Testing asset metrics...")
        
        btc_metrics = AssetMetrics(
            symbol="BTC/USD",
            expected_return=0.30,  # 30% expected return
            volatility=0.60,       # 60% volatility  
            sharpe_ratio=0.50,
            correlation_matrix_position=0,
            cluster=AssetCluster.CRYPTO_MAJOR,
            liquidity_score=0.95,
            momentum_score=0.8,
            mean_reversion_score=-0.2
        )
        
        eth_metrics = AssetMetrics(
            symbol="ETH/USD",
            expected_return=0.25,  # 25% expected return
            volatility=0.70,       # 70% volatility
            sharpe_ratio=0.36,
            correlation_matrix_position=1, 
            cluster=AssetCluster.CRYPTO_MAJOR,
            liquidity_score=0.90,
            momentum_score=0.6,
            mean_reversion_score=-0.1
        )
        
        sizer.update_asset_metrics("BTC/USD", btc_metrics)
        sizer.update_asset_metrics("ETH/USD", eth_metrics)
        
        print(f"   Added BTC metrics: Return={btc_metrics.expected_return:.1%}, Vol={btc_metrics.volatility:.1%}")
        print(f"   Added ETH metrics: Return={eth_metrics.expected_return:.1%}, Vol={eth_metrics.volatility:.1%}")
        print("   ✅ Asset metrics updated")
        
        # Test 2: Kelly weight calculation
        print("\n2. Testing Kelly weight calculation...")
        
        kelly_btc = sizer.calculate_kelly_weight(btc_metrics)
        kelly_eth = sizer.calculate_kelly_weight(eth_metrics)
        
        print(f"   BTC Kelly weight: {kelly_btc:.1%}")
        print(f"   ETH Kelly weight: {kelly_eth:.1%}")
        
        # Validate Kelly calculations
        # Kelly formula: f = (μ - r) / σ² * kelly_fraction
        risk_free = 0.05
        expected_kelly_btc = ((btc_metrics.expected_return - risk_free) / (btc_metrics.volatility ** 2)) * limits.kelly_fraction
        expected_kelly_btc = min(abs(expected_kelly_btc), limits.max_kelly_position)
        
        assert abs(kelly_btc - expected_kelly_btc) < 0.01, f"Kelly BTC calculation error: {kelly_btc:.3f} vs {expected_kelly_btc:.3f}"
        assert 0 <= kelly_btc <= limits.max_kelly_position, f"Kelly BTC out of bounds: {kelly_btc:.1%}"
        assert 0 <= kelly_eth <= limits.max_kelly_position, f"Kelly ETH out of bounds: {kelly_eth:.1%}"
        
        print("   ✅ Kelly weights calculated correctly")
        
        # Test 3: Volatility targeting
        print("\n3. Testing volatility targeting...")
        
        vol_target_btc = sizer.calculate_vol_target_weight(btc_metrics)
        vol_target_eth = sizer.calculate_vol_target_weight(eth_metrics)
        
        print(f"   BTC vol-target weight: {vol_target_btc:.1%}")
        print(f"   ETH vol-target weight: {vol_target_eth:.1%}")
        
        # Vol targeting: w = target_vol / asset_vol
        expected_vol_btc = min(limits.target_portfolio_vol / btc_metrics.volatility, limits.max_single_asset)
        expected_vol_eth = min(limits.target_portfolio_vol / eth_metrics.volatility, limits.max_single_asset)
        
        assert abs(vol_target_btc - expected_vol_btc) < 0.01, f"Vol target BTC error: {vol_target_btc:.3f} vs {expected_vol_btc:.3f}"
        assert vol_target_btc > vol_target_eth, "Lower vol asset should get higher weight"
        
        print("   ✅ Volatility targeting working")
        
        # Test 4: Position sizing with signals
        print("\n4. Testing position sizing...")
        
        signals = {
            "BTC/USD": 0.8,   # Strong buy signal
            "ETH/USD": 0.6    # Moderate buy signal
        }
        
        results = sizer.calculate_position_sizes(signals, SizingMethod.FRACTIONAL_KELLY)
        
        print(f"   Calculated sizes for {len(results)} assets:")
        total_weight = 0.0
        
        for symbol, result in results.items():
            print(f"     {symbol}:")
            print(f"       Target weight: {result.target_weight:.1%}")
            print(f"       Target amount: ${result.target_dollar_amount:,.0f}")
            print(f"       Kelly component: {result.kelly_weight:.1%}")
            print(f"       Vol-adjusted: {result.vol_adjusted_weight:.1%}")
            print(f"       Confidence: {result.confidence_score:.2f}")
            print(f"       Method: {result.method.value}")
            
            total_weight += result.target_weight
            
            # Validation
            assert result.target_weight >= 0, f"Weight should be non-negative: {result.target_weight}"
            assert result.target_weight <= limits.max_single_asset, f"Weight exceeds single asset limit: {result.target_weight:.1%}"
            assert result.confidence_score >= 0, f"Confidence should be non-negative: {result.confidence_score}"
            assert result.target_dollar_amount >= 0, f"Dollar amount should be non-negative: {result.target_dollar_amount}"
        
        print(f"   Total portfolio weight: {total_weight:.1%}")
        assert len(results) == 2, f"Should have 2 results, got {len(results)}"
        print("   ✅ Position sizing working correctly")
        
        # Test 5: Correlation constraints
        print("\n5. Testing correlation constraints...")
        
        # Test cluster constraint by adding another CRYPTO_MAJOR asset
        sol_metrics = AssetMetrics(
            symbol="SOL/USD",
            expected_return=0.35,
            volatility=0.90,
            sharpe_ratio=0.39,
            correlation_matrix_position=2,
            cluster=AssetCluster.CRYPTO_MAJOR,  # Same cluster
            liquidity_score=0.85,
            momentum_score=0.7,
            mean_reversion_score=-0.3
        )
        
        sizer.update_asset_metrics("SOL/USD", sol_metrics)
        
        signals_3assets = {
            "BTC/USD": 0.8,
            "ETH/USD": 0.7,
            "SOL/USD": 0.9   # Very strong signal
        }
        
        results_constrained = sizer.calculate_position_sizes(signals_3assets, SizingMethod.FRACTIONAL_KELLY)
        
        # Check cluster constraint
        crypto_major_weight = sum(
            result.target_weight for symbol, result in results_constrained.items()
            if symbol in ["BTC/USD", "ETH/USD", "SOL/USD"]  # All CRYPTO_MAJOR
        )
        
        print(f"   Total CRYPTO_MAJOR cluster weight: {crypto_major_weight:.1%}")
        print(f"   Cluster limit: {limits.max_cluster_weight:.1%}")
        
        # Should respect cluster limit (with small tolerance for numerical precision)
        assert crypto_major_weight <= limits.max_cluster_weight * 1.02, f"Cluster weight {crypto_major_weight:.1%} exceeds limit {limits.max_cluster_weight:.1%}"
        
        print("   ✅ Correlation constraints applied")
        
        # Test 6: Portfolio summary
        print("\n6. Testing portfolio summary...")
        
        summary = sizer.get_portfolio_summary()
        
        print(f"   Portfolio equity: ${summary['total_equity']:,.0f}")
        print(f"   Position count: {summary['position_count']}")
        print(f"   Total allocation: {summary['total_allocation']:.1%}")
        print(f"   Target vol: {summary['target_portfolio_vol']:.1%}")
        print(f"   Diversification ratio: {summary['diversification_ratio']:.2f}")
        
        # Validate summary structure
        required_keys = ['total_equity', 'position_count', 'limits', 'positions']
        for key in required_keys:
            assert key in summary, f"Summary missing key: {key}"
        
        assert summary['total_equity'] == sizer.portfolio_equity, "Equity mismatch in summary"
        assert summary['target_portfolio_vol'] == limits.target_portfolio_vol, "Target vol mismatch"
        
        print("   ✅ Portfolio summary complete")
        
        # Test 7: Different sizing methods
        print("\n7. Testing different sizing methods...")
        
        # Test vol-targeting method
        vol_results = sizer.calculate_position_sizes(signals, SizingMethod.VOL_TARGET)
        
        print(f"   Vol-targeting results:")
        for symbol, result in vol_results.items():
            print(f"     {symbol}: {result.target_weight:.1%} (method: {result.method.value})")
        
        # Test equal weight method
        equal_results = sizer.calculate_position_sizes(signals, SizingMethod.EQUAL_WEIGHT)
        
        print(f"   Equal-weight results:")
        for symbol, result in equal_results.items():
            print(f"     {symbol}: {result.target_weight:.1%} (method: {result.method.value})")
        
        # Equal weight should be approximately equal (adjusted by signal strength)
        equal_weights = [r.target_weight for r in equal_results.values()]
        weight_std = np.std(equal_weights)
        print(f"   Equal weight std dev: {weight_std:.3f}")
        
        assert weight_std < 0.1, f"Equal weights too different: std={weight_std:.3f}"
        print("   ✅ Multiple sizing methods working")
        
        print(f"\n📊 Final Summary:")
        final_summary = sizer.get_portfolio_summary()
        print(f"   System ready: ✅")
        print(f"   Kelly fraction: {limits.kelly_fraction:.1%}")
        print(f"   Vol target: {limits.target_portfolio_vol:.1%}")
        print(f"   Max single asset: {limits.max_single_asset:.1%}")
        print(f"   Max cluster: {limits.max_cluster_weight:.1%}")
        print(f"   Assets configured: {len(sizer.asset_metrics)}")
        
        print(f"\n🎯 All Kelly sizing tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Kelly sizing system is properly created")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_kelly_sizing_system()