#!/usr/bin/env python3
"""
FASE C ExecutionPolicy Demo - Real Implementation Test
Tests ExecutionPolicy.decide() enforcement with spread, depth, volume gates
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path  
sys.path.insert(0, '/home/runner/workspace')

# Import with fallback handling
try:
    from src.cryptosmarttrader.execution.hard_execution_policy import (
        HardExecutionPolicy,
        OrderRequest,
        MarketConditions,
        ExecutionResult,
        ExecutionDecision,
        OrderSide,
        TimeInForce
    )
    EXECUTION_POLICY_AVAILABLE = True
    print("✅ Hard execution policy imported successfully")
except ImportError as e:
    print(f"⚠️  Hard execution policy import failed: {e}")
    EXECUTION_POLICY_AVAILABLE = False

def test_execution_policy_gates():
    """Test ExecutionPolicy.decide() with various gate scenarios"""
    if not EXECUTION_POLICY_AVAILABLE:
        print("❌ ExecutionPolicy not available for testing")
        return False
    
    print("\n" + "="*50)
    print("TEST: EXECUTION POLICY GATES")
    print("="*50)
    
    # Reset singleton
    if hasattr(HardExecutionPolicy, '_instance'):
        HardExecutionPolicy._instance = None
    
    # Create policy instance with test config
    config = {
        'max_spread_bps': 50,  # 50 bps max spread
        'min_depth_usd': 10000,  # $10k min depth
        'max_slippage_bps': 30,  # 30 bps max slippage
        'min_volume_24h_usd': 1000000,  # $1M min volume
        'max_order_value_usd': 50000,  # $50k max order
    }
    
    policy = HardExecutionPolicy(config)
    
    # Test order
    order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        time_in_force=TimeInForce.POST_ONLY,
        client_order_id="policy_test_001"
    )
    
    print("1. Testing with good market conditions...")
    good_conditions = MarketConditions(
        symbol="BTC/USD",
        bid_price=49950.0,
        ask_price=50050.0,  # 50 bps spread
        spread_bps=50.0,
        bid_depth_usd=15000.0,  # Above $10k requirement
        ask_depth_usd=15000.0,
        volume_24h_usd=5000000.0,  # Above $1M requirement
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    result1 = policy.decide(order, good_conditions)
    print(f"   Good conditions: {result1.decision.value} - {result1.reason}")
    
    print("\n2. Testing spread gate (excessive spread)...")
    bad_spread_conditions = MarketConditions(
        symbol="BTC/USD",
        bid_price=49900.0,
        ask_price=50200.0,  # 60 bps spread (exceeds 50 bps limit)
        spread_bps=60.0,
        bid_depth_usd=15000.0,
        ask_depth_usd=15000.0,
        volume_24h_usd=5000000.0,
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    result2 = policy.decide(order, bad_spread_conditions)
    print(f"   Bad spread: {result2.decision.value} - {result2.reason}")
    
    print("\n3. Testing depth gate (insufficient depth)...")
    bad_depth_conditions = MarketConditions(
        symbol="BTC/USD",
        bid_price=49950.0,
        ask_price=50050.0,
        spread_bps=50.0,
        bid_depth_usd=5000.0,  # Below $10k requirement
        ask_depth_usd=5000.0,
        volume_24h_usd=5000000.0,
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    result3 = policy.decide(order, bad_depth_conditions)
    print(f"   Bad depth: {result3.decision.value} - {result3.reason}")
    
    print("\n4. Testing volume gate (insufficient volume)...")
    bad_volume_conditions = MarketConditions(
        symbol="BTC/USD", 
        bid_price=49950.0,
        ask_price=50050.0,
        spread_bps=50.0,
        bid_depth_usd=15000.0,
        ask_depth_usd=15000.0,
        volume_24h_usd=500000.0,  # Below $1M requirement
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    result4 = policy.decide(order, bad_volume_conditions)
    print(f"   Bad volume: {result4.decision.value} - {result4.reason}")
    
    # Test assertions
    assert result1.decision == ExecutionDecision.APPROVE
    assert result2.decision == ExecutionDecision.REJECT
    assert result3.decision == ExecutionDecision.REJECT  
    assert result4.decision == ExecutionDecision.REJECT
    
    print("\n✅ EXECUTION POLICY GATES TEST: PASSED")
    return True

def test_client_order_id_generation():
    """Test automatic client order ID generation"""
    if not EXECUTION_POLICY_AVAILABLE:
        print("❌ ExecutionPolicy not available for testing")
        return False
        
    print("\n" + "="*50)
    print("TEST: CLIENT ORDER ID GENERATION")
    print("="*50)
    
    # Reset singleton
    if hasattr(HardExecutionPolicy, '_instance'):
        HardExecutionPolicy._instance = None
    
    policy = HardExecutionPolicy()
    
    # Order without client_order_id
    order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        time_in_force=TimeInForce.POST_ONLY,
        client_order_id=None  # No COID provided
    )
    
    good_conditions = MarketConditions(
        symbol="BTC/USD",
        bid_price=49950.0,
        ask_price=50050.0,
        spread_bps=50.0,
        bid_depth_usd=15000.0,
        ask_depth_usd=15000.0,
        volume_24h_usd=5000000.0,
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    print("1. Processing order without client_order_id...")
    result = policy.decide(order, good_conditions)
    
    print(f"   Generated COID: {order.client_order_id}")
    print(f"   Result: {result.decision.value}")
    
    # Test assertions
    assert order.client_order_id is not None
    assert len(order.client_order_id) > 0
    assert result.client_order_id == order.client_order_id
    
    print("\n✅ CLIENT ORDER ID GENERATION TEST: PASSED")
    return True

def test_duplicate_order_detection():
    """Test duplicate order detection"""
    if not EXECUTION_POLICY_AVAILABLE:
        print("❌ ExecutionPolicy not available for testing")
        return False
        
    print("\n" + "="*50)
    print("TEST: DUPLICATE ORDER DETECTION")
    print("="*50)
    
    # Reset singleton
    if hasattr(HardExecutionPolicy, '_instance'):
        HardExecutionPolicy._instance = None
    
    policy = HardExecutionPolicy()
    
    # First order
    order1 = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=0.1,
        price=50000.0,
        time_in_force=TimeInForce.POST_ONLY,
        client_order_id="duplicate_test_123"
    )
    
    # Second order with same COID
    order2 = OrderRequest(
        symbol="ETH/USD",  # Different symbol
        side=OrderSide.SELL,  # Different side
        quantity=1.0,      # Different quantity
        price=3000.0,      # Different price
        time_in_force=TimeInForce.POST_ONLY,
        client_order_id="duplicate_test_123"  # SAME COID
    )
    
    good_conditions = MarketConditions(
        symbol="BTC/USD",
        bid_price=49950.0,
        ask_price=50050.0,
        spread_bps=50.0,
        bid_depth_usd=15000.0,
        ask_depth_usd=15000.0,
        volume_24h_usd=5000000.0,
        volatility_24h=25.0,
        last_update=time.time()
    )
    
    print("1. Processing first order...")
    result1 = policy.decide(order1, good_conditions)
    print(f"   First order: {result1.decision.value} - {result1.reason}")
    
    print("\n2. Processing duplicate order (same COID)...")
    result2 = policy.decide(order2, good_conditions)
    print(f"   Duplicate order: {result2.decision.value} - {result2.reason}")
    
    # Test assertion - should detect duplicate
    assert result2.decision == ExecutionDecision.REJECT
    assert "duplicate" in result2.reason.lower()
    
    print("\n✅ DUPLICATE ORDER DETECTION TEST: PASSED")
    return True

def main():
    """Run ExecutionPolicy test suite"""
    print("FASE C - EXECUTION POLICY ENFORCEMENT DEMO")
    print("Real implementation test with ExecutionPolicy.decide()")
    print("="*60)
    
    if not EXECUTION_POLICY_AVAILABLE:
        print("❌ ExecutionPolicy not available - skipping tests")
        return False
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Gate enforcement
        if test_execution_policy_gates():
            tests_passed += 1
        
        # Test 2: COID generation
        if test_client_order_id_generation():
            tests_passed += 1
        
        # Test 3: Duplicate detection
        if test_duplicate_order_detection():
            tests_passed += 1
        
        print("\n" + "="*60)
        print("EXECUTION POLICY TEST RESULTS")
        print("="*60)
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("\n🎉 ALL EXECUTION POLICY TESTS PASSED!")
            print("✅ Spread Gate: ENFORCED")
            print("✅ Depth Gate: ENFORCED")
            print("✅ Volume Gate: ENFORCED")
            print("✅ Client Order ID: AUTO-GENERATED")
            print("✅ Duplicate Detection: OPERATIONAL")
            print("\nExecutionPolicy.decide() is COMPLETE and OPERATIONAL")
        else:
            print(f"\n❌ {total_tests - tests_passed} tests failed")
        
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)