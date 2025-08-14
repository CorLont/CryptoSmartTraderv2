#!/usr/bin/env python3
"""
Simple test for execution discipline without pytest dependency
Validates double-order prevention and idempotency
"""

import sys
import os
import time
import threading

# Add src to path
sys.path.insert(0, 'src')

try:
    from cryptosmarttrader.execution.execution_discipline import (
        ExecutionPolicy, OrderRequest, MarketConditions, OrderSide,
        TimeInForce, OrderExecutor
    )
    
    def test_execution_discipline():
        """Test execution discipline system"""
        
        print("🧪 Testing Hard Execution Discipline")
        print("=" * 40)
        
        # Setup
        policy = ExecutionPolicy()
        executor = OrderExecutor(policy)
        
        market = MarketConditions(
            spread_bps=15.0,
            bid_depth_usd=100000.0,
            ask_depth_usd=100000.0,
            volume_1m_usd=500000.0,
            last_price=50000.0,
            bid_price=49992.5,
            ask_price=50007.5,
            timestamp=time.time()
        )
        
        # Test 1: Normal order execution
        print("\n1. Testing normal order execution...")
        order1 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            max_slippage_bps=20.0,
            strategy_id="test_strategy"
        )
        
        result1 = policy.decide(order1, market)
        print(f"   Result: {result1.decision.value}")
        print(f"   Reason: {result1.reason}")
        print(f"   Order ID: {order1.client_order_id}")
        
        assert result1.decision.value == "approve", "Normal order should be approved"
        print("   ✅ Normal order approved")
        
        # Mark as executed
        policy.mark_order_executed(order1.client_order_id)
        
        # Test 2: Duplicate order prevention  
        print("\n2. Testing duplicate order prevention...")
        order2 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            max_slippage_bps=20.0,
            strategy_id="test_strategy",
            client_order_id=order1.client_order_id  # Same ID!
        )
        
        result2 = policy.decide(order2, market)
        print(f"   Result: {result2.decision.value}")
        print(f"   Reason: {result2.reason}")
        
        assert result2.decision.value == "reject", "Duplicate order should be rejected"
        assert "Duplicate order ID" in result2.reason, "Should detect duplicate"
        print("   ✅ Duplicate order rejected")
        
        # Test 3: Spread gate
        print("\n3. Testing spread gate...")
        bad_market = MarketConditions(
            spread_bps=100.0,  # Too wide
            bid_depth_usd=100000.0,
            ask_depth_usd=100000.0,
            volume_1m_usd=500000.0,
            last_price=50000.0,
            bid_price=49950.0,
            ask_price=50050.0,
            timestamp=time.time()
        )
        
        order3 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            client_order_id="spread_test"
        )
        
        result3 = policy.decide(order3, bad_market)
        print(f"   Result: {result3.decision.value}")
        print(f"   Reason: {result3.reason}")
        
        assert result3.decision.value == "reject", "Wide spread should be rejected"
        assert "Spread too wide" in result3.reason, "Should detect wide spread"
        print("   ✅ Wide spread rejected")
        
        # Test 4: Depth gate
        print("\n4. Testing depth gate...")
        low_depth_market = MarketConditions(
            spread_bps=10.0,
            bid_depth_usd=5000.0,  # Too low
            ask_depth_usd=5000.0,  # Too low
            volume_1m_usd=500000.0,
            last_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            timestamp=time.time()
        )
        
        order4 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            client_order_id="depth_test"
        )
        
        result4 = policy.decide(order4, low_depth_market)
        print(f"   Result: {result4.decision.value}")
        print(f"   Reason: {result4.reason}")
        
        assert result4.decision.value == "reject", "Low depth should be rejected"
        assert "Insufficient depth" in result4.reason, "Should detect low depth"
        print("   ✅ Low depth rejected")
        
        # Test 5: Volume gate
        print("\n5. Testing volume gate...")
        low_volume_market = MarketConditions(
            spread_bps=10.0,
            bid_depth_usd=100000.0,
            ask_depth_usd=100000.0,
            volume_1m_usd=50000.0,  # Too low
            last_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            timestamp=time.time()
        )
        
        order5 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            client_order_id="volume_test"
        )
        
        result5 = policy.decide(order5, low_volume_market)
        print(f"   Result: {result5.decision.value}")
        print(f"   Reason: {result5.reason}")
        
        assert result5.decision.value == "reject", "Low volume should be rejected"
        assert "Low volume" in result5.reason, "Should detect low volume"
        print("   ✅ Low volume rejected")
        
        # Test 6: Post-only enforcement
        print("\n6. Testing post-only enforcement...")
        order6 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            time_in_force=TimeInForce.IOC,  # Not post-only
            client_order_id="tif_test"
        )
        
        result6 = policy.decide(order6, market)
        print(f"   Result: {result6.decision.value}")
        print(f"   Reason: {result6.reason}")
        
        assert result6.decision.value == "reject", "Non-post-only should be rejected"
        assert "Post-only required" in result6.reason, "Should enforce post-only"
        print("   ✅ Non-post-only rejected")
        
        # Test 7: Concurrent order protection
        print("\n7. Testing concurrent order protection...")
        
        concurrent_results = []
        
        def submit_concurrent_order():
            order = OrderRequest(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                size=0.1,
                client_order_id="concurrent_test"
            )
            result = policy.decide(order, market)
            concurrent_results.append(result.decision.value)
        
        # Submit same order from 3 threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=submit_concurrent_order)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        approvals = concurrent_results.count("approve")
        rejections = concurrent_results.count("reject")
        
        print(f"   Approvals: {approvals}, Rejections: {rejections}")
        
        assert approvals == 1, f"Expected 1 approval, got {approvals}"
        assert rejections == 2, f"Expected 2 rejections, got {rejections}"
        print("   ✅ Concurrent protection working")
        
        # Test 8: Idempotent ID generation
        print("\n8. Testing idempotent ID generation...")
        
        # Same parameters should generate same ID
        order_a = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            strategy_id="id_test"
        )
        
        order_b = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            strategy_id="id_test"
        )
        
        assert order_a.client_order_id == order_b.client_order_id, "Same params should generate same ID"
        print(f"   Generated ID: {order_a.client_order_id}")
        print("   ✅ Idempotent ID generation working")
        
        # Get stats
        stats = policy.get_stats()
        print(f"\n📊 Execution Stats:")
        print(f"   Total evaluations: {stats['total_evaluations']}")
        print(f"   Approvals: {stats['approvals']}")
        print(f"   Rejections: {stats['rejections']}")
        print(f"   Approval rate: {stats['approval_rate']:.1%}")
        
        print(f"\n🎯 All execution discipline tests passed!")
        return True
        
    if __name__ == "__main__":
        test_execution_discipline()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure execution_discipline.py is properly created")
except Exception as e:
    print(f"❌ Test failed: {e}")
    raise