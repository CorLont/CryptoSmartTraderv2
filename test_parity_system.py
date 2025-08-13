#!/usr/bin/env python3
"""
Test backtest-live parity system functionality
"""

import sys
import time
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_execution_simulation():
    """Test execution simulation system"""
    
    print("🎮 Testing Execution Simulation")
    print("=" * 35)
    
    try:
        from cryptosmarttrader.simulation.execution_simulator import (
            ExecutionSimulator, MarketConditions, OrderType, FillType, FeeStructure
        )
        
        # Setup
        simulator = ExecutionSimulator()
        
        # Create market conditions
        market = MarketConditions(
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=10.0,
            ask_size=8.0,
            last_price=50000.0,
            volume_1m=1000000.0,
            volatility=0.02,
            timestamp=time.time()
        )
        
        print(f"Market conditions:")
        print(f"  Spread: {market.spread_bps:.1f} bps")
        print(f"  Mid price: ${market.mid_price:,.0f}")
        
        # Submit market order
        order = simulator.submit_order(
            order_id="test_001",
            symbol="BTC/USD",
            side="buy",
            order_type=OrderType.MARKET,
            size=1.0,
            market_conditions=market
        )
        
        print(f"\nOrder submitted:")
        print(f"  ID: {order.order_id}")
        print(f"  Status: {order.status.value}")
        print(f"  Queue position: {order.queue_position}")
        print(f"  Submit latency: {order.submit_latency_ms:.1f}ms")
        
        # Process execution
        fills = simulator.process_order_execution("test_001", market)
        
        if fills:
            fill = fills[0]
            print(f"\nOrder filled:")
            print(f"  Size: {fill.size}")
            print(f"  Price: ${fill.price:.2f}")
            print(f"  Fee: ${fill.fee:.4f}")
            print(f"  Type: {fill.fill_type.value}")
            print(f"  Latency: {fill.latency_ms:.1f}ms")
        
        # Get statistics
        stats = simulator.get_execution_statistics()
        print(f"\nExecution statistics:")
        print(f"  Total orders: {stats['total_orders']}")
        print(f"  Fill rate: {stats['fill_rate']:.1%}")
        print(f"  Avg slippage: {stats['avg_slippage_bps']:.1f} bps")
        print(f"  Total fees: ${stats['total_fees_usd']:.4f}")
        
        print("✅ Execution simulation working")
        return True
        
    except Exception as e:
        print(f"❌ Execution simulation test failed: {e}")
        return False

def test_parity_tracking():
    """Test parity tracking system"""
    
    print("\n📊 Testing Parity Tracking")
    print("=" * 30)
    
    try:
        from cryptosmarttrader.simulation.parity_tracker import (
            ParityTracker, ParityThresholds
        )
        
        # Setup
        thresholds = ParityThresholds(
            warning_threshold_bps=20.0,
            critical_threshold_bps=50.0,
            disable_threshold_bps=100.0
        )
        
        tracker = ParityTracker("test_strategy", thresholds)
        
        # Record backtest execution
        tracker.record_backtest_execution(
            trade_id="trade_001",
            symbol="BTC/USD",
            side="buy",
            size=1.0,
            price=50000.0,
            timestamp=time.time(),
            fees=5.0
        )
        
        print("Recorded backtest execution")
        
        # Record live execution with slippage
        tracker.record_live_execution(
            trade_id="trade_001",
            price=50015.0,  # 3 bps slippage
            timestamp=time.time() + 0.2,
            fees=7.5,
            slippage=0.0003,
            latency_ms=125.0
        )
        
        print("Recorded live execution")
        
        # Check completed trade
        completed = tracker.completed_trades[0]
        print(f"Price difference: {completed.price_diff_bps:.1f} bps")
        print(f"Fee difference: {completed.fee_diff_bps:.1f} bps")
        
        # Calculate tracking error
        tracking_error = tracker.calculate_daily_tracking_error()
        print(f"Daily tracking error: {tracking_error:.1f} bps")
        
        # Generate daily report
        report = tracker.generate_daily_report()
        print(f"\nDaily report:")
        print(f"  Completed trades: {report.completed_trades}")
        print(f"  Tracking error: {report.tracking_error_bps:.1f} bps")
        print(f"  Status: {report.parity_status.value}")
        
        # Get summary
        summary = tracker.get_parity_summary()
        print(f"\nParity summary:")
        print(f"  Strategy: {summary['strategy_id']}")
        print(f"  Status: {summary['current_status']}")
        print(f"  Disabled: {summary['is_disabled']}")
        
        print("✅ Parity tracking working")
        return True
        
    except Exception as e:
        print(f"❌ Parity tracking test failed: {e}")
        return False

def test_auto_disable():
    """Test auto-disable functionality"""
    
    print("\n🚨 Testing Auto-Disable")
    print("=" * 25)
    
    try:
        from cryptosmarttrader.simulation.parity_tracker import (
            ParityTracker, ParityThresholds
        )
        
        # Setup with low thresholds for testing
        thresholds = ParityThresholds(
            warning_threshold_bps=5.0,
            critical_threshold_bps=10.0,
            disable_threshold_bps=15.0
        )
        
        tracker = ParityTracker("auto_disable_test", thresholds)
        
        # Add trade with large tracking error
        tracker.record_backtest_execution(
            trade_id="large_error_001",
            symbol="BTC/USD",
            side="buy",
            size=1.0,
            price=50000.0,
            timestamp=time.time(),
            fees=5.0
        )
        
        # Large slippage that should trigger auto-disable
        tracker.record_live_execution(
            trade_id="large_error_001",
            price=50100.0,  # 200 bps slippage
            timestamp=time.time() + 0.1,
            fees=8.0,
            slippage=0.002,
            latency_ms=200.0
        )
        
        # Generate report
        report = tracker.generate_daily_report()
        
        print(f"Large tracking error: {report.tracking_error_bps:.1f} bps")
        print(f"Auto-disable triggered: {report.auto_disable_triggered}")
        print(f"Tracker disabled: {tracker.is_disabled}")
        print(f"Disable reason: {tracker.disable_reason}")
        
        if tracker.is_disabled:
            print("✅ Auto-disable working correctly")
            return True
        else:
            print("❌ Auto-disable not triggered")
            return False
        
    except Exception as e:
        print(f"❌ Auto-disable test failed: {e}")
        return False

def test_integration():
    """Test integration between simulation and tracking"""
    
    print("\n🔗 Testing Integration")
    print("=" * 20)
    
    try:
        from cryptosmarttrader.simulation import (
            get_execution_simulator, get_parity_tracker,
            MarketConditions, OrderType
        )
        
        # Setup
        simulator = get_execution_simulator()
        tracker = get_parity_tracker("integration_test")
        
        # Market conditions
        market = MarketConditions(
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=10.0,
            ask_size=8.0,
            last_price=50000.0,
            volume_1m=1000000.0,
            volatility=0.02,
            timestamp=time.time()
        )
        
        # Record backtest
        backtest_price = 50000.0
        tracker.record_backtest_execution(
            trade_id="integration_001",
            symbol="BTC/USD",
            side="buy",
            size=1.0,
            price=backtest_price,
            timestamp=time.time(),
            fees=5.0
        )
        
        # Simulate live execution
        order = simulator.submit_order(
            order_id="integration_001",
            symbol="BTC/USD",
            side="buy",
            order_type=OrderType.MARKET,
            size=1.0,
            market_conditions=market
        )
        
        fills = simulator.process_order_execution("integration_001", market)
        
        if fills:
            fill = fills[0]
            
            # Record in tracker
            tracker.record_live_execution(
                trade_id="integration_001",
                price=fill.price,
                timestamp=fill.timestamp,
                fees=fill.fee,
                slippage=(fill.price - backtest_price) / backtest_price,
                latency_ms=fill.latency_ms
            )
            
            print(f"Integration test:")
            print(f"  Backtest price: ${backtest_price:,.0f}")
            print(f"  Live price: ${fill.price:.2f}")
            print(f"  Slippage: {((fill.price - backtest_price) / backtest_price) * 10000:.1f} bps")
            
            # Generate report
            report = tracker.generate_daily_report()
            print(f"  Tracking error: {report.tracking_error_bps:.1f} bps")
            
            print("✅ Integration working")
            return True
        else:
            print("⚠️ Order not filled")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Backtest-Live Parity System")
    print("=" * 45)
    
    all_passed = True
    
    try:
        # Run all tests
        all_passed &= test_execution_simulation()
        all_passed &= test_parity_tracking() 
        all_passed &= test_auto_disable()
        all_passed &= test_integration()
        
        if all_passed:
            print("\n🎉 ALL PARITY TESTS PASSED!")
            print("\nSummary:")
            print("✅ Execution simulation with realistic fees/latency/slippage")
            print("✅ Parity tracking with daily tracking error calculation")
            print("✅ Auto-disable protection when drift exceeds thresholds")
            print("✅ Integration between simulation and tracking systems")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")