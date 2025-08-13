#!/usr/bin/env python3
"""
Demo: Enterprise Execution System
Comprehensive demonstration of execution controls, tradability gates, and order deduplication.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.execution.execution_policy import (
    ExecutionPolicy,
    ExecutionParams,
    MarketConditions,
    OrderType,
    TimeInForce,
    OrderStatus,
    create_market_conditions,
)


async def demonstrate_execution_system():
    """Comprehensive demonstration of execution system capabilities."""
    print("⚡ ENTERPRISE EXECUTION SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Initialize execution system
    execution_policy = ExecutionPolicy()

    print("✅ Execution Policy System initialized")
    print(f"   Max spread: {execution_policy.limits.max_spread_bps:.0f} bps")
    print(f"   Min depth: ${execution_policy.limits.min_depth_usd:,.0f}")
    print(f"   Min volume: ${execution_policy.limits.min_volume_1m_usd:,.0f}")
    print(f"   Max slippage budget: {execution_policy.limits.max_slippage_budget_bps:.0f} bps")

    # Demo 1: Standard Order Execution (Good Conditions)
    print("\n🎯 DEMO 1: Standard Order Execution (Good Market)")
    print("-" * 50)

    good_conditions = create_market_conditions("BTC-USD", 50000.0, 50010.0, 20000.0)

    standard_params = ExecutionParams(
        symbol="BTC-USD",
        side="buy",
        quantity=0.1,
        order_type=OrderType.LIMIT,
        max_slippage_bps=30.0,
    )

    print(f"   Market: BTC-USD @ ${good_conditions.bid:,.0f}-${good_conditions.ask:,.0f}")
    print(
        f"   Spread: {good_conditions.spread_bps:.1f} bps, Volume: ${good_conditions.volume_1m:,.0f}"
    )
    print(f"   Order: {standard_params.side} {standard_params.quantity} BTC")

    execution = await execution_policy.execute_order(standard_params, good_conditions)

    print(f"   ✅ Execution Status: {execution.status.value.upper()}")
    print(f"   💰 Filled: {execution.filled_quantity} @ ${execution.average_price:,.2f}")
    print(f"   📊 Slippage: {execution.slippage_bps:.1f} bps")
    print(f"   ⏱️ Execution Time: {execution.execution_time_ms:.0f}ms")
    print(f"   💸 Fees: ${execution.fees:.2f}")
    print(f"   🆔 Client Order ID: {execution.client_order_id}")

    # Demo 2: Tradability Gates (Bad Conditions)
    print("\n🚫 DEMO 2: Tradability Gates Rejection (Poor Market)")
    print("-" * 50)

    bad_conditions = MarketConditions(
        symbol="ALTCOIN-USD",
        bid=1.0,
        ask=1.10,  # 1000 bps spread
        spread_bps=1000.0,
        depth_bid=100.0,  # Low depth
        depth_ask=100.0,
        volume_1m=500.0,  # Low volume
        volatility_1m=0.10,  # High volatility
    )

    print(f"   Market: ALTCOIN-USD @ ${bad_conditions.bid:.2f}-${bad_conditions.ask:.2f}")
    print(f"   Spread: {bad_conditions.spread_bps:.0f} bps (too wide)")
    print(f"   Depth: ${bad_conditions.depth_bid:.0f} (too low)")
    print(f"   Volume: ${bad_conditions.volume_1m:.0f} (too low)")
    print(f"   Volatility: {bad_conditions.volatility_1m:.0%} (too high)")

    bad_params = ExecutionParams(
        symbol="ALTCOIN-USD", side="buy", quantity=100.0, order_type=OrderType.MARKET
    )

    is_tradable, violations = execution_policy.check_tradability_gates(bad_conditions)
    print(f"   🚫 Tradability: {'PASS' if is_tradable else 'FAIL'}")
    if violations:
        print(f"   ⚠️ Violations: {[v.value for v in violations]}")

    bad_execution = await execution_policy.execute_order(bad_params, bad_conditions)
    print(f"   ❌ Order Status: {bad_execution.status.value.upper()}")
    print(f"   📝 Rejection Reason: {bad_execution.error_message}")

    # Demo 3: Order Deduplication
    print("\n🔒 DEMO 3: Order Deduplication System")
    print("-" * 50)

    duplicate_params = ExecutionParams(
        symbol="ETH-USD", side="buy", quantity=0.5, client_order_id="DEMO_DUPLICATE_ORDER"
    )

    eth_conditions = create_market_conditions("ETH-USD", 3000.0, 3005.0, 15000.0)

    print("   First submission:")
    first_execution = await execution_policy.execute_order(duplicate_params, eth_conditions)
    print(f"   ✅ Status: {first_execution.status.value}")
    print(f"   🆔 Order ID: {first_execution.client_order_id}")

    print("\n   Immediate resubmission (should be blocked):")
    duplicate_execution = await execution_policy.execute_order(duplicate_params, eth_conditions)
    print(f"   🚫 Status: {duplicate_execution.status.value}")
    print(f"   📝 Reason: {duplicate_execution.error_message}")

    # Demo 4: TWAP Order Execution
    print("\n📈 DEMO 4: TWAP (Time-Weighted Average Price) Order")
    print("-" * 50)

    twap_params = ExecutionParams(
        symbol="BTC-USD",
        side="buy",
        quantity=2.0,  # Large order
        order_type=OrderType.TWAP,
        max_slippage_bps=40.0,
    )

    high_volume_conditions = create_market_conditions("BTC-USD", 50000.0, 50015.0, 100000.0)

    print(
        f"   Large Order: {twap_params.quantity} BTC (~${twap_params.quantity * good_conditions.ask:,.0f})"
    )
    print(f"   Strategy: TWAP (sliced execution)")
    print(f"   Market Volume: ${high_volume_conditions.volume_1m:,.0f}")

    print("   Executing TWAP order...")
    twap_execution = await execution_policy.execute_order(twap_params, high_volume_conditions)

    print(f"   ✅ TWAP Status: {twap_execution.status.value}")
    print(f"   📦 Total Slices: {len(twap_execution.partial_fills)}")
    print(f"   💰 Average Price: ${twap_execution.average_price:,.2f}")
    print(f"   📊 Total Slippage: {twap_execution.slippage_bps:.1f} bps")
    print(f"   💸 Total Fees: ${twap_execution.fees:.2f}")

    if twap_execution.partial_fills:
        print("   📋 Slice Details:")
        for fill in twap_execution.partial_fills[:3]:  # Show first 3 slices
            print(f"      Slice {fill['slice']}: {fill['quantity']:.3f} @ ${fill['price']:,.2f}")
        if len(twap_execution.partial_fills) > 3:
            print(f"      ... and {len(twap_execution.partial_fills) - 3} more slices")

    # Demo 5: Iceberg Order Execution
    print("\n🧊 DEMO 5: Iceberg Order (Hidden Quantity)")
    print("-" * 50)

    iceberg_params = ExecutionParams(
        symbol="ETH-USD",
        side="sell",
        quantity=5.0,  # Large sell order
        order_type=OrderType.ICEBERG,
        max_slippage_bps=25.0,
    )

    print(
        f"   Iceberg Order: {iceberg_params.quantity} ETH (~${iceberg_params.quantity * eth_conditions.bid:,.0f})"
    )
    print(f"   Strategy: Hidden quantity execution")

    print("   Executing iceberg order...")
    iceberg_execution = await execution_policy.execute_order(iceberg_params, eth_conditions)

    print(f"   ✅ Iceberg Status: {iceberg_execution.status.value}")
    print(f"   🔍 Visible Slices: {len(iceberg_execution.partial_fills)}")
    print(f"   💰 Average Price: ${iceberg_execution.average_price:,.2f}")
    print(f"   📊 Total Slippage: {iceberg_execution.slippage_bps:.1f} bps")

    # Demo 6: Execution Strategy Optimization
    print("\n🎛️ DEMO 6: Automatic Execution Strategy Optimization")
    print("-" * 50)

    # Wide spread scenario
    wide_spread_conditions = MarketConditions(
        symbol="DEFI-USD",
        bid=100.0,
        ask=102.0,  # 200 bps spread
        spread_bps=200.0,
        depth_bid=2000.0,
        depth_ask=2000.0,
        volume_1m=25000.0,
        volatility_1m=0.03,
    )

    original_params = ExecutionParams(
        symbol="DEFI-USD", side="buy", quantity=10.0, order_type=OrderType.LIMIT
    )

    optimized_params = execution_policy.calculate_optimal_execution_strategy(
        original_params, wide_spread_conditions
    )

    print(f"   Original Strategy: {original_params.order_type.value}")
    print(f"   Market Spread: {wide_spread_conditions.spread_bps:.0f} bps (wide)")
    print(f"   📈 Optimized Strategy: {optimized_params.order_type.value}")
    print(f"   🎯 Post-Only: {optimized_params.post_only}")
    print(f"   ⏰ Time in Force: {optimized_params.time_in_force.value}")
    print(f"   💲 Optimized Price: ${optimized_params.limit_price:.2f}")

    # Demo 7: Slippage Budget Enforcement
    print("\n💰 DEMO 7: Slippage Budget Enforcement")
    print("-" * 50)

    # Simulate execution with different slippage scenarios
    test_conditions = create_market_conditions("BTC-USD", 50000.0, 50010.0, 20000.0)

    slippage_params = ExecutionParams(
        symbol="BTC-USD",
        side="buy",
        quantity=0.1,
        max_slippage_bps=20.0,  # Tight budget
    )

    print(f"   Slippage Budget: {slippage_params.max_slippage_bps:.0f} bps")
    print(f"   Reference Price: ${test_conditions.ask:,.2f}")

    # Test different execution prices
    test_prices = [50015.0, 50025.0, 50035.0]  # 10, 30, 50 bps slippage

    for price in test_prices:
        within_budget, actual_slippage = execution_policy.validate_slippage_budget(
            slippage_params, price, test_conditions
        )

        status = "✅ WITHIN BUDGET" if within_budget else "❌ EXCEEDS BUDGET"
        print(f"   Execution @ ${price:,.2f}: {actual_slippage:.1f} bps - {status}")

    # Demo 8: Performance and Statistics
    print("\n📊 DEMO 8: Execution Performance Statistics")
    print("-" * 50)

    stats = execution_policy.get_execution_stats()

    print(f"   📈 Total Orders Processed: {stats['total_orders']}")
    print(f"   ✅ Successful Executions: {stats['successful_executions']}")
    print(f"   📊 Success Rate: {stats['success_rate_percent']:.1f}%")
    print(f"   🚫 Rejected by Gates: {stats['rejected_by_gates']}")
    print(f"   ⚠️ Slippage Violations: {stats['slippage_violations']}")
    print(f"   📉 Average Slippage: {stats['average_slippage_bps']:.1f} bps")
    print(f"   ⏱️ Average Execution Time: {stats['average_execution_time_ms']:.0f}ms")
    print(f"   🔄 Active Orders: {stats['active_orders']}")
    print(f"   📋 Completed Orders: {stats['completed_orders']}")
    print(f"   🔒 Deduplication Cache: {stats['deduplication_cache_size']} entries")

    # Demo 9: Order Management
    print("\n🛠️ DEMO 9: Order Management & Status Tracking")
    print("-" * 50)

    # Show recent orders
    print("   Recent Order Status:")
    for order_id, order in list(execution_policy.completed_orders.items())[-3:]:
        print(f"      {order_id[:20]}... -> {order.status.value} ({order.symbol})")

    # Test order cancellation (simulate active order)
    from cryptosmarttrader.execution.execution_policy import OrderExecution
    import time

    mock_active_order = OrderExecution(
        client_order_id="DEMO_CANCEL_ORDER",
        exchange_order_id="EX_DEMO",
        symbol="BTC-USD",
        side="buy",
        quantity=0.1,
        filled_quantity=0.0,
        average_price=0.0,
        status=OrderStatus.SUBMITTED,
        slippage_bps=0.0,
        fees=0.0,
        execution_time_ms=0.0,
        timestamp=datetime.now(),
    )

    execution_policy.active_orders["DEMO_CANCEL_ORDER"] = mock_active_order

    print(f"   📤 Active Order Added: DEMO_CANCEL_ORDER")
    cancel_success = execution_policy.cancel_order("DEMO_CANCEL_ORDER")
    print(f"   🚫 Order Cancellation: {'SUCCESS' if cancel_success else 'FAILED'}")

    print("\n✅ EXECUTION SYSTEM DEMONSTRATION COMPLETED")
    print("=" * 60)

    final_stats = execution_policy.get_execution_stats()
    print(f"📊 FINAL STATISTICS:")
    print(
        f"   Total Executions: {final_stats['successful_executions']}/{final_stats['total_orders']}"
    )
    print(f"   Success Rate: {final_stats['success_rate_percent']:.1f}%")
    print(
        f"   Average Performance: {final_stats['average_slippage_bps']:.1f} bps slippage, {final_stats['average_execution_time_ms']:.0f}ms"
    )


if __name__ == "__main__":
    print("⚡ CRYPTOSMARTTRADER V2 - EXECUTION SYSTEM DEMO")
    print("=" * 60)

    try:
        # Import datetime here for the demo
        from datetime import datetime

        asyncio.run(demonstrate_execution_system())
        print("\n🏆 Execution system demonstration completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
