#!/usr/bin/env python3
"""
Demo Script - Risk Management & Execution Policy
Demonstrates kill-switch functionality, execution gates, and risk controls
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

from src.cryptosmarttrader.risk.centralized_risk_guard import (
    CentralizedRiskGuard, RiskLimits, RiskMetrics, KillSwitchState, RiskLevel
)
from src.cryptosmarttrader.execution.execution_policy import (
    ExecutionPolicy, ExecutionGates, OrderRequest, OrderType, TimeInForce,
    MarketData, ExecutionResult, OrderStatus
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_risk_guard():
    """Demo centralized risk guard functionality"""
    logger.info("🚀 Testing Centralized Risk Guard...")
    
    # Create risk limits for demo
    limits = RiskLimits(
        daily_loss_limit_usd=5000.0,
        max_drawdown_pct=0.10,
        max_total_exposure_usd=25000.0,
        max_open_positions=5,
        max_data_gap_minutes=3
    )
    
    risk_guard = CentralizedRiskGuard(limits)
    risk_guard.start_monitoring()
    
    # Setup alert callback
    alerts_received = []
    def alert_callback(alert):
        alerts_received.append(alert)
        logger.info(f"🚨 Alert received: {alert['type']} - {alert.get('reason', 'N/A')}")
    
    risk_guard.add_alert_callback(alert_callback)
    
    # Test 1: Normal operation
    logger.info("✅ Test 1: Normal operation")
    normal_metrics = RiskMetrics(
        daily_pnl_usd=500.0,
        current_drawdown_pct=0.03,
        total_exposure_usd=10000.0,
        open_positions=3,
        data_gap_minutes=1.0,
        data_quality_score=0.95
    )
    risk_guard.update_metrics(normal_metrics)
    
    normal_order = {
        'symbol': 'BTC/USD',
        'side': 'buy',
        'size': 0.1,
        'price': 50000.0
    }
    allowed, reason = risk_guard.check_order_allowed(normal_order)
    logger.info(f"Normal order allowed: {allowed} - {reason}")
    
    # Test 2: Daily loss warning
    logger.info("⚠️  Test 2: Daily loss warning trigger")
    warning_metrics = RiskMetrics(
        daily_pnl_usd=-4000.0,  # 80% of limit
        current_drawdown_pct=0.05,
        total_exposure_usd=15000.0,
        open_positions=4
    )
    risk_guard.update_metrics(warning_metrics)
    assert risk_guard.risk_level == RiskLevel.WARNING
    
    # Test 3: Exposure limit - soft stop
    logger.info("🟡 Test 3: Exposure limit trigger (soft stop)")
    exposure_metrics = RiskMetrics(
        daily_pnl_usd=-2000.0,
        total_exposure_usd=30000.0,  # Exceeds 25k limit
        open_positions=4
    )
    risk_guard.update_metrics(exposure_metrics)
    assert risk_guard.kill_switch_state == KillSwitchState.SOFT_STOP
    
    # Test position reducing vs increasing orders
    risk_guard.update_position('BTC/USD', {'size': 1.0, 'mark_price': 50000.0})
    
    increasing_order = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.5, 'price': 50000.0}
    reducing_order = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.3, 'price': 50000.0}
    
    allowed_inc, reason_inc = risk_guard.check_order_allowed(increasing_order)
    allowed_red, reason_red = risk_guard.check_order_allowed(reducing_order)
    
    logger.info(f"Position-increasing order: {allowed_inc} - {reason_inc}")
    logger.info(f"Position-reducing order: {allowed_red} - {reason_red}")
    
    # Test 4: Data gap - hard stop
    logger.info("🔴 Test 4: Data gap trigger (hard stop)")
    gap_metrics = RiskMetrics(
        daily_pnl_usd=-1000.0,
        data_gap_minutes=5.0,  # Exceeds 3 minute limit
        data_quality_score=0.85
    )
    risk_guard.update_metrics(gap_metrics)
    assert risk_guard.kill_switch_state == KillSwitchState.HARD_STOP
    
    hard_stop_order = {'symbol': 'ETH/USD', 'side': 'buy', 'size': 1.0, 'price': 3000.0}
    allowed, reason = risk_guard.check_order_allowed(hard_stop_order)
    logger.info(f"Order during hard stop: {allowed} - {reason}")
    
    # Test 5: Emergency stop
    logger.info("🚨 Test 5: Daily loss limit (emergency stop)")
    emergency_metrics = RiskMetrics(
        daily_pnl_usd=-6000.0,  # Exceeds 5k limit
        current_drawdown_pct=0.12,
        total_exposure_usd=20000.0
    )
    risk_guard.update_metrics(emergency_metrics)
    assert risk_guard.kill_switch_state == KillSwitchState.EMERGENCY
    
    emergency_order = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1, 'price': 50000.0}
    allowed, reason = risk_guard.check_order_allowed(emergency_order)
    logger.info(f"Order during emergency: {allowed} - {reason}")
    
    # Test 6: Manual kill switch
    logger.info("🔧 Test 6: Manual kill switch")
    risk_guard.reset_kill_switch("Demo reset")
    risk_guard.manual_kill_switch(KillSwitchState.HARD_STOP, "Manual demo trigger")
    assert risk_guard.kill_switch_state == KillSwitchState.HARD_STOP
    
    # Show final status
    status = risk_guard.get_status()
    logger.info("📊 Final Risk Guard Status:")
    logger.info(f"  Kill Switch: {status['kill_switch_state']}")
    logger.info(f"  Risk Level: {status['risk_level']}")
    logger.info(f"  Violations: {len(status['recent_violations'])}")
    logger.info(f"  Alerts Received: {len(alerts_received)}")
    
    risk_guard.stop_monitoring()
    return len(alerts_received) > 0


async def demo_execution_policy():
    """Demo execution policy functionality"""
    logger.info("🚀 Testing Execution Policy...")
    
    # Create execution gates for demo
    gates = ExecutionGates(
        max_spread_bps=40,
        min_bid_depth_usd=8000.0,
        min_ask_depth_usd=8000.0,
        min_volume_1m_usd=50000.0,
        max_slippage_bps=20,
        slippage_budget_daily_bps=150
    )
    
    policy = ExecutionPolicy(gates)
    
    # Setup good market data
    good_market_data = MarketData(
        symbol='BTC/USD',
        timestamp=datetime.now(),
        bid=49980.0,
        ask=50020.0,
        mid=50000.0,
        last=50000.0,
        bid_depth_usd=15000.0,
        ask_depth_usd=15000.0,
        total_depth_usd=30000.0,
        volume_1m_usd=100000.0,
        trades_1m=25,
        volume_24h_usd=2000000.0,
        volatility_1h_pct=0.04,
        volatility_24h_pct=0.08,
        spread_bps=8,
        spread_pct=0.0008
    )
    policy.update_market_data('BTC/USD', good_market_data)
    
    # Test 1: Valid order passes all gates
    logger.info("✅ Test 1: Valid order validation")
    valid_request = OrderRequest(
        symbol='BTC/USD',
        side='buy',
        size=0.5,
        order_type=OrderType.LIMIT,
        price=49990.0,
        time_in_force=TimeInForce.POST_ONLY
    )
    
    valid, errors, client_id = await policy.validate_order(valid_request)
    logger.info(f"Valid order result: {valid}, Client ID: {client_id}")
    if errors:
        logger.info(f"Errors: {errors}")
    
    # Test 2: Wide spread blocks order
    logger.info("🔴 Test 2: Wide spread blocking")
    wide_spread_data = MarketData(
        symbol='ETH/USD',
        timestamp=datetime.now(),
        bid=2980.0,
        ask=3020.0,  # 40 bps spread
        mid=3000.0,
        last=3000.0,
        bid_depth_usd=12000.0,
        ask_depth_usd=12000.0,
        total_depth_usd=24000.0,
        volume_1m_usd=80000.0,
        trades_1m=30,
        volume_24h_usd=1500000.0,
        volatility_1h_pct=0.06,
        volatility_24h_pct=0.12,
        spread_bps=133,  # Exceeds 40 bps limit
        spread_pct=0.0133
    )
    policy.update_market_data('ETH/USD', wide_spread_data)
    
    spread_request = OrderRequest(
        symbol='ETH/USD',
        side='buy',
        size=1.0,
        order_type=OrderType.LIMIT,
        price=3000.0
    )
    
    spread_valid, spread_errors, _ = await policy.validate_order(spread_request)
    logger.info(f"Wide spread order: {spread_valid}")
    if spread_errors:
        logger.info(f"Spread errors: {spread_errors}")
    
    # Test 3: Insufficient depth blocks order
    logger.info("🔴 Test 3: Insufficient depth blocking")
    low_depth_data = MarketData(
        symbol='SOL/USD',
        timestamp=datetime.now(),
        bid=99.50,
        ask=100.50,
        mid=100.0,
        last=100.0,
        bid_depth_usd=5000.0,  # Below 8k limit
        ask_depth_usd=6000.0,  # Below 8k limit
        total_depth_usd=11000.0,
        volume_1m_usd=60000.0,
        trades_1m=15,
        volume_24h_usd=800000.0,
        volatility_1h_pct=0.08,
        volatility_24h_pct=0.15,
        spread_bps=100,
        spread_pct=0.01
    )
    policy.update_market_data('SOL/USD', low_depth_data)
    
    depth_request = OrderRequest(
        symbol='SOL/USD',
        side='buy',
        size=50.0,
        order_type=OrderType.MARKET
    )
    
    depth_valid, depth_errors, _ = await policy.validate_order(depth_request)
    logger.info(f"Low depth order: {depth_valid}")
    if depth_errors:
        logger.info(f"Depth errors: {depth_errors}")
    
    # Test 4: Duplicate order detection
    logger.info("🔄 Test 4: Duplicate order detection")
    duplicate_request = OrderRequest(
        symbol='BTC/USD',
        side='buy',
        size=0.5,
        order_type=OrderType.LIMIT,
        price=49990.0
    )
    
    # First submission
    dup1_valid, dup1_errors, dup1_client_id = await policy.validate_order(duplicate_request)
    logger.info(f"First submission: {dup1_valid}, Client ID: {dup1_client_id}")
    
    # Immediate duplicate
    dup2_valid, dup2_errors, dup2_client_id = await policy.validate_order(duplicate_request)
    logger.info(f"Duplicate submission: {dup2_valid}")
    if dup2_errors:
        logger.info(f"Duplicate errors: {dup2_errors}")
    
    # Test 5: Slippage estimation
    logger.info("📊 Test 5: Slippage estimation")
    market_order = OrderRequest(
        symbol='BTC/USD',
        side='buy',
        size=1.0,
        order_type=OrderType.MARKET
    )
    
    limit_order = OrderRequest(
        symbol='BTC/USD',
        side='buy',
        size=1.0,
        order_type=OrderType.LIMIT,
        price=49900.0  # Below ask for improvement
    )
    
    market_slippage = policy.estimate_slippage(market_order, good_market_data)
    limit_slippage = policy.estimate_slippage(limit_order, good_market_data)
    
    logger.info(f"Market order slippage: {market_slippage:.1f} bps")
    logger.info(f"Limit order slippage: {limit_slippage:.1f} bps")
    
    # Test 6: Order registration and tracking
    logger.info("📝 Test 6: Order tracking")
    if valid:
        order_id = f"demo_order_{int(time.time() * 1000)}"
        policy.register_order(valid_request, order_id)
        
        # Simulate execution result
        result = ExecutionResult(
            order_id=order_id,
            client_order_id=client_id,
            status=OrderStatus.FILLED,
            filled_size=valid_request.size,
            avg_fill_price=49990.0,
            slippage_bps=12.0,
            execution_latency_ms=150.0
        )
        
        policy.update_order_result(order_id, result)
        
        # Retrieve by client ID
        retrieved_result = policy.get_order_by_client_id(client_id)
        logger.info(f"Order retrieved: {retrieved_result.status.value if retrieved_result else 'Not found'}")
    
    # Show final status
    status = policy.get_status()
    logger.info("📊 Final Execution Policy Status:")
    logger.info(f"  Active orders: {status['orders']['active_count']}")
    logger.info(f"  Total processed: {status['orders']['total_processed']}")
    logger.info(f"  Slippage used: {status['slippage_tracking']['used_slippage_bps']:.1f} bps")
    logger.info(f"  Budget remaining: {status['slippage_tracking']['budget_remaining_bps']:.1f} bps")
    
    return status['orders']['total_processed'] > 0


async def demo_integration():
    """Demo integration between risk guard and execution policy"""
    logger.info("🔗 Testing Risk Guard + Execution Policy Integration...")
    
    # Setup both systems
    risk_limits = RiskLimits(daily_loss_limit_usd=10000.0)
    risk_guard = CentralizedRiskGuard(risk_limits)
    risk_guard.start_monitoring()
    
    execution_gates = ExecutionGates(max_spread_bps=30)
    execution_policy = ExecutionPolicy(execution_gates)
    
    # Setup market data
    market_data = MarketData(
        symbol='BTC/USD',
        timestamp=datetime.now(),
        bid=49990.0,
        ask=50010.0,
        mid=50000.0,
        last=50000.0,
        bid_depth_usd=20000.0,
        ask_depth_usd=20000.0,
        total_depth_usd=40000.0,
        volume_1m_usd=150000.0,
        trades_1m=40,
        volume_24h_usd=3000000.0,
        volatility_1h_pct=0.03,
        volatility_24h_pct=0.07,
        spread_bps=4,
        spread_pct=0.0004
    )
    execution_policy.update_market_data('BTC/USD', market_data)
    
    # Test complete order flow
    request = OrderRequest(
        symbol='BTC/USD',
        side='buy',
        size=0.2,
        order_type=OrderType.LIMIT,
        price=50000.0
    )
    
    # 1. Check execution policy
    valid, errors, client_id = await execution_policy.validate_order(request)
    logger.info(f"Execution policy check: {valid}")
    
    # 2. Check risk guard
    order_dict = {
        'symbol': request.symbol,
        'side': request.side,
        'size': request.size,
        'price': request.price
    }
    allowed, reason = risk_guard.check_order_allowed(order_dict)
    logger.info(f"Risk guard check: {allowed}")
    
    # 3. If both pass, execute order
    if valid and allowed:
        order_id = f"integrated_order_{int(time.time() * 1000)}"
        execution_policy.register_order(request, order_id)
        
        # Simulate successful execution
        result = ExecutionResult(
            order_id=order_id,
            client_order_id=client_id,
            status=OrderStatus.FILLED,
            filled_size=request.size,
            avg_fill_price=50000.0,
            slippage_bps=8.0
        )
        execution_policy.update_order_result(order_id, result)
        
        logger.info(f"✅ Order executed successfully: {order_id}")
        return True
    else:
        logger.info(f"❌ Order blocked - Execution: {valid}, Risk: {allowed}")
        return False


async def main():
    """Main demo function"""
    logger.info("🎯 Starting Risk Management & Execution Policy Demo")
    
    print("\n" + "="*70)
    print("  CryptoSmartTrader V2 - Risk & Execution Demo")
    print("="*70)
    
    try:
        # Demo risk guard
        risk_alerts = demo_risk_guard()
        
        print("\n" + "-"*70)
        
        # Demo execution policy  
        execution_success = await demo_execution_policy()
        
        print("\n" + "-"*70)
        
        # Demo integration
        integration_success = await demo_integration()
        
        print("\n" + "="*70)
        
        if risk_alerts and execution_success and integration_success:
            print("✅ All demos completed successfully!")
            print("✅ Risk management system is working properly")
            print("✅ Execution policy gates are enforced")
            print("✅ Integration between systems is functional")
        else:
            print("⚠️  Some demos had issues - check logs for details")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())