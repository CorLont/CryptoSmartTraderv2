#!/usr/bin/env python3
"""
Demo: Backtest-Live Parity System
Comprehensive demonstration of execution simulation and parity monitoring.
"""

import asyncio
import numpy as np
import pandas as pd
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.parity import (
    create_execution_simulator, create_parity_analyzer,
    SimulationConfig, OrderType, OrderSide, ParityStatus
)


def generate_market_data():
    """Generate synthetic market data for simulation."""
    # Create realistic market data with bid/ask spreads
    np.random.seed(42)
    
    n_days = 30
    dates = pd.date_range(start='2024-12-01', periods=n_days, freq='H')
    
    # Generate price series with realistic volatility
    returns = np.random.normal(0.0001, 0.02, len(dates))  # Hourly returns
    prices = [50000.0]  # Starting BTC price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume data
    volumes = np.random.lognormal(15, 0.5, len(dates))  # Log-normal volume
    
    market_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return market_data


def generate_backtest_data():
    """Generate synthetic backtest performance data."""
    np.random.seed(123)
    
    # Generate 30 days of backtest returns
    n_days = 30
    dates = pd.date_range(start='2024-12-01', periods=n_days, freq='D')
    
    # Simulate strategy returns with some skill
    market_returns = np.random.normal(0.001, 0.03, n_days)  # Daily market returns
    alpha_returns = np.random.normal(0.002, 0.01, n_days)   # Alpha generation
    
    strategy_returns = market_returns + alpha_returns
    
    # Calculate cumulative performance
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    backtest_data = {
        'return': cumulative_returns[-1],  # Total return
        'return_series': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'dates': dates,
        'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252),
        'max_drawdown': np.min(cumulative_returns),
        'win_rate': np.mean(strategy_returns > 0)
    }
    
    return backtest_data


def generate_live_data(backtest_data, execution_results):
    """Generate synthetic live trading data with execution costs."""
    
    # Start with backtest returns
    backtest_returns = backtest_data['return_series'].copy()
    
    # Apply execution costs and slippage
    live_returns = []
    for i, bt_return in enumerate(backtest_returns):
        # Base return matches backtest
        live_return = bt_return
        
        # Subtract execution costs (estimated)
        if abs(bt_return) > 0.005:  # Active trading days
            # Fee impact
            fee_impact = -0.0002  # ~2 bps average
            
            # Slippage impact  
            slippage_impact = -abs(bt_return) * 0.1  # Proportional slippage
            
            # Timing impact (small random)
            timing_impact = np.random.normal(0, 0.0001)
            
            live_return += fee_impact + slippage_impact + timing_impact
        
        live_returns.append(live_return)
    
    live_returns = np.array(live_returns)
    cumulative_live = np.cumprod(1 + live_returns) - 1
    
    live_data = {
        'return': cumulative_live[-1],
        'return_series': live_returns,
        'cumulative_returns': cumulative_live,
        'dates': backtest_data['dates'],
        'execution_results': execution_results,
        'total_fees': sum(result.total_fees for result in execution_results),
        'avg_slippage': np.mean([result.slippage_bps for result in execution_results])
    }
    
    return live_data


async def demonstrate_backtest_live_parity():
    """Comprehensive demonstration of backtest-live parity system."""
    print("🎯 BACKTEST-LIVE PARITY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data
    print("📊 Generating synthetic market and performance data...")
    market_data = generate_market_data()
    backtest_data = generate_backtest_data()
    print(f"   Generated {len(market_data)} hours of market data")
    print(f"   Backtest performance: {backtest_data['return']:.2%} return")
    
    # Initialize systems
    sim_config = SimulationConfig(
        maker_fee_bps=5.0,      # 0.05% maker
        taker_fee_bps=15.0,     # 0.15% taker
        base_slippage_bps=8.0,  # 0.08% base slippage
        partial_fill_probability=0.20
    )
    
    execution_simulator = create_execution_simulator(sim_config)
    parity_analyzer = create_parity_analyzer(
        tracking_error_threshold_bps=20.0,  # 20 bps threshold
        drift_detection_window=7            # 7-day window
    )
    
    print("✅ Execution simulator and parity analyzer initialized")
    
    # Demo 1: Execution Simulation
    print("\n🔧 DEMO 1: Execution Simulation")
    print("-" * 40)
    
    print("   Simulating order executions...")
    execution_results = []
    
    # Simulate various order types
    test_orders = [
        ("BTC-USD", OrderSide.BUY, OrderType.MARKET, 0.5, None),
        ("BTC-USD", OrderSide.SELL, OrderType.LIMIT, 0.3, 51000.0),
        ("BTC-USD", OrderSide.BUY, OrderType.MARKET, 1.2, None),
        ("BTC-USD", OrderSide.SELL, OrderType.LIMIT, 0.8, 49500.0),
        ("BTC-USD", OrderSide.BUY, OrderType.LIMIT, 0.6, 49000.0),
    ]
    
    for i, (symbol, side, order_type, quantity, limit_price) in enumerate(test_orders):
        print(f"\n      Order {i+1}: {side.value} {quantity} {symbol} ({order_type.value})")
        
        # Use recent market data for simulation
        recent_data = market_data.iloc[-24:]  # Last 24 hours
        volume_data = market_data['volume'].iloc[-24:]
        
        result = execution_simulator.simulate_order_execution(
            order_id=f"ORDER_{i+1}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            market_data=recent_data,
            volume_data=volume_data
        )
        
        execution_results.append(result)
        
        print(f"         Executed: {result.executed_quantity:.3f} at ${result.avg_fill_price:.2f}")
        print(f"         Slippage: {result.slippage_bps:.1f} bps")
        print(f"         Fees: ${result.total_fees:.2f}")
        print(f"         Latency: {result.latency_ms:.0f}ms")
        print(f"         Partial Fill: {result.partial_fill}")
        print(f"         Market Impact: {result.market_impact_bps:.1f} bps")
    
    # Show simulation statistics
    sim_stats = execution_simulator.get_simulation_statistics()
    print(f"\n   Execution Statistics:")
    print(f"      Total Orders: {sim_stats['total_orders']}")
    print(f"      Fill Rate: {sim_stats['fill_rate']:.1%}")
    print(f"      Avg Slippage: {sim_stats['average_slippage_bps']:.1f} bps")
    print(f"      Max Slippage: {sim_stats['max_slippage_bps']:.1f} bps")
    print(f"      Avg Latency: {sim_stats['average_latency_ms']:.0f}ms")
    print(f"      Total Fees: ${sim_stats['total_fees']:.2f}")
    print(f"      Partial Fill Rate: {sim_stats['partial_fill_rate']:.1%}")
    
    # Demo 2: Generate Live Data with Execution Costs
    print("\n📈 DEMO 2: Live Performance with Execution Costs")
    print("-" * 50)
    
    live_data = generate_live_data(backtest_data, execution_results)
    
    print(f"   Performance Comparison:")
    print(f"      Backtest Return: {backtest_data['return']:.2%}")
    print(f"      Live Return: {live_data['return']:.2%}")
    print(f"      Tracking Error: {abs(backtest_data['return'] - live_data['return']) * 10000:.1f} bps")
    print(f"      Total Fees: ${live_data['total_fees']:.2f}")
    print(f"      Avg Slippage: {live_data['avg_slippage']:.1f} bps")
    
    # Demo 3: Parity Analysis
    print("\n📊 DEMO 3: Parity Analysis")
    print("-" * 30)
    
    period_start = datetime.now() - timedelta(days=30)
    period_end = datetime.now()
    
    parity_metrics = parity_analyzer.analyze_parity(
        backtest_data=backtest_data,
        live_data=live_data,
        period_start=period_start,
        period_end=period_end,
        execution_results=execution_results
    )
    
    print(f"   Parity Analysis Results:")
    print(f"      Status: {parity_metrics.status.value.upper()}")
    print(f"      Tracking Error: {parity_metrics.tracking_error_bps:.1f} bps")
    print(f"      Correlation: {parity_metrics.correlation:.3f}")
    print(f"      Hit Rate: {parity_metrics.hit_rate:.1%}")
    print(f"      Information Ratio: {parity_metrics.information_ratio:.3f}")
    print(f"      Confidence Score: {parity_metrics.confidence_score:.1%}")
    
    print(f"\n   Component Attribution:")
    for component, value in parity_metrics.component_attribution.items():
        if value > 0.5:  # Only show significant components
            print(f"      {component}: {value:.1f} bps")
    
    # Demo 4: Multi-Day Parity Tracking
    print("\n📅 DEMO 4: Multi-Day Parity Tracking")
    print("-" * 40)
    
    print("   Simulating 7 days of parity tracking...")
    
    # Simulate degrading performance over time
    base_tracking_error = parity_metrics.tracking_error_bps
    
    for day in range(7):
        # Simulate slight degradation
        degradation_factor = 1.0 + (day * 0.1)  # 10% worse each day
        
        # Create modified data for this day
        daily_backtest = backtest_data.copy()
        daily_live = live_data.copy()
        
        # Modify live return to simulate drift
        drift = np.random.normal(0, 0.001) * degradation_factor
        daily_live['return'] = daily_live['return'] + drift
        
        daily_period_start = period_start + timedelta(days=day)
        daily_period_end = daily_period_start + timedelta(days=1)
        
        daily_metrics = parity_analyzer.analyze_parity(
            backtest_data=daily_backtest,
            live_data=daily_live,
            period_start=daily_period_start,
            period_end=daily_period_end,
            execution_results=execution_results
        )
        
        print(f"      Day {day+1}: {daily_metrics.status.value} ({daily_metrics.tracking_error_bps:.1f} bps)")
    
    # Demo 5: Drift Detection
    print("\n🚨 DEMO 5: Drift Detection")
    print("-" * 30)
    
    # Check for drift after simulation
    should_degrade, reason, actions = parity_analyzer.should_degrade_trading()
    
    print(f"   Drift Detection Results:")
    print(f"      Should Degrade Trading: {should_degrade}")
    print(f"      Reason: {reason}")
    if actions:
        print(f"      Recommended Actions:")
        for action in actions:
            print(f"         • {action}")
    
    # Check latest drift alerts
    if parity_analyzer.drift_alerts:
        latest_alert = parity_analyzer.drift_alerts[-1]
        print(f"\n   Latest Drift Alert:")
        print(f"      Drift Detected: {latest_alert.drift_detected}")
        print(f"      Type: {latest_alert.drift_type}")
        print(f"      Magnitude: {latest_alert.drift_magnitude:.2f}")
        print(f"      Confidence: {latest_alert.drift_confidence:.1%}")
        if latest_alert.affected_components:
            print(f"      Affected Components: {', '.join(latest_alert.affected_components)}")
    
    # Demo 6: Daily Report Generation
    print("\n📋 DEMO 6: Daily Report Generation")
    print("-" * 40)
    
    daily_report = parity_analyzer.generate_daily_report()
    
    print(f"   Daily Parity Report:")
    print(f"      Date: {daily_report['date']}")
    print(f"      Overall Status: {daily_report['overall_status'].upper()}")
    
    print(f"\n   Summary Metrics:")
    summary = daily_report['summary']
    print(f"      Tracking Error: {summary['tracking_error_bps']:.1f} bps")
    print(f"      Correlation: {summary['correlation']:.3f}")
    print(f"      Hit Rate: {summary['hit_rate']:.1%}")
    print(f"      Confidence: {summary['confidence_score']:.1%}")
    
    print(f"\n   Performance:")
    performance = daily_report['performance']
    print(f"      Backtest Return: {performance['backtest_return']:.2%}")
    print(f"      Live Return: {performance['live_return']:.2%}")
    print(f"      Excess Return: {performance['excess_return']:.2%}")
    print(f"      Information Ratio: {performance['information_ratio']:.3f}")
    
    print(f"\n   Trends:")
    trends = daily_report['trends']
    print(f"      Weekly Direction: {trends['weekly_direction']}")
    print(f"      Weekly Magnitude: {trends['weekly_magnitude']:.2f}")
    print(f"      Status Changes: {trends['status_changes']}")
    
    if daily_report['drift_alerts']:
        print(f"\n   Drift Alerts: {len(daily_report['drift_alerts'])} active")
        for alert in daily_report['drift_alerts']:
            print(f"      • {alert['type']}: {alert['magnitude']:.2f} magnitude ({alert['confidence']:.1%} confidence)")
    
    if daily_report['recommendations']:
        print(f"\n   Recommendations:")
        for rec in daily_report['recommendations'][:5]:  # Show top 5
            print(f"      • {rec}")
    
    # Demo 7: Advanced Simulation Scenarios
    print("\n🎮 DEMO 7: Advanced Simulation Scenarios")
    print("-" * 45)
    
    print("   Testing extreme market conditions...")
    
    # Reset simulator for clean test
    execution_simulator.reset_simulation()
    
    # Test high volatility scenario
    print("\n      High Volatility Scenario:")
    volatile_config = SimulationConfig(
        maker_fee_bps=10.0,
        taker_fee_bps=25.0,
        base_slippage_bps=25.0,  # Higher slippage
        max_slippage_bps=200.0,
        partial_fill_probability=0.35  # More partial fills
    )
    
    volatile_simulator = create_execution_simulator(volatile_config)
    
    # Simulate large market order during volatility
    volatile_result = volatile_simulator.simulate_order_execution(
        order_id="VOLATILE_TEST",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0,  # Large order
        market_data=market_data.iloc[-1:],  # Recent data
        volume_data=market_data['volume'].iloc[-24:]
    )
    
    print(f"         Large Order (5.0 BTC):")
    print(f"         Executed: {volatile_result.executed_quantity:.3f}")
    print(f"         Slippage: {volatile_result.slippage_bps:.1f} bps")
    print(f"         Market Impact: {volatile_result.market_impact_bps:.1f} bps")
    print(f"         Fees: ${volatile_result.total_fees:.2f}")
    print(f"         Partial Fill: {volatile_result.partial_fill}")
    
    # Test queue position simulation
    print("\n      Limit Order Queue Simulation:")
    for i in range(3):
        queue_result = volatile_simulator.simulate_order_execution(
            order_id=f"QUEUE_TEST_{i}",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.5,
            limit_price=48000.0,  # Below market
            market_data=market_data.iloc[-1:],
            volume_data=market_data['volume'].iloc[-24:]
        )
        
        print(f"         Order {i+1}: Queue Position {queue_result.queue_position}, "
              f"Executed: {queue_result.executed_quantity:.3f}")
    
    # Demo 8: Performance Quality Assessment
    print("\n📈 DEMO 8: Performance Quality Assessment")
    print("-" * 45)
    
    # Calculate comprehensive quality metrics
    latest_parity = parity_analyzer.parity_history[-1] if parity_analyzer.parity_history else parity_metrics
    
    print(f"   Quality Assessment:")
    print(f"      Overall Grade: {latest_parity.status.value.upper()}")
    
    # Grade components
    te_grade = "A" if latest_parity.tracking_error_bps < 10 else "B" if latest_parity.tracking_error_bps < 20 else "C"
    corr_grade = "A" if latest_parity.correlation > 0.8 else "B" if latest_parity.correlation > 0.6 else "C"
    hit_grade = "A" if latest_parity.hit_rate > 0.7 else "B" if latest_parity.hit_rate > 0.5 else "C"
    conf_grade = "A" if latest_parity.confidence_score > 0.8 else "B" if latest_parity.confidence_score > 0.6 else "C"
    
    print(f"      Tracking Error: {te_grade} ({latest_parity.tracking_error_bps:.1f} bps)")
    print(f"      Correlation: {corr_grade} ({latest_parity.correlation:.3f})")
    print(f"      Hit Rate: {hit_grade} ({latest_parity.hit_rate:.1%})")
    print(f"      Confidence: {conf_grade} ({latest_parity.confidence_score:.1%})")
    
    # Execution quality from simulation
    final_sim_stats = execution_simulator.get_simulation_statistics()
    
    avg_slippage = final_sim_stats.get('average_slippage_bps', 0.0)
    fill_rate = final_sim_stats.get('fill_rate', 0.0)
    
    slip_grade = "A" if avg_slippage < 10 else "B" if avg_slippage < 20 else "C"
    fill_grade = "A" if fill_rate > 0.9 else "B" if fill_rate > 0.8 else "C"
    
    print(f"      Slippage Control: {slip_grade} ({avg_slippage:.1f} bps avg)")
    print(f"      Fill Rate: {fill_grade} ({fill_rate:.1%})")
    
    # Overall system assessment
    grades = [te_grade, corr_grade, hit_grade, conf_grade, slip_grade, fill_grade]
    grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    avg_grade = sum(grade_values[g] for g in grades) / len(grades)
    
    if avg_grade >= 3.5:
        overall_grade = "A"
        assessment = "EXCELLENT"
    elif avg_grade >= 2.5:
        overall_grade = "B"
        assessment = "GOOD"
    elif avg_grade >= 1.5:
        overall_grade = "C"
        assessment = "ACCEPTABLE"
    else:
        overall_grade = "D"
        assessment = "NEEDS IMPROVEMENT"
    
    print(f"\n   Overall System Grade: {overall_grade} ({assessment})")
    
    print("\n✅ BACKTEST-LIVE PARITY DEMONSTRATION COMPLETED")
    print("=" * 60)
    
    # Final summary
    print(f"🎯 PARITY SYSTEM ACHIEVEMENTS:")
    total_orders = final_sim_stats.get('total_orders', 0)
    print(f"   ✅ Execution simulation with {total_orders} orders processed")
    print(f"   ✅ Parity tracking with {latest_parity.tracking_error_bps:.1f} bps tracking error")
    print(f"   ✅ Drift detection with {len(parity_analyzer.drift_alerts)} alerts generated")
    print(f"   ✅ Component attribution analysis across {len(latest_parity.component_attribution)} components")
    print(f"   ✅ Daily reporting and recommendation system operational")
    print(f"   ✅ Quality assessment: {overall_grade} grade ({assessment})")


if __name__ == "__main__":
    print("🎯 CRYPTOSMARTTRADER V2 - BACKTEST-LIVE PARITY DEMO")
    print("=" * 60)
    
    try:
        asyncio.run(demonstrate_backtest_live_parity())
        print("\n🏆 Backtest-live parity demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)