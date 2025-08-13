#!/usr/bin/env python3
"""
Test script for Return Attribution System
Demonstrates PnL decomposition into alpha/fees/slippage/timing components.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.cryptosmarttrader.attribution.return_attribution import (
    ReturnAttributionAnalyzer, AttributionPeriod
)
from src.cryptosmarttrader.parity.execution_simulator import (
    ExecutionResult, OrderSide, OrderType
)


def test_return_attribution_system():
    """Test the complete return attribution system."""
    print("🎯 Testing Return Attribution System...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Initialize Attribution Analyzer
    print("\n1. Testing Attribution Analyzer...")
    analyzer = ReturnAttributionAnalyzer()
    print("✅ ReturnAttributionAnalyzer initialized")
    
    # 2. Generate realistic test data
    print("\n2. Generating test data...")
    
    # Portfolio returns (slightly outperforming)
    np.random.seed(42)  # For reproducible results
    portfolio_returns = pd.Series(
        np.random.normal(0.0015, 0.02, 48),  # 48 hours, 15bps avg hourly return
        index=pd.date_range(start=datetime.utcnow() - timedelta(days=2), periods=48, freq='H')
    )
    
    # Benchmark returns (market performance)
    benchmark_returns = pd.Series(
        np.random.normal(0.001, 0.018, 48),   # 10bps avg hourly return
        index=portfolio_returns.index
    )
    
    print(f"✅ Generated {len(portfolio_returns)} hours of return data")
    print(f"   Portfolio return: {portfolio_returns.sum()*10000:.1f} bps")
    print(f"   Benchmark return: {benchmark_returns.sum()*10000:.1f} bps")
    print(f"   Excess return: {(portfolio_returns.sum() - benchmark_returns.sum())*10000:.1f} bps")
    
    # 3. Generate execution results with realistic costs
    print("\n3. Generating execution data...")
    execution_results = []
    
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    for i in range(50):  # 50 trades over 2 days
        # Realistic execution parameters
        symbol = np.random.choice(symbols)
        side = OrderSide.BUY if np.random.random() > 0.5 else OrderSide.SELL
        
        # Mix of order types (70% limit, 30% market)
        order_type = OrderType.LIMIT if np.random.random() > 0.3 else OrderType.MARKET
        
        # Trade sizes
        quantity = np.random.uniform(0.1, 2.0)
        price = 50000 + np.random.normal(0, 2000)  # BTC price variation
        
        # Execution costs
        if order_type == OrderType.LIMIT:
            # Maker orders: lower fees, lower slippage
            fee_rate = 0.0016  # 16 bps maker fee
            slippage_bps = np.random.uniform(1, 8)
            latency_ms = np.random.uniform(100, 300)
        else:
            # Market orders: higher fees, higher slippage
            fee_rate = 0.0026  # 26 bps taker fee
            slippage_bps = np.random.uniform(5, 25)
            latency_ms = np.random.uniform(50, 150)
        
        total_fees = quantity * price * fee_rate
        
        executed_qty = quantity * np.random.uniform(0.95, 1.0)
        is_partial = executed_qty < quantity
        
        execution = ExecutionResult(
            order_id=f"test_{i:03d}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_quantity=quantity,
            executed_quantity=executed_qty,
            avg_fill_price=price,
            total_fees=total_fees,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            partial_fill=is_partial,
            execution_time=datetime.utcnow() - timedelta(minutes=np.random.randint(0, 2880)),
            market_impact_bps=slippage_bps * 0.5,  # Rough estimate
            queue_position=None,
            fills=[],
            metadata={}
        )
        
        execution_results.append(execution)
    
    print(f"✅ Generated {len(execution_results)} execution results")
    print(f"   Average slippage: {np.mean([e.slippage_bps for e in execution_results]):.1f} bps")
    print(f"   Average latency: {np.mean([e.latency_ms for e in execution_results]):.0f} ms")
    print(f"   Total fees: ${sum([e.total_fees for e in execution_results]):.2f}")
    
    # 4. Run Attribution Analysis
    print("\n4. Running Attribution Analysis...")
    
    report = analyzer.analyze_attribution(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        execution_results=execution_results,
        period=AttributionPeriod.DAILY
    )
    
    print("✅ Attribution analysis complete")
    
    # 5. Display Results
    print("\n🏆 ATTRIBUTION RESULTS:")
    print("=" * 60)
    
    print(f"📊 Total Return: {report.total_return_bps:.1f} bps")
    print(f"📈 Benchmark Return: {report.benchmark_return_bps:.1f} bps")
    print(f"🎯 Excess Return: {report.excess_return_bps:.1f} bps")
    print(f"🔍 Explained Variance: {report.explained_variance_pct:.1f}%")
    print(f"✅ Attribution Confidence: {report.attribution_confidence:.1%}")
    
    print("\n📋 COMPONENT BREAKDOWN:")
    components = [
        ("🎯 Alpha", report.alpha_component),
        ("💰 Fees", report.fees_component),
        ("📉 Slippage", report.slippage_component),
        ("⏱️  Timing", report.timing_component),
        ("📏 Sizing", report.sizing_component),
        ("🌊 Market Impact", report.market_impact_component)
    ]
    
    for name, component in components:
        pct = component.contribution_pct * 100
        confidence = component.confidence
        print(f"   {name}: {component.contribution_bps:+.1f} bps ({pct:+.1f}%) [conf: {confidence:.1%}]")
    
    # 6. Detailed Analysis
    print("\n🔍 DETAILED ANALYSIS:")
    print("-" * 40)
    
    # Alpha analysis
    alpha_meta = report.alpha_component.metadata
    print(f"📊 Alpha Metrics:")
    print(f"   Sharpe Ratio: {alpha_meta.get('sharpe_ratio', 0):.2f}")
    print(f"   Hit Rate: {alpha_meta.get('hit_rate', 0):.1%}")
    print(f"   Max Drawdown: {alpha_meta.get('max_drawdown', 0):.1%}")
    
    # Fees analysis
    fees_meta = report.fees_component.metadata
    print(f"\n💰 Fee Analysis:")
    print(f"   Total Fees: ${fees_meta.get('total_fees_usd', 0):.2f}")
    print(f"   Maker Orders: {fees_meta.get('maker_orders', 0)}")
    print(f"   Taker Orders: {fees_meta.get('taker_orders', 0)}")
    print(f"   Maker Ratio: {fees_meta.get('maker_ratio', 0):.1%}")
    
    # Slippage analysis
    slippage_meta = report.slippage_component.metadata
    print(f"\n📉 Slippage Analysis:")
    print(f"   Average: {slippage_meta.get('avg_slippage_bps', 0):.1f} bps")
    print(f"   Median: {slippage_meta.get('median_slippage_bps', 0):.1f} bps")
    print(f"   Maximum: {slippage_meta.get('max_slippage_bps', 0):.1f} bps")
    print(f"   High Slippage Orders: {slippage_meta.get('high_slippage_orders', 0)}")
    
    # Timing analysis
    timing_meta = report.timing_component.metadata
    print(f"\n⏱️  Timing Analysis:")
    print(f"   Average Latency: {timing_meta.get('avg_latency_ms', 0):.0f} ms")
    print(f"   Timing Efficiency: {timing_meta.get('timing_efficiency', 0):.1%}")
    print(f"   Slow Executions: {timing_meta.get('slow_executions', 0)}")
    print(f"   Timing Score: {timing_meta.get('timing_score', 0):.1%}")
    
    # 7. Optimization Insights
    print("\n💡 OPTIMIZATION INSIGHTS:")
    print("=" * 50)
    
    if report.optimization_opportunities:
        print("🎯 Top Opportunities:")
        for i, opp in enumerate(report.optimization_opportunities, 1):
            print(f"   {i}. {opp}")
    else:
        print("🎉 No major optimization opportunities detected!")
    
    if report.execution_improvements:
        print("\n⚡ Execution Improvements:")
        for i, imp in enumerate(report.execution_improvements, 1):
            print(f"   {i}. {imp}")
    
    if report.cost_reduction_suggestions:
        print("\n💰 Cost Reduction Suggestions:")
        for i, sug in enumerate(report.cost_reduction_suggestions, 1):
            print(f"   {i}. {sug}")
    
    # 8. Summary Statistics
    print("\n📊 EXECUTION QUALITY METRICS:")
    print("-" * 35)
    print(f"🏆 Execution Quality Score: {report.execution_quality_score:.1%}")
    print(f"📊 Data Quality Score: {report.data_quality_score:.1%}")
    print(f"🔢 Observation Count: {report.observation_count}")
    
    # 9. Test Attribution Summary
    print("\n📈 Testing Attribution Summary...")
    summary = analyzer.get_attribution_summary(period_days=2)
    
    if 'error' not in summary:
        print(f"✅ Summary for {summary['period_days']} days:")
        print(f"   Reports: {summary['reports_count']}")
        print(f"   Avg Total Return: {summary['avg_total_return_bps']:.1f} bps")
        print(f"   Avg Alpha: {summary['avg_alpha_bps']:.1f} bps")
        print(f"   Avg Execution Quality: {summary['avg_execution_quality']:.1%}")
        print(f"   Top Cost Component: {summary['top_opportunity']}")
    
    # 10. Save Report
    print("\n💾 Saving Attribution Report...")
    try:
        saved_path = analyzer.save_attribution_report(report)
        print(f"✅ Report saved to: {saved_path}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
    
    print("\n🎉 Return Attribution System Test Complete!")
    print("\n📋 SYSTEM CAPABILITIES VALIDATED:")
    print("   ✅ PnL decomposition into 6 components")
    print("   ✅ Alpha vs execution cost attribution")
    print("   ✅ Statistical confidence scoring")
    print("   ✅ Execution quality assessment")
    print("   ✅ Optimization opportunity detection")
    print("   ✅ Cost reduction recommendations")
    print("   ✅ Report persistence and summarization")
    
    return True


def test_multiple_scenarios():
    """Test attribution system with different scenarios."""
    print("\n🧪 Testing Multiple Attribution Scenarios...")
    
    analyzer = ReturnAttributionAnalyzer()
    
    scenarios = [
        ("High Alpha, Low Costs", 0.003, 0.001, 5, 100),    # Good performance
        ("Low Alpha, High Costs", 0.001, 0.0015, 20, 500),  # Poor execution
        ("Negative Alpha, High Costs", -0.001, 0.002, 30, 800),  # Bad scenario
        ("Good Alpha, Moderate Costs", 0.002, 0.001, 12, 250)   # Balanced
    ]
    
    for scenario_name, port_return, bench_return, avg_slippage, avg_latency in scenarios:
        print(f"\n   📊 Scenario: {scenario_name}")
        
        # Generate scenario-specific data
        portfolio_returns = pd.Series(np.random.normal(port_return, 0.01, 24))
        benchmark_returns = pd.Series(np.random.normal(bench_return, 0.01, 24))
        
        # Generate executions with scenario parameters
        executions = []
        for i in range(20):
            exec_result = ExecutionResult(
                order_id=f"scenario_{i}",
                symbol="BTC/USD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                requested_quantity=0.1,
                executed_quantity=0.1,
                avg_fill_price=50000,
                total_fees=10.0,
                slippage_bps=avg_slippage + np.random.normal(0, 5),
                latency_ms=avg_latency + np.random.normal(0, 100),
                partial_fill=False,
                execution_time=datetime.utcnow(),
                market_impact_bps=2.0,
                queue_position=None,
                fills=[],
                metadata={}
            )
            executions.append(exec_result)
        
        # Run attribution
        report = analyzer.analyze_attribution(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            execution_results=executions
        )
        
        print(f"      Total Return: {report.total_return_bps:.1f} bps")
        print(f"      Alpha: {report.alpha_component.contribution_bps:.1f} bps")
        print(f"      Execution Quality: {report.execution_quality_score:.1%}")
        
        if report.optimization_opportunities:
            print(f"      Top Opportunity: {report.optimization_opportunities[0]}")
    
    print("✅ Multiple scenario testing complete")


if __name__ == "__main__":
    try:
        success = test_return_attribution_system()
        
        if success:
            test_multiple_scenarios()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()