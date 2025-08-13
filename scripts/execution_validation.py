#!/usr/bin/env python3
"""
Execution Validation Script
Automated validation of execution performance with slippage, fill rates, and latency
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.execution_simulator import (
    get_execution_simulator, ExecutionConfig, OrderType, ExecutionQuality
)
from core.logging_manager import get_logger

async def run_execution_validation(
    num_orders: int = 1000,
    config_file: Optional[str] = None
):
    """Run comprehensive execution validation"""
    
    logger = get_logger()
    
    print(f"⚡ Execution Validation Starting")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Orders to simulate: {num_orders}")
    print("=" * 70)
    
    try:
        # Load configuration
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = ExecutionConfig(**config_data)
            print(f"📋 Loaded configuration from: {config_file}")
        else:
            config = ExecutionConfig()
            print(f"📋 Using default configuration")
        
        print(f"🎯 Execution Targets:")
        print(f"   Slippage p50: ≤{config.target_slippage_p50:.0f} bps")
        print(f"   Slippage p90: ≤{config.target_slippage_p90:.0f} bps")
        print(f"   Fill rate: ≥{config.target_fill_rate:.1%}")
        print(f"   Latency p95: ≤{config.target_latency_p95:.1f}s")
        print()
        
        # Initialize execution simulator
        simulator = get_execution_simulator(config)
        
        # Generate diverse test orders
        print(f"🔄 Generating {num_orders} test orders...")
        test_orders = generate_test_orders(num_orders)
        print(f"   Generated orders across {len(set(o['symbol'] for o in test_orders))} symbols")
        print(f"   Order types: {len([o for o in test_orders if o['order_type'] == OrderType.MARKET])} market, {len([o for o in test_orders if o['order_type'] == OrderType.LIMIT])} limit")
        print()
        
        # Execute orders
        print(f"⚡ Executing orders...")
        execution_results = []
        
        for i, order in enumerate(test_orders):
            if i % 100 == 0:
                print(f"   Progress: {i}/{num_orders} ({i/num_orders:.1%})")
            
            try:
                result = await simulator.# REMOVED: Mock data pattern not allowed in production
                    symbol=order["symbol"],
                    side=order["side"],
                    quantity=order["quantity"],
                    order_type=order["order_type"],
                    signal_timestamp=order["signal_timestamp"],
                    volatility=order["volatility"]
                )
                
                execution_results.append(result)
                
            except Exception as e:
                logger.warning(f"Order {i} execution failed: {e}")
        
        print(f"   Completed: {len(execution_results)}/{num_orders} orders")
        print()
        
        # Validate execution performance
        print(f"📈 Validating execution performance...")
        validation_report = await simulator.validate_execution_performance(execution_results)
        
        # Display results
        print(f"📊 EXECUTION VALIDATION RESULTS:")
        print(f"   Validation ID: {validation_report.validation_id}")
        print(f"   Total Orders: {validation_report.total_orders}")
        print(f"   Execution Quality: {validation_report.execution_quality.value.upper()}")
        print(f"   Quality Score: {validation_report.quality_score:.2f}/1.00")
        print()
        
        # Slippage metrics
        print(f"💹 SLIPPAGE METRICS:")
        status_p50 = "✅" if validation_report.slippage_p50 <= config.target_slippage_p50 else "❌"
        status_p90 = "✅" if validation_report.slippage_p90 <= config.target_slippage_p90 else "❌"
        
        print(f"   {status_p50} p50: {validation_report.slippage_p50:.1f} bps (target: ≤{config.target_slippage_p50:.0f} bps)")
        print(f"   {status_p90} p90: {validation_report.slippage_p90:.1f} bps (target: ≤{config.target_slippage_p90:.0f} bps)")
        print(f"   📊 p95: {validation_report.slippage_p95:.1f} bps")
        print(f"   📊 Mean: {validation_report.slippage_mean:.1f} ± {validation_report.slippage_std:.1f} bps")
        print()
        
        # Fill rate metrics
        print(f"📋 FILL RATE METRICS:")
        status_fill = "✅" if validation_report.overall_fill_rate >= config.target_fill_rate else "❌"
        
        print(f"   {status_fill} Overall: {validation_report.overall_fill_rate:.1%} (target: ≥{config.target_fill_rate:.1%})")
        print(f"   📊 Market orders: {validation_report.market_order_fill_rate:.1%}")
        print(f"   📊 Limit orders: {validation_report.limit_order_fill_rate:.1%}")
        print()
        
        # Latency metrics
        print(f"⏱️  LATENCY METRICS:")
        status_latency = "✅" if validation_report.latency_p95_ms <= (config.target_latency_p95 * 1000) else "❌"
        
        print(f"   📊 p50: {validation_report.latency_p50_ms:.0f}ms")
        print(f"   📊 p90: {validation_report.latency_p90_ms:.0f}ms")
        print(f"   {status_latency} p95: {validation_report.latency_p95_ms:.0f}ms (target: ≤{config.target_latency_p95*1000:.0f}ms)")
        print(f"   📊 Mean: {validation_report.latency_mean_ms:.0f}ms")
        print()
        
        # Performance by order size
        print(f"📏 PERFORMANCE BY ORDER SIZE:")
        
        small = validation_report.small_order_performance
        if small.get("count", 0) > 0:
            print(f"   Small orders (≤1K): {small['count']} orders")
            print(f"     Avg slippage: {small['avg_slippage_bps']:.1f} bps")
            print(f"     Avg fill rate: {small['avg_fill_rate']:.1%}")
            print(f"     Avg latency: {small['avg_latency_ms']:.0f}ms")
        
        medium = validation_report.medium_order_performance
        if medium.get("count", 0) > 0:
            print(f"   Medium orders (1K-10K): {medium['count']} orders")
            print(f"     Avg slippage: {medium['avg_slippage_bps']:.1f} bps")
            print(f"     Avg fill rate: {medium['avg_fill_rate']:.1%}")
            print(f"     Avg latency: {medium['avg_latency_ms']:.0f}ms")
        
        large = validation_report.large_order_performance
        if large.get("count", 0) > 0:
            print(f"   Large orders (>10K): {large['count']} orders")
            print(f"     Avg slippage: {large['avg_slippage_bps']:.1f} bps")
            print(f"     Avg fill rate: {large['avg_fill_rate']:.1%}")
            print(f"     Avg latency: {large['avg_latency_ms']:.0f}ms")
        
        print()
        
        # Market impact analysis
        if not np.isnan(validation_report.impact_correlation):
            print(f"📊 MARKET IMPACT ANALYSIS:")
            print(f"   Size-impact correlation: {validation_report.impact_correlation:.3f}")
            print(f"   Impact decay rate: {validation_report.impact_decay_rate:.3f}")
            print()
        
        # Validation status
        if validation_report.validation_passed:
            print(f"✅ VALIDATION PASSED: All execution targets met")
        else:
            print(f"❌ VALIDATION FAILED: {len(validation_report.failed_criteria)} criteria failed")
            print(f"   Failed: {', '.join(validation_report.failed_criteria)}")
            if validation_report.passed_criteria:
                print(f"   Passed: {', '.join(validation_report.passed_criteria)}")
        
        print()
        
        # Recommendations
        if validation_report.recommendations:
            print(f"💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(validation_report.recommendations, 1):
                print(f"   {i}. {recommendation}")
            print()
        
        # Save detailed report
        await save_execution_report(validation_report, execution_results)
        print(f"📝 Detailed execution report saved to data/execution_reports/")
        
        return 0 if validation_report.validation_passed else 1
        
    except Exception as e:
        logger.error(f"Execution validation failed: {e}")
        print(f"❌ VALIDATION ERROR: {e}")
        return 2

def generate_test_orders(num_orders: int) -> List[Dict[str, Any]]:
    """Generate diverse test orders for validation"""
    
    symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD", "LINK/USD", "UNI/USD"]
    order_types = [OrderType.MARKET, OrderType.LIMIT]
    sides = ["buy", "sell"]
    
    orders = []
    
    for i in range(num_orders):
        # Generate order parameters
        symbol = np.# REMOVED: Mock data pattern not allowed in production(symbols)
        side = np.# REMOVED: Mock data pattern not allowed in production(sides)
        order_type = np.# REMOVED: Mock data pattern not allowed in production(order_types, p=[0.7, 0.3])  # 70% market orders
        
        # Generate quantity with realistic distribution
        # Most orders small, some medium, few large
        size_category = np.# REMOVED: Mock data pattern not allowed in production(['small', 'medium', 'large'], p=[0.7, 0.25, 0.05])
        
        if size_category == 'small':
            quantity = np.random.exponential(100) + 10  # 10-500 range
        elif size_category == 'medium':
            quantity = np.random.exponential(2000) + 500  # 500-5000 range
        else:
            quantity = np.random.exponential(10000) + 5000  # 5000+ range
        
        quantity = min(quantity, 50000)  # Cap at 50K
        
        # Generate volatility
        volatility = np.random.gamma(2, 0.01)  # ~2% average volatility
        
        # Generate signal timestamp (within last hour)
        signal_timestamp = datetime.now() - timedelta(
            seconds=np.random.exponential(1800)  # Average 30 minutes ago
        )
        
        orders.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "signal_timestamp": signal_timestamp,
            "volatility": volatility
        })
    
    return orders

async def save_execution_report(validation_report, execution_results):
    """Save detailed execution report"""
    
    # Create directory
    reports_dir = Path("data/execution_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare report data
    report_data = {
        "validation_report": {
            "validation_id": validation_report.validation_id,
            "timestamp": validation_report.timestamp.isoformat(),
            "total_orders": validation_report.total_orders,
            "execution_quality": validation_report.execution_quality.value,
            "quality_score": validation_report.quality_score,
            "slippage_metrics": {
                "p50": validation_report.slippage_p50,
                "p90": validation_report.slippage_p90,
                "p95": validation_report.slippage_p95,
                "mean": validation_report.slippage_mean,
                "std": validation_report.slippage_std
            },
            "fill_metrics": {
                "overall_fill_rate": validation_report.overall_fill_rate,
                "market_order_fill_rate": validation_report.market_order_fill_rate,
                "limit_order_fill_rate": validation_report.limit_order_fill_rate
            },
            "latency_metrics": {
                "p50_ms": validation_report.latency_p50_ms,
                "p90_ms": validation_report.latency_p90_ms,
                "p95_ms": validation_report.latency_p95_ms,
                "mean_ms": validation_report.latency_mean_ms
            },
            "size_performance": {
                "small_orders": validation_report.small_order_performance,
                "medium_orders": validation_report.medium_order_performance,
                "large_orders": validation_report.large_order_performance
            },
            "validation_passed": validation_report.validation_passed,
            "failed_criteria": validation_report.failed_criteria,
            "passed_criteria": validation_report.passed_criteria,
            "recommendations": validation_report.recommendations
        },
        "execution_summary": {
            "total_executions": len(execution_results),
            "successful_executions": len([r for r in execution_results if r.fill_rate > 0]),
            "average_slippage": np.mean([r.slippage_bps for r in execution_results if not np.isinf(r.slippage_bps)]),
            "average_fill_rate": np.mean([r.fill_rate for r in execution_results])
        }
    }
    
    # Write main report
    report_file = reports_dir / f"execution_report_{validation_report.validation_id}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Write latest
    latest_file = reports_dir / "latest_execution_report.json"
    with open(latest_file, 'w') as f:
        json.dump(report_data, f, indent=2)

async def run_quick_execution_test(orders: int = 100):
    """Run quick execution test"""
    
    print(f"⚡ Quick Execution Test - {orders} orders")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 40)
    
    try:
        simulator = get_execution_simulator()
        
        # Generate small test set
        test_orders = generate_test_orders(orders)
        
        # Execute orders
        results = []
        for order in test_orders:
            result = await simulator.# REMOVED: Mock data pattern not allowed in production
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
                order_type=order["order_type"]
            )
            results.append(result)
        
        # Quick metrics
        valid_results = [r for r in results if not np.isinf(r.slippage_bps)]
        
        if valid_results:
            avg_slippage = np.mean([r.slippage_bps for r in valid_results])
            p90_slippage = np.percentile([r.slippage_bps for r in valid_results], 90)
            avg_fill_rate = np.mean([r.fill_rate for r in results])
            avg_latency = np.mean([r.end_to_end_latency_ms for r in valid_results if not np.isinf(r.end_to_end_latency_ms)])
            
            print(f"📊 Results:")
            print(f"   Orders executed: {len(results)}")
            print(f"   Avg slippage: {avg_slippage:.1f} bps")
            print(f"   p90 slippage: {p90_slippage:.1f} bps")
            print(f"   Avg fill rate: {avg_fill_rate:.1%}")
            print(f"   Avg latency: {avg_latency:.0f}ms")
        
        return 0
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return 1

def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Full execution validation:")
    print("  python scripts/execution_validation.py validate")
    print()
    print("Custom number of orders:")
    print("  python scripts/execution_validation.py validate --orders 2000")
    print()
    print("Custom configuration:")
    print("  python scripts/execution_validation.py validate --config config/execution.json")
    print()
    print("Quick execution test:")
    print("  python scripts/execution_validation.py quick")
    print("  python scripts/execution_validation.py quick --orders 50")
    print()
    print("Automated scheduling (cron):")
    print("  # Daily execution validation")
    print("  0 3 * * * /usr/bin/python3 /path/to/scripts/execution_validation.py validate")
    print("  # Hourly quick tests")
    print("  0 * * * * /usr/bin/python3 /path/to/scripts/execution_validation.py quick --orders 50")
    print()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Execution Validation - Slippage, fill rates, and latency testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/execution_validation.py validate               # Full validation
  python scripts/execution_validation.py validate --orders 2000    # Custom order count
  python scripts/execution_validation.py quick                 # Quick test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run full execution validation')
    validate_parser.add_argument(
        '--orders',
        type=int,
        default=1000,
        help='Number of orders to simulate (default: 1000)'
    )
    validate_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Run quick execution test')
    quick_parser.add_argument(
        '--orders',
        type=int,
        default=100,
        help='Number of orders to simulate (default: 100)'
    )
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show usage examples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    if args.command == 'validate':
        exit_code = await run_execution_validation(args.orders, args.config)
    elif args.command == 'quick':
        exit_code = await run_quick_execution_test(args.orders)
    elif args.command == 'examples':
        print_usage_examples()
        exit_code = 0
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Execution validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)