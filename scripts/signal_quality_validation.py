#!/usr/bin/env python3
"""
Signal Quality Validation Script
Automated validation of signal quality across all horizons
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

from core.signal_quality_validator import (
    get_signal_quality_validator, SignalQualityConfig, SignalHorizon, ValidationStatus
)
from core.logging_manager import get_logger

async def run_signal_quality_validation(config_file: Optional[str] = None):
    """Run comprehensive signal quality validation"""
    
    logger = get_logger()
    
    print(f"🎯 Signal Quality Validation Starting")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Load configuration
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = SignalQualityConfig(**config_data)
            print(f"📋 Loaded configuration from: {config_file}")
        else:
            config = SignalQualityConfig()
            print(f"📋 Using default configuration")
        
        print(f"⚙️  Configuration:")
        print(f"   Precision@K target: {config.minimum_precision_at_k:.1%}")
        print(f"   Hit rate target: {config.minimum_hit_rate:.1%}")
        print(f"   MAE ratio limit: {config.maximum_mae_ratio:.2f}")
        print(f"   Sharpe ratio target: {config.minimum_sharpe_ratio:.1f}")
        print(f"   Max drawdown limit: {config.maximum_drawdown:.1%}")
        print(f"   Out-of-sample period: {config.target_out_of_sample_days} days")
        print()
        
        # Generate mock historical signals for demonstration
        print(f"📊 Loading historical signals...")
        historical_signals = await generate_mock_historical_signals(config.target_out_of_sample_days)
        print(f"   Loaded {len(historical_signals)} historical signals")
        
        # Generate mock market data
        market_data = await generate_mock_market_data()
        print(f"   Loaded market data for {len(market_data)} symbols")
        print()
        
        # Run validation
        validator = get_signal_quality_validator(config)
        validation_report = await validator.validate_signal_quality(historical_signals, market_data)
        
        # Display results
        print(f"📈 SIGNAL QUALITY VALIDATION RESULTS:")
        print(f"   Validation ID: {validation_report.validation_id}")
        print(f"   Overall Status: {validation_report.overall_status.value.upper()}")
        print(f"   Out-of-sample period: {validation_report.out_of_sample_period_days} days")
        print(f"   Horizons passed: {validation_report.horizons_passed}/{validation_report.horizons_passed + validation_report.horizons_failed}")
        print()
        
        # Show summary statistics
        print(f"📊 SUMMARY METRICS:")
        print(f"   Average Precision@K: {validation_report.average_precision_at_k:.1%}")
        print(f"   Average Hit Rate: {validation_report.average_hit_rate:.1%}")
        print(f"   Average Sharpe Ratio: {validation_report.average_sharpe_ratio:.2f}")
        print(f"   Worst Max Drawdown: {validation_report.worst_max_drawdown:.1%}")
        print()
        
        # Show detailed horizon results
        print(f"🔍 DETAILED HORIZON RESULTS:")
        
        for horizon, metrics in validation_report.horizon_metrics.items():
            status_icon = "✅" if metrics.validation_status == ValidationStatus.PASSED else "❌"
            print(f"\n   {status_icon} {horizon.value} HORIZON:")
            print(f"      Status: {metrics.validation_status.value.upper()}")
            print(f"      Total signals: {metrics.total_signals}")
            print(f"      High-confidence signals: {metrics.high_confidence_signals}")
            
            if metrics.validation_status != ValidationStatus.INSUFFICIENT_DATA:
                print(f"      Precision@{metrics.top_k_used}: {metrics.precision_at_k:.1%} (target: {config.minimum_precision_at_k:.1%})")
                print(f"      Hit rate: {metrics.hit_rate:.1%} (target: {config.minimum_hit_rate:.1%})")
                print(f"      MAE ratio: {metrics.mae_ratio:.2f} (limit: {config.maximum_mae_ratio:.2f})")
                print(f"      Calibration: {metrics.calibration_success_rate:.1%} (target: {config.calibration_success_rate:.1%})")
                print(f"      Sharpe ratio: {metrics.sharpe_ratio:.2f} (target: {config.minimum_sharpe_ratio:.1f})")
                print(f"      Max drawdown: {metrics.max_drawdown:.1%} (limit: {config.maximum_drawdown:.1%})")
                
                if metrics.failed_criteria:
                    print(f"      Failed criteria: {', '.join(metrics.failed_criteria)}")
        
        print()
        
        # Show critical issues
        if validation_report.critical_issues:
            print(f"🚨 CRITICAL ISSUES:")
            for i, issue in enumerate(validation_report.critical_issues, 1):
                print(f"   {i}. {issue}")
            print()
        
        # Show recommendations
        if validation_report.recommendations:
            print(f"💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(validation_report.recommendations, 1):
                print(f"   {i}. {recommendation}")
            print()
        
        # Determine exit code
        if validation_report.overall_status == ValidationStatus.PASSED:
            print(f"✅ VALIDATION PASSED: All signal quality targets met")
            exit_code = 0
        elif validation_report.overall_status == ValidationStatus.FAILED:
            print(f"❌ VALIDATION FAILED: {validation_report.horizons_failed} horizon(s) failed")
            exit_code = 1
        else:
            print(f"⚠️  VALIDATION INCOMPLETE: Insufficient data")
            exit_code = 2
        
        print(f"\n📝 Validation report saved to data/signal_quality_reports/")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Signal quality validation failed: {e}")
        print(f"❌ VALIDATION ERROR: {e}")
        return 3

async def generate_mock_historical_signals(days: int) -> Dict[str, Any]:
    """Generate mock historical signals for demonstration"""
    
    signals = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate signals for each horizon
    horizons = [SignalHorizon.H1, SignalHorizon.H24, SignalHorizon.D7, SignalHorizon.D30]
    symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD", "LINK/USD", "UNI/USD"]
    
    signal_count = 0
    
    for horizon in horizons:
        # Determine signal frequency based on horizon
        if horizon == SignalHorizon.H1:
            signal_interval = timedelta(hours=2)  # Every 2 hours
        elif horizon == SignalHorizon.H24:
            signal_interval = timedelta(hours=8)  # Every 8 hours
        elif horizon == SignalHorizon.D7:
            signal_interval = timedelta(days=1)   # Daily
        else:  # D30
            signal_interval = timedelta(days=3)   # Every 3 days
        
        current_date = start_date
        while current_date <= end_date:
            # Generate signals for random symbols
            selected_symbols = np.random.choice(symbols, size=np.random.randint(1, 4), replace=False)
            
            for symbol in selected_symbols:
                signal_id = f"signal_{signal_count:05d}"
                
                # Generate realistic signal data
                confidence = np.random.beta(2, 2) * 0.4 + 0.6  # 0.6-1.0 range
                predicted_return = np.random.normal(0.05, 0.15)  # 5% mean, 15% std
                
                # Generate "actual" return with some correlation to prediction
                noise = np.random.normal(0, 0.1)
                actual_return = predicted_return * 0.6 + noise  # 60% correlation + noise
                
                signals[signal_id] = {
                    "timestamp": current_date.isoformat(),
                    "symbol": symbol,
                    "horizon": horizon.value,
                    "predicted_return": predicted_return,
                    "confidence": confidence,
                    "actual_return": actual_return,
                    "realized": True,
                    "metadata": {
                        "model_version": "v1.0",
                        "features_used": ["technical", "sentiment", "volume"],
                        "market_regime": np.random.choice(["bull", "bear", "sideways"])
                    }
                }
                
                signal_count += 1
            
            current_date += signal_interval
    
    return signals

async def generate_mock_market_data() -> Dict[str, Any]:
    """Generate mock market data for strategy simulation"""
    
    return {
        "BTC/USD": {"price": 45000, "volume": 1000000000},
        "ETH/USD": {"price": 3000, "volume": 500000000},
        "ADA/USD": {"price": 0.5, "volume": 100000000},
        "SOL/USD": {"price": 100, "volume": 200000000},
        "DOT/USD": {"price": 7.5, "volume": 50000000},
        "LINK/USD": {"price": 15, "volume": 75000000},
        "UNI/USD": {"price": 8, "volume": 60000000}
    }

async def run_horizon_specific_validation(horizon: str):
    """Run validation for specific horizon only"""
    
    print(f"🎯 {horizon} Horizon Signal Quality Validation")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        # Generate mock data
        historical_signals = await generate_mock_historical_signals(56)
        market_data = await generate_mock_market_data()
        
        # Filter for specific horizon
        horizon_signals = {
            k: v for k, v in historical_signals.items() 
            if v.get("horizon") == horizon
        }
        
        if not horizon_signals:
            print(f"❌ No signals found for horizon {horizon}")
            return 1
        
        print(f"📊 Found {len(horizon_signals)} signals for {horizon} horizon")
        
        # Run validation
        validator = get_signal_quality_validator()
        validation_report = await validator.validate_signal_quality(horizon_signals, market_data)
        
        # Display horizon-specific results
        horizon_enum = SignalHorizon(horizon)
        if horizon_enum in validation_report.horizon_metrics:
            metrics = validation_report.horizon_metrics[horizon_enum]
            
            print(f"\n📈 {horizon} HORIZON RESULTS:")
            print(f"   Status: {metrics.validation_status.value.upper()}")
            print(f"   Precision@{metrics.top_k_used}: {metrics.precision_at_k:.1%}")
            print(f"   Hit Rate: {metrics.hit_rate:.1%}")
            print(f"   MAE Ratio: {metrics.mae_ratio:.2f}")
            print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {metrics.max_drawdown:.1%}")
            
            if metrics.failed_criteria:
                print(f"\n❌ Failed Criteria: {', '.join(metrics.failed_criteria)}")
                
            if metrics.passed_criteria:
                print(f"✅ Passed Criteria: {', '.join(metrics.passed_criteria)}")
        
        return 0 if validation_report.overall_status == ValidationStatus.PASSED else 1
        
    except Exception as e:
        print(f"❌ Horizon validation failed: {e}")
        return 1

def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Full signal quality validation:")
    print("  python scripts/signal_quality_validation.py validate")
    print()
    print("Validation with custom config:")
    print("  python scripts/signal_quality_validation.py validate --config config/signal_quality.json")
    print()
    print("Specific horizon validation:")
    print("  python scripts/signal_quality_validation.py horizon --horizon 1H")
    print("  python scripts/signal_quality_validation.py horizon --horizon 24H")
    print("  python scripts/signal_quality_validation.py horizon --horizon 7D")
    print("  python scripts/signal_quality_validation.py horizon --horizon 30D")
    print()
    print("Automated scheduling (cron):")
    print("  # Weekly signal quality validation")
    print("  0 2 * * 0 /usr/bin/python3 /path/to/scripts/signal_quality_validation.py validate")
    print("  # Daily horizon-specific checks")
    print("  0 6 * * * /usr/bin/python3 /path/to/scripts/signal_quality_validation.py horizon --horizon 24H")
    print()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Signal Quality Validation - Multi-horizon performance validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/signal_quality_validation.py validate               # Full validation
  python scripts/signal_quality_validation.py validate --config custom.json  # Custom config
  python scripts/signal_quality_validation.py horizon --horizon 1H   # Specific horizon
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run full signal quality validation')
    validate_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    # Horizon command
    horizon_parser = subparsers.add_parser('horizon', help='Validate specific horizon')
    horizon_parser.add_argument(
        '--horizon',
        choices=['1H', '24H', '7D', '30D'],
        required=True,
        help='Horizon to validate'
    )
    
    # Examples command
    examples_parser = subparsers.add_parser('examples', help='Show usage examples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    if args.command == 'validate':
        exit_code = await run_signal_quality_validation(args.config)
    elif args.command == 'horizon':
        exit_code = await run_horizon_specific_validation(args.horizon)
    elif args.command == 'examples':
        print_usage_examples()
        exit_code = 0
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Signal quality validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)