#!/usr/bin/env python3
"""
Shadow Trading Monitor - Real-time monitoring and validation
Automated script for monitoring shadow trading performance and soak period progress
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shadow_trading_engine import (
    get_shadow_trading_engine,
    SoakPeriodConfig,
    TradingMode,
    SoakStatus,
)
from core.logging_manager import get_logger


async def run_shadow_trading_monitor(interval_minutes: int = 60):
    """Run continuous shadow trading monitoring"""

    logger = get_logger()

    print(f"🔍 Shadow Trading Monitor Started")
    print(f"⏰ Monitoring Interval: {interval_minutes} minutes")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    shadow_engine = get_shadow_trading_engine()

    while True:
        try:
            print(f"\n🔄 Monitor Check: {datetime.now().strftime('%H:%M:%S')}")

            # Get current status
            status = shadow_engine.get_soak_status_summary()

            # Display key metrics
            print(f"📊 SOAK PERIOD STATUS:")
            print(f"   Mode: {status['current_mode'].upper()}")
            print(f"   Status: {status['soak_status'].upper()}")
            print(
                f"   Duration: {status['soak_period']['current_duration_days']}/{status['soak_period']['target_duration_days']} days"
            )
            print(f"   Progress: {status['soak_period']['progress_percentage']:.1f}%")
            print(
                f"   Trades: {status['trading_statistics']['filled_trades']}/{status['trading_statistics']['minimum_required_trades']}"
            )
            print(f"   Portfolio Value: ${status['portfolio_status']['portfolio_value']:,.2f}")

            # Show market regimes
            regimes = status["market_exposure"]["regimes_encountered"]
            if regimes:
                print(
                    f"   Market Regimes: {', '.join(regimes)} ({len(regimes)}/{status['market_exposure']['regimes_required']})"
                )

            # Check if ready for validation
            if (
                status["soak_period"]["current_duration_days"]
                >= shadow_engine.config.minimum_duration_days
                and status["trading_statistics"]["filled_trades"]
                >= shadow_engine.config.minimum_trades
            ):
                print(f"\n🎯 READY FOR VALIDATION!")

                # Run validation
                validation_result = await shadow_engine.run_soak_period_validation()

                print(f"📈 VALIDATION RESULTS:")
                print(f"   Status: {validation_result.status.value.upper()}")
                print(f"   Win Rate: {validation_result.win_rate:.1%}")
                print(f"   False Positive Ratio: {validation_result.false_positive_ratio:.1%}")
                print(f"   Sharpe Ratio: {validation_result.sharpe_ratio:.2f}")
                print(f"   Max Drawdown: {validation_result.max_drawdown:.1%}")
                print(f"   Total Return: {validation_result.total_return_percent:.1f}%")

                if validation_result.status == SoakStatus.PASSED:
                    print(f"✅ LIVE TRADING AUTHORIZED!")
                    print(
                        f"   Authorization Date: {validation_result.end_date.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    # Save authorization certificate
                    await save_authorization_certificate(validation_result)

                    break  # Exit monitoring

                elif validation_result.status == SoakStatus.FAILED:
                    print(f"❌ SOAK PERIOD FAILED")
                    print(f"   Failed Criteria: {', '.join(validation_result.failed_criteria)}")
                    print(f"   Recommendation: Restart soak period with improved strategy")

                    # Log failure details
                    await log_validation_failure(validation_result)

            else:
                # Show remaining requirements
                days_remaining = max(
                    0,
                    shadow_engine.config.minimum_duration_days
                    - status["soak_period"]["current_duration_days"],
                )
                trades_remaining = max(
                    0,
                    shadow_engine.config.minimum_trades
                    - status["trading_statistics"]["filled_trades"],
                )

                print(f"\n⏳ REQUIREMENTS REMAINING:")
                if days_remaining > 0:
                    print(f"   Days: {days_remaining}")
                if trades_remaining > 0:
                    print(f"   Trades: {trades_remaining}")
                if status["market_exposure"]["regimes_remaining"] > 0:
                    print(f"   Market Regimes: {status['market_exposure']['regimes_remaining']}")

            # Update unrealized P&L
            await shadow_engine.update_unrealized_pnl()

            # Sleep for next check
            print(f"😴 Next check in {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print(f"\n⏹️  Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            print(f"❌ Monitor error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry


async def save_authorization_certificate(validation_result):
    """Save live trading authorization certificate"""

    certificate = {
        "authorization_type": "LIVE_TRADING_AUTHORIZATION",
        "validation_id": validation_result.validation_id,
        "authorization_date": validation_result.end_date.isoformat(),
        "soak_period": {
            "start_date": validation_result.start_date.isoformat(),
            "end_date": validation_result.end_date.isoformat(),
            "duration_days": validation_result.duration_days,
        },
        "performance_metrics": {
            "total_trades": validation_result.total_trades,
            "win_rate": validation_result.win_rate,
            "false_positive_ratio": validation_result.false_positive_ratio,
            "sharpe_ratio": validation_result.sharpe_ratio,
            "max_drawdown": validation_result.max_drawdown,
            "total_return_percent": validation_result.total_return_percent,
            "market_regimes_traded": validation_result.market_regimes_traded,
        },
        "validation_criteria": {
            "passed_criteria": validation_result.passed_criteria,
            "failed_criteria": validation_result.failed_criteria,
        },
        "certificate_hash": f"cert_{validation_result.validation_id}_{hash(str(validation_result.__dict__))}",
    }

    # Save certificate
    cert_dir = Path("certificates")
    cert_dir.mkdir(exist_ok=True)

    cert_file = cert_dir / f"live_trading_authorization_{validation_result.validation_id}.json"

    with open(cert_file, "w") as f:
        json.dump(certificate, f, indent=2)

    print(f"📜 Authorization certificate saved: {cert_file}")


async def log_validation_failure(validation_result):
    """Log validation failure details for analysis"""

    failure_log = {
        "failure_type": "SOAK_PERIOD_VALIDATION_FAILURE",
        "validation_id": validation_result.validation_id,
        "failure_date": validation_result.end_date.isoformat(),
        "soak_period": {
            "duration_days": validation_result.duration_days,
            "total_trades": validation_result.total_trades,
        },
        "failed_criteria": validation_result.failed_criteria,
        "performance_metrics": {
            "win_rate": validation_result.win_rate,
            "false_positive_ratio": validation_result.false_positive_ratio,
            "sharpe_ratio": validation_result.sharpe_ratio,
            "max_drawdown": validation_result.max_drawdown,
        },
        "recommendations": [
            "Review strategy parameters and thresholds",
            "Analyze failed criteria for improvement opportunities",
            "Consider extending shadow trading period",
            "Implement risk management improvements",
        ],
    }

    # Save failure log
    log_dir = Path("logs/validation_failures")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"validation_failure_{validation_result.validation_id}.json"

    with open(log_file, "w") as f:
        json.dump(failure_log, f, indent=2)

    print(f"📝 Failure log saved: {log_file}")


async def run_daily_soak_report():
    """Generate daily soak period progress report"""

    print(f"📊 DAILY SOAK PERIOD REPORT")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 50)

    shadow_engine = get_shadow_trading_engine()
    status = shadow_engine.get_soak_status_summary()

    # Overall progress
    print(f"🎯 PROGRESS OVERVIEW:")
    print(f"   Current Mode: {status['current_mode']}")
    print(f"   Soak Status: {status['soak_status']}")
    print(
        f"   Live Trading Authorized: {'✅ YES' if status['live_trading_authorized'] else '❌ NO'}"
    )
    print()

    # Time progress
    print(f"⏰ TIME PROGRESS:")
    soak = status["soak_period"]
    print(f"   Start Date: {soak['start_date'][:10]}")
    print(f"   Duration: {soak['current_duration_days']} days")
    print(f"   Target: {soak['target_duration_days']} days")
    print(f"   Progress: {soak['progress_percentage']:.1f}%")
    if soak["days_remaining"] > 0:
        print(f"   Remaining: {soak['days_remaining']} days")
    print()

    # Trading progress
    print(f"📈 TRADING PROGRESS:")
    trading = status["trading_statistics"]
    print(f"   Total Trades: {trading['total_trades']}")
    print(f"   Filled Trades: {trading['filled_trades']}")
    print(f"   Required: {trading['minimum_required_trades']}")
    if trading["trades_remaining"] > 0:
        print(f"   Remaining: {trading['trades_remaining']}")
    print()

    # Portfolio status
    print(f"💰 PORTFOLIO STATUS:")
    portfolio = status["portfolio_status"]
    print(f"   Initial Capital: $100,000.00")
    print(f"   Current Cash: ${portfolio['current_cash']:,.2f}")
    print(f"   Portfolio Value: ${portfolio['portfolio_value']:,.2f}")
    print(f"   Open Positions: {portfolio['positions_count']}")
    print()

    # Market exposure
    print(f"🌍 MARKET EXPOSURE:")
    market = status["market_exposure"]
    print(
        f"   Regimes Encountered: {len(market['regimes_encountered'])}/{market['regimes_required']}"
    )
    if market["regimes_encountered"]:
        print(f"   Regimes: {', '.join(market['regimes_encountered'])}")
    if market["regimes_remaining"] > 0:
        print(f"   Regimes Needed: {market['regimes_remaining']}")
    print()

    # Recent trades
    recent_trades = [
        t for t in shadow_engine.shadow_trades if (datetime.now() - t.entry_time).days <= 7
    ]

    if recent_trades:
        print(f"📋 RECENT TRADES (Last 7 days): {len(recent_trades)}")
        for trade in recent_trades[-5:]:  # Show last 5
            print(
                f"   {trade.entry_time.strftime('%m-%d')} {trade.symbol} {trade.side} "
                f"{trade.quantity:.2f} @ ${trade.entry_price:.4f} "
                f"(conf: {trade.confidence_score:.0%})"
            )
    print()

    # Recommendations
    print(f"💡 RECOMMENDATIONS:")

    if status["live_trading_authorized"]:
        print(f"   ✅ Live trading is authorized - ready for production deployment")
    elif soak["days_remaining"] > 0:
        print(f"   ⏳ Continue shadow trading for {soak['days_remaining']} more days")
    elif trading["trades_remaining"] > 0:
        print(f"   📈 Need {trading['trades_remaining']} more trades for statistical significance")
    elif market["regimes_remaining"] > 0:
        print(f"   🌍 Need exposure to {market['regimes_remaining']} more market regimes")
    else:
        print(f"   🎯 Ready for soak period validation!")


def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Start continuous monitoring:")
    print("  python scripts/shadow_trading_monitor.py monitor")
    print()
    print("Custom monitoring interval:")
    print("  python scripts/shadow_trading_monitor.py monitor --interval 30")
    print()
    print("Generate daily report:")
    print("  python scripts/shadow_trading_monitor.py report")
    print()
    print("Automated scheduling (cron):")
    print("  # Daily report at 9:00 AM")
    print("  0 9 * * * /usr/bin/python3 /path/to/scripts/shadow_trading_monitor.py report")
    print("  # Continuous monitoring (run once)")
    print("  @reboot /usr/bin/python3 /path/to/scripts/shadow_trading_monitor.py monitor")
    print()


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Shadow Trading Monitor - Monitor soak period progress and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/shadow_trading_monitor.py monitor           # Start monitoring
  python scripts/shadow_trading_monitor.py monitor --interval 30  # 30-min intervals
  python scripts/shadow_trading_monitor.py report            # Daily report
        """,
    )

    parser.add_argument("command", choices=["monitor", "report"], help="Command to execute")

    parser.add_argument(
        "--interval", type=int, default=60, help="Monitoring interval in minutes (default: 60)"
    )

    parser.add_argument("--examples", action="store_true", help="Show usage examples")

    args = parser.parse_args()

    if args.examples:
        print_usage_examples()
        return 0

    # Execute command
    if args.command == "monitor":
        await run_shadow_trading_monitor(args.interval)
    elif args.command == "report":
        await run_daily_soak_report()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Shadow trading monitor interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
