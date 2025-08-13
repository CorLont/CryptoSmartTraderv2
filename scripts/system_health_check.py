#!/usr/bin/env python3
"""
System Health Check Script
Automated system health monitoring with GO/NO-GO authorization gates
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

from core.system_health_monitor import (
    get_system_health_monitor,
    HealthConfig,
    HealthStatus,
    ComponentStatus,
)
from core.logging_manager import get_logger


async def run_system_health_check(detailed: bool = True):
    """Run comprehensive system health check"""

    logger = get_logger()

    print(f"🏥 System Health Check Starting")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        # Initialize health monitor
        health_monitor = get_system_health_monitor()

        # Run health assessment
        print(f"🔍 Assessing system health...")
        health_report = await health_monitor.assess_system_health()

        # Display overall status
        status_icon = {
            HealthStatus.GO: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.NOGO: "❌",
            HealthStatus.CRITICAL: "🚨",
        }.get(health_report.overall_health_status, "❓")

        print(f"\n{status_icon} OVERALL SYSTEM HEALTH:")
        print(f"   Status: {health_report.overall_health_status.value.upper()}")
        print(f"   Health Score: {health_report.overall_health_score:.1f}/100")
        print(f"   Report ID: {health_report.report_id}")
        print()

        # Trading authorization status
        print(f"🚦 TRADING AUTHORIZATION:")
        live_icon = "✅" if health_report.live_trading_authorized else "❌"
        paper_icon = "✅" if health_report.paper_trading_authorized else "❌"

        print(
            f"   {live_icon} Live Trading: {'AUTHORIZED' if health_report.live_trading_authorized else 'BLOCKED'}"
        )
        print(
            f"   {paper_icon} Paper Trading: {'AUTHORIZED' if health_report.paper_trading_authorized else 'BLOCKED'}"
        )
        print(f"   📊 Confidence Gate: {health_report.confidence_gate_status.upper()}")
        print()

        # System metrics
        print(f"📊 SYSTEM METRICS:")
        print(f"   Total coins analyzed: {health_report.total_coins_analyzed}")
        print(f"   High-confidence coins: {health_report.high_confidence_coins}")
        print(f"   Actionable recommendations: {health_report.actionable_recommendations}")
        print()

        # Component health breakdown
        if detailed:
            print(f"🔧 COMPONENT HEALTH BREAKDOWN:")

            for component_name, health in health_report.component_health.items():
                status_icon = {
                    ComponentStatus.HEALTHY: "✅",
                    ComponentStatus.DEGRADED: "⚠️",
                    ComponentStatus.FAILED: "❌",
                    ComponentStatus.UNKNOWN: "❓",
                }.get(health.status, "❓")

                component_display = component_name.replace("_", " ").title()
                print(f"   {status_icon} {component_display}:")
                print(f"      Score: {health.score:.1f}/100")
                print(f"      Raw Value: {health.raw_value:.3f}")
                print(f"      Target: {health.target_value:.3f}")
                print(f"      Status: {health.status.value}")

                if health.error_message:
                    print(f"      Error: {health.error_message}")

                print()
        else:
            # Summary view
            print(f"🔧 COMPONENT HEALTH SUMMARY:")

            healthy_count = sum(
                1
                for h in health_report.component_health.values()
                if h.status == ComponentStatus.HEALTHY
            )
            degraded_count = sum(
                1
                for h in health_report.component_health.values()
                if h.status == ComponentStatus.DEGRADED
            )
            failed_count = sum(
                1
                for h in health_report.component_health.values()
                if h.status == ComponentStatus.FAILED
            )

            print(f"   ✅ Healthy: {healthy_count} components")
            print(f"   ⚠️  Degraded: {degraded_count} components")
            print(f"   ❌ Failed: {failed_count} components")
            print()

        # Critical issues
        if health_report.critical_issues:
            print(f"🚨 CRITICAL ISSUES:")
            for i, issue in enumerate(health_report.critical_issues, 1):
                print(f"   {i}. {issue}")
            print()

        # Warnings
        if health_report.warnings:
            print(f"⚠️  WARNINGS:")
            for i, warning in enumerate(health_report.warnings, 1):
                print(f"   {i}. {warning}")
            print()

        # Recommendations
        if health_report.recommendations:
            print(f"💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(health_report.recommendations, 1):
                print(f"   {i}. {recommendation}")
            print()

        # Health score breakdown
        if detailed:
            print(f"📈 HEALTH SCORE BREAKDOWN:")
            config = health_monitor.config

            print(f"   Component Weights:")
            print(f"     Validation Accuracy: {config.validation_accuracy_weight:.1%}")
            print(f"     Sharpe Ratio: {config.sharpe_ratio_weight:.1%}")
            print(f"     Feedback Hit Rate: {config.feedback_hit_rate_weight:.1%}")
            print(f"     Error Ratio: {config.error_ratio_weight:.1%}")
            print(f"     Data Completeness: {config.data_completeness_weight:.1%}")
            print(f"     Tuning Freshness: {config.tuning_freshness_weight:.1%}")
            print()

            print(f"   Score Thresholds:")
            print(f"     GO (Live Trading): ≥{config.go_threshold:.0f}")
            print(
                f"     WARNING (Paper Only): {config.warning_threshold:.0f}-{config.go_threshold:.0f}"
            )
            print(f"     NO-GO (No Trading): <{config.warning_threshold:.0f}")
            print()

        # Determine exit code
        if health_report.overall_health_status == HealthStatus.GO:
            print(f"✅ HEALTH CHECK PASSED: System ready for live trading")
            exit_code = 0
        elif health_report.overall_health_status == HealthStatus.WARNING:
            print(f"⚠️  HEALTH CHECK WARNING: Paper trading only")
            exit_code = 1
        elif health_report.overall_health_status == HealthStatus.NOGO:
            print(f"❌ HEALTH CHECK FAILED: No trading authorized")
            exit_code = 2
        else:
            print(f"🚨 SYSTEM CRITICAL: Immediate intervention required")
            exit_code = 3

        print(f"\n📝 Health report saved to data/health_reports/")

        return exit_code

    except Exception as e:
        logger.error(f"System health check failed: {e}")
        print(f"🚨 HEALTH CHECK ERROR: {e}")
        return 4


async def run_quick_health_check():
    """Run quick health check without detailed breakdown"""

    print(f"⚡ Quick Health Check")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 40)

    try:
        health_monitor = get_system_health_monitor()
        health_report = await health_monitor.assess_system_health()

        # Quick status display
        status_map = {
            HealthStatus.GO: ("✅", "GO"),
            HealthStatus.WARNING: ("⚠️", "WARNING"),
            HealthStatus.NOGO: ("❌", "NO-GO"),
            HealthStatus.CRITICAL: ("🚨", "CRITICAL"),
        }

        icon, status_text = status_map.get(health_report.overall_health_status, ("❓", "UNKNOWN"))

        print(f"{icon} Status: {status_text}")
        print(f"📊 Score: {health_report.overall_health_score:.1f}/100")
        print(f"🚦 Live Trading: {'✅ YES' if health_report.live_trading_authorized else '❌ NO'}")
        print(
            f"📝 Paper Trading: {'✅ YES' if health_report.paper_trading_authorized else '❌ NO'}"
        )
        print(f"🎯 High-Conf Coins: {health_report.high_confidence_coins}")

        # Quick component summary
        failed_components = [
            name
            for name, health in health_report.component_health.items()
            if health.status == ComponentStatus.FAILED
        ]

        if failed_components:
            print(f"❌ Failed: {', '.join(failed_components)}")

        return 0 if health_report.overall_health_status == HealthStatus.GO else 1

    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        return 1


async def run_trading_authorization_check():
    """Check trading authorization status only"""

    print(f"🚦 Trading Authorization Check")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 40)

    try:
        health_monitor = get_system_health_monitor()
        health_report = await health_monitor.assess_system_health()

        # Get authorization details
        auth_status = health_monitor.check_trading_authorization(health_report)

        print(f"📊 Health Score: {auth_status['health_score']:.1f}")
        print(f"🎯 Health Status: {auth_status['health_status'].upper()}")
        print()

        print(f"🚦 AUTHORIZATION STATUS:")
        print(
            f"   Live Trading: {'✅ AUTHORIZED' if auth_status['live_trading_authorized'] else '❌ BLOCKED'}"
        )
        print(
            f"   Paper Trading: {'✅ AUTHORIZED' if auth_status['paper_trading_authorized'] else '❌ BLOCKED'}"
        )
        print(f"   Confidence Gate: {auth_status['confidence_gate_status'].upper()}")
        print(f"   High-Conf Coins: {auth_status['high_confidence_coins']}")
        print()

        print(f"💭 Reason: {auth_status['authorization_reason']}")
        print(f"⏰ Next Assessment: {auth_status['next_assessment_recommended'][:16]}")

        return 0 if auth_status["live_trading_authorized"] else 1

    except Exception as e:
        print(f"❌ Authorization check failed: {e}")
        return 1


async def monitor_health_continuously(interval_minutes: int = 15):
    """Monitor system health continuously"""

    print(f"🔄 Continuous Health Monitoring")
    print(f"⏰ Interval: {interval_minutes} minutes")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    while True:
        try:
            print(f"\n🔍 Health Check: {datetime.now().strftime('%H:%M:%S')}")

            health_monitor = get_system_health_monitor()
            health_report = await health_monitor.assess_system_health()

            # Quick status display
            status_icons = {
                HealthStatus.GO: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.NOGO: "❌",
                HealthStatus.CRITICAL: "🚨",
            }

            icon = status_icons.get(health_report.overall_health_status, "❓")

            print(
                f"   {icon} {health_report.overall_health_status.value.upper()} - Score: {health_report.overall_health_score:.1f}"
            )
            print(
                f"   🚦 Live: {'✅' if health_report.live_trading_authorized else '❌'} | Paper: {'✅' if health_report.paper_trading_authorized else '❌'}"
            )
            print(f"   🎯 High-Confidence: {health_report.high_confidence_coins} coins")

            # Alert on status changes or critical issues
            if health_report.overall_health_status == HealthStatus.CRITICAL:
                print(f"   🚨 CRITICAL ALERT: {health_report.critical_issues}")
            elif health_report.critical_issues:
                print(f"   ⚠️  Issues: {len(health_report.critical_issues)} critical")

            print(f"   😴 Next check in {interval_minutes} minutes...")

            # Sleep until next check
            await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"   ❌ Monitoring error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry


def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Full system health check:")
    print("  python scripts/system_health_check.py check")
    print()
    print("Quick health status:")
    print("  python scripts/system_health_check.py quick")
    print()
    print("Trading authorization only:")
    print("  python scripts/system_health_check.py auth")
    print()
    print("Continuous monitoring:")
    print("  python scripts/system_health_check.py monitor")
    print("  python scripts/system_health_check.py monitor --interval 30")
    print()
    print("Summary view (less detailed):")
    print("  python scripts/system_health_check.py check --summary")
    print()
    print("Automated scheduling (cron):")
    print("  # Health check every 15 minutes")
    print("  */15 * * * * /usr/bin/python3 /path/to/scripts/system_health_check.py quick")
    print("  # Full health assessment hourly")
    print("  0 * * * * /usr/bin/python3 /path/to/scripts/system_health_check.py check")
    print()


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="System Health Check - GO/NO-GO authorization monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/system_health_check.py check              # Full health check
  python scripts/system_health_check.py quick              # Quick status
  python scripts/system_health_check.py auth               # Authorization only
  python scripts/system_health_check.py monitor            # Continuous monitoring
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Run full system health check")
    check_parser.add_argument(
        "--summary", action="store_true", help="Show summary view instead of detailed breakdown"
    )

    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Run quick health check")

    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Check trading authorization only")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Continuous health monitoring")
    monitor_parser.add_argument(
        "--interval", type=int, default=15, help="Monitoring interval in minutes (default: 15)"
    )

    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Show usage examples")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "check":
        detailed = not args.summary
        exit_code = await run_system_health_check(detailed)
    elif args.command == "quick":
        exit_code = await run_quick_health_check()
    elif args.command == "auth":
        exit_code = await run_trading_authorization_check()
    elif args.command == "monitor":
        exit_code = await monitor_health_continuously(args.interval)
    elif args.command == "examples":
        print_usage_examples()
        exit_code = 0

    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  System health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
