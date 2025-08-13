#!/usr/bin/env python3
"""
Demo: Enterprise Risk Management System
Real-time demonstration of hard blockers and kill switch functionality.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.risk.risk_guard import RiskGuard, RiskLevel, TradingMode, RiskMonitor


async def demonstrate_risk_management():
    """Live demonstration of risk management capabilities."""
    print("🛡️ ENTERPRISE RISK MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    # Initialize risk management system
    risk_guard = RiskGuard()
    risk_monitor = RiskMonitor(risk_guard)

    print("✅ Risk management system initialized")
    print(
        f"   Daily loss limits: {risk_guard.limits.daily_loss_warning:.1%} warn, {risk_guard.limits.daily_loss_critical:.1%} critical, {risk_guard.limits.daily_loss_emergency:.1%} emergency"
    )
    print(
        f"   Drawdown limits: {risk_guard.limits.max_drawdown_warning:.1%} warn, {risk_guard.limits.max_drawdown_critical:.1%} critical, {risk_guard.limits.max_drawdown_emergency:.1%} emergency"
    )
    print(
        f"   Position limits: {risk_guard.limits.max_position_size:.1%} max size, {risk_guard.limits.max_total_positions} max count"
    )

    # Start monitoring
    await risk_monitor.start_monitoring(interval_seconds=2)
    print("✅ Risk monitoring service started")

    # Demo 1: Progressive daily loss escalation
    print("\n🔥 DEMO 1: Progressive Daily Loss Escalation")
    print("-" * 50)

    scenarios = [
        (98000.0, "2% daily loss - Normal operations"),
        (97000.0, "3% daily loss - WARNING level"),
        (95000.0, "5% daily loss - CRITICAL level"),
        (92000.0, "8% daily loss - EMERGENCY + Kill Switch"),
    ]

    for equity, description in scenarios:
        risk_guard.update_portfolio_state(equity, {"BTC-USD": 0.015})
        status = risk_guard.get_risk_status()

        print(f"   💰 ${equity:,} - {description}")
        print(f"      Risk Level: {status['risk_level'].upper()}")
        print(f"      Trading Mode: {status['trading_mode'].upper()}")
        print(
            f"      Kill Switch: {'🚨 ACTIVE' if status['kill_switch_active'] else '✅ Inactive'}"
        )
        print(
            f"      Trading Allowed: {'❌ NO' if not risk_guard.is_trading_allowed() else '✅ YES'}"
        )

        if risk_guard.kill_switch_active:
            print("      🚨 TRADING HALTED - Manual intervention required!")
            break

        await asyncio.sleep(1)

    # Reset for next demo
    risk_guard.reset_kill_switch("Demo reset")
    risk_guard.daily_start_equity = 100000.0
    risk_guard.current_equity = 100000.0
    risk_guard.peak_equity = 110000.0  # Higher peak for drawdown demo

    print("\n📉 DEMO 2: Maximum Drawdown Protection")
    print("-" * 50)

    drawdown_scenarios = [
        (104500.0, "5% drawdown - WARNING level"),
        (99000.0, "10% drawdown - CRITICAL level"),
        (93500.0, "15% drawdown - EMERGENCY + Kill Switch"),
    ]

    for equity, description in drawdown_scenarios:
        risk_guard.update_portfolio_state(equity, {"ETH-USD": 0.01})
        drawdown = (risk_guard.peak_equity - equity) / risk_guard.peak_equity

        print(f"   📊 ${equity:,} - {description} (actual: {drawdown:.1%})")
        print(f"      Risk Level: {risk_guard.current_risk_level.value.upper()}")
        print(
            f"      Kill Switch: {'🚨 ACTIVE' if risk_guard.kill_switch_active else '✅ Inactive'}"
        )

        if risk_guard.kill_switch_active:
            print("      🚨 MAXIMUM DRAWDOWN EXCEEDED - System protection activated!")
            break

        await asyncio.sleep(1)

    # Reset for position demo
    risk_guard.reset_kill_switch("Demo reset")
    risk_guard.current_equity = 100000.0

    print("\n⚖️ DEMO 3: Position Size & Exposure Controls")
    print("-" * 50)

    # Demo oversized position
    risk_guard.update_portfolio_state(
        100000.0,
        {"BTC-USD": 0.03},  # 3% position exceeds 2% limit
        asset_exposures={"BTC": 0.06},  # 6% asset exposure exceeds 5% limit
    )

    print(f"   🔍 Oversized position test:")
    print(f"      Position Size: 3.0% (limit: {risk_guard.limits.max_position_size:.1%})")
    print(f"      Asset Exposure: 6.0% (limit: {risk_guard.limits.max_asset_exposure:.1%})")
    print(f"      Risk Level: {risk_guard.current_risk_level.value.upper()}")

    print("\n🌐 DEMO 4: Data Quality Monitoring")
    print("-" * 50)

    # Reset for data quality demo
    risk_guard.reset_kill_switch("Demo reset")

    # Demo API reliability monitoring
    print("   📡 Testing API reliability monitoring...")
    for i in range(20):
        success = i < 15  # 75% success rate (below 90% threshold)
        risk_guard.update_data_quality(datetime.now(), success, 100.0)
        if i % 5 == 0:
            current_rate = risk_guard.api_success_count / max(1, risk_guard.api_total_count)
            print(
                f"      API Success Rate: {current_rate:.1%} ({risk_guard.api_success_count}/{risk_guard.api_total_count})"
            )

    if risk_guard.kill_switch_active:
        print("      🚨 API RELIABILITY KILL SWITCH ACTIVATED")
        print("      📉 Success rate below 90% threshold")

    # Reset for latency demo
    risk_guard.reset_kill_switch("Demo reset")

    print("   ⚡ Testing latency monitoring...")
    for i in range(10):
        latency = 8000.0  # 8 second latency (above 5 second limit)
        risk_guard.update_data_quality(datetime.now(), True, latency)
        if i % 3 == 0:
            avg_latency = sum(risk_guard.recent_latencies[-5:]) / min(
                5, len(risk_guard.recent_latencies)
            )
            print(
                f"      Average Latency: {avg_latency:.0f}ms (limit: {risk_guard.limits.max_latency_ms}ms)"
            )

    if risk_guard.kill_switch_active:
        print("      🚨 HIGH LATENCY KILL SWITCH ACTIVATED")
        print("      📡 Latency exceeds 5 second threshold")

    print("\n🔄 DEMO 5: Manual Kill Switch & Recovery")
    print("-" * 50)

    # Manual kill switch
    risk_guard.reset_kill_switch("Demo reset")
    risk_guard.manual_kill_switch("Emergency manual intervention")

    print("   🛑 Manual kill switch activated")
    print(f"      Trading Allowed: {'❌ NO' if not risk_guard.is_trading_allowed() else '✅ YES'}")
    print(f"      Trading Mode: {risk_guard.trading_mode.value.upper()}")

    await asyncio.sleep(2)

    # Manual recovery
    risk_guard.reset_kill_switch("Manual recovery complete")
    print("   🔄 Manual recovery executed")
    print(f"      Trading Allowed: {'✅ YES' if risk_guard.is_trading_allowed() else '❌ NO'}")
    print(f"      Trading Mode: {risk_guard.trading_mode.value.upper()}")

    # Show final risk status
    print("\n📊 FINAL RISK STATUS REPORT")
    print("-" * 50)

    final_status = risk_guard.get_risk_status()
    print(f"   Risk Level: {final_status['risk_level'].upper()}")
    print(f"   Trading Mode: {final_status['trading_mode'].upper()}")
    print(f"   Kill Switch: {'🚨 ACTIVE' if final_status['kill_switch_active'] else '✅ Inactive'}")
    print(f"   Daily PnL: {final_status['daily_pnl']:.2%}")
    print(f"   Total Drawdown: {final_status['total_drawdown']:.2%}")
    print(f"   Position Count: {final_status['position_count']}")
    print(f"   Recent Events: {final_status['recent_events']}")

    # Stop monitoring
    await risk_monitor.stop_monitoring()
    print("\n✅ Risk monitoring demonstration completed")

    # Show event log summary
    print(f"\n📋 RISK EVENTS SUMMARY ({len(risk_guard.risk_events)} events)")
    print("-" * 50)

    for event in risk_guard.risk_events[-5:]:  # Show last 5 events
        print(
            f"   {event.timestamp.strftime('%H:%M:%S')} [{event.severity.value.upper()}] {event.description}"
        )


def show_configuration():
    """Show current risk management configuration."""
    print("\n⚙️ RISK MANAGEMENT CONFIGURATION")
    print("=" * 60)

    guard = RiskGuard()
    limits = guard.limits.dict()

    print("🚨 HARD BLOCKERS (Auto Kill Switch)")
    print("-" * 30)
    print(f"Daily Loss Emergency: {limits['daily_loss_emergency']:.1%}")
    print(f"Max Drawdown Emergency: {limits['max_drawdown_emergency']:.1%}")
    print(f"Data Gap Threshold: {limits['max_data_gap_seconds']}s")
    print(f"Max Latency: {limits['max_latency_ms']}ms")
    print(f"Min API Success Rate: {limits['min_api_success_rate']:.1%}")

    print("\n⚠️ WARNING LEVELS")
    print("-" * 30)
    print(f"Daily Loss Warning: {limits['daily_loss_warning']:.1%}")
    print(f"Daily Loss Critical: {limits['daily_loss_critical']:.1%}")
    print(f"Max Drawdown Warning: {limits['max_drawdown_warning']:.1%}")
    print(f"Max Drawdown Critical: {limits['max_drawdown_critical']:.1%}")

    print("\n📊 POSITION LIMITS")
    print("-" * 30)
    print(f"Max Position Size: {limits['max_position_size']:.1%}")
    print(f"Max Asset Exposure: {limits['max_asset_exposure']:.1%}")
    print(f"Max Cluster Exposure: {limits['max_cluster_exposure']:.1%}")
    print(f"Max Total Positions: {limits['max_total_positions']}")


if __name__ == "__main__":
    print("🛡️ CRYPTOSMARTTRADER V2 - RISK MANAGEMENT DEMO")
    print("=" * 60)

    # Show configuration
    show_configuration()

    # Run demonstration
    try:
        asyncio.run(demonstrate_risk_management())
        print("\n🏆 Risk management demonstration completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        sys.exit(1)
