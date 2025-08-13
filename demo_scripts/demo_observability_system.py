#!/usr/bin/env python3
"""
Demo: Enterprise Observability System
Comprehensive demonstration of metrics collection, alerting, and monitoring.
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.observability import (
    MetricsCollector,
    AlertManager,
    get_metrics_collector,
    create_alert_manager,
    AlertSeverity,
)


async def demonstrate_observability_system():
    """Comprehensive demonstration of observability system."""
    print("📊 ENTERPRISE OBSERVABILITY SYSTEM DEMONSTRATION")
    print("=" * 65)

    # Initialize observability components
    metrics_collector = get_metrics_collector()
    alert_manager = create_alert_manager(metrics_collector)

    print("✅ Observability System initialized")
    print(f"   Metrics collector: {len(metrics_collector.registry._collector_to_names)} collectors")
    print(f"   Alert rules: {len(alert_manager.alert_rules)} configured")

    # Demo 1: Order Execution Metrics
    print("\n📈 DEMO 1: Order Execution Metrics Collection")
    print("-" * 50)

    # Simulate order execution sequence
    symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]

    print("   Simulating order execution sequence...")
    for i in range(15):
        symbol = random.choice(symbols)
        side = random.choice(["buy", "sell"])

        # Record order sent
        request_id = metrics_collector.start_request(
            "execution", "place_order", symbol=symbol, side=side
        )
        metrics_collector.record_order_sent(symbol, side)

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Simulate order outcome
        if random.random() > 0.05:  # 95% success rate
            # Order filled
            quantity = random.uniform(0.1, 2.0)
            price = (
                random.uniform(30000, 60000) if symbol == "BTC-USD" else random.uniform(2000, 4000)
            )
            slippage_bps = random.uniform(1.0, 25.0)
            latency = random.uniform(0.05, 0.5)

            metrics_collector.record_order_filled(
                symbol, side, quantity, price, slippage_bps, latency
            )
            metrics_collector.end_request(request_id, success=True, filled_quantity=quantity)

            print(
                f"      Order {i + 1}: {side} {quantity:.2f} {symbol} @ ${price:,.2f} - {slippage_bps:.1f} bps slippage"
            )
        else:
            # Order error
            error_type = random.choice(["insufficient_balance", "market_closed", "rate_limit"])
            metrics_collector.record_order_error(
                symbol, error_type, f"Simulated {error_type} error"
            )
            metrics_collector.end_request(request_id, success=False, error=error_type)

            print(f"      Order {i + 1}: {side} {symbol} - ERROR: {error_type}")

    # Demo 2: Trading Performance Metrics
    print("\n💰 DEMO 2: Trading Performance Metrics")
    print("-" * 50)

    # Simulate portfolio updates
    initial_equity = 100000.0
    current_equity = initial_equity
    peak_equity = initial_equity

    print("   Simulating portfolio performance updates...")
    for day in range(10):
        # Simulate daily performance
        daily_return = random.uniform(-0.03, 0.05)  # -3% to +5% daily
        current_equity *= 1 + daily_return
        peak_equity = max(peak_equity, current_equity)

        # Calculate drawdown
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
        daily_pnl = current_equity - initial_equity

        # Update metrics
        metrics_collector.update_equity(current_equity)
        metrics_collector.update_drawdown(drawdown_pct)
        metrics_collector.update_daily_pnl(daily_pnl)

        print(
            f"      Day {day + 1}: Equity ${current_equity:,.2f}, Drawdown {drawdown_pct:.1f}%, PnL ${daily_pnl:,.2f}"
        )

        # Update risk score
        risk_score = min(100, drawdown_pct * 5 + random.uniform(10, 30))
        metrics_collector.update_risk_score(risk_score)

        await asyncio.sleep(0.1)

    # Demo 3: Signal Reception Tracking
    print("\n📡 DEMO 3: Signal Reception Tracking")
    print("-" * 50)

    signal_sources = [
        "technical_analysis",
        "sentiment_analysis",
        "regime_detector",
        "whale_tracker",
    ]
    signal_types = ["buy", "sell", "neutral", "high_confidence", "low_confidence"]

    print("   Simulating trading signal reception...")
    for i in range(20):
        source = random.choice(signal_sources)
        signal_type = random.choice(signal_types)
        symbol = random.choice(symbols)
        confidence = random.uniform(0.6, 0.95)

        metrics_collector.record_signal_received(
            source,
            signal_type,
            symbol=symbol,
            confidence=confidence,
            strength=random.uniform(0.3, 1.0),
        )

        print(
            f"      Signal {i + 1}: {source} -> {signal_type} for {symbol} (confidence: {confidence:.2f})"
        )

        await asyncio.sleep(0.05)

    # Demo 4: API Request Monitoring
    print("\n🌐 DEMO 4: API Request Monitoring")
    print("-" * 50)

    api_endpoints = ["/market/orderbook", "/trading/orders", "/account/balance", "/market/trades"]
    exchanges = ["kraken", "binance", "kucoin"]

    print("   Simulating API request monitoring...")
    for i in range(25):
        endpoint = random.choice(api_endpoints)
        method = random.choice(["GET", "POST"])
        exchange = random.choice(exchanges)

        # Simulate request
        start_time = time.time()
        await asyncio.sleep(random.uniform(0.01, 0.3))  # Simulate API latency
        duration = time.time() - start_time

        # Simulate response status
        status_code = random.choices([200, 400, 429, 500], weights=[90, 5, 3, 2])[0]

        metrics_collector.record_api_request(endpoint, method, status_code, duration)

        # Update exchange connectivity
        connected = status_code < 500
        metrics_collector.update_exchange_connectivity(exchange, connected)

        status_text = "✅" if status_code == 200 else "❌"
        print(
            f"      API {i + 1}: {method} {endpoint} -> {status_code} ({duration * 1000:.0f}ms) {status_text}"
        )

    # Demo 5: Alert System Evaluation
    print("\n🚨 DEMO 5: Alert System Evaluation")
    print("-" * 50)

    print("   Evaluating alert rules against current metrics...")

    # Get current metrics summary
    metrics_summary = metrics_collector.get_metrics_summary()

    print("   Current metrics summary:")
    print(
        f"      Orders: {metrics_summary['orders']['total_sent']} sent, {metrics_summary['orders']['total_filled']} filled"
    )
    print(f"      Equity: ${metrics_summary['trading']['equity_usd']:,.2f}")
    print(f"      Drawdown: {metrics_summary['trading']['drawdown_percent']:.1f}%")
    print(f"      Signals: {metrics_summary['system']['signals_received']} received")
    print(
        f"      Last signal: {metrics_summary['system']['minutes_since_last_signal']:.1f} minutes ago"
    )

    # Evaluate alert rules
    alert_manager.evaluate_rules(metrics_summary)

    # Get alert summary
    alert_summary = alert_manager.get_alert_summary()
    print(f"\n   Alert System Status:")
    print(f"      Total rules: {alert_summary['total_rules']}")
    print(f"      Enabled rules: {alert_summary['enabled_rules']}")
    print(f"      Active alerts: {alert_summary['active_alerts']}")
    print(f"      Severity distribution: {alert_summary['severity_distribution']}")

    # Show active alerts
    active_alerts = alert_manager.get_active_alerts()
    if active_alerts:
        print("\n   Active Alerts:")
        for alert in active_alerts:
            print(
                f"      🚨 {alert['name']} ({alert['severity']}) - triggered {alert['trigger_count']} times"
            )
    else:
        print("   ✅ No active alerts")

    # Demo 6: Correlation ID Tracking
    print("\n🔗 DEMO 6: Request Correlation Tracking")
    print("-" * 50)

    print("   Demonstrating correlated request tracking...")

    # Simulate complex trading workflow with correlation
    workflow_requests = []

    # Start market analysis request
    req1 = metrics_collector.start_request(
        "market_analysis", "analyze_opportunity", symbol="BTC-USD"
    )
    workflow_requests.append(req1)
    await asyncio.sleep(0.1)

    # Start risk assessment request
    req2 = metrics_collector.start_request(
        "risk_management", "assess_risk", symbol="BTC-USD", parent_request=req1
    )
    workflow_requests.append(req2)
    await asyncio.sleep(0.05)

    # Start execution request
    req3 = metrics_collector.start_request(
        "execution", "place_order", symbol="BTC-USD", parent_request=req2
    )
    workflow_requests.append(req3)
    await asyncio.sleep(0.2)

    # Complete requests in reverse order (simulating completion)
    metrics_collector.end_request(req3, success=True, order_id="ORD123456")
    metrics_collector.end_request(req2, success=True, risk_score=25.5)
    metrics_collector.end_request(req1, success=True, opportunity_score=0.85)

    print("   ✅ Correlated workflow completed with full traceability")
    print(f"      Market Analysis -> Risk Assessment -> Order Execution")
    print(f"      All requests tracked with correlation IDs for debugging")

    # Demo 7: Alerting Scenarios
    print("\n⚠️ DEMO 7: Alert Triggering Scenarios")
    print("-" * 50)

    print("   Testing various alert scenarios...")

    # Trigger high drawdown alert
    print("   🔴 Triggering high drawdown alert...")
    metrics_collector.update_drawdown(12.5)  # Above 10% threshold

    # Force no signals alert by manipulating timestamp
    print("   🔴 Triggering no signals alert...")
    original_time = metrics_collector.last_signal_time
    metrics_collector.last_signal_time = time.time() - 3600  # 1 hour ago

    # Re-evaluate alerts
    updated_metrics = metrics_collector.get_metrics_summary()
    alert_manager.evaluate_rules(updated_metrics)

    # Show new alerts
    final_alerts = alert_manager.get_active_alerts()
    print(f"   Alert count after scenarios: {len(final_alerts)}")
    for alert in final_alerts:
        severity_emoji = {"warning": "⚠️", "critical": "🔴", "emergency": "🚨"}.get(
            alert["severity"], "ℹ️"
        )
        print(f"      {severity_emoji} {alert['name']} - {alert['severity']}")

    # Restore original timestamp
    metrics_collector.last_signal_time = original_time

    # Demo 8: Metrics Export
    print("\n📤 DEMO 8: Prometheus Metrics Export")
    print("-" * 50)

    print("   Generating Prometheus metrics export...")

    # Get Prometheus metrics
    prometheus_metrics = metrics_collector.get_metrics()

    # Count metrics
    metric_lines = [
        line for line in prometheus_metrics.split("\n") if line and not line.startswith("#")
    ]
    help_lines = [line for line in prometheus_metrics.split("\n") if line.startswith("# HELP")]

    print(f"   📊 Metrics exported:")
    print(f"      Total metric families: {len(help_lines)}")
    print(f"      Total data points: {len(metric_lines)}")
    print(f"      Export size: {len(prometheus_metrics)} bytes")

    # Show sample metrics
    print("\n   Sample metrics:")
    sample_lines = metric_lines[:5]
    for line in sample_lines:
        if line.strip():
            print(f"      {line}")
    if len(metric_lines) > 5:
        print(f"      ... and {len(metric_lines) - 5} more metrics")

    # Demo 9: Performance Summary
    print("\n📊 DEMO 9: Final Performance Summary")
    print("-" * 50)

    final_summary = metrics_collector.get_metrics_summary()
    alert_summary = alert_manager.get_alert_summary()

    print(f"   📈 Final Metrics Summary:")
    print(f"      Total Orders: {final_summary['orders']['total_sent']}")
    print(
        f"      Fill Rate: {(final_summary['orders']['total_filled'] / max(1, final_summary['orders']['total_sent']) * 100):.1f}%"
    )
    print(
        f"      Error Rate: {(final_summary['orders']['total_errors'] / max(1, final_summary['orders']['total_sent']) * 100):.1f}%"
    )
    print(f"      Current Equity: ${final_summary['trading']['equity_usd']:,.2f}")
    print(f"      Max Drawdown: {final_summary['trading']['drawdown_percent']:.1f}%")
    print(f"      Signals Received: {final_summary['system']['signals_received']}")
    print(f"      Active Requests: {final_summary['system']['active_requests']}")

    print(f"\n   🚨 Alert System Summary:")
    print(f"      Total Rules: {alert_summary['total_rules']}")
    print(f"      Active Alerts: {alert_summary['active_alerts']}")
    print(f"      Escalated Alerts: {alert_summary['escalated_alerts']}")
    print(
        f"      Critical/Emergency: {alert_summary['severity_distribution']['critical'] + alert_summary['severity_distribution']['emergency']}"
    )

    print("\n✅ OBSERVABILITY SYSTEM DEMONSTRATION COMPLETED")
    print("=" * 65)

    print(f"📊 COMPREHENSIVE MONITORING ACHIEVED:")
    print(f"   ✅ Order execution metrics with latency and slippage tracking")
    print(f"   ✅ Trading performance monitoring with equity and drawdown")
    print(f"   ✅ Signal reception tracking with source attribution")
    print(f"   ✅ API monitoring with error rates and connectivity status")
    print(f"   ✅ Alert system with severity levels and escalation")
    print(f"   ✅ Request correlation with end-to-end traceability")
    print(f"   ✅ Structured JSON logging with correlation IDs")
    print(f"   ✅ Prometheus metrics export for external monitoring")


if __name__ == "__main__":
    print("📊 CRYPTOSMARTTRADER V2 - OBSERVABILITY SYSTEM DEMO")
    print("=" * 65)

    try:
        asyncio.run(demonstrate_observability_system())
        print("\n🏆 Observability system demonstration completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
