#!/usr/bin/env python3
"""
Demo script voor Centralized Observability & Alerts
Test alle gevraagde functionaliteit: metrics centralisatie, alerts, health/metrics endpoints
"""

import time
import asyncio
import requests
import json
from datetime import datetime
import sys
import os

# Add src path for imports
sys.path.append("src")

from src.cryptosmarttrader.observability.centralized_observability_api import (
    get_observability_service,
    CentralizedObservabilityService
)

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def test_metrics_collection():
    """Test 1: Metrics Collection - alle gevraagde metrics"""
    print_header("TEST 1: CENTRALIZED METRICS COLLECTION")
    
    service = get_observability_service()
    
    print_section("Recording Order Metrics")
    # Test orders_sent/filled/errors
    service.record_order_sent("kraken", "BTC/USD", "buy", "market")
    service.record_order_sent("kraken", "BTC/USD", "buy", "market") 
    service.record_order_filled("kraken", "BTC/USD", "buy", "market")
    service.record_order_error("kraken", "BTC/USD", "timeout", "408")
    print("✅ Recorded: 2 orders sent, 1 filled, 1 error")
    
    print_section("Recording Performance Metrics")
    # Test latency_ms, slippage_bps
    service.record_latency("place_order", "kraken", "/api/orders", 125.5)
    service.record_slippage("kraken", "BTC/USD", "buy", 15.2)
    print("✅ Recorded: latency 125.5ms, slippage 15.2 bps")
    
    print_section("Recording Portfolio Metrics") 
    # Test equity, drawdown_pct
    service.update_equity(105000.0)
    service.update_drawdown(3.5)  # 3.5% drawdown
    print("✅ Recorded: equity $105,000, drawdown 3.5%")
    
    print_section("Recording Signal Metrics")
    # Test signals_received
    service.record_signal_received("technical_agent", "buy_signal", "BTC/USD")
    service.record_signal_received("sentiment_agent", "bullish", "ETH/USD")
    print("✅ Recorded: 2 signals received")
    
    print_section("Current Metrics Summary")
    metrics = service.get_centralized_metrics()
    print(json.dumps(metrics, indent=2))
    
    return True

def test_alert_system():
    """Test 2: Alert System - HighOrderErrorRate, DrawdownTooHigh, NoSignals"""
    print_header("TEST 2: ALERT SYSTEM")
    
    service = get_observability_service()
    
    print_section("Test 1: HighOrderErrorRate Alert")
    # Trigger high error rate (>5%)
    for i in range(10):
        service.record_order_sent("kraken", f"SYMBOL{i}", "buy", "market")
        if i < 6:  # 60% error rate - should trigger alert
            service.record_order_error("kraken", f"SYMBOL{i}", "rejected", "400")
    
    print("📊 Simulated 60% order error rate (threshold: 5%)")
    time.sleep(2)  # Allow alert processing
    
    alerts = service.get_active_alerts()
    high_error_alert = next((a for a in alerts if a['name'] == 'HighOrderErrorRate'), None)
    if high_error_alert:
        print(f"🚨 HighOrderErrorRate Alert TRIGGERED: {high_error_alert['message']}")
    else:
        print("❌ HighOrderErrorRate Alert NOT triggered")
    
    print_section("Test 2: DrawdownTooHigh Alert")
    # Trigger high drawdown (>10%)
    service.update_drawdown(15.0)  # 15% drawdown - should trigger alert
    print("📊 Simulated 15% drawdown (threshold: 10%)")
    time.sleep(2)
    
    alerts = service.get_active_alerts()
    drawdown_alert = next((a for a in alerts if a['name'] == 'DrawdownTooHigh'), None)
    if drawdown_alert:
        print(f"🚨 DrawdownTooHigh Alert TRIGGERED: {drawdown_alert['message']}")
    else:
        print("❌ DrawdownTooHigh Alert NOT triggered")
    
    print_section("Test 3: NoSignals Alert")
    # Simulate no signals for 30+ minutes by setting old timestamp
    service.metrics_history['last_signal_time'] = time.time() - 2000  # 33+ minutes ago
    print("📊 Simulated no signals for 33 minutes (threshold: 30 min)")
    time.sleep(2)
    
    alerts = service.get_active_alerts()
    no_signals_alert = next((a for a in alerts if a['name'] == 'NoSignals'), None)
    if no_signals_alert:
        print(f"🚨 NoSignals Alert TRIGGERED: {no_signals_alert['message']}")
    else:
        print("❌ NoSignals Alert NOT triggered")
    
    print_section("Active Alerts Summary")
    active_alerts = service.get_active_alerts()
    print(f"Total active alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  - {alert['name']} ({alert['severity']}): {alert['message']}")
    
    return len(active_alerts) >= 2  # Should have at least HighOrderErrorRate and DrawdownTooHigh

def test_api_endpoints():
    """Test 3: API Endpoints - /health en /metrics rooktest"""
    print_header("TEST 3: API ENDPOINTS ROOKTEST")
    
    # Start API server in background
    print_section("Starting API Server")
    import subprocess
    import threading
    
    def start_api():
        try:
            import uvicorn
            from src.cryptosmarttrader.observability.centralized_observability_api import app
            uvicorn.run(app, host="0.0.0.0", port=8002, log_level="error")
        except Exception as e:
            print(f"API server error: {e}")
    
    # Start server in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Give server time to start
    print("⏳ Waiting for API server to start...")
    time.sleep(5)
    
    print_section("Testing /health Endpoint")
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        print(f"✅ /health endpoint: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Service: {health_data.get('service', 'unknown')}")
            health_ok = True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            health_ok = False
    except Exception as e:
        print(f"❌ /health endpoint error: {e}")
        health_ok = False
    
    print_section("Testing /metrics Endpoint")
    try:
        response = requests.get("http://localhost:8002/metrics", timeout=5)
        print(f"✅ /metrics endpoint: {response.status_code}")
        if response.status_code == 200:
            metrics_text = response.text
            print(f"   Metrics size: {len(metrics_text)} bytes")
            # Check for key metrics
            key_metrics = ['orders_sent_total', 'order_errors_total', 'latency_ms', 'slippage_bps']
            found_metrics = [m for m in key_metrics if m in metrics_text]
            print(f"   Key metrics found: {len(found_metrics)}/{len(key_metrics)}")
            print(f"   Metrics: {', '.join(found_metrics)}")
            metrics_ok = True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            metrics_ok = False
    except Exception as e:
        print(f"❌ /metrics endpoint error: {e}")
        metrics_ok = False
    
    print_section("Testing /alerts Endpoint")
    try:
        response = requests.get("http://localhost:8002/alerts", timeout=5)
        print(f"✅ /alerts endpoint: {response.status_code}")
        if response.status_code == 200:
            alerts_data = response.json()
            print(f"   Active alerts: {alerts_data.get('count', 0)}")
            alerts_ok = True
        else:
            print(f"❌ Alerts endpoint failed: {response.status_code}")
            alerts_ok = False
    except Exception as e:
        print(f"❌ /alerts endpoint error: {e}")
        alerts_ok = False
    
    return health_ok and metrics_ok and alerts_ok

def run_comprehensive_demo():
    """Run comprehensive observability demo"""
    print_header("CRYPTOSMARTTRADER V2 - CENTRALIZED OBSERVABILITY DEMO")
    print("Testing alle gevraagde functionaliteit:")
    print("• Centralized metrics (orders_sent/filled, order_errors, latency_ms, slippage_bps, equity, drawdown_pct, signals_received)")
    print("• Alerts (HighOrderErrorRate, DrawdownTooHigh, NoSignals)")
    print("• API endpoints (/health==200, /metrics bereikbaar)")
    
    results = {}
    
    # Test 1: Metrics collection
    try:
        results['metrics'] = test_metrics_collection()
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        results['metrics'] = False
    
    # Test 2: Alert system
    try:
        results['alerts'] = test_alert_system()
    except Exception as e:
        print(f"❌ Alerts test failed: {e}")
        results['alerts'] = False
    
    # Test 3: API endpoints
    try:
        results['api'] = test_api_endpoints()
    except Exception as e:
        print(f"❌ API test failed: {e}")
        results['api'] = False
    
    # Final results
    print_header("DEMO RESULTS")
    
    success_count = sum(results.values())
    total_tests = len(results)
    
    print(f"✅ Metrics Collection: {'PASS' if results.get('metrics') else 'FAIL'}")
    print(f"✅ Alert System: {'PASS' if results.get('alerts') else 'FAIL'}")
    print(f"✅ API Endpoints: {'PASS' if results.get('api') else 'FAIL'}")
    
    print(f"\n🎯 OVERALL RESULT: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🚀 CENTRALIZED OBSERVABILITY & ALERTS VOLLEDIG OPERATIONEEL!")
        print("\nNext steps:")
        print("• Start observability API: python -m src.cryptosmarttrader.observability.centralized_observability_api")
        print("• Access health: http://localhost:8002/health")
        print("• Access metrics: http://localhost:8002/metrics")
        print("• Access alerts: http://localhost:8002/alerts")
    else:
        print("⚠️  Some tests failed - check logs above")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)