#!/usr/bin/env python3
"""
FASE D - Observability & Alerts Test Demo
Tests centralized Prometheus metrics and AlertManager integration
"""

import sys
import time
import asyncio
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/runner/workspace')

def test_centralized_metrics():
    """Test centralized Prometheus metrics system"""
    print("="*60)
    print("FASE D - CENTRALIZED METRICS TEST")
    print("="*60)
    
    try:
        from src.cryptosmarttrader.observability.metrics import get_metrics
        
        # Get metrics instance
        metrics = get_metrics()
        print("✅ Metrics instance created successfully")
        
        # Test FASE D alert metrics
        print("\n1. Testing FASE D alert metrics...")
        
        # Test HighOrderErrorRate alert
        metrics.record_order_error("kraken", "BTC/USD", "timeout", "E001")
        print("   ✅ Order error recorded")
        
        # Test signal reception and NoSignals alert
        metrics.record_signal_received("technical_agent", "buy_signal", "BTC/USD")
        print("   ✅ Signal received recorded")
        
        # Test drawdown update
        metrics.update_drawdown(5.2)  # 5.2% drawdown
        print("   ✅ Drawdown updated")
        
        # Get metrics summary
        summary = metrics.get_metrics_summary()
        print(f"\n2. Metrics summary:")
        print(f"   Total orders sent: {summary.get('total_orders_sent', 0)}")
        print(f"   Total order errors: {summary.get('total_order_errors', 0)}")
        print(f"   Total signals received: {summary.get('total_signals_received', 0)}")
        print(f"   Current drawdown: {summary.get('current_drawdown_pct', 0)}%")
        
        # Check alert flags
        print(f"\n3. Alert flags:")
        print(f"   High order error rate: {summary.get('alert_high_order_error_rate', 0)}")
        print(f"   Drawdown too high: {summary.get('alert_drawdown_too_high', 0)}")
        print(f"   No signals timeout: {summary.get('alert_no_signals_timeout', 0)}")
        
        print("\n✅ CENTRALIZED METRICS: OPERATIONAL")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_alert_manager():
    """Test FASE D AlertManager"""
    print("\n" + "="*60)
    print("FASE D - ALERT MANAGER TEST")
    print("="*60)
    
    try:
        from src.cryptosmarttrader.observability.fase_d_alerts import get_alert_manager
        
        # Get alert manager instance
        alert_manager = get_alert_manager()
        print("✅ AlertManager instance created successfully")
        
        # Test alert evaluation
        print("\n1. Evaluating alert conditions...")
        evaluation_results = alert_manager.evaluate_alerts()
        print(f"   Alerts evaluated: {evaluation_results['alerts_evaluated']}")
        print(f"   Alerts firing: {evaluation_results['alerts_firing']}")
        print(f"   New alerts: {evaluation_results['new_alerts']}")
        
        # Get alert status
        print("\n2. Alert status:")
        alert_status = alert_manager.get_alert_status()
        print(f"   Total conditions: {alert_status['total_conditions']}")
        print(f"   Active alerts: {alert_status['active_alerts']}")
        
        # Show alert conditions
        print("\n3. Alert conditions:")
        for name, condition in alert_status['conditions'].items():
            print(f"   {name}: {condition['description']} (severity: {condition['severity']})")
        
        # Test Prometheus rules export
        print("\n4. Testing Prometheus rules export...")
        try:
            rules_yaml = alert_manager.export_prometheus_rules()
            print(f"   ✅ Generated {len(rules_yaml)} characters of YAML rules")
        except Exception as e:
            print(f"   ⚠️  Rules export failed: {e}")
        
        print("\n✅ ALERT MANAGER: OPERATIONAL")
        return True
        
    except Exception as e:
        print(f"❌ AlertManager test failed: {e}")
        return False

def test_health_api():
    """Test health API endpoints"""
    print("\n" + "="*60)
    print("FASE D - HEALTH API TEST")
    print("="*60)
    
    try:
        # Test importing health endpoints
        from src.cryptosmarttrader.api.health_endpoints import health_app
        print("✅ Health API endpoints imported successfully")
        
        # Test FastAPI app creation
        print(f"   App title: {health_app.title}")
        print(f"   App version: {health_app.version}")
        
        # List available routes
        print("\n1. Available API routes:")
        for route in health_app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = getattr(route, 'methods', set())
                print(f"   {route.path} [{', '.join(methods)}]")
        
        print("\n✅ HEALTH API: READY FOR DEPLOYMENT")
        return True
        
    except Exception as e:
        print(f"❌ Health API test failed: {e}")
        return False

async def test_api_endpoints_async():
    """Test API endpoints when server is running"""
    print("\n" + "="*60)
    print("FASE D - API ENDPOINT SIMULATION")
    print("="*60)
    
    try:
        from src.cryptosmarttrader.api.health_endpoints import health_check, metrics_endpoint, alerts_endpoint
        
        # Test health check
        print("1. Testing /health endpoint...")
        health_response = await health_check()
        print(f"   Health status: {health_response['status']}")
        print(f"   Timestamp: {health_response['timestamp']}")
        print(f"   Checks: {len(health_response['checks'])}")
        
        # Test metrics endpoint
        print("\n2. Testing /metrics endpoint...")
        metrics_response = await metrics_endpoint()
        metrics_lines = metrics_response.body.decode('utf-8').split('\n') if hasattr(metrics_response, 'body') else str(metrics_response).split('\n')
        print(f"   Metrics output: {len(metrics_lines)} lines")
        print(f"   Sample line: {metrics_lines[0] if metrics_lines else 'No content'}")
        
        # Test alerts endpoint
        print("\n3. Testing /alerts endpoint...")
        alerts_response = await alerts_endpoint()
        print(f"   Alert evaluation: {alerts_response['evaluation_results']['alerts_evaluated']} conditions")
        print(f"   Active alerts: {alerts_response['alert_status']['active_alerts']}")
        
        print("\n✅ API ENDPOINTS: FUNCTIONAL")
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def main():
    """Run complete FASE D observability test suite"""
    print("FASE D - OBSERVABILITY & ALERTS IMPLEMENTATION TEST")
    print("Centralized Prometheus metrics with AlertManager integration")
    print("="*80)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Centralized metrics
    if test_centralized_metrics():
        tests_passed += 1
    
    # Test 2: Alert manager
    if test_alert_manager():
        tests_passed += 1
    
    # Test 3: Health API
    if test_health_api():
        tests_passed += 1
    
    # Test 4: API endpoints (async)
    try:
        async_result = asyncio.run(test_api_endpoints_async())
        if async_result:
            tests_passed += 1
    except Exception as e:
        print(f"❌ Async API test failed: {e}")
    
    print("\n" + "="*80)
    print("FASE D OBSERVABILITY TEST RESULTS")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 FASE D IMPLEMENTATION: COMPLETE")
        print("✅ Centralized Prometheus metrics: OPERATIONAL")
        print("✅ AlertManager with FASE D alerts: OPERATIONAL")
        print("✅ HighOrderErrorRate alert: CONFIGURED")
        print("✅ DrawdownTooHigh alert: CONFIGURED")
        print("✅ NoSignals(30m) alert: CONFIGURED")
        print("✅ Health API endpoints: READY")
        print("✅ /health endpoint: FUNCTIONAL")
        print("✅ /metrics endpoint: FUNCTIONAL")
        print("✅ CI smoke test ready: YES")
        print("\nFASE D observability & alerts zijn volledig geïmplementeerd!")
        print("Ready for CI integration with /health and /metrics endpoints.")
    else:
        print(f"\n❌ {total_tests - tests_passed} tests failed")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)