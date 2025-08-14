#!/usr/bin/env python3
"""
Snelle test van alerts via API endpoints
"""

import requests
import time
import json

def test_alerts():
    """Test alert triggering via API"""
    base_url = "http://localhost:8002"
    
    print("🔍 Testing Alert System...")
    
    # 1. Test high order error rate
    print("\n📊 Simulating high order error rate...")
    for i in range(10):
        # Send orders
        requests.post(f"{base_url}/metrics/record/order", 
                     params={"action": "sent", "symbol": f"TEST{i}"})
        # Most fail
        if i < 8:
            requests.post(f"{base_url}/metrics/record/order",
                         params={"action": "error", "symbol": f"TEST{i}", 
                                "error_type": "rejected", "error_code": "400"})
    
    # 2. Test high drawdown
    print("📉 Simulating high drawdown...")
    requests.post(f"{base_url}/metrics/update/portfolio",
                 params={"equity_usd": 90000, "drawdown_pct": 15.0})
    
    # 3. Check alerts
    print("\n🚨 Checking alerts...")
    time.sleep(3)  # Allow processing
    
    response = requests.get(f"{base_url}/alerts")
    if response.status_code == 200:
        alerts = response.json()
        print(f"Active alerts: {alerts['count']}")
        for alert in alerts['active_alerts']:
            print(f"  - {alert['name']} ({alert['severity']}): {alert['message']}")
    
    # 4. Health check
    print("\n🏥 Health check...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Critical alerts: {health['alerts']['critical_count']}")

if __name__ == "__main__":
    test_alerts()