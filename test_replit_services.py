#!/usr/bin/env python3
"""
Test Replit Multi-Service Configuration
"""

import requests
import time
import json
from typing import Dict, Any


def test_service_health(service_name: str, port: int, health_path: str) -> Dict[str, Any]:
    """Test health endpoint of a service"""
    url = f"http://localhost:{port}{health_path}"

    try:
        response = requests.get(url, timeout=5)
        status = "healthy" if response.status_code == 200 else "unhealthy"

        try:
            data = response.json() if response.content else {}
        except Exception:
            data = {"response": response.text[:200]}

        return {
            "service": service_name,
            "port": port,
            "status": status,
            "status_code": response.status_code,
            "url": url,
            "data": data,
        }
    except Exception as e:
        return {
            "service": service_name,
            "port": port,
            "status": "error",
            "error": str(e),
            "url": url,
        }


def test_all_services():
    """Test all configured services"""
    services = [
        ("Dashboard", 5000, "/_stcore/health"),
        ("API", 8001, "/health"),
        ("Metrics", 8000, "/health"),
    ]

    print("🧪 CryptoSmartTrader V2 - Multi-Service Health Test")
    print("=" * 60)

    results = []
    for service_name, port, health_path in services:
        print(f"\n🔍 Testing {service_name} (port {port})...")
        result = test_service_health(service_name, port, health_path)
        results.append(result)

        if result["status"] == "healthy":
            print(f"✅ {service_name}: {result['status'].upper()}")
            if "data" in result and isinstance(result["data"], dict):
                if "service" in result["data"]:
                    print(f"   Service Type: {result['data'].get('service', 'unknown')}")
                if "timestamp" in result["data"]:
                    print(f"   Response Time: {result['data']['timestamp']}")
        else:
            print(f"❌ {service_name}: {result['status'].upper()}")
            if "error" in result:
                print(f"   Error: {result['error']}")

    # Test additional endpoints
    print(f"\n🔗 Testing Additional Endpoints...")

    # Test API docs
    try:
        response = requests.get("http://localhost:8001/api/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API Documentation: ACCESSIBLE")
        else:
            print(f"⚠️  API Documentation: {response.status_code}")
    except Exception:
        print("❌ API Documentation: UNAVAILABLE")

    # Test metrics endpoint
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus Metrics: ACCESSIBLE")
        else:
            print(f"⚠️  Prometheus Metrics: {response.status_code}")
    except Exception:
        print("❌ Prometheus Metrics: UNAVAILABLE")

    # Test detailed health
    try:
        response = requests.get("http://localhost:8001/health/detailed", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Detailed Health: ACCESSIBLE")
            print(f"   System Status: {data.get('status', 'unknown')}")
            if "uptime_formatted" in data:
                print(f"   Uptime: {data['uptime_formatted']}")
        else:
            print(f"⚠️  Detailed Health: {response.status_code}")
    except Exception:
        print("❌ Detailed Health: UNAVAILABLE")

    # Summary
    print(f"\n📊 Summary:")
    print(f"-" * 30)
    healthy_count = sum(1 for r in results if r["status"] == "healthy")
    total_count = len(results)

    print(f"Services Healthy: {healthy_count}/{total_count}")
    print(
        f"Overall Status: {'✅ ALL HEALTHY' if healthy_count == total_count else '⚠️  SOME ISSUES'}"
    )

    # Service URLs for Replit
    print(f"\n🌐 Service URLs (Replit):")
    print(f"-" * 30)
    print(f"Dashboard:      http://localhost:5000")
    print(f"API:            http://localhost:8001")
    print(f"Metrics:        http://localhost:8000")
    print(f"API Docs:       http://localhost:8001/api/docs")
    print(f"Health Check:   http://localhost:8001/health")
    print(f"Detailed Health: http://localhost:8001/health/detailed")

    return results


if __name__ == "__main__":
    test_all_services()
