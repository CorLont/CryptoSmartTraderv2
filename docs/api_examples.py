#!/usr/bin/env python3
"""
CryptoSmartTrader V2 API Examples
Comprehensive examples for using the REST API with httpx and requests.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import httpx
import requests
from pydantic import BaseModel


# Configuration
API_BASE_URL = "http://localhost:8001"
API_KEY = "your-api-key-here"  # Replace with actual API key

# Headers for authenticated requests
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


class APIClient:
    """
    Synchronous API client for CryptoSmartTrader V2.
    """

    def __init__(self, base_url: str = API_BASE_URL, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if api_key:
            self.session.headers.update(
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            )

    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request with error handling."""
        response = self.session.get(f"{self.base_url}{endpoint}", **kwargs)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """POST request with error handling."""
        response = self.session.post(f"{self.base_url}{endpoint}", json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """PUT request with error handling."""
        response = self.session.put(f"{self.base_url}{endpoint}", json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """DELETE request with error handling."""
        response = self.session.delete(f"{self.base_url}{endpoint}", **kwargs)
        response.raise_for_status()
        return response.json()


class AsyncAPIClient:
    """
    Asynchronous API client for CryptoSmartTrader V2.
    """

    def __init__(self, base_url: str = API_BASE_URL, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Async GET request with error handling."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{endpoint}", headers=self.headers, **kwargs
            )
            response.raise_for_status()
            return response.json()

    async def post(self, endpoint: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Async POST request with error handling."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{endpoint}", headers=self.headers, json=data, **kwargs
            )
            response.raise_for_status()
            return response.json()


def example_health_checks():
    """Example: Health check endpoints."""
    print("🏥 HEALTH CHECK EXAMPLES")
    print("=" * 50)

    client = APIClient()

    # Basic health check
    print("1. Basic Health Check:")
    health = client.get("/health")
    print(f"   Status: {health['status']}")
    print(f"   Uptime: {health['uptime_seconds']:.1f}s")
    print(f"   Version: {health['version']}")

    # Detailed health check
    print("\n2. Detailed Health Check:")
    detailed_health = client.get("/health/detailed")
    for component, status in detailed_health["components"].items():
        status_emoji = "✅" if status == "healthy" else "❌"
        print(f"   {status_emoji} {component}: {status}")

    # Database health
    print("\n3. Database Health:")
    db_health = client.get("/health/database")
    print(f"   Connection: {db_health['connection_status']}")
    print(f"   Query time: {db_health['query_time_ms']}ms")

    # Exchange connectivity
    print("\n4. Exchange Health:")
    exchange_health = client.get("/health/exchanges")
    for exchange, info in exchange_health["exchanges"].items():
        status_emoji = "✅" if info["status"] == "connected" else "❌"
        print(f"   {status_emoji} {exchange}: {info['status']} ({info['latency_ms']}ms)")


def example_market_data():
    """Example: Market data endpoints."""
    print("\n📊 MARKET DATA EXAMPLES")
    print("=" * 50)

    client = APIClient()

    # Get market overview
    print("1. Market Overview:")
    market_overview = client.get("/api/v1/market/overview")
    print(f"   Total pairs: {market_overview['total_pairs']}")
    print(f"   Active signals: {market_overview['active_signals']}")
    print(f"   Last update: {market_overview['last_update']}")

    # Get specific pair data
    print("\n2. Bitcoin Price Data:")
    btc_data = client.get("/api/v1/market/pair/BTC-USD")
    print(f"   Price: ${btc_data['price']:,.2f}")
    print(f"   24h Change: {btc_data['change_24h']:.2f}%")
    print(f"   Volume: ${btc_data['volume_24h']:,.0f}")

    # Get top performers
    print("\n3. Top Performers:")
    top_performers = client.get("/api/v1/market/top_performers", params={"limit": 5})
    for i, pair in enumerate(top_performers["pairs"], 1):
        print(f"   {i}. {pair['symbol']}: {pair['change_24h']:.2f}%")

    # Get market predictions
    print("\n4. ML Predictions:")
    predictions = client.get("/api/v1/market/predictions")
    for prediction in predictions["predictions"][:3]:
        confidence_emoji = "🔥" if prediction["confidence"] > 0.8 else "📈"
        print(
            f"   {confidence_emoji} {prediction['symbol']}: {prediction['direction']} "
            f"({prediction['confidence']:.1%} confidence)"
        )


def example_portfolio_management():
    """Example: Portfolio management endpoints."""
    print("\n💼 PORTFOLIO MANAGEMENT EXAMPLES")
    print("=" * 50)

    client = APIClient(api_key=API_KEY)  # Requires authentication

    # Get portfolio overview
    print("1. Portfolio Overview:")
    portfolio = client.get("/api/v1/portfolio/overview")
    print(f"   Total Value: ${portfolio['total_value']:,.2f}")
    print(f"   Daily P&L: ${portfolio['daily_pnl']:,.2f} ({portfolio['daily_pnl_pct']:.2f}%)")
    print(f"   Positions: {portfolio['position_count']}")

    # Get current positions
    print("\n2. Current Positions:")
    positions = client.get("/api/v1/portfolio/positions")
    for position in positions["positions"][:5]:
        pnl_emoji = "📈" if position["unrealized_pnl"] > 0 else "📉"
        print(
            f"   {pnl_emoji} {position['symbol']}: {position['quantity']} "
            f"(${position['unrealized_pnl']:,.2f} P&L)"
        )

    # Get performance metrics
    print("\n3. Performance Metrics:")
    performance = client.get("/api/v1/portfolio/performance")
    print(f"   Total Return: {performance['total_return']:.2f}%")
    print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"   Win Rate: {performance['win_rate']:.1%}")

    # Get risk metrics
    print("\n4. Risk Metrics:")
    risk = client.get("/api/v1/portfolio/risk")
    print(f"   Current Drawdown: {risk['current_drawdown']:.2f}%")
    print(f"   VaR (95%): ${risk['var_95']:,.2f}")
    print(f"   Beta: {risk['beta']:.2f}")
    print(f"   Volatility: {risk['volatility']:.2f}%")


def example_trading_operations():
    """Example: Trading operations."""
    print("\n🔄 TRADING OPERATIONS EXAMPLES")
    print("=" * 50)

    client = APIClient(api_key=API_KEY)  # Requires authentication

    # Get trading status
    print("1. Trading Status:")
    trading_status = client.get("/api/v1/trading/status")
    status_emoji = "✅" if trading_status["active"] else "⏸️"
    print(f"   {status_emoji} Trading Active: {trading_status['active']}")
    print(f"   Mode: {trading_status['mode']}")
    print(f"   Risk Level: {trading_status['risk_level']}")

    # Place a demo order (use demo mode to avoid real trading)
    print("\n2. Place Demo Order:")
    demo_order = {
        "symbol": "BTC-USD",
        "side": "buy",
        "type": "limit",
        "quantity": 0.001,
        "price": 45000.00,
        "demo_mode": True,
    }

    order_response = client.post("/api/v1/trading/orders", data=demo_order)
    print(f"   Order ID: {order_response['order_id']}")
    print(f"   Status: {order_response['status']}")
    print(f"   Demo Mode: {order_response['demo_mode']}")

    # Get order history
    print("\n3. Recent Orders:")
    orders = client.get("/api/v1/trading/orders", params={"limit": 5})
    for order in orders["orders"]:
        status_emoji = "✅" if order["status"] == "filled" else "⏳"
        print(
            f"   {status_emoji} {order['symbol']} {order['side']}: "
            f"{order['quantity']} @ ${order['price']:.2f}"
        )

    # Get execution statistics
    print("\n4. Execution Stats:")
    execution_stats = client.get("/api/v1/trading/execution_stats")
    print(f"   Average Slippage: {execution_stats['avg_slippage']:.3f}%")
    print(f"   Fill Rate: {execution_stats['fill_rate']:.1%}")
    print(f"   Average Fill Time: {execution_stats['avg_fill_time']:.1f}s")


def example_agent_management():
    """Example: Multi-agent system management."""
    print("\n🤖 AGENT MANAGEMENT EXAMPLES")
    print("=" * 50)

    client = APIClient(api_key=API_KEY)

    # Get all agent status
    print("1. Agent Status Overview:")
    agents = client.get("/api/v1/agents/status")
    for agent in agents["agents"]:
        status_emoji = "✅" if agent["status"] == "running" else "❌"
        print(
            f"   {status_emoji} {agent['name']}: {agent['status']} "
            f"(uptime: {agent['uptime_seconds']:.0f}s)"
        )

    # Get specific agent details
    print("\n2. Technical Agent Details:")
    tech_agent = client.get("/api/v1/agents/technical_agent")
    print(f"   Signals Generated: {tech_agent['signals_generated']}")
    print(f"   Last Signal: {tech_agent['last_signal_time']}")
    print(f"   Success Rate: {tech_agent['success_rate']:.1%}")

    # Get agent logs
    print("\n3. Recent Agent Logs:")
    logs = client.get("/api/v1/agents/logs", params={"limit": 3})
    for log in logs["logs"]:
        level_emoji = "🔥" if log["level"] == "ERROR" else "ℹ️"
        print(f"   {level_emoji} [{log['agent']}] {log['message']}")

    # Restart an agent (if needed)
    print("\n4. Agent Operations:")
    restart_response = client.post("/api/v1/agents/technical_agent/restart")
    print(f"   Restart Status: {restart_response['status']}")


def example_security_operations():
    """Example: Security and compliance operations."""
    print("\n🔒 SECURITY OPERATIONS EXAMPLES")
    print("=" * 50)

    client = APIClient(api_key=API_KEY)

    # Get security status
    print("1. Security Status:")
    security_status = client.get("/api/v1/security/status")
    print(f"   Encryption: {security_status['encryption_enabled']}")
    print(f"   Audit Logging: {security_status['audit_logging']}")
    print(f"   Secret Rotation: {security_status['secret_rotation_active']}")

    # Check secret rotation status
    print("\n2. Secret Rotation Status:")
    rotation_status = client.get("/api/v1/security/secrets/rotation_status")
    print(f"   Secrets Due Rotation: {rotation_status['due_rotation']}")
    for secret in rotation_status["secrets"][:3]:
        days_emoji = "⚠️" if secret["days_overdue"] > 0 else "✅"
        print(f"   {days_emoji} {secret['name']}: {secret['days_since_rotation']} days")

    # Get audit log summary
    print("\n3. Recent Audit Events:")
    audit_events = client.get("/api/v1/security/audit", params={"limit": 5})
    for event in audit_events["events"]:
        event_emoji = "✅" if event["success"] else "❌"
        print(f"   {event_emoji} {event['event_type']}: {event['user_id']} ({event['timestamp']})")

    # Emergency procedures status
    print("\n4. Emergency Procedures:")
    emergency_status = client.get("/api/v1/security/emergency_status")
    print(f"   Kill Switch: {emergency_status['kill_switch_armed']}")
    print(f"   Emergency Contacts: {emergency_status['emergency_contacts_configured']}")


async def example_async_operations():
    """Example: Asynchronous API operations."""
    print("\n⚡ ASYNC OPERATIONS EXAMPLES")
    print("=" * 50)

    client = AsyncAPIClient(api_key=API_KEY)

    # Concurrent health checks
    print("1. Concurrent Health Checks:")
    health_tasks = [
        client.get("/health"),
        client.get("/health/database"),
        client.get("/health/exchanges"),
    ]

    health_results = await asyncio.gather(*health_tasks)

    print(f"   System: {health_results[0]['status']}")
    print(f"   Database: {health_results[1]['connection_status']}")
    print(f"   Exchanges: {len(health_results[2]['exchanges'])} connected")

    # Batch market data requests
    print("\n2. Batch Market Data:")
    symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    market_tasks = [client.get(f"/api/v1/market/pair/{symbol}") for symbol in symbols]

    market_results = await asyncio.gather(*market_tasks)

    for symbol, data in zip(symbols, market_results):
        change_emoji = "📈" if data["change_24h"] > 0 else "📉"
        print(f"   {change_emoji} {symbol}: ${data['price']:,.2f} ({data['change_24h']:+.2f}%)")


def example_error_handling():
    """Example: Error handling and best practices."""
    print("\n⚠️ ERROR HANDLING EXAMPLES")
    print("=" * 50)

    client = APIClient(api_key="invalid-key")

    # Handle authentication errors
    print("1. Authentication Error Handling:")
    try:
        portfolio = client.get("/api/v1/portfolio/overview")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("   ❌ Authentication failed - check API key")
            error_detail = e.response.json()
            print(f"   Error: {error_detail['message']}")
            print(f"   Request ID: {error_detail['request_id']}")

    # Handle rate limiting
    print("\n2. Rate Limiting Handling:")
    try:
        # Simulate rate limiting with many requests
        for i in range(5):
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", 60)
                print(f"   ⏳ Rate limited - retry after {retry_after}s")
                break
            else:
                print(f"   ✅ Request {i + 1}: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Handle network errors
    print("\n3. Network Error Handling:")
    try:
        # Try to connect to invalid endpoint
        response = requests.get("http://localhost:9999/invalid", timeout=1)
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed - service may be down")
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out - check network connectivity")


def example_monitoring_integration():
    """Example: Monitoring and metrics integration."""
    print("\n📊 MONITORING INTEGRATION EXAMPLES")
    print("=" * 50)

    # Get Prometheus metrics
    print("1. Prometheus Metrics:")
    try:
        metrics_response = requests.get("http://localhost:8000/metrics")
        metrics_lines = metrics_response.text.split("\n")

        # Extract key metrics
        for line in metrics_lines:
            if "trading_signals_received_total" in line and not line.startswith("#"):
                print(f"   📈 Signals Received: {line.split()[-1]}")
            elif "trading_equity_usd" in line and not line.startswith("#"):
                print(f"   💰 Portfolio Value: ${float(line.split()[-1]):,.2f}")
            elif "http_requests_total" in line and not line.startswith("#"):
                print(f"   🌐 API Requests: {line.split()[-1]}")
    except Exception as e:
        print(f"   ❌ Failed to get metrics: {e}")

    # Performance monitoring
    print("\n2. Performance Monitoring:")
    start_time = time.time()

    client = APIClient()
    health = client.get("/health")

    response_time = time.time() - start_time
    print(f"   ⏱️ API Response Time: {response_time:.3f}s")
    print(f"   🔄 System Uptime: {health['uptime_seconds']:.0f}s")


def main():
    """Run all API examples."""
    print("🚀 CRYPTOSMARTTRADER V2 API EXAMPLES")
    print("=" * 70)
    print("Comprehensive examples for REST API integration")
    print("=" * 70)

    try:
        # Basic examples
        example_health_checks()
        example_market_data()

        # Authenticated examples (may fail without valid API key)
        try:
            example_portfolio_management()
            example_trading_operations()
            example_agent_management()
            example_security_operations()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("\n⚠️ Authentication required for trading endpoints")
                print("   Set API_KEY variable to test authenticated endpoints")

        # Advanced examples
        print("\n" + "=" * 50)
        print("Advanced Examples:")

        # Run async examples
        asyncio.run(example_async_operations())

        example_error_handling()
        example_monitoring_integration()

        print("\n✅ All API examples completed successfully!")
        print("\n📚 Additional Resources:")
        print("   - API Documentation: http://localhost:8001/docs")
        print("   - OpenAPI Schema: http://localhost:8001/api/v1/openapi.json")
        print("   - Health Check: http://localhost:8001/health")
        print("   - Metrics: http://localhost:8000/metrics")

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection failed - ensure the API server is running:")
        print("   1. Start services: uv run python start_replit_services.py")
        print("   2. Check health: curl http://localhost:8001/health")
        print("   3. Verify API documentation: http://localhost:8001/docs")

    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("   Check logs and system status for details")


if __name__ == "__main__":
    main()
