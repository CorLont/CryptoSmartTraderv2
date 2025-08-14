#!/usr/bin/env python3
"""
Realistic Execution Simulation
Implement real slippage & latency in backtest
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class RealisticExecutionSimulator:
    """
    Simulate realistic trading execution with slippage and latency
    """

    def __init__(self):
        self.execution_logs = []
        self.slippage_model = SlippageModel()
        self.latency_model = LatencyModel()
        self.execution_stats = {
            "total_orders": 0,
            "successful_executions": 0,
            "partial_fills": 0,
            "failed_orders": 0,
            "total_slippage_bps": 0,
            "total_latency_ms": 0,
        }

    def execute_order(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Any],
        orderbook_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute single order with realistic constraints - HARD WIRED TO GATEWAY"""

        # MANDATORY GATEWAY ENFORCEMENT
        try:
            from src.cryptosmarttrader.core.mandatory_execution_gateway import enforce_mandatory_gateway, UniversalOrderRequest
            
            gateway_order = UniversalOrderRequest(
                symbol=order.get("symbol", "BTC/USD"),
                side=order.get("side", "buy"),
                size=order.get("size", 0.0),
                order_type=order.get("type", "market"),
                limit_price=order.get("target_price"),
                strategy_id=order.get("strategy_id", "realistic_execution"),
                source_module="trading.realistic_execution",
                source_function="execute_order"
            )
            
            gateway_result = enforce_mandatory_gateway(gateway_order, market_data)
            
            if not gateway_result.approved:
                return {
                    "order_id": order.get("id", "unknown"),
                    "timestamp": datetime.now(),
                    "symbol": order.get("symbol", "unknown"),
                    "side": order.get("side", "unknown"),
                    "order_type": order.get("type", "market"),
                    "requested_size": order.get("size", 0.0),
                    "executed_size": 0.0,
                    "fill_ratio": 0.0,
                    "target_price": order.get("target_price", 0.0),
                    "execution_price": 0.0,
                    "slippage_bps": 0.0,
                    "latency_ms": 0.0,
                    "fees": 0.0,
                    "total_cost": 0.0,
                    "execution_quality": 0.0,
                    "market_impact": 0.0,
                    "liquidity_consumed": 0.0,
                    "rejected": True,
                    "rejection_reason": gateway_result.reason
                }
            
            # Use approved size
            order["size"] = gateway_result.approved_size
            
        except Exception as e:
            return {
                "order_id": order.get("id", "unknown"),
                "rejected": True,
                "rejection_reason": f"Gateway error: {str(e)}"
            }

        self.execution_stats["total_orders"] += 1

        # Simulate latency
        latency_ms = self.latency_model.calculate_latency(
            order_type=order.get("type", "market"), market_conditions=market_data
        )

        # Apply latency delay (price may have moved)
        delayed_price = self._apply_latency_impact(order["target_price"], latency_ms, market_data)

        # Calculate slippage
        slippage_result = self.slippage_model.calculate_slippage(
            order_size=order["size"], market_data=market_data, orderbook_data=orderbook_data
        )

        # Determine execution price
        if order["side"] == "buy":
            execution_price = delayed_price * (1 + slippage_result["slippage_bps"] / 10000)
        else:  # sell
            execution_price = delayed_price * (1 - slippage_result["slippage_bps"] / 10000)

        # Simulate partial fills
        fill_result = self._simulate_fill(order, market_data, orderbook_data)

        # Calculate actual executed amount
        executed_size = order["size"] * fill_result["fill_ratio"]
        executed_value = executed_size * execution_price

        # Calculate fees
        fees = self._calculate_fees(executed_value, order.get("type", "market"))

        # Update statistics
        self.execution_stats["total_slippage_bps"] += slippage_result["slippage_bps"]
        self.execution_stats["total_latency_ms"] += latency_ms

        if fill_result["fill_ratio"] >= 0.95:  # Consider >95% as successful
            self.execution_stats["successful_executions"] += 1
        elif fill_result["fill_ratio"] > 0:
            self.execution_stats["partial_fills"] += 1
        else:
            self.execution_stats["failed_orders"] += 1

        # Create execution result
        execution_result = {
            "order_id": order.get("id", f"order_{len(self.execution_logs)}"),
            "timestamp": datetime.now(),
            "symbol": order["symbol"],
            "side": order["side"],
            "order_type": order.get("type", "market"),
            "requested_size": order["size"],
            "executed_size": executed_size,
            "fill_ratio": fill_result["fill_ratio"],
            "target_price": order["target_price"],
            "execution_price": execution_price,
            "slippage_bps": slippage_result["slippage_bps"],
            "latency_ms": latency_ms,
            "fees": fees,
            "total_cost": executed_value + fees,
            "execution_quality": self._calculate_execution_quality(
                slippage_result, latency_ms, fill_result
            ),
            "market_impact": slippage_result.get("market_impact", 0),
            "liquidity_consumed": slippage_result.get("liquidity_consumed", 0),
        }

        self.execution_logs.append(execution_result)

        return execution_result

    def _apply_latency_impact(
        self, original_price: float, latency_ms: float, market_data: Dict[str, Any]
    ) -> float:
        """Apply price movement during latency period"""

        # Estimate price movement during latency
        volatility = market_data.get("volatility", 0.02)  # 2% default volatility
        time_factor = latency_ms / (1000 * 60 * 60)  # Convert to hours

        # Random walk during latency period
        price_change = np.random.normal(0, volatility * np.sqrt(time_factor))

        return original_price * (1 + price_change)

    def _simulate_fill(
        self,
        order: Dict[str, Any],
        market_data: Dict[str, Any],
        orderbook_data: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Simulate order fill based on liquidity"""

        # Base fill probability
        fill_probability = 0.95  # 95% base probability

        # Adjust based on order size
        size_impact = min(order["size"] / market_data.get("avg_volume", 1e6), 0.1)
        fill_probability -= size_impact * 2  # Larger orders harder to fill

        # Adjust based on market conditions
        spread = market_data.get("spread_bps", 10) / 10000
        if spread > 0.005:  # Wide spread
            fill_probability -= 0.1

        # Adjust based on volatility
        volatility = market_data.get("volatility", 0.02)
        if volatility > 0.05:  # High volatility
            fill_probability -= 0.15

        # Simulate partial fills
        if np.random.random() < fill_probability:
            # Full or partial fill
            if np.random.random() < 0.8:  # 80% chance of full fill
                fill_ratio = 1.0
            else:
                fill_ratio = np.random.uniform(0.3, 0.95)  # Partial fill
        else:
            fill_ratio = 0.0  # Failed order

        return {"fill_ratio": fill_ratio, "fill_probability": fill_probability}

    def _calculate_fees(self, executed_value: float, order_type: str) -> float:
        """Calculate trading fees"""

        # Typical crypto exchange fees
        if order_type == "limit":
            fee_rate = 0.001  # 0.1% maker fee
        else:  # market order
            fee_rate = 0.002  # 0.2% taker fee

        return executed_value * fee_rate

    def _calculate_execution_quality(
        self, slippage_result: Dict[str, Any], latency_ms: float, fill_result: Dict[str, float]
    ) -> float:
        """Calculate overall execution quality score (0-1)"""

        # Base score
        quality_score = 1.0

        # Penalize slippage
        slippage_penalty = min(slippage_result["slippage_bps"] / 100, 0.5)  # Max 50% penalty
        quality_score -= slippage_penalty

        # Penalize latency
        latency_penalty = min(latency_ms / 1000, 0.2)  # Max 20% penalty for 1s+ latency
        quality_score -= latency_penalty

        # Penalize partial fills
        fill_penalty = (1 - fill_result["fill_ratio"]) * 0.3  # Max 30% penalty
        quality_score -= fill_penalty

        return max(0, quality_score)

    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""

        if not self.execution_logs:
            return {"error": "No executions recorded"}

        logs_df = pd.DataFrame(self.execution_logs)

        analytics = {
            "execution_summary": {
                "total_orders": self.execution_stats["total_orders"],
                "success_rate": self.execution_stats["successful_executions"]
                / max(1, self.execution_stats["total_orders"]),
                "partial_fill_rate": self.execution_stats["partial_fills"]
                / max(1, self.execution_stats["total_orders"]),
                "failure_rate": self.execution_stats["failed_orders"]
                / max(1, self.execution_stats["total_orders"]),
            },
            "slippage_analysis": {
                "avg_slippage_bps": logs_df["slippage_bps"].mean(),
                "median_slippage_bps": logs_df["slippage_bps"].median(),
                "p90_slippage_bps": logs_df["slippage_bps"].quantile(0.9),
                "p95_slippage_bps": logs_df["slippage_bps"].quantile(0.95),
                "max_slippage_bps": logs_df["slippage_bps"].max(),
            },
            "latency_analysis": {
                "avg_latency_ms": logs_df["latency_ms"].mean(),
                "median_latency_ms": logs_df["latency_ms"].median(),
                "p90_latency_ms": logs_df["latency_ms"].quantile(0.9),
                "p95_latency_ms": logs_df["latency_ms"].quantile(0.95),
                "max_latency_ms": logs_df["latency_ms"].max(),
            },
            "execution_quality": {
                "avg_quality_score": logs_df["execution_quality"].mean(),
                "median_quality_score": logs_df["execution_quality"].median(),
                "high_quality_rate": (logs_df["execution_quality"] > 0.8).mean(),
            },
            "cost_analysis": {
                "total_fees": logs_df["fees"].sum(),
                "avg_fee_bps": (logs_df["fees"] / logs_df["total_cost"] * 10000).mean(),
                "total_slippage_cost": logs_df["slippage_bps"].sum()
                / 10000
                * logs_df["total_cost"].sum(),
            },
        }

        return analytics


class SlippageModel:
    """Model for realistic slippage calculation"""

    def __init__(self):
        self.base_slippage_bps = 5  # Base slippage 5 bps

    def calculate_slippage(
        self,
        order_size: float,
        market_data: Dict[str, Any],
        orderbook_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate realistic slippage"""

        # Base slippage
        slippage_bps = self.base_slippage_bps

        # Size impact
        daily_volume = market_data.get("volume_24h", 1e6)
        size_ratio = order_size / daily_volume
        size_impact = min(size_ratio * 1000, 50)  # Max 50 bps from size
        slippage_bps += size_impact

        # Spread impact
        spread_bps = market_data.get("spread_bps", 10)
        slippage_bps += spread_bps * 0.5  # 50% of spread

        # Volatility impact
        volatility = market_data.get("volatility", 0.02)
        vol_impact = volatility * 200  # 2% vol = 4 bps additional
        slippage_bps += vol_impact

        # Market depth impact (if orderbook available)
        if orderbook_data:
            depth_impact = self._calculate_depth_impact(order_size, orderbook_data)
            slippage_bps += depth_impact

        # Add random component
        random_impact = np.random.uniform(-2, 5)  # Slight positive bias
        slippage_bps += random_impact

        # Cap maximum slippage
        slippage_bps = min(slippage_bps, 200)  # Max 200 bps (2%)

        return {
            "slippage_bps": max(0, slippage_bps),
            "size_impact": size_impact,
            "spread_impact": spread_bps * 0.5,
            "volatility_impact": vol_impact,
            "market_impact": size_impact + vol_impact,
            "liquidity_consumed": size_ratio,
        }

    def _calculate_depth_impact(self, order_size: float, orderbook_data: Dict[str, Any]) -> float:
        """Calculate impact from orderbook depth"""

        # Simplified depth calculation
        depth = orderbook_data.get("depth_10", order_size)

        if order_size > depth:
            # Order larger than available depth
            excess_ratio = (order_size - depth) / order_size
            return excess_ratio * 20  # 20 bps per ratio of excess

        return 0


class LatencyModel:
    """Model for realistic latency simulation"""

    def __init__(self):
        self.base_latency_ms = 50  # 50ms base latency

    def calculate_latency(self, order_type: str, market_conditions: Dict[str, Any]) -> float:
        """Calculate realistic latency"""

        # Base latency
        latency_ms = self.base_latency_ms

        # Order type impact
        if order_type == "market":
            latency_ms += 20  # Market orders slightly slower

        # Market conditions impact
        volatility = market_conditions.get("volatility", 0.02)
        if volatility > 0.05:  # High volatility
            latency_ms += 30  # Systems slower during stress

        # Network jitter
        jitter = np.random.exponential(15)  # Exponential distribution for jitter
        latency_ms += jitter

        # Occasional spikes
        if np.random.random() < 0.05:  # 5% chance of spike
            latency_ms += np.random.uniform(200, 500)

        return latency_ms


def create_realistic_backtest(
    orders: List[Dict[str, Any]], market_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create realistic backtest with execution simulation"""

    simulator = RealisticExecutionSimulator()

    execution_results = []

    for order, market_datum in zip(orders, market_data):
        # Execute order with realistic constraints
        result = simulator.execute_order(order, market_datum)
        execution_results.append(result)

    # Get analytics
    analytics = simulator.get_execution_analytics()

    return {"execution_results": execution_results, "analytics": analytics, "simulator": simulator}


if __name__ == "__main__":
    print("⚡ TESTING REALISTIC EXECUTION SIMULATION")
    print("=" * 50)

    # Create sample orders and market data
    np.random.seed(42)

    sample_orders = []
    sample_market_data = []

    symbols = ["BTC", "ETH", "SOL", "ADA", "MATIC"]

    for i in range(20):
        symbol = np.random.choice(symbols)

        order = {
            "id": f"order_{i}",
            "symbol": symbol,
            "side": np.random.choice(["buy", "sell"]),
            "type": np.random.choice(["market", "limit"]),
            "size": np.random.uniform(0.1, 10),
            "target_price": np.random.uniform(100, 70000),
        }

        market_data = {
            "symbol": symbol,
            "volume_24h": np.random.uniform(1e6, 1e9),
            "spread_bps": np.random.uniform(5, 50),
            "volatility": np.random.uniform(0.01, 0.08),
            "avg_volume": np.random.uniform(1e5, 1e7),
        }

        sample_orders.append(order)
        sample_market_data.append(market_data)

    print(f"Created {len(sample_orders)} sample orders for testing")

    # Run realistic backtest
    backtest_results = create_realistic_backtest(sample_orders, sample_market_data)

    # Display results
    analytics = backtest_results["analytics"]

    print(f"\nExecution Summary:")
    summary = analytics["execution_summary"]
    print(f"   Success rate: {summary['success_rate']:.2%}")
    print(f"   Partial fill rate: {summary['partial_fill_rate']:.2%}")
    print(f"   Failure rate: {summary['failure_rate']:.2%}")

    print(f"\nSlippage Analysis:")
    slippage = analytics["slippage_analysis"]
    print(f"   Average: {slippage['avg_slippage_bps']:.2f} bps")
    print(f"   P90: {slippage['p90_slippage_bps']:.2f} bps")
    print(f"   P95: {slippage['p95_slippage_bps']:.2f} bps")

    print(f"\nLatency Analysis:")
    latency = analytics["latency_analysis"]
    print(f"   Average: {latency['avg_latency_ms']:.1f} ms")
    print(f"   P90: {latency['p90_latency_ms']:.1f} ms")
    print(f"   P95: {latency['p95_latency_ms']:.1f} ms")

    print(f"\nExecution Quality:")
    quality = analytics["execution_quality"]
    print(f"   Average quality: {quality['avg_quality_score']:.3f}")
    print(f"   High quality rate: {quality['high_quality_rate']:.2%}")

    print("✅ Realistic execution simulation testing completed")
