#!/usr/bin/env python3
"""
Slippage Estimator
Calculates p50/p90 slippage estimates for realistic trading evaluation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger
from core.orderbook_simulator import OrderBookSimulator, OrderSide, OrderType, OrderBookSnapshot


@dataclass
class SlippageObservation:
    """Single slippage observation"""

    timestamp: datetime
    symbol: str
    side: OrderSide
    order_size: float
    intended_price: float
    executed_price: float
    slippage_bps: float
    market_impact_bps: float
    latency_ms: float


@dataclass
class SlippageEstimate:
    """Slippage estimate with confidence intervals"""

    symbol: str
    side: OrderSide
    order_size_bucket: str
    p50_slippage_bps: float
    p90_slippage_bps: float
    p95_slippage_bps: float
    mean_slippage_bps: float
    std_slippage_bps: float
    sample_count: int
    confidence_interval_95: Tuple[float, float]
    last_updated: datetime


class SlippageEstimator:
    """Real-time slippage estimation and prediction system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_structured_logger("SlippageEstimator")

        # Configuration
        self.config = {
            "max_observations": 10000,
            "min_observations_for_estimate": 20,
            "size_buckets": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # Order size buckets
            "lookback_hours": 168,  # 1 week
            "update_interval_minutes": 15,
            "outlier_threshold_std": 3.0,
        }

        if config:
            self.config.update(config)

        # Slippage observations by symbol
        self.observations: Dict[str, deque] = {}

        # Current estimates by symbol and side
        self.estimates: Dict[str, Dict[str, SlippageEstimate]] = {}

        # Market data for slippage calculation
        self.orderbook_snapshots: Dict[str, OrderBookSnapshot] = {}

    def record_execution(
        self,
        symbol: str,
        side: OrderSide,
        order_size: float,
        intended_price: float,
        executed_price: float,
        latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record actual execution for slippage calculation"""

        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Calculate slippage
            if intended_price <= 0:
                self.logger.warning(f"Invalid intended price {intended_price} for {symbol}")
                return

            # Slippage calculation (positive = worse than expected)
            if side == OrderSide.BUY:
                slippage_bps = (executed_price - intended_price) / intended_price * 10000
            else:  # SELL
                slippage_bps = (intended_price - executed_price) / intended_price * 10000

            # Estimate market impact component
            market_impact_bps = self._estimate_market_impact(symbol, order_size, side)

            # Create observation
            observation = SlippageObservation(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                order_size=order_size,
                intended_price=intended_price,
                executed_price=executed_price,
                slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                latency_ms=latency_ms,
            )

            # Store observation
            if symbol not in self.observations:
                self.observations[symbol] = deque(maxlen=self.config["max_observations"])

            self.observations[symbol].append(observation)

            self.logger.debug(
                f"Recorded slippage: {symbol} {side.value} {order_size} - {slippage_bps:.2f} bps"
            )

        except Exception as e:
            self.logger.error(f"Failed to record execution: {e}")

    def _estimate_market_impact(self, symbol: str, order_size: float, side: OrderSide) -> float:
        """Estimate market impact component of slippage"""

        try:
            # Simple market impact model based on order size
            # This would typically use order book depth analysis

            if symbol in self.orderbook_snapshots:
                orderbook = self.orderbook_snapshots[symbol]

                # Get relevant side of book
                levels = orderbook.asks if side == OrderSide.BUY else orderbook.bids

                if levels:
                    # Calculate available liquidity
                    total_liquidity = sum(level.volume for level in levels[:10])
                    relative_size = order_size / max(total_liquidity, 0.001)

                    # Simple square root impact model
                    impact_bps = 2.5 * np.sqrt(relative_size) * 10000

                    return min(impact_bps, 100.0)  # Cap at 100 bps

            # Fallback estimation based on order size
            if order_size < 0.1:
                return 0.5  # 0.5 bps for small orders
            elif order_size < 1.0:
                return 2.0  # 2 bps for medium orders
            else:
                return 5.0 * np.sqrt(order_size)  # Increasing for large orders

        except Exception as e:
            self.logger.error(f"Market impact estimation failed: {e}")
            return 1.0  # Default 1 bps

    def update_orderbook(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """Update order book snapshot for impact calculation"""
        self.orderbook_snapshots[symbol] = snapshot

    def calculate_slippage_estimates(self, symbol: str) -> Dict[str, SlippageEstimate]:
        """Calculate p50/p90 slippage estimates for symbol"""

        if symbol not in self.observations:
            return {}

        try:
            # Get recent observations
            cutoff_time = datetime.now() - timedelta(hours=self.config["lookback_hours"])
            recent_obs = [obs for obs in self.observations[symbol] if obs.timestamp > cutoff_time]

            if len(recent_obs) < self.config["min_observations_for_estimate"]:
                self.logger.debug(f"Insufficient observations for {symbol}: {len(recent_obs)}")
                return {}

            # Remove outliers
            filtered_obs = self._remove_outliers(recent_obs)

            # Group by side and size bucket
            estimates = {}

            for side in [OrderSide.BUY, OrderSide.SELL]:
                side_obs = [obs for obs in filtered_obs if obs.side == side]

                if len(side_obs) < self.config["min_observations_for_estimate"]:
                    continue

                # Calculate estimates by size bucket
                for i, bucket_size in enumerate(self.config["size_buckets"]):
                    # Define bucket range
                    min_size = self.config["size_buckets"][i - 1] if i > 0 else 0
                    max_size = bucket_size

                    bucket_obs = [obs for obs in side_obs if min_size < obs.order_size <= max_size]

                    if len(bucket_obs) < 5:  # Minimum for meaningful estimate
                        continue

                    # Calculate statistics
                    slippages = [obs.slippage_bps for obs in bucket_obs]

                    estimate = SlippageEstimate(
                        symbol=symbol,
                        side=side,
                        order_size_bucket=f"{min_size:.2f}-{max_size:.2f}",
                        p50_slippage_bps=float(np.percentile(slippages, 50)),
                        p90_slippage_bps=float(np.percentile(slippages, 90)),
                        p95_slippage_bps=float(np.percentile(slippages, 95)),
                        mean_slippage_bps=float(np.mean(slippages)),
                        std_slippage_bps=float(np.std(slippages)),
                        sample_count=len(slippages),
                        confidence_interval_95=self._calculate_confidence_interval(slippages),
                        last_updated=datetime.now(),
                    )

                    key = f"{side.value}_{min_size}_{max_size}"
                    estimates[key] = estimate

            # Store estimates
            self.estimates[symbol] = estimates

            self.logger.info(f"Updated slippage estimates for {symbol}: {len(estimates)} buckets")

            return estimates

        except Exception as e:
            self.logger.error(f"Slippage estimation failed for {symbol}: {e}")
            return {}

    def _remove_outliers(
        self, observations: List[SlippageObservation]
    ) -> List[SlippageObservation]:
        """Remove statistical outliers from observations"""

        try:
            if len(observations) < 10:
                return observations

            slippages = [obs.slippage_bps for obs in observations]
            mean_slippage = np.mean(slippages)
            std_slippage = np.std(slippages)

            threshold = self.config["outlier_threshold_std"] * std_slippage

            filtered = [
                obs for obs in observations if abs(obs.slippage_bps - mean_slippage) <= threshold
            ]

            removed_count = len(observations) - len(filtered)
            if removed_count > 0:
                self.logger.debug(f"Removed {removed_count} outlier observations")

            return filtered

        except Exception as e:
            self.logger.error(f"Outlier removal failed: {e}")
            return observations

    def _calculate_confidence_interval(
        self, values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for values"""

        try:
            if len(values) < 3:
                return (float(np.min(values)), float(np.max(values)))

            mean_val = np.mean(values)
            std_val = np.std(values)
            n = len(values)

            # Use t-distribution for small samples
            if n < 30:
                # Approximate t-value for 95% confidence
                t_value = 2.0 if n > 10 else 2.5
            else:
                t_value = 1.96  # z-value for 95% confidence

            margin_error = t_value * std_val / np.sqrt(n)

            return (float(mean_val - margin_error), float(mean_val + margin_error))

        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)

    def predict_slippage(
        self, symbol: str, side: OrderSide, order_size: float, percentile: int = 90
    ) -> Optional[float]:
        """Predict slippage for given order"""

        if symbol not in self.estimates:
            self.calculate_slippage_estimates(symbol)

        if symbol not in self.estimates:
            self.logger.warning(f"No slippage estimates available for {symbol}")
            return None

        try:
            # Find appropriate size bucket
            bucket_key = None
            for i, bucket_size in enumerate(self.config["size_buckets"]):
                min_size = self.config["size_buckets"][i - 1] if i > 0 else 0

                if min_size < order_size <= bucket_size:
                    bucket_key = f"{side.value}_{min_size}_{bucket_size}"
                    break

            if bucket_key and bucket_key in self.estimates[symbol]:
                estimate = self.estimates[symbol][bucket_key]

                if percentile == 50:
                    return estimate.p50_slippage_bps
                elif percentile == 90:
                    return estimate.p90_slippage_bps
                elif percentile == 95:
                    return estimate.p95_slippage_bps
                else:
                    # Linear interpolation for other percentiles
                    if percentile < 50:
                        return estimate.p50_slippage_bps * (percentile / 50)
                    else:
                        ratio = (percentile - 50) / 40  # 50 to 90
                        return estimate.p50_slippage_bps + ratio * (
                            estimate.p90_slippage_bps - estimate.p50_slippage_bps
                        )

            # Fallback: interpolate from available buckets
            return self._interpolate_slippage_estimate(symbol, side, order_size, percentile)

        except Exception as e:
            self.logger.error(f"Slippage prediction failed: {e}")
            return None

    def _interpolate_slippage_estimate(
        self, symbol: str, side: OrderSide, order_size: float, percentile: int
    ) -> Optional[float]:
        """Interpolate slippage estimate from available data"""

        try:
            # Get all estimates for this side
            side_estimates = {
                k: v for k, v in self.estimates[symbol].items() if k.startswith(side.value)
            }

            if not side_estimates:
                return None

            # Find closest size buckets
            bucket_sizes = []
            bucket_slippages = []

            for key, estimate in side_estimates.items():
                parts = key.split("_")
                max_size = float(parts[2])
                bucket_sizes.append(max_size)

                if percentile == 50:
                    bucket_slippages.append(estimate.p50_slippage_bps)
                elif percentile == 90:
                    bucket_slippages.append(estimate.p90_slippage_bps)
                else:
                    bucket_slippages.append(estimate.p95_slippage_bps)

            if len(bucket_sizes) == 1:
                return bucket_slippages[0]

            # Linear interpolation
            bucket_sizes = np.array(bucket_sizes)
            bucket_slippages = np.array(bucket_slippages)

            # Sort by size
            sort_idx = np.argsort(bucket_sizes)
            bucket_sizes = bucket_sizes[sort_idx]
            bucket_slippages = bucket_slippages[sort_idx]

            # Interpolate
            interpolated = np.interp(order_size, bucket_sizes, bucket_slippages)

            return float(interpolated)

        except Exception as e:
            self.logger.error(f"Slippage interpolation failed: {e}")
            return None

    def get_slippage_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive slippage summary for symbol"""

        if symbol not in self.estimates:
            self.calculate_slippage_estimates(symbol)

        summary = {
            "symbol": symbol,
            "last_updated": datetime.now().isoformat(),
            "total_observations": len(self.observations.get(symbol, [])),
            "estimates_available": len(self.estimates.get(symbol, {})),
        }

        if symbol in self.estimates:
            estimates = self.estimates[symbol]

            # Aggregate statistics
            all_p50 = [est.p50_slippage_bps for est in estimates.values()]
            all_p90 = [est.p90_slippage_bps for est in estimates.values()]

            if all_p50:
                summary.update(
                    {
                        "overall_p50_range": [float(np.min(all_p50)), float(np.max(all_p50))],
                        "overall_p90_range": [float(np.min(all_p90)), float(np.max(all_p90))],
                        "average_p50": float(np.mean(all_p50)),
                        "average_p90": float(np.mean(all_p90)),
                    }
                )

            # Detailed estimates by bucket
            summary["detailed_estimates"] = {}
            for key, estimate in estimates.items():
                summary["detailed_estimates"][key] = {
                    "order_size_bucket": estimate.order_size_bucket,
                    "p50_slippage_bps": estimate.p50_slippage_bps,
                    "p90_slippage_bps": estimate.p90_slippage_bps,
                    "p95_slippage_bps": estimate.p95_slippage_bps,
                    "sample_count": estimate.sample_count,
                }

        return summary


if __name__ == "__main__":

    async def test_slippage_estimator():
        """Test slippage estimator"""

        print("🔍 TESTING SLIPPAGE ESTIMATOR")
        print("=" * 60)

        # Create estimator
        estimator = SlippageEstimator()

        print("📊 Generating simulated executions...")

        # Simulate realistic execution data
        symbol = "BTC/USD"
        np.random.seed(42)  # For reproducible results

        for i in range(200):
            # Simulate different order sizes and sides
            order_size = np.random.choice([0.05, 0.1, 0.5, 1.0, 2.0])
            side = np.random.choice([OrderSide.BUY, OrderSide.SELL])

            # Base price
            intended_price = 50000.0 + np.random.normal(0, 500)

            # Simulate slippage (larger orders have more slippage)
            base_slippage_bps = 1.0 + order_size * 2.0
            slippage_bps = np.random.exponential(base_slippage_bps)

            # Calculate executed price
            if side == OrderSide.BUY:
                executed_price = intended_price * (1 + slippage_bps / 10000)
            else:
                executed_price = intended_price * (1 - slippage_bps / 10000)

            # Random latency
            latency_ms = np.random.exponential(30.0)

            estimator.record_execution(
                symbol=symbol,
                side=side,
                order_size=order_size,
                intended_price=intended_price,
                executed_price=executed_price,
                latency_ms=latency_ms,
            )

        print(f"   Generated 200 execution records")

        print("\n📈 Calculating slippage estimates...")

        estimates = estimator.calculate_slippage_estimates(symbol)
        print(f"   Generated {len(estimates)} slippage estimates")

        # Test slippage predictions
        print("\n🔮 Testing slippage predictions...")

        test_cases = [
            (OrderSide.BUY, 0.1, 50),
            (OrderSide.BUY, 0.1, 90),
            (OrderSide.BUY, 1.0, 90),
            (OrderSide.SELL, 0.5, 90),
            (OrderSide.SELL, 2.0, 95),
        ]

        for side, size, percentile in test_cases:
            predicted = estimator.predict_slippage(symbol, side, size, percentile)
            if predicted is not None:
                print(f"   {side.value} {size} BTC p{percentile}: {predicted:.2f} bps")
            else:
                print(f"   {side.value} {size} BTC p{percentile}: No estimate available")

        # Get comprehensive summary
        print("\n📋 Slippage summary:")
        summary = estimator.get_slippage_summary(symbol)

        print(f"   Total observations: {summary['total_observations']}")
        print(f"   Estimates available: {summary['estimates_available']}")

        if "average_p50" in summary:
            print(f"   Average p50 slippage: {summary['average_p50']:.2f} bps")
            print(f"   Average p90 slippage: {summary['average_p90']:.2f} bps")

        print("\n📊 Detailed estimates by size bucket:")
        for key, estimate in summary.get("detailed_estimates", {}).items():
            print(
                f"   {key}: p50={estimate['p50_slippage_bps']:.2f}, "
                f"p90={estimate['p90_slippage_bps']:.2f} bps "
                f"(n={estimate['sample_count']})"
            )

        print("\n✅ SLIPPAGE ESTIMATOR TEST COMPLETED")

        return len(estimates) > 0 and summary["total_observations"] > 0

    # Run test
    import asyncio

    success = asyncio.run(test_slippage_estimator())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
