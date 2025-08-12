"""
TWAP (Time-Weighted Average Price) Executor

Implements sophisticated TWAP strategies for executing large orders
with minimal market impact in thin or volatile markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class TWAPStrategy(Enum):
    """TWAP execution strategies"""
    LINEAR = "linear"                # Equal time intervals
    VOLUME_WEIGHTED = "volume_weighted"  # Based on historical volume patterns
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Adjust for volatility
    ADAPTIVE = "adaptive"            # Dynamic strategy switching


class SliceStatus(Enum):
    """Status of TWAP slice execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TWAPConfig:
    """Configuration for TWAP execution"""
    total_quantity: float           # Total order size
    duration_minutes: int           # Total execution duration
    slice_count: int               # Number of slices
    strategy: TWAPStrategy = TWAPStrategy.ADAPTIVE
    
    # Timing parameters
    randomize_timing: bool = True   # Add randomness to slice timing
    max_time_variance_pct: float = 0.2  # Max timing randomness (20%)
    
    # Size parameters  
    randomize_sizes: bool = True    # Add randomness to slice sizes
    max_size_variance_pct: float = 0.15  # Max size randomness (15%)
    
    # Risk management
    max_slice_participation_rate: float = 0.1  # Max 10% of recent volume per slice
    max_spread_tolerance_bp: int = 30   # Cancel slice if spread too wide
    
    # Adaptive parameters
    volume_lookback_minutes: int = 60   # Volume pattern analysis window
    volatility_threshold: float = 0.03  # High volatility threshold


@dataclass
class TWAPSlice:
    """Individual TWAP slice"""
    slice_id: str
    target_quantity: float
    target_time: datetime
    actual_quantity: float = 0.0
    actual_price: float = 0.0
    status: SliceStatus = SliceStatus.PENDING
    
    # Execution details
    orders_placed: List[Dict[str, Any]] = None
    total_fees: float = 0.0
    slippage_bp: float = 0.0
    execution_time_ms: int = 0
    
    def __post_init__(self):
        if self.orders_placed is None:
            self.orders_placed = []


@dataclass
class TWAPResult:
    """TWAP execution result"""
    twap_id: str
    total_filled: float
    average_price: float
    total_fees: float
    execution_time_minutes: float
    
    # Performance metrics
    volume_weighted_price: float    # VWAP during execution period
    implementation_shortfall_bp: float  # Cost vs arrival price
    slices_completed: int
    slices_failed: int
    
    # Detailed breakdown
    slices: List[TWAPSlice]
    market_impact_bp: float
    timing_alpha_bp: float          # Alpha from timing vs naive execution


class TWAPExecutor:
    """
    Advanced TWAP executor with adaptive strategies
    """
    
    def __init__(self):
        self.active_twaps = {}      # twap_id -> TWAPExecution
        self.execution_history = []
        self.market_data_cache = {}
        
        # Performance tracking
        self.strategy_performance = {}  # strategy -> performance metrics
        
    async def execute_twap(self, 
                          pair: str,
                          side: str,
                          config: TWAPConfig,
                          market_data_provider: Any) -> TWAPResult:
        """
        Execute TWAP order with specified configuration
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            config: TWAP configuration
            market_data_provider: Provider for real-time market data
            
        Returns:
            TWAP execution result
        """
        try:
            twap_id = f"{pair}_{side}_{datetime.now().timestamp()}"
            start_time = datetime.now()
            
            logger.info(f"Starting TWAP execution: {twap_id}")
            logger.info(f"Total: {config.total_quantity}, Duration: {config.duration_minutes}min, Slices: {config.slice_count}")
            
            # Generate execution plan
            slices = self._generate_execution_plan(config, market_data_provider)
            
            # Execute slices
            completed_slices = []
            failed_slices = []
            
            for slice_obj in slices:
                try:
                    # Wait for slice target time
                    await self._wait_for_slice_time(slice_obj.target_time)
                    
                    # Execute slice
                    result = await self._execute_slice(
                        pair, side, slice_obj, market_data_provider
                    )
                    
                    if result["success"]:
                        slice_obj.status = SliceStatus.COMPLETED
                        slice_obj.actual_quantity = result["filled_quantity"]
                        slice_obj.actual_price = result["average_price"]
                        slice_obj.total_fees = result["fees"]
                        slice_obj.slippage_bp = result["slippage_bp"]
                        slice_obj.execution_time_ms = result["execution_time_ms"]
                        
                        completed_slices.append(slice_obj)
                    else:
                        slice_obj.status = SliceStatus.FAILED
                        failed_slices.append(slice_obj)
                        
                        # Decide whether to continue or abort
                        if not self._should_continue_after_failure(failed_slices, completed_slices):
                            logger.warning(f"TWAP {twap_id} aborted after slice failures")
                            break
                    
                except Exception as e:
                    logger.error(f"Slice execution failed: {e}")
                    slice_obj.status = SliceStatus.FAILED
                    failed_slices.append(slice_obj)
            
            # Calculate results
            result = self._calculate_twap_result(
                twap_id, completed_slices, failed_slices, start_time
            )
            
            # Record for analysis
            self._record_twap_execution(pair, side, config, result)
            
            logger.info(f"TWAP {twap_id} completed: {result.total_filled}/{config.total_quantity} filled")
            
            return result
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            raise
    
    def _generate_execution_plan(self, 
                               config: TWAPConfig,
                               market_data_provider: Any) -> List[TWAPSlice]:
        """Generate TWAP execution plan"""
        try:
            slices = []
            
            # Get historical data for planning
            volume_data = self._get_volume_pattern_data(market_data_provider)
            volatility_data = self._get_volatility_data(market_data_provider)
            
            # Determine slice timing based on strategy
            if config.strategy == TWAPStrategy.LINEAR:
                slice_times = self._generate_linear_timing(config)
                slice_sizes = self._generate_equal_sizes(config)
            elif config.strategy == TWAPStrategy.VOLUME_WEIGHTED:
                slice_times = self._generate_volume_weighted_timing(config, volume_data)
                slice_sizes = self._generate_volume_weighted_sizes(config, volume_data)
            elif config.strategy == TWAPStrategy.VOLATILITY_ADJUSTED:
                slice_times = self._generate_volatility_adjusted_timing(config, volatility_data)
                slice_sizes = self._generate_volatility_adjusted_sizes(config, volatility_data)
            else:  # ADAPTIVE
                slice_times, slice_sizes = self._generate_adaptive_plan(
                    config, volume_data, volatility_data
                )
            
            # Apply randomization if enabled
            if config.randomize_timing:
                slice_times = self._randomize_timing(slice_times, config.max_time_variance_pct)
            
            if config.randomize_sizes:
                slice_sizes = self._randomize_sizes(slice_sizes, config.max_size_variance_pct)
            
            # Create slice objects
            for i, (timing, size) in enumerate(zip(slice_times, slice_sizes)):
                slice_obj = TWAPSlice(
                    slice_id=f"slice_{i+1}",
                    target_quantity=size,
                    target_time=timing
                )
                slices.append(slice_obj)
            
            return slices
            
        except Exception as e:
            logger.error(f"TWAP plan generation failed: {e}")
            # Fallback to linear plan
            return self._generate_linear_plan(config)
    
    def _generate_linear_timing(self, config: TWAPConfig) -> List[datetime]:
        """Generate linear timing intervals"""
        start_time = datetime.now()
        interval_minutes = config.duration_minutes / config.slice_count
        
        times = []
        for i in range(config.slice_count):
            slice_time = start_time + timedelta(minutes=i * interval_minutes)
            times.append(slice_time)
        
        return times
    
    def _generate_equal_sizes(self, config: TWAPConfig) -> List[float]:
        """Generate equal slice sizes"""
        slice_size = config.total_quantity / config.slice_count
        return [slice_size] * config.slice_count
    
    def _generate_volume_weighted_timing(self, 
                                       config: TWAPConfig,
                                       volume_data: Dict[str, Any]) -> List[datetime]:
        """Generate timing based on historical volume patterns"""
        try:
            # Get hourly volume pattern
            hourly_volumes = volume_data.get("hourly_pattern", {})
            
            if not hourly_volumes:
                return self._generate_linear_timing(config)
            
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=config.duration_minutes)
            
            # Calculate volume weights for execution period
            period_weights = []
            current_time = start_time
            
            while current_time < end_time:
                hour = current_time.hour
                weight = hourly_volumes.get(hour, 1.0)
                period_weights.append(weight)
                current_time += timedelta(minutes=config.duration_minutes / config.slice_count)
            
            # Normalize weights and convert to timing
            total_weight = sum(period_weights)
            cumulative_weight = 0
            
            times = []
            for i, weight in enumerate(period_weights):
                cumulative_weight += weight
                time_fraction = cumulative_weight / total_weight
                slice_time = start_time + timedelta(minutes=config.duration_minutes * time_fraction)
                times.append(slice_time)
            
            return times[:config.slice_count]
            
        except Exception as e:
            logger.error(f"Volume weighted timing generation failed: {e}")
            return self._generate_linear_timing(config)
    
    def _generate_volume_weighted_sizes(self, 
                                      config: TWAPConfig,
                                      volume_data: Dict[str, Any]) -> List[float]:
        """Generate sizes based on expected volume"""
        try:
            hourly_volumes = volume_data.get("hourly_pattern", {})
            
            if not hourly_volumes:
                return self._generate_equal_sizes(config)
            
            # Calculate expected volume for each slice time
            slice_weights = []
            start_time = datetime.now()
            
            for i in range(config.slice_count):
                slice_time = start_time + timedelta(minutes=i * config.duration_minutes / config.slice_count)
                hour = slice_time.hour
                weight = hourly_volumes.get(hour, 1.0)
                slice_weights.append(weight)
            
            # Normalize to total quantity
            total_weight = sum(slice_weights)
            sizes = [(weight / total_weight) * config.total_quantity for weight in slice_weights]
            
            return sizes
            
        except Exception as e:
            logger.error(f"Volume weighted size generation failed: {e}")
            return self._generate_equal_sizes(config)
    
    def _generate_adaptive_plan(self, 
                              config: TWAPConfig,
                              volume_data: Dict[str, Any],
                              volatility_data: Dict[str, Any]) -> Tuple[List[datetime], List[float]]:
        """Generate adaptive execution plan"""
        try:
            current_volatility = volatility_data.get("current_1h", 0.02)
            
            # Choose base strategy based on market conditions
            if current_volatility > config.volatility_threshold:
                # High volatility - use more conservative approach
                base_times = self._generate_linear_timing(config)
                base_sizes = self._generate_equal_sizes(config)
                
                # Smaller slices in volatile markets
                adjusted_sizes = [size * 0.8 for size in base_sizes]
                # Add extra slice for remainder
                remainder = config.total_quantity - sum(adjusted_sizes)
                if remainder > 0:
                    adjusted_sizes.append(remainder)
                    extra_time = base_times[-1] + timedelta(minutes=config.duration_minutes / config.slice_count)
                    base_times.append(extra_time)
                
                return base_times, adjusted_sizes
            else:
                # Normal volatility - use volume weighting
                return (
                    self._generate_volume_weighted_timing(config, volume_data),
                    self._generate_volume_weighted_sizes(config, volume_data)
                )
            
        except Exception as e:
            logger.error(f"Adaptive plan generation failed: {e}")
            return (
                self._generate_linear_timing(config),
                self._generate_equal_sizes(config)
            )
    
    def _randomize_timing(self, times: List[datetime], variance_pct: float) -> List[datetime]:
        """Add randomness to slice timing"""
        try:
            randomized_times = []
            
            for i, target_time in enumerate(times):
                if i == 0:
                    # Don't randomize first slice
                    randomized_times.append(target_time)
                    continue
                
                # Calculate max variance in minutes
                prev_time = randomized_times[i-1]
                next_time = times[i+1] if i+1 < len(times) else target_time + timedelta(minutes=10)
                
                min_interval = (target_time - prev_time).total_seconds() / 60 * 0.3  # Min 30% of interval
                max_interval = (next_time - target_time).total_seconds() / 60 * 0.7  # Max 70% of next interval
                
                max_variance_minutes = min(min_interval, max_interval) * variance_pct
                
                # Apply random variance
                variance_minutes = np.random.uniform(-max_variance_minutes, max_variance_minutes)
                randomized_time = target_time + timedelta(minutes=variance_minutes)
                
                # Ensure ordering
                randomized_time = max(randomized_time, prev_time + timedelta(minutes=1))
                
                randomized_times.append(randomized_time)
            
            return randomized_times
            
        except Exception as e:
            logger.error(f"Timing randomization failed: {e}")
            return times
    
    def _randomize_sizes(self, sizes: List[float], variance_pct: float) -> List[float]:
        """Add randomness to slice sizes while maintaining total"""
        try:
            total_quantity = sum(sizes)
            randomized_sizes = []
            
            for size in sizes[:-1]:  # Don't randomize last slice
                variance = size * variance_pct * np.random.uniform(-1, 1)
                randomized_size = size + variance
                randomized_size = max(randomized_size, size * 0.5)  # Min 50% of target
                randomized_sizes.append(randomized_size)
            
            # Last slice gets remainder
            remainder = total_quantity - sum(randomized_sizes)
            randomized_sizes.append(max(remainder, 0))
            
            return randomized_sizes
            
        except Exception as e:
            logger.error(f"Size randomization failed: {e}")
            return sizes
    
    async def _wait_for_slice_time(self, target_time: datetime) -> None:
        """Wait until target slice execution time"""
        try:
            now = datetime.now()
            if target_time > now:
                wait_seconds = (target_time - now).total_seconds()
                await asyncio.sleep(wait_seconds)
        except Exception as e:
            logger.error(f"Failed to wait for slice time: {e}")
    
    async def _execute_slice(self, 
                           pair: str,
                           side: str,
                           slice_obj: TWAPSlice,
                           market_data_provider: Any) -> Dict[str, Any]:
        """Execute individual TWAP slice"""
        try:
            slice_obj.status = SliceStatus.EXECUTING
            start_time = datetime.now()
            
            # Get current market data
            market_data = await self._get_current_market_data(pair, market_data_provider)
            
            # Check execution conditions
            if not self._check_slice_conditions(slice_obj, market_data):
                return {
                    "success": False,
                    "reason": "Slice execution conditions not met"
                }
            
            # Determine optimal execution approach for slice
            execution_strategy = self._determine_slice_strategy(slice_obj, market_data)
            
            # Execute the slice
            if execution_strategy == "post_only":
                result = await self._execute_slice_post_only(pair, side, slice_obj, market_data)
            elif execution_strategy == "aggressive":
                result = await self._execute_slice_aggressive(pair, side, slice_obj, market_data)
            else:  # iceberg
                result = await self._execute_slice_iceberg(pair, side, slice_obj, market_data)
            
            # Calculate metrics
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            result["execution_time_ms"] = execution_time_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Slice execution failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def _get_volume_pattern_data(self, market_data_provider: Any) -> Dict[str, Any]:
        """Get historical volume patterns"""
        try:
            # This would integrate with the market data provider
            # For now, return mock data
            hourly_pattern = {
                hour: 1.0 + 0.3 * np.sin(hour * np.pi / 12)  # Simple pattern
                for hour in range(24)
            }
            
            return {
                "hourly_pattern": hourly_pattern,
                "current_volume_1h": 1000000  # Mock current volume
            }
            
        except Exception as e:
            logger.error(f"Failed to get volume pattern data: {e}")
            return {}
    
    def _get_volatility_data(self, market_data_provider: Any) -> Dict[str, Any]:
        """Get volatility data"""
        try:
            # Mock volatility data
            return {
                "current_1h": 0.025,  # 2.5% hourly volatility
                "avg_24h": 0.02,
                "trend": "increasing"
            }
            
        except Exception as e:
            logger.error(f"Failed to get volatility data: {e}")
            return {"current_1h": 0.02}
    
    async def _get_current_market_data(self, pair: str, market_data_provider: Any) -> Dict[str, Any]:
        """Get current market data"""
        try:
            # Mock market data - would integrate with real provider
            return {
                "bid": 45000,
                "ask": 45010,
                "spread_bp": 2.2,
                "depth_quote": 150000,
                "volume_1h": 2000000
            }
            
        except Exception as e:
            logger.error(f"Failed to get current market data: {e}")
            return {}
    
    def _check_slice_conditions(self, slice_obj: TWAPSlice, market_data: Dict[str, Any]) -> bool:
        """Check if conditions are suitable for slice execution"""
        try:
            spread_bp = market_data.get("spread_bp", 0)
            
            # Check spread tolerance
            if spread_bp > 50:  # Very wide spread
                logger.warning(f"Slice {slice_obj.slice_id}: spread too wide ({spread_bp}bp)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Slice condition check failed: {e}")
            return False
    
    def _determine_slice_strategy(self, slice_obj: TWAPSlice, market_data: Dict[str, Any]) -> str:
        """Determine execution strategy for slice"""
        try:
            spread_bp = market_data.get("spread_bp", 0)
            depth = market_data.get("depth_quote", 0)
            
            # Strategy selection logic
            if spread_bp < 10 and depth > slice_obj.target_quantity * 10:
                return "post_only"  # Good conditions for maker orders
            elif depth < slice_obj.target_quantity * 3:
                return "iceberg"    # Use iceberg for thin books
            else:
                return "aggressive" # Take liquidity
            
        except Exception as e:
            logger.error(f"Slice strategy determination failed: {e}")
            return "post_only"  # Conservative default
    
    async def _execute_slice_post_only(self, pair: str, side: str, slice_obj: TWAPSlice, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute slice using post-only orders"""
        # Mock implementation - would integrate with exchange API
        return {
            "success": True,
            "filled_quantity": slice_obj.target_quantity,
            "average_price": market_data.get("bid" if side == "sell" else "ask", 45000),
            "fees": slice_obj.target_quantity * 45000 * 0.0016,  # 16bp maker fee
            "slippage_bp": 0,  # No slippage for maker orders
            "orders": [{"type": "limit", "status": "filled"}]
        }
    
    async def _execute_slice_aggressive(self, pair: str, side: str, slice_obj: TWAPSlice, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute slice aggressively (market orders)"""
        # Mock implementation
        slippage_bp = np.random.uniform(2, 8)  # Random slippage
        
        return {
            "success": True,
            "filled_quantity": slice_obj.target_quantity,
            "average_price": market_data.get("ask" if side == "buy" else "bid", 45000),
            "fees": slice_obj.target_quantity * 45000 * 0.0026,  # 26bp taker fee
            "slippage_bp": slippage_bp,
            "orders": [{"type": "market", "status": "filled"}]
        }
    
    async def _execute_slice_iceberg(self, pair: str, side: str, slice_obj: TWAPSlice, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute slice using iceberg orders"""
        # Mock implementation
        return {
            "success": True,
            "filled_quantity": slice_obj.target_quantity,
            "average_price": market_data.get("bid" if side == "sell" else "ask", 45000),
            "fees": slice_obj.target_quantity * 45000 * 0.0020,  # Mixed fees
            "slippage_bp": np.random.uniform(1, 4),
            "orders": [{"type": "iceberg", "status": "filled"}]
        }
    
    def _should_continue_after_failure(self, failed_slices: List[TWAPSlice], completed_slices: List[TWAPSlice]) -> bool:
        """Decide whether to continue TWAP after slice failure"""
        try:
            failure_rate = len(failed_slices) / (len(failed_slices) + len(completed_slices))
            
            # Stop if failure rate is too high
            if failure_rate > 0.3:  # 30% failure threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Continue decision failed: {e}")
            return False
    
    def _calculate_twap_result(self, 
                             twap_id: str,
                             completed_slices: List[TWAPSlice],
                             failed_slices: List[TWAPSlice],
                             start_time: datetime) -> TWAPResult:
        """Calculate final TWAP result"""
        try:
            if not completed_slices:
                # No successful slices
                return TWAPResult(
                    twap_id=twap_id,
                    total_filled=0,
                    average_price=0,
                    total_fees=0,
                    execution_time_minutes=0,
                    volume_weighted_price=0,
                    implementation_shortfall_bp=0,
                    slices_completed=0,
                    slices_failed=len(failed_slices),
                    slices=completed_slices + failed_slices,
                    market_impact_bp=0,
                    timing_alpha_bp=0
                )
            
            # Calculate metrics
            total_filled = sum(slice_obj.actual_quantity for slice_obj in completed_slices)
            total_value = sum(slice_obj.actual_quantity * slice_obj.actual_price for slice_obj in completed_slices)
            average_price = total_value / total_filled if total_filled > 0 else 0
            
            total_fees = sum(slice_obj.total_fees for slice_obj in completed_slices)
            
            execution_time_minutes = (datetime.now() - start_time).total_seconds() / 60
            
            # Calculate VWAP (simplified)
            vwap = average_price  # Would calculate based on market volume during execution
            
            # Implementation shortfall (simplified)
            arrival_price = completed_slices[0].actual_price if completed_slices else 0
            shortfall_bp = abs(average_price - arrival_price) / arrival_price * 10000 if arrival_price > 0 else 0
            
            # Market impact estimate
            avg_slippage = np.mean([slice_obj.slippage_bp for slice_obj in completed_slices])
            market_impact_bp = avg_slippage * 0.7  # Portion attributed to market impact
            
            # Timing alpha (difference vs naive equal timing)
            timing_alpha_bp = 0  # Would calculate based on price movements during execution
            
            return TWAPResult(
                twap_id=twap_id,
                total_filled=total_filled,
                average_price=average_price,
                total_fees=total_fees,
                execution_time_minutes=execution_time_minutes,
                volume_weighted_price=vwap,
                implementation_shortfall_bp=shortfall_bp,
                slices_completed=len(completed_slices),
                slices_failed=len(failed_slices),
                slices=completed_slices + failed_slices,
                market_impact_bp=market_impact_bp,
                timing_alpha_bp=timing_alpha_bp
            )
            
        except Exception as e:
            logger.error(f"TWAP result calculation failed: {e}")
            raise
    
    def _record_twap_execution(self, pair: str, side: str, config: TWAPConfig, result: TWAPResult) -> None:
        """Record TWAP execution for analysis"""
        try:
            execution_record = {
                "timestamp": datetime.now(),
                "pair": pair,
                "side": side,
                "strategy": config.strategy.value,
                "target_quantity": config.total_quantity,
                "filled_quantity": result.total_filled,
                "fill_rate": result.total_filled / config.total_quantity,
                "average_price": result.average_price,
                "total_fees": result.total_fees,
                "implementation_shortfall_bp": result.implementation_shortfall_bp,
                "market_impact_bp": result.market_impact_bp,
                "slices_completed": result.slices_completed,
                "slices_failed": result.slices_failed
            }
            
            self.execution_history.append(execution_record)
            
            # Update strategy performance
            strategy = config.strategy.value
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    "executions": 0,
                    "avg_fill_rate": 0,
                    "avg_shortfall_bp": 0,
                    "avg_market_impact_bp": 0
                }
            
            perf = self.strategy_performance[strategy]
            perf["executions"] += 1
            perf["avg_fill_rate"] = (
                perf["avg_fill_rate"] * (perf["executions"] - 1) + result.total_filled / config.total_quantity
            ) / perf["executions"]
            perf["avg_shortfall_bp"] = (
                perf["avg_shortfall_bp"] * (perf["executions"] - 1) + result.implementation_shortfall_bp
            ) / perf["executions"]
            perf["avg_market_impact_bp"] = (
                perf["avg_market_impact_bp"] * (perf["executions"] - 1) + result.market_impact_bp
            ) / perf["executions"]
            
        except Exception as e:
            logger.error(f"Failed to record TWAP execution: {e}")
    
    def _generate_linear_plan(self, config: TWAPConfig) -> List[TWAPSlice]:
        """Generate simple linear TWAP plan as fallback"""
        try:
            slices = []
            start_time = datetime.now()
            slice_size = config.total_quantity / config.slice_count
            interval_minutes = config.duration_minutes / config.slice_count
            
            for i in range(config.slice_count):
                slice_time = start_time + timedelta(minutes=i * interval_minutes)
                slice_obj = TWAPSlice(
                    slice_id=f"linear_slice_{i+1}",
                    target_quantity=slice_size,
                    target_time=slice_time
                )
                slices.append(slice_obj)
            
            return slices
            
        except Exception as e:
            logger.error(f"Linear plan generation failed: {e}")
            raise