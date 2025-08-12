"""
Execution Policy Management

Defines execution strategies per trading pair and market regime,
optimizing for maker ratios, fill quality, and cost minimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution mode strategies"""
    POST_ONLY = "post_only"           # Maker-only orders
    AGGRESSIVE = "aggressive"         # Immediate execution (taker)
    TWAP = "twap"                    # Time-weighted average price
    ICEBERG = "iceberg"              # Hidden size orders
    ADAPTIVE = "adaptive"            # Dynamic strategy switching


class TimeInForce(Enum):
    """Time-in-force options"""
    GTC = "gtc"                      # Good-till-cancelled
    IOC = "ioc"                      # Immediate-or-cancel
    FOK = "fok"                      # Fill-or-kill
    GTD = "gtd"                      # Good-till-date
    POST = "post"                    # Post-only (maker)


class MarketCondition(Enum):
    """Market condition classifications"""
    LIQUID = "liquid"                # Deep, tight spreads
    ILLIQUID = "illiquid"           # Shallow, wide spreads
    VOLATILE = "volatile"           # High price movement
    CALM = "calm"                   # Low volatility
    FUNDING = "funding"             # Near funding times
    CLOSING = "closing"             # Market close periods


@dataclass
class ExecutionConstraints:
    """Execution constraints for a trading pair"""
    max_order_size: float           # Maximum single order size
    min_order_size: float           # Minimum order size
    max_participation_rate: float   # Max % of volume
    max_spread_tolerance_bp: int    # Maximum spread to trade through
    
    # Timing constraints
    avoid_funding_minutes: int = 10  # Minutes to avoid around funding
    avoid_close_minutes: int = 15   # Minutes to avoid around close
    preferred_hours: List[int] = None  # Preferred trading hours (UTC)
    
    # Fee constraints
    target_maker_ratio: float = 0.8  # Target percentage of maker fills
    max_taker_fee_bp: int = 25      # Maximum acceptable taker fee


@dataclass
class ExecutionSettings:
    """Execution settings per regime and pair"""
    primary_mode: ExecutionMode
    fallback_mode: ExecutionMode
    time_in_force: TimeInForce
    
    # Order parameters
    post_only_timeout_seconds: int = 30
    price_improvement_ticks: int = 1
    iceberg_show_ratio: float = 0.1  # Show 10% of size
    
    # TWAP parameters
    twap_duration_minutes: int = 15
    twap_slice_count: int = 5
    
    # Adaptive thresholds
    spread_threshold_aggressive_bp: int = 20
    volume_threshold_aggressive: float = 0.05  # 5% of recent volume


class ExecutionPolicy:
    """
    Manages execution policies per trading pair and market regime
    """
    
    def __init__(self):
        # Policy storage
        self.pair_policies = {}     # pair -> regime -> ExecutionSettings
        self.pair_constraints = {}  # pair -> ExecutionConstraints
        
        # Market condition tracking
        self.current_conditions = {}  # pair -> MarketCondition
        self.condition_history = []
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
        
        # Initialize default policies
        self._initialize_default_policies()
        
    def set_pair_policy(self, 
                       pair: str,
                       regime: str,
                       settings: ExecutionSettings,
                       constraints: ExecutionConstraints) -> None:
        """Set execution policy for a specific pair and regime"""
        try:
            if pair not in self.pair_policies:
                self.pair_policies[pair] = {}
            
            self.pair_policies[pair][regime] = settings
            self.pair_constraints[pair] = constraints
            
            logger.info(f"Set execution policy for {pair} in {regime} regime: {settings.primary_mode.value}")
            
        except Exception as e:
            logger.error(f"Failed to set execution policy for {pair}: {e}")
    
    def get_execution_strategy(self, 
                             pair: str,
                             regime: str,
                             market_data: Dict[str, Any],
                             order_size: float) -> Dict[str, Any]:
        """
        Get optimal execution strategy for current conditions
        
        Args:
            pair: Trading pair
            regime: Market regime
            market_data: Current market data
            order_size: Order size in base currency
            
        Returns:
            Execution strategy configuration
        """
        try:
            # Get base policy
            settings = self._get_policy_settings(pair, regime)
            constraints = self.pair_constraints.get(pair)
            
            if not settings or not constraints:
                return self._get_default_strategy(pair, order_size)
            
            # Assess current market conditions
            condition = self._assess_market_condition(pair, market_data)
            self.current_conditions[pair] = condition
            
            # Check execution constraints
            if not self._check_execution_constraints(pair, constraints, market_data, order_size):
                return {"execute": False, "reason": "Execution constraints not met"}
            
            # Determine optimal mode
            optimal_mode = self._determine_optimal_mode(
                settings, condition, market_data, order_size, constraints
            )
            
            # Build execution strategy
            strategy = self._build_execution_strategy(
                pair, optimal_mode, settings, constraints, market_data, order_size
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to get execution strategy for {pair}: {e}")
            return self._get_default_strategy(pair, order_size)
    
    def record_execution(self, 
                        pair: str,
                        execution_data: Dict[str, Any]) -> None:
        """Record execution results for analysis"""
        try:
            execution_record = {
                "timestamp": datetime.now(),
                "pair": pair,
                "regime": execution_data.get("regime"),
                "mode_used": execution_data.get("mode"),
                "intended_size": execution_data.get("intended_size"),
                "filled_size": execution_data.get("filled_size"),
                "average_price": execution_data.get("average_price"),
                "market_price": execution_data.get("market_price"),
                "fee_paid": execution_data.get("fee_paid"),
                "fee_type": execution_data.get("fee_type"),  # maker/taker
                "slippage_bp": execution_data.get("slippage_bp"),
                "execution_time_ms": execution_data.get("execution_time_ms"),
                "partial_fills": execution_data.get("partial_fills", 0)
            }
            
            self.execution_history.append(execution_record)
            
            # Keep only recent history
            if len(self.execution_history) > 10000:
                self.execution_history = self.execution_history[-10000:]
            
            # Update performance metrics
            self._update_performance_metrics(pair, execution_record)
            
        except Exception as e:
            logger.error(f"Failed to record execution for {pair}: {e}")
    
    def get_execution_analytics(self, 
                              pair: Optional[str] = None,
                              hours_back: int = 24) -> Dict[str, Any]:
        """Get execution performance analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter recent executions
            recent_executions = [
                ex for ex in self.execution_history
                if ex["timestamp"] >= cutoff_time and (pair is None or ex["pair"] == pair)
            ]
            
            if not recent_executions:
                return {"status": "No recent executions"}
            
            analytics = {
                "period_hours": hours_back,
                "total_executions": len(recent_executions),
                "pairs_traded": len(set(ex["pair"] for ex in recent_executions))
            }
            
            # Calculate key metrics
            analytics.update(self._calculate_execution_metrics(recent_executions))
            
            # Per-pair breakdown if not filtered
            if pair is None and len(set(ex["pair"] for ex in recent_executions)) > 1:
                analytics["per_pair"] = {}
                for p in set(ex["pair"] for ex in recent_executions):
                    pair_execs = [ex for ex in recent_executions if ex["pair"] == p]
                    analytics["per_pair"][p] = self._calculate_execution_metrics(pair_execs)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate execution analytics: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _initialize_default_policies(self) -> None:
        """Initialize default execution policies"""
        try:
            # Default settings for different regimes
            default_settings = {
                "trend_up": ExecutionSettings(
                    primary_mode=ExecutionMode.POST_ONLY,
                    fallback_mode=ExecutionMode.TWAP,
                    time_in_force=TimeInForce.GTC,
                    post_only_timeout_seconds=45,
                    twap_duration_minutes=20
                ),
                "trend_down": ExecutionSettings(
                    primary_mode=ExecutionMode.POST_ONLY,
                    fallback_mode=ExecutionMode.TWAP,
                    time_in_force=TimeInForce.GTC,
                    post_only_timeout_seconds=45,
                    twap_duration_minutes=20
                ),
                "mean_reversion": ExecutionSettings(
                    primary_mode=ExecutionMode.ADAPTIVE,
                    fallback_mode=ExecutionMode.POST_ONLY,
                    time_in_force=TimeInForce.GTC,
                    post_only_timeout_seconds=30,
                    twap_duration_minutes=15
                ),
                "high_vol_chop": ExecutionSettings(
                    primary_mode=ExecutionMode.TWAP,
                    fallback_mode=ExecutionMode.ICEBERG,
                    time_in_force=TimeInForce.GTC,
                    twap_duration_minutes=10,
                    twap_slice_count=8
                ),
                "risk_off": ExecutionSettings(
                    primary_mode=ExecutionMode.AGGRESSIVE,
                    fallback_mode=ExecutionMode.POST_ONLY,
                    time_in_force=TimeInForce.IOC,
                    post_only_timeout_seconds=15
                )
            }
            
            # Default constraints
            default_constraints = ExecutionConstraints(
                max_order_size=100000.0,  # $100k max
                min_order_size=100.0,     # $100 min
                max_participation_rate=0.1,  # 10% of volume
                max_spread_tolerance_bp=50,   # 50bp max spread
                target_maker_ratio=0.75,      # 75% maker fills
                max_taker_fee_bp=25           # 25bp max taker fee
            )
            
            # Store as defaults (can be overridden per pair)
            self.default_settings = default_settings
            self.default_constraints = default_constraints
            
        except Exception as e:
            logger.error(f"Failed to initialize default policies: {e}")
    
    def _get_policy_settings(self, pair: str, regime: str) -> Optional[ExecutionSettings]:
        """Get execution settings for pair and regime"""
        try:
            if pair in self.pair_policies and regime in self.pair_policies[pair]:
                return self.pair_policies[pair][regime]
            elif hasattr(self, 'default_settings') and regime in self.default_settings:
                return self.default_settings[regime]
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get policy settings for {pair}, {regime}: {e}")
            return None
    
    def _assess_market_condition(self, pair: str, market_data: Dict[str, Any]) -> MarketCondition:
        """Assess current market conditions"""
        try:
            # Extract key metrics
            spread_bp = market_data.get("spread_bp", 0)
            depth = market_data.get("depth_quote", 0)
            volume = market_data.get("volume_1h", 0)
            volatility = market_data.get("volatility_1h", 0)
            
            # Check for funding period
            current_time = datetime.now()
            if self._is_funding_period(current_time):
                return MarketCondition.FUNDING
            
            # Check for market close
            if self._is_market_close_period(current_time):
                return MarketCondition.CLOSING
            
            # Assess liquidity
            if spread_bp > 30 or depth < 10000:
                return MarketCondition.ILLIQUID
            elif spread_bp < 10 and depth > 50000:
                return MarketCondition.LIQUID
            
            # Assess volatility
            if volatility > 0.05:  # 5% hourly volatility
                return MarketCondition.VOLATILE
            else:
                return MarketCondition.CALM
            
        except Exception as e:
            logger.error(f"Failed to assess market condition for {pair}: {e}")
            return MarketCondition.CALM
    
    def _is_funding_period(self, timestamp: datetime) -> bool:
        """Check if current time is near funding period"""
        try:
            # Funding typically happens at 00:00, 08:00, 16:00 UTC
            funding_hours = [0, 8, 16]
            current_hour = timestamp.hour
            current_minute = timestamp.minute
            
            for funding_hour in funding_hours:
                # Check if within 10 minutes of funding
                if current_hour == funding_hour and current_minute <= 10:
                    return True
                elif current_hour == (funding_hour - 1) % 24 and current_minute >= 50:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check funding period: {e}")
            return False
    
    def _is_market_close_period(self, timestamp: datetime) -> bool:
        """Check if current time is near market close (for traditional markets)"""
        try:
            # Crypto markets are 24/7, but some pairs have reduced liquidity
            # during traditional market close periods
            hour = timestamp.hour
            
            # Reduced liquidity periods (rough approximation)
            if 22 <= hour <= 23 or 0 <= hour <= 6:  # Late night / early morning UTC
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check market close period: {e}")
            return False
    
    def _check_execution_constraints(self, 
                                   pair: str,
                                   constraints: ExecutionConstraints,
                                   market_data: Dict[str, Any],
                                   order_size: float) -> bool:
        """Check if execution constraints are satisfied"""
        try:
            # Size constraints
            if order_size < constraints.min_order_size or order_size > constraints.max_order_size:
                return False
            
            # Spread constraint
            spread_bp = market_data.get("spread_bp", float('inf'))
            if spread_bp > constraints.max_spread_tolerance_bp:
                return False
            
            # Volume participation constraint
            recent_volume = market_data.get("volume_1h", 0)
            if recent_volume > 0:
                participation_rate = order_size / recent_volume
                if participation_rate > constraints.max_participation_rate:
                    return False
            
            # Time constraints
            current_time = datetime.now()
            
            if constraints.avoid_funding_minutes > 0 and self._is_funding_period(current_time):
                return False
            
            if constraints.avoid_close_minutes > 0 and self._is_market_close_period(current_time):
                return False
            
            # Preferred hours constraint
            if constraints.preferred_hours and current_time.hour not in constraints.preferred_hours:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check execution constraints for {pair}: {e}")
            return False
    
    def _determine_optimal_mode(self, 
                              settings: ExecutionSettings,
                              condition: MarketCondition,
                              market_data: Dict[str, Any],
                              order_size: float,
                              constraints: ExecutionConstraints) -> ExecutionMode:
        """Determine optimal execution mode based on conditions"""
        try:
            # Start with primary mode
            mode = settings.primary_mode
            
            # Adaptive mode logic
            if mode == ExecutionMode.ADAPTIVE:
                spread_bp = market_data.get("spread_bp", 0)
                depth = market_data.get("depth_quote", 0)
                
                if condition == MarketCondition.ILLIQUID:
                    mode = ExecutionMode.TWAP
                elif condition == MarketCondition.VOLATILE:
                    mode = ExecutionMode.ICEBERG
                elif condition == MarketCondition.FUNDING or condition == MarketCondition.CLOSING:
                    mode = ExecutionMode.POST_ONLY
                elif spread_bp < settings.spread_threshold_aggressive_bp and depth > order_size * 5:
                    mode = ExecutionMode.POST_ONLY
                else:
                    mode = ExecutionMode.TWAP
            
            # Override for specific conditions
            if condition == MarketCondition.FUNDING:
                # Avoid aggressive trading during funding
                if mode == ExecutionMode.AGGRESSIVE:
                    mode = settings.fallback_mode
            
            elif condition == MarketCondition.ILLIQUID:
                # Use patient strategies for illiquid markets
                if mode == ExecutionMode.AGGRESSIVE:
                    mode = ExecutionMode.TWAP
            
            elif condition == MarketCondition.VOLATILE:
                # Avoid post-only in volatile conditions (may not fill)
                if mode == ExecutionMode.POST_ONLY:
                    mode = ExecutionMode.ICEBERG
            
            return mode
            
        except Exception as e:
            logger.error(f"Failed to determine optimal execution mode: {e}")
            return settings.primary_mode
    
    def _build_execution_strategy(self, 
                                 pair: str,
                                 mode: ExecutionMode,
                                 settings: ExecutionSettings,
                                 constraints: ExecutionConstraints,
                                 market_data: Dict[str, Any],
                                 order_size: float) -> Dict[str, Any]:
        """Build complete execution strategy"""
        try:
            strategy = {
                "execute": True,
                "pair": pair,
                "mode": mode.value,
                "order_size": order_size,
                "time_in_force": settings.time_in_force.value
            }
            
            # Mode-specific parameters
            if mode == ExecutionMode.POST_ONLY:
                strategy.update({
                    "post_only": True,
                    "timeout_seconds": settings.post_only_timeout_seconds,
                    "price_improvement_ticks": settings.price_improvement_ticks
                })
                
            elif mode == ExecutionMode.TWAP:
                strategy.update({
                    "twap_duration_minutes": settings.twap_duration_minutes,
                    "slice_count": settings.twap_slice_count,
                    "randomize_timing": True
                })
                
            elif mode == ExecutionMode.ICEBERG:
                strategy.update({
                    "iceberg": True,
                    "show_size": order_size * settings.iceberg_show_ratio,
                    "refresh_on_fill": True
                })
                
            elif mode == ExecutionMode.AGGRESSIVE:
                strategy.update({
                    "aggressive": True,
                    "max_slippage_bp": min(50, constraints.max_spread_tolerance_bp)
                })
            
            # Fee optimization hints
            current_maker_ratio = self._get_current_maker_ratio(pair)
            if current_maker_ratio < constraints.target_maker_ratio:
                strategy["prefer_maker"] = True
            
            # Risk management
            strategy.update({
                "max_participation_rate": constraints.max_participation_rate,
                "max_spread_tolerance_bp": constraints.max_spread_tolerance_bp
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to build execution strategy: {e}")
            return {"execute": False, "reason": "Strategy building failed"}
    
    def _get_default_strategy(self, pair: str, order_size: float) -> Dict[str, Any]:
        """Get conservative default strategy"""
        return {
            "execute": True,
            "pair": pair,
            "mode": "post_only",
            "order_size": order_size,
            "time_in_force": "gtc",
            "post_only": True,
            "timeout_seconds": 60,
            "price_improvement_ticks": 1,
            "max_participation_rate": 0.05,  # Conservative 5%
            "reason": "Default conservative strategy"
        }
    
    def _get_current_maker_ratio(self, pair: str) -> float:
        """Get current maker fill ratio for pair"""
        try:
            recent_executions = [
                ex for ex in self.execution_history[-100:]  # Last 100 executions
                if ex["pair"] == pair and ex.get("fee_type")
            ]
            
            if not recent_executions:
                return 0.5  # Default assumption
            
            maker_count = sum(1 for ex in recent_executions if ex["fee_type"] == "maker")
            return maker_count / len(recent_executions)
            
        except Exception as e:
            logger.error(f"Failed to get current maker ratio for {pair}: {e}")
            return 0.5
    
    def _update_performance_metrics(self, pair: str, execution_record: Dict[str, Any]) -> None:
        """Update performance metrics"""
        try:
            if pair not in self.performance_metrics:
                self.performance_metrics[pair] = {
                    "total_executions": 0,
                    "maker_fills": 0,
                    "total_slippage_bp": 0,
                    "total_fees": 0,
                    "total_notional": 0
                }
            
            metrics = self.performance_metrics[pair]
            
            metrics["total_executions"] += 1
            
            if execution_record.get("fee_type") == "maker":
                metrics["maker_fills"] += 1
            
            if execution_record.get("slippage_bp") is not None:
                metrics["total_slippage_bp"] += execution_record["slippage_bp"]
            
            if execution_record.get("fee_paid"):
                metrics["total_fees"] += execution_record["fee_paid"]
            
            notional = (execution_record.get("filled_size", 0) * 
                       execution_record.get("average_price", 0))
            metrics["total_notional"] += notional
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics for {pair}: {e}")
    
    def _calculate_execution_metrics(self, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate execution performance metrics"""
        try:
            if not executions:
                return {}
            
            # Basic counts
            total = len(executions)
            maker_fills = sum(1 for ex in executions if ex.get("fee_type") == "maker")
            successful_fills = sum(1 for ex in executions if ex.get("filled_size", 0) > 0)
            
            # Ratios
            maker_ratio = maker_fills / total if total > 0 else 0
            fill_rate = successful_fills / total if total > 0 else 0
            
            # Slippage analysis
            slippages = [ex.get("slippage_bp", 0) for ex in executions if ex.get("slippage_bp") is not None]
            avg_slippage = np.mean(slippages) if slippages else 0
            
            # Fee analysis
            total_fees = sum(ex.get("fee_paid", 0) for ex in executions)
            total_notional = sum(
                (ex.get("filled_size", 0) * ex.get("average_price", 0))
                for ex in executions
            )
            avg_fee_bp = (total_fees / total_notional * 10000) if total_notional > 0 else 0
            
            # Execution time analysis
            execution_times = [ex.get("execution_time_ms", 0) for ex in executions]
            avg_execution_time = np.mean(execution_times) if execution_times else 0
            
            return {
                "maker_ratio": maker_ratio,
                "fill_rate": fill_rate,
                "avg_slippage_bp": avg_slippage,
                "avg_fee_bp": avg_fee_bp,
                "avg_execution_time_ms": avg_execution_time,
                "total_notional": total_notional,
                "successful_executions": successful_fills
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate execution metrics: {e}")
            return {}