"""
Price Improvement Ladder

Implements intelligent price improvement strategies to enhance
execution quality through dynamic pricing ladders.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ImprovementStrategy(Enum):
    """Price improvement strategies"""
    TICK_IMPROVEMENT = "tick_improvement"    # Improve by tick size
    SPREAD_BASED = "spread_based"           # Based on current spread
    VOLUME_WEIGHTED = "volume_weighted"     # Based on order book depth
    ADAPTIVE = "adaptive"                   # Dynamic strategy selection
    MOMENTUM_BASED = "momentum_based"       # Based on price momentum


class LadderType(Enum):
    """Types of price ladders"""
    STATIC = "static"                       # Fixed price levels
    DYNAMIC = "dynamic"                     # Adjusting price levels
    ICEBERG = "iceberg"                     # Hidden quantity ladders
    TIME_BASED = "time_based"              # Time-priority ladders


@dataclass
class PriceLadderLevel:
    """Individual level in price ladder"""
    level_id: str
    price: float
    quantity: float
    priority: int  # Lower number = higher priority

    # Status tracking
    active: bool = True
    filled_quantity: float = 0.0
    created_at: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def remaining_quantity(self) -> float:
        return max(0, self.quantity - self.filled_quantity)

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0


@dataclass
class ImprovementResult:
    """Result of price improvement analysis"""
    recommended_strategy: ImprovementStrategy
    suggested_prices: List[float]
    expected_fill_rate: float
    expected_time_minutes: int
    confidence: float
    reasoning: str

    # Cost-benefit analysis
    potential_savings_bp: float
    execution_risk: float  # Risk of non-execution
    market_impact_bp: float


class PriceImprovementLadder:
    """
    Advanced price improvement ladder system
    """

    def __init__(self):
        self.active_ladders = {}  # ladder_id -> List[PriceLadderLevel]
        self.ladder_history = []

        # Performance tracking
        self.improvement_performance = {}  # strategy -> performance metrics

        # Configuration
        self.default_ladder_levels = 3
        self.max_improvement_bp = 50  # Maximum price improvement
        self.min_tick_improvement = 1  # Minimum tick improvement

    def create_price_ladder(self,
                          ladder_id: str,
                          pair: str,
                          side: str,
                          total_quantity: float,
                          reference_price: float,
                          market_data: Dict[str, Any],
                          strategy: ImprovementStrategy = ImprovementStrategy.ADAPTIVE) -> List[PriceLadderLevel]:
        """
        Create optimized price improvement ladder

        Args:
            ladder_id: Unique identifier for ladder
            pair: Trading pair
            side: 'buy' or 'sell'
            total_quantity: Total quantity to execute
            reference_price: Reference price (mid, bid, ask)
            market_data: Current market conditions
            strategy: Improvement strategy to use

        Returns:
            List of price ladder levels
        """
        try:
            logger.info(f"Creating price ladder {ladder_id}: {total_quantity} {pair} @ {reference_price}")

            # Analyze market conditions for optimal ladder
            analysis = self._analyze_ladder_conditions(pair, market_data, total_quantity)

            # Determine optimal strategy if adaptive
            if strategy == ImprovementStrategy.ADAPTIVE:
                strategy = self._select_optimal_strategy(analysis, side)

            # Generate ladder levels
            levels = self._generate_ladder_levels(
                ladder_id, side, total_quantity, reference_price,
                market_data, strategy, analysis
            )

            # Store ladder
            self.active_ladders[ladder_id] = levels

            # Record for analytics
            self._record_ladder_creation(ladder_id, pair, side, levels, strategy)

            logger.info(f"Created {len(levels)} level price ladder using {strategy.value} strategy")

            return levels

        except Exception as e:
            logger.error(f"Failed to create price ladder {ladder_id}: {e}")
            raise

    def update_ladder(self,
                     ladder_id: str,
                     market_data: Dict[str, Any],
                     fills: Optional[List[Dict[str, Any]]] = None) -> List[PriceLadderLevel]:
        """Update existing price ladder based on market changes and fills"""
        try:
            if ladder_id not in self.active_ladders:
                logger.warning(f"Ladder {ladder_id} not found for update")
                return []

            levels = self.active_ladders[ladder_id]

            # Update fills if provided
            if fills:
                self._update_ladder_fills(levels, fills)

            # Check if ladder needs adjustment
            if self._should_adjust_ladder(levels, market_data):
                adjusted_levels = self._adjust_ladder_prices(levels, market_data)
                self.active_ladders[ladder_id] = adjusted_levels
                return adjusted_levels

            return levels

        except Exception as e:
            logger.error(f"Failed to update ladder {ladder_id}: {e}")
            return self.active_ladders.get(ladder_id, [])

    def get_improvement_recommendation(self,
                                     pair: str,
                                     side: str,
                                     quantity: float,
                                     current_price: float,
                                     market_data: Dict[str, Any]) -> ImprovementResult:
        """Get price improvement recommendation"""
        try:
            # Analyze current conditions
            analysis = self._analyze_improvement_opportunity(
                pair, side, quantity, current_price, market_data
            )

            # Determine optimal strategy
            strategy = self._select_optimal_strategy(analysis, side)

            # Calculate improvement prices
            improvement_prices = self._calculate_improvement_prices(
                side, current_price, market_data, strategy
            )

            # Estimate outcomes
            expected_fill_rate = self._estimate_fill_rate(
                improvement_prices, market_data, quantity
            )

            expected_time = self._estimate_execution_time(
                improvement_prices, market_data, strategy
            )

            # Cost-benefit analysis
            potential_savings = self._calculate_potential_savings(
                improvement_prices, current_price, side
            )

            execution_risk = self._calculate_execution_risk(
                improvement_prices, market_data, expected_fill_rate
            )

            market_impact = self._estimate_market_impact(
                quantity, market_data
            )

            return ImprovementResult(
                recommended_strategy=strategy,
                suggested_prices=improvement_prices,
                expected_fill_rate=expected_fill_rate,
                expected_time_minutes=expected_time,
                confidence=analysis.get("confidence", 0.7),
                reasoning=analysis.get("reasoning", f"Using {strategy.value} strategy"),
                potential_savings_bp=potential_savings,
                execution_risk=execution_risk,
                market_impact_bp=market_impact
            )

        except Exception as e:
            logger.error(f"Failed to generate improvement recommendation: {e}")
            return self._get_default_improvement_result(current_price)

    def get_ladder_analytics(self,
                           ladder_id: Optional[str] = None,
                           days_back: int = 7) -> Dict[str, Any]:
        """Get price improvement analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter relevant history
            if ladder_id:
                history = [h for h in self.ladder_history if h.get("ladder_id") == ladder_id]
            else:
                history = [h for h in self.ladder_history if h.get("created_at", datetime.min) >= cutoff_time]

            if not history:
                return {"status": "No ladder history found"}

            analytics = self._calculate_improvement_analytics(history)

            return analytics

        except Exception as e:
            logger.error(f"Failed to generate ladder analytics: {e}")
            return {"status": "Error", "error": str(e)}

    def _analyze_ladder_conditions(self,
                                 pair: str,
                                 market_data: Dict[str, Any],
                                 quantity: float) -> Dict[str, Any]:
        """Analyze market conditions for ladder optimization"""
        try:
            spread_bp = market_data.get("spread_bp", 10)
            depth_quote = market_data.get("depth_quote", 0)
            volume_1h = market_data.get("volume_1h", 0)
            volatility = market_data.get("volatility_1h", 0.02)

            # Liquidity analysis
            liquidity_score = min(1.0, depth_quote / (quantity * market_data.get("mid_price", 1000)))

            # Market activity analysis
            activity_score = min(1.0, volume_1h / 1000000) if volume_1h > 0 else 0.1

            # Volatility impact
            volatility_score = max(0.1, min(1.0, 1 - volatility * 10))  # Lower vol = higher score

            # Spread analysis
            spread_score = max(0.1, min(1.0, 1 - spread_bp / 100))  # Tighter spread = higher score

            # Overall market suitability for improvement
            improvement_suitability = (
                0.3 * liquidity_score +
                0.2 * activity_score +
                0.2 * volatility_score +
                0.3 * spread_score
            )

            return {
                "spread_bp": spread_bp,
                "liquidity_score": liquidity_score,
                "activity_score": activity_score,
                "volatility_score": volatility_score,
                "spread_score": spread_score,
                "improvement_suitability": improvement_suitability,
                "recommended_levels": min(5, max(2, int(improvement_suitability * 4))),
                "confidence": improvement_suitability
            }

        except Exception as e:
            logger.error(f"Ladder condition analysis failed: {e}")
            return {"improvement_suitability": 0.5, "recommended_levels": 3}

    def _select_optimal_strategy(self,
                               analysis: Dict[str, Any],
                               side: str) -> ImprovementStrategy:
        """Select optimal improvement strategy based on analysis"""
        try:
            liquidity_score = analysis.get("liquidity_score", 0.5)
            spread_bp = analysis.get("spread_bp", 20)
            volatility_score = analysis.get("volatility_score", 0.5)

            # Strategy selection logic
            if spread_bp < 10 and liquidity_score > 0.7:
                return ImprovementStrategy.TICK_IMPROVEMENT
            elif spread_bp > 30:
                return ImprovementStrategy.SPREAD_BASED
            elif liquidity_score < 0.3:
                return ImprovementStrategy.VOLUME_WEIGHTED
            elif volatility_score < 0.4:
                return ImprovementStrategy.MOMENTUM_BASED
            else:
                return ImprovementStrategy.SPREAD_BASED  # Good general strategy

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return ImprovementStrategy.SPREAD_BASED

    def _generate_ladder_levels(self,
                              ladder_id: str,
                              side: str,
                              total_quantity: float,
                              reference_price: float,
                              market_data: Dict[str, Any],
                              strategy: ImprovementStrategy,
                              analysis: Dict[str, Any]) -> List[PriceLadderLevel]:
        """Generate price ladder levels"""
        try:
            num_levels = analysis.get("recommended_levels", self.default_ladder_levels)
            spread_bp = market_data.get("spread_bp", 10)

            levels = []

            if strategy == ImprovementStrategy.TICK_IMPROVEMENT:
                levels = self._generate_tick_improvement_levels(
                    ladder_id, side, total_quantity, reference_price, num_levels
                )
            elif strategy == ImprovementStrategy.SPREAD_BASED:
                levels = self._generate_spread_based_levels(
                    ladder_id, side, total_quantity, reference_price, spread_bp, num_levels
                )
            elif strategy == ImprovementStrategy.VOLUME_WEIGHTED:
                levels = self._generate_volume_weighted_levels(
                    ladder_id, side, total_quantity, reference_price, market_data, num_levels
                )
            elif strategy == ImprovementStrategy.MOMENTUM_BASED:
                levels = self._generate_momentum_based_levels(
                    ladder_id, side, total_quantity, reference_price, market_data, num_levels
                )
            else:
                # Default to spread-based
                levels = self._generate_spread_based_levels(
                    ladder_id, side, total_quantity, reference_price, spread_bp, num_levels
                )

            return levels

        except Exception as e:
            logger.error(f"Ladder level generation failed: {e}")
            return self._generate_fallback_levels(ladder_id, side, total_quantity, reference_price)

    def _generate_tick_improvement_levels(self,
                                        ladder_id: str,
                                        side: str,
                                        total_quantity: float,
                                        reference_price: float,
                                        num_levels: int) -> List[PriceLadderLevel]:
        """Generate tick-based improvement levels"""
        try:
            levels = []
            tick_size = self._estimate_tick_size(reference_price)
            quantity_per_level = total_quantity / num_levels

            for i in range(num_levels):
                if side == "buy":
                    # For buy orders, improve by going higher (more aggressive)
                    price = reference_price + (i + 1) * tick_size
                else:
                    # For sell orders, improve by going lower (more aggressive)
                    price = reference_price - (i + 1) * tick_size

                level = PriceLadderLevel(
                    level_id=f"{ladder_id}_tick_{i+1}",
                    price=price,
                    quantity=quantity_per_level,
                    priority=i + 1
                )
                levels.append(level)

            return levels

        except Exception as e:
            logger.error(f"Tick improvement level generation failed: {e}")
            return []

    def _generate_spread_based_levels(self,
                                    ladder_id: str,
                                    side: str,
                                    total_quantity: float,
                                    reference_price: float,
                                    spread_bp: float,
                                    num_levels: int) -> List[PriceLadderLevel]:
        """Generate spread-based improvement levels"""
        try:
            levels = []
            quantity_per_level = total_quantity / num_levels

            # Use spread as basis for improvements
            spread_absolute = reference_price * spread_bp / 10000
            improvement_per_level = spread_absolute / (num_levels + 1)

            for i in range(num_levels):
                if side == "buy":
                    # Improve by bidding higher
                    price = reference_price + (i + 1) * improvement_per_level
                else:
                    # Improve by asking lower
                    price = reference_price - (i + 1) * improvement_per_level

                level = PriceLadderLevel(
                    level_id=f"{ladder_id}_spread_{i+1}",
                    price=price,
                    quantity=quantity_per_level,
                    priority=i + 1
                )
                levels.append(level)

            return levels

        except Exception as e:
            logger.error(f"Spread-based level generation failed: {e}")
            return []

    def _generate_volume_weighted_levels(self,
                                       ladder_id: str,
                                       side: str,
                                       total_quantity: float,
                                       reference_price: float,
                                       market_data: Dict[str, Any],
                                       num_levels: int) -> List[PriceLadderLevel]:
        """Generate volume-weighted improvement levels"""
        try:
            levels = []

            # Get order book data for volume weighting
            depth_levels = market_data.get("order_book_levels", [])

            if not depth_levels:
                # Fallback to spread-based if no order book data
                return self._generate_spread_based_levels(
                    ladder_id, side, total_quantity, reference_price,
                    market_data.get("spread_bp", 10), num_levels
                )

            # Calculate volume-weighted prices
            cumulative_quantity = 0
            target_quantities = [total_quantity * (i + 1) / num_levels for i in range(num_levels)]

            for i, target_qty in enumerate(target_quantities):
                # Find price level for target quantity
                level_price = self._find_price_for_quantity(depth_levels, target_qty, side)

                if level_price > 0:
                    level_quantity = target_qty - cumulative_quantity

                    level = PriceLadderLevel(
                        level_id=f"{ladder_id}_volume_{i+1}",
                        price=level_price,
                        quantity=level_quantity,
                        priority=i + 1
                    )
                    levels.append(level)
                    cumulative_quantity = target_qty

            return levels

        except Exception as e:
            logger.error(f"Volume-weighted level generation failed: {e}")
            return []

    def _generate_momentum_based_levels(self,
                                      ladder_id: str,
                                      side: str,
                                      total_quantity: float,
                                      reference_price: float,
                                      market_data: Dict[str, Any],
                                      num_levels: int) -> List[PriceLadderLevel]:
        """Generate momentum-based improvement levels"""
        try:
            levels = []
            quantity_per_level = total_quantity / num_levels

            # Get price momentum
            momentum_bp = market_data.get("momentum_1h_bp", 0)

            # Adjust improvement based on momentum
            base_improvement_bp = 5  # 5bp base improvement
            momentum_adjustment = abs(momentum_bp) * 0.1  # 10% of momentum

            improvement_bp = base_improvement_bp + momentum_adjustment
            improvement_per_level_bp = improvement_bp / num_levels

            for i in range(num_levels):
                level_improvement_bp = (i + 1) * improvement_per_level_bp

                if side == "buy":
                    if momentum_bp > 0:  # Positive momentum - be more aggressive
                        price = reference_price * (1 + level_improvement_bp / 10000)
                    else:  # Negative momentum - be more conservative
                        price = reference_price * (1 + level_improvement_bp * 0.5 / 10000)
                else:
                    if momentum_bp < 0:  # Negative momentum - be more aggressive on sells
                        price = reference_price * (1 - level_improvement_bp / 10000)
                    else:  # Positive momentum - be more conservative
                        price = reference_price * (1 - level_improvement_bp * 0.5 / 10000)

                level = PriceLadderLevel(
                    level_id=f"{ladder_id}_momentum_{i+1}",
                    price=price,
                    quantity=quantity_per_level,
                    priority=i + 1
                )
                levels.append(level)

            return levels

        except Exception as e:
            logger.error(f"Momentum-based level generation failed: {e}")
            return []

    def _estimate_tick_size(self, price: float) -> float:
        """Estimate appropriate tick size for price level"""
        try:
            if price < 100:
                return 0.01
            elif price < 1000:
                return 0.1
            elif price < 10000:
                return 1.0
            else:
                return 10.0
        except Exception:
            return 0.01

    def _find_price_for_quantity(self,
                                depth_levels: List[Dict[str, Any]],
                                target_quantity: float,
                                side: str) -> float:
        """Find price level that provides target quantity"""
        try:
            cumulative_quantity = 0

            for level in depth_levels:
                if side == "buy":
                    # For buy orders, look at ask side
                    price = level.get("ask_price", 0)
                    quantity = level.get("ask_quantity", 0)
                else:
                    # For sell orders, look at bid side
                    price = level.get("bid_price", 0)
                    quantity = level.get("bid_quantity", 0)

                cumulative_quantity += quantity

                if cumulative_quantity >= target_quantity:
                    return price

            # If not enough depth, return last price
            if depth_levels:
                last_level = depth_levels[-1]
                return last_level.get(f"{'ask' if side == 'buy' else 'bid'}_price", 0)

            return 0

        except Exception as e:
            logger.error(f"Price finding failed: {e}")
            return 0

    def _should_adjust_ladder(self,
                            levels: List[PriceLadderLevel],
                            market_data: Dict[str, Any]) -> bool:
        """Check if ladder should be adjusted"""
        try:
            if not levels:
                return False

            # Check if market has moved significantly
            current_mid = market_data.get("mid_price", 0)
            if current_mid <= 0:
                return False

            # Get average ladder price
            avg_ladder_price = np.mean([level.price for level in levels if level.active])

            # Adjust if market moved > 20bp from ladder average
            price_change_bp = abs(current_mid - avg_ladder_price) / avg_ladder_price * 10000

            return price_change_bp > 20

        except Exception as e:
            logger.error(f"Ladder adjustment check failed: {e}")
            return False

    def _adjust_ladder_prices(self,
                            levels: List[PriceLadderLevel],
                            market_data: Dict[str, Any]) -> List[PriceLadderLevel]:
        """Adjust ladder prices based on market changes"""
        try:
            current_mid = market_data.get("mid_price", 0)
            if current_mid <= 0:
                return levels

            # Calculate adjustment factor
            avg_ladder_price = np.mean([level.price for level in levels if level.active])
            adjustment_ratio = current_mid / avg_ladder_price

            # Adjust each level
            for level in levels:
                if level.active and level.remaining_quantity > 0:
                    level.price *= adjustment_ratio
                    level.last_updated = datetime.now()

            return levels

        except Exception as e:
            logger.error(f"Ladder price adjustment failed: {e}")
            return levels

    def _update_ladder_fills(self,
                           levels: List[PriceLadderLevel],
                           fills: List[Dict[str, Any]]) -> None:
        """Update ladder with fill information"""
        try:
            for fill in fills:
                level_id = fill.get("level_id")
                filled_quantity = fill.get("quantity", 0)

                # Find matching level
                for level in levels:
                    if level.level_id == level_id:
                        level.filled_quantity += filled_quantity
                        level.last_updated = datetime.now()

                        # Deactivate if fully filled
                        if level.remaining_quantity <= 0:
                            level.active = False

                        break

        except Exception as e:
            logger.error(f"Ladder fill update failed: {e}")

    def _analyze_improvement_opportunity(self,
                                       pair: str,
                                       side: str,
                                       quantity: float,
                                       current_price: float,
                                       market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price improvement opportunity"""
        try:
            spread_bp = market_data.get("spread_bp", 10)
            depth = market_data.get("depth_quote", 0)
            volatility = market_data.get("volatility_1h", 0.02)

            # Calculate improvement opportunity score
            spread_opportunity = min(1.0, spread_bp / 20)  # Higher spread = more opportunity
            depth_support = min(1.0, depth / (quantity * current_price * 5))  # Adequate depth
            volatility_risk = max(0.1, 1 - volatility * 20)  # Lower vol = less risk

            opportunity_score = (
                0.4 * spread_opportunity +
                0.4 * depth_support +
                0.2 * volatility_risk
            )

            return {
                "opportunity_score": opportunity_score,
                "spread_opportunity": spread_opportunity,
                "depth_support": depth_support,
                "volatility_risk": volatility_risk,
                "confidence": opportunity_score,
                "reasoning": f"Spread: {spread_bp}bp, Depth: ${depth:,.0f}, Vol: {volatility:.2%}"
            }

        except Exception as e:
            logger.error(f"Improvement opportunity analysis failed: {e}")
            return {"opportunity_score": 0.5, "confidence": 0.3}

    def _calculate_improvement_prices(self,
                                    side: str,
                                    current_price: float,
                                    market_data: Dict[str, Any],
                                    strategy: ImprovementStrategy) -> List[float]:
        """Calculate improvement price levels"""
        try:
            prices = []
            spread_bp = market_data.get("spread_bp", 10)

            if strategy == ImprovementStrategy.TICK_IMPROVEMENT:
                tick_size = self._estimate_tick_size(current_price)
                for i in range(3):
                    if side == "buy":
                        price = current_price + (i + 1) * tick_size
                    else:
                        price = current_price - (i + 1) * tick_size
                    prices.append(price)

            elif strategy == ImprovementStrategy.SPREAD_BASED:
                spread_absolute = current_price * spread_bp / 10000
                for i in range(3):
                    improvement = spread_absolute * (i + 1) / 4  # 1/4, 2/4, 3/4 of spread
                    if side == "buy":
                        price = current_price + improvement
                    else:
                        price = current_price - improvement
                    prices.append(price)

            else:
                # Default improvement
                base_improvement_bp = 5
                for i in range(3):
                    improvement_bp = base_improvement_bp * (i + 1)
                    if side == "buy":
                        price = current_price * (1 + improvement_bp / 10000)
                    else:
                        price = current_price * (1 - improvement_bp / 10000)
                    prices.append(price)

            return prices

        except Exception as e:
            logger.error(f"Improvement price calculation failed: {e}")
            return [current_price]

    def _estimate_fill_rate(self,
                          prices: List[float],
                          market_data: Dict[str, Any],
                          quantity: float) -> float:
        """Estimate fill rate for improvement prices"""
        try:
            spread_bp = market_data.get("spread_bp", 10)
            depth = market_data.get("depth_quote", 0)

            # Base fill rate starts high and decreases with more aggressive pricing
            base_fill_rate = 0.9

            # Adjust for spread and depth
            if spread_bp > 30:
                base_fill_rate -= 0.2
            if depth < quantity * prices[0] * 2:
                base_fill_rate -= 0.3

            # Decrease with number of aggressive levels
            fill_rate = max(0.3, base_fill_rate - len(prices) * 0.1)

            return fill_rate

        except Exception as e:
            logger.error(f"Fill rate estimation failed: {e}")
            return 0.7

    def _estimate_execution_time(self,
                               prices: List[float],
                               market_data: Dict[str, Any],
                               strategy: ImprovementStrategy) -> int:
        """Estimate execution time in minutes"""
        try:
            base_time = 10  # 10 minutes base

            # Adjust for strategy
            if strategy == ImprovementStrategy.TICK_IMPROVEMENT:
                return base_time + 5  # More patient
            elif strategy == ImprovementStrategy.MOMENTUM_BASED:
                return base_time - 3  # More aggressive timing

            return base_time

        except Exception as e:
            logger.error(f"Execution time estimation failed: {e}")
            return 10

    def _calculate_potential_savings(self,
                                   improvement_prices: List[float],
                                   reference_price: float,
                                   side: str) -> float:
        """Calculate potential savings in basis points"""
        try:
            if not improvement_prices:
                return 0

            avg_improvement_price = np.mean(improvement_prices)

            if side == "buy":
                # For buys, lower price is better
                if avg_improvement_price < reference_price:
                    savings_bp = (reference_price - avg_improvement_price) / reference_price * 10000
                    return savings_bp
            else:
                # For sells, higher price is better
                if avg_improvement_price > reference_price:
                    savings_bp = (avg_improvement_price - reference_price) / reference_price * 10000
                    return savings_bp

            return 0

        except Exception as e:
            logger.error(f"Potential savings calculation failed: {e}")
            return 0

    def _calculate_execution_risk(self,
                                improvement_prices: List[float],
                                market_data: Dict[str, Any],
                                expected_fill_rate: float) -> float:
        """Calculate execution risk (0-1)"""
        try:
            # Base risk from fill rate
            base_risk = 1 - expected_fill_rate

            # Add volatility risk
            volatility = market_data.get("volatility_1h", 0.02)
            volatility_risk = min(0.3, volatility * 10)

            # Add spread risk
            spread_bp = market_data.get("spread_bp", 10)
            spread_risk = min(0.2, spread_bp / 100)

            total_risk = base_risk + volatility_risk + spread_risk

            return max(0, min(1, total_risk))

        except Exception as e:
            logger.error(f"Execution risk calculation failed: {e}")
            return 0.3

    def _estimate_market_impact(self,
                              quantity: float,
                              market_data: Dict[str, Any]) -> float:
        """Estimate market impact in basis points"""
        try:
            depth = market_data.get("depth_quote", 1000000)
            volume_1h = market_data.get("volume_1h", 1000000)

            # Simple market impact model
            depth_impact = (quantity * market_data.get("mid_price", 1000)) / depth * 100
            volume_impact = quantity / (volume_1h / 60) * 10  # Per minute volume

            market_impact_bp = min(50, max(0.1, depth_impact + volume_impact))

            return market_impact_bp

        except Exception as e:
            logger.error(f"Market impact estimation failed: {e}")
            return 5.0

    def _get_default_improvement_result(self, current_price: float) -> ImprovementResult:
        """Get default improvement result"""
        return ImprovementResult(
            recommended_strategy=ImprovementStrategy.TICK_IMPROVEMENT,
            suggested_prices=[current_price],
            expected_fill_rate=0.8,
            expected_time_minutes=10,
            confidence=0.5,
            reasoning="Default recommendation (insufficient data)",
            potential_savings_bp=2.0,
            execution_risk=0.3,
            market_impact_bp=3.0
        )

    def _generate_fallback_levels(self,
                                ladder_id: str,
                                side: str,
                                total_quantity: float,
                                reference_price: float) -> List[PriceLadderLevel]:
        """Generate simple fallback ladder levels"""
        try:
            levels = []
            quantity_per_level = total_quantity / 2
            tick_size = self._estimate_tick_size(reference_price)

            for i in range(2):
                if side == "buy":
                    price = reference_price + (i + 1) * tick_size
                else:
                    price = reference_price - (i + 1) * tick_size

                level = PriceLadderLevel(
                    level_id=f"{ladder_id}_fallback_{i+1}",
                    price=price,
                    quantity=quantity_per_level,
                    priority=i + 1
                )
                levels.append(level)

            return levels

        except Exception as e:
            logger.error(f"Fallback level generation failed: {e}")
            return []

    def _record_ladder_creation(self,
                              ladder_id: str,
                              pair: str,
                              side: str,
                              levels: List[PriceLadderLevel],
                              strategy: ImprovementStrategy) -> None:
        """Record ladder creation for analytics"""
        try:
            ladder_record = {
                "ladder_id": ladder_id,
                "pair": pair,
                "side": side,
                "strategy": strategy.value,
                "level_count": len(levels),
                "total_quantity": sum(level.quantity for level in levels),
                "price_range": {
                    "min": min(level.price for level in levels),
                    "max": max(level.price for level in levels)
                },
                "created_at": datetime.now()
            }

            self.ladder_history.append(ladder_record)

            # Keep only recent history
            if len(self.ladder_history) > 1000:
                self.ladder_history = self.ladder_history[-1000:]

        except Exception as e:
            logger.error(f"Failed to record ladder creation: {e}")

    def _calculate_improvement_analytics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate improvement performance analytics"""
        try:
            if not history:
                return {"status": "No history available"}

            total_ladders = len(history)

            # Strategy performance
            strategies = {}
            for record in history:
                strategy = record.get("strategy", "unknown")
                if strategy not in strategies:
                    strategies[strategy] = {"count": 0, "avg_levels": 0}

                strategies[strategy]["count"] += 1
                strategies[strategy]["avg_levels"] += record.get("level_count", 0)

            # Finalize averages
            for strategy_data in strategies.values():
                if strategy_data["count"] > 0:
                    strategy_data["avg_levels"] /= strategy_data["count"]

            return {
                "total_ladders": total_ladders,
                "strategy_distribution": strategies,
                "avg_levels_per_ladder": np.mean([r.get("level_count", 0) for r in history]),
                "pairs_traded": len(set(r.get("pair", "") for r in history))
            }

        except Exception as e:
            logger.error(f"Improvement analytics calculation failed: {e}")
            return {"status": "Error", "error": str(e)}
