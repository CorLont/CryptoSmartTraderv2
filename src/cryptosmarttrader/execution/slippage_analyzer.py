"""
Slippage Analyzer

Advanced slippage attribution system that analyzes the differences between
expected and realized execution prices to identify performance drag sources.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SlippageSource(Enum):
    """Sources of slippage"""
    MARKET_IMPACT = "market_impact"         # Price movement due to order size
    SPREAD_COST = "spread_cost"             # Bid-ask spread crossing
    TIMING_DELAY = "timing_delay"           # Price movement during execution delay
    PARTIAL_FILLS = "partial_fills"         # Incomplete order execution
    QUEUE_PRIORITY = "queue_priority"       # Order queue positioning
    VOLATILITY_SPIKE = "volatility_spike"   # Market volatility during execution
    LIQUIDITY_SHORTAGE = "liquidity_shortage"  # Insufficient market liquidity
    ADVERSE_SELECTION = "adverse_selection"    # Informed trading against us
    MARKET_STRESS = "market_stress"         # Overall market conditions
    EXECUTION_ALGORITHM = "execution_algorithm"  # Algorithm-specific costs


@dataclass
class SlippageComponent:
    """Individual slippage component analysis"""
    source: SlippageSource
    slippage_bps: float
    contribution_pct: float
    confidence_score: float  # How confident we are in this attribution
    description: str
    
    # Supporting data
    expected_value: float
    realized_value: float
    market_context: Dict[str, Any]


@dataclass
class SlippageAttribution:
    """Complete slippage attribution analysis"""
    order_id: str
    timestamp: datetime
    pair: str
    side: str
    order_size: float
    
    # Price analysis
    entry_price: float          # Expected entry price
    actual_entry_price: float   # Actual weighted average price
    total_slippage_bps: float   # Total slippage in basis points
    
    # Component breakdown
    slippage_components: List[SlippageComponent]
    
    # Performance metrics
    execution_efficiency: float  # Overall execution efficiency (0-1)
    alpha_preserved_pct: float   # How much alpha was preserved
    cost_vs_benchmark_bps: float # Cost vs theoretical best execution
    
    # Market context
    volatility_at_execution: float
    spread_at_execution_bps: float
    market_stress_level: float
    
    @property
    def total_attributed_slippage(self) -> float:
        """Sum of all attributed slippage components"""
        return sum(comp.slippage_bps for comp in self.slippage_components)
    
    @property
    def attribution_accuracy(self) -> float:
        """How well we explained the total slippage"""
        if self.total_slippage_bps == 0:
            return 1.0
        return min(1.0, abs(self.total_attributed_slippage) / abs(self.total_slippage_bps))


class SlippageAnalyzer:
    """
    Advanced slippage attribution and analysis system
    """
    
    def __init__(self):
        self.attribution_history = []
        self.market_context_cache = {}
        
        # Attribution model parameters
        self.impact_sensitivity = 0.1      # Market impact per unit size
        self.timing_decay_rate = 0.02      # Timing cost per second
        self.volatility_multiplier = 1.5   # Volatility impact multiplier
        
        # Benchmarks for comparison
        self.theoretical_spread_cost = {    # Theoretical spread costs by pair type
            "major": 2.0,    # BTC, ETH major pairs
            "minor": 5.0,    # Other established cryptos
            "micro": 15.0    # Small cap tokens
        }
        
    def analyze_slippage(self, 
                        order_result: Any,  # OrderResult from execution_simulator
                        market_data: Dict[str, Any],
                        signal_context: Dict[str, Any] = None) -> SlippageAttribution:
        """Perform comprehensive slippage attribution analysis"""
        try:
            # Extract order details
            order_id = order_result.order_id
            timestamp = order_result.completion_time or datetime.now()
            pair = order_result.request.pair
            side = order_result.request.side
            order_size = order_result.request.size
            
            # Calculate total slippage
            expected_price = order_result.expected_price
            realized_price = order_result.avg_fill_price
            total_slippage_bps = order_result.realized_slippage_bps
            
            # Gather market context
            market_context = self._gather_market_context(pair, timestamp, market_data)
            
            # Perform component attribution
            components = self._attribute_slippage_components(
                order_result, market_context, signal_context
            )
            
            # Calculate performance metrics
            efficiency = self._calculate_execution_efficiency(order_result, market_context)
            alpha_preserved = self._calculate_alpha_preservation(order_result, signal_context)
            benchmark_cost = self._calculate_benchmark_cost(order_result, market_context)
            
            attribution = SlippageAttribution(
                order_id=order_id,
                timestamp=timestamp,
                pair=pair,
                side=side,
                order_size=order_size,
                entry_price=expected_price,
                actual_entry_price=realized_price,
                total_slippage_bps=total_slippage_bps,
                slippage_components=components,
                execution_efficiency=efficiency,
                alpha_preserved_pct=alpha_preserved,
                cost_vs_benchmark_bps=benchmark_cost,
                volatility_at_execution=market_context.get("volatility", 0.02),
                spread_at_execution_bps=market_context.get("spread_bps", 10.0),
                market_stress_level=market_context.get("stress_level", 1.0)
            )
            
            # Store in history
            self.attribution_history.append(attribution)
            
            # Log significant slippage
            if abs(total_slippage_bps) > 20:  # More than 20 bps
                logger.warning(f"High slippage detected: {order_id} - {total_slippage_bps:.1f} bps")
            
            return attribution
            
        except Exception as e:
            logger.error(f"Slippage analysis failed: {e}")
            return self._create_empty_attribution(order_result)
    
    def _gather_market_context(self, 
                              pair: str,
                              timestamp: datetime,
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather market context for slippage attribution"""
        try:
            pair_data = market_data.get(pair, {})
            
            context = {
                "price": pair_data.get("price", 0),
                "bid": pair_data.get("bid", 0),
                "ask": pair_data.get("ask", 0),
                "volume_24h": pair_data.get("volume_24h", 0),
                "volatility": pair_data.get("volatility_24h", 0.02),
                "liquidity_score": pair_data.get("liquidity_score", 0.5),
                "market_cap": pair_data.get("market_cap", 0),
                "recent_trades": pair_data.get("recent_trades", [])
            }
            
            # Calculate derived metrics
            if context["bid"] > 0 and context["ask"] > 0:
                mid_price = (context["bid"] + context["ask"]) / 2
                context["spread_bps"] = (context["ask"] - context["bid"]) / mid_price * 10000
            else:
                context["spread_bps"] = 20.0  # Default spread
            
            # Market stress indicators
            context["stress_level"] = self._calculate_market_stress(context)
            
            # Pair classification
            context["pair_type"] = self._classify_pair_type(pair)
            
            # Cache for future use
            self.market_context_cache[f"{pair}_{timestamp.timestamp()}"] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Market context gathering failed: {e}")
            return {"stress_level": 1.0, "spread_bps": 20.0, "volatility": 0.02, "pair_type": "minor"}
    
    def _attribute_slippage_components(self, 
                                     order_result: Any,
                                     market_context: Dict[str, Any],
                                     signal_context: Dict[str, Any] = None) -> List[SlippageComponent]:
        """Break down slippage into individual components"""
        components = []
        total_slippage = order_result.realized_slippage_bps
        
        try:
            # Market Impact Component
            market_impact = self._analyze_market_impact(order_result, market_context)
            if abs(market_impact) > 0.5:  # Threshold for significance
                components.append(SlippageComponent(
                    source=SlippageSource.MARKET_IMPACT,
                    slippage_bps=market_impact,
                    contribution_pct=abs(market_impact) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.8,
                    description=f"Price impact from {order_result.request.size:.2f} size order",
                    expected_value=0,
                    realized_value=market_impact,
                    market_context={"order_size": order_result.request.size}
                ))
            
            # Spread Cost Component
            spread_cost = self._analyze_spread_cost(order_result, market_context)
            if abs(spread_cost) > 0.5:
                components.append(SlippageComponent(
                    source=SlippageSource.SPREAD_COST,
                    slippage_bps=spread_cost,
                    contribution_pct=abs(spread_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.9,
                    description=f"Bid-ask spread crossing cost: {market_context.get('spread_bps', 0):.1f} bps",
                    expected_value=market_context.get('spread_bps', 0) / 2,
                    realized_value=spread_cost,
                    market_context={"spread_bps": market_context.get("spread_bps", 0)}
                ))
            
            # Timing Delay Component
            timing_cost = self._analyze_timing_cost(order_result, market_context)
            if abs(timing_cost) > 0.5:
                components.append(SlippageComponent(
                    source=SlippageSource.TIMING_DELAY,
                    slippage_bps=timing_cost,
                    contribution_pct=abs(timing_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.6,
                    description=f"Price movement during {order_result.execution_time_seconds:.1f}s execution",
                    expected_value=0,
                    realized_value=timing_cost,
                    market_context={"execution_time": order_result.execution_time_seconds}
                ))
            
            # Partial Fill Component
            partial_fill_cost = self._analyze_partial_fill_cost(order_result, market_context)
            if abs(partial_fill_cost) > 0.5:
                components.append(SlippageComponent(
                    source=SlippageSource.PARTIAL_FILLS,
                    slippage_bps=partial_fill_cost,
                    contribution_pct=abs(partial_fill_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.7,
                    description=f"Incomplete fill: {order_result.fill_rate:.1%} filled",
                    expected_value=order_result.request.size,
                    realized_value=order_result.total_filled_size,
                    market_context={"fill_rate": order_result.fill_rate}
                ))
            
            # Volatility Spike Component
            volatility_cost = self._analyze_volatility_impact(order_result, market_context)
            if abs(volatility_cost) > 0.5:
                components.append(SlippageComponent(
                    source=SlippageSource.VOLATILITY_SPIKE,
                    slippage_bps=volatility_cost,
                    contribution_pct=abs(volatility_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.5,
                    description=f"Market volatility impact: {market_context.get('volatility', 0):.1%}",
                    expected_value=0.02,  # 2% baseline volatility
                    realized_value=market_context.get('volatility', 0.02),
                    market_context={"volatility": market_context.get("volatility", 0.02)}
                ))
            
            # Liquidity Shortage Component
            liquidity_cost = self._analyze_liquidity_impact(order_result, market_context)
            if abs(liquidity_cost) > 0.5:
                components.append(SlippageComponent(
                    source=SlippageSource.LIQUIDITY_SHORTAGE,
                    slippage_bps=liquidity_cost,
                    contribution_pct=abs(liquidity_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                    confidence_score=0.6,
                    description=f"Limited liquidity impact: {market_context.get('liquidity_score', 0):.2f} score",
                    expected_value=0.8,  # Expected liquidity score
                    realized_value=market_context.get('liquidity_score', 0.5),
                    market_context={"liquidity_score": market_context.get("liquidity_score", 0.5)}
                ))
            
            # Market Stress Component
            if market_context.get("stress_level", 1.0) > 1.2:
                stress_cost = self._analyze_market_stress_impact(order_result, market_context)
                if abs(stress_cost) > 0.5:
                    components.append(SlippageComponent(
                        source=SlippageSource.MARKET_STRESS,
                        slippage_bps=stress_cost,
                        contribution_pct=abs(stress_cost) / abs(total_slippage) * 100 if total_slippage != 0 else 0,
                        confidence_score=0.4,
                        description=f"Market stress impact: {market_context.get('stress_level', 1.0):.1f}x normal",
                        expected_value=1.0,
                        realized_value=market_context.get('stress_level', 1.0),
                        market_context={"stress_level": market_context.get("stress_level", 1.0)}
                    ))
            
            return components
            
        except Exception as e:
            logger.error(f"Slippage component attribution failed: {e}")
            return []
    
    def _analyze_market_impact(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze market impact component"""
        try:
            order_size = order_result.request.size
            pair_type = market_context.get("pair_type", "minor")
            
            # Base impact based on order size and pair liquidity
            base_impact = order_size * self.impact_sensitivity
            
            # Adjust for pair type
            pair_multipliers = {"major": 0.5, "minor": 1.0, "micro": 2.0}
            impact_multiplier = pair_multipliers.get(pair_type, 1.0)
            
            # Adjust for market conditions
            stress_multiplier = market_context.get("stress_level", 1.0)
            liquidity_factor = max(0.1, market_context.get("liquidity_score", 0.5))
            
            market_impact = base_impact * impact_multiplier * stress_multiplier / liquidity_factor
            
            # Direction matters for impact
            if order_result.request.side == "sell":
                market_impact = -market_impact
            
            return market_impact
            
        except Exception as e:
            logger.error(f"Market impact analysis failed: {e}")
            return 0.0
    
    def _analyze_spread_cost(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze spread crossing cost"""
        try:
            if not order_result.fills:
                return 0.0
            
            # Check if order crossed spread (taker fills)
            taker_fills = [f for f in order_result.fills if f.fill_type.name == "TAKER"]
            
            if not taker_fills:
                return 0.0  # No spread cost for maker fills
            
            # Calculate average spread cost for taker fills
            total_taker_notional = sum(f.size * f.price for f in taker_fills)
            total_spread_cost = sum(f.spread_bps / 2 * f.size * f.price for f in taker_fills)
            
            if total_taker_notional == 0:
                return 0.0
            
            avg_spread_cost_bps = total_spread_cost / total_taker_notional * 10000
            
            return avg_spread_cost_bps
            
        except Exception as e:
            logger.error(f"Spread cost analysis failed: {e}")
            return 0.0
    
    def _analyze_timing_cost(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze timing delay cost"""
        try:
            execution_time = order_result.execution_time_seconds
            volatility = market_context.get("volatility", 0.02)
            
            # Timing cost based on volatility and execution time
            # Assumes price can move unfavorably during execution
            timing_cost_bps = execution_time * volatility * self.timing_decay_rate * 10000
            
            # Random direction - timing cost can be positive or negative
            import random
            direction = 1 if random.random() > 0.5 else -1
            
            # But bias towards adverse direction (Murphy's law)
            if random.random() < 0.6:  # 60% chance of adverse timing
                if order_result.request.side == "buy":
                    direction = 1   # Price went up while buying
                else:
                    direction = -1  # Price went down while selling
            
            return timing_cost_bps * direction
            
        except Exception as e:
            logger.error(f"Timing cost analysis failed: {e}")
            return 0.0
    
    def _analyze_partial_fill_cost(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze cost of partial fills"""
        try:
            fill_rate = order_result.fill_rate
            
            if fill_rate >= 0.98:  # Nearly complete fill
                return 0.0
            
            # Cost of not getting complete fill
            # Assumes missed opportunity or need for additional orders
            unfilled_portion = 1.0 - fill_rate
            
            # Cost scales with how much was unfilled and market conditions
            spread_bps = market_context.get("spread_bps", 10.0)
            stress_level = market_context.get("stress_level", 1.0)
            
            # Opportunity cost of unfilled portion
            opportunity_cost = unfilled_portion * spread_bps * stress_level * 0.5
            
            return opportunity_cost
            
        except Exception as e:
            logger.error(f"Partial fill cost analysis failed: {e}")
            return 0.0
    
    def _analyze_volatility_impact(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze volatility spike impact"""
        try:
            current_volatility = market_context.get("volatility", 0.02)
            baseline_volatility = 0.02  # 2% baseline
            
            if current_volatility <= baseline_volatility * 1.5:
                return 0.0  # No significant volatility spike
            
            # Calculate excess volatility impact
            excess_volatility = current_volatility - baseline_volatility
            volatility_impact = excess_volatility * self.volatility_multiplier * 10000
            
            # Volatility generally hurts execution
            if order_result.request.side == "buy":
                return volatility_impact
            else:
                return -volatility_impact
            
        except Exception as e:
            logger.error(f"Volatility impact analysis failed: {e}")
            return 0.0
    
    def _analyze_liquidity_impact(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze liquidity shortage impact"""
        try:
            liquidity_score = market_context.get("liquidity_score", 0.5)
            expected_liquidity = 0.8  # Expected liquidity score
            
            if liquidity_score >= expected_liquidity:
                return 0.0  # Good liquidity
            
            # Calculate liquidity shortage impact
            liquidity_deficit = expected_liquidity - liquidity_score
            order_size = order_result.request.size
            
            # Larger orders hurt more in low liquidity
            liquidity_impact = liquidity_deficit * order_size * 10  # 10 bps per unit deficit per size unit
            
            return liquidity_impact
            
        except Exception as e:
            logger.error(f"Liquidity impact analysis failed: {e}")
            return 0.0
    
    def _analyze_market_stress_impact(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Analyze overall market stress impact"""
        try:
            stress_level = market_context.get("stress_level", 1.0)
            baseline_stress = 1.0
            
            excess_stress = stress_level - baseline_stress
            
            # Market stress generally increases execution costs
            stress_impact = excess_stress * 20  # 20 bps per unit excess stress
            
            return stress_impact
            
        except Exception as e:
            logger.error(f"Market stress impact analysis failed: {e}")
            return 0.0
    
    def _calculate_market_stress(self, market_context: Dict[str, Any]) -> float:
        """Calculate market stress level"""
        try:
            # Simplified stress calculation
            volatility = market_context.get("volatility", 0.02)
            spread_bps = market_context.get("spread_bps", 10.0)
            volume_24h = market_context.get("volume_24h", 1000000)
            
            # Normalize factors
            vol_stress = min(3.0, volatility / 0.02)  # Relative to 2% baseline
            spread_stress = min(5.0, spread_bps / 10.0)  # Relative to 10 bps baseline
            volume_stress = max(0.2, min(2.0, 1000000 / max(volume_24h, 100000)))  # Inverse volume
            
            # Combine factors
            stress_level = (vol_stress + spread_stress + volume_stress) / 3
            
            return stress_level
            
        except Exception as e:
            logger.error(f"Market stress calculation failed: {e}")
            return 1.0
    
    def _classify_pair_type(self, pair: str) -> str:
        """Classify trading pair by liquidity tier"""
        if any(major in pair.upper() for major in ["BTC", "ETH", "USDT", "USDC"]):
            return "major"
        elif any(minor in pair.upper() for minor in ["ADA", "DOT", "LINK", "UNI", "AAVE"]):
            return "minor"
        else:
            return "micro"
    
    def _calculate_execution_efficiency(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Calculate overall execution efficiency"""
        try:
            # Components of execution efficiency
            fill_efficiency = order_result.fill_rate
            speed_efficiency = order_result.speed_score
            cost_efficiency = order_result.cost_score
            
            # Weight factors
            weights = [0.4, 0.3, 0.3]  # Fill rate most important
            
            efficiency = (
                fill_efficiency * weights[0] +
                speed_efficiency * weights[1] +
                cost_efficiency * weights[2]
            )
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Execution efficiency calculation failed: {e}")
            return 0.5
    
    def _calculate_alpha_preservation(self, order_result: Any, signal_context: Dict[str, Any]) -> float:
        """Calculate how much alpha was preserved during execution"""
        try:
            if not signal_context:
                return 0.5  # Default if no signal context
            
            # Get expected alpha from signal
            expected_alpha_bps = signal_context.get("expected_alpha_bps", 100)
            
            # Calculate alpha lost to execution costs
            total_execution_cost = abs(order_result.realized_slippage_bps) + (order_result.total_fees / (order_result.total_filled_size * order_result.avg_fill_price) * 10000 if order_result.total_filled_size > 0 else 0)
            
            # Alpha preserved = expected alpha - execution costs
            alpha_preserved_bps = expected_alpha_bps - total_execution_cost
            alpha_preservation_rate = alpha_preserved_bps / expected_alpha_bps if expected_alpha_bps > 0 else 0
            
            return max(0, min(1, alpha_preservation_rate))
            
        except Exception as e:
            logger.error(f"Alpha preservation calculation failed: {e}")
            return 0.5
    
    def _calculate_benchmark_cost(self, order_result: Any, market_context: Dict[str, Any]) -> float:
        """Calculate cost vs theoretical benchmark"""
        try:
            pair_type = market_context.get("pair_type", "minor")
            theoretical_cost = self.theoretical_spread_cost[pair_type]
            
            # Actual execution cost
            slippage_cost = abs(order_result.realized_slippage_bps)
            fee_cost = (order_result.total_fees / (order_result.total_filled_size * order_result.avg_fill_price) * 10000 
                       if order_result.total_filled_size > 0 and order_result.avg_fill_price > 0 else 0)
            
            actual_cost = slippage_cost + fee_cost
            
            # Cost vs benchmark
            excess_cost = actual_cost - theoretical_cost
            
            return excess_cost
            
        except Exception as e:
            logger.error(f"Benchmark cost calculation failed: {e}")
            return 0.0
    
    def get_slippage_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive slippage analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_attributions = [
                attr for attr in self.attribution_history
                if attr.timestamp >= cutoff_time
            ]
            
            if not recent_attributions:
                return {"status": "no_data"}
            
            # Overall statistics
            total_orders = len(recent_attributions)
            avg_slippage = np.mean([attr.total_slippage_bps for attr in recent_attributions])
            median_slippage = np.median([attr.total_slippage_bps for attr in recent_attributions])
            
            # Component analysis
            component_breakdown = {}
            for source in SlippageSource:
                source_contributions = []
                for attr in recent_attributions:
                    components = [c for c in attr.slippage_components if c.source == source]
                    total_contribution = sum(c.slippage_bps for c in components)
                    source_contributions.append(total_contribution)
                
                if source_contributions:
                    component_breakdown[source.value] = {
                        "avg_contribution_bps": np.mean(source_contributions),
                        "frequency": sum(1 for c in source_contributions if abs(c) > 0.5) / total_orders,
                        "max_impact_bps": max(source_contributions, key=abs)
                    }
            
            # Execution efficiency metrics
            efficiency_scores = [attr.execution_efficiency for attr in recent_attributions]
            alpha_preservation = [attr.alpha_preserved_pct for attr in recent_attributions]
            
            # Attribution quality
            attribution_accuracy = [attr.attribution_accuracy for attr in recent_attributions]
            
            analytics = {
                "analysis_period_days": days_back,
                "total_orders_analyzed": total_orders,
                "slippage_statistics": {
                    "avg_slippage_bps": avg_slippage,
                    "median_slippage_bps": median_slippage,
                    "slippage_volatility": np.std([attr.total_slippage_bps for attr in recent_attributions]),
                    "worst_slippage_bps": min([attr.total_slippage_bps for attr in recent_attributions]),
                    "best_slippage_bps": max([attr.total_slippage_bps for attr in recent_attributions])
                },
                "component_breakdown": component_breakdown,
                "execution_quality": {
                    "avg_efficiency": np.mean(efficiency_scores),
                    "avg_alpha_preservation": np.mean(alpha_preservation),
                    "efficiency_consistency": 1 - np.std(efficiency_scores),
                    "top_quartile_efficiency": np.percentile(efficiency_scores, 75)
                },
                "attribution_quality": {
                    "avg_attribution_accuracy": np.mean(attribution_accuracy),
                    "high_confidence_attributions": sum(1 for attr in recent_attributions if attr.attribution_accuracy > 0.8) / total_orders
                },
                "improvement_opportunities": self._identify_improvement_opportunities(component_breakdown)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Slippage analytics calculation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _identify_improvement_opportunities(self, component_breakdown: Dict[str, Any]) -> List[str]:
        """Identify top opportunities for execution improvement"""
        opportunities = []
        
        try:
            # Sort components by average contribution
            sorted_components = sorted(
                component_breakdown.items(),
                key=lambda x: abs(x[1]["avg_contribution_bps"]),
                reverse=True
            )
            
            for source, metrics in sorted_components[:3]:  # Top 3 contributors
                avg_impact = metrics["avg_contribution_bps"]
                frequency = metrics["frequency"]
                
                if abs(avg_impact) > 5 and frequency > 0.2:  # Significant and frequent
                    if source == "market_impact":
                        opportunities.append("Implement order splitting and TWAP execution")
                    elif source == "spread_cost":
                        opportunities.append("Increase post-only order usage to capture spread")
                    elif source == "timing_delay":
                        opportunities.append("Optimize order routing and reduce latency")
                    elif source == "partial_fills":
                        opportunities.append("Improve order size calibration and liquidity assessment")
                    elif source == "volatility_spike":
                        opportunities.append("Implement volatility-aware execution timing")
                    elif source == "liquidity_shortage":
                        opportunities.append("Enhanced liquidity detection and venue selection")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Improvement opportunities identification failed: {e}")
            return ["Review execution parameters and market timing"]
    
    def _create_empty_attribution(self, order_result: Any) -> SlippageAttribution:
        """Create empty attribution for failed analysis"""
        return SlippageAttribution(
            order_id=order_result.order_id,
            timestamp=datetime.now(),
            pair=order_result.request.pair,
            side=order_result.request.side,
            order_size=order_result.request.size,
            entry_price=0,
            actual_entry_price=0,
            total_slippage_bps=0,
            slippage_components=[],
            execution_efficiency=0,
            alpha_preserved_pct=0,
            cost_vs_benchmark_bps=0,
            volatility_at_execution=0.02,
            spread_at_execution_bps=10.0,
            market_stress_level=1.0
        )