"""
Return Attribution System for CryptoSmartTrader
Comprehensive PnL decomposition: alpha/fees/slippage/timing/sizing components.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from ..parity.execution_simulator import ExecutionResult
from ..analysis.backtest_parity import PerformanceAttribution


class AttributionPeriod(Enum):
    """Attribution analysis periods."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class AttributionComponent:
    """Individual attribution component."""
    name: str
    contribution_bps: float
    contribution_pct: float
    confidence: float
    data_points: int
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any]


@dataclass
class ReturnAttributionReport:
    """Comprehensive return attribution report."""
    period: AttributionPeriod
    period_start: datetime
    period_end: datetime
    
    # Total performance
    total_return_bps: float
    benchmark_return_bps: float
    excess_return_bps: float
    
    # Attribution components
    alpha_component: AttributionComponent
    fees_component: AttributionComponent
    slippage_component: AttributionComponent
    timing_component: AttributionComponent
    sizing_component: AttributionComponent
    market_impact_component: AttributionComponent
    
    # Summary metrics
    explained_variance_pct: float
    attribution_confidence: float
    execution_quality_score: float
    
    # Recommendations
    optimization_opportunities: List[str]
    execution_improvements: List[str]
    cost_reduction_suggestions: List[str]
    
    # Metadata
    observation_count: int
    data_quality_score: float
    report_generation_time: datetime


class ReturnAttributionAnalyzer:
    """
    Enterprise return attribution system.
    
    Decomposes portfolio returns into:
    - Alpha: Pure strategy performance
    - Fees: Trading and management costs  
    - Slippage: Market impact and timing costs
    - Timing: Execution timing differences
    - Sizing: Position sizing impacts
    - Market Impact: Large order impacts
    """
    
    def __init__(self):
        self.logger = get_logger("return_attribution")
        
        # Data storage
        self.attribution_history: List[ReturnAttributionReport] = []
        self.execution_data: List[ExecutionResult] = []
        self.performance_data: List[Dict[str, Any]] = []
        
        # Configuration
        self.fee_tier_schedule = {
            'maker': 0.0016,  # 16 bps maker fee
            'taker': 0.0026,  # 26 bps taker fee
            'vip': 0.0010     # 10 bps VIP fee
        }
        
        # Attribution models
        self.alpha_baseline_window = 30  # days
        self.execution_cost_threshold_bps = 5.0
        self.timing_threshold_ms = 1000
        
        self.logger.info("Return Attribution Analyzer initialized")
    
    def analyze_attribution(self,
                          portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          execution_results: List[ExecutionResult],
                          period: AttributionPeriod = AttributionPeriod.DAILY,
                          period_start: Optional[datetime] = None,
                          period_end: Optional[datetime] = None) -> ReturnAttributionReport:
        """
        Perform comprehensive return attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return time series
            benchmark_returns: Benchmark return time series  
            execution_results: List of execution results
            period: Attribution analysis period
            period_start: Analysis start date
            period_end: Analysis end date
            
        Returns:
            ReturnAttributionReport with full decomposition
        """
        # Set default period if not provided
        if period_end is None:
            period_end = datetime.utcnow()
        if period_start is None:
            if period == AttributionPeriod.DAILY:
                period_start = period_end - timedelta(days=1)
            elif period == AttributionPeriod.WEEKLY:
                period_start = period_end - timedelta(weeks=1)
            elif period == AttributionPeriod.MONTHLY:
                period_start = period_end - timedelta(days=30)
            else:
                period_start = period_end - timedelta(days=1)
        
        self.logger.info("Starting return attribution analysis",
                        period=period.value,
                        start=period_start.isoformat(),
                        end=period_end.isoformat())
        
        # Filter data to period
        period_portfolio_returns = self._filter_returns_to_period(
            portfolio_returns, period_start, period_end
        )
        period_benchmark_returns = self._filter_returns_to_period(
            benchmark_returns, period_start, period_end
        )
        period_executions = self._filter_executions_to_period(
            execution_results, period_start, period_end
        )
        
        # Calculate total performance
        total_return_bps = period_portfolio_returns.sum() * 10000
        benchmark_return_bps = period_benchmark_returns.sum() * 10000
        excess_return_bps = total_return_bps - benchmark_return_bps
        
        # Perform component attribution
        alpha_component = self._analyze_alpha_component(
            period_portfolio_returns, period_benchmark_returns, period_start, period_end
        )
        
        fees_component = self._analyze_fees_component(
            period_executions, total_return_bps, period_start, period_end
        )
        
        slippage_component = self._analyze_slippage_component(
            period_executions, total_return_bps, period_start, period_end
        )
        
        timing_component = self._analyze_timing_component(
            period_executions, period_portfolio_returns, period_start, period_end
        )
        
        sizing_component = self._analyze_sizing_component(
            period_executions, period_portfolio_returns, period_start, period_end
        )
        
        market_impact_component = self._analyze_market_impact_component(
            period_executions, total_return_bps, period_start, period_end
        )
        
        # Calculate summary metrics
        explained_variance = self._calculate_explained_variance([
            alpha_component, fees_component, slippage_component,
            timing_component, sizing_component, market_impact_component
        ], excess_return_bps)
        
        attribution_confidence = self._calculate_attribution_confidence([
            alpha_component, fees_component, slippage_component,
            timing_component, sizing_component, market_impact_component
        ])
        
        execution_quality_score = self._calculate_execution_quality_score(period_executions)
        
        # Generate recommendations
        optimization_opportunities = self._generate_optimization_opportunities([
            alpha_component, fees_component, slippage_component,
            timing_component, sizing_component, market_impact_component
        ])
        
        execution_improvements = self._generate_execution_improvements(period_executions)
        cost_reduction_suggestions = self._generate_cost_reduction_suggestions([
            fees_component, slippage_component, market_impact_component
        ])
        
        # Create report
        report = ReturnAttributionReport(
            period=period,
            period_start=period_start,
            period_end=period_end,
            total_return_bps=total_return_bps,
            benchmark_return_bps=benchmark_return_bps,
            excess_return_bps=excess_return_bps,
            alpha_component=alpha_component,
            fees_component=fees_component,
            slippage_component=slippage_component,
            timing_component=timing_component,
            sizing_component=sizing_component,
            market_impact_component=market_impact_component,
            explained_variance_pct=explained_variance,
            attribution_confidence=attribution_confidence,
            execution_quality_score=execution_quality_score,
            optimization_opportunities=optimization_opportunities,
            execution_improvements=execution_improvements,
            cost_reduction_suggestions=cost_reduction_suggestions,
            observation_count=len(period_executions),
            data_quality_score=self._calculate_data_quality_score(period_executions),
            report_generation_time=datetime.utcnow()
        )
        
        # Store report
        self.attribution_history.append(report)
        
        self.logger.info("Return attribution analysis complete",
                        total_return_bps=total_return_bps,
                        alpha_bps=alpha_component.contribution_bps,
                        fees_bps=fees_component.contribution_bps,
                        slippage_bps=slippage_component.contribution_bps,
                        execution_quality=execution_quality_score)
        
        return report
    
    def _filter_returns_to_period(self, returns: pd.Series, start: datetime, end: datetime) -> pd.Series:
        """Filter returns series to specified period."""
        if hasattr(returns.index, 'to_pydatetime'):
            return returns[(returns.index >= start) & (returns.index <= end)]
        else:
            # If no datetime index, return full series (for demo)
            return returns
    
    def _filter_executions_to_period(self, executions: List[ExecutionResult], 
                                   start: datetime, end: datetime) -> List[ExecutionResult]:
        """Filter executions to specified period."""
        return [
            exec for exec in executions
            if start <= exec.execution_time <= end
        ]
    
    def _analyze_alpha_component(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series,
                               period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze pure alpha component."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return AttributionComponent(
                name="Alpha",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=0.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'insufficient_data'}
            )
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        alpha_bps = float(excess_returns.sum() * 10000)
        
        # Calculate confidence based on statistical significance
        if len(excess_returns) > 10:
            t_stat = excess_returns.mean() / (excess_returns.std() / np.sqrt(len(excess_returns)))
            confidence = min(0.99, float(abs(t_stat) / 3.0))  # Rough confidence conversion
        else:
            confidence = 0.5
        
        return AttributionComponent(
            name="Alpha",
            contribution_bps=alpha_bps,
            contribution_pct=alpha_bps / (portfolio_returns.sum() * 10000) if portfolio_returns.sum() != 0 else 0.0,
            confidence=confidence,
            data_points=len(excess_returns),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'sharpe_ratio': excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0,
                'hit_rate': (excess_returns > 0).mean(),
                'max_drawdown': excess_returns.cumsum().expanding().max().sub(excess_returns.cumsum()).max()
            }
        )
    
    def _analyze_fees_component(self, executions: List[ExecutionResult],
                              total_return_bps: float,
                              period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze fee impact component."""
        if not executions:
            return AttributionComponent(
                name="Fees",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=1.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'no_executions'}
            )
        
        # Calculate total fees
        total_fees = sum(exec.total_fees for exec in executions)
        total_volume = sum(exec.executed_quantity * exec.avg_fill_price for exec in executions)
        
        # Convert to bps
        fees_bps = -(total_fees / total_volume * 10000) if total_volume > 0 else 0.0
        
        # Fee breakdown (estimate based on order type)
        maker_orders = [e for e in executions if e.order_type.value == 'limit']
        taker_orders = [e for e in executions if e.order_type.value == 'market']
        
        return AttributionComponent(
            name="Fees",
            contribution_bps=fees_bps,
            contribution_pct=fees_bps / total_return_bps if total_return_bps != 0 else 0.0,
            confidence=0.99,  # Fees are precisely known
            data_points=len(executions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'total_fees_usd': total_fees,
                'total_volume_usd': total_volume,
                'avg_fee_bps': fees_bps,
                'maker_orders': len(maker_orders),
                'taker_orders': len(taker_orders),
                'maker_ratio': len(maker_orders) / len(executions) if executions else 0.0
            }
        )
    
    def _analyze_slippage_component(self, executions: List[ExecutionResult],
                                  total_return_bps: float,
                                  period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze slippage impact component."""
        if not executions:
            return AttributionComponent(
                name="Slippage",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=0.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'no_executions'}
            )
        
        # Calculate weighted average slippage
        total_slippage_cost = 0.0
        total_volume = 0.0
        
        for exec in executions:
            volume = exec.executed_quantity * exec.avg_fill_price
            slippage_cost = exec.slippage_bps / 10000 * volume
            total_slippage_cost += slippage_cost
            total_volume += volume
        
        avg_slippage_bps = float(-(total_slippage_cost / total_volume * 10000)) if total_volume > 0 else 0.0
        
        # Slippage distribution analysis
        slippage_values = [exec.slippage_bps for exec in executions]
        
        return AttributionComponent(
            name="Slippage",
            contribution_bps=avg_slippage_bps,
            contribution_pct=avg_slippage_bps / total_return_bps if total_return_bps != 0 else 0.0,
            confidence=0.8,  # Slippage estimation has some uncertainty
            data_points=len(executions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'avg_slippage_bps': np.mean(slippage_values),
                'median_slippage_bps': np.median(slippage_values),
                'max_slippage_bps': np.max(slippage_values),
                'slippage_std_bps': np.std(slippage_values),
                'high_slippage_orders': len([s for s in slippage_values if s > 20]),
                'total_slippage_cost_usd': total_slippage_cost
            }
        )
    
    def _analyze_timing_component(self, executions: List[ExecutionResult],
                                portfolio_returns: pd.Series,
                                period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze timing impact component."""
        if not executions:
            return AttributionComponent(
                name="Timing",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=0.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'no_executions'}
            )
        
        # Estimate timing impact from execution latency
        latencies = [exec.latency_ms for exec in executions]
        avg_latency = np.mean(latencies)
        
        # Rough timing cost estimation: higher latency = higher timing cost
        # Assume 1ms latency = 0.1 bps timing cost (rough estimate)
        timing_cost_bps = -(avg_latency * 0.1)
        
        # Calculate timing efficiency
        fast_executions = len([l for l in latencies if l < 100])  # < 100ms
        timing_efficiency = fast_executions / len(executions)
        
        return AttributionComponent(
            name="Timing",
            contribution_bps=timing_cost_bps,
            contribution_pct=timing_cost_bps / portfolio_returns.sum() / 10000 if portfolio_returns.sum() != 0 else 0.0,
            confidence=0.6,  # Timing impact estimation is uncertain
            data_points=len(executions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'avg_latency_ms': avg_latency,
                'median_latency_ms': np.median(latencies),
                'max_latency_ms': np.max(latencies),
                'timing_efficiency': timing_efficiency,
                'slow_executions': len([l for l in latencies if l > 1000]),
                'timing_score': max(0, min(1, 1 - avg_latency / 1000))
            }
        )
    
    def _analyze_sizing_component(self, executions: List[ExecutionResult],
                                portfolio_returns: pd.Series,
                                period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze position sizing impact component."""
        if not executions:
            return AttributionComponent(
                name="Sizing",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=0.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'no_executions'}
            )
        
        # Estimate sizing impact from order size variance
        order_sizes = [exec.executed_quantity * exec.avg_fill_price for exec in executions]
        avg_order_size = np.mean(order_sizes)
        size_variance = np.var(order_sizes)
        
        # Rough sizing impact: high variance in sizing can hurt performance
        # Normalize by average size
        sizing_impact_bps = -(size_variance / (avg_order_size ** 2) * 100) if avg_order_size > 0 else 0.0
        
        return AttributionComponent(
            name="Sizing",
            contribution_bps=sizing_impact_bps,
            contribution_pct=sizing_impact_bps / portfolio_returns.sum() / 10000 if portfolio_returns.sum() != 0 else 0.0,
            confidence=0.5,  # Sizing impact is very difficult to estimate
            data_points=len(executions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'avg_order_size_usd': avg_order_size,
                'order_size_std': np.std(order_sizes),
                'size_coefficient_of_variation': np.std(order_sizes) / avg_order_size if avg_order_size > 0 else 0.0,
                'large_orders': len([s for s in order_sizes if s > avg_order_size * 2]),
                'small_orders': len([s for s in order_sizes if s < avg_order_size * 0.5])
            }
        )
    
    def _analyze_market_impact_component(self, executions: List[ExecutionResult],
                                       total_return_bps: float,
                                       period_start: datetime, period_end: datetime) -> AttributionComponent:
        """Analyze market impact component."""
        if not executions:
            return AttributionComponent(
                name="Market Impact",
                contribution_bps=0.0,
                contribution_pct=0.0,
                confidence=0.0,
                data_points=0,
                period_start=period_start,
                period_end=period_end,
                metadata={'reason': 'no_executions'}
            )
        
        # Market impact is typically embedded in slippage, but we can estimate
        # separate component based on order size relative to market
        large_orders = [exec for exec in executions 
                       if exec.executed_quantity * exec.avg_fill_price > 10000]  # >$10k orders
        
        if large_orders:
            # Estimate additional impact for large orders
            large_order_impact = len(large_orders) * 2.0  # 2 bps per large order
            market_impact_bps = -large_order_impact
        else:
            market_impact_bps = 0.0
        
        return AttributionComponent(
            name="Market Impact",
            contribution_bps=market_impact_bps,
            contribution_pct=market_impact_bps / total_return_bps if total_return_bps != 0 else 0.0,
            confidence=0.4,  # Market impact is difficult to isolate
            data_points=len(executions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                'large_orders_count': len(large_orders),
                'large_orders_ratio': len(large_orders) / len(executions) if executions else 0.0,
                'avg_large_order_size': np.mean([e.executed_quantity * e.avg_fill_price for e in large_orders]) if large_orders else 0.0,
                'market_impact_estimate_bps': market_impact_bps
            }
        )
    
    def _calculate_explained_variance(self, components: List[AttributionComponent], 
                                    total_excess_return_bps: float) -> float:
        """Calculate how much of total return is explained by components."""
        total_attributed_bps = sum(comp.contribution_bps for comp in components)
        
        if total_excess_return_bps == 0:
            return 0.0
        
        explained_ratio = abs(total_attributed_bps / total_excess_return_bps)
        return min(1.0, explained_ratio) * 100  # Convert to percentage
    
    def _calculate_attribution_confidence(self, components: List[AttributionComponent]) -> float:
        """Calculate overall attribution confidence."""
        if not components:
            return 0.0
        
        # Weight confidence by absolute contribution
        total_abs_contribution = sum(abs(comp.contribution_bps) for comp in components)
        
        if total_abs_contribution == 0:
            return 0.0
        
        weighted_confidence = sum(
            comp.confidence * abs(comp.contribution_bps) / total_abs_contribution 
            for comp in components
        )
        
        return weighted_confidence
    
    def _calculate_execution_quality_score(self, executions: List[ExecutionResult]) -> float:
        """Calculate overall execution quality score."""
        if not executions:
            return 0.0
        
        # Factors: low slippage, low latency, high fill rate
        avg_slippage = np.mean([exec.slippage_bps for exec in executions])
        avg_latency = np.mean([exec.latency_ms for exec in executions])
        fill_rate = np.mean([exec.executed_quantity / exec.requested_quantity for exec in executions 
                           if exec.requested_quantity > 0])
        
        # Score components (0-1 scale)
        slippage_score = max(0, 1 - avg_slippage / 50)  # Good if < 50 bps
        latency_score = max(0, 1 - avg_latency / 1000)   # Good if < 1000ms  
        fill_score = fill_rate if fill_rate > 0 else 1.0
        
        # Weighted average
        quality_score = (slippage_score * 0.4 + latency_score * 0.3 + fill_score * 0.3)
        
        return float(quality_score)
    
    def _calculate_data_quality_score(self, executions: List[ExecutionResult]) -> float:
        """Calculate data quality score."""
        if not executions:
            return 0.0
        
        # Check for complete data
        complete_executions = len([
            exec for exec in executions
            if exec.executed_quantity > 0 and exec.avg_fill_price > 0
        ])
        
        completeness_score = complete_executions / len(executions)
        
        # Check for reasonable values
        reasonable_executions = len([
            exec for exec in executions
            if 0 <= exec.slippage_bps <= 1000 and 0 <= exec.latency_ms <= 10000
        ])
        
        reasonableness_score = reasonable_executions / len(executions)
        
        return (completeness_score + reasonableness_score) / 2
    
    def _generate_optimization_opportunities(self, components: List[AttributionComponent]) -> List[str]:
        """Generate optimization opportunities based on attribution."""
        opportunities = []
        
        for comp in components:
            if comp.contribution_bps < -10:  # Significant negative contribution
                if comp.name == "Fees":
                    opportunities.append(f"High fee impact ({comp.contribution_bps:.1f} bps): Optimize maker/taker ratio")
                elif comp.name == "Slippage":
                    opportunities.append(f"High slippage cost ({comp.contribution_bps:.1f} bps): Improve order routing")
                elif comp.name == "Timing":
                    opportunities.append(f"Timing cost detected ({comp.contribution_bps:.1f} bps): Reduce execution latency")
                elif comp.name == "Market Impact":
                    opportunities.append(f"Market impact cost ({comp.contribution_bps:.1f} bps): Break up large orders")
        
        # Alpha opportunities
        alpha_comp = next((c for c in components if c.name == "Alpha"), None)
        if alpha_comp and alpha_comp.contribution_bps < 5:
            opportunities.append("Low alpha generation: Review signal quality and model performance")
        
        return opportunities
    
    def _generate_execution_improvements(self, executions: List[ExecutionResult]) -> List[str]:
        """Generate execution improvement suggestions."""
        if not executions:
            return ["No execution data available for analysis"]
        
        improvements = []
        
        # Latency analysis
        avg_latency = np.mean([exec.latency_ms for exec in executions])
        if avg_latency > 500:
            improvements.append(f"High average latency ({avg_latency:.0f}ms): Optimize network connectivity")
        
        # Slippage analysis
        high_slippage_orders = [exec for exec in executions if exec.slippage_bps > 20]
        if len(high_slippage_orders) > len(executions) * 0.2:
            improvements.append("High slippage on 20%+ of orders: Review order types and timing")
        
        # Fill rate analysis
        partial_fills = [exec for exec in executions 
                        if exec.executed_quantity < exec.requested_quantity]
        if len(partial_fills) > len(executions) * 0.3:
            improvements.append("High partial fill rate: Adjust order sizes for market liquidity")
        
        return improvements
    
    def _generate_cost_reduction_suggestions(self, cost_components: List[AttributionComponent]) -> List[str]:
        """Generate cost reduction suggestions."""
        suggestions = []
        
        for comp in cost_components:
            if comp.contribution_bps < -5:  # Significant cost
                if comp.name == "Fees":
                    maker_ratio = comp.metadata.get('maker_ratio', 0.0)
                    if maker_ratio < 0.7:
                        suggestions.append(f"Increase maker ratio (currently {maker_ratio:.1%}): Use limit orders")
                
                elif comp.name == "Slippage":
                    avg_slippage = comp.metadata.get('avg_slippage_bps', 0.0)
                    if avg_slippage > 15:
                        suggestions.append("High slippage: Use smaller order sizes and better timing")
                
                elif comp.name == "Market Impact":
                    large_orders = comp.metadata.get('large_orders_count', 0)
                    if large_orders > 0:
                        suggestions.append(f"Break up {large_orders} large orders to reduce market impact")
        
        return suggestions
    
    def get_attribution_summary(self, period_days: int = 7) -> Dict[str, Any]:
        """Get attribution summary for specified period."""
        recent_reports = [
            report for report in self.attribution_history
            if (datetime.utcnow() - report.report_generation_time).days <= period_days
        ]
        
        if not recent_reports:
            return {'error': 'No attribution data available'}
        
        # Calculate averages
        avg_alpha = np.mean([r.alpha_component.contribution_bps for r in recent_reports])
        avg_fees = np.mean([r.fees_component.contribution_bps for r in recent_reports])
        avg_slippage = np.mean([r.slippage_component.contribution_bps for r in recent_reports])
        avg_timing = np.mean([r.timing_component.contribution_bps for r in recent_reports])
        
        return {
            'period_days': period_days,
            'reports_count': len(recent_reports),
            'avg_total_return_bps': np.mean([r.total_return_bps for r in recent_reports]),
            'avg_alpha_bps': avg_alpha,
            'avg_fees_bps': avg_fees,
            'avg_slippage_bps': avg_slippage,
            'avg_timing_bps': avg_timing,
            'avg_execution_quality': np.mean([r.execution_quality_score for r in recent_reports]),
            'top_opportunity': self._get_top_cost_component([float(avg_fees), float(avg_slippage), float(avg_timing)]),
            'cost_breakdown_pct': {
                'fees': abs(avg_fees) / (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) * 100 if (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) > 0 else 0,
                'slippage': abs(avg_slippage) / (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) * 100 if (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) > 0 else 0,
                'timing': abs(avg_timing) / (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) * 100 if (abs(avg_fees) + abs(avg_slippage) + abs(avg_timing)) > 0 else 0
            }
        }
    
    def _get_top_cost_component(self, costs: List[float]) -> str:
        """Identify the largest cost component."""
        cost_names = ['fees', 'slippage', 'timing']
        abs_costs = [abs(cost) for cost in costs]
        
        if max(abs_costs) == 0:
            return 'none'
        
        max_index = abs_costs.index(max(abs_costs))
        return cost_names[max_index]
    
    def save_attribution_report(self, report: ReturnAttributionReport, 
                              filepath: Optional[str] = None) -> str:
        """Save attribution report to file."""
        if filepath is None:
            reports_dir = Path("data/attribution_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(reports_dir / f"attribution_report_{report.period_start.strftime('%Y%m%d')}_{report.period.value}.json")
        
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        
        # Handle datetime serialization
        for key, value in report_dict.items():
            if isinstance(value, datetime):
                report_dict[key] = value.isoformat()
        
        # Handle nested datetime in components
        for component_name in ['alpha_component', 'fees_component', 'slippage_component',
                              'timing_component', 'sizing_component', 'market_impact_component']:
            if component_name in report_dict:
                comp = report_dict[component_name]
                comp['period_start'] = comp['period_start'].isoformat() if isinstance(comp['period_start'], datetime) else comp['period_start']
                comp['period_end'] = comp['period_end'].isoformat() if isinstance(comp['period_end'], datetime) else comp['period_end']
        
        try:
            with open(str(filepath), 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info("Attribution report saved", filepath=str(filepath))
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save attribution report: {e}")
            raise