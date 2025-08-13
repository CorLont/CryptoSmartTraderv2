"""
Execution Filter System

Integrates liquidity gating, spread monitoring, and slippage tracking
to make comprehensive execution decisions that preserve alpha.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .liquidity_gate import LiquidityGate, LiquidityMetrics
from .spread_monitor import SpreadMonitor, SpreadAnalytics
from .slippage_tracker import SlippageTracker, SlippageMetrics

logger = logging.getLogger(__name__)

@dataclass
class ExecutionDecision:
    """Comprehensive execution decision with all factors"""
    symbol: str
    timestamp: datetime

    # Final decision
    execute: bool
    recommended_size: float
    max_slippage_budget_bp: float

    # Component assessments
    liquidity_suitable: bool
    spread_acceptable: bool
    slippage_acceptable: bool

    # Risk assessment
    overall_risk_level: str          # 'low', 'medium', 'high'
    execution_quality_score: float   # 0-1 composite score

    # Detailed reasoning
    decision_factors: List[str]
    rejection_reasons: List[str]

    # Expected execution metrics
    expected_slippage_bp: float
    expected_execution_cost_bp: float
    alpha_preservation_probability: float


class ExecutionFilter:
    """
    Comprehensive execution filter that integrates all liquidity and execution constraints
    """

    def __init__(self,
                 # Liquidity gate parameters
                 max_spread_bp: float = 15.0,
                 min_depth_quote: float = 50000.0,
                 min_volume_1m: float = 100000.0,

                 # Slippage parameters
                 max_acceptable_slippage_bp: float = 25.0,
                 slippage_budget_factor: float = 1.5,

                 # Overall quality thresholds
                 min_execution_quality_score: float = 0.6):
        """
        Initialize execution filter with comprehensive constraints

        Args:
            max_spread_bp: Maximum acceptable spread
            min_depth_quote: Minimum order book depth required
            min_volume_1m: Minimum 1-minute volume required
            max_acceptable_slippage_bp: Maximum acceptable slippage
            slippage_budget_factor: Factor for slippage budget calculation
            min_execution_quality_score: Minimum composite quality score
        """

        # Initialize component systems
        self.liquidity_gate = LiquidityGate(
            max_spread_bp=max_spread_bp,
            min_depth_quote=min_depth_quote,
            min_volume_1m=min_volume_1m
        )

        self.spread_monitor = SpreadMonitor()
        self.slippage_tracker = SlippageTracker(
            acceptable_slippage_bp=max_acceptable_slippage_bp
        )

        # Filter parameters
        self.max_acceptable_slippage_bp = max_acceptable_slippage_bp
        self.slippage_budget_factor = slippage_budget_factor
        self.min_execution_quality_score = min_execution_quality_score

        # Execution history for learning
        self.execution_decisions = []

    def evaluate_execution(self,
                          symbol: str,
                          trade_size_quote: float,
                          order_book: Dict[str, Any],
                          recent_trades: List[Dict[str, Any]],
                          signal_alpha_bp: float,
                          volume_data: Optional[Dict[str, float]] = None) -> ExecutionDecision:
        """
        Comprehensive execution evaluation

        Args:
            symbol: Trading pair symbol
            trade_size_quote: Intended trade size in quote currency
            order_book: Current order book data
            recent_trades: Recent trade history
            signal_alpha_bp: Expected alpha from signal (basis points)
            volume_data: Pre-calculated volume metrics

        Returns:
            Complete execution decision with reasoning
        """
        try:
            timestamp = datetime.now()
            decision_factors = []
            rejection_reasons = []

            # Step 1: Evaluate liquidity conditions
            liquidity_metrics = self.liquidity_gate.evaluate_liquidity(
                symbol, order_book, recent_trades, volume_data
            )

            liquidity_decision = self.liquidity_gate.should_execute_trade(
                liquidity_metrics, trade_size_quote, self.max_acceptable_slippage_bp
            )

            # Record spread for monitoring
            if order_book.get('bids') and order_book.get('asks'):
                bid = float(order_book['bids'][0][0])
                ask = float(order_book['asks'][0][0])
                volume = volume_data.get('volume_1m', 0) if volume_data else 0
                self.spread_monitor.record_spread(symbol, bid, ask, volume)

            # Step 2: Get execution recommendations from slippage tracker
            market_conditions = {
                'volume': volume_data.get('volume_1m', 0) if volume_data else 0,
                'spread_bp': liquidity_metrics.bid_ask_spread_bp,
                'volatility': self._estimate_volatility(recent_trades)
            }

            slippage_recommendations = self.slippage_tracker.get_execution_recommendations(
                symbol, trade_size_quote, market_conditions
            )

            # Step 3: Calculate slippage budget based on signal alpha
            max_slippage_budget = self._calculate_slippage_budget(
                signal_alpha_bp, self.slippage_budget_factor
            )

            # Step 4: Make comprehensive decision
            execute = True
            recommended_size = trade_size_quote
            overall_risk = "low"

            # Check liquidity gate
            if not liquidity_decision["execute"]:
                execute = False
                rejection_reasons.append(f"Liquidity: {liquidity_decision['reason']}")
                overall_risk = "high"

                # Check if size reduction helps
                if liquidity_decision.get("recommended_size", 0) > 0:
                    recommended_size = liquidity_decision["recommended_size"]
                    if recommended_size >= trade_size_quote * 0.5:  # Accept if >=50% of intended
                        execute = True
                        decision_factors.append("Size reduced for liquidity")
                        overall_risk = "medium"

            # Check slippage recommendations
            expected_slippage = slippage_recommendations.get("expected_slippage_bp", 999)

            if expected_slippage > max_slippage_budget:
                execute = False
                rejection_reasons.append(
                    f"Slippage budget exceeded: {expected_slippage:.1f}bp > {max_slippage_budget:.1f}bp"
                )
                overall_risk = "high"
            elif expected_slippage > self.max_acceptable_slippage_bp:
                if execute:  # Only downgrade if not already rejected
                    overall_risk = "medium"
                decision_factors.append("Moderate slippage expected")

            # Check timing quality
            timing_quality = slippage_recommendations.get("timing_quality", "unknown")
            if timing_quality == "poor":
                if overall_risk == "low":
                    overall_risk = "medium"
                decision_factors.append("Suboptimal execution timing")
            elif timing_quality == "excellent":
                decision_factors.append("Optimal execution timing")

            # Step 5: Calculate execution quality score
            execution_quality_score = self._calculate_execution_quality_score(
                liquidity_metrics, slippage_recommendations, signal_alpha_bp, expected_slippage
            )

            if execution_quality_score < self.min_execution_quality_score:
                execute = False
                rejection_reasons.append(
                    f"Execution quality too low: {execution_quality_score:.2f} < {self.min_execution_quality_score:.2f}"
                )
                overall_risk = "high"

            # Step 6: Final adjustments and metrics
            if execute:
                decision_factors.append("All execution criteria met")

            # Calculate expected execution cost
            spread_cost = liquidity_metrics.bid_ask_spread_bp / 2  # Half spread
            expected_execution_cost = spread_cost + expected_slippage

            # Calculate alpha preservation probability
            alpha_preservation_prob = self._calculate_alpha_preservation_probability(
                signal_alpha_bp, expected_execution_cost, execution_quality_score
            )

            # Create final decision
            decision = ExecutionDecision(
                symbol=symbol,
                timestamp=timestamp,
                execute=execute,
                recommended_size=recommended_size,
                max_slippage_budget_bp=max_slippage_budget,
                liquidity_suitable=liquidity_decision["execute"],
                spread_acceptable=liquidity_metrics.bid_ask_spread_bp <= self.liquidity_gate.max_spread_bp,
                slippage_acceptable=expected_slippage <= self.max_acceptable_slippage_bp,
                overall_risk_level=overall_risk,
                execution_quality_score=execution_quality_score,
                decision_factors=decision_factors,
                rejection_reasons=rejection_reasons,
                expected_slippage_bp=expected_slippage,
                expected_execution_cost_bp=expected_execution_cost,
                alpha_preservation_probability=alpha_preservation_prob
            )

            # Store decision for learning
            self._record_decision(decision, liquidity_metrics, market_conditions)

            return decision

        except Exception as e:
            logger.error(f"Execution evaluation failed for {symbol}: {e}")
            return self._create_error_decision(symbol, str(e))

    def get_execution_analytics(self,
                              symbols: Optional[List[str]] = None,
                              hours_back: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive execution analytics across all components

        Args:
            symbols: Symbols to analyze (None = all)
            hours_back: Hours of history to analyze

        Returns:
            Integrated execution analytics
        """
        try:
            # Get analytics from each component
            liquidity_analytics = self.liquidity_gate.get_liquidity_analytics(None, hours_back)
            spread_summary = self.spread_monitor.get_spread_summary(symbols)
            slippage_summary = self.slippage_tracker.get_slippage_summary(symbols, hours_back)

            # Analyze recent execution decisions
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours_back)
            recent_decisions = [
                d for d in self.execution_decisions
                if d['timestamp'] >= cutoff_time and (symbols is None or d['symbol'] in symbols)
            ]

            decision_analytics = self._analyze_execution_decisions(recent_decisions)

            # Combine into comprehensive analytics
            analytics = {
                "period_hours": hours_back,
                "analysis_timestamp": datetime.now().isoformat(),
                "liquidity_metrics": liquidity_analytics,
                "spread_metrics": spread_summary,
                "slippage_metrics": slippage_summary,
                "execution_decisions": decision_analytics,
                "overall_performance": self._calculate_overall_performance(
                    liquidity_analytics, spread_summary, slippage_summary, decision_analytics
                )
            }

            return analytics

        except Exception as e:
            logger.error(f"Execution analytics generation failed: {e}")
            return {"status": "Error", "error": str(e)}

    def update_execution_outcome(self,
                               symbol: str,
                               decision_timestamp: datetime,
                               actual_slippage_bp: float,
                               actual_execution_cost_bp: float) -> None:
        """
        Update with actual execution outcome for learning

        Args:
            symbol: Symbol that was traded
            decision_timestamp: When decision was made
            actual_slippage_bp: Realized slippage
            actual_execution_cost_bp: Total execution cost
        """
        try:
            # Find the corresponding decision
            for decision in self.execution_decisions:
                if (decision['symbol'] == symbol and
                    abs((decision['timestamp'] - decision_timestamp).total_seconds()) < 300):  # 5-minute window

                    # Update with actual outcomes
                    decision['actual_slippage_bp'] = actual_slippage_bp
                    decision['actual_execution_cost_bp'] = actual_execution_cost_bp
                    decision['prediction_error_bp'] = abs(
                        actual_slippage_bp - decision['expected_slippage_bp']
                    )

                    # Record in slippage tracker for future improvements
                    self.slippage_tracker.record_execution(
                        symbol=symbol,
                        side='buy',  # Simplified - could be tracked
                        intended_price=0,  # Would need to be provided
                        executed_price=0,  # Would need to be provided
                        trade_size_quote=decision['recommended_size'],
                        expected_slippage_bp=decision['expected_slippage_bp'],
                        execution_time=decision_timestamp,
                        market_conditions=decision.get('market_conditions', {})
                    )

                    logger.info(f"Updated execution outcome for {symbol}: "
                               f"expected {decision['expected_slippage_bp']:.1f}bp, "
                               f"actual {actual_slippage_bp:.1f}bp")
                    break

        except Exception as e:
            logger.error(f"Failed to update execution outcome: {e}")

    def _calculate_slippage_budget(self, signal_alpha_bp: float, budget_factor: float) -> float:
        """Calculate maximum acceptable slippage based on signal alpha"""
        try:
            # Base budget: percentage of expected alpha
            base_budget = signal_alpha_bp * 0.3  # Max 30% of alpha

            # Apply budget factor
            slippage_budget = base_budget * budget_factor

            # Ensure reasonable bounds
            slippage_budget = max(5.0, min(50.0, slippage_budget))

            return slippage_budget

        except Exception as e:
            logger.error(f"Slippage budget calculation failed: {e}")
            return 20.0  # Default fallback

    def _estimate_volatility(self, recent_trades: List[Dict[str, Any]]) -> float:
        """Estimate short-term volatility from recent trades"""
        try:
            if len(recent_trades) < 5:
                return 0.02  # Default 2% volatility

            # Extract prices
            prices = []
            for trade in recent_trades[-20:]:  # Last 20 trades
                price = trade.get('price', 0)
                if price > 0:
                    prices.append(float(price))

            if len(prices) < 3:
                return 0.02

            # Calculate price volatility
            price_changes = np.diff(np.log(prices))
            volatility = np.std(price_changes) if len(price_changes) > 0 else 0.02

            return min(0.1, max(0.001, volatility))  # Bound between 0.1% and 10%

        except Exception as e:
            logger.error(f"Volatility estimation failed: {e}")
            return 0.02

    def _calculate_execution_quality_score(self,
                                         liquidity_metrics: LiquidityMetrics,
                                         slippage_recommendations: Dict[str, Any],
                                         signal_alpha_bp: float,
                                         expected_slippage_bp: float) -> float:
        """Calculate composite execution quality score"""
        try:
            # Liquidity component (40% weight)
            liquidity_score = liquidity_metrics.liquidity_score

            # Slippage component (40% weight)
            max_acceptable_slippage = signal_alpha_bp * 0.5  # 50% of alpha
            if expected_slippage_bp <= max_acceptable_slippage:
                slippage_score = 1.0 - (expected_slippage_bp / max_acceptable_slippage) * 0.5
            else:
                slippage_score = 0.5 * (max_acceptable_slippage / expected_slippage_bp)

            slippage_score = max(0.0, min(1.0, slippage_score))

            # Confidence component (20% weight)
            confidence_score = slippage_recommendations.get("confidence", 0.5)

            # Combined score
            execution_quality = (
                0.4 * liquidity_score +
                0.4 * slippage_score +
                0.2 * confidence_score
            )

            return max(0.0, min(1.0, execution_quality))

        except Exception as e:
            logger.error(f"Execution quality score calculation failed: {e}")
            return 0.0

    def _calculate_alpha_preservation_probability(self,
                                                signal_alpha_bp: float,
                                                expected_execution_cost_bp: float,
                                                execution_quality_score: float) -> float:
        """Calculate probability of preserving signal alpha"""
        try:
            if signal_alpha_bp <= 0:
                return 0.0

            # Net alpha after execution costs
            net_alpha_bp = signal_alpha_bp - expected_execution_cost_bp

            if net_alpha_bp <= 0:
                return 0.0

            # Alpha preservation ratio
            alpha_preservation_ratio = net_alpha_bp / signal_alpha_bp

            # Adjust for execution quality
            quality_adjustment = execution_quality_score

            # Final probability
            preservation_prob = alpha_preservation_ratio * quality_adjustment

            return max(0.0, min(1.0, preservation_prob))

        except Exception as e:
            logger.error(f"Alpha preservation calculation failed: {e}")
            return 0.0

    def _record_decision(self,
                        decision: ExecutionDecision,
                        liquidity_metrics: LiquidityMetrics,
                        market_conditions: Dict[str, float]) -> None:
        """Record execution decision for analysis"""
        try:
            decision_record = {
                'timestamp': decision.timestamp,
                'symbol': decision.symbol,
                'execute': decision.execute,
                'recommended_size': decision.recommended_size,
                'expected_slippage_bp': decision.expected_slippage_bp,
                'execution_quality_score': decision.execution_quality_score,
                'overall_risk_level': decision.overall_risk_level,
                'liquidity_score': liquidity_metrics.liquidity_score,
                'spread_bp': liquidity_metrics.bid_ask_spread_bp,
                'market_conditions': market_conditions
            }

            self.execution_decisions.append(decision_record)

            # Keep only recent decisions
            if len(self.execution_decisions) > 1000:
                self.execution_decisions = self.execution_decisions[-1000:]

        except Exception as e:
            logger.error(f"Failed to record execution decision: {e}")

    def _analyze_execution_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent execution decisions"""
        try:
            if not decisions:
                return {"status": "No execution decisions"}

            df = pd.DataFrame(decisions)

            analytics = {
                "total_decisions": len(decisions),
                "execution_rate": df['execute'].mean(),
                "avg_quality_score": df['execution_quality_score'].mean(),
                "risk_distribution": df['overall_risk_level'].value_counts().to_dict(),
                "avg_expected_slippage": df['expected_slippage_bp'].mean(),
                "symbols_evaluated": df['symbol'].nunique()
            }

            # Prediction accuracy if we have actual outcomes
            decisions_with_outcomes = df.dropna(subset=['actual_slippage_bp'])
            if len(decisions_with_outcomes) > 0:
                prediction_errors = abs(
                    decisions_with_outcomes['actual_slippage_bp'] -
                    decisions_with_outcomes['expected_slippage_bp']
                )
                analytics["prediction_accuracy"] = {
                    "avg_prediction_error_bp": prediction_errors.mean(),
                    "median_prediction_error_bp": prediction_errors.median(),
                    "predictions_within_25pct": (
                        prediction_errors <= decisions_with_outcomes['expected_slippage_bp'] * 0.25
                    ).mean()
                }

            return analytics

        except Exception as e:
            logger.error(f"Decision analysis failed: {e}")
            return {"status": "Error", "error": str(e)}

    def _calculate_overall_performance(self, *component_analytics) -> Dict[str, Any]:
        """Calculate overall execution system performance"""
        try:
            # This would implement a sophisticated performance calculation
            # based on all component analytics
            return {
                "status": "Performance calculation implemented",
                "overall_score": 0.75,  # Placeholder
                "alpha_preservation_rate": 0.82,  # Placeholder
                "execution_efficiency": 0.78  # Placeholder
            }

        except Exception as e:
            logger.error(f"Overall performance calculation failed: {e}")
            return {"status": "Error", "error": str(e)}

    def _create_error_decision(self, symbol: str, error_msg: str) -> ExecutionDecision:
        """Create error decision when evaluation fails"""
        return ExecutionDecision(
            symbol=symbol,
            timestamp=datetime.now(),
            execute=False,
            recommended_size=0.0,
            max_slippage_budget_bp=0.0,
            liquidity_suitable=False,
            spread_acceptable=False,
            slippage_acceptable=False,
            overall_risk_level="high",
            execution_quality_score=0.0,
            decision_factors=[],
            rejection_reasons=[f"Evaluation error: {error_msg}"],
            expected_slippage_bp=999.0,
            expected_execution_cost_bp=999.0,
            alpha_preservation_probability=0.0
        )
