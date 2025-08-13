"""
Slippage Tracking and Analysis

Tracks realized vs expected slippage to optimize execution
and preserve alpha through better execution timing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SlippageMetrics:
    """Container for slippage analysis results"""
    symbol: str
    period_hours: int

    # Basic slippage statistics
    avg_slippage_bp: float
    median_slippage_bp: float
    p95_slippage_bp: float
    p99_slippage_bp: float

    # Slippage vs predictions
    avg_prediction_error_bp: float    # |realized - expected|
    prediction_accuracy: float        # % of predictions within 25% of actual

    # Size impact analysis
    size_impact_coefficient: float    # bp per $1k trade size
    size_impact_r_squared: float      # How well size explains slippage

    # Timing analysis
    best_execution_hours: List[str]   # Hours with lowest slippage
    worst_execution_hours: List[str]  # Hours with highest slippage

    # Market condition correlation
    volume_correlation: float         # Slippage vs volume correlation
    spread_correlation: float         # Slippage vs spread correlation
    volatility_correlation: float     # Slippage vs volatility correlation

    # Performance metrics
    alpha_preservation_rate: float    # % of trades with acceptable slippage
    total_slippage_cost_bp: float     # Total slippage cost over period


class SlippageTracker:
    """
    Tracks and analyzes execution slippage for optimization
    """

    def __init__(self,
                 max_history_days: int = 30,
                 acceptable_slippage_bp: float = 20.0):
        """
        Initialize slippage tracker

        Args:
            max_history_days: Days of slippage history to retain
            acceptable_slippage_bp: Threshold for acceptable slippage
        """
        self.max_history_days = max_history_days
        self.acceptable_slippage_bp = acceptable_slippage_bp

        # Slippage history: symbol -> list of execution records
        self.slippage_history = defaultdict(list)

    def record_execution(self,
                        symbol: str,
                        side: str,                    # 'buy' or 'sell'
                        intended_price: float,
                        executed_price: float,
                        trade_size_quote: float,
                        expected_slippage_bp: float,
                        execution_time: Optional[datetime] = None,
                        market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Record a trade execution for slippage analysis

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            intended_price: Price we intended to trade at (e.g., mid price)
            executed_price: Actual execution price
            trade_size_quote: Trade size in quote currency
            expected_slippage_bp: Pre-trade slippage estimate
            execution_time: When trade was executed
            market_conditions: Market state during execution

        Returns:
            Calculated slippage metrics
        """
        try:
            execution_time = execution_time or datetime.now()

            # Calculate realized slippage
            if side.lower() == 'buy':
                # For buys, paying more than intended = positive slippage
                slippage_bp = ((executed_price - intended_price) / intended_price) * 10000
            else:
                # For sells, receiving less than intended = positive slippage
                slippage_bp = ((intended_price - executed_price) / intended_price) * 10000

            # Calculate prediction error
            prediction_error_bp = abs(slippage_bp - expected_slippage_bp)

            # Record execution
            execution_record = {
                'timestamp': execution_time,
                'symbol': symbol,
                'side': side,
                'intended_price': intended_price,
                'executed_price': executed_price,
                'trade_size_quote': trade_size_quote,
                'expected_slippage_bp': expected_slippage_bp,
                'realized_slippage_bp': slippage_bp,
                'prediction_error_bp': prediction_error_bp,
                'market_conditions': market_conditions or {}
            }

            self.slippage_history[symbol].append(execution_record)

            # Cleanup old history
            self._cleanup_history(symbol)

            return {
                'realized_slippage_bp': slippage_bp,
                'prediction_error_bp': prediction_error_bp,
                'acceptable': slippage_bp <= self.acceptable_slippage_bp
            }

        except Exception as e:
            logger.error(f"Failed to record execution for {symbol}: {e}")
            return {
                'realized_slippage_bp': 999.0,
                'prediction_error_bp': 999.0,
                'acceptable': False
            }

    def analyze_slippage_patterns(self,
                                symbol: str,
                                period_hours: int = 24) -> Optional[SlippageMetrics]:
        """
        Analyze slippage patterns for execution optimization

        Args:
            symbol: Symbol to analyze
            period_hours: Analysis period in hours

        Returns:
            Comprehensive slippage analysis
        """
        try:
            if symbol not in self.slippage_history:
                return None

            # Get relevant data
            cutoff_time = datetime.now() - timedelta(hours=period_hours)
            relevant_data = [
                record for record in self.slippage_history[symbol]
                if record['timestamp'] >= cutoff_time
            ]

            if len(relevant_data) < 5:
                logger.warning(f"Insufficient slippage data for {symbol}: {len(relevant_data)} executions")
                return None

            # Convert to DataFrame for analysis
            df = pd.DataFrame(relevant_data)

            # Basic slippage statistics
            slippages = df['realized_slippage_bp'].values
            avg_slippage = np.mean(slippages)
            median_slippage = np.median(slippages)
            p95_slippage = np.percentile(slippages, 95)
            p99_slippage = np.percentile(slippages, 99)

            # Prediction accuracy
            prediction_errors = df['prediction_error_bp'].values
            avg_prediction_error = np.mean(prediction_errors)

            # Prediction accuracy: % within 25% of actual
            acceptable_predictions = np.sum(prediction_errors <= np.abs(slippages) * 0.25)
            prediction_accuracy = acceptable_predictions / len(prediction_errors)

            # Size impact analysis
            sizes = df['trade_size_quote'].values
            size_impact_coef, size_r_squared = self._analyze_size_impact(sizes, slippages)

            # Timing analysis
            df['hour'] = df['timestamp'].dt.hour
            if df['hour'].nunique() > 1:
                hourly_slippage = df.groupby('hour')['realized_slippage_bp'].mean()
                best_hours = hourly_slippage.nsmallest(min(3, len(hourly_slippage))).index.tolist()
                worst_hours = hourly_slippage.nlargest(min(3, len(hourly_slippage))).index.tolist()
            else:
                best_hours = worst_hours = [df['hour'].iloc[0]]

            best_execution_times = [f"{hour:02d}:00" for hour in best_hours]
            worst_execution_times = [f"{hour:02d}:00" for hour in worst_hours]

            # Market condition correlations
            volume_correlation = self._calculate_market_correlation(df, 'volume', slippages)
            spread_correlation = self._calculate_market_correlation(df, 'spread_bp', slippages)
            volatility_correlation = self._calculate_market_correlation(df, 'volatility', slippages)

            # Performance metrics
            alpha_preservation_rate = np.sum(slippages <= self.acceptable_slippage_bp) / len(slippages)
            total_slippage_cost = np.sum(slippages * sizes) / np.sum(sizes)  # Weighted average

            return SlippageMetrics(
                symbol=symbol,
                period_hours=period_hours,
                avg_slippage_bp=avg_slippage,
                median_slippage_bp=median_slippage,
                p95_slippage_bp=p95_slippage,
                p99_slippage_bp=p99_slippage,
                avg_prediction_error_bp=avg_prediction_error,
                prediction_accuracy=prediction_accuracy,
                size_impact_coefficient=size_impact_coef,
                size_impact_r_squared=size_r_squared,
                best_execution_hours=best_execution_times,
                worst_execution_hours=worst_execution_times,
                volume_correlation=volume_correlation,
                spread_correlation=spread_correlation,
                volatility_correlation=volatility_correlation,
                alpha_preservation_rate=alpha_preservation_rate,
                total_slippage_cost_bp=total_slippage_cost
            )

        except Exception as e:
            logger.error(f"Slippage analysis failed for {symbol}: {e}")
            return None

    def get_execution_recommendations(self,
                                    symbol: str,
                                    trade_size_quote: float,
                                    current_market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get execution recommendations based on historical slippage patterns

        Args:
            symbol: Symbol to trade
            trade_size_quote: Intended trade size
            current_market_conditions: Current market state

        Returns:
            Execution recommendations and risk assessment
        """
        try:
            # Analyze recent patterns
            analytics = self.analyze_slippage_patterns(symbol, 24)

            if analytics is None:
                return {
                    "recommended": False,
                    "reason": "Insufficient slippage history",
                    "expected_slippage_bp": 50.0,
                    "confidence": 0.0
                }

            # Estimate slippage for this trade size
            base_slippage = analytics.median_slippage_bp
            size_impact = analytics.size_impact_coefficient * (trade_size_quote / 1000)  # Per $1k
            estimated_slippage = base_slippage + size_impact

            # Adjust for current market conditions
            if current_market_conditions:
                estimated_slippage = self._adjust_for_market_conditions(
                    estimated_slippage, current_market_conditions, analytics
                )

            # Execution timing recommendation
            current_hour = datetime.now().hour
            hour_str = f"{current_hour:02d}:00"

            timing_quality = "good"
            if hour_str in analytics.worst_execution_hours:
                timing_quality = "poor"
                estimated_slippage *= 1.3  # Penalty for bad timing
            elif hour_str in analytics.best_execution_hours:
                timing_quality = "excellent"
                estimated_slippage *= 0.9  # Bonus for good timing

            # Overall recommendation
            recommended = (
                estimated_slippage <= self.acceptable_slippage_bp * 1.5 and
                analytics.alpha_preservation_rate > 0.6 and
                timing_quality != "poor"
            )

            # Confidence based on prediction accuracy and data quality
            confidence = min(1.0, analytics.prediction_accuracy *
                           min(1.0, len(self.slippage_history[symbol]) / 20))

            return {
                "recommended": recommended,
                "expected_slippage_bp": estimated_slippage,
                "confidence": confidence,
                "timing_quality": timing_quality,
                "size_impact_bp": size_impact,
                "alpha_preservation_rate": analytics.alpha_preservation_rate,
                "reason": self._generate_recommendation_reason(
                    estimated_slippage, timing_quality, analytics
                )
            }

        except Exception as e:
            logger.error(f"Failed to generate execution recommendations for {symbol}: {e}")
            return {
                "recommended": False,
                "reason": f"Analysis error: {e}",
                "expected_slippage_bp": 999.0,
                "confidence": 0.0
            }

    def get_slippage_summary(self,
                           symbols: Optional[List[str]] = None,
                           period_hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregate slippage summary across symbols

        Args:
            symbols: Symbols to include (None = all)
            period_hours: Analysis period

        Returns:
            Comprehensive slippage summary
        """
        try:
            if symbols is None:
                symbols = list(self.slippage_history.keys())

            # Collect data across symbols
            all_slippages = []
            all_prediction_errors = []
            symbol_summaries = {}

            for symbol in symbols:
                analytics = self.analyze_slippage_patterns(symbol, period_hours)

                if analytics:
                    # Get raw data
                    cutoff_time = datetime.now() - timedelta(hours=period_hours)
                    symbol_data = [
                        record for record in self.slippage_history[symbol]
                        if record['timestamp'] >= cutoff_time
                    ]

                    symbol_slippages = [r['realized_slippage_bp'] for r in symbol_data]
                    symbol_errors = [r['prediction_error_bp'] for r in symbol_data]

                    all_slippages.extend(symbol_slippages)
                    all_prediction_errors.extend(symbol_errors)

                    symbol_summaries[symbol] = {
                        'executions': len(symbol_data),
                        'avg_slippage_bp': analytics.avg_slippage_bp,
                        'p95_slippage_bp': analytics.p95_slippage_bp,
                        'alpha_preservation_rate': analytics.alpha_preservation_rate,
                        'prediction_accuracy': analytics.prediction_accuracy
                    }

            if not all_slippages:
                return {"status": "No slippage data available"}

            # Aggregate statistics
            summary = {
                "period_hours": period_hours,
                "total_executions": len(all_slippages),
                "symbols_analyzed": len(symbol_summaries),
                "aggregate_metrics": {
                    "avg_slippage_bp": np.mean(all_slippages),
                    "median_slippage_bp": np.median(all_slippages),
                    "p95_slippage_bp": np.percentile(all_slippages, 95),
                    "p99_slippage_bp": np.percentile(all_slippages, 99),
                    "avg_prediction_error_bp": np.mean(all_prediction_errors),
                    "overall_alpha_preservation_rate": np.sum([
                        s <= self.acceptable_slippage_bp for s in all_slippages
                    ]) / len(all_slippages)
                },
                "performance_categories": {
                    "excellent_execution": sum(1 for s in all_slippages if s <= 10),
                    "acceptable_execution": sum(1 for s in all_slippages if 10 < s <= self.acceptable_slippage_bp),
                    "poor_execution": sum(1 for s in all_slippages if s > self.acceptable_slippage_bp)
                },
                "symbol_details": symbol_summaries
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate slippage summary: {e}")
            return {"status": "Error", "error": str(e)}

    def _analyze_size_impact(self, sizes: np.ndarray, slippages: np.ndarray) -> Tuple[float, float]:
        """Analyze relationship between trade size and slippage"""
        try:
            if len(sizes) < 5 or np.std(sizes) == 0:
                return 0.0, 0.0

            # Linear regression: slippage = a * size + b
            size_coeffs = np.polyfit(sizes, slippages, 1)
            size_impact_coef = size_coeffs[0]  # bp per unit of trade size

            # Calculate R-squared
            predicted_slippages = np.polyval(size_coeffs, sizes)
            ss_res = np.sum((slippages - predicted_slippages) ** 2)
            ss_tot = np.sum((slippages - np.mean(slippages)) ** 2)

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))

            return float(size_impact_coef), float(r_squared)

        except Exception as e:
            logger.error(f"Size impact analysis failed: {e}")
            return 0.0, 0.0

    def _calculate_market_correlation(self, df: pd.DataFrame,
                                    condition_key: str,
                                    slippages: np.ndarray) -> float:
        """Calculate correlation between market condition and slippage"""
        try:
            # Extract market condition values
            condition_values = []
            for _, row in df.iterrows():
                market_conditions = row.get('market_conditions', {})
                if isinstance(market_conditions, dict):
                    condition_values.append(market_conditions.get(condition_key, 0))
                else:
                    condition_values.append(0)

            condition_array = np.array(condition_values)

            if len(condition_array) != len(slippages) or np.std(condition_array) == 0:
                return 0.0

            correlation = np.corrcoef(condition_array, slippages)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.error(f"Market correlation calculation failed: {e}")
            return 0.0

    def _adjust_for_market_conditions(self,
                                    base_slippage: float,
                                    current_conditions: Dict[str, float],
                                    analytics: SlippageMetrics) -> float:
        """Adjust slippage estimate based on current market conditions"""
        try:
            adjusted_slippage = base_slippage

            # Volume adjustment
            volume = current_conditions.get('volume', 0)
            if volume > 0 and analytics.volume_correlation != 0:
                # Normalize volume impact
                volume_factor = 1 + (analytics.volume_correlation * 0.1)
                adjusted_slippage *= volume_factor

            # Spread adjustment
            spread = current_conditions.get('spread_bp', 0)
            if spread > 0 and analytics.spread_correlation != 0:
                spread_factor = 1 + (analytics.spread_correlation * spread / 100)
                adjusted_slippage *= max(0.5, min(2.0, spread_factor))

            # Volatility adjustment
            volatility = current_conditions.get('volatility', 0)
            if volatility > 0 and analytics.volatility_correlation != 0:
                vol_factor = 1 + (analytics.volatility_correlation * volatility)
                adjusted_slippage *= max(0.8, min(1.5, vol_factor))

            return max(0, adjusted_slippage)

        except Exception as e:
            logger.error(f"Market condition adjustment failed: {e}")
            return base_slippage

    def _generate_recommendation_reason(self,
                                      estimated_slippage: float,
                                      timing_quality: str,
                                      analytics: SlippageMetrics) -> str:
        """Generate human-readable recommendation reason"""
        try:
            reasons = []

            if estimated_slippage <= 10:
                reasons.append("Low expected slippage")
            elif estimated_slippage <= self.acceptable_slippage_bp:
                reasons.append("Acceptable expected slippage")
            else:
                reasons.append("High expected slippage")

            if timing_quality == "excellent":
                reasons.append("optimal timing")
            elif timing_quality == "poor":
                reasons.append("poor timing")

            if analytics.alpha_preservation_rate > 0.8:
                reasons.append("strong historical performance")
            elif analytics.alpha_preservation_rate < 0.5:
                reasons.append("weak historical performance")

            return "; ".join(reasons)

        except Exception as e:
            logger.error(f"Reason generation failed: {e}")
            return "Analysis completed"

    def _cleanup_history(self, symbol: str) -> None:
        """Remove old history entries"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.max_history_days)

            self.slippage_history[symbol] = [
                record for record in self.slippage_history[symbol]
                if record['timestamp'] >= cutoff_time
            ]

        except Exception as e:
            logger.error(f"History cleanup failed for {symbol}: {e}")
