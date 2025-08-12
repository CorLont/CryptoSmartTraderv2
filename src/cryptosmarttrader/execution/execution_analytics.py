"""
Execution Analytics Engine

Comprehensive analytics for execution quality measurement and optimization.
Tracks and analyzes execution performance across all strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExecutionGrade(Enum):
    """Execution quality grades"""
    A_PLUS = "A+"     # Exceptional execution
    A = "A"           # Excellent execution  
    B_PLUS = "B+"     # Good execution
    B = "B"           # Average execution
    C_PLUS = "C+"     # Below average
    C = "C"           # Poor execution
    D = "D"           # Very poor execution
    F = "F"           # Failed execution


@dataclass
class ExecutionMetrics:
    """Core execution performance metrics"""
    # Fill metrics
    maker_ratio: float                    # Percentage of maker fills
    avg_fill_vs_mid_bp: float            # Average fill improvement vs mid
    fill_rate: float                     # Percentage of orders filled
    
    # Cost metrics
    avg_fees_bp: float                   # Average fees in basis points
    total_slippage_bp: float            # Total slippage cost
    execution_cost_bp: float            # Total execution cost
    
    # Timing metrics
    avg_execution_time_minutes: float   # Average time to fill
    time_to_first_fill_minutes: float   # Time to first partial fill
    
    # Quality metrics
    execution_quality_score: float      # Overall quality score (0-1)
    alpha_preservation_rate: float      # Alpha preserved after costs
    market_impact_bp: float             # Estimated market impact
    
    # Volume metrics
    total_notional_usd: float           # Total notional traded
    avg_order_size_usd: float           # Average order size
    participation_rate: float           # Average participation rate


@dataclass
class ExecutionBenchmark:
    """Execution benchmarks for comparison"""
    period_start: datetime
    period_end: datetime
    
    # Benchmark metrics
    market_maker_ratio: float = 0.5     # Market average maker ratio
    market_avg_fees_bp: float = 20      # Market average fees
    market_avg_slippage_bp: float = 8   # Market average slippage
    
    # Performance thresholds
    excellent_maker_ratio: float = 0.8  # Excellent maker ratio threshold
    good_fill_improvement_bp: float = 3 # Good fill improvement threshold
    acceptable_slippage_bp: float = 15  # Acceptable slippage threshold


class ExecutionAnalytics:
    """
    Advanced execution analytics and performance measurement
    """
    
    def __init__(self):
        self.execution_records = []
        self.benchmarks = {}  # period -> ExecutionBenchmark
        self.performance_history = []
        
        # Grading weights
        self.grading_weights = {
            "maker_ratio": 0.25,
            "fill_improvement": 0.20,
            "cost_efficiency": 0.20,
            "execution_speed": 0.15,
            "alpha_preservation": 0.20
        }
        
    def record_execution(self, 
                        execution_data: Dict[str, Any]) -> None:
        """Record execution for analytics"""
        try:
            # Standardize execution record
            record = {
                "timestamp": execution_data.get("timestamp", datetime.now()),
                "pair": execution_data.get("pair", ""),
                "side": execution_data.get("side", ""),
                "strategy": execution_data.get("strategy", ""),
                "regime": execution_data.get("regime", ""),
                
                # Order details
                "intended_size": execution_data.get("intended_size", 0),
                "filled_size": execution_data.get("filled_size", 0),
                "average_price": execution_data.get("average_price", 0),
                "market_price_at_time": execution_data.get("market_price", 0),
                
                # Costs and fees
                "total_fees": execution_data.get("total_fees", 0),
                "fee_type": execution_data.get("fee_type", ""),
                "slippage_bp": execution_data.get("slippage_bp", 0),
                
                # Timing
                "execution_time_ms": execution_data.get("execution_time_ms", 0),
                "time_to_first_fill_ms": execution_data.get("time_to_first_fill_ms", 0),
                
                # Market conditions
                "spread_bp": execution_data.get("spread_bp", 0),
                "volume_participation": execution_data.get("volume_participation", 0),
                "market_volatility": execution_data.get("market_volatility", 0),
                
                # Quality indicators
                "partial_fills": execution_data.get("partial_fills", 0),
                "order_modifications": execution_data.get("order_modifications", 0),
                "cancellations": execution_data.get("cancellations", 0)
            }
            
            self.execution_records.append(record)
            
            # Keep only recent records (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            self.execution_records = [
                r for r in self.execution_records 
                if r["timestamp"] >= cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
    
    def calculate_metrics(self, 
                         pair: Optional[str] = None,
                         strategy: Optional[str] = None,
                         days_back: int = 7) -> ExecutionMetrics:
        """Calculate comprehensive execution metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter records
            filtered_records = [
                r for r in self.execution_records
                if (r["timestamp"] >= cutoff_time and
                    (pair is None or r["pair"] == pair) and
                    (strategy is None or r["strategy"] == strategy))
            ]
            
            if not filtered_records:
                return self._get_empty_metrics()
            
            # Calculate metrics
            metrics = self._calculate_core_metrics(filtered_records)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return self._get_empty_metrics()
    
    def grade_execution_performance(self, 
                                  metrics: ExecutionMetrics,
                                  benchmark: Optional[ExecutionBenchmark] = None) -> ExecutionGrade:
        """Grade execution performance"""
        try:
            if benchmark is None:
                benchmark = self._get_default_benchmark()
            
            score = 0.0
            
            # Maker ratio score
            maker_score = min(1.0, metrics.maker_ratio / benchmark.excellent_maker_ratio)
            score += self.grading_weights["maker_ratio"] * maker_score
            
            # Fill improvement score
            fill_improvement_score = max(0, min(1.0, metrics.avg_fill_vs_mid_bp / benchmark.good_fill_improvement_bp))
            score += self.grading_weights["fill_improvement"] * fill_improvement_score
            
            # Cost efficiency score (lower is better)
            total_cost = metrics.avg_fees_bp + metrics.total_slippage_bp
            cost_efficiency_score = max(0, 1 - total_cost / (benchmark.market_avg_fees_bp + benchmark.market_avg_slippage_bp))
            score += self.grading_weights["cost_efficiency"] * cost_efficiency_score
            
            # Execution speed score (faster is better, up to a point)
            speed_score = max(0, min(1.0, 1 - metrics.avg_execution_time_minutes / 30))  # 30 min baseline
            score += self.grading_weights["execution_speed"] * speed_score
            
            # Alpha preservation score
            alpha_score = metrics.alpha_preservation_rate
            score += self.grading_weights["alpha_preservation"] * alpha_score
            
            # Convert score to grade
            return self._score_to_grade(score)
            
        except Exception as e:
            logger.error(f"Performance grading failed: {e}")
            return ExecutionGrade.C
    
    def get_performance_report(self, 
                             days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Overall metrics
            overall_metrics = self.calculate_metrics(days_back=days_back)
            overall_grade = self.grade_execution_performance(overall_metrics)
            
            # Per-strategy breakdown
            strategies = set(r["strategy"] for r in self.execution_records if r["strategy"])
            strategy_performance = {}
            
            for strategy in strategies:
                strategy_metrics = self.calculate_metrics(strategy=strategy, days_back=days_back)
                strategy_grade = self.grade_execution_performance(strategy_metrics)
                
                strategy_performance[strategy] = {
                    "metrics": strategy_metrics,
                    "grade": strategy_grade.value,
                    "score": self._grade_to_score(strategy_grade)
                }
            
            # Per-pair breakdown
            pairs = set(r["pair"] for r in self.execution_records if r["pair"])
            pair_performance = {}
            
            for pair in list(pairs)[:10]:  # Top 10 pairs by volume
                pair_metrics = self.calculate_metrics(pair=pair, days_back=days_back)
                pair_grade = self.grade_execution_performance(pair_metrics)
                
                pair_performance[pair] = {
                    "metrics": pair_metrics,
                    "grade": pair_grade.value,
                    "notional_usd": pair_metrics.total_notional_usd
                }
            
            # Trend analysis
            trend_analysis = self._analyze_performance_trends(days_back)
            
            # Recommendations
            recommendations = self._generate_recommendations(overall_metrics, strategy_performance)
            
            return {
                "period_days": days_back,
                "overall_performance": {
                    "metrics": overall_metrics,
                    "grade": overall_grade.value,
                    "score": self._grade_to_score(overall_grade)
                },
                "strategy_performance": strategy_performance,
                "pair_performance": pair_performance,
                "trends": trend_analysis,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {"status": "Error", "error": str(e)}
    
    def get_execution_dashboard_data(self) -> Dict[str, Any]:
        """Get data for execution quality dashboard"""
        try:
            # Recent performance (last 24 hours)
            recent_metrics = self.calculate_metrics(days_back=1)
            recent_grade = self.grade_execution_performance(recent_metrics)
            
            # Key performance indicators
            kpis = {
                "maker_ratio": {
                    "value": recent_metrics.maker_ratio,
                    "target": 0.8,
                    "status": "good" if recent_metrics.maker_ratio >= 0.7 else "warning"
                },
                "avg_fees_bp": {
                    "value": recent_metrics.avg_fees_bp,
                    "target": 15.0,
                    "status": "good" if recent_metrics.avg_fees_bp <= 20 else "warning"
                },
                "fill_improvement_bp": {
                    "value": recent_metrics.avg_fill_vs_mid_bp,
                    "target": 3.0,
                    "status": "good" if recent_metrics.avg_fill_vs_mid_bp >= 2 else "warning"
                },
                "execution_cost_bp": {
                    "value": recent_metrics.execution_cost_bp,
                    "target": 25.0,
                    "status": "good" if recent_metrics.execution_cost_bp <= 30 else "warning"
                }
            }
            
            # Historical trend (7 days)
            daily_metrics = []
            for days_ago in range(7):
                start_date = datetime.now() - timedelta(days=days_ago+1)
                end_date = datetime.now() - timedelta(days=days_ago)
                
                daily_records = [
                    r for r in self.execution_records
                    if start_date <= r["timestamp"] < end_date
                ]
                
                if daily_records:
                    daily_metric = self._calculate_core_metrics(daily_records)
                    daily_metrics.append({
                        "date": start_date.date().isoformat(),
                        "maker_ratio": daily_metric.maker_ratio,
                        "avg_fees_bp": daily_metric.avg_fees_bp,
                        "execution_cost_bp": daily_metric.execution_cost_bp,
                        "notional_usd": daily_metric.total_notional_usd
                    })
            
            # Top performing strategies
            strategy_rankings = []
            strategies = set(r["strategy"] for r in self.execution_records if r["strategy"])
            
            for strategy in strategies:
                strategy_metrics = self.calculate_metrics(strategy=strategy, days_back=7)
                grade = self.grade_execution_performance(strategy_metrics)
                
                strategy_rankings.append({
                    "strategy": strategy,
                    "grade": grade.value,
                    "score": self._grade_to_score(grade),
                    "maker_ratio": strategy_metrics.maker_ratio,
                    "avg_cost_bp": strategy_metrics.execution_cost_bp
                })
            
            # Sort by score
            strategy_rankings.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "current_grade": recent_grade.value,
                "kpis": kpis,
                "daily_trends": daily_metrics[-7:],  # Last 7 days
                "strategy_rankings": strategy_rankings[:5],  # Top 5 strategies
                "total_executions_24h": len([
                    r for r in self.execution_records
                    if r["timestamp"] >= datetime.now() - timedelta(days=1)
                ]),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _calculate_core_metrics(self, records: List[Dict[str, Any]]) -> ExecutionMetrics:
        """Calculate core execution metrics from records"""
        try:
            if not records:
                return self._get_empty_metrics()
            
            # Fill metrics
            maker_fills = [r for r in records if r.get("fee_type") == "maker"]
            maker_ratio = len(maker_fills) / len(records)
            
            filled_records = [r for r in records if r.get("filled_size", 0) > 0]
            fill_rate = len(filled_records) / len(records) if records else 0
            
            # Fill improvement calculation
            fill_improvements = []
            for record in filled_records:
                if record.get("average_price", 0) > 0 and record.get("market_price_at_time", 0) > 0:
                    avg_price = record["average_price"]
                    market_price = record["market_price_at_time"]
                    
                    if record["side"] == "buy" and avg_price < market_price:
                        improvement_bp = (market_price - avg_price) / market_price * 10000
                        fill_improvements.append(improvement_bp)
                    elif record["side"] == "sell" and avg_price > market_price:
                        improvement_bp = (avg_price - market_price) / market_price * 10000
                        fill_improvements.append(improvement_bp)
            
            avg_fill_vs_mid_bp = np.mean(fill_improvements) if fill_improvements else 0
            
            # Cost metrics
            total_fees = sum(r.get("total_fees", 0) for r in records)
            total_notional = sum(
                r.get("filled_size", 0) * r.get("average_price", 0) 
                for r in records
            )
            avg_fees_bp = (total_fees / total_notional * 10000) if total_notional > 0 else 0
            
            slippages = [r.get("slippage_bp", 0) for r in records if r.get("slippage_bp")]
            total_slippage_bp = np.mean(slippages) if slippages else 0
            
            execution_cost_bp = avg_fees_bp + total_slippage_bp
            
            # Timing metrics
            execution_times = [r.get("execution_time_ms", 0) / (1000 * 60) for r in records]  # Convert to minutes
            avg_execution_time_minutes = np.mean(execution_times) if execution_times else 0
            
            first_fill_times = [
                r.get("time_to_first_fill_ms", 0) / (1000 * 60) 
                for r in records if r.get("time_to_first_fill_ms")
            ]
            time_to_first_fill_minutes = np.mean(first_fill_times) if first_fill_times else 0
            
            # Quality metrics
            execution_quality_score = self._calculate_quality_score(records)
            
            # Alpha preservation (simplified calculation)
            alpha_preservation_rate = max(0, 1 - execution_cost_bp / 50)  # Assume 50bp baseline alpha
            
            # Market impact estimation
            participation_rates = [r.get("volume_participation", 0) for r in records]
            avg_participation = np.mean(participation_rates) if participation_rates else 0
            market_impact_bp = min(20, avg_participation * 100)  # Simple impact model
            
            # Volume metrics
            avg_order_size_usd = total_notional / len(filled_records) if filled_records else 0
            
            return ExecutionMetrics(
                maker_ratio=maker_ratio,
                avg_fill_vs_mid_bp=avg_fill_vs_mid_bp,
                fill_rate=fill_rate,
                avg_fees_bp=avg_fees_bp,
                total_slippage_bp=total_slippage_bp,
                execution_cost_bp=execution_cost_bp,
                avg_execution_time_minutes=avg_execution_time_minutes,
                time_to_first_fill_minutes=time_to_first_fill_minutes,
                execution_quality_score=execution_quality_score,
                alpha_preservation_rate=alpha_preservation_rate,
                market_impact_bp=market_impact_bp,
                total_notional_usd=total_notional,
                avg_order_size_usd=avg_order_size_usd,
                participation_rate=avg_participation
            )
            
        except Exception as e:
            logger.error(f"Core metrics calculation failed: {e}")
            return self._get_empty_metrics()
    
    def _calculate_quality_score(self, records: List[Dict[str, Any]]) -> float:
        """Calculate overall execution quality score"""
        try:
            if not records:
                return 0.0
            
            quality_factors = []
            
            # Fill success rate
            fill_success_rate = len([r for r in records if r.get("filled_size", 0) > 0]) / len(records)
            quality_factors.append(fill_success_rate)
            
            # Low cancellation rate
            cancellation_rate = np.mean([r.get("cancellations", 0) for r in records])
            quality_factors.append(max(0, 1 - cancellation_rate / 3))  # Penalize > 3 cancellations
            
            # Low modification rate
            modification_rate = np.mean([r.get("order_modifications", 0) for r in records])
            quality_factors.append(max(0, 1 - modification_rate / 5))  # Penalize > 5 modifications
            
            # Spread efficiency
            spreads = [r.get("spread_bp", 0) for r in records if r.get("spread_bp")]
            spread_efficiency = max(0, 1 - np.mean(spreads) / 50) if spreads else 0.5
            quality_factors.append(spread_efficiency)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5
    
    def _score_to_grade(self, score: float) -> ExecutionGrade:
        """Convert numeric score to execution grade"""
        if score >= 0.95:
            return ExecutionGrade.A_PLUS
        elif score >= 0.90:
            return ExecutionGrade.A
        elif score >= 0.85:
            return ExecutionGrade.B_PLUS
        elif score >= 0.75:
            return ExecutionGrade.B
        elif score >= 0.65:
            return ExecutionGrade.C_PLUS
        elif score >= 0.55:
            return ExecutionGrade.C
        elif score >= 0.40:
            return ExecutionGrade.D
        else:
            return ExecutionGrade.F
    
    def _grade_to_score(self, grade: ExecutionGrade) -> float:
        """Convert execution grade to numeric score"""
        grade_scores = {
            ExecutionGrade.A_PLUS: 0.95,
            ExecutionGrade.A: 0.90,
            ExecutionGrade.B_PLUS: 0.85,
            ExecutionGrade.B: 0.75,
            ExecutionGrade.C_PLUS: 0.65,
            ExecutionGrade.C: 0.55,
            ExecutionGrade.D: 0.40,
            ExecutionGrade.F: 0.20
        }
        return grade_scores.get(grade, 0.50)
    
    def _get_default_benchmark(self) -> ExecutionBenchmark:
        """Get default execution benchmark"""
        return ExecutionBenchmark(
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now()
        )
    
    def _get_empty_metrics(self) -> ExecutionMetrics:
        """Get empty metrics for no-data scenarios"""
        return ExecutionMetrics(
            maker_ratio=0.0,
            avg_fill_vs_mid_bp=0.0,
            fill_rate=0.0,
            avg_fees_bp=0.0,
            total_slippage_bp=0.0,
            execution_cost_bp=0.0,
            avg_execution_time_minutes=0.0,
            time_to_first_fill_minutes=0.0,
            execution_quality_score=0.0,
            alpha_preservation_rate=0.0,
            market_impact_bp=0.0,
            total_notional_usd=0.0,
            avg_order_size_usd=0.0,
            participation_rate=0.0
        )
    
    def _analyze_performance_trends(self, days_back: int) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            # Calculate metrics for first and second half of period
            mid_point = days_back // 2
            
            early_metrics = self.calculate_metrics(days_back=days_back - mid_point)
            recent_metrics = self.calculate_metrics(days_back=mid_point)
            
            trends = {
                "maker_ratio": {
                    "direction": "improving" if recent_metrics.maker_ratio > early_metrics.maker_ratio else "declining",
                    "change": recent_metrics.maker_ratio - early_metrics.maker_ratio
                },
                "execution_cost": {
                    "direction": "improving" if recent_metrics.execution_cost_bp < early_metrics.execution_cost_bp else "declining",
                    "change": recent_metrics.execution_cost_bp - early_metrics.execution_cost_bp
                },
                "alpha_preservation": {
                    "direction": "improving" if recent_metrics.alpha_preservation_rate > early_metrics.alpha_preservation_rate else "declining",
                    "change": recent_metrics.alpha_preservation_rate - early_metrics.alpha_preservation_rate
                }
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    def _generate_recommendations(self, 
                                 overall_metrics: ExecutionMetrics,
                                 strategy_performance: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        try:
            recommendations = []
            
            # Maker ratio recommendations
            if overall_metrics.maker_ratio < 0.7:
                recommendations.append(
                    f"Increase maker ratio from {overall_metrics.maker_ratio:.1%} to target 80% "
                    "by using more post-only orders and improving price levels"
                )
            
            # Fee optimization
            if overall_metrics.avg_fees_bp > 20:
                recommendations.append(
                    f"Reduce average fees from {overall_metrics.avg_fees_bp:.1f}bp "
                    "by increasing maker fills and optimizing fee tiers"
                )
            
            # Execution cost
            if overall_metrics.execution_cost_bp > 30:
                recommendations.append(
                    f"Reduce total execution cost from {overall_metrics.execution_cost_bp:.1f}bp "
                    "through better timing and reduced slippage"
                )
            
            # Strategy optimization
            if strategy_performance:
                best_strategy = max(strategy_performance.items(), key=lambda x: x[1]["score"])
                worst_strategy = min(strategy_performance.items(), key=lambda x: x[1]["score"])
                
                if best_strategy[1]["score"] - worst_strategy[1]["score"] > 0.2:
                    recommendations.append(
                        f"Consider increasing allocation to {best_strategy[0]} strategy "
                        f"(grade: {best_strategy[1]['grade']}) and reducing {worst_strategy[0]} "
                        f"(grade: {worst_strategy[1]['grade']})"
                    )
            
            # Alpha preservation
            if overall_metrics.alpha_preservation_rate < 0.8:
                recommendations.append(
                    f"Improve alpha preservation from {overall_metrics.alpha_preservation_rate:.1%} "
                    "by reducing execution costs and improving timing"
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to analysis error"]