#!/usr/bin/env python3
"""
FASE F - Backtest-Live Parity Validation System
Comprehensive tracking error monitoring with fees/partial fills/latency analysis
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ParityStatus(Enum):
    """Parity validation status"""
    PASS = "pass"
    WARN = "warn" 
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class OrderExecution:
    """Order execution details for parity comparison"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    latency_ms: float
    partial_fill: bool
    execution_source: str  # 'backtest' or 'live'
    order_id: str
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0


@dataclass
class ParityMetrics:
    """Parity comparison metrics"""
    tracking_error_bps: float
    fee_difference_bps: float
    latency_difference_ms: float
    partial_fill_rate_diff: float
    execution_quality_score: float
    total_orders_compared: int
    parity_status: ParityStatus


@dataclass
class ParityReport:
    """Daily parity validation report"""
    date: datetime
    symbol: str
    metrics: ParityMetrics
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    breach_events: List[Dict[str, Any]]


class BacktestLiveParityValidator:
    """
    FASE F Parity Validator
    Validates backtest-live execution parity with comprehensive tracking error analysis
    """
    
    def __init__(self, 
                 max_tracking_error_bps: float = 20.0,
                 max_fee_difference_bps: float = 5.0,
                 max_latency_difference_ms: float = 100.0):
        self.max_tracking_error_bps = max_tracking_error_bps
        self.max_fee_difference_bps = max_fee_difference_bps
        self.max_latency_difference_ms = max_latency_difference_ms
        
        # Storage for execution data
        self.backtest_executions: List[OrderExecution] = []
        self.live_executions: List[OrderExecution] = []
        
        # Parity reports storage
        self.reports_dir = Path("exports/parity_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert thresholds
        self.alert_thresholds = {
            'tracking_error_warning': 15.0,  # bps
            'tracking_error_critical': 25.0,  # bps
            'fee_deviation_warning': 3.0,  # bps
            'latency_deviation_warning': 75.0,  # ms
        }
        
        logger.info(f"Parity validator initialized - max tracking error: {max_tracking_error_bps} bps")
    
    def record_execution(self, execution: OrderExecution) -> None:
        """Record order execution for parity analysis"""
        if execution.execution_source == 'backtest':
            self.backtest_executions.append(execution)
        elif execution.execution_source == 'live':
            self.live_executions.append(execution)
        else:
            logger.warning(f"Unknown execution source: {execution.execution_source}")
    
    def calculate_tracking_error(self, 
                                backtest_returns: List[float], 
                                live_returns: List[float]) -> float:
        """Calculate tracking error between backtest and live returns"""
        if len(backtest_returns) != len(live_returns) or len(backtest_returns) == 0:
            logger.warning("Cannot calculate tracking error: mismatched or empty return arrays")
            return float('inf')
        
        # Convert to numpy arrays
        bt_returns = np.array(backtest_returns)
        live_returns = np.array(live_returns)
        
        # Calculate return differences
        return_diff = bt_returns - live_returns
        
        # Tracking error = standard deviation of return differences * sqrt(252) * 10000 (for bps)
        tracking_error_daily = np.std(return_diff, ddof=1) if len(return_diff) > 1 else 0.0
        tracking_error_annualized = tracking_error_daily * np.sqrt(252) * 10000  # bps
        
        return tracking_error_annualized
    
    def analyze_fee_differences(self, 
                               backtest_fees: List[float], 
                               live_fees: List[float]) -> Dict[str, float]:
        """Analyze fee differences between backtest and live execution"""
        if not backtest_fees or not live_fees:
            return {'mean_diff_bps': 0.0, 'max_diff_bps': 0.0, 'std_diff_bps': 0.0}
        
        bt_fees = np.array(backtest_fees)
        live_fees = np.array(live_fees)
        
        # Calculate fee differences in basis points
        fee_diff = (live_fees - bt_fees) * 10000  # bps
        
        return {
            'mean_diff_bps': np.mean(fee_diff),
            'max_diff_bps': np.max(np.abs(fee_diff)),
            'std_diff_bps': np.std(fee_diff, ddof=1) if len(fee_diff) > 1 else 0.0,
            'median_diff_bps': np.median(fee_diff)
        }
    
    def analyze_latency_differences(self, 
                                   backtest_latencies: List[float], 
                                   live_latencies: List[float]) -> Dict[str, float]:
        """Analyze latency differences between backtest and live execution"""
        if not backtest_latencies or not live_latencies:
            return {'mean_diff_ms': 0.0, 'max_diff_ms': 0.0, 'p95_diff_ms': 0.0}
        
        bt_latencies = np.array(backtest_latencies)
        live_latencies = np.array(live_latencies)
        
        # Calculate latency differences
        latency_diff = live_latencies - bt_latencies
        
        return {
            'mean_diff_ms': np.mean(latency_diff),
            'max_diff_ms': np.max(latency_diff),
            'p95_diff_ms': np.percentile(latency_diff, 95),
            'median_diff_ms': np.median(latency_diff)
        }
    
    def analyze_partial_fills(self, 
                             backtest_executions: List[OrderExecution], 
                             live_executions: List[OrderExecution]) -> Dict[str, float]:
        """Analyze partial fill rate differences"""
        bt_partial_rate = sum(1 for ex in backtest_executions if ex.partial_fill) / len(backtest_executions) if backtest_executions else 0.0
        live_partial_rate = sum(1 for ex in live_executions if ex.partial_fill) / len(live_executions) if live_executions else 0.0
        
        return {
            'backtest_partial_rate': bt_partial_rate,
            'live_partial_rate': live_partial_rate,
            'rate_difference': live_partial_rate - bt_partial_rate
        }
    
    def calculate_execution_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall execution quality score (0-100)"""
        # Base score
        score = 100.0
        
        # Penalize tracking error
        tracking_error = metrics.get('tracking_error_bps', 0.0)
        if tracking_error > self.alert_thresholds['tracking_error_warning']:
            score -= min(30.0, (tracking_error - self.alert_thresholds['tracking_error_warning']) * 2)
        
        # Penalize fee differences
        fee_diff = abs(metrics.get('fee_analysis', {}).get('mean_diff_bps', 0.0))
        if fee_diff > self.alert_thresholds['fee_deviation_warning']:
            score -= min(20.0, (fee_diff - self.alert_thresholds['fee_deviation_warning']) * 3)
        
        # Penalize latency differences
        latency_diff = metrics.get('latency_analysis', {}).get('mean_diff_ms', 0.0)
        if latency_diff > self.alert_thresholds['latency_deviation_warning']:
            score -= min(25.0, (latency_diff - self.alert_thresholds['latency_deviation_warning']) * 0.2)
        
        # Penalize partial fill rate differences
        partial_diff = abs(metrics.get('partial_fill_analysis', {}).get('rate_difference', 0.0))
        if partial_diff > 0.05:  # 5% difference
            score -= min(15.0, partial_diff * 200)
        
        return max(0.0, score)
    
    def generate_parity_report(self, 
                              symbol: str, 
                              date: Optional[datetime] = None) -> ParityReport:
        """Generate comprehensive parity validation report"""
        if date is None:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Filter executions for the specified date and symbol
        start_date = date
        end_date = date + timedelta(days=1)
        
        bt_executions = [
            ex for ex in self.backtest_executions 
            if ex.symbol == symbol and start_date <= ex.timestamp < end_date
        ]
        
        live_executions = [
            ex for ex in self.live_executions 
            if ex.symbol == symbol and start_date <= ex.timestamp < end_date
        ]
        
        if not bt_executions or not live_executions:
            logger.warning(f"Insufficient execution data for parity analysis: {symbol} on {date.date()}")
            return ParityReport(
                date=date,
                symbol=symbol,
                metrics=ParityMetrics(
                    tracking_error_bps=float('inf'),
                    fee_difference_bps=0.0,
                    latency_difference_ms=0.0,
                    partial_fill_rate_diff=0.0,
                    execution_quality_score=0.0,
                    total_orders_compared=0,
                    parity_status=ParityStatus.UNKNOWN
                ),
                detailed_analysis={},
                recommendations=[],
                breach_events=[]
            )
        
        # Calculate returns for tracking error
        bt_returns = [(ex.price * ex.quantity) for ex in bt_executions]
        live_returns = [(ex.price * ex.quantity) for ex in live_executions]
        
        # Normalize to same length (match by timestamp)
        bt_returns_matched, live_returns_matched = self._match_executions_by_time(
            bt_executions, live_executions
        )
        
        # Calculate metrics
        tracking_error = self.calculate_tracking_error(bt_returns_matched, live_returns_matched)
        
        fee_analysis = self.analyze_fee_differences(
            [ex.fee for ex in bt_executions],
            [ex.fee for ex in live_executions]
        )
        
        latency_analysis = self.analyze_latency_differences(
            [ex.latency_ms for ex in bt_executions],
            [ex.latency_ms for ex in live_executions]
        )
        
        partial_fill_analysis = self.analyze_partial_fills(bt_executions, live_executions)
        
        # Detailed analysis
        detailed_analysis = {
            'tracking_error_bps': tracking_error,
            'fee_analysis': fee_analysis,
            'latency_analysis': latency_analysis,
            'partial_fill_analysis': partial_fill_analysis,
            'execution_count': {
                'backtest': len(bt_executions),
                'live': len(live_executions)
            },
            'volume_analysis': {
                'backtest_total': sum(ex.quantity for ex in bt_executions),
                'live_total': sum(ex.quantity for ex in live_executions)
            }
        }
        
        # Calculate execution quality score
        execution_quality = self.calculate_execution_quality_score(detailed_analysis)
        
        # Determine parity status
        parity_status = self._determine_parity_status(tracking_error, fee_analysis, latency_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detailed_analysis, parity_status)
        
        # Identify breach events
        breach_events = self._identify_breach_events(detailed_analysis)
        
        # Create metrics object
        metrics = ParityMetrics(
            tracking_error_bps=tracking_error,
            fee_difference_bps=fee_analysis.get('mean_diff_bps', 0.0),
            latency_difference_ms=latency_analysis.get('mean_diff_ms', 0.0),
            partial_fill_rate_diff=partial_fill_analysis.get('rate_difference', 0.0),
            execution_quality_score=execution_quality,
            total_orders_compared=min(len(bt_executions), len(live_executions)),
            parity_status=parity_status
        )
        
        return ParityReport(
            date=date,
            symbol=symbol,
            metrics=metrics,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            breach_events=breach_events
        )
    
    def _match_executions_by_time(self, 
                                 bt_executions: List[OrderExecution], 
                                 live_executions: List[OrderExecution]) -> Tuple[List[float], List[float]]:
        """Match executions by timestamp for comparison"""
        # Simple matching by order (assumes similar execution timing)
        min_length = min(len(bt_executions), len(live_executions))
        
        bt_returns = [(ex.price * ex.quantity) for ex in bt_executions[:min_length]]
        live_returns = [(ex.price * ex.quantity) for ex in live_executions[:min_length]]
        
        return bt_returns, live_returns
    
    def _determine_parity_status(self, 
                               tracking_error: float, 
                               fee_analysis: Dict[str, float], 
                               latency_analysis: Dict[str, float]) -> ParityStatus:
        """Determine overall parity status"""
        if tracking_error == float('inf'):
            return ParityStatus.UNKNOWN
        
        # Check critical thresholds
        if (tracking_error > self.alert_thresholds['tracking_error_critical'] or
            abs(fee_analysis.get('mean_diff_bps', 0.0)) > self.max_fee_difference_bps or
            latency_analysis.get('mean_diff_ms', 0.0) > self.max_latency_difference_ms):
            return ParityStatus.FAIL
        
        # Check warning thresholds
        if (tracking_error > self.alert_thresholds['tracking_error_warning'] or
            abs(fee_analysis.get('mean_diff_bps', 0.0)) > self.alert_thresholds['fee_deviation_warning'] or
            latency_analysis.get('mean_diff_ms', 0.0) > self.alert_thresholds['latency_deviation_warning']):
            return ParityStatus.WARN
        
        return ParityStatus.PASS
    
    def _generate_recommendations(self, 
                                detailed_analysis: Dict[str, Any], 
                                parity_status: ParityStatus) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        tracking_error = detailed_analysis.get('tracking_error_bps', 0.0)
        fee_analysis = detailed_analysis.get('fee_analysis', {})
        latency_analysis = detailed_analysis.get('latency_analysis', {})
        
        if tracking_error > self.alert_thresholds['tracking_error_warning']:
            recommendations.append(
                f"High tracking error ({tracking_error:.1f} bps) - Review execution timing and market impact models"
            )
        
        if abs(fee_analysis.get('mean_diff_bps', 0.0)) > self.alert_thresholds['fee_deviation_warning']:
            recommendations.append(
                f"Fee model deviation ({fee_analysis.get('mean_diff_bps', 0.0):.1f} bps) - Calibrate fee estimation models"
            )
        
        if latency_analysis.get('mean_diff_ms', 0.0) > self.alert_thresholds['latency_deviation_warning']:
            recommendations.append(
                f"Latency deviation ({latency_analysis.get('mean_diff_ms', 0.0):.1f} ms) - Optimize execution latency modeling"
            )
        
        if parity_status == ParityStatus.FAIL:
            recommendations.append("CRITICAL: Disable automated trading until parity issues are resolved")
        
        return recommendations
    
    def _identify_breach_events(self, detailed_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific breach events for investigation"""
        breach_events = []
        
        tracking_error = detailed_analysis.get('tracking_error_bps', 0.0)
        if tracking_error > self.max_tracking_error_bps:
            breach_events.append({
                'type': 'tracking_error_breach',
                'severity': 'critical' if tracking_error > self.alert_thresholds['tracking_error_critical'] else 'high',
                'value': tracking_error,
                'threshold': self.max_tracking_error_bps,
                'timestamp': datetime.now().isoformat()
            })
        
        fee_diff = abs(detailed_analysis.get('fee_analysis', {}).get('mean_diff_bps', 0.0))
        if fee_diff > self.max_fee_difference_bps:
            breach_events.append({
                'type': 'fee_difference_breach',
                'severity': 'high',
                'value': fee_diff,
                'threshold': self.max_fee_difference_bps,
                'timestamp': datetime.now().isoformat()
            })
        
        return breach_events
    
    def save_report(self, report: ParityReport) -> Path:
        """Save parity report to file"""
        # Replace / with _ for safe filename
        safe_symbol = report.symbol.replace('/', '_')
        filename = f"parity_report_{safe_symbol}_{report.date.strftime('%Y%m%d')}.json"
        filepath = self.reports_dir / filename
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            'date': report.date.isoformat(),
            'symbol': report.symbol,
            'metrics': asdict(report.metrics),
            'detailed_analysis': report.detailed_analysis,
            'recommendations': report.recommendations,
            'breach_events': report.breach_events,
            'generated_at': datetime.now().isoformat()
        }
        
        # Handle enum serialization
        report_dict['metrics']['parity_status'] = report.metrics.parity_status.value
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Parity report saved: {filepath}")
        return filepath
    
    def get_daily_tracking_error(self, symbol: str, days: int = 7) -> List[float]:
        """Get daily tracking errors for the past N days"""
        tracking_errors = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            report = self.generate_parity_report(symbol, date)
            
            if report.metrics.tracking_error_bps != float('inf'):
                tracking_errors.append(report.metrics.tracking_error_bps)
        
        return tracking_errors
    
    def is_parity_within_threshold(self, symbol: str) -> bool:
        """Check if current parity is within acceptable thresholds"""
        report = self.generate_parity_report(symbol)
        return report.metrics.parity_status in [ParityStatus.PASS, ParityStatus.WARN]
    
    def get_parity_summary(self) -> Dict[str, Any]:
        """Get overall parity validation summary"""
        # Get unique symbols from recent executions
        recent_symbols = set()
        cutoff_time = datetime.now() - timedelta(days=1)
        
        for ex in self.backtest_executions + self.live_executions:
            if ex.timestamp > cutoff_time:
                recent_symbols.add(ex.symbol)
        
        summary = {
            'total_symbols_monitored': len(recent_symbols),
            'symbols_within_threshold': 0,
            'symbols_at_warning': 0,
            'symbols_failed': 0,
            'avg_tracking_error_bps': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        if not recent_symbols:
            return summary
        
        tracking_errors = []
        for symbol in recent_symbols:
            report = self.generate_parity_report(symbol)
            
            if report.metrics.parity_status == ParityStatus.PASS:
                summary['symbols_within_threshold'] += 1
            elif report.metrics.parity_status == ParityStatus.WARN:
                summary['symbols_at_warning'] += 1
            elif report.metrics.parity_status == ParityStatus.FAIL:
                summary['symbols_failed'] += 1
            
            if report.metrics.tracking_error_bps != float('inf'):
                tracking_errors.append(report.metrics.tracking_error_bps)
        
        if tracking_errors:
            summary['avg_tracking_error_bps'] = np.mean(tracking_errors)
        
        return summary


# Global parity validator instance
_parity_validator: Optional[BacktestLiveParityValidator] = None


def get_parity_validator() -> BacktestLiveParityValidator:
    """Get global parity validator instance"""
    global _parity_validator
    if _parity_validator is None:
        _parity_validator = BacktestLiveParityValidator()
    return _parity_validator


def reset_parity_validator():
    """Reset parity validator instance (for testing)"""
    global _parity_validator
    _parity_validator = None