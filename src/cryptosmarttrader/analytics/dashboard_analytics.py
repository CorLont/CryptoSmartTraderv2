"""
Dashboard Analytics

Analytics backend for the Return Attribution dashboard with
real-time data processing and visualization support.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class HeatmapType(Enum):
    """Types of attribution heatmaps"""
    PAIR_REGIME = "pair_regime"
    PAIR_TIME = "pair_time"
    COMPONENT_TIME = "component_time"
    REGIME_TIME = "regime_time"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"


class TimeGranularity(Enum):
    """Time granularity options"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class RealtimeMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    
    # Current performance
    current_pnl_bps: float = 0.0
    current_sharpe: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Attribution breakdown
    alpha_contribution: float = 0.0
    cost_drag: float = 0.0
    execution_impact: float = 0.0
    
    # Quality metrics
    attribution_accuracy: float = 0.0
    data_quality_score: float = 1.0
    
    # Alert status
    active_alerts: int = 0
    critical_alerts: int = 0
    
    # System health
    system_status: str = "healthy"
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class AnalyticsCache:
    """Analytics data cache for performance optimization"""
    
    # Cached heatmaps
    heatmap_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Cached aggregations
    aggregation_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Cache metadata
    cache_timestamps: Dict[str, datetime] = field(default_factory=dict)
    cache_ttl_minutes: int = 15
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = datetime.now() - self.cache_timestamps[cache_key]
        return age.total_seconds() < (self.cache_ttl_minutes * 60)
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            self.heatmap_cache.clear()
            self.aggregation_cache.clear()
            self.cache_timestamps.clear()
        else:
            # Pattern-based invalidation
            keys_to_remove = [key for key in self.cache_timestamps.keys() if pattern in key]
            for key in keys_to_remove:
                self.heatmap_cache.pop(key, None)
                self.aggregation_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)


class DashboardAnalytics:
    """
    Advanced analytics backend for Return Attribution dashboard
    """
    
    def __init__(self, 
                 cache_ttl_minutes: int = 15,
                 max_data_points: int = 10000):
        
        self.cache = AnalyticsCache(cache_ttl_minutes=cache_ttl_minutes)
        self.max_data_points = max_data_points
        
        # Real-time metrics
        self.current_metrics = RealtimeMetrics(datetime.now())
        
        # Data processing executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Attribution color schemes
        self.color_schemes = {
            "attribution": {
                "alpha": "#2E8B57",      # Sea green (positive)
                "fees": "#DC143C",       # Crimson (negative)
                "slippage": "#FF4500",   # Orange red (negative)
                "timing": "#4169E1",     # Royal blue (variable)
                "sizing": "#9370DB",     # Medium purple (variable)
                "market_impact": "#FF6347"  # Tomato (negative)
            },
            "performance": {
                "excellent": "#006400",   # Dark green
                "good": "#32CD32",       # Lime green
                "neutral": "#FFD700",    # Gold
                "poor": "#FF4500",       # Orange red
                "critical": "#DC143C"    # Crimson
            }
        }
    
    def update_realtime_metrics(self, 
                               performance_data: Dict[str, Any],
                               attribution_result: Any = None,
                               alerts: List[Any] = None) -> RealtimeMetrics:
        """Update real-time metrics display"""
        try:
            timestamp = datetime.now()
            
            # Extract performance metrics
            current_pnl = performance_data.get('total_pnl_bps', 0.0)
            current_sharpe = performance_data.get('sharpe_ratio', 0.0)
            current_drawdown = performance_data.get('drawdown_pct', 0.0)
            
            # Extract attribution metrics
            alpha_contrib = 0.0
            cost_drag = 0.0
            execution_impact = 0.0
            attribution_accuracy = 0.0
            
            if attribution_result:
                alpha_contrib = attribution_result.alpha_contribution
                cost_drag = attribution_result.cost_drag
                execution_impact = cost_drag  # Simplified
                attribution_accuracy = attribution_result.attribution_accuracy
            
            # Alert metrics
            active_alerts = len(alerts) if alerts else 0
            critical_alerts = len([a for a in alerts if getattr(a, 'severity', None) == 'critical']) if alerts else 0
            
            # System status
            if critical_alerts > 0:
                system_status = "critical"
            elif active_alerts > 5:
                system_status = "degraded"
            elif active_alerts > 0:
                system_status = "monitoring"
            else:
                system_status = "healthy"
            
            self.current_metrics = RealtimeMetrics(
                timestamp=timestamp,
                current_pnl_bps=current_pnl,
                current_sharpe=current_sharpe,
                current_drawdown_pct=current_drawdown,
                alpha_contribution=alpha_contrib,
                cost_drag=cost_drag,
                execution_impact=execution_impact,
                attribution_accuracy=attribution_accuracy,
                data_quality_score=1.0,  # Would be calculated from data quality metrics
                active_alerts=active_alerts,
                critical_alerts=critical_alerts,
                system_status=system_status,
                last_update=timestamp
            )
            
            return self.current_metrics
            
        except Exception as e:
            logger.error(f"Real-time metrics update failed: {e}")
            return self.current_metrics
    
    def generate_attribution_heatmap(self,
                                   trade_data: pd.DataFrame,
                                   heatmap_type: HeatmapType = HeatmapType.PAIR_REGIME,
                                   time_granularity: TimeGranularity = TimeGranularity.DAY,
                                   days_back: int = 30) -> Dict[str, Any]:
        """Generate interactive attribution heatmap"""
        try:
            cache_key = f"heatmap_{heatmap_type.value}_{time_granularity.value}_{days_back}"
            
            # Check cache
            if self.cache.is_cache_valid(cache_key):
                return self.cache.heatmap_cache[cache_key]
            
            # Filter data to time period
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_data = trade_data[trade_data['timestamp'] >= cutoff_date].copy()
            
            if len(filtered_data) == 0:
                return self._create_empty_heatmap(heatmap_type)
            
            # Generate heatmap based on type
            if heatmap_type == HeatmapType.PAIR_REGIME:
                heatmap_data = self._create_pair_regime_heatmap(filtered_data)
            elif heatmap_type == HeatmapType.PAIR_TIME:
                heatmap_data = self._create_pair_time_heatmap(filtered_data, time_granularity)
            elif heatmap_type == HeatmapType.COMPONENT_TIME:
                heatmap_data = self._create_component_time_heatmap(filtered_data, time_granularity)
            elif heatmap_type == HeatmapType.REGIME_TIME:
                heatmap_data = self._create_regime_time_heatmap(filtered_data, time_granularity)
            else:
                heatmap_data = self._create_performance_attribution_heatmap(filtered_data)
            
            # Cache result
            self.cache.heatmap_cache[cache_key] = heatmap_data
            self.cache.cache_timestamps[cache_key] = datetime.now()
            
            return heatmap_data
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            return self._create_empty_heatmap(heatmap_type)
    
    def _create_pair_regime_heatmap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create pair vs regime attribution heatmap"""
        try:
            # Aggregate by pair and regime
            if 'pair' not in data.columns or 'regime' not in data.columns:
                return self._create_empty_heatmap(HeatmapType.PAIR_REGIME)
            
            pivot_data = data.groupby(['pair', 'regime'])['realized_pnl'].sum().unstack(fill_value=0)
            
            # Create plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                text=pivot_data.values,
                texttemplate="%{text:.1f}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Regime: %{x}<br>PnL: %{z:.1f} bps<extra></extra>'
            ))
            
            fig.update_layout(
                title="Return Attribution: Pair vs Regime",
                xaxis_title="Market Regime",
                yaxis_title="Trading Pair",
                height=max(400, len(pivot_data.index) * 25),
                width=800
            )
            
            return {
                "figure": fig.to_dict(),
                "data_summary": {
                    "total_pairs": len(pivot_data.index),
                    "total_regimes": len(pivot_data.columns),
                    "best_pair": pivot_data.sum(axis=1).idxmax() if not pivot_data.empty else "N/A",
                    "best_regime": pivot_data.sum(axis=0).idxmax() if not pivot_data.empty else "N/A",
                    "total_pnl": pivot_data.values.sum()
                }
            }
            
        except Exception as e:
            logger.error(f"Pair-regime heatmap creation failed: {e}")
            return self._create_empty_heatmap(HeatmapType.PAIR_REGIME)
    
    def _create_component_time_heatmap(self, 
                                     data: pd.DataFrame,
                                     time_granularity: TimeGranularity) -> Dict[str, Any]:
        """Create attribution component vs time heatmap"""
        try:
            # Add time grouping column
            data_copy = data.copy()
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            
            if time_granularity == TimeGranularity.HOUR:
                data_copy['time_group'] = data_copy['timestamp'].dt.strftime('%Y-%m-%d %H:00')
            elif time_granularity == TimeGranularity.DAY:
                data_copy['time_group'] = data_copy['timestamp'].dt.strftime('%Y-%m-%d')
            elif time_granularity == TimeGranularity.WEEK:
                data_copy['time_group'] = data_copy['timestamp'].dt.strftime('%Y-W%U')
            else:  # MONTH
                data_copy['time_group'] = data_copy['timestamp'].dt.strftime('%Y-%m')
            
            # Create attribution components matrix
            components = ['alpha', 'fees', 'slippage', 'timing', 'sizing']
            time_groups = sorted(data_copy['time_group'].unique())
            
            # Initialize matrix
            attribution_matrix = np.zeros((len(components), len(time_groups)))
            
            for i, component in enumerate(components):
                if f'{component}_contribution' in data_copy.columns:
                    time_series = data_copy.groupby('time_group')[f'{component}_contribution'].sum()
                    for j, time_group in enumerate(time_groups):
                        attribution_matrix[i, j] = time_series.get(time_group, 0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attribution_matrix,
                x=time_groups,
                y=components,
                colorscale='RdYlGn',
                text=attribution_matrix,
                texttemplate="%{text:.1f}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Contribution: %{z:.1f} bps<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Attribution Components Over Time ({time_granularity.value.title()})",
                xaxis_title=f"Time ({time_granularity.value.title()})",
                yaxis_title="Attribution Component",
                height=400,
                width=max(800, len(time_groups) * 40)
            )
            
            return {
                "figure": fig.to_dict(),
                "data_summary": {
                    "time_periods": len(time_groups),
                    "components": len(components),
                    "best_component": components[np.argmax(np.sum(attribution_matrix, axis=1))],
                    "worst_component": components[np.argmin(np.sum(attribution_matrix, axis=1))],
                    "total_attribution": float(np.sum(attribution_matrix))
                }
            }
            
        except Exception as e:
            logger.error(f"Component-time heatmap creation failed: {e}")
            return self._create_empty_heatmap(HeatmapType.COMPONENT_TIME)
    
    def _create_pair_time_heatmap(self, 
                                data: pd.DataFrame,
                                time_granularity: TimeGranularity) -> Dict[str, Any]:
        """Create pair performance vs time heatmap"""
        # Implementation similar to component_time but for pairs
        return self._create_empty_heatmap(HeatmapType.PAIR_TIME)
    
    def _create_regime_time_heatmap(self, 
                                  data: pd.DataFrame,
                                  time_granularity: TimeGranularity) -> Dict[str, Any]:
        """Create regime performance vs time heatmap"""
        # Implementation similar to component_time but for regimes
        return self._create_empty_heatmap(HeatmapType.REGIME_TIME)
    
    def _create_performance_attribution_heatmap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create overall performance attribution heatmap"""
        # Implementation for comprehensive attribution view
        return self._create_empty_heatmap(HeatmapType.PERFORMANCE_ATTRIBUTION)
    
    def _create_empty_heatmap(self, heatmap_type: HeatmapType) -> Dict[str, Any]:
        """Create empty heatmap placeholder"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this time period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            title=f"{heatmap_type.value.replace('_', ' ').title()} Attribution",
            height=400,
            width=800,
            showlegend=False
        )
        
        return {
            "figure": fig.to_dict(),
            "data_summary": {
                "status": "no_data",
                "message": "No data available for the selected time period"
            }
        }
    
    def generate_performance_trends(self, 
                                  attribution_history: List[Any],
                                  days_back: int = 30) -> Dict[str, Any]:
        """Generate performance trend analysis"""
        try:
            cache_key = f"trends_{days_back}"
            
            if self.cache.is_cache_valid(cache_key):
                return self.cache.heatmap_cache[cache_key]
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_results = [r for r in attribution_history if r.period_start >= cutoff_date]
            
            if not recent_results:
                return {"status": "no_data"}
            
            # Extract trend data
            timestamps = [r.period_end for r in recent_results]
            total_pnl = [r.total_pnl_bps for r in recent_results]
            alpha_contrib = [r.alpha_contribution for r in recent_results]
            cost_drag = [r.cost_drag for r in recent_results]
            
            # Create multi-line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=total_pnl,
                mode='lines+markers',
                name='Total PnL',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=alpha_contrib,
                mode='lines+markers',
                name='Alpha Contribution',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps, y=cost_drag,
                mode='lines+markers',
                name='Cost Drag',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Performance Attribution Trends",
                xaxis_title="Date",
                yaxis_title="Contribution (bps)",
                height=400,
                width=800,
                hovermode='x unified'
            )
            
            # Calculate trend statistics
            if len(total_pnl) > 1:
                pnl_trend = np.polyfit(range(len(total_pnl)), total_pnl, 1)[0]
                alpha_trend = np.polyfit(range(len(alpha_contrib)), alpha_contrib, 1)[0]
            else:
                pnl_trend = 0
                alpha_trend = 0
            
            result = {
                "figure": fig.to_dict(),
                "trend_analysis": {
                    "pnl_trend_bps_per_day": pnl_trend,
                    "alpha_trend_bps_per_day": alpha_trend,
                    "current_pnl": total_pnl[-1] if total_pnl else 0,
                    "current_alpha": alpha_contrib[-1] if alpha_contrib else 0,
                    "data_points": len(recent_results)
                }
            }
            
            # Cache result
            self.cache.heatmap_cache[cache_key] = result
            self.cache.cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Performance trends generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_alert_dashboard(self, alerts: List[Any]) -> Dict[str, Any]:
        """Generate alert monitoring dashboard"""
        try:
            if not alerts:
                return {"status": "no_alerts"}
            
            # Alert distribution by type
            alert_types = {}
            severity_dist = {}
            
            for alert in alerts:
                alert_type = getattr(alert, 'alert_type', 'unknown')
                severity = getattr(alert, 'severity', 'unknown')
                
                if hasattr(alert_type, 'value'):
                    alert_type = alert_type.value
                if hasattr(severity, 'value'):
                    severity = severity.value
                
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                severity_dist[severity] = severity_dist.get(severity, 0) + 1
            
            # Create alert type pie chart
            fig_types = go.Figure(data=[go.Pie(
                labels=list(alert_types.keys()),
                values=list(alert_types.values()),
                hole=0.3
            )])
            
            fig_types.update_layout(
                title="Alert Distribution by Type",
                height=400,
                width=400
            )
            
            # Create severity bar chart
            fig_severity = go.Figure(data=[go.Bar(
                x=list(severity_dist.keys()),
                y=list(severity_dist.values()),
                marker_color=['green', 'yellow', 'orange', 'red'][:len(severity_dist)]
            )])
            
            fig_severity.update_layout(
                title="Alert Distribution by Severity",
                xaxis_title="Severity Level",
                yaxis_title="Number of Alerts",
                height=400,
                width=400
            )
            
            return {
                "alert_type_chart": fig_types.to_dict(),
                "severity_chart": fig_severity.to_dict(),
                "summary": {
                    "total_alerts": len(alerts),
                    "alert_types": alert_types,
                    "severity_distribution": severity_dist,
                    "most_common_type": max(alert_types, key=alert_types.get) if alert_types else "none"
                }
            }
            
        except Exception as e:
            logger.error(f"Alert dashboard generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        try:
            return {
                "realtime_metrics": {
                    "current_pnl_bps": self.current_metrics.current_pnl_bps,
                    "current_sharpe": self.current_metrics.current_sharpe,
                    "current_drawdown_pct": self.current_metrics.current_drawdown_pct,
                    "alpha_contribution": self.current_metrics.alpha_contribution,
                    "cost_drag": self.current_metrics.cost_drag,
                    "attribution_accuracy": self.current_metrics.attribution_accuracy,
                    "system_status": self.current_metrics.system_status,
                    "last_update": self.current_metrics.last_update.isoformat()
                },
                "cache_status": {
                    "cached_items": len(self.cache.cache_timestamps),
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "oldest_cache_age_minutes": self._get_oldest_cache_age()
                },
                "system_health": {
                    "active_alerts": self.current_metrics.active_alerts,
                    "critical_alerts": self.current_metrics.critical_alerts,
                    "data_quality_score": self.current_metrics.data_quality_score,
                    "system_status": self.current_metrics.system_status
                }
            }
            
        except Exception as e:
            logger.error(f"Dashboard summary generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder)"""
        # Would track cache hits vs misses in production
        return 0.85
    
    def _get_oldest_cache_age(self) -> float:
        """Get age of oldest cache entry in minutes"""
        if not self.cache.cache_timestamps:
            return 0.0
        
        oldest_time = min(self.cache.cache_timestamps.values())
        age = datetime.now() - oldest_time
        return age.total_seconds() / 60