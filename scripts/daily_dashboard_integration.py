#!/usr/bin/env python3
"""
Daily Dashboard Integration - Historical Metrics Visualization
Integrates daily metrics logs with dashboard for historical trend analysis
"""

import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger

class DailyDashboardIntegrator:
    """Integrates daily metrics with dashboard visualizations"""
    
    def __init__(self):
        self.logger = get_logger()
        self.daily_logs_dir = Path("logs/daily")
    
    def load_historical_metrics(self, days_back: int = 30) -> pd.DataFrame:
        """Load historical daily metrics for trend analysis"""
        
        self.logger.info(f"Loading {days_back} days of historical metrics")
        
        # Collect all daily metric files
        historical_data = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            
            date_dir = self.daily_logs_dir / date_str
            
            if date_dir.exists():
                # Look for latest.json in this date directory
                latest_file = date_dir / "latest.json"
                
                if latest_file.exists():
                    try:
                        with open(latest_file, 'r') as f:
                            daily_data = json.load(f)
                        
                        # Extract key metrics
                        flattened_data = self._flatten_daily_metrics(daily_data, date_str)
                        historical_data.append(flattened_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load metrics for {date_str}: {e}")
        
        # Convert to DataFrame
        if historical_data:
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            self.logger.info(f"Loaded metrics for {len(df)} days")
            return df
        else:
            self.logger.warning("No historical metrics found")
            return pd.DataFrame()
    
    def _flatten_daily_metrics(self, daily_data: Dict[str, Any], date_str: str) -> Dict[str, Any]:
        """Flatten nested daily metrics into flat structure for DataFrame"""
        
        flattened = {
            "date": date_str,
            "timestamp": daily_data.get("timestamp", ""),
            "collection_status": daily_data.get("collection_status", "unknown")
        }
        
        metrics = daily_data.get("metrics", {})
        
        # Health metrics
        health = metrics.get("health", {})
        if health.get("status") != "error":
            flattened.update({
                "health_score": health.get("health_score", 0),
                "health_decision": health.get("decision", "unknown"),
                "validation_accuracy_score": health.get("component_scores", {}).get("validation_accuracy", 0),
                "sharpe_norm_score": health.get("component_scores", {}).get("sharpe_norm", 0),
                "feedback_success_score": health.get("component_scores", {}).get("feedback_success", 0),
                "error_ratio_score": health.get("component_scores", {}).get("error_ratio", 0),
                "data_completeness_score": health.get("component_scores", {}).get("data_completeness", 0),
                "tuning_freshness_score": health.get("component_scores", {}).get("tuning_freshness", 0)
            })
        
        # Coverage metrics
        coverage = metrics.get("coverage", {})
        if coverage.get("status") != "error":
            flattened.update({
                "coverage_percentage": coverage.get("coverage_percentage", 0),
                "total_symbols": coverage.get("total_live_symbols", 0),
                "covered_symbols": coverage.get("covered_symbols", 0),
                "missing_symbols": coverage.get("missing_symbols", 0),
                "coverage_status": coverage.get("quality_status", "unknown")
            })
        
        # Performance metrics
        performance = metrics.get("performance", {})
        if performance.get("status") != "error":
            flattened.update({
                "precision_at_k": performance.get("precision_at_k", 0),
                "hit_rate_conf80": performance.get("hit_rate_conf80", 0),
                "mae_calibration": performance.get("mae_calibration", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "performance_sample_size": performance.get("sample_size", 0)
            })
        
        # Calibration metrics
        calibration = metrics.get("calibration", {})
        if calibration.get("status") != "error":
            flattened.update({
                "expected_calibration_error": calibration.get("expected_calibration_error", 0),
                "max_calibration_error": calibration.get("max_calibration_error", 0),
                "well_calibrated_bins": calibration.get("well_calibrated_bins", 0)
            })
        
        # Signal quality metrics
        signal_quality = metrics.get("signal_quality", {})
        if signal_quality.get("status") != "error":
            flattened.update({
                "sq_average_precision": signal_quality.get("average_precision_at_k", 0),
                "sq_average_hit_rate": signal_quality.get("average_hit_rate", 0),
                "sq_average_sharpe": signal_quality.get("average_sharpe_ratio", 0),
                "sq_worst_drawdown": signal_quality.get("worst_max_drawdown", 0),
                "sq_horizons_passed": signal_quality.get("horizons_passed", 0)
            })
        
        # Execution metrics
        execution = metrics.get("execution", {})
        if execution.get("status") != "error":
            flattened.update({
                "slippage_p50": execution.get("slippage_p50", 0),
                "slippage_p90": execution.get("slippage_p90", 0),
                "fill_rate": execution.get("overall_fill_rate", 0),
                "latency_p95": execution.get("latency_p95_ms", 0),
                "execution_quality_score": execution.get("quality_score", 0)
            })
        
        # Composite metrics
        composite = metrics.get("composite", {})
        if composite:
            flattened.update({
                "overall_readiness_score": composite.get("overall_readiness_score", 0),
                "operational_status": composite.get("operational_status", "unknown")
            })
        
        return flattened
    
    def generate_health_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Generate health score trend chart"""
        
        if df.empty or 'health_score' not in df.columns:
            return go.Figure().add_annotation(text="No health data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Health score line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['health_score'],
            mode='lines+markers',
            name='Health Score',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add threshold lines
        fig.add_hline(y=85, line_dash="dash", line_color="green", 
                     annotation_text="GO Threshold (85)")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold (60)")
        
        fig.update_layout(
            title="System Health Score Trend",
            xaxis_title="Date",
            yaxis_title="Health Score",
            yaxis=dict(range=[0, 100]),
            showlegend=True
        )
        
        return fig
    
    def generate_performance_metrics_chart(self, df: pd.DataFrame) -> go.Figure:
        """Generate performance metrics trend chart"""
        
        if df.empty:
            return go.Figure().add_annotation(text="No performance data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Precision@K
        if 'precision_at_k' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['precision_at_k'] * 100,  # Convert to percentage
                mode='lines+markers',
                name='Precision@K (%)',
                line=dict(color='blue')
            ))
        
        # Hit Rate
        if 'hit_rate_conf80' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['hit_rate_conf80'] * 100,  # Convert to percentage
                mode='lines+markers',
                name='Hit Rate (%)',
                line=dict(color='green')
            ))
        
        # Add target lines
        fig.add_hline(y=60, line_dash="dash", line_color="blue", 
                     annotation_text="Precision Target (60%)")
        fig.add_hline(y=55, line_dash="dash", line_color="green", 
                     annotation_text="Hit Rate Target (55%)")
        
        fig.update_layout(
            title="Performance Metrics Trend",
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=True
        )
        
        return fig
    
    def generate_coverage_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Generate coverage trend chart"""
        
        if df.empty or 'coverage_percentage' not in df.columns:
            return go.Figure().add_annotation(text="No coverage data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Coverage percentage
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['coverage_percentage'],
            mode='lines+markers',
            name='Coverage %',
            line=dict(color='purple', width=2),
            marker=dict(size=6),
            fill='tonexty'
        ))
        
        # Add target lines
        fig.add_hline(y=99, line_dash="dash", line_color="green", 
                     annotation_text="Target (99%)")
        fig.add_hline(y=95, line_dash="dash", line_color="orange", 
                     annotation_text="Minimum (95%)")
        
        fig.update_layout(
            title="Exchange Coverage Trend",
            xaxis_title="Date",
            yaxis_title="Coverage Percentage (%)",
            yaxis=dict(range=[90, 100]),
            showlegend=True
        )
        
        return fig
    
    def generate_operational_status_chart(self, df: pd.DataFrame) -> go.Figure:
        """Generate operational status over time"""
        
        if df.empty or 'operational_status' not in df.columns:
            return go.Figure().add_annotation(text="No operational status data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Map status to numeric values for plotting
        status_mapping = {"GO": 3, "WARNING": 2, "NO-GO": 1}
        df_plot = df.copy()
        df_plot['status_numeric'] = df_plot['operational_status'].map(status_mapping)
        
        # Create color mapping
        colors = []
        for status in df_plot['operational_status']:
            if status == "GO":
                colors.append('green')
            elif status == "WARNING":
                colors.append('orange')
            else:
                colors.append('red')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_plot['date'],
            y=df_plot['status_numeric'],
            mode='lines+markers',
            name='Operational Status',
            line=dict(color='gray', width=1),
            marker=dict(size=10, color=colors),
            text=df_plot['operational_status'],
            textposition="middle center"
        ))
        
        fig.update_layout(
            title="Operational Status Over Time",
            xaxis_title="Date",
            yaxis_title="Status Level",
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3],
                ticktext=['NO-GO', 'WARNING', 'GO'],
                range=[0.5, 3.5]
            ),
            showlegend=False
        )
        
        return fig
    
    def generate_component_scores_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Generate health component scores heatmap"""
        
        component_cols = [
            'validation_accuracy_score',
            'sharpe_norm_score', 
            'feedback_success_score',
            'error_ratio_score',
            'data_completeness_score',
            'tuning_freshness_score'
        ]
        
        # Check if component data exists
        available_cols = [col for col in component_cols if col in df.columns]
        
        if df.empty or not available_cols:
            return go.Figure().add_annotation(text="No component score data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for heatmap
        heatmap_data = df[['date'] + available_cols].set_index('date')
        
        # Create nicer column names
        nice_names = {
            'validation_accuracy_score': 'Validation Accuracy',
            'sharpe_norm_score': 'Sharpe Ratio',
            'feedback_success_score': 'Feedback Success',
            'error_ratio_score': 'Error Ratio',
            'data_completeness_score': 'Data Completeness',
            'tuning_freshness_score': 'Tuning Freshness'
        }
        
        heatmap_data.columns = [nice_names.get(col, col) for col in heatmap_data.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values.T,
            x=heatmap_data.index,
            y=heatmap_data.columns,
            colorscale='RdYlGn',
            zmin=0,
            zmax=25,  # Max component score
            colorbar=dict(title="Component Score")
        ))
        
        fig.update_layout(
            title="Health Component Scores Over Time",
            xaxis_title="Date",
            yaxis_title="Component"
        )
        
        return fig
    
    def generate_daily_metrics_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from daily metrics"""
        
        if df.empty:
            return {"error": "No data available"}
        
        summary = {
            "period": {
                "start_date": df['date'].min().strftime('%Y-%m-%d') if not df.empty else "N/A",
                "end_date": df['date'].max().strftime('%Y-%m-%d') if not df.empty else "N/A",
                "total_days": len(df)
            }
        }
        
        # Health metrics summary
        if 'health_score' in df.columns:
            health_data = df['health_score'].dropna()
            if not health_data.empty:
                summary["health"] = {
                    "average_score": round(health_data.mean(), 1),
                    "current_score": round(health_data.iloc[-1], 1) if len(health_data) > 0 else 0,
                    "days_above_85": int((health_data >= 85).sum()),
                    "days_60_to_85": int(((health_data >= 60) & (health_data < 85)).sum()),
                    "days_below_60": int((health_data < 60).sum())
                }
        
        # Performance metrics summary
        if 'precision_at_k' in df.columns:
            precision_data = df['precision_at_k'].dropna()
            if not precision_data.empty:
                summary["performance"] = {
                    "avg_precision_at_k": round(precision_data.mean(), 3),
                    "current_precision": round(precision_data.iloc[-1], 3) if len(precision_data) > 0 else 0,
                    "days_above_target": int((precision_data >= 0.60).sum())
                }
        
        if 'hit_rate_conf80' in df.columns:
            hit_rate_data = df['hit_rate_conf80'].dropna()
            if not hit_rate_data.empty:
                summary.setdefault("performance", {}).update({
                    "avg_hit_rate": round(hit_rate_data.mean(), 3),
                    "current_hit_rate": round(hit_rate_data.iloc[-1], 3) if len(hit_rate_data) > 0 else 0
                })
        
        # Coverage summary
        if 'coverage_percentage' in df.columns:
            coverage_data = df['coverage_percentage'].dropna()
            if not coverage_data.empty:
                summary["coverage"] = {
                    "average_coverage": round(coverage_data.mean(), 1),
                    "current_coverage": round(coverage_data.iloc[-1], 1) if len(coverage_data) > 0 else 0,
                    "days_above_99": int((coverage_data >= 99).sum()),
                    "days_above_95": int((coverage_data >= 95).sum())
                }
        
        # Operational status summary
        if 'operational_status' in df.columns:
            status_data = df['operational_status'].dropna()
            if not status_data.empty:
                status_counts = status_data.value_counts()
                summary["operational"] = {
                    "current_status": status_data.iloc[-1] if len(status_data) > 0 else "Unknown",
                    "days_go": int(status_counts.get("GO", 0)),
                    "days_warning": int(status_counts.get("WARNING", 0)),
                    "days_nogo": int(status_counts.get("NO-GO", 0))
                }
        
        return summary

def main():
    """Main entry point for dashboard integration"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Daily Dashboard Integration - Historical Metrics"
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to load (default: 30)'
    )
    
    parser.add_argument(
        '--export-charts',
        action='store_true',
        help='Export charts as HTML files'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"📊 DAILY DASHBOARD INTEGRATION")
        print(f"📅 Loading {args.days} days of metrics")
        print("=" * 50)
        
        # Initialize integrator
        integrator = DailyDashboardIntegrator()
        
        # Load historical data
        df = integrator.load_historical_metrics(args.days)
        
        if df.empty:
            print("❌ No historical data found")
            return 1
        
        print(f"✅ Loaded metrics for {len(df)} days")
        
        # Generate summary
        summary = integrator.generate_daily_metrics_summary(df)
        print(f"📊 Summary: {summary.get('period', {}).get('total_days', 0)} days analyzed")
        
        if args.export_charts:
            # Generate and export charts
            charts_dir = Path("data/dashboard_exports")
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Health trend
            health_chart = integrator.generate_health_trend_chart(df)
            health_chart.write_html(charts_dir / "health_trend.html")
            
            # Performance metrics
            performance_chart = integrator.generate_performance_metrics_chart(df)
            performance_chart.write_html(charts_dir / "performance_trend.html")
            
            # Coverage trend
            coverage_chart = integrator.generate_coverage_trend_chart(df)
            coverage_chart.write_html(charts_dir / "coverage_trend.html")
            
            # Operational status
            status_chart = integrator.generate_operational_status_chart(df)
            status_chart.write_html(charts_dir / "operational_status.html")
            
            # Component heatmap
            heatmap_chart = integrator.generate_component_scores_heatmap(df)
            heatmap_chart.write_html(charts_dir / "component_heatmap.html")
            
            print(f"📈 Charts exported to: {charts_dir}")
        
        # Save summary
        summary_file = Path("logs/daily/historical_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📝 Summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Dashboard integration error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())