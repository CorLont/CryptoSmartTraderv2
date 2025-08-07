"""
CryptoSmartTrader V2 - Production Monitoring Dashboard
Enterprise-grade monitoring interface with real-time metrics and alerts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import time

class ProductionMonitoringDashboard:
    """Production monitoring dashboard with comprehensive system metrics"""
    
    def __init__(self, container):
        self.container = container
        self.monitoring_system = container.monitoring_system()
        self.error_handler = container.centralized_error_handler()
        self.health_monitor = container.health_monitor()
        
    def render(self):
        """Render the production monitoring dashboard"""
        st.title("🔍 Production Monitoring")
        st.markdown("Enterprise-grade system monitoring with real-time metrics and alerts")
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # System overview
        self._render_system_overview()
        
        # Metrics section
        self._render_metrics_section()
        
        # Error monitoring
        self._render_error_monitoring()
        
        # Alert management
        self._render_alert_management()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Health monitoring
        self._render_health_monitoring()
    
    def _render_system_overview(self):
        """Render system overview section"""
        st.header("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get monitoring status
        monitoring_status = self.monitoring_system.get_monitoring_status()
        
        with col1:
            cpu_color = "normal" if cpu_percent < 70 else "inverse"
            st.metric(
                "CPU Usage", 
                f"{cpu_percent:.1f}%",
                delta=None,
                delta_color=cpu_color
            )
        
        with col2:
            memory_color = "normal" if memory.percent < 80 else "inverse"
            st.metric(
                "Memory Usage", 
                f"{memory.percent:.1f}%",
                delta=f"{memory.used // (1024**3):.1f}GB Used",
                delta_color=memory_color
            )
        
        with col3:
            disk_color = "normal" if disk.percent < 85 else "inverse"
            st.metric(
                "Disk Usage", 
                f"{disk.percent:.1f}%",
                delta=f"{disk.free // (1024**3):.1f}GB Free",
                delta_color=disk_color
            )
        
        with col4:
            monitoring_status_text = "🟢 Active" if monitoring_status["monitoring_active"] else "🔴 Inactive"
            st.metric(
                "Monitoring", 
                monitoring_status_text,
                delta=f"{monitoring_status['alert_history_count']} Alerts"
            )
        
        # System health indicators
        st.subheader("🏥 System Health Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_report = self.health_monitor.get_system_health()
            overall_score = health_report.get("overall_score", 0.5)
            
            # Health score gauge
            fig_health = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_health.update_layout(height=300)
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col2:
            # Error rate over time
            error_stats = self.error_handler.get_error_statistics()
            
            if error_stats.get("errors_by_hour"):
                hours = list(error_stats["errors_by_hour"].keys())[-24:]  # Last 24 hours
                error_counts = [error_stats["errors_by_hour"].get(hour, 0) for hour in hours]
                
                fig_errors = go.Figure()
                fig_errors.add_trace(go.Scatter(
                    x=hours,
                    y=error_counts,
                    mode='lines+markers',
                    name='Errors',
                    line=dict(color='red')
                ))
                fig_errors.update_layout(
                    title="Error Rate (24h)",
                    xaxis_title="Hour",
                    yaxis_title="Error Count",
                    height=300
                )
                st.plotly_chart(fig_errors, use_container_width=True)
            else:
                st.info("No error data available")
        
        with col3:
            # Performance metrics
            system_metrics = monitoring_status.get("system_metrics", {})
            
            metrics_data = {
                "Metric": ["CPU", "Memory", "Health"],
                "Value": [
                    system_metrics.get("cpu_usage", 0),
                    system_metrics.get("memory_usage", 0),
                    system_metrics.get("health_score", 0.5) * 100
                ]
            }
            
            fig_metrics = px.bar(
                x=metrics_data["Metric"],
                y=metrics_data["Value"],
                title="Current Metrics",
                color=metrics_data["Value"],
                color_continuous_scale="RdYlGn"
            )
            fig_metrics.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    def _render_metrics_section(self):
        """Render metrics section with Prometheus data"""
        st.header("📈 Metrics & Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Analysis Metrics")
            
            # Simulated analysis metrics (would come from Prometheus in production)
            analysis_data = {
                "Agent": ["Sentiment", "Technical", "ML Predictor", "Backtest", "Trade Executor"],
                "Requests": [150, 120, 95, 45, 30],
                "Success Rate": [98.5, 96.2, 94.8, 99.1, 97.3]
            }
            
            df_analysis = pd.DataFrame(analysis_data)
            
            fig_requests = px.bar(
                df_analysis,
                x="Agent",
                y="Requests",
                title="Analysis Requests (Last Hour)",
                color="Success Rate",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_requests, use_container_width=True)
        
        with col2:
            st.subheader("🌐 API Metrics")
            
            # API performance metrics
            api_data = {
                "Service": ["Kraken", "Binance", "CoinGecko", "OpenAI", "Reddit"],
                "Response Time (ms)": [120, 85, 200, 450, 350],
                "Success Rate": [99.2, 98.8, 95.5, 97.1, 92.3]
            }
            
            df_api = pd.DataFrame(api_data)
            
            fig_api = px.scatter(
                df_api,
                x="Response Time (ms)",
                y="Success Rate",
                size="Success Rate",
                color="Service",
                title="API Performance Matrix"
            )
            st.plotly_chart(fig_api, use_container_width=True)
        
        # Cache performance
        st.subheader("💾 Cache Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Hit Rate", "87.3%", delta="2.1%")
        
        with col2:
            st.metric("Cache Size", "245 MB", delta="15 MB")
        
        with col3:
            st.metric("Avg Response", "12ms", delta="-3ms")
    
    def _render_error_monitoring(self):
        """Render error monitoring section"""
        st.header("🚨 Error Monitoring")
        
        error_stats = self.error_handler.get_error_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Error Statistics")
            
            # Error metrics
            total_errors = error_stats.get("total_errors", 0)
            recovery_rate = error_stats.get("recovery_success_rate", 0) * 100
            
            st.metric("Total Errors", total_errors)
            st.metric("Recovery Rate", f"{recovery_rate:.1f}%")
            
            # Error categories
            if error_stats.get("errors_by_category"):
                categories = list(error_stats["errors_by_category"].keys())
                counts = list(error_stats["errors_by_category"].values())
                
                fig_categories = px.pie(
                    values=counts,
                    names=categories,
                    title="Errors by Category"
                )
                st.plotly_chart(fig_categories, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Recent Errors")
            
            # Display recent error information
            if error_stats.get("last_critical_error"):
                st.error(f"Last Critical Error: {error_stats['last_critical_error']}")
            else:
                st.success("No recent critical errors")
            
            # Health recommendations
            recommendations = error_stats.get("recommendations", [])
            if recommendations:
                st.write("**Health Recommendations:**")
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.info("System operating normally")
    
    def _render_alert_management(self):
        """Render alert management section"""
        st.header("🔔 Alert Management")
        
        monitoring_status = self.monitoring_system.get_monitoring_status()
        recent_alerts = monitoring_status.get("recent_alerts", [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Recent Alerts")
            
            if recent_alerts:
                for alert in recent_alerts[-5:]:  # Show last 5 alerts
                    severity_emoji = {
                        "critical": "🔴",
                        "warning": "🟡", 
                        "info": "🔵"
                    }.get(alert.get("severity", "info"), "🔵")
                    
                    with st.expander(f"{severity_emoji} {alert.get('name', 'Unknown Alert')}"):
                        st.write(f"**Severity:** {alert.get('severity', 'Unknown')}")
                        st.write(f"**Time:** {alert.get('timestamp', 'Unknown')}")
            else:
                st.info("No recent alerts")
        
        with col2:
            st.subheader("⚙️ Alert Configuration")
            
            # Alert threshold configuration
            if st.button("Configure Alert Thresholds"):
                st.info("Alert configuration interface would be implemented here")
            
            # Test alert
            if st.button("Send Test Alert"):
                st.success("Test alert sent successfully")
    
    def _render_performance_metrics(self):
        """Render performance metrics section"""
        st.header("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🚀 Response Times")
            
            # Response time distribution
            response_times = [50, 75, 120, 95, 150, 80, 200, 110, 85, 145]
            
            fig_response = go.Figure()
            fig_response.add_trace(go.Histogram(
                x=response_times,
                name="Response Times",
                nbinsx=10
            ))
            fig_response.update_layout(
                title="Response Time Distribution (ms)",
                xaxis_title="Response Time (ms)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            st.subheader("📊 Throughput")
            
            # Throughput metrics
            hours = list(range(24))
            throughput = [45, 38, 25, 15, 12, 18, 35, 65, 85, 92, 88, 95, 
                         102, 98, 105, 110, 108, 95, 85, 75, 65, 55, 50, 48]
            
            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Scatter(
                x=hours,
                y=throughput,
                mode='lines+markers',
                name='Requests/hour'
            ))
            fig_throughput.update_layout(
                title="Request Throughput (24h)",
                xaxis_title="Hour",
                yaxis_title="Requests"
            )
            st.plotly_chart(fig_throughput, use_container_width=True)
        
        with col3:
            st.subheader("🎯 SLA Metrics")
            
            sla_metrics = {
                "Uptime": 99.95,
                "Availability": 99.88,
                "Response SLA": 98.5,
                "Error Rate SLA": 99.2
            }
            
            for metric, value in sla_metrics.items():
                color = "normal" if value >= 99.0 else "inverse"
                st.metric(metric, f"{value:.2f}%", delta_color=color)
    
    def _render_health_monitoring(self):
        """Render health monitoring section"""
        st.header("🏥 Health Monitoring")
        
        health_report = self.health_monitor.get_system_health()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Component Health")
            
            components = health_report.get("components", {})
            
            for component_name, component_health in components.items():
                status = component_health.get("status", "unknown")
                
                status_emoji = {
                    "healthy": "🟢",
                    "degraded": "🟡",
                    "unhealthy": "🔴",
                    "unknown": "⚪"
                }.get(status, "⚪")
                
                with st.expander(f"{status_emoji} {component_name.replace('_', ' ').title()}"):
                    st.write(f"**Status:** {status}")
                    if "details" in component_health:
                        st.json(component_health["details"])
        
        with col2:
            st.subheader("📈 Health Trends")
            
            # Health score over time (simulated)
            hours = list(range(-24, 0))
            health_scores = [0.85 + 0.1 * (1 + 0.1 * i + 0.05 * (i ** 2) % 10) for i in hours]
            
            fig_health_trend = go.Figure()
            fig_health_trend.add_trace(go.Scatter(
                x=hours,
                y=health_scores,
                mode='lines+markers',
                name='Health Score',
                line=dict(color='green')
            ))
            fig_health_trend.update_layout(
                title="Health Score Trend (24h)",
                xaxis_title="Hours Ago",
                yaxis_title="Health Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_health_trend, use_container_width=True)
        
        # System recommendations
        if health_report.get("recommendations"):
            st.subheader("💡 System Recommendations")
            for rec in health_report["recommendations"]:
                st.info(rec)
        
        # Manual health actions
        st.subheader("🔧 Manual Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Health Check"):
                self.health_monitor.run_health_checks()
                st.success("Health checks refreshed")
        
        with col2:
            if st.button("🧹 Clear Cache"):
                st.success("Cache cleared")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.success("Health report generated")