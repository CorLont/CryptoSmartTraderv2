import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

class MainDashboard:
    """Main dashboard for comprehensive system overview"""
    
    def __init__(self, config_manager, health_monitor):
        self.config_manager = config_manager
        self.health_monitor = health_monitor
        
        # Initialize session state for auto-refresh
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30  # seconds
    
    def render(self):
        """Render main dashboard"""
        st.title("🎯 Main Dashboard")
        
        # Auto-refresh controls
        self._render_refresh_controls()
        
        # System overview
        self._render_system_overview()
        
        # Market overview
        self._render_market_overview()
        
        # Agent performance
        self._render_agent_performance()
        
        # Recent alerts and activities
        self._render_alerts_activities()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Auto-refresh logic
        self._handle_auto_refresh()
    
    def _render_refresh_controls(self):
        """Render auto-refresh controls"""
        col1, col2, col3, col4 = st.columns([2, 2, 2, 6])
        
        with col1:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh", 
                value=st.session_state.auto_refresh
            )
        
        with col3:
            st.session_state.refresh_interval = st.selectbox(
                "Interval (s)",
                options=[10, 30, 60, 120, 300],
                index=1,
                key="refresh_interval_select"
            )
        
        if st.session_state.auto_refresh:
            st.info(f"🔄 Auto-refreshing every {st.session_state.refresh_interval} seconds")
    
    def _render_system_overview(self):
        """Render system health overview"""
        st.subheader("🖥️ System Overview")
        
        # Get system health data
        health_data = self.health_monitor.get_detailed_health()
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            grade = health_data.get('grade', 'F')
            grade_color = {
                'A': '🟢', 'B': '🟡', 'C': '🟡', 
                'D': '🟠', 'E': '🔴', 'F': '🔴'
            }.get(grade, '🔴')
            st.metric("System Grade", f"{grade_color} {grade}")
        
        with col2:
            score = health_data.get('score', 0)
            st.metric("Health Score", f"{score:.1f}%", 
                     delta=f"{health_data.get('score_change', 0):.1f}%")
        
        with col3:
            active_agents = health_data.get('active_agents', 0)
            st.metric("Active Agents", active_agents,
                     delta=health_data.get('agent_change', 0))
        
        with col4:
            coverage = health_data.get('data_coverage', 0)
            st.metric("Data Coverage", f"{coverage:.1f}%",
                     delta=f"{health_data.get('coverage_change', 0):.1f}%")
        
        with col5:
            uptime = health_data.get('uptime', 0)
            st.metric("Uptime", f"{uptime:.1f}h")
        
        # System resources
        resources = health_data.get('resources', {})
        if resources:
            st.subheader("💻 System Resources")
            
            resource_col1, resource_col2, resource_col3 = st.columns(3)
            
            with resource_col1:
                cpu_usage = resources.get('cpu_percent', 0)
                st.metric("CPU Usage", f"{cpu_usage:.1f}%")
                st.progress(min(cpu_usage / 100, 1.0))
            
            with resource_col2:
                memory_usage = resources.get('memory_percent', 0)
                st.metric("Memory Usage", f"{memory_usage:.1f}%")
                st.progress(min(memory_usage / 100, 1.0))
            
            with resource_col3:
                disk_usage = resources.get('disk_percent', 0)
                st.metric("Disk Usage", f"{disk_usage:.1f}%")
                st.progress(min(disk_usage / 100, 1.0))
    
    def _render_market_overview(self):
        """Render market data overview"""
        st.subheader("📊 Market Overview")
        
        # Mock market data for demonstration
        # In production, this would get real data from data_manager
        try:
            # Generate sample market overview
            market_data = self._generate_sample_market_data()
            
            # Market summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Symbols", market_data.get('total_symbols', 0))
            
            with col2:
                gainers = market_data.get('gainers', 0)
                st.metric("Gainers (24h)", gainers, delta=f"+{gainers}")
            
            with col3:
                losers = market_data.get('losers', 0)
                st.metric("Losers (24h)", losers, delta=f"-{losers}")
            
            with col4:
                avg_change = market_data.get('avg_change', 0)
                st.metric("Avg Change", f"{avg_change:+.2f}%", delta=f"{avg_change:+.2f}%")
            
            # Market heatmap
            self._render_market_heatmap(market_data)
            
            # Top movers
            self._render_top_movers(market_data)
            
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
            st.info("Market data will be displayed when data sources are connected.")
    
    def _generate_sample_market_data(self) -> dict:
        """Generate sample market data for demonstration"""
        import random
        
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD', 'DOT/USD', 
                  'AVAX/USD', 'MATIC/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD']
        
        market_data = {
            'total_symbols': len(symbols),
            'symbols': []
        }
        
        gainers = 0
        losers = 0
        total_change = 0
        
        for symbol in symbols:
            change = random.uniform(-15, 15)
            price = random.uniform(0.1, 50000)
            volume = random.uniform(1000000, 100000000)
            
            if change > 0:
                gainers += 1
            elif change < 0:
                losers += 1
            
            total_change += change
            
            market_data['symbols'].append({
                'symbol': symbol,
                'price': price,
                'change_24h': change,
                'volume_24h': volume,
                'market_cap': price * random.uniform(1000000, 10000000000)
            })
        
        market_data['gainers'] = gainers
        market_data['losers'] = losers
        market_data['avg_change'] = total_change / len(symbols)
        
        return market_data
    
    def _render_market_heatmap(self, market_data: dict):
        """Render market heatmap"""
        if not market_data.get('symbols'):
            return
        
        symbols_data = market_data['symbols']
        
        # Create heatmap data
        symbols = [s['symbol'] for s in symbols_data]
        changes = [s['change_24h'] for s in symbols_data]
        sizes = [s['market_cap'] for s in symbols_data]
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=symbols,
            values=sizes,
            parents=[""] * len(symbols),
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="24h Change %"),
            text=[f"{s}<br>{c:+.2f}%" for s, c in zip(symbols, changes)],
            textinfo="text",
            hovertemplate='<b>%{label}</b><br>Change: %{color:+.2f}%<br>Market Cap: $%{value:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Market Heatmap (24h Performance)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_top_movers(self, market_data: dict):
        """Render top gainers and losers"""
        if not market_data.get('symbols'):
            return
        
        symbols_data = market_data['symbols']
        
        # Sort by change
        sorted_symbols = sorted(symbols_data, key=lambda x: x['change_24h'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Top Gainers")
            gainers = sorted_symbols[-5:]  # Top 5 gainers
            gainers.reverse()
            
            for symbol_data in gainers:
                if symbol_data['change_24h'] > 0:
                    st.success(f"**{symbol_data['symbol']}**: +{symbol_data['change_24h']:.2f}% (${symbol_data['price']:.2f})")
        
        with col2:
            st.subheader("📉 Top Losers")
            losers = sorted_symbols[:5]  # Top 5 losers
            
            for symbol_data in losers:
                if symbol_data['change_24h'] < 0:
                    st.error(f"**{symbol_data['symbol']}**: {symbol_data['change_24h']:.2f}% (${symbol_data['price']:.2f})")
    
    def _render_agent_performance(self):
        """Render agent performance overview"""
        st.subheader("🤖 Agent Performance")
        
        # Get agent health data
        health_data = self.health_monitor.get_detailed_health()
        agent_health = health_data.get('agent_health', {})
        
        if not agent_health:
            st.info("Agent performance data will be displayed once agents are active.")
            return
        
        # Agent status overview
        agent_cols = st.columns(min(len(agent_health), 6))
        
        for idx, (agent_name, status) in enumerate(agent_health.items()):
            if idx < len(agent_cols):
                with agent_cols[idx]:
                    status_icon = "🟢" if status.get('status') == 'healthy' else "🔴"
                    display_name = agent_name.replace('_', ' ').title()
                    
                    st.metric(
                        display_name,
                        f"{status_icon} {status.get('status', 'unknown').title()}",
                        delta=f"{status.get('uptime', 0):.1f}h"
                    )
        
        # Agent performance chart
        self._render_agent_performance_chart(agent_health)
    
    def _render_agent_performance_chart(self, agent_health: dict):
        """Render agent performance metrics chart"""
        if not agent_health:
            return
        
        # Prepare data for chart
        agent_names = list(agent_health.keys())
        uptimes = [status.get('uptime', 0) for status in agent_health.values()]
        error_rates = [status.get('error_rate', 0) * 100 for status in agent_health.values()]
        processed_items = [status.get('processed_items', 0) for status in agent_health.values()]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Uptime (hours)', 'Error Rate (%)', 'Processed Items', 'Agent Status'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Uptime chart
        fig.add_trace(
            go.Bar(x=agent_names, y=uptimes, name="Uptime", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Error rate chart
        fig.add_trace(
            go.Bar(x=agent_names, y=error_rates, name="Error Rate", marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Processed items chart
        fig.add_trace(
            go.Bar(x=agent_names, y=processed_items, name="Processed Items", marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Overall agent health indicator
        healthy_agents = sum(1 for status in agent_health.values() if status.get('status') == 'healthy')
        total_agents = len(agent_health)
        health_percentage = (healthy_agents / total_agents * 100) if total_agents > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode = "gauge+number",
                value = health_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Agent Health %"},
                gauge = {
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
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Agent Performance Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts_activities(self):
        """Render recent alerts and system activities"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Recent Alerts")
            
            # Get recent alerts
            recent_alerts = self.health_monitor.get_alerts(hours=24)
            
            if recent_alerts:
                for alert in recent_alerts[:5]:  # Show last 5 alerts
                    alert_type = alert.get('type', 'info')
                    alert_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(alert_type, "🔵")
                    
                    with st.expander(f"{alert_icon} {alert.get('timestamp', 'Unknown time')}"):
                        st.write(f"**Type:** {alert_type.title()}")
                        st.write(f"**Message:** {alert.get('message', 'No message')}")
                        if alert.get('details'):
                            st.json(alert['details'])
            else:
                st.success("✅ No recent alerts - system running smoothly!")
        
        with col2:
            st.subheader("📈 System Activities")
            
            # System activity timeline
            activities = [
                {"time": "2 min ago", "activity": "Data collection completed", "status": "success"},
                {"time": "5 min ago", "activity": "ML models updated", "status": "success"},
                {"time": "12 min ago", "activity": "Whale activity detected", "status": "info"},
                {"time": "18 min ago", "activity": "Technical analysis completed", "status": "success"},
                {"time": "25 min ago", "activity": "Sentiment analysis updated", "status": "success"}
            ]
            
            for activity in activities:
                status_icon = {
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                    "info": "ℹ️"
                }.get(activity['status'], "ℹ️")
                
                st.write(f"{status_icon} **{activity['time']}** - {activity['activity']}")
    
    def _render_performance_metrics(self):
        """Render system performance metrics"""
        st.subheader("⚡ Performance Metrics")
        
        # Generate sample performance data
        performance_data = self._generate_performance_data()
        
        # Performance overview
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("API Response Time", f"{performance_data['api_response_ms']:.0f}ms")
        
        with perf_col2:
            st.metric("Data Processing Rate", f"{performance_data['processing_rate']:.0f}/min")
        
        with perf_col3:
            st.metric("ML Prediction Accuracy", f"{performance_data['ml_accuracy']:.1f}%")
        
        with perf_col4:
            st.metric("System Throughput", f"{performance_data['throughput']:.0f} ops/s")
        
        # Performance trends chart
        self._render_performance_trends(performance_data)
    
    def _generate_performance_data(self) -> dict:
        """Generate sample performance data"""
        import random
        
        return {
            'api_response_ms': random.uniform(50, 200),
            'processing_rate': random.uniform(800, 1200),
            'ml_accuracy': random.uniform(75, 95),
            'throughput': random.uniform(150, 300),
            'timestamps': [datetime.now() - timedelta(minutes=i*5) for i in range(12, 0, -1)],
            'response_times': [random.uniform(50, 200) for _ in range(12)],
            'throughput_history': [random.uniform(150, 300) for _ in range(12)]
        }
    
    def _render_performance_trends(self, performance_data: dict):
        """Render performance trend charts"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('API Response Time (ms)', 'System Throughput (ops/s)')
        )
        
        timestamps = performance_data['timestamps']
        
        # Response time trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=performance_data['response_times'],
                mode='lines+markers',
                name='Response Time',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Throughput trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=performance_data['throughput_history'],
                mode='lines+markers',
                name='Throughput',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Performance Trends (Last Hour)"
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Response Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Throughput (ops/s)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _handle_auto_refresh(self):
        """Handle auto-refresh functionality"""
        if st.session_state.auto_refresh:
            current_time = time.time()
            if current_time - st.session_state.last_refresh >= st.session_state.refresh_interval:
                st.session_state.last_refresh = current_time
                time.sleep(0.1)  # Small delay to prevent too frequent refreshes
                st.rerun()
            
            # Add a small delay and rerun check
            placeholder = st.empty()
            with placeholder.container():
                time_until_refresh = st.session_state.refresh_interval - (current_time - st.session_state.last_refresh)
                if time_until_refresh > 0:
                    st.caption(f"⏱️ Next refresh in {int(time_until_refresh)} seconds")
                else:
                    st.caption("🔄 Refreshing...")
