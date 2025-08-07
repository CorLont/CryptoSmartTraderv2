import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

class AgentDashboard:
    """Agent monitoring and management dashboard"""
    
    def __init__(self, config_manager, health_monitor):
        self.config_manager = config_manager
        self.health_monitor = health_monitor
    
    def render(self):
        """Render agent dashboard"""
        st.title("🤖 Agent Dashboard")
        
        # Agent selection and controls
        selected_agent = self._render_agent_selector()
        
        if selected_agent:
            # Specific agent view
            self._render_agent_details(selected_agent)
        else:
            # Overview of all agents
            self._render_agents_overview()
        
        # Agent performance comparison
        self._render_performance_comparison()
        
        # Agent logs and activities
        self._render_agent_logs()
    
    def _render_agent_selector(self):
        """Render agent selection controls"""
        st.subheader("🎛️ Agent Controls")
        
        # Get available agents
        agents = self.config_manager.get("agents", {})
        agent_names = list(agents.keys()) + ["All Agents"]
        
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            selected_agent = st.selectbox(
                "Select Agent",
                options=agent_names,
                index=len(agent_names)-1  # Default to "All Agents"
            )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col3:
            if st.button("⚙️ Configure Agents"):
                st.info("Agent configuration will open in a modal")
        
        # Return selected agent (None if "All Agents")
        return selected_agent if selected_agent != "All Agents" else None
    
    def _render_agent_details(self, agent_name: str):
        """Render detailed view for specific agent"""
        st.subheader(f"📊 {agent_name.replace('_', ' ').title()} Agent Details")
        
        # Get agent configuration and status
        agent_config = self.config_manager.get("agents", {}).get(agent_name, {})
        health_data = self.health_monitor.get_detailed_health()
        agent_health = health_data.get('agent_health', {}).get(agent_name, {})
        
        # Agent status and metrics
        self._render_agent_status(agent_name, agent_config, agent_health)
        
        # Agent-specific metrics and visualizations
        if agent_name == 'sentiment':
            self._render_sentiment_agent_details()
        elif agent_name == 'technical':
            self._render_technical_agent_details()
        elif agent_name == 'ml_predictor':
            self._render_ml_predictor_details()
        elif agent_name == 'backtest':
            self._render_backtest_agent_details()
        elif agent_name == 'trade_executor':
            self._render_trade_executor_details()
        elif agent_name == 'whale_detector':
            self._render_whale_detector_details()
        
        # Agent configuration
        self._render_agent_configuration(agent_name, agent_config)
    
    def _render_agent_status(self, agent_name: str, config: dict, health: dict):
        """Render agent status overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = health.get('status', 'unknown')
            status_icon = "🟢" if status == 'healthy' else "🔴"
            st.metric("Status", f"{status_icon} {status.title()}")
        
        with col2:
            enabled = config.get('enabled', False)
            enabled_icon = "✅" if enabled else "❌"
            st.metric("Enabled", f"{enabled_icon} {enabled}")
        
        with col3:
            uptime = health.get('uptime', 0)
            st.metric("Uptime", f"{uptime:.1f}h")
        
        with col4:
            processed = health.get('processed_items', 0)
            st.metric("Processed", f"{processed:,}")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            error_rate = health.get('error_rate', 0)
            st.metric("Error Rate", f"{error_rate:.2%}")
        
        with col6:
            update_interval = config.get('update_interval', 0)
            st.metric("Update Interval", f"{update_interval}s")
        
        with col7:
            last_update = health.get('last_update', 'Never')
            if last_update != 'Never':
                try:
                    last_update_dt = datetime.fromisoformat(last_update)
                    time_ago = (datetime.now() - last_update_dt).total_seconds() / 60
                    last_update = f"{time_ago:.0f}m ago"
                except:
                    pass
            st.metric("Last Update", last_update)
        
        with col8:
            memory_usage = health.get('memory_usage_mb', 0)
            st.metric("Memory", f"{memory_usage:.1f}MB")
    
    def _render_agents_overview(self):
        """Render overview of all agents"""
        st.subheader("📋 All Agents Overview")
        
        # Get agents data
        agents_config = self.config_manager.get("agents", {})
        health_data = self.health_monitor.get_detailed_health()
        agents_health = health_data.get('agent_health', {})
        
        # Create agents summary table
        agents_data = []
        
        for agent_name, config in agents_config.items():
            health = agents_health.get(agent_name, {})
            
            agents_data.append({
                'Agent': agent_name.replace('_', ' ').title(),
                'Status': health.get('status', 'unknown').title(),
                'Enabled': '✅' if config.get('enabled', False) else '❌',
                'Uptime (h)': f"{health.get('uptime', 0):.1f}",
                'Processed': f"{health.get('processed_items', 0):,}",
                'Error Rate': f"{health.get('error_rate', 0):.2%}",
                'Update Interval (s)': config.get('update_interval', 0)
            })
        
        if agents_data:
            df = pd.DataFrame(agents_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No agent data available")
        
        # Agent status distribution
        self._render_agent_status_distribution(agents_health)
    
    def _render_agent_status_distribution(self, agents_health: dict):
        """Render agent status distribution chart"""
        if not agents_health:
            return
        
        # Count statuses
        status_counts = {}
        for health in agents_health.values():
            status = health.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3,
            marker_colors=['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        )])
        
        fig.update_layout(
            title="Agent Status Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_comparison(self):
        """Render agent performance comparison"""
        st.subheader("⚡ Performance Comparison")
        
        # Generate sample performance data for visualization
        performance_data = self._generate_agent_performance_data()
        
        # Create comparison charts
        self._render_performance_charts(performance_data)
    
    def _generate_agent_performance_data(self) -> dict:
        """Generate sample agent performance data"""
        import random
        
        agents = ['sentiment', 'technical', 'ml_predictor', 'backtest', 'trade_executor', 'whale_detector']
        
        return {
            'agents': agents,
            'processing_times': [random.uniform(0.1, 2.0) for _ in agents],
            'success_rates': [random.uniform(85, 99) for _ in agents],
            'throughput': [random.uniform(50, 500) for _ in agents],
            'memory_usage': [random.uniform(10, 200) for _ in agents]
        }
    
    def _render_performance_charts(self, performance_data: dict):
        """Render performance comparison charts"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Time (s)', 'Success Rate (%)', 'Throughput (ops/min)', 'Memory Usage (MB)'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        agents = performance_data['agents']
        agent_labels = [agent.replace('_', ' ').title() for agent in agents]
        
        # Processing time
        fig.add_trace(
            go.Bar(x=agent_labels, y=performance_data['processing_times'], 
                   name="Processing Time", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Success rate
        fig.add_trace(
            go.Bar(x=agent_labels, y=performance_data['success_rates'], 
                   name="Success Rate", marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Throughput
        fig.add_trace(
            go.Bar(x=agent_labels, y=performance_data['throughput'], 
                   name="Throughput", marker_color='lightyellow'),
            row=2, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Bar(x=agent_labels, y=performance_data['memory_usage'], 
                   name="Memory Usage", marker_color='lightcoral'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Agent Performance Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_sentiment_agent_details(self):
        """Render sentiment agent specific details"""
        st.subheader("💭 Sentiment Analysis Details")
        
        # Generate sample sentiment data
        sentiment_data = self._generate_sample_sentiment_data()
        
        # Sentiment overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Analyzed Currencies", sentiment_data['total_currencies'])
        
        with col2:
            st.metric("Avg Sentiment Score", f"{sentiment_data['avg_sentiment']:.2f}")
        
        with col3:
            st.metric("High Confidence", f"{sentiment_data['high_confidence']}%")
        
        # Sentiment distribution
        self._render_sentiment_distribution(sentiment_data)
    
    def _generate_sample_sentiment_data(self) -> dict:
        """Generate sample sentiment analysis data"""
        import random
        
        return {
            'total_currencies': random.randint(20, 50),
            'avg_sentiment': random.uniform(0.3, 0.7),
            'high_confidence': random.randint(60, 90),
            'bullish': random.randint(5, 15),
            'bearish': random.randint(3, 12),
            'neutral': random.randint(10, 25)
        }
    
    def _render_sentiment_distribution(self, sentiment_data: dict):
        """Render sentiment distribution chart"""
        categories = ['Bullish', 'Neutral', 'Bearish']
        values = [sentiment_data['bullish'], sentiment_data['neutral'], sentiment_data['bearish']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])
        fig.update_layout(title="Sentiment Distribution", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_technical_agent_details(self):
        """Render technical analysis agent details"""
        st.subheader("📈 Technical Analysis Details")
        
        # Sample technical analysis metrics
        tech_data = {
            'indicators_calculated': 45,
            'symbols_analyzed': 35,
            'buy_signals': 8,
            'sell_signals': 5,
            'hold_signals': 22
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Indicators", tech_data['indicators_calculated'])
        
        with col2:
            st.metric("Symbols", tech_data['symbols_analyzed'])
        
        with col3:
            st.metric("Buy Signals", tech_data['buy_signals'], delta=f"+{tech_data['buy_signals']}")
        
        with col4:
            st.metric("Sell Signals", tech_data['sell_signals'], delta=f"-{tech_data['sell_signals']}")
        
        with col5:
            st.metric("Hold Signals", tech_data['hold_signals'])
        
        # Technical signals distribution
        signal_data = [tech_data['buy_signals'], tech_data['hold_signals'], tech_data['sell_signals']]
        signal_labels = ['Buy', 'Hold', 'Sell']
        
        fig = go.Figure(data=[go.Pie(labels=signal_labels, values=signal_data, hole=0.3)])
        fig.update_layout(title="Trading Signals Distribution", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ml_predictor_details(self):
        """Render ML predictor agent details"""
        st.subheader("🧠 ML Prediction Details")
        
        # Sample ML metrics
        ml_data = {
            'models_trained': 15,
            'predictions_made': 120,
            'avg_accuracy': 82.5,
            'model_types': ['XGBoost', 'LightGBM', 'Random Forest']
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Trained", ml_data['models_trained'])
        
        with col2:
            st.metric("Predictions Made", ml_data['predictions_made'])
        
        with col3:
            st.metric("Avg Accuracy", f"{ml_data['avg_accuracy']}%")
        
        # Model performance comparison
        model_performance = {
            'XGBoost': 85.2,
            'LightGBM': 83.8,
            'Random Forest': 78.9
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(model_performance.keys()), y=list(model_performance.values()))
        ])
        fig.update_layout(title="Model Accuracy Comparison (%)", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_backtest_agent_details(self):
        """Render backtest agent details"""
        st.subheader("🔄 Backtest Details")
        
        # Sample backtest metrics
        backtest_data = {
            'strategies_tested': 8,
            'total_backtests': 45,
            'avg_return': 12.5,
            'win_rate': 68.2
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategies", backtest_data['strategies_tested'])
        
        with col2:
            st.metric("Total Tests", backtest_data['total_backtests'])
        
        with col3:
            st.metric("Avg Return", f"{backtest_data['avg_return']}%")
        
        with col4:
            st.metric("Win Rate", f"{backtest_data['win_rate']}%")
    
    def _render_trade_executor_details(self):
        """Render trade executor agent details"""
        st.subheader("⚡ Trade Execution Details")
        
        # Sample execution metrics
        execution_data = {
            'signals_generated': 28,
            'high_confidence_signals': 12,
            'risk_alerts': 3,
            'portfolio_positions': 5
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Signals Generated", execution_data['signals_generated'])
        
        with col2:
            st.metric("High Confidence", execution_data['high_confidence_signals'])
        
        with col3:
            st.metric("Risk Alerts", execution_data['risk_alerts'])
        
        with col4:
            st.metric("Portfolio Positions", execution_data['portfolio_positions'])
    
    def _render_whale_detector_details(self):
        """Render whale detector agent details"""
        st.subheader("🐋 Whale Detection Details")
        
        # Sample whale detection metrics
        whale_data = {
            'whale_activities': 15,
            'critical_alerts': 3,
            'symbols_monitored': 100,
            'large_transactions': 8
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Whale Activities", whale_data['whale_activities'])
        
        with col2:
            st.metric("Critical Alerts", whale_data['critical_alerts'])
        
        with col3:
            st.metric("Symbols Monitored", whale_data['symbols_monitored'])
        
        with col4:
            st.metric("Large Transactions", whale_data['large_transactions'])
    
    def _render_agent_configuration(self, agent_name: str, config: dict):
        """Render agent configuration settings"""
        with st.expander("⚙️ Agent Configuration", expanded=False):
            st.subheader(f"Configuration for {agent_name.replace('_', ' ').title()}")
            
            # Display current configuration
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.json(config)
            
            with config_col2:
                st.subheader("Update Settings")
                
                # Basic settings
                enabled = st.checkbox("Enabled", value=config.get('enabled', False))
                update_interval = st.number_input(
                    "Update Interval (seconds)",
                    min_value=10,
                    max_value=3600,
                    value=config.get('update_interval', 300)
                )
                
                # Agent-specific settings
                if agent_name == 'sentiment':
                    st.selectbox("Sentiment Sources", ['Twitter', 'Reddit', 'News'], index=0)
                elif agent_name == 'technical':
                    st.multiselect("Technical Indicators", 
                                 ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages'],
                                 default=['RSI', 'MACD'])
                elif agent_name == 'ml_predictor':
                    st.multiselect("ML Models", 
                                 ['XGBoost', 'LightGBM', 'Random Forest'],
                                 default=['XGBoost', 'LightGBM'])
                
                if st.button(f"Update {agent_name.title()} Config"):
                    # Update configuration logic would go here
                    st.success(f"Configuration updated for {agent_name}")
    
    def _render_agent_logs(self):
        """Render agent logs and activities"""
        st.subheader("📝 Agent Logs")
        
        # Sample log entries
        log_entries = [
            {"timestamp": "2024-01-15 10:30:25", "agent": "sentiment", "level": "INFO", "message": "Sentiment analysis completed for 45 currencies"},
            {"timestamp": "2024-01-15 10:29:18", "agent": "technical", "level": "INFO", "message": "Technical indicators calculated for BTC/USD"},
            {"timestamp": "2024-01-15 10:28:45", "agent": "ml_predictor", "level": "INFO", "message": "ML prediction model updated"},
            {"timestamp": "2024-01-15 10:27:32", "agent": "whale_detector", "level": "WARNING", "message": "Large transaction detected on ETH/USD"},
            {"timestamp": "2024-01-15 10:26:15", "agent": "backtest", "level": "INFO", "message": "Backtest completed for moving average strategy"},
        ]
        
        # Log level filter
        col1, col2 = st.columns([3, 1])
        
        with col2:
            log_level_filter = st.selectbox(
                "Filter Level",
                ["All", "INFO", "WARNING", "ERROR"],
                index=0
            )
        
        # Display logs
        for entry in log_entries:
            if log_level_filter == "All" or entry["level"] == log_level_filter:
                level_color = {
                    "INFO": "🔵",
                    "WARNING": "🟡", 
                    "ERROR": "🔴"
                }.get(entry["level"], "⚪")
                
                st.write(f"{level_color} **{entry['timestamp']}** [{entry['agent']}] {entry['message']}")
