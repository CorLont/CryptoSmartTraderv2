import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup daily logging system first
from config.daily_logging_config import setup_daily_logging
daily_logger = setup_daily_logging()

from containers import ApplicationContainer
from config.logging_config import setup_logging
from config.settings import config
from dashboards.main_dashboard import MainDashboard
from dashboards.agent_dashboard import AgentDashboard  
from dashboards.portfolio_dashboard import PortfolioDashboard
from dashboards.performance_dashboard import PerformanceDashboard
from dashboards.analysis_control_dashboard import AnalysisControlDashboard
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize dependency injection container
container = ApplicationContainer()

def main():
    """Main application entry point with dependency injection"""
    st.set_page_config(
        page_title="CryptoSmartTrader V2",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize core systems using dependency injection
    try:
        # Get dependencies from container
        config_manager = container.config()
        health_monitor = container.health_monitor()
        market_scanner = container.market_scanner()
        monitoring_system = container.monitoring_system()
        
        # Store in session state for access across pages
        if 'container' not in st.session_state:
            st.session_state.container = container
        if 'config_manager' not in st.session_state:
            st.session_state.config_manager = config_manager
        if 'health_monitor' not in st.session_state:
            st.session_state.health_monitor = health_monitor
        
        # Auto-start comprehensive market scanning
        if not st.session_state.get('market_scanner_started', False):
            market_scanner.start_comprehensive_scanning()
            monitoring_system.start_monitoring()
            st.session_state.market_scanner_started = True
            logger.info("Auto-started comprehensive market scanning and production monitoring")
            
    except Exception as e:
        st.error(f"Failed to initialize core systems: {str(e)}")
        logger.error(f"Core system initialization failed: {str(e)}", 
                    extra={"component": "app_initialization", "error": str(e)})
        return
    
    # Sidebar navigation
    st.sidebar.title("🚀 CryptoSmartTrader V2")
    st.sidebar.markdown("---")
    
    # System health display
    health_status = health_monitor.get_system_health()
    health_color = {
        'A': '🟢', 'B': '🟡', 'C': '🟡', 
        'D': '🟠', 'E': '🔴', 'F': '🔴'
    }.get(health_status.get('grade', 'F'), '🔴')
    
    st.sidebar.metric(
        "System Health", 
        f"{health_color} {health_status.get('grade', 'F')}",
        f"{health_status.get('score', 0):.1f}%"
    )
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🚀 Alpha Opportunities",
            "🎯 Main Dashboard",
            "🌍 Comprehensive Market",
            "🧠 Advanced Analytics",
            "🤖 AI/ML Engine",
            "🧠 Crypto AI System",
            "🚀 ML/AI Differentiators",
            "🏥 System Health",
            "🎛️ Analysis Control",
            "🔧 Agent Dashboard", 
            "💼 Portfolio Dashboard",
            "🔍 Production Monitoring",
            "📊 Performance Dashboard",
            "🔧 Automated Feature Engineering",
            "🌍 Market Regime Detection",
            "🧠 Causal Inference",
            "🤖 RL Portfolio Allocation",
            "⚙️ System Configuration",
            "📈 Health Monitor"
        ]
    )
    
    # Main content area
    if page == "🚀 Alpha Opportunities":
        from dashboards.alpha_opportunities_dashboard import AlphaOpportunitiesDashboard
        alpha_dashboard = AlphaOpportunitiesDashboard(container)
        alpha_dashboard.render()
        
    elif page == "🎯 Main Dashboard":
        main_dashboard = MainDashboard(config_manager, health_monitor)
        main_dashboard.render()
        
    elif page == "🌍 Comprehensive Market":
        from dashboards.comprehensive_market_dashboard import ComprehensiveMarketDashboard
        market_dashboard = ComprehensiveMarketDashboard(container)
        market_dashboard.render()
        
    elif page == "🧠 Advanced Analytics":
        from dashboards.advanced_analytics_dashboard import AdvancedAnalyticsDashboard
        advanced_dashboard = AdvancedAnalyticsDashboard(container)
        advanced_dashboard.render()
        
    elif page == "🤖 AI/ML Engine":
        from dashboards.ai_ml_dashboard import AIMLDashboard
        ai_ml_dashboard = AIMLDashboard(container)
        ai_ml_dashboard.render()
        
    elif page == "🧠 Crypto AI System":
        from dashboards.crypto_ai_system_dashboard import CryptoAISystemDashboard
        crypto_ai_dashboard = CryptoAISystemDashboard(container)
        crypto_ai_dashboard.render()
        
    elif page == "🚀 ML/AI Differentiators":
        from dashboards.ml_ai_differentiators_dashboard import MLAIDifferentiatorsDashboard
        ml_ai_dashboard = MLAIDifferentiatorsDashboard(container)
        ml_ai_dashboard.render()
        
    elif page == "🏥 System Health":
        from dashboards.system_health_dashboard import SystemHealthDashboard
        health_dashboard = SystemHealthDashboard(container)
        health_dashboard.render()
        
    elif page == "🎛️ Analysis Control":
        analysis_control = AnalysisControlDashboard(container)
        analysis_control.render()
        
    elif page == "🔧 Agent Dashboard":
        agent_dashboard = AgentDashboard(config_manager, health_monitor)
        agent_dashboard.render()
        
    elif page == "💼 Portfolio Dashboard":
        portfolio_dashboard = PortfolioDashboard(config_manager, health_monitor)
        portfolio_dashboard.render()
        
    elif page == "🔍 Production Monitoring":
        from dashboards.production_monitoring_dashboard import ProductionMonitoringDashboard
        monitoring_dashboard = ProductionMonitoringDashboard(container)
        monitoring_dashboard.render()
        
    elif page == "📊 Performance Dashboard":
        performance_dashboard = PerformanceDashboard(container)
        performance_dashboard.render()
        
    elif page == "🔧 Automated Feature Engineering":
        from dashboards.automated_feature_engineering_dashboard import AutomatedFeatureEngineeringDashboard
        fe_dashboard = AutomatedFeatureEngineeringDashboard()
        fe_dashboard.render()
        
    elif page == "🌍 Market Regime Detection":
        from dashboards.market_regime_dashboard import MarketRegimeDashboard
        regime_dashboard = MarketRegimeDashboard()
        regime_dashboard.render()
        
    elif page == "🧠 Causal Inference":
        from dashboards.causal_inference_dashboard import CausalInferenceDashboard
        causal_dashboard = CausalInferenceDashboard()
        causal_dashboard.render()
        
    elif page == "🤖 RL Portfolio Allocation":
        from dashboards.rl_portfolio_dashboard import RLPortfolioDashboard
        rl_dashboard = RLPortfolioDashboard()
        rl_dashboard.render()
        
    elif page == "⚙️ System Configuration":
        render_config_page(config_manager)
        
    elif page == "📈 Health Monitor":
        render_health_page(health_monitor)

def render_config_page(config_manager):
    """Render system configuration page"""
    st.title("⚙️ System Configuration")
    
    # Configuration sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Exchange Settings")
        exchanges = st.multiselect(
            "Active Exchanges",
            ["kraken", "binance", "kucoin", "huobi"],
            default=config_manager.get("exchanges", ["kraken"])
        )
        
        api_rate_limit = st.number_input(
            "API Rate Limit (req/min)",
            min_value=10,
            max_value=1000,
            value=config_manager.get("api_rate_limit", 100)
        )
        
        st.subheader("ML Model Settings")
        prediction_horizons = st.multiselect(
            "Prediction Horizons",
            ["1h", "4h", "1d", "3d", "7d", "30d"],
            default=config_manager.get("prediction_horizons", ["1d", "7d"])
        )
        
    with col2:
        st.subheader("Monitoring Settings")
        health_check_interval = st.number_input(
            "Health Check Interval (minutes)",
            min_value=1,
            max_value=60,
            value=config_manager.get("health_check_interval", 5)
        )
        
        alert_thresholds = st.slider(
            "Alert Threshold (%)",
            min_value=0,
            max_value=100,
            value=config_manager.get("alert_threshold", 80)
        )
        
        st.subheader("Performance Settings")
        enable_gpu = st.checkbox(
            "Enable GPU Acceleration",
            value=config_manager.get("enable_gpu", False)
        )
        
        max_coins = st.number_input(
            "Maximum Coins to Process",
            min_value=50,
            max_value=1000,
            value=config_manager.get("max_coins", 453)
        )
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        new_config = {
            "exchanges": exchanges,
            "api_rate_limit": api_rate_limit,
            "prediction_horizons": prediction_horizons,
            "health_check_interval": health_check_interval,
            "alert_threshold": alert_thresholds,
            "enable_gpu": enable_gpu,
            "max_coins": max_coins
        }
        
        if config_manager.update_config(new_config):
            st.success("✅ Configuration saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save configuration")

def render_health_page(health_monitor):
    """Render system health monitoring page"""
    st.title("📊 System Health Monitor")
    
    # Get comprehensive health data
    health_data = health_monitor.get_detailed_health()
    
    # Overall health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Grade",
            health_data.get('grade', 'F'),
            delta=health_data.get('grade_change', 0)
        )
    
    with col2:
        st.metric(
            "System Score",
            f"{health_data.get('score', 0):.1f}%",
            delta=f"{health_data.get('score_change', 0):.1f}%"
        )
    
    with col3:
        st.metric(
            "Active Agents",
            health_data.get('active_agents', 0),
            delta=health_data.get('agent_change', 0)
        )
    
    with col4:
        st.metric(
            "Data Coverage",
            f"{health_data.get('data_coverage', 0):.1f}%",
            delta=f"{health_data.get('coverage_change', 0):.1f}%"
        )
    
    # Detailed health breakdown
    st.subheader("Detailed Health Analysis")
    
    # Agent health status
    agent_health = health_data.get('agent_health', {})
    if agent_health:
        st.subheader("🤖 Agent Status")
        agent_cols = st.columns(len(agent_health))
        
        for idx, (agent_name, status) in enumerate(agent_health.items()):
            with agent_cols[idx]:
                status_icon = "🟢" if status.get('status') == 'healthy' else "🔴"
                st.metric(
                    agent_name.replace('_', ' ').title(),
                    f"{status_icon} {status.get('status', 'unknown')}",
                    delta=f"{status.get('uptime', 0):.1f}h uptime"
                )
    
    # System resources
    st.subheader("💻 System Resources")
    resource_data = health_data.get('resources', {})
    
    resource_col1, resource_col2, resource_col3 = st.columns(3)
    
    with resource_col1:
        cpu_usage = resource_data.get('cpu_percent', 0)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%")
        st.progress(cpu_usage / 100)
    
    with resource_col2:
        memory_usage = resource_data.get('memory_percent', 0)
        st.metric("Memory Usage", f"{memory_usage:.1f}%")
        st.progress(memory_usage / 100)
    
    with resource_col3:
        disk_usage = resource_data.get('disk_percent', 0)
        st.metric("Disk Usage", f"{disk_usage:.1f}%")
        st.progress(disk_usage / 100)
    
    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    alerts = health_data.get('recent_alerts', [])
    
    if alerts:
        for alert in alerts[:5]:  # Show last 5 alerts
            alert_type = alert.get('type', 'info')
            alert_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(alert_type, "🔵")
            
            st.info(f"{alert_icon} **{alert.get('timestamp', 'Unknown time')}**: {alert.get('message', 'No message')}")
    else:
        st.success("✅ No recent alerts - system running smoothly!")

if __name__ == "__main__":
    main()
