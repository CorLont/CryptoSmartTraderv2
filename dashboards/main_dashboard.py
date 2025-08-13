#!/usr/bin/env python3
"""
Main Dashboard - Enterprise Streamlit application with performance optimization
"""

import streamlit as st
import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dashboards.pages import market_page, agents_page, portfolio_page, health_page
from dashboards.utils.cache_manager import CacheManager
from dashboards.utils.session_state import SessionStateManager
from dashboards.utils.async_updater import AsyncDataUpdater
from core.logging_config import setup_logging, set_correlation_id
from cryptosmarttrader.config import settings


# Configure Streamlit
st.set_page_config(
    page_title="CryptoSmartTrader V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/help",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": f"CryptoSmartTrader V2 - Enterprise Trading Intelligence",
    },
)

# Initialize logging
setup_logging(log_level=settings.LOG_LEVEL, enable_json=True, enable_trading_logs=True)

# Initialize managers
cache_manager = CacheManager()
session_manager = SessionStateManager()
data_updater = AsyncDataUpdater()


def initialize_dashboard():
    """Initialize dashboard state and warm up caches"""

    # Set correlation ID for session
    if "correlation_id" not in st.session_state:
        st.session_state.correlation_id = set_correlation_id()
    else:
        set_correlation_id(st.session_state.correlation_id)

    # Initialize session state
    session_manager.initialize_session()

    # Warm up critical caches
    with st.spinner("Initializing dashboard..."):
        cache_manager.warm_up_caches()

    # Start async data updates if not already running
    if not st.session_state.get("async_updater_started", False):
        data_updater.start_background_updates()
        st.session_state.async_updater_started = True


def render_sidebar():
    """Render navigation sidebar"""

    st.sidebar.title("📈 CryptoSmartTrader V2")
    st.sidebar.markdown("---")

    # System status indicator
    health_score = st.session_state.get("system_health_score", 0.0)
    health_grade = st.session_state.get("system_health_grade", "F")

    if health_score >= 90:
        status_color = "🟢"
    elif health_score >= 70:
        status_color = "🟡"
    else:
        status_color = "🔴"

    st.sidebar.metric(
        "System Health", f"Grade {health_grade}", f"{status_color} {health_score:.1f}%"
    )

    # Navigation
    st.sidebar.markdown("### Navigation")

    pages = {
        "🏪 Market Overview": "market",
        "🤖 AI Agents": "agents",
        "💼 Portfolio": "portfolio",
        "🏥 System Health": "health",
    }

    selected_page = st.sidebar.radio("Select Page", list(pages.keys()), key="page_selection")

    st.session_state.current_page = pages[selected_page]

    # Quick stats
    st.sidebar.markdown("### Quick Stats")

    trading_enabled = st.session_state.get("trading_enabled", False)
    st.sidebar.metric("Trading Status", "🟢 Enabled" if trading_enabled else "🔴 Disabled")

    active_positions = st.session_state.get("active_positions", 0)
    st.sidebar.metric("Active Positions", active_positions)

    daily_pnl = st.session_state.get("daily_pnl", 0.0)
    pnl_color = "normal" if daily_pnl >= 0 else "inverse"
    st.sidebar.metric("Daily P&L", f"${daily_pnl:,.2f}", delta_color=pnl_color)

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=st.session_state.get("auto_refresh", True),
        help="Automatically refresh data every 30 seconds",
    )
    st.session_state.auto_refresh = auto_refresh

    # Refresh interval
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.get("refresh_interval", 30),
            step=10,
        )
        st.session_state.refresh_interval = refresh_interval

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        cache_manager.clear_all_caches()
        st.rerun()

    # Environment info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Environment: {settings.ENVIRONMENT}")
    st.sidebar.caption(f"Version: {settings.APP_VERSION}")


def render_main_content():
    """Render main dashboard content based on selected page"""

    current_page = st.session_state.get("current_page", "market")

    # Page routing
    if current_page == "market":
        market_page.render()
    elif current_page == "agents":
        agents_page.render()
    elif current_page == "portfolio":
        portfolio_page.render()
    elif current_page == "health":
        health_page.render()
    else:
        st.error(f"Unknown page: {current_page}")


def handle_auto_refresh():
    """Handle automatic data refresh"""

    if not st.session_state.get("auto_refresh", True):
        return

    refresh_interval = st.session_state.get("refresh_interval", 30)
    last_refresh = st.session_state.get("last_refresh_time", 0)
    current_time = time.time()

    if current_time - last_refresh >= refresh_interval:
        # Update data asynchronously
        data_updater.trigger_refresh()
        st.session_state.last_refresh_time = current_time

        # Show refresh indicator
        with st.empty():
            st.info("🔄 Refreshing data...")
            time.sleep(1)


def render_performance_metrics():
    """Render performance metrics in expander"""

    with st.expander("⚡ Performance Metrics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            page_load_time = st.session_state.get("page_load_time", 0)
            st.metric("Page Load Time", f"{page_load_time:.2f}s")

        with col2:
            cache_hit_rate = cache_manager.get_cache_hit_rate()
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")

        with col3:
            active_connections = st.session_state.get("active_connections", 0)
            st.metric("API Connections", active_connections)

        with col4:
            memory_usage = st.session_state.get("memory_usage_mb", 0)
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")


def main():
    """Main dashboard application"""

    start_time = time.time()

    try:
        # Initialize dashboard
        initialize_dashboard()

        # Custom CSS for better styling
        st.markdown(
            """
        <style>
        .main-header {
            padding: 1rem 0;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007acc;
        }
        
        .status-success { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-danger { color: #dc3545; }
        
        .data-table {
            font-size: 0.9rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Render sidebar
        render_sidebar()

        # Main content area
        with st.container():
            # Header
            st.markdown('<div class="main-header">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])

            with col1:
                st.title("CryptoSmartTrader V2 Dashboard")
                st.caption("Enterprise Cryptocurrency Trading Intelligence Platform")

            with col2:
                # Real-time clock
                current_time = time.strftime("%H:%M:%S UTC", time.gmtime())
                st.metric("Current Time", current_time)

            st.markdown("</div>", unsafe_allow_html=True)

            # Handle auto-refresh
            handle_auto_refresh()

            # Render main content
            render_main_content()

            # Performance metrics
            render_performance_metrics()

        # Store page load time
        page_load_time = time.time() - start_time
        st.session_state.page_load_time = page_load_time

        # Footer
        st.markdown("---")
        st.markdown(
            f"Dashboard loaded in {page_load_time:.2f}s | "
            f"Session ID: {st.session_state.correlation_id[:8]}... | "
            f"Environment: {settings.ENVIRONMENT}"
        )

    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logging.error(f"Dashboard error: {e}", exc_info=True)

        # Show error details in debug mode
        if settings.DEBUG:
            st.exception(e)


if __name__ == "__main__":
    main()
