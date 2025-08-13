#!/usr/bin/env python3
"""
Return Attribution Dashboard Demo
Demo interactive PnL attribution analysis with execution optimization insights.
"""

import streamlit as st
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now import our attribution dashboard
from cryptosmarttrader.attribution.attribution_dashboard import AttributionDashboard

def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Return Attribution Demo",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎯 Return Attribution System Demo")
    st.markdown("**PnL Decomposition: Alpha • Fees • Slippage • Timing • Sizing**")
    
    # Introduction
    with st.expander("ℹ️ About Return Attribution Analysis", expanded=False):
        st.markdown("""
        **Return Attribution Analysis** splits portfolio performance into components:
        
        - **🎯 Alpha**: Pure strategy performance vs benchmark
        - **💰 Fees**: Trading and management costs (maker/taker fees)
        - **📉 Slippage**: Market impact and execution costs
        - **⏱️ Timing**: Execution timing differences and latency costs
        - **📏 Sizing**: Position sizing impact on performance
        - **🌊 Market Impact**: Large order impact on markets
        
        **Goal**: Optimize where the execution pain points are (vaak execution).
        """)
    
    # Create and render the dashboard
    dashboard = AttributionDashboard()
    
    # Add demo warning
    st.info("📊 **Demo Mode**: Using simulated trading data for demonstration purposes")
    
    # Render the main dashboard
    dashboard.render_dashboard()
    
    # Footer with system info
    st.markdown("---")
    st.markdown("**🚀 CryptoSmartTrader V2 - Enterprise Return Attribution System**")


if __name__ == "__main__":
    main()