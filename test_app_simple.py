#!/usr/bin/env python3
"""
Simple test app to verify basic Streamlit functionality
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append('.')

def main():
    """Simple test application"""
    st.set_page_config(
        page_title="CryptoSmartTrader V2 - Test",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 CryptoSmartTrader V2 - System Test")
    st.markdown("Testing basic Streamlit functionality...")
    
    # Basic content
    st.header("System Status")
    st.success("✅ Streamlit is working correctly!")
    
    # Test components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Metric 1", "100%", delta="5%")
    
    with col2:
        st.metric("Test Metric 2", "99.9%", delta="0.1%")
    
    with col3:
        st.metric("Test Metric 3", "Active", delta="OK")
    
    # Test charts
    import pandas as pd
    import numpy as np
    
    st.header("Test Chart")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    st.line_chart(chart_data)
    
    # Test sidebar
    st.sidebar.title("🧪 Test Sidebar")
    st.sidebar.success("Sidebar is working!")
    
    test_value = st.sidebar.slider("Test Slider", 0, 100, 50)
    st.sidebar.write(f"Selected value: {test_value}")
    
    # Test advanced components import
    st.header("Advanced Component Tests")
    
    try:
        # Test if our core modules can be imported
        st.subheader("Core Module Imports")
        
        import_results = []
        
        try:
            from containers import ApplicationContainer
            import_results.append("✅ ApplicationContainer imported")
        except Exception as e:
            import_results.append(f"❌ ApplicationContainer failed: {e}")
        
        try:
            from config.daily_logging_config import setup_daily_logging
            import_results.append("✅ Daily logging imported")
        except Exception as e:
            import_results.append(f"❌ Daily logging failed: {e}")
        
        try:
            from dashboards.main_dashboard import MainDashboard
            import_results.append("✅ MainDashboard imported")
        except Exception as e:
            import_results.append(f"❌ MainDashboard failed: {e}")
        
        try:
            from core.causal_inference_engine import get_causal_inference_engine
            import_results.append("✅ Causal Inference imported")
        except Exception as e:
            import_results.append(f"❌ Causal Inference failed: {e}")
        
        try:
            from core.reinforcement_portfolio_allocator import get_rl_portfolio_allocator
            import_results.append("✅ RL Portfolio imported")
        except Exception as e:
            import_results.append(f"❌ RL Portfolio failed: {e}")
        
        # Display results
        for result in import_results:
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)
        
    except Exception as e:
        st.error(f"Advanced component test failed: {e}")
    
    # Test button
    if st.button("🚀 Test Button"):
        st.balloons()
        st.success("Button clicked successfully!")
    
    st.markdown("---")
    st.info("If you can see this page, Streamlit is working correctly!")

if __name__ == "__main__":
    main()