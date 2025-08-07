"""
CryptoSmartTrader V2 - Dashboard Manager
Centralized dashboard management for all system components
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class DashboardManager:
    """Centralized dashboard manager for all system components"""
    
    def __init__(self, container):
        self.container = container
        
    def render_ai_ml_dashboard(self):
        """Render AI/ML Engine dashboard"""
        try:
            from dashboards.ai_ml_dashboard import AIMLDashboard
            dashboard = AIMLDashboard(self.container)
            dashboard.render()
        except Exception as e:
            st.error(f"Failed to load AI/ML dashboard: {e}")
    
    def render_crypto_ai_system_dashboard(self):
        """Render Complete Crypto AI System dashboard"""
        try:
            from dashboards.crypto_ai_system_dashboard import CryptoAISystemDashboard
            dashboard = CryptoAISystemDashboard(self.container)
            dashboard.render()
        except Exception as e:
            st.error(f"Failed to load Crypto AI System dashboard: {e}")
            st.exception(e)
    
    def render_analysis_dashboard(self):
        """Render analysis control dashboard"""
        try:
            from dashboards.analysis_control_dashboard import AnalysisControlDashboard
            dashboard = AnalysisControlDashboard(self.container)
            dashboard.render()
        except Exception as e:
            st.error(f"Failed to load analysis dashboard: {e}")
    
    def render_comprehensive_market_dashboard(self):
        """Render comprehensive market dashboard"""
        try:
            from dashboards.comprehensive_market_dashboard import ComprehensiveMarketDashboard
            dashboard = ComprehensiveMarketDashboard(self.container)
            dashboard.render()
        except Exception as e:
            st.error(f"Failed to load market dashboard: {e}")
    
    def render_production_monitoring_dashboard(self):
        """Render production monitoring dashboard"""
        try:
            from dashboards.production_monitoring_dashboard import ProductionMonitoringDashboard
            dashboard = ProductionMonitoringDashboard(self.container)
            dashboard.render()
        except Exception as e:
            st.error(f"Failed to load production monitoring dashboard: {e}")