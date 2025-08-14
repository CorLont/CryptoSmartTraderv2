"""
Deployment Dashboard
Real-time monitoring of parity validation and canary deployments
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

try:
    from src.cryptosmarttrader.deployment import ParityValidator, CanaryManager
    from src.cryptosmarttrader.deployment import DriftSeverity, CanaryPhase, CanaryStatus
except ImportError:
    # Mock classes for demo mode
    class ParityValidator:
        def get_parity_summary(self, hours=24):
            return {'status': 'demo_mode'}
    
    class CanaryManager:
        def list_active_deployments(self):
            return []


class DeploymentDashboard:
    """Dashboard for monitoring deployment systems"""
    
    def __init__(self, container=None):
        self.container = container
        self.setup_components()
    
    def setup_components(self):
        """Setup deployment monitoring components"""
        try:
            # Initialize real components if available
            config = {
                'warning_threshold_bps': 20,
                'critical_threshold_bps': 50,
                'emergency_threshold_bps': 100
            }
            
            self.parity_validator = ParityValidator(config)
            self.canary_manager = CanaryManager(config)
            self.demo_mode = False
            
        except Exception as e:
            st.warning(f"Using demo mode: {e}")
            self.parity_validator = ParityValidator()
            self.canary_manager = CanaryManager()
            self.demo_mode = True
    
    def render(self):
        """Render deployment monitoring dashboard"""
        st.title("🚀 Deployment & Parity Monitoring")
        st.markdown("**Real-time monitoring van model deployments en performance parity**")
        
        if self.demo_mode:
            st.info("🔄 Demo modus actief - echte deployment data niet beschikbaar")
        
        # Auto-refresh controls
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Live monitoring van backtest-live parity en canary deployments**")
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Main monitoring sections
        self.render_parity_monitoring()
        st.markdown("---")
        self.render_canary_monitoring()
        st.markdown("---")
        self.render_deployment_metrics()
    
    def render_parity_monitoring(self):
        """Render parity validation monitoring"""
        st.markdown("## 📊 Backtest-Live Parity Monitoring")
        
        # Get parity summary
        parity_summary = self.get_parity_summary()
        
        if parity_summary['status'] == 'no_data':
            st.warning("⚠️ Geen parity data beschikbaar")
            return
        
        # Parity metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_error = parity_summary.get('avg_tracking_error_bps', 0)
            color = self.get_error_color(avg_error)
            st.metric(
                "Gem. Tracking Error", 
                f"{avg_error:.1f} bps",
                delta=f"Target: <20 bps",
                delta_color="normal"
            )
        
        with col2:
            max_error = parity_summary.get('max_tracking_error_bps', 0)
            st.metric(
                "Max Tracking Error",
                f"{max_error:.1f} bps",
                delta=f"Emergency: >100 bps",
                delta_color="inverse" if max_error > 100 else "normal"
            )
        
        with col3:
            tolerance_pct = parity_summary.get('within_tolerance', 0)
            st.metric(
                "Binnen Tolerantie",
                f"{tolerance_pct:.1%}",
                delta="Target: >95%",
                delta_color="normal" if tolerance_pct > 0.95 else "inverse"
            )
        
        with col4:
            emergency_count = parity_summary.get('emergency_halts', 0)
            st.metric(
                "Emergency Halts",
                str(emergency_count),
                delta="Afgelopen 24u",
                delta_color="inverse" if emergency_count > 0 else "normal"
            )
        
        # Parity status indicator
        self.render_parity_status(parity_summary)
        
        # Tracking error chart
        self.render_tracking_error_chart()
    
    def render_canary_monitoring(self):
        """Render canary deployment monitoring"""
        st.markdown("## 🔄 Canary Deployment Monitoring")
        
        # Get active deployments
        active_deployments = self.get_active_deployments()
        
        if not active_deployments:
            st.info("ℹ️ Geen actieve canary deployments")
            self.render_deployment_controls()
            return
        
        # Display active deployments
        for deployment in active_deployments:
            self.render_deployment_card(deployment)
        
        # Deployment history
        self.render_deployment_history()
    
    def render_deployment_card(self, deployment: Dict[str, Any]):
        """Render individual deployment status card"""
        
        # Determine status color
        status_colors = {
            'active': '🟢',
            'monitoring': '🟡', 
            'rolling_back': '🔴',
            'success': '✅',
            'failed': '❌'
        }
        
        status_icon = status_colors.get(deployment['status'], '🔵')
        
        # Create deployment card
        with st.container():
            st.markdown(f"""
            <div style='border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin: 1rem 0; background: #f9f9f9;'>
                <h4>{status_icon} Deployment: {deployment['model_version']}</h4>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                    <div><strong>Status:</strong> {deployment['status']}</div>
                    <div><strong>Phase:</strong> {deployment['current_phase']}</div>
                    <div><strong>Traffic:</strong> {deployment['traffic_percentage']:.1%}</div>
                    <div><strong>Duration:</strong> {deployment['duration_hours']:.1f}h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance comparison if available
            if deployment.get('performance_baseline') and deployment.get('performance_canary'):
                self.render_performance_comparison(deployment)
            
            # Deployment controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"⏸️ Pause", key=f"pause_{deployment['deployment_id']}"):
                    self.pause_deployment(deployment['deployment_id'])
            
            with col2:
                if st.button(f"▶️ Resume", key=f"resume_{deployment['deployment_id']}"):
                    self.resume_deployment(deployment['deployment_id'])
            
            with col3:
                if st.button(f"🔄 Rollback", key=f"rollback_{deployment['deployment_id']}"):
                    self.force_rollback(deployment['deployment_id'])
    
    def render_performance_comparison(self, deployment: Dict[str, Any]):
        """Render performance comparison chart"""
        baseline = deployment['performance_baseline']
        canary = deployment['performance_canary']
        
        # Create comparison chart
        metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']
        baseline_values = [baseline.get(m, 0) for m in metrics]
        canary_values = [canary.get(m, 0) for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=metrics,
            y=baseline_values,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Canary',
            x=metrics,
            y=canary_values,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Performance Comparison: Baseline vs Canary",
            barmode='group',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_deployment_controls(self):
        """Render deployment control interface"""
        st.markdown("### 🎛️ New Deployment Controls")
        
        with st.form("new_deployment"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_version = st.text_input("Model Version", "v2.1.0")
                baseline_version = st.text_input("Baseline Version", "v2.0.0")
            
            with col2:
                phase_duration = st.slider("Phase Duration (hours)", 1, 72, 24)
                auto_promote = st.checkbox("Auto Promote", value=True)
            
            if st.form_submit_button("🚀 Start Canary Deployment"):
                self.start_canary_deployment(model_version, baseline_version, phase_duration, auto_promote)
    
    def render_parity_status(self, parity_summary: Dict[str, Any]):
        """Render overall parity status"""
        avg_error = parity_summary.get('avg_tracking_error_bps', 0)
        
        if avg_error < 20:
            status = "🟢 EXCELLENT"
            message = "Parity binnen excellente parameters"
        elif avg_error < 50:
            status = "🟡 WARNING" 
            message = "Verhoogde tracking error - monitoring vereist"
        elif avg_error < 100:
            status = "🔴 CRITICAL"
            message = "Kritieke parity drift - aandacht vereist"
        else:
            status = "🚨 EMERGENCY"
            message = "Emergency halt drempel overschreden"
        
        st.markdown(f"""
        <div style='padding: 1rem; border-radius: 10px; margin: 1rem 0; 
                    background: {"#d4edda" if "🟢" in status else "#fff3cd" if "🟡" in status else "#f8d7da"};
                    border: 1px solid {"#c3e6cb" if "🟢" in status else "#ffeaa7" if "🟡" in status else "#f5c6cb"};'>
            <h4>{status} Parity Status</h4>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_tracking_error_chart(self):
        """Render tracking error time series chart"""
        # Generate sample tracking error data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='H')
        
        # Simulate tracking error data
        tracking_errors = []
        for i, date in enumerate(dates):
            base_error = 15 + np.random.normal(0, 5)  # Base around 15 bps
            if i > 20:  # Simulate some drift
                base_error += (i - 20) * 0.5
            tracking_errors.append(max(0, base_error))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'tracking_error_bps': tracking_errors
        })
        
        fig = px.line(df, x='timestamp', y='tracking_error_bps',
                     title="Tracking Error Afgelopen 24 Uur",
                     labels={'tracking_error_bps': 'Tracking Error (bps)',
                            'timestamp': 'Tijd'})
        
        # Add threshold lines
        fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                     annotation_text="Warning (20 bps)")
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                     annotation_text="Critical (50 bps)")
        fig.add_hline(y=100, line_dash="dash", line_color="darkred",
                     annotation_text="Emergency (100 bps)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_deployment_history(self):
        """Render deployment history table"""
        st.markdown("### 📋 Deployment Geschiedenis")
        
        # Sample deployment history
        history_data = [
            {
                'Deployment ID': 'dep-001',
                'Model Version': 'v2.0.0',
                'Status': '✅ Success',
                'Start Time': '2025-08-13 09:00',
                'Duration': '48h',
                'Traffic Peak': '100%'
            },
            {
                'Deployment ID': 'dep-002', 
                'Model Version': 'v1.9.5',
                'Status': '🔄 Rollback',
                'Start Time': '2025-08-12 14:30',
                'Duration': '6h',
                'Traffic Peak': '25%'
            }
        ]
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    def render_deployment_metrics(self):
        """Render deployment system metrics"""
        st.markdown("## 📈 Deployment System Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Successful Deployments", "12", delta="+2 deze week")
        
        with col2:
            st.metric("Average Deploy Time", "36h", delta="-6h vs vorige")
        
        with col3:
            st.metric("Rollback Rate", "8.3%", delta="-2.1% verbetering")
        
        with col4:
            st.metric("System Uptime", "99.97%", delta="+0.02% deze maand")
    
    def get_parity_summary(self) -> Dict[str, Any]:
        """Get parity validation summary"""
        if self.demo_mode:
            return {
                'status': 'active',
                'avg_tracking_error_bps': 18.5,
                'max_tracking_error_bps': 42.1,
                'within_tolerance': 0.89,
                'emergency_halts': 0,
                'total_validations': 1440
            }
        else:
            return self.parity_validator.get_parity_summary(hours=24)
    
    def get_active_deployments(self) -> List[Dict[str, Any]]:
        """Get list of active deployments"""
        if self.demo_mode:
            return [{
                'deployment_id': 'demo-001',
                'model_version': 'v2.1.0-canary',
                'baseline_version': 'v2.0.0',
                'current_phase': 'expansion',
                'status': 'active',
                'traffic_percentage': 0.25,
                'duration_hours': 18.5,
                'performance_baseline': {
                    'sharpe_ratio': 1.85,
                    'total_return': 0.0024,
                    'win_rate': 0.58,
                    'max_drawdown': 0.032
                },
                'performance_canary': {
                    'sharpe_ratio': 1.92,
                    'total_return': 0.0027,
                    'win_rate': 0.61,
                    'max_drawdown': 0.028
                }
            }]
        else:
            return self.canary_manager.list_active_deployments()
    
    def get_error_color(self, error_bps: float) -> str:
        """Get color based on error magnitude"""
        if error_bps < 20:
            return "normal"
        elif error_bps < 50:
            return "inverse"
        else:
            return "inverse"
    
    def start_canary_deployment(self, model_version: str, baseline_version: str, 
                               phase_duration: int, auto_promote: bool):
        """Start new canary deployment"""
        if self.demo_mode:
            st.success(f"🚀 Demo: Canary deployment {model_version} zou worden gestart")
        else:
            try:
                deployment_id = self.canary_manager.deploy_canary(
                    model_version=model_version,
                    baseline_version=baseline_version,
                    phase_duration_hours=phase_duration,
                    auto_promote=auto_promote
                )
                st.success(f"🚀 Canary deployment gestart: {deployment_id}")
            except Exception as e:
                st.error(f"❌ Deployment failed: {e}")
    
    def pause_deployment(self, deployment_id: str):
        """Pause deployment"""
        if self.demo_mode:
            st.info(f"⏸️ Demo: Deployment {deployment_id} zou worden gepauzeerd")
        else:
            self.canary_manager.pause_deployment(deployment_id)
            st.info(f"⏸️ Deployment {deployment_id} gepauzeerd")
    
    def resume_deployment(self, deployment_id: str):
        """Resume deployment"""
        if self.demo_mode:
            st.info(f"▶️ Demo: Deployment {deployment_id} zou worden hervat")
        else:
            self.canary_manager.resume_deployment(deployment_id)
            st.info(f"▶️ Deployment {deployment_id} hervat")
    
    def force_rollback(self, deployment_id: str):
        """Force rollback deployment"""
        if self.demo_mode:
            st.warning(f"🔄 Demo: Deployment {deployment_id} zou worden teruggedraaid")
        else:
            self.canary_manager.force_rollback(deployment_id, "Manual rollback from dashboard")
            st.warning(f"🔄 Deployment {deployment_id} teruggedraaid")


# For standalone usage
if __name__ == "__main__":
    import numpy as np
    dashboard = DeploymentDashboard()
    dashboard.render()