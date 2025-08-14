"""
Secrets Management Dashboard voor CryptoSmartTrader V2
Centraal overzicht van secrets configuratie, health status en security events
"""

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Import onze secrets manager
try:
    from config.secrets_manager import get_secrets_manager, SecretsManager
    from config.security import security_manager
except ImportError:
    st.error("⚠️ Kan secrets management modules niet importeren. Check config/ directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="CryptoSmartTrader V2 - Secrets Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f1f2e 0%, #2d2d44 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-healthy { 
        background-color: #d4edda; 
        border: 1px solid #c3e6cb; 
        color: #155724; 
        padding: 0.5rem; 
        border-radius: 0.25rem; 
    }
    .status-degraded { 
        background-color: #fff3cd; 
        border: 1px solid #ffeaa7; 
        color: #856404; 
        padding: 0.5rem; 
        border-radius: 0.25rem; 
    }
    .status-error { 
        background-color: #f8d7da; 
        border: 1px solid #f5c6cb; 
        color: #721c24; 
        padding: 0.5rem; 
        border-radius: 0.25rem; 
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔐 CryptoSmartTrader V2 - Secrets Management</h1>
    <p>Centraal beheer van API keys, environment configuratie en security status</p>
</div>
""", unsafe_allow_html=True)

# Initialiseer secrets manager
@st.cache_resource
def get_cached_secrets_manager():
    """Cache de secrets manager voor performance"""
    try:
        return get_secrets_manager()
    except Exception as e:
        st.error(f"❌ Fout bij laden secrets manager: {e}")
        return None

secrets = get_cached_secrets_manager()

if secrets is None:
    st.error("Kan secrets manager niet laden. Check je configuratie.")
    st.stop()

# Sidebar - Environment Info
with st.sidebar:
    st.header("🌍 Environment Info")
    
    try:
        system_config = secrets.get_system_config()
        
        st.metric("Environment", system_config.get("environment", "unknown"))
        st.metric("Trading Mode", system_config.get("trading_mode", "unknown"))
        st.metric("Debug Mode", "Enabled" if system_config.get("debug", False) else "Disabled")
        st.metric("Log Level", system_config.get("log_level", "unknown"))
        
        # Ports info
        st.subheader("🔌 Service Ports")
        st.text(f"Dashboard: {system_config.get('dashboard_port', 'N/A')}")
        st.text(f"Metrics: {system_config.get('metrics_port', 'N/A')}")  
        st.text(f"Health: {system_config.get('health_port', 'N/A')}")
        
    except Exception as e:
        st.error(f"Fout bij laden system config: {e}")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Secrets Health Status")
    
    try:
        health_status = secrets.get_health_status()
        
        # Overall status
        status = health_status.get("status", "unknown")
        if status == "healthy":
            st.markdown('<div class="status-healthy">✅ Alle secrets zijn gezond en geldig</div>', unsafe_allow_html=True)
        elif status == "degraded":
            st.markdown('<div class="status-degraded">⚠️ Sommige secrets ontbreken of zijn ongeldig</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Kritieke fouten in secrets configuratie</div>', unsafe_allow_html=True)
        
        # Metrics row
        col1a, col1b, col1c, col1d = st.columns(4)
        
        with col1a:
            st.metric(
                "Environment", 
                health_status.get("environment", "N/A")
            )
        
        with col1b:
            st.metric(
                "Total Secrets", 
                health_status.get("secrets_count", 0)
            )
        
        with col1c:
            st.metric(
                "All Present", 
                "✅" if health_status.get("all_present", False) else "❌"
            )
        
        with col1d:
            st.metric(
                "All Valid", 
                "✅" if health_status.get("all_valid", False) else "❌"
            )
        
        # Detailed secrets status
        st.subheader("🔍 Gedetailleerde Secrets Status")
        
        details = health_status.get("details", {})
        if details:
            # Convert naar DataFrame voor mooie weergave
            secrets_data = []
            for secret_name, status_info in details.items():
                secrets_data.append({
                    "Secret Name": secret_name,
                    "Present": "✅" if status_info.get("present", False) else "❌",
                    "Valid Format": "✅" if status_info.get("valid_format", False) else "❌",
                    "Status": "OK" if status_info.get("present", False) and status_info.get("valid_format", False) else "ISSUE"
                })
            
            df_secrets = pd.DataFrame(secrets_data)
            st.dataframe(df_secrets, use_container_width=True)
            
            # Status distribution chart
            status_counts = df_secrets["Status"].value_counts()
            if not status_counts.empty:
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Secrets Status Verdeling",
                    color_discrete_map={"OK": "#28a745", "ISSUE": "#dc3545"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Fout bij laden health status: {e}")
        st.exception(e)

with col2:
    st.header("🔒 Security Events")
    
    try:
        # Load security audit log
        audit_log_path = Path("logs/security_audit.log")
        
        if audit_log_path.exists():
            # Read recent events
            events = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with open(audit_log_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event.get("timestamp", ""))
                        if event_time > cutoff_time:
                            events.append(event)
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            if events:
                # Recent events count
                st.metric("Events (24h)", len(events))
                
                # Severity breakdown
                severity_counts = {}
                for event in events:
                    severity = event.get("severity", "INFO")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Show severity metrics
                for severity, count in severity_counts.items():
                    delta_value = None
                    if severity in ["ERROR", "CRITICAL"]:
                        delta_value = count
                    
                    st.metric(
                        f"{severity} Events",
                        count,
                        delta=delta_value
                    )
                
                # Recent events table
                st.subheader("🕒 Recente Events")
                
                events_data = []
                for event in events[-10:]:  # Laatste 10 events
                    events_data.append({
                        "Time": event.get("timestamp", "")[:19],  # Cut off milliseconds
                        "Type": event.get("event_type", ""),
                        "Severity": event.get("severity", ""),
                        "Details": str(event.get("details", {}))[:50] + "..."
                    })
                
                if events_data:
                    df_events = pd.DataFrame(events_data)
                    st.dataframe(df_events, use_container_width=True, height=300)
                
                # Events timeline
                if len(events) > 1:
                    st.subheader("📈 Events Timeline")
                    
                    # Group events by hour
                    hourly_counts = {}
                    for event in events:
                        hour = event.get("timestamp", "")[:13]  # YYYY-MM-DDTHH
                        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
                    
                    if hourly_counts:
                        times = list(hourly_counts.keys())
                        counts = list(hourly_counts.values())
                        
                        fig_timeline = go.Figure()
                        fig_timeline.add_trace(go.Scatter(
                            x=times,
                            y=counts,
                            mode='lines+markers',
                            name='Security Events',
                            line=dict(color='#dc3545', width=2)
                        ))
                        
                        fig_timeline.update_layout(
                            title="Security Events per Hour",
                            xaxis_title="Time",
                            yaxis_title="Event Count",
                            height=300
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("Geen recente security events (laatste 24h)")
        else:
            st.info("Geen security audit log gevonden")
            
    except Exception as e:
        st.error(f"❌ Fout bij laden security events: {e}")

# Footer met configuratie details
st.header("⚙️ Configuratie Details")

tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "🛡️ Security", "💹 Trading"])

with tab1:
    st.subheader("Exchange API Status")
    
    # Check exchange credentials
    exchanges_status = []
    
    for exchange in ["kraken", "binance"]:
        try:
            creds = secrets.get_exchange_credentials(exchange)
            status = "✅ Configured"
        except ValueError:
            status = "❌ Missing"
        except Exception as e:
            status = f"⚠️ Error: {str(e)[:50]}"
        
        exchanges_status.append({
            "Exchange": exchange.title(),
            "Status": status,
            "Required": "Yes" if exchange == "kraken" else "No"
        })
    
    df_exchanges = pd.DataFrame(exchanges_status)
    st.dataframe(df_exchanges, use_container_width=True)
    
    st.subheader("AI Providers Status")
    
    # Check AI API keys
    ai_status = []
    
    for provider in ["openai", "anthropic", "gemini"]:
        try:
            key = secrets.get_ai_api_key(provider)
            status = "✅ Configured" if key else "❌ Missing"
        except ValueError:
            status = "❌ Missing"
        except Exception as e:
            status = f"⚠️ Error: {str(e)[:50]}"
        
        ai_status.append({
            "Provider": provider.title(),
            "Status": status,
            "Required": "Yes" if provider == "openai" else "No"
        })
    
    df_ai = pd.DataFrame(ai_status)
    st.dataframe(df_ai, use_container_width=True)

with tab2:
    st.subheader("Security Configuration")
    
    try:
        security_config = secrets.get_security_config()
        
        security_data = []
        for key, value in security_config.items():
            security_data.append({
                "Setting": key.replace("_", " ").title(),
                "Status": "✅ Configured" if value else "❌ Missing",
                "Length": len(str(value)) if value else 0
            })
        
        df_security = pd.DataFrame(security_data)
        st.dataframe(df_security, use_container_width=True)
        
        # Security manager summary
        security_summary = security_manager.get_security_summary()
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.metric(
                "Security Status",
                security_summary.get("status", "unknown").title()
            )
        
        with col2b:
            st.metric(
                "Events (1h)",
                security_summary.get("events_last_hour", 0)
            )
        
        with col2c:
            st.metric(
                "Active Lockouts",
                security_summary.get("active_lockouts", 0)
            )
        
    except Exception as e:
        st.error(f"❌ Fout bij laden security config: {e}")

with tab3:
    st.subheader("Trading Configuration")
    
    try:
        system_config = secrets.get_system_config()
        
        # Trading mode validation
        is_valid_mode = secrets.validate_trading_mode()
        
        trading_data = [
            {
                "Setting": "Trading Mode",
                "Value": system_config.get("trading_mode", "unknown"),
                "Status": "✅ Valid" if is_valid_mode else "❌ Invalid"
            },
            {
                "Setting": "Initial Portfolio", 
                "Value": f"${system_config.get('initial_portfolio', 0):,.2f}",
                "Status": "✅ Set"
            },
            {
                "Setting": "Max Daily Loss",
                "Value": f"{system_config.get('max_daily_loss', 0):.1f}%",
                "Status": "✅ Set"
            },
            {
                "Setting": "Max Drawdown",
                "Value": f"{system_config.get('max_drawdown', 0):.1f}%", 
                "Status": "✅ Set"
            }
        ]
        
        df_trading = pd.DataFrame(trading_data)
        st.dataframe(df_trading, use_container_width=True)
        
        # Warning voor live trading
        if system_config.get("trading_mode") == "live":
            st.warning("⚠️ LIVE TRADING MODE GEDETECTEERD! Zorg ervoor dat alle configuratie correct is.")
        
    except Exception as e:
        st.error(f"❌ Fout bij laden trading config: {e}")

# Refresh button
if st.button("🔄 Refresh Dashboard", type="primary"):
    st.cache_resource.clear()
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    CryptoSmartTrader V2 - Enterprise Secrets Management Dashboard<br>
    Voor ondersteuning: check SECRETS_DISCIPLINE_GUIDE.md
</div>
""", unsafe_allow_html=True)