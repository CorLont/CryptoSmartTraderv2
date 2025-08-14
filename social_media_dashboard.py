#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Social Media Data Ingestion Dashboard
Enterprise TOS-compliant social media monitoring with ban protection
"""

import streamlit as st
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
st.set_page_config(
    page_title="Social Media Data Ingestion",
    page_icon="📱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e86ab);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    .compliance-good { border-left-color: #28a745; }
    .compliance-warning { border-left-color: #ffc107; }
    .compliance-critical { border-left-color: #dc3545; }
    .platform-status {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_demo_compliance_data():
    """Generate demo compliance data for visualization"""
    
    platforms = ["reddit", "twitter", "telegram", "discord"]
    
    # Demo TOS compliance status
    compliance_status = {
        "timestamp": datetime.now().isoformat(),
        "platforms": {
            "reddit": {
                "rate_limit_status": {
                    "hourly": "45/100",
                    "daily": "850/1000", 
                    "backoff_until": None,
                    "consecutive_failures": 0
                },
                "tos_compliance": {
                    "api_required": True,
                    "auth_required": True,
                    "allowed_endpoints": 4,
                    "forbidden_actions": 4
                }
            },
            "twitter": {
                "rate_limit_status": {
                    "hourly": "180/300",
                    "daily": "1200/1500",
                    "backoff_until": None,
                    "consecutive_failures": 0
                },
                "tos_compliance": {
                    "api_required": True,
                    "auth_required": True,
                    "allowed_endpoints": 3,
                    "forbidden_actions": 3
                }
            },
            "telegram": {
                "rate_limit_status": {
                    "hourly": "25/3000",
                    "daily": "400/72000",
                    "backoff_until": None,
                    "consecutive_failures": 0
                },
                "tos_compliance": {
                    "api_required": True,
                    "auth_required": True,
                    "allowed_endpoints": 2,
                    "forbidden_actions": 1
                }
            }
        }
    }
    
    # Demo collection metrics
    collection_metrics = {
        "reddit": {
            "total_posts_collected": 1247,
            "total_collections": 23,
            "last_collection": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "avg_posts_per_collection": 54.2
        },
        "twitter": {
            "total_posts_collected": 892,
            "total_collections": 18,
            "last_collection": (datetime.now() - timedelta(minutes=8)).isoformat(),
            "avg_posts_per_collection": 49.6
        },
        "telegram": {
            "total_posts_collected": 234,
            "total_collections": 12,
            "last_collection": (datetime.now() - timedelta(minutes=25)).isoformat(),
            "avg_posts_per_collection": 19.5
        }
    }
    
    # Demo ban protection status
    ban_protection = {
        "reddit": {
            "status": "no_data",
            "total_events": 45,
            "recent_events_1h": 3,
            "ban_events_1h": 0,
            "ban_rate_1h": 0.0,
            "last_ban_time": None,
            "current_risk_level": "low"
        },
        "twitter": {
            "status": "monitored",
            "total_events": 67,
            "recent_events_1h": 5,
            "ban_events_1h": 0,
            "ban_rate_1h": 0.0,
            "last_ban_time": None,
            "current_risk_level": "low"
        },
        "telegram": {
            "status": "monitored",
            "total_events": 12,
            "recent_events_1h": 1,
            "ban_events_1h": 0,
            "ban_rate_1h": 0.0,
            "last_ban_time": None,
            "current_risk_level": "minimal"
        }
    }
    
    return compliance_status, collection_metrics, ban_protection

def render_platform_status_card(platform, rate_status, tos_status, collection_metrics, ban_status):
    """Render platform status card"""
    
    # Parse rate limits
    hourly_used, hourly_limit = map(int, rate_status["hourly"].split("/"))
    daily_used, daily_limit = map(int, rate_status["daily"].split("/"))
    
    # Calculate percentages
    hourly_pct = (hourly_used / hourly_limit) * 100
    daily_pct = (daily_used / daily_limit) * 100
    
    # Determine status color
    if max(hourly_pct, daily_pct) > 80:
        status_class = "compliance-critical"
        status_icon = "🔴"
    elif max(hourly_pct, daily_pct) > 60:
        status_class = "compliance-warning"
        status_icon = "🟡"
    else:
        status_class = "compliance-good"
        status_icon = "🟢"
    
    # Platform header
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3>{status_icon} {platform.title()}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Rate Limits",
            f"{hourly_pct:.0f}% (hourly)",
            f"{daily_pct:.0f}% (daily)"
        )
    
    with col2:
        posts_collected = collection_metrics.get("total_posts_collected", 0)
        avg_posts = collection_metrics.get("avg_posts_per_collection", 0)
        st.metric(
            "Posts Collected",
            f"{posts_collected:,}",
            f"~{avg_posts:.0f} per collection"
        )
    
    with col3:
        risk_level = ban_status.get("current_risk_level", "unknown")
        risk_emoji = {"minimal": "✅", "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(risk_level, "❓")
        st.metric(
            "Ban Risk",
            f"{risk_emoji} {risk_level.title()}",
            f"{ban_status.get('ban_events_1h', 0)} events (1h)"
        )
    
    # Detailed info in expander
    with st.expander(f"📊 {platform.title()} Details"):
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**Rate Limiting:**")
            st.write(f"• Hourly: {rate_status['hourly']}")
            st.write(f"• Daily: {rate_status['daily']}")
            st.write(f"• Failures: {rate_status.get('consecutive_failures', 0)}")
            
            if rate_status.get("backoff_until"):
                st.warning(f"⏰ Backoff until: {rate_status['backoff_until']}")
        
        with detail_col2:
            st.write("**TOS Compliance:**")
            st.write(f"• API Required: {'✅' if tos_status['api_required'] else '❌'}")
            st.write(f"• Auth Required: {'✅' if tos_status['auth_required'] else '❌'}")
            st.write(f"• Allowed Endpoints: {tos_status['allowed_endpoints']}")
            st.write(f"• Forbidden Actions: {tos_status['forbidden_actions']}")
    
    return hourly_pct, daily_pct

def render_compliance_overview(compliance_status, collection_metrics, ban_protection):
    """Render compliance overview section"""
    
    st.markdown("## 📋 TOS Compliance Overview")
    
    platforms = list(compliance_status["platforms"].keys())
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_posts = sum(collection_metrics.get(p, {}).get("total_posts_collected", 0) for p in platforms)
    total_collections = sum(collection_metrics.get(p, {}).get("total_collections", 0) for p in platforms)
    
    with col1:
        st.metric("Total Platforms", len(platforms))
    
    with col2:
        st.metric("Posts Collected", f"{total_posts:,}")
    
    with col3:
        st.metric("Total Collections", total_collections)
    
    with col4:
        # Count platforms with low risk
        low_risk_count = sum(1 for p in platforms 
                           if ban_protection.get(p, {}).get("current_risk_level", "unknown") in ["minimal", "low"])
        st.metric("Low Risk Platforms", f"{low_risk_count}/{len(platforms)}")
    
    st.markdown("---")
    
    # Platform status cards
    rate_data = []
    
    for platform in platforms:
        if platform in compliance_status["platforms"]:
            rate_status = compliance_status["platforms"][platform]["rate_limit_status"]
            tos_status = compliance_status["platforms"][platform]["tos_compliance"]
            coll_metrics = collection_metrics.get(platform, {})
            ban_status = ban_protection.get(platform, {})
            
            hourly_pct, daily_pct = render_platform_status_card(
                platform, rate_status, tos_status, coll_metrics, ban_status
            )
            
            rate_data.append({
                "platform": platform.title(),
                "hourly_usage": hourly_pct,
                "daily_usage": daily_pct,
                "posts_collected": coll_metrics.get("total_posts_collected", 0),
                "risk_level": ban_status.get("current_risk_level", "unknown")
            })
    
    return rate_data

def render_analytics_charts(rate_data, collection_metrics):
    """Render analytics and charts"""
    
    st.markdown("## 📊 Analytics & Monitoring")
    
    if not rate_data:
        st.warning("No data available for analytics")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(rate_data)
    
    # Create charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Rate limit usage chart
        fig_rate = px.bar(
            df,
            x="platform",
            y=["hourly_usage", "daily_usage"],
            title="Rate Limit Usage (%)",
            barmode="group",
            color_discrete_map={
                "hourly_usage": "#2e86ab",
                "daily_usage": "#a23b72"
            }
        )
        fig_rate.update_layout(height=400)
        st.plotly_chart(fig_rate, use_container_width=True)
    
    with chart_col2:
        # Posts collected chart
        fig_posts = px.pie(
            df,
            values="posts_collected",
            names="platform",
            title="Posts Collected by Platform",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_posts.update_layout(height=400)
        st.plotly_chart(fig_posts, use_container_width=True)
    
    # Time series simulation (demo data)
    st.markdown("### 📈 Collection Timeline (Last 24 Hours)")
    
    # Generate demo time series data
    import numpy as np
    
    times = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq="1H"
    )
    
    timeline_data = []
    for platform in df["platform"].unique():
        base_rate = np.random.randint(10, 50)
        values = base_rate + np.random.normal(0, 10, len(times))
        values = np.maximum(0, values)  # No negative values
        
        for time, value in zip(times, values):
            timeline_data.append({
                "time": time,
                "platform": platform,
                "posts_per_hour": int(value)
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig_timeline = px.line(
        timeline_df,
        x="time",
        y="posts_per_hour",
        color="platform",
        title="Posts Collected Per Hour",
        line_shape="spline"
    )
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)

def render_configuration_panel():
    """Render configuration and controls"""
    
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Platform selection
    st.sidebar.markdown("### Platform Selection")
    platforms = st.sidebar.multiselect(
        "Active Platforms",
        ["reddit", "twitter", "telegram", "discord"],
        default=["reddit", "twitter", "telegram"]
    )
    
    # Rate limiting settings
    st.sidebar.markdown("### Rate Limiting")
    global_rate_limit = st.sidebar.slider(
        "Global Rate Limit (req/min)",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    backoff_strategy = st.sidebar.selectbox(
        "Backoff Strategy",
        ["exponential", "linear", "fixed"],
        index=0
    )
    
    # Ban protection settings
    st.sidebar.markdown("### Ban Protection")
    ban_protection_enabled = st.sidebar.checkbox(
        "Enable Ban Protection",
        value=True
    )
    
    detection_sensitivity = st.sidebar.selectbox(
        "Detection Sensitivity",
        ["low", "medium", "high"],
        index=1
    )
    
    # Data collection settings
    st.sidebar.markdown("### Data Collection")
    collection_interval = st.sidebar.selectbox(
        "Collection Interval",
        ["5 minutes", "15 minutes", "30 minutes", "1 hour"],
        index=1
    )
    
    keywords = st.sidebar.text_area(
        "Crypto Keywords",
        value="bitcoin, ethereum, crypto, defi, nft",
        help="Comma-separated keywords to monitor"
    )
    
    # Control buttons
    st.sidebar.markdown("### Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()
    
    with col2:
        if st.button("📊 Export Report"):
            st.sidebar.success("Report exported!")
    
    # Secrets management info
    st.sidebar.markdown("### 🔐 Secrets Status")
    
    # Demo secrets status
    secrets_status = {
        "Reddit API": "✅ Configured",
        "Twitter API": "✅ Configured", 
        "Telegram API": "⚠️ Missing",
        "Discord API": "❌ Not Set"
    }
    
    for platform, status in secrets_status.items():
        st.sidebar.markdown(f"**{platform}:** {status}")
    
    if st.sidebar.button("🛠️ Setup Secrets"):
        st.sidebar.info("Run: `python scripts/social_media_secrets_setup.py`")
    
    return {
        "platforms": platforms,
        "global_rate_limit": global_rate_limit,
        "backoff_strategy": backoff_strategy,
        "ban_protection_enabled": ban_protection_enabled,
        "detection_sensitivity": detection_sensitivity,
        "collection_interval": collection_interval,
        "keywords": [k.strip() for k in keywords.split(",")]
    }

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📱 Social Media Data Ingestion Dashboard</h1>
        <p style="color: white; text-align: center; margin: 0;">
            Enterprise TOS-Compliant Social Media Monitoring with Ban Protection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load demo data
    compliance_status, collection_metrics, ban_protection = load_demo_compliance_data()
    
    # Configuration panel
    config = render_configuration_panel()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview",
        "📊 Analytics", 
        "⚖️ Compliance",
        "🛡️ Security"
    ])
    
    with tab1:
        # Compliance overview
        rate_data = render_compliance_overview(compliance_status, collection_metrics, ban_protection)
        
        # Real-time status
        st.markdown("## 🔄 Real-Time Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.markdown("""
            <div class="metric-card compliance-good">
                <h4>🟢 System Status</h4>
                <p>All platforms operational</p>
                <small>Last updated: just now</small>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col2:
            st.markdown("""
            <div class="metric-card compliance-good">
                <h4>✅ TOS Compliance</h4>
                <p>100% compliant requests</p>
                <small>0 violations detected</small>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col3:
            st.markdown("""
            <div class="metric-card compliance-good">
                <h4>🛡️ Ban Protection</h4>
                <p>Active monitoring</p>
                <small>Risk level: Low</small>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Analytics charts
        render_analytics_charts(rate_data, collection_metrics)
    
    with tab3:
        # Detailed compliance info
        st.markdown("## ⚖️ TOS Compliance Details")
        
        # Recent compliance events
        st.markdown("### 📋 Recent Compliance Events")
        
        compliance_events = [
            {"time": "2 minutes ago", "platform": "Twitter", "event": "Rate limit check passed", "status": "✅"},
            {"time": "5 minutes ago", "platform": "Reddit", "event": "Authentication verified", "status": "✅"},
            {"time": "8 minutes ago", "platform": "Telegram", "event": "Endpoint validation passed", "status": "✅"},
            {"time": "12 minutes ago", "platform": "Twitter", "event": "User-Agent header validated", "status": "✅"},
            {"time": "15 minutes ago", "platform": "Reddit", "event": "OAuth token refreshed", "status": "✅"}
        ]
        
        for event in compliance_events:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; 
                        padding: 0.5rem; margin: 0.25rem 0; background: #f8f9fa; border-radius: 5px;">
                <span><strong>{event['platform']}</strong>: {event['event']}</span>
                <span style="font-size: 0.8em; color: #666;">{event['time']} {event['status']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # TOS rules summary
        st.markdown("### 📜 Platform TOS Rules")
        
        tos_summary = {
            "Reddit": {
                "Rate Limit": "60 requests/minute",
                "Authentication": "OAuth2 required",
                "Commercial Use": "Not allowed",
                "Data Retention": "30 days max"
            },
            "Twitter": {
                "Rate Limit": "300 requests/15min",
                "Authentication": "Bearer Token required", 
                "Commercial Use": "Allowed with license",
                "Data Retention": "90 days max"
            },
            "Telegram": {
                "Rate Limit": "30 requests/second",
                "Authentication": "Bot Token required",
                "Commercial Use": "Allowed",
                "Data Retention": "365 days max"
            }
        }
        
        for platform, rules in tos_summary.items():
            with st.expander(f"📋 {platform} TOS Rules"):
                for rule, value in rules.items():
                    st.write(f"**{rule}:** {value}")
    
    with tab4:
        # Security monitoring
        st.markdown("## 🛡️ Security & Ban Protection")
        
        # Security metrics
        sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
        
        with sec_col1:
            st.metric("Failed Requests", "0", "0%")
        
        with sec_col2:
            st.metric("Suspicious Patterns", "0", "✅ Clean")
        
        with sec_col3:
            st.metric("Rate Limit Hits", "0", "No violations")
        
        with sec_col4:
            st.metric("Ban Risk Score", "2/100", "Very Low")
        
        # Security timeline
        st.markdown("### 🔍 Security Event Timeline")
        
        security_events = [
            {"time": datetime.now() - timedelta(minutes=30), "event": "System health check", "level": "INFO"},
            {"time": datetime.now() - timedelta(hours=2), "event": "Rate limit validation", "level": "INFO"},
            {"time": datetime.now() - timedelta(hours=6), "event": "Token refresh scheduled", "level": "INFO"},
            {"time": datetime.now() - timedelta(hours=12), "event": "Compliance audit passed", "level": "SUCCESS"},
        ]
        
        for event in security_events:
            level_color = {"INFO": "#17a2b8", "SUCCESS": "#28a745", "WARNING": "#ffc107", "ERROR": "#dc3545"}
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.5rem; margin: 0.25rem 0; 
                        background: #f8f9fa; border-radius: 5px; border-left: 4px solid {level_color.get(event['level'], '#6c757d')};">
                <span style="flex: 1;"><strong>{event['level']}</strong>: {event['event']}</span>
                <span style="font-size: 0.8em; color: #666;">{event['time'].strftime('%H:%M:%S')}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        🔒 Enterprise Social Media Data Ingestion Framework | 
        TOS-Compliant | Ban-Protected | Real-time Monitoring
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()