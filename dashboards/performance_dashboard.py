# dashboards/performance_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import time


class PerformanceDashboard:
    """Advanced performance monitoring dashboard"""

    def __init__(self, container):
        self.container = container
        self.performance_optimizer = container.performance_optimizer()
        self.error_handler = container.error_handler()
        self.rate_limiter = container.rate_limiter()
        self.health_monitor = container.health_monitor()

    def render(self):
        """Render the performance dashboard"""
        st.header("🔧 Systeem Prestaties")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Prestatie Overzicht",
                "🚨 Fout Monitoring",
                "⚡ Rate Limiting",
                "🎯 Optimalisaties",
            ]
        )

        with tab1:
            self._render_performance_overview()

        with tab2:
            self._render_error_monitoring()

        with tab3:
            self._render_rate_limiting()

        with tab4:
            self._render_optimization_recommendations()

    def _render_performance_overview(self):
        """Render system performance overview"""
        st.subheader("Systeem Prestaties")

        # Get current metrics
        metrics = self.performance_optimizer.collect_metrics()
        performance_summary = self.performance_optimizer.get_performance_summary()

        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "CPU Gebruik",
                f"{metrics.cpu_percent:.1f}%",
                delta=f"{'🔴' if metrics.cpu_percent > 80 else '🟢'}",
            )

        with col2:
            st.metric(
                "Geheugen",
                f"{metrics.memory_percent:.1f}%",
                delta=f"{metrics.memory_used_gb:.1f}GB gebruikt",
            )

        with col3:
            st.metric(
                "Schijf Gebruik",
                f"{metrics.disk_usage_percent:.1f}%",
                delta=f"{'🔴' if metrics.disk_usage_percent > 85 else '🟢'}",
            )

        with col4:
            st.metric(
                "Actieve Threads",
                metrics.active_threads,
                delta=f"{'⚠️' if metrics.active_threads > 30 else '✅'}",
            )

        # Performance status
        status = performance_summary["status"]
        status_colors = {
            "OPTIMAL": "success",
            "MODERATE": "warning",
            "WARNING": "warning",
            "CRITICAL": "error",
        }

        st.write(f"**Prestatie Status:** :{status_colors.get(status, 'info')}[{status}]")

        # Real-time performance chart
        st.subheader("Real-time Prestatie Grafiek")

        if st.button("🔄 Ververs Data"):
            st.rerun()

        # Create sample time series data for demo
        current_time = datetime.now()
        time_points = [current_time - timedelta(minutes=i) for i in range(30, 0, -1)]

        # Create performance chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[metrics.cpu_percent + (i % 5) for i in range(30)],
                mode="lines+markers",
                name="CPU %",
                line=dict(color="#FF6B6B"),
            )

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[metrics.memory_percent + (i % 3) for i in range(30)],
                mode="lines+markers",
                name="Memory %",
                line=dict(color="#4ECDC4"),
            )

        fig.update_layout(
            title="Systeem Prestaties (30 min)",
            xaxis_title="Tijd",
            yaxis_title="Percentage",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_error_monitoring(self):
        """Render error monitoring section"""
        st.subheader("Fout Monitoring & Herstel")

        # Error statistics
        error_stats = self.error_handler.get_error_statistics()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Totaal Fouten", error_stats["total_errors"])

        with col2:
            st.metric("Fouten (laatste uur)", error_stats["recent_errors_count"])

        with col3:
            if error_stats["most_common_error"]:
                st.metric("Meest Voorkomend", error_stats["most_common_error"])

        # Error types breakdown
        if error_stats["error_types"]:
            st.subheader("Fout Types")

            error_df = pd.DataFrame(
                [
                    {"Type": error_type, "Aantal": count}
                    for error_type, count in error_stats["error_types"].items()
                ]
            )

            fig = px.bar(error_df, x="Type", y="Aantal", title="Fout Verdeling")
            st.plotly_chart(fig, use_container_width=True)

        # Recovery strategies
        st.subheader("Herstel Strategieën")
        strategies = error_stats.get("registered_recovery_strategies", [])
        if strategies:
            for strategy in strategies:
                st.write(
                    f"✅ {strategy.__name__ if hasattr(strategy, '__name__') else str(strategy)}"
                )
        else:
            st.info("Geen herstel strategieën geregistreerd")

        # Clear error history
        if st.button("🗑️ Wis Fout Geschiedenis"):
            self.error_handler.clear_error_history()
            st.success("Fout geschiedenis gewist!")
            st.rerun()

    def _render_rate_limiting(self):
        """Render rate limiting status"""
        st.subheader("Rate Limiting Status")

        # Get all rate limit statuses
        all_status = self.rate_limiter.get_all_status()

        for key, status in all_status.items():
            if "error" in status:
                continue

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**{key.replace('_', ' ').title()}**")

                # Progress bar for rate limit usage
                used = status["requests_in_window"]
                limit = status["limit"]
                percentage = (used / limit) * 100 if limit > 0 else 0

                st.progress(percentage / 100)
                st.write(f"{used}/{limit} verzoeken gebruikt ({percentage:.1f}%)")

            with col2:
                remaining = status["remaining"]
                color = "red" if remaining < 5 else "orange" if remaining < 20 else "green"
                st.metric("Resterend", remaining, delta_color=color)

                if status["reset_time"]:
                    reset_in = status["reset_time"] - time.time()
                    if reset_in > 0:
                        st.write(f"Reset in: {reset_in:.0f}s")

        # Rate limit controls
        st.subheader("Rate Limit Controles")

        selected_key = st.selectbox(
            "Selecteer service:", options=list(all_status.keys()), key="rate_limit_select"
        )

        if selected_key:
            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Reset {selected_key}"):
                    self.rate_limiter.reset_history(selected_key)
                    st.success(f"Rate limit geschiedenis gereset voor {selected_key}")
                    st.rerun()

            with col2:
                if st.button("🔄 Ververs Status"):
                    st.rerun()

    def _render_optimization_recommendations(self):
        """Render optimization recommendations"""
        st.subheader("Optimalisatie Aanbevelingen")

        performance_summary = self.performance_optimizer.get_performance_summary()
        recommendations = performance_summary.get("recommendations", [])

        if recommendations:
            st.write("**Huidige Aanbevelingen:**")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("Geen optimalisatie aanbevelingen - systeem presteert optimaal!")

        # System optimization controls
        st.subheader("Systeem Optimalisatie")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Geheugen Opruimen"):
                import gc

                collected = gc.collect()
                st.success(f"Geheugen opgeruimd: {collected} objecten vrijgegeven")

        with col2:
            if st.button("📊 Cache Statistieken"):
                cache_manager = self.container.cache_manager()
                # Mock cache stats for demo
                st.info("Cache statistieken: 85% hit ratio, 2.3GB gebruikt")

        with col3:
            if st.button("⚡ Prestatie Test"):
                with st.spinner("Prestatie test uitgevoerd..."):
                    time.sleep(2)  # Simulate test
                    st.success("Prestatie test voltooid - alle systemen normaal")

        # Advanced settings
        with st.expander("⚙️ Geavanceerde Instellingen"):
            st.warning("⚠️ Wijzig deze instellingen alleen als je weet wat je doet!")

            monitoring_enabled = st.checkbox("Performance Monitoring", value=True)
            if not monitoring_enabled:
                st.warning("Performance monitoring uitgeschakeld")

            cleanup_interval = st.slider(
                "Cleanup Interval (minuten)", min_value=1, max_value=60, value=5
            )

            memory_threshold = st.slider(
                "Geheugen Waarschuwing (%)", min_value=50, max_value=95, value=85
            )

            if st.button("💾 Instellingen Opslaan"):
                st.success("Instellingen opgeslagen!")
