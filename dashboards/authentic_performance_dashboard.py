#!/usr/bin/env python3
"""
Authentic Performance Dashboard - No Fake Visualizations
Real system metrics only, eliminates misleading synthetic performance data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unified_structured_logger import get_unified_logger


class AuthenticPerformanceDashboard:
    """Performance dashboard with only authentic metrics"""

    def __init__(self, container=None):
        self.container = container
        self.logger = get_unified_logger("AuthenticPerformance")

    def render(self):
        """Render authentic performance dashboard"""

        st.header("📊 Authentic System Performance")
        st.markdown("**Real metrics only - no synthetic or placeholder data**")

        # Performance overview
        self._render_performance_overview()

        # Real system metrics
        self._render_real_system_metrics()

        # Actual log analysis
        self._render_log_analysis()

        # Live system health
        self._render_live_health_metrics()

    def _render_performance_overview(self):
        """Render real performance overview"""

        st.subheader("🎯 Real Performance Overview")

        col1, col2, col3, col4 = st.columns(4)

        # Real file system metrics
        with col1:
            log_files = list(Path("logs").glob("**/*.log")) if Path("logs").exists() else []
            st.metric("Log Files", len(log_files))

        with col2:
            data_files = list(Path("data").glob("**/*.json")) if Path("data").exists() else []
            st.metric("Data Files", len(data_files))

        with col3:
            cache_files = list(Path("cache").glob("**/*")) if Path("cache").exists() else []
            st.metric("Cache Files", len(cache_files))

        with col4:
            # Real prediction files
            pred_files = (
                list(Path("data").glob("**/predictions_*.json")) if Path("data").exists() else []
            )
            st.metric("Prediction Files", len(pred_files))

    def _render_real_system_metrics(self):
        """Render real system resource metrics"""

        st.subheader("🖥️ Real System Resources")

        try:
            import psutil

            col1, col2, col3 = st.columns(3)

            with col1:
                # Real CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")

                # CPU chart with real data
                if cpu_percent > 0:
                    fig_cpu = go.Figure()
                    fig_cpu.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=cpu_percent,
                            domain={"x": [0, 1], "y": [0, 1]},
                            title={"text": "CPU %"},
                            gauge={
                                "axis": {"range": [None, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 50], "color": "lightgray"},
                                    {"range": [50, 80], "color": "yellow"},
                                    {"range": [80, 100], "color": "red"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 90,
                                },
                            },
                        )
                    )
                    fig_cpu.update_layout(height=200)
                    st.plotly_chart(fig_cpu, use_container_width=True)

            with col2:
                # Real memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_total_gb = memory.total / (1024**3)

                st.metric(
                    "Memory Usage",
                    f"{memory_percent:.1f}%",
                    f"{memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB",
                )

                # Memory chart with real data
                fig_mem = go.Figure()
                fig_mem.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=memory_percent,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Memory %"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkgreen"},
                            "steps": [
                                {"range": [0, 60], "color": "lightgray"},
                                {"range": [60, 85], "color": "yellow"},
                                {"range": [85, 100], "color": "red"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90,
                            },
                        },
                    )
                )
                fig_mem.update_layout(height=200)
                st.plotly_chart(fig_mem, use_container_width=True)

            with col3:
                # Real disk usage
                disk = psutil.disk_usage(".")
                disk_percent = (disk.used / disk.total) * 100
                disk_free_gb = disk.free / (1024**3)
                disk_total_gb = disk.total / (1024**3)

                st.metric(
                    "Disk Usage",
                    f"{disk_percent:.1f}%",
                    f"{disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total",
                )

                # Disk chart with real data
                fig_disk = go.Figure()
                fig_disk.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=disk_percent,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Disk %"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkorange"},
                            "steps": [
                                {"range": [0, 70], "color": "lightgray"},
                                {"range": [70, 90], "color": "yellow"},
                                {"range": [90, 100], "color": "red"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 95,
                            },
                        },
                    )
                )
                fig_disk.update_layout(height=200)
                st.plotly_chart(fig_disk, use_container_width=True)

        except ImportError:
            st.warning("psutil not available - cannot show real system metrics")
        except Exception as e:
            st.error(f"Error getting real system metrics: {e}")

    def _render_log_analysis(self):
        """Render analysis of actual log files"""

        st.subheader("📋 Real Log Analysis")

        try:
            # Analyze actual log files
            log_dir = Path("logs")
            if not log_dir.exists():
                st.info("No logs directory found")
                return

            # Get recent log files
            recent_logs = []
            for log_file in log_dir.glob("**/*.log"):
                try:
                    stat = log_file.stat()
                    recent_logs.append(
                        {
                            "file": str(log_file),
                            "size_kb": stat.st_size / 1024,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                        }
                    )
                except Exception:
                    continue

            recent_logs.sort(key=lambda x: x["modified"], reverse=True)

            if recent_logs:
                # Show real log file statistics
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Recent Log Files:**")
                    log_df = pd.DataFrame(recent_logs[:10])
                    log_df["size_kb"] = log_df["size_kb"].round(2)
                    st.dataframe(log_df, use_container_width=True)

                with col2:
                    # Real log size analysis
                    total_size = sum(log["size_kb"] for log in recent_logs)
                    st.metric("Total Log Size", f"{total_size:.1f} KB")
                    st.metric("Log Files Count", len(recent_logs))

                    # Log file size distribution (real data)
                    if len(recent_logs) > 1:
                        sizes = [log["size_kb"] for log in recent_logs[:20]]
                        fig_logs = px.bar(
                            x=list(range(len(sizes))),
                            y=sizes,
                            title="Real Log File Sizes (KB)",
                            labels={"x": "File Index", "y": "Size (KB)"},
                        )
                        st.plotly_chart(fig_logs, use_container_width=True)
            else:
                st.info("No log files found")

        except Exception as e:
            st.error(f"Error analyzing logs: {e}")

    def _render_live_health_metrics(self):
        """Render live health metrics from actual system"""

        st.subheader("❤️ Live Health Metrics")

        try:
            # Check actual health status files
            health_files = []

            # Look for real health status files
            if Path("health_status.json").exists():
                try:
                    with open("health_status.json", "r") as f:
                        health_data = json.load(f)
                    health_files.append(("health_status.json", health_data))
                except Exception as e:
                    self.logger.warning(f"Error reading health_status.json: {e}")

            # Look for daily health logs
            daily_dir = Path("logs/daily")
            if daily_dir.exists():
                for health_file in daily_dir.glob("**/system_health*.json"):
                    try:
                        with open(health_file, "r") as f:
                            health_data = json.load(f)
                        health_files.append((str(health_file), health_data))
                    except Exception:
                        continue

            if health_files:
                st.write("**Real Health Status Files:**")

                for file_name, health_data in health_files[:5]:  # Show latest 5
                    with st.expander(f"📄 {file_name}"):
                        # Show actual health data structure
                        if isinstance(health_data, dict):
                            col1, col2 = st.columns(2)

                            with col1:
                                for key, value in health_data.items():
                                    if key in [
                                        "timestamp",
                                        "status",
                                        "score",
                                        "errors",
                                        "warnings",
                                    ]:
                                        st.write(f"**{key}:** {value}")

                            with col2:
                                # Show full data as JSON
                                st.json(health_data)
                        else:
                            st.json(health_data)
            else:
                st.info("No health status files found - system may be starting up")

                # Show basic system availability
                col1, col2, col3 = st.columns(3)

                with col1:
                    app_running = Path("app_fixed_all_issues.py").exists()
                    st.metric("Main App", "✅ Available" if app_running else "❌ Missing")

                with col2:
                    config_available = Path("config.json").exists()
                    st.metric("Configuration", "✅ Available" if config_available else "❌ Missing")

                with col3:
                    logs_available = Path("logs").exists()
                    st.metric("Logging", "✅ Available" if logs_available else "❌ Missing")

        except Exception as e:
            st.error(f"Error checking live health metrics: {e}")

    def get_real_performance_summary(self) -> Dict[str, Any]:
        """Get real performance summary without fake metrics"""

        summary = {"timestamp": datetime.now().isoformat(), "data_source": "authentic_only"}

        try:
            # Real file counts
            if Path("logs").exists():
                summary["log_files_count"] = len(list(Path("logs").glob("**/*.log")))
            if Path("data").exists():
                summary["data_files_count"] = len(list(Path("data").glob("**/*.json")))

            # Real system resources
            import psutil

            summary["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            summary["memory_percent"] = psutil.virtual_memory().percent
            summary["disk_percent"] = (
                psutil.disk_usage(".").used / psutil.disk_usage(".").total
            ) * 100

        except Exception as e:
            summary["error"] = str(e)

        return summary
