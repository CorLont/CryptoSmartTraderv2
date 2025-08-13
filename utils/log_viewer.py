#!/usr/bin/env python3
"""
Log Viewer Utility voor CryptoSmartTrader V2
Biedt functionaliteiten voor het bekijken en analyseren van dagelijkse logs
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st


class LogViewer:
    """Utility class voor het bekijken van logs"""

    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)

    def get_available_dates(self) -> List[str]:
        """Krijg lijst van beschikbare log datums"""
        if not self.base_log_dir.exists():
            return []

        dates = []
        for log_dir in self.base_log_dir.iterdir():
            if log_dir.is_dir() and log_dir.name.count("-") == 2:
                try:
                    # Valideer datum format
                    datetime.strptime(log_dir.name, "%Y-%m-%d")
                    dates.append(log_dir.name)
                except ValueError:
                    continue

        return sorted(dates, reverse=True)

    def get_log_files_for_date(self, date: str) -> Dict[str, Path]:
        """Krijg log files voor specifieke datum"""
        log_dir = self.base_log_dir / date
        if not log_dir.exists():
            return {}

        log_files = {}
        for log_file in log_dir.iterdir():
            if log_file.suffix in [".json", ".log"]:
                component = log_file.stem
                log_files[component] = log_file

        return log_files

    def read_json_logs(self, log_file: Path) -> List[Dict]:
        """Lees JSON log file"""
        if not log_file.exists():
            return []

        logs = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.error(f"Error reading {log_file}: {e}")

        return logs

    def get_prediction_summary(self, date: str) -> Dict[str, Any]:
        """Krijg samenvatting van predictions voor datum"""
        log_files = self.get_log_files_for_date(date)
        predictions_file = log_files.get("predictions")

        if not predictions_file:
            return {"total": 0, "coins": [], "avg_confidence": 0}

        logs = self.read_json_logs(predictions_file)

        if not logs:
            return {"total": 0, "coins": [], "avg_confidence": 0}

        coins = []
        confidences = []

        for log in logs:
            if "data" in log and "coin" in log["data"]:
                coins.append(log["data"]["coin"])
                # Extract confidence from message or data
                if "confidence" in log["data"]:
                    confidences.append(log["data"]["confidence"])

        return {
            "total": len(logs),
            "coins": list(set(coins)),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "logs": logs,
        }

    def get_performance_summary(self, date: str) -> Dict[str, Any]:
        """Krijg performance samenvatting voor datum"""
        log_files = self.get_log_files_for_date(date)
        performance_file = log_files.get("performance")

        if not performance_file:
            return {"operations": {}, "total_time": 0}

        logs = self.read_json_logs(performance_file)

        operations = {}
        total_time = 0

        for log in logs:
            if "data" in log and "operation" in log["data"]:
                operation = log["data"]["operation"]
                exec_time = log["data"].get("execution_time", 0)

                if operation not in operations:
                    operations[operation] = {"count": 0, "total_time": 0, "times": []}

                operations[operation]["count"] += 1
                operations[operation]["total_time"] += exec_time
                operations[operation]["times"].append(exec_time)
                total_time += exec_time

        # Calculate averages
        for op_data in operations.values():
            op_data["avg_time"] = op_data["total_time"] / op_data["count"]
            op_data["min_time"] = min(op_data["times"])
            op_data["max_time"] = max(op_data["times"])

        return {"operations": operations, "total_time": total_time, "logs": logs}

    def get_error_summary(self, date: str) -> Dict[str, Any]:
        """Krijg error samenvatting voor datum"""
        log_files = self.get_log_files_for_date(date)
        errors_file = log_files.get("errors")

        if not errors_file:
            return {"total": 0, "by_component": {}, "by_type": {}}

        logs = self.read_json_logs(errors_file)

        by_component = {}
        by_type = {}

        for log in logs:
            if "data" in log:
                component = log["data"].get("component", "unknown")
                error_type = log["data"].get("error_type", "unknown")

                by_component[component] = by_component.get(component, 0) + 1
                by_type[error_type] = by_type.get(error_type, 0) + 1

        return {"total": len(logs), "by_component": by_component, "by_type": by_type, "logs": logs}

    def display_log_dashboard(self):
        """Streamlit dashboard voor logs"""
        st.header("📊 Log Analysis Dashboard")

        available_dates = self.get_available_dates()

        if not available_dates:
            st.warning("Geen log files gevonden")
            return

        # Date selector
        selected_date = st.selectbox("Selecteer datum:", available_dates, index=0)

        if not selected_date:
            return

        # Toon log samenvatting
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Predictions")
            pred_summary = self.get_prediction_summary(selected_date)
            st.metric("Total Predictions", pred_summary["total"])
            st.metric("Unique Coins", len(pred_summary["coins"]))
            if pred_summary["avg_confidence"] > 0:
                st.metric("Avg Confidence", f"{pred_summary['avg_confidence']:.3f}")

        with col2:
            st.subheader("⚡ Performance")
            perf_summary = self.get_performance_summary(selected_date)
            st.metric("Operations", len(perf_summary["operations"]))
            st.metric("Total Time", f"{perf_summary['total_time']:.1f}s")

            if perf_summary["operations"]:
                slowest_op = max(perf_summary["operations"].items(), key=lambda x: x[1]["avg_time"])
                st.metric(
                    "Slowest Operation", f"{slowest_op[0]} ({slowest_op[1]['avg_time']:.2f}s)"
                )

        with col3:
            st.subheader("❌ Errors")
            error_summary = self.get_error_summary(selected_date)
            st.metric("Total Errors", error_summary["total"])

            if error_summary["by_component"]:
                top_error_component = max(error_summary["by_component"].items(), key=lambda x: x[1])
                st.metric("Most Errors", f"{top_error_component[0]} ({top_error_component[1]})")

        # Detailed views
        st.subheader("🔍 Detailed Views")

        view_type = st.selectbox(
            "Select view:", ["Predictions", "Performance", "Errors", "API Calls", "User Actions"]
        )

        if view_type == "Predictions" and pred_summary["logs"]:
            self._display_predictions_table(pred_summary["logs"])
        elif view_type == "Performance" and perf_summary["logs"]:
            self._display_performance_table(perf_summary["logs"])
        elif view_type == "Errors" and error_summary["logs"]:
            self._display_errors_table(error_summary["logs"])

    def _display_predictions_table(self, logs: List[Dict]):
        """Display predictions in table format"""
        data = []
        for log in logs:
            if "data" in log:
                data.append(
                    {
                        "Time": log["timestamp"],
                        "Coin": log["data"].get("coin", "N/A"),
                        "Confidence": log["data"].get("confidence", 0),
                        "Expected Returns": str(
                            log["data"].get("prediction_data", {}).get("expected_returns", {})
                        ),
                        "Sentiment": log["data"]
                        .get("prediction_data", {})
                        .get("sentiment_score", 0),
                        "Whale Detected": log["data"]
                        .get("prediction_data", {})
                        .get("whale_detected", False),
                    }
                )

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

    def _display_performance_table(self, logs: List[Dict]):
        """Display performance in table format"""
        data = []
        for log in logs:
            if "data" in log:
                data.append(
                    {
                        "Time": log["timestamp"],
                        "Operation": log["data"].get("operation", "N/A"),
                        "Execution Time": f"{log['data'].get('execution_time', 0):.3f}s",
                        "Additional Info": str(log["data"].get("additional_metrics", {})),
                    }
                )

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

    def _display_errors_table(self, logs: List[Dict]):
        """Display errors in table format"""
        data = []
        for log in logs:
            if "data" in log:
                data.append(
                    {
                        "Time": log["timestamp"],
                        "Component": log["data"].get("component", "N/A"),
                        "Error Type": log["data"].get("error_type", "N/A"),
                        "Error Message": log["data"].get("error_message", "N/A"),
                        "Context": str(log["data"].get("context", {})),
                    }
                )

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)


def create_log_viewer_page():
    """Create log viewer page for Streamlit"""
    log_viewer = LogViewer()
    log_viewer.display_log_dashboard()


if __name__ == "__main__":
    # Voor standalone gebruik
    import streamlit as st

    st.set_page_config(page_title="CryptoSmartTrader Log Viewer", page_icon="📊", layout="wide")

    create_log_viewer_page()
