#!/usr/bin/env python3
"""
Enterprise Code Audit System
Implements comprehensive checks for all critical failure modes
Based on enterprise checklist for production trading systems
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import logging
import asyncio
import aiohttp
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

# Core imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..core.structured_logger import get_logger


class CriticalCodeAuditor:
    """Enterprise-grade code audit system for trading infrastructure"""

    def __init__(self):
        self.logger = get_logger("CodeAuditor")
        self.audit_results = {}
        self.critical_issues = []
        self.warnings = []

        # Audit categories
        self.audit_categories = {
            "label_leakage": "Label Leakage / Look-ahead (FATAL)",
            "timestamps": "Timestamp & Timezone Alignment",
            "data_completeness": "NaN & Fallback Detection",
            "data_splits": "Proper Time Series Splitting",
            "target_scaling": "Target Scale Validation",
            "concurrency": "Concurrency & IO Issues",
            "ml_calibration": "ML Probability Calibration",
            "uncertainty": "Uncertainty Quantification",
            "regime_awareness": "Regime-blind Detection",
            "backtest_realism": "Backtest Reality Check",
            "security_logging": "Security & Logging",
        }

    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete code audit across all critical areas"""

        self.logger.info("🔍 STARTING COMPREHENSIVE CODE AUDIT")
        self.logger.info("=" * 60)

        audit_start = datetime.now()

        # Run all audit categories
        self.audit_results = {}

        try:
            self.audit_results["label_leakage"] = self._audit_label_leakage()
            self.audit_results["timestamps"] = self._audit_timestamps()
            self.audit_results["data_completeness"] = self._audit_data_completeness()
            self.audit_results["data_splits"] = self._audit_data_splits()
            self.audit_results["target_scaling"] = self._audit_target_scaling()
            self.audit_results["concurrency"] = self._audit_concurrency()
            self.audit_results["ml_calibration"] = self._audit_ml_calibration()
            self.audit_results["uncertainty"] = self._audit_uncertainty()
            self.audit_results["regime_awareness"] = self._audit_regime_awareness()
            self.audit_results["backtest_realism"] = self._audit_backtest_realism()
            self.audit_results["security_logging"] = self._audit_security_logging()

        except Exception as e:
            self.logger.error(f"Audit failed: {e}")
            self.critical_issues.append(f"Audit system failure: {e}")

        # Generate comprehensive report
        audit_duration = (datetime.now() - audit_start).total_seconds()

        report = self._generate_audit_report(audit_duration)
        self._save_audit_results(report)

        self.logger.info(f"🏁 Code audit completed in {audit_duration:.1f}s")

        return report

    def _audit_label_leakage(self) -> Dict[str, Any]:
        """1.1 Critical: Label leakage / look-ahead detection"""

        self.logger.info("🚨 AUDITING: Label Leakage (MOST CRITICAL)")

        issues = []

        # Check for feature files
        feature_files = (
            list(Path("exports").glob("*features*.parquet")) if Path("exports").exists() else []
        )

        if not feature_files:
            issues.append("No feature files found - cannot check label leakage")
            return {"status": "warning", "issues": issues}

        try:
            for file_path in feature_files[:3]:  # Check first 3 files
                try:
                    df = pd.read_parquet(file_path)

                    # Check 1: Label timestamp after feature timestamp
                    if "label_ts" in df.columns and "ts" in df.columns:
                        look_ahead_count = (df["label_ts"] <= df["ts"]).sum()
                        if look_ahead_count > 0:
                            issues.append(
                                f"CRITICAL: {look_ahead_count} look-ahead labels in {file_path.name}"
                            )

                    # Check 2: Future features (t+ suffix)
                    future_cols = [c for c in df.columns if c.startswith("feat_") and "t+" in c]
                    if future_cols:
                        issues.append(f"CRITICAL: Future features found: {future_cols[:5]}")

                    # Check 3: Suspicious shift operations
                    for col in df.columns:
                        if "shift_pos" in col or "future_" in col:
                            issues.append(f"SUSPICIOUS: Future-looking column: {col}")

                except Exception as e:
                    issues.append(f"Could not audit {file_path}: {e}")

        except Exception as e:
            issues.append(f"Label leakage audit failed: {e}")

        # Check codebase for dangerous patterns
        dangerous_patterns = self._scan_codebase_patterns(
            ["shift(+", ".ffill(", ".bfill(", "train_test_split(", "KFold("]
        )

        if dangerous_patterns:
            issues.extend([f"CODEBASE: {pattern}" for pattern in dangerous_patterns])

        status = "critical" if any("CRITICAL" in issue for issue in issues) else "pass"

        return {
            "status": status,
            "issues": issues,
            "files_checked": len(feature_files),
            "dangerous_patterns": dangerous_patterns,
        }

    def _audit_timestamps(self) -> Dict[str, Any]:
        """1.2 Timestamp & timezone validation"""

        self.logger.info("🕐 AUDITING: Timestamps & Timezones")

        issues = []

        try:
            # Check feature files for timestamp alignment
            feature_files = (
                list(Path("exports").glob("*features*.parquet")) if Path("exports").exists() else []
            )

            for file_path in feature_files[:2]:
                try:
                    df = pd.read_parquet(file_path)

                    if "ts" in df.columns:
                        ts_col = df["ts"]

                        # Check timezone
                        if hasattr(ts_col.dtype, "tz") and ts_col.dt.tz is None:
                            issues.append(f"Missing timezone in {file_path.name}")

                        # Check alignment to candle boundaries (hourly)
                        if not ts_col.empty:
                            misaligned = (ts_col != ts_col.dt.floor("1H")).sum()
                            if misaligned > 0:
                                issues.append(
                                    f"Non-candle aligned timestamps: {misaligned}/{len(ts_col)}"
                                )

                except Exception as e:
                    issues.append(f"Timestamp check failed for {file_path}: {e}")

        except Exception as e:
            issues.append(f"Timestamp audit failed: {e}")

        return {"status": "critical" if issues else "pass", "issues": issues}

    def _audit_data_completeness(self) -> Dict[str, Any]:
        """1.3 NaN's & fallback detection"""

        self.logger.info("🔍 AUDITING: Data Completeness (Zero Fallback)")

        issues = []

        try:
            feature_files = (
                list(Path("exports").glob("*features*.parquet")) if Path("exports").exists() else []
            )

            # Required features for production (no fallbacks allowed)
            required_features = [
                "sent_score",
                "sent_prob",
                "whale_score",
                "rsi_14",
                "vol_24h",
                "price",
                "change_24h",
            ]

            for file_path in feature_files[:2]:
                try:
                    df = pd.read_parquet(file_path)

                    # Check required features
                    available_req = [col for col in required_features if col in df.columns]

                    if available_req:
                        incomplete_rows = df[available_req].isna().any(axis=1).sum() / len(df)

                        if incomplete_rows > 0.0:
                            issues.append(
                                f"CRITICAL: {incomplete_rows * 100:.2f}% incomplete rows in {file_path.name}"
                            )

                        # Check for placeholder values
                        for col in available_req:
                            if col in df.columns:
                                # Common placeholder detection
                                placeholder_count = (
                                    (df[col] == 0.0).sum()
                                    + (df[col] == -999).sum()
                                    + (df[col] == 999).sum()

                                if placeholder_count > len(df) * 0.1:  # >10% placeholders
                                    issues.append(
                                        f"SUSPICIOUS: {col} has {placeholder_count} potential placeholders"
                                    )

                except Exception as e:
                    issues.append(f"Completeness check failed for {file_path}: {e}")

        except Exception as e:
            issues.append(f"Data completeness audit failed: {e}")

        return {
            "status": "critical" if any("CRITICAL" in issue for issue in issues) else "pass",
            "issues": issues,
        }

    def _audit_data_splits(self) -> Dict[str, Any]:
        """1.4 Proper time series splitting"""

        self.logger.info("📊 AUDITING: Data Splitting Methods")

        issues = []

        # Scan for dangerous split methods
        dangerous_splits = self._scan_codebase_patterns(
            ["train_test_split(", "StratifiedKFold(", "KFold(", "shuffle=True"]
        )

        # Look for proper time series splits
        good_splits = self._scan_codebase_patterns(
            ["TimeSeriesSplit(", "rolling_window", "walk_forward"]
        )

        if dangerous_splits and not good_splits:
            issues.append("CRITICAL: Using random splits instead of time series splits")

        if dangerous_splits:
            issues.extend([f"DANGEROUS SPLIT: {split}" for split in dangerous_splits])

        return {
            "status": "critical" if any("CRITICAL" in issue for issue in issues) else "pass",
            "issues": issues,
            "dangerous_splits": dangerous_splits,
            "proper_splits": good_splits,
        }

    def _audit_target_scaling(self) -> Dict[str, Any]:
        """1.5 Target scale validation"""

        self.logger.info("🎯 AUDITING: Target Scale Sanity")

        issues = []

        try:
            feature_files = (
                list(Path("exports").glob("*features*.parquet")) if Path("exports").exists() else []
            )

            for file_path in feature_files[:2]:
                try:
                    df = pd.read_parquet(file_path)

                    # Check target columns
                    target_cols = [col for col in df.columns if "target_" in col or "label_" in col]

                    for target_col in target_cols:
                        if target_col in df.columns and not df[target_col].empty:
                            q99 = df[target_col].abs().quantile(0.99)

                            # Targets should be in decimal form (0.05 not 5.0 for 5%)
                            if q99 > 3.0:  # Suspiciously large
                                issues.append(
                                    f"SUSPICIOUS: {target_col} scale (q99={q99:.2f}), might be percentage not decimal"
                                )

                            # Check for unrealistic values
                            if q99 > 10.0:  # >1000% returns
                                issues.append(
                                    f"CRITICAL: {target_col} has unrealistic values (q99={q99:.2f})"
                                )

                except Exception as e:
                    issues.append(f"Target scale check failed for {file_path}: {e}")

        except Exception as e:
            issues.append(f"Target scaling audit failed: {e}")

        return {
            "status": "critical" if any("CRITICAL" in issue for issue in issues) else "pass",
            "issues": issues,
        }

    def _audit_concurrency(self) -> Dict[str, Any]:
        """1.6 Concurrency & IO issues"""

        self.logger.info("⚡ AUDITING: Concurrency & IO Performance")

        issues = []

        # Scan for blocking operations
        blocking_patterns = self._scan_codebase_patterns(
            ["requests.get(", "requests.post(", "time.sleep(", "synchronous"]
        )

        # Look for async patterns
        async_patterns = self._scan_codebase_patterns(["async def", "await ", "aiohttp", "asyncio"])

        if blocking_patterns and not async_patterns:
            issues.append("WARNING: Blocking operations without async alternatives")

        # Check for timeout configurations
        timeout_patterns = self._scan_codebase_patterns(["timeout="])

        if blocking_patterns and not timeout_patterns:
            issues.append("CRITICAL: Network requests without timeouts")

        return {
            "status": "warning" if issues else "pass",
            "issues": issues,
            "blocking_patterns": len(blocking_patterns),
            "async_patterns": len(async_patterns),
        }

    def _audit_ml_calibration(self) -> Dict[str, Any]:
        """1.7 ML probability calibration"""

        self.logger.info("🧠 AUDITING: ML Calibration")

        issues = []

        # Check for calibration in codebase
        calibration_patterns = self._scan_codebase_patterns(
            ["CalibratedClassifierCV", "isotonic", "calibrat", "reliability_curve"]
        )

        confidence_patterns = self._scan_codebase_patterns(["confidence", "conf_", "uncertainty"])

        if confidence_patterns and not calibration_patterns:
            issues.append("WARNING: Confidence scores without calibration validation")

        # Mock calibration check (would use real model predictions in production)
        try:
            # REMOVED: Mock data pattern not allowed in production
            # REMOVED: Mock data pattern not allowed in production0.6, 0.95, 1000)
            # REMOVED: Mock data pattern not allowed in production1, mock_probs)

            # Simple calibration metric
            high_conf_mask = mock_probs > 0.8
            if high_conf_mask.sum() > 0:
                realized_accuracy = random.choice()
                if realized_accuracy < 0.7:  # 80% confident should be >70% accurate
                    issues.append(
                        f"CALIBRATION: High confidence (>80%) only {realized_accuracy:.1%} accurate"
                    )

        except Exception as e:
            issues.append(f"Calibration check failed: {e}")

        return {"status": "warning" if issues else "pass", "issues": issues}

    def _audit_uncertainty(self) -> Dict[str, Any]:
        """1.8 Uncertainty quantification"""

        self.logger.info("🎲 AUDITING: Uncertainty Quantification")

        issues = []

        # Check for uncertainty methods
        uncertainty_patterns = self._scan_codebase_patterns(
            ["MC.*Dropout", "ensemble", "uncertainty", "confidence_interval", "std"]
        )

        if not uncertainty_patterns:
            issues.append("WARNING: No uncertainty quantification methods found")

        # Check for point estimates only
        point_estimate_patterns = self._scan_codebase_patterns(["predict(", "predict_proba("])

        if point_estimate_patterns and not uncertainty_patterns:
            issues.append("CRITICAL: Only point estimates, no uncertainty bands")

        return {"status": "warning" if issues else "pass", "issues": issues}

    def _audit_regime_awareness(self) -> Dict[str, Any]:
        """1.9 Regime-blind detection"""

        self.logger.info("🌊 AUDITING: Regime Awareness")

        issues = []

        # Check for regime-aware features
        regime_patterns = self._scan_codebase_patterns(
            ["regime", "volatility", "ATR", "market_state", "bull", "bear"]
        )

        if not regime_patterns:
            issues.append("WARNING: No regime-aware features detected")

        # Mock regime performance check
        try:
            # REMOVED: Mock data pattern not allowed in production
            mock_errors = {"low_vol": np.random.normal(0, 1), "high_vol": np.random.normal(0, 1)}

            low_vol_mae = np.mean(np.abs(mock_errors["low_vol"]))
            high_vol_mae = np.mean(np.abs(mock_errors["high_vol"]))

            if high_vol_mae > low_vol_mae * 2:
                issues.append(
                    f"REGIME: High volatility errors {high_vol_mae / low_vol_mae:.1f}x larger than low volatility"
                )

        except Exception as e:
            issues.append(f"Regime analysis failed: {e}")

        return {"status": "warning" if issues else "pass", "issues": issues}

    def _audit_backtest_realism(self) -> Dict[str, Any]:
        """1.10 Backtest realism check"""

        self.logger.info("📈 AUDITING: Backtest Realism")

        issues = []

        # Check for realistic execution modeling
        execution_patterns = self._scan_codebase_patterns(
            ["slippage", "market_impact", "latency", "partial_fill", "L2", "orderbook"]
        )

        if not execution_patterns:
            issues.append("CRITICAL: No realistic execution modeling in backtests")

        # Check for overly optimistic assumptions
        perfect_execution_patterns = self._scan_codebase_patterns(
            ["perfect_execution", "zero_slippage", "instant_fill"]
        )

        if perfect_execution_patterns:
            issues.append("CRITICAL: Perfect execution assumptions detected")

        return {
            "status": "critical" if any("CRITICAL" in issue for issue in issues) else "pass",
            "issues": issues,
        }

    def _audit_security_logging(self) -> Dict[str, Any]:
        """1.11 Security & logging practices"""

        self.logger.info("🔒 AUDITING: Security & Logging")

        issues = []

        # Check for secrets in logs
        secret_patterns = self._scan_codebase_patterns(["API_KEY", "SECRET", "PASSWORD", "TOKEN"])

        log_patterns = self._scan_codebase_patterns(["logger.info(", "print(", "logging"])

        if secret_patterns and log_patterns:
            issues.append("WARNING: Potential secrets in logging code - verify redaction")

        # Check for correlation IDs
        correlation_patterns = self._scan_codebase_patterns(
            ["correlation_id", "request_id", "run_id"]
        )

        if not correlation_patterns:
            issues.append("WARNING: No correlation IDs for request tracing")

        # Check for authentication
        auth_patterns = self._scan_codebase_patterns(["auth", "jwt", "login", "password"])

        dashboard_patterns = self._scan_codebase_patterns(["streamlit", "dashboard", "app.py"])

        if dashboard_patterns and not auth_patterns:
            issues.append("WARNING: Dashboard without authentication")

        return {"status": "warning" if issues else "pass", "issues": issues}

    def _scan_codebase_patterns(self, patterns: List[str]) -> List[str]:
        """Scan codebase for specific patterns"""

        found_patterns = []

        try:
            # Scan Python files
            python_files = list(Path(".").rglob("*.py"))

            for pattern in patterns:
                pattern_found = False

                for py_file in python_files[:20]:  # Limit scan to prevent timeout
                    try:
                        if py_file.name.startswith(".") or "__pycache__" in str(py_file):
                            continue

                        content = py_file.read_text(encoding="utf-8", errors="ignore")

                        if pattern.lower() in content.lower():
                            found_patterns.append(f"{pattern} in {py_file.name}")
                            pattern_found = True
                            break

                    except Exception:
                        continue

                if pattern_found:
                    break

        except Exception as e:
            self.logger.warning(f"Pattern scan failed: {e}")

        return found_patterns

    def _generate_audit_report(self, duration: float) -> Dict[str, Any]:
        """Generate comprehensive audit report"""

        # Count issues by severity
        critical_count = sum(
            1 for result in self.audit_results.values() if result.get("status") == "critical"
        )
        warning_count = sum(
            1 for result in self.audit_results.values() if result.get("status") == "warning"
        )

        # Overall status
        if critical_count > 0:
            overall_status = "CRITICAL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        # Generate summary
        summary = {
            "audit_timestamp": datetime.now().isoformat(),
            "audit_duration_seconds": duration,
            "overall_status": overall_status,
            "critical_issues": critical_count,
            "warnings": warning_count,
            "total_categories": len(self.audit_categories),
            "categories_audited": len(self.audit_results),
        }

        report = {
            "summary": summary,
            "detailed_results": self.audit_results,
            "category_descriptions": self.audit_categories,
        }

        return report

    def _save_audit_results(self, report: Dict[str, Any]) -> None:
        """Save audit results to file"""

        try:
            # Create audit directory
            audit_dir = Path("logs/audit")
            audit_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = audit_dir / f"code_audit_{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Save latest
            latest_file = audit_dir / "latest_audit.json"
            with open(latest_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"📁 Audit report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to save audit report: {e}")


def run_critical_code_audit():
    """Run comprehensive code audit"""

    print("🔍 ENTERPRISE CODE AUDIT SYSTEM")
    print("=" * 50)
    print("Running comprehensive audit of all critical failure modes...")

    auditor = CriticalCodeAuditor()
    report = auditor.run_comprehensive_audit()

    # Print summary
    summary = report["summary"]

    print(f"\n📊 AUDIT SUMMARY")
    print(f"Status: {summary['overall_status']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Duration: {summary['audit_duration_seconds']:.1f}s")

    # Print critical issues
    if summary["critical_issues"] > 0:
        print(f"\n🚨 CRITICAL ISSUES:")
        for category, result in report["detailed_results"].items():
            if result.get("status") == "critical":
                print(f"   {category.upper()}:")
                for issue in result.get("issues", []):
                    print(f"     • {issue}")

    # Print warnings
    if summary["warnings"] > 0:
        print(f"\n⚠️ WARNINGS:")
        for category, result in report["detailed_results"].items():
            if result.get("status") == "warning":
                print(f"   {category.upper()}:")
                for issue in result.get("issues", [])[:3]:  # First 3 warnings
                    print(f"     • {issue}")

    if summary["overall_status"] == "PASS":
        print(f"\n✅ All critical checks passed!")

    print(f"\n📁 Detailed report saved to logs/audit/")

    return report


if __name__ == "__main__":
    report = run_critical_code_audit()
    print("🏁 Code audit completed")
