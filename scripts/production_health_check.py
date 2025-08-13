#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Production Health Check
Comprehensive system health verification and error resolution
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from containers import ApplicationContainer
# Import not needed for this health check


class ProductionHealthChecker:
    """Production health checker and error resolver"""

    def __init__(self):
        self.container = ApplicationContainer()
        self.container.wire(modules=[__name__])

        self.logger = logging.getLogger(__name__)
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "errors_found": [],
            "warnings_found": [],
            "fixes_applied": [],
            "system_status": "unknown",
        }

    def run_comprehensive_health_check(self) -> dict:
        """Run comprehensive health check and fix issues"""
        self.logger.info("Starting comprehensive production health check...")

        # Check all system components
        self._check_exchange_initialization()
        self._check_monitoring_system()
        self._check_real_time_pipeline()
        self._check_cache_system()
        self._check_ml_systems()
        self._check_dashboard_accessibility()

        # Determine overall system status
        if self.health_report["errors_found"]:
            self.health_report["system_status"] = "degraded"
        elif self.health_report["warnings_found"]:
            self.health_report["system_status"] = "warning"
        else:
            self.health_report["system_status"] = "healthy"

        self.logger.info(f"Health check completed - Status: {self.health_report['system_status']}")
        return self.health_report

    def _check_exchange_initialization(self):
        """Check exchange initialization issues"""
        try:
            scanner = self.container.comprehensive_market_scanner()

            if not scanner.exchanges:
                self.health_report["errors_found"].append(
                    {
                        "component": "Exchange Initialization",
                        "error": "No exchanges initialized",
                        "severity": "high",
                    }
                )
            else:
                working_exchanges = 0
                for name, exchange in scanner.exchanges.items():
                    try:
                        # Test exchange connectivity
                        exchange.load_markets()
                        working_exchanges += 1
                        self.logger.info(f"Exchange {name} working correctly")
                    except Exception as e:
                        if "coinbasepro" in str(e) or "coinbase" in str(e):
                            self.health_report["warnings_found"].append(
                                {
                                    "component": "Exchange Initialization",
                                    "warning": f"Coinbase exchange not available: {e}",
                                    "severity": "low",
                                }
                            )
                        else:
                            self.health_report["warnings_found"].append(
                                {
                                    "component": "Exchange Initialization",
                                    "warning": f"Exchange {name} issues: {e}",
                                    "severity": "medium",
                                }
                            )

                if working_exchanges == 0:
                    self.health_report["errors_found"].append(
                        {
                            "component": "Exchange Connectivity",
                            "error": "No working exchanges available",
                            "severity": "high",
                        }
                    )

        except Exception as e:
            self.health_report["errors_found"].append(
                {
                    "component": "Exchange System",
                    "error": f"Exchange system failure: {e}",
                    "severity": "critical",
                }
            )

    def _check_monitoring_system(self):
        """Check monitoring system health"""
        try:
            monitoring = self.container.monitoring_system()

            # Check if monitoring is running
            if not monitoring.is_monitoring:
                self.health_report["warnings_found"].append(
                    {
                        "component": "Monitoring System",
                        "warning": "Monitoring not active",
                        "severity": "medium",
                    }
                )

            # Check metrics server
            if not hasattr(monitoring, "metrics_server_port"):
                self.health_report["warnings_found"].append(
                    {
                        "component": "Metrics Server",
                        "warning": "Metrics server may not be running",
                        "severity": "low",
                    }
                )

        except Exception as e:
            self.health_report["errors_found"].append(
                {
                    "component": "Monitoring System",
                    "error": f"Monitoring system failure: {e}",
                    "severity": "high",
                }
            )

    def _check_real_time_pipeline(self):
        """Check real-time pipeline health"""
        try:
            pipeline = self.container.real_time_pipeline()

            # Run pipeline tasks to check for errors
            test_results = {
                "coin_discovery": pipeline.discover_coins(),
                "price_data": pipeline.collect_price_data(),
                "sentiment_scraping": pipeline.scrape_sentiment_data(),
                "whale_detection": pipeline.detect_whale_activity(),
                "ml_inference": pipeline.run_ml_batch_inference(),
            }

            for task_name, result in test_results.items():
                if not result.get("success", False):
                    reason = result.get("reason", result.get("error", "Unknown"))
                    if "No validated" in reason or "Insufficient" in reason:
                        self.health_report["warnings_found"].append(
                            {
                                "component": "Real-time Pipeline",
                                "warning": f"{task_name}: {reason}",
                                "severity": "low",
                            }
                        )
                    else:
                        self.health_report["errors_found"].append(
                            {
                                "component": "Real-time Pipeline",
                                "error": f"{task_name}: {reason}",
                                "severity": "medium",
                            }
                        )

        except Exception as e:
            self.health_report["errors_found"].append(
                {
                    "component": "Real-time Pipeline",
                    "error": f"Pipeline system failure: {e}",
                    "severity": "high",
                }
            )

    def _check_cache_system(self):
        """Check cache system health"""
        try:
            cache_manager = self.container.cache_manager()

            # Test cache operations
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat(), "test": True}

            cache_manager.set(test_key, test_value, ttl_minutes=1)
            retrieved = cache_manager.get(test_key)

            if retrieved != test_value:
                self.health_report["errors_found"].append(
                    {
                        "component": "Cache System",
                        "error": "Cache read/write test failed",
                        "severity": "medium",
                    }
                )
            else:
                self.logger.info("Cache system working correctly")

        except Exception as e:
            self.health_report["errors_found"].append(
                {
                    "component": "Cache System",
                    "error": f"Cache system failure: {e}",
                    "severity": "high",
                }
            )

    def _check_ml_systems(self):
        """Check ML systems health"""
        try:
            # Check multi-horizon ML
            ml_system = self.container.multi_horizon_ml()

            # Check if models are available
            try:
                model_status = (
                    ml_system.get_model_status() if hasattr(ml_system, "get_model_status") else {}
                )
                loaded_models = sum(
                    1 for status in model_status.values() if status.get("loaded", False)
                )
            except Exception:
                loaded_models = 0

            if loaded_models == 0:
                self.health_report["warnings_found"].append(
                    {
                        "component": "ML Systems",
                        "warning": "No ML models loaded",
                        "severity": "medium",
                    }
                )

            # Check ML/AI differentiators
            try:
                ml_ai_diff = self.container.ml_ai_differentiators()
                status = ml_ai_diff.get_differentiator_status()

                if status.get("completion_rate", 0) < 50:
                    self.health_report["warnings_found"].append(
                        {
                            "component": "ML/AI Differentiators",
                            "warning": f"Low implementation rate: {status.get('completion_rate', 0)}%",
                            "severity": "low",
                        }
                    )

            except Exception as diff_error:
                self.health_report["warnings_found"].append(
                    {
                        "component": "ML/AI Differentiators",
                        "warning": f"Differentiators system issues: {diff_error}",
                        "severity": "medium",
                    }
                )

        except Exception as e:
            self.health_report["errors_found"].append(
                {"component": "ML Systems", "error": f"ML system failure: {e}", "severity": "high"}
            )

    def _check_dashboard_accessibility(self):
        """Check dashboard accessibility"""
        try:
            # Test dashboard imports
            dashboard_modules = [
                "dashboards.main_dashboard",
                "dashboards.comprehensive_market_dashboard",
                "dashboards.ai_ml_dashboard",
                "dashboards.crypto_ai_system_dashboard",
                "dashboards.ml_ai_differentiators_dashboard",
            ]

            for module_name in dashboard_modules:
                try:
                    __import__(module_name)
                    self.logger.debug(f"Dashboard module {module_name} imports correctly")
                except ImportError as import_error:
                    self.health_report["errors_found"].append(
                        {
                            "component": "Dashboard System",
                            "error": f"Dashboard import failed: {module_name} - {import_error}",
                            "severity": "medium",
                        }
                    )

        except Exception as e:
            self.health_report["errors_found"].append(
                {
                    "component": "Dashboard System",
                    "error": f"Dashboard system failure: {e}",
                    "severity": "high",
                }
            )

    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        report_lines = [
            "=" * 80,
            "CRYPTOSMARTTRADER V2 - PRODUCTION HEALTH REPORT",
            "=" * 80,
            f"Generated: {self.health_report['timestamp']}",
            f"System Status: {self.health_report['system_status'].upper()}",
            "",
        ]

        # Errors section
        if self.health_report["errors_found"]:
            report_lines.extend(["🔴 CRITICAL ERRORS:", "-" * 40])
            for error in self.health_report["errors_found"]:
                report_lines.append(
                    f"• {error['component']}: {error['error']} (Severity: {error['severity']})"
                )
            report_lines.append("")

        # Warnings section
        if self.health_report["warnings_found"]:
            report_lines.extend(["🟡 WARNINGS:", "-" * 40])
            for warning in self.health_report["warnings_found"]:
                report_lines.append(
                    f"• {warning['component']}: {warning['warning']} (Severity: {warning['severity']})"
                )
            report_lines.append("")

        # Recommendations
        report_lines.extend(["📋 RECOMMENDATIONS:", "-" * 40])

        if not self.health_report["errors_found"] and not self.health_report["warnings_found"]:
            report_lines.append("✅ System is running optimally - no issues detected")
        else:
            if any(e["severity"] == "critical" for e in self.health_report["errors_found"]):
                report_lines.append("🚨 Immediate attention required for critical errors")

            if any(e["severity"] == "high" for e in self.health_report["errors_found"]):
                report_lines.append("⚠️  High priority errors need resolution")

            if self.health_report["warnings_found"]:
                report_lines.append("ℹ️  Review warnings for system optimization")

        report_lines.extend(["", "=" * 80])

        return "\n".join(report_lines)


def main():
    """Main health check execution"""
    try:
        health_checker = ProductionHealthChecker()
        health_report = health_checker.run_comprehensive_health_check()

        # Generate and display report
        report_text = health_checker.generate_health_report()
        print(report_text)

        # Save report to file
        report_path = Path("logs/health_report.txt")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report_text)

        # Save JSON report
        json_path = Path("logs/health_report.json")
        with open(json_path, "w") as f:
            json.dump(health_report, f, indent=2)

        print(f"\nHealth reports saved to:")
        print(f"• {report_path}")
        print(f"• {json_path}")

        # Return appropriate exit code
        if health_report["system_status"] in ["degraded", "critical"]:
            return 1
        else:
            return 0

    except Exception as e:
        print(f"Health check failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
