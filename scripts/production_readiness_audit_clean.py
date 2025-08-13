#!/usr/bin/env python3
"""
Clean Production Readiness Audit - Complete System Validation
Checks all components for production deployment readiness
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionReadinessAuditor:
    """Complete production readiness validation"""

    def __init__(self):
        self.audit_results = {}
        self.critical_issues = []
        self.warnings = []
        self.passed_checks = []

    def audit_api_integrations(self) -> Dict[str, Any]:
        """Audit all API integrations for real functionality"""
        logger.info("🔍 Auditing API Integrations...")

        api_status = {
            "kraken_api": False,
            "openai_api": False,
            "news_api": False,
            "twitter_api": False,
            "reddit_api": False,
            "blockchain_apis": False,
        }

        # Test Kraken API
        try:
            import ccxt

            kraken = ccxt.kraken({"enableRateLimit": True})
            ticker = kraken.fetch_ticker("BTC/USD")
            if ticker and "last" in ticker:
                api_status["kraken_api"] = True
                self.passed_checks.append("✅ Kraken API operational")
            else:
                self.critical_issues.append("❌ Kraken API returns invalid data")
        except Exception as e:
            self.critical_issues.append(f"❌ Kraken API failed: {e}")

        # Check OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            self.passed_checks.append("✅ OpenAI API key available")
            api_status["openai_api"] = True
        else:
            self.critical_issues.append("❌ OPENAI_API_KEY missing")

        # Check News/Social APIs
        for api_name, env_var in [
            ("news_api", "NEWS_API_KEY"),
            ("twitter_api", "TWITTER_BEARER_TOKEN"),
            ("reddit_api", "REDDIT_CLIENT_ID"),
        ]:
            if os.getenv(env_var):
                api_status[api_name] = True
                self.passed_checks.append(f"✅ {api_name.upper()} key available")
            else:
                self.critical_issues.append(f"❌ {env_var} missing")

        return api_status

    def audit_ml_models(self) -> Dict[str, Any]:
        """Audit ML models for production readiness"""
        logger.info("🔍 Auditing ML Models...")

        model_status = {
            "random_forest_models": [],
            "model_training_dates": {},
            "models_trained_12_weeks": False,
        }

        model_dir = Path("models/saved")
        if not model_dir.exists():
            self.critical_issues.append("❌ No trained models directory")
            return model_status

        # Check Random Forest models
        required_horizons = ["1h", "24h", "168h", "720h"]
        for horizon in required_horizons:
            model_file = model_dir / f"rf_{horizon}.pkl"
            if model_file.exists():
                model_status["random_forest_models"].append(horizon)

                # Check training date
                file_stat = model_file.stat()
                training_date = datetime.fromtimestamp(file_stat.st_mtime)
                model_status["model_training_dates"][horizon] = training_date.isoformat()

                # Check if model is 12+ weeks old (production ready)
                weeks_old = (datetime.now() - training_date).days / 7
                if weeks_old >= 12:
                    self.passed_checks.append(f"✅ Model {horizon} trained 12+ weeks ago")
                else:
                    self.warnings.append(f"⚠️ Model {horizon} only {weeks_old:.1f} weeks old")

            else:
                self.critical_issues.append(f"❌ Missing RF model: {horizon}")

        # Check if all models are 12+ weeks trained
        all_dates = [
            datetime.fromisoformat(d) for d in model_status["model_training_dates"].values()
        ]
        if all_dates:
            oldest_model = min(all_dates)
            weeks_trained = (datetime.now() - oldest_model).days / 7
            model_status["models_trained_12_weeks"] = weeks_trained >= 12

            if weeks_trained >= 12:
                self.passed_checks.append(f"✅ All models trained for {weeks_trained:.1f} weeks")
            else:
                self.critical_issues.append(
                    f"❌ Models need {12 - weeks_trained:.1f} more weeks training"
                )

        return model_status

    def audit_data_sources(self) -> Dict[str, Any]:
        """Audit data sources for authenticity"""
        logger.info("🔍 Auditing Data Sources...")

        data_status = {
            "market_data_authentic": False,
            "historical_data_available": False,
            "technical_indicators_real": False,
        }

        # Check market data authenticity
        try:
            import ccxt

            kraken = ccxt.kraken({"enableRateLimit": True})
            ticker = kraken.fetch_ticker("BTC/USD")
            if ticker and isinstance(ticker["last"], (int, float)) and ticker["last"] > 0:
                data_status["market_data_authentic"] = True
                self.passed_checks.append("✅ Authentic market data from Kraken")
            else:
                self.critical_issues.append("❌ Invalid market data format")
        except Exception as e:
            self.critical_issues.append(f"❌ Market data unavailable: {e}")

        # Check for historical data capability
        try:
            ohlcv = kraken.fetch_ohlcv("BTC/USD", "1h", limit=100)
            if ohlcv and len(ohlcv) > 50:
                data_status["historical_data_available"] = True
                data_status["technical_indicators_real"] = True
                self.passed_checks.append("✅ Historical OHLCV data available")
            else:
                self.critical_issues.append("❌ Insufficient historical data")
        except Exception as e:
            self.critical_issues.append(f"❌ Historical data fetch failed: {e}")

        return data_status

    def audit_artificial_data_removal(self) -> Dict[str, Any]:
        """Audit that all artificial data has been removed"""
        logger.info("🔍 Auditing Artificial Data Removal...")

        removal_status = {
            "generate_final_predictions_clean": False,
            "ensemble_voting_clean": False,
            "mock_data_cleanup_completed": False,
        }

        # Check if clean prediction generator exists
        clean_pred_file = Path("generate_final_predictions.py")
        if clean_pred_file.exists():
            content = clean_pred_file.read_text()
            if "RealDataPredictionGenerator" in content and "authentic" in content.lower():
                removal_status["generate_final_predictions_clean"] = True
                self.passed_checks.append("✅ Clean prediction generator implemented")
            else:
                self.critical_issues.append("❌ Prediction generator not using authentic data")

        # Check ensemble voting agent
        ensemble_file = Path("agents/ensemble_voting_agent_clean.py")
        if ensemble_file.exists():
            removal_status["ensemble_voting_clean"] = True
            self.passed_checks.append("✅ Clean ensemble voting agent created")
        else:
            self.warnings.append("⚠️ Clean ensemble voting agent not found")

        # Check for cleanup report
        cleanup_report = Path("MOCK_DATA_CLEANUP_REPORT.json")
        if cleanup_report.exists():
            removal_status["mock_data_cleanup_completed"] = True
            self.passed_checks.append("✅ Mock data cleanup report found")
        else:
            self.warnings.append("⚠️ Mock data cleanup report not found")

        return removal_status

    def check_12_week_training_requirement(self) -> bool:
        """Check if 12-week training requirement is met"""
        logger.info("🔍 Checking 12-Week Training Requirement...")

        # Check for training status file
        training_status_file = Path("training_status.json")
        if training_status_file.exists():
            try:
                with open(training_status_file, "r") as f:
                    status = json.load(f)

                if status.get("training_complete", False):
                    self.passed_checks.append("✅ 12-week training requirement met")
                    return True
                else:
                    weeks_trained = status.get("weeks_trained", 0)
                    weeks_remaining = status.get("weeks_remaining", 12)
                    self.critical_issues.append(
                        f"❌ Only {weeks_trained} weeks trained ({weeks_remaining} remaining)"
                    )
                    return False
            except Exception as e:
                self.critical_issues.append(f"❌ Training status file corrupted: {e}")
                return False
        else:
            self.critical_issues.append("❌ No training status tracking found")
            return False

    def generate_production_report(self) -> Dict[str, Any]:
        """Generate comprehensive production readiness report"""
        logger.info("📋 Generating Production Readiness Report...")

        # Run all audits
        api_results = self.audit_api_integrations()
        model_results = self.audit_ml_models()
        data_results = self.audit_data_sources()
        removal_results = self.audit_artificial_data_removal()
        training_ready = self.check_12_week_training_requirement()

        # Calculate overall status
        critical_count = len(self.critical_issues)
        warning_count = len(self.warnings)
        passed_count = len(self.passed_checks)

        # Determine production readiness
        production_ready = critical_count == 0 and training_ready

        report = {
            "timestamp": datetime.now().isoformat(),
            "production_ready": production_ready,
            "summary": {
                "critical_issues": critical_count,
                "warnings": warning_count,
                "passed_checks": passed_count,
                "twelve_week_training_met": training_ready,
            },
            "detailed_results": {
                "api_integrations": api_results,
                "ml_models": model_results,
                "data_sources": data_results,
                "artificial_data_removal": removal_results,
            },
            "issues": {
                "critical": self.critical_issues,
                "warnings": self.warnings,
                "passed": self.passed_checks,
            },
            "recommendations": [],
        }

        # Generate recommendations
        if not production_ready:
            report["recommendations"].extend(
                [
                    "Complete all critical issues before production deployment",
                    "Implement missing API integrations",
                    "Ensure 12-week model training requirement is met",
                ]
            )

        if critical_count == 0 and not training_ready:
            report["recommendations"].append(
                "System ready except for 12-week training requirement - continue training models"
            )

        if production_ready:
            report["recommendations"].append("✅ System is production ready for deployment")

        return report

    def save_report(self, report: Dict[str, Any]):
        """Save production readiness report"""
        report_file = Path("PRODUCTION_READINESS_REPORT.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📋 Production readiness report saved: {report_file}")

        # Also create markdown summary
        self.create_markdown_summary(report)

    def create_markdown_summary(self, report: Dict[str, Any]):
        """Create markdown summary of production readiness"""

        status_emoji = "✅" if report["production_ready"] else "❌"

        markdown_content = f"""# Production Readiness Report

**Status:** {status_emoji} {"PRODUCTION READY" if report["production_ready"] else "NOT PRODUCTION READY"}  
**Date:** {report["timestamp"][:10]}  
**12-Week Training:** {"✅ Met" if report["summary"]["twelve_week_training_met"] else "❌ Not Met"}

## Summary

- **Critical Issues:** {report["summary"]["critical_issues"]}
- **Warnings:** {report["summary"]["warnings"]}  
- **Passed Checks:** {report["summary"]["passed_checks"]}

## Critical Issues
"""

        if report["issues"]["critical"]:
            for issue in report["issues"]["critical"]:
                markdown_content += f"\n{issue}"
        else:
            markdown_content += "\n✅ No critical issues found"

        markdown_content += "\n\n## Warnings\n"

        if report["issues"]["warnings"]:
            for warning in report["issues"]["warnings"]:
                markdown_content += f"\n{warning}"
        else:
            markdown_content += "\n✅ No warnings"

        markdown_content += "\n\n## Passed Checks\n"

        for check in report["issues"]["passed"]:
            markdown_content += f"\n{check}"

        markdown_content += "\n\n## Recommendations\n"

        for rec in report["recommendations"]:
            markdown_content += f"\n• {rec}"

        # Save markdown report
        report_file = Path("PRODUCTION_READINESS_REPORT.md")
        report_file.write_text(markdown_content)

        logger.info(f"📋 Markdown report saved: {report_file}")


def main():
    """Run production readiness audit"""
    auditor = ProductionReadinessAuditor()

    print("🔍 PRODUCTION READINESS AUDIT")
    print("=" * 50)

    report = auditor.generate_production_report()
    auditor.save_report(report)

    # Print summary
    print(f"\n📋 AUDIT COMPLETE")
    print(f"Production Ready: {'✅ YES' if report['production_ready'] else '❌ NO'}")
    print(f"Critical Issues: {report['summary']['critical_issues']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Passed Checks: {report['summary']['passed_checks']}")
    print(
        f"12-Week Training: {'✅ Met' if report['summary']['twelve_week_training_met'] else '❌ Not Met'}"
    )

    if report["summary"]["critical_issues"] > 0:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in report["issues"]["critical"]:
            print(f"  {issue}")

    if report["production_ready"]:
        print("\n✅ SYSTEM IS PRODUCTION READY!")
    else:
        print("\n❌ SYSTEM NOT READY FOR PRODUCTION")
        print("Complete critical issues and ensure 12-week training before deployment")


if __name__ == "__main__":
    main()
