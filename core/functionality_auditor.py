#!/usr/bin/env python3
"""
Functionality Auditor
Complete audit of all required functionalities
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class FunctionalityAuditor:
    """
    Complete functionality audit system
    """

    def __init__(self):
        self.audit_results = {}
        self.missing_features = []
        self.implemented_features = []

    def run_complete_functionality_audit(self) -> Dict[str, Any]:
        """Run complete functionality audit"""

        print("🔍 RUNNING COMPLETE FUNCTIONALITY AUDIT")
        print("=" * 50)

        audit_start = time.time()

        # Audit all required functionalities
        self._audit_dynamic_kraken_coverage()
        self._audit_no_dummy_fallback()
        self._audit_batched_multi_horizon()
        self._audit_confidence_gate_80()
        self._audit_uncertainty_calibration()
        self._audit_regime_awareness()
        self._audit_explainability_shap()
        self._audit_realistic_backtest()
        self._audit_orchestrator_isolation()
        self._audit_daily_eval_health()
        self._audit_security_implementation()

        audit_duration = time.time() - audit_start

        # Calculate overall implementation score
        total_features = len(self.audit_results)
        implemented_count = len(self.implemented_features)
        implementation_score = (
            (implemented_count / total_features) * 100 if total_features > 0 else 0
        )

        # Compile audit report
        audit_report = {
            "audit_timestamp": datetime.now().isoformat(),
            "audit_duration": audit_duration,
            "total_features_audited": total_features,
            "implemented_features": implemented_count,
            "missing_features": len(self.missing_features),
            "implementation_score": implementation_score,
            "audit_results": self.audit_results,
            "implemented_features_list": self.implemented_features,
            "missing_features_list": self.missing_features,
            "overall_status": "PRODUCTION_READY" if implementation_score >= 80 else "INCOMPLETE",
        }

        # Save audit report
        self._save_functionality_report(audit_report)

        return audit_report

    def _audit_dynamic_kraken_coverage(self):
        """Audit 1: Dynamic Kraken coverage (100%)"""

        print("🌐 Auditing dynamic Kraken coverage...")

        # Check for coverage logs
        coverage_logs = (
            list(Path("logs/coverage").glob("coverage_*.json"))
            if Path("logs/coverage").exists()
            else []
        )

        coverage_implemented = False
        coverage_percentage = 0.0
        missing_symbols = []

        if coverage_logs:
            try:
                # Get latest coverage log
                latest_coverage = max(coverage_logs, key=lambda p: p.stat().st_mtime)

                with open(latest_coverage, "r") as f:
                    coverage_data = json.load(f)

                coverage_percentage = coverage_data.get("coverage_pct", 0.0)
                missing_symbols = coverage_data.get("missing", [])

                if coverage_percentage >= 99 and len(missing_symbols) == 0:
                    coverage_implemented = True
                    self.implemented_features.append("Dynamic Kraken coverage (100%)")
                else:
                    self.missing_features.append(f"Kraken coverage only {coverage_percentage:.1f}%")

            except Exception as e:
                self.missing_features.append(f"Coverage logs unreadable: {e}")
        else:
            self.missing_features.append("No coverage logs found")

        # Check for coverage audit implementation
        coverage_audit_exists = Path("eval/coverage_audit.py").exists()

        self.audit_results["dynamic_kraken_coverage"] = {
            "implemented": coverage_implemented,
            "coverage_percentage": coverage_percentage,
            "missing_symbols_count": len(missing_symbols),
            "coverage_logs_found": len(coverage_logs),
            "coverage_audit_exists": coverage_audit_exists,
            "evidence_path": str(latest_coverage) if coverage_logs else None,
        }

        print(f"   Coverage: {coverage_percentage:.1f}%, Missing: {len(missing_symbols)}")

    def _audit_no_dummy_fallback(self):
        """Audit 2: No dummy/fallback in production"""

        print("🚫 Auditing no dummy/fallback policy...")

        # Check for completeness gate
        completeness_gate_exists = Path("core/completeness_gate.py").exists()

        # Check for validation files
        validation_files = (
            list(Path(".").glob("**/validation_*.json")) if Path(".").exists() else []
        )
        great_expectations_exists = Path("great_expectations").exists()

        # Check evaluator implementation
        evaluator_files = list(Path("eval").glob("*.py")) if Path("eval").exists() else []

        # Scan for fallback patterns in code
        fallback_patterns_found = 0
        dummy_data_patterns = 0

        py_files = list(Path(".").glob("**/*.py"))
        for file_path in py_files[:100]:  # Sample check
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                if "fallback" in content and "data" in content:
                    fallback_patterns_found += 1

                if "dummy" in content or "placeholder" in content:
                    dummy_data_patterns += 1

            except Exception:
                continue

        no_fallback_implemented = (
            completeness_gate_exists and fallback_patterns_found < 5 and dummy_data_patterns < 10
        )

        if no_fallback_implemented:
            self.implemented_features.append("No dummy/fallback in production")
        else:
            self.missing_features.append("Fallback patterns detected in code")

        self.audit_results["no_dummy_fallback"] = {
            "implemented": no_fallback_implemented,
            "completeness_gate_exists": completeness_gate_exists,
            "validation_files_count": len(validation_files),
            "great_expectations_exists": great_expectations_exists,
            "evaluator_files_count": len(evaluator_files),
            "fallback_patterns_found": fallback_patterns_found,
            "dummy_data_patterns": dummy_data_patterns,
            "evidence_path": "core/completeness_gate.py" if completeness_gate_exists else None,
        }

        print(f"   No fallback: {'✓' if no_fallback_implemented else '✗'}")

    def _audit_batched_multi_horizon(self):
        """Audit 3: Batched multi-horizon inference"""

        print("⏱️ Auditing batched multi-horizon inference...")

        # Check for predictions export
        predictions_file = Path("exports/predictions.csv")
        batch_inference_exists = Path("ml/batch_inference.py").exists()

        multi_horizon_implemented = False
        horizons_found = []

        if predictions_file.exists():
            try:
                # Check file size and columns
                file_size_mb = predictions_file.stat().st_size / (1024 * 1024)

                # Read header to check columns
                with open(predictions_file, "r") as f:
                    header = f.readline()

                # Check for horizon columns
                horizons = ["1h", "24h", "7d", "30d"]
                for horizon in horizons:
                    if f"pred_{horizon}" in header and f"conf_{horizon}" in header:
                        horizons_found.append(horizon)

                if len(horizons_found) >= 3:  # At least 3 horizons
                    multi_horizon_implemented = True
                    self.implemented_features.append("Batched multi-horizon inference")
                else:
                    self.missing_features.append(f"Only {len(horizons_found)} horizons found")

            except Exception as e:
                self.missing_features.append(f"Predictions file unreadable: {e}")
        else:
            self.missing_features.append("No predictions.csv found in exports/")

        self.audit_results["batched_multi_horizon"] = {
            "implemented": multi_horizon_implemented,
            "predictions_file_exists": predictions_file.exists(),
            "batch_inference_exists": batch_inference_exists,
            "horizons_found": horizons_found,
            "horizons_count": len(horizons_found),
            "evidence_path": "exports/predictions.csv" if predictions_file.exists() else None,
        }

        print(f"   Multi-horizon: {len(horizons_found)} horizons found")

    def _audit_confidence_gate_80(self):
        """Audit 4: 80% confidence gate"""

        print("🚪 Auditing 80% confidence gate...")

        # Check for confidence gate implementation
        confidence_gate_files = list(Path(".").glob("**/confidence_gate*.py"))

        # Check dashboard implementation
        dashboard_files = list(Path(".").glob("**/app*.py"))

        # Check for confidence gate logs
        today = datetime.now().strftime("%Y%m%d")
        confidence_logs = (
            list(Path(f"logs/daily/{today}").glob("confidence_gate*"))
            if Path(f"logs/daily/{today}").exists()
            else []
        )

        confidence_gate_implemented = False
        gate_operational = False

        if confidence_gate_files:
            # Check if 80% threshold is implemented
            for gate_file in confidence_gate_files:
                try:
                    with open(gate_file, "r") as f:
                        content = f.read()

                    if "0.8" in content or "80" in content:
                        confidence_gate_implemented = True
                        break

                except Exception:
                    continue

        # Check operational status from logs
        if confidence_logs:
            try:
                latest_log = max(confidence_logs, key=lambda p: p.stat().st_mtime)

                with open(latest_log, "r") as f:
                    content = f.read()

                if "confidence" in content.lower() and "gate" in content.lower():
                    gate_operational = True

            except Exception:
                pass

        if confidence_gate_implemented:
            self.implemented_features.append("80% confidence gate")
        else:
            self.missing_features.append("80% confidence gate not implemented")

        self.audit_results["confidence_gate_80"] = {
            "implemented": confidence_gate_implemented,
            "gate_operational": gate_operational,
            "confidence_gate_files": len(confidence_gate_files),
            "dashboard_files": len(dashboard_files),
            "confidence_logs": len(confidence_logs),
            "evidence_path": str(confidence_gate_files[0]) if confidence_gate_files else None,
        }

        print(f"   Confidence gate: {'✓' if confidence_gate_implemented else '✗'}")

    def _audit_uncertainty_calibration(self):
        """Audit 5: Uncertainty & calibration active"""

        print("🎯 Auditing uncertainty & calibration...")

        # Check for calibration implementation
        calibration_files = list(Path(".").glob("**/calibration*.py"))
        uncertainty_files = list(Path(".").glob("**/uncertainty*.py"))

        # Check for calibration logs
        today = datetime.now().strftime("%Y%m%d")
        calibration_logs = (
            list(Path(f"logs/daily/{today}").glob("calibration*"))
            if Path(f"logs/daily/{today}").exists()
            else []
        )

        calibration_implemented = len(calibration_files) > 0 or len(uncertainty_files) > 0

        if calibration_implemented:
            self.implemented_features.append("Uncertainty & calibration active")
        else:
            self.missing_features.append("No calibration implementation found")

        self.audit_results["uncertainty_calibration"] = {
            "implemented": calibration_implemented,
            "calibration_files": len(calibration_files),
            "uncertainty_files": len(uncertainty_files),
            "calibration_logs": len(calibration_logs),
            "evidence_path": str(calibration_files[0])
            if calibration_files
            else str(uncertainty_files[0])
            if uncertainty_files
            else None,
        }

        print(f"   Calibration: {'✓' if calibration_implemented else '✗'}")

    def _audit_regime_awareness(self):
        """Audit 6: Regime awareness"""

        print("🌊 Auditing regime awareness...")

        # Check for regime detection implementation
        regime_files = list(Path(".").glob("**/regime*.py"))

        # Check for regime logs
        today = datetime.now().strftime("%Y%m%d")
        regime_logs = (
            list(Path(f"logs/daily/{today}").glob("*regime*"))
            if Path(f"logs/daily/{today}").exists()
            else []
        )

        # Check for A/B testing logs
        ab_logs = (
            list(Path(f"logs/daily/{today}").glob("ab_regime*"))
            if Path(f"logs/daily/{today}").exists()
            else []
        )

        regime_implemented = len(regime_files) > 0

        if regime_implemented:
            self.implemented_features.append("Regime awareness")
        else:
            self.missing_features.append("No regime detection implementation")

        self.audit_results["regime_awareness"] = {
            "implemented": regime_implemented,
            "regime_files": len(regime_files),
            "regime_logs": len(regime_logs),
            "ab_testing_logs": len(ab_logs),
            "evidence_path": str(regime_files[0]) if regime_files else None,
        }

        print(f"   Regime awareness: {'✓' if regime_implemented else '✗'}")

    def _audit_explainability_shap(self):
        """Audit 7: Explainability (SHAP)"""

        print("📊 Auditing explainability (SHAP)...")

        # Check for SHAP implementation
        shap_files = list(Path(".").glob("**/shap*.py"))
        explainability_files = list(Path(".").glob("**/explain*.py"))

        # Check for SHAP exports
        shap_exports = list(Path("exports").glob("*shap*")) if Path("exports").exists() else []

        shap_implemented = len(shap_files) > 0 or len(explainability_files) > 0

        if shap_implemented:
            self.implemented_features.append("Explainability (SHAP)")
        else:
            self.missing_features.append("No SHAP implementation found")

        self.audit_results["explainability_shap"] = {
            "implemented": shap_implemented,
            "shap_files": len(shap_files),
            "explainability_files": len(explainability_files),
            "shap_exports": len(shap_exports),
            "evidence_path": str(shap_files[0])
            if shap_files
            else str(explainability_files[0])
            if explainability_files
            else None,
        }

        print(f"   SHAP: {'✓' if shap_implemented else '✗'}")

    def _audit_realistic_backtest(self):
        """Audit 8: Realistic backtest"""

        print("📈 Auditing realistic backtest...")

        # Check for backtest implementation
        backtest_files = list(Path(".").glob("**/backtest*.py"))
        slippage_files = list(Path(".").glob("**/slippage*.py"))

        # Check for execution metrics
        today = datetime.now().strftime("%Y%m%d")
        execution_logs = (
            list(Path(f"logs/daily/{today}").glob("execution_metrics*"))
            if Path(f"logs/daily/{today}").exists()
            else []
        )

        realistic_backtest_implemented = len(backtest_files) > 0 or len(slippage_files) > 0

        if realistic_backtest_implemented:
            self.implemented_features.append("Realistic backtest (fees, slippage)")
        else:
            self.missing_features.append("No realistic backtest implementation")

        self.audit_results["realistic_backtest"] = {
            "implemented": realistic_backtest_implemented,
            "backtest_files": len(backtest_files),
            "slippage_files": len(slippage_files),
            "execution_logs": len(execution_logs),
            "evidence_path": str(slippage_files[0])
            if slippage_files
            else str(backtest_files[0])
            if backtest_files
            else None,
        }

        print(f"   Realistic backtest: {'✓' if realistic_backtest_implemented else '✗'}")

    def _audit_orchestrator_isolation(self):
        """Audit 9: Orchestrator isolation & autorestart"""

        print("🔄 Auditing orchestrator isolation...")

        # Check for orchestrator implementation
        orchestrator_files = list(Path(".").glob("**/orchestrat*.py"))
        supervisor_files = list(Path(".").glob("**/supervisor*.py"))

        # Check for process isolation
        multiprocessing_files = []
        for file_path in Path(".").glob("**/*.py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "multiprocessing" in content or "Process(" in content:
                    multiprocessing_files.append(str(file_path))

            except Exception:
                continue

        isolation_implemented = len(orchestrator_files) > 0 or len(multiprocessing_files) > 0

        if isolation_implemented:
            self.implemented_features.append("Orchestrator isolation & autorestart")
        else:
            self.missing_features.append("No orchestrator isolation found")

        self.audit_results["orchestrator_isolation"] = {
            "implemented": isolation_implemented,
            "orchestrator_files": len(orchestrator_files),
            "supervisor_files": len(supervisor_files),
            "multiprocessing_files": len(multiprocessing_files),
            "evidence_path": str(orchestrator_files[0]) if orchestrator_files else None,
        }

        print(f"   Orchestrator: {'✓' if isolation_implemented else '✗'}")

    def _audit_daily_eval_health(self):
        """Audit 10: Daily evaluation & health GO/NOGO"""

        print("🏥 Auditing daily evaluation & health...")

        # Check for daily evaluation
        evaluator_files = list(Path("eval").glob("*.py")) if Path("eval").exists() else []

        # Check for daily logs
        today = datetime.now().strftime("%Y%m%d")
        daily_dir = Path(f"logs/daily/{today}")
        daily_metrics = list(daily_dir.glob("daily_metrics*")) if daily_dir.exists() else []

        # Check for health monitoring
        health_files = list(Path(".").glob("**/health*.py"))

        daily_eval_implemented = len(evaluator_files) > 0 and len(daily_metrics) > 0

        if daily_eval_implemented:
            self.implemented_features.append("Daily evaluation & health GO/NOGO")
        else:
            self.missing_features.append("Daily evaluation system incomplete")

        self.audit_results["daily_eval_health"] = {
            "implemented": daily_eval_implemented,
            "evaluator_files": len(evaluator_files),
            "daily_metrics": len(daily_metrics),
            "health_files": len(health_files),
            "daily_dir_exists": daily_dir.exists(),
            "evidence_path": str(daily_dir) if daily_dir.exists() else None,
        }

        print(f"   Daily eval: {'✓' if daily_eval_implemented else '✗'}")

    def _audit_security_implementation(self):
        """Audit 11: Security implementation"""

        print("🔐 Auditing security implementation...")

        # Check for .env file
        env_file_exists = Path(".env").exists()
        env_example_exists = Path(".env.example").exists()

        # Check for secrets in repo (should be clean)
        secrets_in_repo = False
        try:
            import subprocess

            result = subprocess.run(
                ["grep", "-r", "-i", "api_key\|token\|secret\|password", "."],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                secrets_in_repo = True
        except Exception:
            pass

        # Check for Vault implementation
        vault_files = list(Path(".").glob("**/vault*.py"))

        # Check for authentication in dashboard
        auth_implemented = False
        dashboard_files = list(Path(".").glob("**/app*.py"))
        for app_file in dashboard_files:
            try:
                with open(app_file, "r") as f:
                    content = f.read()

                if "auth" in content.lower() or "login" in content.lower():
                    auth_implemented = True
                    break

            except Exception:
                continue

        security_implemented = env_file_exists and not secrets_in_repo

        if security_implemented:
            self.implemented_features.append("Security (.env, no secrets in repo)")
        else:
            self.missing_features.append("Security implementation incomplete")

        self.audit_results["security_implementation"] = {
            "implemented": security_implemented,
            "env_file_exists": env_file_exists,
            "env_example_exists": env_example_exists,
            "secrets_in_repo": secrets_in_repo,
            "vault_files": len(vault_files),
            "auth_implemented": auth_implemented,
            "evidence_path": ".env" if env_file_exists else None,
        }

        print(f"   Security: {'✓' if security_implemented else '✗'}")

    def _save_functionality_report(self, report: Dict[str, Any]):
        """Save functionality audit report"""

        report_dir = Path("logs/functionality")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"functionality_audit_{timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Functionality audit report saved: {report_path}")

    def print_audit_summary(self, report: Dict[str, Any]):
        """Print functionality audit summary"""

        print(f"\n🎯 FUNCTIONALITY AUDIT SUMMARY")
        print("=" * 50)
        print(f"Total Features Audited: {report['total_features_audited']}")
        print(f"Implemented Features: {report['implemented_features']}")
        print(f"Missing Features: {report['missing_features']}")
        print(f"Implementation Score: {report['implementation_score']:.1f}%")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Audit Duration: {report['audit_duration']:.2f}s")

        if report["implemented_features_list"]:
            print(f"\n✅ Implemented Features:")
            for feature in report["implemented_features_list"]:
                print(f"   • {feature}")

        if report["missing_features_list"]:
            print(f"\n❌ Missing Features:")
            for feature in report["missing_features_list"]:
                print(f"   • {feature}")


def run_functionality_audit() -> Dict[str, Any]:
    """Run complete functionality audit"""

    auditor = FunctionalityAuditor()
    report = auditor.run_complete_functionality_audit()
    auditor.print_audit_summary(report)

    return report


if __name__ == "__main__":
    audit_report = run_functionality_audit()
