#!/usr/bin/env python3
"""
System Health Check - Production readiness validation
Fixed version with proper validation instead of string searches
"""

import os
import sys
import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple


class SystemHealthChecker:
    """System health checker with proper validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_tests = 0
        self.warnings = 0
        self.results = []

    def check_streamlit_components(self) -> bool:
        """Check Streamlit components using AST parsing instead of string search"""

        app_file = Path("app_fixed_all_issues.py")
        if not app_file.exists():
            self.failed_tests += 1
            self.results.append("❌ Main app file not found")
            return False

        try:
            with open(app_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST to find actual Streamlit usage
            tree = ast.parse(content)

            streamlit_imports = False
            streamlit_usage = False

            for node in ast.walk(tree):
                # Check for streamlit import
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "streamlit":
                            streamlit_imports = True
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "streamlit":
                        streamlit_imports = True

                # Check for actual streamlit usage (st.something calls)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "st":
                        streamlit_usage = True

            if streamlit_imports and streamlit_usage:
                self.results.append("✅ Streamlit components properly integrated")
                return True
            else:
                self.warnings += 1
                self.results.append("⚠️ Streamlit imports/usage incomplete")
                return False

        except Exception as e:
            self.failed_tests += 1
            self.results.append(f"❌ Failed to parse Streamlit components: {e}")
            return False

    def check_advanced_engines(self) -> bool:
        """Check advanced engines with proper import validation"""

        advanced_engines = [
            "core.ml_predictor_agent",
            "core.sentiment_analyzer",
            "core.technical_agent",
            "core.whale_detector_agent",
            "core.risk_manager",
        ]

        working_engines = 0

        for engine_module in advanced_engines:
            try:
                # Try actual import instead of string search
                module = importlib.import_module(engine_module)

                # Check for coordinator or main class
                has_coordinator = any(
                    hasattr(module, attr)
                    for attr in ["coordinator", "process", "analyze", "predict", "detect"]
                )

                if has_coordinator:
                    working_engines += 1
                    self.results.append(f"✅ {engine_module} operational")
                else:
                    self.warnings += 1
                    self.results.append(f"⚠️ {engine_module} missing coordinator functions")

            except ImportError as e:
                self.warnings += 1
                self.results.append(f"⚠️ {engine_module} not available: {e}")
            except Exception as e:
                self.failed_tests += 1
                self.results.append(f"❌ {engine_module} failed validation: {e}")

        # At least 60% of engines should work
        success_rate = working_engines / len(advanced_engines)
        if success_rate >= 0.6:
            self.results.append(
                f"✅ Advanced engines status: {working_engines}/{len(advanced_engines)} operational"
            )
            return True
        else:
            self.failed_tests += 1
            self.results.append(
                f"❌ Advanced engines insufficient: {working_engines}/{len(advanced_engines)} operational"
            )
            return False

    def check_production_files(self) -> bool:
        """Check critical production files exist"""

        critical_files = ["config.json", "logs/", "data/", "pyproject.toml"]

        missing_files = []

        for file_path in critical_files:
            path = Path(file_path)
            if not path.exists():
                missing_files.append(file_path)

        if missing_files:
            if len(missing_files) <= 1:
                self.warnings += 1
                self.results.append(f"⚠️ Missing files: {', '.join(missing_files)}")
            else:
                self.failed_tests += 1
                self.results.append(f"❌ Critical files missing: {', '.join(missing_files)}")
            return False
        else:
            self.results.append("✅ All critical production files present")
            return True

    def check_dependencies(self) -> bool:
        """Check critical dependencies are available"""

        critical_deps = [
            "streamlit",
            "pandas",
            "numpy",
            "plotly",
            "scikit-learn",
            "ccxt",
            "aiohttp",
            "pydantic",
        ]

        missing_deps = []

        for dep in critical_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            self.failed_tests += 1
            self.results.append(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            return False
        else:
            self.results.append("✅ All critical dependencies available")
            return True

    def run_complete_health_check(self) -> Dict[str, Any]:
        """Run complete health check with proper evaluation"""

        print("🏥 System Health Check Starting")
        print("=" * 50)

        # Reset counters
        self.failed_tests = 0
        self.warnings = 0
        self.results = []

        # Run all checks
        streamlit_ok = self.check_streamlit_components()
        engines_ok = self.check_advanced_engines()
        files_ok = self.check_production_files()
        deps_ok = self.check_dependencies()

        # Display results
        print("\n📋 Health Check Results:")
        for result in self.results:
            print(f"   {result}")

        # Proper evaluation considering warnings
        print(f"\n📊 Summary: {self.failed_tests} failed tests, {self.warnings} warnings")

        # Realistic end status evaluation
        ready = self.failed_tests == 0

        print(f"\n📊 Summary:")
        print(f"   Failed Tests: {self.failed_tests}")
        print(f"   Warnings: {self.warnings}")
        print(f"   Production Ready: {'Yes' if ready else 'No'}")

        # Fixed final assessment that includes warnings
        if ready and self.warnings == 0:
            print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
            status = "PRODUCTION_READY"
        elif ready and self.warnings > 0:
            print("\n⚠️ TESTS PASSED WITH WARNINGS - Review issues before production")
            status = "READY_WITH_WARNINGS"
        else:
            print("\n❌ SYSTEM NOT READY - Critical issues detected")
            status = "NOT_READY"

        return {
            "status": status,
            "failed_tests": self.failed_tests,
            "warnings": self.warnings,
            "production_ready": ready and self.warnings == 0,
            "results": self.results,
        }


def main():
    """Main health check execution"""
    checker = SystemHealthChecker()
    result = checker.run_complete_health_check()

    # Exit codes for automation
    if result["status"] == "PRODUCTION_READY":
        sys.exit(0)
    elif result["status"] == "READY_WITH_WARNINGS":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
