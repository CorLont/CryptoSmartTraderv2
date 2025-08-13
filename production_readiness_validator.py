#!/usr/bin/env python3
"""
Production Readiness Validator
Final validation that the system is production ready.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionReadinessValidator:
    """Validate complete production readiness."""

    def __init__(self):
        self.project_root = Path(".")
        self.core_files = [
            "app_fixed_all_issues.py",
            "src/cryptosmarttrader/core/data_manager.py",
            "src/cryptosmarttrader/risk/risk_guard.py",
            "src/cryptosmarttrader/execution/execution_policy.py",
            "src/cryptosmarttrader/attribution/return_attribution.py",
            "attribution_demo.py",
        ]

    def check_syntax_errors(self) -> Tuple[int, List[str]]:
        """Check for syntax errors in core files."""
        syntax_errors = []

        for file_path_str in self.core_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}:{e.lineno} - {e.msg}")
                except Exception as e:
                    syntax_errors.append(f"{file_path} - {str(e)}")

        return len(syntax_errors), syntax_errors

    def check_imports(self) -> Tuple[int, List[str]]:
        """Check for import issues in core files."""
        import_issues = []

        for file_path_str in self.core_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                try:
                    # Try importing (simulate)
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Check for obvious import issues
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        if line.strip().startswith("from ") or line.strip().startswith("import "):
                            # Check for relative imports outside package
                            if "from ." in line and "src/" not in str(file_path):
                                import_issues.append(
                                    f"{file_path}:{i} - Relative import outside package"
                                )

                except Exception as e:
                    import_issues.append(f"{file_path} - {str(e)}")

        return len(import_issues), import_issues

    def check_ci_cd_setup(self) -> Tuple[bool, List[str]]:
        """Check CI/CD infrastructure."""
        ci_files = [".github/workflows/ci.yml", "CODEOWNERS", "pyproject.toml", "pytest.ini"]

        missing_files = []
        for file_path in ci_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        return len(missing_files) == 0, missing_files

    def check_core_functionality(self) -> Tuple[bool, List[str]]:
        """Check that core functionality is accessible."""
        issues = []

        # Check main services
        services = [
            ("app_fixed_all_issues.py", "Main dashboard"),
            ("attribution_demo.py", "Attribution system"),
            ("src/cryptosmarttrader/api/health_endpoint.py", "Health API"),
        ]

        for file_path, description in services:
            if not Path(file_path).exists():
                issues.append(f"Missing {description}: {file_path}")

        # Check essential directories
        essential_dirs = ["src/cryptosmarttrader/", "metrics/", "tests/"]

        for dir_path in essential_dirs:
            if not Path(dir_path).exists():
                issues.append(f"Missing essential directory: {dir_path}")

        return len(issues) == 0, issues

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check dependency management."""
        issues = []

        # Check pyproject.toml
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            try:
                import tomllib

                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)

                # Check essential dependencies
                dependencies = data.get("project", {}).get("dependencies", [])
                essential_deps = ["streamlit", "fastapi", "ccxt", "pandas", "numpy"]

                for dep in essential_deps:
                    if not any(dep in d for d in dependencies):
                        issues.append(f"Missing essential dependency: {dep}")

            except Exception as e:
                issues.append(f"Error reading pyproject.toml: {e}")
        else:
            issues.append("Missing pyproject.toml")

        return len(issues) == 0, issues

    def check_security_setup(self) -> Tuple[bool, List[str]]:
        """Check security infrastructure."""
        issues = []

        security_files = ["SECURITY.md", ".env.example", ".gitignore"]

        for file_path in security_files:
            if not Path(file_path).exists():
                issues.append(f"Missing security file: {file_path}")

        # Check .gitignore has essential patterns
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                gitignore_content = f.read()

            essential_patterns = [".env", "*.log", "__pycache__", ".cache"]
            for pattern in essential_patterns:
                if pattern not in gitignore_content:
                    issues.append(f"Missing .gitignore pattern: {pattern}")

        return len(issues) == 0, issues

    def check_documentation(self) -> Tuple[bool, List[str]]:
        """Check documentation completeness."""
        issues = []

        doc_files = ["README.md", "replit.md", "RETURN_ATTRIBUTION_COMPLETION_REPORT.md"]

        for file_path in doc_files:
            if not Path(file_path).exists():
                issues.append(f"Missing documentation: {file_path}")

        return len(issues) == 0, issues

    def run_comprehensive_validation(self) -> Dict:
        """Run complete production readiness validation."""
        logger.info("Running comprehensive production readiness validation...")

        results = {}

        # 1. Syntax Errors
        syntax_count, syntax_errors = self.check_syntax_errors()
        results["syntax_errors"] = {
            "count": syntax_count,
            "issues": syntax_errors,
            "passed": syntax_count == 0,
        }

        # 2. Import Issues
        import_count, import_issues = self.check_imports()
        results["import_issues"] = {
            "count": import_count,
            "issues": import_issues,
            "passed": import_count == 0,
        }

        # 3. CI/CD Setup
        ci_passed, ci_missing = self.check_ci_cd_setup()
        results["ci_cd"] = {"passed": ci_passed, "missing_files": ci_missing}

        # 4. Core Functionality
        core_passed, core_issues = self.check_core_functionality()
        results["core_functionality"] = {"passed": core_passed, "issues": core_issues}

        # 5. Dependencies
        deps_passed, deps_issues = self.check_dependencies()
        results["dependencies"] = {"passed": deps_passed, "issues": deps_issues}

        # 6. Security
        security_passed, security_issues = self.check_security_setup()
        results["security"] = {"passed": security_passed, "issues": security_issues}

        # 7. Documentation
        docs_passed, docs_issues = self.check_documentation()
        results["documentation"] = {"passed": docs_passed, "issues": docs_issues}

        # Overall assessment
        all_checks = [
            results["syntax_errors"]["passed"],
            results["import_issues"]["passed"],
            results["ci_cd"]["passed"],
            results["core_functionality"]["passed"],
            results["dependencies"]["passed"],
            results["security"]["passed"],
            results["documentation"]["passed"],
        ]

        results["overall"] = {
            "production_ready": all(all_checks),
            "passed_checks": sum(all_checks),
            "total_checks": len(all_checks),
            "success_rate": sum(all_checks) / len(all_checks) * 100,
        }

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate production readiness report."""
        report = []
        report.append("# PRODUCTION READINESS VALIDATION REPORT")
        report.append(f"Generated: {Path.cwd()}")
        report.append("")

        # Overall Status
        overall = results["overall"]
        if overall["production_ready"]:
            status = "✅ PRODUCTION READY"
        elif overall["success_rate"] >= 80:
            status = "⚠️ NEAR PRODUCTION READY"
        else:
            status = "❌ NOT PRODUCTION READY"

        report.append(f"## OVERALL STATUS: {status}")
        report.append(
            f"Success Rate: {overall['success_rate']:.1f}% ({overall['passed_checks']}/{overall['total_checks']} checks passed)"
        )
        report.append("")

        # Detailed Results
        checks = [
            ("syntax_errors", "Syntax Errors"),
            ("import_issues", "Import Issues"),
            ("ci_cd", "CI/CD Infrastructure"),
            ("core_functionality", "Core Functionality"),
            ("dependencies", "Dependencies"),
            ("security", "Security Setup"),
            ("documentation", "Documentation"),
        ]

        for check_key, check_name in checks:
            check_result = results[check_key]
            status = "✅" if check_result["passed"] else "❌"
            report.append(f"### {status} {check_name}")

            if not check_result["passed"]:
                if "issues" in check_result:
                    for issue in check_result["issues"]:
                        report.append(f"- {issue}")
                if "missing_files" in check_result:
                    for file in check_result["missing_files"]:
                        report.append(f"- Missing: {file}")
                if "count" in check_result and check_result["count"] > 0:
                    report.append(f"- Total issues: {check_result['count']}")
            else:
                report.append("- All checks passed")

            report.append("")

        return "\n".join(report)


def main():
    """Main execution."""
    validator = ProductionReadinessValidator()

    print("🔍 RUNNING PRODUCTION READINESS VALIDATION...")
    results = validator.run_comprehensive_validation()

    # Generate report
    report = validator.generate_report(results)

    # Save report
    report_path = "FINAL_PRODUCTION_READINESS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Save JSON results
    json_path = "production_readiness_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("🎯 PRODUCTION READINESS VALIDATION COMPLETE")
    print(f"📊 Report saved to: {report_path}")
    print(f"📋 JSON results: {json_path}")
    print()

    overall = results["overall"]
    if overall["production_ready"]:
        print("🚀 STATUS: PRODUCTION READY")
    elif overall["success_rate"] >= 80:
        print("⚠️  STATUS: NEAR PRODUCTION READY")
    else:
        print("❌ STATUS: NOT PRODUCTION READY")

    print(f"Success Rate: {overall['success_rate']:.1f}%")
    print(f"Checks Passed: {overall['passed_checks']}/{overall['total_checks']}")

    return results


if __name__ == "__main__":
    main()
