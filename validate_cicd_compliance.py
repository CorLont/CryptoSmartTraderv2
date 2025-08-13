#!/usr/bin/env python3
"""Validate CI/CD compliance with enterprise requirements."""

import yaml
from pathlib import Path


def validate_cicd_compliance():
    """Validate all CI/CD requirements are met."""
    print("🔍 Validating CI/CD Enterprise Compliance")
    print("=" * 50)

    compliance_checks = {}

    # 1. GitHub Actions with UV
    print("\n1. 📦 UV Integration")
    ci_file = Path(".github/workflows/ci.yml")
    if ci_file.exists():
        ci_content = yaml.safe_load(ci_file.read_text())

        # Check UV usage
        uv_sync_found = any(
            "uv sync --frozen" in str(step.get("run", ""))
            for job in ci_content.get("jobs", {}).values()
            for step in job.get("steps", [])
        )
        compliance_checks["uv_sync_frozen"] = uv_sync_found
        print(f"   {'✅' if uv_sync_found else '❌'} uv sync --frozen usage")

        # Check UV setup action
        uv_setup_found = any(
            "astral-sh/setup-uv@v4" in step.get("uses", "")
            for job in ci_content.get("jobs", {}).values()
            for step in job.get("steps", [])
        )
        compliance_checks["uv_setup"] = uv_setup_found
        print(f"   {'✅' if uv_setup_found else '❌'} astral-sh/setup-uv@v4 action")

    # 2. Actions Up-to-Date
    print("\n2. 🔄 Actions Versions")
    actions_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/release.yml",
    ]

    up_to_date_actions = True
    for file_path in actions_files:
        if Path(file_path).exists():
            content = Path(file_path).read_text()

            # Check for up-to-date actions
            required_actions = {
                "actions/checkout@v4": "@v4" in content and "checkout" in content,
                "actions/setup-python@v5": "@v5" in content and "setup-python" in content,
                "actions/upload-artifact@v4": "@v4" in content and "upload-artifact" in content,
                "actions/download-artifact@v4": "@v4" in content and "download-artifact" in content,
            }

            for action, found in required_actions.items():
                if action.split("@")[0] in content:  # Action is used
                    if not found:
                        up_to_date_actions = False
                        print(f"   ❌ {action} - outdated version found")
                    else:
                        print(f"   ✅ {action}")

    compliance_checks["actions_up_to_date"] = up_to_date_actions

    # 3. Python Matrix Testing
    print("\n3. 🐍 Python Matrix")
    if ci_file.exists():
        ci_content = yaml.safe_load(ci_file.read_text())

        matrix_found = False
        python_versions = []

        for job in ci_content.get("jobs", {}).values():
            strategy = job.get("strategy", {})
            if "matrix" in strategy:
                matrix = strategy["matrix"]
                if "python-version" in matrix:
                    python_versions = matrix["python-version"]
                    matrix_found = True

        has_311 = "3.11" in python_versions
        has_312 = "3.12" in python_versions

        compliance_checks["python_matrix"] = matrix_found and has_311 and has_312
        print(f"   {'✅' if matrix_found else '❌'} Matrix strategy configured")
        print(f"   {'✅' if has_311 else '❌'} Python 3.11 in matrix")
        print(f"   {'✅' if has_312 else '❌'} Python 3.12 in matrix")

    # 4. Coverage Gates
    print("\n4. 📊 Coverage Gates")
    coverage_gate_found = False

    if ci_file.exists():
        content = ci_file.read_text()
        if "--cov-fail-under=70" in content or "--fail-under" in content:
            coverage_gate_found = True

    # Also check pyproject.toml
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        content = pyproject_file.read_text()
        if "fail_under = 70" in content or "--cov-fail-under=70" in content:
            coverage_gate_found = True

    compliance_checks["coverage_gates"] = coverage_gate_found
    print(f"   {'✅' if coverage_gate_found else '❌'} Coverage fail-under gates")

    # 5. Security Scanning
    print("\n5. 🔒 Security Scanning")
    security_file = Path(".github/workflows/security.yml")
    security_checks = {"gitleaks": False, "pip_audit": False, "osv_scanner": False, "bandit": False}

    if security_file.exists():
        content = security_file.read_text()
        security_checks["gitleaks"] = "gitleaks" in content.lower()
        security_checks["pip_audit"] = "pip-audit" in content
        security_checks["osv_scanner"] = "osv-scanner" in content
        security_checks["bandit"] = "bandit" in content

    # Also check main CI file
    if ci_file.exists():
        content = ci_file.read_text()
        if "gitleaks" in content.lower():
            security_checks["gitleaks"] = True
        if "pip-audit" in content:
            security_checks["pip_audit"] = True
        if "bandit" in content:
            security_checks["bandit"] = True

    for check, passed in security_checks.items():
        print(f"   {'✅' if passed else '❌'} {check.replace('_', '-')} scanning")

    compliance_checks["security_scanning"] = all(security_checks.values())

    # 6. Branch Protection
    print("\n6. 🛡️ Branch Protection")
    codeowners_file = Path(".github/CODEOWNERS")
    protection_script = Path(".github/workflows/branch-protection.yml")

    codeowners_exists = codeowners_file.exists()
    protection_automation = protection_script.exists()

    compliance_checks["codeowners"] = codeowners_exists
    compliance_checks["branch_protection"] = protection_automation

    print(f"   {'✅' if codeowners_exists else '❌'} CODEOWNERS file")
    print(f"   {'✅' if protection_automation else '❌'} Branch protection automation")

    # 7. Cache Configuration
    print("\n7. 💾 Dependency Caching")
    cache_found = False

    if ci_file.exists():
        content = ci_file.read_text()
        if "cache" in content.lower() and "uv" in content:
            cache_found = True

    compliance_checks["dependency_caching"] = cache_found
    print(f"   {'✅' if cache_found else '❌'} UV dependency caching")

    # Summary
    print("\n" + "=" * 50)
    print("📊 COMPLIANCE SUMMARY")
    print("=" * 50)

    total_checks = len(compliance_checks)
    passed_checks = sum(1 for passed in compliance_checks.values() if passed)
    compliance_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    for check, passed in compliance_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check.replace('_', ' ').title()}")

    print(f"\n🎯 OVERALL COMPLIANCE: {passed_checks}/{total_checks} ({compliance_rate:.1f}%)")

    if compliance_rate >= 90:
        print("🎉 EXCELLENT - Enterprise CI/CD standards fully met")
        return 0
    elif compliance_rate >= 80:
        print("✅ GOOD - Most requirements met, minor improvements needed")
        return 1
    else:
        print("⚠️ NEEDS WORK - Significant compliance gaps")
        return 2


if __name__ == "__main__":
    exit_code = validate_cicd_compliance()
    exit(exit_code)
