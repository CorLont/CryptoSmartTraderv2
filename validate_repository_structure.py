#!/usr/bin/env python3
"""Validation script voor enterprise repository structuur en package layout."""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class RepositoryValidator:
    """Valideer enterprise repository structuur en configuratie."""

    def __init__(self):
        """Initialiseer validator."""
        self.root_path = Path(".")
        self.src_path = self.root_path / "src" / "cryptosmarttrader"
        self.results = {}

    def validate_package_structure(self) -> Dict[str, Any]:
        """Valideer clean package structuur."""
        print("🏗️  VALIDATING PACKAGE STRUCTURE")
        print("-" * 50)

        structure_results = {}

        # Check src/cryptosmarttrader/ exists
        if self.src_path.exists():
            print("✅ src/cryptosmarttrader/ directory exists")
            structure_results["main_package"] = True
        else:
            print("❌ src/cryptosmarttrader/ directory missing")
            structure_results["main_package"] = False
            return structure_results

        # Check key modules
        required_modules = ["core", "analysis", "deployment", "monitoring", "testing"]

        for module in required_modules:
            module_path = self.src_path / module
            if module_path.exists():
                print(f"✅ {module}/ module exists")
                structure_results[f"module_{module}"] = True
            else:
                print(f"❌ {module}/ module missing")
                structure_results[f"module_{module}"] = False

        # Check __init__.py files
        init_files = [
            self.src_path / "__init__.py",
            self.src_path / "core" / "__init__.py",
            self.src_path / "analysis" / "__init__.py",
        ]

        for init_file in init_files:
            if init_file.exists():
                print(f"✅ {init_file.relative_to(self.root_path)} exists")
                structure_results[f"init_{init_file.parent.name}"] = True
            else:
                print(f"❌ {init_file.relative_to(self.root_path)} missing")
                structure_results[f"init_{init_file.parent.name}"] = False

        print()
        return structure_results

    def validate_import_system(self) -> Dict[str, Any]:
        """Valideer package import functionaliteit."""
        print("📦 VALIDATING IMPORT SYSTEM")
        print("-" * 50)

        import_results = {}

        try:
            # Test main package import
            sys.path.insert(0, str(self.root_path / "src"))
            import cryptosmarttrader

            print("✅ Main package import successful")
            import_results["main_import"] = True

            # Test module availability
            available_modules = getattr(cryptosmarttrader, "__all__", [])
            print(f"✅ Available exports: {len(available_modules)} modules")
            import_results["exports_available"] = len(available_modules) > 0

            # Test specific imports
            test_imports = [
                "ConfigManager",
                "get_logger",
                "RiskGuard",
                "RegimeDetector",
                "StrategySwitcher",
                "BacktestParityAnalyzer",
            ]

            for import_name in test_imports:
                try:
                    module = getattr(cryptosmarttrader, import_name)
                    print(f"✅ {import_name} import successful")
                    import_results[f"import_{import_name}"] = True
                except AttributeError:
                    print(f"❌ {import_name} not available")
                    import_results[f"import_{import_name}"] = False

        except ImportError as e:
            print(f"❌ Package import failed: {e}")
            import_results["main_import"] = False

        print()
        return import_results

    def validate_configuration_files(self) -> Dict[str, Any]:
        """Valideer configuratie bestanden."""
        print("⚙️  VALIDATING CONFIGURATION FILES")
        print("-" * 50)

        config_results = {}

        # Check pyproject.toml
        pyproject_path = self.root_path / "pyproject.toml"
        if pyproject_path.exists():
            print("✅ pyproject.toml exists")
            config_results["pyproject_toml"] = True

            # Validate content
            content = pyproject_path.read_text()
            required_sections = [
                "[project]",
                "[build-system]",
                "[tool.pytest.ini_options]",
                "[tool.mypy]",
                "[dependency-groups]",
            ]

            for section in required_sections:
                if section in content:
                    print(f"✅ {section} section present")
                    config_results[f"section_{section[1:-1].replace('.', '_')}"] = True
                else:
                    print(f"❌ {section} section missing")
                    config_results[f"section_{section[1:-1].replace('.', '_')}"] = False
        else:
            print("❌ pyproject.toml missing")
            config_results["pyproject_toml"] = False

        # Check .env.example
        env_example_path = self.root_path / ".env.example"
        if env_example_path.exists():
            print("✅ .env.example exists")
            config_results["env_example"] = True

            content = env_example_path.read_text()
            required_vars = ["OPENAI_API_KEY", "KRAKEN_API_KEY", "DATABASE_URL", "ENVIRONMENT"]

            for var in required_vars:
                if var in content:
                    print(f"✅ {var} documented in .env.example")
                    config_results[f"env_var_{var}"] = True
                else:
                    print(f"❌ {var} missing from .env.example")
                    config_results[f"env_var_{var}"] = False
        else:
            print("❌ .env.example missing")
            config_results["env_example"] = False

        print()
        return config_results

    def validate_gitignore(self) -> Dict[str, Any]:
        """Valideer .gitignore configuratie."""
        print("🚫 VALIDATING ARTIFACT EXCLUSIONS")
        print("-" * 50)

        gitignore_results = {}

        gitignore_path = self.root_path / ".gitignore"
        if gitignore_path.exists():
            print("✅ .gitignore exists")
            gitignore_results["gitignore_exists"] = True

            content = gitignore_path.read_text()

            # Check critical exclusions
            critical_exclusions = [
                "logs/",
                "models/",
                "*.egg-info/",
                "__pycache__/",
                "exports/",
                "cache/",
                ".env",
            ]

            for exclusion in critical_exclusions:
                if exclusion in content:
                    print(f"✅ {exclusion} excluded")
                    gitignore_results[
                        f"exclude_{exclusion.replace('/', '_').replace('*', 'star')}"
                    ] = True
                else:
                    print(f"❌ {exclusion} not excluded")
                    gitignore_results[
                        f"exclude_{exclusion.replace('/', '_').replace('*', 'star')}"
                    ] = False
        else:
            print("❌ .gitignore missing")
            gitignore_results["gitignore_exists"] = False

        print()
        return gitignore_results

    def validate_development_tools(self) -> Dict[str, Any]:
        """Valideer development tools configuratie."""
        print("🔧 VALIDATING DEVELOPMENT TOOLS")
        print("-" * 50)

        tools_results = {}

        # Check if uv is available
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ uv package manager available: {result.stdout.strip()}")
                tools_results["uv_available"] = True
            else:
                print("❌ uv package manager not available")
                tools_results["uv_available"] = False
        except FileNotFoundError:
            print("❌ uv package manager not found")
            tools_results["uv_available"] = False

        # Check uv.lock
        uv_lock_path = self.root_path / "uv.lock"
        if uv_lock_path.exists():
            print("✅ uv.lock dependency lock file exists")
            tools_results["uv_lock"] = True
        else:
            print("❌ uv.lock dependency lock file missing")
            tools_results["uv_lock"] = False

        # Check tests directory
        tests_path = self.root_path / "tests"
        if tests_path.exists():
            print("✅ tests/ directory exists")
            tools_results["tests_dir"] = True

            # Count test files
            test_files = list(tests_path.glob("test_*.py"))
            print(f"✅ {len(test_files)} test files found")
            tools_results["test_files_count"] = len(test_files)
        else:
            print("❌ tests/ directory missing")
            tools_results["tests_dir"] = False

        print()
        return tools_results

    def validate_secrets_safety(self) -> Dict[str, Any]:
        """Valideer dat er geen secrets in de repository staan."""
        print("🔒 VALIDATING SECRETS SAFETY")
        print("-" * 50)

        secrets_results = {}

        # Check for .env files (should not exist)
        env_files = [".env", ".env.local", ".env.production"]
        has_real_env = False

        for env_file in env_files:
            env_path = self.root_path / env_file
            if env_path.exists():
                print(f"⚠️  {env_file} found (should not be in repo)")
                has_real_env = True
            else:
                print(f"✅ {env_file} correctly excluded")

        secrets_results["no_real_env_files"] = not has_real_env

        # Check for common secret patterns in tracked files
        secret_patterns = [
            "sk-",  # OpenAI API keys
            "xoxb-",  # Slack bot tokens
            "arn:aws:",  # AWS ARNs
            "AKIA",  # AWS access keys
        ]

        tracked_files = [".env.example", "pyproject.toml", "README.md"]
        secrets_found = False

        for file_name in tracked_files:
            file_path = self.root_path / file_name
            if file_path.exists():
                content = file_path.read_text()
                for pattern in secret_patterns:
                    if pattern in content and "example" not in content.lower():
                        print(f"⚠️  Potential secret pattern '{pattern}' found in {file_name}")
                        secrets_found = True

        if not secrets_found:
            print("✅ No secret patterns found in tracked files")

        secrets_results["no_secrets_in_tracked_files"] = not secrets_found

        print()
        return secrets_results

    def run_full_validation(self) -> Dict[str, Any]:
        """Voer volledige repository validatie uit."""
        print("🚀 CryptoSmartTrader V2 - Repository Structure Validation")
        print("=" * 80)
        print()

        # Run all validations
        self.results["package_structure"] = self.validate_package_structure()
        self.results["import_system"] = self.validate_import_system()
        self.results["configuration"] = self.validate_configuration_files()
        self.results["gitignore"] = self.validate_gitignore()
        self.results["development_tools"] = self.validate_development_tools()
        self.results["secrets_safety"] = self.validate_secrets_safety()

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self) -> None:
        """Genereer validatie samenvatting."""
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)

        total_checks = 0
        passed_checks = 0

        for category, results in self.results.items():
            category_passed = sum(1 for v in results.values() if v is True)
            category_total = len(results)

            total_checks += category_total
            passed_checks += category_passed

            success_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"

            print(
                f"{status} {category.replace('_', ' ').title()}: {category_passed}/{category_total} "
                f"({success_rate:.1f}%)"
            )

        overall_success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        overall_status = (
            "✅" if overall_success_rate >= 90 else "⚠️" if overall_success_rate >= 70 else "❌"
        )

        print(
            f"\n{overall_status} OVERALL: {passed_checks}/{total_checks} ({overall_success_rate:.1f}%)"
        )

        if overall_success_rate >= 90:
            print("\n🎉 ENTERPRISE REPOSITORY STRUCTURE: EXCELLENT")
            print("   Repository voldoet volledig aan enterprise standaarden!")
        elif overall_success_rate >= 70:
            print("\n⚠️  ENTERPRISE REPOSITORY STRUCTURE: GOOD")
            print("   Repository voldoet grotendeels aan enterprise standaarden.")
        else:
            print("\n❌ ENTERPRISE REPOSITORY STRUCTURE: NEEDS IMPROVEMENT")
            print("   Repository vereist aanpassingen voor enterprise standaarden.")

        print(f"\n✨ Validation completed - {passed_checks} checks passed")


def main():
    """Hoofdfunctie."""
    validator = RepositoryValidator()
    results = validator.run_full_validation()

    # Return exit code based on results
    total_checks = sum(len(category) for category in results.values())
    passed_checks = sum(
        sum(1 for v in category.values() if v is True) for category in results.values()

    success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

    return 0 if success_rate >= 90 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
