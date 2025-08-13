#!/usr/bin/env python3
"""
Production Readiness Fixer
Systematically fix all production readiness issues.
"""

import os
import re
import glob
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionReadinessFixer:
    """Fix all production readiness issues systematically."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues_found = {
            "syntax_errors": [],
            "bare_excepts": [],
            "module_duplicates": [],
            "import_issues": [],
            "code_quality": [],
        }

        # Exclude patterns
        self.exclude_patterns = [
            "*.git*",
            "*__pycache__*",
            "*.cache*",
            "*venv*",
            "*node_modules*",
            "*mlartifacts*",
            "*models*",
            "*exports*",
            "*logs*",
            "*.pyc",
            "*dist*",
            "*build*",
            "*.egg-info*",
        ]

    def should_exclude(self, filepath: Path) -> bool:
        """Check if file should be excluded."""
        for pattern in self.exclude_patterns:
            if filepath.match(pattern) or any(part.startswith(".") for part in filepath.parts):
                return True
        return False

    def find_python_files(self) -> List[Path]:
        """Find all Python files to analyze."""
        files = []
        for py_file in self.project_root.rglob("*.py"):
            if not self.should_exclude(py_file):
                files.append(py_file)
        return files

    def check_syntax_errors(self, files: List[Path]) -> List[Dict]:
        """Check for syntax errors in Python files."""
        syntax_errors = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Try to compile
                ast.parse(content)

            except SyntaxError as e:
                syntax_errors.append(
                    {
                        "file": str(file_path),
                        "line": e.lineno,
                        "error": str(e),
                        "text": e.text.strip() if e.text else None,
                    }
                )
            except UnicodeDecodeError:
                logger.warning(f"Could not decode {file_path}")
            except Exception as e:
                logger.warning(f"Error checking {file_path}: {e}")

        return syntax_errors

    def find_bare_excepts(self, files: List[Path]) -> List[Dict]:
        """Find bare except statements."""
        bare_excepts = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    # Check for bare except
                    if re.search(r"except\s*:", line):
                        bare_excepts.append(
                            {"file": str(file_path), "line": i, "content": line.strip()}
                        )

            except Exception as e:
                logger.warning(f"Error checking {file_path}: {e}")

        return bare_excepts

    def fix_bare_excepts(self, files: List[Path]) -> int:
        """Fix bare except statements."""
        fixes_made = 0

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace bare excepts with Exception
                original_content = content
                content = re.sub(r"except\s*:", "except Exception:", content)

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixes_made += 1
                    logger.info(f"Fixed bare excepts in {file_path}")

            except Exception as e:
                logger.warning(f"Error fixing {file_path}: {e}")

        return fixes_made

    def find_duplicate_modules(self, files: List[Path]) -> List[Dict]:
        """Find duplicate module definitions."""
        module_names = {}
        duplicates = []

        for file_path in files:
            # Skip __init__.py files
            if file_path.name == "__init__.py":
                continue

            module_name = file_path.stem

            if module_name in module_names:
                duplicates.append(
                    {"module": module_name, "files": [module_names[module_name], str(file_path)]}
                )
            else:
                module_names[module_name] = str(file_path)

        return duplicates

    def analyze_import_issues(self, files: List[Path]) -> List[Dict]:
        """Analyze import issues."""
        import_issues = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Look for relative imports that might cause issues
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if re.search(r"from\s+\.\.?\.?\w+\s+import", line):
                        import_issues.append(
                            {
                                "file": str(file_path),
                                "line": i,
                                "import": line.strip(),
                                "issue": "relative_import",
                            }
                        )

            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        return import_issues

    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive production readiness analysis."""
        logger.info("Starting comprehensive production readiness analysis...")

        # Find all Python files
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")

        # Check syntax errors
        logger.info("Checking syntax errors...")
        syntax_errors = self.check_syntax_errors(python_files)

        # Find bare excepts
        logger.info("Finding bare except statements...")
        bare_excepts = self.find_bare_excepts(python_files)

        # Find duplicate modules
        logger.info("Finding duplicate modules...")
        duplicates = self.find_duplicate_modules(python_files)

        # Analyze imports
        logger.info("Analyzing import issues...")
        import_issues = self.analyze_import_issues(python_files)

        results = {
            "total_files": len(python_files),
            "syntax_errors": syntax_errors,
            "bare_excepts": bare_excepts,
            "duplicates": duplicates,
            "import_issues": import_issues,
        }

        return results

    def fix_issues(self, analysis: Dict) -> Dict:
        """Fix identified issues."""
        logger.info("Starting to fix issues...")

        fixes = {"bare_excepts_fixed": 0, "files_processed": 0}

        # Fix bare excepts
        python_files = self.find_python_files()
        fixes["bare_excepts_fixed"] = self.fix_bare_excepts(python_files)
        fixes["files_processed"] = len(python_files)

        return fixes

    def generate_report(self, analysis: Dict, fixes: Dict = None) -> str:
        """Generate production readiness report."""
        report = []
        report.append("# PRODUCTION READINESS ANALYSIS REPORT")
        report.append(f"Generated: {Path.cwd()}")
        report.append("")

        # Summary
        report.append("## SUMMARY")
        report.append(f"Total Python files analyzed: {analysis['total_files']}")
        report.append(f"Syntax errors found: {len(analysis['syntax_errors'])}")
        report.append(f"Bare except statements: {len(analysis['bare_excepts'])}")
        report.append(f"Duplicate modules: {len(analysis['duplicates'])}")
        report.append(f"Import issues: {len(analysis['import_issues'])}")
        report.append("")

        # Syntax Errors
        if analysis["syntax_errors"]:
            report.append("## SYNTAX ERRORS")
            for error in analysis["syntax_errors"]:
                report.append(f"- {error['file']}:{error['line']} - {error['error']}")
            report.append("")

        # Bare Excepts
        if analysis["bare_excepts"]:
            report.append("## BARE EXCEPT STATEMENTS")
            file_counts = {}
            for bare in analysis["bare_excepts"]:
                file_path = bare["file"]
                if file_path not in file_counts:
                    file_counts[file_path] = 0
                file_counts[file_path] += 1

            for file_path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- {file_path}: {count} bare excepts")
            report.append("")

        # Duplicates
        if analysis["duplicates"]:
            report.append("## DUPLICATE MODULES")
            for dup in analysis["duplicates"]:
                report.append(f"- {dup['module']}: {', '.join(dup['files'])}")
            report.append("")

        # Import Issues
        if analysis["import_issues"]:
            report.append("## IMPORT ISSUES")
            for issue in analysis["import_issues"]:
                report.append(f"- {issue['file']}:{issue['line']} - {issue['import']}")
            report.append("")

        # Fixes Applied
        if fixes:
            report.append("## FIXES APPLIED")
            report.append(f"Bare excepts fixed: {fixes['bare_excepts_fixed']} files")
            report.append(f"Files processed: {fixes['files_processed']}")
            report.append("")

        # Production Readiness Status
        total_issues = (
            len(analysis["syntax_errors"])
            + len(analysis["bare_excepts"])
            + len(analysis["duplicates"])
        )

        report.append("## PRODUCTION READINESS STATUS")
        if total_issues == 0:
            report.append("✅ PRODUCTION READY - No critical issues found")
        elif total_issues <= 10:
            report.append("⚠️  NEAR PRODUCTION READY - Minor issues to address")
        else:
            report.append("❌ NOT PRODUCTION READY - Critical issues need resolution")

        report.append(f"Critical issues remaining: {total_issues}")
        report.append("")

        return "\n".join(report)


def main():
    """Main execution."""
    fixer = ProductionReadinessFixer()

    # Run analysis
    analysis = fixer.run_comprehensive_analysis()

    # Apply fixes
    fixes = fixer.fix_issues(analysis)

    # Generate report
    report = fixer.generate_report(analysis, fixes)

    # Save report
    report_path = "PRODUCTION_READINESS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print("🔍 PRODUCTION READINESS ANALYSIS COMPLETE")
    print(f"📊 Report saved to: {report_path}")
    print()
    print("📋 SUMMARY:")
    print(f"   Files analyzed: {analysis['total_files']}")
    print(f"   Syntax errors: {len(analysis['syntax_errors'])}")
    print(f"   Bare excepts: {len(analysis['bare_excepts'])}")
    print(f"   Duplicates: {len(analysis['duplicates'])}")
    print(f"   Fixes applied: {fixes['bare_excepts_fixed']} files")

    return analysis, fixes


if __name__ == "__main__":
    main()
