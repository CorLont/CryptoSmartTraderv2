#!/usr/bin/env python3
"""
Syntax Error Fixer
Focus on fixing the 155 syntax errors identified.
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntaxErrorFixer:
    """Fix syntax errors systematically."""

    def __init__(self):
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
            "*experiments*",
        ]

    def should_exclude(self, filepath: Path) -> bool:
        """Check if file should be excluded."""
        for pattern in self.exclude_patterns:
            if filepath.match(pattern) or any(part.startswith(".") for part in filepath.parts):
                return True
        return False

    def find_and_fix_syntax_errors(self) -> Dict:
        """Find and fix syntax errors."""
        project_root = Path(".")
        python_files = []

        for py_file in project_root.rglob("*.py"):
            if not self.should_exclude(py_file):
                python_files.append(py_file)

        syntax_errors = []
        fixed_files = []

        for file_path in python_files[:200]:  # Limit to first 200 files for main project
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for syntax
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append(
                        {"file": str(file_path), "line": e.lineno, "error": str(e)}
                    )

                    # Try common fixes
                    fixed_content = self.apply_common_fixes(content, str(e))
                    if fixed_content != content:
                        try:
                            ast.parse(fixed_content)
                            # Syntax is now valid, save the fix
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(fixed_content)
                            fixed_files.append(str(file_path))
                            logger.info(f"Fixed syntax error in {file_path}")
                        except SyntaxError:
                            # Fix didn't work, keep original
                            pass

            except Exception as e:
                logger.warning(f"Could not process {file_path}: {e}")

        return {
            "files_checked": len(python_files[:200]),
            "syntax_errors": syntax_errors,
            "fixed_files": fixed_files,
        }

    def apply_common_fixes(self, content: str, error_msg: str) -> str:
        """Apply common syntax fixes."""
        fixed_content = content

        # Fix 1: Missing quotes in f-strings
        if "f-string expression part cannot include a backslash" in error_msg:
            # Replace problematic f-string patterns
            fixed_content = re.sub(r'f"([^"]*){([^}]*\\[^}]*)}"', r'f"\1{repr(\2)}"', fixed_content)

        # Fix 2: Unmatched parentheses/brackets
        if "EOF while scanning" in error_msg or "unmatched" in error_msg:
            lines = fixed_content.split("\n")

            # Check for unmatched brackets/parens
            for i, line in enumerate(lines):
                open_parens = line.count("(") - line.count(")")
                open_brackets = line.count("[") - line.count("]")
                open_braces = line.count("{") - line.count("}")

                # If line has unmatched opening, try to close on next line
                if open_parens > 0 and i < len(lines) - 1:
                    if not lines[i + 1].strip():
                        lines[i] += ")" * open_parens

                if open_brackets > 0 and i < len(lines) - 1:
                    if not lines[i + 1].strip():
                        lines[i] += "]" * open_brackets

                if open_braces > 0 and i < len(lines) - 1:
                    if not lines[i + 1].strip():
                        lines[i] += "}" * open_braces

            fixed_content = "\n".join(lines)

        # Fix 3: Invalid escape sequences
        if "invalid escape sequence" in error_msg:
            # Fix common escape sequence issues
            fixed_content = re.sub(r'\\([^\\nr\'"tfbv])', r"\\\\\\1", fixed_content)

        # Fix 4: Missing colons after if/for/while/def/class
        if "invalid syntax" in error_msg:
            lines = fixed_content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(
                    (
                        "if ",
                        "for ",
                        "while ",
                        "def ",
                        "class ",
                        "try",
                        "except ",
                        "finally",
                        "with ",
                    )
                ) and not stripped.endswith(":"):
                    lines[i] = line + ":"
            fixed_content = "\n".join(lines)

        # Fix 5: Indentation errors
        if "IndentationError" in error_msg or "expected an indented block" in error_msg:
            lines = fixed_content.split("\n")
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    # Check if previous line ends with colon
                    if i > 0 and lines[i - 1].strip().endswith(":"):
                        lines[i] = "    " + line  # Add 4 spaces
            fixed_content = "\n".join(lines)

        return fixed_content


def main():
    """Main execution."""
    fixer = SyntaxErrorFixer()
    results = fixer.find_and_fix_syntax_errors()

    print("🔧 SYNTAX ERROR FIXING COMPLETE")
    print(f"Files checked: {results['files_checked']}")
    print(f"Syntax errors found: {len(results['syntax_errors'])}")
    print(f"Files fixed: {len(results['fixed_files'])}")

    if results["fixed_files"]:
        print("\n✅ Fixed files:")
        for file in results["fixed_files"]:
            print(f"  - {file}")

    if results["syntax_errors"]:
        print(f"\n⚠️  Remaining syntax errors: {len(results['syntax_errors'])}")
        for error in results["syntax_errors"][:10]:  # Show first 10
            print(f"  - {error['file']}:{error['line']} - {error['error']}")

    return results


if __name__ == "__main__":
    main()
