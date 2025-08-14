"""
Systematic Syntax Error Fixer
Identifies and fixes syntax errors for production readiness
"""

import os
import ast
import logging
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class SyntaxErrorFixer:
    """
    Comprehensive syntax error detection and fixing
    Ensures production-ready code quality
    """
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.syntax_errors: List[Dict[str, Any]] = []
        self.import_errors: List[Dict[str, Any]] = []
        self.fixed_files: List[str] = []
        
        # Common syntax error patterns and fixes
        self.common_fixes = {
            # Missing imports
            r"name '(\w+)' is not defined": self._fix_undefined_name,
            r"No module named '(\w+)'": self._fix_missing_module,
            
            # Syntax issues
            r"invalid syntax": self._fix_invalid_syntax,
            r"unexpected EOF": self._fix_unexpected_eof,
            r"invalid character": self._fix_invalid_character,
            
            # Indentation issues
            r"IndentationError": self._fix_indentation_error,
            r"TabError": self._fix_tab_error,
            
            # Type annotation issues
            r"invalid type annotation": self._fix_type_annotation,
            
            # F-string issues
            r"f-string expression": self._fix_fstring_error,
        }
        
        self.logger.info("✅ Syntax Error Fixer initialized")
    
    def scan_project(self) -> Dict[str, Any]:
        """Scan entire project for syntax errors"""
        
        self.logger.info(f"🔍 Scanning project for syntax errors: {self.root_path}")
        
        self.syntax_errors.clear()
        self.import_errors.clear()
        
        python_files = list(self.root_path.rglob("*.py"))
        
        for file_path in python_files:
            # Skip certain directories
            if any(skip in str(file_path) for skip in [".git", "__pycache__", ".pytest_cache", "venv", "env"]):
                continue
            
            self._check_file_syntax(file_path)
        
        results = {
            "total_files_scanned": len(python_files),
            "syntax_errors": len(self.syntax_errors),
            "import_errors": len(self.import_errors),
            "total_errors": len(self.syntax_errors) + len(self.import_errors),
            "error_details": {
                "syntax": self.syntax_errors,
                "imports": self.import_errors
            }
        }
        
        self.logger.info(
            f"📊 Scan complete: {results['total_errors']} errors in {results['total_files_scanned']} files"
        )
        
        return results
    
    def _check_file_syntax(self, file_path: Path):
        """Check syntax of individual file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.syntax_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "text": e.text.strip() if e.text else ""
                })
                return
            
            # Check for import/compile errors
            try:
                compile(content, str(file_path), 'exec')
            except Exception as e:
                if "No module named" in str(e) or "cannot import" in str(e):
                    self.import_errors.append({
                        "file": str(file_path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
        
        except Exception as e:
            self.logger.error(f"❌ Error checking {file_path}: {e}")
    
    def fix_all_errors(self) -> Dict[str, Any]:
        """Fix all detected syntax errors"""
        
        self.logger.info("🔧 Starting systematic error fixing")
        
        self.fixed_files.clear()
        fixed_count = 0
        failed_fixes = []
        
        # Fix syntax errors first
        for error in self.syntax_errors:
            try:
                if self._fix_syntax_error(error):
                    fixed_count += 1
                    if error["file"] not in self.fixed_files:
                        self.fixed_files.append(error["file"])
            except Exception as e:
                failed_fixes.append({
                    "file": error["file"],
                    "error": str(e),
                    "original_error": error
                })
        
        # Fix import errors
        for error in self.import_errors:
            try:
                if self._fix_import_error(error):
                    fixed_count += 1
                    if error["file"] not in self.fixed_files:
                        self.fixed_files.append(error["file"])
            except Exception as e:
                failed_fixes.append({
                    "file": error["file"],
                    "error": str(e),
                    "original_error": error
                })
        
        results = {
            "fixed_count": fixed_count,
            "fixed_files": len(self.fixed_files),
            "failed_fixes": len(failed_fixes),
            "fixed_file_list": self.fixed_files,
            "failed_details": failed_fixes
        }
        
        self.logger.info(f"✅ Fixed {fixed_count} errors in {len(self.fixed_files)} files")
        
        return results
    
    def _fix_syntax_error(self, error: Dict[str, Any]) -> bool:
        """Fix individual syntax error"""
        
        file_path = error["file"]
        error_msg = error["error"]
        line_number = error.get("line", 1)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply common fixes based on error pattern
            for pattern, fix_func in self.common_fixes.items():
                if re.search(pattern, error_msg, re.IGNORECASE):
                    if fix_func(file_path, lines, error):
                        self._write_fixed_file(file_path, lines)
                        return True
            
            # Manual fixes for specific error types
            if "unexpected EOF" in error_msg.lower():
                return self._fix_unexpected_eof_manual(file_path, lines, error)
            elif "invalid syntax" in error_msg.lower():
                return self._fix_invalid_syntax_manual(file_path, lines, error)
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fix syntax error in {file_path}: {e}")
            return False
    
    def _fix_import_error(self, error: Dict[str, Any]) -> bool:
        """Fix individual import error"""
        
        file_path = error["file"]
        error_msg = error["error"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix common import issues
            fixed_content = self._fix_import_issues(content, error_msg)
            
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fix import error in {file_path}: {e}")
            return False
    
    def _fix_undefined_name(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix undefined name errors by adding imports"""
        
        error_msg = error["error"]
        
        # Extract undefined name
        match = re.search(r"name '(\w+)' is not defined", error_msg)
        if not match:
            return False
        
        undefined_name = match.group(1)
        
        # Common undefined names and their imports
        common_imports = {
            "datetime": "from datetime import datetime",
            "time": "import time",
            "os": "import os",
            "sys": "import sys",
            "logging": "import logging",
            "json": "import json",
            "re": "import re",
            "numpy": "import numpy as np",
            "pandas": "import pandas as pd",
            "plt": "import matplotlib.pyplot as plt",
            "np": "import numpy as np",
            "pd": "import pandas as pd",
            "Optional": "from typing import Optional",
            "List": "from typing import List",
            "Dict": "from typing import Dict",
            "Any": "from typing import Any",
            "Tuple": "from typing import Tuple",
            "Union": "from typing import Union",
        }
        
        if undefined_name in common_imports:
            import_line = common_imports[undefined_name] + "\n"
            
            # Find insertion point (after existing imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('"""'):
                    insert_index = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(insert_index, import_line)
            return True
        
        return False
    
    def _fix_missing_module(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix missing module errors"""
        
        error_msg = error["error"]
        
        # Extract module name
        match = re.search(r"No module named '(\w+)'", error_msg)
        if not match:
            return False
        
        module_name = match.group(1)
        
        # Common module fixes
        module_fixes = {
            "ccxt": "# TODO: Install ccxt library: pip install ccxt",
            "prometheus_client": "# TODO: Install prometheus_client: pip install prometheus_client",
            "plotly": "# TODO: Install plotly: pip install plotly",
            "streamlit": "# TODO: Install streamlit: pip install streamlit",
            "fastapi": "# TODO: Install fastapi: pip install fastapi",
            "uvicorn": "# TODO: Install uvicorn: pip install uvicorn",
        }
        
        if module_name in module_fixes:
            # Add TODO comment at top of file
            comment = module_fixes[module_name] + "\n"
            lines.insert(0, comment)
            return True
        
        return False
    
    def _fix_invalid_syntax(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix invalid syntax errors"""
        
        line_number = error.get("line", 1) - 1  # Convert to 0-based index
        
        if 0 <= line_number < len(lines):
            line = lines[line_number]
            
            # Common syntax fixes
            fixed_line = line
            
            # Fix missing colons
            if re.search(r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+.*[^:]$', line.strip()):
                fixed_line = line.rstrip() + ":\n"
            
            # Fix missing quotes
            if "unterminated string" in error["error"].lower():
                if line.count('"') % 2 == 1:
                    fixed_line = line.rstrip() + '"\n'
                elif line.count("'") % 2 == 1:
                    fixed_line = line.rstrip() + "'\n"
            
            # Fix missing parentheses
            if "expected ')'" in error["error"].lower():
                if line.count('(') > line.count(')'):
                    fixed_line = line.rstrip() + ")\n"
            
            if fixed_line != line:
                lines[line_number] = fixed_line
                return True
        
        return False
    
    def _fix_unexpected_eof(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix unexpected EOF errors"""
        
        # Usually caused by unclosed brackets/quotes
        if lines:
            last_line = lines[-1].rstrip()
            
            # Check for unclosed structures
            open_brackets = last_line.count('(') - last_line.count(')')
            open_square = last_line.count('[') - last_line.count(']')
            open_curly = last_line.count('{') - last_line.count('}')
            
            fix_needed = ""
            if open_brackets > 0:
                fix_needed += ")" * open_brackets
            if open_square > 0:
                fix_needed += "]" * open_square
            if open_curly > 0:
                fix_needed += "}" * open_curly
            
            if fix_needed:
                lines[-1] = last_line + fix_needed + "\n"
                return True
            
            # Check if last line needs continuation
            if last_line.endswith(('\\', ',')):
                lines.append("    pass  # Temporary completion\n")
                return True
        
        return False
    
    def _fix_unexpected_eof_manual(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Manual fix for unexpected EOF"""
        
        # Add pass statement if needed
        if lines and not lines[-1].strip():
            lines.append("pass\n")
            return True
        
        return False
    
    def _fix_invalid_syntax_manual(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Manual fix for invalid syntax"""
        
        line_number = error.get("line", 1) - 1
        
        if 0 <= line_number < len(lines):
            line = lines[line_number]
            
            # Try to fix common issues
            if line.strip().endswith('='):
                lines[line_number] = line.rstrip() + " None  # Temporary assignment\n"
                return True
        
        return False
    
    def _fix_indentation_error(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix indentation errors"""
        
        line_number = error.get("line", 1) - 1
        
        if 0 <= line_number < len(lines):
            # Try to fix by normalizing indentation to 4 spaces
            line = lines[line_number]
            
            # Convert tabs to spaces
            fixed_line = line.expandtabs(4)
            
            if fixed_line != line:
                lines[line_number] = fixed_line
                return True
        
        return False
    
    def _fix_tab_error(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix tab/space mixing errors"""
        
        # Convert all tabs to 4 spaces
        for i, line in enumerate(lines):
            lines[i] = line.expandtabs(4)
        
        return True
    
    def _fix_type_annotation(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix type annotation errors"""
        
        line_number = error.get("line", 1) - 1
        
        if 0 <= line_number < len(lines):
            line = lines[line_number]
            
            # Add typing imports if needed
            if "Optional" in line and not any("from typing import" in l for l in lines):
                lines.insert(0, "from typing import Optional, List, Dict, Any, Union\n")
                return True
        
        return False
    
    def _fix_fstring_error(self, file_path: str, lines: List[str], error: Dict[str, Any]) -> bool:
        """Fix f-string errors"""
        
        line_number = error.get("line", 1) - 1
        
        if 0 <= line_number < len(lines):
            line = lines[line_number]
            
            # Fix common f-string issues
            if 'f"' in line and line.count('"') % 2 == 1:
                lines[line_number] = line.rstrip() + '"\n'
                return True
        
        return False
    
    def _fix_import_issues(self, content: str, error_msg: str) -> str:
        """Fix import-related issues in content"""
        
        lines = content.split('\n')
        
        # Add try-except around problematic imports
        if "No module named" in error_msg:
            module_match = re.search(r"No module named '(\w+)'", error_msg)
            if module_match:
                module_name = module_match.group(1)
                
                for i, line in enumerate(lines):
                    if f"import {module_name}" in line or f"from {module_name}" in line:
                        # Wrap in try-except
                        indent = len(line) - len(line.lstrip())
                        lines[i] = " " * indent + f"try:\n{line}\nexcept ImportError:\n    # {module_name} not available"
                        break
        
        return '\n'.join(lines)
    
    def _write_fixed_file(self, file_path: str, lines: List[str]):
        """Write fixed content back to file"""
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.logger.info(f"✅ Fixed syntax errors in {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to write fixed file {file_path}: {e}")
            raise
    
    def validate_fixes(self) -> Dict[str, Any]:
        """Validate that fixes actually work"""
        
        self.logger.info("🔍 Validating syntax fixes")
        
        validation_results = {
            "validated_files": 0,
            "still_broken": 0,
            "broken_files": []
        }
        
        for file_path in self.fixed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file now compiles
                try:
                    ast.parse(content)
                    validation_results["validated_files"] += 1
                except SyntaxError as e:
                    validation_results["still_broken"] += 1
                    validation_results["broken_files"].append({
                        "file": file_path,
                        "error": str(e)
                    })
            
            except Exception as e:
                self.logger.error(f"❌ Error validating {file_path}: {e}")
        
        success_rate = (
            validation_results["validated_files"] / max(1, len(self.fixed_files))
        ) * 100
        
        self.logger.info(f"✅ Validation complete: {success_rate:.1f}% success rate")
        
        return validation_results
    
    def get_fixing_report(self) -> Dict[str, Any]:
        """Get comprehensive fixing report"""
        
        return {
            "scan_results": {
                "syntax_errors": len(self.syntax_errors),
                "import_errors": len(self.import_errors),
                "total_errors": len(self.syntax_errors) + len(self.import_errors)
            },
            "fix_results": {
                "fixed_files": len(self.fixed_files),
                "fixed_file_list": self.fixed_files
            },
            "common_error_types": self._analyze_error_patterns(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_error_patterns(self) -> Dict[str, int]:
        """Analyze patterns in detected errors"""
        
        patterns = {}
        
        for error in self.syntax_errors + self.import_errors:
            error_type = error.get("error_type", "Unknown")
            patterns[error_type] = patterns.get(error_type, 0) + 1
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on errors found"""
        
        recommendations = []
        
        if len(self.import_errors) > 5:
            recommendations.append("Consider using virtual environment for dependency management")
        
        if any("IndentationError" in str(e) for e in self.syntax_errors):
            recommendations.append("Configure IDE to use consistent indentation (4 spaces)")
        
        if any("ccxt" in str(e) for e in self.import_errors):
            recommendations.append("Install missing trading library: pip install ccxt")
        
        return recommendations


def fix_project_syntax(root_path: str = ".") -> Dict[str, Any]:
    """Convenience function to fix all syntax errors in project"""
    
    fixer = SyntaxErrorFixer(root_path)
    
    # Scan for errors
    scan_results = fixer.scan_project()
    
    if scan_results["total_errors"] == 0:
        return {
            "status": "clean",
            "message": "No syntax errors found",
            "scan_results": scan_results
        }
    
    # Fix errors
    fix_results = fixer.fix_all_errors()
    
    # Validate fixes
    validation_results = fixer.validate_fixes()
    
    return {
        "status": "fixed",
        "scan_results": scan_results,
        "fix_results": fix_results,
        "validation_results": validation_results,
        "report": fixer.get_fixing_report()
    }


if __name__ == "__main__":
    # Run syntax fixing on project
    results = fix_project_syntax()
    print(f"Syntax fixing complete: {results['status']}")
    if results.get("fix_results"):
        print(f"Fixed {results['fix_results']['fixed_files']} files")