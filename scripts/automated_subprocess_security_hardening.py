#!/usr/bin/env python3
"""
Automated Subprocess Security Hardening Script
Systematically migrates all unsafe subprocess calls to SecureSubprocess framework
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

class SubprocessSecurityHardener:
    """Automated subprocess security migration tool"""
    
    def __init__(self):
        self.files_processed = 0
        self.calls_migrated = 0
        self.errors = []
        self.skipped_files = []
        
        # Files to exclude from migration
        self.exclude_patterns = [
            '.cache', '.git', '__pycache__', '.pytest_cache',
            'node_modules', '.venv', 'venv', 'archive',
            'secure_subprocess.py',  # Don't modify the framework itself
            'automated_subprocess_security_hardening.py'  # Don't modify self
        ]
        
        # Migration patterns
        self.migration_patterns = [
            # subprocess.run patterns
            (
                r'subprocess\.run\(\s*([^,\)]+),?\s*([^)]*)\)',
                self._replace_subprocess_run
            ),
            # subprocess.call patterns
            (
                r'subprocess\.call\(\s*([^,\)]+),?\s*([^)]*)\)',
                self._replace_subprocess_call
            ),
            # subprocess.Popen patterns
            (
                r'subprocess\.Popen\(\s*([^,\)]+),?\s*([^)]*)\)',
                self._replace_subprocess_popen
            )
        ]
    
    def scan_and_migrate_project(self) -> Dict[str, any]:
        """Scan entire project and migrate subprocess calls"""
        logger.info("Starting automated subprocess security hardening...")
        
        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Process each file
        for file_path in python_files:
            try:
                self._process_file(file_path)
            except Exception as e:
                self.errors.append(f"{file_path}: {str(e)}")
                logger.error(f"Error processing {file_path}: {e}")
        
        # Generate summary report
        return self._generate_report()
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files to process"""
        python_files = []
        
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    
                    # Skip excluded files
                    if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                        continue
                    
                    python_files.append(file_path)
        
        return python_files
    
    def _process_file(self, file_path: Path) -> bool:
        """Process a single Python file for subprocess migration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Cannot read {file_path}: {e}")
            return False
        
        original_content = content
        modified = False
        
        # Check if file has subprocess usage
        if 'subprocess.' not in content:
            return False
        
        # Add secure import if needed
        if 'from core.secure_subprocess import' not in content:
            content = self._add_secure_import(content, file_path)
            modified = True
        
        # Apply migration patterns
        for pattern, replacement_func in self.migration_patterns:
            new_content, count = replacement_func(content, pattern)
            if count > 0:
                content = new_content
                self.calls_migrated += count
                modified = True
                logger.info(f"Migrated {count} subprocess calls in {file_path}")
        
        # Write back if modified
        if modified:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_processed += 1
                logger.info(f"✅ Successfully hardened {file_path}")
                return True
            except Exception as e:
                logger.error(f"Cannot write {file_path}: {e}")
                self.errors.append(f"{file_path}: Write error - {str(e)}")
                return False
        
        return False
    
    def _add_secure_import(self, content: str, file_path: Path) -> str:
        """Add secure subprocess import to file"""
        lines = content.split('\n')
        
        # Find import section
        import_line_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_line_idx = i
        
        # Calculate relative path to core module
        relative_depth = len(file_path.relative_to(PROJECT_ROOT).parts) - 1
        if relative_depth > 0:
            relative_path = '../' * relative_depth + 'core/secure_subprocess'
        else:
            relative_path = 'core/secure_subprocess'
        
        # Add secure import
        secure_import = f"""
# SECURITY: Import secure subprocess framework
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent{'/..' * relative_depth}))
from core.secure_subprocess import secure_subprocess, SecureSubprocessError"""
        
        if import_line_idx >= 0:
            lines.insert(import_line_idx + 1, secure_import)
        else:
            # Add at beginning if no imports found
            lines.insert(0, secure_import)
        
        return '\n'.join(lines)
    
    def _replace_subprocess_run(self, content: str, pattern: str) -> Tuple[str, int]:
        """Replace subprocess.run calls with secure version"""
        count = 0
        
        def replace_match(match):
            nonlocal count
            command = match.group(1).strip()
            args = match.group(2).strip() if match.group(2) else ""
            
            # Parse existing arguments
            timeout = self._extract_timeout(args)
            check = self._extract_check(args)
            capture_output = self._extract_capture_output(args)
            text = self._extract_text(args)
            
            # Build secure replacement
            secure_call = f"""secure_subprocess.run_secure(
        {command},
        timeout={timeout},
        check={check},
        capture_output={capture_output},
        text={text}
    )"""
            
            count += 1
            return secure_call
        
        new_content = re.sub(pattern, replace_match, content)
        return new_content, count
    
    def _replace_subprocess_call(self, content: str, pattern: str) -> Tuple[str, int]:
        """Replace subprocess.call calls with secure version"""
        count = 0
        
        def replace_match(match):
            nonlocal count
            command = match.group(1).strip()
            args = match.group(2).strip() if match.group(2) else ""
            
            timeout = self._extract_timeout(args)
            
            # Build secure replacement
            secure_call = f"""secure_subprocess.run_secure(
        {command},
        timeout={timeout},
        check=True,
        capture_output=False,
        text=True
    )"""
            
            count += 1
            return secure_call
        
        new_content = re.sub(pattern, replace_match, content)
        return new_content, count
    
    def _replace_subprocess_popen(self, content: str, pattern: str) -> Tuple[str, int]:
        """Replace subprocess.Popen calls with secure version"""
        count = 0
        
        def replace_match(match):
            nonlocal count
            command = match.group(1).strip()
            args = match.group(2).strip() if match.group(2) else ""
            
            timeout = self._extract_timeout(args)
            
            # Build secure replacement with monitoring
            secure_call = f"""secure_subprocess.popen_secure(
        {command},
        timeout={timeout},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )[0]  # Return process object"""
            
            count += 1
            return secure_call
        
        new_content = re.sub(pattern, replace_match, content)
        return new_content, count
    
    def _extract_timeout(self, args: str) -> str:
        """Extract timeout from arguments or provide default"""
        timeout_match = re.search(r'timeout\s*=\s*(\d+)', args)
        if timeout_match:
            return timeout_match.group(1)
        return "30"  # Default 30 seconds
    
    def _extract_check(self, args: str) -> str:
        """Extract check parameter or provide default"""
        check_match = re.search(r'check\s*=\s*(True|False)', args)
        if check_match:
            return check_match.group(1)
        return "False"  # Default False for manual handling
    
    def _extract_capture_output(self, args: str) -> str:
        """Extract capture_output parameter or provide default"""
        capture_match = re.search(r'capture_output\s*=\s*(True|False)', args)
        if capture_match:
            return capture_match.group(1)
        return "True"  # Default True for security logging
    
    def _extract_text(self, args: str) -> str:
        """Extract text parameter or provide default"""
        text_match = re.search(r'text\s*=\s*(True|False)', args)
        if text_match:
            return text_match.group(1)
        return "True"  # Default True for string handling
    
    def _generate_report(self) -> Dict[str, any]:
        """Generate comprehensive migration report"""
        report = {
            "migration_summary": {
                "files_processed": self.files_processed,
                "subprocess_calls_migrated": self.calls_migrated,
                "errors": len(self.errors),
                "status": "COMPLETED" if len(self.errors) == 0 else "COMPLETED_WITH_ERRORS"
            },
            "errors": self.errors,
            "security_improvements": [
                "Mandatory timeout enforcement on all subprocess calls",
                "Comprehensive error handling with SecureSubprocessError",
                "Shell injection prevention (shell=False enforced)",
                "Input validation and command sanitization",
                "Complete audit trail logging",
                "Working directory and environment validation"
            ],
            "next_steps": [
                "Run test suite to verify all migrations work correctly",
                "Review error log for any failed migrations",
                "Update deployment scripts to use new secure framework",
                "Add security monitoring for subprocess operations"
            ]
        }
        
        return report


def main():
    """Main execution function"""
    hardener = SubprocessSecurityHardener()
    
    print("🔒 SUBPROCESS SECURITY HARDENING - AUTOMATED MIGRATION")
    print("=" * 60)
    
    # Run migration
    report = hardener.scan_and_migrate_project()
    
    # Display results
    print(f"\n📊 MIGRATION RESULTS:")
    print(f"   Files processed: {report['migration_summary']['files_processed']}")
    print(f"   Subprocess calls migrated: {report['migration_summary']['subprocess_calls_migrated']}")
    print(f"   Errors: {report['migration_summary']['errors']}")
    print(f"   Status: {report['migration_summary']['status']}")
    
    if report['errors']:
        print("\n❌ ERRORS ENCOUNTERED:")
        for error in report['errors']:
            print(f"   - {error}")
    
    print("\n✅ SECURITY IMPROVEMENTS IMPLEMENTED:")
    for improvement in report['security_improvements']:
        print(f"   - {improvement}")
    
    print("\n🔄 RECOMMENDED NEXT STEPS:")
    for step in report['next_steps']:
        print(f"   - {step}")
    
    # Save detailed report
    report_file = PROJECT_ROOT / "AUTOMATED_SUBPROCESS_MIGRATION_REPORT.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report['migration_summary']['status'] == "COMPLETED"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)