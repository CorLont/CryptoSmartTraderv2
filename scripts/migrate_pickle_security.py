#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Pickle Security Migration Script
Systematically migrate all pickle usage to secure alternatives
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cryptosmarttrader.security.secure_serialization import SecureSerializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PickleMigrator:
    """Migrate pickle usage to secure alternatives"""
    
    def __init__(self):
        self.serializer = SecureSerializer()
        self.migration_report = {
            'files_scanned': 0,
            'files_modified': 0,
            'pickle_imports_removed': 0,
            'pickle_calls_replaced': 0,
            'errors': [],
            'warnings': []
        }
        
        # Patterns to identify pickle usage
        self.pickle_patterns = {
            'import_pickle': r'import\s+pickle',
            'from_pickle': r'from\s+pickle\s+import',
            'pickle_load': r'pickle\.load\s*\(',
            'pickle_dump': r'pickle\.dump\s*\(',
            'pickle_loads': r'pickle\.loads\s*\(',
            'pickle_dumps': r'pickle\.dumps\s*\(',
        }
        
        # Trusted internal directories (pickle allowed with security)
        self.trusted_dirs = {
            'src/cryptosmarttrader',
            'models',
            'ml',
            'cache',
            'exports/models',
            'model_backup',
            'mlartifacts'
        }
        
        # External directories (pickle forbidden)
        self.external_dirs = {
            'data',
            'configs',
            'exports/data',
            'integrations',
            'scripts',
            'utils'
        }
    
    def is_trusted_path(self, filepath: Path) -> bool:
        """Check if file is in trusted internal directory"""
        path_str = str(filepath)
        return any(path_str.startswith(trusted) for trusted in self.trusted_dirs)
    
    def is_external_path(self, filepath: Path) -> bool:
        """Check if file is in external directory"""
        path_str = str(filepath)
        return any(path_str.startswith(external) for external in self.external_dirs)
    
    def analyze_pickle_usage(self, content: str) -> Dict[str, List[int]]:
        """Analyze pickle usage in file content"""
        usage = {pattern: [] for pattern in self.pickle_patterns}
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern_name, pattern in self.pickle_patterns.items():
                if re.search(pattern, line):
                    usage[pattern_name].append(i)
        
        return usage
    
    def generate_secure_replacement(self, filepath: Path, content: str) -> str:
        """Generate secure replacement for pickle usage"""
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            original_line = line
            
            # Replace imports
            if re.search(self.pickle_patterns['import_pickle'], line):
                if self.is_trusted_path(filepath):
                    # For trusted paths, use secure serialization
                    line = re.sub(
                        r'import\s+pickle',
                        'from cryptosmarttrader.security.secure_serialization import SecureSerializer',
                        line
                    )
                    line += '\n_secure_serializer = SecureSerializer()  # SECURITY: Restricted to trusted internal files'
                else:
                    # For external paths, use JSON/msgpack
                    line = re.sub(
                        r'import\s+pickle',
                        'import json  # SECURITY: Replaced pickle with JSON for external data',
                        line
                    )
            
            # Replace pickle.load calls
            if re.search(self.pickle_patterns['pickle_load'], line):
                if self.is_trusted_path(filepath):
                    line = re.sub(
                        r'pickle\.load\s*\(',
                        '_secure_serializer.load_secure_pickle(',
                        line
                    )
                else:
                    line = re.sub(
                        r'pickle\.load\s*\(',
                        'json.load(',
                        line
                    )
                    self.migration_report['warnings'].append(
                        f"{filepath}: Replaced pickle.load with json.load - verify data compatibility"
                    )
            
            # Replace pickle.dump calls
            if re.search(self.pickle_patterns['pickle_dump'], line):
                if self.is_trusted_path(filepath):
                    line = re.sub(
                        r'pickle\.dump\s*\(',
                        '_secure_serializer.save_secure_pickle(',
                        line
                    )
                else:
                    line = re.sub(
                        r'pickle\.dump\s*\(',
                        'json.dump(',
                        line
                    )
                    self.migration_report['warnings'].append(
                        f"{filepath}: Replaced pickle.dump with json.dump - verify data compatibility"
                    )
            
            # Replace pickle.loads calls
            if re.search(self.pickle_patterns['pickle_loads'], line):
                if self.is_trusted_path(filepath):
                    # For trusted paths, this requires file-based secure storage
                    self.migration_report['warnings'].append(
                        f"{filepath}: pickle.loads requires manual migration to file-based secure storage"
                    )
                else:
                    line = re.sub(
                        r'pickle\.loads\s*\(',
                        'json.loads(',
                        line
                    )
            
            # Replace pickle.dumps calls
            if re.search(self.pickle_patterns['pickle_dumps'], line):
                if self.is_trusted_path(filepath):
                    # For trusted paths, this requires file-based secure storage
                    self.migration_report['warnings'].append(
                        f"{filepath}: pickle.dumps requires manual migration to file-based secure storage"
                    )
                else:
                    line = re.sub(
                        r'pickle\.dumps\s*\(',
                        'json.dumps(',
                        line
                    )
            
            if line != original_line:
                self.migration_report['pickle_calls_replaced'] += 1
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)
    
    def add_security_header(self, filepath: Path, content: str) -> str:
        """Add security policy header to files"""
        if self.is_trusted_path(filepath):
            header = '''"""
SECURITY POLICY: PICKLE USAGE RESTRICTED
This file is in a trusted internal directory.
Pickle usage is allowed ONLY with SecureSerializer for internal data.
External data must use JSON/msgpack formats.
"""

'''
        else:
            header = '''"""
SECURITY POLICY: NO PICKLE ALLOWED
This file handles external data.
Pickle usage is FORBIDDEN for security reasons.
Use JSON or msgpack for all serialization.
"""

'''
        
        # Check if header already exists
        if 'SECURITY POLICY:' not in content:
            # Insert after shebang and initial docstring
            lines = content.split('\n')
            insert_pos = 0
            
            # Skip shebang
            if lines and lines[0].startswith('#!'):
                insert_pos = 1
            
            # Skip module docstring
            if len(lines) > insert_pos and lines[insert_pos].strip().startswith('"""'):
                in_docstring = True
                insert_pos += 1
                while insert_pos < len(lines) and in_docstring:
                    if '"""' in lines[insert_pos] and not lines[insert_pos].strip().startswith('"""'):
                        insert_pos += 1
                        break
                    insert_pos += 1
            
            lines.insert(insert_pos, header)
            content = '\n'.join(lines)
        
        return content
    
    def migrate_file(self, filepath: Path) -> bool:
        """Migrate single file from pickle to secure alternatives"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.migration_report['files_scanned'] += 1
            
            # Analyze current pickle usage
            usage = self.analyze_pickle_usage(content)
            has_pickle = any(lines for lines in usage.values())
            
            if not has_pickle:
                return True  # No pickle usage found
            
            logger.info(f"Migrating pickle usage in {filepath}")
            
            # Generate secure replacement
            new_content = self.generate_secure_replacement(filepath, content)
            
            # Add security policy header
            new_content = self.add_security_header(filepath, new_content)
            
            # Write back modified content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.migration_report['files_modified'] += 1
            logger.info(f"Successfully migrated {filepath}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to migrate {filepath}: {e}"
            self.migration_report['errors'].append(error_msg)
            logger.error(error_msg)
            return False
    
    def scan_and_migrate(self, root_path: Path = None) -> Dict:
        """Scan and migrate all Python files"""
        if root_path is None:
            root_path = Path('.')
        
        logger.info(f"Starting pickle security migration from {root_path}")
        
        # Find all Python files
        python_files = list(root_path.rglob('*.py'))
        
        # Filter out hidden files and test files for now
        python_files = [
            f for f in python_files 
            if not any(part.startswith('.') for part in f.parts)
            and not str(f).startswith('venv')
            and not str(f).startswith('env')
        ]
        
        logger.info(f"Found {len(python_files)} Python files to scan")
        
        for filepath in python_files:
            self.migrate_file(filepath)
        
        # Generate migration report
        self.generate_migration_report()
        
        return self.migration_report
    
    def generate_migration_report(self):
        """Generate comprehensive migration report"""
        report_path = Path('PICKLE_SECURITY_MIGRATION_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# Pickle Security Migration Report\n\n")
            f.write(f"**Migration Date:** {Path(__file__).stat().st_mtime}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Files Scanned:** {self.migration_report['files_scanned']}\n")
            f.write(f"- **Files Modified:** {self.migration_report['files_modified']}\n")
            f.write(f"- **Pickle Calls Replaced:** {self.migration_report['pickle_calls_replaced']}\n")
            f.write(f"- **Errors:** {len(self.migration_report['errors'])}\n")
            f.write(f"- **Warnings:** {len(self.migration_report['warnings'])}\n\n")
            
            f.write("## Security Policy\n\n")
            f.write("### Trusted Internal Directories (Secure Pickle Allowed)\n")
            for trusted_dir in sorted(self.trusted_dirs):
                f.write(f"- `{trusted_dir}/`\n")
            
            f.write("\n### External Directories (JSON/msgpack Only)\n")
            for external_dir in sorted(self.external_dirs):
                f.write(f"- `{external_dir}/`\n")
            
            if self.migration_report['warnings']:
                f.write("\n## Warnings\n\n")
                for warning in self.migration_report['warnings']:
                    f.write(f"- {warning}\n")
            
            if self.migration_report['errors']:
                f.write("\n## Errors\n\n")
                for error in self.migration_report['errors']:
                    f.write(f"- {error}\n")
            
            f.write("\n## Security Benefits\n\n")
            f.write("1. **HMAC Integrity Validation**: All pickle files include cryptographic signatures\n")
            f.write("2. **Path Restrictions**: Pickle limited to trusted internal directories only\n")
            f.write("3. **Audit Trail**: Complete logging of all serialization operations\n")
            f.write("4. **External Data Safety**: JSON/msgpack for all external data sources\n")
            f.write("5. **ML Model Security**: Enhanced joblib integration with integrity checks\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review all warnings for manual data format compatibility\n")
            f.write("2. Test migrated files thoroughly\n")
            f.write("3. Update CI/CD to include pickle security scanning\n")
            f.write("4. Train team on new secure serialization practices\n")
        
        logger.info(f"Migration report generated: {report_path}")


def main():
    """Main migration function"""
    migrator = PickleMigrator()
    
    # Run migration
    report = migrator.scan_and_migrate()
    
    print("\n" + "="*60)
    print("PICKLE SECURITY MIGRATION COMPLETE")
    print("="*60)
    print(f"Files Scanned: {report['files_scanned']}")
    print(f"Files Modified: {report['files_modified']}")
    print(f"Pickle Calls Replaced: {report['pickle_calls_replaced']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    
    if report['errors']:
        print("\nERRORS:")
        for error in report['errors']:
            print(f"  - {error}")
    
    if report['warnings']:
        print("\nWARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    print("\nSee PICKLE_SECURITY_MIGRATION_REPORT.md for full details")
    
    return len(report['errors']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)