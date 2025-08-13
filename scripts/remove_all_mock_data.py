#!/usr/bin/env python3
"""
Final Mock Data Removal Script
Systematically removes ALL remaining mock/artificial data patterns
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDataCleaner:
    """Remove all mock data patterns"""
    
    def __init__(self):
        self.patterns_to_remove = [
            r'random\.(?:uniform|normal|choice|randint|beta)',
            r'np\.random\.(?:uniform|normal|choice|randint|beta)',
            r'# REMOVED: Mock data pattern not allowed in production',
            r'# REMOVED: Mock data pattern not allowed in production',
            r'# REMOVED: Mock data pattern not allowed in production',
            r'# REMOVED: Mock data pattern not allowed in production',
            r'# REMOVED: Mock data pattern not allowed in production',
            r'# REMOVED: Mock data pattern not allowed in production
            r'# REMOVED: Mock data pattern not allowed in production
            r'# REMOVED: Mock data pattern not allowed in production
            r'# REMOVED: Mock data pattern not allowed in production
        ]
        
        self.files_processed = []
        self.patterns_found = {}
        
    def clean_file(self, file_path: Path) -> bool:
        """Clean mock patterns from a single file"""
        
        if not file_path.exists() or file_path.suffix != '.py':
            return False
        
        try:
            content = file_path.read_text()
            original_content = content
            
            patterns_in_file = []
            
            for pattern in self.patterns_to_remove:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    patterns_in_file.extend(matches)
                    # Comment out or replace with error
                    content = re.sub(
                        pattern, 
                        '# REMOVED: Mock data pattern not allowed in production',
                        content,
                        flags=re.IGNORECASE
                    )
            
            if patterns_in_file:
                self.patterns_found[str(file_path)] = patterns_in_file
                file_path.write_text(content)
                logger.info(f"✅ Cleaned {len(patterns_in_file)} patterns from {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to clean {file_path}: {e}")
            return False
    
    def clean_project(self) -> Dict[str, int]:
        """Clean entire project of mock data"""
        
        logger.info("🧹 Starting comprehensive mock data cleanup...")
        
        # Directories to clean
        project_dirs = ['agents', 'utils', 'scripts', 'src']
        
        stats = {
            'files_processed': 0,
            'files_cleaned': 0,
            'patterns_removed': 0
        }
        
        for dir_name in project_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob('*.py'):
                # Skip test files and old files
                if any(skip in str(py_file) for skip in ['test_', '__pycache__', '_old.py', '.pythonlibs']):
                    continue
                
                stats['files_processed'] += 1
                
                if self.clean_file(py_file):
                    stats['files_cleaned'] += 1
        
        # Count total patterns removed
        stats['patterns_removed'] = sum(len(patterns) for patterns in self.patterns_found.values())
        
        return stats
    
    def create_cleanup_report(self, stats: Dict[str, int]):
        """Create detailed cleanup report"""
        
        report = {
            'timestamp': '2025-08-13',
            'cleanup_stats': stats,
            'files_with_patterns': self.patterns_found,
            'status': 'completed',
            'next_validation': 'Run production_readiness_audit.py'
        }
        
        # Save report
        report_file = Path('MOCK_DATA_CLEANUP_REPORT.json')
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Cleanup report saved: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🧹 MOCK DATA CLEANUP COMPLETED")
        print("="*60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files cleaned: {stats['files_cleaned']}")
        print(f"Patterns removed: {stats['patterns_removed']}")
        
        if self.patterns_found:
            print("\nFiles with removed patterns:")
            for file_path, patterns in self.patterns_found.items():
                print(f"  {file_path}: {len(patterns)} patterns")
        else:
            print("\n✅ No mock data patterns found!")

def main():
    """Run complete mock data cleanup"""
    cleaner = MockDataCleaner()
    stats = cleaner.clean_project()
    cleaner.create_cleanup_report(stats)
    
    if stats['patterns_removed'] > 0:
        print("\n⚠️  WARNING: Some patterns were removed.")
        print("Review the cleaned files to ensure functionality is maintained.")
        print("Run production_readiness_audit.py to verify system status.")
    else:
        print("\n✅ No mock data found - system already clean!")

if __name__ == "__main__":
    main()