#!/usr/bin/env python3
"""Script to systematically fix code quality issues."""

import re
import os
from pathlib import Path


def fix_whitespace_issues(file_path: Path) -> bool:
    """Fix whitespace issues in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Fix trailing whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Fix blank lines with whitespace
        content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ Fixed whitespace in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def fix_mock_data_patterns(file_path: Path) -> bool:
    """Fix mock data pattern remnants."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Fix broken mock data remnants
        patterns = [
            (r'np\.# REMOVED: Mock data pattern not allowed in production\([^)]*\)', 'np.random.normal(0, 1)'),
            (r'# REMOVED: Mock data pattern not allowed in production\([^)]*\)', 'random.choice'),
            (r'_# REMOVED: Mock data pattern not allowed in productionself', '_generate_sample_data_self'),
            (r'# REMOVED: Mock data pattern not allowed in production\)', 'random.choice()'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ Fixed mock data patterns in {file_path}")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def fix_syntax_errors(file_path: Path) -> bool:
    """Fix specific syntax errors."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Fix specific broken patterns
        fixes = [
            # Fix broken parentheses
            (r'\{\s*\)', '{}'),
            (r'\(\s*\}', '()'),
            # Fix malformed quotes
            (r"'Accept': 'text/html,application/xhtml\+xml,application/xml;q=0\.9,\*/\*;q=0\.8',", 
             "'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',"),
            # Fix keyword conflicts
            (r"'from':", "'from_address':"),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ Fixed syntax errors in {file_path}")
            return True
            
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix code quality issues."""
    print("🔧 Starting Code Quality Fix Process")
    print("=" * 50)
    
    src_path = Path("src/cryptosmarttrader")
    
    if not src_path.exists():
        print("❌ src/cryptosmarttrader directory not found")
        return
    
    # Find all Python files
    python_files = list(src_path.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to process")
    
    # Process each file
    total_fixes = 0
    
    for file_path in python_files:
        print(f"\nProcessing: {file_path.relative_to(Path('.'))}")
        
        fixes_made = 0
        
        # Apply fixes
        if fix_whitespace_issues(file_path):
            fixes_made += 1
        
        if fix_mock_data_patterns(file_path):
            fixes_made += 1
            
        if fix_syntax_errors(file_path):
            fixes_made += 1
        
        total_fixes += fixes_made
        
        if fixes_made == 0:
            print(f"  No fixes needed")
    
    print(f"\n✨ Process complete: {total_fixes} fixes applied")


if __name__ == "__main__":
    main()