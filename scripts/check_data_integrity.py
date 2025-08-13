#!/usr/bin/env python3
'''
Data Integrity Checker - Ensures no artificial data is used
'''

import sys
from pathlib import Path

def check_for_artificial_data():
    violations = []
    
    # Check for random/mock patterns in project code only (exclude libraries)
    py_files = []
    project_dirs = ['agents', 'utils', 'scripts', 'src', '.']
    for dir_name in project_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            py_files.extend(dir_path.glob('**/*.py'))
    
    for py_file in py_files:
        if ('test_' in py_file.name or '__pycache__' in str(py_file) or 
            '.pythonlibs' in str(py_file) or '.cache' in str(py_file)):
            continue
            
        try:
            content = py_file.read_text()
            if any(pattern in content for pattern in [
                'np.random.', 'random.', 'mock_', 'fake_', 'dummy_',
                'simulate_', 'artificial_', 'test_data'
            ]):
                violations.append(f"Artificial data patterns found in {py_file}")
        except Exception:
            pass
    
    # Check for artificial prediction files
    if Path('exports/production/predictions.csv').exists():
        violations.append("Artificial predictions.csv exists")
    
    if violations:
        print("DATA INTEGRITY VIOLATIONS FOUND:")
        for violation in violations:
            print(f"  - {violation}")
        return False
    else:
        print("✅ No artificial data detected")
        return True

if __name__ == "__main__":
    success = check_for_artificial_data()
    sys.exit(0 if success else 1)
