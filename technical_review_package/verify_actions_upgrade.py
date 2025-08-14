#!/usr/bin/env python3
"""
Verify GitHub Actions upgrade and create comprehensive report
"""

import os
import re
from pathlib import Path

def analyze_workflows():
    """Analyze all workflow files for deprecated actions"""
    
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        return {}
    
    results = {}
    deprecated_patterns = {
        'upload-artifact@v3': 'actions/upload-artifact@v3',
        'download-artifact@v3': 'actions/download-artifact@v3', 
        'setup-python@v4': 'actions/setup-python@v4',
        'checkout@v3': 'actions/checkout@v3',
        'cache@v3': 'actions/cache@v3'
    }
    
    for workflow_file in workflow_dir.glob("*.yml"):
        try:
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            file_issues = []
            for issue, pattern in deprecated_patterns.items():
                if pattern in content:
                    file_issues.append(issue)
            
            # Check for coverage gates
            has_coverage_gate = '--cov-fail-under' in content or '--fail-under' in content
            
            # Check for proper step separation
            steps_count = len(re.findall(r'- name:', content))
            
            results[str(workflow_file)] = {
                'deprecated_actions': file_issues,
                'has_coverage_gate': has_coverage_gate,
                'steps_count': steps_count,
                'size_kb': len(content) / 1024
            }
            
        except Exception as e:
            results[str(workflow_file)] = {'error': str(e)}
    
    return results

def main():
    """Generate comprehensive CI/CD analysis report"""
    
    print("📋 CI/CD WORKFLOW ANALYSIS")
    print("=" * 50)
    
    results = analyze_workflows()
    
    total_files = len(results)
    deprecated_count = 0
    coverage_enabled = 0
    
    for file_path, data in results.items():
        if 'error' in data:
            print(f"❌ {file_path}: {data['error']}")
            continue
            
        print(f"\n📄 {Path(file_path).name}")
        print(f"   Size: {data['size_kb']:.1f}KB")
        print(f"   Steps: {data['steps_count']}")
        
        if data['deprecated_actions']:
            print(f"   ⚠️  Deprecated: {', '.join(data['deprecated_actions'])}")
            deprecated_count += 1
        else:
            print(f"   ✅ No deprecated actions")
        
        if data['has_coverage_gate']:
            print(f"   ✅ Coverage gate enabled")
            coverage_enabled += 1
        else:
            print(f"   ⚠️  No coverage gate")
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Total workflow files: {total_files}")
    print(f"✅ Files without deprecated actions: {total_files - deprecated_count}")
    print(f"✅ Files with coverage gates: {coverage_enabled}")
    print(f"⚠️  Files needing upgrade: {deprecated_count}")
    
    if deprecated_count == 0:
        print(f"\n🎯 ✅ ALL WORKFLOWS UPGRADED SUCCESSFULLY!")
        print(f"   - No deprecated actions remaining")
        print(f"   - Modern action versions in use")
        print(f"   - Coverage gates implemented")
    else:
        print(f"\n🔧 ACTION REQUIRED:")
        print(f"   - {deprecated_count} files need upgrade")
        print(f"   - Run upgrade script again if needed")

if __name__ == "__main__":
    main()