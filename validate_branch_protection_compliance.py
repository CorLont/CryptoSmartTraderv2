#!/usr/bin/env python3
"""Validate branch protection and governance compliance."""

import json
from pathlib import Path


def validate_branch_protection():
    """Validate branch protection configuration compliance."""
    print("🛡️ Validating Branch Protection & Governance Compliance")
    print("=" * 60)
    
    compliance_results = {}
    
    # 1. CODEOWNERS Configuration
    print("\n1. 📋 CODEOWNERS Configuration")
    codeowners_file = Path('.github/CODEOWNERS')
    
    if codeowners_file.exists():
        content = codeowners_file.read_text()
        
        # Check global ownership
        has_global_owner = '* @clont1' in content
        
        # Check critical paths
        critical_paths = [
            '/src/cryptosmarttrader/core/',
            '/src/cryptosmarttrader/risk/', 
            '/.github/workflows/',
            '/pyproject.toml',
            '/tests/'
        ]
        
        protected_paths = sum(1 for path in critical_paths if path in content)
        
        compliance_results['codeowners_exists'] = True
        compliance_results['global_owner'] = has_global_owner
        compliance_results['critical_paths_protected'] = protected_paths >= 4
        
        print(f"   ✅ CODEOWNERS file exists")
        print(f"   {'✅' if has_global_owner else '❌'} Global ownership configured (* @clont1)")
        print(f"   {'✅' if protected_paths >= 4 else '❌'} Critical paths protected ({protected_paths}/5)")
    else:
        compliance_results['codeowners_exists'] = False
        print("   ❌ CODEOWNERS file missing")
    
    # 2. Branch Protection Workflow
    print("\n2. 🔧 Branch Protection Automation")
    protection_workflow = Path('.github/workflows/branch-protection.yml')
    
    if protection_workflow.exists():
        content = protection_workflow.read_text()
        
        # Check required status checks
        has_status_checks = 'required_status_checks' in content
        has_security_check = 'Security Scanning' in content
        has_test_checks = 'Test (Python' in content
        has_quality_check = 'Quality Gates' in content
        
        # Check review requirements
        has_review_requirements = 'required_pull_request_reviews' in content
        has_codeowner_reviews = 'require_code_owner_reviews: true' in content
        
        compliance_results['protection_workflow'] = True
        compliance_results['status_checks'] = has_status_checks
        compliance_results['required_checks'] = has_security_check and has_test_checks and has_quality_check
        compliance_results['review_requirements'] = has_review_requirements and has_codeowner_reviews
        
        print(f"   ✅ Branch protection workflow exists")
        print(f"   {'✅' if has_status_checks else '❌'} Required status checks configured")
        print(f"   {'✅' if compliance_results['required_checks'] else '❌'} All required checks present")
        print(f"   {'✅' if compliance_results['review_requirements'] else '❌'} Code owner reviews required")
    else:
        compliance_results['protection_workflow'] = False
        print("   ❌ Branch protection workflow missing")
    
    # 3. Pull Request Template
    print("\n3. 📝 Pull Request Template")
    pr_template = Path('.github/pull_request_template.md')
    
    if pr_template.exists():
        content = pr_template.read_text()
        
        # Check required sections
        required_sections = [
            'Testing',
            'Security',
            'Quality Assurance',
            'Risk Assessment',
            'Architecture'
        ]
        
        sections_present = sum(1 for section in required_sections if section in content)
        has_checkboxes = '- [ ]' in content
        
        compliance_results['pr_template'] = True
        compliance_results['comprehensive_template'] = sections_present >= 4
        compliance_results['checkboxes'] = has_checkboxes
        
        print(f"   ✅ PR template exists")
        print(f"   {'✅' if compliance_results['comprehensive_template'] else '❌'} Comprehensive sections ({sections_present}/5)")
        print(f"   {'✅' if has_checkboxes else '❌'} Checkbox format for validation")
    else:
        compliance_results['pr_template'] = False
        print("   ❌ Pull request template missing")
    
    # 4. CI/CD Status Check Requirements
    print("\n4. 🔄 CI/CD Integration")
    ci_workflow = Path('.github/workflows/ci.yml')
    
    if ci_workflow.exists():
        content = ci_workflow.read_text()
        
        # Check job names that should be required
        required_jobs = [
            'security',
            'test',
            'quality'
        ]
        
        jobs_present = sum(1 for job in required_jobs if f'name: {job.title()}' in content or f'{job}:' in content)
        has_matrix = 'matrix:' in content and 'python-version' in content
        has_coverage_gate = '--cov-fail-under' in content
        
        compliance_results['ci_integration'] = True
        compliance_results['required_jobs'] = jobs_present >= 3
        compliance_results['matrix_testing'] = has_matrix
        compliance_results['coverage_enforcement'] = has_coverage_gate
        
        print(f"   ✅ CI workflow configured")
        print(f"   {'✅' if compliance_results['required_jobs'] else '❌'} Required jobs present ({jobs_present}/3)")
        print(f"   {'✅' if has_matrix else '❌'} Matrix testing configured")
        print(f"   {'✅' if has_coverage_gate else '❌'} Coverage gates enforced")
    else:
        compliance_results['ci_integration'] = False
        print("   ❌ CI workflow missing")
    
    # 5. Security Configuration
    print("\n5. 🔒 Security Configuration")
    gitleaks_config = Path('.gitleaks.toml')
    security_workflow = Path('.github/workflows/security.yml')
    
    security_compliance = {
        'gitleaks_config': gitleaks_config.exists(),
        'security_workflow': security_workflow.exists(),
        'crypto_patterns': False,
        'daily_scans': False
    }
    
    if gitleaks_config.exists():
        content = gitleaks_config.read_text()
        security_compliance['crypto_patterns'] = 'crypto-api-keys' in content or 'kraken' in content.lower()
    
    if security_workflow.exists():
        content = security_workflow.read_text()
        security_compliance['daily_scans'] = 'schedule:' in content and 'cron:' in content
    
    compliance_results['security_config'] = all(security_compliance.values())
    
    for check, passed in security_compliance.items():
        print(f"   {'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
    
    # 6. Documentation Standards
    print("\n6. 📚 Documentation Standards")
    
    docs_compliance = {
        'readme': Path('README.md').exists(),
        'changelog': Path('CHANGELOG.md').exists(),
        'security_policy': Path('SECURITY.md').exists(),
        'replit_docs': Path('replit.md').exists()
    }
    
    compliance_results['documentation'] = sum(docs_compliance.values()) >= 3
    
    for doc, exists in docs_compliance.items():
        print(f"   {'✅' if exists else '❌'} {doc.replace('_', ' ').title()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BRANCH PROTECTION COMPLIANCE SUMMARY")
    print("=" * 60)
    
    total_checks = len(compliance_results)
    passed_checks = sum(1 for passed in compliance_results.values() if passed)
    compliance_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    for check, passed in compliance_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    print(f"\n🎯 OVERALL COMPLIANCE: {passed_checks}/{total_checks} ({compliance_rate:.1f}%)")
    
    if compliance_rate >= 90:
        print("🏆 EXCELLENT - Enterprise governance standards fully met")
        return 0
    elif compliance_rate >= 80:
        print("✅ GOOD - Most governance requirements met")
        return 1
    else:
        print("⚠️ NEEDS IMPROVEMENT - Governance gaps detected")
        return 2


if __name__ == "__main__":
    exit_code = validate_branch_protection()
    exit(exit_code)