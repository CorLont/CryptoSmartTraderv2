#!/usr/bin/env python3
"""
Settings Validation Script for CryptoSmartTrader V2
Standalone validation without starting services
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cryptosmarttrader.config import load_and_validate_settings
import json


def main():
    """Validate settings and display results"""
    print("🔍 CryptoSmartTrader V2 - Settings Validation")
    print("=" * 50)
    
    try:
        # Load and validate settings
        settings = load_and_validate_settings()
        
        # Display configuration summary
        summary = settings.get_summary()
        print("\n✅ Configuration Valid")
        print(json.dumps(summary, indent=2, default=str))
        
        # Display validation results
        validation = settings.validate_startup_requirements()
        
        if validation["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        if validation["critical_issues"]:
            print("\n❌ Critical Issues:")
            for issue in validation["critical_issues"]:
                print(f"  - {issue}")
            return 1
        
        print(f"\n🎯 System Status: {'✅ READY' if validation['startup_ready'] else '❌ NOT READY'}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())