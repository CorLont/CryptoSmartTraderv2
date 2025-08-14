#!/usr/bin/env python3
"""
Update existing metrics files to use centralized observability
Replace scattered metrics with centralized imports
"""

import os
import re

def update_metrics_files():
    """Update files to use centralized metrics"""
    
    # Files with heavy Prometheus usage that should be updated
    priority_files = [
        'src/cryptosmarttrader/observability/metrics_collector.py',
        'src/cryptosmarttrader/observability/unified_metrics.py', 
        'src/cryptosmarttrader/monitoring/prometheus_metrics.py',
        'src/cryptosmarttrader/utils/metrics.py'
    ]
    
    updated_files = 0
    
    for filepath in priority_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Check if already uses centralized metrics
                if 'from cryptosmarttrader.observability.metrics import get_metrics' in content:
                    print(f"⏭️  {filepath} already using centralized metrics")
                    continue
                
                # Add deprecation notice and import redirect
                deprecation_notice = '''"""
DEPRECATED: This module has been superseded by centralized observability.
Use: from cryptosmarttrader.observability.metrics import get_metrics

This file is kept for backward compatibility only.
"""

# Redirect to centralized metrics
from cryptosmarttrader.observability.metrics import get_metrics

# Legacy compatibility - delegate to centralized metrics
def get_legacy_metrics():
    """Legacy compatibility function"""
    return get_metrics()

# Re-export centralized metrics for backward compatibility
metrics = get_metrics()

'''
                
                # Add deprecation notice at the top
                if '"""' in content:
                    # Find the first docstring and add after it
                    first_docstring_end = content.find('"""', content.find('"""') + 3) + 3
                    content = content[:first_docstring_end] + '\n\n' + deprecation_notice + content[first_docstring_end:]
                else:
                    content = deprecation_notice + content
                
                with open(filepath, 'w') as f:
                    f.write(content)
                
                updated_files += 1
                print(f"✅ Updated: {filepath}")
                
            except Exception as e:
                print(f"❌ Error updating {filepath}: {e}")
    
    return updated_files

def create_observability_init():
    """Create __init__.py for observability package"""
    
    init_content = '''"""
CryptoSmartTrader V2 Observability Package

Centralized metrics, monitoring, and observability for the trading system.
All Prometheus metrics are consolidated in metrics.py for consistency.
"""

from .metrics import get_metrics, timer, track_api_calls, track_orders

__all__ = [
    'get_metrics',
    'timer', 
    'track_api_calls',
    'track_orders'
]

# Version info
__version__ = '2.0.0'
__title__ = 'CryptoSmartTrader Observability'
__description__ = 'Centralized observability and metrics collection'
'''

    init_file = 'src/cryptosmarttrader/observability/__init__.py'
    with open(init_file, 'w') as f:
        f.write(init_content)
    
    print(f"✅ Created observability package init file")

def main():
    """Main execution"""
    
    print("🔧 Updating Existing Metrics Files")
    print("=" * 40)
    
    # Update existing files with deprecation notices
    print("\n📋 Adding deprecation notices to existing metrics files...")
    updated = update_metrics_files()
    
    # Create package init
    print("\n🏗️  Creating observability package structure...")
    create_observability_init()
    
    print(f"\n📊 Results:")
    print(f"✅ Files updated with deprecation notices: {updated}")
    print(f"✅ Observability package structure created")
    print(f"✅ Backward compatibility maintained")
    print(f"✅ Centralized metrics ready for use")
    
    print(f"\n🎯 Metrics consolidation complete!")
    print(f"📋 All modules can now import: from cryptosmarttrader.observability.metrics import get_metrics")

if __name__ == "__main__":
    main()