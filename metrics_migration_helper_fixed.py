#!/usr/bin/env python3
"""
Metrics Migration Helper - Fixed Version
Assists in migrating existing Prometheus metrics to centralized observability
"""

import os
import re
from pathlib import Path

def find_and_replace_metrics():
    """Find and suggest replacements for existing metrics usage"""
    
    # Simple patterns without complex regex
    patterns_to_find = [
        'Counter(',
        'Gauge(',
        'Histogram(',
        'Summary(',
        'prometheus_client'
    ]
    
    files_to_update = []
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py') and 'observability/metrics.py' not in file:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    patterns_found = []
                    for pattern in patterns_to_find:
                        if pattern in content:
                            patterns_found.append(pattern)
                    
                    if patterns_found:
                        files_to_update.append({
                            'file': filepath,
                            'patterns': patterns_found
                        })
                        
                except Exception:
                    pass
    
    return files_to_update

def generate_migration_report():
    """Generate migration report"""
    
    files_to_update = find_and_replace_metrics()
    
    report = f"""
# OBSERVABILITY MIGRATION REPORT

## Summary
Found {len(files_to_update)} files that need migration to centralized metrics.

## Migration Steps:

### 1. Add centralized import:
```python
from cryptosmarttrader.observability.metrics import get_metrics
```

### 2. Replace individual metric definitions with centralized calls

## Files to Update:
"""
    
    for file_info in files_to_update:
        report += f"\n### {file_info['file']}\n"
        report += f"Patterns found: {', '.join(file_info['patterns'])}\n"
    
    report += """
## Standard Usage Patterns:

```python
# Trading metrics
get_metrics().record_order_sent("kraken", "BTC/USD", "buy", "market")
get_metrics().record_order_filled("kraken", "BTC/USD", "buy", "market") 
get_metrics().record_order_error("kraken", "BTC/USD", "timeout")

# Performance metrics  
get_metrics().record_latency("place_order", "kraken", "/api/orders", 45.2)
get_metrics().record_slippage("kraken", "BTC/USD", "buy", 2.5)
get_metrics().update_equity("momentum", "main", 100000.0)
get_metrics().update_drawdown("momentum", "1h", 5.2)

# Signal metrics
get_metrics().record_signal("ml_agent", "entry", "BTC/USD")
get_metrics().record_api_call("kraken", "/api/balance", "GET", 200)
```

## Benefits:
✅ Consistent metric naming across all modules
✅ Centralized registry for easy monitoring
✅ Standard labels and conventions
✅ Thread-safe singleton pattern
"""

    with open('OBSERVABILITY_MIGRATION_REPORT.md', 'w') as f:
        f.write(report)
    
    print("📋 Migration report created: OBSERVABILITY_MIGRATION_REPORT.md")
    return len(files_to_update)

if __name__ == "__main__":
    print("🔍 Analyzing metrics migration needs...")
    files_needing_migration = generate_migration_report()
    print(f"📊 Found {files_needing_migration} files needing migration")
    print("📋 See OBSERVABILITY_MIGRATION_REPORT.md for detailed migration steps")