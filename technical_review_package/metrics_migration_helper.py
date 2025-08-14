#!/usr/bin/env python3
"""
Metrics Migration Helper
Assists in migrating existing Prometheus metrics to centralized observability
"""

import os
import re
from pathlib import Path

def find_and_replace_metrics():
    """Find and suggest replacements for existing metrics usage"""
    
    replacements = {
        # Old patterns -> New centralized patterns
        r'Counter\s*\(\s*["\']orders_total["\']': 'get_metrics().orders_sent',
        r'Counter\s*\(\s*["\']filled_orders["\']': 'get_metrics().orders_filled',
        r'Counter\s*\(\s*["\']order_errors["\']': 'get_metrics().order_errors',
        r'Histogram\s*\(\s*["\']latency["\']': 'get_metrics().latency_ms',
        r'Histogram\s*\(\s*["\']slippage["\']': 'get_metrics().slippage_bps',
        r'Gauge\s*\(\s*["\']equity["\']': 'get_metrics().equity',
        r'Gauge\s*\(\s*["\']drawdown["\']': 'get_metrics().drawdown_pct',
        r'Counter\s*\(\s*["\']signals["\']': 'get_metrics().signals_received',
    }
    
    files_to_update = []
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py') and 'observability/metrics.py' not in file:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    changes_needed = []
                    for old_pattern, new_usage in replacements.items():
                        if re.search(old_pattern, content):
                            changes_needed.append((old_pattern, new_usage))
                    
                    if changes_needed:
                        files_to_update.append({
                            'file': filepath,
                            'changes': changes_needed
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

### 2. Replace individual metric definitions with centralized calls:

"""
    
    for file_info in files_to_update:
        report += f"### {file_info['file']}\n"
        for old_pattern, new_usage in file_info['changes']:
            report += f"- Replace `{old_pattern}` with `{new_usage}`\n"
        report += "\n"
    
    report += """
### 3. Standard Usage Patterns:

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

### 4. Context Manager Usage:

```python
# Automatic latency tracking
with timer("place_order", "kraken", "/api/orders"):
    result = exchange.place_order(...)

# Decorator usage
@track_orders("kraken")
def place_order(symbol, side, order_type):
    return exchange_api.place_order(symbol, side, order_type)
```

## Benefits:
✅ Consistent metric naming across all modules
✅ Centralized registry for easy monitoring
✅ Standard labels and conventions
✅ Context managers and decorators for automation
✅ Thread-safe singleton pattern
✅ Easy testing with reset functionality
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
