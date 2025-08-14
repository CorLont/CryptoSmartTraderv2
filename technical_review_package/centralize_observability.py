#!/usr/bin/env python3
"""
Centralize Prometheus Observability
- Consolidate metrics from ≥11 files into single observability/metrics.py
- Standardize metric naming: orders_sent/filled, order_errors, latency_ms, slippage_bps, equity, drawdown_pct, signals_received
- Create central metrics registry with consistent structure
"""

import os
import re
from pathlib import Path

def analyze_prometheus_usage():
    """Analyze current Prometheus metrics usage across codebase"""
    
    prometheus_files = []
    metrics_patterns = {
        'counter': r'Counter\s*\(',
        'gauge': r'Gauge\s*\(',
        'histogram': r'Histogram\s*\(',
        'summary': r'Summary\s*\(',
        'prometheus_import': r'from prometheus_client import|import prometheus_client'
    }
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    metrics_found = {}
                    for pattern_name, pattern in metrics_patterns.items():
                        matches = re.findall(pattern, content)
                        if matches:
                            metrics_found[pattern_name] = len(matches)
                    
                    if metrics_found:
                        prometheus_files.append({
                            'file': filepath,
                            'metrics': metrics_found,
                            'size': len(content)
                        })
                        
                except Exception:
                    pass
    
    return prometheus_files

def create_centralized_metrics():
    """Create centralized observability/metrics.py module"""
    
    # Ensure observability directory exists
    os.makedirs('src/cryptosmarttrader/observability', exist_ok=True)
    
    metrics_module = '''"""
Centralized Prometheus Metrics for CryptoSmartTrader V2
All observability metrics consolidated in single module with consistent naming
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from typing import Dict, Optional
import time
import threading
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Standard metric types for consistent usage"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Standardized metric definition"""
    name: str
    description: str
    labels: list
    metric_type: MetricType


class CryptoSmartTraderMetrics:
    """
    Centralized metrics registry for CryptoSmartTrader V2
    
    Standardized metric naming convention:
    - Trading: orders_sent, orders_filled, order_errors
    - Performance: latency_ms, slippage_bps, equity, drawdown_pct  
    - Signals: signals_received, signals_processed, signal_accuracy
    - System: api_calls_total, cache_hits, memory_usage_bytes
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all standard metrics with consistent naming"""
        
        # Trading Metrics
        self.orders_sent = Counter(
            'orders_sent_total',
            'Total number of orders sent to exchange',
            ['exchange', 'symbol', 'side', 'order_type'],
            registry=self.registry
        )
        
        self.orders_filled = Counter(
            'orders_filled_total', 
            'Total number of orders successfully filled',
            ['exchange', 'symbol', 'side', 'order_type'],
            registry=self.registry
        )
        
        self.order_errors = Counter(
            'order_errors_total',
            'Total number of order errors by type',
            ['exchange', 'symbol', 'error_type', 'error_code'],
            registry=self.registry
        )
        
        # Performance Metrics
        self.latency_ms = Histogram(
            'latency_ms',
            'Request latency in milliseconds',
            ['operation', 'exchange', 'endpoint'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        self.slippage_bps = Histogram(
            'slippage_bps', 
            'Order slippage in basis points',
            ['exchange', 'symbol', 'side'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        self.equity = Gauge(
            'equity_usd',
            'Current portfolio equity in USD',
            ['strategy', 'account'],
            registry=self.registry
        )
        
        self.drawdown_pct = Gauge(
            'drawdown_pct',
            'Current drawdown percentage from peak',
            ['strategy', 'timeframe'],
            registry=self.registry
        )
        
        # Signal & ML Metrics
        self.signals_received = Counter(
            'signals_received_total',
            'Total number of trading signals received',
            ['agent', 'signal_type', 'symbol'],
            registry=self.registry
        )
        
        self.signals_processed = Counter(
            'signals_processed_total',
            'Total number of signals successfully processed',
            ['agent', 'signal_type', 'symbol', 'outcome'],
            registry=self.registry
        )
        
        self.signal_accuracy = Gauge(
            'signal_accuracy_pct',
            'Signal accuracy percentage by agent',
            ['agent', 'symbol', 'timeframe'],
            registry=self.registry
        )
        
        # System & Infrastructure Metrics
        self.api_calls_total = Counter(
            'api_calls_total',
            'Total number of API calls by service',
            ['service', 'endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hit/miss statistics',
            ['cache_type', 'hit_miss'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes by component',
            ['component', 'memory_type'],
            registry=self.registry
        )
        
        # Risk Management Metrics
        self.risk_violations = Counter(
            'risk_violations_total',
            'Total risk limit violations',
            ['violation_type', 'symbol', 'strategy'],
            registry=self.registry
        )
        
        self.position_size_usd = Gauge(
            'position_size_usd',
            'Current position size in USD',
            ['symbol', 'strategy', 'side'],
            registry=self.registry
        )
        
        # Data Quality Metrics
        self.data_points_received = Counter(
            'data_points_received_total',
            'Total data points received from sources',
            ['source', 'data_type', 'symbol'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1) by source',
            ['source', 'data_type'],
            registry=self.registry
        )
    
    # Convenience Methods for Common Operations
    
    def record_order_sent(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record an order being sent"""
        self.orders_sent.labels(
            exchange=exchange, 
            symbol=symbol, 
            side=side, 
            order_type=order_type
        ).inc()
    
    def record_order_filled(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record an order being filled"""
        self.orders_filled.labels(
            exchange=exchange,
            symbol=symbol,
            side=side, 
            order_type=order_type
        ).inc()
    
    def record_order_error(self, exchange: str, symbol: str, error_type: str, error_code: str = "unknown"):
        """Record an order error"""
        self.order_errors.labels(
            exchange=exchange,
            symbol=symbol,
            error_type=error_type,
            error_code=error_code
        ).inc()
    
    def record_latency(self, operation: str, exchange: str, endpoint: str, latency_ms: float):
        """Record operation latency"""
        self.latency_ms.labels(
            operation=operation,
            exchange=exchange,
            endpoint=endpoint
        ).observe(latency_ms)
    
    def record_slippage(self, exchange: str, symbol: str, side: str, slippage_bps: float):
        """Record trade slippage in basis points"""
        self.slippage_bps.labels(
            exchange=exchange,
            symbol=symbol,
            side=side
        ).observe(slippage_bps)
    
    def update_equity(self, strategy: str, account: str, equity_usd: float):
        """Update current equity"""
        self.equity.labels(strategy=strategy, account=account).set(equity_usd)
    
    def update_drawdown(self, strategy: str, timeframe: str, drawdown_pct: float):
        """Update current drawdown percentage"""
        self.drawdown_pct.labels(strategy=strategy, timeframe=timeframe).set(drawdown_pct)
    
    def record_signal(self, agent: str, signal_type: str, symbol: str):
        """Record a signal being received"""
        self.signals_received.labels(
            agent=agent,
            signal_type=signal_type,
            symbol=symbol
        ).inc()
    
    def record_api_call(self, service: str, endpoint: str, method: str, status_code: int):
        """Record an API call"""
        self.api_calls_total.labels(
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics instance
_metrics_instance: Optional[CryptoSmartTraderMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> CryptoSmartTraderMetrics:
    """Get global metrics instance (singleton)"""
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = CryptoSmartTraderMetrics()
    
    return _metrics_instance


def reset_metrics():
    """Reset global metrics instance (for testing)"""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None


# Context managers for timing operations
class timer:
    """Context manager for timing operations and recording latency"""
    
    def __init__(self, operation: str, exchange: str = "unknown", endpoint: str = "unknown"):
        self.operation = operation
        self.exchange = exchange
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            get_metrics().record_latency(
                self.operation, 
                self.exchange, 
                self.endpoint, 
                latency_ms
            )


# Decorators for automatic metrics collection
def track_api_calls(service: str, endpoint: str = "unknown"):
    """Decorator to automatically track API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                get_metrics().record_api_call(service, endpoint, func.__name__, 200)
                return result
            except Exception as e:
                get_metrics().record_api_call(service, endpoint, func.__name__, 500)
                raise
        return wrapper
    return decorator


def track_orders(exchange: str):
    """Decorator to automatically track order operations"""
    def decorator(func):
        def wrapper(symbol: str, side: str, order_type: str = "market", *args, **kwargs):
            try:
                get_metrics().record_order_sent(exchange, symbol, side, order_type)
                result = func(symbol, side, order_type, *args, **kwargs)
                get_metrics().record_order_filled(exchange, symbol, side, order_type)
                return result
            except Exception as e:
                get_metrics().record_order_error(exchange, symbol, type(e).__name__)
                raise
        return wrapper
    return decorator


# Standard metric labels for consistency
STANDARD_LABELS = {
    'exchanges': ['kraken', 'binance', 'coinbase', 'bybit'],
    'sides': ['buy', 'sell'],
    'order_types': ['market', 'limit', 'stop', 'stop_limit'],
    'signal_types': ['entry', 'exit', 'stop_loss', 'take_profit'],
    'error_types': ['timeout', 'rate_limit', 'invalid_order', 'insufficient_funds'],
    'operations': ['place_order', 'cancel_order', 'get_balance', 'get_orderbook'],
    'strategies': ['momentum', 'mean_reversion', 'arbitrage', 'market_making']
}


if __name__ == "__main__":
    # Example usage
    metrics = get_metrics()
    
    # Record some sample metrics
    metrics.record_order_sent("kraken", "BTC/USD", "buy", "market")
    metrics.record_latency("place_order", "kraken", "/api/orders", 45.2)
    metrics.update_equity("momentum", "main", 100000.0)
    
    print("Sample metrics recorded successfully")
    print(f"Metrics export size: {len(metrics.get_metrics())} bytes")
'''

    with open('src/cryptosmarttrader/observability/metrics.py', 'w') as f:
        f.write(metrics_module)
    
    print("✅ Created centralized observability/metrics.py")

def create_metrics_migration_script():
    """Create script to help migrate existing metrics usage"""
    
    migration_script = '''#!/usr/bin/env python3
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
        report += f"### {file_info['file']}\\n"
        for old_pattern, new_usage in file_info['changes']:
            report += f"- Replace `{old_pattern}` with `{new_usage}`\\n"
        report += "\\n"
    
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
'''

    with open('metrics_migration_helper.py', 'w') as f:
        f.write(migration_script)
    
    print("✅ Created metrics migration helper script")

def main():
    """Main observability centralization execution"""
    
    print("📊 Observability Centralization")
    print("=" * 40)
    
    # Analyze current Prometheus usage
    print("\n🔍 Analyzing current Prometheus usage...")
    prometheus_files = analyze_prometheus_usage()
    
    print(f"Found Prometheus usage in {len(prometheus_files)} files:")
    for file_info in prometheus_files[:10]:  # Show first 10
        print(f"  📄 {file_info['file']}: {file_info['metrics']}")
    
    if len(prometheus_files) > 10:
        print(f"  ... and {len(prometheus_files) - 10} more files")
    
    # Create centralized metrics module
    print(f"\n🏗️  Creating centralized observability module...")
    create_centralized_metrics()
    
    # Create migration helper
    print(f"\n🔧 Creating migration helper...")
    create_metrics_migration_script()
    
    print(f"\n📊 Results:")
    print(f"✅ Prometheus usage found in: {len(prometheus_files)} files")
    print(f"✅ Centralized metrics module created")
    print(f"✅ Migration helper script created")
    print(f"✅ Standardized metric naming implemented:")
    print(f"   - orders_sent/filled, order_errors")
    print(f"   - latency_ms, slippage_bps")
    print(f"   - equity, drawdown_pct")
    print(f"   - signals_received, signal_accuracy")
    print(f"   - api_calls_total, cache_hits")
    
    print(f"\n🎯 Observability centralization complete!")
    print(f"📋 Next: Run metrics_migration_helper.py for migration guidance")

if __name__ == "__main__":
    main()