
# OBSERVABILITY MIGRATION REPORT

## Summary
Found 10 files that need migration to centralized metrics.

## Migration Steps:

### 1. Add centralized import:
```python
from cryptosmarttrader.observability.metrics import get_metrics
```

### 2. Replace individual metric definitions with centralized calls

## Files to Update:

### src/cryptosmarttrader/core/final_system_integrator.py
Patterns found: prometheus_client

### src/cryptosmarttrader/core/system_validator.py
Patterns found: prometheus_client

### src/cryptosmarttrader/core/metrics_collector.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

### src/cryptosmarttrader/agents/agents/sentiment_agent.py
Patterns found: Summary(

### src/cryptosmarttrader/utils/metrics.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

### src/cryptosmarttrader/deployment/go_live_system.py
Patterns found: Counter(, Gauge(, prometheus_client

### src/cryptosmarttrader/observability/metrics.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

### src/cryptosmarttrader/observability/metrics_collector.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

### src/cryptosmarttrader/observability/unified_metrics.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

### src/cryptosmarttrader/monitoring/prometheus_metrics.py
Patterns found: Counter(, Gauge(, Histogram(, prometheus_client

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
