"""
Prometheus Metrics - Alias to canonical implementation
This file redirects to the single source of truth in src/cryptosmarttrader/observability
"""

# Import from canonical source
from src.cryptosmarttrader.observability.metrics import *

# Maintain backward compatibility
from src.cryptosmarttrader.observability.metrics import PrometheusMetrics as PrometheusMetricsSystem