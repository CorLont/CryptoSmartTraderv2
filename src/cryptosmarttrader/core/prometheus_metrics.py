"""
Prometheus Metrics - Alias to canonical implementation
This file redirects to the single source of truth in observability module
"""

# Import from canonical source
from ..observability.metrics import *  # type: ignore

# Maintain backward compatibility
from ..observability.metrics import PrometheusMetrics as PrometheusMetricsSystem