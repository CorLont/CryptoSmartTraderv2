"""
Analytics Module

Advanced return attribution and performance analytics system
for understanding what drives trading performance.
"""

from .return_attribution import ReturnAttributor, AttributionResult, AttributionComponent
from .performance_monitor import PerformanceMonitor, PerformanceAlert, AlertType
from .degradation_detector import DegradationDetector, DegradationAlert, DriftMetrics
from .dashboard_analytics import DashboardAnalytics, AnalyticsCache, RealtimeMetrics

__all__ = [
    "ReturnAttributor",
    "AttributionResult",
    "AttributionComponent",
    "PerformanceMonitor",
    "PerformanceAlert",
    "AlertType",
    "DegradationDetector",
    "DegradationAlert",
    "DriftMetrics",
    "DashboardAnalytics",
    "AnalyticsCache",
    "RealtimeMetrics"
]
