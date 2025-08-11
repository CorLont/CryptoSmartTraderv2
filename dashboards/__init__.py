"""
Dashboards Module
Interactive dashboard components and visualizations
"""

# Robuuste lazy imports - voorkom package-import failures bij missende submodules
try:
    from .main_dashboard import MainDashboard
except Exception as e:
    # Beperk de impact bij incomplete installs
    MainDashboard = None

try:
    from .advanced_analytics_dashboard import AdvancedAnalyticsDashboard as AnalyticsDashboard
except Exception as e:
    AnalyticsDashboard = None

try:
    from .system_health_dashboard import SystemHealthDashboard as HealthDashboard
except Exception as e:
    HealthDashboard = None

# ChartComponents doesn't exist - create placeholder
ChartComponents = None

# Dynamisch genereren van __all__ gebaseerd op beschikbare imports
__all__ = []
if MainDashboard is not None:
    __all__.append('MainDashboard')
if AnalyticsDashboard is not None:
    __all__.append('AnalyticsDashboard')
if HealthDashboard is not None:
    __all__.append('HealthDashboard')
if ChartComponents is not None:
    __all__.append('ChartComponents')

# Fallback voor lege __all__
if not __all__:
    __all__ = ['MainDashboard', 'AnalyticsDashboard', 'HealthDashboard', 'ChartComponents']