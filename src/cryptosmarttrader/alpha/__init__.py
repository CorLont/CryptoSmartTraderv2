"""
Alpha & Portfolio Management Module
Advanced alpha generation and portfolio optimization for CryptoSmartTrader.
"""

from .regime_detector import RegimeDetector, RegimeSignal, MarketRegime, create_regime_detector

from .kelly_sizing import (
    KellyOptimizer,
    PositionSizing,
    PortfolioAllocation,
    create_kelly_optimizer,
)

from .cluster_manager import (
    ClusterManager,
    AssetCluster,
    ClusterLimit,
    AssetClassification,
    ClusterExposure,
    create_cluster_manager,
)

from .return_attribution import (
    ReturnAttributor,
    ReturnAttribution,
    PortfolioAttribution,
    ReturnComponent,
    create_return_attributor,
)

__all__ = [
    "RegimeDetector",
    "RegimeSignal",
    "MarketRegime",
    "KellyOptimizer",
    "PositionSizing",
    "PortfolioAllocation",
    "ClusterManager",
    "AssetCluster",
    "ClusterLimit",
    "AssetClassification",
    "ClusterExposure",
    "ReturnAttributor",
    "ReturnAttribution",
    "PortfolioAttribution",
    "ReturnComponent",
    "create_regime_detector",
    "create_kelly_optimizer",
    "create_cluster_manager",
    "create_return_attributor",
]
