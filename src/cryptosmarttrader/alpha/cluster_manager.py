"""
Cluster & Correlation Management for CryptoSmartTrader
Advanced cluster caps and correlation limits for portfolio risk management.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AssetCluster(Enum):
    """Asset cluster types."""

    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    DEFI = "defi"
    LAYER1 = "layer1"
    LAYER2 = "layer2"
    MEME = "meme"
    STABLE = "stable"
    PRIVACY = "privacy"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class ClusterLimit:
    """Cluster exposure limit configuration."""

    cluster: AssetCluster
    max_allocation: float
    max_correlation: float
    max_assets: int
    rebalance_threshold: float


@dataclass
class AssetClassification:
    """Asset cluster classification."""

    symbol: str
    primary_cluster: AssetCluster
    secondary_clusters: List[AssetCluster]
    market_cap_tier: str
    correlation_group: int
    risk_score: float
    metadata: Dict[str, Any]


@dataclass
class ClusterExposure:
    """Current cluster exposure summary."""

    cluster: AssetCluster
    total_allocation: float
    asset_count: int
    avg_correlation: float
    risk_contribution: float
    assets: List[str]
    limit_utilization: float


class ClusterManager:
    """
    Enterprise cluster and correlation management system.

    Features:
    - Dynamic asset clustering using returns correlation
    - Cluster exposure limits and monitoring
    - Correlation-based position caps
    - Factor exposure management
    - Risk decomposition by cluster
    - Automatic rebalancing triggers
    """

    def __init__(
        self,
        max_cluster_allocation: float = 0.30,
        max_correlation_threshold: float = 0.70,
        max_assets_per_cluster: int = 8,
        rebalance_threshold: float = 0.05,
    ):
        self.max_cluster_allocation = max_cluster_allocation
        self.max_correlation_threshold = max_correlation_threshold
        self.max_assets_per_cluster = max_assets_per_cluster
        self.rebalance_threshold = rebalance_threshold

        # Initialize cluster limits
        self.cluster_limits = self._initialize_cluster_limits()

        # Asset classifications cache
        self.asset_classifications: Dict[str, AssetClassification] = {}

        # Correlation tracking
        self.correlation_history: Dict[str, pd.DataFrame] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("ClusterManager initialized with enterprise risk controls")

    def _initialize_cluster_limits(self) -> Dict[AssetCluster, ClusterLimit]:
        """Initialize default cluster limits."""
        return {
            AssetCluster.LARGE_CAP: ClusterLimit(
                cluster=AssetCluster.LARGE_CAP,
                max_allocation=0.50,  # 50% max in large caps
                max_correlation=0.80,
                max_assets=10,
                rebalance_threshold=0.05,
            ),
            AssetCluster.MID_CAP: ClusterLimit(
                cluster=AssetCluster.MID_CAP,
                max_allocation=0.30,
                max_correlation=0.75,
                max_assets=8,
                rebalance_threshold=0.04,
            ),
            AssetCluster.SMALL_CAP: ClusterLimit(
                cluster=AssetCluster.SMALL_CAP,
                max_allocation=0.20,
                max_correlation=0.70,
                max_assets=6,
                rebalance_threshold=0.03,
            ),
            AssetCluster.DEFI: ClusterLimit(
                cluster=AssetCluster.DEFI,
                max_allocation=0.25,
                max_correlation=0.80,
                max_assets=8,
                rebalance_threshold=0.04,
            ),
            AssetCluster.LAYER1: ClusterLimit(
                cluster=AssetCluster.LAYER1,
                max_allocation=0.40,
                max_correlation=0.75,
                max_assets=6,
                rebalance_threshold=0.05,
            ),
            AssetCluster.LAYER2: ClusterLimit(
                cluster=AssetCluster.LAYER2,
                max_allocation=0.20,
                max_correlation=0.70,
                max_assets=5,
                rebalance_threshold=0.03,
            ),
            AssetCluster.MEME: ClusterLimit(
                cluster=AssetCluster.MEME,
                max_allocation=0.10,  # Limited meme exposure
                max_correlation=0.60,
                max_assets=4,
                rebalance_threshold=0.02,
            ),
            AssetCluster.STABLE: ClusterLimit(
                cluster=AssetCluster.STABLE,
                max_allocation=0.20,
                max_correlation=0.90,  # Stables can be highly correlated
                max_assets=3,
                rebalance_threshold=0.05,
            ),
            AssetCluster.PRIVACY: ClusterLimit(
                cluster=AssetCluster.PRIVACY,
                max_allocation=0.15,
                max_correlation=0.65,
                max_assets=4,
                rebalance_threshold=0.03,
            ),
            AssetCluster.INFRASTRUCTURE: ClusterLimit(
                cluster=AssetCluster.INFRASTRUCTURE,
                max_allocation=0.25,
                max_correlation=0.70,
                max_assets=6,
                rebalance_threshold=0.04,
            ),
        }

    def classify_assets(
        self, asset_data: Dict[str, Dict[str, Any]], price_returns: pd.DataFrame
    ) -> Dict[str, AssetClassification]:
        """
        Classify assets into clusters based on multiple factors.

        Args:
            asset_data: Asset metadata (market_cap, category, etc.)
            price_returns: Historical returns for correlation analysis

        Returns:
            Dictionary of asset classifications
        """
        classifications = {}

        # Calculate correlation matrix
        correlation_matrix = price_returns.corr()

        # Perform clustering if sklearn available
        correlation_groups = self._perform_correlation_clustering(price_returns)

        for symbol in asset_data:
            if symbol not in asset_data:
                continue

            metadata = asset_data[symbol]

            # Determine primary cluster
            primary_cluster = self._determine_primary_cluster(symbol, metadata)

            # Determine secondary clusters
            secondary_clusters = self._determine_secondary_clusters(symbol, metadata)

            # Get market cap tier
            market_cap_tier = self._get_market_cap_tier(metadata.get("market_cap", 0))

            # Get correlation group
            corr_group = correlation_groups.get(symbol, 0)

            # Calculate risk score
            risk_score = self._calculate_asset_risk_score(
                symbol, metadata, price_returns.get(symbol, pd.Series())
            )

            classification = AssetClassification(
                symbol=symbol,
                primary_cluster=primary_cluster,
                secondary_clusters=secondary_clusters,
                market_cap_tier=market_cap_tier,
                correlation_group=corr_group,
                risk_score=risk_score,
                metadata=metadata,
            )

            classifications[symbol] = classification

        self.asset_classifications = classifications
        return classifications

    def _determine_primary_cluster(self, symbol: str, metadata: Dict[str, Any]) -> AssetCluster:
        """Determine primary cluster for asset."""

        # Check explicit category mappings
        category = metadata.get("category", "").lower()

        if "defi" in category or "decentralized" in category:
            return AssetCluster.DEFI
        elif "layer 1" in category or "platform" in category:
            return AssetCluster.LAYER1
        elif "layer 2" in category or "scaling" in category:
            return AssetCluster.LAYER2
        elif "meme" in category or "dog" in category:
            return AssetCluster.MEME
        elif "stable" in category or "usd" in symbol.lower():
            return AssetCluster.STABLE
        elif "privacy" in category or "anonymous" in category:
            return AssetCluster.PRIVACY
        elif "infrastructure" in category or "oracle" in category:
            return AssetCluster.INFRASTRUCTURE

        # Fallback to market cap classification
        market_cap = metadata.get("market_cap", 0)
        if market_cap > 10_000_000_000:  # $10B+
            return AssetCluster.LARGE_CAP
        elif market_cap > 1_000_000_000:  # $1B+
            return AssetCluster.MID_CAP
        else:
            return AssetCluster.SMALL_CAP

    def _determine_secondary_clusters(
        self, symbol: str, metadata: Dict[str, Any]
    ) -> List[AssetCluster]:
        """Determine secondary cluster classifications."""
        secondary = []

        # Assets can belong to multiple clusters
        category = metadata.get("category", "").lower()
        market_cap = metadata.get("market_cap", 0)

        # Add market cap tier if not primary
        if market_cap > 10_000_000_000:
            secondary.append(AssetCluster.LARGE_CAP)
        elif market_cap > 1_000_000_000:
            secondary.append(AssetCluster.MID_CAP)

        # Add functional categories
        if "exchange" in category:
            secondary.append(AssetCluster.INFRASTRUCTURE)
        if "gaming" in category or "nft" in category:
            secondary.append(AssetCluster.INFRASTRUCTURE)

        return secondary

    def _get_market_cap_tier(self, market_cap: float) -> str:
        """Get market cap tier classification."""
        if market_cap > 50_000_000_000:  # $50B+
            return "mega_cap"
        elif market_cap > 10_000_000_000:  # $10B+
            return "large_cap"
        elif market_cap > 1_000_000_000:  # $1B+
            return "mid_cap"
        elif market_cap > 100_000_000:  # $100M+
            return "small_cap"
        else:
            return "micro_cap"

    def _perform_correlation_clustering(self, returns: pd.DataFrame) -> Dict[str, int]:
        """Perform correlation-based clustering of assets."""
        if not HAS_SKLEARN or len(returns.columns) < 3:
            # Simple fallback clustering
            return {symbol: 0 for symbol in returns.columns}

        try:
            # Use correlation as features for clustering
            correlation_matrix = returns.corr()

            # Convert to distance matrix
            distance_matrix = 1 - correlation_matrix.abs()

            # Perform clustering
            n_clusters = min(5, len(returns.columns) // 2)
            if n_clusters < 2:
                return {symbol: 0 for symbol in returns.columns}

            # Use correlation values as features
            features = correlation_matrix.values

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)

            return {symbol: int(label) for symbol, label in zip(returns.columns, cluster_labels)}

        except Exception as e:
            self.logger.warning(f"Error in correlation clustering: {e}")
            return {symbol: 0 for symbol in returns.columns}

    def _calculate_asset_risk_score(
        self, symbol: str, metadata: Dict[str, Any], returns: pd.Series
    ) -> float:
        """Calculate comprehensive risk score for asset."""
        risk_score = 0.0

        # Volatility component (40% weight)
        if len(returns) > 20:
            volatility = returns.std() * np.sqrt(365)
            vol_score = min(1.0, volatility / 2.0)  # Normalize to 200% vol
            risk_score += vol_score * 0.4
        else:
            risk_score += 0.3  # Default for missing data

        # Market cap component (30% weight)
        market_cap = metadata.get("market_cap", 0)
        if market_cap < 100_000_000:  # < $100M
            risk_score += 0.3
        elif market_cap < 1_000_000_000:  # < $1B
            risk_score += 0.2
        elif market_cap < 10_000_000_000:  # < $10B
            risk_score += 0.1

        # Category-based risk (20% weight)
        category = metadata.get("category", "").lower()
        if "meme" in category:
            risk_score += 0.2
        elif "defi" in category:
            risk_score += 0.15
        elif "layer2" in category:
            risk_score += 0.1
        elif "stable" in category:
            risk_score += 0.02
        else:
            risk_score += 0.08

        # Age/maturity component (10% weight)
        age_days = metadata.get("age_days", 365)
        if age_days < 90:
            risk_score += 0.1
        elif age_days < 365:
            risk_score += 0.05

        return min(1.0, risk_score)

    def check_cluster_limits(
        self, proposed_allocations: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if proposed allocations violate cluster limits.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Calculate cluster exposures
        cluster_exposures = self.calculate_cluster_exposures(proposed_allocations)

        for cluster, exposure in cluster_exposures.items():
            limit = self.cluster_limits.get(cluster)
            if not limit:
                continue

            # Check allocation limit
            if exposure.total_allocation > limit.max_allocation:
                violations.append(
                    f"Cluster {cluster.value} allocation {exposure.total_allocation:.1%} "
                    f"exceeds limit {limit.max_allocation:.1%}"
                )

            # Check asset count limit
            if exposure.asset_count > limit.max_assets:
                violations.append(
                    f"Cluster {cluster.value} has {exposure.asset_count} assets "
                    f"exceeding limit {limit.max_assets}"
                )

            # Check correlation limit
            if exposure.avg_correlation > limit.max_correlation:
                violations.append(
                    f"Cluster {cluster.value} avg correlation {exposure.avg_correlation:.2f} "
                    f"exceeds limit {limit.max_correlation:.2f}"
                )

        return len(violations) == 0, violations

    def calculate_cluster_exposures(
        self, allocations: Dict[str, float]
    ) -> Dict[AssetCluster, ClusterExposure]:
        """Calculate current exposure to each cluster."""
        cluster_data = {}

        # Initialize cluster tracking
        for cluster in AssetCluster:
            cluster_data[cluster] = {"total_allocation": 0.0, "assets": [], "correlations": []}

        # Aggregate by cluster
        for symbol, allocation in allocations.items():
            if allocation <= 0:
                continue

            classification = self.asset_classifications.get(symbol)
            if not classification:
                continue

            # Add to primary cluster
            primary = classification.primary_cluster
            cluster_data[primary]["total_allocation"] += allocation
            cluster_data[primary]["assets"].append(symbol)

            # Add to secondary clusters (with reduced weight)
            for secondary in classification.secondary_clusters:
                cluster_data[secondary]["total_allocation"] += allocation * 0.3
                if symbol not in cluster_data[secondary]["assets"]:
                    cluster_data[secondary]["assets"].append(symbol)

        # Calculate exposures
        exposures = {}
        for cluster, data in cluster_data.items():
            # Calculate average correlation within cluster
            avg_correlation = self._calculate_cluster_correlation(data["assets"])

            # Calculate risk contribution
            risk_contribution = self._calculate_cluster_risk_contribution(
                data["assets"], allocations
            )

            # Get limit utilization
            limit = self.cluster_limits.get(cluster)
            limit_utilization = (
                data["total_allocation"] / limit.max_allocation
                if limit and limit.max_allocation > 0
                else 0.0
            )

            exposures[cluster] = ClusterExposure(
                cluster=cluster,
                total_allocation=data["total_allocation"],
                asset_count=len(data["assets"]),
                avg_correlation=avg_correlation,
                risk_contribution=risk_contribution,
                assets=data["assets"],
                limit_utilization=limit_utilization,
            )

        return exposures

    def _calculate_cluster_correlation(self, assets: List[str]) -> float:
        """Calculate average correlation within cluster."""
        if len(assets) < 2:
            return 0.0

        # Get latest correlation data
        correlations = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]

                # Look up correlation from history
                correlation = self._get_asset_correlation(asset1, asset2)
                if correlation is not None:
                    correlations.append(abs(correlation))

        return np.mean(correlations) if correlations else 0.0

    def _get_asset_correlation(self, asset1: str, asset2: str) -> Optional[float]:
        """Get correlation between two assets."""
        # This would ideally use real correlation data
        # For now, return estimated correlation based on cluster similarity

        class1 = self.asset_classifications.get(asset1)
        class2 = self.asset_classifications.get(asset2)

        if not class1 or not class2:
            return None

        # Same primary cluster = higher correlation
        if class1.primary_cluster == class2.primary_cluster:
            return 0.7

        # Same secondary cluster = moderate correlation
        if (
            class1.primary_cluster in class2.secondary_clusters
            or class2.primary_cluster in class1.secondary_clusters
        ):
            return 0.4

        # Same correlation group = moderate correlation
        if class1.correlation_group == class2.correlation_group:
            return 0.3

        # Different clusters = low correlation
        return 0.1

    def _calculate_cluster_risk_contribution(
        self, assets: List[str], allocations: Dict[str, float]
    ) -> float:
        """Calculate risk contribution of cluster to portfolio."""
        total_risk = 0.0

        for asset in assets:
            allocation = allocations.get(asset, 0.0)
            classification = self.asset_classifications.get(asset)

            if classification:
                risk_contribution = allocation * classification.risk_score
                total_risk += risk_contribution

        return total_risk

    def apply_cluster_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply cluster constraints to allocation and return adjusted allocations."""

        adjusted_allocations = allocations.copy()

        # Calculate current exposures
        cluster_exposures = self.calculate_cluster_exposures(adjusted_allocations)

        # Apply constraints iteratively
        max_iterations = 5
        for iteration in range(max_iterations):
            constraints_applied = False

            for cluster, exposure in cluster_exposures.items():
                limit = self.cluster_limits.get(cluster)
                if not limit:
                    continue

                # Check allocation constraint
                if exposure.total_allocation > limit.max_allocation:
                    # Scale down all assets in cluster proportionally
                    scale_factor = limit.max_allocation / exposure.total_allocation

                    for asset in exposure.assets:
                        if asset in adjusted_allocations:
                            adjusted_allocations[asset] *= scale_factor

                    constraints_applied = True

                # Check asset count constraint
                if exposure.asset_count > limit.max_assets:
                    # Remove lowest allocation assets from cluster
                    cluster_allocations = [
                        (asset, adjusted_allocations[asset])
                        for asset in exposure.assets
                        if asset in adjusted_allocations
                    ]

                    # Sort by allocation and keep only top assets
                    cluster_allocations.sort(key=lambda x: x[1], reverse=True)
                    assets_to_remove = cluster_allocations[limit.max_assets :]

                    for asset, _ in assets_to_remove:
                        adjusted_allocations[asset] = 0.0

                    constraints_applied = True

            if not constraints_applied:
                break

            # Recalculate exposures for next iteration
            cluster_exposures = self.calculate_cluster_exposures(adjusted_allocations)

        return adjusted_allocations

    def get_cluster_summary(self, allocations: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive cluster analysis summary."""

        cluster_exposures = self.calculate_cluster_exposures(allocations)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_allocation": sum(allocations.values()),
            "cluster_count": len([c for c in cluster_exposures.values() if c.asset_count > 0]),
            "cluster_exposures": {},
            "constraint_violations": [],
            "diversification_metrics": {},
        }

        # Cluster exposure details
        for cluster, exposure in cluster_exposures.items():
            if exposure.asset_count > 0:
                summary["cluster_exposures"][cluster.value] = {
                    "allocation": exposure.total_allocation,
                    "asset_count": exposure.asset_count,
                    "avg_correlation": exposure.avg_correlation,
                    "risk_contribution": exposure.risk_contribution,
                    "limit_utilization": exposure.limit_utilization,
                    "assets": exposure.assets,
                }

        # Check violations
        is_valid, violations = self.check_cluster_limits(allocations)
        summary["constraint_violations"] = violations

        # Diversification metrics
        total_assets = len([a for a in allocations.values() if a > 0])
        active_clusters = len([c for c in cluster_exposures.values() if c.asset_count > 0])

        summary["diversification_metrics"] = {
            "total_assets": total_assets,
            "active_clusters": active_clusters,
            "avg_assets_per_cluster": total_assets / max(1, active_clusters),
            "max_single_allocation": max(allocations.values()) if allocations else 0.0,
            "concentration_score": self._calculate_concentration_score(allocations),
        }

        return summary

    def _calculate_concentration_score(self, allocations: Dict[str, float]) -> float:
        """Calculate portfolio concentration score (0 = diversified, 1 = concentrated)."""
        if not allocations:
            return 0.0

        # Calculate Herfindahl-Hirschman Index
        weights = np.array(list(allocations.values()))
        weights = weights / weights.sum()  # Normalize
        hhi = np.sum(weights**2)

        # Convert to 0-1 scale (1/n = fully diversified, 1 = fully concentrated)
        n = len(weights)
        min_hhi = 1.0 / n
        concentration = (hhi - min_hhi) / (1.0 - min_hhi)

        return max(0.0, min(1.0, concentration))


def create_cluster_manager(
    max_cluster_allocation: float = 0.30, max_correlation_threshold: float = 0.70
) -> ClusterManager:
    """Create cluster manager with specified parameters."""
    return ClusterManager(max_cluster_allocation, max_correlation_threshold)
