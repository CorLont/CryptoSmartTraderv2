"""
Correlation and Cluster Management System

Advanced correlation monitoring with cluster caps and
correlation shock stress testing for portfolio protection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class CorrelationLevel(Enum):
    """Correlation risk levels"""

    LOW = "low"  # < 0.3
    MODERATE = "moderate"  # 0.3 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    EXTREME = "extreme"  # > 0.8


class ClusterType(Enum):
    """Asset cluster types"""

    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    DEFI = "defi"
    LAYER1 = "layer1"
    GAMING = "gaming"
    MEME = "meme"
    STABLECOIN = "stablecoin"


@dataclass
class CorrelationWindow:
    """Correlation calculation window"""

    lookback_days: int = 30
    min_observations: int = 20
    rolling_window: bool = True


@dataclass
class ClusterLimit:
    """Cluster exposure limit"""

    cluster_type: ClusterType
    max_weight: float = 0.15  # Max 15% per cluster
    max_positions: int = 5  # Max 5 positions per cluster
    current_weight: float = 0.0
    current_positions: int = 0

    @property
    def weight_utilization(self) -> float:
        return self.current_weight / self.max_weight if self.max_weight > 0 else 0.0

    @property
    def position_utilization(self) -> float:
        return self.current_positions / self.max_positions if self.max_positions > 0 else 0.0

    @property
    def is_at_limit(self) -> bool:
        return (
            self.current_weight >= self.max_weight or self.current_positions >= self.max_positions
        )


@dataclass
class CorrelationShockScenario:
    """Correlation shock test scenario"""

    name: str
    correlation_increase: float = 0.3  # Increase all correlations by 30%
    volatility_spike: float = 2.0  # 2x volatility increase
    duration_days: int = 5  # Shock duration
    affected_clusters: List[ClusterType] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Portfolio stress test result"""

    scenario_name: str
    max_drawdown: float
    portfolio_var: float
    correlation_metrics: Dict[str, float]
    cluster_impacts: Dict[str, float]
    risk_limit_breaches: List[str]
    recommended_actions: List[str]


class CorrelationManager:
    """
    Advanced correlation and cluster management system
    """

    def __init__(self):
        # Correlation parameters
        self.correlation_window = CorrelationWindow()
        self.correlation_threshold = 0.7  # Alert threshold
        self.max_cluster_correlation = 0.8  # Max avg correlation within cluster

        # Portfolio limits
        self.max_single_asset_weight = 0.05  # 5% max per asset
        self.max_total_crypto_weight = 0.25  # 25% max crypto allocation
        self.max_high_correlation_weight = 0.15  # 15% max in highly correlated assets

        # Cluster definitions and limits
        self.cluster_limits: Dict[ClusterType, ClusterLimit] = self._initialize_cluster_limits()
        self.asset_cluster_mapping: Dict[str, ClusterType] = {}

        # Correlation tracking
        self.correlation_history: List[Dict[str, Any]] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[datetime] = None

        # Risk monitoring
        self.correlation_alerts: List[Dict[str, Any]] = []
        self.cluster_breach_alerts: List[Dict[str, Any]] = []

    def _initialize_cluster_limits(self) -> Dict[ClusterType, ClusterLimit]:
        """Initialize default cluster limits"""

        return {
            ClusterType.LARGE_CAP: ClusterLimit(
                cluster_type=ClusterType.LARGE_CAP,
                max_weight=0.15,  # 15% in large caps (BTC, ETH)
                max_positions=3,
            ),
            ClusterType.MID_CAP: ClusterLimit(
                cluster_type=ClusterType.MID_CAP,
                max_weight=0.10,  # 10% in mid caps
                max_positions=5,
            ),
            ClusterType.SMALL_CAP: ClusterLimit(
                cluster_type=ClusterType.SMALL_CAP,
                max_weight=0.08,  # 8% in small caps
                max_positions=8,
            ),
            ClusterType.DEFI: ClusterLimit(
                cluster_type=ClusterType.DEFI,
                max_weight=0.12,  # 12% in DeFi
                max_positions=6,
            ),
            ClusterType.LAYER1: ClusterLimit(
                cluster_type=ClusterType.LAYER1,
                max_weight=0.10,  # 10% in Layer 1s
                max_positions=4,
            ),
            ClusterType.GAMING: ClusterLimit(
                cluster_type=ClusterType.GAMING,
                max_weight=0.06,  # 6% in gaming
                max_positions=4,
            ),
            ClusterType.MEME: ClusterLimit(
                cluster_type=ClusterType.MEME,
                max_weight=0.03,  # 3% in meme coins
                max_positions=3,
            ),
            ClusterType.STABLECOIN: ClusterLimit(
                cluster_type=ClusterType.STABLECOIN,
                max_weight=0.05,  # 5% in stablecoins
                max_positions=2,
            ),
        }

    def update_asset_clusters(self, asset_clusters: Dict[str, str]):
        """Update asset to cluster mapping"""

        try:
            cluster_mapping = {}

            for asset, cluster_name in asset_clusters.items():
                try:
                    cluster_type = ClusterType(cluster_name.lower())
                    cluster_mapping[asset] = cluster_type
                except ValueError:
                    logger.warning(f"Unknown cluster type: {cluster_name}")
                    cluster_mapping[asset] = ClusterType.MID_CAP  # Default

            self.asset_cluster_mapping = cluster_mapping
            logger.info(f"Updated cluster mapping for {len(cluster_mapping)} assets")

        except Exception as e:
            logger.error(f"Asset cluster update failed: {e}")

    def calculate_correlation_matrix(
        self, price_data: Dict[str, pd.DataFrame], window: Optional[CorrelationWindow] = None
    ) -> pd.DataFrame:
        """Calculate rolling correlation matrix for assets"""

        try:
            if not price_data:
                return pd.DataFrame()

            window = window or self.correlation_window

            # Prepare return data
            returns_data = {}

            for symbol, data in price_data.items():
                if "close" not in data.columns or len(data) < window.min_observations:
                    continue

                # Calculate returns
                returns = data["close"].pct_change().dropna()

                # Use last N days
                if len(returns) > window.lookback_days:
                    returns = returns.tail(window.lookback_days)

                if len(returns) >= window.min_observations:
                    returns_data[symbol] = returns

            if len(returns_data) < 2:
                logger.warning("Insufficient data for correlation calculation")
                return pd.DataFrame()

            # Align data by timestamp
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if len(returns_df) < window.min_observations:
                logger.warning("Insufficient aligned data for correlation")
                return pd.DataFrame()

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()

            # Store results
            self.correlation_matrix = correlation_matrix
            self.last_correlation_update = datetime.now()

            logger.info(f"Correlation matrix calculated for {len(correlation_matrix)} assets")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return pd.DataFrame()

    def identify_correlation_clusters(
        self, correlation_matrix: pd.DataFrame, cluster_threshold: float = 0.7
    ) -> Dict[int, List[str]]:
        """Identify asset clusters based on correlation"""

        try:
            if correlation_matrix.empty:
                return {}

            # Convert correlation to distance matrix
            distance_matrix = 1 - correlation_matrix.abs()

            # Hierarchical clustering
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method="ward")

            # Form clusters
            cluster_labels = fcluster(linkage_matrix, 1 - cluster_threshold, criterion="distance")

            # Group assets by cluster
            clusters = {}
            for i, asset in enumerate(correlation_matrix.index):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(asset)

            # Filter clusters with multiple assets
            significant_clusters = {
                cid: assets for cid, assets in clusters.items() if len(assets) > 1
            }

            logger.info(f"Identified {len(significant_clusters)} correlation clusters")
            return significant_clusters

        except Exception as e:
            logger.error(f"Cluster identification failed: {e}")
            return {}

    def check_correlation_limits(
        self, current_positions: Dict[str, float], correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check portfolio against correlation limits"""

        try:
            if correlation_matrix.empty or not current_positions:
                return {"status": "no_data"}

            violations = []
            warnings = []

            # Filter positions that are in correlation matrix
            relevant_positions = {
                symbol: weight
                for symbol, weight in current_positions.items()
                if symbol in correlation_matrix.index and weight > 0
            }

            if not relevant_positions:
                return {"status": "no_overlap"}

            # Check pairwise correlations
            for symbol1, weight1 in relevant_positions.items():
                for symbol2, weight2 in relevant_positions.items():
                    if symbol1 >= symbol2:  # Avoid duplicates
                        continue

                    correlation = correlation_matrix.loc[symbol1, symbol2]
                    combined_weight = weight1 + weight2

                    # High correlation warning
                    if abs(correlation) > self.correlation_threshold and combined_weight > 0.05:
                        warnings.append(
                            {
                                "type": "high_correlation",
                                "assets": [symbol1, symbol2],
                                "correlation": correlation,
                                "combined_weight": combined_weight,
                                "message": f"High correlation ({correlation:.2f}) between {symbol1} and {symbol2}",
                            }
                        )

                    # Extreme correlation violation
                    if abs(correlation) > 0.85 and combined_weight > 0.03:
                        violations.append(
                            {
                                "type": "extreme_correlation",
                                "assets": [symbol1, symbol2],
                                "correlation": correlation,
                                "combined_weight": combined_weight,
                                "severity": "high",
                            }
                        )

            # Check total high-correlation exposure
            high_corr_assets = set()
            for symbol1 in relevant_positions:
                for symbol2 in relevant_positions:
                    if symbol1 != symbol2:
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        if abs(correlation) > self.correlation_threshold:
                            high_corr_assets.add(symbol1)
                            high_corr_assets.add(symbol2)

            high_corr_exposure = sum(
                weight
                for symbol, weight in relevant_positions.items()
                if symbol in high_corr_assets
            )

            if high_corr_exposure > self.max_high_correlation_weight:
                violations.append(
                    {
                        "type": "total_correlation_exposure",
                        "exposure": high_corr_exposure,
                        "limit": self.max_high_correlation_weight,
                        "severity": "medium",
                    }
                )

            return {
                "status": "checked",
                "violations": violations,
                "warnings": warnings,
                "high_correlation_exposure": high_corr_exposure,
                "total_positions_checked": len(relevant_positions),
            }

        except Exception as e:
            logger.error(f"Correlation limit check failed: {e}")
            return {"status": "error", "message": str(e)}

    def check_cluster_limits(self, current_positions: Dict[str, float]) -> Dict[str, Any]:
        """Check portfolio against cluster limits"""

        try:
            if not self.asset_cluster_mapping:
                return {"status": "no_cluster_mapping"}

            # Reset cluster tracking
            for cluster_limit in self.cluster_limits.values():
                cluster_limit.current_weight = 0.0
                cluster_limit.current_positions = 0

            # Calculate current cluster exposures
            for symbol, weight in current_positions.items():
                if weight <= 0:
                    continue

                cluster_type = self.asset_cluster_mapping.get(symbol)
                if cluster_type and cluster_type in self.cluster_limits:
                    cluster_limit = self.cluster_limits[cluster_type]
                    cluster_limit.current_weight += weight
                    cluster_limit.current_positions += 1

            # Check for violations
            violations = []
            warnings = []

            for cluster_type, cluster_limit in self.cluster_limits.items():
                # Weight violations
                if cluster_limit.current_weight > cluster_limit.max_weight:
                    violations.append(
                        {
                            "type": "cluster_weight_violation",
                            "cluster": cluster_type.value,
                            "current_weight": cluster_limit.current_weight,
                            "max_weight": cluster_limit.max_weight,
                            "excess": cluster_limit.current_weight - cluster_limit.max_weight,
                            "severity": "high",
                        }
                    )
                elif cluster_limit.current_weight > cluster_limit.max_weight * 0.85:
                    warnings.append(
                        {
                            "type": "cluster_weight_warning",
                            "cluster": cluster_type.value,
                            "current_weight": cluster_limit.current_weight,
                            "max_weight": cluster_limit.max_weight,
                            "utilization": cluster_limit.weight_utilization,
                        }
                    )

                # Position count violations
                if cluster_limit.current_positions > cluster_limit.max_positions:
                    violations.append(
                        {
                            "type": "cluster_position_violation",
                            "cluster": cluster_type.value,
                            "current_positions": cluster_limit.current_positions,
                            "max_positions": cluster_limit.max_positions,
                            "excess": cluster_limit.current_positions - cluster_limit.max_positions,
                            "severity": "medium",
                        }
                    )

            # Generate cluster summary
            cluster_summary = {}
            for cluster_type, cluster_limit in self.cluster_limits.items():
                cluster_summary[cluster_type.value] = {
                    "weight": cluster_limit.current_weight,
                    "max_weight": cluster_limit.max_weight,
                    "positions": cluster_limit.current_positions,
                    "max_positions": cluster_limit.max_positions,
                    "weight_utilization": cluster_limit.weight_utilization,
                    "position_utilization": cluster_limit.position_utilization,
                    "at_limit": cluster_limit.is_at_limit,
                }

            return {
                "status": "checked",
                "violations": violations,
                "warnings": warnings,
                "cluster_summary": cluster_summary,
                "total_clusters": len(self.cluster_limits),
            }

        except Exception as e:
            logger.error(f"Cluster limit check failed: {e}")
            return {"status": "error", "message": str(e)}

    def stress_test_correlation_shock(
        self,
        current_positions: Dict[str, float],
        scenarios: List[CorrelationShockScenario],
        historical_volatility: Dict[str, float],
    ) -> Dict[str, StressTestResult]:
        """Stress test portfolio against correlation shock scenarios"""

        try:
            if not self.correlation_matrix or self.correlation_matrix.empty:
                logger.error("No correlation matrix available for stress testing")
                return {}

            stress_results = {}

            for scenario in scenarios:
                logger.info(f"Running correlation shock test: {scenario.name}")

                # Create shocked correlation matrix
                shocked_corr_matrix = self.correlation_matrix.copy()

                # Apply correlation shock
                matrix_values = shocked_corr_matrix.values.copy()
                for i in range(len(matrix_values)):
                    for j in range(len(matrix_values)):
                        if i != j:
                            current_corr = matrix_values[i, j]
                            # Increase correlation but cap at 0.95
                            shocked_corr = min(
                                0.95, abs(current_corr) + scenario.correlation_increase
                            )
                            # Maintain sign of original correlation
                            if current_corr < 0:
                                shocked_corr = -shocked_corr
                            matrix_values[i, j] = shocked_corr

                # Update the DataFrame with new values
                shocked_corr_matrix = pd.DataFrame(
                    matrix_values,
                    index=shocked_corr_matrix.index,
                    columns=shocked_corr_matrix.columns,
                )

                # Calculate portfolio metrics under shock
                portfolio_var = self._calculate_portfolio_var(
                    current_positions,
                    shocked_corr_matrix,
                    historical_volatility,
                    scenario.volatility_spike,
                )

                # Estimate maximum drawdown
                max_drawdown = min(0.5, portfolio_var * 2.0)  # Rough estimate

                # Check risk limit breaches
                risk_breaches = []
                if max_drawdown > 0.1:  # 10% max DD policy
                    risk_breaches.append(f"Max drawdown ({max_drawdown:.1%}) exceeds 10% limit")

                if portfolio_var > 0.2:  # 20% daily VaR limit
                    risk_breaches.append(f"Portfolio VaR ({portfolio_var:.1%}) exceeds 20% limit")

                # Calculate cluster impacts
                cluster_impacts = self._calculate_cluster_impacts(
                    current_positions, shocked_corr_matrix
                )

                # Generate recommendations
                recommendations = self._generate_stress_recommendations(
                    max_drawdown, portfolio_var, risk_breaches
                )

                # Correlation metrics
                avg_correlation = shocked_corr_matrix.values[
                    np.triu_indices_from(shocked_corr_matrix.values, k=1)
                ].mean()
                max_correlation = shocked_corr_matrix.values[
                    np.triu_indices_from(shocked_corr_matrix.values, k=1)
                ].max()

                correlation_metrics = {
                    "average_correlation": avg_correlation,
                    "max_correlation": max_correlation,
                    "correlation_increase": scenario.correlation_increase,
                }

                stress_results[scenario.name] = StressTestResult(
                    scenario_name=scenario.name,
                    max_drawdown=max_drawdown,
                    portfolio_var=portfolio_var,
                    correlation_metrics=correlation_metrics,
                    cluster_impacts=cluster_impacts,
                    risk_limit_breaches=risk_breaches,
                    recommended_actions=recommendations,
                )

            return stress_results

        except Exception as e:
            logger.error(f"Correlation shock stress test failed: {e}")
            return {}

    def _calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        volatilities: Dict[str, float],
        vol_multiplier: float = 1.0,
    ) -> float:
        """Calculate portfolio Value at Risk"""

        try:
            # Filter positions in correlation matrix
            relevant_positions = {
                symbol: weight
                for symbol, weight in positions.items()
                if symbol in correlation_matrix.index and weight > 0
            }

            if len(relevant_positions) < 2:
                return 0.0

            # Create weight vector
            symbols = list(relevant_positions.keys())
            weights = np.array([relevant_positions[symbol] for symbol in symbols])

            # Create volatility vector
            vols = np.array([volatilities.get(symbol, 0.02) * vol_multiplier for symbol in symbols])

            # Get correlation submatrix
            corr_sub = correlation_matrix.loc[symbols, symbols].values

            # Calculate covariance matrix
            cov_matrix = np.outer(vols, vols) * corr_sub

            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)

            # VaR (95% confidence, 1-day)
            var_95 = portfolio_vol * 1.645  # 95% quantile of normal distribution

            return var_95

        except Exception as e:
            logger.error(f"Portfolio VaR calculation failed: {e}")
            return 0.0

    def _calculate_cluster_impacts(
        self, positions: Dict[str, float], correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate impact by cluster under correlation shock"""

        try:
            cluster_impacts = {}

            for cluster_type in ClusterType:
                cluster_assets = [
                    symbol
                    for symbol, cluster in self.asset_cluster_mapping.items()
                    if cluster == cluster_type and symbol in positions and positions[symbol] > 0
                ]

                if len(cluster_assets) < 2:
                    cluster_impacts[cluster_type.value] = 0.0
                    continue

                # Calculate average correlation within cluster
                correlations = []
                for i, asset1 in enumerate(cluster_assets):
                    for asset2 in cluster_assets[i + 1 :]:
                        if (
                            asset1 in correlation_matrix.index
                            and asset2 in correlation_matrix.index
                        ):
                            correlations.append(correlation_matrix.loc[asset1, asset2])

                avg_correlation = np.mean(correlations) if correlations else 0.0

                # Weight impact by cluster exposure
                cluster_weight = sum(positions.get(asset, 0) for asset in cluster_assets)
                impact = avg_correlation * cluster_weight

                cluster_impacts[cluster_type.value] = impact

            return cluster_impacts

        except Exception as e:
            logger.error(f"Cluster impact calculation failed: {e}")
            return {}

    def _generate_stress_recommendations(
        self, max_dd: float, var: float, breaches: List[str]
    ) -> List[str]:
        """Generate recommendations based on stress test results"""

        recommendations = []

        if max_dd > 0.1:
            recommendations.append("Reduce position sizes to limit maximum drawdown")

        if var > 0.15:
            recommendations.append("Diversify across uncorrelated assets")

        if max_dd > 0.15:
            recommendations.append("Consider hedging with inverse correlated assets")

        if len(breaches) > 2:
            recommendations.append("Implement emergency risk reduction protocols")

        if not recommendations:
            recommendations.append("Portfolio correlation risk within acceptable limits")

        return recommendations

    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get comprehensive correlation and cluster summary"""

        try:
            summary = {
                "last_update": self.last_correlation_update.isoformat()
                if self.last_correlation_update
                else None,
                "correlation_matrix_size": len(self.correlation_matrix)
                if self.correlation_matrix is not None
                else 0,
                "cluster_mapping_size": len(self.asset_cluster_mapping),
                "correlation_threshold": self.correlation_threshold,
                "max_cluster_correlation": self.max_cluster_correlation,
            }

            # Correlation statistics
            if self.correlation_matrix is not None and not self.correlation_matrix.empty:
                corr_values = self.correlation_matrix.values[
                    np.triu_indices_from(self.correlation_matrix.values, k=1)
                ]
                summary["correlation_stats"] = {
                    "mean": float(np.mean(corr_values)),
                    "median": float(np.median(corr_values)),
                    "max": float(np.max(corr_values)),
                    "min": float(np.min(corr_values)),
                    "std": float(np.std(corr_values)),
                }

            # Cluster limits summary
            cluster_limits_summary = {}
            for cluster_type, limit in self.cluster_limits.items():
                cluster_limits_summary[cluster_type.value] = {
                    "max_weight": limit.max_weight,
                    "max_positions": limit.max_positions,
                    "current_weight": limit.current_weight,
                    "current_positions": limit.current_positions,
                }

            summary["cluster_limits"] = cluster_limits_summary

            return summary

        except Exception as e:
            logger.error(f"Correlation summary failed: {e}")
            return {"status": "error", "message": str(e)}
