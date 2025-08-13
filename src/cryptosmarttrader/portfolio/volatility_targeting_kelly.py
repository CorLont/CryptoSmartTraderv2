#!/usr/bin/env python3
"""
Volatility Targeting & Kelly Sizing System
Implementeert: sizing = fractional Kelly × vol-target met correlatie-based asset/cluster caps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
import threading
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from ..observability.unified_metrics import UnifiedMetrics


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    VERY_LOW = "very_low"  # < 10% annualized
    LOW = "low"  # 10-20% annualized
    MEDIUM = "medium"  # 20-40% annualized
    HIGH = "high"  # 40-70% annualized
    VERY_HIGH = "very_high"  # > 70% annualized


@dataclass
class AssetMetrics:
    """Asset-specific metrics for sizing calculations."""

    symbol: str

    # Price and return data
    current_price: float
    returns_1d: List[float] = field(default_factory=list)
    returns_7d: List[float] = field(default_factory=list)
    returns_30d: List[float] = field(default_factory=list)

    # Volatility metrics
    realized_vol_1d: float = 0.0
    realized_vol_7d: float = 0.0
    realized_vol_30d: float = 0.0
    implied_vol: Optional[float] = None
    vol_regime: VolatilityRegime = VolatilityRegime.MEDIUM

    # Signal metrics
    expected_return: float = 0.0
    signal_confidence: float = 0.0
    signal_horizon_days: int = 1

    # Risk metrics
    max_drawdown_30d: float = 0.0
    sharpe_ratio_30d: float = 0.0
    correlation_to_portfolio: float = 0.0

    # Market structure
    liquidity_score: float = 0.0
    volume_24h: float = 0.0
    spread_bps: float = 0.0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterMetrics:
    """Cluster-specific correlation and exposure metrics."""

    cluster_id: str
    cluster_name: str

    # Assets in cluster
    assets: List[str] = field(default_factory=list)

    # Correlation matrix
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_correlation: float = 0.0
    max_correlation: float = 0.0

    # Exposure metrics
    total_exposure: float = 0.0
    target_exposure: float = 0.0
    max_exposure: float = 0.0

    # Risk metrics
    cluster_beta: float = 1.0
    cluster_vol: float = 0.0
    diversification_ratio: float = 1.0

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SizingResult:
    """Result from volatility targeting & Kelly sizing calculation."""

    symbol: str
    signal_confidence: float
    expected_return: float
    kelly_fraction: float
    fractional_kelly: float
    target_volatility: float
    realized_volatility: float
    vol_scaling_factor: float
    base_size: float
    vol_adjusted_size: float
    correlation_adjusted_size: float
    final_position_size: float

    # Default values
    kelly_multiplier: float = 0.25  # Default 25% of Kelly
    asset_cap_applied: bool = False
    cluster_cap_applied: bool = False
    vol_limit_applied: bool = False
    asset_cap_pct: float = 2.0
    cluster_cap_pct: float = 20.0
    max_leverage: float = 3.0
    calculation_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class VolatilityTargetingKelly:
    """
    Volatility Targeting & Kelly Sizing System.

    Formule: sizing = fractional_kelly × volatility_scaling × correlation_adjustment

    Features:
    - Fractional Kelly criterion (default 25% of Kelly)
    - Volatility targeting (15% default target)
    - Asset-level caps (2% per asset)
    - Cluster-level caps (20% per cluster)
    - Correlation-based adjustment
    - Regime-adaptive scaling
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        kelly_fraction: float = 0.25,
        max_asset_exposure_pct: float = 2.0,
        max_cluster_exposure_pct: float = 20.0,
        correlation_threshold: float = 0.7,
        max_leverage: float = 3.0,
    ):
        """Initialize volatility targeting & Kelly sizing system."""

        self.logger = get_logger("volatility_targeting_kelly")

        # Core parameters
        self.target_volatility = target_volatility
        self.kelly_fraction = kelly_fraction
        self.max_asset_exposure_pct = max_asset_exposure_pct
        self.max_cluster_exposure_pct = max_cluster_exposure_pct
        self.correlation_threshold = correlation_threshold
        self.max_leverage = max_leverage

        # Asset and cluster tracking
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        self.cluster_metrics: Dict[str, ClusterMetrics] = {}

        # Portfolio state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = 100000.0  # Default
        self.portfolio_volatility: float = 0.0

        # Correlation analysis
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_clusters: Dict[str, List[str]] = {}

        # Performance tracking
        self.sizing_history: List[SizingResult] = []

        # Metrics and monitoring
        self.metrics = UnifiedMetrics("volatility_targeting_kelly")

        # Threading
        self._lock = threading.RLock()

        # Data persistence
        self.data_path = Path("data/volatility_targeting_kelly")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "VolatilityTargetingKelly initialized",
            target_volatility=target_volatility,
            kelly_fraction=kelly_fraction,
            max_asset_exposure_pct=max_asset_exposure_pct,
            max_cluster_exposure_pct=max_cluster_exposure_pct,
            correlation_threshold=correlation_threshold,
        )

    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        expected_return: float,
        current_price: float,
        portfolio_value: float,
    ) -> SizingResult:
        """
        Calculate optimal position size using volatility targeting & Kelly sizing.

        Formula: sizing = fractional_kelly × vol_target_scaling × correlation_adjustment
        """

        calc_start = time.time()

        with self._lock:
            try:
                # Initialize result
                result = SizingResult(
                    symbol=symbol,
                    signal_confidence=signal_confidence,
                    expected_return=expected_return,
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    target_volatility=self.target_volatility,
                    realized_volatility=0.0,
                    vol_scaling_factor=1.0,
                    base_size=0.0,
                    vol_adjusted_size=0.0,
                    correlation_adjusted_size=0.0,
                    final_position_size=0.0,
                    kelly_multiplier=self.kelly_fraction,
                )

                # Get or create asset metrics
                asset_metrics = self._get_asset_metrics(symbol, current_price)

                # Step 1: Calculate Kelly fraction
                kelly_result = self._calculate_kelly_fraction(
                    expected_return, asset_metrics, signal_confidence
                )
                result.kelly_fraction = kelly_result["kelly_fraction"]
                result.fractional_kelly = kelly_result["fractional_kelly"]

                # Step 2: Apply volatility targeting
                vol_result = self._apply_volatility_targeting(
                    symbol, asset_metrics, result.fractional_kelly
                )
                result.realized_volatility = vol_result["realized_volatility"]
                result.vol_scaling_factor = vol_result["vol_scaling_factor"]
                result.vol_adjusted_size = vol_result["vol_adjusted_size"]

                # Step 3: Apply correlation adjustments
                corr_result = self._apply_correlation_adjustment(
                    symbol, result.vol_adjusted_size, portfolio_value
                )
                result.correlation_adjusted_size = corr_result["correlation_adjusted_size"]

                # Step 4: Apply caps and limits
                limit_result = self._apply_caps_and_limits(
                    symbol, result.correlation_adjusted_size, portfolio_value
                )
                result.final_position_size = limit_result["final_size"]
                result.asset_cap_applied = limit_result["asset_cap_applied"]
                result.cluster_cap_applied = limit_result["cluster_cap_applied"]
                result.vol_limit_applied = limit_result["vol_limit_applied"]
                result.warnings.extend(limit_result["warnings"])

                # Set caps and limits
                result.asset_cap_pct = self.max_asset_exposure_pct
                result.cluster_cap_pct = self.max_cluster_exposure_pct
                result.max_leverage = self.max_leverage

                # Record base size for reference
                result.base_size = result.fractional_kelly

                # Finalize calculation
                result.calculation_time_ms = (time.time() - calc_start) * 1000

                # Log result
                self._log_sizing_result(result)

                # Update tracking
                self._update_sizing_history(result)

                return result

            except Exception as e:
                calc_time = (time.time() - calc_start) * 1000

                self.logger.error(
                    "Position sizing calculation failed",
                    symbol=symbol,
                    error=str(e),
                    calculation_time_ms=calc_time,
                )

                # Return minimal size on error
                return SizingResult(
                    symbol=symbol,
                    signal_confidence=signal_confidence,
                    expected_return=expected_return,
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    target_volatility=self.target_volatility,
                    realized_volatility=0.0,
                    vol_scaling_factor=0.0,
                    base_size=0.0,
                    vol_adjusted_size=0.0,
                    correlation_adjusted_size=0.0,
                    final_position_size=0.0,
                    warnings=[f"Calculation error: {str(e)}"],
                    calculation_time_ms=calc_time,
                )

    def _calculate_kelly_fraction(
        self, expected_return: float, asset_metrics: AssetMetrics, signal_confidence: float
    ) -> Dict[str, float]:
        """Calculate Kelly fraction with confidence adjustment."""

        # Use realized volatility as risk measure
        volatility = max(asset_metrics.realized_vol_30d, 0.01)  # Min 1% vol

        # Basic Kelly formula: f = (μ - r) / σ²
        # Simplified: f = expected_return / volatility²
        kelly_fraction = expected_return / (volatility**2)

        # Adjust for signal confidence
        confidence_adjusted_kelly = kelly_fraction * signal_confidence

        # Apply fractional Kelly (default 25%)
        fractional_kelly = confidence_adjusted_kelly * self.kelly_fraction

        # Cap Kelly fraction at reasonable levels
        fractional_kelly = np.clip(fractional_kelly, -0.5, 0.5)  # Max 50% Kelly

        return {
            "kelly_fraction": kelly_fraction,
            "fractional_kelly": fractional_kelly,
            "confidence_adjustment": signal_confidence,
            "volatility_used": volatility,
        }

    def _apply_volatility_targeting(
        self, symbol: str, asset_metrics: AssetMetrics, base_size: float
    ) -> Dict[str, float]:
        """Apply volatility targeting to position size."""

        # Get realized volatility
        realized_vol = asset_metrics.realized_vol_30d
        if realized_vol <= 0:
            realized_vol = 0.20  # Default 20% if no data

        # Calculate volatility scaling factor
        vol_scaling_factor = self.target_volatility / realized_vol

        # Apply regime-based adjustment
        regime_adjustment = self._get_regime_adjustment(asset_metrics.vol_regime)
        vol_scaling_factor *= regime_adjustment

        # Cap scaling factor to prevent extreme leverage
        vol_scaling_factor = np.clip(vol_scaling_factor, 0.1, 5.0)

        # Apply to base size
        vol_adjusted_size = base_size * vol_scaling_factor

        return {
            "realized_volatility": realized_vol,
            "vol_scaling_factor": vol_scaling_factor,
            "vol_adjusted_size": vol_adjusted_size,
            "regime_adjustment": regime_adjustment,
        }

    def _apply_correlation_adjustment(
        self, symbol: str, base_size: float, portfolio_value: float
    ) -> Dict[str, float]:
        """Apply correlation-based position adjustment."""

        # Get asset's cluster
        cluster_id = self._get_asset_cluster(symbol)

        if not cluster_id or cluster_id not in self.cluster_metrics:
            # No correlation data - return unchanged
            return {
                "correlation_adjusted_size": base_size,
                "cluster_correlation": 0.0,
                "adjustment_factor": 1.0,
            }

        cluster = self.cluster_metrics[cluster_id]

        # Calculate correlation adjustment factor
        avg_correlation = cluster.avg_correlation

        if avg_correlation > self.correlation_threshold:
            # High correlation - reduce size
            correlation_penalty = 1.0 - (avg_correlation - self.correlation_threshold) * 0.5
            adjustment_factor = max(correlation_penalty, 0.5)  # Min 50% size
        else:
            # Low correlation - no penalty
            adjustment_factor = 1.0

        # Apply diversification bonus for low correlation
        if avg_correlation < 0.3:
            diversification_bonus = 1.1  # 10% bonus
            adjustment_factor *= diversification_bonus

        correlation_adjusted_size = base_size * adjustment_factor

        return {
            "correlation_adjusted_size": correlation_adjusted_size,
            "cluster_correlation": avg_correlation,
            "adjustment_factor": adjustment_factor,
        }

    def _apply_caps_and_limits(
        self, symbol: str, base_size: float, portfolio_value: float
    ) -> Dict[str, Any]:
        """Apply asset and cluster caps and other limits."""

        warnings = []
        final_size = base_size
        asset_cap_applied = False
        cluster_cap_applied = False
        vol_limit_applied = False

        # Calculate position value
        asset_metrics = self.asset_metrics.get(symbol)
        if not asset_metrics:
            return {
                "final_size": 0.0,
                "asset_cap_applied": False,
                "cluster_cap_applied": False,
                "vol_limit_applied": False,
                "warnings": ["No asset metrics available"],
            }

        position_value = abs(final_size) * asset_metrics.current_price

        # Asset-level cap (2% default)
        max_asset_value = portfolio_value * (self.max_asset_exposure_pct / 100)
        if position_value > max_asset_value:
            final_size = (max_asset_value / asset_metrics.current_price) * np.sign(final_size)
            asset_cap_applied = True
            warnings.append(f"Asset cap applied: {self.max_asset_exposure_pct}%")

        # Cluster-level cap (20% default)
        cluster_id = self._get_asset_cluster(symbol)
        if cluster_id and cluster_id in self.cluster_metrics:
            cluster = self.cluster_metrics[cluster_id]
            current_cluster_exposure = cluster.total_exposure
            new_position_value = abs(final_size) * asset_metrics.current_price
            total_cluster_exposure = current_cluster_exposure + new_position_value

            max_cluster_value = portfolio_value * (self.max_cluster_exposure_pct / 100)
            if total_cluster_exposure > max_cluster_value:
                available_cluster_space = max_cluster_value - current_cluster_exposure
                if available_cluster_space > 0:
                    final_size = (available_cluster_space / asset_metrics.current_price) * np.sign(
                        final_size
                    )
                else:
                    final_size = 0.0
                cluster_cap_applied = True
                warnings.append(f"Cluster cap applied: {self.max_cluster_exposure_pct}%")

        # Volatility limit (max 3x leverage default)
        position_leverage = (abs(final_size) * asset_metrics.current_price) / portfolio_value
        if position_leverage > self.max_leverage:
            final_size = (
                portfolio_value * self.max_leverage / asset_metrics.current_price
            ) * np.sign(final_size)
            vol_limit_applied = True
            warnings.append(f"Leverage cap applied: {self.max_leverage}x")

        return {
            "final_size": final_size,
            "asset_cap_applied": asset_cap_applied,
            "cluster_cap_applied": cluster_cap_applied,
            "vol_limit_applied": vol_limit_applied,
            "warnings": warnings,
        }

    def _get_asset_metrics(self, symbol: str, current_price: float) -> AssetMetrics:
        """Get or create asset metrics."""

        if symbol not in self.asset_metrics:
            # Create new asset metrics with mock data
            self.asset_metrics[symbol] = AssetMetrics(
                symbol=symbol,
                current_price=current_price,
                realized_vol_30d=0.25,  # Default 25% vol for crypto
                vol_regime=VolatilityRegime.MEDIUM,
                liquidity_score=0.8,
                volume_24h=1000000.0,  # $1M default
                spread_bps=10.0,
            )
        else:
            # Update current price
            self.asset_metrics[symbol].current_price = current_price
            self.asset_metrics[symbol].last_updated = datetime.now()

        return self.asset_metrics[symbol]

    def _get_asset_cluster(self, symbol: str) -> Optional[str]:
        """Get cluster ID for asset."""

        # Simple clustering based on symbol patterns
        if "BTC" in symbol.upper():
            return "btc_cluster"
        elif "ETH" in symbol.upper():
            return "eth_cluster"
        elif any(alt in symbol.upper() for alt in ["ADA", "DOT", "LINK", "UNI"]):
            return "altcoin_cluster"
        elif any(stable in symbol.upper() for stable in ["USDT", "USDC", "DAI"]):
            return "stablecoin_cluster"
        else:
            return "other_cluster"

    def _get_regime_adjustment(self, regime: VolatilityRegime) -> float:
        """Get regime-based sizing adjustment."""

        regime_adjustments = {
            VolatilityRegime.VERY_LOW: 1.2,  # 20% increase in low vol
            VolatilityRegime.LOW: 1.1,  # 10% increase
            VolatilityRegime.MEDIUM: 1.0,  # No adjustment
            VolatilityRegime.HIGH: 0.8,  # 20% decrease in high vol
            VolatilityRegime.VERY_HIGH: 0.6,  # 40% decrease
        }

        return regime_adjustments.get(regime, 1.0)

    def update_correlation_matrix(self, correlation_data: Dict[str, Dict[str, float]]) -> None:
        """Update correlation matrix and cluster metrics."""

        with self._lock:
            # Convert to pandas DataFrame
            symbols = list(correlation_data.keys())
            correlation_matrix = np.zeros((len(symbols), len(symbols)))

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    correlation_matrix[i, j] = correlation_data[symbol1].get(symbol2, 0.0)

            self.correlation_matrix = pd.DataFrame(
                correlation_matrix, index=symbols, columns=symbols
            )

            # Update cluster metrics
            self._update_cluster_metrics()

            self.logger.info(
                "Correlation matrix updated",
                symbols_count=len(symbols),
                avg_correlation=np.mean(correlation_matrix[correlation_matrix != 1.0]),
            )

    def _update_cluster_metrics(self) -> None:
        """Update cluster metrics based on current correlations."""

        if self.correlation_matrix is None:
            return

        # Group symbols by cluster
        cluster_groups = {}
        for symbol in self.correlation_matrix.index:
            cluster_id = self._get_asset_cluster(symbol)
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(symbol)

        # Calculate cluster metrics
        for cluster_id, symbols in cluster_groups.items():
            if len(symbols) < 2:
                continue

            # Get correlation submatrix for cluster
            cluster_corr = self.correlation_matrix.loc[symbols, symbols]

            # Calculate cluster statistics
            cluster_correlations = cluster_corr.values[
                np.triu_indices_from(cluster_corr.values, k=1)
            ]
            avg_correlation = np.mean(cluster_correlations)
            max_correlation = np.max(cluster_correlations)

            # Create or update cluster metrics
            if cluster_id not in self.cluster_metrics:
                self.cluster_metrics[cluster_id] = ClusterMetrics(
                    cluster_id=cluster_id, cluster_name=cluster_id.replace("_", " ").title()
                )

            cluster = self.cluster_metrics[cluster_id]
            cluster.assets = symbols
            cluster.correlation_matrix = cluster_corr.values
            cluster.avg_correlation = avg_correlation
            cluster.max_correlation = max_correlation
            cluster.last_updated = datetime.now()

    def _log_sizing_result(self, result: SizingResult) -> None:
        """Log position sizing result."""

        self.logger.info(
            "Position size calculated",
            symbol=result.symbol,
            signal_confidence=result.signal_confidence,
            expected_return=result.expected_return,
            kelly_fraction=result.kelly_fraction,
            fractional_kelly=result.fractional_kelly,
            vol_scaling=result.vol_scaling_factor,
            final_size=result.final_position_size,
            caps_applied=result.asset_cap_applied or result.cluster_cap_applied,
            calculation_time_ms=result.calculation_time_ms,
        )

    def _update_sizing_history(self, result: SizingResult) -> None:
        """Update sizing history for analysis."""

        self.sizing_history.append(result)

        # Keep last 1000 calculations
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]

    def get_sizing_statistics(self) -> Dict[str, Any]:
        """Get sizing system statistics."""

        if not self.sizing_history:
            return {
                "total_calculations": 0,
                "recent_calculations": 0,
                "avg_calculation_time_ms": 0.0,
                "avg_kelly_fraction": 0.0,
                "avg_fractional_kelly": 0.0,
                "avg_vol_scaling": 0.0,
                "avg_final_size": 0.0,
                "caps_applied_pct": 0.0,
                "target_volatility": self.target_volatility,
                "kelly_fraction": self.kelly_fraction,
                "max_asset_exposure_pct": self.max_asset_exposure_pct,
                "max_cluster_exposure_pct": self.max_cluster_exposure_pct,
                "assets_tracked": 0,
                "clusters_tracked": 0,
            }

        recent_results = self.sizing_history[-100:]  # Last 100 calculations

        return {
            "total_calculations": len(self.sizing_history),
            "recent_calculations": len(recent_results),
            "avg_calculation_time_ms": np.mean([r.calculation_time_ms for r in recent_results]),
            "avg_kelly_fraction": np.mean([r.kelly_fraction for r in recent_results]),
            "avg_fractional_kelly": np.mean([r.fractional_kelly for r in recent_results]),
            "avg_vol_scaling": np.mean([r.vol_scaling_factor for r in recent_results]),
            "avg_final_size": np.mean([abs(r.final_position_size) for r in recent_results]),
            "caps_applied_pct": np.mean(
                [r.asset_cap_applied or r.cluster_cap_applied for r in recent_results]
            )
            * 100,
            "target_volatility": self.target_volatility,
            "kelly_fraction": self.kelly_fraction,
            "max_asset_exposure_pct": self.max_asset_exposure_pct,
            "max_cluster_exposure_pct": self.max_cluster_exposure_pct,
            "assets_tracked": len(self.asset_metrics),
            "clusters_tracked": len(self.cluster_metrics),
        }
