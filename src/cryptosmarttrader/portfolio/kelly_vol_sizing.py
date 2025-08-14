"""
Kelly Vol-Targeting & Portfolio Sizing System
Fractional Kelly × vol-target + cluster/correlatie-caps + regime throttling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    CHOP = "chop"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class AssetMetrics:
    """Asset-specific metrics voor sizing"""
    symbol: str
    expected_return: float  # Annualized
    volatility: float  # Annualized
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win_loss_ratio: float
    correlation_to_market: float
    cluster_id: str  # "crypto_large", "crypto_alt", "stablecoins", etc.


@dataclass
class ClusterLimits:
    """Cluster exposure limits"""
    cluster_id: str
    max_exposure_pct: float
    max_correlation: float
    assets: List[str] = field(default_factory=list)
    current_exposure: float = 0.0


@dataclass
class SizingParameters:
    """Portfolio sizing parameters"""
    fractional_kelly: float = 0.25  # 25% of full Kelly
    vol_target_annual: float = 0.20  # 20% annual vol target
    max_position_size: float = 0.10  # 10% max per position
    max_cluster_exposure: float = 0.30  # 30% max per cluster
    correlation_threshold: float = 0.7  # High correlation threshold
    regime_throttle_factors: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.TREND: 1.0,
        MarketRegime.MEAN_REVERSION: 0.8,
        MarketRegime.CHOP: 0.5,
        MarketRegime.HIGH_VOL: 0.3,
        MarketRegime.CRISIS: 0.1
    })


@dataclass
class PositionSize:
    """Position sizing result"""
    symbol: str
    target_size_pct: float
    target_size_usd: float
    kelly_size_pct: float
    vol_adjusted_size_pct: float
    regime_adjusted_size_pct: float
    cluster_adjusted_size_pct: float
    final_size_pct: float
    reasoning: List[str] = field(default_factory=list)


class KellyVolSizer:
    """
    Advanced portfolio sizing met Kelly criterion, vol-targeting, 
    cluster caps en regime-aware throttling
    """
    
    def __init__(self, sizing_params: Optional[SizingParameters] = None):
        self.sizing_params = sizing_params or SizingParameters()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.total_equity = 100000.0
        self.current_positions: Dict[str, float] = {}  # symbol -> size_usd
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        self.cluster_limits: Dict[str, ClusterLimits] = {}
        self.current_regime = MarketRegime.TREND
        
        # Correlation matrix
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update = 0.0
        
        self._initialize_cluster_limits()
    
    def _initialize_cluster_limits(self):
        """Initialize default cluster limits"""
        clusters = {
            "crypto_large": ClusterLimits("crypto_large", 0.40, 0.8, ["BTC/USD", "ETH/USD"]),
            "crypto_alt": ClusterLimits("crypto_alt", 0.30, 0.7, ["SOL/USD", "ADA/USD", "DOT/USD"]),
            "crypto_defi": ClusterLimits("crypto_defi", 0.20, 0.6, ["UNI/USD", "AAVE/USD", "COMP/USD"]),
            "crypto_layer1": ClusterLimits("crypto_layer1", 0.25, 0.7, ["AVAX/USD", "NEAR/USD", "FTM/USD"]),
            "stablecoins": ClusterLimits("stablecoins", 0.10, 0.9, ["USDC/USD", "USDT/USD"])
        }
        self.cluster_limits = clusters
    
    def update_asset_metrics(self, metrics: Dict[str, AssetMetrics]):
        """Update asset metrics voor sizing calculations"""
        self.asset_metrics.update(metrics)
        
        # Update cluster assignments
        for symbol, asset_metric in metrics.items():
            cluster_id = asset_metric.cluster_id
            if cluster_id in self.cluster_limits:
                if symbol not in self.cluster_limits[cluster_id].assets:
                    self.cluster_limits[cluster_id].assets.append(symbol)
        
        self.logger.info(f"📊 Updated metrics for {len(metrics)} assets")
    
    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update correlation matrix"""
        self.correlation_matrix = correlation_matrix
        self.last_correlation_update = time.time()
        self.logger.info(f"📈 Updated correlation matrix: {correlation_matrix.shape}")
    
    def update_portfolio_state(self, total_equity: float, positions: Dict[str, float]):
        """Update current portfolio state"""
        self.total_equity = total_equity
        self.current_positions = positions.copy()
        
        # Update cluster exposures
        for cluster in self.cluster_limits.values():
            cluster.current_exposure = sum(
                positions.get(symbol, 0.0) for symbol in cluster.assets
            ) / total_equity
    
    def set_market_regime(self, regime: MarketRegime):
        """Update current market regime"""
        self.current_regime = regime
        self.logger.info(f"🌍 Market regime updated: {regime.value}")
    
    def calculate_kelly_size(self, asset: AssetMetrics) -> float:
        """
        Calculate Kelly criterion sizing
        
        Kelly = (p*b - q) / b
        where p = win rate, q = loss rate, b = avg_win/avg_loss
        """
        if asset.win_rate <= 0 or asset.avg_win_loss_ratio <= 0:
            return 0.0
        
        p = asset.win_rate
        q = 1 - p
        b = asset.avg_win_loss_ratio
        
        kelly_fraction = (p * b - q) / b
        
        # Apply fractional Kelly
        kelly_size = max(0, kelly_fraction * self.sizing_params.fractional_kelly)
        
        return min(kelly_size, self.sizing_params.max_position_size)
    
    def calculate_vol_adjusted_size(self, asset: AssetMetrics, kelly_size: float) -> float:
        """
        Adjust Kelly size for volatility targeting
        
        Vol-adjusted size = (vol_target / asset_vol) * kelly_size
        """
        if asset.volatility <= 0:
            return 0.0
        
        vol_scalar = self.sizing_params.vol_target_annual / asset.volatility
        vol_adjusted_size = kelly_size * vol_scalar
        
        return min(vol_adjusted_size, self.sizing_params.max_position_size)
    
    def apply_regime_throttle(self, size: float) -> float:
        """Apply regime-based throttling"""
        throttle_factor = self.sizing_params.regime_throttle_factors.get(
            self.current_regime, 1.0
        )
        return size * throttle_factor
    
    def calculate_cluster_adjusted_size(
        self, 
        symbol: str, 
        proposed_size: float
    ) -> Tuple[float, List[str]]:
        """
        Apply cluster and correlation limits
        
        Returns (adjusted_size, reasoning)
        """
        reasoning = []
        
        # Find asset's cluster
        asset_cluster = None
        for cluster in self.cluster_limits.values():
            if symbol in cluster.assets:
                asset_cluster = cluster
                break
        
        if not asset_cluster:
            reasoning.append("No cluster limits - using proposed size")
            return proposed_size, reasoning
        
        # Calculate new cluster exposure
        current_cluster_exposure = asset_cluster.current_exposure
        proposed_addition = proposed_size
        new_cluster_exposure = current_cluster_exposure + proposed_addition
        
        # Check cluster limit
        if new_cluster_exposure > asset_cluster.max_exposure_pct:
            available_cluster_space = asset_cluster.max_exposure_pct - current_cluster_exposure
            adjusted_size = max(0, available_cluster_space)
            reasoning.append(
                f"Cluster {asset_cluster.cluster_id} limit: "
                f"{new_cluster_exposure:.1%} → {asset_cluster.max_exposure_pct:.1%}, "
                f"reduced to {adjusted_size:.1%}"
            )
            proposed_size = adjusted_size
        
        # Check correlation limits
        if self.correlation_matrix is not None and symbol in self.correlation_matrix.columns:
            correlated_exposure = self._calculate_correlated_exposure(
                symbol, proposed_size
            )
            
            max_corr_exposure = self.sizing_params.max_cluster_exposure
            if correlated_exposure > max_corr_exposure:
                correlation_adjustment = max_corr_exposure / correlated_exposure
                proposed_size *= correlation_adjustment
                reasoning.append(
                    f"Correlation limit: reduced by {correlation_adjustment:.1%} "
                    f"due to high correlation exposure"
                )
        
        return proposed_size, reasoning
    
    def _calculate_correlated_exposure(self, symbol: str, proposed_size: float) -> float:
        """Calculate total exposure to highly correlated assets"""
        if self.correlation_matrix is None:
            return proposed_size
        
        total_correlated_exposure = proposed_size
        
        for other_symbol, position_size in self.current_positions.items():
            if other_symbol == symbol or other_symbol not in self.correlation_matrix.columns:
                continue
            
            correlation = abs(self.correlation_matrix.loc[symbol, other_symbol])
            if correlation > self.sizing_params.correlation_threshold:
                total_correlated_exposure += (position_size / self.total_equity)
        
        return total_correlated_exposure
    
    def calculate_position_size(
        self, 
        symbol: str, 
        signal_strength: float = 1.0
    ) -> PositionSize:
        """
        Calculate optimal position size voor een asset
        
        Args:
            symbol: Asset symbol
            signal_strength: Signal confidence (0-1)
            
        Returns:
            PositionSize with detailed breakdown
        """
        if symbol not in self.asset_metrics:
            self.logger.warning(f"⚠️  No metrics available for {symbol}")
            return PositionSize(
                symbol=symbol,
                target_size_pct=0.0,
                target_size_usd=0.0,
                kelly_size_pct=0.0,
                vol_adjusted_size_pct=0.0,
                regime_adjusted_size_pct=0.0,
                cluster_adjusted_size_pct=0.0,
                final_size_pct=0.0,
                reasoning=["No asset metrics available"]
            )
        
        asset = self.asset_metrics[symbol]
        reasoning = []
        
        # Step 1: Kelly sizing
        kelly_size = self.calculate_kelly_size(asset)
        reasoning.append(f"Kelly size: {kelly_size:.1%}")
        
        # Step 2: Vol-adjust
        vol_adjusted_size = self.calculate_vol_adjusted_size(asset, kelly_size)
        reasoning.append(
            f"Vol-adjusted (target {self.sizing_params.vol_target_annual:.0%}): {vol_adjusted_size:.1%}"
        )
        
        # Step 3: Regime throttle
        regime_adjusted_size = self.apply_regime_throttle(vol_adjusted_size)
        throttle_factor = self.sizing_params.regime_throttle_factors.get(self.current_regime, 1.0)
        reasoning.append(
            f"Regime throttle ({self.current_regime.value}, {throttle_factor:.1%}): {regime_adjusted_size:.1%}"
        )
        
        # Step 4: Signal strength adjustment
        signal_adjusted_size = regime_adjusted_size * signal_strength
        if signal_strength != 1.0:
            reasoning.append(f"Signal strength ({signal_strength:.1%}): {signal_adjusted_size:.1%}")
        else:
            signal_adjusted_size = regime_adjusted_size
        
        # Step 5: Cluster and correlation limits
        cluster_adjusted_size, cluster_reasoning = self.calculate_cluster_adjusted_size(
            symbol, signal_adjusted_size
        )
        reasoning.extend(cluster_reasoning)
        
        # Final size
        final_size_pct = max(0, min(cluster_adjusted_size, self.sizing_params.max_position_size))
        target_size_usd = final_size_pct * self.total_equity
        
        if final_size_pct < cluster_adjusted_size:
            reasoning.append(f"Position size cap: {final_size_pct:.1%}")
        
        return PositionSize(
            symbol=symbol,
            target_size_pct=final_size_pct,
            target_size_usd=target_size_usd,
            kelly_size_pct=kelly_size,
            vol_adjusted_size_pct=vol_adjusted_size,
            regime_adjusted_size_pct=regime_adjusted_size,
            cluster_adjusted_size_pct=cluster_adjusted_size,
            final_size_pct=final_size_pct,
            reasoning=reasoning
        )
    
    def calculate_portfolio_sizes(
        self, 
        signals: Dict[str, float]  # symbol -> signal_strength
    ) -> Dict[str, PositionSize]:
        """
        Calculate position sizes for entire portfolio
        
        Args:
            signals: Dict van symbol -> signal_strength (0-1)
            
        Returns:
            Dict van symbol -> PositionSize
        """
        position_sizes = {}
        
        # Calculate individual sizes
        for symbol, signal_strength in signals.items():
            position_size = self.calculate_position_size(symbol, signal_strength)
            position_sizes[symbol] = position_size
        
        # Portfolio-level adjustments
        position_sizes = self._apply_portfolio_constraints(position_sizes)
        
        return position_sizes
    
    def _apply_portfolio_constraints(
        self, 
        position_sizes: Dict[str, PositionSize]
    ) -> Dict[str, PositionSize]:
        """Apply portfolio-level constraints and scaling"""
        
        # Calculate total allocation
        total_allocation = sum(ps.final_size_pct for ps in position_sizes.values())
        
        # If over-allocated, scale down proportionally
        if total_allocation > 1.0:
            scale_factor = 0.95 / total_allocation  # Leave 5% cash
            
            for symbol, position_size in position_sizes.items():
                scaled_size = position_size.final_size_pct * scale_factor
                scaled_usd = scaled_size * self.total_equity
                
                # Update position size
                position_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    target_size_pct=scaled_size,
                    target_size_usd=scaled_usd,
                    kelly_size_pct=position_size.kelly_size_pct,
                    vol_adjusted_size_pct=position_size.vol_adjusted_size_pct,
                    regime_adjusted_size_pct=position_size.regime_adjusted_size_pct,
                    cluster_adjusted_size_pct=position_size.cluster_adjusted_size_pct,
                    final_size_pct=scaled_size,
                    reasoning=position_size.reasoning + [
                        f"Portfolio scaled by {scale_factor:.1%} (total allocation: {total_allocation:.1%})"
                    ]
                )
        
        return position_sizes
    
    def get_cluster_exposures(self) -> Dict[str, Dict[str, float]]:
        """Get current cluster exposures"""
        cluster_exposures = {}
        
        for cluster_id, cluster in self.cluster_limits.items():
            exposure = sum(
                self.current_positions.get(symbol, 0.0) for symbol in cluster.assets
            ) / self.total_equity
            
            cluster_exposures[cluster_id] = {
                "current_exposure": exposure,
                "max_exposure": cluster.max_exposure_pct,
                "utilization": exposure / cluster.max_exposure_pct,
                "available": cluster.max_exposure_pct - exposure,
                "assets": cluster.assets
            }
        
        return cluster_exposures
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get comprehensive sizing system summary"""
        return {
            "sizing_parameters": {
                "fractional_kelly": self.sizing_params.fractional_kelly,
                "vol_target_annual": self.sizing_params.vol_target_annual,
                "max_position_size": self.sizing_params.max_position_size,
                "max_cluster_exposure": self.sizing_params.max_cluster_exposure
            },
            "market_regime": {
                "current": self.current_regime.value,
                "throttle_factor": self.sizing_params.regime_throttle_factors.get(
                    self.current_regime, 1.0
                )
            },
            "portfolio_state": {
                "total_equity": self.total_equity,
                "positions_count": len(self.current_positions),
                "total_allocation": sum(self.current_positions.values()) / self.total_equity
            },
            "cluster_exposures": self.get_cluster_exposures(),
            "correlation_status": {
                "matrix_available": self.correlation_matrix is not None,
                "last_update": self.last_correlation_update,
                "symbols_count": len(self.correlation_matrix.columns) if self.correlation_matrix is not None else 0
            }
        }


# Global Kelly Vol Sizer instance
_global_kelly_sizer: Optional[KellyVolSizer] = None


def get_global_kelly_sizer() -> KellyVolSizer:
    """Get or create global Kelly Vol Sizer"""
    global _global_kelly_sizer
    if _global_kelly_sizer is None:
        _global_kelly_sizer = KellyVolSizer()
        logger.info("✅ Global KellyVolSizer initialized")
    return _global_kelly_sizer


def reset_global_kelly_sizer():
    """Reset global Kelly sizer (for testing)"""
    global _global_kelly_sizer
    _global_kelly_sizer = None