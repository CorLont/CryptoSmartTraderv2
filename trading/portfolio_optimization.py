#!/usr/bin/env python3
"""
Advanced Portfolio Optimization
Uncertainty-aware sizing, correlation caps, and risk overlays
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class UncertaintyAwarePositionSizer:
    """
    Kelly-lite position sizing with uncertainty awareness
    """

    def __init__(
        self,
        max_position_size: float = 0.05,  # 5% max per position
        min_confidence: float = 0.7,  # Minimum confidence threshold
        risk_free_rate: float = 0.02,
    ):  # 2% risk-free rate
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.risk_free_rate = risk_free_rate

    def calculate_position_size(
        self,
        prediction: float,
        confidence: float,
        uncertainty: float,
        volatility: float,
        liquidity_score: float,
    ) -> Dict[str, float]:
        """Calculate optimal position size"""

        # Base Kelly fraction
        if volatility <= 0:
            return {"position_size": 0, "reason": "zero_volatility"}

        # Expected return above risk-free rate
        excess_return = prediction - self.risk_free_rate / 365  # Daily

        # Basic Kelly fraction: f = (μ - r) / σ²
        kelly_fraction = excess_return / (volatility**2)

        # Apply confidence scaling
        confidence_scalar = max(0, (confidence - self.min_confidence) / (1 - self.min_confidence))

        # Apply uncertainty penalty
        uncertainty_penalty = 1 / (1 + uncertainty * 10)  # Reduce size for high uncertainty

        # Apply liquidity constraint
        liquidity_constraint = min(1.0, liquidity_score)

        # Combine all factors
        adjusted_size = (
            kelly_fraction * confidence_scalar * uncertainty_penalty * liquidity_constraint
        )

        # Apply bounds
        position_size = np.clip(adjusted_size, 0, self.max_position_size)

        return {
            "position_size": position_size,
            "kelly_fraction": kelly_fraction,
            "confidence_scalar": confidence_scalar,
            "uncertainty_penalty": uncertainty_penalty,
            "liquidity_constraint": liquidity_constraint,
            "reason": "optimal" if position_size > 0 else "filtered_out",
        }


class CorrelationManager:
    """
    Manage portfolio correlation and clustering
    """

    def __init__(
        self,
        max_cluster_exposure: float = 0.15,  # 15% max per cluster
        correlation_threshold: float = 0.7,
    ):  # High correlation threshold
        self.max_cluster_exposure = max_cluster_exposure
        self.correlation_threshold = correlation_threshold
        self.asset_clusters = {}
        self.correlation_matrix = None

    def update_correlations(self, price_data: pd.DataFrame, lookback_days: int = 30) -> np.ndarray:
        """Update correlation matrix from price data"""

        # Calculate returns
        returns = price_data.pct_change().dropna()

        # Use recent data
        if len(returns) > lookback_days:
            returns = returns.tail(lookback_days)

        # Calculate correlation matrix
        self.correlation_matrix = returns.corr().values

        return self.correlation_matrix

    def cluster_assets(self, feature_data: pd.DataFrame, n_clusters: int = 5) -> Dict[str, int]:
        """Cluster assets based on features"""

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data.fillna(0))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Map assets to clusters
        self.asset_clusters = {
            asset: int(cluster) for asset, cluster in zip(feature_data.index, cluster_labels)
        }

        return self.asset_clusters

    def check_correlation_constraints(self, proposed_positions: Dict[str, float]) -> Dict[str, Any]:
        """Check if proposed positions violate correlation constraints"""

        violations = []
        cluster_exposures = {}

        # Calculate cluster exposures
        for asset, position in proposed_positions.items():
            if asset in self.asset_clusters:
                cluster = self.asset_clusters[asset]
                if cluster not in cluster_exposures:
                    cluster_exposures[cluster] = 0
                cluster_exposures[cluster] += abs(position)

        # Check cluster limits
        for cluster, exposure in cluster_exposures.items():
            if exposure > self.max_cluster_exposure:
                violations.append(
                    {
                        "type": "cluster_exposure",
                        "cluster": cluster,
                        "exposure": exposure,
                        "limit": self.max_cluster_exposure,
                    }
                )

        # Check pairwise correlations
        if self.correlation_matrix is not None:
            asset_list = list(proposed_positions.keys())

            for i, asset1 in enumerate(asset_list):
                for j, asset2 in enumerate(asset_list[i + 1 :], i + 1):
                    if (
                        abs(proposed_positions[asset1]) > 0.01
                        and abs(proposed_positions[asset2]) > 0.01
                    ):
                        # Find correlation (simplified lookup)
                        correlation = 0.5  # Placeholder - would need proper mapping

                        if abs(correlation) > self.correlation_threshold:
                            combined_exposure = abs(proposed_positions[asset1]) + abs(
                                proposed_positions[asset2]
                            )

                            violations.append(
                                {
                                    "type": "high_correlation",
                                    "assets": [asset1, asset2],
                                    "correlation": correlation,
                                    "combined_exposure": combined_exposure,
                                }
                            )

        return {
            "violations": violations,
            "cluster_exposures": cluster_exposures,
            "is_valid": len(violations) == 0,
        }

    def adjust_positions_for_constraints(
        self, proposed_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust positions to satisfy correlation constraints"""

        constraints_check = self.check_correlation_constraints(proposed_positions)

        if constraints_check["is_valid"]:
            return proposed_positions

        adjusted_positions = proposed_positions.copy()

        # Handle cluster exposure violations
        for violation in constraints_check["violations"]:
            if violation["type"] == "cluster_exposure":
                cluster = violation["cluster"]
                scale_factor = self.max_cluster_exposure / violation["exposure"]

                # Scale down all positions in this cluster
                for asset in adjusted_positions:
                    if self.asset_clusters.get(asset) == cluster:
                        adjusted_positions[asset] *= scale_factor

            elif violation["type"] == "high_correlation":
                # Reduce the smaller position
                asset1, asset2 = violation["assets"]
                pos1, pos2 = adjusted_positions[asset1], adjusted_positions[asset2]

                if abs(pos1) < abs(pos2):
                    adjusted_positions[asset1] *= 0.5
                else:
                    adjusted_positions[asset2] *= 0.5

        return adjusted_positions


class RiskOverlaySystem:
    """
    Hard risk overlays for regime protection
    """

    def __init__(self):
        self.risk_rules = {
            "btc_drawdown_limit": 0.15,  # 15% BTC drawdown limit
            "health_score_limit": 60,  # Health score limit
            "volatility_spike_limit": 3.0,  # 3x normal volatility
            "correlation_spike_limit": 0.9,  # 90% correlation to BTC
        }

        self.risk_states = {
            "btc_drawdown_active": False,
            "low_health_active": False,
            "high_volatility_active": False,
            "high_correlation_active": False,
        }

    def evaluate_risk_conditions(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate current risk conditions"""

        risk_signals = {}

        # BTC drawdown check
        btc_price = market_data.get("btc_price", 0)
        btc_high = market_data.get("btc_recent_high", btc_price)

        if btc_high > 0:
            btc_drawdown = (btc_high - btc_price) / btc_high
            risk_signals["btc_drawdown"] = btc_drawdown
            self.risk_states["btc_drawdown_active"] = (
                btc_drawdown > self.risk_rules["btc_drawdown_limit"]
            )

        # Health score check
        health_score = market_data.get("system_health_score", 100)
        risk_signals["health_score"] = health_score
        self.risk_states["low_health_active"] = health_score < self.risk_rules["health_score_limit"]

        # Volatility spike check
        current_vol = market_data.get("market_volatility", 0)
        normal_vol = market_data.get("normal_volatility", current_vol)

        if normal_vol > 0:
            vol_ratio = current_vol / normal_vol
            risk_signals["volatility_ratio"] = vol_ratio
            self.risk_states["high_volatility_active"] = (
                vol_ratio > self.risk_rules["volatility_spike_limit"]
            )

        # Correlation spike check
        btc_correlation = market_data.get("avg_btc_correlation", 0)
        risk_signals["btc_correlation"] = btc_correlation
        self.risk_states["high_correlation_active"] = (
            btc_correlation > self.risk_rules["correlation_spike_limit"]
        )

        return {
            "risk_signals": risk_signals,
            "risk_states": self.risk_states,
            "any_risk_active": any(self.risk_states.values()),
            "active_risks": [k for k, v in self.risk_states.items() if v],
        }

    def apply_risk_overlays(
        self, proposed_positions: Dict[str, float], market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply risk overlays to proposed positions"""

        risk_evaluation = self.evaluate_risk_conditions(market_data)

        if not risk_evaluation["any_risk_active"]:
            return proposed_positions

        adjusted_positions = proposed_positions.copy()

        # Calculate risk reduction factor
        risk_reduction = 1.0

        if self.risk_states["btc_drawdown_active"]:
            risk_reduction *= 0.5  # 50% position reduction

        if self.risk_states["low_health_active"]:
            risk_reduction *= 0.3  # 70% position reduction

        if self.risk_states["high_volatility_active"]:
            risk_reduction *= 0.6  # 40% position reduction

        if self.risk_states["high_correlation_active"]:
            risk_reduction *= 0.7  # 30% position reduction

        # Apply reduction
        for asset in adjusted_positions:
            adjusted_positions[asset] *= risk_reduction

        return adjusted_positions


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimizer combining all components
    """

    def __init__(self):
        self.position_sizer = UncertaintyAwarePositionSizer()
        self.correlation_manager = CorrelationManager()
        self.risk_overlay = RiskOverlaySystem()

    def optimize_portfolio(
        self,
        opportunities: pd.DataFrame,
        market_data: Dict[str, Any],
        current_positions: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Optimize complete portfolio"""

        if current_positions is None:
            current_positions = {}

        # Step 1: Calculate individual position sizes
        position_recommendations = {}

        for _, opportunity in opportunities.iterrows():
            sizing_result = self.position_sizer.calculate_position_size(
                prediction=opportunity.get("prediction", 0),
                confidence=opportunity.get("confidence", 0),
                uncertainty=opportunity.get("uncertainty", 1),
                volatility=opportunity.get("volatility", 0.1),
                liquidity_score=opportunity.get("liquidity_score", 1),
            )

            if sizing_result["position_size"] > 0:
                position_recommendations[opportunity["coin"]] = sizing_result["position_size"]

        # Step 2: Apply correlation constraints
        correlation_adjusted = self.correlation_manager.adjust_positions_for_constraints(
            position_recommendations
        )

        # Step 3: Apply risk overlays
        final_positions = self.risk_overlay.apply_risk_overlays(correlation_adjusted, market_data)

        # Step 4: Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            final_positions, opportunities, market_data
        )

        return {
            "final_positions": final_positions,
            "initial_recommendations": position_recommendations,
            "correlation_adjusted": correlation_adjusted,
            "portfolio_metrics": portfolio_metrics,
            "risk_evaluation": self.risk_overlay.evaluate_risk_conditions(market_data),
        }

    def _calculate_portfolio_metrics(
        self, positions: Dict[str, float], opportunities: pd.DataFrame, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""

        total_exposure = sum(abs(pos) for pos in positions.values())

        # Expected return
        expected_return = 0
        for asset, position in positions.items():
            asset_data = opportunities[opportunities["coin"] == asset]
            if not asset_data.empty:
                expected_return += position * asset_data.iloc[0].get("prediction", 0)

        # Portfolio concentration
        max_position = max(abs(pos) for pos in positions.values()) if positions else 0
        concentration = max_position / total_exposure if total_exposure > 0 else 0

        # Number of positions
        n_positions = len([p for p in positions.values() if abs(p) > 0.001])

        return {
            "total_exposure": total_exposure,
            "expected_return": expected_return,
            "concentration": concentration,
            "n_positions": n_positions,
            "max_position_size": max_position,
            "portfolio_diversification": 1 - concentration,
        }


if __name__ == "__main__":
    print("💼 TESTING ADVANCED PORTFOLIO OPTIMIZATION")
    print("=" * 50)

    # Create sample opportunities
    sample_opportunities = pd.DataFrame(
        {
            "coin": ["BTC", "ETH", "SOL", "ADA", "MATIC"],
            "prediction": [0.05, 0.08, 0.12, 0.03, 0.07],
            "confidence": [0.85, 0.82, 0.78, 0.75, 0.80],
            "uncertainty": [0.02, 0.03, 0.05, 0.08, 0.04],
            "volatility": [0.04, 0.05, 0.08, 0.06, 0.07],
            "liquidity_score": [1.0, 0.95, 0.85, 0.70, 0.80],
        }
    )

    # Sample market data
    market_data = {
        "btc_price": 67500,
        "btc_recent_high": 70000,
        "system_health_score": 85,
        "market_volatility": 0.05,
        "normal_volatility": 0.04,
        "avg_btc_correlation": 0.6,
    }

    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer()

    # Test position sizing
    print("Testing individual position sizing:")
    for _, opp in sample_opportunities.iterrows():
        sizing = optimizer.position_sizer.calculate_position_size(
            opp["prediction"],
            opp["confidence"],
            opp["uncertainty"],
            opp["volatility"],
            opp["liquidity_score"],
        )
        print(f"   {opp['coin']}: {sizing['position_size']:.4f} ({sizing['reason']})")

    # Test portfolio optimization
    result = optimizer.optimize_portfolio(sample_opportunities, market_data)

    print(f"\nPortfolio Optimization Results:")
    print(f"   Final positions:")
    for asset, position in result["final_positions"].items():
        print(f"      {asset}: {position:.4f}")

    print(f"\n   Portfolio metrics:")
    metrics = result["portfolio_metrics"]
    for key, value in metrics.items():
        print(f"      {key}: {value:.4f}")

    print(f"\n   Risk evaluation:")
    risk_eval = result["risk_evaluation"]
    print(f"      Any risks active: {risk_eval['any_risk_active']}")
    if risk_eval["active_risks"]:
        print(f"      Active risks: {risk_eval['active_risks']}")

    print("✅ Advanced portfolio optimization testing completed")
