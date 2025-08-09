#!/usr/bin/env python3
"""
Kelly-Lite Position Sizing with Uncertainty Awareness
Advanced position sizing using modified Kelly criterion with correlation caps and risk overlays
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from core.structured_logger import get_structured_logger

class UncertaintyAwareKellyOptimizer:
    """Kelly-lite position sizing with uncertainty quantification and risk controls"""
    
    def __init__(self, max_position_size: float = 0.15, max_correlation: float = 0.3, 
                 kelly_fraction: float = 0.25, uncertainty_penalty: float = 0.5):
        """
        Initialize Kelly-lite optimizer
        
        Args:
            max_position_size: Maximum position size per asset (e.g., 15%)
            max_correlation: Maximum correlation between positions
            kelly_fraction: Fraction of Kelly criterion to use (for safety)
            uncertainty_penalty: Penalty factor for high uncertainty predictions
        """
        self.logger = get_structured_logger("KellyLiteOptimizer")
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.kelly_fraction = kelly_fraction
        self.uncertainty_penalty = uncertainty_penalty
        
        # Risk management parameters
        self.min_win_rate = 0.51  # Minimum required win rate
        self.min_reward_risk = 1.2  # Minimum reward/risk ratio
        self.max_drawdown_risk = 0.02  # Max 2% portfolio risk per position
        
    def optimize_portfolio_sizing(self, predictions: pd.DataFrame, 
                                historical_returns: pd.DataFrame = None,
                                uncertainty_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Optimize portfolio position sizes using Kelly-lite with uncertainty awareness"""
        
        self.logger.info(f"Optimizing position sizes for {len(predictions)} assets")
        
        try:
            # Validate inputs
            if len(predictions) == 0:
                return self._get_empty_portfolio()
            
            # Calculate Kelly-lite scores for each asset
            kelly_scores = self._calculate_kelly_scores(predictions, uncertainty_data)
            
            # Apply correlation constraints
            if historical_returns is not None and len(historical_returns) > 30:
                correlation_adjusted_scores = self._apply_correlation_constraints(
                    kelly_scores, historical_returns
                )
            else:
                correlation_adjusted_scores = kelly_scores.copy()
            
            # Apply hard risk overlays
            final_positions = self._apply_risk_overlays(correlation_adjusted_scores, predictions)
            
            # Portfolio optimization
            optimized_portfolio = self._optimize_portfolio_weights(final_positions, predictions)
            
            # Calculate portfolio risk metrics
            portfolio_metrics = self._calculate_portfolio_metrics(optimized_portfolio, predictions)
            
            self.logger.info(f"Portfolio optimization complete - {len(optimized_portfolio['positions'])} positions")
            
            return {
                'positions': optimized_portfolio['positions'],
                'total_allocation': optimized_portfolio['total_allocation'],
                'expected_return': portfolio_metrics['expected_return'],
                'portfolio_risk': portfolio_metrics['portfolio_risk'],
                'sharpe_estimate': portfolio_metrics['sharpe_estimate'],
                'max_correlation': portfolio_metrics['max_correlation'],
                'diversification_score': portfolio_metrics['diversification_score'],
                'optimization_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return self._get_empty_portfolio()
    
    def _calculate_kelly_scores(self, predictions: pd.DataFrame, 
                               uncertainty_data: pd.DataFrame = None) -> pd.Series:
        """Calculate Kelly criterion scores for each asset"""
        
        kelly_scores = {}
        
        for _, pred in predictions.iterrows():
            symbol = pred.get('symbol', pred.get('coin', 'UNKNOWN'))
            
            # Extract prediction data
            predicted_return = pred.get('predicted_return', 0.0)
            confidence = pred.get('confidence', 0.5)
            win_probability = pred.get('win_probability', confidence)
            
            # Get uncertainty penalty
            uncertainty_penalty = 0.0
            if uncertainty_data is not None and symbol in uncertainty_data.index:
                uncertainty = uncertainty_data.loc[symbol, 'uncertainty']
                uncertainty_penalty = uncertainty * self.uncertainty_penalty
            
            # Calculate Kelly fraction
            kelly_score = self._calculate_individual_kelly(
                predicted_return, win_probability, uncertainty_penalty
            )
            
            kelly_scores[symbol] = kelly_score
        
        return pd.Series(kelly_scores)
    
    def _calculate_individual_kelly(self, predicted_return: float, win_probability: float, 
                                  uncertainty_penalty: float) -> float:
        """Calculate Kelly criterion for individual asset"""
        
        try:
            # Basic Kelly formula: f = (bp - q) / b
            # where b = odds received, p = win probability, q = lose probability
            
            if predicted_return <= 0 or win_probability <= 0.5:
                return 0.0  # No position for negative expected value
            
            # Estimate win/loss amounts
            avg_win = abs(predicted_return) if predicted_return > 0 else 0.02
            avg_loss = 0.01  # Assume 1% average loss (can be refined)
            
            # Kelly calculation
            edge = win_probability - (1 - win_probability) * (avg_loss / avg_win)
            if edge <= 0:
                return 0.0
            
            kelly_fraction = edge / (avg_win / avg_loss)
            
            # Apply uncertainty penalty
            adjusted_kelly = kelly_fraction * (1 - uncertainty_penalty)
            
            # Apply Kelly fraction safety multiplier
            safe_kelly = adjusted_kelly * self.kelly_fraction
            
            # Cap at maximum position size
            final_kelly = min(safe_kelly, self.max_position_size)
            
            return max(0.0, final_kelly)
            
        except Exception as e:
            self.logger.error(f"Kelly calculation failed: {e}")
            return 0.0
    
    def _apply_correlation_constraints(self, kelly_scores: pd.Series, 
                                     historical_returns: pd.DataFrame) -> pd.Series:
        """Apply correlation constraints to position sizes"""
        
        try:
            self.logger.info("Applying correlation constraints")
            
            # Calculate correlation matrix
            correlation_matrix = historical_returns.corr()
            
            # Find high correlation pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > self.max_correlation:
                        asset1 = correlation_matrix.columns[i]
                        asset2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((asset1, asset2, corr))
            
            # Adjust positions for high correlation
            adjusted_scores = kelly_scores.copy()
            
            for asset1, asset2, corr in high_corr_pairs:
                if asset1 in adjusted_scores and asset2 in adjusted_scores:
                    # Reduce positions proportionally
                    total_exposure = adjusted_scores[asset1] + adjusted_scores[asset2]
                    correlation_factor = 1 - (abs(corr) - self.max_correlation) * 0.5
                    
                    # Redistribute while maintaining relative weights
                    if total_exposure > 0:
                        ratio1 = adjusted_scores[asset1] / total_exposure
                        ratio2 = adjusted_scores[asset2] / total_exposure
                        
                        reduced_total = total_exposure * correlation_factor
                        adjusted_scores[asset1] = reduced_total * ratio1
                        adjusted_scores[asset2] = reduced_total * ratio2
            
            self.logger.info(f"Applied correlation constraints to {len(high_corr_pairs)} pairs")
            return adjusted_scores
            
        except Exception as e:
            self.logger.error(f"Correlation constraint application failed: {e}")
            return kelly_scores
    
    def _apply_risk_overlays(self, position_scores: pd.Series, 
                           predictions: pd.DataFrame) -> pd.Series:
        """Apply hard risk overlays and safety checks"""
        
        risk_adjusted_scores = position_scores.copy()
        
        for symbol in position_scores.index:
            # Find prediction data for this symbol
            pred_data = predictions[predictions.get('symbol', predictions.get('coin', '')) == symbol]
            
            if len(pred_data) == 0:
                risk_adjusted_scores[symbol] = 0.0
                continue
            
            pred = pred_data.iloc[0]
            
            # Risk check 1: Minimum confidence
            confidence = pred.get('confidence', 0.0)
            if confidence < 0.6:  # 60% minimum confidence
                risk_adjusted_scores[symbol] *= 0.5  # Halve position
            
            # Risk check 2: Minimum expected return
            expected_return = pred.get('predicted_return', 0.0)
            if expected_return < 0.01:  # 1% minimum expected return
                risk_adjusted_scores[symbol] = 0.0
            
            # Risk check 3: Maximum position size
            risk_adjusted_scores[symbol] = min(
                risk_adjusted_scores[symbol], 
                self.max_position_size
            )
            
            # Risk check 4: Drawdown risk
            estimated_volatility = pred.get('volatility', 0.02)
            max_safe_position = self.max_drawdown_risk / (2 * estimated_volatility)
            risk_adjusted_scores[symbol] = min(
                risk_adjusted_scores[symbol],
                max_safe_position
            )
        
        return risk_adjusted_scores
    
    def _optimize_portfolio_weights(self, position_scores: pd.Series, 
                                  predictions: pd.DataFrame) -> Dict[str, Any]:
        """Final portfolio weight optimization"""
        
        # Filter out zero positions
        active_positions = position_scores[position_scores > 0.001]
        
        if len(active_positions) == 0:
            return {'positions': {}, 'total_allocation': 0.0}
        
        # Normalize to ensure total doesn't exceed 100%
        total_raw_allocation = active_positions.sum()
        
        if total_raw_allocation > 1.0:
            # Scale down proportionally
            scaling_factor = 0.95 / total_raw_allocation  # Leave 5% cash
            active_positions *= scaling_factor
        
        # Create final position dictionary
        final_positions = {}
        for symbol, weight in active_positions.items():
            if weight > 0.001:  # Minimum 0.1% position
                # Get additional data for this position
                pred_data = predictions[predictions.get('symbol', predictions.get('coin', '')) == symbol]
                
                position_info = {
                    'weight': weight,
                    'confidence': pred_data.iloc[0].get('confidence', 0.5) if len(pred_data) > 0 else 0.5,
                    'expected_return': pred_data.iloc[0].get('predicted_return', 0.0) if len(pred_data) > 0 else 0.0,
                    'kelly_score': position_scores.get(symbol, 0.0)
                }
                
                final_positions[symbol] = position_info
        
        return {
            'positions': final_positions,
            'total_allocation': sum(pos['weight'] for pos in final_positions.values())
        }
    
    def _calculate_portfolio_metrics(self, portfolio: Dict[str, Any], 
                                   predictions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio-level risk and return metrics"""
        
        try:
            positions = portfolio['positions']
            
            if not positions:
                return {
                    'expected_return': 0.0,
                    'portfolio_risk': 0.0,
                    'sharpe_estimate': 0.0,
                    'max_correlation': 0.0,
                    'diversification_score': 0.0
                }
            
            # Calculate expected portfolio return
            portfolio_expected_return = 0.0
            position_volatilities = []
            
            for symbol, pos_info in positions.items():
                weight = pos_info['weight']
                expected_return = pos_info['expected_return']
                portfolio_expected_return += weight * expected_return
                
                # Estimate volatility
                pred_data = predictions[predictions.get('symbol', predictions.get('coin', '')) == symbol]
                volatility = pred_data.iloc[0].get('volatility', 0.02) if len(pred_data) > 0 else 0.02
                position_volatilities.append(volatility * weight)
            
            # Estimate portfolio risk (simplified)
            portfolio_risk = np.sqrt(sum(vol**2 for vol in position_volatilities))
            
            # Estimate Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_estimate = (portfolio_expected_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Diversification score
            position_weights = [pos['weight'] for pos in positions.values()]
            diversification_score = 1 - sum(w**2 for w in position_weights)  # Herfindahl index
            
            # Maximum correlation (placeholder)
            max_correlation = 0.3  # Would need actual correlation calculation
            
            return {
                'expected_return': portfolio_expected_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_estimate': sharpe_estimate,
                'max_correlation': max_correlation,
                'diversification_score': diversification_score
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {e}")
            return {
                'expected_return': 0.0,
                'portfolio_risk': 0.0,
                'sharpe_estimate': 0.0,
                'max_correlation': 0.0,
                'diversification_score': 0.0
            }
    
    def _get_empty_portfolio(self) -> Dict[str, Any]:
        """Return empty portfolio structure"""
        return {
            'positions': {},
            'total_allocation': 0.0,
            'expected_return': 0.0,
            'portfolio_risk': 0.0,
            'sharpe_estimate': 0.0,
            'max_correlation': 0.0,
            'diversification_score': 0.0,
            'optimization_timestamp': pd.Timestamp.now().isoformat()
        }

class CorrelationCapManager:
    """Manages correlation limits and cluster-based position limits"""
    
    def __init__(self, max_cluster_allocation: float = 0.4):
        self.logger = get_structured_logger("CorrelationCapManager")
        self.max_cluster_allocation = max_cluster_allocation
        
    def apply_cluster_caps(self, positions: Dict[str, Any], 
                          asset_clusters: Dict[str, str] = None) -> Dict[str, Any]:
        """Apply cluster-based allocation caps"""
        
        try:
            if not asset_clusters:
                # Simple clustering based on asset names
                asset_clusters = self._simple_asset_clustering(list(positions.keys()))
            
            # Calculate cluster allocations
            cluster_allocations = {}
            for symbol, pos_info in positions.items():
                cluster = asset_clusters.get(symbol, 'other')
                cluster_allocations.setdefault(cluster, 0.0)
                cluster_allocations[cluster] += pos_info['weight']
            
            # Apply caps to over-allocated clusters
            adjusted_positions = positions.copy()
            
            for cluster, total_allocation in cluster_allocations.items():
                if total_allocation > self.max_cluster_allocation:
                    # Scale down positions in this cluster
                    scaling_factor = self.max_cluster_allocation / total_allocation
                    
                    for symbol, pos_info in adjusted_positions.items():
                        if asset_clusters.get(symbol, 'other') == cluster:
                            pos_info['weight'] *= scaling_factor
            
            self.logger.info(f"Applied cluster caps to {len(cluster_allocations)} clusters")
            return adjusted_positions
            
        except Exception as e:
            self.logger.error(f"Cluster cap application failed: {e}")
            return positions
    
    def _simple_asset_clustering(self, symbols: List[str]) -> Dict[str, str]:
        """Simple asset clustering based on naming patterns"""
        
        clusters = {}
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            
            # Layer 1 blockchains
            if any(token in symbol_upper for token in ['BTC', 'ETH', 'ADA', 'DOT', 'SOL', 'AVAX']):
                clusters[symbol] = 'layer1'
            # DeFi tokens
            elif any(token in symbol_upper for token in ['UNI', 'AAVE', 'COMP', 'MKR', 'YFI']):
                clusters[symbol] = 'defi'
            # Meme tokens
            elif any(token in symbol_upper for token in ['DOGE', 'SHIB', 'PEPE']):
                clusters[symbol] = 'meme'
            # Stablecoins
            elif any(token in symbol_upper for token in ['USDT', 'USDC', 'DAI', 'BUSD']):
                clusters[symbol] = 'stable'
            else:
                clusters[symbol] = 'other'
        
        return clusters

# Example usage
def example_kelly_optimization():
    """Example of Kelly-lite optimization usage"""
    
    # Sample prediction data
    predictions = pd.DataFrame({
        'symbol': ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD'],
        'predicted_return': [0.05, 0.03, 0.08, 0.04],
        'confidence': [0.85, 0.75, 0.90, 0.70],
        'volatility': [0.03, 0.04, 0.06, 0.05]
    })
    
    # Sample uncertainty data
    uncertainty_data = pd.DataFrame({
        'uncertainty': [0.1, 0.15, 0.08, 0.12]
    }, index=['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD'])
    
    # Initialize optimizer
    optimizer = UncertaintyAwareKellyOptimizer()
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio_sizing(
        predictions=predictions,
        uncertainty_data=uncertainty_data
    )
    
    return result

if __name__ == "__main__":
    result = example_kelly_optimization()
    print("Kelly-lite optimization result:")
    for key, value in result.items():
        print(f"{key}: {value}")