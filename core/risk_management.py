"""
Strict Position Sizing & Risk Management
Advanced correlation analysis, orderbook simulation, and slippage modeling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import asyncio

@dataclass
class PositionSizing:
    """Position sizing calculation result"""
    coin: str
    recommended_size: float
    max_position_size: float
    risk_score: float
    correlation_risk: float
    liquidity_risk: float
    volatility_risk: float
    reasoning: str

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio and individual positions"""
    portfolio_var: float  # Value at Risk
    portfolio_correlation: float
    max_drawdown_estimate: float
    sharpe_ratio_estimate: float
    concentration_risk: float
    liquidity_score: float

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RiskManagementEngine:
    """
    Advanced risk management with strict position sizing
    Includes correlation analysis, orderbook simulation, and slippage modeling
    """
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Risk configuration
        self.config = {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk
            'max_single_position': 0.10,  # 10% max single position
            'correlation_threshold': 0.7,  # High correlation threshold
            'liquidity_threshold': 1000000,  # Minimum daily volume
            'max_slippage': 0.005,  # 0.5% max acceptable slippage
            'var_confidence': 0.95,  # 95% VaR confidence
            'lookback_days': 30,  # Historical data lookback
            'rebalance_threshold': 0.05  # 5% deviation triggers rebalance
        }
        
        # Portfolio state
        self.current_positions = {}
        self.correlation_matrix = None
        self.risk_metrics_history = []
        
        # Orderbook simulation
        self.orderbook_cache = {}
        
        self.logger.critical("RISK MANAGEMENT ENGINE INITIALIZED - Strict position sizing active")
    
    async def calculate_position_size(self, 
                                    coin: str,
                                    predicted_return: float,
                                    confidence: float,
                                    market_data: Dict[str, Any],
                                    portfolio_value: float) -> PositionSizing:
        """Calculate optimal position size with strict risk controls"""
        
        try:
            # Get risk metrics for coin
            risk_metrics = await self._analyze_coin_risk(coin, market_data)
            
            # Calculate correlation risk with existing positions
            correlation_risk = await self._calculate_correlation_risk(coin, market_data)
            
            # Assess liquidity risk
            liquidity_risk = self._assess_liquidity_risk(market_data)
            
            # Calculate volatility risk
            volatility_risk = self._calculate_volatility_risk(market_data)
            
            # Combined risk score
            combined_risk = max(correlation_risk, liquidity_risk, volatility_risk)
            
            # Kelly Criterion base calculation
            kelly_fraction = self._calculate_kelly_fraction(
                predicted_return, confidence, risk_metrics['volatility']
            )
            
            # Apply risk adjustments
            risk_adjusted_fraction = self._apply_risk_adjustments(
                kelly_fraction, combined_risk, correlation_risk
            )
            
            # Position size limits
            max_position = min(
                self.config['max_single_position'],
                self.config['max_portfolio_risk'] / risk_metrics['volatility']
            )
            
            # Final position size
            recommended_size = min(risk_adjusted_fraction, max_position)
            
            # Convert to actual size
            actual_size = recommended_size * portfolio_value
            
            # Check minimum position threshold
            if actual_size < 100:  # Minimum $100 position
                actual_size = 0
                recommended_size = 0
                reasoning = "Position too small (< $100)"
            else:
                reasoning = self._generate_sizing_reasoning(
                    kelly_fraction, risk_adjusted_fraction, combined_risk
                )
            
            result = PositionSizing(
                coin=coin,
                recommended_size=actual_size,
                max_position_size=max_position * portfolio_value,
                risk_score=combined_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                volatility_risk=volatility_risk,
                reasoning=reasoning
            )
            
            self.logger.info(
                f"Position sizing: {coin} = ${actual_size:.2f} "
                f"(Risk: {combined_risk:.3f}, Kelly: {kelly_fraction:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position sizing failed for {coin}: {e}")
            return PositionSizing(
                coin=coin,
                recommended_size=0,
                max_position_size=0,
                risk_score=1.0,
                correlation_risk=1.0,
                liquidity_risk=1.0,
                volatility_risk=1.0,
                reasoning=f"Error in calculation: {e}"
            )
    
    async def _analyze_coin_risk(self, coin: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze risk metrics for specific coin"""
        
        try:
            # Get price history
            price_history = market_data.get('price_history', [])
            
            if len(price_history) < 10:
                return {
                    'volatility': 0.5,  # High default volatility
                    'max_drawdown': 0.3,
                    'var_95': 0.1
                }
            
            # Calculate returns
            prices = [p.get('close', 0) for p in price_history[-30:]]
            returns = np.diff(prices) / prices[:-1]
            
            # Remove any invalid returns
            returns = returns[~np.isnan(returns)]
            
            if len(returns) < 5:
                return {
                    'volatility': 0.5,
                    'max_drawdown': 0.3,
                    'var_95': 0.1
                }
            
            # Calculate risk metrics
            volatility = float(np.std(returns)) * np.sqrt(365)  # Annualized
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdowns))
            
            # Value at Risk (95%)
            var_95 = float(np.percentile(returns, 5))  # 5th percentile
            
            return {
                'volatility': min(volatility, 2.0),  # Cap at 200%
                'max_drawdown': abs(max_drawdown),
                'var_95': abs(var_95)
            }
            
        except Exception as e:
            self.logger.error(f"Coin risk analysis failed for {coin}: {e}")
            return {
                'volatility': 0.5,
                'max_drawdown': 0.3,
                'var_95': 0.1
            }
    
    async def _calculate_correlation_risk(self, coin: str, market_data: Dict[str, Any]) -> float:
        """Calculate correlation risk with existing portfolio"""
        
        try:
            if not self.current_positions:
                return 0.1  # Low risk if no positions
            
            # Get coin's price history
            coin_history = market_data.get('price_history', [])
            
            if len(coin_history) < 10:
                return 0.8  # High risk if insufficient data
            
            coin_prices = [p.get('close', 0) for p in coin_history[-20:]]
            coin_returns = np.diff(coin_prices) / coin_prices[:-1]
            
            # Calculate correlations with existing positions
            correlations = []
            
            for position_coin, position_data in self.current_positions.items():
                if position_coin == coin:
                    continue
                
                # Get position coin history (would come from data manager)
                position_history = position_data.get('price_history', [])
                
                if len(position_history) >= 10:
                    position_prices = [p.get('close', 0) for p in position_history[-20:]]
                    position_returns = np.diff(position_prices) / position_prices[:-1]
                    
                    # Calculate correlation
                    min_length = min(len(coin_returns), len(position_returns))
                    if min_length >= 5:
                        correlation = np.corrcoef(
                            coin_returns[-min_length:],
                            position_returns[-min_length:]
                        )[0, 1]
                        
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))
            
            if not correlations:
                return 0.2  # Low risk if no correlations calculated
            
            # Maximum correlation risk
            max_correlation = max(correlations)
            
            # Risk penalty for high correlation
            if max_correlation > self.config['correlation_threshold']:
                correlation_risk = 0.8 + 0.2 * max_correlation
            else:
                correlation_risk = 0.1 + 0.3 * max_correlation
            
            return min(correlation_risk, 1.0)
            
        except Exception as e:
            self.logger.error(f"Correlation risk calculation failed: {e}")
            return 0.5  # Medium risk on error
    
    def _assess_liquidity_risk(self, market_data: Dict[str, Any]) -> float:
        """Assess liquidity risk based on volume and spread"""
        
        try:
            # Get recent volume data
            price_history = market_data.get('price_history', [])
            
            if not price_history:
                return 0.9  # Very high risk if no data
            
            # Calculate average daily volume
            recent_volumes = [p.get('volume', 0) for p in price_history[-7:]]
            avg_volume = np.mean(recent_volumes)
            
            # Volume-based liquidity risk
            if avg_volume >= self.config['liquidity_threshold']:
                volume_risk = 0.1  # Low risk
            elif avg_volume >= self.config['liquidity_threshold'] * 0.5:
                volume_risk = 0.3  # Medium risk
            elif avg_volume >= self.config['liquidity_threshold'] * 0.1:
                volume_risk = 0.6  # High risk
            else:
                volume_risk = 0.9  # Very high risk
            
            # Bid-ask spread risk (if available)
            spread_risk = 0.2  # Default moderate spread risk
            
            # Recent orderbook data
            orderbook = market_data.get('orderbook', {})
            if orderbook:
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if bids and asks:
                    best_bid = bids[0][0] if bids[0] else 0
                    best_ask = asks[0][0] if asks[0] else 0
                    
                    if best_bid > 0 and best_ask > 0:
                        spread = (best_ask - best_bid) / best_bid
                        
                        if spread <= 0.001:  # 0.1%
                            spread_risk = 0.1
                        elif spread <= 0.005:  # 0.5%
                            spread_risk = 0.3
                        elif spread <= 0.02:  # 2%
                            spread_risk = 0.6
                        else:
                            spread_risk = 0.9
            
            # Combined liquidity risk
            liquidity_risk = max(volume_risk, spread_risk)
            
            return liquidity_risk
            
        except Exception as e:
            self.logger.error(f"Liquidity risk assessment failed: {e}")
            return 0.7  # High risk on error
    
    def _calculate_volatility_risk(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based risk"""
        
        try:
            price_history = market_data.get('price_history', [])
            
            if len(price_history) < 5:
                return 0.8  # High risk if insufficient data
            
            # Calculate recent volatility
            prices = [p.get('close', 0) for p in price_history[-10:]]
            returns = np.diff(prices) / prices[:-1]
            
            # Remove invalid returns
            returns = returns[~np.isnan(returns)]
            
            if len(returns) < 3:
                return 0.8
            
            # Daily volatility
            daily_vol = np.std(returns)
            
            # Risk categories based on daily volatility
            if daily_vol <= 0.02:  # 2% daily vol
                vol_risk = 0.1
            elif daily_vol <= 0.05:  # 5% daily vol
                vol_risk = 0.3
            elif daily_vol <= 0.10:  # 10% daily vol
                vol_risk = 0.6
            else:
                vol_risk = 0.9  # Very high volatility
            
            return vol_risk
            
        except Exception as e:
            self.logger.error(f"Volatility risk calculation failed: {e}")
            return 0.6
    
    def _calculate_kelly_fraction(self, 
                                expected_return: float,
                                confidence: float,
                                volatility: float) -> float:
        """Calculate Kelly Criterion position size"""
        
        try:
            # Adjust expected return by confidence
            adjusted_return = expected_return * confidence
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = probability of win, q = probability of loss
            
            # Estimate win probability from confidence and expected return
            if adjusted_return > 0:
                win_prob = 0.5 + confidence * 0.3  # Base 50% + confidence bonus
            else:
                win_prob = 0.5 - confidence * 0.3
            
            loss_prob = 1 - win_prob
            
            # Calculate odds (expected return / expected loss)
            if loss_prob > 0 and volatility > 0:
                odds = abs(adjusted_return) / (volatility * loss_prob)
                kelly_fraction = (odds * win_prob - loss_prob) / odds
            else:
                kelly_fraction = 0
            
            # Cap Kelly fraction at 25% for safety
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Kelly calculation failed: {e}")
            return 0.05  # Conservative default
    
    def _apply_risk_adjustments(self, 
                              kelly_fraction: float,
                              combined_risk: float,
                              correlation_risk: float) -> float:
        """Apply risk adjustments to position size"""
        
        # Risk penalty multiplier
        risk_penalty = 1.0 - combined_risk * 0.8  # Up to 80% reduction
        
        # Correlation penalty
        correlation_penalty = 1.0 - correlation_risk * 0.5  # Up to 50% reduction
        
        # Apply adjustments
        adjusted_fraction = kelly_fraction * risk_penalty * correlation_penalty
        
        # Minimum position filter
        if adjusted_fraction < 0.001:  # Less than 0.1%
            adjusted_fraction = 0
        
        return adjusted_fraction
    
    def _generate_sizing_reasoning(self, 
                                 kelly_fraction: float,
                                 adjusted_fraction: float,
                                 risk_score: float) -> str:
        """Generate human-readable reasoning for position size"""
        
        if adjusted_fraction == 0:
            return "Position rejected due to high risk or insufficient confidence"
        
        reduction = (kelly_fraction - adjusted_fraction) / kelly_fraction if kelly_fraction > 0 else 0
        
        reasoning_parts = []
        
        if kelly_fraction > 0:
            reasoning_parts.append(f"Kelly optimal: {kelly_fraction:.1%}")
        
        if reduction > 0.1:
            reasoning_parts.append(f"Risk-adjusted down {reduction:.1%}")
        
        if risk_score > 0.6:
            reasoning_parts.append("High risk environment")
        elif risk_score > 0.4:
            reasoning_parts.append("Moderate risk environment")
        else:
            reasoning_parts.append("Low risk environment")
        
        return "; ".join(reasoning_parts)
    
    async def simulate_order_execution(self, 
                                     coin: str,
                                     order_size: float,
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order execution with slippage estimation"""
        
        try:
            orderbook = market_data.get('orderbook', {})
            
            if not orderbook:
                # Estimate slippage without orderbook
                avg_volume = np.mean([
                    p.get('volume', 0) for p in market_data.get('price_history', [])[-5:]
                ])
                
                if avg_volume > 0:
                    volume_ratio = order_size / avg_volume
                    estimated_slippage = min(volume_ratio * 0.001, 0.02)  # Max 2%
                else:
                    estimated_slippage = 0.01  # 1% default
                
                return {
                    'estimated_slippage': estimated_slippage,
                    'execution_feasible': estimated_slippage <= self.config['max_slippage'],
                    'method': 'volume_estimation'
                }
            
            # Orderbook-based simulation
            asks = orderbook.get('asks', [])
            
            if not asks:
                return {
                    'estimated_slippage': 0.02,
                    'execution_feasible': False,
                    'method': 'no_orderbook'
                }
            
            # Simulate market buy order
            remaining_size = order_size
            total_cost = 0
            
            for ask_price, ask_size in asks:
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, ask_size)
                total_cost += fill_size * ask_price
                remaining_size -= fill_size
            
            if remaining_size > 0:
                # Could not fill entire order
                return {
                    'estimated_slippage': 0.05,  # 5% penalty for partial fill
                    'execution_feasible': False,
                    'method': 'orderbook_simulation',
                    'fill_ratio': (order_size - remaining_size) / order_size
                }
            
            # Calculate average execution price
            avg_execution_price = total_cost / order_size
            best_ask = asks[0][0]
            
            slippage = (avg_execution_price - best_ask) / best_ask
            
            return {
                'estimated_slippage': slippage,
                'execution_feasible': slippage <= self.config['max_slippage'],
                'method': 'orderbook_simulation',
                'avg_execution_price': avg_execution_price,
                'fill_ratio': 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Order simulation failed for {coin}: {e}")
            return {
                'estimated_slippage': 0.02,
                'execution_feasible': False,
                'method': 'error',
                'error': str(e)
            }
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            if not positions:
                return RiskMetrics(
                    portfolio_var=0,
                    portfolio_correlation=0,
                    max_drawdown_estimate=0,
                    sharpe_ratio_estimate=0,
                    concentration_risk=0,
                    liquidity_score=1.0
                )
            
            # Extract position weights and returns
            weights = []
            returns_matrix = []
            
            for coin, position in positions.items():
                weight = position.get('weight', 0)
                returns = position.get('returns', [])
                
                if weight > 0 and len(returns) >= 10:
                    weights.append(weight)
                    returns_matrix.append(returns[-20:])  # Last 20 returns
            
            if len(weights) < 2:
                return RiskMetrics(
                    portfolio_var=0.02,
                    portfolio_correlation=0,
                    max_drawdown_estimate=0.1,
                    sharpe_ratio_estimate=0,
                    concentration_risk=max(weights) if weights else 0,
                    liquidity_score=0.8
                )
            
            weights = np.array(weights)
            returns_matrix = np.array(returns_matrix)
            
            # Portfolio variance calculation
            cov_matrix = np.cov(returns_matrix)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_var = np.sqrt(portfolio_variance) * 1.96  # 95% VaR
            
            # Average correlation
            corr_matrix = np.corrcoef(returns_matrix)
            upper_triangle = np.triu(corr_matrix, k=1)
            avg_correlation = np.mean(upper_triangle[upper_triangle != 0])
            
            # Concentration risk (Herfindahl index)
            concentration_risk = np.sum(weights ** 2)
            
            # Portfolio returns for Sharpe calculation
            portfolio_returns = np.dot(returns_matrix.T, weights)
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
            
            # Estimate max drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown_estimate = abs(np.min(drawdowns))
            
            # Liquidity score (average of position liquidity)
            liquidity_scores = [pos.get('liquidity_score', 0.5) for pos in positions.values()]
            avg_liquidity = np.mean(liquidity_scores)
            
            return RiskMetrics(
                portfolio_var=float(portfolio_var),
                portfolio_correlation=float(avg_correlation) if not np.isnan(avg_correlation) else 0,
                max_drawdown_estimate=float(max_drawdown_estimate),
                sharpe_ratio_estimate=float(sharpe_ratio),
                concentration_risk=float(concentration_risk),
                liquidity_score=float(avg_liquidity)
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio risk calculation failed: {e}")
            return RiskMetrics(
                portfolio_var=0.05,
                portfolio_correlation=0.5,
                max_drawdown_estimate=0.2,
                sharpe_ratio_estimate=0,
                concentration_risk=0.5,
                liquidity_score=0.5
            )
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk management status"""
        
        return {
            'risk_config': self.config,
            'current_positions_count': len(self.current_positions),
            'correlation_matrix_available': self.correlation_matrix is not None,
            'risk_metrics_history_size': len(self.risk_metrics_history),
            'orderbook_cache_size': len(self.orderbook_cache),
            'last_updated': datetime.now().isoformat()
        }

# Global risk management instance
risk_manager = None

def get_risk_manager(container):
    """Get or create risk management instance"""
    global risk_manager
    
    if risk_manager is None:
        risk_manager = RiskManagementEngine(container)
    
    return risk_manager