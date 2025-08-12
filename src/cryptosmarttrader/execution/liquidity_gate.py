"""
Liquidity Gating System

Evaluates market liquidity conditions before trade execution to ensure
sufficient depth and volume for alpha preservation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class LiquidityMetrics:
    """Container for liquidity measurement results"""
    symbol: str
    timestamp: datetime
    
    # Spread metrics
    bid_ask_spread_bp: float      # Bid-ask spread in basis points
    mid_price: float              # Mid price (bid + ask) / 2
    
    # Order book depth
    bid_depth_quote: float        # Total bid depth in quote currency
    ask_depth_quote: float        # Total ask depth in quote currency
    depth_imbalance: float        # (bid_depth - ask_depth) / total_depth
    
    # Volume metrics
    volume_1m: float              # 1-minute volume in quote currency
    volume_5m: float              # 5-minute volume in quote currency
    volume_1h: float              # 1-hour volume in quote currency
    
    # Liquidity quality scores
    spread_score: float           # 0-1 score (1 = tight spread)
    depth_score: float            # 0-1 score (1 = deep book)
    volume_score: float           # 0-1 score (1 = high volume)
    liquidity_score: float        # Combined liquidity score
    
    # Execution suitability
    suitable_for_execution: bool
    max_recommended_size: float   # Maximum size for good execution (quote)
    estimated_slippage_bp: float  # Estimated slippage for this size


class LiquidityGate:
    """
    Evaluates and gates trades based on liquidity conditions
    """
    
    def __init__(self,
                 max_spread_bp: float = 15.0,           # Max 15 basis points spread
                 min_depth_quote: float = 50000.0,      # Min $50k depth each side
                 min_volume_1m: float = 100000.0,       # Min $100k 1-minute volume
                 min_liquidity_score: float = 0.6):     # Min combined liquidity score
        """
        Initialize liquidity gate with thresholds
        
        Args:
            max_spread_bp: Maximum allowed bid-ask spread (basis points)
            min_depth_quote: Minimum order book depth (quote currency)
            min_volume_1m: Minimum 1-minute volume (quote currency)
            min_liquidity_score: Minimum combined liquidity score (0-1)
        """
        self.max_spread_bp = max_spread_bp
        self.min_depth_quote = min_depth_quote
        self.min_volume_1m = min_volume_1m
        self.min_liquidity_score = min_liquidity_score
        
        # Tracking for analytics
        self.liquidity_history = []
        self.rejected_trades = []
        
    def evaluate_liquidity(self, 
                          symbol: str,
                          order_book: Dict[str, Any],
                          recent_trades: List[Dict[str, Any]],
                          volume_data: Optional[Dict[str, float]] = None) -> LiquidityMetrics:
        """
        Evaluate current liquidity conditions for a symbol
        
        Args:
            symbol: Trading pair symbol
            order_book: Order book data with 'bids' and 'asks'
            recent_trades: Recent trade data for volume calculation
            volume_data: Pre-calculated volume metrics
            
        Returns:
            Complete liquidity assessment
        """
        try:
            timestamp = datetime.now()
            
            # Extract order book data
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return self._create_empty_metrics(symbol, timestamp)
            
            # Calculate spread metrics
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            if best_bid <= 0 or best_ask <= 0:
                return self._create_empty_metrics(symbol, timestamp)
            
            mid_price = (best_bid + best_ask) / 2
            spread_absolute = best_ask - best_bid
            spread_bp = (spread_absolute / mid_price) * 10000
            
            # Calculate order book depth
            bid_depth = self._calculate_depth(bids, mid_price)
            ask_depth = self._calculate_depth(asks, mid_price)
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(recent_trades, volume_data)
            
            # Calculate quality scores
            spread_score = self._calculate_spread_score(spread_bp)
            depth_score = self._calculate_depth_score(min(bid_depth, ask_depth))
            volume_score = self._calculate_volume_score(volume_metrics['volume_1m'])
            
            # Combined liquidity score (weighted average)
            liquidity_score = (
                0.4 * spread_score +
                0.4 * depth_score +
                0.2 * volume_score
            )
            
            # Determine execution suitability
            suitable = self._is_suitable_for_execution(
                spread_bp, min(bid_depth, ask_depth), 
                volume_metrics['volume_1m'], liquidity_score
            )
            
            # Estimate maximum recommended size and slippage
            max_size, estimated_slippage = self._estimate_execution_impact(
                bids, asks, mid_price, volume_metrics['volume_1m']
            )
            
            metrics = LiquidityMetrics(
                symbol=symbol,
                timestamp=timestamp,
                bid_ask_spread_bp=spread_bp,
                mid_price=mid_price,
                bid_depth_quote=bid_depth,
                ask_depth_quote=ask_depth,
                depth_imbalance=depth_imbalance,
                volume_1m=volume_metrics['volume_1m'],
                volume_5m=volume_metrics['volume_5m'],
                volume_1h=volume_metrics['volume_1h'],
                spread_score=spread_score,
                depth_score=depth_score,
                volume_score=volume_score,
                liquidity_score=liquidity_score,
                suitable_for_execution=suitable,
                max_recommended_size=max_size,
                estimated_slippage_bp=estimated_slippage
            )
            
            # Store for analytics
            self.liquidity_history.append(metrics)
            if len(self.liquidity_history) > 1000:
                self.liquidity_history = self.liquidity_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Liquidity evaluation failed for {symbol}: {e}")
            return self._create_empty_metrics(symbol, datetime.now())
    
    def should_execute_trade(self, 
                           liquidity_metrics: LiquidityMetrics,
                           trade_size_quote: float,
                           max_slippage_bp: float = 25.0) -> Dict[str, Any]:
        """
        Determine if trade should be executed based on liquidity conditions
        
        Args:
            liquidity_metrics: Current liquidity assessment
            trade_size_quote: Intended trade size in quote currency
            max_slippage_bp: Maximum acceptable slippage (basis points)
            
        Returns:
            Execution decision with reasoning
        """
        try:
            decision = {
                "execute": False,
                "reason": "",
                "risk_level": "high",
                "recommended_size": 0.0,
                "estimated_slippage": liquidity_metrics.estimated_slippage_bp,
                "liquidity_score": liquidity_metrics.liquidity_score
            }
            
            # Check basic liquidity suitability
            if not liquidity_metrics.suitable_for_execution:
                decision["reason"] = "Failed basic liquidity requirements"
                self._log_rejected_trade(liquidity_metrics, trade_size_quote, "liquidity_fail")
                return decision
            
            # Check spread constraint
            if liquidity_metrics.bid_ask_spread_bp > self.max_spread_bp:
                decision["reason"] = f"Spread too wide: {liquidity_metrics.bid_ask_spread_bp:.1f}bp > {self.max_spread_bp}bp"
                self._log_rejected_trade(liquidity_metrics, trade_size_quote, "spread_wide")
                return decision
            
            # Check size constraint
            if trade_size_quote > liquidity_metrics.max_recommended_size:
                # Offer size reduction
                decision["recommended_size"] = liquidity_metrics.max_recommended_size * 0.8
                decision["reason"] = f"Size too large: ${trade_size_quote:,.0f} > ${liquidity_metrics.max_recommended_size:,.0f}"
                decision["risk_level"] = "medium"
                
                # Still reject if even recommended size is too small
                if decision["recommended_size"] < trade_size_quote * 0.3:
                    self._log_rejected_trade(liquidity_metrics, trade_size_quote, "size_large")
                    return decision
            
            # Check estimated slippage
            estimated_slippage = self._estimate_trade_slippage(
                liquidity_metrics, trade_size_quote
            )
            
            if estimated_slippage > max_slippage_bp:
                decision["reason"] = f"Estimated slippage too high: {estimated_slippage:.1f}bp > {max_slippage_bp}bp"
                decision["estimated_slippage"] = estimated_slippage
                self._log_rejected_trade(liquidity_metrics, trade_size_quote, "slippage_high")
                return decision
            
            # Trade approved
            decision.update({
                "execute": True,
                "reason": "Liquidity conditions acceptable",
                "risk_level": "low" if liquidity_metrics.liquidity_score > 0.8 else "medium",
                "recommended_size": trade_size_quote,
                "estimated_slippage": estimated_slippage
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Trade execution decision failed: {e}")
            return {
                "execute": False,
                "reason": f"Decision error: {e}",
                "risk_level": "high",
                "recommended_size": 0.0,
                "estimated_slippage": 999.0,
                "liquidity_score": 0.0
            }
    
    def get_liquidity_analytics(self, 
                              symbol: Optional[str] = None,
                              hours_back: int = 24) -> Dict[str, Any]:
        """
        Get liquidity analytics for monitoring and optimization
        
        Args:
            symbol: Filter by specific symbol (optional)
            hours_back: Hours of history to analyze
            
        Returns:
            Comprehensive liquidity analytics
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter relevant data
            relevant_history = [
                m for m in self.liquidity_history
                if m.timestamp >= cutoff_time and (symbol is None or m.symbol == symbol)
            ]
            
            relevant_rejects = [
                r for r in self.rejected_trades
                if r['timestamp'] >= cutoff_time and (symbol is None or r['symbol'] == symbol)
            ]
            
            if not relevant_history:
                return {"status": "No liquidity data available"}
            
            # Calculate analytics
            df = pd.DataFrame([{
                'symbol': m.symbol,
                'timestamp': m.timestamp,
                'spread_bp': m.bid_ask_spread_bp,
                'depth_quote': min(m.bid_depth_quote, m.ask_depth_quote),
                'volume_1m': m.volume_1m,
                'liquidity_score': m.liquidity_score,
                'suitable': m.suitable_for_execution
            } for m in relevant_history])
            
            analytics = {
                "period_hours": hours_back,
                "total_assessments": len(relevant_history),
                "unique_symbols": df['symbol'].nunique(),
                "avg_spread_bp": df['spread_bp'].mean(),
                "median_spread_bp": df['spread_bp'].median(),
                "p95_spread_bp": df['spread_bp'].quantile(0.95),
                "avg_depth": df['depth_quote'].mean(),
                "median_depth": df['depth_quote'].median(),
                "avg_volume_1m": df['volume_1m'].mean(),
                "avg_liquidity_score": df['liquidity_score'].mean(),
                "execution_rate": df['suitable'].mean(),
                "rejected_trades": len(relevant_rejects),
                "rejection_reasons": {}
            }
            
            # Rejection reason breakdown
            if relevant_rejects:
                rejection_df = pd.DataFrame(relevant_rejects)
                analytics["rejection_reasons"] = rejection_df['reason'].value_counts().to_dict()
            
            # Symbol-specific analytics if requested
            if symbol and symbol in df['symbol'].values:
                symbol_data = df[df['symbol'] == symbol]
                analytics["symbol_specific"] = {
                    "assessments": len(symbol_data),
                    "avg_spread_bp": symbol_data['spread_bp'].mean(),
                    "avg_liquidity_score": symbol_data['liquidity_score'].mean(),
                    "execution_rate": symbol_data['suitable'].mean()
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Liquidity analytics generation failed: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _calculate_depth(self, book_side: List[List], mid_price: float, 
                        depth_threshold: float = 0.02) -> float:
        """Calculate order book depth within threshold of mid price"""
        try:
            total_depth = 0.0
            
            for price_str, size_str in book_side:
                price = float(price_str)
                size = float(size_str)
                
                # Only count orders within depth_threshold of mid price
                price_deviation = abs(price - mid_price) / mid_price
                if price_deviation <= depth_threshold:
                    total_depth += price * size  # Value in quote currency
                else:
                    break  # Order book is sorted, so we can break early
            
            return total_depth
            
        except Exception as e:
            logger.error(f"Depth calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_metrics(self, 
                                recent_trades: List[Dict[str, Any]],
                                volume_data: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate volume metrics from trades or provided data"""
        try:
            if volume_data:
                return {
                    'volume_1m': volume_data.get('volume_1m', 0.0),
                    'volume_5m': volume_data.get('volume_5m', 0.0),
                    'volume_1h': volume_data.get('volume_1h', 0.0)
                }
            
            # Calculate from recent trades
            now = datetime.now()
            volume_1m = volume_5m = volume_1h = 0.0
            
            for trade in recent_trades:
                try:
                    trade_time = datetime.fromtimestamp(trade.get('timestamp', 0))
                    trade_volume = float(trade.get('cost', 0))  # Volume in quote currency
                    
                    time_diff = (now - trade_time).total_seconds()
                    
                    if time_diff <= 60:  # 1 minute
                        volume_1m += trade_volume
                    if time_diff <= 300:  # 5 minutes
                        volume_5m += trade_volume
                    if time_diff <= 3600:  # 1 hour
                        volume_1h += trade_volume
                        
                except (ValueError, TypeError):
                    continue
            
            return {
                'volume_1m': volume_1m,
                'volume_5m': volume_5m,
                'volume_1h': volume_1h
            }
            
        except Exception as e:
            logger.error(f"Volume calculation failed: {e}")
            return {'volume_1m': 0.0, 'volume_5m': 0.0, 'volume_1h': 0.0}
    
    def _calculate_spread_score(self, spread_bp: float) -> float:
        """Calculate normalized spread score (1 = excellent, 0 = poor)"""
        # Sigmoid function: good spreads (0-10bp) get high scores
        if spread_bp <= 5:
            return 1.0
        elif spread_bp >= 50:
            return 0.0
        else:
            # Linear decay from 5bp to 50bp
            return max(0.0, 1.0 - (spread_bp - 5) / 45)
    
    def _calculate_depth_score(self, min_depth: float) -> float:
        """Calculate normalized depth score (1 = excellent, 0 = poor)"""
        if min_depth >= self.min_depth_quote * 5:  # 5x minimum
            return 1.0
        elif min_depth <= self.min_depth_quote * 0.5:  # Half minimum
            return 0.0
        else:
            # Log scale for depth scoring
            return min(1.0, np.log10(min_depth / self.min_depth_quote) / np.log10(5))
    
    def _calculate_volume_score(self, volume_1m: float) -> float:
        """Calculate normalized volume score (1 = excellent, 0 = poor)"""
        if volume_1m >= self.min_volume_1m * 10:  # 10x minimum
            return 1.0
        elif volume_1m <= self.min_volume_1m * 0.1:  # 10% of minimum
            return 0.0
        else:
            # Log scale for volume scoring
            return min(1.0, np.log10(volume_1m / self.min_volume_1m) / np.log10(10))
    
    def _is_suitable_for_execution(self, spread_bp: float, min_depth: float,
                                  volume_1m: float, liquidity_score: float) -> bool:
        """Determine if conditions meet minimum execution requirements"""
        return (
            spread_bp <= self.max_spread_bp and
            min_depth >= self.min_depth_quote and
            volume_1m >= self.min_volume_1m and
            liquidity_score >= self.min_liquidity_score
        )
    
    def _estimate_execution_impact(self, bids: List[List], asks: List[List],
                                  mid_price: float, volume_1m: float) -> Tuple[float, float]:
        """Estimate maximum size and slippage for execution"""
        try:
            # Conservative sizing: max 10% of 1-minute volume
            volume_based_size = volume_1m * 0.1
            
            # Order book based sizing: sum depth within reasonable slippage
            max_slippage_threshold = 0.005  # 50bp max slippage
            
            ask_depth_within_threshold = 0.0
            for price_str, size_str in asks:
                price = float(price_str)
                size = float(size_str)
                
                if (price - mid_price) / mid_price <= max_slippage_threshold:
                    ask_depth_within_threshold += price * size
                else:
                    break
            
            bid_depth_within_threshold = 0.0
            for price_str, size_str in bids:
                price = float(price_str)
                size = float(size_str)
                
                if (mid_price - price) / mid_price <= max_slippage_threshold:
                    bid_depth_within_threshold += price * size
                else:
                    break
            
            depth_based_size = min(ask_depth_within_threshold, bid_depth_within_threshold) * 0.5
            
            # Take minimum of volume and depth constraints
            max_recommended_size = min(volume_based_size, depth_based_size)
            
            # Estimate slippage for this size
            estimated_slippage = self._estimate_slippage_for_size(
                asks if max_recommended_size > 0 else bids, 
                mid_price, max_recommended_size
            )
            
            return max_recommended_size, estimated_slippage
            
        except Exception as e:
            logger.error(f"Execution impact estimation failed: {e}")
            return 0.0, 999.0
    
    def _estimate_slippage_for_size(self, book_side: List[List], 
                                   mid_price: float, size_quote: float) -> float:
        """Estimate slippage for given trade size"""
        try:
            if size_quote <= 0:
                return 0.0
            
            remaining_size = size_quote
            weighted_price = 0.0
            total_filled = 0.0
            
            for price_str, available_str in book_side:
                price = float(price_str)
                available_quote = price * float(available_str)
                
                if remaining_size <= available_quote:
                    # Partial fill at this level
                    fill_amount = remaining_size
                    weighted_price += fill_amount * price
                    total_filled += fill_amount
                    break
                else:
                    # Full fill at this level
                    weighted_price += available_quote * price
                    total_filled += available_quote
                    remaining_size -= available_quote
            
            if total_filled <= 0:
                return 999.0  # Cannot fill
            
            avg_fill_price = weighted_price / total_filled
            slippage_bp = abs(avg_fill_price - mid_price) / mid_price * 10000
            
            return slippage_bp
            
        except Exception as e:
            logger.error(f"Slippage estimation failed: {e}")
            return 999.0
    
    def _estimate_trade_slippage(self, metrics: LiquidityMetrics, 
                                trade_size: float) -> float:
        """Estimate slippage for specific trade size"""
        # Simple model based on size relative to depth and volume
        try:
            min_depth = min(metrics.bid_depth_quote, metrics.ask_depth_quote)
            
            if min_depth <= 0:
                return 999.0
            
            # Size impact factor
            size_impact = trade_size / min_depth
            
            # Base slippage from spread
            base_slippage = metrics.bid_ask_spread_bp / 2
            
            # Additional slippage from market impact
            impact_slippage = size_impact * 50  # 50bp per depth ratio
            
            total_slippage = base_slippage + impact_slippage
            
            return min(999.0, total_slippage)
            
        except Exception as e:
            logger.error(f"Trade slippage estimation failed: {e}")
            return 999.0
    
    def _log_rejected_trade(self, metrics: LiquidityMetrics, 
                          trade_size: float, reason: str) -> None:
        """Log rejected trade for analytics"""
        try:
            rejection = {
                'timestamp': metrics.timestamp,
                'symbol': metrics.symbol,
                'trade_size': trade_size,
                'reason': reason,
                'spread_bp': metrics.bid_ask_spread_bp,
                'liquidity_score': metrics.liquidity_score,
                'depth_quote': min(metrics.bid_depth_quote, metrics.ask_depth_quote)
            }
            
            self.rejected_trades.append(rejection)
            
            # Keep only recent rejections
            if len(self.rejected_trades) > 500:
                self.rejected_trades = self.rejected_trades[-500:]
                
        except Exception as e:
            logger.error(f"Failed to log rejected trade: {e}")
    
    def _create_empty_metrics(self, symbol: str, timestamp: datetime) -> LiquidityMetrics:
        """Create empty metrics for error cases"""
        return LiquidityMetrics(
            symbol=symbol,
            timestamp=timestamp,
            bid_ask_spread_bp=999.0,
            mid_price=0.0,
            bid_depth_quote=0.0,
            ask_depth_quote=0.0,
            depth_imbalance=0.0,
            volume_1m=0.0,
            volume_5m=0.0,
            volume_1h=0.0,
            spread_score=0.0,
            depth_score=0.0,
            volume_score=0.0,
            liquidity_score=0.0,
            suitable_for_execution=False,
            max_recommended_size=0.0,
            estimated_slippage_bp=999.0
        )