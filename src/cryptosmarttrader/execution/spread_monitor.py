"""
Spread Monitoring and Analysis

Tracks bid-ask spreads across symbols and time to identify
execution opportunities and market microstructure patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SpreadAnalytics:
    """Container for spread analysis results"""
    symbol: str
    period_minutes: int
    
    # Basic spread statistics
    avg_spread_bp: float
    median_spread_bp: float
    min_spread_bp: float
    max_spread_bp: float
    p95_spread_bp: float
    
    # Spread volatility
    spread_volatility: float      # Standard deviation of spreads
    spread_trend: float          # Linear trend over period
    
    # Time-based patterns
    best_execution_times: List[str]  # Hours with tightest spreads
    worst_execution_times: List[str] # Hours with widest spreads
    
    # Market state correlation
    volume_correlation: float    # Correlation between volume and spread
    volatility_correlation: float # Correlation between price vol and spread
    
    # Quality metrics
    execution_quality_score: float  # Overall execution quality (0-1)
    recommended_for_trading: bool


class SpreadMonitor:
    """
    Monitors and analyzes bid-ask spreads for optimal execution timing
    """
    
    def __init__(self, 
                 max_history_hours: int = 24,
                 analysis_window_minutes: int = 60):
        """
        Initialize spread monitor
        
        Args:
            max_history_hours: Maximum hours of spread history to retain
            analysis_window_minutes: Window size for rolling analysis
        """
        self.max_history_hours = max_history_hours
        self.analysis_window_minutes = analysis_window_minutes
        
        # Spread history: symbol -> list of (timestamp, spread_bp, volume, price)
        self.spread_history = defaultdict(list)
        
        # Current market state
        self.current_spreads = {}
        
    def record_spread(self, 
                     symbol: str,
                     bid: float,
                     ask: float,
                     volume: float = 0.0,
                     timestamp: Optional[datetime] = None) -> float:
        """
        Record a new spread observation
        
        Args:
            symbol: Trading pair symbol
            bid: Current best bid price
            ask: Current best ask price
            volume: Current volume (optional)
            timestamp: Observation timestamp (defaults to now)
            
        Returns:
            Calculated spread in basis points
        """
        try:
            timestamp = timestamp or datetime.now()
            
            if bid <= 0 or ask <= 0 or ask <= bid:
                logger.warning(f"Invalid bid/ask for {symbol}: {bid}/{ask}")
                return 999.0
            
            # Calculate spread
            mid_price = (bid + ask) / 2
            spread_bp = ((ask - bid) / mid_price) * 10000
            
            # Record observation
            observation = {
                'timestamp': timestamp,
                'spread_bp': spread_bp,
                'bid': bid,
                'ask': ask,
                'mid_price': mid_price,
                'volume': volume
            }
            
            self.spread_history[symbol].append(observation)
            self.current_spreads[symbol] = spread_bp
            
            # Cleanup old history
            self._cleanup_history(symbol)
            
            return spread_bp
            
        except Exception as e:
            logger.error(f"Failed to record spread for {symbol}: {e}")
            return 999.0
    
    def get_current_spread(self, symbol: str) -> Optional[float]:
        """Get most recent spread for symbol"""
        return self.current_spreads.get(symbol)
    
    def analyze_spread_patterns(self, 
                              symbol: str,
                              period_minutes: int = 60) -> Optional[SpreadAnalytics]:
        """
        Analyze spread patterns for execution optimization
        
        Args:
            symbol: Symbol to analyze
            period_minutes: Analysis period in minutes
            
        Returns:
            Comprehensive spread analysis
        """
        try:
            if symbol not in self.spread_history:
                return None
            
            # Get relevant data
            cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
            relevant_data = [
                obs for obs in self.spread_history[symbol]
                if obs['timestamp'] >= cutoff_time
            ]
            
            if len(relevant_data) < 10:
                logger.warning(f"Insufficient spread data for {symbol}: {len(relevant_data)} observations")
                return None
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(relevant_data)
            
            # Basic statistics
            spreads = df['spread_bp'].values
            avg_spread = np.mean(spreads)
            median_spread = np.median(spreads)
            min_spread = np.min(spreads)
            max_spread = np.max(spreads)
            p95_spread = np.percentile(spreads, 95)
            
            # Volatility and trend
            spread_volatility = np.std(spreads)
            
            # Linear trend (positive = widening spreads)
            if len(spreads) > 1:
                x = np.arange(len(spreads))
                trend_coef = np.polyfit(x, spreads, 1)[0]
            else:
                trend_coef = 0.0
            
            # Time-based patterns
            df['hour'] = df['timestamp'].dt.hour
            hourly_spreads = df.groupby('hour')['spread_bp'].mean()
            
            best_hours = hourly_spreads.nsmallest(3).index.tolist()
            worst_hours = hourly_spreads.nlargest(3).index.tolist()
            
            best_execution_times = [f"{hour:02d}:00" for hour in best_hours]
            worst_execution_times = [f"{hour:02d}:00" for hour in worst_hours]
            
            # Correlations
            volume_correlation = self._calculate_correlation(
                df['volume'].values, spreads
            )
            
            # Price volatility correlation
            if len(df) > 5:
                price_returns = df['mid_price'].pct_change().rolling(5).std()
                volatility_correlation = self._calculate_correlation(
                    price_returns.dropna().values, 
                    spreads[-len(price_returns.dropna()):]
                )
            else:
                volatility_correlation = 0.0
            
            # Execution quality score
            quality_score = self._calculate_execution_quality_score(
                avg_spread, spread_volatility, trend_coef
            )
            
            # Trading recommendation
            recommended = (
                avg_spread < 25.0 and           # Reasonable average spread
                spread_volatility < 10.0 and    # Stable spreads
                quality_score > 0.6             # Good overall quality
            )
            
            return SpreadAnalytics(
                symbol=symbol,
                period_minutes=period_minutes,
                avg_spread_bp=avg_spread,
                median_spread_bp=median_spread,
                min_spread_bp=min_spread,
                max_spread_bp=max_spread,
                p95_spread_bp=p95_spread,
                spread_volatility=spread_volatility,
                spread_trend=trend_coef,
                best_execution_times=best_execution_times,
                worst_execution_times=worst_execution_times,
                volume_correlation=volume_correlation,
                volatility_correlation=volatility_correlation,
                execution_quality_score=quality_score,
                recommended_for_trading=recommended
            )
            
        except Exception as e:
            logger.error(f"Spread analysis failed for {symbol}: {e}")
            return None
    
    def get_best_execution_opportunities(self, 
                                       symbols: Optional[List[str]] = None,
                                       max_spread_bp: float = 20.0) -> List[Dict[str, Any]]:
        """
        Identify current best execution opportunities
        
        Args:
            symbols: List of symbols to check (None = all)
            max_spread_bp: Maximum acceptable spread
            
        Returns:
            List of execution opportunities sorted by quality
        """
        try:
            if symbols is None:
                symbols = list(self.current_spreads.keys())
            
            opportunities = []
            
            for symbol in symbols:
                current_spread = self.get_current_spread(symbol)
                if current_spread is None or current_spread > max_spread_bp:
                    continue
                
                # Analyze recent patterns
                analytics = self.analyze_spread_patterns(symbol, 30)  # 30-minute window
                
                if analytics is None:
                    continue
                
                # Calculate opportunity score
                opportunity_score = self._calculate_opportunity_score(
                    current_spread, analytics
                )
                
                opportunities.append({
                    'symbol': symbol,
                    'current_spread_bp': current_spread,
                    'avg_spread_bp': analytics.avg_spread_bp,
                    'spread_percentile': self._calculate_spread_percentile(
                        symbol, current_spread
                    ),
                    'opportunity_score': opportunity_score,
                    'recommended': analytics.recommended_for_trading,
                    'trend': 'tightening' if analytics.spread_trend < -0.1 else 
                            'widening' if analytics.spread_trend > 0.1 else 'stable'
                })
            
            # Sort by opportunity score (highest first)
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to get execution opportunities: {e}")
            return []
    
    def get_spread_summary(self, 
                          symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary statistics across all monitored symbols
        
        Args:
            symbols: Symbols to include (None = all)
            
        Returns:
            Aggregate spread statistics
        """
        try:
            if symbols is None:
                symbols = list(self.spread_history.keys())
            
            all_spreads = []
            symbol_stats = {}
            
            for symbol in symbols:
                if symbol in self.current_spreads:
                    current_spread = self.current_spreads[symbol]
                    analytics = self.analyze_spread_patterns(symbol, 60)
                    
                    if analytics:
                        all_spreads.append(current_spread)
                        symbol_stats[symbol] = {
                            'current_spread_bp': current_spread,
                            'avg_spread_bp': analytics.avg_spread_bp,
                            'recommended': analytics.recommended_for_trading
                        }
            
            if not all_spreads:
                return {"status": "No spread data available"}
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "symbols_monitored": len(symbols),
                "symbols_with_data": len(all_spreads),
                "market_spread_stats": {
                    "avg_spread_bp": np.mean(all_spreads),
                    "median_spread_bp": np.median(all_spreads),
                    "min_spread_bp": np.min(all_spreads),
                    "max_spread_bp": np.max(all_spreads),
                    "p95_spread_bp": np.percentile(all_spreads, 95)
                },
                "execution_quality": {
                    "tight_spreads_count": sum(1 for s in all_spreads if s < 10),
                    "acceptable_spreads_count": sum(1 for s in all_spreads if s < 25),
                    "wide_spreads_count": sum(1 for s in all_spreads if s >= 25),
                    "recommended_symbols": [
                        symbol for symbol, stats in symbol_stats.items()
                        if stats.get('recommended', False)
                    ]
                },
                "symbol_details": symbol_stats
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate spread summary: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _cleanup_history(self, symbol: str) -> None:
        """Remove old history entries"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            
            self.spread_history[symbol] = [
                obs for obs in self.spread_history[symbol]
                if obs['timestamp'] >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"History cleanup failed for {symbol}: {e}")
    
    def _calculate_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation between two arrays"""
        try:
            if len(x) != len(y) or len(x) < 3:
                return 0.0
            
            # Remove any NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(mask) < 3:
                return 0.0
            
            correlation = np.corrcoef(x[mask], y[mask])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0
    
    def _calculate_execution_quality_score(self, 
                                         avg_spread: float,
                                         spread_volatility: float,
                                         trend: float) -> float:
        """Calculate overall execution quality score (0-1)"""
        try:
            # Spread tightness score (lower spread = higher score)
            if avg_spread <= 5:
                tightness_score = 1.0
            elif avg_spread >= 50:
                tightness_score = 0.0
            else:
                tightness_score = 1.0 - (avg_spread - 5) / 45
            
            # Stability score (lower volatility = higher score)
            if spread_volatility <= 2:
                stability_score = 1.0
            elif spread_volatility >= 20:
                stability_score = 0.0
            else:
                stability_score = 1.0 - (spread_volatility - 2) / 18
            
            # Trend score (tightening spreads = higher score)
            if trend <= -0.5:  # Tightening
                trend_score = 1.0
            elif trend >= 0.5:  # Widening
                trend_score = 0.0
            else:
                trend_score = 0.5 - trend  # Neutral around 0
            
            # Weighted combination
            quality_score = (
                0.5 * tightness_score +
                0.3 * stability_score +
                0.2 * trend_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_opportunity_score(self, 
                                   current_spread: float,
                                   analytics: SpreadAnalytics) -> float:
        """Calculate execution opportunity score"""
        try:
            # How good is current spread vs recent average?
            spread_advantage = max(0, analytics.avg_spread_bp - current_spread) / analytics.avg_spread_bp
            
            # Stability bonus
            stability_bonus = 1.0 - min(1.0, analytics.spread_volatility / 20)
            
            # Trend bonus (prefer tightening spreads)
            trend_bonus = max(0, -analytics.spread_trend / 2)
            
            # Combined score
            opportunity_score = (
                0.6 * spread_advantage +
                0.2 * stability_bonus +
                0.2 * trend_bonus
            )
            
            return max(0.0, min(1.0, opportunity_score))
            
        except Exception as e:
            logger.error(f"Opportunity score calculation failed: {e}")
            return 0.0
    
    def _calculate_spread_percentile(self, symbol: str, current_spread: float) -> float:
        """Calculate what percentile the current spread represents"""
        try:
            if symbol not in self.spread_history:
                return 50.0
            
            # Get recent spreads
            recent_spreads = [
                obs['spread_bp'] for obs in self.spread_history[symbol][-100:]  # Last 100 observations
            ]
            
            if len(recent_spreads) < 10:
                return 50.0
            
            # Calculate percentile
            percentile = (np.sum(np.array(recent_spreads) >= current_spread) / len(recent_spreads)) * 100
            
            return 100 - percentile  # Invert so low spreads = high percentile
            
        except Exception as e:
            logger.error(f"Percentile calculation failed: {e}")
            return 50.0