"""
Regime Features Calculator

Calculates key features for market regime identification:
- Hurst Exponent (trend persistence)
- ADX (trend strength)
- Realized Volatility/ATR
- Market Breadth (dominance/alt breadth)
- Funding/Open Interest impulse
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
import logging

# Try to import talib, fallback to custom implementation if not available
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatureSet:
    """Container for regime detection features"""
    hurst_exponent: float
    adx: float
    realized_vol: float
    atr_normalized: float
    btc_dominance: float
    alt_breadth: float
    funding_impulse: float
    oi_impulse: float
    volatility_regime: str  # 'low', 'high'
    trend_strength: str     # 'weak', 'strong'
    

class RegimeFeatures:
    """
    Advanced feature calculator for regime detection
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'hurst': 100,
            'adx': 14,
            'vol': 20,
            'atr': 14,
            'funding': 24,  # hours
            'oi': 48       # hours
        }
        
    def calculate_hurst_exponent(self, prices: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent for trend persistence
        H > 0.5: trending (persistent)
        H < 0.5: mean-reverting (anti-persistent)
        H = 0.5: random walk
        """
        try:
            prices = np.array(prices.dropna())
            if len(prices) < max_lag * 2:
                return 0.5  # neutral/unknown
                
            # Calculate log returns
            log_returns = np.diff(np.log(prices))
            
            # R/S Analysis
            lags = range(2, max_lag + 1)
            rs_values = []
            
            for lag in lags:
                # Split series into chunks of length 'lag'
                chunks = [log_returns[i:i+lag] for i in range(0, len(log_returns), lag)]
                chunks = [chunk for chunk in chunks if len(chunk) == lag]
                
                if not chunks:
                    continue
                    
                rs_chunk_values = []
                for chunk in chunks:
                    # Mean-adjusted series
                    mean_chunk = np.mean(chunk)
                    adjusted = chunk - mean_chunk
                    
                    # Cumulative sum
                    cumsum = np.cumsum(adjusted)
                    
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    
                    # Standard deviation
                    S = np.std(chunk, ddof=1) if len(chunk) > 1 else 1e-8
                    
                    # R/S ratio
                    if S > 0:
                        rs_chunk_values.append(R / S)
                
                if rs_chunk_values:
                    rs_values.append(np.mean(rs_chunk_values))
            
            if len(rs_values) < 3:
                return 0.5
                
            # Linear regression of log(R/S) vs log(lag)
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove any inf or nan values
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5
                
            log_lags = log_lags[valid_mask]
            log_rs = log_rs[valid_mask]
            
            # Linear regression
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Bound between 0 and 1
            return max(0, min(1, hurst))
            
        except Exception as e:
            logger.warning(f"Hurst calculation failed: {e}")
            return 0.5
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = None) -> float:
        """Calculate Average Directional Index for trend strength"""
        try:
            period = period or self.lookback_periods['adx']
            
            if TALIB_AVAILABLE and talib is not None:
                adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
                return float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            else:
                # Custom ADX implementation
                return self._calculate_adx_custom(high, low, close, period)
                
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            return 0.0
            
    def _calculate_adx_custom(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Custom ADX calculation when talib is not available"""
        try:
            # True Range calculation
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
            
            # Smoothed averages
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
            
            # ADX calculation
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.warning(f"Custom ADX calculation failed: {e}")
            return 20.0  # Default moderate trend strength
    
    def calculate_realized_volatility(self, returns: pd.Series, 
                                     period: int = None) -> float:
        """Calculate realized volatility (annualized)"""
        try:
            period = period or self.lookback_periods['vol']
            recent_returns = returns.tail(period).dropna()
            
            if len(recent_returns) < 5:
                return 0.0
                
            # Annualized volatility (assuming hourly data)
            vol = recent_returns.std() * np.sqrt(365 * 24)
            return float(vol) if not np.isnan(vol) else 0.0
            
        except Exception as e:
            logger.warning(f"Realized volatility calculation failed: {e}")
            return 0.0
    
    def calculate_atr_normalized(self, high: pd.Series, low: pd.Series, 
                                close: pd.Series, period: int = None) -> float:
        """Calculate ATR normalized by price (as percentage)"""
        try:
            period = period or self.lookback_periods['atr']
            
            if TALIB_AVAILABLE and talib is not None:
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                
                if np.isnan(atr[-1]) or close.iloc[-1] == 0:
                    return 0.0
                    
                # Normalize by current price
                atr_pct = (atr[-1] / close.iloc[-1]) * 100
                return float(atr_pct)
            else:
                # Custom ATR implementation
                return self._calculate_atr_custom(high, low, close, period)
            
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.0
            
    def _calculate_atr_custom(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Custom ATR calculation when talib is not available"""
        try:
            # True Range calculation
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR as rolling mean of True Range
            atr = tr.rolling(period).mean()
            
            if np.isnan(atr.iloc[-1]) or close.iloc[-1] == 0:
                return 0.0
                
            # Normalize by current price
            atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100
            return float(atr_pct)
            
        except Exception as e:
            logger.warning(f"Custom ATR calculation failed: {e}")
            return 2.0  # Default moderate volatility
    
    def calculate_market_breadth(self, dominance_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate market breadth metrics
        
        Args:
            dominance_data: {'BTC': 45.2, 'ETH': 18.5, 'others': 36.3}
        """
        try:
            btc_dominance = dominance_data.get('BTC', 50.0)
            eth_dominance = dominance_data.get('ETH', 15.0)
            
            # Alt breadth = how spread the non-BTC market is
            alt_market_cap = 100 - btc_dominance
            major_alts = eth_dominance  # Could include more top alts
            
            # Higher score means more distributed alt market
            alt_breadth = (alt_market_cap - major_alts) / alt_market_cap if alt_market_cap > 0 else 0
            
            return {
                'btc_dominance': btc_dominance,
                'alt_breadth': alt_breadth * 100  # Convert to percentage
            }
            
        except Exception as e:
            logger.warning(f"Market breadth calculation failed: {e}")
            return {'btc_dominance': 50.0, 'alt_breadth': 50.0}
    
    def calculate_funding_impulse(self, funding_rates: pd.Series, 
                                 period: int = None) -> float:
        """
        Calculate funding rate impulse (change acceleration)
        
        High positive = leveraged longs building (risk on)
        High negative = leveraged shorts building (risk off)
        """
        try:
            period = period or self.lookback_periods['funding']
            recent_funding = funding_rates.tail(period).dropna()
            
            if len(recent_funding) < 5:
                return 0.0
                
            # Calculate first and second derivatives
            first_diff = recent_funding.diff().dropna()
            second_diff = first_diff.diff().dropna()
            
            # Funding impulse = recent change in funding rate momentum
            impulse = second_diff.tail(3).mean() * 10000  # Scale for readability
            
            return float(impulse) if not np.isnan(impulse) else 0.0
            
        except Exception as e:
            logger.warning(f"Funding impulse calculation failed: {e}")
            return 0.0
    
    def calculate_oi_impulse(self, open_interest: pd.Series, 
                            volume: pd.Series, period: int = None) -> float:
        """
        Calculate Open Interest impulse relative to volume
        
        High OI growth + High volume = strong directional conviction
        High OI growth + Low volume = potential squeeze setup
        """
        try:
            period = period or self.lookback_periods['oi']
            
            oi_recent = open_interest.tail(period).dropna()
            vol_recent = volume.tail(period).dropna()
            
            if len(oi_recent) < 5 or len(vol_recent) < 5:
                return 0.0
            
            # OI growth rate
            oi_growth = oi_recent.pct_change().dropna()
            
            # Volume normalization 
            vol_ma = vol_recent.rolling(7).mean()
            vol_ratio = vol_recent / vol_ma
            
            # Combine OI growth with volume context
            # Higher when OI grows with high volume (conviction)
            # Lower when OI grows with low volume (potential reversal setup)
            min_len = min(len(oi_growth), len(vol_ratio))
            if min_len < 3:
                return 0.0
                
            oi_impulse = (oi_growth.tail(min_len) * vol_ratio.tail(min_len)).mean()
            
            return float(oi_impulse * 100) if not np.isnan(oi_impulse) else 0.0
            
        except Exception as e:
            logger.warning(f"OI impulse calculation failed: {e}")
            return 0.0
    
    def classify_volatility_regime(self, realized_vol: float, 
                                  historical_vols: pd.Series) -> str:
        """Classify current volatility as low/high vs historical"""
        try:
            if len(historical_vols) < 20:
                return 'unknown'
                
            vol_percentile = (historical_vols < realized_vol).mean() * 100
            
            if vol_percentile > 70:
                return 'high'
            elif vol_percentile < 30:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            logger.warning(f"Volatility regime classification failed: {e}")
            return 'unknown'
    
    def classify_trend_strength(self, hurst: float, adx: float) -> str:
        """Classify trend strength based on Hurst and ADX"""
        try:
            # Strong trend: high persistence + high directional movement
            if hurst > 0.55 and adx > 25:
                return 'strong'
            elif hurst < 0.45 or adx < 15:
                return 'weak'
            else:
                return 'medium'
                
        except Exception as e:
            logger.warning(f"Trend strength classification failed: {e}")
            return 'unknown'
    
    def calculate_all_features(self, market_data: Dict[str, pd.DataFrame],
                              dominance_data: Dict[str, float],
                              funding_data: Optional[pd.Series] = None,
                              oi_data: Optional[pd.Series] = None) -> RegimeFeatureSet:
        """
        Calculate all regime features from market data
        
        Args:
            market_data: {'BTC/USD': DataFrame with OHLCV, ...}
            dominance_data: Current dominance percentages
            funding_data: Funding rates time series (optional)
            oi_data: Open interest time series (optional)
        """
        try:
            # Use BTC as primary market proxy
            btc_data = market_data.get('BTC/USD')
            if btc_data is None or len(btc_data) < 50:
                logger.warning("Insufficient BTC data for regime analysis")
                return self._default_feature_set()
            
            # Calculate returns for volatility and Hurst
            returns = btc_data['close'].pct_change().dropna()
            
            # Core features
            hurst = self.calculate_hurst_exponent(btc_data['close'])
            adx = self.calculate_adx(btc_data['high'], btc_data['low'], btc_data['close'])
            realized_vol = self.calculate_realized_volatility(returns)
            atr_norm = self.calculate_atr_normalized(
                btc_data['high'], btc_data['low'], btc_data['close']
            )
            
            # Market breadth
            breadth = self.calculate_market_breadth(dominance_data)
            
            # Derivatives data (optional)
            funding_impulse = 0.0
            oi_impulse = 0.0
            
            if funding_data is not None and len(funding_data) > 24:
                funding_impulse = self.calculate_funding_impulse(funding_data)
                
            if oi_data is not None and len(oi_data) > 48:
                volume_proxy = btc_data['volume'] if 'volume' in btc_data.columns else None
                if volume_proxy is not None:
                    oi_impulse = self.calculate_oi_impulse(oi_data, volume_proxy)
            
            # Classifications
            historical_vols = returns.rolling(100).std() * np.sqrt(365 * 24)
            vol_regime = self.classify_volatility_regime(realized_vol, historical_vols.dropna())
            trend_strength = self.classify_trend_strength(hurst, adx)
            
            return RegimeFeatureSet(
                hurst_exponent=hurst,
                adx=adx,
                realized_vol=realized_vol,
                atr_normalized=atr_norm,
                btc_dominance=breadth['btc_dominance'],
                alt_breadth=breadth['alt_breadth'],
                funding_impulse=funding_impulse,
                oi_impulse=oi_impulse,
                volatility_regime=vol_regime,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            return self._default_feature_set()
    
    def _default_feature_set(self) -> RegimeFeatureSet:
        """Return default/neutral feature set when calculation fails"""
        return RegimeFeatureSet(
            hurst_exponent=0.5,
            adx=20.0,
            realized_vol=50.0,
            atr_normalized=2.0,
            btc_dominance=45.0,
            alt_breadth=50.0,
            funding_impulse=0.0,
            oi_impulse=0.0,
            volatility_regime='medium',
            trend_strength='medium'
        )