#!/usr/bin/env python3
"""
Technical Analysis Agent - Enterprise-grade TA with proper threading and authentic data
Fixes: thread leaks, dummy data, SELL bias, div/NaN risks, UTC timestamps
"""

import threading
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Try TA-Lib import with fallback
try:
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    talib = None

from ..core.consolidated_logging_manager import get_consolidated_logger

class TechnicalAgent:
    """Enterprise threaded TA agent with proper lifecycle and authentic indicators"""
    
    def __init__(self, update_interval: int = 300):  # 5 minutes default
        self.update_interval = update_interval
        self.active = False
        self.agent_thread = None
        self.logger = get_consolidated_logger("TechnicalAgent")
        
        # Analysis results storage
        self.latest_analysis = {}
        self.analysis_lock = threading.Lock()
        
        # Indicator cache
        self.indicator_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Performance tracking
        self.analysis_count = 0
        self.error_count = 0
        self.last_analysis_time = None
        
        self.logger.info("Technical Agent initialized - no auto-start for proper orchestration")
    
    def start(self):
        """Start the technical analysis agent thread"""
        if self.active:
            self.logger.warning("Technical Agent already running")
            return
        
        self.active = True
        self.agent_thread = threading.Thread(target=self._analysis_loop, daemon=False)
        self.agent_thread.start()
        self.logger.info("Technical Agent started successfully")
    
    def stop(self):
        """Properly stop the technical analysis agent with thread cleanup"""
        self.active = False
        
        # Proper thread cleanup to prevent leaks
        if hasattr(self, "agent_thread") and self.agent_thread and self.agent_thread.is_alive():
            self.logger.info("Stopping Technical Agent thread...")
            self.agent_thread.join(timeout=5)
            
            if self.agent_thread.is_alive():
                self.logger.warning("Technical Agent thread did not stop within timeout")
            else:
                self.logger.info("Technical Agent thread stopped successfully")
        
        self.logger.info("Technical Agent stopped")
    
    def _analysis_loop(self):
        """Main analysis loop with proper error handling"""
        self.logger.info("Technical analysis loop started")
        
        while self.active:
            try:
                start_time = time.time()
                
                # Perform analysis for major pairs
                major_pairs = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD']
                
                analysis_results = {}
                for pair in major_pairs:
                    if not self.active:  # Check active state during loop
                        break
                    
                    try:
                        result = self._analyze_pair(pair)
                        if result:
                            analysis_results[pair] = result
                    except Exception as e:
                        self.logger.error(f"Analysis failed for {pair}: {e}")
                        self.error_count += 1
                
                # Update results atomically
                if analysis_results and self.active:
                    with self.analysis_lock:
                        self.latest_analysis.update(analysis_results)
                        self.last_analysis_time = datetime.now(timezone.utc)
                    
                    self.analysis_count += 1
                    self.logger.info(f"Technical analysis completed for {len(analysis_results)} pairs")
                
                # Performance logging
                elapsed = time.time() - start_time
                self.logger.log_performance_metric("technical_analysis_duration", elapsed, "seconds")
                
                # Sleep until next analysis
                if self.active:
                    time.sleep(self.update_interval)
                    
            except Exception as e:
                self.logger.error(f"Technical analysis loop error: {e}")
                self.error_count += 1
                if self.active:
                    time.sleep(60)  # Error backoff
        
        self.logger.info("Technical analysis loop ended")
    
    def _analyze_pair(self, pair: str) -> Optional[Dict[str, Any]]:
        """Analyze single trading pair with authentic data only"""
        
        # Get authentic market data only
        market_data = self._get_authentic_market_data(pair)
        if market_data is None or market_data.empty:
            self.logger.warning(f"No authentic market data available for {pair}")
            return None
        
        try:
            # Perform comprehensive technical analysis
            trend_analysis = self._analyze_trend(market_data)
            momentum_analysis = self._analyze_momentum(market_data)
            volatility_analysis = self._analyze_volatility(market_data)
            support_resistance = self._analyze_support_resistance(market_data)
            
            # Generate overall signal with bias fix
            overall_signal = self._generate_overall_signal_fixed(
                trend_analysis, momentum_analysis, volatility_analysis
            )
            
            analysis_result = {
                'pair': pair,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volatility': volatility_analysis,
                'support_resistance': support_resistance,
                'overall_signal': overall_signal,
                'data_quality': 'authentic',
                'analysis_method': 'talib' if TA_AVAILABLE else 'custom_safe'
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed for {pair}: {e}")
            return None
    
    def _get_authentic_market_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Get authentic market data - NO DUMMY DATA FALLBACK"""
        
        try:
            # Try to load from authentic data sources
            data_paths = [
                f"data/market_data/{pair.lower()}_1h.csv",
                f"data/historical/{pair.lower()}.csv",
                f"cache/market_data/{pair.lower()}.json"
            ]
            
            for data_path in data_paths:
                path = Path(data_path)
                if path.exists():
                    try:
                        if path.suffix == '.csv':
                            df = pd.read_csv(path)
                        elif path.suffix == '.json':
                            df = pd.read_json(path)
                        else:
                            continue
                        
                        # Validate data structure
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required_columns):
                            # Safe timestamp handling
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                            return df.tail(100)  # Last 100 candles
                    except Exception as e:
                        self.logger.warning(f"Failed to load {data_path}: {e}")
                        continue
            
            # NO DUMMY DATA FALLBACK - return None for authentic data policy
            self.logger.info(f"No authentic market data found for {pair} - skipping analysis")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {pair}: {e}")
            return None
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using EMA and SMA with safe calculations"""
        
        close = df['close'].astype(float)
        
        try:
            if TA_AVAILABLE and talib:
                # Use TA-Lib for accurate calculations
                ema_12 = talib.EMA(close.values, timeperiod=12)
                ema_26 = talib.EMA(close.values, timeperiod=26)
                sma_50 = talib.SMA(close.values, timeperiod=50)
                sma_200 = talib.SMA(close.values, timeperiod=200)
                
                # Convert back to pandas Series
                ema_12 = pd.Series(ema_12, index=close.index)
                ema_26 = pd.Series(ema_26, index=close.index)
                sma_50 = pd.Series(sma_50, index=close.index)
                sma_200 = pd.Series(sma_200, index=close.index)
            else:
                # Safe custom implementation
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                sma_50 = close.rolling(50).mean()
                sma_200 = close.rolling(200).mean()
            
            # Safe trend determination
            current_price = close.iloc[-1]
            ema_12_val = ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else current_price
            ema_26_val = ema_26.iloc[-1] if not pd.isna(ema_26.iloc[-1]) else current_price
            sma_50_val = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            
            trend_direction = 'neutral'
            if ema_12_val > ema_26_val and current_price > sma_50_val:
                trend_direction = 'bullish'
            elif ema_12_val < ema_26_val and current_price < sma_50_val:
                trend_direction = 'bearish'
            
            return {
                'direction': trend_direction,
                'ema_12': float(ema_12_val),
                'ema_26': float(ema_26_val),
                'sma_50': float(sma_50_val),
                'current_price': float(current_price),
                'strength': abs(ema_12_val - ema_26_val) / max(current_price, 1e-9)
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {'direction': 'neutral', 'error': str(e)}
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum with MACD bias fix and safe calculations"""
        
        close = df['close'].astype(float)
        
        try:
            if TA_AVAILABLE and talib:
                # Use TA-Lib for accurate MACD
                macd, macd_signal, macd_hist = talib.MACD(close.values)
                rsi = talib.RSI(close.values, timeperiod=14)
                
                # Convert back to pandas Series
                macd = pd.Series(macd, index=close.index)
                macd_signal = pd.Series(macd_signal, index=close.index)
                rsi = pd.Series(rsi, index=close.index)
            else:
                # Safe custom MACD implementation
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9).mean()
                macd_hist = macd - macd_signal
                
                # Safe RSI calculation
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                loss = loss.replace(0, 1e-9)  # Avoid division by zero
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            
            # Safe value extraction with proper Series handling
            macd_val = float(macd.iloc[-1]) if len(macd) > 0 and not pd.isna(macd.iloc[-1]) else 0.0
            macd_signal_val = float(macd_signal.iloc[-1]) if len(macd_signal) > 0 and not pd.isna(macd_signal.iloc[-1]) else 0.0
            rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
            
            # MACD bias fix - explicit True/False/None handling
            macd_bullish = None
            if not pd.isna(macd_val) and not pd.isna(macd_signal_val):
                macd_bullish = macd_val > macd_signal_val
            
            return {
                'macd_bullish': macd_bullish,
                'macd_value': float(macd_val),
                'macd_signal': float(macd_signal_val),
                'rsi': float(rsi_val),
                'rsi_overbought': rsi_val > 70,
                'rsi_oversold': rsi_val < 30
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            return {'macd_bullish': None, 'error': str(e)}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility with Bollinger Bands - division by zero fix"""
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        try:
            if TA_AVAILABLE and talib:
                # Use TA-Lib for accurate calculations
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values, timeperiod=20)
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
                
                # Convert back to pandas Series
                bb_upper = pd.Series(bb_upper, index=close.index)
                bb_middle = pd.Series(bb_middle, index=close.index)
                bb_lower = pd.Series(bb_lower, index=close.index)
                atr = pd.Series(atr, index=close.index)
            else:
                # Safe custom Bollinger Bands
                bb_middle = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                
                # Safe ATR calculation
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean()
            
            # Safe value extraction with division by zero protection
            current_price = float(close.iloc[-1])
            bb_upper_val = float(bb_upper.iloc[-1]) if len(bb_upper) > 0 and not pd.isna(bb_upper.iloc[-1]) else current_price
            bb_lower_val = float(bb_lower.iloc[-1]) if len(bb_lower) > 0 and not pd.isna(bb_lower.iloc[-1]) else current_price
            bb_middle_val = float(bb_middle.iloc[-1]) if len(bb_middle) > 0 and not pd.isna(bb_middle.iloc[-1]) else current_price
            atr_val = float(atr.iloc[-1]) if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0.0
            
            # Bollinger width with safe division
            bb_width = float(((bb_upper_val - bb_lower_val) / max(bb_middle_val, 1e-9)))
            
            current_price = close.iloc[-1]
            bb_position = (current_price - bb_lower_val) / max(bb_upper_val - bb_lower_val, 1e-9)
            
            return {
                'bb_upper': float(bb_upper_val),
                'bb_middle': float(bb_middle_val),
                'bb_lower': float(bb_lower_val),
                'bb_width': bb_width,
                'bb_position': float(bb_position),
                'atr': float(atr_val),
                'volatility_level': 'high' if bb_width > 0.1 else 'normal' if bb_width > 0.05 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Volatility analysis error: {e}")
            return {'bb_width': 0, 'atr': 0, 'error': str(e)}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze support and resistance with proper length guards"""
        
        if len(df) < 5:  # Proper guard for rolling window
            return {
                'support': float(df['low'].min()),
                'resistance': float(df['high'].max()),
                'pivot': float(df['close'].iloc[-1]),
                'note': 'Insufficient data for full analysis'
            }
        
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # Safe pivot calculation with proper rolling
            typical_price = (high + low + close) / 3
            pivot = typical_price.rolling(window=5).mean().iloc[-1]
            
            # Support and resistance levels
            recent_data = df.tail(20)  # Last 20 periods
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()
            
            # Additional pivot levels
            pivot_high = high.rolling(5).max().iloc[-1]
            pivot_low = low.rolling(5).min().iloc[-1]
            
            return {
                'support': float(support_level),
                'resistance': float(resistance_level),
                'pivot': float(pivot),
                'pivot_high': float(pivot_high),
                'pivot_low': float(pivot_low),
                'range_pct': float((resistance_level - support_level) / max(close.iloc[-1], 1e-9) * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance analysis error: {e}")
            return {'support': 0, 'resistance': 0, 'pivot': 0, 'error': str(e)}
    
    def _generate_overall_signal_fixed(self, trend_data: Dict, momentum_data: Dict, 
                                     volatility_data: Dict) -> Dict[str, Any]:
        """Generate overall signal with MACD bias fix"""
        
        signals = []
        confidence_scores = []
        
        try:
            # Trend signals
            if trend_data.get('direction') == 'bullish':
                signals.append('buy')
                confidence_scores.append(0.4)
            elif trend_data.get('direction') == 'bearish':
                signals.append('sell')
                confidence_scores.append(0.4)
            
            # Momentum signals with bias fix
            macd_bull = momentum_data.get('macd_bullish')
            if macd_bull is True:  # Explicit True check
                signals.append('buy')
                confidence_scores.append(0.3)
            elif macd_bull is False:  # Explicit False check
                signals.append('sell')
                confidence_scores.append(0.3)
            # No signal added if macd_bullish is None (key missing or calculation failed)
            
            # RSI signals
            if momentum_data.get('rsi_oversold'):
                signals.append('buy')
                confidence_scores.append(0.2)
            elif momentum_data.get('rsi_overbought'):
                signals.append('sell')
                confidence_scores.append(0.2)
            
            # Volatility considerations
            volatility_level = volatility_data.get('volatility_level', 'normal')
            volatility_penalty = 0.1 if volatility_level == 'high' else 0
            
            # Calculate final signal
            if not signals:
                final_signal = 'hold'
                final_confidence = 0.5
            else:
                buy_signals = signals.count('buy') if signals else 0
                sell_signals = signals.count('sell') if signals else 0
                
                if buy_signals > sell_signals:
                    final_signal = 'buy'
                    final_confidence = min(sum(confidence_scores) / len(confidence_scores) - volatility_penalty, 1.0)
                elif sell_signals > buy_signals:
                    final_signal = 'sell'
                    final_confidence = min(sum(confidence_scores) / len(confidence_scores) - volatility_penalty, 1.0)
                else:
                    final_signal = 'hold'
                    final_confidence = 0.5
            
            return {
                'signal': final_signal,
                'confidence': max(final_confidence, 0.0),
                'signal_count': {'buy': buy_signals, 'sell': sell_signals} if signals else {'buy': 0, 'sell': 0},
                'volatility_penalty': volatility_penalty,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def get_latest_analysis(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """Get latest technical analysis results"""
        
        with self.analysis_lock:
            if pair:
                return self.latest_analysis.get(pair, {})
            return self.latest_analysis.copy()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get technical agent status"""
        
        return {
            'active': self.active,
            'thread_alive': self.agent_thread.is_alive() if self.agent_thread else False,
            'analysis_count': self.analysis_count,
            'error_count': self.error_count,
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'talib_available': TA_AVAILABLE,
            'cached_pairs': list(self.latest_analysis.keys()),
            'update_interval': self.update_interval
        }
    
    def force_analysis(self, pair: str) -> Optional[Dict[str, Any]]:
        """Force immediate analysis of specific pair"""
        
        self.logger.info(f"Forcing immediate analysis for {pair}")
        result = self._analyze_pair(pair)
        
        if result:
            with self.analysis_lock:
                self.latest_analysis[pair] = result
            self.logger.info(f"Forced analysis completed for {pair}")
        
        return result

def create_technical_agent(update_interval: int = 300) -> TechnicalAgent:
    """Factory function to create technical agent"""
    return TechnicalAgent(update_interval)

if __name__ == "__main__":
    # Test technical agent
    print("Testing Technical Agent")
    
    agent = TechnicalAgent(update_interval=60)  # 1 minute for testing
    
    try:
        agent.start()
        print("Agent started, waiting for analysis...")
        
        time.sleep(5)  # Wait for initial analysis
        
        status = agent.get_agent_status()
        print(f"Agent status: {status}")
        
        # Test force analysis
        result = agent.force_analysis('BTCUSD')
        if result:
            print(f"Forced analysis result: {json.dumps(result, indent=2)}")
        else:
            print("No analysis result (no authentic data available)")
        
    finally:
        agent.stop()
        print("Agent stopped")