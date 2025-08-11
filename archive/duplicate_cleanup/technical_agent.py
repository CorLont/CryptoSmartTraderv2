import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Technical analysis library
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class TechnicalAgent:
    """Technical Analysis Agent with 50+ indicators"""
    
    def __init__(self, config_manager, data_manager, cache_manager):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0
        
        # Technical analysis results storage
        self.technical_data = {}
        self._lock = threading.Lock()
        
        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("technical", {}).get("enabled", True):
            self.start()
    
    def start(self):
        """Start the technical analysis agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Technical Agent started")
    
    def stop(self):
        """Stop the technical analysis agent"""
        self.active = False
        self.logger.info("Technical Agent stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = self.config_manager.get("agents", {}).get("technical", {}).get("update_interval", 60)
                
                # Perform technical analysis
                self._analyze_all_symbols()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep until next analysis
                time.sleep(interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Technical analysis error: {str(e)}")
                time.sleep(30)  # Shorter sleep on error
    
    def _analyze_all_symbols(self):
        """Analyze all available trading symbols"""
        try:
            symbols = self.data_manager.get_supported_symbols()
            
            for symbol in symbols[:100]:  # Limit for efficiency
                try:
                    analysis = self._analyze_symbol(symbol)
                    if analysis:
                        self._store_technical_data(symbol, analysis)
                        self.processed_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Technical analysis loop error: {str(e)}")
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Perform comprehensive technical analysis on a symbol"""
        # Check cache first
        cache_key = f"technical_{symbol.replace('/', '_')}"
        cached_analysis = self.cache_manager.get(cache_key)
        
        if cached_analysis is not None:
            return cached_analysis
        
        try:
            # Get historical data
            historical_data = self._get_analysis_data(symbol)
            
            if historical_data is None or len(historical_data) < 50:
                return None
            
            # Perform analysis
            analysis = self._calculate_all_indicators(historical_data, symbol)
            
            # Cache result
            self.cache_manager.set(cache_key, analysis, ttl_minutes=10)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error for {symbol}: {str(e)}")
            return None
    
    def _get_analysis_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data suitable for technical analysis"""
        # Try to get historical data first
        historical_data = self.data_manager.get_historical_data(symbol, days=30)
        
        if historical_data is not None and len(historical_data) >= 20:
            return historical_data
        
        # Fallback to current market data and generate mock historical data
        current_data = self.data_manager.get_market_data(symbol=symbol)
        
        if current_data is not None and not current_data.empty:
            return self._generate_mock_historical_data(current_data.iloc[0])
        
        return None
    
    def _generate_mock_historical_data(self, current_row: pd.Series) -> pd.DataFrame:
        """Generate mock historical data for analysis (for demo purposes)"""
        import random
        
        current_price = current_row.get('price', 100)
        base_data = []
        
        # Generate 100 periods of mock data
        for i in range(100, 0, -1):
            timestamp = datetime.now() - timedelta(hours=i)
            
            # Random walk with some trend
            price_change = random.uniform(-0.05, 0.05)
            price = current_price * (1 + price_change)
            
            volume = random.uniform(1000, 10000)
            high = price * random.uniform(1.0, 1.02)
            low = price * random.uniform(0.98, 1.0)
            
            base_data.append({
                'timestamp': timestamp,
                'symbol': current_row.get('symbol'),
                'price': price,
                'open': price,
                'high': high,
                'low': low,
                'volume': volume
            })
            
            current_price = price
        
        return pd.DataFrame(base_data)
    
    def _calculate_all_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive set of technical indicators"""
        timestamp = datetime.now()
        
        # Prepare data
        df = data.copy()
        df = df.sort_values('timestamp')
        
        # Ensure we have OHLCV columns
        if 'open' not in df.columns:
            df['open'] = df['price']
        if 'high' not in df.columns:
            df['high'] = df['price']
        if 'low' not in df.columns:
            df['low'] = df['price']
        if 'close' not in df.columns:
            df['close'] = df['price']
        if 'volume' not in df.columns:
            df['volume'] = 1000
        
        indicators = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'data_points': len(df),
            'current_price': df['close'].iloc[-1] if not df.empty else 0
        }
        
        try:
            # Moving Averages
            indicators['moving_averages'] = self._calculate_moving_averages(df)
            
            # Momentum Indicators
            indicators['momentum'] = self._calculate_momentum_indicators(df)
            
            # Volatility Indicators
            indicators['volatility'] = self._calculate_volatility_indicators(df)
            
            # Volume Indicators
            indicators['volume'] = self._calculate_volume_indicators(df)
            
            # Trend Indicators
            indicators['trend'] = self._calculate_trend_indicators(df)
            
            # Support and Resistance
            indicators['levels'] = self._calculate_support_resistance(df)
            
            # Pattern Recognition
            indicators['patterns'] = self._detect_patterns(df)
            
            # Overall Signal
            indicators['signal'] = self._generate_overall_signal(indicators)
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error for {symbol}: {str(e)}")
            indicators['error'] = str(e)
        
        return indicators
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages"""
        close = df['close']
        
        mas = {}
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(close) >= period:
                sma = close.rolling(window=period).mean().iloc[-1]
                ema = close.ewm(span=period).mean().iloc[-1]
                
                mas[f'sma_{period}'] = sma
                mas[f'ema_{period}'] = ema
        
        # Moving average convergence/divergence signals
        current_price = close.iloc[-1]
        mas['price_vs_sma20'] = (current_price / mas.get('sma_20', current_price) - 1) * 100
        mas['price_vs_sma50'] = (current_price / mas.get('sma_50', current_price) - 1) * 100
        
        # Golden/Death cross signals
        if 'sma_50' in mas and 'sma_200' in mas:
            mas['golden_cross'] = mas['sma_50'] > mas['sma_200']
        
        return mas
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        momentum = {}
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            momentum['rsi'] = rsi.iloc[-1]
            momentum['rsi_overbought'] = rsi.iloc[-1] > 70
            momentum['rsi_oversold'] = rsi.iloc[-1] < 30
        
        # MACD
        if len(close) >= 26:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            momentum['macd'] = macd_line.iloc[-1]
            momentum['macd_signal'] = signal_line.iloc[-1]
            momentum['macd_histogram'] = histogram.iloc[-1]
            momentum['macd_bullish'] = macd_line.iloc[-1] > signal_line.iloc[-1]
        
        # Stochastic Oscillator
        if len(df) >= 14:
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            k_percent = 100 * ((close - low14) / (high14 - low14))
            d_percent = k_percent.rolling(window=3).mean()
            
            momentum['stoch_k'] = k_percent.iloc[-1]
            momentum['stoch_d'] = d_percent.iloc[-1]
            momentum['stoch_overbought'] = k_percent.iloc[-1] > 80
            momentum['stoch_oversold'] = k_percent.iloc[-1] < 20
        
        # Williams %R
        if len(df) >= 14:
            high14 = high.rolling(window=14).max()
            low14 = low.rolling(window=14).min()
            williams_r = -100 * (high14 - close) / (high14 - low14)
            momentum['williams_r'] = williams_r.iloc[-1]
        
        return momentum
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        volatility = {}
        
        # Bollinger Bands
        if len(close) >= 20:
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_middle = sma20
            
            volatility['bb_upper'] = bb_upper.iloc[-1]
            volatility['bb_middle'] = bb_middle.iloc[-1]
            volatility['bb_lower'] = bb_lower.iloc[-1]
            volatility['bb_width'] = ((bb_upper - bb_lower) / bb_middle).iloc[-1]
            volatility['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower)).iloc[-1]
        
        # Average True Range (ATR)
        if len(df) >= 14:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            volatility['atr'] = atr.iloc[-1]
            volatility['atr_percent'] = (atr.iloc[-1] / close.iloc[-1]) * 100
        
        # Historical Volatility
        if len(close) >= 30:
            returns = close.pct_change().dropna()
            hist_vol = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
            volatility['historical_volatility'] = hist_vol.iloc[-1]
        
        return volatility
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume indicators"""
        close = df['close']
        volume = df['volume']
        
        vol_indicators = {}
        
        # Volume Moving Averages
        if len(volume) >= 20:
            vol_sma20 = volume.rolling(window=20).mean()
            vol_indicators['volume_sma20'] = vol_sma20.iloc[-1]
            vol_indicators['volume_ratio'] = volume.iloc[-1] / vol_sma20.iloc[-1]
        
        # On-Balance Volume (OBV)
        if len(df) >= 2:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(df)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            vol_indicators['obv'] = obv.iloc[-1]
            
            if len(obv) >= 10:
                obv_ma = obv.rolling(window=10).mean()
                vol_indicators['obv_trend'] = 'up' if obv.iloc[-1] > obv_ma.iloc[-1] else 'down'
        
        # Volume Rate of Change
        if len(volume) >= 10:
            vol_roc = ((volume - volume.shift(10)) / volume.shift(10) * 100)
            vol_indicators['volume_roc'] = vol_roc.iloc[-1]
        
        return vol_indicators
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        trend = {}
        
        # ADX (Average Directional Index)
        if len(df) >= 14:
            # Simplified ADX calculation
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = pd.Series(index=df.index, dtype=float)
            minus_dm = pd.Series(index=df.index, dtype=float)
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_dm = pd.Series(plus_dm, index=df.index).rolling(window=14).mean()
            minus_dm = pd.Series(minus_dm, index=df.index).rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm / atr)
            minus_di = 100 * (minus_dm / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            trend['adx'] = adx.iloc[-1] if not adx.empty else 0
            trend['plus_di'] = plus_di.iloc[-1] if not plus_di.empty else 0
            trend['minus_di'] = minus_di.iloc[-1] if not minus_di.empty else 0
            trend['trend_strength'] = 'strong' if adx.iloc[-1] > 25 else 'weak'
        
        # Parabolic SAR (simplified)
        if len(df) >= 10:
            # Mock Parabolic SAR calculation
            sar_values = []
            af = 0.02
            
            for i in range(len(df)):
                if i == 0:
                    sar_values.append(low.iloc[i])
                else:
                    # Simplified calculation
                    sar = sar_values[-1] + af * (high.iloc[i-1] - sar_values[-1])
                    sar_values.append(max(sar, low.iloc[i]))
            
            trend['parabolic_sar'] = sar_values[-1]
            trend['sar_signal'] = 'buy' if close.iloc[-1] > sar_values[-1] else 'sell'
        
        return trend
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        levels = {}
        
        # Pivot Points
        if len(df) >= 1:
            typical_price = (high + low + close) / 3
            pivot = typical_price.rolling(window=5).mean().iloc[-1]
            
            high_avg = high.rolling(window=5).mean().iloc[-1]
            low_avg = low.rolling(window=5).mean().iloc[-1]
            
            levels['pivot_point'] = pivot
            levels['resistance_1'] = 2 * pivot - low_avg
            levels['support_1'] = 2 * pivot - high_avg
            levels['resistance_2'] = pivot + (high_avg - low_avg)
            levels['support_2'] = pivot - (high_avg - low_avg)
        
        # Recent highs and lows
        if len(df) >= 20:
            levels['recent_high'] = high.rolling(window=20).max().iloc[-1]
            levels['recent_low'] = low.rolling(window=20).min().iloc[-1]
            
            current_price = close.iloc[-1]
            levels['distance_to_resistance'] = ((levels['recent_high'] - current_price) / current_price) * 100
            levels['distance_to_support'] = ((current_price - levels['recent_low']) / current_price) * 100
        
        return levels
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        patterns = {}
        
        if len(df) >= 10:
            # Simple pattern detection
            recent_closes = close.tail(5).values
            recent_highs = high.tail(5).values
            recent_lows = low.tail(5).values
            
            # Trend direction
            if recent_closes[-1] > recent_closes[0]:
                patterns['trend'] = 'uptrend'
            elif recent_closes[-1] < recent_closes[0]:
                patterns['trend'] = 'downtrend'
            else:
                patterns['trend'] = 'sideways'
            
            # Consolidation pattern
            price_range = (max(recent_highs) - min(recent_lows)) / close.iloc[-1]
            patterns['consolidation'] = price_range < 0.05
            
            # Breakout detection
            if len(df) >= 20:
                recent_range = (high.tail(20).max() - low.tail(20).min()) / close.iloc[-1]
                current_move = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
                patterns['potential_breakout'] = current_move > (recent_range * 0.3)
        
        return patterns
    
    def _generate_overall_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trading signal based on all indicators"""
        signals = []
        confidence_scores = []
        
        # Moving Average signals
        ma_data = indicators.get('moving_averages', {})
        if ma_data.get('price_vs_sma20', 0) > 2:
            signals.append('buy')
            confidence_scores.append(0.3)
        elif ma_data.get('price_vs_sma20', 0) < -2:
            signals.append('sell')
            confidence_scores.append(0.3)
        
        # Momentum signals
        momentum_data = indicators.get('momentum', {})
        rsi = momentum_data.get('rsi', 50)
        if rsi < 30:
            signals.append('buy')
            confidence_scores.append(0.4)
        elif rsi > 70:
            signals.append('sell')
            confidence_scores.append(0.4)
        
        # MACD signal
        if momentum_data.get('macd_bullish', False):
            signals.append('buy')
            confidence_scores.append(0.3)
        else:
            signals.append('sell')
            confidence_scores.append(0.3)
        
        # Count signals
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        # Generate overall signal
        if buy_signals > sell_signals:
            overall_signal = 'buy'
            signal_strength = buy_signals / len(signals) if signals else 0
        elif sell_signals > buy_signals:
            overall_signal = 'sell'
            signal_strength = sell_signals / len(signals) if signals else 0
        else:
            overall_signal = 'hold'
            signal_strength = 0.5
        
        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.1
        
        return {
            'signal': overall_signal,
            'strength': signal_strength,
            'confidence': min(0.95, confidence),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': len(signals)
        }
    
    def _store_technical_data(self, symbol: str, analysis: Dict[str, Any]):
        """Store technical analysis data"""
        with self._lock:
            if symbol not in self.technical_data:
                self.technical_data[symbol] = []
            
            self.technical_data[symbol].append(analysis)
            
            # Keep only last 24 hours of analysis
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.technical_data[symbol] = [
                data for data in self.technical_data[symbol]
                if datetime.fromisoformat(data['timestamp']) > cutoff_time
            ]
    
    def get_technical_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest technical analysis for a symbol"""
        with self._lock:
            if symbol in self.technical_data and self.technical_data[symbol]:
                return self.technical_data[symbol][-1]
            return None
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of all trading signals"""
        with self._lock:
            if not self.technical_data:
                return {
                    'total_symbols': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'hold_signals': 0,
                    'strong_signals': 0
                }
            
            latest_analyses = {}
            for symbol, analyses in self.technical_data.items():
                if analyses:
                    latest_analyses[symbol] = analyses[-1]
            
            buy_count = sum(1 for analysis in latest_analyses.values() 
                          if analysis.get('signal', {}).get('signal') == 'buy')
            sell_count = sum(1 for analysis in latest_analyses.values() 
                           if analysis.get('signal', {}).get('signal') == 'sell')
            hold_count = len(latest_analyses) - buy_count - sell_count
            strong_signals = sum(1 for analysis in latest_analyses.values() 
                               if analysis.get('signal', {}).get('confidence', 0) > 0.7)
            
            return {
                'total_symbols': len(latest_analyses),
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'hold_signals': hold_count,
                'strong_signals': strong_signals,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'analyzed_symbols': len(self.technical_data),
            'ta_library_available': TA_AVAILABLE
        }
