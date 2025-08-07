"""
CryptoSmartTrader V2 - Comprehensive Market Scanner
Complete market coverage with dynamic cryptocurrency discovery and multi-timeframe analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import sys
import ccxt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GPU accelerator
# GPU accelerator is now managed via dependency injection

class ComprehensiveMarketScanner:
    """Comprehensive market scanner for complete cryptocurrency coverage"""
    
    def __init__(self, config_manager=None, cache_manager=None, error_handler=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchanges
        self.exchanges = self._initialize_exchanges()
        
        # Comprehensive timeframes for analysis
        self.timeframes = {
            '1m': '1 minute',
            '5m': '5 minutes', 
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day',
            '1w': '1 week',
            '1M': '1 month'
        }
        
        # Market scanning configuration
        self.scanning_config = {
            'discovery_interval': 3600,  # Discover new coins every hour
            'analysis_threads': 10,      # Parallel analysis threads
            'batch_size': 50,           # Process coins in batches
            'min_volume_usd': 10000,    # Minimum daily volume filter
            'exclude_stablecoins': True,
            'include_new_listings': True,
            'max_age_hours': 24         # Maximum data age before refresh
        }
        
        # Dynamic coin registry
        self.discovered_coins: Set[str] = set()
        self.coin_metadata: Dict[str, Dict] = {}
        self.analysis_cache: Dict[str, Dict] = {}
        
        # Threading controls
        self.scanning_active = False
        self.scanner_thread = None
        
        # Performance tracking
        self.scan_statistics = {
            'total_coins_discovered': 0,
            'active_coins_analyzed': 0,
            'last_full_scan': None,
            'scan_duration_seconds': 0,
            'opportunities_found': 0
        }
        
        self.logger.info("Comprehensive Market Scanner initialized")
    
    def _initialize_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """Initialize all supported exchanges"""
        exchanges = {}
        
        try:
            # Kraken - primary exchange for comprehensive coverage
            exchanges['kraken'] = ccxt.kraken({
                'apiKey': '',
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
            # Binance - for additional coverage
            exchanges['binance'] = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
            # Coinbase Pro - for US market coverage
            exchanges['coinbasepro'] = ccxt.coinbasepro({
                'apiKey': '',
                'secret': '',
                'passphrase': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': False
            })
            
            self.logger.info(f"Initialized {len(exchanges)} exchanges")
            return exchanges
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            return exchanges
    
    def start_comprehensive_scanning(self):
        """Start comprehensive market scanning"""
        if self.scanning_active:
            self.logger.warning("Market scanning already active")
            return
        
        self.scanning_active = True
        self.scanner_thread = threading.Thread(target=self._scanning_loop, daemon=True)
        self.scanner_thread.start()
        
        self.logger.info("Comprehensive market scanning started")
    
    def stop_comprehensive_scanning(self):
        """Stop comprehensive market scanning"""
        self.scanning_active = False
        if self.scanner_thread:
            self.scanner_thread.join(timeout=10)
        
        self.logger.info("Comprehensive market scanning stopped")
    
    def _scanning_loop(self):
        """Main scanning loop for continuous market analysis"""
        while self.scanning_active:
            try:
                scan_start = time.time()
                
                # Discover all available cryptocurrencies
                self._discover_all_cryptocurrencies()
                
                # Analyze all discovered coins across timeframes
                self._analyze_all_timeframes()
                
                # Update scan statistics
                scan_duration = time.time() - scan_start
                self.scan_statistics.update({
                    'last_full_scan': datetime.now().isoformat(),
                    'scan_duration_seconds': scan_duration,
                    'total_coins_discovered': len(self.discovered_coins),
                    'active_coins_analyzed': len([c for c in self.coin_metadata.values() if c.get('active', False)])
                })
                
                self.logger.info(f"Full market scan completed in {scan_duration:.1f}s - {len(self.discovered_coins)} coins analyzed")
                
                # Wait before next scan
                time.sleep(self.scanning_config['discovery_interval'])
                
            except Exception as e:
                self.logger.error(f"Market scanning error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _discover_all_cryptocurrencies(self):
        """Discover all available cryptocurrencies across exchanges"""
        try:
            all_markets = {}
            
            # Get markets from all exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    markets = exchange.load_markets()
                    all_markets[exchange_name] = markets
                    
                    self.logger.info(f"Loaded {len(markets)} markets from {exchange_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load markets from {exchange_name}: {e}")
                    continue
            
            # Process and combine all discovered coins
            new_coins = set()
            
            for exchange_name, markets in all_markets.items():
                for symbol, market_info in markets.items():
                    try:
                        # Filter valid trading pairs
                        if self._is_valid_trading_pair(symbol, market_info):
                            new_coins.add(symbol)
                            
                            # Store metadata
                            self.coin_metadata[symbol] = {
                                'exchange': exchange_name,
                                'base': market_info.get('base', ''),
                                'quote': market_info.get('quote', ''),
                                'active': market_info.get('active', False),
                                'type': market_info.get('type', 'spot'),
                                'spot': market_info.get('spot', True),
                                'margin': market_info.get('margin', False),
                                'future': market_info.get('future', False),
                                'option': market_info.get('option', False),
                                'contract': market_info.get('contract', False),
                                'linear': market_info.get('linear', None),
                                'inverse': market_info.get('inverse', None),
                                'taker': market_info.get('taker', 0),
                                'maker': market_info.get('maker', 0),
                                'percentage': market_info.get('percentage', True),
                                'tierBased': market_info.get('tierBased', False),
                                'discovered_at': datetime.now().isoformat(),
                                'last_updated': datetime.now().isoformat()
                            }
                    
                    except Exception as e:
                        self.logger.debug(f"Error processing market {symbol}: {e}")
                        continue
            
            # Update discovered coins
            previously_discovered = len(self.discovered_coins)
            self.discovered_coins.update(new_coins)
            newly_discovered = len(self.discovered_coins) - previously_discovered
            
            if newly_discovered > 0:
                self.logger.info(f"Discovered {newly_discovered} new trading pairs")
            
            # Cache discovered coins
            if self.cache_manager:
                self.cache_manager.set(
                    'comprehensive_coin_discovery',
                    {
                        'coins': list(self.discovered_coins),
                        'metadata': self.coin_metadata,
                        'last_discovery': datetime.now().isoformat()
                    },
                    ttl_minutes=60
                )
                
        except Exception as e:
            self.logger.error(f"Cryptocurrency discovery failed: {e}")
    
    def _is_valid_trading_pair(self, symbol: str, market_info: Dict) -> bool:
        """Validate if trading pair should be analyzed"""
        try:
            # Check if market is active
            if not market_info.get('active', False):
                return False
            
            # Only spot trading pairs
            if not market_info.get('spot', True):
                return False
            
            # Exclude certain types
            if market_info.get('future', False) or market_info.get('option', False):
                return False
            
            base = market_info.get('base', '').upper()
            quote = market_info.get('quote', '').upper()
            
            # Must have USD, USDT, EUR, or BTC quote
            valid_quotes = {'USD', 'USDT', 'USDC', 'EUR', 'BTC', 'ETH'}
            if quote not in valid_quotes:
                return False
            
            # Exclude obvious stablecoins if configured
            if self.scanning_config['exclude_stablecoins']:
                stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'PAX', 'GUSD'}
                if base in stablecoins:
                    return False
            
            # Exclude test/demo coins
            test_patterns = ['TEST', 'DEMO', 'FAKE']
            if any(pattern in base for pattern in test_patterns):
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Validation error for {symbol}: {e}")
            return False
    
    def _analyze_all_timeframes(self):
        """Analyze all discovered coins across all timeframes"""
        try:
            active_coins = [
                symbol for symbol, metadata in self.coin_metadata.items()
                if metadata.get('active', False)
            ]
            
            if not active_coins:
                self.logger.warning("No active coins found for analysis")
                return
            
            # Process coins in batches with parallel execution
            batch_size = self.scanning_config['batch_size']
            total_batches = (len(active_coins) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, len(active_coins))
                batch_coins = active_coins[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_coins)} coins)")
                
                # Parallel analysis of batch
                self._analyze_coin_batch_parallel(batch_coins)
            
            self.logger.info(f"Completed analysis of {len(active_coins)} coins across {len(self.timeframes)} timeframes")
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis failed: {e}")
    
    def _analyze_coin_batch_parallel(self, coin_batch: List[str]):
        """Analyze a batch of coins in parallel across all timeframes"""
        try:
            with ThreadPoolExecutor(max_workers=self.scanning_config['analysis_threads']) as executor:
                # Submit analysis tasks for each coin-timeframe combination
                futures = {}
                
                for coin in coin_batch:
                    for timeframe in self.timeframes:
                        future = executor.submit(self._analyze_coin_timeframe, coin, timeframe)
                        futures[future] = (coin, timeframe)
                
                # Collect results
                for future in as_completed(futures):
                    coin, timeframe = futures[future]
                    try:
                        analysis_result = future.result(timeout=30)
                        
                        if analysis_result:
                            # Store analysis result
                            cache_key = f"analysis_{coin}_{timeframe}"
                            if self.cache_manager:
                                self.cache_manager.set(cache_key, analysis_result, ttl_minutes=60)
                            
                            # Check for trading opportunities
                            self._evaluate_trading_opportunity(coin, timeframe, analysis_result)
                            
                    except Exception as e:
                        self.logger.debug(f"Analysis failed for {coin} {timeframe}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
    
    def _analyze_coin_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze a single coin on a specific timeframe"""
        try:
            metadata = self.coin_metadata.get(symbol, {})
            exchange_name = metadata.get('exchange', 'kraken')
            
            if exchange_name not in self.exchanges:
                return None
            
            exchange = self.exchanges[exchange_name]
            
            # Get OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                limit=200  # Get enough data for technical analysis
            )
            
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate technical indicators
            analysis = self._calculate_comprehensive_indicators(df)
            
            # Add metadata
            analysis.update({
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange_name,
                'last_price': float(df['close'].iloc[-1]),
                'volume_24h': float(df['volume'].tail(24).sum()) if len(df) >= 24 else float(df['volume'].sum()),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(df)
            })
            
            return analysis
            
        except Exception as e:
            self.logger.debug(f"Analysis failed for {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_comprehensive_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators with GPU acceleration"""
        try:
            analysis = {}
            
            # Use GPU-accelerated technical indicators
            # Remove GPU accelerator reference - handled via dependency injection now
            gpu_indicators = {}
            
            # Price action analysis
            analysis['price_change_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            analysis['price_change_24h_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) >= 24 else 0
            
            # Volatility using GPU acceleration
            # Calculate volatility using numpy
            analysis['volatility'] = np.std(df['close'].pct_change().dropna().values) * 100
            
            # Volume analysis
            if 'volume_sma' in gpu_indicators and len(gpu_indicators['volume_sma']) > 0:
                volume_sma = gpu_indicators['volume_sma'][-1]
                analysis['volume_sma_10'] = volume_sma
                analysis['volume_ratio'] = df['volume'].iloc[-1] / max(volume_sma, 1e-10)  # Avoid division by zero
            else:
                analysis['volume_sma_10'] = df['volume'].rolling(10).mean().iloc[-1]
                analysis['volume_ratio'] = df['volume'].iloc[-1] / max(analysis['volume_sma_10'], 1e-10)
            
            # Moving averages from GPU calculations
            if gpu_indicators:
                analysis['sma_20'] = gpu_indicators['sma_20'][-1] if len(gpu_indicators.get('sma_20', [])) > 0 else df['close'].rolling(20).mean().iloc[-1]
                analysis['sma_50'] = gpu_indicators['sma_50'][-1] if len(gpu_indicators.get('sma_50', [])) > 0 and len(df) >= 50 else None
                analysis['ema_12'] = gpu_indicators['ema_12'][-1] if len(gpu_indicators.get('ema_12', [])) > 0 else df['close'].ewm(span=12).mean().iloc[-1]
                analysis['ema_26'] = gpu_indicators['ema_26'][-1] if len(gpu_indicators.get('ema_26', [])) > 0 else df['close'].ewm(span=26).mean().iloc[-1]
            else:
                # Fallback to pandas calculations
                analysis['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
                analysis['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
                analysis['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
                analysis['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
            
            # RSI from GPU calculations
            if 'rsi' in gpu_indicators and len(gpu_indicators['rsi']) > 0:
                analysis['rsi'] = float(gpu_indicators['rsi'][-1])
            else:
                # Fallback RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-10)  # Avoid division by zero
                analysis['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # MACD
            analysis['macd'] = analysis['ema_12'] - analysis['ema_26']
            analysis['macd_signal'] = pd.Series([analysis['macd']]).ewm(span=9).mean().iloc[0]
            analysis['macd_histogram'] = analysis['macd'] - analysis['macd_signal']
            
            # Bollinger Bands from GPU calculations
            if all(key in gpu_indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
                analysis['bb_upper'] = float(gpu_indicators['bb_upper'][-1])
                analysis['bb_lower'] = float(gpu_indicators['bb_lower'][-1])
                analysis['bb_middle'] = float(gpu_indicators['bb_middle'][-1])
            else:
                # Fallback Bollinger Bands
                bb_sma = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                analysis['bb_upper'] = float((bb_sma + (bb_std * 2)).iloc[-1])
                analysis['bb_lower'] = float((bb_sma - (bb_std * 2)).iloc[-1])
                analysis['bb_middle'] = float(bb_sma.iloc[-1])
            
            # Bollinger Band position
            bb_range = analysis['bb_upper'] - analysis['bb_lower']
            if bb_range > 0:
                analysis['bb_position'] = (df['close'].iloc[-1] - analysis['bb_lower']) / bb_range
            else:
                analysis['bb_position'] = 0.5
            
            # Support and Resistance
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            analysis['resistance'] = float(highs.iloc[-1])
            analysis['support'] = float(lows.iloc[-1])
            
            # Trend analysis using GPU-accelerated correlation
            recent_closes = df['close'].tail(10).values
            if len(recent_closes) > 1:
                time_series = list(range(len(recent_closes)))
                # Calculate correlation using numpy
                analysis['trend_strength'] = np.corrcoef(time_series, recent_closes)[0, 1] if len(time_series) == len(recent_closes) else 0.0
                
                # Handle NaN values
                if np.isnan(analysis['trend_strength']):
                    analysis['trend_strength'] = 0.0
                
                analysis['trend_direction'] = 'bullish' if analysis['trend_strength'] > 0.1 else 'bearish' if analysis['trend_strength'] < -0.1 else 'sideways'
            else:
                analysis['trend_strength'] = 0.0
                analysis['trend_direction'] = 'sideways'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"GPU-accelerated indicator calculation failed: {e}")
            return {}
    
    def _evaluate_trading_opportunity(self, symbol: str, timeframe: str, analysis: Dict[str, Any]):
        """Evaluate if analysis indicates a trading opportunity"""
        try:
            opportunity_score = 0
            signals = []
            
            # RSI signals
            rsi = analysis.get('rsi', 50)
            if rsi < 30:
                opportunity_score += 2
                signals.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                opportunity_score += 1
                signals.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD signals
            macd = analysis.get('macd', 0)
            macd_signal = analysis.get('macd_signal', 0)
            if macd > macd_signal:
                opportunity_score += 1
                signals.append("MACD bullish crossover")
            
            # Volume surge
            volume_ratio = analysis.get('volume_ratio', 1)
            if volume_ratio > 2:
                opportunity_score += 2
                signals.append(f"Volume surge ({volume_ratio:.1f}x)")
            
            # Bollinger Band signals
            bb_position = analysis.get('bb_position', 0.5)
            if bb_position < 0.1:
                opportunity_score += 1
                signals.append("Near lower Bollinger Band")
            elif bb_position > 0.9:
                opportunity_score += 1
                signals.append("Near upper Bollinger Band")
            
            # Price momentum
            price_change = analysis.get('price_change_pct', 0)
            if abs(price_change) > 5:
                opportunity_score += 1
                signals.append(f"Strong momentum ({price_change:+.1f}%)")
            
            # Trend strength
            trend_strength = analysis.get('trend_strength', 0)
            if abs(trend_strength) > 0.7:
                opportunity_score += 1
                signals.append(f"Strong {analysis.get('trend_direction', 'unknown')} trend")
            
            # Record significant opportunities
            if opportunity_score >= 3:
                opportunity = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'score': opportunity_score,
                    'signals': signals,
                    'analysis': analysis,
                    'detected_at': datetime.now().isoformat()
                }
                
                # Cache opportunity
                if self.cache_manager:
                    cache_key = f"opportunity_{symbol}_{timeframe}_{int(time.time())}"
                    self.cache_manager.set(cache_key, opportunity, ttl_minutes=240)
                
                self.scan_statistics['opportunities_found'] += 1
                self.logger.info(f"Trading opportunity detected: {symbol} {timeframe} (score: {opportunity_score})")
                
        except Exception as e:
            self.logger.error(f"Opportunity evaluation failed: {e}")
    
    def get_all_discovered_coins(self) -> Dict[str, Any]:
        """Get all discovered coins with metadata"""
        return {
            'total_coins': len(self.discovered_coins),
            'coins': list(self.discovered_coins),
            'metadata': self.coin_metadata,
            'scan_statistics': self.scan_statistics,
            'timeframes': self.timeframes
        }
    
    def get_trading_opportunities(self, min_score: int = 3) -> List[Dict[str, Any]]:
        """Get current trading opportunities above minimum score"""
        try:
            opportunities = []
            
            if not self.cache_manager:
                return opportunities
            
            # Search cache for opportunities
            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith('opportunity_'):
                    opportunity = self.cache_manager.get(cache_key)
                    if opportunity and opportunity.get('score', 0) >= min_score:
                        opportunities.append(opportunity)
            
            # Sort by score descending
            opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to get trading opportunities: {e}")
            return []
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a specific coin across all timeframes"""
        try:
            analysis = {
                'symbol': symbol,
                'metadata': self.coin_metadata.get(symbol, {}),
                'timeframe_analysis': {},
                'aggregated_signals': {},
                'opportunities': []
            }
            
            # Collect analysis from all timeframes
            for timeframe in self.timeframes:
                cache_key = f"analysis_{symbol}_{timeframe}"
                if self.cache_manager:
                    timeframe_data = self.cache_manager.get(cache_key)
                    if timeframe_data:
                        analysis['timeframe_analysis'][timeframe] = timeframe_data
            
            # Aggregate signals across timeframes
            if analysis['timeframe_analysis']:
                analysis['aggregated_signals'] = self._aggregate_signals(analysis['timeframe_analysis'])
            
            # Get related opportunities
            opportunities = self.get_trading_opportunities()
            analysis['opportunities'] = [
                opp for opp in opportunities 
                if opp.get('symbol') == symbol
            ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to get comprehensive analysis for {symbol}: {e}")
            return {}
    
    def _aggregate_signals(self, timeframe_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate signals across multiple timeframes"""
        try:
            aggregated = {
                'overall_trend': 'neutral',
                'strength_score': 0,
                'risk_level': 'medium',
                'timeframe_consensus': {}
            }
            
            trend_votes = {'bullish': 0, 'bearish': 0, 'sideways': 0}
            rsi_values = []
            volume_ratios = []
            
            for timeframe, data in timeframe_analysis.items():
                # Collect trend votes
                trend = data.get('trend_direction', 'sideways')
                trend_votes[trend] += 1
                
                # Collect indicator values
                if 'rsi' in data:
                    rsi_values.append(data['rsi'])
                if 'volume_ratio' in data:
                    volume_ratios.append(data['volume_ratio'])
            
            # Determine overall trend
            max_votes = max(trend_votes.values())
            for trend, votes in trend_votes.items():
                if votes == max_votes:
                    aggregated['overall_trend'] = trend
                    break
            
            # Calculate consensus strength
            total_timeframes = len(timeframe_analysis)
            if total_timeframes > 0:
                consensus_strength = max_votes / total_timeframes
                aggregated['strength_score'] = consensus_strength
                
                if consensus_strength > 0.7:
                    aggregated['risk_level'] = 'low'
                elif consensus_strength < 0.4:
                    aggregated['risk_level'] = 'high'
            
            # Average RSI across timeframes
            if rsi_values:
                aggregated['avg_rsi'] = sum(rsi_values) / len(rsi_values)
            
            # Average volume ratio
            if volume_ratios:
                aggregated['avg_volume_ratio'] = sum(volume_ratios) / len(volume_ratios)
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Signal aggregation failed: {e}")
            return {}
    
    def force_full_scan(self):
        """Force immediate full market scan"""
        try:
            self.logger.info("Starting forced full market scan")
            
            # Discover all cryptocurrencies
            self._discover_all_cryptocurrencies()
            
            # Analyze all timeframes
            self._analyze_all_timeframes()
            
            self.logger.info("Forced full market scan completed")
            
        except Exception as e:
            self.logger.error(f"Forced scan failed: {e}")
    
    def get_scan_status(self) -> Dict[str, Any]:
        """Get current scanning status and statistics"""
        return {
            'scanning_active': self.scanning_active,
            'statistics': self.scan_statistics,
            'configuration': self.scanning_config,
            'exchanges': list(self.exchanges.keys()),
            'timeframes': self.timeframes,
            'discovered_coins_count': len(self.discovered_coins),
            'active_coins_count': len([c for c in self.coin_metadata.values() if c.get('active', False)])
        }