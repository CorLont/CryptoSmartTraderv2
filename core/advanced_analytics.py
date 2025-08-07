"""
CryptoSmartTrader V2 - Advanced Analytics Engine
Geavanceerde functies voor perfecte analyse van snelle groeiers
"""

import logging
import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from pathlib import Path
import sys
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class OnChainSignal:
    """On-chain data signal"""
    coin: str
    signal_type: str  # whale_accumulation, smart_money_flow, holder_distribution, etc.
    strength: float  # 0-1
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class NewsEvent:
    """News/event data"""
    coin: str
    title: str
    content: str
    source: str
    sentiment: float  # -1 to 1
    impact_score: float  # 0-1
    timestamp: datetime
    tags: List[str]

@dataclass
class OrderBookAnomaly:
    """Order book anomaly detection"""
    coin: str
    anomaly_type: str  # spoofing, thin_liquidity, wall_detected, etc.
    severity: float  # 0-1
    timestamp: datetime
    details: Dict[str, Any]

class OnChainAnalyzer:
    """Advanced on-chain data analysis beyond basic whale detection"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
        # Smart money addresses (known VCs, institutions, etc.)
        self.smart_money_addresses = {
            'btc': ['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'],  # Genesis address
            'eth': ['0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045'],  # Vitalik
            # Add more known addresses
        }
        
    async def analyze_smart_money_flow(self, coin: str) -> List[OnChainSignal]:
        """Analyze smart money movements"""
        try:
            signals = []
            
            # Get recent transactions from known smart money addresses
            smart_addresses = self.smart_money_addresses.get(coin.lower(), [])
            
            for address in smart_addresses:
                # Simulate smart money analysis (replace with real API calls)
                flow_data = await self._get_address_flow(address, coin)
                
                if flow_data and flow_data.get('net_flow', 0) > 0:
                    signal = OnChainSignal(
                        coin=coin,
                        signal_type='smart_money_accumulation',
                        strength=min(flow_data['net_flow'] / 1000000, 1.0),  # Normalize
                        timestamp=datetime.now(),
                        metadata={
                            'address': address,
                            'net_flow': flow_data['net_flow'],
                            'transaction_count': flow_data.get('tx_count', 0)
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Smart money analysis failed for {coin}: {e}")
            return []
    
    async def analyze_holder_distribution(self, coin: str) -> List[OnChainSignal]:
        """Analyze token holder distribution changes"""
        try:
            signals = []
            
            # Get holder distribution data
            distribution_data = await self._get_holder_distribution(coin)
            
            if distribution_data:
                # Check for concentration changes
                top_10_concentration = distribution_data.get('top_10_percent', 0)
                
                if top_10_concentration < 50:  # Decentralized holding = bullish
                    signal = OnChainSignal(
                        coin=coin,
                        signal_type='decentralized_holding',
                        strength=(50 - top_10_concentration) / 50,
                        timestamp=datetime.now(),
                        metadata={
                            'top_10_concentration': top_10_concentration,
                            'total_holders': distribution_data.get('total_holders', 0)
                        }
                    )
                    signals.append(signal)
                
                # Check for new whale accumulation
                new_large_holders = distribution_data.get('new_large_holders', 0)
                if new_large_holders > 0:
                    signal = OnChainSignal(
                        coin=coin,
                        signal_type='new_whale_accumulation',
                        strength=min(new_large_holders / 10, 1.0),
                        timestamp=datetime.now(),
                        metadata={'new_large_holders': new_large_holders}
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Holder distribution analysis failed for {coin}: {e}")
            return []
    
    async def _get_address_flow(self, address: str, coin: str) -> Optional[Dict]:
        """Get address flow data (replace with real API)"""
        # Simulate API call with cache
        cache_key = f"address_flow_{address}_{coin}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Simulate smart money flow data
        flow_data = {
            'net_flow': np.random.uniform(-1000000, 2000000),  # Random flow
            'tx_count': np.random.randint(1, 50),
            'last_activity': datetime.now().isoformat()
        }
        
        self.cache_manager.set(cache_key, flow_data, ttl_minutes=15)
        return flow_data
    
    async def _get_holder_distribution(self, coin: str) -> Optional[Dict]:
        """Get holder distribution data (replace with real API)"""
        cache_key = f"holder_distribution_{coin}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Simulate holder distribution
        distribution_data = {
            'top_10_percent': np.random.uniform(20, 80),
            'total_holders': np.random.randint(1000, 100000),
            'new_large_holders': np.random.randint(0, 5)
        }
        
        self.cache_manager.set(cache_key, distribution_data, ttl_minutes=30)
        return distribution_data

class OrderBookAnalyzer:
    """Real-time order book analysis for anomaly detection"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
    async def analyze_order_book(self, coin: str, exchange: str = 'kraken') -> List[OrderBookAnomaly]:
        """Analyze order book for anomalies"""
        try:
            anomalies = []
            
            # Get order book data
            order_book = await self._get_order_book(coin, exchange)
            
            if not order_book:
                return anomalies
            
            # Detect spoofing
            spoofing_anomaly = self._detect_spoofing(order_book, coin)
            if spoofing_anomaly:
                anomalies.append(spoofing_anomaly)
            
            # Detect thin liquidity
            liquidity_anomaly = self._detect_thin_liquidity(order_book, coin)
            if liquidity_anomaly:
                anomalies.append(liquidity_anomaly)
            
            # Detect large walls
            wall_anomaly = self._detect_large_walls(order_book, coin)
            if wall_anomaly:
                anomalies.append(wall_anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Order book analysis failed for {coin}: {e}")
            return []
    
    def _detect_spoofing(self, order_book: Dict, coin: str) -> Optional[OrderBookAnomaly]:
        """Detect order book spoofing"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return None
            
            # Look for unusually large orders far from mid-price
            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            
            for bid in bids[:10]:
                price, size = float(bid[0]), float(bid[1])
                if size > 1000 and abs(price - mid_price) / mid_price > 0.05:  # Large order >5% away
                    return OrderBookAnomaly(
                        coin=coin,
                        anomaly_type='potential_spoofing',
                        severity=0.7,
                        timestamp=datetime.now(),
                        details={
                            'side': 'bid',
                            'price': price,
                            'size': size,
                            'distance_from_mid': abs(price - mid_price) / mid_price
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Spoofing detection failed: {e}")
            return None
    
    def _detect_thin_liquidity(self, order_book: Dict, coin: str) -> Optional[OrderBookAnomaly]:
        """Detect thin liquidity conditions"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 5 or len(asks) < 5:
                return OrderBookAnomaly(
                    coin=coin,
                    anomaly_type='thin_liquidity',
                    severity=0.8,
                    timestamp=datetime.now(),
                    details={
                        'bid_levels': len(bids),
                        'ask_levels': len(asks),
                        'total_bid_size': sum(float(bid[1]) for bid in bids[:5]),
                        'total_ask_size': sum(float(ask[1]) for ask in asks[:5])
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Liquidity detection failed: {e}")
            return None
    
    def _detect_large_walls(self, order_book: Dict, coin: str) -> Optional[OrderBookAnomaly]:
        """Detect large buy/sell walls"""
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            # Calculate average order size
            all_sizes = []
            for bid in bids[:20]:
                all_sizes.append(float(bid[1]))
            for ask in asks[:20]:
                all_sizes.append(float(ask[1]))
            
            if not all_sizes:
                return None
            
            avg_size = np.mean(all_sizes)
            
            # Look for orders 5x larger than average
            for bid in bids[:5]:
                size = float(bid[1])
                if size > avg_size * 5:
                    return OrderBookAnomaly(
                        coin=coin,
                        anomaly_type='large_buy_wall',
                        severity=0.6,
                        timestamp=datetime.now(),
                        details={
                            'price': float(bid[0]),
                            'size': size,
                            'size_ratio': size / avg_size
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Wall detection failed: {e}")
            return None
    
    async def _get_order_book(self, coin: str, exchange: str) -> Optional[Dict]:
        """Get order book data"""
        cache_key = f"orderbook_{exchange}_{coin}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Use CCXT to get real order book data
            data_manager = self.container.data_manager()
            exchanges = data_manager.exchanges
            
            if exchange in exchanges:
                exchange_obj = exchanges[exchange]
                symbol = f"{coin}/USD"  # Adjust as needed
                
                order_book = await exchange_obj.fetch_order_book(symbol, limit=50)
                self.cache_manager.set(cache_key, order_book, ttl_minutes=1)  # Short TTL
                return order_book
                
        except Exception as e:
            self.logger.error(f"Failed to fetch order book for {coin}: {e}")
        
        return None

class NewsEventTracker:
    """Advanced news and event tracking"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
        # News sources configuration
        self.news_sources = {
            'coindesk': 'https://www.coindesk.com/api/v1/news',
            'cointelegraph': 'https://cointelegraph.com/api/news',
            'cryptopanic': 'https://cryptopanic.com/api/v1/posts/'
        }
        
    async def track_coin_events(self, coin: str) -> List[NewsEvent]:
        """Track news events for specific coin"""
        try:
            events = []
            
            # Search across multiple sources
            for source, api_url in self.news_sources.items():
                source_events = await self._fetch_coin_news(coin, source, api_url)
                events.extend(source_events)
            
            # Sort by impact score
            events.sort(key=lambda x: x.impact_score, reverse=True)
            
            return events[:10]  # Top 10 most impactful
            
        except Exception as e:
            self.logger.error(f"Event tracking failed for {coin}: {e}")
            return []
    
    async def _fetch_coin_news(self, coin: str, source: str, api_url: str) -> List[NewsEvent]:
        """Fetch news from specific source"""
        cache_key = f"news_{source}_{coin}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Simulate news fetching (replace with real API calls)
            news_items = await self._simulate_news_fetch(coin, source)
            
            events = []
            for item in news_items:
                event = NewsEvent(
                    coin=coin,
                    title=item['title'],
                    content=item.get('content', ''),
                    source=source,
                    sentiment=item.get('sentiment', 0),
                    impact_score=item.get('impact_score', 0.5),
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    tags=item.get('tags', [])
                )
                events.append(event)
            
            self.cache_manager.set(cache_key, events, ttl_minutes=15)
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to fetch news from {source}: {e}")
            return []
    
    async def _simulate_news_fetch(self, coin: str, source: str) -> List[Dict]:
        """Simulate news API response"""
        # Generate realistic news events
        news_templates = [
            f"{coin} announces major partnership with tech giant",
            f"New {coin} upgrade promises 10x performance improvement",
            f"{coin} listed on major exchange",
            f"Institutional investor adds {coin} to portfolio",
            f"{coin} integrates with popular DeFi protocol"
        ]
        
        news_items = []
        for i, template in enumerate(news_templates[:3]):  # Max 3 items
            news_items.append({
                'title': template,
                'content': f"Detailed content about {template}...",
                'sentiment': np.random.uniform(-0.5, 1.0),  # Mostly positive news
                'impact_score': np.random.uniform(0.3, 0.9),
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'tags': ['partnership', 'technology', 'listing']
            })
        
        return news_items

class AnomalyDetector:
    """Advanced anomaly and regime change detection"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()
        
    def detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            # Get BTC dominance and correlation data
            btc_dominance = self._get_btc_dominance()
            sp500_correlation = self._get_sp500_correlation()
            volatility_index = self._get_volatility_index()
            
            # Determine regime
            if btc_dominance > 50 and sp500_correlation < 0.3:
                regime = "crypto_native"
                risk_level = "medium"
            elif sp500_correlation > 0.7:
                regime = "risk_off"
                risk_level = "high"
            elif volatility_index > 0.8:
                regime = "high_volatility"
                risk_level = "high"
            else:
                regime = "risk_on"
                risk_level = "low"
            
            return {
                'regime': regime,
                'risk_level': risk_level,
                'btc_dominance': btc_dominance,
                'sp500_correlation': sp500_correlation,
                'volatility_index': volatility_index,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed: {e}")
            return {
                'regime': 'unknown',
                'risk_level': 'medium',
                'timestamp': datetime.now()
            }
    
    def detect_price_anomalies(self, coin: str, price_data: pd.DataFrame) -> List[Dict]:
        """Detect price anomalies using statistical methods"""
        try:
            anomalies = []
            
            if len(price_data) < 20:
                return anomalies
            
            # Calculate rolling statistics
            price_data['returns'] = price_data['close'].pct_change()
            price_data['volume_ma'] = price_data['volume'].rolling(window=20).mean()
            price_data['price_ma'] = price_data['close'].rolling(window=20).mean()
            price_data['price_std'] = price_data['close'].rolling(window=20).std()
            
            # Detect volume spikes
            latest_volume = price_data['volume'].iloc[-1]
            avg_volume = price_data['volume_ma'].iloc[-1]
            
            if latest_volume > avg_volume * 3:  # 3x volume spike
                anomalies.append({
                    'type': 'volume_spike',
                    'severity': min((latest_volume / avg_volume) / 10, 1.0),
                    'timestamp': datetime.now(),
                    'details': {
                        'current_volume': latest_volume,
                        'average_volume': avg_volume,
                        'ratio': latest_volume / avg_volume
                    }
                })
            
            # Detect price breakouts
            latest_price = price_data['close'].iloc[-1]
            upper_bound = price_data['price_ma'].iloc[-1] + 2 * price_data['price_std'].iloc[-1]
            lower_bound = price_data['price_ma'].iloc[-1] - 2 * price_data['price_std'].iloc[-1]
            
            if latest_price > upper_bound:
                anomalies.append({
                    'type': 'upward_breakout',
                    'severity': min((latest_price - upper_bound) / upper_bound, 1.0),
                    'timestamp': datetime.now(),
                    'details': {
                        'current_price': latest_price,
                        'upper_bound': upper_bound,
                        'distance': (latest_price - upper_bound) / upper_bound
                    }
                })
            elif latest_price < lower_bound:
                anomalies.append({
                    'type': 'downward_breakout',
                    'severity': min((lower_bound - latest_price) / lower_bound, 1.0),
                    'timestamp': datetime.now(),
                    'details': {
                        'current_price': latest_price,
                        'lower_bound': lower_bound,
                        'distance': (lower_bound - latest_price) / lower_bound
                    }
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Price anomaly detection failed for {coin}: {e}")
            return []
    
    def _get_btc_dominance(self) -> float:
        """Get BTC market dominance"""
        # Simulate BTC dominance (replace with real API)
        return np.random.uniform(40, 60)
    
    def _get_sp500_correlation(self) -> float:
        """Get S&P 500 correlation"""
        # Simulate correlation (replace with real calculation)
        return np.random.uniform(0.2, 0.8)
    
    def _get_volatility_index(self) -> float:
        """Get market volatility index"""
        # Simulate volatility index (replace with real calculation)
        return np.random.uniform(0.3, 0.9)

class AdvancedAnalyticsEngine:
    """Main coordinator for all advanced analytics"""
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.onchain_analyzer = OnChainAnalyzer(container)
        self.orderbook_analyzer = OrderBookAnalyzer(container)
        self.news_tracker = NewsEventTracker(container)
        self.anomaly_detector = AnomalyDetector(container)
        
        # Analysis state
        self.is_running = False
        self.analysis_thread = None
    
    def start_advanced_analysis(self):
        """Start continuous advanced analysis"""
        if self.is_running:
            self.logger.warning("Advanced analysis already running")
            return
        
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        self.logger.info("Advanced analytics engine started")
    
    def stop_advanced_analysis(self):
        """Stop advanced analysis"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        self.logger.info("Advanced analytics engine stopped")
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            try:
                # Run analysis cycle
                asyncio.run(self._run_analysis_cycle())
                
                # Wait before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Analysis cycle failed: {e}")
                time.sleep(60)  # Wait longer on error
    
    async def _run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        try:
            self.logger.info("Starting advanced analysis cycle")
            
            # Get coins to analyze
            coins = await self._get_analysis_coins()
            
            # Analyze market regime
            market_regime = self.anomaly_detector.detect_market_regime()
            self.logger.info(f"Market regime: {market_regime['regime']}")
            
            # Analyze each coin
            for coin in coins[:20]:  # Limit to top 20 coins
                try:
                    await self._analyze_coin_comprehensive(coin, market_regime)
                except Exception as e:
                    self.logger.error(f"Coin analysis failed for {coin}: {e}")
            
            self.logger.info("Advanced analysis cycle completed")
            
        except Exception as e:
            self.logger.error(f"Analysis cycle error: {e}")
    
    async def _analyze_coin_comprehensive(self, coin: str, market_regime: Dict):
        """Comprehensive analysis for single coin"""
        try:
            # Parallel analysis
            tasks = [
                self.onchain_analyzer.analyze_smart_money_flow(coin),
                self.onchain_analyzer.analyze_holder_distribution(coin),
                self.orderbook_analyzer.analyze_order_book(coin),
                self.news_tracker.track_coin_events(coin)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            onchain_signals = results[0] if not isinstance(results[0], Exception) else []
            holder_signals = results[1] if not isinstance(results[1], Exception) else []
            orderbook_anomalies = results[2] if not isinstance(results[2], Exception) else []
            news_events = results[3] if not isinstance(results[3], Exception) else []
            
            # Ensure all results are lists
            if not isinstance(onchain_signals, list):
                onchain_signals = []
            if not isinstance(holder_signals, list):
                holder_signals = []
            if not isinstance(orderbook_anomalies, list):
                orderbook_anomalies = []
            if not isinstance(news_events, list):
                news_events = []
            
            # Combine and score
            combined_signals = onchain_signals + holder_signals
            total_score = self._calculate_advanced_score(
                combined_signals,
                orderbook_anomalies,
                news_events,
                market_regime
            )
            
            # Store results
            self._store_advanced_analysis(coin, {
                'onchain_signals': combined_signals,
                'orderbook_anomalies': orderbook_anomalies,
                'news_events': news_events,
                'advanced_score': total_score,
                'market_regime': market_regime,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed for {coin}: {e}")
    
    def _calculate_advanced_score(self, onchain_signals: List, orderbook_anomalies: List, 
                                 news_events: List, market_regime: Dict) -> float:
        """Calculate comprehensive advanced score"""
        try:
            score = 0.0
            
            # On-chain signals weight
            for signal in onchain_signals:
                if signal.signal_type in ['smart_money_accumulation', 'decentralized_holding']:
                    score += signal.strength * 0.3
                elif signal.signal_type == 'new_whale_accumulation':
                    score += signal.strength * 0.2
            
            # Order book signals
            for anomaly in orderbook_anomalies:
                if anomaly.anomaly_type == 'large_buy_wall':
                    score += anomaly.severity * 0.2
                elif anomaly.anomaly_type == 'thin_liquidity':
                    score -= anomaly.severity * 0.1  # Negative signal
            
            # News events
            for event in news_events:
                score += event.sentiment * event.impact_score * 0.3
            
            # Market regime adjustment
            if market_regime['risk_level'] == 'high':
                score *= 0.7  # Reduce score in high risk
            elif market_regime['risk_level'] == 'low':
                score *= 1.2  # Boost score in low risk
            
            return max(0, min(1, score))  # Clamp to 0-1
            
        except Exception as e:
            self.logger.error(f"Score calculation failed: {e}")
            return 0.0
    
    def _store_advanced_analysis(self, coin: str, analysis_data: Dict):
        """Store advanced analysis results"""
        try:
            cache_manager = self.container.cache_manager()
            cache_key = f"advanced_analysis_{coin}"
            
            cache_manager.set(cache_key, analysis_data, ttl_minutes=60)
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis for {coin}: {e}")
    
    async def _get_analysis_coins(self) -> List[str]:
        """Get list of coins to analyze"""
        try:
            # Get coins from cache or data manager
            cache_manager = self.container.cache_manager()
            cached_coins = cache_manager.get("discovered_coins")
            
            if cached_coins:
                return list(cached_coins.keys())[:50]  # Top 50
            
            # Fallback to default coins
            return ['BTC', 'ETH', 'ADA', 'DOT', 'SOL', 'MATIC', 'AVAX', 'LUNA', 'ATOM', 'ALGO']
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis coins: {e}")
            return ['BTC', 'ETH']
    
    def get_advanced_analysis(self, coin: str) -> Optional[Dict]:
        """Get stored advanced analysis for coin"""
        try:
            cache_manager = self.container.cache_manager()
            cache_key = f"advanced_analysis_{coin}"
            
            return cache_manager.get(cache_key)
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis for {coin}: {e}")
            return None