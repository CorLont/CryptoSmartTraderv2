"""
Whale Detection Agent for Large Transaction Monitoring

Monitors large transactions, whale wallets, and unusual market movements
to identify potential market-moving events and early signals.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import hashlib
import json

try:
    import aiohttp
    import ccxt
    HAS_EXCHANGE_LIBS = True
except ImportError:
    HAS_EXCHANGE_LIBS = False
    logging.warning("Exchange libraries not available")

logger = logging.getLogger(__name__)

class WhaleActivityType(Enum):
    """Types of whale activities"""
    LARGE_BUY = "large_buy"
    LARGE_SELL = "large_sell"
    WALLET_ACCUMULATION = "wallet_accumulation"
    WALLET_DISTRIBUTION = "wallet_distribution"
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    WHALE_CLUSTER_MOVEMENT = "whale_cluster_movement"

class WhaleSize(Enum):
    """Whale size categories"""
    SMALL_WHALE = "small_whale"      # $100k - $1M
    MEDIUM_WHALE = "medium_whale"    # $1M - $10M
    LARGE_WHALE = "large_whale"      # $10M - $100M
    MEGA_WHALE = "mega_whale"        # $100M+

@dataclass
class WhaleTransaction:
    """Individual whale transaction"""
    timestamp: datetime
    symbol: str
    transaction_hash: str
    activity_type: WhaleActivityType
    whale_size: WhaleSize
    
    # Transaction details
    amount_usd: float
    amount_tokens: float
    price_at_time: float
    
    # Market context
    volume_ratio: float  # Transaction volume vs 24h volume
    order_book_impact: float  # Estimated order book impact
    price_impact: float  # Actual price impact observed
    
    # Whale identification
    wallet_address: str
    whale_confidence: float  # 0-1 confidence this is a whale
    previous_activity: int  # Number of previous large transactions
    wallet_balance_usd: float
    
    # Market timing
    market_hours: bool  # During main trading hours
    weekend_activity: bool
    news_correlation: float  # 0-1 correlation with recent news
    
    # Technical context
    support_resistance_level: Optional[float]
    volume_profile_significance: float
    momentum_alignment: bool  # Aligned with current momentum

@dataclass
class WhaleAlert:
    """Whale activity alert"""
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: int  # 1-5 scale
    
    # Alert details
    message: str
    confidence: float
    estimated_impact: float
    time_sensitivity: str  # "immediate", "short_term", "medium_term"
    
    # Supporting data
    transactions: List[WhaleTransaction]
    market_context: Dict[str, Any]
    follow_up_recommendations: List[str]

@dataclass
class WhaleMetrics:
    """Aggregated whale metrics for a symbol"""
    symbol: str
    timestamp: datetime
    
    # Activity metrics
    large_transactions_24h: int
    whale_buy_volume_24h: float
    whale_sell_volume_24h: float
    net_whale_flow: float  # Buys - Sells
    
    # Whale sentiment
    whale_sentiment_score: float  # -1 to 1
    accumulation_ratio: float  # Accumulation vs Distribution
    exchange_flow_ratio: float  # Inflow vs Outflow
    
    # Historical context
    whale_activity_trend: str  # "increasing", "decreasing", "stable"
    unusual_activity_detected: bool
    days_since_last_mega_whale: int
    
    # Market impact
    estimated_price_pressure: float
    liquidity_impact_score: float
    correlation_with_price_moves: float

class WhaleDetectorAgent:
    """
    Advanced Whale Detection Agent for Large Transaction Monitoring
    """
    
    def __init__(self, config_manager=None, data_manager=None):
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.active = False
        self.last_update = None
        self.processed_count = 0
        self.error_count = 0
        
        # Data storage
        self.whale_transactions: deque = deque(maxlen=10000)
        self.whale_alerts: deque = deque(maxlen=1000)
        self.whale_metrics: Dict[str, WhaleMetrics] = {}
        
        # Known whale wallets (would be populated from blockchain analysis)
        self.known_whales: Dict[str, Dict[str, Any]] = {}
        self.whale_watching_list: Set[str] = set()
        
        # Configuration
        self.update_interval = 120  # 2 minutes
        self.whale_thresholds = {
            'small_whale_usd': 100000,    # $100k
            'medium_whale_usd': 1000000,   # $1M
            'large_whale_usd': 10000000,   # $10M
            'mega_whale_usd': 100000000,   # $100M
        }
        
        # Volume impact thresholds
        self.volume_impact_thresholds = {
            'significant': 0.05,  # 5% of 24h volume
            'major': 0.10,        # 10% of 24h volume
            'massive': 0.20,      # 20% of 24h volume
        }
        
        # Tracked symbols
        self.tracked_symbols = [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD", 
            "SOL/USD", "AVAX/USD", "DOT/USD", "MATIC/USD", "LINK/USD",
            "UNI/USD", "AAVE/USD", "COMP/USD", "MKR/USD", "SNX/USD"
        ]
        
        # Exchange connections
        self.exchanges = {}
        self._initialize_exchanges()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_transactions_monitored': 0,
            'whale_transactions_detected': 0,
            'alerts_generated': 0,
            'mega_whale_events': 0,
            'exchange_flows_detected': 0,
            'accumulation_events': 0,
            'distribution_events': 0
        }
        
        # Data directory
        self.data_path = Path("data/whale_detection")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Load known whale data
        self._load_whale_database()
        
        logger.info("Whale Detector Agent initialized")
    
    def start(self):
        """Start the whale detection agent"""
        if not self.active and HAS_EXCHANGE_LIBS:
            self.active = True
            self.agent_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Whale Detector Agent started")
        else:
            self.logger.warning("Whale Detector Agent not started - missing dependencies")
    
    def stop(self):
        """Stop the whale detection agent"""
        self.active = False
        self.logger.info("Whale Detector Agent stopped")
    
    def _initialize_exchanges(self):
        """Initialize exchange connections for whale detection"""
        if not HAS_EXCHANGE_LIBS:
            return
        
        try:
            # Initialize exchanges for whale monitoring
            # We'll use exchanges that provide good volume and transaction data
            self.exchanges['kraken'] = ccxt.kraken({
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Add more exchanges as needed
            # self.exchanges['binance'] = ccxt.binance({...})
            
            self.logger.info(f"Initialized {len(self.exchanges)} exchanges for whale detection")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")
    
    def _detection_loop(self):
        """Main whale detection loop"""
        while self.active:
            try:
                # Monitor large transactions
                self._monitor_large_transactions()
                
                # Analyze whale wallet movements
                self._analyze_whale_wallets()
                
                # Detect exchange flows
                self._detect_exchange_flows()
                
                # Update whale metrics
                self._update_whale_metrics()
                
                # Generate alerts
                self._generate_whale_alerts()
                
                # Save data
                self._save_whale_data()
                
                # Update statistics
                self.last_update = datetime.now()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Whale detection error: {e}")
                time.sleep(60)  # Sleep 1 minute on error
    
    def _monitor_large_transactions(self):
        """Monitor large transactions across exchanges"""
        
        for symbol in self.tracked_symbols:
            try:
                # Get recent trades for symbol
                recent_trades = self._get_recent_large_trades(symbol)
                
                # Analyze each large trade
                for trade in recent_trades:
                    whale_transaction = self._analyze_trade_for_whale_activity(symbol, trade)
                    
                    if whale_transaction:
                        with self._lock:
                            self.whale_transactions.append(whale_transaction)
                            self.stats['whale_transactions_detected'] += 1
                            
                            # Check for mega whale activity
                            if whale_transaction.whale_size == WhaleSize.MEGA_WHALE:
                                self.stats['mega_whale_events'] += 1
                                self.logger.warning(f"MEGA WHALE detected: {symbol} ${whale_transaction.amount_usd:,.0f}")
                
                self.stats['total_transactions_monitored'] += len(recent_trades)
                
            except Exception as e:
                self.logger.error(f"Error monitoring transactions for {symbol}: {e}")
    
    def _get_recent_large_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent large trades from exchanges"""
        
        all_trades = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get recent trades
                trades = exchange.fetch_trades(symbol, limit=100)
                
                # Filter for large trades (simplified simulation)
                large_trades = []
                
                for trade in trades:
                    trade_value_usd = trade['amount'] * trade['price']
                    
                    # Consider trades over $50k as potentially whale-worthy
                    if trade_value_usd > 50000:
                        trade_data = {
                            'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000),
                            'amount': trade['amount'],
                            'price': trade['price'],
                            'value_usd': trade_value_usd,
                            'side': trade['side'],
                            'exchange': exchange_name,
                            'id': trade['id']
                        }
                        large_trades.append(trade_data)
                
                all_trades.extend(large_trades)
                
            except Exception as e:
                self.logger.error(f"Error fetching trades from {exchange_name} for {symbol}: {e}")
        
        return all_trades
    
    def _analyze_trade_for_whale_activity(self, symbol: str, trade: Dict[str, Any]) -> Optional[WhaleTransaction]:
        """Analyze a trade to determine if it's whale activity"""
        
        trade_value = trade['value_usd']
        
        # Determine whale size
        whale_size = self._classify_whale_size(trade_value)
        if not whale_size:
            return None  # Not large enough to be considered whale activity
        
        # Get market context
        market_context = self._get_market_context(symbol, trade['timestamp'])
        
        # Calculate volume ratio
        volume_ratio = self._calculate_volume_ratio(symbol, trade_value)
        
        # Estimate price impact
        estimated_impact = self._estimate_price_impact(symbol, trade_value, trade['side'])
        
        # Generate pseudo wallet address (in real implementation, would come from blockchain)
        trade_identifier = f"{trade['id']}{trade['timestamp']}"
        wallet_address = f"whale_{hashlib.md5(trade_identifier.encode()).hexdigest()[:8]}"
        
        # Determine activity type
        activity_type = WhaleActivityType.LARGE_BUY if trade['side'] == 'buy' else WhaleActivityType.LARGE_SELL
        
        # Calculate whale confidence (simplified heuristic)
        whale_confidence = min(1.0, (trade_value / self.whale_thresholds['small_whale_usd']) * 0.2 + 0.5)
        
        # Create whale transaction
        whale_transaction = WhaleTransaction(
            timestamp=trade['timestamp'],
            symbol=symbol,
            transaction_hash=trade['id'],
            activity_type=activity_type,
            whale_size=whale_size,
            amount_usd=trade_value,
            amount_tokens=trade['amount'],
            price_at_time=trade['price'],
            volume_ratio=volume_ratio,
            order_book_impact=estimated_impact,
            price_impact=0.0,  # Would be calculated post-trade
            wallet_address=wallet_address,
            whale_confidence=whale_confidence,
            previous_activity=0,  # Would track historical activity
            wallet_balance_usd=trade_value * 10,  # Estimated based on transaction size
            market_hours=self._is_market_hours(trade['timestamp']),
            weekend_activity=self._is_weekend(trade['timestamp']),
            news_correlation=0.0,  # Would correlate with news events
            support_resistance_level=None,  # Would calculate from technical analysis
            volume_profile_significance=volume_ratio,
            momentum_alignment=True  # Simplified
        )
        
        return whale_transaction
    
    def _classify_whale_size(self, trade_value_usd: float) -> Optional[WhaleSize]:
        """Classify transaction as whale size category"""
        
        if trade_value_usd >= self.whale_thresholds['mega_whale_usd']:
            return WhaleSize.MEGA_WHALE
        elif trade_value_usd >= self.whale_thresholds['large_whale_usd']:
            return WhaleSize.LARGE_WHALE
        elif trade_value_usd >= self.whale_thresholds['medium_whale_usd']:
            return WhaleSize.MEDIUM_WHALE
        elif trade_value_usd >= self.whale_thresholds['small_whale_usd']:
            return WhaleSize.SMALL_WHALE
        else:
            return None
    
    def _get_market_context(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Get market context at time of transaction"""
        # Simplified market context
        return {
            'market_cap': 1000000000,  # Would fetch real market cap
            'daily_volume': 50000000,  # Would fetch real daily volume
            'price_change_24h': 0.02,  # Would calculate real price change
            'volatility': 0.05,        # Would calculate real volatility
        }
    
    def _calculate_volume_ratio(self, symbol: str, trade_value: float) -> float:
        """Calculate trade value as ratio of daily volume"""
        # Simplified - would use real volume data
        estimated_daily_volume = 50000000  # $50M daily volume
        return min(1.0, trade_value / estimated_daily_volume)
    
    def _estimate_price_impact(self, symbol: str, trade_value: float, side: str) -> float:
        """Estimate potential price impact of trade"""
        # Simplified price impact model
        base_impact = trade_value / 100000000  # $100M causes 100% impact (very simplified)
        
        # Adjust for market depth (simplified)
        market_depth_multiplier = 2.0  # Assume thin markets
        
        estimated_impact = base_impact * market_depth_multiplier
        return min(0.5, estimated_impact)  # Cap at 50% impact
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if transaction occurred during main trading hours"""
        # Crypto trades 24/7, but can still identify peak hours
        hour = timestamp.hour
        return 6 <= hour <= 22  # Consider 6 AM to 10 PM as main hours
    
    def _is_weekend(self, timestamp: datetime) -> bool:
        """Check if transaction occurred on weekend"""
        return timestamp.weekday() >= 5  # Saturday (5) or Sunday (6)
    
    def _analyze_whale_wallets(self):
        """Analyze known whale wallet movements"""
        
        # This would connect to blockchain APIs to monitor specific whale wallets
        # For now, simulate whale wallet analysis
        
        simulated_whale_activities = [
            {
                'wallet': 'whale_btc_001',
                'symbol': 'BTC/USD',
                'activity': 'accumulation',
                'amount_usd': 5000000,
                'confidence': 0.9
            },
            {
                'wallet': 'whale_eth_002', 
                'symbol': 'ETH/USD',
                'activity': 'distribution',
                'amount_usd': 2000000,
                'confidence': 0.8
            }
        ]
        
        for activity in simulated_whale_activities:
            if activity['activity'] == 'accumulation':
                self.stats['accumulation_events'] += 1
            else:
                self.stats['distribution_events'] += 1
    
    def _detect_exchange_flows(self):
        """Detect large flows into/out of exchanges"""
        
        # This would monitor blockchain data for large transfers to/from exchange wallets
        # Simulate exchange flow detection
        
        simulated_flows = [
            {
                'symbol': 'BTC/USD',
                'flow_type': 'inflow',
                'amount_usd': 10000000,
                'exchange': 'binance',
                'significance': 'major'
            }
        ]
        
        for flow in simulated_flows:
            self.stats['exchange_flows_detected'] += 1
            
            # Large exchange inflows can be bearish (preparation to sell)
            # Large exchange outflows can be bullish (moving to cold storage)
            self.logger.info(f"Exchange flow detected: {flow['symbol']} {flow['flow_type']} ${flow['amount_usd']:,.0f}")
    
    def _update_whale_metrics(self):
        """Update aggregated whale metrics for each symbol"""
        
        with self._lock:
            # Get recent whale transactions (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_transactions = [
                tx for tx in self.whale_transactions 
                if tx.timestamp > cutoff_time
            ]
            
            # Group by symbol
            symbol_transactions = defaultdict(list)
            for tx in recent_transactions:
                symbol_transactions[tx.symbol].append(tx)
            
            # Calculate metrics for each symbol
            for symbol, transactions in symbol_transactions.items():
                metrics = self._calculate_whale_metrics(symbol, transactions)
                self.whale_metrics[symbol] = metrics
    
    def _calculate_whale_metrics(self, symbol: str, transactions: List[WhaleTransaction]) -> WhaleMetrics:
        """Calculate comprehensive whale metrics for a symbol"""
        
        if not transactions:
            return self._create_neutral_whale_metrics(symbol)
        
        # Basic metrics
        large_transactions_24h = len(transactions)
        
        # Volume metrics
        buy_transactions = [tx for tx in transactions if tx.activity_type == WhaleActivityType.LARGE_BUY]
        sell_transactions = [tx for tx in transactions if tx.activity_type == WhaleActivityType.LARGE_SELL]
        
        whale_buy_volume_24h = sum(tx.amount_usd for tx in buy_transactions)
        whale_sell_volume_24h = sum(tx.amount_usd for tx in sell_transactions)
        net_whale_flow = whale_buy_volume_24h - whale_sell_volume_24h
        
        # Sentiment calculation
        if whale_buy_volume_24h + whale_sell_volume_24h > 0:
            whale_sentiment_score = net_whale_flow / (whale_buy_volume_24h + whale_sell_volume_24h)
        else:
            whale_sentiment_score = 0.0
        
        # Accumulation ratio
        total_volume = whale_buy_volume_24h + whale_sell_volume_24h
        accumulation_ratio = whale_buy_volume_24h / total_volume if total_volume > 0 else 0.5
        
        # Unusual activity detection
        avg_transaction_size = total_volume / large_transactions_24h if large_transactions_24h > 0 else 0
        unusual_activity_detected = (
            large_transactions_24h > 10 or  # Many transactions
            avg_transaction_size > self.whale_thresholds['large_whale_usd'] or  # Very large average
            abs(whale_sentiment_score) > 0.7  # Strong directional bias
        )
        
        # Price pressure estimation
        estimated_price_pressure = min(1.0, net_whale_flow / 50000000)  # $50M causes 100% pressure
        
        # Trend analysis (simplified)
        whale_activity_trend = "stable"
        if large_transactions_24h > 5:
            whale_activity_trend = "increasing"
        elif large_transactions_24h == 0:
            whale_activity_trend = "decreasing"
        
        return WhaleMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            large_transactions_24h=large_transactions_24h,
            whale_buy_volume_24h=whale_buy_volume_24h,
            whale_sell_volume_24h=whale_sell_volume_24h,
            net_whale_flow=net_whale_flow,
            whale_sentiment_score=whale_sentiment_score,
            accumulation_ratio=accumulation_ratio,
            exchange_flow_ratio=0.5,  # Would calculate from exchange flow data
            whale_activity_trend=whale_activity_trend,
            unusual_activity_detected=unusual_activity_detected,
            days_since_last_mega_whale=0,  # Would track from historical data
            estimated_price_pressure=estimated_price_pressure,
            liquidity_impact_score=min(1.0, total_volume / 100000000),  # $100M = 100% impact
            correlation_with_price_moves=0.6  # Would calculate correlation
        )
    
    def _create_neutral_whale_metrics(self, symbol: str) -> WhaleMetrics:
        """Create neutral whale metrics when no activity detected"""
        return WhaleMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            large_transactions_24h=0,
            whale_buy_volume_24h=0.0,
            whale_sell_volume_24h=0.0,
            net_whale_flow=0.0,
            whale_sentiment_score=0.0,
            accumulation_ratio=0.5,
            exchange_flow_ratio=0.5,
            whale_activity_trend="stable",
            unusual_activity_detected=False,
            days_since_last_mega_whale=999,
            estimated_price_pressure=0.0,
            liquidity_impact_score=0.0,
            correlation_with_price_moves=0.0
        )
    
    def _generate_whale_alerts(self):
        """Generate alerts for significant whale activities"""
        
        for symbol, metrics in self.whale_metrics.items():
            
            # High-value transaction alert
            if metrics.large_transactions_24h > 0:
                recent_large_tx = [
                    tx for tx in self.whale_transactions 
                    if tx.symbol == symbol and 
                    tx.timestamp > datetime.now() - timedelta(hours=1) and
                    tx.whale_size in [WhaleSize.LARGE_WHALE, WhaleSize.MEGA_WHALE]
                ]
                
                for tx in recent_large_tx:
                    alert = WhaleAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="large_transaction",
                        severity=5 if tx.whale_size == WhaleSize.MEGA_WHALE else 3,
                        message=f"{tx.whale_size.value.title()} {tx.activity_type.value} detected: ${tx.amount_usd:,.0f}",
                        confidence=tx.whale_confidence,
                        estimated_impact=tx.order_book_impact,
                        time_sensitivity="immediate",
                        transactions=[tx],
                        market_context={},
                        follow_up_recommendations=[
                            "Monitor price action closely",
                            "Check for follow-up transactions",
                            "Assess market depth"
                        ]
                    )
                    
                    with self._lock:
                        self.whale_alerts.append(alert)
                        self.stats['alerts_generated'] += 1
            
            # Unusual activity alert
            if metrics.unusual_activity_detected:
                alert = WhaleAlert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type="unusual_activity",
                    severity=4,
                    message=f"Unusual whale activity: {metrics.large_transactions_24h} large transactions, net flow ${metrics.net_whale_flow:,.0f}",
                    confidence=0.8,
                    estimated_impact=metrics.estimated_price_pressure,
                    time_sensitivity="short_term",
                    transactions=[],
                    market_context={'metrics': metrics},
                    follow_up_recommendations=[
                        "Investigate underlying causes",
                        "Monitor for trend continuation",
                        "Check related symbols"
                    ]
                )
                
                with self._lock:
                    self.whale_alerts.append(alert)
                    self.stats['alerts_generated'] += 1
    
    def get_whale_metrics(self, symbol: str) -> Optional[WhaleMetrics]:
        """Get whale metrics for a specific symbol"""
        with self._lock:
            return self.whale_metrics.get(symbol)
    
    def get_all_whale_metrics(self) -> Dict[str, WhaleMetrics]:
        """Get all whale metrics"""
        with self._lock:
            return self.whale_metrics.copy()
    
    def get_recent_whale_transactions(self, symbol: str = None, hours: int = 24) -> List[WhaleTransaction]:
        """Get recent whale transactions"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            transactions = [
                tx for tx in self.whale_transactions 
                if tx.timestamp > cutoff_time and (symbol is None or tx.symbol == symbol)
            ]
        
        return sorted(transactions, key=lambda x: x.timestamp, reverse=True)
    
    def get_whale_alerts(self, symbol: str = None, severity_min: int = 1) -> List[WhaleAlert]:
        """Get whale alerts"""
        with self._lock:
            alerts = [
                alert for alert in self.whale_alerts 
                if alert.severity >= severity_min and (symbol is None or alert.symbol == symbol)
            ]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_whale_signals(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Get actionable whale signals for trading"""
        signals = []
        
        for symbol, metrics in self.whale_metrics.items():
            
            # Strong accumulation signal
            if (metrics.accumulation_ratio > 0.7 and 
                metrics.large_transactions_24h >= 3 and
                metrics.net_whale_flow > 1000000):  # $1M+ net buying
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'whale_accumulation',
                    'strength': metrics.accumulation_ratio,
                    'confidence': min_confidence,
                    'reasoning': f"Strong whale accumulation: {metrics.large_transactions_24h} transactions, ${metrics.net_whale_flow:,.0f} net flow",
                    'estimated_impact': metrics.estimated_price_pressure,
                    'time_horizon': 'short_term'
                })
            
            # Strong distribution signal
            elif (metrics.accumulation_ratio < 0.3 and 
                  metrics.large_transactions_24h >= 3 and
                  metrics.net_whale_flow < -1000000):  # $1M+ net selling
                
                signals.append({
                    'symbol': symbol,
                    'signal_type': 'whale_distribution',
                    'strength': 1 - metrics.accumulation_ratio,
                    'confidence': min_confidence,
                    'reasoning': f"Strong whale distribution: {metrics.large_transactions_24h} transactions, ${metrics.net_whale_flow:,.0f} net flow",
                    'estimated_impact': abs(metrics.estimated_price_pressure),
                    'time_horizon': 'short_term'
                })
        
        # Sort by estimated impact
        signals.sort(key=lambda x: x['estimated_impact'], reverse=True)
        return signals
    
    def _load_whale_database(self):
        """Load known whale wallet database"""
        try:
            whale_db_file = self.data_path / "whale_database.json"
            if whale_db_file.exists():
                with open(whale_db_file, 'r') as f:
                    self.known_whales = json.load(f)
                    self.logger.info(f"Loaded {len(self.known_whales)} known whale wallets")
        except Exception as e:
            self.logger.error(f"Error loading whale database: {e}")
    
    def _save_whale_data(self):
        """Save whale data to disk"""
        try:
            # Save recent metrics
            metrics_file = self.data_path / "whale_metrics.json"
            metrics_data = {
                symbol: {
                    'timestamp': metrics.timestamp.isoformat(),
                    'large_transactions_24h': metrics.large_transactions_24h,
                    'net_whale_flow': metrics.net_whale_flow,
                    'whale_sentiment_score': metrics.whale_sentiment_score,
                    'unusual_activity_detected': metrics.unusual_activity_detected,
                    'estimated_price_pressure': metrics.estimated_price_pressure
                }
                for symbol, metrics in self.whale_metrics.items()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving whale data: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'tracked_symbols': len(self.tracked_symbols),
            'whale_metrics': len(self.whale_metrics),
            'whale_transactions': len(self.whale_transactions),
            'whale_alerts': len(self.whale_alerts),
            'known_whales': len(self.known_whales),
            'exchanges_connected': len(self.exchanges),
            'statistics': self.stats,
            'dependencies_available': HAS_EXCHANGE_LIBS
        }