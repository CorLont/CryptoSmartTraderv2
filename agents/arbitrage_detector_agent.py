"""
Cross-Exchange Arbitrage & Funding Rate Opportunities Agent

Monitors price differences, funding rates, and trading opportunities across
multiple cryptocurrency exchanges to identify profitable arbitrage situations.
"""

import asyncio
import time
import json
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
import math

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    logging.warning("CCXT library not available")

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    SPOT_ARBITRAGE = "spot_arbitrage"           # Simple price differences
    TRIANGULAR_ARBITRAGE = "triangular"         # Three-way arbitrage
    FUNDING_RATE_ARBITRAGE = "funding_rate"     # Perpetual funding arbitrage
    BASIS_ARBITRAGE = "basis"                   # Spot vs futures basis
    CROSS_EXCHANGE_SPREAD = "cross_spread"      # Exchange spread differences
    WITHDRAWAL_ARBITRAGE = "withdrawal"         # Considering withdrawal costs

class OpportunityRisk(Enum):
    """Risk levels for arbitrage opportunities"""
    LOW = "low"           # <1% risk, high liquidity
    MODERATE = "moderate" # 1-3% risk, good liquidity  
    HIGH = "high"         # 3-5% risk, limited liquidity
    EXTREME = "extreme"   # >5% risk, low liquidity

@dataclass
class ArbitrageOpportunity:
    """Individual arbitrage opportunity"""
    timestamp: datetime
    opportunity_id: str
    arbitrage_type: ArbitrageType
    
    # Trading pair and exchanges
    symbol: str
    buy_exchange: str
    sell_exchange: str
    
    # Price information
    buy_price: float
    sell_price: float
    price_difference: float
    spread_percentage: float
    
    # Profit calculation
    gross_profit_percentage: float
    estimated_fees: float
    net_profit_percentage: float
    minimum_trade_amount: float
    maximum_trade_amount: float
    
    # Risk assessment
    risk_level: OpportunityRisk
    liquidity_score: float  # 0-1 score
    execution_difficulty: float  # 0-1 score (1 = very difficult)
    slippage_risk: float
    
    # Market context
    volume_24h_buy: float
    volume_24h_sell: float
    bid_ask_spread_buy: float
    bid_ask_spread_sell: float
    
    # Timing information
    opportunity_window: float  # Expected window in minutes
    execution_urgency: int  # 1-5 scale
    decay_rate: float  # How fast opportunity disappears
    
    # Additional costs and considerations
    withdrawal_fees: Dict[str, float]
    deposit_time_minutes: Dict[str, int]
    withdrawal_time_minutes: Dict[str, int]
    minimum_withdrawal: Dict[str, float]
    
    # Performance prediction
    expected_return: float
    confidence: float
    success_probability: float

@dataclass
class FundingRateOpportunity:
    """Funding rate arbitrage opportunity"""
    timestamp: datetime
    symbol: str
    exchange: str
    
    # Funding rate information
    current_funding_rate: float  # 8-hour rate
    predicted_funding_rate: float
    annual_funding_rate: float  # Annualized
    
    # Market position
    recommended_position: str  # "long", "short"
    position_size_recommendation: float
    leverage_recommendation: float
    
    # Risk metrics
    basis_risk: float  # Risk of spot-futures basis change
    liquidation_risk: float
    funding_volatility: float
    
    # Profit estimation
    expected_daily_return: float
    expected_weekly_return: float
    risk_adjusted_return: float

class ArbitrageDetectorAgent:
    """
    Advanced Cross-Exchange Arbitrage Detection Agent
    Identifies and evaluates arbitrage opportunities across multiple exchanges
    """
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.active = False
        self.last_update = None
        self.opportunities_found = 0
        self.error_count = 0
        
        # Initialize stats before calling _initialize_exchanges
        self.stats = {
            'exchanges_monitored': 0,
            'arbitrage_opportunities_found': 0,
            'funding_rate_opportunities': 0,
            'triangular_arbitrage_found': 0,
            'average_spread': 0.0,
            'best_spread_today': 0.0,
            'execution_success_rate': 0.0
        }
        
        # Data storage
        self.arbitrage_opportunities: deque = deque(maxlen=1000)
        self.funding_opportunities: deque = deque(maxlen=500)
        self.price_cache: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.exchange_info_cache: Dict[str, Dict] = {}
        
        # Configuration
        self.update_interval = 30  # 30 seconds for arbitrage detection
        self.min_profit_threshold = 0.5  # 0.5% minimum profit
        self.max_execution_time = 300  # 5 minutes max execution window
        self.min_liquidity_threshold = 10000  # $10k minimum liquidity
        
        # Exchange clients and configuration
        self.exchange_clients = {}
        self.exchange_fees = {}
        self.exchange_limits = {}
        self._initialize_exchanges()
        
        # Symbols to monitor
        self.monitored_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'MATIC/USDT', 'AVAX/USDT', 'DOT/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'FTM/USDT', 'ALGO/USDT'
        ]
        
        # Additional statistics tracking
        self.extended_stats = {
            'total_opportunities_found': 0,
            'profitable_opportunities': 0,
            'executed_trades': 0,
            'successful_arbitrages': 0,
            'average_profit': 0.0,
            'total_profit': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Data directory
        self.data_path = Path("data/arbitrage")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Arbitrage Detector Agent initialized")
    
    def start(self):
        """Start the arbitrage detection agent"""
        if not self.active and HAS_CCXT:
            self.active = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            self.logger.info("Arbitrage Detection Agent started")
        else:
            self.logger.warning("Arbitrage Detection Agent not started - CCXT library required")
    
    def stop(self):
        """Stop the arbitrage detection agent"""
        self.active = False
        self.logger.info("Arbitrage Detection Agent stopped")
    
    def _initialize_exchanges(self):
        """Initialize exchange clients and cache their information"""
        if not HAS_CCXT:
            return
        
        # Define exchanges to monitor
        exchange_configs = {
            'binance': {
                'class': ccxt.binance,
                'config': {'sandbox': False, 'enableRateLimit': True}
            },
            'kraken': {
                'class': ccxt.kraken,
                'config': {'sandbox': False, 'enableRateLimit': True}
            },
            'kucoin': {
                'class': ccxt.kucoin,
                'config': {'sandbox': False, 'enableRateLimit': True}
            },
            'huobi': {
                'class': ccxt.huobi,
                'config': {'sandbox': False, 'enableRateLimit': True}
            }
        }
        
        # Initialize clients
        for exchange_name, config in exchange_configs.items():
            try:
                client = config['class'](config['config'])
                client.load_markets()
                
                self.exchange_clients[exchange_name] = client
                self.exchange_fees[exchange_name] = self._get_exchange_fees(client)
                self.exchange_limits[exchange_name] = self._get_exchange_limits(client)
                
                self.logger.info(f"Initialized {exchange_name} for arbitrage monitoring")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")
        
        self.stats['exchanges_monitored'] = len(self.exchange_clients)
        
    def _get_exchange_fees(self, client) -> Dict[str, float]:
        """Get trading fees for an exchange"""
        try:
            # Get fee structure from exchange
            fees = client.describe().get('fees', {})
            trading_fees = fees.get('trading', {})
            
            return {
                'maker': trading_fees.get('maker', 0.001),  # 0.1% default
                'taker': trading_fees.get('taker', 0.001),
                'withdrawal': 0.0  # Would need to fetch per-symbol
            }
        except:
            return {'maker': 0.001, 'taker': 0.001, 'withdrawal': 0.0}
    
    def _get_exchange_limits(self, client) -> Dict[str, Any]:
        """Get trading limits for an exchange"""
        try:
            limits = client.describe().get('limits', {})
            return {
                'amount': limits.get('amount', {'min': 0.01, 'max': 1000000}),
                'cost': limits.get('cost', {'min': 10, 'max': 10000000})
            }
        except:
            return {
                'amount': {'min': 0.01, 'max': 1000000},
                'cost': {'min': 10, 'max': 10000000}
            }
    
    def _detection_loop(self):
        """Main arbitrage detection loop"""
        while self.active:
            try:
                # Update price data from all exchanges
                self._update_price_data()
                
                # Detect spot arbitrage opportunities
                spot_opportunities = self._detect_spot_arbitrage()
                
                # Detect funding rate opportunities
                funding_opportunities = self._detect_funding_rate_opportunities()
                
                # Detect triangular arbitrage
                triangular_opportunities = self._detect_triangular_arbitrage()
                
                # Evaluate and store opportunities
                self._evaluate_opportunities(
                    spot_opportunities + triangular_opportunities
                )
                
                self._evaluate_funding_opportunities(funding_opportunities)
                
                # Clean expired opportunities
                self._cleanup_expired_opportunities()
                
                # Update statistics
                self._update_statistics()
                
                # Save data
                self._save_arbitrage_data()
                
                self.last_update = datetime.now()
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Arbitrage detection error: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _update_price_data(self):
        """Update price data from all exchanges"""
        for exchange_name, client in self.exchange_clients.items():
            try:
                # Fetch tickers for monitored symbols
                for symbol in self.monitored_symbols:
                    if symbol in client.markets:
                        ticker = client.fetch_ticker(symbol)
                        order_book = client.fetch_order_book(symbol, limit=5)
                        
                        price_data = {
                            'timestamp': datetime.now(),
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'last': ticker['last'],
                            'volume': ticker['quoteVolume'],
                            'bid_volume': order_book['bids'][0][1] if order_book['bids'] else 0,
                            'ask_volume': order_book['asks'][0][1] if order_book['asks'] else 0,
                            'spread': (ticker['ask'] - ticker['bid']) / ticker['bid'] if ticker['bid'] else 0
                        }
                        
                        self.price_cache[symbol][exchange_name] = price_data
                        
            except Exception as e:
                self.logger.error(f"Error updating prices for {exchange_name}: {e}")
    
    def _detect_spot_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Detect spot arbitrage opportunities between exchanges"""
        opportunities = []
        
        for symbol in self.monitored_symbols:
            exchanges_with_data = list(self.price_cache.get(symbol, {}).keys())
            
            # Check all exchange pairs
            for i, buy_exchange in enumerate(exchanges_with_data):
                for sell_exchange in exchanges_with_data[i+1:]:
                    
                    buy_data = self.price_cache[symbol][buy_exchange]
                    sell_data = self.price_cache[symbol][sell_exchange]
                    
                    # Calculate arbitrage in both directions
                    opportunities.extend([
                        self._calculate_arbitrage_opportunity(
                            symbol, buy_exchange, sell_exchange, buy_data, sell_data
                        ),
                        self._calculate_arbitrage_opportunity(
                            symbol, sell_exchange, buy_exchange, sell_data, buy_data
                        )
                    ])
        
        return [opp for opp in opportunities if opp is not None]
    
    def _calculate_arbitrage_opportunity(
        self, symbol: str, buy_exchange: str, sell_exchange: str,
        buy_data: Dict, sell_data: Dict
    ) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity between two exchanges"""
        
        buy_price = buy_data['ask']  # Price to buy at
        sell_price = sell_data['bid']  # Price to sell at
        
        if not buy_price or not sell_price or buy_price >= sell_price:
            return None
        
        # Calculate profit
        price_difference = sell_price - buy_price
        spread_percentage = (price_difference / buy_price) * 100
        
        # Calculate fees
        buy_fee = self.exchange_fees[buy_exchange]['taker']
        sell_fee = self.exchange_fees[sell_exchange]['taker']
        total_fees = buy_fee + sell_fee
        
        gross_profit = spread_percentage
        net_profit = gross_profit - (total_fees * 100)
        
        # Skip if not profitable enough
        if net_profit < self.min_profit_threshold:
            return None
        
        # Calculate trade amounts
        buy_volume = buy_data['ask_volume']
        sell_volume = sell_data['bid_volume']
        max_volume = min(buy_volume, sell_volume) * 0.8  # Use 80% of available volume
        
        min_trade_usd = max(
            self.exchange_limits[buy_exchange]['cost']['min'],
            self.exchange_limits[sell_exchange]['cost']['min']
        )
        max_trade_usd = min(max_volume * buy_price, self.min_liquidity_threshold)
        
        if max_trade_usd < min_trade_usd:
            return None  # Not enough liquidity
        
        # Risk assessment
        liquidity_score = min(1.0, max_trade_usd / self.min_liquidity_threshold)
        spread_risk = max(buy_data['spread'], sell_data['spread'])
        slippage_risk = spread_risk * 2  # Rough estimate
        
        risk_level = self._classify_risk_level(net_profit, liquidity_score, slippage_risk)
        
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.now(),
            opportunity_id=f"spot_{symbol}_{buy_exchange}_{sell_exchange}_{int(time.time())}",
            arbitrage_type=ArbitrageType.SPOT_ARBITRAGE,
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            price_difference=price_difference,
            spread_percentage=spread_percentage,
            gross_profit_percentage=gross_profit,
            estimated_fees=total_fees * 100,
            net_profit_percentage=net_profit,
            minimum_trade_amount=min_trade_usd,
            maximum_trade_amount=max_trade_usd,
            risk_level=risk_level,
            liquidity_score=liquidity_score,
            execution_difficulty=0.3,  # Spot arbitrage is relatively easy
            slippage_risk=slippage_risk,
            volume_24h_buy=buy_data['volume'],
            volume_24h_sell=sell_data['volume'],
            bid_ask_spread_buy=buy_data['spread'],
            bid_ask_spread_sell=sell_data['spread'],
            opportunity_window=10.0,  # 10 minutes typical window
            execution_urgency=4,  # High urgency
            decay_rate=0.1,  # 10% per minute decay
            withdrawal_fees={},
            deposit_time_minutes={buy_exchange: 15, sell_exchange: 15},
            withdrawal_time_minutes={buy_exchange: 30, sell_exchange: 30},
            minimum_withdrawal={},
            expected_return=net_profit,
            confidence=0.8,  # High confidence for spot arbitrage
            success_probability=0.7
        )
        
        return opportunity
    
    def _classify_risk_level(self, net_profit: float, liquidity_score: float, slippage_risk: float) -> OpportunityRisk:
        """Classify risk level of arbitrage opportunity"""
        
        risk_score = (slippage_risk * 2) + ((1 - liquidity_score) * 0.5) - (net_profit * 0.1)
        
        if risk_score < 0.01:
            return OpportunityRisk.LOW
        elif risk_score < 0.03:
            return OpportunityRisk.MODERATE
        elif risk_score < 0.05:
            return OpportunityRisk.HIGH
        else:
            return OpportunityRisk.EXTREME
    
    def _detect_funding_rate_opportunities(self) -> List[FundingRateOpportunity]:
        """Detect funding rate arbitrage opportunities"""
        opportunities = []
        
        # Simulate funding rate data (in production, fetch from exchange APIs)
        funding_data = [
            {
                'symbol': 'BTC/USDT',
                'exchange': 'binance',
                'current_rate': 0.0002,  # 0.02% per 8 hours
                'predicted_rate': 0.0003,
                'basis': -0.0001  # Negative basis = backwardation
            },
            {
                'symbol': 'ETH/USDT', 
                'exchange': 'binance',
                'current_rate': 0.00015,
                'predicted_rate': 0.00025,
                'basis': 0.0001  # Positive basis = contango
            }
        ]
        
        for data in funding_data:
            if np.random.random() < 0.3:  # 30% chance of funding opportunity
                
                # Calculate annualized funding rate
                daily_rate = data['current_rate'] * 3  # 3 funding periods per day
                annual_rate = daily_rate * 365
                
                # Determine position recommendation
                if data['current_rate'] > 0.0001:  # High positive rate
                    position = 'short'  # Short futures, long spot
                    expected_daily = daily_rate * 100  # Percentage
                else:
                    position = 'long'   # Long futures, short spot
                    expected_daily = abs(daily_rate) * 100
                
                # Risk assessment
                basis_risk = abs(data['basis'])
                funding_volatility = 0.0001  # Assume some volatility
                
                opportunity = FundingRateOpportunity(
                    timestamp=datetime.now(),
                    symbol=data['symbol'],
                    exchange=data['exchange'],
                    current_funding_rate=data['current_rate'],
                    predicted_funding_rate=data['predicted_rate'],
                    annual_funding_rate=annual_rate,
                    recommended_position=position,
                    position_size_recommendation=0.1,  # 10% of portfolio
                    leverage_recommendation=2.0,  # 2x leverage
                    basis_risk=basis_risk,
                    liquidation_risk=0.05,  # 5% liquidation risk
                    funding_volatility=funding_volatility,
                    expected_daily_return=expected_daily,
                    expected_weekly_return=expected_daily * 7,
                    risk_adjusted_return=expected_daily * 0.8  # Adjust for risk
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _detect_triangular_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage opportunities within single exchange"""
        opportunities = []
        
        # Define triangular paths (simplified)
        triangular_paths = [
            ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            ['BNB/USDT', 'BTC/BNB', 'BTC/USDT'],
            ['ADA/USDT', 'ADA/BTC', 'BTC/USDT']
        ]
        
        for exchange_name, client in self.exchange_clients.items():
            for path in triangular_paths:
                try:
                    if all(symbol in self.price_cache and exchange_name in self.price_cache[symbol] for symbol in path):
                        
                        # Calculate triangular arbitrage
                        opportunity = self._calculate_triangular_opportunity(exchange_name, path)
                        if opportunity:
                            opportunities.append(opportunity)
                            
                except Exception as e:
                    self.logger.error(f"Triangular arbitrage calculation error: {e}")
        
        return opportunities
    
    def _calculate_triangular_opportunity(self, exchange: str, path: List[str]) -> Optional[ArbitrageOpportunity]:
        """Calculate triangular arbitrage opportunity"""
        
        # Get prices for the triangular path
        prices = []
        for symbol in path:
            price_data = self.price_cache[symbol][exchange]
            prices.append({
                'symbol': symbol,
                'bid': price_data['bid'],
                'ask': price_data['ask']
            })
        
        # Calculate forward and reverse paths
        # Forward: USDT -> BTC -> ETH -> USDT
        forward_result = 1.0
        forward_result /= prices[0]['ask']  # USDT to BTC
        forward_result *= prices[1]['bid']  # BTC to ETH  
        forward_result *= prices[2]['bid']  # ETH to USDT
        
        forward_profit = (forward_result - 1.0) * 100
        
        # Reverse: USDT -> ETH -> BTC -> USDT  
        reverse_result = 1.0
        reverse_result /= prices[2]['ask']  # USDT to ETH
        reverse_result /= prices[1]['ask']  # ETH to BTC
        reverse_result *= prices[0]['bid']  # BTC to USDT
        
        reverse_profit = (reverse_result - 1.0) * 100
        
        # Take the more profitable direction
        if forward_profit > reverse_profit and forward_profit > self.min_profit_threshold:
            profit = forward_profit
            direction = "forward"
        elif reverse_profit > self.min_profit_threshold:
            profit = reverse_profit
            direction = "reverse"
        else:
            return None  # Not profitable
        
        # Estimate fees (3 trades)
        fee_rate = self.exchange_fees[exchange]['taker']
        total_fees = fee_rate * 3 * 100  # 3 trades
        net_profit = profit - total_fees
        
        if net_profit < self.min_profit_threshold:
            return None
        
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.now(),
            opportunity_id=f"triangular_{exchange}_{'_'.join(path)}_{int(time.time())}",
            arbitrage_type=ArbitrageType.TRIANGULAR_ARBITRAGE,
            symbol=' -> '.join(path),
            buy_exchange=exchange,
            sell_exchange=exchange,
            buy_price=0.0,  # Not applicable for triangular
            sell_price=0.0,
            price_difference=0.0,
            spread_percentage=profit,
            gross_profit_percentage=profit,
            estimated_fees=total_fees,
            net_profit_percentage=net_profit,
            minimum_trade_amount=100,  # $100 minimum
            maximum_trade_amount=10000,  # $10k maximum for triangular
            risk_level=OpportunityRisk.MODERATE,
            liquidity_score=0.7,
            execution_difficulty=0.8,  # Triangular is more complex
            slippage_risk=0.02,
            volume_24h_buy=0.0,
            volume_24h_sell=0.0,
            bid_ask_spread_buy=0.0,
            bid_ask_spread_sell=0.0,
            opportunity_window=5.0,  # 5 minutes window
            execution_urgency=5,  # Very high urgency
            decay_rate=0.2,  # 20% per minute decay
            withdrawal_fees={},
            deposit_time_minutes={},
            withdrawal_time_minutes={},
            minimum_withdrawal={},
            expected_return=net_profit,
            confidence=0.6,  # Lower confidence for triangular
            success_probability=0.5
        )
        
        return opportunity
    
    def _evaluate_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        """Evaluate and store arbitrage opportunities"""
        
        with self._lock:
            for opportunity in opportunities:
                # Additional validation
                if self._validate_opportunity(opportunity):
                    self.arbitrage_opportunities.append(opportunity)
                    self.stats['total_opportunities_found'] += 1
                    
                    if opportunity.net_profit_percentage > 1.0:  # >1% profit
                        self.stats['profitable_opportunities'] += 1
                        
                        self.logger.info(
                            f"ARBITRAGE OPPORTUNITY: {opportunity.symbol} "
                            f"{opportunity.buy_exchange} -> {opportunity.sell_exchange} "
                            f"{opportunity.net_profit_percentage:.2f}% profit"
                        )
    
    def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate arbitrage opportunity"""
        
        # Check minimum profit threshold
        if opportunity.net_profit_percentage < self.min_profit_threshold:
            return False
        
        # Check liquidity requirements
        if opportunity.maximum_trade_amount < opportunity.minimum_trade_amount:
            return False
        
        # Check risk level
        if opportunity.risk_level == OpportunityRisk.EXTREME:
            return False  # Skip extreme risk opportunities
        
        return True
    
    def _evaluate_funding_opportunities(self, opportunities: List[FundingRateOpportunity]):
        """Evaluate and store funding rate opportunities"""
        
        with self._lock:
            for opportunity in opportunities:
                if opportunity.expected_daily_return > 0.1:  # >0.1% daily return
                    self.funding_opportunities.append(opportunity)
                    
                    self.logger.info(
                        f"FUNDING OPPORTUNITY: {opportunity.symbol} on {opportunity.exchange} "
                        f"{opportunity.expected_daily_return:.2f}% daily return"
                    )
    
    def _cleanup_expired_opportunities(self):
        """Remove expired arbitrage opportunities"""
        cutoff_time = datetime.now() - timedelta(minutes=self.max_execution_time)
        
        with self._lock:
            # Filter arbitrage opportunities
            self.arbitrage_opportunities = deque([
                opp for opp in self.arbitrage_opportunities 
                if opp.timestamp > cutoff_time
            ], maxlen=1000)
            
            # Filter funding opportunities (longer timeframe)
            funding_cutoff = datetime.now() - timedelta(hours=24)
            self.funding_opportunities = deque([
                opp for opp in self.funding_opportunities
                if opp.timestamp > funding_cutoff
            ], maxlen=500)
    
    def _update_statistics(self):
        """Update performance statistics"""
        
        with self._lock:
            if self.arbitrage_opportunities:
                profits = [opp.net_profit_percentage for opp in self.arbitrage_opportunities]
                self.stats['average_profit'] = np.mean(profits)
    
    def get_best_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Get best arbitrage opportunities sorted by profit"""
        
        with self._lock:
            sorted_opportunities = sorted(
                self.arbitrage_opportunities,
                key=lambda x: x.net_profit_percentage,
                reverse=True
            )
            
            return sorted_opportunities[:limit]
    
    def get_funding_opportunities(self) -> List[FundingRateOpportunity]:
        """Get active funding rate opportunities"""
        
        with self._lock:
            return list(self.funding_opportunities)
    
    def get_arbitrage_summary(self) -> Dict[str, Any]:
        """Get summary of arbitrage opportunities"""
        
        with self._lock:
            active_opportunities = len(self.arbitrage_opportunities)
            
            if active_opportunities > 0:
                profits = [opp.net_profit_percentage for opp in self.arbitrage_opportunities]
                avg_profit = np.mean(profits)
                max_profit = max(profits)
                
                # Count by type
                type_counts = defaultdict(int)
                for opp in self.arbitrage_opportunities:
                    type_counts[opp.arbitrage_type.value] += 1
                
            else:
                avg_profit = 0.0
                max_profit = 0.0
                type_counts = {}
            
            return {
                'total_opportunities': active_opportunities,
                'funding_opportunities': len(self.funding_opportunities),
                'average_profit': avg_profit,
                'maximum_profit': max_profit,
                'opportunities_by_type': dict(type_counts),
                'exchanges_monitored': len(self.exchange_clients),
                'symbols_monitored': len(self.monitored_symbols)
            }
    
    def _save_arbitrage_data(self):
        """Save arbitrage data to disk"""
        try:
            # Save best opportunities
            opportunities_file = self.data_path / "arbitrage_opportunities.json"
            best_opportunities = self.get_best_opportunities(50)
            
            opportunities_data = []
            for opp in best_opportunities:
                opportunities_data.append({
                    'timestamp': opp.timestamp.isoformat(),
                    'symbol': opp.symbol,
                    'type': opp.arbitrage_type.value,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'profit_percentage': opp.net_profit_percentage,
                    'risk_level': opp.risk_level.value,
                    'confidence': opp.confidence
                })
            
            with open(opportunities_file, 'w') as f:
                json.dump(opportunities_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving arbitrage data: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'opportunities_found': self.opportunities_found,
            'error_count': self.error_count,
            'active_opportunities': len(self.arbitrage_opportunities),
            'funding_opportunities': len(self.funding_opportunities),
            'exchanges_connected': len(self.exchange_clients),
            'symbols_monitored': len(self.monitored_symbols),
            'statistics': self.stats,
            'has_ccxt': HAS_CCXT
        }