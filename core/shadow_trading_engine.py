#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Shadow Trading Engine
Paper trading and model validation in live market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import threading
import time

# ML imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class ShadowTradeStatus(Enum):
    """Status of shadow trades"""
    PENDING = "pending"
    EXECUTED = "executed"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class ShadowOrderType(Enum):
    """Types of shadow orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class ShadowTrade:
    """Represents a shadow trade for paper trading"""
    trade_id: str
    coin: str
    side: str  # buy/sell
    order_type: ShadowOrderType
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: ShadowTradeStatus = ShadowTradeStatus.PENDING
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    ml_prediction: Optional[float] = None
    confidence: Optional[float] = None
    strategy: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pnl(self) -> float:
        """Calculate PnL of the trade"""
        if self.exit_price is None:
            return 0.0
        
        if self.side == "buy":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # sell
            return (self.entry_price - self.exit_price) / self.entry_price
    
    @property
    def holding_period_hours(self) -> float:
        """Calculate holding period in hours"""
        if self.exit_time is None:
            return (datetime.now() - self.entry_time).total_seconds() / 3600
        return (self.exit_time - self.entry_time).total_seconds() / 3600
    
    @property
    def is_profitable(self) -> bool:
        """Check if trade is profitable"""
        return self.pnl > 0

@dataclass
class ShadowPortfolio:
    """Represents a shadow trading portfolio"""
    portfolio_id: str
    initial_capital: float
    current_capital: float
    positions: Dict[str, float] = field(default_factory=dict)  # coin -> quantity
    trades: List[ShadowTrade] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_pnl(self) -> float:
        """Calculate total PnL"""
        return (self.current_capital - self.initial_capital) / self.initial_capital
    
    @property
    def total_trades(self) -> int:
        """Get total number of trades"""
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        """Get number of winning trades"""
        return len([t for t in self.trades if t.is_profitable and t.status == ShadowTradeStatus.CLOSED])
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        closed_trades = [t for t in self.trades if t.status == ShadowTradeStatus.CLOSED]
        if not closed_trades:
            return 0.0
        return self.winning_trades / len(closed_trades)

class MarketDataProvider:
    """Provides live market data for shadow trading"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_cache = {}
        self.last_update = {}
    
    def get_current_price(self, coin: str) -> float:
        """Get current market price for a coin"""
        
        # Check cache freshness (5-minute cache)
        if coin in self.last_update:
            if datetime.now() - self.last_update[coin] < timedelta(minutes=5):
                return self.price_cache.get(coin, 0.0)
        
        # In real implementation, this would fetch from exchange API
        # For demo, generate realistic prices
        base_prices = {
            'BTC': 45000,
            'ETH': 2800,
            'ADA': 0.85,
            'DOT': 15.2,
            'SOL': 180,
            'MATIC': 1.2,
            'LINK': 25.5,
            'UNI': 12.8
        }
        
        base_price = base_prices.get(coin, 100)
        
        # Add some realistic price movement
        volatility = 0.02  # 2% volatility
        price_change = np.random.normal(0, volatility)
        current_price = base_price * (1 + price_change)
        
        # Update cache
        self.price_cache[coin] = current_price
        self.last_update[coin] = datetime.now()
        
        return current_price
    
    def get_historical_price(self, coin: str, timestamp: datetime) -> float:
        """Get historical price for a coin at specific timestamp"""
        
        # For demo, return current price with some variation
        current_price = self.get_current_price(coin)
        
        # Add time-based variation
        hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
        drift = np.random.normal(0, 0.001 * hours_ago)  # Price drift over time
        
        return current_price * (1 + drift)

class ShadowOrderExecutor:
    """Executes shadow orders based on market conditions"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.logger = logging.getLogger(__name__)
        self.market_data = market_data
    
    def can_execute_order(self, trade: ShadowTrade, current_price: float) -> bool:
        """Check if order can be executed at current price"""
        
        if trade.order_type == ShadowOrderType.MARKET:
            return True
        
        elif trade.order_type == ShadowOrderType.LIMIT:
            if trade.side == "buy":
                return current_price <= trade.entry_price
            else:  # sell
                return current_price >= trade.entry_price
        
        elif trade.order_type == ShadowOrderType.STOP_LOSS:
            if trade.side == "buy":
                return current_price >= trade.entry_price
            else:  # sell
                return current_price <= trade.entry_price
        
        return False
    
    def execute_entry(self, trade: ShadowTrade) -> bool:
        """Execute trade entry"""
        
        current_price = self.market_data.get_current_price(trade.coin)
        
        if self.can_execute_order(trade, current_price):
            if trade.order_type == ShadowOrderType.MARKET:
                trade.entry_price = current_price
            
            trade.status = ShadowTradeStatus.EXECUTED
            trade.entry_time = datetime.now()
            
            self.logger.info(f"Executed shadow trade: {trade.side} {trade.quantity} {trade.coin} at {trade.entry_price}")
            return True
        
        return False
    
    def check_exit_conditions(self, trade: ShadowTrade) -> Optional[float]:
        """Check if trade should be exited"""
        
        if trade.status != ShadowTradeStatus.EXECUTED:
            return None
        
        current_price = self.market_data.get_current_price(trade.coin)
        
        # Check stop loss
        if trade.stop_loss is not None:
            if trade.side == "buy" and current_price <= trade.stop_loss:
                return current_price
            elif trade.side == "sell" and current_price >= trade.stop_loss:
                return current_price
        
        # Check take profit
        if trade.take_profit is not None:
            if trade.side == "buy" and current_price >= trade.take_profit:
                return current_price
            elif trade.side == "sell" and current_price <= trade.take_profit:
                return current_price
        
        return None
    
    def execute_exit(self, trade: ShadowTrade, exit_price: float):
        """Execute trade exit"""
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = ShadowTradeStatus.CLOSED
        
        self.logger.info(f"Closed shadow trade: {trade.trade_id} at {exit_price}, PnL: {trade.pnl:.2%}")

class ShadowTradingEngine:
    """Main shadow trading engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.market_data = MarketDataProvider()
        self.order_executor = ShadowOrderExecutor(self.market_data)
        
        # Initialize portfolio
        self.portfolio = ShadowPortfolio(
            portfolio_id=f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            initial_capital=initial_capital,
            current_capital=initial_capital
        )
        
        # Trading state
        self.is_running = False
        self.monitoring_thread = None
        self.trade_counter = 0
        
        # Performance tracking
        self.performance_history = []
        self.daily_pnl = []
    
    def create_shadow_trade(self, coin: str, side: str, quantity: float, 
                          order_type: ShadowOrderType = ShadowOrderType.MARKET,
                          entry_price: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          strategy: str = "manual",
                          ml_prediction: Optional[float] = None,
                          confidence: Optional[float] = None) -> ShadowTrade:
        """Create a new shadow trade"""
        
        self.trade_counter += 1
        trade_id = f"shadow_{self.trade_counter}_{datetime.now().strftime('%H%M%S')}"
        
        # Get market price for market orders
        if order_type == ShadowOrderType.MARKET:
            entry_price = self.market_data.get_current_price(coin)
        
        trade = ShadowTrade(
            trade_id=trade_id,
            coin=coin,
            side=side,
            order_type=order_type,
            quantity=quantity,
            entry_price=entry_price or 0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            ml_prediction=ml_prediction,
            confidence=confidence
        )
        
        return trade
    
    def submit_trade(self, trade: ShadowTrade) -> bool:
        """Submit a shadow trade for execution"""
        
        # Validate trade
        if not self._validate_trade(trade):
            return False
        
        # Add to portfolio
        self.portfolio.trades.append(trade)
        
        # Try immediate execution for market orders
        if trade.order_type == ShadowOrderType.MARKET:
            success = self.order_executor.execute_entry(trade)
            if success:
                self._update_portfolio_position(trade)
            return success
        
        self.logger.info(f"Submitted shadow trade: {trade.trade_id}")
        return True
    
    def _validate_trade(self, trade: ShadowTrade) -> bool:
        """Validate trade parameters"""
        
        if trade.quantity <= 0:
            self.logger.error("Trade quantity must be positive")
            return False
        
        # Check if we have enough capital
        current_price = trade.entry_price or self.market_data.get_current_price(trade.coin)
        trade_value = trade.quantity * current_price
        
        if trade.side == "buy" and trade_value > self.portfolio.current_capital * 0.1:  # Max 10% per trade
            self.logger.error("Trade size too large relative to capital")
            return False
        
        return True
    
    def _update_portfolio_position(self, trade: ShadowTrade):
        """Update portfolio position after trade execution"""
        
        if trade.side == "buy":
            if trade.coin not in self.portfolio.positions:
                self.portfolio.positions[trade.coin] = 0
            self.portfolio.positions[trade.coin] += trade.quantity
            
            # Reduce cash
            trade_value = trade.quantity * trade.entry_price
            self.portfolio.current_capital -= trade_value
            
        elif trade.side == "sell":
            if trade.coin in self.portfolio.positions:
                self.portfolio.positions[trade.coin] -= trade.quantity
                if self.portfolio.positions[trade.coin] <= 0:
                    del self.portfolio.positions[trade.coin]
            
            # Increase cash
            trade_value = trade.quantity * trade.entry_price
            self.portfolio.current_capital += trade_value
        
        self.portfolio.last_updated = datetime.now()
    
    def start_monitoring(self):
        """Start the shadow trading monitoring loop"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started shadow trading monitoring")
    
    def stop_monitoring(self):
        """Stop the shadow trading monitoring loop"""
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped shadow trading monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop for shadow trades"""
        
        while self.is_running:
            try:
                # Check pending orders
                for trade in self.portfolio.trades:
                    if trade.status == ShadowTradeStatus.PENDING:
                        self.order_executor.execute_entry(trade)
                        if trade.status == ShadowTradeStatus.EXECUTED:
                            self._update_portfolio_position(trade)
                    
                    elif trade.status == ShadowTradeStatus.EXECUTED:
                        exit_price = self.order_executor.check_exit_conditions(trade)
                        if exit_price is not None:
                            self.order_executor.execute_exit(trade, exit_price)
                            self._update_portfolio_position_on_exit(trade)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for monitoring interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _update_portfolio_position_on_exit(self, trade: ShadowTrade):
        """Update portfolio after trade exit"""
        
        if trade.side == "buy":
            # Remove position
            if trade.coin in self.portfolio.positions:
                self.portfolio.positions[trade.coin] -= trade.quantity
                if self.portfolio.positions[trade.coin] <= 0:
                    del self.portfolio.positions[trade.coin]
            
            # Add cash from sale
            trade_value = trade.quantity * trade.exit_price
            self.portfolio.current_capital += trade_value
        
        self.portfolio.last_updated = datetime.now()
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        
        # Calculate current portfolio value
        portfolio_value = self.portfolio.current_capital
        
        for coin, quantity in self.portfolio.positions.items():
            current_price = self.market_data.get_current_price(coin)
            portfolio_value += quantity * current_price
        
        # Update current capital
        self.portfolio.current_capital = portfolio_value
        
        # Track daily performance
        today = datetime.now().date()
        daily_return = (portfolio_value - self.portfolio.initial_capital) / self.portfolio.initial_capital
        
        # Add to history
        perf_entry = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'total_return': daily_return,
            'daily_return': daily_return,  # Simplified
            'positions': len(self.portfolio.positions),
            'total_trades': len(self.portfolio.trades)
        }
        
        self.performance_history.append(perf_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        closed_trades = [t for t in self.portfolio.trades if t.status == ShadowTradeStatus.CLOSED]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'message': 'No closed trades yet'
            }
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.is_profitable]
        losing_trades = [t for t in closed_trades if not t.is_profitable]
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate Sharpe ratio (simplified)
        if closed_trades:
            returns = [t.pnl for t in closed_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        summary = {
            'portfolio_id': self.portfolio.portfolio_id,
            'total_trades': len(closed_trades),
            'win_rate': len(winning_trades) / len(closed_trades),
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'current_capital': self.portfolio.current_capital,
            'portfolio_return': self.portfolio.total_pnl,
            'active_positions': len(self.portfolio.positions),
            'pending_orders': len([t for t in self.portfolio.trades if t.status == ShadowTradeStatus.PENDING]),
            'last_updated': datetime.now()
        }
        
        return summary
    
    def get_trade_history(self, status_filter: Optional[ShadowTradeStatus] = None) -> List[Dict[str, Any]]:
        """Get trade history with optional status filter"""
        
        trades = self.portfolio.trades
        
        if status_filter:
            trades = [t for t in trades if t.status == status_filter]
        
        trade_data = []
        for trade in trades:
            trade_dict = {
                'trade_id': trade.trade_id,
                'coin': trade.coin,
                'side': trade.side,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl if trade.status == ShadowTradeStatus.CLOSED else None,
                'status': trade.status.value,
                'strategy': trade.strategy,
                'confidence': trade.confidence,
                'holding_period_hours': trade.holding_period_hours,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time
            }
            trade_data.append(trade_dict)
        
        return trade_data
    
    def close_all_positions(self):
        """Close all open positions"""
        
        for trade in self.portfolio.trades:
            if trade.status == ShadowTradeStatus.EXECUTED:
                current_price = self.market_data.get_current_price(trade.coin)
                self.order_executor.execute_exit(trade, current_price)
                self._update_portfolio_position_on_exit(trade)
        
        self.logger.info("Closed all shadow trading positions")

# Global instance
_shadow_engine = None

def get_shadow_trading_engine() -> ShadowTradingEngine:
    """Get or create shadow trading engine"""
    global _shadow_engine
    
    if _shadow_engine is None:
        _shadow_engine = ShadowTradingEngine()
    
    return _shadow_engine

def submit_shadow_trade(coin: str, side: str, quantity: float, strategy: str = "ml_signal",
                       ml_prediction: Optional[float] = None, confidence: Optional[float] = None) -> bool:
    """Submit a shadow trade for paper trading"""
    
    engine = get_shadow_trading_engine()
    
    # Calculate position sizing based on confidence (if provided)
    if confidence is not None and confidence > 0:
        # Reduce quantity for low confidence trades
        quantity *= max(0.1, confidence)
    
    trade = engine.create_shadow_trade(
        coin=coin,
        side=side,
        quantity=quantity,
        strategy=strategy,
        ml_prediction=ml_prediction,
        confidence=confidence
    )
    
    return engine.submit_trade(trade)

def start_shadow_trading():
    """Start shadow trading monitoring"""
    engine = get_shadow_trading_engine()
    engine.start_monitoring()

def get_shadow_performance() -> Dict[str, Any]:
    """Get shadow trading performance summary"""
    engine = get_shadow_trading_engine()
    return engine.get_performance_summary()

if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    engine = get_shadow_trading_engine()
    engine.start_monitoring()
    
    # Submit some demo trades
    submit_shadow_trade("BTC", "buy", 0.01, "demo_strategy", 0.05, 0.8)
    submit_shadow_trade("ETH", "buy", 0.1, "demo_strategy", 0.03, 0.7)
    
    # Wait a bit
    time.sleep(5)
    
    # Get performance
    performance = get_shadow_performance()
    print("Shadow trading performance:", performance)
    
    engine.stop_monitoring()