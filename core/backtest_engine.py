#!/usr/bin/env python3
"""
Advanced Backtesting Engine with Realistic Execution Simulation
Integrates execution simulator for comprehensive strategy validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger
from core.execution_simulator import (
    get_execution_simulator, Order, OrderSide, OrderType, 
    MarketConditions, ExecutionSimulator
)
from core.ml_slo_monitor import get_slo_monitor

class BacktestMode(str, Enum):
    """Backtesting modes"""
    SIMPLE = "simple"          # Basic price-based simulation
    REALISTIC = "realistic"    # Full execution simulation
    STRESS_TEST = "stress"     # Extreme market conditions

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    max_position_size: float
    commission_rate: float
    slippage_rate: float
    mode: BacktestMode = BacktestMode.REALISTIC
    exchange: str = 'kraken'
    rebalance_frequency: str = 'daily'  # 'hourly', 'daily', 'weekly'
    risk_free_rate: float = 0.02  # Annual risk-free rate
    benchmark_symbol: str = 'BTC/USD'

@dataclass
class Position:
    """Trading position tracking"""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class BacktestTrade:
    """Individual trade record"""
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float
    slippage: float
    duration_hours: float

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    config: BacktestConfig
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_metrics: Dict[str, float]
    trades: List[BacktestTrade]
    portfolio_history: pd.DataFrame
    drawdown_periods: List[Dict[str, Any]]
    monthly_returns: pd.Series
    
class StrategyInterface:
    """Interface for trading strategies"""
    
    def generate_signals(
        self, 
        market_data: pd.DataFrame, 
        portfolio: Dict[str, Position],
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Generate trading signals
        
        Returns:
            List of signal dictionaries with keys:
            - symbol: str
            - action: 'buy', 'sell', 'hold'
            - size: float (position size)
            - confidence: float (0-1)
            - reason: str (signal explanation)
        """
        raise NotImplementedError
    
    def calculate_position_size(
        self, 
        signal: Dict[str, Any], 
        portfolio_value: float,
        risk_budget: float
    ) -> float:
        """Calculate position size based on signal and risk management"""
        raise NotImplementedError

class BacktestEngine:
    """Advanced backtesting engine with realistic execution"""
    
    def __init__(self):
        self.logger = get_logger()
        self.execution_simulator = get_execution_simulator()
        self.slo_monitor = get_slo_monitor()
        
        # Backtest state
        self.current_backtest = None
        self.backtest_history = []
        
    def run_backtest(
        self, 
        strategy: StrategyInterface,
        market_data: pd.DataFrame,
        config: BacktestConfig,
        ml_predictions: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """Run comprehensive backtest with realistic execution"""
        
        self.logger.info(
            f"Starting backtest: {config.start_date} to {config.end_date}",
            extra={
                'strategy': strategy.__class__.__name__,
                'mode': config.mode.value,
                'initial_capital': config.initial_capital,
                'data_points': len(market_data)
            }
        )
        
        # Initialize backtest state
        portfolio = {}
        portfolio_history = []
        trades = []
        cash = config.initial_capital
        
        # Performance tracking
        peak_value = config.initial_capital
        max_drawdown = 0.0
        drawdown_periods = []
        
        # Filter market data to backtest period
        mask = (market_data.index >= config.start_date) & (market_data.index <= config.end_date)
        backtest_data = market_data.loc[mask].copy()
        
        if backtest_data.empty:
            raise ValueError("No market data available for backtest period")
        
        # Main backtest loop
        for timestamp, row in backtest_data.iterrows():
            try:
                # Update portfolio positions with current prices
                portfolio_value = self._update_portfolio_positions(portfolio, row, cash)
                
                # Generate trading signals
                signals = strategy.generate_signals(
                    backtest_data.loc[:timestamp], portfolio, timestamp
                )
                
                # Execute trades based on signals
                new_trades, cash = self._execute_signals(
                    signals, portfolio, row, timestamp, cash, config
                )
                trades.extend(new_trades)
                
                # Update performance tracking
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                else:
                    current_drawdown = (peak_value - portfolio_value) / peak_value
                    max_drawdown = max(max_drawdown, current_drawdown)
                
                # Record portfolio state
                portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'positions_value': portfolio_value - cash,
                    'drawdown': (peak_value - portfolio_value) / peak_value,
                    'position_count': len(portfolio)
                })
                
                # Risk management checks
                if portfolio_value < config.initial_capital * 0.5:  # 50% drawdown limit
                    self.logger.warning(
                        f"Risk limit breached at {timestamp}",
                        extra={
                            'portfolio_value': portfolio_value,
                            'drawdown': (config.initial_capital - portfolio_value) / config.initial_capital
                        }
                    )
                    break
                
            except Exception as e:
                self.logger.error(f"Backtest error at {timestamp}: {e}")
                continue
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Calculate comprehensive metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_df, config)
        risk_metrics = self._calculate_risk_metrics(portfolio_df, trades, config)
        execution_metrics = self._calculate_execution_metrics(trades)
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(portfolio_df)
        
        # Identify drawdown periods
        drawdown_periods = self._identify_drawdown_periods(portfolio_df)
        
        # Create results
        results = BacktestResults(
            config=config,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            execution_metrics=execution_metrics,
            trades=trades,
            portfolio_history=portfolio_df,
            drawdown_periods=drawdown_periods,
            monthly_returns=monthly_returns
        )
        
        # Store backtest for analysis
        self.current_backtest = results
        self.backtest_history.append(results)
        
        # Record performance for SLO monitoring
        if ml_predictions is not None:
            self._record_slo_performance(results, ml_predictions)
        
        self.logger.info(
            f"Backtest completed",
            extra={
                'total_trades': len(trades),
                'final_value': portfolio_df['portfolio_value'].iloc[-1],
                'total_return': performance_metrics.get('total_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0)
            }
        )
        
        return results
    
    def _update_portfolio_positions(
        self, 
        portfolio: Dict[str, Position], 
        market_row: pd.Series, 
        cash: float
    ) -> float:
        """Update portfolio positions with current market prices"""
        
        total_value = cash
        
        for symbol, position in portfolio.items():
            # Get current price for the symbol
            price_col = f'{symbol}_close' if f'{symbol}_close' in market_row else 'close'
            current_price = market_row.get(price_col, position.current_price)
            
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
            
            position_value = current_price * abs(position.size)
            total_value += position_value
        
        return total_value
    
    def _execute_signals(
        self, 
        signals: List[Dict[str, Any]], 
        portfolio: Dict[str, Position],
        market_row: pd.Series,
        timestamp: datetime,
        cash: float,
        config: BacktestConfig
    ) -> Tuple[List[BacktestTrade], float]:
        """Execute trading signals with realistic execution simulation"""
        
        executed_trades = []
        
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']
                confidence = signal.get('confidence', 1.0)
                
                if action == 'hold':
                    continue
                
                # Calculate position size
                position_size = signal.get('size', 0)
                if position_size == 0:
                    continue
                
                # Get current price
                price_col = f'{symbol}_close' if f'{symbol}_close' in market_row else 'close'
                current_price = market_row.get(price_col, 0)
                
                if current_price <= 0:
                    continue
                
                # Create order
                side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
                order = Order(
                    order_id=f"bt_{timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}",
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    size=position_size,
                    timestamp=timestamp
                )
                
                # Create market conditions based on signal confidence
                market_conditions = MarketConditions(
                    volatility=0.02 * (1 - confidence),  # Lower confidence = higher volatility
                    volume_ratio=confidence,  # Higher confidence = better liquidity
                    spread_ratio=1.0,
                    depth_ratio=confidence,
                    market_impact_factor=1.0
                )
                
                # Execute order
                if config.mode == BacktestMode.REALISTIC:
                    executed_order = self.execution_simulator.execute_order(
                        order, current_price, market_conditions, config.exchange
                    )
                else:
                    # Simple execution with fixed slippage and fees
                    executed_order = self._simple_execute_order(order, current_price, config)
                
                # Process execution results
                if executed_order.status.value in ['filled', 'partially_filled']:
                    trade, cash = self._process_executed_order(
                        executed_order, portfolio, cash, timestamp
                    )
                    if trade:
                        executed_trades.append(trade)
                
            except Exception as e:
                self.logger.warning(f"Failed to execute signal: {e}")
        
        return executed_trades, cash
    
    def _simple_execute_order(
        self, 
        order: Order, 
        current_price: float, 
        config: BacktestConfig
    ) -> Order:
        """Simple order execution without full simulation"""
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + config.slippage_rate)
        else:
            execution_price = current_price * (1 - config.slippage_rate)
        
        # Calculate fees
        trade_value = order.size * execution_price
        fees = trade_value * config.commission_rate
        
        # Simple fill
        from core.execution_simulator import ExecutionFill, OrderStatus
        fill = ExecutionFill(
            price=execution_price,
            size=order.size,
            fee=fees,
            timestamp=order.timestamp,
            fee_currency='USD',
            order_book_level=0
        )
        
        order.fills = [fill]
        order.status = OrderStatus.FILLED
        order.average_fill_price = execution_price
        order.total_fees = fees
        order.remaining_size = 0.0
        order.slippage_bps = abs((execution_price - current_price) / current_price) * 10000
        
        return order
    
    def _process_executed_order(
        self, 
        executed_order: Order, 
        portfolio: Dict[str, Position], 
        cash: float,
        timestamp: datetime
    ) -> Tuple[Optional[BacktestTrade], float]:
        """Process executed order and update portfolio"""
        
        symbol = executed_order.symbol
        filled_size = sum(fill.size for fill in executed_order.fills)
        avg_price = executed_order.average_fill_price
        total_fees = executed_order.total_fees
        
        if filled_size == 0:
            return None, cash
        
        # Adjust for order side
        if executed_order.side == OrderSide.SELL:
            filled_size = -filled_size
        
        trade = None
        
        # Update portfolio
        if symbol in portfolio:
            # Existing position
            position = portfolio[symbol]
            
            if (position.size > 0 and filled_size < 0) or (position.size < 0 and filled_size > 0):
                # Closing or reducing position
                close_size = min(abs(filled_size), abs(position.size))
                
                if executed_order.side == OrderSide.SELL:
                    # Selling long position
                    pnl = (avg_price - position.entry_price) * close_size
                    cash += avg_price * close_size - total_fees
                else:
                    # Buying to cover short position
                    pnl = (position.entry_price - avg_price) * close_size
                    cash -= avg_price * close_size + total_fees
                
                # Create trade record
                trade = BacktestTrade(
                    symbol=symbol,
                    side='long' if position.size > 0 else 'short',
                    size=close_size,
                    entry_price=position.entry_price,
                    exit_price=avg_price,
                    entry_time=position.entry_time,
                    exit_time=timestamp,
                    pnl=pnl - total_fees,
                    fees=total_fees,
                    slippage=executed_order.slippage_bps,
                    duration_hours=(timestamp - position.entry_time).total_seconds() / 3600
                )
                
                # Update position
                position.size += filled_size
                position.realized_pnl += pnl
                
                if abs(position.size) < 1e-8:  # Position closed
                    del portfolio[symbol]
            else:
                # Adding to position
                position.size += filled_size
                if executed_order.side == OrderSide.BUY:
                    cash -= avg_price * abs(filled_size) + total_fees
                else:
                    cash += avg_price * abs(filled_size) - total_fees
        else:
            # New position
            portfolio[symbol] = Position(
                symbol=symbol,
                size=filled_size,
                entry_price=avg_price,
                entry_time=timestamp,
                current_price=avg_price
            )
            
            if executed_order.side == OrderSide.BUY:
                cash -= avg_price * abs(filled_size) + total_fees
            else:
                cash += avg_price * abs(filled_size) - total_fees
        
        return trade, cash
    
    def _calculate_performance_metrics(
        self, 
        portfolio_df: pd.DataFrame, 
        config: BacktestConfig
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if portfolio_df.empty:
            return {}
        
        initial_value = config.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        # Returns calculation
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        
        # Basic metrics
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized metrics
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = max(days / 365.25, 1/365.25)  # Minimum 1 day
        
        annualized_return = (final_value / initial_value) ** (1 / years) - 1
        annualized_volatility = returns.std() * np.sqrt(252)  # Assuming daily data
        
        # Risk-adjusted metrics
        excess_returns = returns - config.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'best_day': returns.max() if len(returns) > 0 else 0,
            'worst_day': returns.min() if len(returns) > 0 else 0,
            'final_portfolio_value': final_value
        }
    
    def _calculate_risk_metrics(
        self, 
        portfolio_df: pd.DataFrame, 
        trades: List[BacktestTrade],
        config: BacktestConfig
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        if portfolio_df.empty:
            return {}
        
        # Maximum drawdown
        peak = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        var_99 = returns.quantile(0.01) if len(returns) > 0 else 0
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Trade-based metrics
        if trades:
            trade_pnls = [trade.pnl for trade in trades]
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            win_loss_ratio = 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'profit_factor': profit_factor,
            'avg_winning_trade': avg_win,
            'avg_losing_trade': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'total_trades': len(trades)
        }
    
    def _calculate_execution_metrics(self, trades: List[BacktestTrade]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        
        if not trades:
            return {}
        
        slippages = [trade.slippage for trade in trades]
        fees = [trade.fees for trade in trades]
        durations = [trade.duration_hours for trade in trades]
        
        return {
            'avg_slippage_bps': np.mean(slippages),
            'max_slippage_bps': np.max(slippages),
            'total_fees': sum(fees),
            'avg_fees_per_trade': np.mean(fees),
            'avg_trade_duration_hours': np.mean(durations),
            'median_trade_duration_hours': np.median(durations)
        }
    
    def _calculate_monthly_returns(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns"""
        
        if portfolio_df.empty:
            return pd.Series()
        
        monthly_values = portfolio_df['portfolio_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns
    
    def _identify_drawdown_periods(self, portfolio_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify significant drawdown periods"""
        
        if portfolio_df.empty:
            return []
        
        peak = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak
        
        # Find drawdown periods > 5%
        in_drawdown = drawdown < -0.05
        
        drawdown_periods = []
        start_date = None
        
        for date, is_dd in in_drawdown.items():
            if is_dd and start_date is None:
                start_date = date
            elif not is_dd and start_date is not None:
                max_dd = drawdown[start_date:date].min()
                duration = (date - start_date).days
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'max_drawdown': abs(max_dd),
                    'duration_days': duration
                })
                start_date = None
        
        return drawdown_periods
    
    def _record_slo_performance(
        self, 
        results: BacktestResults, 
        ml_predictions: pd.DataFrame
    ):
        """Record backtest performance for SLO monitoring"""
        
        try:
            # Extract predictions and actuals for SLO monitoring
            if 'prediction' in ml_predictions.columns and 'actual' in ml_predictions.columns:
                predictions = ml_predictions['prediction'].tolist()
                actuals = ml_predictions['actual'].tolist()
                
                # Record for different horizons
                for horizon in ['1h', '24h']:
                    if len(predictions) > 0 and len(actuals) > 0:
                        self.slo_monitor.record_performance(
                            horizon=horizon,
                            model_version='backtest_v1',
                            predictions=predictions[:100],  # Limit size
                            actuals=actuals[:100],
                            confidence_scores=[0.8] * min(100, len(predictions))  # Default confidence
                        )
        
        except Exception as e:
            self.logger.warning(f"Failed to record SLO performance: {e}")
    
    def compare_backtests(
        self, 
        backtest_results: List[BacktestResults]
    ) -> Dict[str, Any]:
        """Compare multiple backtest results"""
        
        if not backtest_results:
            return {}
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'backtest_count': len(backtest_results),
            'metrics_comparison': {},
            'ranking': []
        }
        
        # Compare key metrics
        metrics_to_compare = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in metrics_to_compare:
            values = []
            for i, result in enumerate(backtest_results):
                value = result.performance_metrics.get(metric, 0)
                if metric == 'max_drawdown':
                    value = -value  # Lower is better for drawdown
                values.append((i, value))
            
            # Sort by metric value (descending for most metrics)
            values.sort(key=lambda x: x[1], reverse=True)
            comparison['metrics_comparison'][metric] = values
        
        # Overall ranking (simple average of rankings)
        rankings = {}
        for i in range(len(backtest_results)):
            rankings[i] = 0
        
        for metric_rankings in comparison['metrics_comparison'].values():
            for rank, (backtest_idx, _) in enumerate(metric_rankings):
                rankings[backtest_idx] += rank
        
        # Sort by average ranking
        overall_ranking = sorted(rankings.items(), key=lambda x: x[1])
        comparison['ranking'] = [
            {
                'rank': rank + 1,
                'backtest_index': backtest_idx,
                'avg_ranking_score': score
            }
            for rank, (backtest_idx, score) in enumerate(overall_ranking)
        ]
        
        return comparison
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtest summary"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_backtests_run': len(self.backtest_history),
            'current_backtest': {
                'total_return': self.current_backtest.performance_metrics.get('total_return', 0) if self.current_backtest else 0,
                'sharpe_ratio': self.current_backtest.performance_metrics.get('sharpe_ratio', 0) if self.current_backtest else 0,
                'max_drawdown': self.current_backtest.risk_metrics.get('max_drawdown', 0) if self.current_backtest else 0,
                'total_trades': len(self.current_backtest.trades) if self.current_backtest else 0
            } if self.current_backtest else None,
            'execution_simulator_stats': self.execution_simulator.get_execution_summary()
        }

# Global instance
_backtest_engine = None

def get_backtest_engine() -> BacktestEngine:
    """Get global backtest engine instance"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine