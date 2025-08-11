import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from pathlib import Path

class BacktestAgent:
    """Backtester Evaluator Agent for strategy validation"""
    
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
        
        # Backtest results storage
        self.backtest_results = {}
        self.strategy_performance = {}
        self._lock = threading.Lock()
        
        # Results path
        self.results_path = Path("backtest_results")
        self.results_path.mkdir(exist_ok=True)
        
        # Start agent if enabled
        if self.config_manager.get("agents", {}).get("backtest", {}).get("enabled", True):
            self.start()
    
    def start(self):
        """Start the backtest agent"""
        if not self.active:
            self.active = True
            self.agent_thread = threading.Thread(target=self._backtest_loop, daemon=True)
            self.agent_thread.start()
            self.logger.info("Backtest Agent started")
    
    def stop(self):
        """Stop the backtest agent"""
        self.active = False
        self.logger.info("Backtest Agent stopped")
    
    def _backtest_loop(self):
        """Main backtest loop"""
        while self.active:
            try:
                # Get update interval from config
                interval = self.config_manager.get("agents", {}).get("backtest", {}).get("update_interval", 3600)
                
                # Run backtests
                self._run_backtests()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep until next backtest
                time.sleep(interval)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Backtest error: {str(e)}")
                time.sleep(600)  # Sleep 10 minutes on error
    
    def _run_backtests(self):
        """Run backtests for all available strategies"""
        try:
            symbols = self.data_manager.get_supported_symbols()
            
            # Define strategies to test
            strategies = [
                "moving_average_crossover",
                "rsi_mean_reversion",
                "momentum_breakout",
                "bollinger_bands",
                "trend_following"
            ]
            
            for symbol in symbols[:20]:  # Limit for efficiency
                for strategy in strategies:
                    try:
                        result = self._backtest_strategy(symbol, strategy)
                        if result:
                            self._store_backtest_result(symbol, strategy, result)
                            self.processed_count += 1
                    
                    except Exception as e:
                        self.logger.error(f"Error backtesting {strategy} on {symbol}: {str(e)}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Backtest run error: {str(e)}")
    
    def _backtest_strategy(self, symbol: str, strategy: str, lookback_days: int = 30) -> Optional[Dict[str, Any]]:
        """Backtest a specific strategy on a symbol"""
        # Check cache first
        cache_key = f"backtest_{symbol.replace('/', '_')}_{strategy}_{lookback_days}"
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        try:
            # Get historical data
            historical_data = self.data_manager.get_historical_data(symbol, days=lookback_days + 10)
            
            if historical_data is None or len(historical_data) < 20:
                return None
            
            # Prepare data
            df = historical_data.copy()
            df = df.sort_values('timestamp')
            df.reset_index(drop=True, inplace=True)
            
            # Generate trading signals based on strategy
            signals = self._generate_strategy_signals(df, strategy)
            
            if signals is None:
                return None
            
            # Run backtest simulation
            backtest_result = self._simulate_trading(df, signals, symbol, strategy)
            
            # Cache result
            self.cache_manager.set(cache_key, backtest_result, ttl_minutes=120)
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"Strategy backtest error: {str(e)}")
            return None
    
    def _generate_strategy_signals(self, df: pd.DataFrame, strategy: str) -> Optional[pd.DataFrame]:
        """Generate trading signals for a specific strategy"""
        try:
            df = df.copy()
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                df['close'] = df['price']
            if 'high' not in df.columns:
                df['high'] = df['price']
            if 'low' not in df.columns:
                df['low'] = df['price']
            if 'volume' not in df.columns:
                df['volume'] = 1000
            
            # Initialize signal column
            df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
            df['position'] = 0  # Current position
            
            if strategy == "moving_average_crossover":
                return self._ma_crossover_strategy(df)
            elif strategy == "rsi_mean_reversion":
                return self._rsi_strategy(df)
            elif strategy == "momentum_breakout":
                return self._momentum_strategy(df)
            elif strategy == "bollinger_bands":
                return self._bollinger_strategy(df)
            elif strategy == "trend_following":
                return self._trend_following_strategy(df)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Signal generation error for {strategy}: {str(e)}")
            return None
    
    def _ma_crossover_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving Average Crossover Strategy"""
        # Calculate moving averages
        df['ma_fast'] = df['close'].rolling(window=10).mean()
        df['ma_slow'] = df['close'].rolling(window=20).mean()
        
        # Generate signals
        df['ma_fast_prev'] = df['ma_fast'].shift(1)
        df['ma_slow_prev'] = df['ma_slow'].shift(1)
        
        # Buy when fast MA crosses above slow MA
        buy_condition = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast_prev'] <= df['ma_slow_prev'])
        # Sell when fast MA crosses below slow MA
        sell_condition = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast_prev'] >= df['ma_slow_prev'])
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _rsi_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI Mean Reversion Strategy"""
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        # Buy when RSI < 30 (oversold)
        buy_condition = df['rsi'] < 30
        # Sell when RSI > 70 (overbought)
        sell_condition = df['rsi'] > 70
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _momentum_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum Breakout Strategy"""
        # Calculate momentum indicators
        df['returns'] = df['close'].pct_change()
        df['momentum'] = df['returns'].rolling(window=5).mean()
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Calculate dynamic thresholds
        df['momentum_upper'] = df['momentum'].rolling(window=20).quantile(0.8)
        df['momentum_lower'] = df['momentum'].rolling(window=20).quantile(0.2)
        
        # Generate signals
        # Buy on positive momentum breakout
        buy_condition = df['momentum'] > df['momentum_upper']
        # Sell on negative momentum breakout
        sell_condition = df['momentum'] < df['momentum_lower']
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _bollinger_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands Strategy"""
        # Calculate Bollinger Bands
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
        
        # Generate signals
        # Buy when price touches lower band (oversold)
        buy_condition = df['close'] <= df['bb_lower']
        # Sell when price touches upper band (overbought)
        sell_condition = df['close'] >= df['bb_upper']
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _trend_following_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Following Strategy"""
        # Calculate trend indicators
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['price_trend'] = df['close'] / df['ma50'] - 1
        
        # Calculate ADX-like trend strength
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Generate signals based on trend strength and direction
        # Buy when uptrend and strong momentum
        buy_condition = (df['price_trend'] > 0.02) & (df['atr'] > df['atr'].rolling(window=20).mean())
        # Sell when downtrend and strong momentum
        sell_condition = (df['price_trend'] < -0.02) & (df['atr'] > df['atr'].rolling(window=20).mean())
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def _simulate_trading(self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str, strategy: str) -> Dict[str, Any]:
        """Simulate trading based on signals"""
        try:
            # Trading parameters
            initial_capital = 10000
            transaction_cost = 0.001  # 0.1% per trade
            
            # Initialize tracking variables
            capital = initial_capital
            position = 0  # 0 = no position, 1 = long
            entry_price = 0
            trades = []
            equity_curve = []
            
            for i, row in signals.iterrows():
                current_price = row['close']
                signal = row['signal']
                timestamp = row['timestamp']
                
                # Execute trades based on signals
                if signal == 1 and position == 0:  # Buy signal, no current position
                    # Enter long position
                    position = 1
                    entry_price = current_price
                    trade_value = capital * 0.95  # Use 95% of capital
                    shares = trade_value / current_price
                    transaction_costs = trade_value * transaction_cost
                    capital -= transaction_costs
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'value': trade_value,
                        'costs': transaction_costs
                    })
                
                elif signal == -1 and position == 1:  # Sell signal, currently long
                    # Exit long position
                    position = 0
                    exit_price = current_price
                    trade_value = shares * current_price
                    transaction_costs = trade_value * transaction_cost
                    capital = trade_value - transaction_costs
                    
                    # Calculate trade result
                    trade_return = (exit_price - entry_price) / entry_price
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': current_price,
                        'shares': shares,
                        'value': trade_value,
                        'costs': transaction_costs,
                        'return': trade_return
                    })
                
                # Track equity curve
                if position == 1:
                    current_equity = shares * current_price
                else:
                    current_equity = capital
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_equity,
                    'position': position
                })
            
            # Calculate performance metrics
            final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
            total_return = (final_equity - initial_capital) / initial_capital
            
            # Calculate additional metrics
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1]['equity']
                curr_equity = equity_curve[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            returns = np.array(returns)
            
            # Risk metrics
            if len(returns) > 0:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
                max_drawdown = self._calculate_max_drawdown(equity_curve)
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Win rate
            winning_trades = [t for t in trades if t.get('return', 0) > 0]
            losing_trades = [t for t in trades if t.get('return', 0) < 0]
            total_trades = len([t for t in trades if 'return' in t])
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'strategy': strategy,
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
        except Exception as e:
            self.logger.error(f"Trading simulation error: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = equity_curve[0]['equity']
            max_dd = 0
            
            for point in equity_curve:
                equity = point['equity']
                if equity > peak:
                    peak = equity
                
                drawdown = (peak - equity) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
            
            return max_dd
            
        except Exception:
            return 0
    
    def _store_backtest_result(self, symbol: str, strategy: str, result: Dict[str, Any]):
        """Store backtest result"""
        with self._lock:
            key = f"{symbol}_{strategy}"
            
            if key not in self.backtest_results:
                self.backtest_results[key] = []
            
            self.backtest_results[key].append(result)
            
            # Keep only last 10 results per strategy/symbol
            self.backtest_results[key] = self.backtest_results[key][-10:]
            
            # Update strategy performance summary
            self._update_strategy_performance(strategy, result)
    
    def _update_strategy_performance(self, strategy: str, result: Dict[str, Any]):
        """Update overall strategy performance metrics"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'total_backtests': 0,
                'avg_return': 0,
                'avg_sharpe': 0,
                'avg_win_rate': 0,
                'success_rate': 0,
                'best_return': float('-inf'),
                'worst_return': float('inf')
            }
        
        perf = self.strategy_performance[strategy]
        
        # Update running averages
        total = perf['total_backtests']
        new_total = total + 1
        
        perf['avg_return'] = (perf['avg_return'] * total + result['total_return']) / new_total
        perf['avg_sharpe'] = (perf['avg_sharpe'] * total + result['sharpe_ratio']) / new_total
        perf['avg_win_rate'] = (perf['avg_win_rate'] * total + result['win_rate']) / new_total
        
        # Update extremes
        if result['total_return'] > perf['best_return']:
            perf['best_return'] = result['total_return']
        if result['total_return'] < perf['worst_return']:
            perf['worst_return'] = result['total_return']
        
        # Update success rate (positive returns)
        successful_backtests = perf['success_rate'] * total + (1 if result['total_return'] > 0 else 0)
        perf['success_rate'] = successful_backtests / new_total
        
        perf['total_backtests'] = new_total
    
    def get_backtest_result(self, symbol: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Get latest backtest result for symbol and strategy"""
        with self._lock:
            key = f"{symbol}_{strategy}"
            if key in self.backtest_results and self.backtest_results[key]:
                return self.backtest_results[key][-1]
            return None
    
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get strategies ranked by performance"""
        with self._lock:
            rankings = []
            
            for strategy, perf in self.strategy_performance.items():
                rankings.append({
                    'strategy': strategy,
                    'avg_return': perf['avg_return'],
                    'avg_sharpe': perf['avg_sharpe'],
                    'success_rate': perf['success_rate'],
                    'total_backtests': perf['total_backtests']
                })
            
            # Sort by average return
            rankings.sort(key=lambda x: x['avg_return'], reverse=True)
            
            return rankings
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get comprehensive backtest summary"""
        with self._lock:
            total_backtests = sum(len(results) for results in self.backtest_results.values())
            
            all_returns = []
            all_sharpe = []
            
            for results in self.backtest_results.values():
                for result in results:
                    all_returns.append(result['total_return'])
                    all_sharpe.append(result['sharpe_ratio'])
            
            return {
                'total_backtests': total_backtests,
                'total_strategies': len(self.strategy_performance),
                'total_symbols': len(set(key.split('_')[0] for key in self.backtest_results.keys())),
                'avg_return_all': np.mean(all_returns) if all_returns else 0,
                'avg_sharpe_all': np.mean(all_sharpe) if all_sharpe else 0,
                'best_strategy': max(self.strategy_performance.items(), 
                                   key=lambda x: x[1]['avg_return'])[0] if self.strategy_performance else None,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            'active': self.active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'total_backtests': sum(len(results) for results in self.backtest_results.values()),
            'tracked_strategies': len(self.strategy_performance)
        }
