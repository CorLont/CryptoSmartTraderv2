#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Shadow Trading Engine
Implements live shadow testing and out-of-sample validation for production readiness
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

class ShadowMode(Enum):
    """Shadow trading modes"""
    SILENT = "silent"  # Monitor only, no logging of trades
    MONITORING = "monitoring"  # Log all trades but don't execute
    PAPER_TRADING = "paper_trading"  # Full simulation with portfolio tracking
    LIVE_VALIDATION = "live_validation"  # Compare with live market but don't trade

@dataclass
class ShadowTrade:
    """Shadow trade execution record"""
    timestamp: datetime
    symbol: str
    action: str  # BUY/SELL
    quantity: float
    price: float
    confidence: float
    model_version: str
    market_regime: str
    execution_latency: float
    fees: float
    expected_profit: float
    actual_profit: Optional[float] = None
    trade_id: str = ""
    
    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{self.action}_{int(self.timestamp.timestamp())}"

@dataclass
class ShadowPortfolio:
    """Shadow portfolio tracking"""
    total_value: float
    positions: Dict[str, float]  # symbol -> quantity
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    
class OutOfSampleValidator:
    """Comprehensive out-of-sample validation system"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.holdout_data = {}
        
    def create_validation_split(self, data: pd.DataFrame, 
                              holdout_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation split"""
        try:
            split_idx = int(len(data) * (1 - holdout_ratio))
            
            train_data = data.iloc[:split_idx].copy()
            validation_data = data.iloc[split_idx:].copy()
            
            self.logger.info(f"Created validation split: {len(train_data)} train, {len(validation_data)} validation")
            
            return train_data, validation_data
            
        except Exception as e:
            self.logger.error(f"Validation split creation failed: {e}")
            return data, pd.DataFrame()
    
    def validate_model_performance(self, model, validation_data: pd.DataFrame, 
                                 target_col: str = 'target') -> Dict[str, Any]:
        """Comprehensive out-of-sample model validation"""
        try:
            if target_col not in validation_data.columns:
                return {'error': 'target_column_missing'}
            
            # Prepare features and target
            feature_cols = [col for col in validation_data.columns if col != target_col]
            X_val = validation_data[feature_cols].fillna(0)
            y_true = validation_data[target_col].fillna(0)
            
            if len(X_val) < 5:
                return {'error': 'insufficient_validation_data'}
            
            # Generate predictions
            try:
                y_pred = model.predict(X_val)
            except Exception as e:
                return {'error': f'prediction_failed: {e}'}
            
            # Calculate comprehensive metrics
            metrics = self._calculate_validation_metrics(y_true, y_pred)
            
            # Time-based analysis
            time_analysis = self._analyze_temporal_performance(
                validation_data.index, y_true, y_pred
            )
            
            # Confidence analysis
            confidence_analysis = self._analyze_prediction_confidence(y_pred)
            
            results = {
                'validation_metrics': metrics,
                'temporal_analysis': time_analysis,
                'confidence_analysis': confidence_analysis,
                'data_period': {
                    'start': validation_data.index[0].isoformat() if hasattr(validation_data.index[0], 'isoformat') else str(validation_data.index[0]),
                    'end': validation_data.index[-1].isoformat() if hasattr(validation_data.index[-1], 'isoformat') else str(validation_data.index[-1]),
                    'samples': len(validation_data)
                },
                'model_info': {
                    'type': type(model).__name__,
                    'features_used': len(feature_cols)
                }
            }
            
            self.validation_results[datetime.now().isoformat()] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive validation metrics"""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Additional metrics
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'prediction_std': float(np.std(y_pred)),
                'target_std': float(np.std(y_true))
            }
            
        except Exception as e:
            self.logger.warning(f"Metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_temporal_performance(self, timestamps, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze model performance over time"""
        try:
            # Performance degradation analysis
            window_size = max(10, len(y_true) // 5)
            performance_over_time = []
            
            for i in range(0, len(y_true) - window_size, window_size // 2):
                window_true = y_true[i:i + window_size]
                window_pred = y_pred[i:i + window_size]
                
                window_mse = np.mean((window_true - window_pred) ** 2)
                performance_over_time.append({
                    'start_idx': i,
                    'mse': float(window_mse),
                    'timestamp': timestamps[i] if hasattr(timestamps[i], 'isoformat') else str(timestamps[i])
                })
            
            # Detect performance degradation
            if len(performance_over_time) >= 2:
                first_half_mse = np.mean([p['mse'] for p in performance_over_time[:len(performance_over_time)//2]])
                second_half_mse = np.mean([p['mse'] for p in performance_over_time[len(performance_over_time)//2:]])
                degradation_ratio = second_half_mse / (first_half_mse + 1e-10)
            else:
                degradation_ratio = 1.0
            
            return {
                'performance_windows': performance_over_time,
                'degradation_ratio': float(degradation_ratio),
                'degradation_detected': degradation_ratio > 1.5
            }
            
        except Exception as e:
            self.logger.warning(f"Temporal analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_prediction_confidence(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence and uncertainty"""
        try:
            # Simple confidence metrics based on prediction distribution
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            pred_range = np.max(y_pred) - np.min(y_pred)
            
            # Stability metric
            stability = 1.0 / (1.0 + pred_std)  # Higher stability = lower std
            
            return {
                'prediction_mean': float(pred_mean),
                'prediction_std': float(pred_std),
                'prediction_range': float(pred_range),
                'stability_score': float(stability),
                'extreme_predictions': int(np.sum(np.abs(y_pred) > 2 * pred_std))
            }
            
        except Exception as e:
            self.logger.warning(f"Confidence analysis failed: {e}")
            return {'error': str(e)}

class ShadowTradingEngine:
    """Shadow trading engine for live validation without capital risk"""
    
    def __init__(self, config_manager=None, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Shadow trading configuration
        self.shadow_mode = ShadowMode.MONITORING
        self.initial_capital = 100000.0  # Virtual capital for shadow trading
        
        # Portfolio tracking
        self.shadow_portfolio = ShadowPortfolio(
            total_value=self.initial_capital,
            positions={},
            cash_balance=self.initial_capital,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            max_drawdown=0.0,
            sharpe_ratio=0.0
        )
        
        # Trading history
        self.shadow_trades = []
        self.performance_history = []
        
        # Live monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        self.logger.info(f"Shadow Trading Engine initialized in {self.shadow_mode.value} mode")
    
    def start_shadow_trading(self, mode: ShadowMode = ShadowMode.MONITORING):
        """Start shadow trading in specified mode"""
        try:
            self.shadow_mode = mode
            self.is_monitoring = True
            
            if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
            
            self.logger.info(f"Shadow trading started in {mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to start shadow trading: {e}")
    
    def stop_shadow_trading(self):
        """Stop shadow trading"""
        try:
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Shadow trading stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop shadow trading: {e}")
    
    def execute_shadow_trade(self, signal: Dict[str, Any], current_price: float) -> Optional[ShadowTrade]:
        """Execute a shadow trade based on trading signal"""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            action = signal.get('action', 'BUY')
            confidence = signal.get('confidence', 0.5)
            quantity = self._calculate_position_size(symbol, action, confidence, current_price)
            
            if quantity <= 0:
                return None
            
            # Calculate fees (typical crypto exchange fees)
            fees = quantity * current_price * 0.001  # 0.1% fee
            
            # Create shadow trade
            shadow_trade = ShadowTrade(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=current_price,
                confidence=confidence,
                model_version=signal.get('model_version', 'unknown'),
                market_regime=signal.get('market_regime', 'unknown'),
                execution_latency=signal.get('latency', 0.0),
                fees=fees,
                expected_profit=signal.get('expected_profit', 0.0)
            )
            
            # Update shadow portfolio
            self._update_shadow_portfolio(shadow_trade)
            
            # Record trade
            self.shadow_trades.append(shadow_trade)
            
            # Log trade (if not in silent mode)
            if self.shadow_mode != ShadowMode.SILENT:
                self.logger.info(f"Shadow trade executed: {action} {quantity:.6f} {symbol} @ {current_price:.2f}")
            
            return shadow_trade
            
        except Exception as e:
            self.logger.error(f"Shadow trade execution failed: {e}")
            return None
    
    def _calculate_position_size(self, symbol: str, action: str, 
                               confidence: float, price: float) -> float:
        """Calculate appropriate position size for shadow trade"""
        try:
            # Risk-based position sizing
            max_position_value = self.shadow_portfolio.total_value * 0.05  # Max 5% per position
            confidence_multiplier = min(confidence, 1.0)  # Cap at 1.0
            
            # Adjust for confidence
            position_value = max_position_value * confidence_multiplier
            
            # Calculate quantity
            if action == 'BUY':
                if self.shadow_portfolio.cash_balance < position_value:
                    position_value = self.shadow_portfolio.cash_balance * 0.9  # Leave some cash
                
                quantity = position_value / price
                
            else:  # SELL
                current_position = self.shadow_portfolio.positions.get(symbol, 0.0)
                quantity = min(current_position, position_value / price)
            
            return max(0.0, quantity)
            
        except Exception as e:
            self.logger.warning(f"Position size calculation failed: {e}")
            return 0.0
    
    def _update_shadow_portfolio(self, trade: ShadowTrade):
        """Update shadow portfolio with executed trade"""
        try:
            symbol = trade.symbol
            trade_value = trade.quantity * trade.price
            
            if trade.action == 'BUY':
                # Add position
                self.shadow_portfolio.positions[symbol] = (
                    self.shadow_portfolio.positions.get(symbol, 0.0) + trade.quantity
                )
                # Reduce cash
                self.shadow_portfolio.cash_balance -= (trade_value + trade.fees)
                
            else:  # SELL
                # Reduce position
                current_position = self.shadow_portfolio.positions.get(symbol, 0.0)
                self.shadow_portfolio.positions[symbol] = max(0.0, current_position - trade.quantity)
                
                # Add cash
                self.shadow_portfolio.cash_balance += (trade_value - trade.fees)
                
                # Calculate realized P&L
                # Simplified: assume average cost basis
                if current_position > 0:
                    profit = trade_value - (trade.quantity * trade.price)  # Simplified
                    self.shadow_portfolio.realized_pnl += profit
                    
                    if profit > 0:
                        self.shadow_portfolio.winning_trades += 1
                    else:
                        self.shadow_portfolio.losing_trades += 1
            
            self.shadow_portfolio.total_trades += 1
            
            # Clean up zero positions
            self.shadow_portfolio.positions = {
                k: v for k, v in self.shadow_portfolio.positions.items() if v > 1e-8
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio update failed: {e}")
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value with current market prices"""
        try:
            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            
            for symbol, quantity in self.shadow_portfolio.positions.items():
                if symbol in current_prices:
                    position_value = quantity * current_prices[symbol]
                    # Simplified: assume average cost basis of current price (for demo)
                    unrealized_pnl += position_value - (quantity * current_prices[symbol])
            
            self.shadow_portfolio.unrealized_pnl = unrealized_pnl
            
            # Calculate total portfolio value
            portfolio_value = self.shadow_portfolio.cash_balance
            for symbol, quantity in self.shadow_portfolio.positions.items():
                if symbol in current_prices:
                    portfolio_value += quantity * current_prices[symbol]
            
            self.shadow_portfolio.total_value = portfolio_value
            
            # Update max drawdown
            peak_value = max(self.initial_capital, self.shadow_portfolio.total_value)
            current_drawdown = (peak_value - self.shadow_portfolio.total_value) / peak_value
            self.shadow_portfolio.max_drawdown = max(self.shadow_portfolio.max_drawdown, current_drawdown)
            
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'total_value': portfolio_value,
                'realized_pnl': self.shadow_portfolio.realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'positions_count': len(self.shadow_portfolio.positions)
            })
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Portfolio value update failed: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop for shadow trading"""
        try:
            while self.is_monitoring:
                # Simulate monitoring cycle
                time.sleep(60)  # Check every minute
                
                # Log status periodically
                if len(self.shadow_trades) % 10 == 0 and len(self.shadow_trades) > 0:
                    self._log_performance_summary()
                
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")
    
    def _log_performance_summary(self):
        """Log shadow trading performance summary"""
        try:
            total_return = (self.shadow_portfolio.total_value - self.initial_capital) / self.initial_capital * 100
            win_rate = (self.shadow_portfolio.winning_trades / 
                       max(1, self.shadow_portfolio.winning_trades + self.shadow_portfolio.losing_trades)) * 100
            
            self.logger.info(f"Shadow Trading Summary: "
                           f"Return: {total_return:.2f}%, "
                           f"Trades: {self.shadow_portfolio.total_trades}, "
                           f"Win Rate: {win_rate:.1f}%, "
                           f"Max DD: {self.shadow_portfolio.max_drawdown:.2f}%")
            
        except Exception as e:
            self.logger.warning(f"Performance summary logging failed: {e}")
    
    def get_shadow_trading_report(self) -> Dict[str, Any]:
        """Generate comprehensive shadow trading report"""
        try:
            total_return = (self.shadow_portfolio.total_value - self.initial_capital) / self.initial_capital * 100
            
            if self.shadow_portfolio.total_trades > 0:
                win_rate = (self.shadow_portfolio.winning_trades / 
                           max(1, self.shadow_portfolio.winning_trades + self.shadow_portfolio.losing_trades)) * 100
            else:
                win_rate = 0.0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.performance_history) > 1:
                returns = []
                for i in range(1, len(self.performance_history)):
                    prev_value = self.performance_history[i-1]['total_value']
                    curr_value = self.performance_history[i]['total_value']
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
                
                if len(returns) > 0 and np.std(returns) > 0:
                    self.shadow_portfolio.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)  # Annualized
                else:
                    self.shadow_portfolio.sharpe_ratio = 0.0
            
            report = {
                'mode': self.shadow_mode.value,
                'performance': {
                    'total_return_pct': total_return,
                    'total_value': self.shadow_portfolio.total_value,
                    'realized_pnl': self.shadow_portfolio.realized_pnl,
                    'unrealized_pnl': self.shadow_portfolio.unrealized_pnl,
                    'max_drawdown_pct': self.shadow_portfolio.max_drawdown * 100,
                    'sharpe_ratio': self.shadow_portfolio.sharpe_ratio
                },
                'trading_stats': {
                    'total_trades': self.shadow_portfolio.total_trades,
                    'winning_trades': self.shadow_portfolio.winning_trades,
                    'losing_trades': self.shadow_portfolio.losing_trades,
                    'win_rate_pct': win_rate
                },
                'portfolio': {
                    'cash_balance': self.shadow_portfolio.cash_balance,
                    'positions': dict(self.shadow_portfolio.positions),
                    'positions_count': len(self.shadow_portfolio.positions)
                },
                'recent_trades': [asdict(trade) for trade in self.shadow_trades[-10:]],  # Last 10 trades
                'monitoring_status': self.is_monitoring,
                'report_timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Shadow trading report generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# Main coordinator class
class LiveValidationCoordinator:
    """Coordinates shadow trading and out-of-sample validation"""
    
    def __init__(self, config_manager=None, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.shadow_engine = ShadowTradingEngine(config_manager, cache_manager)
        self.validator = OutOfSampleValidator(config_manager)
        
        self.logger.info("Live Validation Coordinator initialized")
    
    def start_live_validation(self, shadow_mode: ShadowMode = ShadowMode.MONITORING):
        """Start comprehensive live validation"""
        try:
            self.shadow_engine.start_shadow_trading(shadow_mode)
            self.logger.info(f"Live validation started in {shadow_mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to start live validation: {e}")
    
    def stop_live_validation(self):
        """Stop live validation"""
        try:
            self.shadow_engine.stop_shadow_trading()
            self.logger.info("Live validation stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop live validation: {e}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive validation and shadow trading report"""
        try:
            shadow_report = self.shadow_engine.get_shadow_trading_report()
            
            return {
                'shadow_trading': shadow_report,
                'validation_results': self.validator.validation_results,
                'system_status': {
                    'shadow_engine_active': self.shadow_engine.is_monitoring,
                    'total_validations': len(self.validator.validation_results),
                    'total_shadow_trades': len(self.shadow_engine.shadow_trades)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive report generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# Convenience function
def get_live_validation_coordinator(config_manager=None, cache_manager=None) -> LiveValidationCoordinator:
    """Get configured live validation coordinator"""
    return LiveValidationCoordinator(config_manager, cache_manager)

if __name__ == "__main__":
    # Test the shadow trading engine
    coordinator = get_live_validation_coordinator()
    
    print("Testing Shadow Trading Engine...")
    
    # Start shadow trading
    coordinator.start_live_validation(ShadowMode.PAPER_TRADING)
    
    # Simulate some trades
    test_signal = {
        'symbol': 'BTC/USD',
        'action': 'BUY',
        'confidence': 0.8,
        'model_version': 'test_v1',
        'market_regime': 'bull_trending'
    }
    
    shadow_trade = coordinator.shadow_engine.execute_shadow_trade(test_signal, 50000.0)
    if shadow_trade:
        print(f"Shadow trade executed: {shadow_trade.trade_id}")
    
    # Update portfolio with current prices
    coordinator.shadow_engine.update_portfolio_value({'BTC/USD': 51000.0})
    
    # Get report
    report = coordinator.get_comprehensive_report()
    print(f"\nShadow Trading Report:")
    print(f"  Total Value: ${report['shadow_trading']['performance']['total_value']:,.2f}")
    print(f"  Total Return: {report['shadow_trading']['performance']['total_return_pct']:.2f}%")
    print(f"  Total Trades: {report['shadow_trading']['trading_stats']['total_trades']}")
    
    # Stop
    coordinator.stop_live_validation()
    print("Shadow trading test completed")