#!/usr/bin/env python3
"""
Realistic Execution & Slippage Modeling
Enterprise-grade execution simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class OrderExecutionResult:
    """Result of order execution simulation"""
    executed_size: float
    executed_price: float
    slippage_bps: float
    latency_ms: float
    success: bool
    partial_fill: bool

class RealisticExecutionEngine:
    """Realistic execution simulation with L2 orderbook modeling"""
    
    def __init__(self):
        self.execution_history = []
    
    def execute_order(self, 
                     order_size: float,
                     market_price: float,
                     volatility: float = 0.02,
                     volume_24h: float = 1000000,
                     spread_bps: float = 5) -> OrderExecutionResult:
        """Execute order with realistic slippage and latency"""
        
        # Calculate base slippage (in basis points)
        base_slippage = spread_bps / 2  # Half spread
        
        # Size impact (larger orders have more impact)
        size_impact = min(50, (order_size / volume_24h) * 10000)  # Max 50 bps
        
        # Volatility impact
        volatility_impact = volatility * 100  # Convert to bps
        
        # Market stress factor (random)
        stress_factor = np.random.uniform(0.8, 1.5)
        
        # Total slippage
        total_slippage = (base_slippage + size_impact + volatility_impact) * stress_factor
        total_slippage = min(total_slippage, 200)  # Cap at 200 bps
        
        # Calculate execution price
        slippage_factor = total_slippage / 10000  # Convert bps to decimal
        executed_price = market_price * (1 + slippage_factor)
        
        # Latency modeling
        base_latency = 50  # Base 50ms
        network_jitter = np.random.exponential(30)  # Exponential jitter
        market_stress_latency = volatility * 200  # Higher volatility = more latency
        
        total_latency = base_latency + network_jitter + market_stress_latency
        
        # Execution success probability
        success_prob = max(0.7, 1 - (total_slippage / 500))  # Lower success for high slippage
        success = np.random.random() < success_prob
        
        # Partial fill probability
        partial_prob = min(0.3, total_slippage / 100)  # Higher slippage = more partial fills
        partial_fill = np.random.random() < partial_prob and success
        
        executed_size = order_size * (0.5 + np.random.random() * 0.5) if partial_fill else order_size
        
        result = OrderExecutionResult(
            executed_size=executed_size if success else 0,
            executed_price=executed_price if success else market_price,
            slippage_bps=total_slippage if success else 0,
            latency_ms=total_latency,
            success=success,
            partial_fill=partial_fill
        )
        
        self.execution_history.append(result)
        return result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        if not self.execution_history:
            return {"error": "No execution history"}
        
        successes = [ex for ex in self.execution_history if ex.success]
        
        if not successes:
            return {"success_rate": 0.0}
        
        return {
            "total_executions": len(self.execution_history),
            "success_rate": len(successes) / len(self.execution_history),
            "avg_slippage_bps": np.mean([ex.slippage_bps for ex in successes]),
            "p90_slippage_bps": np.percentile([ex.slippage_bps for ex in successes], 90),
            "avg_latency_ms": np.mean([ex.latency_ms for ex in self.execution_history]),
            "partial_fill_rate": sum(1 for ex in successes if ex.partial_fill) / len(successes)
        }

class PortfolioBacktestEngine:
    """Realistic portfolio backtesting with execution costs"""
    
    def __init__(self):
        self.execution_engine = RealisticExecutionEngine()
        self.portfolio_history = []
    
    def backtest_strategy(self, 
                         signals_df: pd.DataFrame,
                         initial_capital: float = 100000) -> Dict[str, Any]:
        """Run realistic backtest with execution costs"""
        
        portfolio_value = initial_capital
        positions = {}
        trades = []
        
        for _, signal in signals_df.iterrows():
            # Simulate trade execution
            execution = self.execution_engine.execute_order(
                order_size=signal.get('position_size', 1000),
                market_price=signal.get('price', 100),
                volatility=signal.get('volatility', 0.02),
                volume_24h=signal.get('volume_24h', 1000000)
            )
            
            if execution.success:
                # Apply execution costs
                execution_cost = execution.executed_size * execution.executed_price
                slippage_cost = execution.executed_size * execution.executed_price * (execution.slippage_bps / 10000)
                
                trades.append({
                    'timestamp': signal.get('timestamp'),
                    'symbol': signal.get('symbol'),
                    'size': execution.executed_size,
                    'price': execution.executed_price,
                    'slippage_bps': execution.slippage_bps,
                    'cost': execution_cost + slippage_cost
                })
        
        # Calculate performance metrics
        total_slippage = sum(trade['slippage_bps'] * trade['size'] for trade in trades) / 10000
        
        return {
            "total_trades": len(trades),
            "successful_executions": len(trades),
            "total_slippage_cost": total_slippage,
            "avg_slippage_bps": np.mean([trade['slippage_bps'] for trade in trades]) if trades else 0,
            "execution_stats": self.execution_engine.get_execution_stats()
        }
