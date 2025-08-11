#!/usr/bin/env python3
"""
Technical Analysis Agent - Technical indicators and analysis
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.structured_logger import get_structured_logger

class TechnicalAnalysisAgent:
    """Agent for technical analysis with multiple indicators"""
    
    def __init__(self):
        self.logger = get_structured_logger("TechnicalAnalysisAgent")
        self.initialized = False
        
    async def initialize(self):
        """Initialize the technical analysis agent"""
        try:
            self.logger.info("Initializing Technical Analysis Agent")
            self.initialized = True
            self.logger.info("Technical Analysis Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Technical Analysis Agent initialization failed: {e}")
            raise
    
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process technical analysis"""
        
        try:
            market_data = data.get('market_data', {})
            
            technical_analysis = {}
            
            for symbol, symbol_data in market_data.items():
                if isinstance(symbol_data, list) and len(symbol_data) > 0:
                    analysis = await self.analyze_symbol(symbol, symbol_data)
                    technical_analysis[symbol] = analysis
            
            return {
                "technical_analysis": technical_analysis,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {"technical_analysis": {}, "status": "error", "error": str(e)}
    
    async def analyze_symbol(self, symbol: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze single symbol with technical indicators"""
        
        try:
            if len(data) < 20:
                return {"error": "Insufficient data for technical analysis"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Calculate technical indicators
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "indicators": {}
            }
            
            # Moving averages
            analysis["indicators"]["sma_20"] = df['close'].rolling(20).mean().iloc[-1]
            analysis["indicators"]["sma_50"] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
            
            # RSI
            analysis["indicators"]["rsi"] = self.calculate_rsi(df['close'].values)
            
            # MACD
            macd_result = self.calculate_macd(df['close'].values)
            analysis["indicators"]["macd"] = macd_result
            
            # Bollinger Bands
            bb_result = self.calculate_bollinger_bands(df['close'].values)
            analysis["indicators"]["bollinger_bands"] = bb_result
            
            # Overall signal
            analysis["overall_signal"] = self.generate_overall_signal(analysis["indicators"])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD indicator"""
        
        if len(prices) < 26:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        # Simple approximation
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:])
        macd = ema_12 - ema_26
        
        signal = macd * 0.9  # Simplified signal line
        histogram = macd - signal
        
        return {
            "macd": float(macd),
            "signal": float(signal),
            "histogram": float(histogram)
        }
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        
        if len(prices) < period:
            current_price = prices[-1] if len(prices) > 0 else 0
            return {
                "upper": current_price * 1.02,
                "middle": current_price,
                "lower": current_price * 0.98
            }
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        return {
            "upper": float(upper),
            "middle": float(sma),
            "lower": float(lower)
        }
    
    def generate_overall_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trading signal from indicators"""
        
        signals = []
        
        # RSI signal
        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            signals.append("BUY")
        elif rsi > 70:
            signals.append("SELL")
        else:
            signals.append("NEUTRAL")
        
        # MACD signal
        macd_data = indicators.get("macd", {})
        if isinstance(macd_data, dict):
            macd = macd_data.get("macd", 0)
            signal = macd_data.get("signal", 0)
            
            if macd > signal:
                signals.append("BUY")
            else:
                signals.append("SELL")
        
        # Count signals
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        if buy_count > sell_count:
            overall = "BUY"
            strength = buy_count / len(signals)
        elif sell_count > buy_count:
            overall = "SELL"
            strength = sell_count / len(signals)
        else:
            overall = "NEUTRAL"
            strength = 0.5
        
        return {
            "signal": overall,
            "strength": strength,
            "confidence": min(strength * 1.2, 1.0)
        }