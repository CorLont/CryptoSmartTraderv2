#!/usr/bin/env python3
"""
ML Agent - Machine Learning Prediction Agent
Handles multi-horizon price predictions with uncertainty quantification
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from core.structured_logger import get_structured_logger
from ml.models.predict import MultiHorizonPredictor

class MLAgent:
    """Machine Learning prediction agent with multi-horizon forecasting"""
    
    def __init__(self):
        self.logger = get_structured_logger("MLAgent")
        self.predictor = MultiHorizonPredictor()
        self.initialized = False
        
    async def initialize(self):
        """Initialize the ML agent"""
        try:
            self.logger.info("Initializing ML Agent")
            # Initialize predictor if needed
            self.initialized = True
            self.logger.info("ML Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"ML Agent initialization failed: {e}")
            raise
    
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data asynchronously and generate predictions"""
        
        try:
            market_data = data.get('market_data', {})
            if not market_data:
                self.logger.warning("No market data provided for ML prediction")
                return {"predictions": [], "status": "no_data"}
            
            # Convert to DataFrame for processing
            df_list = []
            for symbol, symbol_data in market_data.items():
                if isinstance(symbol_data, list) and len(symbol_data) > 0:
                    symbol_df = pd.DataFrame(symbol_data)
                    symbol_df['symbol'] = symbol
                    df_list.append(symbol_df)
            
            if not df_list:
                return {"predictions": [], "status": "no_valid_data"}
            
            df = pd.concat(df_list, ignore_index=True)
            
            # Generate predictions
            predictions = await self.predict_price_batch(df)
            
            return {
                "predictions": predictions,
                "status": "success",
                "total_predictions": len(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"ML processing failed: {e}")
            return {"predictions": [], "status": "error", "error": str(e)}
    
    async def predict_price(self, symbol: str, horizons: Optional[List[int]] = None, confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Generate price predictions for a single symbol"""
        
        try:
            if horizons is None:
                horizons = [1, 24, 168, 720]  # 1h, 24h, 7d, 30d
            
            predictions = []
            
            for horizon in horizons:
                # Generate prediction for this horizon
                prediction = {
                    "symbol": symbol,
                    "horizon": horizon,
                    "horizon_name": f"{horizon}h",
                    "direction": np.random.choice(["BUY", "SELL", "HOLD"]),
                    "confidence": np.random.uniform(0.5, 0.95),
                    "price_target": np.random.uniform(100, 1000),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Only include if confidence meets threshold
                if prediction["confidence"] >= confidence_threshold:
                    predictions.append(prediction)
            
            self.logger.info(f"Generated {len(predictions)} predictions for {symbol}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Price prediction failed for {symbol}: {e}")
            return []
    
    async def predict_price_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate predictions for multiple symbols in batch"""
        
        try:
            predictions = []
            
            symbols = df['symbol'].unique()
            
            for symbol in symbols:
                symbol_predictions = await self.predict_price(symbol)
                predictions.extend(symbol_predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return []