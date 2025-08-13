#!/usr/bin/env python3
"""
Portfolio Optimizer Agent - Optimal portfolio allocation and rebalancing
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.structured_logger import get_logger

class PortfolioOptimizerAgent:
    """Agent for portfolio optimization and allocation"""

    def __init__(self):
        self.logger = get_logger("PortfolioOptimizerAgent")
        self.max_positions = 10
        self.min_allocation = 0.01  # 1% minimum allocation
        self.max_allocation = 0.20  # 20% maximum allocation per position
        self.initialized = False

    async def initialize(self):
        """Initialize the portfolio optimizer agent"""
        try:
            self.logger.info("Initializing Portfolio Optimizer Agent")
            self.initialized = True
            self.logger.info("Portfolio Optimizer Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Portfolio Optimizer Agent initialization failed: {e}")
            raise

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and optimize portfolio allocation"""

        try:
            predictions = data.get('predictions', [])
            risk_data = data.get('risk_data', {})

            optimization_result = await self.optimize_portfolio(predictions, risk_data)

            return {
                "portfolio_optimization": optimization_result,
                "status": "success"
            }

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {"portfolio_optimization": {}, "status": "error", "error": str(e)}

    async def optimize_portfolio(self, predictions: List[Dict], risk_data: Dict) -> Dict[str, Any]:
        """Optimize portfolio allocation based on predictions and risk"""

        try:
            if not predictions:
                return {"allocations": {}, "total_allocation": 0.0}

            # Sort predictions by confidence
            sorted_predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)

            # Take top predictions up to max_positions
            top_predictions = sorted_predictions[:self.max_positions]

            allocations = {}
            total_confidence = sum(p.get('confidence', 0) for p in top_predictions)

            for prediction in top_predictions:
                symbol = prediction.get('symbol', 'UNKNOWN')
                confidence = prediction.get('confidence', 0.0)
                direction = prediction.get('direction', 'HOLD')

                if direction != 'HOLD' and confidence > 0.5:
                    # Calculate allocation based on confidence and risk
                    base_allocation = (confidence / total_confidence) if total_confidence > 0 else 0

                    # Apply risk adjustment
                    symbol_risk = risk_data.get('symbol_risks', {}).get(symbol, {})
                    risk_score = symbol_risk.get('risk_score', 0.5)

                    # Reduce allocation for higher risk
                    risk_adjusted_allocation = base_allocation * (1 - risk_score * 0.5)

                    # Apply min/max constraints
                    final_allocation = max(self.min_allocation,
                                         min(self.max_allocation, risk_adjusted_allocation))

                    allocations[symbol] = {
                        "allocation": final_allocation,
                        "confidence": confidence,
                        "direction": direction,
                        "risk_score": risk_score,
                        "recommended_action": direction.lower()
                    }

            # Normalize allocations to sum to 1.0 (or target allocation)
            total_allocation = sum(a["allocation"] for a in allocations.values())

            if total_allocation > 1.0:
                # Scale down to 100%
                for symbol in allocations:
                    allocations[symbol]["allocation"] /= total_allocation
                total_allocation = 1.0

            optimization_result = {
                "allocations": allocations,
                "total_allocation": total_allocation,
                "number_of_positions": len(allocations),
                "optimization_timestamp": datetime.utcnow().isoformat(),
                "optimization_method": "confidence_risk_weighted"
            }

            self.logger.info(f"Portfolio optimized: {len(allocations)} positions, {total_allocation:.2%} allocated")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {}
