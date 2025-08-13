#!/usr/bin/env python3
"""
Production Model Training Script
Train and validate production-ready ML models with complete pipeline
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from core.structured_logger import get_structured_logger
from ml.models.predict import MultiHorizonPredictor
from agents.data_collector_agent import DataCollectorAgent


async def train_production_models():
    """Train complete production model suite"""

    logger = get_structured_logger("ProductionModelTraining")
    logger.info("Starting production model training pipeline")

    try:
        # 1. Collect training data
        logger.info("Collecting training data")
        data_collector = DataCollectorAgent()
        await data_collector.initialize()

        # Get comprehensive market data
        symbols = [
            "BTC/USD",
            "ETH/USD",
            "ADA/USD",
            "DOT/USD",
            "LINK/USD",
            "LTC/USD",
            "XRP/USD",
            "BCH/USD",
            "XLM/USD",
            "ETC/USD",
            "ATOM/USD",
            "ALGO/USD",
            "TRX/USD",
            "MATIC/USD",
            "AVAX/USD",
        ]

        market_data = {}
        for symbol in symbols:
            symbol_data = await data_collector.collect_symbol_data(symbol, ["1h"])
            if symbol_data:
                market_data[symbol] = symbol_data

        # Convert to training DataFrame
        training_data = []
        for symbol, data in market_data.items():
            for candle in data:
                training_data.append(
                    {
                        "timestamp": candle["timestamp"],
                        "symbol": symbol,
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle["volume"],
                    }
                )

        df = pd.DataFrame(training_data)
        logger.info(f"Collected {len(df)} training samples for {len(symbols)} symbols")

        # 2. Initialize predictor
        predictor = MultiHorizonPredictor()

        # 3. Train models for all horizons
        logger.info("Training multi-horizon models")
        training_results = await predictor.train_and_save_production_models(df)

        if training_results.get("success", False):
            logger.info("Production models trained successfully")

            # 4. Validate model performance
            performance_results = training_results.get("performance_validation", {})

            logger.info("Model Performance Summary:")
            for horizon, results in performance_results.items():
                mae = results.get("mae", "N/A")
                ready = results.get("production_ready", False)
                logger.info(f"  {horizon}: MAE={mae}, Production Ready={ready}")

            # 5. Test predictions
            logger.info("Testing prediction pipeline")
            test_predictions = predictor.predict_all(df.head(10))

            if len(test_predictions) > 0:
                logger.info(f"Generated {len(test_predictions)} test predictions")

                # Test confidence gate
                from core.strict_gate import StrictConfidenceGate

                gate = StrictConfidenceGate(threshold=0.8)

                # Convert to prediction format
                predictions = []
                for _, row in test_predictions.iterrows():
                    for horizon in [1, 24, 168, 720]:
                        if f"conf_{horizon}" in row:
                            predictions.append(
                                {
                                    "symbol": row.get("coin", "TEST"),
                                    "confidence": row[f"conf_{horizon}"],
                                    "prediction": row.get(f"pred_{horizon}", 0.0),
                                    "horizon": horizon,
                                }
                            )

                # Apply confidence gate
                gate_result = gate.apply_gate(predictions)
                logger.info(
                    f"Confidence gate test: {gate_result['filtered_count']}/{gate_result['original_count']} passed"
                )

                return {
                    "success": True,
                    "models_trained": True,
                    "training_results": training_results,
                    "test_predictions": len(test_predictions),
                    "confidence_gate_passed": gate_result["gate_passed"],
                }
            else:
                logger.error("No test predictions generated")
                return {"success": False, "error": "No test predictions generated"}
        else:
            logger.error("Model training failed")
            return {"success": False, "error": "Model training failed"}

    except Exception as e:
        logger.error(f"Production model training failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(train_production_models())
    print(f"Training result: {result}")
