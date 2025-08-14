#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - ML Background Analysis Service
Continuous machine learning analysis and prediction service
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from containers import ApplicationContainer
from config.structured_logging import setup_structured_logging
from utils.orchestrator import Task, TaskStatus, Priority


class MLBackgroundService:
    """Background service for continuous ML analysis"""

    def __init__(self):
        # Initialize dependency injection container
        self.container = ApplicationContainer()

        # Setup structured logging
        setup_structured_logging(
            service_name="ml_background_service",
            log_level="INFO",
            enable_console=True,
            enable_file=True,
        )

        self.logger = logging.getLogger(__name__)
        self.running = False
        self.last_analysis = {}

        # Service configuration
        self.analysis_interval = 900  # 15 minutes
        self.prediction_interval = 1800  # 30 minutes
        self.model_retrain_interval = 86400  # 24 hours

        # Initialize components
        self.config_manager = self.container.config()
        self.orchestrator = self.container.orchestrator()
        self.ml_agent = self.container.ml_predictor_agent()
        self.data_manager = self.container.data_manager()
        self.health_monitor = self.container.health_monitor()

        self.logger.info("ML Background Service initialized")

    async def start_service(self):
        """Start the background ML analysis service"""
        self.running = True
        self.logger.info("Starting ML Background Service")

        # Start orchestrator
        await self.orchestrator.start()

        # Schedule initial tasks
        await self._schedule_initial_tasks()

        # Main service loop
        try:
            while self.running:
                # Check system health
                health_status = self.health_monitor.get_system_health()

                if health_status.get("grade") in ["E", "F"]:
                    self.logger.warning("System health critical, pausing ML analysis")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue

                # Schedule periodic tasks
                await self._schedule_periodic_tasks()

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            self.logger.error(f"ML Background Service error: {e}")
        finally:
            await self.orchestrator.stop()
            self.logger.info("ML Background Service stopped")

    async def _schedule_initial_tasks(self):
        """Schedule initial ML analysis tasks"""
        self.logger.info("Scheduling initial ML analysis tasks")

        # Data collection task
        data_task = Task(
            id="initial_data_collection",
            name="Initial Data Collection",
            agent_type="data_manager",
            function=self._collect_market_data,
            priority=Priority.HIGH,
            timeout=600,
        )

        # Model initialization task
        model_task = Task(
            id="model_initialization",
            name="Model Initialization",
            agent_type="ml_predictor",
            function=self._initialize_models,
            depends_on=["initial_data_collection"],
            priority=Priority.HIGH,
            timeout=1200,
        )

        # Submit tasks
        await self.orchestrator.submit_task(data_task)
        await self.orchestrator.submit_task(model_task)

    async def _schedule_periodic_tasks(self):
        """Schedule periodic ML analysis tasks"""
        current_time = datetime.now()

        # Check if analysis is due
        if self._is_analysis_due("market_analysis", self.analysis_interval):
            analysis_task = Task(
                id=f"market_analysis_{int(current_time.timestamp())}",
                name="Market Analysis",
                agent_type="ml_predictor",
                function=self._run_market_analysis,
                priority=Priority.NORMAL,
                timeout=900,
            )
            await self.orchestrator.submit_task(analysis_task)
            self.last_analysis["market_analysis"] = current_time

        # Check if predictions are due
        if self._is_analysis_due("predictions", self.prediction_interval):
            prediction_task = Task(
                id=f"predictions_{int(current_time.timestamp())}",
                name="Price Predictions",
                agent_type="ml_predictor",
                function=self._generate_predictions,
                priority=Priority.HIGH,
                timeout=1200,
            )
            await self.orchestrator.submit_task(prediction_task)
            self.last_analysis["predictions"] = current_time

        # Check if model retraining is due
        if self._is_analysis_due("model_retrain", self.model_retrain_interval):
            retrain_task = Task(
                id=f"model_retrain_{int(current_time.timestamp())}",
                name="Model Retraining",
                agent_type="ml_predictor",
                function=self._retrain_models,
                priority=Priority.LOW,
                timeout=3600,
            )
            await self.orchestrator.submit_task(retrain_task)
            self.last_analysis["model_retrain"] = current_time

    def _is_analysis_due(self, analysis_type: str, interval: int) -> bool:
        """Check if a specific analysis type is due"""
        if analysis_type not in self.last_analysis:
            return True

        time_since_last = datetime.now() - self.last_analysis[analysis_type]
        return time_since_last.total_seconds() >= interval

    async def _collect_market_data(self):
        """Collect latest market data for analysis"""
        self.logger.info("Collecting market data for ML analysis")

        try:
            # Get top cryptocurrencies
            coins = self.container.coin_registry().get_active_coins()[:50]

            data_collected = 0
            for coin in coins:
                try:
                    # Fetch market data
                    market_data = self.data_manager.get_market_data(symbol=f"{coin}/USD")
                    if market_data is not None and not market_data.empty:
                        data_collected += 1

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    self.logger.warning(f"Failed to collect data for {coin}: {e}")

            self.logger.info(
                f"Market data collection completed: {data_collected}/{len(coins)} coins"
            )
            return {"coins_processed": data_collected, "total_coins": len(coins)}

        except Exception as e:
            self.logger.error(f"Market data collection failed: {e}")
            raise

    async def _initialize_models(self):
        """Initialize ML models for predictions"""
        self.logger.info("Initializing ML models")

        try:
            # Initialize ML predictor agent
            await asyncio.get_event_loop().run_in_executor(None, self.ml_agent.initialize_models)

            self.logger.info("ML models initialized successfully")
            return {"status": "models_initialized"}

        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    async def _run_market_analysis(self):
        """Run comprehensive market analysis"""
        self.logger.info("Running market analysis")

        try:
            # Run technical analysis
            technical_results = await asyncio.get_event_loop().run_in_executor(
                None, self._run_technical_analysis
            )

            # Run sentiment analysis
            sentiment_results = await asyncio.get_event_loop().run_in_executor(
                None, self._run_sentiment_analysis
            )

            # Combine results
            analysis_results = {
                "timestamp": datetime.now().isoformat(),
                "technical": technical_results,
                "sentiment": sentiment_results,
            }

            # Save results
            self._save_analysis_results("market_analysis", analysis_results)

            self.logger.info("Market analysis completed")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            raise

    def _run_technical_analysis(self):
        """Run technical analysis on market data"""
        try:
            technical_agent = self.container.technical_agent()

            # Get top coins for analysis
            coins = self.container.coin_registry().get_active_coins()[:20]
            results = {}

            for coin in coins:
                try:
                    analysis = technical_agent.analyze_market(coin)
                    if analysis:
                        results[coin] = analysis
                except Exception as e:
                    self.logger.warning(f"Technical analysis failed for {coin}: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {}

    def _run_await get_sentiment_analyzer().analyze_text(self):
        """Run sentiment analysis on market data"""
        try:
            sentiment_agent = self.container.sentiment_agent()

            # Get top coins for sentiment analysis
            coins = self.container.coin_registry().get_active_coins()[:20]
            results = {}

            for coin in coins:
                try:
                    sentiment = sentiment_agent.await get_sentiment_analyzer().analyze_text(coin)
                    if sentiment:
                        results[coin] = sentiment
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed for {coin}: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {}

    async def _generate_predictions(self):
        """Generate price predictions for cryptocurrencies"""
        self.logger.info("Generating price predictions")

        try:
            # Run ML predictions
            predictions = await asyncio.get_event_loop().run_in_executor(
                None, self._run_ml_predictions
            )

            # Save predictions
            self._save_analysis_results("predictions", predictions)

            self.logger.info(f"Generated predictions for {len(predictions)} cryptocurrencies")
            return predictions

        except Exception as e:
            self.logger.error(f"Price prediction failed: {e}")
            raise

    def _run_ml_predictions(self):
        """Run ML predictions for cryptocurrencies"""
        try:
            # Get top coins for prediction
            coins = self.container.coin_registry().get_active_coins()[:30]
            predictions = {}

            for coin in coins:
                try:
                    # Generate predictions for different horizons
                    coin_predictions = self.ml_agent.predict_prices(coin)
                    if coin_predictions:
                        predictions[coin] = coin_predictions
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {coin}: {e}")

            return {"timestamp": datetime.now().isoformat(), "predictions": predictions}

        except Exception as e:
            self.logger.error(f"ML predictions failed: {e}")
            return {}

    async def _retrain_models(self):
        """Retrain ML models with latest data"""
        self.logger.info("Starting model retraining")

        try:
            # Run model retraining
            retrain_results = await asyncio.get_event_loop().run_in_executor(
                None, self.ml_agent.retrain_models
            )

            self.logger.info("Model retraining completed")
            return retrain_results

        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            raise

    def _save_analysis_results(self, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to file"""
        try:
            results_dir = Path("data") / "analysis"
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"{analysis_type}_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.debug(f"Analysis results saved: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")

    def stop_service(self):
        """Stop the background service"""
        self.logger.info("Stopping ML Background Service")
        self.running = False


# Signal handlers for graceful shutdown
service_instance = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    if service_instance:
        service_instance.stop_service()
    sys.exit(0)


async def main():
    """Main function to run the ML background service"""
    global service_instance

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start service
    service_instance = MLBackgroundService()

    try:
        await service_instance.start_service()
    except KeyboardInterrupt:
        service_instance.stop_service()
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
