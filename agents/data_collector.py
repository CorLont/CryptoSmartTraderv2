#!/usr/bin/env python3
"""
Async Data Collector Agent - Fully Async with Dependency Injection
Replaces blocking I/O with concurrent async operations
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
from dependency_injector.wiring import Provide, inject

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .base_agent import BaseAgent
from core.async_data_manager import AsyncDataManager, RateLimitConfig
from core.dependency_container import Container
from core.logging_manager import get_logger
from core.secrets_manager import get_secrets_manager, get_secure_logger
from config.settings import AppSettings

class AsyncDataCollectorAgent(BaseAgent):
    @inject
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        settings: AppSettings = Provide[Container.config],
        data_manager: AsyncDataManager = Provide[Container.async_data_manager],
        rate_limit_config: RateLimitConfig = Provide[Container.rate_limit_config]
    ):
        super().__init__("async_data_collector", config)
        
        # Inject dependencies
        self.settings = settings
        self.data_manager = data_manager
        self.rate_limit_config = rate_limit_config
        
        # Initialize structured logger with secret redaction
        self.structured_logger = get_logger({
            'log_dir': str(settings.logging.log_dir),
            'metrics_port': settings.logging.metrics_port
        })
        
        # Initialize secrets manager
        self.secrets_manager = get_secrets_manager()
        
        # Wrap logger for automatic secret redaction
        self.secure_logger = get_secure_logger(self.logger)
        
        # Configuration from settings
        self.collection_interval = self.config.get('collection_interval', 45)
        self.data_dir = settings.data.data_dir / "market_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_async_components(self):
        """Initialize async data manager (if not already injected)"""
        try:
            if not hasattr(self.data_manager, 'session') or self.data_manager.session is None:
                await self.data_manager.initialize()
            self.logger.info("Async data manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize async components: {e}")
            raise
    
    async def perform_health_check(self):
        """Async health check of all exchanges"""
        if not self.data_manager:
            await self.initialize_async_components()
        
        try:
            # Quick concurrent health check
            health_tasks = []
            for exchange_name in self.data_manager.exchanges.keys():
                task = self.check_single_exchange_health(exchange_name)
                health_tasks.append(task)
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            healthy_count = sum(1 for result in health_results if result is True)
            total_count = len(health_results)
            
            if healthy_count == 0:
                raise Exception("No exchanges are healthy")
            
            self.logger.info(f"Health check: {healthy_count}/{total_count} exchanges healthy")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise
    
    async def check_single_exchange_health(self, exchange_name: str) -> bool:
        """Check health of single exchange"""
        try:
            if exchange_name not in self.data_manager.exchanges:
                return False
            
            exchange = self.data_manager.exchanges[exchange_name]
            
            # Quick status check with timeout
            status = await asyncio.wait_for(
                exchange.fetch_status(),
                timeout=10.0
            )
            
            return status.get('status') == 'ok'
            
        except Exception as e:
            self.logger.warning(f"Exchange {exchange_name} health check failed: {e}")
            return False
    
    async def collect_comprehensive_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data using async batch operations"""
        with self.structured_logger.correlation_context(
            operation="data_collection",
            agent_name=self.name
        ) as ctx:
            try:
                # Use the async data manager for concurrent collection
                collection_start = time.time()
                market_data = await self.data_manager.batch_collect_all_exchanges()
                collection_duration = time.time() - collection_start
                
                # Calculate data completeness
                completeness = self.calculate_data_completeness(market_data)
                
                # Enhance with collection metadata
                market_data.update({
                    "collection_method": "async_concurrent",
                    "agent": self.name,
                    "collection_duration": collection_duration,
                    "data_completeness": completeness
                })
                
                # Log completeness metrics for each exchange
                for exchange_name, exchange_data in market_data.get("exchanges", {}).items():
                    if isinstance(exchange_data, dict) and "error" not in exchange_data:
                        exchange_completeness = 1.0 if exchange_data.get("tickers") else 0.0
                        self.structured_logger.log_data_completeness(
                            exchange_name, "tickers", exchange_completeness
                        )
                
                self.structured_logger.info(
                    f"Market data collection completed successfully",
                    extra={
                        "successful_exchanges": market_data['summary']['successful'],
                        "failed_exchanges": market_data['summary']['failed'],
                        "collection_duration": collection_duration,
                        "overall_completeness": completeness["overall"]
                    }
                )
                
                return market_data
                
            except Exception as e:
                self.structured_logger.error(
                    f"Failed to collect market data: {str(e)}",
                    extra={"error_type": type(e).__name__},
                    exc_info=True
                )
                return {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "exchanges": {},
                    "summary": {"successful": 0, "failed": 1}
                }
    
    def calculate_data_completeness(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate completeness metrics for collected data"""
        completeness = {
            "overall": 0.0,
            "tickers": 0.0,
            "ohlcv": 0.0,
            "order_books": 0.0
        }
        
        try:
            exchanges = market_data.get("exchanges", {})
            total_exchanges = len(exchanges)
            
            if total_exchanges == 0:
                return completeness
            
            successful_exchanges = 0
            tickers_count = 0
            ohlcv_count = 0
            order_books_count = 0
            
            for exchange_data in exchanges.values():
                if isinstance(exchange_data, dict) and "error" not in exchange_data:
                    successful_exchanges += 1
                    
                    if exchange_data.get("tickers"):
                        tickers_count += 1
                    if exchange_data.get("ohlcv"):
                        ohlcv_count += 1
                    if exchange_data.get("order_books"):
                        order_books_count += 1
            
            completeness["overall"] = successful_exchanges / total_exchanges
            completeness["tickers"] = tickers_count / total_exchanges
            completeness["ohlcv"] = ohlcv_count / total_exchanges
            completeness["order_books"] = order_books_count / total_exchanges
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate completeness: {e}")
        
        return completeness
    
    async def store_data_async(self, data: Dict[str, Any]):
        """Async idempotent data storage with atomic writes"""
        try:
            # Current data file (atomic write)
            current_file = self.data_dir / "current_market_data.json"
            await self.data_manager.store_data_async(data, current_file)
            
            # Timestamped file for historical data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            historical_file = self.data_dir / f"market_data_{timestamp}.json"
            await self.data_manager.store_data_async(data, historical_file)
            
            # Concurrent cleanup of old files
            asyncio.create_task(self.cleanup_old_files())
            
            self.logger.info("Market data stored successfully with async I/O")
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
    
    async def cleanup_old_files(self):
        """Async cleanup of old data files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=48)  # Keep 48 hours
            
            for file_path in self.data_dir.glob("market_data_*.json"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = file_path.stem.split('_', 2)[-1]
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        self.logger.debug(f"Cleaned up old file: {file_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def main_loop(self):
        """Main async collection loop with comprehensive monitoring"""
        with self.structured_logger.correlation_context(
            operation="main_collection_loop",
            agent_name=self.name
        ):
            self.structured_logger.info("Starting async data collection loop")
            
            # Initialize async components
            await self.initialize_async_components()
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.running:
                loop_start = time.time()
                
                with self.structured_logger.correlation_context(
                    operation="collection_cycle",
                    agent_name=self.name
                ) as cycle_ctx:
                    try:
                        # Collect comprehensive market data
                        market_data = await self.collect_comprehensive_market_data()
                        
                        # Store data asynchronously
                        await self.store_data_async(market_data)
                        
                        # Calculate total cycle duration
                        cycle_duration = time.time() - loop_start
                        
                        # Reset error counter on success
                        consecutive_errors = 0
                        
                        # Log cycle completion
                        self.structured_logger.info(
                            "Collection cycle completed successfully",
                            extra={
                                "cycle_duration": cycle_duration,
                                "next_collection_in": self.collection_interval
                            }
                        )
                        
                        # Dynamic interval adjustment based on performance
                        if cycle_duration > self.collection_interval * 0.8:
                            adjusted_sleep = max(30, self.collection_interval - cycle_duration)
                            self.structured_logger.warning(
                                "Collection cycle took longer than expected, adjusting sleep",
                                extra={
                                    "cycle_duration": cycle_duration,
                                    "interval_threshold": self.collection_interval * 0.8,
                                    "adjusted_sleep": adjusted_sleep
                                }
                            )
                            await asyncio.sleep(adjusted_sleep)
                        else:
                            await asyncio.sleep(self.collection_interval)
                        
                    except Exception as e:
                        consecutive_errors += 1
                        
                        self.structured_logger.error(
                            "Collection cycle failed",
                            extra={
                                "consecutive_errors": consecutive_errors,
                                "max_consecutive_errors": max_consecutive_errors,
                                "error_type": type(e).__name__
                            },
                            exc_info=True
                        )
                        
                        # Circuit breaker logic with exponential backoff
                        if consecutive_errors >= max_consecutive_errors:
                            circuit_breaker_duration = 300  # 5 minutes
                            
                            self.structured_logger.critical(
                                "Circuit breaker activated due to consecutive errors",
                                extra={
                                    "consecutive_errors": consecutive_errors,
                                    "circuit_breaker_duration": circuit_breaker_duration
                                }
                            )
                            
                            # Set circuit breaker state metric
                            self.structured_logger.metrics.circuit_breaker_state.labels(
                                component=self.name
                            ).set(1)  # Open state
                            
                            await asyncio.sleep(circuit_breaker_duration)
                            consecutive_errors = 0
                            
                            # Reset circuit breaker state
                            self.structured_logger.metrics.circuit_breaker_state.labels(
                                component=self.name
                            ).set(0)  # Closed state
                            
                        else:
                            backoff_duration = min(60, 10 * consecutive_errors)
                            self.structured_logger.warning(
                                "Applying exponential backoff",
                                extra={"backoff_duration": backoff_duration}
                            )
                            await asyncio.sleep(backoff_duration)
    
    async def cleanup(self):
        """Cleanup async resources"""
        if self.data_manager:
            await self.data_manager.cleanup()
        await super().cleanup()

@inject
def run(
    settings: AppSettings = Provide[Container.config]
):
    """Entry point for the async data collector agent with DI"""
    import asyncio
    from core.dependency_container import wire_container
    
    # Wire the container
    wire_container([__name__])
    
    config = {
        'collection_interval': 45,
        'health_check_interval': settings.agents.health_check_interval
    }
    
    agent = AsyncDataCollectorAgent(config)
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.logger.info("Agent interrupted by user")
    except Exception as e:
        agent.logger.error(f"Agent failed: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    run()