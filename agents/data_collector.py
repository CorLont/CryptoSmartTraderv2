#!/usr/bin/env python3
"""
Async Data Collector Agent - Fully Async with Rate Limiting
Replaces blocking I/O with concurrent async operations
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .base_agent import BaseAgent
from core.async_data_manager import AsyncDataManager, RateLimitConfig

class AsyncDataCollectorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("async_data_collector", config)
        
        self.collection_interval = self.config.get('collection_interval', 45)  # Faster collection
        self.data_dir = Path("data/market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure aggressive rate limiting for high-frequency collection
        self.rate_limit_config = RateLimitConfig(
            requests_per_second=15.0,  # Higher throughput
            burst_size=75,             # Larger burst capacity
            cool_down_period=30        # Shorter cooldown
        )
        
        self.data_manager: AsyncDataManager = None
    
    async def initialize_async_components(self):
        """Initialize async data manager"""
        try:
            self.data_manager = AsyncDataManager(self.rate_limit_config)
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
        try:
            # Use the async data manager for concurrent collection
            market_data = await self.data_manager.batch_collect_all_exchanges()
            
            # Enhance with collection metadata
            market_data.update({
                "collection_method": "async_concurrent",
                "agent": self.name,
                "collection_duration": None,  # Will be calculated
                "data_completeness": self.calculate_data_completeness(market_data)
            })
            
            self.logger.info(
                f"Collected data from {market_data['summary']['successful']} exchanges "
                f"({market_data['summary']['failed']} failed)"
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect market data: {e}")
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
        """Main async collection loop with robust error handling"""
        self.logger.info("Starting async data collection loop")
        
        # Initialize async components
        await self.initialize_async_components()
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            collection_start = datetime.now()
            
            try:
                # Collect comprehensive market data
                market_data = await self.collect_comprehensive_market_data()
                
                # Calculate collection duration
                collection_duration = (datetime.now() - collection_start).total_seconds()
                market_data["collection_duration"] = collection_duration
                
                # Store data asynchronously
                await self.store_data_async(market_data)
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Dynamic interval adjustment based on performance
                if collection_duration > self.collection_interval * 0.8:
                    self.logger.warning(f"Collection took {collection_duration:.1f}s, adjusting interval")
                    await asyncio.sleep(max(30, self.collection_interval - collection_duration))
                else:
                    await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Collection error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                # Exponential backoff on repeated errors
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Too many consecutive errors, circuit breaker activated")
                    await asyncio.sleep(300)  # 5 minute circuit breaker
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(min(60, 10 * consecutive_errors))
    
    async def cleanup(self):
        """Cleanup async resources"""
        if self.data_manager:
            await self.data_manager.cleanup()
        await super().cleanup()

def run():
    """Entry point for the async data collector agent"""
    import asyncio
    
    config = {
        'collection_interval': 45,  # Faster with async
        'health_check_interval': 30
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