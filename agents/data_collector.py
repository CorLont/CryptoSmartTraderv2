#!/usr/bin/env python3
"""
Data Collector Agent - Isolated Process
Collects market data from exchanges with circuit breakers
"""

import asyncio
import ccxt
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path
from .base_agent import BaseAgent

class DataCollectorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("data_collector", config)
        
        self.exchanges = {}
        self.collection_interval = self.config.get('collection_interval', 60)
        self.data_dir = Path("data/market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_exchanges()
    
    def setup_exchanges(self):
        """Initialize exchange connections with error handling"""
        try:
            # Kraken with API keys if available
            kraken_key = os.getenv('KRAKEN_API_KEY')
            kraken_secret = os.getenv('KRAKEN_SECRET')
            
            if kraken_key and kraken_secret:
                self.exchanges['kraken'] = ccxt.kraken({
                    'apiKey': kraken_key,
                    'secret': kraken_secret,
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
            else:
                self.exchanges['kraken'] = ccxt.kraken({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
            
            # Additional exchanges for redundancy
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            self.logger.info(f"Initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            self.logger.error(f"Failed to setup exchanges: {e}")
    
    async def perform_health_check(self):
        """Check exchange connectivity"""
        healthy_exchanges = 0
        
        for name, exchange in self.exchanges.items():
            try:
                # Test connection with a simple call
                await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_status
                )
                healthy_exchanges += 1
            except Exception as e:
                self.logger.warning(f"Exchange {name} health check failed: {e}")
        
        if healthy_exchanges == 0:
            raise Exception("No exchanges are healthy")
        
        self.logger.info(f"{healthy_exchanges}/{len(self.exchanges)} exchanges healthy")
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """Collect data from all available exchanges"""
        collected_data = {
            "timestamp": datetime.now().isoformat(),
            "exchanges": {}
        }
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get tickers for major USD pairs
                tickers = await asyncio.get_event_loop().run_in_executor(
                    None, exchange.fetch_tickers
                )
                
                # Filter for USD pairs
                usd_pairs = {k: v for k, v in tickers.items() if k.endswith('/USD')}
                
                # Store only essential data to reduce storage
                exchange_data = {}
                for symbol, ticker in usd_pairs.items():
                    exchange_data[symbol] = {
                        'price': ticker['last'],
                        'volume': ticker['quoteVolume'],
                        'change': ticker['percentage'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'high': ticker['high'],
                        'low': ticker['low'],
                        'timestamp': ticker['timestamp']
                    }
                
                collected_data["exchanges"][exchange_name] = {
                    "status": "success",
                    "pairs_count": len(exchange_data),
                    "data": exchange_data
                }
                
                self.logger.info(f"Collected {len(exchange_data)} pairs from {exchange_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to collect from {exchange_name}: {e}")
                collected_data["exchanges"][exchange_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return collected_data
    
    async def store_data(self, data: Dict[str, Any]):
        """Store collected data with rotation"""
        try:
            # Current data file
            current_file = self.data_dir / "current_market_data.json"
            
            # Store current data
            with open(current_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Hourly snapshot
            hour_file = self.data_dir / f"market_data_{datetime.now().strftime('%Y%m%d_%H')}.json"
            with open(hour_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Cleanup old files (keep last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            for file_path in self.data_dir.glob("market_data_*.json"):
                try:
                    file_time = datetime.strptime(file_path.stem.split('_')[-2:], ['%Y%m%d', '%H'])
                    if file_time < cutoff_time:
                        file_path.unlink()
                except:
                    pass
            
            self.logger.info("Market data stored successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
    
    async def main_loop(self):
        """Main data collection loop"""
        self.logger.info("Starting data collection loop")
        
        while self.running:
            try:
                # Collect data from all exchanges
                market_data = await self.collect_market_data()
                
                # Store the data
                await self.store_data(market_data)
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Data collection loop error: {e}")
                await asyncio.sleep(30)  # Back off on error

def run():
    """Entry point for the data collector agent"""
    from .base_agent import run_agent
    
    config = {
        'collection_interval': 60,  # 1 minute
        'health_check_interval': 30
    }
    
    run_agent(DataCollectorAgent, config)

if __name__ == "__main__":
    run()