#!/usr/bin/env python3
"""
Async Data Manager with Global Rate Limiting and Robust Error Handling
Replaces blocking I/O with full async/await implementation
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from asyncio_throttle.throttler import Throttler
import ccxt.async_support as ccxt_async
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_second: float = 10.0
    burst_size: int = 50
    cool_down_period: int = 60

class AsyncDataManager:
    def __init__(self, rate_limit_config: Optional[RateLimitConfig] = None):
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.throttler = Throttler(rate_limit=int(self.rate_limit_config.requests_per_second))
        self.session: Optional[aiohttp.ClientSession] = None
        self.exchanges: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Global rate limiter for all API calls
        self.api_semaphore = asyncio.Semaphore(self.rate_limit_config.burst_size)
        self.last_api_calls: List[float] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize async session and exchanges"""
        # Create aiohttp session with optimized settings
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'CryptoSmartTrader-AsyncClient/2.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        
        # Initialize async exchanges
        await self.setup_async_exchanges()
        
        self.logger.info("Async Data Manager initialized")
    
    async def setup_async_exchanges(self):
        """Setup async exchange connections"""
        import os
        
        try:
            # Kraken async with API credentials
            kraken_key = os.getenv('KRAKEN_API_KEY')
            kraken_secret = os.getenv('KRAKEN_SECRET')
            
            if kraken_key and kraken_secret:
                self.exchanges['kraken'] = ccxt_async.kraken({
                    'apiKey': kraken_key,
                    'secret': kraken_secret,
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'session': self.session
                })
            else:
                self.exchanges['kraken'] = ccxt_async.kraken({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'session': self.session
                })
            
            # Additional async exchanges
            self.exchanges['binance'] = ccxt_async.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'session': self.session
            })
            
            self.logger.info(f"Initialized {len(self.exchanges)} async exchanges")
            
        except Exception as e:
            self.logger.error(f"Failed to setup async exchanges: {e}")
    
    async def global_rate_limit(self):
        """Global rate limiter with burst control"""
        async with self.api_semaphore:
            current_time = time.time()
            
            # Clean old timestamps (older than 1 second)
            self.last_api_calls = [
                call_time for call_time in self.last_api_calls 
                if current_time - call_time < 1.0
            ]
            
            # Check if we're within rate limits
            if len(self.last_api_calls) >= self.rate_limit_config.requests_per_second:
                sleep_time = 1.0 - (current_time - self.last_api_calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.last_api_calls.append(current_time)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, ccxt_async.BaseError))
    )
    async def fetch_market_data_async(self, exchange_name: str) -> Dict[str, Any]:
        """Fetch market data with async retries and rate limiting"""
        await self.global_rate_limit()
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            # Concurrent fetch of multiple data types
            tasks = [
                self.fetch_tickers_async(exchange),
                self.fetch_ohlcv_batch_async(exchange),
                self.fetch_order_book_async(exchange)
            ]
            
            tickers, ohlcv_data, order_books = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # Handle partial failures gracefully
            result = {
                "exchange": exchange_name,
                "timestamp": datetime.now().isoformat(),
                "tickers": tickers if not isinstance(tickers, Exception) else None,
                "ohlcv": ohlcv_data if not isinstance(ohlcv_data, Exception) else None,
                "order_books": order_books if not isinstance(order_books, Exception) else None,
                "errors": []
            }
            
            # Collect any errors
            for data, name in [(tickers, "tickers"), (ohlcv_data, "ohlcv"), (order_books, "order_books")]:
                if isinstance(data, Exception):
                    result["errors"].append(f"{name}: {str(data)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data from {exchange_name}: {e}")
            raise
    
    async def fetch_tickers_async(self, exchange) -> Dict[str, Any]:
        """Fetch tickers with async throttling"""
        async with self.throttler:
            tickers = await exchange.fetch_tickers()
            
            # Filter for USD pairs and add enhanced data
            usd_pairs = {}
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USD'):
                    usd_pairs[symbol] = {
                        'price': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'spread': ((ticker['ask'] - ticker['bid']) / ticker['bid'] * 100) if ticker['bid'] else 0,
                        'volume': ticker['quoteVolume'],
                        'change': ticker['percentage'],
                        'high': ticker['high'],
                        'low': ticker['low'],
                        'vwap': ticker.get('vwap'),
                        'timestamp': ticker['timestamp']
                    }
            
            return usd_pairs
    
    async def fetch_ohlcv_batch_async(self, exchange, timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch OHLCV data for multiple timeframes concurrently"""
        timeframes = timeframes or ['1h', '1d']
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'SOL/USD']
        
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = self.fetch_single_ohlcv_async(exchange, symbol, timeframe)
                tasks.append((symbol, timeframe, task))
        
        # Execute all OHLCV fetches concurrently
        results = await asyncio.gather(
            *[task for _, _, task in tasks], 
            return_exceptions=True
        )
        
        # Organize results
        ohlcv_data = {}
        for (symbol, timeframe, _), result in zip(tasks, results):
            if not isinstance(result, Exception):
                if symbol not in ohlcv_data:
                    ohlcv_data[symbol] = {}
                ohlcv_data[symbol][timeframe] = result
        
        return ohlcv_data
    
    async def fetch_single_ohlcv_async(self, exchange, symbol: str, timeframe: str) -> List[List]:
        """Fetch single OHLCV with rate limiting"""
        async with self.throttler:
            try:
                return await exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            except Exception as e:
                self.logger.warning(f"Failed to fetch OHLCV for {symbol} {timeframe}: {e}")
                return []
    
    async def fetch_order_book_async(self, exchange) -> Dict[str, Any]:
        """Fetch order books for major pairs"""
        major_pairs = ['BTC/USD', 'ETH/USD', 'ADA/USD']
        
        tasks = [
            self.fetch_single_order_book_async(exchange, symbol) 
            for symbol in major_pairs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        order_books = {}
        for symbol, result in zip(major_pairs, results):
            if not isinstance(result, Exception):
                order_books[symbol] = result
        
        return order_books
    
    async def fetch_single_order_book_async(self, exchange, symbol: str) -> Dict[str, Any]:
        """Fetch single order book with throttling"""
        async with self.throttler:
            try:
                book = await exchange.fetch_order_book(symbol, limit=20)
                return {
                    'bids': book['bids'][:10],  # Top 10 bids
                    'asks': book['asks'][:10],  # Top 10 asks
                    'timestamp': book['timestamp']
                }
            except Exception as e:
                self.logger.warning(f"Failed to fetch order book for {symbol}: {e}")
                return {}
    
    async def store_data_async(self, data: Dict[str, Any], file_path: Path):
        """Idempotent async file writes with atomic operations"""
        try:
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first (atomic operation)
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))
            
            # Atomic rename (idempotent)
            temp_path.replace(file_path)
            
            self.logger.debug(f"Data stored to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to store data to {file_path}: {e}")
            raise
    
    async def batch_collect_all_exchanges(self) -> Dict[str, Any]:
        """Collect data from all exchanges concurrently"""
        tasks = [
            self.fetch_market_data_async(exchange_name)
            for exchange_name in self.exchanges.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        collected_data = {
            "timestamp": datetime.now().isoformat(),
            "exchanges": {},
            "summary": {
                "total_exchanges": len(self.exchanges),
                "successful": 0,
                "failed": 0
            }
        }
        
        for exchange_name, result in zip(self.exchanges.keys(), results):
            if isinstance(result, Exception):
                collected_data["exchanges"][exchange_name] = {
                    "status": "error",
                    "error": str(result)
                }
                collected_data["summary"]["failed"] += 1
            else:
                collected_data["exchanges"][exchange_name] = result
                collected_data["summary"]["successful"] += 1
        
        return collected_data
    
    async def cleanup(self):
        """Cleanup async resources"""
        # Close all exchange connections
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing exchange: {e}")
        
        # Close aiohttp session
        if self.session:
            await self.session.close()
        
        self.logger.info("Async Data Manager cleaned up")

# Global async data manager instance
async_data_manager = None

async def get_async_data_manager() -> AsyncDataManager:
    """Get or create global async data manager"""
    global async_data_manager
    
    if async_data_manager is None:
        async_data_manager = AsyncDataManager()
        await async_data_manager.initialize()
    
    return async_data_manager