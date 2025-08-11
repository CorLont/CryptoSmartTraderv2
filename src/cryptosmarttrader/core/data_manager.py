import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import threading
import time
from pathlib import Path
import json
from datetime import datetime, timedelta
import ccxt
from concurrent.futures import ThreadPoolExecutor
import logging

class DataManager:
    """Centralized data management with real-time feeds and monitoring"""
    
    def __init__(self, config_manager, cache_manager):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.data_path = Path("data")
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize exchange connections
        self.exchanges = {}
        self._init_exchanges()
        
        # Data tracking
        self.data_freshness = {}
        self.data_quality = {}
        self._lock = threading.Lock()
        
        # Background data collection
        self.collection_active = False
        self.start_data_collection()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _init_exchanges(self):
        """Initialize exchange connections"""
        enabled_exchanges = self.config_manager.get("exchanges", ["kraken"])
        
        exchange_classes = {
            "kraken": ccxt.kraken,
            "binance": ccxt.binance,
            "kucoin": ccxt.kucoin,
            "huobi": ccxt.huobi
        }
        
        for exchange_name in enabled_exchanges:
            try:
                exchange_class = exchange_classes.get(exchange_name)
                if exchange_class:
                    # Get API keys from config
                    api_key = self.config_manager.get("api_keys", {}).get(exchange_name, "")
                    secret = self.config_manager.get("api_keys", {}).get(f"{exchange_name}_secret", "")
                    
                    self.exchanges[exchange_name] = exchange_class({
                        'apiKey': api_key,
                        'secret': secret,
                        'timeout': self.config_manager.get("timeout_seconds", 30) * 1000,
                        'rateLimit': 60000 / self.config_manager.get("api_rate_limit", 100),
                        'enableRateLimit': True,
                        'sandbox': False
                    })
                    
                    self.logger.info(f"Initialized {exchange_name} exchange")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {str(e)}")
    
    def start_data_collection(self):
        """Start background data collection"""
        if not self.collection_active:
            self.collection_active = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
    
    def stop_data_collection(self):
        """Stop background data collection"""
        self.collection_active = False
    
    def _collection_loop(self):
        """Main data collection loop"""
        while self.collection_active:
            try:
                # Collect market data
                self._collect_market_data()
                
                # Update data freshness tracking
                self._update_freshness_tracking()
                
                # Sleep between collections
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Data collection error: {str(e)}")
                time.sleep(30)  # Shorter sleep on error
    
    def _collect_market_data(self):
        """Collect market data from exchanges"""
        max_coins = self.config_manager.get("max_coins", 453)
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Fetch tickers for all markets
                tickers = exchange.fetch_tickers()
                
                # Process and store data
                processed_data = self._process_ticker_data(tickers, exchange_name)
                self._store_market_data(processed_data, exchange_name)
                
                # Update data quality metrics
                self._update_data_quality(exchange_name, len(processed_data), max_coins)
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from {exchange_name}: {str(e)}")
                self._update_data_quality(exchange_name, 0, max_coins)
    
    def _process_ticker_data(self, tickers: Dict, exchange_name: str) -> List[Dict]:
        """Process raw ticker data"""
        processed = []
        timestamp = datetime.now()
        
        for symbol, ticker in tickers.items():
            try:
                # Extract base currency (e.g., BTC from BTC/USD)
                if '/' in symbol:
                    base_currency = symbol.split('/')[0]
                else:
                    continue
                
                processed_ticker = {
                    'symbol': symbol,
                    'base_currency': base_currency,
                    'exchange': exchange_name,
                    'timestamp': timestamp.isoformat(),
                    'price': ticker.get('last', 0),
                    'volume': ticker.get('baseVolume', 0),
                    'high': ticker.get('high', 0),
                    'low': ticker.get('low', 0),
                    'open': ticker.get('open', 0),
                    'change': ticker.get('change', 0),
                    'change_percent': ticker.get('percentage', 0),
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0),
                    'spread': (ticker.get('ask', 0) - ticker.get('bid', 0)) if ticker.get('ask') and ticker.get('bid') else 0
                }
                
                processed.append(processed_ticker)
                
            except Exception as e:
                self.logger.error(f"Error processing ticker {symbol}: {str(e)}")
                continue
        
        return processed
    
    def _store_market_data(self, data: List[Dict], exchange_name: str):
        """Store market data to file system and cache"""
        if not data:
            return
        
        try:
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Store to cache for immediate access
            cache_key = f"market_data_{exchange_name}"
            self.cache_manager.set(cache_key, df, ttl_minutes=5)
            
            # Store to file system for historical data
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_path = self.data_path / f"market_{exchange_name}_{date_str}.csv"
            
            # Append to daily file
            if file_path.exists():
                existing_df = pd.read_csv(file_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last').to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
                
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
    
    def _update_data_quality(self, exchange_name: str, collected_count: int, max_expected: int):
        """Update data quality metrics"""
        with self._lock:
            quality_score = (collected_count / max_expected) * 100 if max_expected > 0 else 0
            
            self.data_quality[exchange_name] = {
                'timestamp': datetime.now().isoformat(),
                'collected_count': collected_count,
                'expected_count': max_expected,
                'quality_score': quality_score,
                'status': 'good' if quality_score >= 90 else 'poor' if quality_score < 50 else 'fair'
            }
    
    def _update_freshness_tracking(self):
        """Update data freshness tracking"""
        with self._lock:
            for exchange_name in self.exchanges.keys():
                cache_key = f"market_data_{exchange_name}"
                cached_data = self.cache_manager.get(cache_key)
                
                if cached_data is not None:
                    self.data_freshness[exchange_name] = {
                        'last_update': datetime.now().isoformat(),
                        'status': 'fresh',
                        'age_minutes': 0
                    }
                else:
                    # Calculate age from last known update
                    last_update = self.data_freshness.get(exchange_name, {}).get('last_update')
                    if last_update:
                        try:
                            last_time = datetime.fromisoformat(last_update)
                            age_minutes = (datetime.now() - last_time).total_seconds() / 60
                            
                            self.data_freshness[exchange_name] = {
                                'last_update': last_update,
                                'status': 'stale' if age_minutes > 10 else 'fresh',
                                'age_minutes': age_minutes
                            }
                        except:
                            pass
    
    def get_market_data(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get current market data"""
        if exchange:
            cache_key = f"market_data_{exchange}"
            data = self.cache_manager.get(cache_key)
            
            if data is not None and symbol:
                return data[data['symbol'] == symbol]
            return data
        else:
            # Aggregate data from all exchanges
            all_data = []
            for exchange_name in self.exchanges.keys():
                cache_key = f"market_data_{exchange_name}"
                data = self.cache_manager.get(cache_key)
                if data is not None:
                    all_data.append(data)
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                if symbol:
                    return combined[combined['symbol'] == symbol]
                return combined
            
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            historical_data = []
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                
                for exchange_name in self.exchanges.keys():
                    file_path = self.data_path / f"market_{exchange_name}_{date_str}.csv"
                    
                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path)
                            symbol_data = df[df['symbol'] == symbol]
                            if not symbol_data.empty:
                                historical_data.append(symbol_data)
                        except Exception as e:
                            self.logger.error(f"Error reading historical data: {str(e)}")
                
                current_date += timedelta(days=1)
            
            if historical_data:
                combined = pd.concat(historical_data, ignore_index=True)
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                return combined.sort_values('timestamp')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get comprehensive data status"""
        with self._lock:
            return {
                'freshness': self.data_freshness.copy(),
                'quality': self.data_quality.copy(),
                'exchanges': list(self.exchanges.keys()),
                'collection_active': self.collection_active
            }
    
    def get_supported_symbols(self, exchange: Optional[str] = None) -> List[str]:
        """Get list of supported trading symbols"""
        symbols = set()
        
        if exchange and exchange in self.exchanges:
            cache_key = f"market_data_{exchange}"
            data = self.cache_manager.get(cache_key)
            if data is not None:
                symbols.update(data['symbol'].unique())
        else:
            # Get symbols from all exchanges
            for exchange_name in self.exchanges.keys():
                cache_key = f"market_data_{exchange_name}"
                data = self.cache_manager.get(cache_key)
                if data is not None:
                    symbols.update(data['symbol'].unique())
        
        return sorted(list(symbols))
    
    def cleanup_old_data(self, retention_days: int = None):
        """Clean up old data files"""
        if retention_days is None:
            retention_days = self.config_manager.get("data_retention_days", 365)
        
        cutoff_date = datetime.now().date() - timedelta(days=retention_days)
        
        try:
            for file_path in self.data_path.glob("market_*.csv"):
                # Extract date from filename
                filename = file_path.stem
                date_part = filename.split('_')[-1]
                
                try:
                    file_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                    if file_date < cutoff_date:
                        file_path.unlink()
                        self.logger.info(f"Deleted old data file: {file_path}")
                except ValueError:
                    # Skip files that don't match the expected format
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {str(e)}")
