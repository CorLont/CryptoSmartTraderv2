#!/usr/bin/env python3
"""
Unit tests for AsyncDataManager
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import json

from core.async_data_manager import AsyncDataManager, RateLimitConfig
from core.secrets_manager import get_secrets_manager

@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncDataManager:
    """Test AsyncDataManager functionality"""
    
    async def test_initialization(self, test_rate_limit_config):
        """Test AsyncDataManager initialization"""
        manager = AsyncDataManager(test_rate_limit_config)
        
        assert manager.rate_limit_config == test_rate_limit_config
        assert manager.session is None
        assert len(manager.exchanges) == 0
    
    async def test_async_context_manager(self, test_rate_limit_config):
        """Test async context manager functionality"""
        manager = AsyncDataManager(test_rate_limit_config)
        
        with patch.object(manager, 'initialize') as mock_init, \
             patch.object(manager, 'cleanup') as mock_cleanup:
            
            async with manager:
                mock_init.assert_called_once()
            
            mock_cleanup.assert_called_once()
    
    @patch('aiohttp.ClientSession')
    async def test_initialize_creates_session(self, mock_session_class, test_rate_limit_config):
        """Test that initialize creates aiohttp session"""
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        manager = AsyncDataManager(test_rate_limit_config)
        
        with patch.object(manager, 'setup_async_exchanges') as mock_setup:
            await manager.initialize()
        
        assert manager.session == mock_session
        mock_session_class.assert_called_once()
        mock_setup.assert_called_once()
    
    async def test_cleanup_closes_session(self, test_rate_limit_config):
        """Test that cleanup properly closes session and exchanges"""
        manager = AsyncDataManager(test_rate_limit_config)
        
        # Mock session and exchanges
        mock_session = AsyncMock()
        mock_exchange = AsyncMock()
        manager.session = mock_session
        manager.exchanges = {'kraken': mock_exchange}
        
        await manager.cleanup()
        
        mock_session.close.assert_called_once()
        mock_exchange.close.assert_called_once()
    
    @patch('core.async_data_manager.get_secrets_manager')
    @patch('ccxt.async_support.kraken')
    @patch('ccxt.async_support.binance')
    async def test_setup_async_exchanges_with_credentials(
        self, mock_binance, mock_kraken, mock_get_secrets, test_rate_limit_config
    ):
        """Test exchange setup with valid credentials"""
        # Mock secrets manager
        mock_secrets = Mock()
        mock_secrets.get_secret.side_effect = lambda key: {
            'KRAKEN_API_KEY': 'test_kraken_key',
            'KRAKEN_SECRET': 'test_kraken_secret',
            'BINANCE_API_KEY': 'test_binance_key',
            'BINANCE_SECRET': 'test_binance_secret'
        }.get(key)
        mock_get_secrets.return_value = mock_secrets
        
        # Mock exchange constructors
        mock_kraken_instance = AsyncMock()
        mock_binance_instance = AsyncMock()
        mock_kraken.return_value = mock_kraken_instance
        mock_binance.return_value = mock_binance_instance
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.session = AsyncMock()  # Mock session
        
        # Mock structured logger
        manager.structured_logger = Mock()
        manager.structured_logger.info = Mock()
        manager.structured_logger.warning = Mock()
        
        await manager.setup_async_exchanges()
        
        # Verify exchanges were created with credentials
        mock_kraken.assert_called_once_with({
            'apiKey': 'test_kraken_key',
            'secret': 'test_kraken_secret',
            'enableRateLimit': True,
            'timeout': 30000,
            'session': manager.session
        })
        
        mock_binance.assert_called_once_with({
            'apiKey': 'test_binance_key',
            'secret': 'test_binance_secret',
            'enableRateLimit': True,
            'timeout': 30000,
            'session': manager.session
        })
        
        assert 'kraken' in manager.exchanges
        assert 'binance' in manager.exchanges
    
    @patch('core.async_data_manager.get_secrets_manager')
    @patch('ccxt.async_support.kraken')
    async def test_setup_async_exchanges_without_credentials(
        self, mock_kraken, mock_get_secrets, test_rate_limit_config
    ):
        """Test exchange setup without credentials (public mode)"""
        # Mock secrets manager returning None for credentials
        mock_secrets = Mock()
        mock_secrets.get_secret.return_value = None
        mock_get_secrets.return_value = mock_secrets
        
        mock_kraken_instance = AsyncMock()
        mock_kraken.return_value = mock_kraken_instance
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.session = AsyncMock()
        manager.structured_logger = Mock()
        manager.structured_logger.warning = Mock()
        manager.structured_logger.info = Mock()
        
        await manager.setup_async_exchanges()
        
        # Verify exchange was created without credentials
        mock_kraken.assert_called_once_with({
            'enableRateLimit': True,
            'timeout': 30000,
            'session': manager.session
        })
        
        # Should log warning about public mode
        manager.structured_logger.warning.assert_called()
    
    async def test_global_rate_limit(self, test_rate_limit_config):
        """Test global rate limiting functionality"""
        # Use faster rate limit for testing
        config = RateLimitConfig(requests_per_second=2.0, burst_size=2)
        manager = AsyncDataManager(config)
        
        start_time = asyncio.get_event_loop().time()
        
        # Make requests that should trigger rate limiting
        await manager.global_rate_limit()
        await manager.global_rate_limit()
        await manager.global_rate_limit()  # This should be rate limited
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Should take at least some time due to rate limiting
        assert duration > 0.4  # Should be slowed down by rate limiting
    
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_with_retry_success(self, mock_get, test_rate_limit_config):
        """Test successful HTTP fetch with retry mechanism"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'success': True, 'data': 'test'}
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None
        mock_get.return_value = mock_response
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.session = AsyncMock()
        manager.session.get = mock_get
        
        result = await manager.fetch_with_retry('https://api.example.com/data')
        
        assert result == {'success': True, 'data': 'test'}
        mock_get.assert_called_once()
    
    @patch('aiohttp.ClientSession.get')
    async def test_fetch_with_retry_failure(self, mock_get, test_rate_limit_config):
        """Test HTTP fetch with retry on failure"""
        # Mock failing response
        mock_get.side_effect = aiohttp.ClientError("Connection failed")
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.session = AsyncMock()
        manager.session.get = mock_get
        
        result = await manager.fetch_with_retry('https://api.example.com/data')
        
        assert result is None
        assert mock_get.call_count > 1  # Should have retried
    
    async def test_fetch_single_ohlcv_async_success(self, test_rate_limit_config, mock_ohlcv_data):
        """Test successful OHLCV data fetch"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_ohlcv_data
        mock_exchange.name = 'kraken'
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        
        result = await manager.fetch_single_ohlcv_async(mock_exchange, 'BTC/USD', '1h')
        
        assert result == mock_ohlcv_data
        mock_exchange.fetch_ohlcv.assert_called_once_with('BTC/USD', '1h', limit=100)
        manager.structured_logger.log_api_request.assert_called_once()
    
    async def test_fetch_single_ohlcv_async_failure(self, test_rate_limit_config):
        """Test OHLCV fetch failure handling"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        mock_exchange.name = 'kraken'
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        manager.structured_logger.warning = Mock()
        
        result = await manager.fetch_single_ohlcv_async(mock_exchange, 'BTC/USD', '1h')
        
        assert result == []
        manager.structured_logger.warning.assert_called_once()
    
    async def test_batch_collect_tickers_async(self, test_rate_limit_config, mock_kraken_data):
        """Test batch ticker collection"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_tickers.return_value = mock_kraken_data['result']
        mock_exchange.name = 'kraken'
        mock_exchange.id = 'kraken'
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.exchanges = {'kraken': mock_exchange}
        
        result = await manager.batch_collect_tickers_async()
        
        assert 'kraken' in result
        assert result['kraken'] == mock_kraken_data['result']
        mock_exchange.fetch_tickers.assert_called_once()
    
    async def test_batch_collect_all_exchanges(self, test_rate_limit_config, mock_kraken_data):
        """Test comprehensive data collection from all exchanges"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_tickers.return_value = mock_kraken_data['result']
        mock_exchange.name = 'kraken'
        mock_exchange.id = 'kraken'
        
        manager = AsyncDataManager(test_rate_limit_config)
        manager.exchanges = {'kraken': mock_exchange}
        
        with patch.object(manager, 'batch_collect_ohlcv_async') as mock_ohlcv:
            mock_ohlcv.return_value = {'BTC/USD': {'1h': []}}
            
            result = await manager.batch_collect_all_exchanges()
        
        assert 'timestamp' in result
        assert 'exchanges' in result
        assert 'summary' in result
        assert 'kraken' in result['exchanges']
        assert result['summary']['successful'] == 1
        assert result['summary']['failed'] == 0
    
    async def test_store_data_async(self, test_rate_limit_config, temp_dir):
        """Test async data storage"""
        manager = AsyncDataManager(test_rate_limit_config)
        
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'test': 'data'
        }
        
        file_path = temp_dir / 'test_data.json'
        
        # Mock aiofiles
        with patch('aiofiles.open', create=True) as mock_open:
            mock_file = AsyncMock()
            mock_open.return_value.__aenter__.return_value = mock_file
            mock_open.return_value.__aexit__.return_value = None
            
            await manager.store_data_async(test_data, file_path)
            
            mock_open.assert_called_once()
            mock_file.write.assert_called_once()
    
    async def test_concurrent_operations_thread_safety(self, test_rate_limit_config):
        """Test that concurrent operations are thread-safe"""
        manager = AsyncDataManager(test_rate_limit_config)
        
        # Create multiple concurrent rate limit operations
        tasks = [manager.global_rate_limit() for _ in range(10)]
        
        # All should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)

@pytest.mark.unit
class TestRateLimitConfig:
    """Test RateLimitConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 15.0
        assert config.burst_size == 50
        assert config.timeout_seconds == 30
        assert config.cool_down_period == 60
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=25,
            timeout_seconds=15,
            cool_down_period=30
        )
        
        assert config.requests_per_second == 10.0
        assert config.burst_size == 25
        assert config.timeout_seconds == 15
        assert config.cool_down_period == 30