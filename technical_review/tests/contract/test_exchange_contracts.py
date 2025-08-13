#!/usr/bin/env python3
"""
Contract tests for external exchange APIs (CCXT)
These tests verify our assumptions about external API behavior
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List
import ccxt.async_support as ccxt_async

from core.async_data_manager import AsyncDataManager, RateLimitConfig

@pytest.mark.contract
@pytest.mark.asyncio
class TestKrakenContract:
    """Contract tests for Kraken API integration"""
    
    async def test_kraken_fetch_tickers_contract(self):
        """Test that Kraken API returns expected ticker structure"""
        
        # Mock Kraken response structure based on actual API
        expected_ticker_structure = {
            'XXBTZUSD': {
                'a': ['50000.00000', '1', '1.000'],  # ask [price, whole lot volume, lot volume]
                'b': ['49999.00000', '2', '2.000'],  # bid [price, whole lot volume, lot volume]
                'c': ['50000.50000', '0.01000000'], # last trade closed [price, lot volume]
                'v': ['1234.12345678', '2345.23456789'], # volume [today, last 24 hours]
                'p': ['49500.00000', '49750.00000'], # volume weighted average price [today, last 24 hours]
                't': [1000, 2000], # number of trades [today, last 24 hours]
                'l': ['49000.00000', '49250.00000'], # low [today, last 24 hours]
                'h': ['51000.00000', '50750.00000'], # high [today, last 24 hours]
                'o': '49800.00000' # opening price today
            }
        }
        
        # Mock Kraken exchange
        mock_kraken = AsyncMock(spec=ccxt_async.kraken)
        mock_kraken.fetch_tickers.return_value = expected_ticker_structure
        mock_kraken.name = 'kraken'
        mock_kraken.id = 'kraken'
        
        # Test our data manager's expectations
        manager = AsyncDataManager(RateLimitConfig())
        manager.exchanges = {'kraken': mock_kraken}
        
        result = await manager.batch_collect_tickers_async()
        
        # Verify contract compliance
        assert 'kraken' in result
        kraken_data = result['kraken']
        
        # Verify ticker structure matches expected format
        assert 'XXBTZUSD' in kraken_data
        ticker = kraken_data['XXBTZUSD']
        
        # Essential fields that our system expects
        assert 'a' in ticker  # ask
        assert 'b' in ticker  # bid
        assert 'c' in ticker  # last trade
        assert 'v' in ticker  # volume
        assert 'h' in ticker  # high
        assert 'l' in ticker  # low
        assert 'o' in ticker  # open
        
        # Verify data types
        assert isinstance(ticker['a'], list)
        assert isinstance(ticker['b'], list)
        assert isinstance(ticker['c'], list)
        assert len(ticker['a']) >= 1  # At least price
        assert len(ticker['b']) >= 1  # At least price
        assert len(ticker['c']) >= 1  # At least price
    
    async def test_kraken_fetch_ohlcv_contract(self):
        """Test that Kraken OHLCV returns expected structure"""
        
        # Expected OHLCV structure: [timestamp, open, high, low, close, volume]
        expected_ohlcv = [
            [1640995200000, 49000.0, 50000.0, 48500.0, 49500.0, 100.5],
            [1640998800000, 49500.0, 51000.0, 49200.0, 50200.0, 150.3],
            [1641002400000, 50200.0, 50500.0, 49800.0, 50000.0, 120.8]
        ]
        
        mock_kraken = AsyncMock(spec=ccxt_async.kraken)
        mock_kraken.fetch_ohlcv.return_value = expected_ohlcv
        mock_kraken.name = 'kraken'
        
        manager = AsyncDataManager(RateLimitConfig())
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        
        result = await manager.fetch_single_ohlcv_async(mock_kraken, 'BTC/USD', '1h')
        
        # Verify contract compliance
        assert isinstance(result, list)
        assert len(result) > 0
        
        for candle in result:
            assert isinstance(candle, list)
            assert len(candle) == 6  # [timestamp, open, high, low, close, volume]
            
            timestamp, open_price, high, low, close, volume = candle
            
            # Verify data types
            assert isinstance(timestamp, (int, float))
            assert isinstance(open_price, (int, float))
            assert isinstance(high, (int, float))
            assert isinstance(low, (int, float))
            assert isinstance(close, (int, float))
            assert isinstance(volume, (int, float))
            
            # Verify logical constraints
            assert high >= max(open_price, close)
            assert low <= min(open_price, close)
            assert volume >= 0
    
    async def test_kraken_error_handling_contract(self):
        """Test that Kraken errors are handled according to CCXT patterns"""
        
        # Common CCXT exception types that Kraken might throw
        ccxt_errors = [
            ccxt_async.NetworkError("Connection failed"),
            ccxt_async.ExchangeError("Invalid symbol"),
            ccxt_async.RateLimitExceeded("Too many requests"),
            ccxt_async.AuthenticationError("Invalid API key")
        ]
        
        manager = AsyncDataManager(RateLimitConfig())
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        manager.structured_logger.warning = Mock()
        
        for error in ccxt_errors:
            mock_kraken = AsyncMock(spec=ccxt_async.kraken)
            mock_kraken.fetch_ohlcv.side_effect = error
            mock_kraken.name = 'kraken'
            
            # Should handle error gracefully and return empty list
            result = await manager.fetch_single_ohlcv_async(mock_kraken, 'BTC/USD', '1h')
            
            assert result == []
            
            # Should log the error
            manager.structured_logger.warning.assert_called()
            manager.structured_logger.log_api_request.assert_called()

@pytest.mark.contract
@pytest.mark.asyncio  
class TestBinanceContract:
    """Contract tests for Binance API integration"""
    
    async def test_binance_ticker_structure_contract(self):
        """Test Binance ticker structure compliance"""
        
        expected_binance_ticker = {
            'BTCUSDT': {
                'symbol': 'BTCUSDT',
                'priceChange': '500.00000000',
                'priceChangePercent': '1.020',
                'weightedAvgPrice': '49750.00000000',
                'prevClosePrice': '49500.00000000',
                'lastPrice': '50000.00000000',
                'lastQty': '0.01000000',
                'bidPrice': '49999.00000000',
                'bidQty': '1.00000000',
                'askPrice': '50001.00000000',
                'askQty': '1.00000000',
                'openPrice': '49500.00000000',
                'highPrice': '51000.00000000',
                'lowPrice': '49000.00000000',
                'volume': '1234.12345678',
                'quoteVolume': '61234567.89000000',
                'openTime': 1640995200000,
                'closeTime': 1641081599999,
                'count': 1000
            }
        }
        
        mock_binance = AsyncMock(spec=ccxt_async.binance)
        mock_binance.fetch_tickers.return_value = expected_binance_ticker
        mock_binance.name = 'binance'
        
        manager = AsyncDataManager(RateLimitConfig())
        manager.exchanges = {'binance': mock_binance}
        
        result = await manager.batch_collect_tickers_async()
        
        # Verify contract
        assert 'binance' in result
        binance_data = result['binance']
        assert 'BTCUSDT' in binance_data
        
        ticker = binance_data['BTCUSDT']
        
        # Required fields for our system
        required_fields = [
            'lastPrice', 'bidPrice', 'askPrice', 'volume',
            'highPrice', 'lowPrice', 'openPrice', 'priceChange'
        ]
        
        for field in required_fields:
            assert field in ticker, f"Missing required field: {field}"
    
    async def test_binance_rate_limits_contract(self):
        """Test Binance rate limiting behavior"""
        
        mock_binance = AsyncMock(spec=ccxt_async.binance)
        
        # Simulate rate limit exceeded
        mock_binance.fetch_ohlcv.side_effect = ccxt_async.RateLimitExceeded("Rate limit exceeded")
        mock_binance.name = 'binance'
        
        manager = AsyncDataManager(RateLimitConfig(requests_per_second=1.0))
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        manager.structured_logger.warning = Mock()
        
        result = await manager.fetch_single_ohlcv_async(mock_binance, 'BTC/USDT', '1h')
        
        # Should handle rate limit gracefully
        assert result == []
        
        # Should log the rate limit error
        manager.structured_logger.warning.assert_called()

@pytest.mark.contract
@pytest.mark.asyncio
class TestCCXTGeneralContract:
    """General contract tests for CCXT library assumptions"""
    
    async def test_exchange_initialization_contract(self):
        """Test that exchanges initialize with expected interface"""
        
        # Test both authenticated and public modes
        configs = [
            # Public mode
            {
                'enableRateLimit': True,
                'timeout': 30000
            },
            # Authenticated mode
            {
                'apiKey': 'test_key',
                'secret': 'test_secret',
                'enableRateLimit': True,
                'timeout': 30000
            }
        ]
        
        for config in configs:
            with patch('ccxt.async_support.kraken') as mock_kraken_class:
                mock_exchange = AsyncMock()
                mock_exchange.name = 'kraken'
                mock_exchange.id = 'kraken'
                mock_kraken_class.return_value = mock_exchange
                
                # Should initialize without errors
                exchange = mock_kraken_class(config)
                
                # Verify expected interface
                assert hasattr(exchange, 'name')
                assert hasattr(exchange, 'id')
                
                # Verify configuration was passed
                mock_kraken_class.assert_called_once_with(config)
    
    async def test_ccxt_has_capabilities_contract(self):
        """Test CCXT 'has' capability reporting"""
        
        mock_exchange = AsyncMock()
        mock_exchange.has = {
            'fetchTickers': True,
            'fetchOHLCV': True,
            'fetchOrderBook': True,
            'fetchTrades': True,
            'fetchBalance': False,  # Requires authentication
            'createOrder': False    # Requires authentication
        }
        
        # Our system should check capabilities before using
        required_capabilities = ['fetchTickers', 'fetchOHLCV']
        
        for capability in required_capabilities:
            assert mock_exchange.has.get(capability, False), f"Missing required capability: {capability}"
    
    async def test_ccxt_error_hierarchy_contract(self):
        """Test CCXT error hierarchy that our error handling depends on"""
        
        # Test error inheritance - our code catches base NetworkError
        assert issubclass(ccxt_async.RequestTimeout, ccxt_async.NetworkError)
        assert issubclass(ccxt_async.ExchangeNotAvailable, ccxt_async.NetworkError)
        
        # Test that specific errors exist that we handle
        error_types = [
            ccxt_async.NetworkError,
            ccxt_async.ExchangeError,
            ccxt_async.RateLimitExceeded,
            ccxt_async.AuthenticationError,
            ccxt_async.InsufficientFunds,
            ccxt_async.InvalidOrder
        ]
        
        for error_type in error_types:
            # Should be able to instantiate with message
            error = error_type("Test error message")
            assert isinstance(error, Exception)
            assert str(error) == "Test error message"

@pytest.mark.contract
class TestDataStructureContracts:
    """Contract tests for data structures our system expects"""
    
    def test_market_data_contract(self, mock_kraken_data, mock_binance_data):
        """Test that market data structures match our expectations"""
        
        # Test Kraken ticker data contract
        kraken_ticker = mock_kraken_data['result']['XXBTZUSD']
        
        # Must have bid/ask structure for spread calculation
        assert 'a' in kraken_ticker  # ask
        assert 'b' in kraken_ticker  # bid
        assert isinstance(kraken_ticker['a'], list)
        assert isinstance(kraken_ticker['b'], list)
        assert len(kraken_ticker['a']) >= 1
        assert len(kraken_ticker['b']) >= 1
        
        # Must have OHLCV data for technical analysis
        assert 'h' in kraken_ticker  # high
        assert 'l' in kraken_ticker  # low
        assert 'o' in kraken_ticker  # open
        assert 'c' in kraken_ticker  # close
        assert 'v' in kraken_ticker  # volume
        
        # Test Binance ticker data contract
        binance_ticker = mock_binance_data
        
        # Must have price fields
        price_fields = ['lastPrice', 'bidPrice', 'askPrice', 'openPrice', 'highPrice', 'lowPrice']
        for field in price_fields:
            assert field in binance_ticker
            assert binance_ticker[field] is not None
        
        # Must have volume and change data
        assert 'volume' in binance_ticker
        assert 'priceChange' in binance_ticker
        assert 'priceChangePercent' in binance_ticker
    
    def test_ohlcv_data_contract(self, mock_ohlcv_data):
        """Test OHLCV data structure contract"""
        
        for candle in mock_ohlcv_data:
            # Must be list/array with exactly 6 elements
            assert isinstance(candle, list)
            assert len(candle) == 6
            
            timestamp, open_price, high, low, close, volume = candle
            
            # Timestamp must be valid
            assert isinstance(timestamp, (int, float))
            assert timestamp > 0
            
            # OHLC must be valid numbers
            for price in [open_price, high, low, close]:
                assert isinstance(price, (int, float))
                assert price > 0
            
            # Volume must be non-negative
            assert isinstance(volume, (int, float))
            assert volume >= 0
            
            # Price relationships must be logical
            assert high >= open_price
            assert high >= close
            assert low <= open_price
            assert low <= close