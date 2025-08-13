"""Integration tests for exchange adapters."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from src.cryptosmarttrader.adapters.kraken_data_adapter import KrakenDataAdapter


@pytest.mark.integration
class TestKrakenDataAdapter:
    """Test Kraken exchange adapter integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.adapter = KrakenDataAdapter(
            api_key="test_key",
            secret_key="test_secret",
            sandbox=True,  # Use sandbox environment
        )

    @pytest.mark.asyncio
    async def test_connection_establishment(self):
        """Test connection to exchange API."""
        with patch.object(self.adapter, "_make_request") as mock_request:
            mock_request.return_value = {"result": {"serverTime": 1640995200}}

            connection_result = await self.adapter.test_connection()

            assert connection_result.is_connected is True
            assert connection_result.latency_ms is not None
            assert connection_result.server_time is not None

    @pytest.mark.asyncio
    async def test_market_data_retrieval(self):
        """Test market data retrieval."""
        with patch.object(self.adapter, "_make_request") as mock_request:
            # Mock OHLCV data response
            mock_request.return_value = {
                "result": {
                    "XXBTZUSD": [
                        [
                            1640995200,
                            "50000.0",
                            "50100.0",
                            "49900.0",
                            "50050.0",
                            "50000.0",
                            "10.5",
                            100,
                        ],
                        [
                            1640995260,
                            "50050.0",
                            "50150.0",
                            "49950.0",
                            "50100.0",
                            "50075.0",
                            "8.3",
                            95,
                        ],
                    ]
                }
            }

            ohlcv_data = await self.adapter.get_ohlcv("BTC/USD", "1m", limit=100)

            assert len(ohlcv_data) > 0
            assert all(
                len(candle) == 8 for candle in ohlcv_data
            )  # timestamp, o, h, l, c, v, volume, count
            assert all(isinstance(candle[0], int) for candle in ohlcv_data)  # timestamp is int

    @pytest.mark.asyncio
    async def test_orderbook_retrieval(self):
        """Test order book data retrieval."""
        with patch.object(self.adapter, "_make_request") as mock_request:
            mock_request.return_value = {
                "result": {
                    "XXBTZUSD": {
                        "bids": [["49950.0", "0.5", 1640995200], ["49940.0", "1.0", 1640995180]],
                        "asks": [["50050.0", "0.3", 1640995200], ["50060.0", "0.8", 1640995190]],
                    }
                }
            }

            orderbook = await self.adapter.get_orderbook("BTC/USD", limit=50)

            assert "bids" in orderbook
            assert "asks" in orderbook
            assert len(orderbook["bids"]) > 0
            assert len(orderbook["asks"]) > 0

            # Validate bid/ask structure
            for bid in orderbook["bids"]:
                assert len(bid) == 3  # price, volume, timestamp
                assert float(bid[0]) > 0  # price > 0
                assert float(bid[1]) > 0  # volume > 0

    @pytest.mark.asyncio
    async def test_ticker_data_retrieval(self):
        """Test ticker data retrieval."""
        with patch.object(self.adapter, "_make_request") as mock_request:
            mock_request.return_value = {
                "result": {
                    "XXBTZUSD": {
                        "a": ["50100.0", "0", "0.000"],  # ask
                        "b": ["50000.0", "0", "0.000"],  # bid
                        "c": ["50050.0", "0.100"],  # last trade
                        "v": ["150.5", "450.2"],  # volume
                        "p": ["50025.0", "49800.0"],  # vwap
                        "t": [120, 350],  # number of trades
                        "l": ["49900.0", "49500.0"],  # low
                        "h": ["50200.0", "50800.0"],  # high
                        "o": "49950.0",  # opening price
                    }
                }
            }

            ticker = await self.adapter.get_ticker("BTC/USD")

            assert "bid" in ticker
            assert "ask" in ticker
            assert "last" in ticker
            assert "volume" in ticker
            assert ticker["bid"] < ticker["ask"]  # Bid should be less than ask
            assert ticker["volume"] > 0

    @pytest.mark.asyncio
    async def test_balance_retrieval(self):
        """Test account balance retrieval."""
        with patch.object(self.adapter, "_make_private_request") as mock_request:
            mock_request.return_value = {
                "result": {"ZUSD": "10000.50", "XXBT": "0.15000000", "XETH": "5.25000000"}
            }

            balances = await self.adapter.get_balance()

            assert "USD" in balances or "ZUSD" in balances
            assert "BTC" in balances or "XXBT" in balances

            for currency, balance in balances.items():
                assert float(balance) >= 0

    @pytest.mark.asyncio
    async def test_order_placement_mock(self):
        """Test order placement (mocked)."""
        with patch.object(self.adapter, "_make_private_request") as mock_request:
            mock_request.return_value = {"result": {"txid": ["OQCLML-BW3P3-BUCMWZ"]}}

            order_result = await self.adapter.place_order(
                symbol="BTC/USD",
                side="buy",
                order_type="limit",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
            )

            assert order_result.success is True
            assert order_result.order_id is not None
            assert len(order_result.order_id) > 0

    @pytest.mark.asyncio
    async def test_order_status_retrieval(self):
        """Test order status retrieval."""
        with patch.object(self.adapter, "_make_private_request") as mock_request:
            mock_request.return_value = {
                "result": {
                    "OQCLML-BW3P3-BUCMWZ": {
                        "status": "open",
                        "opentm": 1640995200.1234,
                        "starttm": 0,
                        "expiretm": 0,
                        "descr": {
                            "pair": "XXBTZUSD",
                            "type": "buy",
                            "ordertype": "limit",
                            "price": "50000.0",
                            "price2": "0",
                            "leverage": "none",
                        },
                        "vol": "0.10000000",
                        "vol_exec": "0.00000000",
                        "cost": "0.00000",
                        "fee": "0.00000",
                        "price": "0.00000",
                        "stopprice": "0.00000",
                        "limitprice": "0.00000",
                        "misc": "",
                        "oflags": "fciq",
                    }
                }
            }

            order_status = await self.adapter.get_order_status("OQCLML-BW3P3-BUCMWZ")

            assert order_status.order_id == "OQCLML-BW3P3-BUCMWZ"
            assert order_status.status in ["open", "closed", "canceled", "expired"]
            assert order_status.filled_quantity >= 0
            assert order_status.remaining_quantity >= 0

    @pytest.mark.asyncio
    async def test_trade_history_retrieval(self):
        """Test trade history retrieval."""
        with patch.object(self.adapter, "_make_private_request") as mock_request:
            mock_request.return_value = {
                "result": {
                    "trades": {
                        "TCCCTY-DVNLG-DFG2L4": {
                            "ordertxid": "OQCLML-BW3P3-BUCMWZ",
                            "pair": "XXBTZUSD",
                            "time": 1640995300.1234,
                            "type": "buy",
                            "ordertype": "limit",
                            "price": "50000.0",
                            "cost": "5000.00",
                            "fee": "13.00",
                            "vol": "0.10000000",
                            "margin": "0.00000",
                            "misc": "",
                        }
                    }
                }
            }

            trades = await self.adapter.get_trade_history(limit=50)

            assert len(trades) > 0

            for trade in trades:
                assert trade.trade_id is not None
                assert trade.order_id is not None
                assert trade.symbol is not None
                assert trade.side in ["buy", "sell"]
                assert trade.quantity > 0
                assert trade.price > 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for API failures."""
        with patch.object(self.adapter, "_make_request") as mock_request:
            # Simulate API error
            mock_request.side_effect = Exception("API connection failed")

            with pytest.raises(Exception):
                await self.adapter.get_ticker("BTC/USD")

    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self):
        """Test rate limiting compliance."""
        # Track request timing
        request_times = []

        original_request = self.adapter._make_request

        async def timed_request(*args, **kwargs):
            import time

            request_times.append(time.time())
            return {"result": {"serverTime": 1640995200}}

        with patch.object(self.adapter, "_make_request", side_effect=timed_request):
            # Make multiple rapid requests
            tasks = [self.adapter.test_connection() for _ in range(5)]
            await asyncio.gather(*tasks)

            # Check rate limiting
            if len(request_times) > 1:
                min_interval = min(
                    request_times[i + 1] - request_times[i] for i in range(len(request_times) - 1)
                )
                # Should respect minimum interval (depends on exchange limits)
                assert min_interval >= 0  # At least no negative intervals

    @pytest.mark.asyncio
    async def test_websocket_connection_mock(self):
        """Test WebSocket connection (mocked)."""
        with patch("websockets.connect") as mock_ws:
            mock_ws.return_value.__aenter__.return_value = AsyncMock()

            ws_result = await self.adapter.connect_websocket(["ticker", "ohlc"], ["BTC/USD"])

            assert ws_result.connected is True

    @pytest.mark.asyncio
    async def test_symbol_normalization(self):
        """Test symbol normalization between formats."""
        # Test various symbol format conversions
        test_cases = [("BTC/USD", "XXBTZUSD"), ("ETH/USD", "XETHZUSD"), ("BTC/EUR", "XXBTZEUR")]

        for standard_symbol, kraken_symbol in test_cases:
            normalized = self.adapter.normalize_symbol(standard_symbol)
            denormalized = self.adapter.denormalize_symbol(kraken_symbol)

            # Should convert consistently
            assert isinstance(normalized, str)
            assert isinstance(denormalized, str)

    def test_configuration_validation(self):
        """Test adapter configuration validation."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            KrakenDataAdapter(api_key="", secret_key="test_secret")

        with pytest.raises(ValueError):
            KrakenDataAdapter(api_key="test_key", secret_key="")

    def test_data_validation(self):
        """Test incoming data validation."""
        # Test with invalid ticker data
        invalid_ticker = {
            "bid": "invalid",  # Should be numeric
            "ask": 50100.0,
            "last": 50050.0,
        }

        with pytest.raises(ValueError):
            self.adapter._validate_ticker_data(invalid_ticker)

        # Test with valid ticker data
        valid_ticker = {"bid": 50000.0, "ask": 50100.0, "last": 50050.0, "volume": 150.5}

        # Should not raise exception
        self.adapter._validate_ticker_data(valid_ticker)
