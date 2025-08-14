import ccxt
import threading
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

# MANDATORY EXECUTION DISCIPLINE IMPORTS
try:
    from src.cryptosmarttrader.execution.execution_discipline import (
        ExecutionPolicy, OrderRequest, MarketConditions, OrderSide, TimeInForce
    )
    from src.cryptosmarttrader.execution.mandatory_enforcement import (
        DisciplinedExchangeManager, get_global_execution_policy
    )
    EXECUTION_DISCIPLINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  ExecutionDiscipline not available: {e}")
    EXECUTION_DISCIPLINE_AVAILABLE = False


class ExchangeManager:
    """Enhanced exchange manager for multi-exchange connectivity"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Exchange connections
        self.exchanges = {}
        self.exchange_status = {}
        self._lock = threading.Lock()

        # Rate limiting
        self.rate_limiters = {}
        self.last_request_times = {}

        # Initialize exchanges
        self._initialize_exchanges()

        # Start health monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_exchanges, daemon=True)
        self.monitor_thread.start()

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        enabled_exchanges = self.config_manager.get("exchanges", ["kraken"])
        api_keys = self.config_manager.get("api_keys", {})

        exchange_classes = {
            "kraken": ccxt.kraken,
            "binance": ccxt.binance,
            "kucoin": ccxt.kucoin,
            "huobi": ccxt.huobi,
            "coinbase": ccxt.coinbasepro,
            "bitfinex": ccxt.bitfinex,
        }

        for exchange_name in enabled_exchanges:
            try:
                exchange_class = exchange_classes.get(exchange_name)
                if not exchange_class:
                    self.logger.warning(f"Unknown exchange: {exchange_name}")
                    continue

                # Get API credentials
                api_key = api_keys.get(exchange_name, "")
                secret = api_keys.get(f"{exchange_name}_secret", "")
                passphrase = api_keys.get(f"{exchange_name}_passphrase", "")

                # Exchange configuration
                config = {
                    "apiKey": api_key,
                    "secret": secret,
                    "timeout": self.config_manager.get("timeout_seconds", 30) * 1000,
                    "rateLimit": 60000 / self.config_manager.get("api_rate_limit", 100),
                    "enableRateLimit": True,
                    "sandbox": False,
                }

                # Add passphrase for exchanges that need it
                if passphrase and exchange_name in ["kucoin", "coinbase"]:
                    config["passphrase"] = passphrase

                # Initialize exchange
                exchange = exchange_class(config)

                # Test connection
                if self._test_exchange_connection(exchange):
                    self.exchanges[exchange_name] = exchange
                    self.exchange_status[exchange_name] = {
                        "status": "connected",
                        "last_check": datetime.now().isoformat(),
                        "error_count": 0,
                        "success_count": 0,
                    }
                    self.logger.info(f"Successfully initialized {exchange_name}")
                else:
                    self.exchange_status[exchange_name] = {
                        "status": "failed",
                        "last_check": datetime.now().isoformat(),
                        "error_count": 1,
                        "success_count": 0,
                    }
                    self.logger.error(f"Failed to connect to {exchange_name}")

            except Exception as e:
                self.logger.error(f"Error initializing {exchange_name}: {str(e)}")
                self.exchange_status[exchange_name] = {
                    "status": "error",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e),
                    "error_count": 1,
                    "success_count": 0,
                }

    def _test_exchange_connection(self, exchange) -> bool:
        """Test exchange connection"""
        try:
            # Try to fetch exchange info or markets
            markets = exchange.load_markets()
            return len(markets) > 0
        except Exception as e:
            self.logger.error(f"Exchange connection test failed: {str(e)}")
            return False

    def _monitor_exchanges(self):
        """Monitor exchange health and connectivity"""
        while self.monitoring_active:
            try:
                for exchange_name, exchange in list(self.exchanges.items()):
                    try:
                        # Test exchange connectivity
                        if self._test_exchange_connection(exchange):
                            with self._lock:
                                self.exchange_status[exchange_name]["status"] = "connected"
                                self.exchange_status[exchange_name]["last_check"] = (
                                    datetime.now().isoformat()
                                )
                                self.exchange_status[exchange_name]["success_count"] += 1
                        else:
                            with self._lock:
                                self.exchange_status[exchange_name]["status"] = "disconnected"
                                self.exchange_status[exchange_name]["error_count"] += 1

                    except Exception as e:
                        with self._lock:
                            self.exchange_status[exchange_name]["status"] = "error"
                            self.exchange_status[exchange_name]["error"] = str(e)
                            self.exchange_status[exchange_name]["error_count"] += 1

                # Sleep for 5 minutes between checks
                time.sleep(300)

            except Exception as e:
                self.logger.error(f"Exchange monitoring error: {str(e)}")
                time.sleep(60)

    def _check_rate_limit(self, exchange_name: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        rate_limit = self.config_manager.get("api_rate_limit", 100)  # requests per minute
        min_interval = 60 / rate_limit  # minimum seconds between requests

        if exchange_name in self.last_request_times:
            time_since_last = current_time - self.last_request_times[exchange_name]
            if time_since_last < min_interval:
                return False

        self.last_request_times[exchange_name] = current_time
        return True

    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all exchanges"""
        with self._lock:
            return self.exchange_status.copy()

    def get_available_exchanges(self) -> List[str]:
        """Get list of available and connected exchanges"""
        with self._lock:
            return [
                name
                for name, status in self.exchange_status.items()
                if status.get("status") == "connected"
            ]

    def fetch_ticker(self, symbol: str, exchange_name: str = None) -> Optional[Dict[str, Any]]:
        """Fetch ticker data from exchange"""
        try:
            if exchange_name:
                exchanges_to_try = [exchange_name] if exchange_name in self.exchanges else []
            else:
                exchanges_to_try = self.get_available_exchanges()

            for exch_name in exchanges_to_try:
                try:
                    if not self._check_rate_limit(exch_name):
                        continue

                    exchange = self.exchanges[exch_name]
                    ticker = exchange.fetch_ticker(symbol)

                    # Add exchange info to ticker
                    ticker["exchange"] = exch_name
                    ticker["timestamp_fetched"] = datetime.now().isoformat()

                    return ticker

                except Exception as e:
                    self.logger.error(f"Error fetching ticker from {exch_name}: {str(e)}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Ticker fetch error: {str(e)}")
            return None

    def fetch_tickers(self, exchange_name: str = None) -> Dict[str, Any]:
        """Fetch all tickers from exchange"""
        try:
            if exchange_name:
                exchanges_to_try = [exchange_name] if exchange_name in self.exchanges else []
            else:
                exchanges_to_try = self.get_available_exchanges()

            all_tickers = {}

            for exch_name in exchanges_to_try:
                try:
                    if not self._check_rate_limit(exch_name):
                        continue

                    exchange = self.exchanges[exch_name]
                    tickers = exchange.fetch_tickers()

                    # Add exchange info to each ticker
                    for symbol, ticker in tickers.items():
                        ticker["exchange"] = exch_name
                        ticker["timestamp_fetched"] = datetime.now().isoformat()

                        # Use exchange-specific key to avoid conflicts
                        key = f"{symbol}_{exch_name}"
                        all_tickers[key] = ticker

                    # For single exchange, just return the tickers
                    if exchange_name:
                        return tickers

                except Exception as e:
                    self.logger.error(f"Error fetching tickers from {exch_name}: {str(e)}")
                    continue

            return all_tickers

        except Exception as e:
            self.logger.error(f"Tickers fetch error: {str(e)}")
            return {}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 100, exchange_name: str = None
    ) -> Optional[List[List]]:
        """Fetch OHLCV data from exchange"""
        try:
            if exchange_name:
                exchanges_to_try = [exchange_name] if exchange_name in self.exchanges else []
            else:
                exchanges_to_try = self.get_available_exchanges()

            for exch_name in exchanges_to_try:
                try:
                    if not self._check_rate_limit(exch_name):
                        continue

                    exchange = self.exchanges[exch_name]

                    # Check if exchange supports OHLCV
                    if not exchange.has["fetchOHLCV"]:
                        continue

                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    return ohlcv

                except Exception as e:
                    self.logger.error(f"Error fetching OHLCV from {exch_name}: {str(e)}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"OHLCV fetch error: {str(e)}")
            return None

    def fetch_order_book(
        self, symbol: str, limit: int = 100, exchange_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch order book from exchange"""
        try:
            if exchange_name:
                exchanges_to_try = [exchange_name] if exchange_name in self.exchanges else []
            else:
                exchanges_to_try = self.get_available_exchanges()

            for exch_name in exchanges_to_try:
                try:
                    if not self._check_rate_limit(exch_name):
                        continue

                    exchange = self.exchanges[exch_name]
                    order_book = exchange.fetch_order_book(symbol, limit)

                    # Add exchange info
                    order_book["exchange"] = exch_name
                    order_book["timestamp_fetched"] = datetime.now().isoformat()

                    return order_book

                except Exception as e:
                    self.logger.error(f"Error fetching order book from {exch_name}: {str(e)}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Order book fetch error: {str(e)}")
            return None

    def get_exchange_markets(self, exchange_name: str) -> Dict[str, Any]:
        """Get markets for a specific exchange"""
        try:
            if exchange_name not in self.exchanges:
                return {}

            exchange = self.exchanges[exchange_name]
            return exchange.markets

        except Exception as e:
            self.logger.error(f"Error getting markets for {exchange_name}: {str(e)}")
            return {}

    def get_all_symbols(self) -> List[str]:
        """Get all available trading symbols across exchanges"""
        all_symbols = set()

        for exchange_name in self.get_available_exchanges():
            try:
                markets = self.get_exchange_markets(exchange_name)
                all_symbols.update(markets.keys())
            except Exception as e:
                self.logger.error(f"Error getting symbols from {exchange_name}: {str(e)}")

        return sorted(list(all_symbols))

    def get_exchange_info(self, exchange_name: str) -> Dict[str, Any]:
        """Get detailed exchange information"""
        if exchange_name not in self.exchanges:
            return {}

        try:
            exchange = self.exchanges[exchange_name]

            return {
                "id": exchange.id,
                "name": exchange.name,
                "countries": getattr(exchange, "countries", []),
                "urls": getattr(exchange, "urls", {}),
                "api": getattr(exchange, "api", {}),
                "has": getattr(exchange, "has", {}),
                "timeframes": getattr(exchange, "timeframes", {}),
                "markets_count": len(exchange.markets) if exchange.markets else 0,
                "status": self.exchange_status.get(exchange_name, {}),
            }

        except Exception as e:
            self.logger.error(f"Error getting exchange info for {exchange_name}: {str(e)}")
            return {"error": str(e)}

    def refresh_exchange_connection(self, exchange_name: str) -> bool:
        """Refresh connection to a specific exchange"""
        try:
            if exchange_name in self.exchanges:
                # Remove old connection
                del self.exchanges[exchange_name]

            # Re-initialize
            self._initialize_exchanges()

            return exchange_name in self.exchanges

        except Exception as e:
            self.logger.error(f"Error refreshing {exchange_name}: {str(e)}")
            return False

    def get_rate_limit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit status for all exchanges"""
        current_time = time.time()
        rate_limit = self.config_manager.get("api_rate_limit", 100)
        min_interval = 60 / rate_limit

        status = {}

        for exchange_name in self.exchanges.keys():
            last_request = self.last_request_times.get(exchange_name, 0)
            time_since_last = current_time - last_request
            can_request = time_since_last >= min_interval

            status[exchange_name] = {
                "last_request": datetime.fromtimestamp(last_request).isoformat()
                if last_request > 0
                else None,
                "time_since_last": time_since_last,
                "can_request": can_request,
                "rate_limit_per_minute": rate_limit,
                "min_interval_seconds": min_interval,
            }

        return status

    def stop_monitoring(self):
        """Stop exchange monitoring"""
        self.monitoring_active = False

    # MANDATORY EXECUTION DISCIPLINE METHODS
    def create_market_conditions(self, symbol: str, exchange_name: str = "kraken") -> Optional[MarketConditions]:
        """Create MarketConditions from real market data"""
        if not EXECUTION_DISCIPLINE_AVAILABLE:
            self.logger.warning("⚠️  ExecutionDiscipline not available - cannot create market conditions")
            return None
            
        try:
            # Get ticker and order book
            ticker = self.fetch_ticker(symbol, exchange_name)
            order_book = self.fetch_order_book(symbol, 100, exchange_name)
            
            if not ticker or not order_book:
                self.logger.error(f"❌ Could not fetch market data for {symbol}")
                return None
            
            # Calculate market conditions
            bid_price = ticker.get('bid', 0)
            ask_price = ticker.get('ask', 0)
            last_price = ticker.get('last', 0)
            
            if bid_price <= 0 or ask_price <= 0 or last_price <= 0:
                self.logger.error(f"❌ Invalid price data for {symbol}")
                return None
            
            # Calculate spread in bps
            spread_bps = ((ask_price - bid_price) / last_price) * 10000
            
            # Calculate depth (sum of top 10 levels)
            bids = order_book.get('bids', [])[:10]
            asks = order_book.get('asks', [])[:10]
            
            bid_depth_usd = sum(bid[0] * bid[1] for bid in bids) if bids else 0
            ask_depth_usd = sum(ask[0] * ask[1] for ask in asks) if asks else 0
            
            # Estimate 1-minute volume (use 24h volume / 1440)
            volume_24h = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0) * last_price
            volume_1m_usd = volume_24h / 1440 if volume_24h > 0 else 0
            
            return MarketConditions(
                spread_bps=spread_bps,
                bid_depth_usd=bid_depth_usd,
                ask_depth_usd=ask_depth_usd,
                volume_1m_usd=volume_1m_usd,
                last_price=last_price,
                bid_price=bid_price,
                ask_price=ask_price,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating market conditions: {e}")
            return None

    def execute_risk_enforced_order(
        self,
        operation_type: str,
        symbol: str,
        side: str,
        size_usd: float,
        limit_price: Optional[float] = None,
        strategy_id: str = "default",
        exchange_name: str = "kraken"
    ) -> Dict[str, Any]:
        """
        🛡️  COMPLETE RISK-ENFORCED EXECUTION PIPELINE
        
        1. CentralRiskGuard evaluation (day-loss, drawdown, exposure, kill-switch)
        2. ExecutionDiscipline gates (spread, depth, volume, slippage)
        3. Exchange execution
        
        This is the PREFERRED method for all trading operations
        """
        
        # Import here to avoid circular dependencies
        try:
            from src.cryptosmarttrader.execution.risk_enforced_execution import get_global_risk_enforced_manager
            
            risk_manager = get_global_risk_enforced_manager(exchange_manager=self)
            
            return risk_manager.execute_trading_operation(
                operation_type=operation_type,
                symbol=symbol,
                side=side,
                size_usd=size_usd,
                limit_price=limit_price,
                strategy_id=strategy_id,
                exchange_name=exchange_name
            )
            
        except ImportError as e:
            self.logger.warning(f"⚠️  RiskEnforcedExecution not available: {e}")
            # Fallback to ExecutionDiscipline only
            return self.execute_disciplined_order(
                symbol=symbol,
                side=side,
                size=size_usd / (limit_price or 50000.0),  # Convert USD to units
                order_type="limit",
                limit_price=limit_price,
                strategy_id=strategy_id,
                exchange_name=exchange_name
            )

    def execute_disciplined_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        strategy_id: str = "default",
        exchange_name: str = "kraken"
    ) -> Dict[str, Any]:
        """
        🛡️  ExecutionDiscipline-only execution (legacy method)
        
        RECOMMENDATION: Use execute_risk_enforced_order() instead
        for complete risk management pipeline
        """
        
        if not EXECUTION_DISCIPLINE_AVAILABLE:
            return {
                "success": False,
                "error": "ExecutionDiscipline not available - cannot place orders safely",
                "order_id": None
            }
        
        try:
            # Create OrderRequest
            order_request = OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                size=size,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force=TimeInForce.POST_ONLY,  # Enforce post-only
                strategy_id=strategy_id
            )
            
            # Get current market conditions
            market_conditions = self.create_market_conditions(symbol, exchange_name)
            if not market_conditions:
                return {
                    "success": False,
                    "error": f"Could not get market conditions for {symbol}",
                    "order_id": None
                }
            
            # MANDATORY ExecutionDiscipline.decide() gate
            execution_policy = get_global_execution_policy()
            result = execution_policy.decide(order_request, market_conditions)
            
            # Check discipline decision
            if result.decision.value == "reject":
                self.logger.warning(f"🚫 ExecutionDiscipline REJECTED: {result.reason}")
                return {
                    "success": False,
                    "error": f"ExecutionDiscipline rejected: {result.reason}",
                    "gate_results": result.gate_results,
                    "order_id": None,
                    "client_order_id": order_request.client_order_id
                }
            
            if result.decision.value == "defer":
                self.logger.info(f"⏳ ExecutionDiscipline DEFERRED: {result.reason}")
                return {
                    "success": False,
                    "error": f"ExecutionDiscipline deferred: {result.reason}",
                    "gate_results": result.gate_results,
                    "order_id": None,
                    "client_order_id": order_request.client_order_id
                }
            
            # Order APPROVED - execute through exchange
            self.logger.info(f"✅ ExecutionDiscipline APPROVED: {order_request.client_order_id}")
            
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return {
                    "success": False,
                    "error": f"Exchange {exchange_name} not available",
                    "gate_results": result.gate_results,
                    "order_id": None
                }
            
            # Prepare order parameters with discipline requirements
            order_params = {
                "timeInForce": order_request.time_in_force.value,
                "clientOrderId": order_request.client_order_id,
                "postOnly": True  # Enforce post-only
            }
            
            # Execute order based on type
            if order_type == "limit" and limit_price:
                exchange_result = exchange.create_limit_order(
                    symbol, side, size, limit_price, order_params
                )
            else:
                # Market orders are generally NOT recommended due to slippage
                self.logger.warning(f"⚠️  Market order requested - consider using limit orders")
                exchange_result = exchange.create_market_order(
                    symbol, side, size, order_params
                )
            
            self.logger.info(f"✅ Order executed successfully: {exchange_result.get('id')}")
            
            return {
                "success": True,
                "exchange_result": exchange_result,
                "gate_results": result.gate_results,
                "order_id": exchange_result.get("id"),
                "client_order_id": order_request.client_order_id,
                "market_conditions": {
                    "spread_bps": market_conditions.spread_bps,
                    "bid_depth_usd": market_conditions.bid_depth_usd,
                    "ask_depth_usd": market_conditions.ask_depth_usd,
                    "volume_1m_usd": market_conditions.volume_1m_usd
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Disciplined order execution failed: {e}")
            return {
                "success": False,
                "error": f"Order execution failed: {str(e)}",
                "order_id": None
            }

    def get_exchange(self, exchange_name: str):
        """Get exchange instance - for use by DisciplinedExchangeManager only"""
        return self.exchanges.get(exchange_name)

# BLOCK ALL DIRECT ORDER METHODS - FORCE THROUGH DISCIPLINE
class OrderExecutionBlocked(Exception):
    """Raised when trying to use blocked order methods"""
    pass

def _block_direct_orders(*args, **kwargs):
    """Block direct order execution - force through ExecutionDiscipline"""
    raise OrderExecutionBlocked(
        "🚫 DIRECT ORDER EXECUTION BLOCKED!\n"
        "Use ExchangeManager.execute_disciplined_order() instead.\n"
        "ALL orders must go through ExecutionDiscipline.decide() gates!"
    )
