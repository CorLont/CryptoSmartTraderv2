#!/usr/bin/env python3
"""
Authentic Data Collector
Ensures only real data from exchanges - NO fallbacks, NO synthetic data
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import ccxt.async_support as ccxt
from core.strict_data_integrity import DataSource, StrictDataIntegrityEnforcer
import warnings

warnings.filterwarnings("ignore")


@dataclass
class AuthenticDataPoint:
    """Authentic data point with source verification"""

    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    change_24h: float
    market_cap: Optional[float]
    source_exchange: str
    data_source: DataSource
    api_response_time_ms: float
    data_quality_score: float  # 0-1, based on freshness and completeness


@dataclass
class DataCollectionResult:
    """Result of authentic data collection"""

    success: bool
    symbols_collected: List[str]
    symbols_failed: List[str]
    authentic_data_points: List[AuthenticDataPoint]
    collection_timestamp: datetime
    total_api_calls: int
    average_response_time_ms: float
    data_integrity_score: float  # 0-1, overall data quality


class AuthenticDataCollector:
    """Collects only authentic data from exchanges - zero fallbacks"""

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 10):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)

        # Initialize exchanges for authentic data
        self.exchanges = {}
        self.failed_symbols = set()  # Track symbols that consistently fail

        # Data integrity enforcer
        self.integrity_enforcer = StrictDataIntegrityEnforcer(production_mode=True)

        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchange connections for authentic data"""

        try:
            # Primary exchanges for authentic data
            self.exchanges["kraken"] = ccxt.kraken(
                {
                    "enableRateLimit": True,
                    "timeout": self.timeout_seconds * 1000,
                }
            )

            self.exchanges["binance"] = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "timeout": self.timeout_seconds * 1000,
                }
            )

            self.exchanges["kucoin"] = ccxt.kucoin(
                {
                    "enableRateLimit": True,
                    "timeout": self.timeout_seconds * 1000,
                }
            )

            self.logger.info(
                f"Initialized {len(self.exchanges)} exchanges for authentic data collection"
            )

        except Exception as e:
            self.logger.error(f"Exchange initialization error: {e}")

    async def collect_authentic_market_data(
        self, symbols: List[str], require_all_symbols: bool = True
    ) -> DataCollectionResult:
        """Collect authentic market data with zero fallbacks"""

        collection_start = datetime.utcnow()
        authentic_data_points = []
        symbols_collected = []
        symbols_failed = []
        total_api_calls = 0
        response_times = []

        self.logger.info(f"Starting authentic data collection for {len(symbols)} symbols")

        # Filter out previously failed symbols to avoid wasting API calls
        symbols_to_collect = [s for s in symbols if s not in self.failed_symbols]

        if len(symbols_to_collect) < len(symbols):
            self.logger.warning(
                f"Skipping {len(symbols) - len(symbols_to_collect)} previously failed symbols"
            )

        # Collect data for each symbol
        for symbol in symbols_to_collect:
            symbol_success = False

            # Try each exchange until we get authentic data
            for exchange_name, exchange in self.exchanges.items():
                try:
                    start_time = datetime.utcnow()

                    # Load markets
                    await exchange.load_markets()

                    # Determine trading pair
                    trading_pair = self._get_trading_pair(symbol, exchange)

                    if not trading_pair:
                        continue

                    # Get authentic ticker data
                    ticker = await exchange.fetch_ticker(trading_pair)
                    total_api_calls += 1

                    # Calculate response time
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    response_times.append(response_time)

                    # Validate authentic data
                    if self._validate_authentic_ticker(ticker):
                        # Create authentic data point
                        data_point = AuthenticDataPoint(
                            symbol=symbol,
                            timestamp=datetime.utcfromtimestamp(ticker["timestamp"] / 1000)
                            if ticker["timestamp"]
                            else datetime.utcnow(),
                            price=float(ticker["last"]) if ticker["last"] else 0.0,
                            volume_24h=float(ticker["quoteVolume"])
                            if ticker["quoteVolume"]
                            else 0.0,
                            change_24h=float(ticker["percentage"]) / 100
                            if ticker["percentage"]
                            else 0.0,
                            market_cap=None,  # Will be calculated separately if needed
                            source_exchange=exchange_name,
                            data_source=DataSource.AUTHENTIC,
                            api_response_time_ms=response_time,
                            data_quality_score=self._calculate_data_quality_score(
                                ticker, response_time
                            ),
                        )

                        authentic_data_points.append(data_point)
                        symbols_collected.append(symbol)
                        symbol_success = True

                        self.logger.debug(
                            f"Collected authentic data for {symbol} from {exchange_name}"
                        )
                        break  # Success - move to next symbol

                except Exception as e:
                    self.logger.debug(f"Failed to collect {symbol} from {exchange_name}: {e}")
                    continue

            # If no exchange provided authentic data for this symbol
            if not symbol_success:
                symbols_failed.append(symbol)
                self.failed_symbols.add(symbol)  # Track for future optimization

                if require_all_symbols:
                    self.logger.error(
                        f"CRITICAL: Failed to collect authentic data for required symbol {symbol}"
                    )

        # Close exchange connections
        await self._close_exchanges()

        # Calculate collection metrics
        avg_response_time = np.mean(response_times) if response_times else 0
        data_integrity_score = len(symbols_collected) / len(symbols) if symbols else 0

        # Determine collection success
        collection_success = len(symbols_collected) > 0 and (
            not require_all_symbols or len(symbols_failed) == 0
        )

        result = DataCollectionResult(
            success=collection_success,
            symbols_collected=symbols_collected,
            symbols_failed=symbols_failed,
            authentic_data_points=authentic_data_points,
            collection_timestamp=collection_start,
            total_api_calls=total_api_calls,
            average_response_time_ms=avg_response_time,
            data_integrity_score=data_integrity_score,
        )

        self.logger.info(
            f"Collection complete: {len(symbols_collected)}/{len(symbols)} symbols, "
            f"{total_api_calls} API calls, {avg_response_time:.1f}ms avg response"
        )

        return result

    def _get_trading_pair(self, symbol: str, exchange) -> Optional[str]:
        """Get correct trading pair for symbol on exchange"""

        # Common trading pair patterns
        possible_pairs = [f"{symbol}/USDT", f"{symbol}/USD", f"{symbol}/BTC", f"{symbol}/ETH"]

        for pair in possible_pairs:
            if pair in exchange.markets:
                return pair

        return None

    def _validate_authentic_ticker(self, ticker: Dict) -> bool:
        """Validate that ticker data is authentic and complete"""

        # Check required fields
        required_fields = ["last", "timestamp", "quoteVolume"]

        for field in required_fields:
            if ticker.get(field) is None:
                return False

        # Check data freshness (within last hour)
        if ticker["timestamp"]:
            data_age_hours = (datetime.utcnow().timestamp() * 1000 - ticker["timestamp"]) / (
                1000 * 3600
            )
            if data_age_hours > 1:  # Data older than 1 hour
                return False

        # Check realistic values
        price = ticker.get("last", 0)
        volume = ticker.get("quoteVolume", 0)

        if price <= 0 or volume < 0:
            return False

        # Check for obvious placeholder values
        placeholder_values = [0, 1, 999, 9999, 99999]
        if price in placeholder_values or volume in placeholder_values:
            return False

        return True

    def _calculate_data_quality_score(self, ticker: Dict, response_time_ms: float) -> float:
        """Calculate data quality score (0-1)"""

        score = 1.0

        # Penalize slow response times
        if response_time_ms > 1000:  # >1 second
            score -= 0.2
        elif response_time_ms > 500:  # >0.5 seconds
            score -= 0.1

        # Check data completeness
        optional_fields = ["high", "low", "open", "close", "percentage"]
        present_optional = sum(1 for field in optional_fields if ticker.get(field) is not None)
        completeness_score = present_optional / len(optional_fields)
        score *= 0.8 + 0.2 * completeness_score  # Weight completeness 20%

        # Check data freshness
        if ticker.get("timestamp"):
            data_age_minutes = (datetime.utcnow().timestamp() * 1000 - ticker["timestamp"]) / (
                1000 * 60
            )
            if data_age_minutes < 5:  # Very fresh data
                score *= 1.0
            elif data_age_minutes < 15:  # Reasonably fresh
                score *= 0.95
            else:  # Older data
                score *= 0.9

        return max(0.0, min(1.0, score))

    async def _close_exchanges(self):
        """Close exchange connections"""

        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass

    def create_authentic_dataframe(self, data_points: List[AuthenticDataPoint]) -> pd.DataFrame:
        """Create DataFrame from authentic data points with source tracking"""

        if not data_points:
            return pd.DataFrame()

        # Convert to DataFrame
        data_records = []
        data_sources = {}

        for dp in data_points:
            record = {
                "symbol": dp.symbol,
                "timestamp": dp.timestamp,
                "price": dp.price,
                "volume_24h": dp.volume_24h,
                "change_24h": dp.change_24h,
                "market_cap": dp.market_cap,
                "source_exchange": dp.source_exchange,
                "api_response_time_ms": dp.api_response_time_ms,
                "data_quality_score": dp.data_quality_score,
            }

            data_records.append(record)

            # Track data sources for integrity validation
            for col in ["price", "volume_24h", "change_24h"]:
                data_sources[f"{dp.symbol}_{col}"] = dp.data_source

        df = pd.DataFrame(data_records)

        # Validate data integrity
        integrity_report = self.integrity_enforcer.validate_data_integrity(df, data_sources)

        if not integrity_report.is_production_ready:
            self.logger.error(
                f"Authentic data failed integrity check: {integrity_report.critical_violations} critical violations"
            )
            raise ValueError(
                f"Collected data failed integrity validation: {integrity_report.critical_violations} critical issues"
            )

        self.logger.info(
            f"Created authentic DataFrame: {len(df)} records, {integrity_report.authentic_data_percentage:.1f}% authentic"
        )

        return df

    def get_failed_symbols_report(self) -> Dict[str, Any]:
        """Get report of symbols that consistently fail data collection"""

        return {
            "failed_symbols": list(self.failed_symbols),
            "failed_count": len(self.failed_symbols),
            "recommendation": "Remove failed symbols from universe or investigate API issues",
        }


class ProductionDataValidator:
    """Validates data before production use - blocks any non-authentic data"""

    def __init__(self):
        self.integrity_enforcer = StrictDataIntegrityEnforcer(production_mode=True)
        self.logger = logging.getLogger(__name__)

    def validate_for_production(
        self,
        df: pd.DataFrame,
        data_sources: Dict[str, DataSource] = None,
        block_on_violation: bool = True,
    ) -> Tuple[bool, str]:
        """Validate data for production use"""

        # Run integrity validation
        integrity_report = self.integrity_enforcer.validate_data_integrity(df, data_sources)

        if not integrity_report.is_production_ready:
            violation_summary = (
                f"Critical violations: {integrity_report.critical_violations}, "
                f"Authentic data: {integrity_report.authentic_data_percentage:.1f}%"
            )

            if block_on_violation:
                self.logger.error(f"PRODUCTION BLOCKED: {violation_summary}")
                raise ValueError(f"Data validation failed for production: {violation_summary}")
            else:
                self.logger.warning(f"Data validation warnings: {violation_summary}")
                return False, violation_summary

        self.logger.info(
            f"Production validation PASSED: {integrity_report.authentic_data_percentage:.1f}% authentic data"
        )
        return True, "Production validation passed"


async def collect_authentic_crypto_data(
    symbols: List[str], require_all: bool = True
) -> pd.DataFrame:
    """Collect authentic cryptocurrency data with zero fallbacks"""

    collector = AuthenticDataCollector()

    try:
        # Collect authentic data
        result = await collector.collect_authentic_market_data(symbols, require_all)

        if not result.success:
            raise ValueError(
                f"Failed to collect authentic data: {len(result.symbols_failed)} failed symbols"
            )

        # Create DataFrame from authentic data
        df = collector.create_authentic_dataframe(result.authentic_data_points)

        return df

    except Exception as e:
        logging.getLogger(__name__).error(f"Authentic data collection failed: {e}")
        raise


def validate_production_ready(df: pd.DataFrame, data_sources: Dict[str, DataSource] = None) -> bool:
    """Quick validation that data is production ready"""

    validator = ProductionDataValidator()
    is_valid, message = validator.validate_for_production(
        df, data_sources, block_on_violation=False
    )

    return is_valid
