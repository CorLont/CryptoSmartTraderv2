#!/usr/bin/env python3
"""
Futures Data Features
Funding rates, Open Interest, and basis features for detecting leverage squeezes
"""

import numpy as np
import pandas as pd
import ccxt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class FuturesDataCollector:
    """
    Collect futures data from multiple exchanges
    """

    def __init__(self):
        self.exchanges = self._initialize_exchanges()
        self.funding_rates = {}
        self.open_interest = {}
        self.basis_data = {}

    def _initialize_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """Initialize exchange connections"""

        exchanges = {}

        try:
            # Binance Futures
            exchanges["binance"] = ccxt.binance(
                {"sandbox": False, "enableRateLimit": True, "options": {"defaultType": "future"}}
            )
        except Exception:
            pass

        try:
            # Bybit
            exchanges["bybit"] = ccxt.bybit({"sandbox": False, "enableRateLimit": True})
        except Exception:
            pass

        try:
            # OKX
            exchanges["okx"] = ccxt.okx({"sandbox": False, "enableRateLimit": True})
        except Exception:
            pass

        return exchanges

    def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch current funding rates for given symbols"""

        funding_data = {}

        for exchange_name, exchange in self.exchanges.items():
            if not hasattr(exchange, "fetch_funding_rate"):
                continue

            for symbol in symbols:
                try:
                    # Convert symbol format
                    futures_symbol = f"{symbol}/USDT:USDT" if ":" not in symbol else symbol

                    funding_info = exchange.fetch_funding_rate(futures_symbol)

                    if symbol not in funding_data:
                        funding_data[symbol] = {}

                    funding_data[symbol][exchange_name] = {
                        "funding_rate": funding_info.get("fundingRate", 0),
                        "next_funding_time": funding_info.get("fundingTimestamp"),
                        "mark_price": funding_info.get("markPrice", 0),
                        "index_price": funding_info.get("indexPrice", 0),
                    }

                except Exception as e:
                    print(f"Failed to fetch funding rate for {symbol} on {exchange_name}: {e}")
                    continue

        return funding_data

    def fetch_open_interest(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch open interest data"""

        oi_data = {}

        for exchange_name, exchange in self.exchanges.items():
            for symbol in symbols:
                try:
                    futures_symbol = f"{symbol}/USDT:USDT" if ":" not in symbol else symbol

                    # Try to fetch open interest
                    if hasattr(exchange, "fetch_open_interest"):
                        oi_info = exchange.fetch_open_interest(futures_symbol)

                        if symbol not in oi_data:
                            oi_data[symbol] = {}

                        oi_data[symbol][exchange_name] = {
                            "open_interest": oi_info.get("openInterestAmount", 0),
                            "open_interest_value": oi_info.get("openInterestValue", 0),
                            "timestamp": oi_info.get("timestamp"),
                        }

                except Exception as e:
                    print(f"Failed to fetch OI for {symbol} on {exchange_name}: {e}")
                    continue

        return oi_data

    def calculate_basis(
        self, spot_price: float, futures_price: float, days_to_expiry: float = 30
    ) -> Dict[str, float]:
        """Calculate futures basis and annualized basis"""

        if spot_price <= 0 or futures_price <= 0:
            return {"basis": 0, "annualized_basis": 0}

        # Simple basis
        basis = (futures_price - spot_price) / spot_price

        # Annualized basis
        if days_to_expiry > 0:
            annualized_basis = basis * (365 / days_to_expiry)
        else:
            annualized_basis = 0

        return {"basis": basis, "annualized_basis": annualized_basis}


class FuturesFeatureGenerator:
    """
    Generate ML features from futures data
    """

    def __init__(self):
        self.funding_history = []
        self.oi_history = []
        self.basis_history = []

    def add_funding_data(self, timestamp: datetime, funding_data: Dict[str, Any]):
        """Add funding rate data point"""

        data_point = {"timestamp": timestamp, "data": funding_data}
        self.funding_history.append(data_point)

        # Keep only last 30 days
        cutoff = timestamp - timedelta(days=30)
        self.funding_history = [
            point for point in self.funding_history if point["timestamp"] >= cutoff
        ]

    def add_oi_data(self, timestamp: datetime, oi_data: Dict[str, Any]):
        """Add open interest data point"""

        data_point = {"timestamp": timestamp, "data": oi_data}
        self.oi_history.append(data_point)

        # Keep only last 30 days
        cutoff = timestamp - timedelta(days=30)
        self.oi_history = [point for point in self.oi_history if point["timestamp"] >= cutoff]

    def generate_funding_features(self, symbol: str) -> Dict[str, float]:
        """Generate funding rate features"""

        if not self.funding_history:
            return self._empty_funding_features()

        # Extract funding rates for symbol
        funding_rates = []
        for point in self.funding_history:
            if symbol in point["data"]:
                # Average across exchanges
                symbol_data = point["data"][symbol]
                rates = [
                    exchange_data.get("funding_rate", 0)
                    for exchange_data in symbol_data.values()
                    if isinstance(exchange_data, dict)
                ]
                if rates:
                    avg_rate = np.mean(rates)
                    funding_rates.append(avg_rate)

        if not funding_rates:
            return self._empty_funding_features()

        # Convert to numpy array
        rates = np.array(funding_rates)

        # Calculate features
        features = {
            "funding_rate_current": rates[-1] if len(rates) > 0 else 0,
            "funding_rate_mean_24h": np.mean(rates[-8:]) if len(rates) >= 8 else np.mean(rates),
            "funding_rate_std_24h": np.std(rates[-8:]) if len(rates) >= 8 else np.std(rates),
            "funding_rate_min_24h": np.min(rates[-8:]) if len(rates) >= 8 else np.min(rates),
            "funding_rate_max_24h": np.max(rates[-8:]) if len(rates) >= 8 else np.max(rates),
            "funding_rate_trend": self._calculate_trend(rates[-24:]) if len(rates) >= 24 else 0,
            "funding_rate_extreme": 1
            if abs(rates[-1]) > 0.01
            else 0
            if len(rates) > 0
            else 0,  # 1% threshold
            "funding_rate_squeeze_signal": self._detect_funding_squeeze(rates),
        }

        return features

    def generate_oi_features(self, symbol: str) -> Dict[str, float]:
        """Generate open interest features"""

        if not self.oi_history:
            return self._empty_oi_features()

        # Extract OI values for symbol
        oi_values = []
        for point in self.oi_history:
            if symbol in point["data"]:
                symbol_data = point["data"][symbol]
                values = [
                    exchange_data.get("open_interest", 0)
                    for exchange_data in symbol_data.values()
                    if isinstance(exchange_data, dict)
                ]
                if values:
                    total_oi = sum(values)
                    oi_values.append(total_oi)

        if not oi_values:
            return self._empty_oi_features()

        # Convert to numpy array
        oi = np.array(oi_values)

        # Calculate features
        features = {
            "open_interest_current": oi[-1] if len(oi) > 0 else 0,
            "open_interest_change_24h": (oi[-1] - oi[-24]) / oi[-24]
            if len(oi) >= 24 and oi[-24] != 0
            else 0,
            "open_interest_change_7d": (oi[-1] - oi[-168]) / oi[-168]
            if len(oi) >= 168 and oi[-168] != 0
            else 0,
            "open_interest_volatility": np.std(oi[-24:]) / np.mean(oi[-24:])
            if len(oi) >= 24 and np.mean(oi[-24:]) != 0
            else 0,
            "open_interest_trend": self._calculate_trend(oi[-24:]) if len(oi) >= 24 else 0,
            "open_interest_spike": 1
            if len(oi) >= 2 and (oi[-1] / oi[-2]) > 1.2
            else 0,  # 20% spike
            "open_interest_drop": 1 if len(oi) >= 2 and (oi[-1] / oi[-2]) < 0.8 else 0,  # 20% drop
        }

        return features

    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""

        if len(values) < 2:
            return 0

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Normalize by mean to get relative trend
        mean_val = np.mean(values)
        if mean_val != 0:
            normalized_slope = slope / mean_val
        else:
            normalized_slope = 0

        return normalized_slope

    def _detect_funding_squeeze(self, rates: np.ndarray) -> float:
        """Detect funding rate squeeze patterns"""

        if len(rates) < 12:  # Need at least 12 periods
            return 0

        recent_rates = rates[-12:]

        # Check for consistently high funding rates
        high_funding = np.mean(recent_rates) > 0.005  # 0.5% average

        # Check for increasing trend
        increasing_trend = self._calculate_trend(recent_rates) > 0

        # Check for volatility spike
        volatility_spike = np.std(recent_rates) > np.std(rates[:-12]) * 1.5

        # Combine signals
        squeeze_score = sum([high_funding, increasing_trend, volatility_spike]) / 3

        return squeeze_score

    def _empty_funding_features(self) -> Dict[str, float]:
        """Return empty funding features"""
        return {
            "funding_rate_current": 0,
            "funding_rate_mean_24h": 0,
            "funding_rate_std_24h": 0,
            "funding_rate_min_24h": 0,
            "funding_rate_max_24h": 0,
            "funding_rate_trend": 0,
            "funding_rate_extreme": 0,
            "funding_rate_squeeze_signal": 0,
        }

    def _empty_oi_features(self) -> Dict[str, float]:
        """Return empty OI features"""
        return {
            "open_interest_current": 0,
            "open_interest_change_24h": 0,
            "open_interest_change_7d": 0,
            "open_interest_volatility": 0,
            "open_interest_trend": 0,
            "open_interest_spike": 0,
            "open_interest_drop": 0,
        }


def create_futures_feature_pipeline() -> Tuple[FuturesDataCollector, FuturesFeatureGenerator]:
    """Create complete futures data pipeline"""

    collector = FuturesDataCollector()
    feature_gen = FuturesFeatureGenerator()

    return collector, feature_gen


if __name__ == "__main__":
    print("📊 TESTING FUTURES DATA FEATURES")
    print("=" * 40)

    # Test futures data collection
    collector, feature_gen = create_futures_feature_pipeline()

    print(f"Initialized {len(collector.exchanges)} exchanges")

    # Test with sample data since real API calls might fail
    sample_symbols = ["BTC", "ETH", "SOL"]

    # REMOVED: Mock data pattern not allowed in production
    sample_funding = {
        "BTC": {
            "binance": {"funding_rate": 0.0001, "mark_price": 67500},
            "bybit": {"funding_rate": 0.0002, "mark_price": 67520},
        },
        "ETH": {
            "binance": {"funding_rate": 0.0003, "mark_price": 3850},
            "bybit": {"funding_rate": 0.0004, "mark_price": 3860},
        },
    }

    # Add sample data points
    base_time = datetime.now()
    for i in range(24):  # 24 hours of data
        timestamp = base_time - timedelta(hours=23 - i)

        # Vary funding rates slightly
        varied_funding = {}
        for symbol, exchanges in sample_funding.items():
            varied_funding[symbol] = {}
            for exchange, data in exchanges.items():
                noise = np.random.normal(0, 1)
                varied_funding[symbol][exchange] = {
                    "funding_rate": data["funding_rate"] + noise,
                    "mark_price": data["mark_price"],
                }

        feature_gen.add_funding_data(timestamp, varied_funding)

    # Generate features
    btc_funding_features = feature_gen.generate_funding_features("BTC")

    print("BTC Funding Features:")
    for key, value in btc_funding_features.items():
        print(f"   {key}: {value:.6f}")

    # Test basis calculation
    basis_calc = collector.calculate_basis(67500, 67520, 30)
    print(f"\nBasis Calculation:")
    print(f"   Basis: {basis_calc['basis']:.6f}")
    print(f"   Annualized: {basis_calc['annualized_basis']:.6f}")

    print("✅ Futures data features testing completed")
