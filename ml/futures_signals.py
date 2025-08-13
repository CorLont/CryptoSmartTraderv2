#!/usr/bin/env python3
"""
Futures Signals System
Detects crowding, squeeze-risk via funding rates, open interest, and basis features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import ccxt.async_support as ccxt
import warnings

warnings.filterwarnings("ignore")


@dataclass
class FuturesSignal:
    """Futures-based trading signal"""

    symbol: str
    signal_type: str  # 'funding_squeeze', 'oi_divergence', 'basis_anomaly', 'crowding_reversal'
    signal_strength: float  # 0-1
    funding_rate: float
    funding_premium_percentile: float  # Historical percentile
    open_interest: float
    oi_change_24h: float
    basis_bps: float  # Basis in basis points
    basis_z_score: float
    crowding_score: float  # 0-1, higher = more crowded
    squeeze_risk: float  # 0-1, higher = more squeeze risk
    signal_confidence: float
    expected_duration_hours: int
    risk_level: str  # 'low', 'medium', 'high'


class FuturesDataCollector:
    """Collects futures data from multiple exchanges"""

    def __init__(self):
        self.exchanges = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize futures exchanges"""
        try:
            # Binance Futures
            self.exchanges["binance"] = ccxt.binance(
                {
                    "apiKey": "",  # Will use environment variables
                    "secret": "",
                    "sandbox": False,
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                }
            )

            # FTX (if available)
            # self.exchanges['ftx'] = ccxt.ftx({
            #     'apiKey': '',
            #     'secret': '',
            #     'enableRateLimit': True
            # })

        except Exception as e:
            self.logger.warning(f"Exchange initialization warning: {e}")

    async def get_funding_rates(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get current funding rates for symbols"""

        funding_data = {}

        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()

                for symbol in symbols:
                    futures_symbol = f"{symbol}/USDT"

                    if futures_symbol in exchange.markets:
                        # Get funding rate
                        funding_rate = await exchange.fetch_funding_rate(futures_symbol)

                        # Get funding rate history
                        funding_history = await exchange.fetch_funding_rate_history(
                            futures_symbol,
                            limit=168,  # 1 week of 1h data
                        )

                        funding_data[symbol] = {
                            "current_rate": funding_rate["fundingRate"],
                            "next_funding_time": funding_rate["fundingTimestamp"],
                            "history": [f["fundingRate"] for f in funding_history],
                            "exchange": exchange_name,
                        }

            except Exception as e:
                self.logger.error(f"Error fetching funding rates from {exchange_name}: {e}")

        return funding_data

    async def get_open_interest(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get open interest data for symbols"""

        oi_data = {}

        for exchange_name, exchange in self.exchanges.items():
            try:
                for symbol in symbols:
                    futures_symbol = f"{symbol}/USDT"

                    if futures_symbol in exchange.markets:
                        # Get current OI
                        ticker = await exchange.fetch_ticker(futures_symbol)

                        # Get OI history (if available)
                        try:
                            oi_history = await exchange.fetch_open_interest_history(
                                futures_symbol, timeframe="1h", limit=168
                            )
                            oi_values = [oi["openInterest"] for oi in oi_history]
                        except FileNotFoundError as e:
                            logger.warning(f"Error in futures_signals.py: {e}")
                            oi_values = []

                        oi_data[symbol] = {
                            "current_oi": ticker.get("info", {}).get("openInterest", 0),
                            "oi_history": oi_values,
                            "volume_24h": ticker["quoteVolume"],
                            "exchange": exchange_name,
                        }

            except Exception as e:
                self.logger.error(f"Error fetching OI from {exchange_name}: {e}")

        return oi_data

    async def get_basis_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get spot-futures basis data"""

        basis_data = {}

        for exchange_name, exchange in self.exchanges.items():
            try:
                for symbol in symbols:
                    spot_symbol = f"{symbol}/USDT"
                    futures_symbol = f"{symbol}/USDT"

                    # Get spot and futures prices
                    spot_ticker = await exchange.fetch_ticker(spot_symbol)
                    futures_ticker = await exchange.fetch_ticker(futures_symbol)

                    spot_price = spot_ticker["last"]
                    futures_price = futures_ticker["last"]

                    # Calculate basis
                    basis_absolute = futures_price - spot_price
                    basis_bps = (basis_absolute / spot_price) * 10000

                    basis_data[symbol] = {
                        "spot_price": spot_price,
                        "futures_price": futures_price,
                        "basis_absolute": basis_absolute,
                        "basis_bps": basis_bps,
                        "exchange": exchange_name,
                    }

            except Exception as e:
                self.logger.error(f"Error fetching basis from {exchange_name}: {e}")

        return basis_data

    async def close_connections(self):
        """Close exchange connections"""
        for exchange in self.exchanges.values():
            await exchange.close()


class FuturesSignalGenerator:
    """Generates trading signals from futures data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.signal_history = {}

        # Signal thresholds
        self.thresholds = {
            "funding_extreme_percentile": 90,  # 90th percentile for extreme funding
            "oi_change_threshold": 0.15,  # 15% OI change threshold
            "basis_zscore_threshold": 2.0,  # 2 std devs for basis anomaly
            "crowding_threshold": 0.7,  # 70% crowding score threshold
            "min_signal_confidence": 0.6,  # 60% minimum confidence
        }

    def generate_funding_squeeze_signals(
        self, funding_data: Dict[str, Dict]
    ) -> List[FuturesSignal]:
        """Generate signals based on funding rate extremes"""

        signals = []

        for symbol, data in funding_data.items():
            current_rate = data["current_rate"]
            history = data.get("history", [])

            if len(history) < 24:  # Need at least 24 hours of data
                continue

            # Calculate percentile of current funding rate
            funding_percentile = self._calculate_percentile(current_rate, history)

            # Extreme funding suggests potential reversal
            if funding_percentile > self.thresholds["funding_extreme_percentile"]:
                # High funding -> shorts pay longs -> potential squeeze up
                signal_strength = min(1.0, (funding_percentile - 50) / 50)
                squeeze_risk = signal_strength

                signal = FuturesSignal(
                    symbol=symbol,
                    signal_type="funding_squeeze",
                    signal_strength=signal_strength,
                    funding_rate=current_rate,
                    funding_premium_percentile=funding_percentile,
                    open_interest=0,  # Will be filled later
                    oi_change_24h=0,
                    basis_bps=0,
                    basis_z_score=0,
                    crowding_score=signal_strength,
                    squeeze_risk=squeeze_risk,
                    signal_confidence=min(0.9, signal_strength + 0.2),
                    expected_duration_hours=8,
                    risk_level="high" if squeeze_risk > 0.8 else "medium",
                )

                signals.append(signal)

            elif funding_percentile < (100 - self.thresholds["funding_extreme_percentile"]):
                # Low/negative funding -> longs pay shorts -> potential squeeze down
                signal_strength = min(1.0, (50 - funding_percentile) / 50)
                squeeze_risk = signal_strength

                signal = FuturesSignal(
                    symbol=symbol,
                    signal_type="funding_squeeze",
                    signal_strength=-signal_strength,  # Negative for short signal
                    funding_rate=current_rate,
                    funding_premium_percentile=funding_percentile,
                    open_interest=0,
                    oi_change_24h=0,
                    basis_bps=0,
                    basis_z_score=0,
                    crowding_score=signal_strength,
                    squeeze_risk=squeeze_risk,
                    signal_confidence=min(0.9, signal_strength + 0.2),
                    expected_duration_hours=8,
                    risk_level="high" if squeeze_risk > 0.8 else "medium",
                )

                signals.append(signal)

        return signals

    def generate_oi_divergence_signals(
        self, oi_data: Dict[str, Dict], price_data: Dict[str, float]
    ) -> List[FuturesSignal]:
        """Generate signals based on OI divergence from price"""

        signals = []

        for symbol, data in oi_data.items():
            oi_history = data.get("oi_history", [])
            current_oi = data["current_oi"]

            if len(oi_history) < 24 or symbol not in price_data:
                continue

            # Calculate OI change
            if len(oi_history) >= 24:
                oi_24h_ago = oi_history[-24]
                oi_change_24h = (current_oi - oi_24h_ago) / oi_24h_ago if oi_24h_ago > 0 else 0
            else:
                oi_change_24h = 0

            # Check for significant OI changes
            if abs(oi_change_24h) > self.thresholds["oi_change_threshold"]:
                # Rising OI + rising price = trend continuation
                # Rising OI + falling price = potential reversal
                # Falling OI + rising price = weak trend
                # Falling OI + falling price = trend exhaustion

                price_change = price_data.get(f"{symbol}_change_24h", 0)

                signal_type = "oi_divergence"

                if oi_change_24h > 0 and price_change < -0.05:
                    # Rising OI, falling price -> potential reversal up
                    signal_strength = min(1.0, abs(oi_change_24h) * 2)
                    signal_direction = 1
                elif oi_change_24h < 0 and price_change > 0.05:
                    # Falling OI, rising price -> potential reversal down
                    signal_strength = min(1.0, abs(oi_change_24h) * 2)
                    signal_direction = -1
                else:
                    continue

                crowding_score = min(1.0, abs(oi_change_24h) * 3)

                signal = FuturesSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    signal_strength=signal_strength * signal_direction,
                    funding_rate=0,
                    funding_premium_percentile=50,
                    open_interest=current_oi,
                    oi_change_24h=oi_change_24h,
                    basis_bps=0,
                    basis_z_score=0,
                    crowding_score=crowding_score,
                    squeeze_risk=crowding_score * 0.8,
                    signal_confidence=min(0.8, signal_strength + 0.3),
                    expected_duration_hours=12,
                    risk_level="medium" if crowding_score < 0.7 else "high",
                )

                signals.append(signal)

        return signals

    def generate_basis_anomaly_signals(self, basis_data: Dict[str, Dict]) -> List[FuturesSignal]:
        """Generate signals based on spot-futures basis anomalies"""

        signals = []

        # First, collect all basis values for z-score calculation
        all_basis_values = [data["basis_bps"] for data in basis_data.values()]

        if len(all_basis_values) < 5:
            return signals

        basis_mean = np.mean(all_basis_values)
        basis_std = np.std(all_basis_values)

        for symbol, data in basis_data.items():
            basis_bps = data["basis_bps"]

            # Calculate z-score
            z_score = (basis_bps - basis_mean) / basis_std if basis_std > 0 else 0

            # Check for basis anomalies
            if abs(z_score) > self.thresholds["basis_zscore_threshold"]:
                # Extreme positive basis -> futures overpriced -> potential convergence (short futures)
                # Extreme negative basis -> futures underpriced -> potential convergence (long futures)

                signal_strength = min(1.0, abs(z_score) / 3)
                signal_direction = -1 if z_score > 0 else 1  # Convergence trade

                signal = FuturesSignal(
                    symbol=symbol,
                    signal_type="basis_anomaly",
                    signal_strength=signal_strength * signal_direction,
                    funding_rate=0,
                    funding_premium_percentile=50,
                    open_interest=0,
                    oi_change_24h=0,
                    basis_bps=basis_bps,
                    basis_z_score=z_score,
                    crowding_score=signal_strength * 0.6,
                    squeeze_risk=signal_strength * 0.4,
                    signal_confidence=min(0.8, signal_strength + 0.2),
                    expected_duration_hours=6,
                    risk_level="low" if abs(z_score) < 3 else "medium",
                )

                signals.append(signal)

        return signals

    def generate_crowding_reversal_signals(
        self, funding_data: Dict[str, Dict], oi_data: Dict[str, Dict], basis_data: Dict[str, Dict]
    ) -> List[FuturesSignal]:
        """Generate reversal signals based on extreme crowding"""

        signals = []

        # Get symbols that have all data types
        common_symbols = set(funding_data.keys()) & set(oi_data.keys()) & set(basis_data.keys())

        for symbol in common_symbols:
            funding = funding_data[symbol]
            oi = oi_data[symbol]
            basis = basis_data[symbol]

            # Calculate crowding score based on multiple factors
            funding_extreme = self._calculate_percentile(
                funding["current_rate"], funding.get("history", [])

            oi_change = oi.get("oi_change_24h", 0)
            basis_bps = basis["basis_bps"]

            # Crowding indicators:
            # - Extreme funding rates
            # - High OI increases
            # - Wide basis spreads

            crowding_factors = []

            # Funding extremity (0-1)
            funding_factor = max(
                (funding_extreme - 50) / 50 if funding_extreme > 50 else 0,
                (50 - funding_extreme) / 50 if funding_extreme < 50 else 0,
            )
            crowding_factors.append(funding_factor)

            # OI increase factor (0-1)
            oi_factor = min(1.0, max(0, oi_change * 3)) if oi_change > 0 else 0
            crowding_factors.append(oi_factor)

            # Basis extremity factor (0-1)
            basis_factor = min(1.0, abs(basis_bps) / 100)  # Normalize by 100 bps
            crowding_factors.append(basis_factor)

            # Combined crowding score
            crowding_score = np.mean(crowding_factors)

            if crowding_score > self.thresholds["crowding_threshold"]:
                # Determine reversal direction based on dominant factor
                if funding_extreme > 70:
                    # High funding -> potential squeeze up
                    signal_direction = 1
                elif funding_extreme < 30:
                    # Low funding -> potential squeeze down
                    signal_direction = -1
                elif basis_bps > 50:
                    # Wide positive basis -> convergence short
                    signal_direction = -1
                elif basis_bps < -50:
                    # Wide negative basis -> convergence long
                    signal_direction = 1
                else:
                    signal_direction = 1 if oi_change > 0 else -1

                signal_strength = crowding_score
                squeeze_risk = crowding_score

                signal = FuturesSignal(
                    symbol=symbol,
                    signal_type="crowding_reversal",
                    signal_strength=signal_strength * signal_direction,
                    funding_rate=funding["current_rate"],
                    funding_premium_percentile=funding_extreme,
                    open_interest=oi["current_oi"],
                    oi_change_24h=oi_change,
                    basis_bps=basis_bps,
                    basis_z_score=0,
                    crowding_score=crowding_score,
                    squeeze_risk=squeeze_risk,
                    signal_confidence=min(0.9, crowding_score + 0.1),
                    expected_duration_hours=4,
                    risk_level="high",
                )

                signals.append(signal)

        return signals

    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of value in history"""
        if not history:
            return 50.0

        sorted_history = sorted(history)
        position = sum(1 for h in sorted_history if h <= value)
        percentile = (position / len(sorted_history)) * 100

        return percentile


class FuturesSignalSystem:
    """Complete futures signal system"""

    def __init__(self):
        self.data_collector = FuturesDataCollector()
        self.signal_generator = FuturesSignalGenerator()
        self.logger = logging.getLogger(__name__)

        # Signal aggregation settings
        self.max_signals_per_symbol = 3
        self.signal_decay_hours = 24

    async def generate_futures_signals(
        self, symbols: List[str], price_data: Dict[str, float] = None
    ) -> List[FuturesSignal]:
        """Generate all futures-based signals"""

        if price_data is None:
            price_data = {}

        all_signals = []

        try:
            # Collect futures data
            funding_data = await self.data_collector.get_funding_rates(symbols)
            oi_data = await self.data_collector.get_open_interest(symbols)
            basis_data = await self.data_collector.get_basis_data(symbols)

            # Generate different signal types
            funding_signals = self.signal_generator.generate_funding_squeeze_signals(funding_data)
            oi_signals = self.signal_generator.generate_oi_divergence_signals(oi_data, price_data)
            basis_signals = self.signal_generator.generate_basis_anomaly_signals(basis_data)
            crowding_signals = self.signal_generator.generate_crowding_reversal_signals(
                funding_data, oi_data, basis_data
            )

            all_signals.extend(funding_signals)
            all_signals.extend(oi_signals)
            all_signals.extend(basis_signals)
            all_signals.extend(crowding_signals)

            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(all_signals)

            self.logger.info(
                f"Generated {len(filtered_signals)} futures signals from {len(symbols)} symbols"
            )

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error generating futures signals: {e}")
            return []

        finally:
            await self.data_collector.close_connections()

    def _filter_and_rank_signals(self, signals: List[FuturesSignal]) -> List[FuturesSignal]:
        """Filter and rank signals by quality"""

        # Filter by minimum confidence
        filtered = [
            s
            for s in signals
            if s.signal_confidence >= self.signal_generator.thresholds["min_signal_confidence"]
        ]

        # Group by symbol and keep only top signals per symbol
        symbol_groups = {}
        for signal in filtered:
            if signal.symbol not in symbol_groups:
                symbol_groups[signal.symbol] = []
            symbol_groups[signal.symbol].append(signal)

        # Keep top signals per symbol
        final_signals = []
        for symbol, symbol_signals in symbol_groups.items():
            # Sort by signal confidence * squeeze risk (quality metric)
            sorted_signals = sorted(
                symbol_signals,
                key=lambda s: s.signal_confidence * (1 + s.squeeze_risk),
                reverse=True,
            )

            # Keep top N signals per symbol
            final_signals.extend(sorted_signals[: self.max_signals_per_symbol])

        # Final sort by overall quality
        final_signals.sort(
            key=lambda s: s.signal_confidence * abs(s.signal_strength) * (1 + s.squeeze_risk),
            reverse=True,
        )

        return final_signals

    def create_futures_features(self, signals: List[FuturesSignal]) -> pd.DataFrame:
        """Create ML features from futures signals"""

        if not signals:
            return pd.DataFrame()

        features_data = []

        for signal in signals:
            features = {
                "symbol": signal.symbol,
                "futures_signal_strength": signal.signal_strength,
                "funding_rate": signal.funding_rate,
                "funding_percentile": signal.funding_premium_percentile,
                "oi_change_24h": signal.oi_change_24h,
                "basis_bps": signal.basis_bps,
                "basis_z_score": signal.basis_z_score,
                "crowding_score": signal.crowding_score,
                "squeeze_risk": signal.squeeze_risk,
                "signal_confidence": signal.signal_confidence,
                "is_funding_squeeze": signal.signal_type == "funding_squeeze",
                "is_oi_divergence": signal.signal_type == "oi_divergence",
                "is_basis_anomaly": signal.signal_type == "basis_anomaly",
                "is_crowding_reversal": signal.signal_type == "crowding_reversal",
                "risk_level_numeric": {"low": 1, "medium": 2, "high": 3}[signal.risk_level],
            }

            features_data.append(features)

        return pd.DataFrame(features_data)


async def main():
    """Test futures signal system"""

    system = FuturesSignalSystem()

    test_symbols = ["BTC", "ETH", "ADA", "SOL", "AVAX"]
    test_prices = {
        "BTC_change_24h": 0.03,
        "ETH_change_24h": -0.02,
        "ADA_change_24h": 0.08,
        "SOL_change_24h": -0.05,
        "AVAX_change_24h": 0.15,
    }

    signals = await system.generate_futures_signals(test_symbols, test_prices)

    print(f"Generated {len(signals)} futures signals")
    for signal in signals[:5]:
        print(
            f"{signal.symbol}: {signal.signal_type} - Strength: {signal.signal_strength:.3f}, "
            f"Confidence: {signal.signal_confidence:.3f}, Risk: {signal.squeeze_risk:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
