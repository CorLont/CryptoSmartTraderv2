#!/usr/bin/env python3
"""
Advanced Features: Futures Data + Order Book Analysis
Funding rates, open interest, basis, order book imbalance, and spoof detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio

from ..core.structured_logger import get_logger


class FuturesDataAnalyzer:
    """Analyzer for futures-specific features: funding, OI, basis"""

    def __init__(self):
        self.logger = get_logger("FuturesAnalyzer")

    def calculate_funding_features(self, funding_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate funding rate features for predictive modeling"""

        try:
            self.logger.info("Calculating funding rate features")

            if len(funding_data) < 10:
                return self._get_default_funding_features()

            # Current funding rate
            current_funding = funding_data["funding_rate"].iloc[-1]

            # Funding rate statistics
            funding_mean = funding_data["funding_rate"].mean()
            funding_std = funding_data["funding_rate"].std()
            funding_zscore = (current_funding - funding_mean) / (funding_std + 1e-8)

            # Funding rate trends
            funding_ma_8h = funding_data["funding_rate"].rolling(8).mean().iloc[-1]
            funding_ma_24h = funding_data["funding_rate"].rolling(24).mean().iloc[-1]
            funding_trend = funding_ma_8h - funding_ma_24h

            # Extreme funding detection
            funding_percentile = (funding_data["funding_rate"] <= current_funding).mean()
            is_extreme_positive = funding_percentile > 0.95  # Top 5%
            is_extreme_negative = funding_percentile < 0.05  # Bottom 5%

            # Funding momentum
            funding_changes = funding_data["funding_rate"].diff()
            funding_momentum = funding_changes.rolling(8).mean().iloc[-1]

            # Funding volatility
            funding_volatility = funding_changes.rolling(24).std().iloc[-1]

            features = {
                "current_funding_rate": current_funding,
                "funding_zscore": funding_zscore,
                "funding_trend": funding_trend,
                "funding_momentum": funding_momentum,
                "funding_volatility": funding_volatility,
                "is_extreme_positive_funding": float(is_extreme_positive),
                "is_extreme_negative_funding": float(is_extreme_negative),
                "funding_percentile": funding_percentile,
                "hours_since_last_extreme": self._hours_since_extreme_funding(funding_data),
            }

            self.logger.info(f"Funding features calculated - current rate: {current_funding:.6f}")
            return features

        except Exception as e:
            self.logger.error(f"Funding feature calculation failed: {e}")
            return self._get_default_funding_features()

    def calculate_open_interest_features(
        self, oi_data: pd.DataFrame, price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate open interest features"""

        try:
            self.logger.info("Calculating open interest features")

            if len(oi_data) < 10:
                return self._get_default_oi_features()

            # Current OI
            current_oi = oi_data["open_interest"].iloc[-1]

            # OI trends
            oi_ma_short = oi_data["open_interest"].rolling(8).mean().iloc[-1]
            oi_ma_long = oi_data["open_interest"].rolling(24).mean().iloc[-1]
            oi_trend = (oi_ma_short - oi_ma_long) / oi_ma_long if oi_ma_long > 0 else 0

            # OI changes
            oi_change_24h = (
                (current_oi - oi_data["open_interest"].iloc[-24])
                / oi_data["open_interest"].iloc[-24]
                if len(oi_data) >= 24
                else 0
            )
            oi_volatility = oi_data["open_interest"].pct_change().rolling(24).std().iloc[-1]

            # Price-OI divergence
            price_change_24h = (
                (price_data["close"].iloc[-1] - price_data["close"].iloc[-24])
                / price_data["close"].iloc[-24]
                if len(price_data) >= 24
                else 0
            )
            price_oi_divergence = abs(price_change_24h - oi_change_24h)

            # OI percentile
            oi_percentile = (oi_data["open_interest"] <= current_oi).mean()

            features = {
                "current_open_interest": current_oi,
                "oi_trend": oi_trend,
                "oi_change_24h": oi_change_24h,
                "oi_volatility": oi_volatility,
                "price_oi_divergence": price_oi_divergence,
                "oi_percentile": oi_percentile,
                "oi_momentum": oi_data["open_interest"].diff().rolling(8).mean().iloc[-1]
                / current_oi
                if current_oi > 0
                else 0,
            }

            self.logger.info(f"OI features calculated - current OI: {current_oi:.0f}")
            return features

        except Exception as e:
            self.logger.error(f"OI feature calculation failed: {e}")
            return self._get_default_oi_features()

    def calculate_basis_features(
        self, spot_price: float, futures_price: float, time_to_expiry_days: float
    ) -> Dict[str, Any]:
        """Calculate basis and carry features"""

        try:
            # Basis calculation
            basis = futures_price - spot_price
            basis_percentage = basis / spot_price if spot_price > 0 else 0

            # Annualized basis (rough carry estimate)
            if time_to_expiry_days > 0:
                annualized_basis = (basis_percentage * 365) / time_to_expiry_days
            else:
                annualized_basis = 0

            # Contango/backwardation
            is_contango = basis > 0
            is_backwardation = basis < 0

            # Basis strength
            basis_strength = abs(basis_percentage)

            features = {
                "basis_absolute": basis,
                "basis_percentage": basis_percentage,
                "annualized_basis": annualized_basis,
                "is_contango": float(is_contango),
                "is_backwardation": float(is_backwardation),
                "basis_strength": basis_strength,
                "time_to_expiry_days": time_to_expiry_days,
            }

            self.logger.info(f"Basis features calculated - basis: {basis_percentage:.4f}")
            return features

        except Exception as e:
            self.logger.error(f"Basis feature calculation failed: {e}")
            return {
                "basis_absolute": 0,
                "basis_percentage": 0,
                "annualized_basis": 0,
                "is_contango": 0,
                "is_backwardation": 0,
                "basis_strength": 0,
                "time_to_expiry_days": 30,
            }

    def _hours_since_extreme_funding(self, funding_data: pd.DataFrame) -> float:
        """Calculate hours since last extreme funding rate"""

        try:
            funding_rates = funding_data["funding_rate"]

            # Define extreme thresholds (adjust based on market)
            extreme_positive = funding_rates.quantile(0.95)
            extreme_negative = funding_rates.quantile(0.05)

            # Find last extreme event
            extreme_mask = (funding_rates >= extreme_positive) | (funding_rates <= extreme_negative)

            if not extreme_mask.any():
                return 168  # Default: 1 week

            last_extreme_idx = extreme_mask[::-1].idxmax()  # Last True value
            hours_since = len(funding_data) - funding_data.index.get_loc(last_extreme_idx) - 1

            return float(hours_since)

        except Exception:
            return 168.0

    def _get_default_funding_features(self) -> Dict[str, Any]:
        """Default funding features when data is insufficient"""
        return {
            "current_funding_rate": 0.0001,  # 0.01% default
            "funding_zscore": 0.0,
            "funding_trend": 0.0,
            "funding_momentum": 0.0,
            "funding_volatility": 0.0001,
            "is_extreme_positive_funding": 0.0,
            "is_extreme_negative_funding": 0.0,
            "funding_percentile": 0.5,
            "hours_since_last_extreme": 168.0,
        }

    def _get_default_oi_features(self) -> Dict[str, Any]:
        """Default OI features when data is insufficient"""
        return {
            "current_open_interest": 1000000,
            "oi_trend": 0.0,
            "oi_change_24h": 0.0,
            "oi_volatility": 0.01,
            "price_oi_divergence": 0.0,
            "oi_percentile": 0.5,
            "oi_momentum": 0.0,
        }


class OrderBookAnalyzer:
    """Advanced order book analysis: imbalance detection and spoof detection"""

    def __init__(self):
        self.logger = get_logger("OrderBookAnalyzer")

    def analyze_orderbook_imbalance(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order book imbalance for timing signals"""

        try:
            self.logger.info("Analyzing order book imbalance")

            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])

            if not bids or not asks:
                return self._get_default_orderbook_features()

            # Convert to arrays for analysis
            bid_prices = np.array([float(bid[0]) for bid in bids])
            bid_volumes = np.array([float(bid[1]) for bid in bids])
            ask_prices = np.array([float(ask[0]) for ask in asks])
            ask_volumes = np.array([float(ask[1]) for ask in asks])

            # Basic imbalance metrics
            total_bid_volume = np.sum(bid_volumes)
            total_ask_volume = np.sum(ask_volumes)

            # Volume imbalance
            volume_imbalance = (total_bid_volume - total_ask_volume) / (
                total_bid_volume + total_ask_volume + 1e-8
            )

            # Depth-weighted imbalance (top 5 levels)
            top_levels = min(5, len(bid_volumes), len(ask_volumes))
            weighted_bid_volume = np.sum(bid_volumes[:top_levels] * np.arange(top_levels, 0, -1))
            weighted_ask_volume = np.sum(ask_volumes[:top_levels] * np.arange(top_levels, 0, -1))
            depth_weighted_imbalance = (weighted_bid_volume - weighted_ask_volume) / (
                weighted_bid_volume + weighted_ask_volume + 1e-8
            )

            # Spread analysis
            best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
            best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            spread_percentage = spread / mid_price if mid_price > 0 else 0

            # Liquidity concentration
            bid_concentration = self._calculate_liquidity_concentration(bid_volumes)
            ask_concentration = self._calculate_liquidity_concentration(ask_volumes)

            # Order book pressure
            pressure_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0

            # Large order detection
            large_bid_threshold = np.percentile(bid_volumes, 90) if len(bid_volumes) > 5 else 0
            large_ask_threshold = np.percentile(ask_volumes, 90) if len(ask_volumes) > 5 else 0
            large_bids_count = np.sum(bid_volumes >= large_bid_threshold)
            large_asks_count = np.sum(ask_volumes >= large_ask_threshold)

            features = {
                "volume_imbalance": volume_imbalance,
                "depth_weighted_imbalance": depth_weighted_imbalance,
                "spread_percentage": spread_percentage,
                "bid_concentration": bid_concentration,
                "ask_concentration": ask_concentration,
                "pressure_ratio": pressure_ratio,
                "large_bids_count": large_bids_count,
                "large_asks_count": large_asks_count,
                "total_bid_volume": total_bid_volume,
                "total_ask_volume": total_ask_volume,
                "orderbook_depth": min(len(bids), len(asks)),
            }

            self.logger.info(f"Order book analysis complete - imbalance: {volume_imbalance:.3f}")
            return features

        except Exception as e:
            self.logger.error(f"Order book analysis failed: {e}")
            return self._get_default_orderbook_features()

    def detect_spoofing(self, orderbook_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect spoofing patterns in order book snapshots"""

        try:
            self.logger.info(
                f"Analyzing {len(orderbook_snapshots)} order book snapshots for spoofing"
            )

            if len(orderbook_snapshots) < 5:
                return {"spoofing_detected": False, "spoof_signals": []}

            spoof_signals = []

            for i in range(1, len(orderbook_snapshots)):
                current_book = orderbook_snapshots[i]
                previous_book = orderbook_snapshots[i - 1]

                # Detect sudden large order appearances/disappearances
                spoof_pattern = self._detect_spoof_pattern(previous_book, current_book)
                if spoof_pattern:
                    spoof_signals.append(
                        {
                            "timestamp": current_book.get(
                                "timestamp", datetime.utcnow().isoformat(),
                            "pattern_type": spoof_pattern["type"],
                            "side": spoof_pattern["side"],
                            "confidence": spoof_pattern["confidence"],
                        }
                    )

            spoofing_detected = len(spoof_signals) > 0
            spoof_intensity = len(spoof_signals) / len(orderbook_snapshots)

            return {
                "spoofing_detected": spoofing_detected,
                "spoof_signals": spoof_signals,
                "spoof_intensity": spoof_intensity,
                "total_snapshots_analyzed": len(orderbook_snapshots),
            }

        except Exception as e:
            self.logger.error(f"Spoofing detection failed: {e}")
            return {"spoofing_detected": False, "spoof_signals": []}

    def _detect_spoof_pattern(
        self, prev_book: Dict[str, Any], curr_book: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect specific spoofing patterns between two order book snapshots"""

        try:
            prev_bids = {float(bid[0]): float(bid[1]) for bid in prev_book.get("bids", [])}
            curr_bids = {float(bid[0]): float(bid[1]) for bid in curr_book.get("bids", [])}
            prev_asks = {float(ask[0]): float(ask[1]) for ask in prev_book.get("asks", [])}
            curr_asks = {float(ask[0]): float(ask[1]) for ask in curr_book.get("asks", [])}

            # Pattern 1: Large order suddenly appears and disappears
            disappeared_bids = set(prev_bids.keys()) - set(curr_bids.keys())
            appeared_bids = set(curr_bids.keys()) - set(prev_bids.keys())
            disappeared_asks = set(prev_asks.keys()) - set(curr_asks.keys())
            appeared_asks = set(curr_asks.keys()) - set(prev_asks.keys())

            # Check for large orders that disappeared
            large_disappeared_bids = [
                price
                for price in disappeared_bids
                if prev_bids[price] > np.median(list(prev_bids.values()) + [1]) * 5
            ]
            large_disappeared_asks = [
                price
                for price in disappeared_asks
                if prev_asks[price] > np.median(list(prev_asks.values()) + [1]) * 5
            ]

            if large_disappeared_bids:
                return {
                    "type": "large_order_withdrawal",
                    "side": "bid",
                    "confidence": min(len(large_disappeared_bids) * 0.3, 1.0),
                }

            if large_disappeared_asks:
                return {
                    "type": "large_order_withdrawal",
                    "side": "ask",
                    "confidence": min(len(large_disappeared_asks) * 0.3, 1.0),
                }

            # Pattern 2: Sudden large order appearance at edge
            if appeared_bids:
                max_bid = max(appeared_bids)
                if (
                    max_bid in curr_bids
                    and curr_bids[max_bid] > np.median(list(curr_bids.values()) + [1]) * 3
                ):
                    return {"type": "large_order_placement", "side": "bid", "confidence": 0.6}

            if appeared_asks:
                min_ask = min(appeared_asks)
                if (
                    min_ask in curr_asks
                    and curr_asks[min_ask] > np.median(list(curr_asks.values()) + [1]) * 3
                ):
                    return {"type": "large_order_placement", "side": "ask", "confidence": 0.6}

            return None

        except Exception:
            return None

    def _calculate_liquidity_concentration(self, volumes: np.ndarray) -> float:
        """Calculate Herfindahl-Hirschman Index for liquidity concentration"""

        if len(volumes) == 0:
            return 0.0

        total_volume = np.sum(volumes)
        if total_volume == 0:
            return 0.0

        proportions = volumes / total_volume
        hhi = np.sum(proportions**2)

        return float(hhi)

    def _get_default_orderbook_features(self) -> Dict[str, Any]:
        """Default order book features when analysis fails"""
        return {
            "volume_imbalance": 0.0,
            "depth_weighted_imbalance": 0.0,
            "spread_percentage": 0.001,
            "bid_concentration": 0.2,
            "ask_concentration": 0.2,
            "pressure_ratio": 1.0,
            "large_bids_count": 0,
            "large_asks_count": 0,
            "total_bid_volume": 1000,
            "total_ask_volume": 1000,
            "orderbook_depth": 10,
        }


class AdvancedTimingSignals:
    """Combines futures and order book analysis for superior timing"""

    def __init__(self):
        self.logger = get_logger("AdvancedTimingSignals")
        self.futures_analyzer = FuturesDataAnalyzer()
        self.orderbook_analyzer = OrderBookAnalyzer()

    def generate_timing_signals(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive timing signals from advanced features"""

        try:
            self.logger.info("Generating advanced timing signals")

            signals = {
                "timestamp": datetime.utcnow().isoformat(),
                "timing_score": 0.0,
                "signal_strength": "NEUTRAL",
                "contributing_factors": [],
            }

            # Futures signals
            if "futures_data" in market_data:
                futures_signals = self._analyze_futures_signals(market_data["futures_data"])
                signals.update(futures_signals)

            # Order book signals
            if "orderbook_data" in market_data:
                orderbook_signals = self._analyze_orderbook_signals(market_data["orderbook_data"])
                signals.update(orderbook_signals)

            # Combine signals
            timing_score = self._calculate_combined_timing_score(signals)
            signals["timing_score"] = timing_score
            signals["signal_strength"] = self._classify_signal_strength(timing_score)

            self.logger.info(f"Timing signals generated - score: {timing_score:.3f}")
            return signals

        except Exception as e:
            self.logger.error(f"Timing signal generation failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "timing_score": 0.0,
                "signal_strength": "NEUTRAL",
                "contributing_factors": ["error_in_analysis"],
            }

    def _analyze_futures_signals(self, futures_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze futures-specific signals"""

        signals = {}

        # Funding rate signals
        if "funding_rates" in futures_data:
            funding_features = self.futures_analyzer.calculate_funding_features(
                futures_data["funding_rates"]
            )

            # Extreme funding = potential reversal
            if funding_features["is_extreme_positive_funding"]:
                signals["funding_signal"] = -0.3  # Bearish
                signals.setdefault("contributing_factors", []).append("extreme_positive_funding")
            elif funding_features["is_extreme_negative_funding"]:
                signals["funding_signal"] = 0.3  # Bullish
                signals.setdefault("contributing_factors", []).append("extreme_negative_funding")
            else:
                signals["funding_signal"] = 0.0

        # Open interest signals
        if "open_interest" in futures_data and "price_data" in futures_data:
            oi_features = self.futures_analyzer.calculate_open_interest_features(
                futures_data["open_interest"], futures_data["price_data"]
            )

            # Rising OI + rising price = bullish continuation
            if oi_features["oi_trend"] > 0.02:  # 2% OI increase
                signals["oi_signal"] = 0.2
                signals.setdefault("contributing_factors", []).append("rising_open_interest")
            elif oi_features["oi_trend"] < -0.02:
                signals["oi_signal"] = -0.1
                signals.setdefault("contributing_factors", []).append("falling_open_interest")
            else:
                signals["oi_signal"] = 0.0

        return signals

    def _analyze_orderbook_signals(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order book signals"""

        signals = {}

        # Imbalance signals
        imbalance_features = self.orderbook_analyzer.analyze_orderbook_imbalance(orderbook_data)

        volume_imbalance = imbalance_features["volume_imbalance"]
        if abs(volume_imbalance) > 0.1:  # 10% imbalance
            signals["imbalance_signal"] = volume_imbalance * 0.5  # Scale down
            if volume_imbalance > 0:
                signals.setdefault("contributing_factors", []).append("bid_volume_dominance")
            else:
                signals.setdefault("contributing_factors", []).append("ask_volume_dominance")
        else:
            signals["imbalance_signal"] = 0.0

        # Spread signals
        spread_pct = imbalance_features["spread_percentage"]
        if spread_pct > 0.005:  # Wide spread = uncertainty
            signals["spread_signal"] = -0.1
            signals.setdefault("contributing_factors", []).append("wide_spread")
        elif spread_pct < 0.001:  # Tight spread = confidence
            signals["spread_signal"] = 0.1
            signals.setdefault("contributing_factors", []).append("tight_spread")
        else:
            signals["spread_signal"] = 0.0

        # Spoofing detection
        if "orderbook_history" in orderbook_data:
            spoof_result = self.orderbook_analyzer.detect_spoofing(
                orderbook_data["orderbook_history"]
            )
            if spoof_result["spoofing_detected"]:
                signals["spoof_signal"] = -0.2  # Negative for market manipulation
                signals.setdefault("contributing_factors", []).append("spoofing_detected")
            else:
                signals["spoof_signal"] = 0.0

        return signals

    def _calculate_combined_timing_score(self, signals: Dict[str, Any]) -> float:
        """Calculate combined timing score from all signals"""

        signal_components = []

        # Collect signal components
        for key, value in signals.items():
            if key.endswith("_signal") and isinstance(value, (int, float)):
                signal_components.append(value)

        if not signal_components:
            return 0.0

        # Weighted average with some non-linearity
        base_score = np.mean(signal_components)

        # Boost extreme scores
        if abs(base_score) > 0.3:
            base_score *= 1.2

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, base_score))

    def _classify_signal_strength(self, timing_score: float) -> str:
        """Classify signal strength based on timing score"""

        abs_score = abs(timing_score)

        if abs_score >= 0.7:
            return "VERY_STRONG"
        elif abs_score >= 0.5:
            return "STRONG"
        elif abs_score >= 0.3:
            return "MODERATE"
        elif abs_score >= 0.1:
            return "WEAK"
        else:
            return "NEUTRAL"
