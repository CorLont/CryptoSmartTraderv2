#!/usr/bin/env python3
"""
Hard Data Integrity Filter
Implements zero-tolerance policy for incomplete data - coins with missing data are BLOCKED
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings

warnings.filterwarnings("ignore")

from core.logging_manager import get_logger
from core.data_quality_manager import get_data_quality_manager, DataCompleteness


class FilterAction(str, Enum):
    """Filter actions for incomplete data"""

    PASS = "pass"  # Data is complete - allow through
    BLOCK = "block"  # Data is incomplete - BLOCK coin
    QUARANTINE = "quarantine"  # Data quality issues - quarantine for review


class BlockReason(str, Enum):
    """Reasons for blocking coins"""

    MISSING_PRICE = "missing_price"
    MISSING_VOLUME = "missing_volume"
    MISSING_OHLCV = "missing_ohlcv"
    MISSING_SENTIMENT = "missing_sentiment"
    MISSING_TECHNICAL = "missing_technical"
    INSUFFICIENT_COMPLETENESS = "insufficient_completeness"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    API_ERROR = "api_error"


@dataclass
class FilterResult:
    """Result of hard data filter check"""

    symbol: str
    action: FilterAction
    completeness_score: float
    missing_components: List[str]
    block_reason: Optional[BlockReason]
    component_scores: Dict[str, float]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterStats:
    """Statistics from filtering operation"""

    timestamp: datetime
    total_coins_processed: int
    coins_passed: int
    coins_blocked: int
    coins_quarantined: int
    block_reasons: Dict[BlockReason, int]
    average_completeness: float
    filter_duration_seconds: float


class HardDataFilter:
    """Hard data integrity filter with zero tolerance for incomplete data"""

    def __init__(self):
        self.logger = get_logger()
        self.data_quality_manager = get_data_quality_manager()

        # Filter configuration
        self.completeness_threshold = 0.8  # 80% minimum completeness
        self.critical_components = {"price_data", "volume_data", "ohlcv_data"}  # Must have these
        self.component_weights = {
            "price_data": 0.35,  # 35% weight - CRITICAL
            "volume_data": 0.25,  # 25% weight - CRITICAL
            "ohlcv_data": 0.20,  # 20% weight - CRITICAL
            "order_book_data": 0.10,  # 10% weight
            "sentiment_data": 0.05,  # 5% weight
            "technical_data": 0.05,  # 5% weight
        }

        # Minimum thresholds per component
        self.component_thresholds = {
            "price_data": 0.9,  # 90% - price must be nearly perfect
            "volume_data": 0.8,  # 80% - volume must be reliable
            "ohlcv_data": 0.7,  # 70% - OHLCV needs reasonable coverage
            "order_book_data": 0.5,  # 50% - order book can be partial
            "sentiment_data": 0.3,  # 30% - sentiment can be sparse
            "technical_data": 0.6,  # 60% - technical indicators need good coverage
        }

        # Statistics tracking
        self.filter_stats_history = []
        self.blocked_coins_history = []

        self.logger.info(
            "Hard Data Filter initialized with zero-tolerance policy",
            extra={
                "completeness_threshold": self.completeness_threshold,
                "critical_components": list(self.critical_components),
            },
        )

    def apply_hard_filter(self, coins_data: Dict[str, Any]) -> Tuple[Dict[str, Any], FilterStats]:
        """Apply hard filter to coin data - BLOCK incomplete coins"""

        start_time = datetime.now()

        filtered_data = {}
        filter_results = []
        block_reasons = {}

        total_processed = len(coins_data)
        passed_count = 0
        blocked_count = 0
        quarantined_count = 0

        completeness_scores = []

        self.logger.info(f"Applying hard filter to {total_processed} coins")

        for symbol, coin_data in coins_data.items():
            try:
                # Check coin completeness
                filter_result = self._check_coin_completeness_strict(symbol, coin_data)
                filter_results.append(filter_result)
                completeness_scores.append(filter_result.completeness_score)

                if filter_result.action == FilterAction.PASS:
                    # Coin passes strict filter - include in output
                    filtered_data[symbol] = coin_data
                    passed_count += 1

                    self.logger.debug(
                        f"PASS: {symbol} - completeness {filter_result.completeness_score:.2f}",
                        extra={
                            "symbol": symbol,
                            "action": "PASS",
                            "completeness": filter_result.completeness_score,
                        },
                    )

                elif filter_result.action == FilterAction.BLOCK:
                    # Coin BLOCKED - incomplete data
                    blocked_count += 1

                    # Track block reason
                    reason = filter_result.block_reason
                    if reason not in block_reasons:
                        block_reasons[reason] = 0
                    block_reasons[reason] += 1

                    # Store blocked coin for analysis
                    self.blocked_coins_history.append(
                        {
                            "symbol": symbol,
                            "block_reason": reason.value,
                            "completeness_score": filter_result.completeness_score,
                            "missing_components": filter_result.missing_components,
                            "timestamp": start_time,
                        }
                    )

                    self.logger.warning(
                        f"BLOCKED: {symbol} - {reason.value} (completeness {filter_result.completeness_score:.2f})",
                        extra={
                            "symbol": symbol,
                            "action": "BLOCKED",
                            "block_reason": reason.value,
                            "completeness": filter_result.completeness_score,
                            "missing_components": filter_result.missing_components,
                        },
                    )

                else:  # QUARANTINE
                    quarantined_count += 1

                    self.logger.info(
                        f"QUARANTINE: {symbol} - requires review",
                        extra={
                            "symbol": symbol,
                            "action": "QUARANTINE",
                            "completeness": filter_result.completeness_score,
                        },
                    )

            except Exception as e:
                # Exception during filtering - BLOCK the coin
                blocked_count += 1
                block_reasons[BlockReason.API_ERROR] = (
                    block_reasons.get(BlockReason.API_ERROR, 0) + 1
                )

                self.logger.error(
                    f"BLOCKED: {symbol} - filter error: {e}",
                    extra={
                        "symbol": symbol,
                        "action": "BLOCKED",
                        "block_reason": "api_error",
                        "error": str(e),
                    },
                )

        # Calculate statistics
        filter_duration = (datetime.now() - start_time).total_seconds()
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0

        filter_stats = FilterStats(
            timestamp=start_time,
            total_coins_processed=total_processed,
            coins_passed=passed_count,
            coins_blocked=blocked_count,
            coins_quarantined=quarantined_count,
            block_reasons=block_reasons,
            average_completeness=avg_completeness,
            filter_duration_seconds=filter_duration,
        )

        # Store statistics
        self.filter_stats_history.append(filter_stats)
        if len(self.filter_stats_history) > 1000:  # Keep last 1000 stats
            self.filter_stats_history = self.filter_stats_history[-1000:]

        # Keep blocked coins history manageable
        if len(self.blocked_coins_history) > 10000:  # Keep last 10000 blocked coins
            self.blocked_coins_history = self.blocked_coins_history[-10000:]

        # Log final results
        self.logger.info(
            f"Hard filter completed: {passed_count}/{total_processed} coins passed",
            extra={
                "total_processed": total_processed,
                "passed": passed_count,
                "blocked": blocked_count,
                "quarantined": quarantined_count,
                "pass_rate": passed_count / total_processed if total_processed > 0 else 0,
                "avg_completeness": avg_completeness,
                "duration_seconds": filter_duration,
            },
        )

        return filtered_data, filter_stats

    def _check_coin_completeness_strict(
        self, symbol: str, coin_data: Dict[str, Any]
    ) -> FilterResult:
        """Strict completeness check with zero tolerance for incomplete data"""

        component_scores = {}
        missing_components = []

        # 1. Price data completeness (CRITICAL)
        price_score = self._check_price_data_completeness(coin_data)
        component_scores["price_data"] = price_score
        if price_score < self.component_thresholds["price_data"]:
            missing_components.append("price_data")

        # 2. Volume data completeness (CRITICAL)
        volume_score = self._check_volume_data_completeness(coin_data)
        component_scores["volume_data"] = volume_score
        if volume_score < self.component_thresholds["volume_data"]:
            missing_components.append("volume_data")

        # 3. OHLCV data completeness (CRITICAL)
        ohlcv_score = self._check_ohlcv_data_completeness(coin_data)
        component_scores["ohlcv_data"] = ohlcv_score
        if ohlcv_score < self.component_thresholds["ohlcv_data"]:
            missing_components.append("ohlcv_data")

        # 4. Order book data completeness
        order_book_score = self._check_order_book_completeness(coin_data)
        component_scores["order_book_data"] = order_book_score
        if order_book_score < self.component_thresholds["order_book_data"]:
            missing_components.append("order_book_data")

        # 5. Sentiment data completeness
        sentiment_score = self._check_sentiment_data_completeness(coin_data)
        component_scores["sentiment_data"] = sentiment_score
        if sentiment_score < self.component_thresholds["sentiment_data"]:
            missing_components.append("sentiment_data")

        # 6. Technical data completeness
        technical_score = self._check_technical_data_completeness(coin_data)
        component_scores["technical_data"] = technical_score
        if technical_score < self.component_thresholds["technical_data"]:
            missing_components.append("technical_data")

        # Calculate weighted completeness score
        total_score = sum(
            component_scores[component] * weight
            for component, weight in self.component_weights.items()

        # Determine filter action
        action, block_reason = self._determine_filter_action(
            total_score, missing_components, component_scores
        )

        return FilterResult(
            symbol=symbol,
            action=action,
            completeness_score=total_score,
            missing_components=missing_components,
            block_reason=block_reason,
            component_scores=component_scores,
            timestamp=datetime.now(),
            details={
                "critical_components_missing": len(
                    [c for c in missing_components if c in self.critical_components]
                ),
                "total_components_missing": len(missing_components),
            },
        )

    def _check_price_data_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check price data completeness - CRITICAL component"""

        score = 0.0

        # Check if basic price data exists
        if not coin_data.get("price"):
            return 0.0

        price = coin_data["price"]
        bid = coin_data.get("bid", 0)
        ask = coin_data.get("ask", 0)

        # Price must be positive and reasonable
        if price <= 0:
            return 0.0

        checks_passed = 0
        total_checks = 4

        # Check 1: Valid price
        if price > 0:
            checks_passed += 1

        # Check 2: Valid bid
        if bid > 0 and bid <= price:
            checks_passed += 1

        # Check 3: Valid ask
        if ask > 0 and ask >= price:
            checks_passed += 1

        # Check 4: Reasonable spread
        if bid > 0 and ask > 0:
            spread = ((ask - bid) / price) * 100
            if spread < 10:  # Spread less than 10%
                checks_passed += 1

        score = checks_passed / total_checks

        return score

    def _check_volume_data_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check volume data completeness - CRITICAL component"""

        volume = coin_data.get("volume", 0)

        if volume <= 0:
            return 0.0

        # Volume quality checks
        score = 0.0

        # Basic volume exists
        if volume > 0:
            score += 0.5

        # Meaningful volume (> $1000)
        if volume > 1000:
            score += 0.3

        # High volume (> $100k)
        if volume > 100000:
            score += 0.2

        return min(score, 1.0)

    def _check_ohlcv_data_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check OHLCV data completeness - CRITICAL component"""

        # Check if OHLCV data exists
        ohlcv_data = coin_data.get("ohlcv", {})

        if not ohlcv_data:
            return 0.0

        valid_timeframes = 0
        total_timeframes = len(ohlcv_data)

        for timeframe, data in ohlcv_data.items():
            if data and isinstance(data, list) and len(data) > 10:
                # Check if data points are valid
                valid_points = 0
                for point in data[:10]:  # Check first 10 points
                    if (
                        isinstance(point, list)
                        and len(point) >= 6
                        and all(isinstance(x, (int, float)) and x > 0 for x in point[1:5]):  # OHLC values
                        valid_points += 1

                if valid_points >= 8:  # At least 80% valid points
                    valid_timeframes += 1

        if total_timeframes == 0:
            return 0.0

        return valid_timeframes / total_timeframes

    def _check_order_book_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check order book data completeness"""

        order_book = coin_data.get("order_book", {})

        if not order_book:
            return 0.0

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            return 0.0

        score = 0.0

        # Check bid depth
        if len(bids) >= 5:
            score += 0.3
        elif len(bids) >= 1:
            score += 0.1

        # Check ask depth
        if len(asks) >= 5:
            score += 0.3
        elif len(asks) >= 1:
            score += 0.1

        # Check data quality
        valid_bids = sum(1 for bid in bids[:5] if len(bid) >= 2 and bid[0] > 0 and bid[1] > 0)
        valid_asks = sum(1 for ask in asks[:5] if len(ask) >= 2 and ask[0] > 0 and ask[1] > 0)

        if valid_bids >= 3 and valid_asks >= 3:
            score += 0.4

        return min(score, 1.0)

    def _check_sentiment_data_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check sentiment data completeness"""

        # For now, estimate based on volume (high volume coins likely have sentiment data)
        volume = coin_data.get("volume", 0)

        if volume > 10000000:  # $10M+ volume
            return 0.9
        elif volume > 1000000:  # $1M+ volume
            return 0.7
        elif volume > 100000:  # $100k+ volume
            return 0.5
        elif volume > 10000:  # $10k+ volume
            return 0.3
        else:
            return 0.1

    def _check_technical_data_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Check technical indicator data completeness"""

        score = 0.0

        # Check basic technical fields
        if coin_data.get("high") and coin_data["high"] > 0:
            score += 0.25

        if coin_data.get("low") and coin_data["low"] > 0:
            score += 0.25

        if coin_data.get("change") is not None:
            score += 0.25

        if coin_data.get("vwap") and coin_data["vwap"] > 0:
            score += 0.25

        return score

    def _determine_filter_action(
        self,
        completeness_score: float,
        missing_components: List[str],
        component_scores: Dict[str, float],
    ) -> Tuple[FilterAction, Optional[BlockReason]]:
        """Determine filter action based on completeness analysis"""

        # Check critical components first
        critical_missing = [c for c in missing_components if c in self.critical_components]

        # BLOCK if any critical component is missing
        if critical_missing:
            if "price_data" in critical_missing:
                return FilterAction.BLOCK, BlockReason.MISSING_PRICE
            elif "volume_data" in critical_missing:
                return FilterAction.BLOCK, BlockReason.MISSING_VOLUME
            elif "ohlcv_data" in critical_missing:
                return FilterAction.BLOCK, BlockReason.MISSING_OHLCV

        # BLOCK if overall completeness below threshold
        if completeness_score < self.completeness_threshold:
            return FilterAction.BLOCK, BlockReason.INSUFFICIENT_COMPLETENESS

        # BLOCK if too many components missing (more than 3)
        if len(missing_components) > 3:
            return FilterAction.BLOCK, BlockReason.DATA_QUALITY_FAILURE

        # QUARANTINE if close to threshold but not quite passing
        if completeness_score < 0.9 and len(missing_components) > 1:
            return FilterAction.QUARANTINE, None

        # PASS - coin meets all requirements
        return FilterAction.PASS, None

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get comprehensive filter summary"""

        if not self.filter_stats_history:
            return {"error": "No filter statistics available"}

        latest_stats = self.filter_stats_history[-1]

        # Calculate trends
        recent_stats = (
            self.filter_stats_history[-10:]
            if len(self.filter_stats_history) >= 10
            else self.filter_stats_history
        )
        avg_pass_rate = np.mean(
            [
                s.coins_passed / s.total_coins_processed
                for s in recent_stats
                if s.total_coins_processed > 0
            ]
        )
        avg_completeness = np.mean([s.average_completeness for s in recent_stats])

        # Most common block reasons
        all_block_reasons = {}
        for stats in recent_stats:
            for reason, count in stats.block_reasons.items():
                if reason not in all_block_reasons:
                    all_block_reasons[reason] = 0
                all_block_reasons[reason] += count

        return {
            "timestamp": datetime.now().isoformat(),
            "latest_filter_run": {
                "timestamp": latest_stats.timestamp.isoformat(),
                "total_processed": latest_stats.total_coins_processed,
                "passed": latest_stats.coins_passed,
                "blocked": latest_stats.coins_blocked,
                "pass_rate": latest_stats.coins_passed / latest_stats.total_coins_processed
                if latest_stats.total_coins_processed > 0
                else 0,
                "average_completeness": latest_stats.average_completeness,
            },
            "trends": {
                "average_pass_rate_last_10": avg_pass_rate,
                "average_completeness_last_10": avg_completeness,
                "total_filter_runs": len(self.filter_stats_history),
            },
            "block_analysis": {
                "most_common_block_reasons": dict(
                    sorted(all_block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                ),
                "recently_blocked_coins": len(
                    [
                        c
                        for c in self.blocked_coins_history
                        if (datetime.now() - c["timestamp"]).total_seconds() < 3600
                    ]
                ),  # Last hour
            },
            "configuration": {
                "completeness_threshold": self.completeness_threshold,
                "critical_components": list(self.critical_components),
                "zero_tolerance_policy": True,
            },
        }


# Global instance
_hard_data_filter = None


def get_hard_data_filter() -> HardDataFilter:
    """Get global hard data filter instance"""
    global _hard_data_filter
    if _hard_data_filter is None:
        _hard_data_filter = HardDataFilter()
    return _hard_data_filter
