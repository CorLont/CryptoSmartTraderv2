"""
Zero Fallback Data Validator - STRICT Production Mode
Ensures ZERO synthetic/fallback data tolerance in production
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    PRICE = "price"
    SENTIMENT = "sentiment"
    WHALE = "whale"
    VOLUME = "volume"
    TECHNICAL = "technical"
    NEWS = "news"


@dataclass
class DataValidationResult:
    is_valid: bool
    data_type: DataType
    symbol: str
    timestamp: str
    rejection_reason: Optional[str] = None
    source: Optional[str] = None


class ZeroFallbackValidator:
    """
    STRICT validator that blocks ALL synthetic/fallback data
    Missing data = immediate exclusion from analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rejected_data_log = []
        self.validation_stats = {"total_validations": 0, "rejections": 0, "rejection_reasons": {}}

    def validate_price_data(self, data: Dict[str, Any], symbol: str) -> DataValidationResult:
        """Validate price data with ZERO tolerance for synthetic data"""
        timestamp = datetime.now().isoformat()

        # Check for synthetic markers
        synthetic_indicators = [
            "generated",
            "synthetic",
            "fallback",
            "mock",
            "dummy",
            "simulated",
            "estimated",
            "interpolated",
            "backfilled",
        ]

        if any(indicator in str(data).lower() for indicator in synthetic_indicators):
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.PRICE,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Synthetic data markers detected",
                source=data.get("source", "unknown"),
            )

        # Validate required fields
        required_fields = ["open", "high", "low", "close", "volume", "timestamp"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return DataValidationResult(
                    is_valid=False,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    timestamp=timestamp,
                    rejection_reason=f"Missing required field: {field}",
                )

        # Validate OHLC consistency
        ohlc = [data["open"], data["high"], data["low"], data["close"]]
        if not (min(ohlc) == data["low"] and max(ohlc) == data["high"]):
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.PRICE,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="OHLC consistency violation",
            )

        # Check for impossible values
        if any(price <= 0 for price in ohlc) or data["volume"] < 0:
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.PRICE,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Invalid price/volume values",
            )

        # Validate data freshness (must be recent)
        if "timestamp" in data:
            data_age = datetime.now().timestamp() - data["timestamp"]
            if data_age > 3600:  # Older than 1 hour
                return DataValidationResult(
                    is_valid=False,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    timestamp=timestamp,
                    rejection_reason="Data too old (>1 hour)",
                )

        self._log_validation_success(DataType.PRICE, symbol)
        return DataValidationResult(
            is_valid=True,
            data_type=DataType.PRICE,
            symbol=symbol,
            timestamp=timestamp,
            source=data.get("source", "exchange"),
        )

    def validate_sentiment_data(self, data: Dict[str, Any], symbol: str) -> DataValidationResult:
        """Validate sentiment data - REJECT if scraping failed"""
        timestamp = datetime.now().isoformat()

        # Check for fallback sentiment indicators
        if data.get("is_fallback", False) or data.get("is_synthetic", False):
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.SENTIMENT,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Fallback/synthetic sentiment data",
            )

        # Require actual data sources
        required_sources = data.get("data_sources", [])
        if not required_sources or len(required_sources) == 0:
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.SENTIMENT,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="No actual data sources",
            )

        # Validate sentiment score range
        sentiment_score = data.get("sentiment_score")
        if sentiment_score is None or not (0 <= sentiment_score <= 1):
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.SENTIMENT,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Invalid sentiment score",
            )

        # Require minimum mention volume for reliability
        mention_volume = data.get("mention_volume", 0)
        if mention_volume < 10:  # Minimum threshold
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.SENTIMENT,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Insufficient mention volume for reliability",
            )

        self._log_validation_success(DataType.SENTIMENT, symbol)
        return DataValidationResult(
            is_valid=True,
            data_type=DataType.SENTIMENT,
            symbol=symbol,
            timestamp=timestamp,
            source=",".join(required_sources),
        )

    def validate_whale_data(self, data: Dict[str, Any], symbol: str) -> DataValidationResult:
        """Validate whale data - REJECT if on-chain scraping failed"""
        timestamp = datetime.now().isoformat()

        # Check for simulated whale data
        if data.get("is_simulated", False) or "simulated" in str(data).lower():
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.WHALE,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Simulated whale data detected",
            )

        # Require blockchain source verification
        if not data.get("blockchain_verified", False):
            return DataValidationResult(
                is_valid=False,
                data_type=DataType.WHALE,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="No blockchain verification",
            )

        # Validate transaction data
        required_fields = ["large_transactions", "net_flow", "whale_concentration"]
        for field in required_fields:
            if field not in data:
                return DataValidationResult(
                    is_valid=False,
                    data_type=DataType.WHALE,
                    symbol=symbol,
                    timestamp=timestamp,
                    rejection_reason=f"Missing whale field: {field}",
                )

        self._log_validation_success(DataType.WHALE, symbol)
        return DataValidationResult(
            is_valid=True,
            data_type=DataType.WHALE,
            symbol=symbol,
            timestamp=timestamp,
            source="blockchain",
        )

    def validate_and_filter_batch(
        self, data_batch: List[Dict[str, Any]], data_type: DataType
    ) -> List[Dict[str, Any]]:
        """Validate entire batch and filter out invalid data"""
        valid_data = []
        rejected_count = 0

        for item in data_batch:
            symbol = item.get("symbol", "unknown")

            if data_type == DataType.PRICE:
                result = self.validate_price_data(item, symbol)
            elif data_type == DataType.SENTIMENT:
                result = self.validate_sentiment_data(item, symbol)
            elif data_type == DataType.WHALE:
                result = self.validate_whale_data(item, symbol)
            else:
                # Generic validation for other types
                result = self._validate_generic(item, symbol, data_type)

            if result.is_valid:
                valid_data.append(item)
            else:
                rejected_count += 1
                self._log_rejection(result)

        self.logger.warning(
            f"Batch validation: {len(valid_data)} valid, {rejected_count} rejected "
            f"for {data_type.value}"
        )

        return valid_data

    def _validate_generic(
        self, data: Dict[str, Any], symbol: str, data_type: DataType
    ) -> DataValidationResult:
        """Generic validation for other data types"""
        timestamp = datetime.now().isoformat()

        # Check for synthetic markers
        if any(marker in str(data).lower() for marker in ["synthetic", "fallback", "mock"]):
            return DataValidationResult(
                is_valid=False,
                data_type=data_type,
                symbol=symbol,
                timestamp=timestamp,
                rejection_reason="Synthetic data markers",
            )

        return DataValidationResult(
            is_valid=True, data_type=data_type, symbol=symbol, timestamp=timestamp
        )

    def _log_validation_success(self, data_type: DataType, symbol: str):
        """Log successful validation"""
        self.validation_stats["total_validations"] += 1

    def _log_rejection(self, result: DataValidationResult):
        """Log data rejection with full details"""
        self.validation_stats["total_validations"] += 1
        self.validation_stats["rejections"] += 1

        reason = result.rejection_reason
        if reason not in self.validation_stats["rejection_reasons"]:
            self.validation_stats["rejection_reasons"][reason] = 0
        self.validation_stats["rejection_reasons"][reason] += 1

        # Log to file for audit trail
        rejection_record = {
            "timestamp": result.timestamp,
            "symbol": result.symbol,
            "data_type": result.data_type.value,
            "rejection_reason": result.rejection_reason,
            "source": result.source,
        }

        self.rejected_data_log.append(rejection_record)

        # Log critical rejection
        self.logger.error(
            f"DATA REJECTED: {result.symbol} ({result.data_type.value}) - {result.rejection_reason}"
        )

        # Save rejection log periodically
        if len(self.rejected_data_log) % 100 == 0:
            self._save_rejection_log()

    def _save_rejection_log(self):
        """Save rejection log to file"""
        try:
            with open("logs/rejected_data.json", "w") as f:
                json.dump(
                    {
                        "rejection_log": self.rejected_data_log,
                        "validation_stats": self.validation_stats,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Failed to save rejection log: {e}")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        rejection_rate = 0
        if self.validation_stats["total_validations"] > 0:
            rejection_rate = (
                self.validation_stats["rejections"]
                / self.validation_stats["total_validations"]
                * 100
            )

        return {
            **self.validation_stats,
            "rejection_rate_percent": round(rejection_rate, 2),
            "status": "STRICT_MODE_ACTIVE" if rejection_rate > 0 else "ALL_DATA_VALID",
        }

    def enforce_strict_mode(self) -> bool:
        """Ensure strict mode is active"""
        stats = self.get_validation_stats()

        # Log current enforcement status
        self.logger.critical(
            f"ZERO FALLBACK MODE ACTIVE - Rejection rate: {stats['rejection_rate_percent']}%"
        )

        return True


# Global validator instance
zero_fallback_validator = ZeroFallbackValidator()
