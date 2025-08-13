#!/usr/bin/env python3
"""
Coverage Audit System - Enterprise Data Coverage & Hygiene
Implements 99%+ exchange coverage with comprehensive diff reporting and hard gates
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import ccxt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..core.logging_manager import get_logger
from ..core.async_data_manager import get_async_data_manager

class CoverageStatus(str, Enum):
    """Coverage audit status"""
    COMPLETE = "complete"      # ≥99% coverage achieved
    PARTIAL = "partial"        # <99% but >95% coverage
    INSUFFICIENT = "insufficient"  # <95% coverage - BLOCK
    ERROR = "error"           # Audit failed

class TickerChangeType(str, Enum):
    """Types of ticker changes"""
    NEW_LISTING = "new_listing"
    DELISTED = "delisted"
    MISSING = "missing"
    RESTORED = "restored"
    DATA_QUALITY_ISSUE = "data_quality_issue"

@dataclass
class TickerChange:
    """Individual ticker change record"""
    symbol: str
    change_type: TickerChangeType
    detected_at: datetime
    previous_status: Optional[str] = None
    current_status: Optional[str] = None
    impact_score: float = 0.0  # 0-1 impact on coverage
    volume_24h: float = 0.0
    market_cap_tier: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoverageReport:
    """Comprehensive coverage audit report"""
    audit_id: str
    timestamp: datetime
    exchange_name: str
    total_tradeable_tickers: int
    covered_tickers: int
    missing_tickers: int
    coverage_percentage: float
    status: CoverageStatus
    ticker_changes: List[TickerChange]
    data_quality_issues: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class DataCompletenessConfig:
    """Configuration for data completeness validation"""
    minimum_completeness_percent: float = 0.98  # 98% minimum
    required_features: List[str] = field(default_factory=lambda: [
        "price", "volume", "high", "low", "open", "close", "change"
    ])
    optional_features: List[str] = field(default_factory=lambda: [
        "bid", "ask", "spread", "volume_quote"
    ])
    max_missing_features: int = 1  # Max 1 missing optional feature
    latency_budget_minutes: int = 30  # 30 min max for batch processing

@dataclass
class CompletenessValidationResult:
    """Result from completeness validation"""
    validation_id: str
    timestamp: datetime
    total_coins_evaluated: int
    passed_coins: int
    excluded_coins: int
    exclusion_reasons: Dict[str, int]
    average_completeness: float
    feature_availability: Dict[str, float]
    processing_latency_minutes: float

class KrakenCoverageAuditor:
    """Kraken exchange coverage auditor"""

    def __init__(self):
        self.logger = get_logger()
        self.exchange = None
        self.last_audit_result = None
        self.ticker_history = {}

    async def initialize_exchange(self):
        """Initialize Kraken exchange connection"""

        try:
            self.exchange = ccxt.kraken({
                'apiKey': '',  # Public data only
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
            })

            await self.exchange.load_markets()

            self.logger.info(f"Kraken exchange initialized with {len(self.exchange.markets)} markets")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Kraken exchange: {e}")
            return False

    async def run_coverage_audit(self) -> CoverageReport:
        """Run comprehensive coverage audit"""

        audit_id = f"coverage_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audit_start = datetime.now()

        self.logger.info(f"Starting coverage audit: {audit_id}")

        try:
            # Initialize exchange if needed
            if not self.exchange:
                if not await self.initialize_exchange():
                    raise Exception("Failed to initialize exchange")

            # Get all tradeable tickers from Kraken
            tradeable_tickers = await self._get_all_tradeable_tickers()

            if not tradeable_tickers:
                raise Exception("No tradeable tickers found")

            # Get current data coverage
            covered_tickers = await self._get_covered_tickers()

            # Calculate coverage metrics
            total_tradeable = len(tradeable_tickers)
            total_covered = len(covered_tickers)
            missing_tickers = set(tradeable_tickers) - set(covered_tickers)
            coverage_percentage = (total_covered / total_tradeable) * 100 if total_tradeable > 0 else 0

            # Determine status
            if coverage_percentage >= 99.0:
                status = CoverageStatus.COMPLETE
            elif coverage_percentage >= 95.0:
                status = CoverageStatus.PARTIAL
            else:
                status = CoverageStatus.INSUFFICIENT

            # Detect ticker changes
            ticker_changes = await self._detect_ticker_changes(tradeable_tickers, covered_tickers)

            # Analyze data quality issues
            data_quality_issues = await self._analyze_data_quality(covered_tickers)

            # Calculate performance metrics
            audit_duration = (datetime.now() - audit_start).total_seconds() / 60
            performance_metrics = {
                "audit_duration_minutes": audit_duration,
                "tickers_per_minute": total_tradeable / max(audit_duration, 0.1),
                "coverage_percentage": coverage_percentage,
                "missing_count": len(missing_tickers)
            }

            # Generate recommendations
            recommendations = self._generate_recommendations(
                coverage_percentage, missing_tickers, ticker_changes, data_quality_issues
            )

            # Create coverage report
            coverage_report = CoverageReport(
                audit_id=audit_id,
                timestamp=audit_start,
                exchange_name="kraken",
                total_tradeable_tickers=total_tradeable,
                covered_tickers=total_covered,
                missing_tickers=len(missing_tickers),
                coverage_percentage=coverage_percentage,
                status=status,
                ticker_changes=ticker_changes,
                data_quality_issues=data_quality_issues,
                performance_metrics=performance_metrics,
                recommendations=recommendations
            )

            # Store audit result
            await self._store_coverage_report(coverage_report)

            # Update ticker history
            self._update_ticker_history(tradeable_tickers, covered_tickers)

            self.last_audit_result = coverage_report

            self.logger.info(
                f"Coverage audit completed: {audit_id}",
                extra={
                    "audit_id": audit_id,
                    "status": status.value,
                    "coverage_percentage": coverage_percentage,
                    "total_tradeable": total_tradeable,
                    "covered": total_covered,
                    "missing": len(missing_tickers),
                    "audit_duration_minutes": audit_duration
                }
            )

            return coverage_report

        except Exception as e:
            self.logger.error(f"Coverage audit failed: {e}")

            # Return error report
            return CoverageReport(
                audit_id=audit_id,
                timestamp=audit_start,
                exchange_name="kraken",
                total_tradeable_tickers=0,
                covered_tickers=0,
                missing_tickers=0,
                coverage_percentage=0.0,
                status=CoverageStatus.ERROR,
                ticker_changes=[],
                data_quality_issues=[{"error": str(e)}],
                performance_metrics={},
                recommendations=["Fix exchange connection and retry audit"]
            )

    async def _get_all_tradeable_tickers(self) -> List[str]:
        """Get all tradeable tickers from Kraken"""

        try:
            # Load markets
            markets = self.exchange.markets

            # Filter for active, spot trading pairs
            tradeable_tickers = []

            for symbol, market in markets.items():
                if (market.get('active', False) and
                    market.get('type') == 'spot' and
                    not market.get('info', {}).get('wsname', '').startswith('.')):  # Skip .d pairs

                    tradeable_tickers.append(symbol)

            # Sort for consistent ordering
            tradeable_tickers.sort()

            self.logger.info(f"Found {len(tradeable_tickers)} tradeable tickers on Kraken")

            return tradeable_tickers

        except Exception as e:
            self.logger.error(f"Failed to get tradeable tickers: {e}")
            return []

    async def _get_covered_tickers(self) -> List[str]:
        """Get tickers currently covered by our data collection"""

        try:
            # Get data from async data manager
            async_data_manager = await get_async_data_manager()

            # Collect current data
            current_data = await async_data_manager.batch_collect_all_exchanges()

            # Extract covered tickers from Kraken
            kraken_data = current_data.get("exchanges", {}).get("kraken", {})

            if not kraken_data or "tickers" not in kraken_data:
                return []

            covered_tickers = list(kraken_data["tickers"].keys())
            covered_tickers.sort()

            self.logger.info(f"Currently covering {len(covered_tickers)} tickers")

            return covered_tickers

        except Exception as e:
            self.logger.error(f"Failed to get covered tickers: {e}")
            return []

    async def _detect_ticker_changes(
        self,
        tradeable_tickers: List[str],
        covered_tickers: List[str]
    ) -> List[TickerChange]:
        """Detect changes in ticker availability"""

        changes = []
        current_time = datetime.now()

        # Get previous state
        previous_tradeable = set(self.ticker_history.get("tradeable", []))
        previous_covered = set(self.ticker_history.get("covered", []))

        current_tradeable = set(tradeable_tickers)
        current_covered = set(covered_tickers)

        # Detect new listings
        new_listings = current_tradeable - previous_tradeable
        for symbol in new_listings:
            volume_24h = await self._get_ticker_volume(symbol)
            market_cap_tier = self._classify_market_cap_tier(volume_24h)

            changes.append(TickerChange(
                symbol=symbol,
                change_type=TickerChangeType.NEW_LISTING,
                detected_at=current_time,
                previous_status="not_listed",
                current_status="tradeable",
                impact_score=min(volume_24h / 1000000, 1.0),  # Impact based on volume
                volume_24h=volume_24h,
                market_cap_tier=market_cap_tier,
                metadata={"first_detected": current_time.isoformat()}
            ))

        # Detect delistings
        delistings = previous_tradeable - current_tradeable
        for symbol in delistings:
            changes.append(TickerChange(
                symbol=symbol,
                change_type=TickerChangeType.DELISTED,
                detected_at=current_time,
                previous_status="tradeable",
                current_status="delisted",
                impact_score=0.5,  # Medium impact for delistings
                metadata={"delisted_at": current_time.isoformat()}
            ))

        # Detect missing coverage
        missing_coverage = current_tradeable - current_covered
        for symbol in missing_coverage:
            volume_24h = await self._get_ticker_volume(symbol)

            changes.append(TickerChange(
                symbol=symbol,
                change_type=TickerChangeType.MISSING,
                detected_at=current_time,
                previous_status="unknown",
                current_status="tradeable_but_not_covered",
                impact_score=min(volume_24h / 1000000, 1.0),
                volume_24h=volume_24h,
                market_cap_tier=self._classify_market_cap_tier(volume_24h),
                metadata={"volume_24h": volume_24h}
            ))

        # Detect restored coverage
        restored_coverage = (current_covered & previous_tradeable) - previous_covered
        for symbol in restored_coverage:
            changes.append(TickerChange(
                symbol=symbol,
                change_type=TickerChangeType.RESTORED,
                detected_at=current_time,
                previous_status="missing_coverage",
                current_status="covered",
                impact_score=0.3,
                metadata={"restored_at": current_time.isoformat()}
            ))

        return changes

    async def _get_ticker_volume(self, symbol: str) -> float:
        """Get 24h volume for ticker"""

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker.get('quoteVolume', 0.0) or ticker.get('baseVolume', 0.0) or 0.0
        except Exception:
            return 0.0

    def _classify_market_cap_tier(self, volume_24h: float) -> str:
        """Classify market cap tier based on volume"""

        if volume_24h > 100000000:  # $100M+
            return "large_cap"
        elif volume_24h > 10000000:  # $10M+
            return "mid_cap"
        elif volume_24h > 1000000:  # $1M+
            return "small_cap"
        else:
            return "micro_cap"

    async def _analyze_data_quality(self, covered_tickers: List[str]) -> List[Dict[str, Any]]:
        """Analyze data quality issues"""

        issues = []

        try:
            # Sample some tickers for quality analysis
            sample_size = min(50, len(covered_tickers))
            sample_tickers = np.random.normal(0, 1)

            for symbol in sample_tickers:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)

                    # Check for data quality issues
                    if not ticker.get('last') or ticker['last'] <= 0:
                        issues.append({
                            "type": "invalid_price",
                            "symbol": symbol,
                            "description": "Price is zero or negative",
                            "severity": "high"
                        })

                    if not ticker.get('baseVolume') and not ticker.get('quoteVolume'):
                        issues.append({
                            "type": "missing_volume",
                            "symbol": symbol,
                            "description": "Volume data missing",
                            "severity": "medium"
                        })

                    if ticker.get('bid') and ticker.get('ask'):
                        spread = abs(ticker['ask'] - ticker['bid']) / ticker['ask']
                        if spread > 0.05:  # 5%+ spread
                            issues.append({
                                "type": "wide_spread",
                                "symbol": symbol,
                                "description": f"Spread too wide: {spread:.1%}",
                                "severity": "medium"
                            })

                except Exception as e:
                    issues.append({
                        "type": "fetch_error",
                        "symbol": symbol,
                        "description": f"Failed to fetch ticker: {e}",
                        "severity": "high"
                    })

        except Exception as e:
            issues.append({
                "type": "analysis_error",
                "description": f"Data quality analysis failed: {e}",
                "severity": "critical"
            })

        return issues

    def _generate_recommendations(
        self,
        coverage_percentage: float,
        missing_tickers: Set[str],
        ticker_changes: List[TickerChange],
        data_quality_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # Coverage recommendations
        if coverage_percentage < 99.0:
            recommendations.append(f"Coverage at {coverage_percentage:.1f}% - target 99%+")

            high_impact_missing = [
                change for change in ticker_changes
                if change.change_type == TickerChangeType.MISSING and change.impact_score > 0.5
            ]

            if high_impact_missing:
                recommendations.append(f"Priority: Add {len(high_impact_missing)} high-impact missing tickers")

        # New listings
        new_listings = [c for c in ticker_changes if c.change_type == TickerChangeType.NEW_LISTING]
        if new_listings:
            recommendations.append(f"Review {len(new_listings)} new listings for integration")

        # Data quality
        critical_issues = [i for i in data_quality_issues if i.get("severity") == "critical"]
        if critical_issues:
            recommendations.append(f"Fix {len(critical_issues)} critical data quality issues")

        high_issues = [i for i in data_quality_issues if i.get("severity") == "high"]
        if high_issues:
            recommendations.append(f"Address {len(high_issues)} high-severity data issues")

        # Performance
        if coverage_percentage >= 99.0 and not critical_issues:
            recommendations.append("Coverage target achieved - maintain monitoring")

        return recommendations

    async def _store_coverage_report(self, report: CoverageReport):
        """Store coverage report to disk"""

        try:
            # Create directories
            reports_dir = Path("data/coverage_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Prepare report data
            report_data = {
                "audit_id": report.audit_id,
                "timestamp": report.timestamp.isoformat(),
                "exchange_name": report.exchange_name,
                "coverage_metrics": {
                    "total_tradeable_tickers": report.total_tradeable_tickers,
                    "covered_tickers": report.covered_tickers,
                    "missing_tickers": report.missing_tickers,
                    "coverage_percentage": report.coverage_percentage,
                    "status": report.status.value
                },
                "ticker_changes": [
                    {
                        "symbol": change.symbol,
                        "change_type": change.change_type.value,
                        "detected_at": change.detected_at.isoformat(),
                        "impact_score": change.impact_score,
                        "volume_24h": change.volume_24h,
                        "market_cap_tier": change.market_cap_tier,
                        "metadata": change.metadata
                    }
                    for change in report.ticker_changes
                ],
                "data_quality_issues": report.data_quality_issues,
                "performance_metrics": report.performance_metrics,
                "recommendations": report.recommendations
            }

            # Write to file
            report_file = reports_dir / f"coverage_report_{report.audit_id}.json"

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            # Also store as latest
            latest_file = reports_dir / "latest_coverage_report.json"
            with open(latest_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            self.logger.info(f"Coverage report stored: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to store coverage report: {e}")

    def _update_ticker_history(self, tradeable_tickers: List[str], covered_tickers: List[str]):
        """Update ticker history for change detection"""

        self.ticker_history = {
            "tradeable": tradeable_tickers,
            "covered": covered_tickers,
            "updated_at": datetime.now().isoformat()
        }

class DataCompletenessValidator:
    """Validates data completeness with hard gates"""

    def __init__(self, config: Optional[DataCompletenessConfig] = None):
        self.config = config or DataCompletenessConfig()
        self.logger = get_logger()

    async def validate_completeness(self, coin_data: Dict[str, Any]) -> CompletenessValidationResult:
        """Validate data completeness with strict gates"""

        validation_id = f"completeness_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validation_start = datetime.now()

        self.logger.info(f"Starting completeness validation: {validation_id}")

        try:
            passed_coins = []
            excluded_coins = []
            exclusion_reasons = {}
            completeness_scores = []
            feature_availability = {feature: 0 for feature in self.config.required_features + self.config.optional_features}

            total_coins = len(coin_data)

            for symbol, data in coin_data.items():
                # Calculate completeness score
                completeness_score = self._calculate_completeness_score(data)
                completeness_scores.append(completeness_score)

                # Check hard gates
                exclusion_reason = self._check_hard_gates(data)

                if exclusion_reason:
                    excluded_coins.append(symbol)
                    if exclusion_reason not in exclusion_reasons:
                        exclusion_reasons[exclusion_reason] = 0
                    exclusion_reasons[exclusion_reason] += 1

                    self.logger.debug(
                        f"Coin excluded: {symbol} - {exclusion_reason}",
                        extra={
                            "symbol": symbol,
                            "reason": exclusion_reason,
                            "completeness_score": completeness_score
                        }
                    )
                else:
                    passed_coins.append(symbol)

                    # Update feature availability
                    for feature in self.config.required_features + self.config.optional_features:
                        if self._has_valid_feature(data, feature):
                            feature_availability[feature] += 1

            # Calculate metrics
            processing_latency = (datetime.now() - validation_start).total_seconds() / 60
            average_completeness = np.mean(completeness_scores) if completeness_scores else 0.0

            # Convert feature availability to percentages
            for feature in feature_availability:
                feature_availability[feature] = (feature_availability[feature] / total_coins) * 100 if total_coins > 0 else 0

            result = CompletenessValidationResult(
                validation_id=validation_id,
                timestamp=validation_start,
                total_coins_evaluated=total_coins,
                passed_coins=len(passed_coins),
                excluded_coins=len(excluded_coins),
                exclusion_reasons=exclusion_reasons,
                average_completeness=average_completeness,
                feature_availability=feature_availability,
                processing_latency_minutes=processing_latency
            )

            self.logger.info(
                f"Completeness validation completed: {validation_id}",
                extra={
                    "validation_id": validation_id,
                    "total_evaluated": total_coins,
                    "passed": len(passed_coins),
                    "excluded": len(excluded_coins),
                    "average_completeness": average_completeness,
                    "processing_latency_minutes": processing_latency
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Completeness validation failed: {e}")

            return CompletenessValidationResult(
                validation_id=validation_id,
                timestamp=validation_start,
                total_coins_evaluated=0,
                passed_coins=0,
                excluded_coins=0,
                exclusion_reasons={"validation_error": 1},
                average_completeness=0.0,
                feature_availability={},
                processing_latency_minutes=0.0
            )

    def _calculate_completeness_score(self, coin_data: Dict[str, Any]) -> float:
        """Calculate completeness score for coin"""

        total_features = len(self.config.required_features) + len(self.config.optional_features)
        available_features = 0

        for feature in self.config.required_features + self.config.optional_features:
            if self._has_valid_feature(coin_data, feature):
                available_features += 1

        return available_features / total_features if total_features > 0 else 0.0

    def _check_hard_gates(self, coin_data: Dict[str, Any]) -> Optional[str]:
        """Check hard gates - return exclusion reason if any"""

        # Check required features
        missing_required = []
        for feature in self.config.required_features:
            if not self._has_valid_feature(coin_data, feature):
                missing_required.append(feature)

        if missing_required:
            return f"missing_required_features: {', '.join(missing_required)}"

        # Check optional features
        missing_optional = []
        for feature in self.config.optional_features:
            if not self._has_valid_feature(coin_data, feature):
                missing_optional.append(feature)

        if len(missing_optional) > self.config.max_missing_features:
            return f"too_many_missing_optional_features: {len(missing_optional)}/{len(self.config.optional_features)}"

        # Check overall completeness
        completeness_score = self._calculate_completeness_score(coin_data)
        if completeness_score < self.config.minimum_completeness_percent:
            return f"completeness_below_threshold: {completeness_score:.1%} < {self.config.minimum_completeness_percent:.1%}"

        return None

    def _has_valid_feature(self, coin_data: Dict[str, Any], feature: str) -> bool:
        """Check if feature has valid data"""

        value = coin_data.get(feature)

        if value is None:
            return False

        if isinstance(value, (int, float)):
            return not np.isnan(value) and not np.isinf(value) and value > 0

        if isinstance(value, str):
            return len(value.strip()) > 0

        return bool(value)

# Global instances
_kraken_coverage_auditor = None
_data_completeness_validator = None

def get_kraken_coverage_auditor() -> KrakenCoverageAuditor:
    """Get global Kraken coverage auditor instance"""
    global _kraken_coverage_auditor
    if _kraken_coverage_auditor is None:
        _kraken_coverage_auditor = KrakenCoverageAuditor()
    return _kraken_coverage_auditor

def get_data_completeness_validator(config: Optional[DataCompletenessConfig] = None) -> DataCompletenessValidator:
    """Get global data completeness validator instance"""
    global _data_completeness_validator
    if _data_completeness_validator is None:
        _data_completeness_validator = DataCompletenessValidator(config)
    return _data_completeness_validator
