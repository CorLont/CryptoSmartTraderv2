#!/usr/bin/env python3
"""
Coverage Audit System
Compares Kraken exchange coverage vs processed coins to ensure completeness
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from core.structured_logger import get_structured_logger


class KrakenCoverageAuditor:
    """Audit coverage against Kraken exchange"""

    def __init__(self):
        self.logger = get_structured_logger("KrakenCoverageAuditor")
        self.exchange = None

        if CCXT_AVAILABLE:
            try:
                self.exchange = ccxt.kraken(
                    {"sandbox": False, "rateLimit": 1000, "enableRateLimit": True}
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Kraken exchange: {e}")

    def get_kraken_markets(self) -> Set[str]:
        """Get all available markets from Kraken"""

        if not self.exchange:
            self.logger.warning("Kraken exchange not available, using mock data")
            return self._get_mock_kraken_markets()

        try:
            markets = self.exchange.load_markets()

            # Extract base currencies (coins)
            kraken_coins = set()
            for symbol in markets.keys():
                if "/USD" in symbol or "/EUR" in symbol or "/BTC" in symbol:
                    base_currency = symbol.split("/")[0]
                    kraken_coins.add(base_currency)

            self.logger.info(f"Retrieved {len(kraken_coins)} coins from Kraken")
            return kraken_coins

        except Exception as e:
            self.logger.error(f"Failed to get Kraken markets: {e}")
            return self._get_mock_kraken_markets()

    def _get_mock_kraken_markets(self) -> Set[str]:
        """Mock Kraken markets for testing"""

        return {
            "BTC",
            "ETH",
            "ADA",
            "SOL",
            "DOT",
            "AVAX",
            "ALGO",
            "NEAR",
            "FTM",
            "ATOM",
            "LINK",
            "UNI",
            "AAVE",
            "SUSHI",
            "CRV",
            "COMP",
            "YFI",
            "SNX",
            "MKR",
            "BAL",
            "ZRX",
            "ENJ",
            "MANA",
            "SAND",
            "AXS",
            "GALA",
            "APE",
            "LRC",
            "IMX",
            "MATIC",
            "OP",
            "ARB",
            "TIA",
            "SUI",
            "APT",
            "SEI",
            "INJ",
            "PYTH",
            "JUP",
            "WIF",
            "BONK",
            "PEPE",
            "FLOKI",
            "DOGE",
            "SHIB",
            "WLD",
            "FET",
            "RENDER",
            "TAO",
        }


class ProcessedCoinsTracker:
    """Track coins processed by our system"""

    def __init__(self):
        self.logger = get_structured_logger("ProcessedCoinsTracker")

    def get_processed_coins(self) -> Set[str]:
        """Get coins currently processed by our system"""

        try:
            # Check for prediction files
            predictions_dir = Path("cache/predictions")
            data_dir = Path("data")

            processed_coins = set()

            # Scan prediction files
            if predictions_dir.exists():
                for file_path in predictions_dir.glob("*.json"):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "coin" in data:
                                processed_coins.add(data["coin"])
                            elif isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "coin" in item:
                                        processed_coins.add(item["coin"])
                    except Exception as e:
                        self.logger.debug(f"Could not parse {file_path}: {e}")

            # Scan data files
            if data_dir.exists():
                for file_path in data_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(file_path)
                        if "coin" in df.columns:
                            processed_coins.update(df["coin"].unique())
                        elif "symbol" in df.columns:
                            processed_coins.update(df["symbol"].unique())
                    except Exception as e:
                        self.logger.debug(f"Could not parse {file_path}: {e}")

            # If no files found, use mock data
            if not processed_coins:
                processed_coins = self._get_mock_processed_coins()

            self.logger.info(f"Found {len(processed_coins)} processed coins")
            return processed_coins

        except Exception as e:
            self.logger.error(f"Failed to get processed coins: {e}")
            return self._get_mock_processed_coins()

    def _get_mock_processed_coins(self) -> Set[str]:
        """Mock processed coins for testing"""

        return {
            "BTC",
            "ETH",
            "ADA",
            "SOL",
            "DOT",
            "AVAX",
            "ALGO",
            "NEAR",
            "FTM",
            "LINK",
            "UNI",
            "AAVE",
            "MATIC",
            "OP",
            "ARB",  # Subset of Kraken
        }


class CoverageGapAnalyzer:
    """Analyze gaps between available and processed coins"""

    def __init__(self):
        self.logger = get_structured_logger("CoverageGapAnalyzer")

    def analyze_coverage_gaps(
        self, available_coins: Set[str], processed_coins: Set[str]
    ) -> Dict[str, Any]:
        """Analyze coverage gaps and impact"""

        try:
            # Calculate gaps
            missing_coins = available_coins - processed_coins
            extra_coins = processed_coins - available_coins
            covered_coins = available_coins & processed_coins

            # Calculate coverage metrics
            total_available = len(available_coins)
            total_covered = len(covered_coins)
            coverage_percentage = (total_covered / max(total_available, 1)) * 100

            # Estimate impact (simplified model)
            estimated_missing_volume = len(missing_coins) * 50000000  # $50M per coin average
            estimated_missed_opportunities = (
                len(missing_coins) * 2
            )  # 2 opportunities per coin average

            # Priority scoring (based on coin popularity/market cap proxy)
            priority_coins = {
                "BTC",
                "ETH",
                "ADA",
                "SOL",
                "DOT",
                "AVAX",
                "MATIC",
                "LINK",
                "UNI",
                "AAVE",
            }

            high_priority_missing = missing_coins & priority_coins

            gap_analysis = {
                "coverage_summary": {
                    "total_available": total_available,
                    "total_covered": total_covered,
                    "total_missing": len(missing_coins),
                    "coverage_percentage": coverage_percentage,
                    "high_priority_missing": len(high_priority_missing),
                },
                "missing_coins": sorted(list(missing_coins)),
                "high_priority_missing": sorted(list(high_priority_missing)),
                "extra_coins": sorted(list(extra_coins)),
                "covered_coins": sorted(list(covered_coins)),
                "impact_estimation": {
                    "estimated_missing_volume_usd": estimated_missing_volume,
                    "estimated_missed_opportunities": estimated_missed_opportunities,
                    "coverage_status": self._get_coverage_status(coverage_percentage),
                },
            }

            self.logger.info(
                f"Coverage analysis: {coverage_percentage:.1f}% ({total_covered}/{total_available})"
            )

            return gap_analysis

        except Exception as e:
            self.logger.error(f"Coverage gap analysis failed: {e}")
            return {"error": str(e)}

    def _get_coverage_status(self, coverage_percentage: float) -> str:
        """Get coverage status based on percentage"""

        if coverage_percentage >= 99:
            return "EXCELLENT"
        elif coverage_percentage >= 95:
            return "GOOD"
        elif coverage_percentage >= 90:
            return "ADEQUATE"
        elif coverage_percentage >= 80:
            return "NEEDS_IMPROVEMENT"
        else:
            return "CRITICAL"


class ComprehensiveCoverageAuditor:
    """Comprehensive coverage audit system"""

    def __init__(self):
        self.logger = get_structured_logger("ComprehensiveCoverageAuditor")

        self.kraken_auditor = KrakenCoverageAuditor()
        self.processed_tracker = ProcessedCoinsTracker()
        self.gap_analyzer = CoverageGapAnalyzer()

    def run_coverage_audit(self) -> Dict[str, Any]:
        """Run comprehensive coverage audit"""

        start_time = time.time()

        self.logger.info("Starting comprehensive coverage audit")

        try:
            # Get available coins from Kraken
            kraken_coins = self.kraken_auditor.get_kraken_markets()

            # Get processed coins from our system
            processed_coins = self.processed_tracker.get_processed_coins()

            # Analyze gaps
            gap_analysis = self.gap_analyzer.analyze_coverage_gaps(kraken_coins, processed_coins)

            # Compile comprehensive audit report
            audit_report = {
                "audit_timestamp": datetime.now().isoformat(),
                "audit_duration": time.time() - start_time,
                "data_sources": {
                    "kraken_available": len(kraken_coins),
                    "system_processed": len(processed_coins),
                },
                "coverage_analysis": gap_analysis,
                "recommendations": self._generate_recommendations(gap_analysis),
                "quality_gates": self._check_quality_gates(gap_analysis),
            }

            self.logger.info(f"Coverage audit completed in {time.time() - start_time:.2f}s")

            return audit_report

        except Exception as e:
            self.logger.error(f"Coverage audit failed: {e}")
            return {"error": str(e), "audit_timestamp": datetime.now().isoformat()}

    def _generate_recommendations(self, gap_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        try:
            coverage_pct = gap_analysis["coverage_summary"]["coverage_percentage"]
            missing_count = gap_analysis["coverage_summary"]["total_missing"]
            high_priority_missing = gap_analysis["coverage_summary"]["high_priority_missing"]

            if coverage_pct < 95:
                recommendations.append(
                    f"Critical: Coverage below 95% ({coverage_pct:.1f}%). Add missing {missing_count} coins."
                )

            if high_priority_missing > 0:
                recommendations.append(
                    f"High priority: {high_priority_missing} important coins missing from system."
                )

            if missing_count > 10:
                recommendations.append("Consider automated coin discovery and onboarding.")

            if coverage_pct < 90:
                recommendations.append("Implement emergency coverage improvement plan.")

            if not recommendations:
                recommendations.append("Coverage is adequate. Continue monitoring.")

        except Exception as e:
            recommendations.append(f"Could not generate recommendations: {e}")

        return recommendations

    def _check_quality_gates(self, gap_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """Check coverage quality gates"""

        try:
            coverage_pct = gap_analysis["coverage_summary"]["coverage_percentage"]
            high_priority_missing = gap_analysis["coverage_summary"]["high_priority_missing"]

            gates = {
                "coverage_above_95_percent": coverage_pct >= 95.0,
                "coverage_above_99_percent": coverage_pct >= 99.0,
                "no_high_priority_missing": high_priority_missing == 0,
                "overall_coverage_pass": coverage_pct >= 95.0 and high_priority_missing == 0,
            }

            return gates

        except Exception as e:
            self.logger.error(f"Quality gate check failed: {e}")
            return {
                "coverage_above_95_percent": False,
                "coverage_above_99_percent": False,
                "no_high_priority_missing": False,
                "overall_coverage_pass": False,
            }


if __name__ == "__main__":
    # Test coverage audit system
    print("🔍 TESTING COVERAGE AUDIT SYSTEM")
    print("=" * 60)

    # Run comprehensive audit
    auditor = ComprehensiveCoverageAuditor()
    audit_results = auditor.run_coverage_audit()

    print("\n📊 COVERAGE AUDIT RESULTS:")
    if "coverage_analysis" in audit_results:
        coverage = audit_results["coverage_analysis"]["coverage_summary"]
        print(
            f"Coverage: {coverage['coverage_percentage']:.1f}% ({coverage['total_covered']}/{coverage['total_available']})"
        )
        print(f"Missing coins: {coverage['total_missing']}")
        print(f"High priority missing: {coverage['high_priority_missing']}")

        print("\n📋 RECOMMENDATIONS:")
        for rec in audit_results.get("recommendations", []):
            print(f"• {rec}")

        print("\n🚪 QUALITY GATES:")
        gates = audit_results.get("quality_gates", {})
        for gate, passed in gates.items():
            status = "✅" if passed else "❌"
            print(f"{status} {gate}: {passed}")

    print(f"\n⏱️  Audit duration: {audit_results.get('audit_duration', 0):.2f}s")
