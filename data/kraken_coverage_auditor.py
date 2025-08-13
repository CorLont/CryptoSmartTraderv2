#!/usr/bin/env python3
"""
Kraken Coverage Auditor - 100% Coverage Enforcement
Daily audit ensuring ALL Kraken coins are monitored
"""

import asyncio
import ccxt
import pandas as pd
import json
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
from pathlib import Path

from core.structured_logger import get_structured_logger
from config.daily_logging_config import get_daily_logger


class KrakenCoverageAuditor:
    """Enforces 100% coverage of all Kraken-listed cryptocurrencies"""

    def __init__(self):
        self.logger = get_structured_logger("KrakenCoverageAuditor")
        self.daily_logger = get_daily_logger()

        # Initialize exchange
        self.exchange = ccxt.kraken(
            {
                "sandbox": True,
                "enableRateLimit": True,
            }
        )

        # Coverage requirements
        self.required_coverage = 1.0  # 100%
        self.coverage_report = {}
        self.audit_results = {}

    async def audit_full_coverage(self) -> Dict[str, Any]:
        """Perform complete coverage audit of all Kraken coins"""

        audit_start = datetime.utcnow()
        self.logger.info("Starting Kraken 100% coverage audit")

        try:
            # 1. Get ALL available markets from Kraken
            all_markets = await self.get_all_kraken_markets()

            # 2. Get currently monitored coins
            monitored_coins = await self.get_monitored_coins()

            # 3. Calculate coverage
            coverage_analysis = self.calculate_coverage(all_markets, monitored_coins)

            # 4. Identify missing coins
            missing_coins = coverage_analysis["missing_coins"]

            # 5. Generate compliance report
            compliance_report = self.generate_compliance_report(coverage_analysis)

            # 6. Take corrective action if needed
            if coverage_analysis["coverage_percentage"] < self.required_coverage:
                await self.enforce_full_coverage(missing_coins)

            audit_time = (datetime.utcnow() - audit_start).total_seconds()

            # Log audit results
            self.daily_logger.log_system_check(
                "kraken_coverage_audit",
                coverage_analysis["coverage_percentage"] >= self.required_coverage,
                f"Coverage: {coverage_analysis['coverage_percentage']:.1%}, Missing: {len(missing_coins)}",
            )

            audit_result = {
                "audit_timestamp": audit_start.isoformat(),
                "audit_duration_seconds": audit_time,
                "total_kraken_markets": len(all_markets),
                "monitored_markets": len(monitored_coins),
                "coverage_percentage": coverage_analysis["coverage_percentage"],
                "missing_coins": missing_coins,
                "compliance_status": "COMPLIANT"
                if coverage_analysis["coverage_percentage"] >= self.required_coverage
                else "NON_COMPLIANT",
                "corrective_actions_taken": len(missing_coins)
                if coverage_analysis["coverage_percentage"] < self.required_coverage
                else 0,
                "detailed_analysis": coverage_analysis,
            }

            # Save audit report
            await self.save_audit_report(audit_result)

            self.logger.info(
                f"Coverage audit completed: {coverage_analysis['coverage_percentage']:.1%} coverage"
            )

            return audit_result

        except Exception as e:
            self.logger.error(f"Coverage audit failed: {e}")
            self.daily_logger.log_error_with_context(e, {"audit_type": "kraken_coverage"})
            return {
                "audit_timestamp": audit_start.isoformat(),
                "compliance_status": "AUDIT_FAILED",
                "error": str(e),
            }

    async def get_all_kraken_markets(self) -> List[str]:
        """Get complete list of all Kraken markets"""

        try:
            # Load markets from exchange
            markets = await asyncio.to_thread(self.exchange.load_markets)

            # Filter for active USD/EUR pairs
            active_markets = []
            for symbol, market_info in markets.items():
                if market_info.get("active", True) and ("USD" in symbol or "EUR" in symbol):
                    active_markets.append(symbol)

            self.logger.info(f"Found {len(active_markets)} active Kraken markets")
            return sorted(active_markets)

        except Exception as e:
            self.logger.error(f"Failed to get Kraken markets: {e}")
            # Fallback with known major pairs
            return [
                "BTC/USD",
                "ETH/USD",
                "ADA/USD",
                "DOT/USD",
                "LINK/USD",
                "LTC/USD",
                "XRP/USD",
                "BCH/USD",
                "XLM/USD",
                "ETC/USD",
                "ATOM/USD",
                "ALGO/USD",
                "TRX/USD",
                "MATIC/USD",
                "AVAX/USD",
            ]

    async def get_monitored_coins(self) -> Set[str]:
        """Get list of currently monitored coins"""

        try:
            # Check monitoring config files
            config_path = Path("config/monitored_coins.json")
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    return set(config.get("monitored_symbols", []))

            # Check data directory for recent data files
            data_dir = Path("data/market_data")
            monitored = set()

            if data_dir.exists():
                for file_path in data_dir.glob("*.json"):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if "symbol" in data:
                                monitored.add(data["symbol"])
                            elif "symbols" in data:
                                monitored.update(data["symbols"])
                    except Exception:
                        continue

            return monitored

        except Exception as e:
            self.logger.error(f"Failed to get monitored coins: {e}")
            return set()

    def calculate_coverage(
        self, all_markets: List[str], monitored_coins: Set[str]
    ) -> Dict[str, Any]:
        """Calculate coverage statistics"""

        all_markets_set = set(all_markets)

        # Calculate coverage
        covered_markets = all_markets_set.intersection(monitored_coins)
        missing_markets = all_markets_set - monitored_coins
        extra_monitored = monitored_coins - all_markets_set

        coverage_percentage = len(covered_markets) / len(all_markets_set) if all_markets_set else 0

        return {
            "total_markets": len(all_markets_set),
            "monitored_markets": len(monitored_coins),
            "covered_markets": len(covered_markets),
            "missing_markets": len(missing_markets),
            "extra_monitored": len(extra_monitored),
            "coverage_percentage": coverage_percentage,
            "missing_coins": sorted(list(missing_markets)),
            "extra_coins": sorted(list(extra_monitored)),
            "covered_coins": sorted(list(covered_markets)),
        }

    def generate_compliance_report(self, coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed compliance report"""

        is_compliant = coverage_analysis["coverage_percentage"] >= self.required_coverage

        report = {
            "compliance_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "required_coverage": self.required_coverage,
            "actual_coverage": coverage_analysis["coverage_percentage"],
            "coverage_gap": max(
                0, self.required_coverage - coverage_analysis["coverage_percentage"]
            ),
            "critical_issues": [],
            "recommendations": [],
        }

        # Add critical issues
        if not is_compliant:
            report["critical_issues"].append(
                f"Coverage below required {self.required_coverage:.1%}"
            )
            report["critical_issues"].append(
                f"{len(coverage_analysis['missing_coins'])} coins not monitored"
            )

        # Add recommendations
        if coverage_analysis["missing_coins"]:
            report["recommendations"].append("Add missing coins to monitoring system")
            report["recommendations"].append("Update data collection configuration")

        if coverage_analysis["extra_coins"]:
            report["recommendations"].append("Review extra monitored coins for validity")

        return report

    async def enforce_full_coverage(self, missing_coins: List[str]) -> None:
        """Enforce 100% coverage by adding missing coins"""

        try:
            self.logger.info(f"Enforcing coverage: adding {len(missing_coins)} missing coins")

            # Update monitoring configuration
            config_path = Path("config/monitored_coins.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Get current monitored coins
            current_monitored = await self.get_monitored_coins()

            # Add missing coins
            updated_monitored = list(current_monitored.union(set(missing_coins)))

            # Save updated configuration
            config = {
                "monitored_symbols": updated_monitored,
                "last_updated": datetime.utcnow().isoformat(),
                "auto_updated_by": "coverage_auditor",
                "total_symbols": len(updated_monitored),
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            self.logger.info(f"Updated monitoring config: {len(updated_monitored)} total symbols")

            # Log corrective action
            self.daily_logger.log_system_check(
                "coverage_enforcement",
                True,
                f"Added {len(missing_coins)} missing coins to achieve 100% coverage",
            )

        except Exception as e:
            self.logger.error(f"Coverage enforcement failed: {e}")
            self.daily_logger.log_error_with_context(e, {"action": "coverage_enforcement"})

    async def save_audit_report(self, audit_result: Dict[str, Any]) -> None:
        """Save audit report to file"""

        try:
            # Create reports directory
            reports_dir = Path("logs/coverage_audits")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed report
            report_file = (
                reports_dir / f"coverage_audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(audit_result, f, indent=2)

            # Update latest report
            latest_file = reports_dir / "latest_coverage_audit.json"
            with open(latest_file, "w") as f:
                json.dump(audit_result, f, indent=2)

            self.logger.info(f"Audit report saved: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to save audit report: {e}")


# Daily audit scheduler
async def run_daily_coverage_audit():
    """Run daily coverage audit"""

    auditor = KrakenCoverageAuditor()
    audit_result = await auditor.audit_full_coverage()

    return audit_result


if __name__ == "__main__":
    asyncio.run(run_daily_coverage_audit())
