#!/usr/bin/env python3
"""
Daily Coverage Audit Script
Automated daily audit of exchange coverage and data completeness
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.coverage_audit_system import (
    get_kraken_coverage_auditor, get_data_completeness_validator,
    CoverageStatus, DataCompletenessConfig
)
from core.async_data_manager import get_async_data_manager
from core.logging_manager import get_logger

async def run_daily_coverage_audit(exchange: str = "kraken", include_completeness: bool = True):
    """Run comprehensive daily coverage audit"""
    
    logger = get_logger()
    
    print(f"🔍 Daily Coverage Audit Starting")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏦 Exchange: {exchange.upper()}")
    print("=" * 70)
    
    try:
        # Run exchange coverage audit
        if exchange.lower() == "kraken":
            coverage_auditor = get_kraken_coverage_auditor()
            coverage_report = await coverage_auditor.run_coverage_audit()
        else:
            raise ValueError(f"Exchange {exchange} not supported yet")
        
        # Display coverage results
        print(f"📊 COVERAGE AUDIT RESULTS:")
        print(f"   Status: {coverage_report.status.value.upper()}")
        print(f"   Coverage: {coverage_report.coverage_percentage:.2f}%")
        print(f"   Tradeable Tickers: {coverage_report.total_tradeable_tickers}")
        print(f"   Covered Tickers: {coverage_report.covered_tickers}")
        print(f"   Missing Tickers: {coverage_report.missing_tickers}")
        print(f"   Audit Duration: {coverage_report.performance_metrics.get('audit_duration_minutes', 0):.1f} minutes")
        print()
        
        # Show ticker changes
        if coverage_report.ticker_changes:
            print(f"📈 TICKER CHANGES DETECTED:")
            
            change_counts = {}
            for change in coverage_report.ticker_changes:
                change_type = change.change_type.value
                if change_type not in change_counts:
                    change_counts[change_type] = 0
                change_counts[change_type] += 1
            
            for change_type, count in change_counts.items():
                print(f"   {change_type.replace('_', ' ').title()}: {count}")
            
            # Show high-impact changes
            high_impact_changes = [c for c in coverage_report.ticker_changes if c.impact_score > 0.5]
            if high_impact_changes:
                print(f"\n   HIGH IMPACT CHANGES:")
                for change in high_impact_changes[:10]:  # Show top 10
                    print(f"   - {change.symbol}: {change.change_type.value} (impact: {change.impact_score:.2f})")
            print()
        
        # Show data quality issues
        if coverage_report.data_quality_issues:
            print(f"⚠️  DATA QUALITY ISSUES:")
            
            severity_counts = {}
            for issue in coverage_report.data_quality_issues:
                severity = issue.get("severity", "unknown")
                if severity not in severity_counts:
                    severity_counts[severity] = 0
                severity_counts[severity] += 1
            
            for severity, count in severity_counts.items():
                print(f"   {severity.title()}: {count}")
            
            # Show critical issues
            critical_issues = [i for i in coverage_report.data_quality_issues if i.get("severity") == "critical"]
            if critical_issues:
                print(f"\n   CRITICAL ISSUES:")
                for issue in critical_issues:
                    print(f"   - {issue.get('description', 'Unknown issue')}")
            print()
        
        # Run data completeness validation if requested
        if include_completeness:
            print(f"🔬 RUNNING DATA COMPLETENESS VALIDATION...")
            
            # Get current data
            async_data_manager = await get_async_data_manager()
            current_data = await async_data_manager.batch_collect_all_exchanges()
            
            # Extract coin data
            exchange_data = current_data.get("exchanges", {}).get(exchange.lower(), {})
            coin_data = exchange_data.get("tickers", {})
            
            if coin_data:
                # Run completeness validation
                completeness_validator = get_data_completeness_validator()
                completeness_result = await completeness_validator.validate_completeness(coin_data)
                
                # Display completeness results
                print(f"📋 DATA COMPLETENESS RESULTS:")
                print(f"   Total Coins Evaluated: {completeness_result.total_coins_evaluated}")
                print(f"   Passed Validation: {completeness_result.passed_coins}")
                print(f"   Excluded: {completeness_result.excluded_coins}")
                print(f"   Pass Rate: {completeness_result.passed_coins / completeness_result.total_coins_evaluated * 100:.1f}%")
                print(f"   Average Completeness: {completeness_result.average_completeness:.1%}")
                print(f"   Processing Time: {completeness_result.processing_latency_minutes:.1f} minutes")
                print()
                
                # Show exclusion reasons
                if completeness_result.exclusion_reasons:
                    print(f"❌ EXCLUSION REASONS:")
                    for reason, count in completeness_result.exclusion_reasons.items():
                        print(f"   {reason}: {count} coins")
                    print()
                
                # Show feature availability
                print(f"📊 FEATURE AVAILABILITY:")
                for feature, availability in completeness_result.feature_availability.items():
                    status = "✅" if availability >= 95.0 else "⚠️" if availability >= 90.0 else "❌"
                    print(f"   {status} {feature}: {availability:.1f}%")
                print()
                
                # Check if completeness validation passed
                if completeness_result.processing_latency_minutes > 30:
                    print(f"⚠️  WARNING: Processing time {completeness_result.processing_latency_minutes:.1f} min > 30 min budget")
                
                if completeness_result.excluded_coins > (completeness_result.total_coins_evaluated * 0.02):
                    print(f"⚠️  WARNING: {completeness_result.excluded_coins} excluded coins > 2% threshold")
            
            else:
                print(f"❌ No coin data available for completeness validation")
        
        # Show recommendations
        if coverage_report.recommendations:
            print(f"💡 RECOMMENDATIONS:")
            for i, recommendation in enumerate(coverage_report.recommendations, 1):
                print(f"   {i}. {recommendation}")
            print()
        
        # Determine overall status
        overall_status = "PASSED"
        
        if coverage_report.status == CoverageStatus.INSUFFICIENT:
            overall_status = "FAILED"
            print(f"❌ AUDIT FAILED: Coverage below 95% threshold")
        elif coverage_report.status == CoverageStatus.ERROR:
            overall_status = "ERROR"
            print(f"❌ AUDIT ERROR: Unable to complete coverage audit")
        elif coverage_report.coverage_percentage < 99.0:
            overall_status = "WARNING"
            print(f"⚠️  AUDIT WARNING: Coverage below 99% target")
        else:
            print(f"✅ AUDIT PASSED: All coverage targets met")
        
        # Save summary report
        await save_daily_summary(coverage_report, completeness_result if include_completeness else None)
        
        print(f"\n📝 Daily audit report saved to data/coverage_reports/")
        print(f"🎯 Overall Status: {overall_status}")
        
        return 0 if overall_status in ["PASSED", "WARNING"] else 1
        
    except Exception as e:
        logger.error(f"Daily coverage audit failed: {e}")
        print(f"❌ AUDIT FAILED: {e}")
        return 2

async def save_daily_summary(coverage_report, completeness_result=None):
    """Save daily summary report"""
    
    summary = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "timestamp": datetime.now().isoformat(),
        "coverage_audit": {
            "exchange": coverage_report.exchange_name,
            "status": coverage_report.status.value,
            "coverage_percentage": coverage_report.coverage_percentage,
            "total_tradeable": coverage_report.total_tradeable_tickers,
            "covered": coverage_report.covered_tickers,
            "missing": coverage_report.missing_tickers,
            "ticker_changes_count": len(coverage_report.ticker_changes),
            "data_quality_issues_count": len(coverage_report.data_quality_issues),
            "audit_duration_minutes": coverage_report.performance_metrics.get('audit_duration_minutes', 0)
        }
    }
    
    if completeness_result:
        summary["completeness_validation"] = {
            "total_evaluated": completeness_result.total_coins_evaluated,
            "passed": completeness_result.passed_coins,
            "excluded": completeness_result.excluded_coins,
            "pass_rate": completeness_result.passed_coins / completeness_result.total_coins_evaluated * 100 if completeness_result.total_coins_evaluated > 0 else 0,
            "average_completeness": completeness_result.average_completeness,
            "processing_latency_minutes": completeness_result.processing_latency_minutes
        }
    
    # Save summary
    summary_dir = Path("data/coverage_reports/daily_summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = summary_dir / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

async def run_quick_coverage_check(exchange: str = "kraken"):
    """Run quick coverage check without full audit"""
    
    print(f"⚡ Quick Coverage Check - {exchange.upper()}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 40)
    
    try:
        if exchange.lower() == "kraken":
            coverage_auditor = get_kraken_coverage_auditor()
            
            # Get current coverage without full audit
            await coverage_auditor.initialize_exchange()
            tradeable_tickers = await coverage_auditor._get_all_tradeable_tickers()
            covered_tickers = await coverage_auditor._get_covered_tickers()
            
            coverage_percentage = (len(covered_tickers) / len(tradeable_tickers)) * 100 if tradeable_tickers else 0
            
            print(f"📊 Coverage: {coverage_percentage:.1f}%")
            print(f"📈 Tradeable: {len(tradeable_tickers)}")
            print(f"✅ Covered: {len(covered_tickers)}")
            print(f"❌ Missing: {len(tradeable_tickers) - len(covered_tickers)}")
            
            if coverage_percentage >= 99.0:
                print(f"✅ Status: TARGET MET")
            elif coverage_percentage >= 95.0:
                print(f"⚠️  Status: NEEDS IMPROVEMENT")
            else:
                print(f"❌ Status: INSUFFICIENT")
        
        return 0
        
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        return 1

def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Full daily audit:")
    print("  python scripts/daily_coverage_audit.py audit")
    print()
    print("Audit without completeness validation:")
    print("  python scripts/daily_coverage_audit.py audit --no-completeness")
    print()
    print("Quick coverage check:")
    print("  python scripts/daily_coverage_audit.py quick")
    print()
    print("Different exchange (future):")
    print("  python scripts/daily_coverage_audit.py audit --exchange binance")
    print()
    print("Automated scheduling (cron):")
    print("  # Daily audit at 6:00 AM")
    print("  0 6 * * * /usr/bin/python3 /path/to/scripts/daily_coverage_audit.py audit")
    print("  # Quick checks every 4 hours")
    print("  0 */4 * * * /usr/bin/python3 /path/to/scripts/daily_coverage_audit.py quick")
    print()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Daily Coverage Audit - Comprehensive exchange coverage and data hygiene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/daily_coverage_audit.py audit              # Full audit
  python scripts/daily_coverage_audit.py audit --no-completeness  # Coverage only
  python scripts/daily_coverage_audit.py quick              # Quick check
        """
    )
    
    parser.add_argument(
        'command',
        choices=['audit', 'quick'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--exchange',
        default='kraken',
        help='Exchange to audit (default: kraken)'
    )
    
    parser.add_argument(
        '--no-completeness',
        action='store_true',
        help='Skip data completeness validation'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show usage examples'
    )
    
    args = parser.parse_args()
    
    if args.examples:
        print_usage_examples()
        return 0
    
    # Execute command
    if args.command == 'audit':
        include_completeness = not args.no_completeness
        exit_code = await run_daily_coverage_audit(args.exchange, include_completeness)
    elif args.command == 'quick':
        exit_code = await run_quick_coverage_check(args.exchange)
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Coverage audit interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)