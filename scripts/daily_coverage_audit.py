#!/usr/bin/env python3
"""
Daily Coverage Audit Script
Automated daily job to verify 100% exchange coverage and alert on missing coins
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.coverage_audit_manager import get_coverage_auditor, CoverageStatus, AlertSeverity
from core.logging_manager import get_logger

async def run_coverage_audit(exchange: str = 'kraken', verbose: bool = False):
    """Run coverage audit for specified exchange"""
    
    logger = get_logger()
    auditor = get_coverage_auditor()
    
    print(f"🔍 Starting daily coverage audit for {exchange.upper()}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Run the audit
        audit_result = await auditor.run_daily_coverage_audit(exchange)
        
        # Print results
        print(f"📊 AUDIT RESULTS:")
        print(f"   Audit ID: {audit_result.audit_id}")
        print(f"   Exchange: {audit_result.exchange}")
        print(f"   Status: {audit_result.status.value.upper()}")
        print(f"   Coverage: {audit_result.coverage_percentage:.2%}")
        print(f"   Live Coins: {audit_result.total_live_coins}")
        print(f"   Analyzed: {audit_result.analyzed_coins}")
        print(f"   Missing: {audit_result.missing_coins}")
        print(f"   Duration: {audit_result.audit_duration_seconds:.2f}s")
        print()
        
        # Show coverage status
        if audit_result.status == CoverageStatus.COMPLETE:
            print("✅ COVERAGE STATUS: COMPLETE")
            print("   All live coins are being analyzed")
        elif audit_result.status == CoverageStatus.PARTIAL:
            print("⚠️  COVERAGE STATUS: PARTIAL")
            print(f"   Missing {audit_result.missing_coins} coins ({100 - audit_result.coverage_percentage * 100:.1f}% gap)")
        elif audit_result.status == CoverageStatus.CRITICAL:
            print("🚨 COVERAGE STATUS: CRITICAL")
            print(f"   Missing {audit_result.missing_coins} coins ({100 - audit_result.coverage_percentage * 100:.1f}% gap)")
        else:
            print("❌ COVERAGE STATUS: FAILED")
            print("   Audit could not be completed")
        
        print()
        
        # Show missing coins if any
        if audit_result.missing_coins > 0:
            print(f"🔍 MISSING COINS ({audit_result.missing_coins}):")
            
            if verbose:
                # Show all missing coins
                for symbol in audit_result.missing_coin_symbols:
                    print(f"   • {symbol}")
            else:
                # Show top 10 missing coins
                missing_to_show = audit_result.missing_coin_symbols[:10]
                for symbol in missing_to_show:
                    print(f"   • {symbol}")
                
                if len(audit_result.missing_coin_symbols) > 10:
                    print(f"   ... and {len(audit_result.missing_coin_symbols) - 10} more")
            
            print()
        
        # Show high-impact gaps
        high_impact_gaps = [g for g in audit_result.coverage_gaps if g.impact_score > 0.3]
        if high_impact_gaps:
            print(f"🎯 HIGH-IMPACT MISSING COINS ({len(high_impact_gaps)}):")
            for gap in high_impact_gaps[:5]:  # Top 5
                volume_str = f"${gap.volume_24h:,.0f}" if gap.volume_24h else "N/A"
                print(f"   • {gap.symbol} (Impact: {gap.impact_score:.2f}, Volume: {volume_str})")
            print()
        
        # Show new listings
        if audit_result.new_listings:
            print(f"🆕 NEW LISTINGS DETECTED ({len(audit_result.new_listings)}):")
            for symbol in audit_result.new_listings[:10]:  # Top 10
                print(f"   • {symbol}")
            if len(audit_result.new_listings) > 10:
                print(f"   ... and {len(audit_result.new_listings) - 10} more")
            print()
        
        # Show delisted coins
        if audit_result.delisted_coins:
            print(f"📤 DELISTED COINS DETECTED ({len(audit_result.delisted_coins)}):")
            for symbol in audit_result.delisted_coins:
                print(f"   • {symbol}")
            print()
        
        # Show active alerts
        active_alerts = [a for a in auditor.coverage_alerts if not a.resolved]
        if active_alerts:
            print(f"🚨 ACTIVE ALERTS ({len(active_alerts)}):")
            
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
            
            if critical_alerts:
                print("   CRITICAL:")
                for alert in critical_alerts[:3]:  # Top 3
                    print(f"     • {alert.message}")
            
            if warning_alerts:
                print("   WARNING:")
                for alert in warning_alerts[:3]:  # Top 3
                    print(f"     • {alert.message}")
            
            print()
        
        # Show recommendations
        recommendations = auditor._get_coverage_recommendations(audit_result)
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
            print()
        
        # Exit code based on status
        if audit_result.status == CoverageStatus.FAILED:
            print("❌ Audit failed - check logs for details")
            return 2
        elif audit_result.status == CoverageStatus.CRITICAL:
            print("🚨 Critical coverage gaps detected - immediate action required")
            return 1
        elif audit_result.status == CoverageStatus.PARTIAL:
            print("⚠️  Partial coverage - review missing coins")
            return 0
        else:
            print("✅ Coverage audit completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Coverage audit failed: {e}")
        print(f"❌ ERROR: Coverage audit failed: {e}")
        return 2

def print_usage_examples():
    """Print usage examples"""
    print("📖 USAGE EXAMPLES:")
    print()
    print("Basic audit:")
    print("  python scripts/daily_coverage_audit.py")
    print()
    print("Verbose output:")
    print("  python scripts/daily_coverage_audit.py --verbose")
    print()
    print("Specific exchange:")
    print("  python scripts/daily_coverage_audit.py --exchange binance")
    print()
    print("Automated daily run (cron):")
    print("  0 6 * * * /usr/bin/python3 /path/to/scripts/daily_coverage_audit.py")
    print()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Daily Coverage Audit - Verify 100% exchange coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/daily_coverage_audit.py                    # Basic audit
  python scripts/daily_coverage_audit.py --verbose          # Detailed output
  python scripts/daily_coverage_audit.py --exchange binance # Specific exchange
        """
    )
    
    parser.add_argument(
        '--exchange', 
        default='kraken',
        help='Exchange to audit (default: kraken)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including all missing coins'
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
    
    # Run the audit
    exit_code = await run_coverage_audit(args.exchange, args.verbose)
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Audit interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)