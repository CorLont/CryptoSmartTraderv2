#!/usr/bin/env python3
"""
Coverage Audit - Live Kraken Symbols vs Processed Symbols
Implements comprehensive coverage validation with diff reporting
"""

import ccxt
import json
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger

def get_kraken_symbols() -> List[str]:
    """Get all tradeable symbols from Kraken exchange"""
    
    logger = get_logger()
    
    try:
        logger.info("Fetching live Kraken symbols...")
        
        # Initialize Kraken exchange
        exchange = ccxt.kraken({
            'apiKey': '',  # Public data only
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
        })
        
        # Load markets
        markets = exchange.load_markets()
        
        # Filter for spot crypto pairs (no fiat, no futures)
        symbols = []
        for symbol, market in markets.items():
            # Include only spot crypto pairs with "/" separator, exclude complex instruments
            if (market.get('type') == 'spot' and 
                '/' in symbol and 
                ':' not in symbol and
                not symbol.endswith('.d') and  # Exclude dark pools
                market.get('active', False)):
                symbols.append(symbol)
        
        # Sort for consistent ordering
        symbols.sort()
        
        logger.info(f"Found {len(symbols)} tradeable Kraken symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to fetch Kraken symbols: {e}")
        return []

def load_processed_symbols(file_path: str) -> List[str]:
    """Load processed symbols from JSON file"""
    
    logger = get_logger()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            symbols = data
        elif isinstance(data, dict):
            # Try common key names
            symbols = data.get('symbols', data.get('processed_symbols', data.get('coins', [])))
        else:
            logger.error(f"Unknown JSON structure in {file_path}")
            return []
        
        # Ensure all symbols are strings
        symbols = [str(s) for s in symbols]
        
        logger.info(f"Loaded {len(symbols)} processed symbols from {file_path}")
        return sorted(symbols)
        
    except FileNotFoundError:
        logger.error(f"Processed symbols file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading processed symbols: {e}")
        return []

def perform_coverage_audit(processed_symbols: List[str]) -> Dict:
    """Perform comprehensive coverage audit"""
    
    logger = get_logger()
    
    # Get live symbols from Kraken
    live_symbols = get_kraken_symbols()
    
    if not live_symbols:
        logger.error("Failed to get live symbols from Kraken")
        return {
            "error": "Failed to fetch live symbols",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Convert to sets for set operations
    live_set = set(live_symbols)
    processed_set = set(processed_symbols)
    
    # Calculate differences
    missing_symbols = sorted(list(live_set - processed_set))  # In Kraken but not processed
    extra_symbols = sorted(list(processed_set - live_set))    # Processed but not in Kraken
    covered_symbols = sorted(list(live_set & processed_set))  # In both
    
    # Calculate coverage percentage
    coverage_pct = round(100 * len(covered_symbols) / max(1, len(live_set)), 2)
    
    # Generate detailed report
    audit_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "exchange": "kraken",
        "coverage_summary": {
            "total_live_symbols": len(live_symbols),
            "total_processed_symbols": len(processed_symbols),
            "covered_symbols": len(covered_symbols),
            "missing_symbols": len(missing_symbols),
            "extra_symbols": len(extra_symbols),
            "coverage_percentage": coverage_pct
        },
        "symbol_analysis": {
            "missing_from_processing": missing_symbols,
            "extra_in_processing": extra_symbols,
            "successfully_covered": covered_symbols
        },
        "quality_assessment": {
            "coverage_status": "excellent" if coverage_pct >= 99 else 
                             "good" if coverage_pct >= 95 else
                             "poor" if coverage_pct >= 90 else "critical",
            "missing_high_volume": [],  # Could be enhanced with volume data
            "recommended_actions": []
        }
    }
    
    # Generate recommendations
    recommendations = []
    if missing_symbols:
        recommendations.append(f"Add {len(missing_symbols)} missing symbols to processing pipeline")
        if len(missing_symbols) <= 10:
            recommendations.append(f"Priority missing symbols: {', '.join(missing_symbols[:5])}")
    
    if extra_symbols:
        recommendations.append(f"Review {len(extra_symbols)} extra symbols - may be delisted")
    
    if coverage_pct < 95:
        recommendations.append("Coverage below 95% threshold - immediate action required")
    elif coverage_pct < 99:
        recommendations.append("Coverage below 99% target - improvement needed")
    else:
        recommendations.append("Coverage target achieved - maintain monitoring")
    
    audit_report["quality_assessment"]["recommended_actions"] = recommendations
    
    logger.info(
        f"Coverage audit completed: {coverage_pct:.1f}% coverage "
        f"({len(covered_symbols)}/{len(live_symbols)} symbols)"
    )
    
    return audit_report

def save_audit_report(report: Dict, output_dir: str = "logs/coverage") -> str:
    """Save audit report to timestamped file"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = report["timestamp"].replace(":", "-").replace(".", "-")
    filename = f"coverage_audit_{timestamp}.json"
    filepath = output_path / filename
    
    # Save report
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as latest
    latest_path = output_path / "latest_coverage_audit.json"
    with open(latest_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger = get_logger()
    logger.info(f"Coverage audit report saved: {filepath}")
    
    return str(filepath)

def print_audit_summary(report: Dict) -> None:
    """Print human-readable audit summary"""
    
    summary = report["coverage_summary"]
    analysis = report["symbol_analysis"]
    quality = report["quality_assessment"]
    
    print(f"📊 KRAKEN COVERAGE AUDIT REPORT")
    print(f"📅 Timestamp: {report['timestamp']}")
    print("=" * 60)
    
    # Coverage metrics
    print(f"📈 COVERAGE METRICS:")
    print(f"   Total Kraken symbols: {summary['total_live_symbols']}")
    print(f"   Processed symbols: {summary['total_processed_symbols']}")
    print(f"   Successfully covered: {summary['covered_symbols']}")
    print(f"   Coverage percentage: {summary['coverage_percentage']:.1f}%")
    
    # Status indicator
    status_icon = {
        "excellent": "✅",
        "good": "🟢", 
        "poor": "🟡",
        "critical": "🔴"
    }.get(quality["coverage_status"], "❓")
    
    print(f"   Status: {status_icon} {quality['coverage_status'].upper()}")
    print()
    
    # Missing symbols
    if analysis["missing_from_processing"]:
        print(f"❌ MISSING SYMBOLS ({len(analysis['missing_from_processing'])}):")
        missing = analysis["missing_from_processing"]
        
        # Show first 10 missing symbols
        for symbol in missing[:10]:
            print(f"   - {symbol}")
        
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")
        print()
    
    # Extra symbols
    if analysis["extra_in_processing"]:
        print(f"⚠️  EXTRA SYMBOLS ({len(analysis['extra_in_processing'])}):")
        extra = analysis["extra_in_processing"]
        
        # Show first 5 extra symbols
        for symbol in extra[:5]:
            print(f"   - {symbol} (possibly delisted)")
        
        if len(extra) > 5:
            print(f"   ... and {len(extra) - 5} more")
        print()
    
    # Recommendations
    if quality["recommended_actions"]:
        print(f"💡 RECOMMENDATIONS:")
        for i, action in enumerate(quality["recommended_actions"], 1):
            print(f"   {i}. {action}")
        print()

def create_sample_processed_file(output_path: str = "last_run_processed_symbols.json") -> None:
    """Create sample processed symbols file for testing"""
    
    logger = get_logger()
    
    try:
        # Get current Kraken symbols
        live_symbols = get_kraken_symbols()
        
        if not live_symbols:
            logger.error("Cannot create sample file - failed to fetch live symbols")
            return
        
        # REMOVED: Mock data pattern not allowed in production
        import random
        processed_count = int(len(live_symbols) * 0.95)
        processed_symbols = random.sample(live_symbols, processed_count)
        
        # Add a few fake symbols to test extra detection
        processed_symbols.extend(["FAKE/USD", "TEST/EUR", "DEMO/BTC"])
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(sorted(processed_symbols), f, indent=2)
        
        logger.info(f"Created sample processed symbols file: {output_path}")
        print(f"📝 Created sample file with {len(processed_symbols)} symbols: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create sample file: {e}")

def main():
    """Main entry point for coverage audit script"""
    
    parser = argparse.ArgumentParser(
        description="Coverage Audit - Live vs Processed Symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/coverage_audit.py --input last_run_processed_symbols.json
  python scripts/coverage_audit.py --input processed.json --output logs/custom
  python scripts/coverage_audit.py --create-sample  # Create test file
        """
    )
    
    parser.add_argument(
        '--input',
        default='last_run_processed_symbols.json',
        help='Path to processed symbols JSON file (default: last_run_processed_symbols.json)'
    )
    
    parser.add_argument(
        '--output',
        default='logs/coverage',
        help='Output directory for audit reports (default: logs/coverage)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample processed symbols file for testing'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        if args.create_sample:
            create_sample_processed_file(args.input)
            return 0
        
        # Load processed symbols
        if not args.quiet:
            print(f"📂 Loading processed symbols from: {args.input}")
        
        processed_symbols = load_processed_symbols(args.input)
        
        if not processed_symbols:
            print(f"❌ No processed symbols loaded from {args.input}")
            return 1
        
        # Perform audit
        if not args.quiet:
            print(f"🔍 Performing coverage audit against live Kraken data...")
        
        audit_report = perform_coverage_audit(processed_symbols)
        
        if "error" in audit_report:
            print(f"❌ Audit failed: {audit_report['error']}")
            return 2
        
        # Save report
        report_path = save_audit_report(audit_report, args.output)
        
        # Display results
        if not args.quiet:
            print_audit_summary(audit_report)
            print(f"📝 Full report saved to: {report_path}")
        
        # Determine exit code based on coverage
        coverage_pct = audit_report["coverage_summary"]["coverage_percentage"]
        
        if coverage_pct >= 99:
            if not args.quiet:
                print("✅ AUDIT PASSED: Coverage target achieved")
            return 0
        elif coverage_pct >= 95:
            if not args.quiet:
                print("⚠️  AUDIT WARNING: Coverage below target but acceptable")
            return 1
        else:
            if not args.quiet:
                print("❌ AUDIT FAILED: Coverage critically low")
            return 2
            
    except Exception as e:
        print(f"❌ Coverage audit error: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())