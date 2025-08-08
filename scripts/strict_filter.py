#!/usr/bin/env python3
"""
Strict Filter - 80% Probability Gate + Sorting
Implements strict confidence filtering with hard gates for recommendations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger

def strict_toplist(
    pred_df: pd.DataFrame, 
    min_conf: float = 0.80, 
    horizon_col: str = "pred_30d", 
    conf_col: str = "conf_30d"
) -> pd.DataFrame:
    """
    Strict confidence filtering - shows only coins meeting confidence threshold
    
    Args:
        pred_df: DataFrame with columns: coin, pred_1h,... pred_30d, conf_1h,... conf_30d, timestamp
        min_conf: Minimum confidence threshold (default: 0.80)
        horizon_col: Column name for predicted returns
        conf_col: Column name for confidence scores
    
    Returns:
        Filtered and sorted DataFrame, empty if no coins meet threshold
    """
    logger = get_logger()
    
    if pred_df.empty:
        logger.warning("Empty prediction DataFrame provided")
        return pd.DataFrame()
    
    # Validate required columns exist
    required_cols = ['coin', horizon_col, conf_col, 'timestamp']
    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    logger.info(f"Applying strict filter: min_conf={min_conf:.1%}, horizon={horizon_col}")
    
    # Apply strict confidence filter
    df = pred_df.copy()
    initial_count = len(df)
    
    # Hard gate: only coins meeting confidence threshold
    df = df[df[conf_col] >= min_conf]
    filtered_count = len(df)
    
    # Sort by predicted return (descending)
    df = df.sort_values(horizon_col, ascending=False)
    
    logger.info(
        f"Strict filter results: {filtered_count}/{initial_count} coins passed "
        f"({filtered_count/initial_count*100:.1f}% pass rate)"
    )
    
    if df.empty:
        logger.warning(f"NO COINS meet confidence threshold {min_conf:.1%} - returning empty result")
    
    return df.reset_index(drop=True)

def multi_horizon_strict_filter(
    pred_df: pd.DataFrame,
    min_conf: float = 0.80,
    horizons: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply strict filtering across multiple horizons
    
    Args:
        pred_df: Predictions DataFrame
        min_conf: Minimum confidence threshold
        horizons: Dictionary mapping horizon names to (pred_col, conf_col) tuples
    
    Returns:
        Dictionary of filtered DataFrames per horizon
    """
    if horizons is None:
        horizons = {
            "1h": ("pred_1h", "conf_1h"),
            "24h": ("pred_24h", "conf_24h"), 
            "7d": ("pred_7d", "conf_7d"),
            "30d": ("pred_30d", "conf_30d")
        }
    
    logger = get_logger()
    results = {}
    
    for horizon_name, (pred_col, conf_col) in horizons.items():
        if pred_col in pred_df.columns and conf_col in pred_df.columns:
            filtered = strict_toplist(pred_df, min_conf, pred_col, conf_col)
            results[horizon_name] = filtered
            
            logger.info(f"Horizon {horizon_name}: {len(filtered)} coins passed strict filter")
        else:
            logger.warning(f"Horizon {horizon_name}: missing columns {pred_col} or {conf_col}")
            results[horizon_name] = pd.DataFrame()
    
    return results

def format_strict_output(df: pd.DataFrame, top_n: int = 10) -> str:
    """Format strict filter output for display"""
    
    if df.empty:
        return "🚫 NO RECOMMENDATIONS - No coins meet confidence threshold"
    
    output = [f"✅ STRICT FILTER RESULTS - Top {min(top_n, len(df))} recommendations:"]
    output.append("=" * 60)
    
    for i, row in df.head(top_n).iterrows():
        coin = row.get('coin', 'Unknown')
        pred_return = row.iloc[1] if len(row) > 1 else 0  # First prediction column
        confidence = row.iloc[2] if len(row) > 2 else 0   # First confidence column
        
        output.append(f"{i+1:2d}. {coin:<12} | Return: {pred_return:+6.1%} | Conf: {confidence:.1%}")
    
    if len(df) > top_n:
        output.append(f"... and {len(df) - top_n} more coins")
    
    return "\n".join(output)

def save_strict_results(
    results: Dict[str, pd.DataFrame],
    output_dir: str = "data/strict_filter"
) -> None:
    """Save strict filter results to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    for horizon, df in results.items():
        if not df.empty:
            filename = f"strict_filter_{horizon}_{timestamp}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
            
            logger = get_logger()
            logger.info(f"Saved {len(df)} {horizon} recommendations to {filepath}")

def main():
    """Main entry point for strict filter script"""
    
    parser = argparse.ArgumentParser(
        description="Strict Filter - 80% Probability Gate + Sorting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/strict_filter.py --input predictions.csv
  python scripts/strict_filter.py --input predictions.csv --confidence 0.85 --top 5
  python scripts/strict_filter.py --input predictions.csv --horizon 24h
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to predictions CSV file'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.80,
        help='Minimum confidence threshold (default: 0.80)'
    )
    
    parser.add_argument(
        '--horizon',
        choices=['1h', '24h', '7d', '30d', 'all'],
        default='all',
        help='Horizon to filter (default: all)'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top recommendations to display (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        help='Output directory for results (default: data/strict_filter)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load predictions
        print(f"📊 Loading predictions from: {args.input}")
        pred_df = pd.read_csv(args.input, parse_dates=['timestamp'])
        print(f"   Loaded {len(pred_df)} prediction records")
        
        # Apply strict filtering
        if args.horizon == 'all':
            results = multi_horizon_strict_filter(pred_df, args.confidence)
            
            print(f"\n🔍 MULTI-HORIZON STRICT FILTER (min_conf={args.confidence:.1%}):")
            print("=" * 70)
            
            for horizon, df in results.items():
                print(f"\n{horizon.upper()} HORIZON:")
                print(format_strict_output(df, args.top))
        
        else:
            # Single horizon
            horizon_map = {
                "1h": ("pred_1h", "conf_1h"),
                "24h": ("pred_24h", "conf_24h"), 
                "7d": ("pred_7d", "conf_7d"),
                "30d": ("pred_30d", "conf_30d")
            }
            
            pred_col, conf_col = horizon_map[args.horizon]
            df = strict_toplist(pred_df, args.confidence, pred_col, conf_col)
            results = {args.horizon: df}
            
            print(f"\n🔍 {args.horizon.upper()} HORIZON STRICT FILTER (min_conf={args.confidence:.1%}):")
            print(format_strict_output(df, args.top))
        
        # Save results if output specified
        if args.output:
            save_strict_results(results, args.output)
            print(f"\n💾 Results saved to: {args.output}")
        
        # Exit code based on whether any recommendations found
        total_recommendations = sum(len(df) for df in results.values())
        
        if total_recommendations == 0:
            print(f"\n🚫 NO RECOMMENDATIONS - No coins meet {args.confidence:.1%} confidence threshold")
            return 1
        else:
            print(f"\n✅ Found {total_recommendations} total recommendations across all horizons")
            return 0
            
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {args.input}")
        return 2
    except Exception as e:
        print(f"❌ Error: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())