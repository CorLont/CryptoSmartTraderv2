#!/usr/bin/env python3
"""
Daily Logging Bundler
Ultra-compact daily metrics logging with automatic bundling and archival
"""

from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional
import gzip
import shutil
import os

def write_daily(metrics: dict, log_type: str = "daily_metrics") -> str:
    """
    Ultra-compact daily logging bundler
    Write metrics to daily directory with timestamp and latest file
    """
    
    d = datetime.datetime.utcnow()
    p = Path(f"logs/daily/{d:%Y%m%d}")
    p.mkdir(parents=True, exist_ok=True)
    
    # Write timestamped file
    timestamped_file = p / f"{log_type}_{d:%H%M%S}.json"
    timestamped_file.write_text(json.dumps(metrics, indent=2))
    
    # Write/update latest file
    latest_file = p / f"{log_type}_latest.json"
    latest_file.write_text(json.dumps(metrics, indent=2))
    
    return str(timestamped_file)

def bundle_daily_logs(date_str: Optional[str] = None, compress: bool = True) -> Dict[str, Any]:
    """
    Bundle all daily logs into a single archive
    """
    
    if date_str is None:
        date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
    
    daily_dir = Path(f"logs/daily/{date_str}")
    
    if not daily_dir.exists():
        return {'error': f'No logs found for date {date_str}'}
    
    # Collect all log files
    log_files = list(daily_dir.glob("*.json"))
    
    bundle_data = {
        'date': date_str,
        'bundle_created': datetime.datetime.utcnow().isoformat(),
        'total_files': len(log_files),
        'logs': {}
    }
    
    # Read and bundle all logs
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_content = json.load(f)
            
            bundle_data['logs'][log_file.name] = log_content
            
        except Exception as e:
            bundle_data['logs'][log_file.name] = {'error': str(e)}
    
    # Write bundle file
    bundle_file = daily_dir / f"daily_bundle_{date_str}.json"
    
    with open(bundle_file, 'w') as f:
        json.dump(bundle_data, f, indent=2)
    
    # Compress if requested
    if compress:
        bundle_compressed = daily_dir / f"daily_bundle_{date_str}.json.gz"
        
        with open(bundle_file, 'rb') as f_in:
            with gzip.open(bundle_compressed, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove uncompressed bundle
        bundle_file.unlink()
        
        bundle_info = {
            'bundle_file': str(bundle_compressed),
            'compressed': True,
            'original_size': bundle_file.stat().st_size if bundle_file.exists() else 0,
            'compressed_size': bundle_compressed.stat().st_size
        }
    else:
        bundle_info = {
            'bundle_file': str(bundle_file),
            'compressed': False,
            'size': bundle_file.stat().st_size
        }
    
    bundle_info.update({
        'date': date_str,
        'total_files_bundled': len(log_files),
        'bundle_created': datetime.datetime.utcnow().isoformat()
    })
    
    return bundle_info

def cleanup_old_daily_logs(retention_days: int = 30, keep_bundles: bool = True) -> Dict[str, Any]:
    """
    Clean up old daily logs while preserving bundles
    """
    
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
    
    logs_dir = Path("logs/daily")
    
    if not logs_dir.exists():
        return {'error': 'Daily logs directory not found'}
    
    cleanup_stats = {
        'retention_days': retention_days,
        'cutoff_date': cutoff_date.strftime("%Y%m%d"),
        'directories_processed': 0,
        'files_removed': 0,
        'bundles_kept': 0,
        'total_size_freed': 0
    }
    
    for date_dir in logs_dir.iterdir():
        if not date_dir.is_dir():
            continue
        
        try:
            # Parse directory name as date
            dir_date = datetime.datetime.strptime(date_dir.name, "%Y%m%d")
            
            if dir_date < cutoff_date:
                cleanup_stats['directories_processed'] += 1
                
                # Process files in old directory
                for log_file in date_dir.iterdir():
                    file_size = log_file.stat().st_size
                    
                    # Keep bundle files if requested
                    if keep_bundles and 'bundle' in log_file.name:
                        cleanup_stats['bundles_kept'] += 1
                        continue
                    
                    # Remove old log file
                    log_file.unlink()
                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['total_size_freed'] += file_size
                
                # Remove directory if empty
                if not any(date_dir.iterdir()):
                    date_dir.rmdir()
        
        except ValueError:
            # Skip directories that don't match date format
            continue
    
    return cleanup_stats

class DailyMetricsLogger:
    """
    Enhanced daily metrics logger with automatic bundling and cleanup
    """
    
    def __init__(self, auto_bundle: bool = True, auto_cleanup: bool = True, 
                 retention_days: int = 30):
        
        self.auto_bundle = auto_bundle
        self.auto_cleanup = auto_cleanup
        self.retention_days = retention_days
        
        # Create base directory
        Path("logs/daily").mkdir(parents=True, exist_ok=True)
    
    def log_metrics(self, metrics: Dict[str, Any], log_type: str = "metrics") -> str:
        """
        Log metrics with automatic bundling and cleanup
        """
        
        # Add metadata
        enhanced_metrics = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'log_type': log_type,
            'metrics': metrics
        }
        
        # Write daily log
        log_file = write_daily(enhanced_metrics, log_type)
        
        # Auto-bundle yesterday's logs
        if self.auto_bundle:
            self._auto_bundle_yesterday()
        
        # Auto-cleanup old logs
        if self.auto_cleanup:
            self._auto_cleanup()
        
        return log_file
    
    def _auto_bundle_yesterday(self):
        """
        Automatically bundle yesterday's logs
        """
        
        yesterday = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
        try:
            bundle_daily_logs(yesterday, compress=True)
        except Exception:
            # Fail silently for auto-bundling
            pass
    
    def _auto_cleanup(self):
        """
        Automatically cleanup old logs
        """
        
        try:
            cleanup_old_daily_logs(self.retention_days, keep_bundles=True)
        except Exception:
            # Fail silently for auto-cleanup
            pass
    
    def get_daily_summary(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of daily logs
        """
        
        if date_str is None:
            date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
        
        daily_dir = Path(f"logs/daily/{date_str}")
        
        if not daily_dir.exists():
            return {'error': f'No logs found for date {date_str}'}
        
        log_files = list(daily_dir.glob("*.json"))
        
        summary = {
            'date': date_str,
            'total_log_files': len(log_files),
            'log_types': {},
            'total_size_bytes': 0
        }
        
        # Categorize log files
        for log_file in log_files:
            file_size = log_file.stat().st_size
            summary['total_size_bytes'] += file_size
            
            # Extract log type from filename
            if '_' in log_file.stem:
                log_type = log_file.stem.split('_')[0]
            else:
                log_type = 'unknown'
            
            if log_type not in summary['log_types']:
                summary['log_types'][log_type] = {
                    'count': 0,
                    'size_bytes': 0
                }
            
            summary['log_types'][log_type]['count'] += 1
            summary['log_types'][log_type]['size_bytes'] += file_size
        
        return summary

# Global instance for easy access
_daily_logger = None

def get_daily_logger() -> DailyMetricsLogger:
    """
    Get singleton daily logger instance
    """
    global _daily_logger
    
    if _daily_logger is None:
        _daily_logger = DailyMetricsLogger()
    
    return _daily_logger

# Convenience functions
def log_daily_metrics(metrics: Dict[str, Any], log_type: str = "metrics") -> str:
    """
    Convenience function for daily metrics logging
    """
    return get_daily_logger().log_metrics(metrics, log_type)

def log_trading_session(session_data: Dict[str, Any]) -> str:
    """
    Log trading session data
    """
    return log_daily_metrics(session_data, "trading_session")

def log_system_health(health_data: Dict[str, Any]) -> str:
    """
    Log system health metrics
    """
    return log_daily_metrics(health_data, "system_health")

def log_confidence_gate(gate_data: Dict[str, Any]) -> str:
    """
    Log confidence gate results
    """
    return log_daily_metrics(gate_data, "confidence_gate")

if __name__ == "__main__":
    print("📝 TESTING DAILY LOGGING BUNDLER")
    print("=" * 50)
    
    # Test basic daily logging
    print("✏️ Testing basic daily logging...")
    
    sample_metrics = {
        'total_trades': 15,
        'win_rate': 0.73,
        'total_pnl': 2450.50,
        'sharpe_ratio': 1.85,
        'max_drawdown': 0.12
    }
    
    log_file = write_daily(sample_metrics, "test_metrics")
    print(f"   Logged to: {log_file}")
    
    # Test enhanced logger
    print("\n🔧 Testing DailyMetricsLogger...")
    
    logger = DailyMetricsLogger(auto_bundle=False, auto_cleanup=False)
    
    # Log different types of metrics
    metrics_types = [
        ("trading_session", {'trades': 5, 'pnl': 123.45}),
        ("system_health", {'cpu': 45.2, 'memory': 67.8, 'health_score': 85}),
        ("confidence_gate", {'passed': 12, 'failed': 88, 'pass_rate': 0.12})
    ]
    
    for log_type, metrics in metrics_types:
        log_file = logger.log_metrics(metrics, log_type)
        print(f"   {log_type}: {Path(log_file).name}")
    
    # Test daily summary
    print("\n📊 Testing daily summary...")
    
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    summary = logger.get_daily_summary(today)
    
    print(f"   Date: {summary['date']}")
    print(f"   Total files: {summary['total_log_files']}")
    print(f"   Total size: {summary['total_size_bytes']} bytes")
    
    for log_type, stats in summary['log_types'].items():
        print(f"   {log_type}: {stats['count']} files, {stats['size_bytes']} bytes")
    
    # Test bundling
    print(f"\n📦 Testing log bundling...")
    
    bundle_info = bundle_daily_logs(today, compress=True)
    
    if 'error' not in bundle_info:
        print(f"   Bundle created: {Path(bundle_info['bundle_file']).name}")
        print(f"   Files bundled: {bundle_info['total_files_bundled']}")
        if bundle_info['compressed']:
            print(f"   Compressed size: {bundle_info['compressed_size']} bytes")
    
    # Test convenience functions
    print(f"\n⚡ Testing convenience functions...")
    
    log_trading_session({'session_id': 'test_123', 'duration_minutes': 45})
    log_system_health({'uptime_hours': 24, 'error_count': 0})
    log_confidence_gate({'gate_status': 'OPEN', 'opportunities': 8})
    
    print("   Convenience functions tested")
    
    print("\n✅ Daily logging bundler test completed")