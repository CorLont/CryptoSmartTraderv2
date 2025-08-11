#!/usr/bin/env python3
"""
Log Cleanup Script for CryptoSmartTrader V2
Automated log rotation and cleanup with configurable retention
"""

import os
import gzip
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


class LogCleaner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "files_processed": 0,
            "files_compressed": 0,
            "files_deleted": 0,
            "space_freed_mb": 0,
            "errors": []
        }
    
    def get_log_files(self, log_dir: Path, pattern: str = "*.log") -> List[Path]:
        """Get all log files matching pattern"""
        try:
            return list(log_dir.glob(pattern))
        except Exception as e:
            self.stats["errors"].append(f"Error scanning {log_dir}: {e}")
            return []
    
    def get_file_age_days(self, file_path: Path) -> float:
        """Get file age in days"""
        try:
            stat = file_path.stat()
            age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
            return age.total_seconds() / 86400
        except Exception as e:
            self.stats["errors"].append(f"Error getting age for {file_path}: {e}")
            return 0
    
    def compress_file(self, file_path: Path) -> bool:
        """Compress a log file with gzip"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            if compressed_path.exists():
                print(f"  ⚠️  Compressed file already exists: {compressed_path}")
                return False
            
            if self.dry_run:
                print(f"  [DRY RUN] Would compress: {file_path}")
                return True
            
            # Get original size
            original_size = file_path.stat().st_size
            
            # Compress file
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Verify compression
            if compressed_path.exists():
                compressed_size = compressed_path.stat().st_size
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                # Remove original file
                file_path.unlink()
                
                # Update stats
                self.stats["space_freed_mb"] += (original_size - compressed_size) / (1024 * 1024)
                
                print(f"  ✅ Compressed: {file_path.name} ({compression_ratio:.1f}% reduction)")
                return True
            else:
                self.stats["errors"].append(f"Compression failed for {file_path}")
                return False
                
        except Exception as e:
            self.stats["errors"].append(f"Error compressing {file_path}: {e}")
            return False
    
    def delete_file(self, file_path: Path) -> bool:
        """Delete a file"""
        try:
            if self.dry_run:
                print(f"  [DRY RUN] Would delete: {file_path}")
                return True
            
            # Get file size for stats
            file_size = file_path.stat().st_size
            
            # Delete file
            file_path.unlink()
            
            # Update stats
            self.stats["space_freed_mb"] += file_size / (1024 * 1024)
            
            print(f"  🗑️  Deleted: {file_path.name}")
            return True
            
        except Exception as e:
            self.stats["errors"].append(f"Error deleting {file_path}: {e}")
            return False
    
    def clean_directory(self, log_dir: Path, retain_days: int, compress_days: int) -> Dict:
        """Clean logs in a specific directory"""
        if not log_dir.exists():
            print(f"⚠️  Directory not found: {log_dir}")
            return {"processed": 0, "compressed": 0, "deleted": 0}
        
        print(f"\n📁 Processing directory: {log_dir}")
        print(f"   Retain: {retain_days} days, Compress: {compress_days} days")
        
        # Get all log files
        log_files = []
        for pattern in ["*.log", "*.json", "*.csv"]:
            log_files.extend(self.get_log_files(log_dir, pattern))
        
        if not log_files:
            print("   No log files found")
            return {"processed": 0, "compressed": 0, "deleted": 0}
        
        # Sort by modification time (oldest first)
        log_files.sort(key=lambda x: x.stat().st_mtime)
        
        dir_stats = {"processed": 0, "compressed": 0, "deleted": 0}
        
        for file_path in log_files:
            try:
                age_days = self.get_file_age_days(file_path)
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                self.stats["files_processed"] += 1
                dir_stats["processed"] += 1
                
                print(f"   📄 {file_path.name} (age: {age_days:.1f}d, size: {file_size_mb:.1f}MB)")
                
                if age_days > retain_days:
                    # Delete old files
                    if self.delete_file(file_path):
                        self.stats["files_deleted"] += 1
                        dir_stats["deleted"] += 1
                        
                elif age_days > compress_days and not file_path.suffix == '.gz':
                    # Compress intermediate files
                    if self.compress_file(file_path):
                        self.stats["files_compressed"] += 1
                        dir_stats["compressed"] += 1
                else:
                    print(f"      ⏳ Keeping (too recent)")
                
            except Exception as e:
                self.stats["errors"].append(f"Error processing {file_path}: {e}")
        
        return dir_stats
    
    def clean_cache_directories(self, max_size_mb: int = 1000) -> None:
        """Clean cache directories if they exceed size limit"""
        cache_dirs = ["cache", ".streamlit", "__pycache__"]
        
        print(f"\n🧹 Cleaning cache directories (max: {max_size_mb}MB each)")
        
        for cache_dir_name in cache_dirs:
            cache_dir = Path(cache_dir_name)
            if not cache_dir.exists():
                continue
            
            try:
                # Calculate directory size
                total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                total_size_mb = total_size / (1024 * 1024)
                
                print(f"   📁 {cache_dir_name}: {total_size_mb:.1f}MB")
                
                if total_size_mb > max_size_mb:
                    if self.dry_run:
                        print(f"      [DRY RUN] Would clean {cache_dir_name}")
                    else:
                        # Remove directory contents
                        for item in cache_dir.iterdir():
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        
                        self.stats["space_freed_mb"] += total_size_mb
                        print(f"      🧹 Cleaned {cache_dir_name}")
                else:
                    print(f"      ✅ Size OK")
                    
            except Exception as e:
                self.stats["errors"].append(f"Error cleaning {cache_dir_name}: {e}")
    
    def clean_temp_files(self) -> None:
        """Clean temporary files"""
        print(f"\n🗑️  Cleaning temporary files")
        
        temp_patterns = [
            "*.tmp",
            "*.temp", 
            "*~",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        temp_count = 0
        for pattern in temp_patterns:
            for temp_file in Path(".").rglob(pattern):
                try:
                    if temp_file.is_file():
                        if self.dry_run:
                            print(f"   [DRY RUN] Would delete: {temp_file}")
                        else:
                            file_size = temp_file.stat().st_size
                            temp_file.unlink()
                            self.stats["space_freed_mb"] += file_size / (1024 * 1024)
                            temp_count += 1
                            
                except Exception as e:
                    self.stats["errors"].append(f"Error deleting temp file {temp_file}: {e}")
        
        if temp_count > 0:
            print(f"   🗑️  Deleted {temp_count} temporary files")
        else:
            print(f"   ✅ No temporary files found")
    
    def print_summary(self) -> None:
        """Print cleanup summary"""
        print(f"\n📊 Cleanup Summary")
        print(f"=" * 40)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files compressed: {self.stats['files_compressed']}")
        print(f"Files deleted: {self.stats['files_deleted']}")
        print(f"Space freed: {self.stats['space_freed_mb']:.1f} MB")
        
        if self.stats["errors"]:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:5]:  # Show first 5
                print(f"   {error}")
            if len(self.stats["errors"]) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more")
        
        if self.dry_run:
            print(f"\n⚠️  DRY RUN MODE - No files were actually modified")


def main():
    parser = argparse.ArgumentParser(description="Clean and rotate CryptoSmartTrader V2 logs")
    parser.add_argument("--days", type=int, default=30, help="Retain logs for N days (default: 30)")
    parser.add_argument("--compress-days", type=int, default=7, help="Compress logs older than N days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--cache-max-mb", type=int, default=1000, help="Max cache size in MB (default: 1000)")
    parser.add_argument("--include-cache", action="store_true", help="Also clean cache directories")
    parser.add_argument("--include-temp", action="store_true", help="Also clean temporary files")
    
    args = parser.parse_args()
    
    print("🧹 CryptoSmartTrader V2 - Log Cleanup")
    print("=" * 50)
    print(f"Retain: {args.days} days")
    print(f"Compress: {args.compress_days} days")
    print(f"Dry run: {args.dry_run}")
    
    cleaner = LogCleaner(dry_run=args.dry_run)
    
    # Define log directories to clean
    log_directories = [
        Path("logs"),
        Path("logs/agents"),
        Path("logs/system"), 
        Path("logs/api"),
        Path("logs/metrics"),
        Path("exports"),
        Path("backups")
    ]
    
    # Clean log directories
    for log_dir in log_directories:
        cleaner.clean_directory(log_dir, args.days, args.compress_days)
    
    # Optional cache cleanup
    if args.include_cache:
        cleaner.clean_cache_directories(args.cache_max_mb)
    
    # Optional temp file cleanup
    if args.include_temp:
        cleaner.clean_temp_files()
    
    # Print summary
    cleaner.print_summary()
    
    # Return appropriate exit code
    if cleaner.stats["errors"]:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())