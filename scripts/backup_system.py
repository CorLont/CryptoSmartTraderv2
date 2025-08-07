"""
CryptoSmartTrader V2 - Automated Backup System
Enterprise-grade backup and recovery for models, data, and configurations
"""

import os
import json
import shutil
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class AutomatedBackupSystem:
    """Enterprise-grade automated backup and recovery system"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Backup configuration
        self.backup_config = {
            "backup_root": Path("backups"),
            "retention_days": 30,
            "max_backups": 100,
            "compress_backups": True,
            "backup_schedule": {
                "models": "daily",
                "data": "hourly", 
                "config": "on_change",
                "logs": "daily"
            }
        }
        
        # Backup categories and their sources
        self.backup_categories = {
            "models": {
                "sources": ["models/", "data/ml_models/"],
                "description": "ML models and training data",
                "priority": "high"
            },
            "configurations": {
                "sources": ["config.json", "config/", ".env.example"],
                "description": "System configurations",
                "priority": "critical"
            },
            "cache_data": {
                "sources": ["data/cache/", "data/processed/"],
                "description": "Processed and cached data",
                "priority": "medium"
            },
            "analysis_results": {
                "sources": ["data/analysis/", "data/reports/", "data/daily_reports/"],
                "description": "Analysis results and reports",
                "priority": "high"
            },
            "system_logs": {
                "sources": ["logs/"],
                "description": "System and error logs",
                "priority": "medium"
            },
            "user_data": {
                "sources": ["data/exports/", "data/user_preferences/"],
                "description": "User exports and preferences",
                "priority": "low"
            }
        }
        
        # Initialize backup system
        self._initialize_backup_system()
    
    def _initialize_backup_system(self):
        """Initialize backup directory structure"""
        try:
            # Create backup directories
            self.backup_config["backup_root"].mkdir(parents=True, exist_ok=True)
            
            for category in self.backup_categories:
                category_dir = self.backup_config["backup_root"] / category
                category_dir.mkdir(exist_ok=True)
            
            # Create backup manifest directory
            manifest_dir = self.backup_config["backup_root"] / "manifests"
            manifest_dir.mkdir(exist_ok=True)
            
            self.logger.info("Backup system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backup system: {e}")
    
    def create_full_backup(self, categories: List[str] = None) -> Dict[str, Any]:
        """Create a full system backup"""
        try:
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_session_id = f"full_backup_{backup_timestamp}"
            
            categories = categories or list(self.backup_categories.keys())
            
            backup_results = {
                "session_id": backup_session_id,
                "timestamp": backup_timestamp,
                "categories": {},
                "overall_status": "success",
                "total_size": 0,
                "duration": 0
            }
            
            start_time = datetime.now()
            
            for category in categories:
                self.logger.info(f"Backing up category: {category}")
                
                category_result = self._backup_category(category, backup_timestamp)
                backup_results["categories"][category] = category_result
                
                if category_result["status"] == "failed":
                    backup_results["overall_status"] = "partial"
                
                backup_results["total_size"] += category_result.get("size_bytes", 0)
            
            # Calculate duration
            end_time = datetime.now()
            backup_results["duration"] = (end_time - start_time).total_seconds()
            
            # Create backup manifest
            self._create_backup_manifest(backup_results)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Full backup completed: {backup_session_id}")
            return backup_results
            
        except Exception as e:
            self.logger.error(f"Full backup failed: {e}")
            return {"session_id": None, "overall_status": "failed", "error": str(e)}
    
    def _backup_category(self, category: str, timestamp: str) -> Dict[str, Any]:
        """Backup a specific category"""
        try:
            if category not in self.backup_categories:
                return {"status": "failed", "error": f"Unknown category: {category}"}
            
            category_config = self.backup_categories[category]
            backup_name = f"{category}_{timestamp}"
            
            if self.backup_config["compress_backups"]:
                backup_file = self.backup_config["backup_root"] / category / f"{backup_name}.zip"
                return self._create_compressed_backup(category_config["sources"], backup_file)
            else:
                backup_dir = self.backup_config["backup_root"] / category / backup_name
                return self._create_directory_backup(category_config["sources"], backup_dir)
                
        except Exception as e:
            self.logger.error(f"Category backup failed for {category}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _create_compressed_backup(self, sources: List[str], backup_file: Path) -> Dict[str, Any]:
        """Create compressed ZIP backup"""
        try:
            backed_up_files = 0
            total_size = 0
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for source in sources:
                    source_path = Path(source)
                    
                    if source_path.exists():
                        if source_path.is_file():
                            zipf.write(source_path, source_path.name)
                            backed_up_files += 1
                            total_size += source_path.stat().st_size
                        elif source_path.is_dir():
                            for file_path in source_path.rglob("*"):
                                if file_path.is_file():
                                    # Create relative path for archive
                                    archive_path = file_path.relative_to(source_path.parent)
                                    zipf.write(file_path, archive_path)
                                    backed_up_files += 1
                                    total_size += file_path.stat().st_size
            
            return {
                "status": "success",
                "backup_file": str(backup_file),
                "files_backed_up": backed_up_files,
                "size_bytes": total_size,
                "compressed_size": backup_file.stat().st_size if backup_file.exists() else 0
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _create_directory_backup(self, sources: List[str], backup_dir: Path) -> Dict[str, Any]:
        """Create directory-based backup"""
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backed_up_files = 0
            total_size = 0
            
            for source in sources:
                source_path = Path(source)
                
                if source_path.exists():
                    destination = backup_dir / source_path.name
                    
                    if source_path.is_file():
                        shutil.copy2(source_path, destination)
                        backed_up_files += 1
                        total_size += source_path.stat().st_size
                    elif source_path.is_dir():
                        shutil.copytree(source_path, destination, dirs_exist_ok=True)
                        
                        # Count files and calculate size
                        for file_path in destination.rglob("*"):
                            if file_path.is_file():
                                backed_up_files += 1
                                total_size += file_path.stat().st_size
            
            return {
                "status": "success",
                "backup_directory": str(backup_dir),
                "files_backed_up": backed_up_files,
                "size_bytes": total_size
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def _create_backup_manifest(self, backup_results: Dict[str, Any]):
        """Create backup manifest file"""
        try:
            manifest_file = (
                self.backup_config["backup_root"] / 
                "manifests" / 
                f"manifest_{backup_results['session_id']}.json"
            )
            
            manifest_data = {
                **backup_results,
                "created_at": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "cwd": str(Path.cwd())
                },
                "backup_config": {
                    "retention_days": self.backup_config["retention_days"],
                    "compressed": self.backup_config["compress_backups"]
                }
            }
            
            with open(manifest_file, 'w') as f:
                json.dump(manifest_data, f, indent=2, default=str)
            
            self.logger.info(f"Backup manifest created: {manifest_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create backup manifest: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_config["retention_days"])
            
            for category in self.backup_categories:
                category_dir = self.backup_config["backup_root"] / category
                
                if not category_dir.exists():
                    continue
                
                # Get all backup files/directories
                backups = []
                for item in category_dir.iterdir():
                    try:
                        # Extract timestamp from filename
                        if item.name.count('_') >= 2:
                            timestamp_str = '_'.join(item.name.split('_')[-2:]).replace('.zip', '')
                            backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            backups.append((backup_date, item))
                    except ValueError:
                        continue
                
                # Sort by date and remove old backups
                backups.sort(key=lambda x: x[0])
                
                # Remove backups older than retention period
                for backup_date, backup_path in backups:
                    if backup_date < cutoff_date:
                        if backup_path.is_file():
                            backup_path.unlink()
                        elif backup_path.is_dir():
                            shutil.rmtree(backup_path)
                        
                        self.logger.info(f"Removed old backup: {backup_path}")
                
                # Keep only max_backups most recent
                if len(backups) > self.backup_config["max_backups"]:
                    excess_backups = backups[:-self.backup_config["max_backups"]]
                    for _, backup_path in excess_backups:
                        if backup_path.exists():
                            if backup_path.is_file():
                                backup_path.unlink()
                            elif backup_path.is_dir():
                                shutil.rmtree(backup_path)
                            
                            self.logger.info(f"Removed excess backup: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")
    
    def restore_from_backup(self, session_id: str, categories: List[str] = None, 
                          target_directory: str = None) -> Dict[str, Any]:
        """Restore system from backup"""
        try:
            # Find backup manifest
            manifest_file = self.backup_config["backup_root"] / "manifests" / f"manifest_{session_id}.json"
            
            if not manifest_file.exists():
                return {"status": "failed", "error": f"Backup manifest not found: {session_id}"}
            
            # Load backup manifest
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            categories = categories or list(manifest_data["categories"].keys())
            target_dir = Path(target_directory) if target_directory else Path.cwd()
            
            restore_results = {
                "session_id": session_id,
                "categories": {},
                "overall_status": "success"
            }
            
            for category in categories:
                if category not in manifest_data["categories"]:
                    continue
                
                self.logger.info(f"Restoring category: {category}")
                
                category_result = self._restore_category(category, session_id, target_dir)
                restore_results["categories"][category] = category_result
                
                if category_result["status"] == "failed":
                    restore_results["overall_status"] = "partial"
            
            self.logger.info(f"Restore completed: {session_id}")
            return restore_results
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _restore_category(self, category: str, session_id: str, target_dir: Path) -> Dict[str, Any]:
        """Restore a specific category"""
        try:
            # Find backup files for the session
            category_dir = self.backup_config["backup_root"] / category
            backup_files = list(category_dir.glob(f"*{session_id.split('_')[-1]}*"))
            
            if not backup_files:
                return {"status": "failed", "error": f"No backup files found for {category}"}
            
            backup_file = backup_files[0]
            restored_files = 0
            
            if backup_file.suffix == '.zip':
                # Restore from ZIP
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(target_dir)
                    restored_files = len(zipf.namelist())
            else:
                # Restore from directory
                for source_file in backup_file.rglob("*"):
                    if source_file.is_file():
                        relative_path = source_file.relative_to(backup_file)
                        target_file = target_dir / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_file)
                        restored_files += 1
            
            return {
                "status": "success",
                "files_restored": restored_files,
                "backup_source": str(backup_file)
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def list_available_backups(self) -> Dict[str, Any]:
        """List all available backups"""
        try:
            manifests_dir = self.backup_config["backup_root"] / "manifests"
            available_backups = []
            
            if manifests_dir.exists():
                for manifest_file in manifests_dir.glob("manifest_*.json"):
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = json.load(f)
                        
                        backup_info = {
                            "session_id": manifest_data["session_id"],
                            "timestamp": manifest_data["timestamp"],
                            "created_at": manifest_data.get("created_at"),
                            "categories": list(manifest_data["categories"].keys()),
                            "total_size": manifest_data.get("total_size", 0),
                            "status": manifest_data["overall_status"]
                        }
                        available_backups.append(backup_info)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to read manifest {manifest_file}: {e}")
            
            # Sort by timestamp (newest first)
            available_backups.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {
                "status": "success",
                "backup_count": len(available_backups),
                "backups": available_backups
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup system statistics"""
        try:
            stats = {
                "backup_root": str(self.backup_config["backup_root"]),
                "total_backups": 0,
                "total_size_bytes": 0,
                "categories": {},
                "oldest_backup": None,
                "newest_backup": None
            }
            
            # Analyze each category
            for category in self.backup_categories:
                category_dir = self.backup_config["backup_root"] / category
                
                if not category_dir.exists():
                    continue
                
                category_stats = {
                    "backup_count": 0,
                    "total_size": 0,
                    "latest_backup": None
                }
                
                # Analyze backup files in category
                for backup_item in category_dir.iterdir():
                    if backup_item.is_file() or backup_item.is_dir():
                        category_stats["backup_count"] += 1
                        stats["total_backups"] += 1
                        
                        # Calculate size
                        if backup_item.is_file():
                            size = backup_item.stat().st_size
                        else:
                            size = sum(f.stat().st_size for f in backup_item.rglob("*") if f.is_file())
                        
                        category_stats["total_size"] += size
                        stats["total_size_bytes"] += size
                        
                        # Track latest backup
                        mtime = datetime.fromtimestamp(backup_item.stat().st_mtime)
                        if not category_stats["latest_backup"] or mtime > category_stats["latest_backup"]:
                            category_stats["latest_backup"] = mtime
                
                stats["categories"][category] = category_stats
            
            # Find oldest and newest backups across all categories
            all_times = []
            for cat_stats in stats["categories"].values():
                if cat_stats["latest_backup"]:
                    all_times.append(cat_stats["latest_backup"])
            
            if all_times:
                stats["oldest_backup"] = min(all_times).isoformat()
                stats["newest_backup"] = max(all_times).isoformat()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get backup statistics: {e}")
            return {"error": str(e)}

def main():
    """CLI interface for backup system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptoSmartTrader Backup System")
    parser.add_argument("action", choices=["backup", "restore", "list", "stats", "cleanup"])
    parser.add_argument("--categories", nargs="+", help="Categories to backup/restore")
    parser.add_argument("--session-id", help="Session ID for restore operation")
    parser.add_argument("--target-dir", help="Target directory for restore")
    
    args = parser.parse_args()
    
    # Initialize backup system
    backup_system = AutomatedBackupSystem()
    
    if args.action == "backup":
        result = backup_system.create_full_backup(args.categories)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "restore":
        if not args.session_id:
            print("Error: --session-id required for restore operation")
            return
        
        result = backup_system.restore_from_backup(
            args.session_id, 
            args.categories, 
            args.target_dir
        )
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "list":
        result = backup_system.list_available_backups()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "stats":
        result = backup_system.get_backup_statistics()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "cleanup":
        backup_system._cleanup_old_backups()
        print("Backup cleanup completed")

if __name__ == "__main__":
    main()