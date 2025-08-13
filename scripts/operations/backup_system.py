#!/usr/bin/env python3
"""
System Backup Script for CryptoSmartTrader V2
Creates comprehensive backups of system state, configurations, and data
"""

import os
import json
import shutil
import tarfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class SystemBackup:
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_name = f"system_backup_{self.timestamp}"

    def get_backup_manifest(self) -> Dict[str, Any]:
        """Create backup manifest with system information"""
        return {
            "backup_name": self.backup_name,
            "timestamp": self.timestamp,
            "version": "2.0.0",
            "type": "full_system",
            "includes": [
                "configuration_files",
                "logs",
                "cached_data",
                "model_files",
                "user_settings",
            ],
            "excludes": ["temporary_files", "large_datasets", "sensitive_credentials"],
        }

    def backup_configurations(self, backup_path: Path) -> None:
        """Backup configuration files"""
        config_dir = backup_path / "configs"
        config_dir.mkdir(exist_ok=True)

        config_files = [".env.example", "pyproject.toml", "config.json", "replit.md", ".replit"]

        for config_file in config_files:
            if Path(config_file).exists():
                shutil.copy2(config_file, config_dir)
                print(f"✓ Backed up: {config_file}")

    def backup_logs(self, backup_path: Path, days: int = 7) -> None:
        """Backup recent logs"""
        logs_backup = backup_path / "logs"

        if Path("logs").exists():
            # Create selective log backup (last N days)
            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)

            for log_file in Path("logs").rglob("*.log"):
                if log_file.stat().st_mtime > cutoff_time:
                    relative_path = log_file.relative_to("logs")
                    dest_file = logs_backup / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(log_file, dest_file)

            print(f"✓ Backed up recent logs ({days} days)")

    def backup_data(self, backup_path: Path) -> None:
        """Backup essential data files"""
        data_backup = backup_path / "data"
        data_backup.mkdir(exist_ok=True)

        # Backup essential data directories
        data_dirs = ["cache", "data", "models"]

        for data_dir in data_dirs:
            if Path(data_dir).exists():
                # Selective backup - only essential files
                for data_file in Path(data_dir).rglob("*"):
                    if data_file.is_file() and data_file.suffix in [".json", ".pkl", ".csv"]:
                        relative_path = data_file.relative_to(data_dir)
                        dest_file = data_backup / data_dir / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(data_file, dest_file)

                print(f"✓ Backed up: {data_dir}")

    def create_backup_archive(self, backup_path: Path) -> Path:
        """Create compressed backup archive"""
        archive_path = self.backup_dir / f"{self.backup_name}.tar.gz"

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=self.backup_name)

        # Calculate file hash
        hash_md5 = hashlib.md5()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        # Save hash file
        hash_file = archive_path.with_suffix(".tar.gz.md5")
        with open(hash_file, "w") as f:
            f.write(f"{hash_md5.hexdigest()}  {archive_path.name}\n")

        return archive_path

    def create_full_backup(self) -> Dict[str, Any]:
        """Create complete system backup"""
        print(f"🔄 Creating system backup: {self.backup_name}")

        # Create temporary backup directory
        temp_backup = self.backup_dir / f"temp_{self.backup_name}"
        temp_backup.mkdir(exist_ok=True)

        try:
            # Create manifest
            manifest = self.get_backup_manifest()
            with open(temp_backup / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Backup different components
            self.backup_configurations(temp_backup)
            self.backup_logs(temp_backup)
            self.backup_data(temp_backup)

            # Create compressed archive
            archive_path = self.create_backup_archive(temp_backup)

            # Cleanup temporary directory
            shutil.rmtree(temp_backup)

            # Get final stats
            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)

            result = {
                "success": True,
                "backup_file": str(archive_path),
                "size_mb": round(archive_size_mb, 2),
                "timestamp": self.timestamp,
                "manifest": manifest,
            }

            print(f"✅ Backup complete: {archive_path.name} ({archive_size_mb:.1f}MB)")
            return result

        except Exception as e:
            # Cleanup on error
            if temp_backup.exists():
                shutil.rmtree(temp_backup)

            return {"success": False, "error": str(e), "timestamp": self.timestamp}


def main():
    """Main backup execution"""
    print("💾 CryptoSmartTrader V2 - System Backup")
    print("=" * 50)

    backup_system = SystemBackup()
    result = backup_system.create_full_backup()

    if result["success"]:
        print(f"\n📊 Backup Summary:")
        print(f"File: {result['backup_file']}")
        print(f"Size: {result['size_mb']} MB")
        print(f"Time: {result['timestamp']}")
        return 0
    else:
        print(f"\n❌ Backup failed: {result['error']}")
        return 1


if __name__ == "__main__":
    exit(main())
