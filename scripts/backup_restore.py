#!/usr/bin/env python3
"""
Backup and Restore Scripts
Complete backup/restore for models, configs, logs, and data
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_backup(
    backup_name: str = None,
    include_logs: bool = True,
    include_data: bool = False,
    output_dir: str = "./backups",
) -> str:
    """Create complete system backup"""

    print("🔄 CREATING SYSTEM BACKUP")
    print("=" * 50)

    # Generate backup name if not provided
    if not backup_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"cryptosmarttrader_backup_{timestamp}"

    # Create backup directory
    backup_dir = Path(output_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_file = backup_dir / f"{backup_name}.zip"

    print(f"📦 Creating backup: {backup_file}")

    # Items to backup
    backup_items = {
        "configs": ["config.json", "config.json.backup", ".env.example"],
        "models": ["models/", "ml/", "model_backup/"],
        "core_system": ["core/", "orchestration/", "agents/", "api/"],
        "evaluation": ["eval/", "exports/", "cache/"],
        "scripts": ["scripts/", "*.py", "*.bat"],
        "documentation": ["*.md", "*.txt", "README*"],
        "data_schemas": ["data/", "dashboards/"],
    }

    if include_logs:
        backup_items["logs"] = ["logs/", "mlruns/", "mlartifacts/"]

    if include_data:
        backup_items["raw_data"] = ["data/raw/", "data/processed/"]

    # Create backup
    try:
        with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as backup_zip:
            # Backup metadata
            backup_metadata = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "include_logs": include_logs,
                "include_data": include_data,
                "version": "2.0",
                "system": "CryptoSmartTrader",
            }

            backup_zip.writestr("backup_metadata.json", json.dumps(backup_metadata, indent=2))

            # Backup each category
            total_files = 0

            for category, patterns in backup_items.items():
                print(f"   📁 Backing up {category}...")
                category_files = 0

                for pattern in patterns:
                    if pattern.endswith("/"):
                        # Directory
                        dir_path = Path(pattern)
                        if dir_path.exists():
                            for file_path in dir_path.rglob("*"):
                                if file_path.is_file():
                                    # Skip sensitive files
                                    if not _should_skip_file(file_path):
                                        arcname = str(file_path)
                                        backup_zip.write(file_path, arcname)
                                        category_files += 1
                    else:
                        # File pattern
                        for file_path in Path(".").glob(pattern):
                            if file_path.is_file() and not _should_skip_file(file_path):
                                backup_zip.write(file_path, str(file_path))
                                category_files += 1

                print(f"      {category_files} files backed up")
                total_files += category_files

            # Create system snapshot
            system_snapshot = _create_system_snapshot()
            backup_zip.writestr("system_snapshot.json", json.dumps(system_snapshot, indent=2))

        print(f"\n✅ Backup created successfully: {backup_file}")
        print(f"   Total files: {total_files}")
        print(f"   Backup size: {backup_file.stat().st_size / 1024 / 1024:.1f} MB")

        return str(backup_file)

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        raise


def restore_backup(
    backup_file: str,
    restore_dir: str = ".",
    restore_configs: bool = True,
    restore_models: bool = True,
    restore_logs: bool = False,
    dry_run: bool = False,
) -> bool:
    """Restore system from backup"""

    print("🔄 RESTORING SYSTEM BACKUP")
    print("=" * 50)

    backup_path = Path(backup_file)
    if not backup_path.exists():
        print(f"❌ Backup file not found: {backup_file}")
        return False

    print(f"📦 Restoring from: {backup_file}")

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")

    try:
        with zipfile.ZipFile(backup_path, "r") as backup_zip:
            # Read backup metadata
            try:
                metadata_content = backup_zip.read("backup_metadata.json")
                backup_metadata = json.loads(metadata_content)

                print(f"   Backup created: {backup_metadata.get('created_at')}")
                print(f"   Version: {backup_metadata.get('version')}")
                print(f"   System: {backup_metadata.get('system')}")

            except KeyError:
                print("⚠️ Backup metadata not found - proceeding with caution")
                backup_metadata = {}

            # List all files in backup
            all_files = backup_zip.namelist()
            print(f"   Total files in backup: {len(all_files)}")

            # Create restore plan
            restore_plan = _create_restore_plan(
                all_files, restore_configs, restore_models, restore_logs
            )

            if not restore_plan:
                print("❌ No files to restore based on selection")
                return False

            print(f"\n📋 Restore plan: {len(restore_plan)} files")

            # Show what will be restored
            categories = {}
            for file_path in restore_plan:
                category = _categorize_file(file_path)
                categories[category] = categories.get(category, 0) + 1

            for category, count in categories.items():
                print(f"   {category}: {count} files")

            if dry_run:
                print("\n🔍 Dry run completed - no files modified")
                return True

            # Confirm restoration
            response = input("\n❓ Proceed with restoration? (yes/no): ").lower().strip()
            if response not in ["yes", "y"]:
                print("❌ Restoration cancelled")
                return False

            # Create backup of current system before restore
            print("\n📦 Creating pre-restore backup...")
            try:
                pre_restore_backup = create_backup(
                    backup_name=f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    include_logs=False,
                    include_data=False,
                )
                print(f"   Pre-restore backup: {pre_restore_backup}")
            except Exception as e:
                print(f"⚠️ Pre-restore backup failed: {e}")
                proceed = input("Continue without pre-restore backup? (yes/no): ").lower().strip()
                if proceed not in ["yes", "y"]:
                    return False

            # Perform restoration
            print(f"\n🔄 Restoring files...")

            restored_files = 0
            restore_dir_path = Path(restore_dir)

            for file_path in restore_plan:
                try:
                    # Extract file
                    target_path = restore_dir_path / file_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    with backup_zip.open(file_path) as source:
                        with open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)

                    restored_files += 1

                    if restored_files % 100 == 0:
                        print(f"   Restored {restored_files} files...")

                except Exception as e:
                    print(f"⚠️ Failed to restore {file_path}: {e}")

        print(f"\n✅ Restoration completed: {restored_files} files restored")

        # Verify critical files
        print("\n🔍 Verifying restoration...")
        verification_result = _verify_restoration()

        if verification_result:
            print("✅ Restoration verification passed")
        else:
            print("⚠️ Restoration verification warnings - check system manually")

        return True

    except Exception as e:
        print(f"❌ Restoration failed: {e}")
        return False


def list_backups(backup_dir: str = "./backups") -> List[Dict[str, Any]]:
    """List available backups"""

    backup_path = Path(backup_dir)
    backups = []

    if not backup_path.exists():
        return backups

    for backup_file in backup_path.glob("*.zip"):
        try:
            # Read backup metadata
            with zipfile.ZipFile(backup_file, "r") as backup_zip:
                try:
                    metadata_content = backup_zip.read("backup_metadata.json")
                    metadata = json.loads(metadata_content)
                except KeyError:
                    metadata = {}

                file_count = len(backup_zip.namelist())

            backup_info = {
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_mb": backup_file.stat().st_size / 1024 / 1024,
                "created_at": metadata.get("created_at", "unknown"),
                "version": metadata.get("version", "unknown"),
                "include_logs": metadata.get("include_logs", False),
                "include_data": metadata.get("include_data", False),
                "file_count": file_count,
            }

            backups.append(backup_info)

        except Exception as e:
            print(f"⚠️ Could not read backup {backup_file}: {e}")

    # Sort by creation date (newest first)
    backups.sort(key=lambda x: x["created_at"], reverse=True)

    return backups


def _should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped during backup"""

    skip_patterns = [
        ".git/",
        "__pycache__/",
        ".pytest_cache/",
        "node_modules/",
        ".env",
        "*.pyc",
        "*.log",
        "temp/",
        "tmp/",
        ".DS_Store",
        "Thumbs.db",
        "*.tmp",
    ]

    file_str = str(file_path)

    for pattern in skip_patterns:
        if pattern in file_str or file_str.endswith(pattern.replace("*", "")):
            return True

    return False


def _create_system_snapshot() -> Dict[str, Any]:
    """Create system snapshot for backup metadata"""

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
    }

    # Check for key configuration files
    config_files = ["config.json", "pyproject.toml", "requirements.txt"]

    for config_file in config_files:
        if Path(config_file).exists():
            snapshot[f"{config_file}_exists"] = True
            if config_file.endswith(".json"):
                try:
                    with open(config_file, "r") as f:
                        snapshot[f"{config_file}_content"] = json.load(f)
                except Exception:
                    snapshot[f"{config_file}_content"] = "failed_to_read"

    return snapshot


def _create_restore_plan(
    all_files: List[str], restore_configs: bool, restore_models: bool, restore_logs: bool
) -> List[str]:
    """Create restoration plan based on user preferences"""

    restore_plan = []

    for file_path in all_files:
        category = _categorize_file(file_path)

        should_restore = False

        if category in ["core_system", "scripts", "documentation"]:
            should_restore = True  # Always restore core system
        elif category == "configs" and restore_configs:
            should_restore = True
        elif category == "models" and restore_models:
            should_restore = True
        elif category == "logs" and restore_logs:
            should_restore = True
        elif category in ["evaluation", "data_schemas"]:
            should_restore = True  # Always restore evaluation tools

        if should_restore:
            restore_plan.append(file_path)

    return restore_plan


def _categorize_file(file_path: str) -> str:
    """Categorize file for restoration planning"""

    path_lower = file_path.lower()

    if any(x in path_lower for x in ["config.json", ".env"]):
        return "configs"
    elif any(x in path_lower for x in ["models/", "ml/", "model_backup/"]):
        return "models"
    elif any(x in path_lower for x in ["logs/", "mlruns/", "mlartifacts/"]):
        return "logs"
    elif any(x in path_lower for x in ["core/", "orchestration/", "agents/", "api/"]):
        return "core_system"
    elif any(x in path_lower for x in ["eval/", "exports/", "cache/"]):
        return "evaluation"
    elif any(x in path_lower for x in ["scripts/", ".py", ".bat"]):
        return "scripts"
    elif any(x in path_lower for x in [".md", ".txt", "readme"]):
        return "documentation"
    elif any(x in path_lower for x in ["data/", "dashboards/"]):
        return "data_schemas"
    else:
        return "other"


def _verify_restoration() -> bool:
    """Verify restoration completed successfully"""

    critical_files = ["core/", "config.json", "pyproject.toml"]

    all_exist = True

    for file_path in critical_files:
        if not Path(file_path).exists():
            print(f"⚠️ Critical file missing: {file_path}")
            all_exist = False

    return all_exist


def main():
    """Command line interface for backup/restore"""

    parser = argparse.ArgumentParser(description="CryptoSmartTrader Backup/Restore Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create system backup")
    backup_parser.add_argument("--name", help="Backup name")
    backup_parser.add_argument("--output-dir", default="./backups", help="Output directory")
    backup_parser.add_argument("--include-logs", action="store_true", help="Include logs")
    backup_parser.add_argument("--include-data", action="store_true", help="Include raw data")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_file", help="Backup file to restore")
    restore_parser.add_argument("--restore-dir", default=".", help="Restore directory")
    restore_parser.add_argument(
        "--no-configs", action="store_true", help="Skip configuration files"
    )
    restore_parser.add_argument("--no-models", action="store_true", help="Skip model files")
    restore_parser.add_argument("--include-logs", action="store_true", help="Restore logs")
    restore_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be restored"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument("--backup-dir", default="./backups", help="Backup directory")

    args = parser.parse_args()

    if args.command == "backup":
        try:
            backup_file = create_backup(
                backup_name=args.name,
                include_logs=args.include_logs,
                include_data=args.include_data,
                output_dir=args.output_dir,
            )
            print(f"\n🎉 Backup completed: {backup_file}")
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            sys.exit(1)

    elif args.command == "restore":
        try:
            success = restore_backup(
                backup_file=args.backup_file,
                restore_dir=args.restore_dir,
                restore_configs=not args.no_configs,
                restore_models=not args.no_models,
                restore_logs=args.include_logs,
                dry_run=args.dry_run,
            )

            if success:
                print(f"\n🎉 Restoration completed successfully")
            else:
                print(f"\n❌ Restoration failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Restoration failed: {e}")
            sys.exit(1)

    elif args.command == "list":
        backups = list_backups(args.backup_dir)

        if not backups:
            print("No backups found")
        else:
            print(f"\n📋 Available backups ({len(backups)}):")
            print("-" * 80)

            for backup in backups:
                print(f"📦 {backup['filename']}")
                print(f"   Created: {backup['created_at']}")
                print(f"   Size: {backup['size_mb']:.1f} MB")
                print(f"   Files: {backup['file_count']}")
                print(f"   Logs: {'✓' if backup['include_logs'] else '✗'}")
                print(f"   Data: {'✓' if backup['include_data'] else '✗'}")
                print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
