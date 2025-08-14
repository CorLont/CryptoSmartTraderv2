#!/usr/bin/env python3
"""
Module Duplicate Resolver
Resolve the 1421 duplicate modules found in the codebase.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModuleDuplicateResolver:
    """Resolve module duplicates systematically."""

    def __init__(self):
        self.project_root = Path(".")
        self.keep_patterns = [
            "src/cryptosmarttrader/",  # Main source code
            "app_fixed_all_issues.py",  # Main app
            "attribution_demo.py",  # New attribution system
        ]

        self.remove_patterns = [
            "attached_assets/",
            "exports/",
            "experiments/",
            "backups/",
            "cache/",
            "mlartifacts/",
            "models/",
            "logs/",
        ]

    def find_duplicate_modules(self) -> Dict[str, List[str]]:
        """Find all duplicate modules."""
        module_files = {}

        for py_file in self.project_root.rglob("*.py"):
            if any(
                pattern in str(py_file) for pattern in [".git", "__pycache__", ".cache", "venv"]
            ):
                continue

            module_name = py_file.stem
            if module_name == "__init__":
                continue

            if module_name not in module_files:
                module_files[module_name] = []
            module_files[module_name].append(str(py_file))

        # Only return modules with duplicates
        duplicates = {name: files for name, files in module_files.items() if len(files) > 1}
        return duplicates

    def get_file_hash(self, filepath: str) -> str:
        """Get SHA256 hash of file content."""
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def prioritize_files(self, files: List[str]) -> List[str]:
        """Prioritize which files to keep vs remove."""
        prioritized = []

        # Sort by priority patterns
        for pattern in self.keep_patterns:
            matching = [f for f in files if pattern in f]
            prioritized.extend(matching)

        # Add remaining files
        remaining = [f for f in files if f not in prioritized]
        prioritized.extend(remaining)

        return prioritized

    def resolve_duplicates(self) -> Dict:
        """Resolve all duplicate modules."""
        duplicates = self.find_duplicate_modules()

        results = {
            "total_duplicates": len(duplicates),
            "files_removed": [],
            "files_kept": [],
            "errors": [],
        }

        for module_name, files in duplicates.items():
            try:
                # Skip critical modules
                if module_name in ["__main__", "main", "app", "config"]:
                    continue

                prioritized = self.prioritize_files(files)
                keep_file = prioritized[0]  # Keep the highest priority file
                remove_files = prioritized[1:]  # Remove the rest

                # Check if files are identical
                keep_hash = self.get_file_hash(keep_file)

                for remove_file in remove_files:
                    # Only remove if it's in a removable directory
                    if any(pattern in remove_file for pattern in self.remove_patterns):
                        try:
                            remove_hash = self.get_file_hash(remove_file)

                            # Remove file (or move to backup)
                            remove_path = Path(remove_file)
                            if remove_path.exists():
                                remove_path.unlink()
                                results["files_removed"].append(remove_file)
                                logger.info(f"Removed duplicate: {remove_file}")

                        except Exception as e:
                            results["errors"].append(f"Error removing {remove_file}: {e}")

                results["files_kept"].append(keep_file)

            except Exception as e:
                results["errors"].append(f"Error processing {module_name}: {e}")

        return results

    def cleanup_empty_directories(self) -> int:
        """Remove empty directories after cleanup."""
        removed_dirs = 0

        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        removed_dirs += 1
                        logger.info(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not remove directory {dir_path}: {e}")

        return removed_dirs


def main():
    """Main execution."""
    resolver = ModuleDuplicateResolver()

    print("🔍 ANALYZING MODULE DUPLICATES...")
    duplicates = resolver.find_duplicate_modules()
    print(f"Found {len(duplicates)} modules with duplicates")

    print("🧹 RESOLVING DUPLICATES...")
    results = resolver.resolve_duplicates()

    print("🗂️ CLEANING EMPTY DIRECTORIES...")
    empty_dirs = resolver.cleanup_empty_directories()

    print("\n📊 DUPLICATE RESOLUTION COMPLETE")
    print(f"Total modules with duplicates: {results['total_duplicates']}")
    print(f"Files removed: {len(results['files_removed'])}")
    print(f"Files kept: {len(results['files_kept'])}")
    print(f"Empty directories removed: {empty_dirs}")
    print(f"Errors: {len(results['errors'])}")

    if results["errors"]:
        print("\n⚠️ Errors encountered:")
        for error in results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")

    # Save report
    report_path = "MODULE_CLEANUP_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Module Duplicate Cleanup Report\n\n")
        f.write(f"- Total duplicates resolved: {results['total_duplicates']}\n")
        f.write(f"- Files removed: {len(results['files_removed'])}\n")
        f.write(f"- Files kept: {len(results['files_kept'])}\n")
        f.write(f"- Empty directories removed: {empty_dirs}\n")
        f.write(f"- Errors: {len(results['errors'])}\n\n")

        if results["files_removed"]:
            f.write("## Files Removed\n")
            for file in results["files_removed"]:
                f.write(f"- {file}\n")

    print(f"\n📄 Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
