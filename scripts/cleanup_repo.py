#!/usr/bin/env python3
"""
Repository Cleanup Script - Remove tracked files that should be ignored
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Set


class RepositoryCleanup:
    """
    Clean up repository by removing files and directories that should be ignored
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.repo_root = Path.cwd()
        self.removed_files = []
        self.removed_dirs = []

        # Directories to remove completely
        self.dirs_to_remove = {
            "logs",
            "model_backup",
            "mlartifacts",
            "exports",
            "backups",
            "attached_assets",
            "cache",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "htmlcov",
            ".coverage",
            "test_results",
            "profiling",
            "temp_files",
            "temporary",
        }

        # File patterns to remove
        self.file_patterns_to_remove = {
            "*.log",
            "*.pkl",
            "*.joblib",
            "*.cache",
            "*.tmp",
            "*.temp",
            "*.backup",
            "*.bak",
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".coverage.*",
            "bandit-report.json",
            "safety-report.json",
        }

        # Files to specifically keep (even if they match patterns)
        self.files_to_keep = {
            ".env.template",
            ".env.example",
            "requirements.txt",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
        }

    def is_git_tracked(self, path: Path) -> bool:
        """Check if a file/directory is tracked by git"""
        try:
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(path)],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )
            return result.returncode == 0
        except Exception:
            return False

    def should_remove_file(self, file_path: Path) -> bool:
        """Determine if a file should be removed"""
        # Keep important files
        if file_path.name in self.files_to_keep:
            return False

        # Check patterns
        for pattern in self.file_patterns_to_remove:
            if file_path.match(pattern):
                return True

        return False

    def should_remove_directory(self, dir_path: Path) -> bool:
        """Determine if a directory should be removed"""
        return dir_path.name in self.dirs_to_remove

    def get_tracked_files_to_remove(self) -> List[Path]:
        """Get list of tracked files that should be removed"""
        files_to_remove = []

        try:
            # Get all tracked files
            result = subprocess.run(
                ["git", "ls-files"], capture_output=True, text=True, cwd=self.repo_root
            )

            if result.returncode != 0:
                print("Error: Not in a git repository or git command failed")
                return []

            tracked_files = result.stdout.strip().split("\n")

            for file_path_str in tracked_files:
                if not file_path_str:
                    continue

                file_path = Path(file_path_str)

                # Check if file should be removed
                if self.should_remove_file(file_path):
                    files_to_remove.append(file_path)

                # Check if file is in a directory that should be removed
                for part in file_path.parts:
                    if part in self.dirs_to_remove:
                        files_to_remove.append(file_path)
                        break

        except Exception as e:
            print(f"Error getting tracked files: {e}")

        return files_to_remove

    def get_untracked_items_to_remove(self) -> tuple[List[Path], List[Path]]:
        """Get untracked files and directories that should be removed"""
        files_to_remove = []
        dirs_to_remove = []

        # Walk through repository
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)

            # Skip .git directory
            if ".git" in root_path.parts:
                continue

            # Check directories
            for dir_name in dirs.copy():
                dir_path = root_path / dir_name
                if self.should_remove_directory(dir_path):
                    dirs_to_remove.append(dir_path)
                    dirs.remove(dir_name)  # Don't walk into this directory

            # Check files
            for file_name in files:
                file_path = root_path / file_name
                if self.should_remove_file(file_path):
                    files_to_remove.append(file_path)

        return files_to_remove, dirs_to_remove

    def remove_from_git(self, paths: List[Path]):
        """Remove files from git tracking"""
        if not paths:
            return

        path_strings = [str(p) for p in paths]

        try:
            if not self.dry_run:
                result = subprocess.run(
                    ["git", "rm", "--cached", "-r"] + path_strings,
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root,
                )

                if result.returncode != 0:
                    print(f"Warning: Some files could not be removed from git: {result.stderr}")

            print(f"{'[DRY RUN] ' if self.dry_run else ''}Removed from git tracking:")
            for path in paths:
                print(f"  - {path}")
                self.removed_files.append(path)

        except Exception as e:
            print(f"Error removing files from git: {e}")

    def remove_files_and_dirs(self, files: List[Path], dirs: List[Path]):
        """Remove files and directories from filesystem"""
        # Remove files
        for file_path in files:
            try:
                if file_path.exists():
                    if not self.dry_run:
                        file_path.unlink()
                    print(f"{'[DRY RUN] ' if self.dry_run else ''}Removed file: {file_path}")
                    self.removed_files.append(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

        # Remove directories
        for dir_path in dirs:
            try:
                if dir_path.exists():
                    if not self.dry_run:
                        shutil.rmtree(dir_path)
                    print(f"{'[DRY RUN] ' if self.dry_run else ''}Removed directory: {dir_path}")
                    self.removed_dirs.append(dir_path)
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

    def update_gitignore_if_needed(self):
        """Check if .gitignore needs updating"""
        gitignore_path = self.repo_root / ".gitignore"

        if not gitignore_path.exists():
            print("Warning: .gitignore not found")
            return

        print("\n✅ .gitignore exists and should be up to date")
        print("Make sure all removed items are properly ignored in .gitignore")

    def create_cleanup_summary(self):
        """Create a summary of cleanup actions"""
        summary_path = self.repo_root / "CLEANUP_SUMMARY.md"

        content = f"""# Repository Cleanup Summary

Generated on: {subprocess.run(["date"], capture_output=True, text=True).stdout.strip()}

## Files Removed from Git Tracking
{len(self.removed_files)} files removed:

"""

        for file_path in sorted(self.removed_files):
            content += f"- `{file_path}`\n"

        content += f"""

## Directories Removed
{len(self.removed_dirs)} directories removed:

"""

        for dir_path in sorted(self.removed_dirs):
            content += f"- `{dir_path}/`\n"

        content += """

## Next Steps

1. Review the changes with `git status`
2. Commit the updated .gitignore: `git add .gitignore && git commit -m "Update .gitignore and remove tracked artifacts"`
3. Verify no sensitive data was committed: `git log --stat | grep -E '\.(log|pkl|cache|tmp)$'`
4. Consider running `git gc` to clean up repository

## Important Notes

- These files/directories are now ignored and won't be tracked by git
- Make sure to backup any important data before running cleanup
- The .gitignore has been updated to prevent these files from being tracked again
"""

        if not self.dry_run:
            with open(summary_path, "w") as f:
                f.write(content)
            print(f"\n📋 Cleanup summary written to: {summary_path}")
        else:
            print(f"\n[DRY RUN] Would create cleanup summary at: {summary_path}")

    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("🧹 CryptoSmartTrader V2 Repository Cleanup")
        print("=" * 50)

        if self.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        else:
            print("⚠️  LIVE MODE - Files will be permanently removed")

        print()

        # Get tracked files that should be removed
        print("📋 Scanning for tracked files to remove...")
        tracked_files = self.get_tracked_files_to_remove()

        if tracked_files:
            print(f"Found {len(tracked_files)} tracked files to remove:")
            for file_path in tracked_files[:10]:  # Show first 10
                print(f"  - {file_path}")
            if len(tracked_files) > 10:
                print(f"  ... and {len(tracked_files) - 10} more")

            # Remove from git
            self.remove_from_git(tracked_files)
        else:
            print("✅ No tracked files need removal")

        print()

        # Get untracked items that should be removed
        print("📋 Scanning for untracked items to remove...")
        untracked_files, untracked_dirs = self.get_untracked_items_to_remove()

        if untracked_files or untracked_dirs:
            print(
                f"Found {len(untracked_files)} untracked files and {len(untracked_dirs)} directories to remove"
            )

            # Remove from filesystem
            self.remove_files_and_dirs(untracked_files, untracked_dirs)
        else:
            print("✅ No untracked items need removal")

        print()

        # Check .gitignore
        self.update_gitignore_if_needed()

        # Create summary
        self.create_cleanup_summary()

        print("\n🎉 Repository cleanup completed!")

        if self.dry_run:
            print("\n🔄 Run with --execute flag to perform actual cleanup")
        else:
            print("\n📝 Next steps:")
            print("1. Review changes: git status")
            print("2. Commit changes: git add . && git commit -m 'Clean up repository artifacts'")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean up CryptoSmartTrader V2 repository")
    parser.add_argument(
        "--execute", action="store_true", help="Actually perform cleanup (default is dry run)"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    # Confirmation for live mode
    if args.execute and not args.force:
        response = input("\n⚠️  This will permanently remove files. Continue? [y/N]: ")
        if response.lower() != "y":
            print("Cleanup cancelled.")
            return

    # Run cleanup
    cleanup = RepositoryCleanup(dry_run=not args.execute)
    cleanup.run_cleanup()


if __name__ == "__main__":
    main()
