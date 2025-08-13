#!/usr/bin/env python3
"""
Release Management Script - Automate versioning, tagging, and CHANGELOG updates
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class ReleaseManager:
    """
    Manage releases with semantic versioning and changelog updates
    """

    def __init__(self):
        self.repo_root = Path.cwd()
        self.changelog_path = self.repo_root / "CHANGELOG.md"
        self.pyproject_path = self.repo_root / "pyproject.toml"

    def get_current_version(self) -> Optional[str]:
        """Get current version from pyproject.toml"""
        try:
            with open(self.pyproject_path) as f:
                content = f.read()
                match = re.search(r'version = "([^"]+)"', content)
                return match.group(1) if match else None
        except FileNotFoundError:
            return None

    def get_latest_git_tag(self) -> Optional[str]:
        """Get the latest git tag"""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def validate_version_format(self, version: str) -> bool:
        """Validate semantic version format"""
        pattern = r"^v?\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$"
        return bool(re.match(pattern, version))

    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse version string into major, minor, patch"""
        # Remove 'v' prefix if present
        version = version.lstrip("v")

        # Handle pre-release versions
        if "-" in version:
            version = version.split("-")[0]

        parts = version.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def increment_version(self, current_version: str, bump_type: str) -> str:
        """Increment version based on bump type"""
        major, minor, patch = self.parse_version(current_version)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

        return f"{major}.{minor}.{patch}"

    def get_commits_since_tag(self, tag: str) -> List[str]:
        """Get commit messages since the last tag"""
        try:
            result = subprocess.run(
                ["git", "log", f"{tag}..HEAD", "--pretty=format:%s"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )

            if result.returncode == 0:
                return [line.strip() for line in result.stdout.split("\n") if line.strip()]
            return []
        except Exception:
            return []

    def categorize_commits(self, commits: List[str]) -> dict:
        """Categorize commits into conventional commit types"""
        categories = {
            "breaking": [],
            "feat": [],
            "fix": [],
            "docs": [],
            "style": [],
            "refactor": [],
            "perf": [],
            "test": [],
            "ci": [],
            "chore": [],
            "other": [],
        }

        for commit in commits:
            commit_lower = commit.lower()

            # Check for breaking changes
            if "breaking" in commit_lower or "!" in commit:
                categories["breaking"].append(commit)
            elif commit_lower.startswith("feat"):
                categories["feat"].append(commit)
            elif commit_lower.startswith("fix"):
                categories["fix"].append(commit)
            elif commit_lower.startswith("docs"):
                categories["docs"].append(commit)
            elif commit_lower.startswith("style"):
                categories["style"].append(commit)
            elif commit_lower.startswith("refactor"):
                categories["refactor"].append(commit)
            elif commit_lower.startswith("perf"):
                categories["perf"].append(commit)
            elif commit_lower.startswith("test"):
                categories["test"].append(commit)
            elif commit_lower.startswith("ci"):
                categories["ci"].append(commit)
            elif commit_lower.startswith("chore"):
                categories["chore"].append(commit)
            else:
                categories["other"].append(commit)

        return categories

    def suggest_version_bump(self, commit_categories: dict) -> str:
        """Suggest version bump type based on commits"""
        if commit_categories["breaking"]:
            return "major"
        elif commit_categories["feat"]:
            return "minor"
        elif any(commit_categories[cat] for cat in ["fix", "perf", "docs", "style", "refactor"]):
            return "patch"
        else:
            return "patch"  # Default to patch for any changes

    def update_pyproject_version(self, new_version: str) -> bool:
        """Update version in pyproject.toml"""
        try:
            with open(self.pyproject_path) as f:
                content = f.read()

            # Update version
            new_content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)

            with open(self.pyproject_path, "w") as f:
                f.write(new_content)

            return True
        except Exception as e:
            print(f"Error updating pyproject.toml: {e}")
            return False

    def update_changelog(self, version: str, commit_categories: dict) -> bool:
        """Update CHANGELOG.md with new version"""
        try:
            # Read current changelog
            if self.changelog_path.exists():
                with open(self.changelog_path) as f:
                    content = f.read()
            else:
                content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"

            # Create new entry
            date = datetime.now().strftime("%Y-%m-%d")
            new_entry = f"## [{version}] - {date}\n\n"

            # Add categories with commits
            if commit_categories["breaking"]:
                new_entry += "### ⚠️ BREAKING CHANGES\n"
                for commit in commit_categories["breaking"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["feat"]:
                new_entry += "### Added\n"
                for commit in commit_categories["feat"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["fix"]:
                new_entry += "### Fixed\n"
                for commit in commit_categories["fix"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["perf"]:
                new_entry += "### Performance\n"
                for commit in commit_categories["perf"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["docs"]:
                new_entry += "### Documentation\n"
                for commit in commit_categories["docs"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["refactor"] or commit_categories["style"]:
                new_entry += "### Changed\n"
                for commit in commit_categories["refactor"] + commit_categories["style"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["test"] or commit_categories["ci"]:
                new_entry += "### Development\n"
                for commit in commit_categories["test"] + commit_categories["ci"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            if commit_categories["other"]:
                new_entry += "### Other\n"
                for commit in commit_categories["other"]:
                    new_entry += f"- {commit}\n"
                new_entry += "\n"

            # Insert new entry after "## [Unreleased]" or at the beginning
            if "## [Unreleased]" in content:
                content = content.replace("## [Unreleased]", f"## [Unreleased]\n\n{new_entry}")
            else:
                # Find first ## heading and insert before it
                lines = content.split("\n")
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith("## "):
                        insert_index = i
                        break

                lines.insert(insert_index, new_entry.rstrip())
                content = "\n".join(lines)

            # Write updated changelog
            with open(self.changelog_path, "w") as f:
                f.write(content)

            return True
        except Exception as e:
            print(f"Error updating CHANGELOG.md: {e}")
            return False

    def create_git_tag(self, version: str, message: str = None) -> bool:
        """Create and push git tag"""
        try:
            tag_name = f"v{version}"
            tag_message = message or f"Release {tag_name}"

            # Create annotated tag
            result = subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message], cwd=self.repo_root
            )

            if result.returncode != 0:
                return False

            print(f"✅ Created git tag: {tag_name}")
            return True
        except Exception as e:
            print(f"Error creating git tag: {e}")
            return False

    def push_changes(self, tag: str) -> bool:
        """Push changes and tags to remote"""
        try:
            # Push commits
            result1 = subprocess.run(["git", "push"], cwd=self.repo_root)

            # Push tags
            result2 = subprocess.run(["git", "push", "--tags"], cwd=self.repo_root)

            return result1.returncode == 0 and result2.returncode == 0
        except Exception as e:
            print(f"Error pushing changes: {e}")
            return False

    def prepare_release(self, version: str = None, bump_type: str = None) -> bool:
        """Prepare a new release"""
        print("🚀 CryptoSmartTrader V2 Release Manager")
        print("=" * 50)

        # Get current version
        current_version = self.get_current_version()
        if not current_version:
            print("❌ Could not find current version in pyproject.toml")
            return False

        print(f"📦 Current version: {current_version}")

        # Get latest tag
        latest_tag = self.get_latest_git_tag()
        if latest_tag:
            print(f"🏷️  Latest tag: {latest_tag}")

        # Get commits since last tag
        commits = self.get_commits_since_tag(latest_tag) if latest_tag else []
        print(f"📝 Commits since last tag: {len(commits)}")

        # Categorize commits
        commit_categories = self.categorize_commits(commits)

        # Determine new version
        if version:
            if not self.validate_version_format(version):
                print(f"❌ Invalid version format: {version}")
                return False
            new_version = version.lstrip("v")
        else:
            if not bump_type:
                bump_type = self.suggest_version_bump(commit_categories)
                print(f"💡 Suggested bump type: {bump_type}")

            new_version = self.increment_version(current_version, bump_type)

        print(f"🔄 New version: {new_version}")

        # Show what will be included
        print("\n📋 Release Notes Preview:")
        for category, commits in commit_categories.items():
            if commits and category != "chore":
                print(f"  {category.title()}: {len(commits)} changes")

        # Confirm
        response = input(f"\n✅ Create release {new_version}? [y/N]: ")
        if response.lower() != "y":
            print("Release cancelled.")
            return False

        # Update files
        print(f"\n📝 Updating pyproject.toml...")
        if not self.update_pyproject_version(new_version):
            return False

        print(f"📝 Updating CHANGELOG.md...")
        if not self.update_changelog(new_version, commit_categories):
            return False

        # Commit changes
        print(f"📝 Committing changes...")
        try:
            subprocess.run(["git", "add", "pyproject.toml", "CHANGELOG.md"], cwd=self.repo_root)
            subprocess.run(["git", "commit", "-m", f"Release {new_version}"], cwd=self.repo_root)
        except Exception as e:
            print(f"Error committing changes: {e}")
            return False

        # Create tag
        print(f"🏷️  Creating git tag...")
        if not self.create_git_tag(new_version):
            return False

        print(f"\n🎉 Release {new_version} prepared successfully!")
        print(f"📝 Next steps:")
        print(f"1. Review changes: git log --oneline -5")
        print(f"2. Push to remote: git push && git push --tags")
        print(f"3. Create GitHub release from tag v{new_version}")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage CryptoSmartTrader V2 releases")
    parser.add_argument("--version", help="Specific version to release (e.g., 2.1.0)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="Version bump type")
    parser.add_argument("--push", action="store_true", help="Automatically push changes and tags")

    args = parser.parse_args()

    # Create release manager
    manager = ReleaseManager()

    # Prepare release
    success = manager.prepare_release(args.version, args.bump)

    if success and args.push:
        print(f"\n🚀 Pushing changes...")
        if manager.push_changes(f"v{args.version}" if args.version else ""):
            print(f"✅ Changes pushed successfully!")
        else:
            print(f"❌ Failed to push changes")
            return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
