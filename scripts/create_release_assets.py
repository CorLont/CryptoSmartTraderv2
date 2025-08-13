#!/usr/bin/env python3
"""
Create Release Assets - Generate model and deployment artifacts for releases
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ReleaseAssetCreator:
    """
    Create release assets including models, configs, and deployment files
    """

    def __init__(self, version: str):
        self.version = version
        self.repo_root = Path.cwd()
        self.assets_dir = self.repo_root / f"release-assets-{version}"
        self.models_dir = self.repo_root / "models"

    def create_assets_directory(self):
        """Create clean assets directory"""
        if self.assets_dir.exists():
            shutil.rmtree(self.assets_dir)

        self.assets_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.assets_dir / "deployment").mkdir()
        (self.assets_dir / "models").mkdir()
        (self.assets_dir / "configs").mkdir()
        (self.assets_dir / "docs").mkdir()

        print(f"📁 Created assets directory: {self.assets_dir}")

    def copy_deployment_scripts(self):
        """Copy deployment scripts"""
        deployment_files = [
            "1_install_all_dependencies.bat",
            "2_start_background_services.bat",
            "3_start_dashboard.bat",
            "README_DEPLOYMENT.md",
            "README_INSTALLATION.md",
            "README_WINDOWS_DEPLOYMENT.md",
        ]

        deployment_dir = self.assets_dir / "deployment"

        for file_name in deployment_files:
            source = self.repo_root / file_name
            if source.exists():
                shutil.copy2(source, deployment_dir / file_name)
                print(f"✅ Copied: {file_name}")

        # Create installation guide
        self.create_installation_guide(deployment_dir)

    def create_installation_guide(self, deployment_dir: Path):
        """Create comprehensive installation guide"""
        guide_content = f"""# CryptoSmartTrader V2 {self.version} - Installation Guide

## Quick Start

### Windows Installation
1. Run `1_install_all_dependencies.bat` as Administrator
2. Configure your API keys in `.env` file
3. Run `2_start_background_services.bat`
4. Run `3_start_dashboard.bat`
5. Open http://localhost:5000 in your browser

### Manual Installation
1. Install Python 3.11 or higher
2. Install uv: `pip install uv`
3. Clone repository and navigate to directory
4. Run: `uv sync --all-extras`
5. Configure environment variables
6. Start services: `uv run python start_agents.py`
7. Start dashboard: `uv run streamlit run dashboards/main_dashboard.py --server.port 5000`

## Requirements

- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space
- **Network**: Internet connection for exchange APIs
- **OS**: Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)

## API Keys Required

You'll need API keys from supported exchanges:

- **Kraken**: KRAKEN_API_KEY, KRAKEN_SECRET
- **Binance**: BINANCE_API_KEY, BINANCE_SECRET (optional)

### Getting Kraken API Keys
1. Login to Kraken.com
2. Go to Settings > API
3. Create new API key with permissions:
   - Query Funds
   - Query Open Orders
   - Query Closed Orders
   - Query Ledger Entries
   - Export Data
4. Copy API Key and Private Key

## Configuration

Create `.env` file in project root:

```
# Kraken API (Required)
KRAKEN_API_KEY=your_api_key_here
KRAKEN_SECRET=your_secret_here

# Optional: Additional exchanges
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## Security Notes

- **Never share your API keys**
- Use read-only API keys when possible
- Enable IP restrictions on exchange API keys
- Monitor API usage regularly
- Keep software updated

## Troubleshooting

### Common Issues

1. **Port 5000 already in use**
   - Change port in dashboard command: `--server.port 5001`

2. **Permission denied on Windows**
   - Run scripts as Administrator
   - Check Windows Defender exclusions

3. **API connection errors**
   - Verify API keys are correct
   - Check internet connection
   - Verify exchange API permissions

4. **Memory errors**
   - Close other applications
   - Increase virtual memory
   - Use smaller dataset limits

### Getting Help

- Check logs in `logs/` directory
- Review system health dashboard
- Contact support with log files

## Advanced Configuration

See `configs/` directory for advanced settings:
- Trading parameters
- Risk management settings
- ML model configurations
- System performance tuning

---

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Version: {self.version}
"""

        with open(deployment_dir / "INSTALLATION_GUIDE.md", "w") as f:
            f.write(guide_content)

        print("✅ Created installation guide")

    def copy_configuration_files(self):
        """Copy configuration templates"""
        config_files = [".env.template", ".env.example", "config.json", "pyproject.toml"]

        configs_dir = self.assets_dir / "configs"

        for file_name in config_files:
            source = self.repo_root / file_name
            if source.exists():
                if file_name == "config.json":
                    # Copy as example
                    shutil.copy2(source, configs_dir / "config.example.json")
                else:
                    shutil.copy2(source, configs_dir / file_name)
                print(f"✅ Copied config: {file_name}")

        # Create production config template
        self.create_production_config(configs_dir)

    def create_production_config(self, configs_dir: Path):
        """Create production configuration template"""
        prod_config = {
            "version": self.version,
            "environment": "production",
            "logging": {
                "level": "INFO",
                "enable_json": True,
                "enable_trading_logs": True,
                "log_retention_days": 30,
            },
            "trading": {
                "confidence_threshold": 0.8,
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
                "enable_stop_loss": True,
                "enable_take_profit": True,
            },
            "exchanges": {
                "kraken": {"enabled": True, "rate_limit": 1.0, "timeout": 30},
                "binance": {"enabled": False, "rate_limit": 0.5, "timeout": 15},
            },
            "ml": {
                "retrain_interval_hours": 24,
                "model_backup_count": 5,
                "uncertainty_threshold": 0.15,
                "enable_drift_detection": True,
            },
            "system": {
                "health_check_interval": 300,
                "metric_retention_days": 90,
                "auto_restart_on_failure": True,
                "max_memory_usage_mb": 8192,
            },
        }

        with open(configs_dir / "production.config.json", "w") as f:
            json.dump(prod_config, f, indent=2)

        print("✅ Created production config template")

    def package_models(self):
        """Package trained models if available"""
        models_asset_dir = self.assets_dir / "models"

        # Check if models directory exists
        if not self.models_dir.exists():
            print("ℹ️  No models directory found - skipping model packaging")
            # Create README explaining model setup
            readme_content = f"""# Models Directory

CryptoSmartTrader V2 {self.version} uses machine learning models for prediction.

## Model Files Not Included

Pre-trained models are not included in this release for the following reasons:
- Large file sizes (>100MB per model)
- Models are trained on specific time periods
- Fresh training produces better results

## Setting Up Models

The system will automatically train models on first run:

1. Start the system normally
2. Wait for initial data collection (15-30 minutes)
3. Models will be trained automatically
4. Check `models/` directory for generated files

## Model Files Generated

- `1h_predictor.pkl` - 1-hour prediction model
- `24h_predictor.pkl` - 24-hour prediction model  
- `7d_predictor.pkl` - 7-day prediction model
- `sentiment_model.pkl` - Sentiment analysis model
- `regime_detector.pkl` - Market regime detection model

## Performance Notes

- Initial training may take 30-60 minutes
- Models improve with more data over time
- Automatic retraining occurs daily
- Monitor model performance in health dashboard

---

For pre-trained models or faster setup, contact support.
"""

            with open(models_asset_dir / "README.md", "w") as f:
                f.write(readme_content)

            return

        # Package available models
        model_files = list(self.models_dir.glob("*.pkl"))
        model_files.extend(self.models_dir.glob("*.joblib"))

        if model_files:
            print(f"📦 Packaging {len(model_files)} model files...")

            for model_file in model_files:
                shutil.copy2(model_file, models_asset_dir / model_file.name)
                print(f"✅ Packaged model: {model_file.name}")

            # Create model manifest
            manifest = {
                "version": self.version,
                "created": datetime.now().isoformat(),
                "models": [
                    {
                        "filename": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                    for f in model_files
                ],
            }

            with open(models_asset_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            print("✅ Created model manifest")
        else:
            print("ℹ️  No model files found")

    def copy_documentation(self):
        """Copy documentation files"""
        docs_asset_dir = self.assets_dir / "docs"

        doc_files = ["README.md", "CHANGELOG.md", "LICENSE"]

        for file_name in doc_files:
            source = self.repo_root / file_name
            if source.exists():
                shutil.copy2(source, docs_asset_dir / file_name)
                print(f"✅ Copied doc: {file_name}")

        # Copy report files if they exist
        report_files = list(self.repo_root.glob("*_REPORT.md"))
        for report_file in report_files:
            shutil.copy2(report_file, docs_asset_dir / report_file.name)
            print(f"✅ Copied report: {report_file.name}")

    def create_release_info(self):
        """Create release information file"""
        release_info = {
            "version": self.version,
            "release_date": datetime.now().isoformat(),
            "description": "CryptoSmartTrader V2 - Enterprise Cryptocurrency Trading Intelligence Platform",
            "requirements": {
                "python": ">=3.11",
                "memory_gb": 8,
                "storage_gb": 5,
                "network": "Internet connection required",
            },
            "features": [
                "Multi-agent cryptocurrency trading intelligence",
                "Real-time analysis of 1457+ cryptocurrencies",
                "Deep learning-powered price predictions",
                "Advanced sentiment analysis and whale detection",
                "Professional Streamlit dashboard interface",
                "Enterprise-grade risk management",
                "Comprehensive backtesting engine",
                "Production-ready monitoring and alerting",
            ],
            "installation": {
                "windows": "Run 1_install_all_dependencies.bat as Administrator",
                "manual": "Install Python 3.11+, run 'uv sync --all-extras'",
                "docker": "docker run -p 5000:5000 cryptosmarttrader:latest",
            },
            "api_keys_required": [
                "Kraken API Key (KRAKEN_API_KEY)",
                "Kraken Secret (KRAKEN_SECRET)",
                "Optional: Binance API credentials",
            ],
            "support": {
                "documentation": "docs/",
                "installation_guide": "deployment/INSTALLATION_GUIDE.md",
                "troubleshooting": "See docs/README.md",
            },
        }

        with open(self.assets_dir / "release_info.json", "w") as f:
            json.dump(release_info, f, indent=2)

        print("✅ Created release info")

    def create_archive(self) -> Path:
        """Create release archive"""
        archive_name = f"cryptosmarttrader-v{self.version}-release.zip"
        archive_path = self.repo_root / archive_name

        print(f"📦 Creating release archive: {archive_name}")

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.assets_dir.rglob("*"):
                if file_path.is_file():
                    arc_path = file_path.relative_to(self.assets_dir)
                    zf.write(file_path, arc_path)

        file_size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"✅ Archive created: {archive_path} ({file_size_mb:.1f} MB)")

        return archive_path

    def create_models_only_archive(self) -> Path:
        """Create models-only archive for separate distribution"""
        models_archive = f"cryptosmarttrader-v{self.version}-models.zip"
        models_archive_path = self.repo_root / models_archive

        models_asset_dir = self.assets_dir / "models"

        if not any(models_asset_dir.glob("*.pkl")) and not any(models_asset_dir.glob("*.joblib")):
            print("ℹ️  No models to package separately")
            return None

        print(f"📦 Creating models-only archive: {models_archive}")

        with zipfile.ZipFile(models_archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in models_asset_dir.rglob("*"):
                if file_path.is_file():
                    arc_path = file_path.relative_to(models_asset_dir)
                    zf.write(file_path, arc_path)

        file_size_mb = models_archive_path.stat().st_size / (1024 * 1024)
        print(f"✅ Models archive created: {models_archive_path} ({file_size_mb:.1f} MB)")

        return models_archive_path

    def generate_checksums(self, archives: List[Path]):
        """Generate SHA256 checksums for archives"""
        import hashlib

        checksums_file = self.repo_root / f"cryptosmarttrader-v{self.version}-checksums.txt"

        with open(checksums_file, "w") as f:
            f.write(f"# CryptoSmartTrader V2 {self.version} - SHA256 Checksums\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for archive_path in archives:
                if archive_path and archive_path.exists():
                    sha256_hash = hashlib.sha256()
                    with open(archive_path, "rb") as f_archive:
                        for byte_block in iter(lambda: f_archive.read(4096), b""):
                            sha256_hash.update(byte_block)

                    checksum = sha256_hash.hexdigest()
                    f.write(f"{checksum}  {archive_path.name}\n")
                    print(f"✅ Checksum: {archive_path.name}")

        print(f"✅ Checksums written to: {checksums_file}")

    def create_release_assets(self):
        """Create complete release assets"""
        print(f"🚀 Creating release assets for CryptoSmartTrader V2 {self.version}")
        print("=" * 60)

        # Create assets directory
        self.create_assets_directory()

        # Copy all components
        print("\n📋 Copying deployment files...")
        self.copy_deployment_scripts()

        print("\n⚙️  Copying configuration files...")
        self.copy_configuration_files()

        print("\n🧠 Packaging models...")
        self.package_models()

        print("\n📚 Copying documentation...")
        self.copy_documentation()

        print("\n📄 Creating release info...")
        self.create_release_info()

        # Create archives
        print("\n📦 Creating archives...")
        main_archive = self.create_archive()
        models_archive = self.create_models_only_archive()

        # Generate checksums
        print("\n🔐 Generating checksums...")
        archives = [main_archive]
        if models_archive:
            archives.append(models_archive)
        self.generate_checksums(archives)

        print(f"\n🎉 Release assets created successfully!")
        print(f"📁 Assets directory: {self.assets_dir}")
        print(f"📦 Main archive: {main_archive}")
        if models_archive:
            print(f"🧠 Models archive: {models_archive}")

        print(f"\n📝 Next steps:")
        print(f"1. Test installation from archive")
        print(f"2. Upload to GitHub releases")
        print(f"3. Update release notes")
        print(f"4. Announce release")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create CryptoSmartTrader V2 release assets")
    parser.add_argument("version", help="Version number (e.g., 2.1.0)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up assets directory after")

    args = parser.parse_args()

    # Validate version format
    import re

    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        print("❌ Invalid version format. Use semantic versioning (e.g., 2.1.0)")
        return 1

    # Create release assets
    creator = ReleaseAssetCreator(args.version)
    creator.create_release_assets()

    # Optional cleanup
    if args.cleanup:
        response = input(f"\n🗑️  Remove assets directory {creator.assets_dir}? [y/N]: ")
        if response.lower() == "y":
            shutil.rmtree(creator.assets_dir)
            print(f"✅ Cleaned up: {creator.assets_dir}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
