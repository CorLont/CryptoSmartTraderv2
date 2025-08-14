#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Social Media API Secrets Setup
Enterprise-grade secrets management voor social media integrations
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import logging

def get_structured_logger(name):
    """Simple logger for setup script"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


class SocialMediaSecretsManager:
    """Manage social media API credentials securely"""
    
    def __init__(self):
        self.logger = get_structured_logger("SocialMediaSecretsManager")
        
        # Required secrets per platform
        self.required_secrets = {
            "reddit": {
                "REDDIT_CLIENT_ID": "Reddit API Client ID",
                "REDDIT_CLIENT_SECRET": "Reddit API Client Secret", 
                "REDDIT_USER_AGENT": "Reddit API User Agent (Format: app_name/version)"
            },
            "twitter": {
                "TWITTER_BEARER_TOKEN": "Twitter API v2 Bearer Token",
                "TWITTER_API_KEY": "Twitter API Key (Optional for v2)",
                "TWITTER_API_SECRET": "Twitter API Secret (Optional for v2)"
            },
            "telegram": {
                "TELEGRAM_BOT_TOKEN": "Telegram Bot API Token",
                "TELEGRAM_CHANNEL_ID": "Telegram Channel/Group ID to monitor"
            },
            "discord": {
                "DISCORD_BOT_TOKEN": "Discord Bot Token",
                "DISCORD_GUILD_ID": "Discord Server/Guild ID"
            }
        }
        
        # Platform-specific setup instructions
        self.setup_instructions = {
            "reddit": {
                "api_url": "https://www.reddit.com/prefs/apps",
                "steps": [
                    "1. Go to https://www.reddit.com/prefs/apps",
                    "2. Click 'Create App' or 'Create Another App'",
                    "3. Choose 'script' application type",
                    "4. Fill in name and description",
                    "5. Set redirect URI to http://localhost:8080",
                    "6. Copy the Client ID (under app name) and Client Secret"
                ],
                "rate_limits": "60 requests per minute",
                "tos_compliance": "Must use OAuth2, respect rate limits, no mass downloading"
            },
            "twitter": {
                "api_url": "https://developer.twitter.com/en/portal/dashboard",
                "steps": [
                    "1. Apply for Twitter Developer Account",
                    "2. Create a new App in the Developer Portal",
                    "3. Generate Bearer Token in App settings",
                    "4. Enable API v2 access",
                    "5. Review Twitter Developer Policy",
                    "6. Copy Bearer Token for API access"
                ],
                "rate_limits": "300 requests per 15 minutes (API v2)",
                "tos_compliance": "Commercial use allowed with proper license, respect rate limits"
            },
            "telegram": {
                "api_url": "https://t.me/BotFather",
                "steps": [
                    "1. Start chat with @BotFather on Telegram", 
                    "2. Send /newbot command",
                    "3. Follow instructions to create bot",
                    "4. Copy the Bot Token provided",
                    "5. Add bot to target channel/group",
                    "6. Get channel ID using @userinfobot"
                ],
                "rate_limits": "30 requests per second",
                "tos_compliance": "Public channels only, respect bot API limits"
            }
        }
    
    def check_existing_secrets(self) -> Dict[str, Dict[str, bool]]:
        """Check which social media secrets are already configured"""
        
        status = {}
        
        for platform, secrets in self.required_secrets.items():
            platform_status = {}
            
            for secret_key, description in secrets.items():
                # Check if secret exists in environment
                value = os.environ.get(secret_key)
                platform_status[secret_key] = {
                    "configured": bool(value),
                    "description": description,
                    "value_length": len(value) if value else 0
                }
            
            status[platform] = platform_status
        
        return status
    
    def generate_secrets_template(self) -> str:
        """Generate .env template for social media secrets"""
        
        template_lines = [
            "# CryptoSmartTrader V2 - Social Media API Secrets",
            "# Generated on: " + datetime.now().isoformat(),
            "",
            "# =============================================================================",
            "# SOCIAL MEDIA API CREDENTIALS",
            "# =============================================================================",
            "",
        ]
        
        for platform, secrets in self.required_secrets.items():
            instructions = self.setup_instructions.get(platform, {})
            
            template_lines.extend([
                f"# {platform.upper()} API Configuration",
                f"# Setup: {instructions.get('api_url', 'See documentation')}",
                f"# Rate Limits: {instructions.get('rate_limits', 'See platform documentation')}",
                f"# TOS: {instructions.get('tos_compliance', 'Review platform terms')}",
                ""
            ])
            
            for secret_key, description in secrets.items():
                template_lines.extend([
                    f"# {description}",
                    f"{secret_key}=",
                    ""
                ])
            
            template_lines.append("")
        
        template_lines.extend([
            "# =============================================================================",
            "# SOCIAL MEDIA COMPLIANCE SETTINGS",
            "# =============================================================================",
            "",
            "# Global rate limiting (requests per minute)",
            "SOCIAL_MEDIA_GLOBAL_RATE_LIMIT=100",
            "",
            "# Ban protection settings",
            "SOCIAL_MEDIA_BAN_PROTECTION_ENABLED=true",
            "SOCIAL_MEDIA_BACKOFF_STRATEGY=exponential",
            "",
            "# Data retention (days)", 
            "SOCIAL_MEDIA_DATA_RETENTION_DAYS=30",
            "",
            "# TOS compliance monitoring",
            "SOCIAL_MEDIA_TOS_COMPLIANCE_STRICT=true"
        ])
        
        return "\n".join(template_lines)
    
    def validate_secrets_format(self, platform: str) -> Dict[str, Any]:
        """Validate format of configured secrets for platform"""
        
        validation_results = {
            "platform": platform,
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        if platform not in self.required_secrets:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Unknown platform: {platform}")
            return validation_results
        
        platform_secrets = self.required_secrets[platform]
        
        for secret_key, description in platform_secrets.items():
            value = os.environ.get(secret_key)
            
            if not value:
                validation_results["issues"].append(f"Missing secret: {secret_key}")
                validation_results["valid"] = False
                continue
            
            # Platform-specific validation
            if platform == "reddit":
                if secret_key == "REDDIT_CLIENT_ID" and len(value) < 10:
                    validation_results["warnings"].append("Reddit Client ID seems too short")
                elif secret_key == "REDDIT_USER_AGENT" and "cryptosmarttrader" not in value.lower():
                    validation_results["warnings"].append("User-Agent should identify your application")
            
            elif platform == "twitter":
                if secret_key == "TWITTER_BEARER_TOKEN":
                    if not value.startswith("AAA"):
                        validation_results["warnings"].append("Twitter Bearer Token format may be invalid")
                    if len(value) < 100:
                        validation_results["warnings"].append("Twitter Bearer Token seems too short")
            
            elif platform == "telegram":
                if secret_key == "TELEGRAM_BOT_TOKEN":
                    if ":" not in value or len(value.split(":")) != 2:
                        validation_results["issues"].append("Telegram Bot Token format invalid (should be number:string)")
                        validation_results["valid"] = False
                elif secret_key == "TELEGRAM_CHANNEL_ID":
                    if not value.startswith("-") and not value.startswith("@"):
                        validation_results["warnings"].append("Telegram Channel ID should start with - or @")
        
        return validation_results
    
    def generate_setup_instructions(self, platform: str) -> str:
        """Generate detailed setup instructions for platform"""
        
        if platform not in self.setup_instructions:
            return f"No setup instructions available for platform: {platform}"
        
        instructions = self.setup_instructions[platform]
        
        instruction_text = f"""
# {platform.upper()} API Setup Instructions

## Quick Setup
{chr(10).join(instructions.get('steps', []))}

## API Information
- Developer Portal: {instructions.get('api_url')}
- Rate Limits: {instructions.get('rate_limits')}
- TOS Compliance: {instructions.get('tos_compliance')}

## Required Environment Variables
"""
        
        platform_secrets = self.required_secrets.get(platform, {})
        for secret_key, description in platform_secrets.items():
            instruction_text += f"- {secret_key}: {description}\n"
        
        instruction_text += f"""
## Testing Your Setup
```bash
# Test {platform} configuration
python -c "
from src.cryptosmarttrader.data.enterprise_social_media_ingestion import get_social_media_manager
import asyncio

async def test_{platform}():
    manager = get_social_media_manager()
    status = manager.get_compliance_status()
    print(f'{platform.title()} Status:', status)

asyncio.run(test_{platform}())
"
```

## Security Best Practices
1. Never commit API keys to version control
2. Use environment variables or secure secret storage
3. Regularly rotate API credentials
4. Monitor API usage and rate limits
5. Review platform TOS periodically

## Troubleshooting
- Check secret formats using the validation tool
- Verify API credentials in platform developer portal
- Ensure rate limits are respected
- Monitor compliance violations
"""
        
        return instruction_text
    
    def create_secrets_report(self) -> Dict[str, Any]:
        """Create comprehensive secrets status report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "platforms": {},
            "summary": {
                "total_platforms": len(self.required_secrets),
                "configured_platforms": 0,
                "missing_secrets": 0,
                "validation_issues": 0
            }
        }
        
        # Check each platform
        for platform in self.required_secrets.keys():
            secrets_status = self.check_existing_secrets()[platform]
            validation = self.validate_secrets_format(platform)
            
            configured_secrets = sum(1 for s in secrets_status.values() if s["configured"])
            total_secrets = len(secrets_status)
            
            platform_report = {
                "secrets_configured": f"{configured_secrets}/{total_secrets}",
                "fully_configured": configured_secrets == total_secrets,
                "validation_valid": validation["valid"],
                "validation_issues": len(validation["issues"]),
                "validation_warnings": len(validation["warnings"]),
                "secrets_details": secrets_status,
                "validation_details": validation
            }
            
            report["platforms"][platform] = platform_report
            
            # Update summary
            if platform_report["fully_configured"]:
                report["summary"]["configured_platforms"] += 1
            
            report["summary"]["missing_secrets"] += (total_secrets - configured_secrets)
            report["summary"]["validation_issues"] += len(validation["issues"])
        
        return report


def main():
    """Main setup process"""
    
    print("🔐 CryptoSmartTrader V2 - Social Media Secrets Setup")
    print("=" * 60)
    
    manager = SocialMediaSecretsManager()
    
    # Generate status report
    print("\n📊 Current Secrets Status:")
    report = manager.create_secrets_report()
    
    summary = report["summary"]
    print(f"Platforms Configured: {summary['configured_platforms']}/{summary['total_platforms']}")
    print(f"Missing Secrets: {summary['missing_secrets']}")
    print(f"Validation Issues: {summary['validation_issues']}")
    
    # Show platform details
    print("\n📋 Platform Details:")
    for platform, details in report["platforms"].items():
        status = "✅" if details["fully_configured"] else "❌"
        print(f"{status} {platform.title()}: {details['secrets_configured']} configured")
        
        if details["validation_issues"] > 0:
            print(f"   ⚠️  {details['validation_issues']} validation issues")
        if details["validation_warnings"] > 0:
            print(f"   ⚠️  {details['validation_warnings']} warnings")
    
    # Generate setup files
    print("\n📄 Generating Setup Files...")
    
    # Create .env template
    env_template = manager.generate_secrets_template()
    with open(".env.social_media_template", "w") as f:
        f.write(env_template)
    print("Created: .env.social_media_template")
    
    # Create setup instructions for each platform
    for platform in manager.required_secrets.keys():
        instructions = manager.generate_setup_instructions(platform)
        filename = f"SOCIAL_MEDIA_SETUP_{platform.upper()}.md"
        with open(filename, "w") as f:
            f.write(instructions)
        print(f"Created: {filename}")
    
    # Save full report
    with open("social_media_secrets_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Created: social_media_secrets_report.json")
    
    print("\n✅ Social Media Secrets Setup Complete!")
    print("\nNext Steps:")
    print("1. Copy .env.social_media_template to .env (if not exists)")
    print("2. Follow platform-specific setup instructions")
    print("3. Add your API credentials to .env file")
    print("4. Run validation: python -m src.cryptosmarttrader.data.tos_compliance_validator")
    print("5. Test social media data collection")


if __name__ == "__main__":
    main()