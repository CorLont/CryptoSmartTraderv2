
# REDDIT API Setup Instructions

## Quick Setup
1. Go to https://www.reddit.com/prefs/apps
2. Click 'Create App' or 'Create Another App'
3. Choose 'script' application type
4. Fill in name and description
5. Set redirect URI to http://localhost:8080
6. Copy the Client ID (under app name) and Client Secret

## API Information
- Developer Portal: https://www.reddit.com/prefs/apps
- Rate Limits: 60 requests per minute
- TOS Compliance: Must use OAuth2, respect rate limits, no mass downloading

## Required Environment Variables
- REDDIT_CLIENT_ID: Reddit API Client ID
- REDDIT_CLIENT_SECRET: Reddit API Client Secret
- REDDIT_USER_AGENT: Reddit API User Agent (Format: app_name/version)

## Testing Your Setup
```bash
# Test reddit configuration
python -c "
from src.cryptosmarttrader.data.enterprise_social_media_ingestion import get_social_media_manager
import asyncio

async def test_reddit():
    manager = get_social_media_manager()
    status = manager.get_compliance_status()
    print(f'Reddit Status:', status)

asyncio.run(test_reddit())
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
