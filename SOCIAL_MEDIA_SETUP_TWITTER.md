
# TWITTER API Setup Instructions

## Quick Setup
1. Apply for Twitter Developer Account
2. Create a new App in the Developer Portal
3. Generate Bearer Token in App settings
4. Enable API v2 access
5. Review Twitter Developer Policy
6. Copy Bearer Token for API access

## API Information
- Developer Portal: https://developer.twitter.com/en/portal/dashboard
- Rate Limits: 300 requests per 15 minutes (API v2)
- TOS Compliance: Commercial use allowed with proper license, respect rate limits

## Required Environment Variables
- TWITTER_BEARER_TOKEN: Twitter API v2 Bearer Token
- TWITTER_API_KEY: Twitter API Key (Optional for v2)
- TWITTER_API_SECRET: Twitter API Secret (Optional for v2)

## Testing Your Setup
```bash
# Test twitter configuration
python -c "
from src.cryptosmarttrader.data.enterprise_social_media_ingestion import get_social_media_manager
import asyncio

async def test_twitter():
    manager = get_social_media_manager()
    status = manager.get_compliance_status()
    print(f'Twitter Status:', status)

asyncio.run(test_twitter())
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
