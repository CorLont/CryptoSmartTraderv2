
# TELEGRAM API Setup Instructions

## Quick Setup
1. Start chat with @BotFather on Telegram
2. Send /newbot command
3. Follow instructions to create bot
4. Copy the Bot Token provided
5. Add bot to target channel/group
6. Get channel ID using @userinfobot

## API Information
- Developer Portal: https://t.me/BotFather
- Rate Limits: 30 requests per second
- TOS Compliance: Public channels only, respect bot API limits

## Required Environment Variables
- TELEGRAM_BOT_TOKEN: Telegram Bot API Token
- TELEGRAM_CHANNEL_ID: Telegram Channel/Group ID to monitor

## Testing Your Setup
```bash
# Test telegram configuration
python -c "
from src.cryptosmarttrader.data.enterprise_social_media_ingestion import get_social_media_manager
import asyncio

async def test_telegram():
    manager = get_social_media_manager()
    status = manager.get_compliance_status()
    print(f'Telegram Status:', status)

asyncio.run(test_telegram())
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
