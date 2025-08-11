#!/usr/bin/env python3
"""
Configuration Startup Demo
Demonstrates enterprise Pydantic Settings with startup logging
"""

import logging
import sys
from pathlib import Path

# Add src to path for import
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main demonstration function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== CryptoSmartTrader V2 Configuration Demo ===")
    
    try:
        # Import and create settings
        from cryptosmarttrader.config import settings, log_startup_config
        
        logger.info("Configuration loaded successfully")
        
        # Log startup configuration
        log_startup_config(settings)
        
        # Demonstrate fail-fast validation
        logger.info("\n=== Validation Results ===")
        missing_secrets = settings.validate_required_secrets()
        
        if missing_secrets:
            logger.warning(f"Missing secrets (non-critical in {settings.ENVIRONMENT}): {missing_secrets}")
        else:
            logger.info("All required secrets configured")
        
        # Show type safety
        logger.info(f"\n=== Type Safety Demo ===")
        logger.info(f"API Port (int): {settings.API_PORT} (type: {type(settings.API_PORT).__name__})")
        logger.info(f"Debug Mode (bool): {settings.DEBUG} (type: {type(settings.DEBUG).__name__})")
        logger.info(f"Risk Percentage (float): {settings.RISK_PERCENTAGE} (type: {type(settings.RISK_PERCENTAGE).__name__})")
        
        # Show environment-specific behavior
        logger.info(f"\n=== Environment-Specific Behavior ===")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Is Production: {settings.is_production()}")
        logger.info(f"Trading Enabled: {settings.TRADING_ENABLED}")
        
        if settings.is_development():
            logger.info("Development safety checks active")
        
        logger.info("\n✅ Configuration demonstration complete")
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()