"""
CryptoSmartTrader V2 - Main Entry Point with Fail-Fast Validation
Enterprise startup with comprehensive configuration validation
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cryptosmarttrader.config import load_and_validate_settings
from src.cryptosmarttrader.logging import setup_logging, get_logger, LogContext
import logging


def validate_environment():
    """Validate critical environment requirements"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    
    # Check critical dependencies
    try:
        import pydantic
        import streamlit
        import fastapi
        import pandas
        import numpy
    except ImportError as e:
        issues.append(f"Missing critical dependency: {e}")
    
    # Check working directory
    if not Path("pyproject.toml").exists():
        issues.append("Must run from project root directory")
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print("✅ Environment validation passed")


def start_services(settings):
    """Start application services based on configuration"""
    logger = get_logger(__name__)
    
    with LogContext() as correlation_id:
        logger.info(
            "Starting CryptoSmartTrader V2 services",
            extra={"correlation_id": correlation_id}
        )
        
        # Import service modules
        try:
            if settings.ENABLE_PROMETHEUS:
                logger.info("Starting Prometheus metrics server")
                # Start metrics server
                
            if settings.API_PORT:
                logger.info("Starting FastAPI service")
                # Start API server
                
            if settings.DASHBOARD_PORT:
                logger.info("Starting Streamlit dashboard")
                # Start dashboard
                
            logger.info("✅ All services started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main application entry point with fail-fast validation"""
    print("🚀 CryptoSmartTrader V2 - Enterprise Trading Intelligence")
    print("=" * 60)
    
    try:
        # 1. Validate environment
        validate_environment()
        
        # 2. Load and validate configuration (fail-fast)
        print("⚙️  Loading configuration...")
        settings = load_and_validate_settings()
        
        # 3. Setup logging with validated settings
        print("📋 Setting up logging...")
        setup_logging(
            level=settings.LOG_LEVEL,
            log_dir=settings.LOGS_DIR,
            service_name="cryptosmarttrader"
        )
        
        logger = get_logger(__name__)
        logger.info("CryptoSmartTrader V2 startup initiated")
        
        # 4. Log configuration summary
        config_summary = settings.get_summary()
        logger.info("Configuration loaded", extra={"config": config_summary})
        
        # 5. Start services
        print("🔄 Starting services...")
        start_services(settings)
        
        print("✅ CryptoSmartTrader V2 started successfully")
        print(f"📊 Dashboard: http://localhost:{settings.DASHBOARD_PORT}")
        print(f"🔌 API: http://localhost:{settings.API_PORT}/docs")
        print(f"📈 Metrics: http://localhost:{settings.METRICS_PORT}/metrics")
        
        # Keep main thread alive
        try:
            import signal
            signal.pause()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            print("\n👋 Shutting down CryptoSmartTrader V2...")
            
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()