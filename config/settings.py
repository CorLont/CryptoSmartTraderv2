#!/usr/bin/env python3
"""
Configuration Settings for CryptoSmartTrader V2
"""

import os
from typing import Dict, Any, List
from pathlib import Path

class Config:
    """Central configuration class"""
    
    def __init__(self):
        # Server settings
        self.server_host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.getenv("SERVER_PORT", "8001"))
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        # API settings
        self.kraken_api_key = os.getenv("KRAKEN_API_KEY", "")
        self.kraken_secret = os.getenv("KRAKEN_SECRET", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Database settings
        self.database_url = os.getenv("DATABASE_URL", "")
        
        # Logging settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_dir = Path(os.getenv("LOG_DIR", "logs"))
        
        # Trading settings
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.max_positions = int(os.getenv("MAX_POSITIONS", "10"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()