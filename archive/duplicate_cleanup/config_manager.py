#!/usr/bin/env python3
"""
Configuration Manager
Centralized configuration management for the CryptoSmartTrader system
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

from .system_settings import (
    get_system_settings, get_exchange_settings, get_ml_settings,
    get_data_settings, get_notification_settings, get_api_settings
)

class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self):
        self._system_settings = None
        self._exchange_settings = None
        self._ml_settings = None
        self._data_settings = None
        self._notification_settings = None
        self._api_settings = None
    
    @property
    def system(self):
        if self._system_settings is None:
            self._system_settings = get_system_settings()
        return self._system_settings
    
    @property
    def exchange(self):
        if self._exchange_settings is None:
            self._exchange_settings = get_exchange_settings()
        return self._exchange_settings
    
    @property
    def ml(self):
        if self._ml_settings is None:
            self._ml_settings = get_ml_settings()
        return self._ml_settings
    
    @property
    def data(self):
        if self._data_settings is None:
            self._data_settings = get_data_settings()
        return self._data_settings
    
    @property
    def notification(self):
        if self._notification_settings is None:
            self._notification_settings = get_notification_settings()
        return self._notification_settings
    
    @property
    def api(self):
        if self._api_settings is None:
            self._api_settings = get_api_settings()
        return self._api_settings

# Global config manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager