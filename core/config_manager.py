import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import logging
from datetime import datetime
from config.settings import config
from config.validation import SystemConfiguration, validate_configuration, get_default_configuration


class ConfigManager:
    """Enhanced configuration management system with Pydantic integration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.backup_path = Path(f"{config_path}.backup")
        self._config: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._pydantic_config = config  # Use the Pydantic config instance
        self._default_config = self._get_default_config()
        
        # Initialize configuration
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values with validation"""
        return get_default_configuration()
    
    def _load_config(self):
        """Load configuration from file with validation"""
        with self._lock:
            try:
                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        loaded_config = json.load(f)
                    
                    # Validate configuration
                    try:
                        validated_config = validate_configuration(loaded_config)
                        self._config = validated_config.model_dump()
                    except Exception as e:
                        logging.warning(f"Configuration validation failed: {e}, using defaults")
                        self._config = self._default_config.copy()
                        self._save_config()
                else:
                    self._config = self._default_config.copy()
                    self._save_config()
                    
            except Exception as e:
                logging.error(f"Error loading configuration: {e}")
                self._config = self._default_config.copy()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Ensure required keys exist
        for key in self._default_config:
            if key not in self._config:
                self._config[key] = self._default_config[key]
        
        # Validate ranges
        self._config["api_rate_limit"] = max(10, min(1000, self._config.get("api_rate_limit", 100)))
        self._config["max_coins"] = max(50, min(1000, self._config.get("max_coins", 453)))
        self._config["alert_threshold"] = max(0, min(100, self._config.get("alert_threshold", 80)))
    
    def _save_config(self):
        """Save configuration to file with backup"""
        try:
            # Create backup if config exists
            if self.config_path.exists():
                import shutil
                shutil.copy(self.config_path, self.backup_path)
            
            # Save new config
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
                
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        with self._lock:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        with self._lock:
            self._config[key] = value
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        with self._lock:
            try:
                # Update config
                self._config.update(updates)
                self._config["last_updated"] = datetime.now().isoformat()
                
                # Validate and save
                self._validate_config()
                return self._save_config()
                
            except Exception as e:
                print(f"Error updating config: {e}")
                return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        with self._lock:
            return self._config.copy()
    
    def rollback(self) -> bool:
        """Rollback to backup configuration"""
        try:
            if self.backup_path.exists():
                import shutil
                shutil.copy(self.backup_path, self.config_path)
                self._load_config()
                return True
        except Exception as e:
            print(f"Error rolling back config: {e}")
        
        return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        with self._lock:
            self._config = self._default_config.copy()
            return self._save_config()
    
    def get_pydantic_value(self, key: str, default: Any = None) -> Any:
        """Get value from Pydantic config with fallback to legacy config"""
        try:
            return getattr(self._pydantic_config, key, default)
        except AttributeError:
            return self.get(key, default)
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        agent_configs = self.get("agents", {})
        return agent_configs.get(agent_name, {"enabled": True})
