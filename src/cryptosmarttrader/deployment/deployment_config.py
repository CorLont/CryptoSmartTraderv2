"""
Deployment Configuration System

Enterprise deployment configuration with RTO/RPO settings,
process management, and health monitoring configuration.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
import json

@dataclass
class DeploymentConfig:
    """Enterprise deployment configuration"""
    
    # Environment settings
    environment: str = "production"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Process management
    max_processes: int = 10
    process_restart_delay: float = 5.0
    max_restart_attempts: int = 3
    health_check_interval: float = 30.0
    
    # RTO/RPO settings
    rto_target_seconds: int = 30      # Recovery Time Objective
    rpo_target_seconds: int = 60      # Recovery Point Objective
    backup_interval_seconds: int = 300  # 5 minutes
    checkpoint_interval_seconds: int = 60  # 1 minute
    
    # Directory configuration
    data_directory: str = "./data"
    logs_directory: str = "./logs" 
    cache_directory: str = "./cache"
    backup_directory: str = "./backups"
    
    # Write directories with rotation
    write_directories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logs": {
            "path": "./logs",
            "max_size_mb": 100,
            "max_files": 10,
            "rotation_enabled": True
        },
        "data": {
            "path": "./data",
            "max_size_mb": 500,
            "max_files": 5,
            "rotation_enabled": True
        },
        "cache": {
            "path": "./cache",
            "max_size_mb": 200,
            "max_files": 3,
            "rotation_enabled": True
        }
    })
    
    # Health check dependencies
    health_dependencies: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "system_memory",
            "type": "system_resource",
            "critical": True,
            "max_memory_percent": 90.0
        },
        {
            "name": "system_disk", 
            "type": "system_resource",
            "critical": True,
            "max_disk_percent": 85.0
        },
        {
            "name": "data_directory",
            "type": "file_system", 
            "critical": True,
            "path": "./data"
        }
    ])
    
    # Service configurations
    services: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "api": {
            "port": 8001,
            "health_endpoint": "/health",
            "auto_restart": True,
            "max_restarts": 5
        },
        "metrics": {
            "port": 8000,
            "health_endpoint": "/metrics", 
            "auto_restart": True,
            "max_restarts": 5
        },
        "dashboard": {
            "port": 5000,
            "health_endpoint": "/",
            "auto_restart": True,
            "max_restarts": 3
        }
    })
    
    def create_directories(self):
        """Create required directories"""
        directories = [
            self.data_directory,
            self.logs_directory,
            self.cache_directory, 
            self.backup_directory
        ]
        
        for write_config in self.write_directories.values():
            directories.append(write_config["path"])
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific service"""
        return self.services.get(service_name)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            "environment": self.environment,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "max_processes": self.max_processes,
            "process_restart_delay": self.process_restart_delay,
            "max_restart_attempts": self.max_restart_attempts,
            "health_check_interval": self.health_check_interval,
            "rto_target_seconds": self.rto_target_seconds,
            "rpo_target_seconds": self.rpo_target_seconds,
            "backup_interval_seconds": self.backup_interval_seconds,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "data_directory": self.data_directory,
            "logs_directory": self.logs_directory,
            "cache_directory": self.cache_directory,
            "backup_directory": self.backup_directory,
            "write_directories": self.write_directories,
            "health_dependencies": self.health_dependencies,
            "services": self.services
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DeploymentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check RTO/RPO targets
        if self.rto_target_seconds <= 0:
            issues.append("RTO target must be positive")
        
        if self.rpo_target_seconds <= 0:
            issues.append("RPO target must be positive")
        
        if self.backup_interval_seconds > self.rpo_target_seconds:
            issues.append("Backup interval exceeds RPO target")
        
        # Check directory paths
        for name, config in self.write_directories.items():
            if not config.get("path"):
                issues.append(f"Write directory '{name}' missing path")
        
        # Check service ports
        used_ports = set()
        for service_name, service_config in self.services.items():
            port = service_config.get("port")
            if not port:
                issues.append(f"Service '{service_name}' missing port")
            elif port in used_ports:
                issues.append(f"Port {port} used by multiple services")
            else:
                used_ports.add(port)
        
        return issues

# Default deployment configuration
DEFAULT_DEPLOYMENT_CONFIG = DeploymentConfig()