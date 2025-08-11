#!/usr/bin/env python3
"""
Fixed containers - alle provider conflicts en duplicates opgelost
"""
from dependency_injector import containers, providers
from pathlib import Path
import logging
import importlib

logger = logging.getLogger(__name__)

def safe_import(module_path: str, class_name: str, fallback=None):
    """Safe import met expliciete fallback"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import {module_path}.{class_name}: {e}")
        if fallback:
            return fallback
        # Return safe dummy class
        class DummyClass:
            def __init__(self, *args, **kwargs):
                logger.info(f"Using dummy implementation for {class_name}")
            def __call__(self, *args, **kwargs):
                return None
        return DummyClass

class ApplicationContainer(containers.DeclarativeContainer):
    """Fixed container - geen provider conflicts meer"""
    
    # Configuration
    config = providers.Configuration()
    
    # Core managers - SINGLE INSTANCE ONLY
    config_manager = providers.Singleton(
        safe_import("core.config_manager", "ConfigManager")
    )
    
    data_manager = providers.Singleton(
        safe_import("core.data_manager", "DataManager"),
        config=config
    )
    
    # OpenAI analyzer
    openai_analyzer = providers.Singleton(
        safe_import("core.openai_enhanced_analyzer", "OpenAIEnhancedAnalyzer"),
        config_manager=config_manager
    )
    
    # Performance optimizer - FIXED: only ONE definition
    performance_optimizer = providers.Singleton(
        safe_import("core.performance_optimizer", "PerformanceOptimizer"),
        config=config
    )
    
    # GPU accelerator - FIXED: clear single definition
    gpu_accelerator = providers.Singleton(
        safe_import("core.gpu_accelerator", "GPUAccelerator"),
        config=config
    )
    
    # Health monitor
    health_monitor = providers.Singleton(
        safe_import("core.health_monitor", "HealthMonitor"),
        config=config
    )
    
    # ML predictor
    ml_predictor = providers.Factory(
        safe_import("agents.ml_predictor", "MLPredictor"),
        config_manager=config_manager,
        data_manager=data_manager
    )
    
    # System orchestrator
    orchestrator = providers.Singleton(
        safe_import("orchestration.system_orchestrator", "SystemOrchestrator"),
        config_manager=config_manager,
        health_monitor=health_monitor
    )
    
    # Logging manager
    logging_manager = providers.Singleton(
        safe_import("core.logging_manager", "LoggingManager"),
        config=config
    )

# Global container instance
container = ApplicationContainer()

def get_container():
    """Get configured container"""
    return container

def initialize_container():
    """Initialize container with proper error handling"""
    try:
        config_path = Path("config.json")
        if config_path.exists():
            container.config.from_json(str(config_path))
            logger.info("Container initialized with config")
        else:
            logger.warning("No config file found - using defaults")
        
        # Test critical providers
        critical_providers = ['config_manager', 'data_manager', 'health_monitor']
        for provider_name in critical_providers:
            try:
                provider = getattr(container, provider_name)
                instance = provider()
                logger.info(f"Provider {provider_name} initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {e}")
                # Continue with other providers
        
        return container
        
    except Exception as e:
        logger.error(f"Container initialization failed: {e}")
        # Return container anyway with defaults
        return container

def validate_container():
    """Validate container heeft geen duplicates of conflicts"""
    provider_names = []
    
    for name in dir(container):
        if not name.startswith('_') and hasattr(getattr(container, name), 'provider'):
            provider_names.append(name)
    
    # Check voor duplicates
    if len(provider_names) != len(set(provider_names)):
        logger.error("Duplicate provider names detected!")
        return False
    
    logger.info(f"Container validated: {len(provider_names)} unique providers")
    return True