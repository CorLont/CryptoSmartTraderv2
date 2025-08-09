#!/usr/bin/env python3
"""
Fixed containers - cleaned up duplicate providers
"""
from dependency_injector import containers, providers
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def safe_import(module_path: str, class_name: str):
    """Safe import with fallback"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import {module_path}.{class_name}: {e}")
        # Return a dummy class that does nothing
        return lambda *args, **kwargs: None

class ApplicationContainer(containers.DeclarativeContainer):
    """Fixed container without duplicate providers"""
    
    # Configuration
    config = providers.Configuration()
    
    # Core managers (single instances only)
    config_manager = providers.Singleton(
        safe_import("core.config_manager", "ConfigManager")
    )
    
    data_manager = providers.Singleton(
        safe_import("core.data_manager", "DataManager"),
        config=config
    )
    
    # OpenAI analyzer (single instance)
    openai_analyzer = providers.Singleton(
        safe_import("core.openai_enhanced_analyzer", "OpenAIEnhancedAnalyzer"),
        config_manager=config_manager
    )
    
    # Performance components (FIXED: removed duplicate)
    performance_optimizer = providers.Singleton(
        safe_import("core.performance_optimizer", "PerformanceOptimizer"),
        config=config
    )
    
    # GPU accelerator (FIXED: single definition)
    gpu_accelerator = providers.Singleton(
        safe_import("core.gpu_accelerator", "GPUAccelerator"),
        config=config
    )
    
    # ML components
    ml_predictor = providers.Factory(
        safe_import("agents.ml_predictor", "MLPredictor"),
        config_manager=config_manager,
        data_manager=data_manager
    )
    
    # Health monitor
    health_monitor = providers.Singleton(
        safe_import("core.health_monitor", "HealthMonitor"),
        config=config
    )
    
    # System orchestrator
    orchestrator = providers.Singleton(
        safe_import("orchestration.system_orchestrator", "SystemOrchestrator"),
        config_manager=config_manager,
        health_monitor=health_monitor
    )

# Global container instance
container = ApplicationContainer()

def get_container():
    """Get configured container"""
    return container

def initialize_container():
    """Initialize container with configuration"""
    config_path = Path("config.json")
    if config_path.exists():
        container.config.from_json(str(config_path))
        logger.info("Container initialized with config")
    else:
        logger.warning("No config file found - using defaults")
    
    return container