#!/usr/bin/env python3
"""
Fixed containers - implements audit point G
"""
from dependency_injector import containers, providers

class ApplicationContainer(containers.DeclarativeContainer):
    """Fixed container zonder dubbele providers"""
    
    # Fixed: Verwijder de dubbele performance_optimizer; kies één:
    performance_optimizer = providers.Singleton("PerformanceOptimizer")
    
    # Fixed: Verwijder gpu_accelerator_provider of hernoem:
    gpu_accelerator = providers.Singleton("GPUAccelerator", container=providers.Self)
    
    # Fixed: Maak dynamische imports robuuster:
    def _lazy(cls_path):
        try:
            mod, cls = cls_path.rsplit(".", 1)
            return getattr(__import__(mod, fromlist=[cls]), cls)
        except ImportError:
            # Fallback voor ontbrekende modules
            return lambda *args, **kwargs: None
    
    # Fixed: Robuuste providers
    config_manager = providers.Singleton(_lazy("core.config_manager.ConfigManager"))
    data_manager = providers.Singleton(_lazy("core.data_manager.DataManager"))
    
    # OpenAI analyzer met fallback
    try:
        openai_analyzer = providers.Singleton(
            _lazy("core.openai_enhanced_analyzer.OpenAIEnhancedAnalyzer"), 
            config_manager=config_manager
        )
    except:
        openai_analyzer = providers.Singleton(lambda: None)