# containers.py
from dependency_injector import containers, providers
from core.config_manager import ConfigManager
from core.data_manager import DataManager
from core.health_monitor import HealthMonitor
from utils.exchange_manager import ExchangeManager
from core.cache_manager import CacheManager
from config.logging_config import setup_logging
from utils.performance_optimizer import PerformanceOptimizer
from utils.error_handler import error_handler
from utils.rate_limiter import rate_limiter
from utils.system_optimizer import SystemOptimizer
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.ml_predictor_agent import MLPredictorAgent
from agents.backtest_agent import BacktestAgent
from agents.trade_executor_agent import TradeExecutorAgent
from agents.whale_detector_agent import WhaleDetectorAgent
from models.ml_models import MLModelManager
from data.coin_registry import CoinRegistry
from utils.metrics import MetricsServer


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for CryptoSmartTrader V2.
    Provides centralized management of all system dependencies.
    """
    
    # Core configuration
    config = providers.Singleton(ConfigManager)
    
    # Logging setup
    logger_setup = providers.Resource(
        setup_logging,
        config_manager=config
    )
    
    # Performance and error handling
    performance_optimizer = providers.Singleton(PerformanceOptimizer)
    
    error_handler = providers.Object(error_handler)
    
    rate_limiter = providers.Object(rate_limiter)
    
    system_optimizer = providers.Singleton(
        SystemOptimizer,
        config_manager=config
    )
    
    # Core infrastructure
    cache_manager = providers.Singleton(
        CacheManager,
        config_manager=config
    )
    
    health_monitor = providers.Singleton(
        HealthMonitor,
        config_manager=config
    )
    
    # Exchange management
    exchange_manager = providers.Singleton(
        ExchangeManager,
        config_manager=config,
        health_monitor=health_monitor
    )
    
    # Data management
    coin_registry = providers.Singleton(
        CoinRegistry,
        config_manager=config
    )
    
    data_manager = providers.Singleton(
        DataManager,
        config_manager=config,
        exchange_manager=exchange_manager,
        health_monitor=health_monitor,
        coin_registry=coin_registry,
        cache_manager=cache_manager
    )
    
    # ML Model management
    ml_model_manager = providers.Singleton(
        MLModelManager,
        config_manager=config,
        cache_manager=cache_manager
    )
    
    # Agent dependencies
    sentiment_agent = providers.Factory(
        SentimentAgent,
        config_manager=config,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    technical_agent = providers.Factory(
        TechnicalAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    ml_predictor_agent = providers.Factory(
        MLPredictorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager,
        ml_model_manager=ml_model_manager
    )
    
    backtest_agent = providers.Factory(
        BacktestAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    trade_executor_agent = providers.Factory(
        TradeExecutorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    whale_detector_agent = providers.Factory(
        WhaleDetectorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    # Metrics server
    metrics_server = providers.Singleton(
        MetricsServer,
        config_manager=config,
        health_monitor=health_monitor
    )