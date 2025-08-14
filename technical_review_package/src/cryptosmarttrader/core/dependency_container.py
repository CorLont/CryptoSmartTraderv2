#!/usr/bin/env python3
"""
Dependency Injection Container
Manages all application dependencies with proper injection
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from config.settings import AppSettings, get_settings
from ..core.async_data_manager import AsyncDataManager, RateLimitConfig
from ..core.secrets_manager import SecretsManager, get_secrets_manager


class Container(containers.DeclarativeContainer):
    """Main DI container for the application"""

    # Configuration
    config = providers.Singleton(get_settings)

    # Secrets management
    secrets_manager = providers.Singleton(get_secrets_manager, config=config.provided.dict())

    # Logging
    logger_factory = providers.Factory(logging.getLogger, name="CryptoSmartTrader")

    # Rate limiting configuration
    rate_limit_config = providers.Factory(
        RateLimitConfig,
        requests_per_second=config.provided.exchange.requests_per_second,
        burst_size=config.provided.exchange.burst_size,
        cool_down_period=60,
    )

    # Async Data Manager
    async_data_manager = providers.Singleton(AsyncDataManager, rate_limit_config=rate_limit_config)

    # Secure exchange configurations
    kraken_config = providers.Factory(
        lambda settings, secrets: {
            **settings.get_exchange_config("kraken"),
            "apiKey": secrets.get_secret("KRAKEN_API_KEY"),
            "secret": secrets.get_secret("KRAKEN_SECRET"),
        },
        settings=config,
        secrets=secrets_manager,
    )

    binance_config = providers.Factory(
        lambda settings, secrets: {
            **settings.get_exchange_config("binance"),
            "apiKey": secrets.get_secret("BINANCE_API_KEY"),
            "secret": secrets.get_secret("BINANCE_SECRET"),
        },
        settings=config,
        secrets=secrets_manager,
    )


class AgentContainer(containers.DeclarativeContainer):
    """Container for agent-specific dependencies"""

    # Parent container
    parent = providers.DependenciesContainer()

    # Agent configuration
    agent_config = providers.Factory(
        dict,
        health_check_interval=parent.config.provided.agents.health_check_interval,
        restart_limit=parent.config.provided.agents.restart_limit,
        circuit_breaker_threshold=parent.config.provided.agents.circuit_breaker_threshold,
        max_memory_mb=parent.config.provided.agents.max_memory_mb,
    )


class MLContainer(containers.DeclarativeContainer):
    """Container for ML-specific dependencies"""

    # Parent container
    parent = providers.DependenciesContainer()

    # ML configuration
    ml_config = providers.Factory(
        dict,
        model_cache_dir=parent.config.provided.ml.model_cache_dir,
        training_data_days=parent.config.provided.ml.training_data_days,
        confidence_threshold=parent.config.provided.ml.prediction_confidence_threshold,
        batch_size=parent.config.provided.ml.batch_size,
        learning_rate=parent.config.provided.ml.learning_rate,
    )


# Global container instance
container = Container()


def get_container() -> Container:
    """Get the global DI container"""
    return container


def wire_container(modules: list) -> None:
    """Wire the container to specified modules"""
    container.wire(modules=modules)


def configure_container(settings: Optional[AppSettings] = None) -> Container:
    """Configure container with custom settings (for testing)"""
    if settings:
        # Override config provider for testing
        container.config.override(settings)

    # Configure sub-containers
    agent_container = AgentContainer(parent=container)
    ml_container = MLContainer(parent=container)

    return container


def reset_container() -> None:
    """Reset container (for testing)"""
    container.reset_singletons()


# Dependency injection decorators and functions
def inject_config() -> AppSettings:
    """Inject application configuration"""
    return Provide[Container.config]


def inject_logger() -> logging.Logger:
    """Inject logger instance"""
    return Provide[Container.logger_factory]


def inject_async_data_manager() -> AsyncDataManager:
    """Inject async data manager"""
    return Provide[Container.async_data_manager]


def inject_rate_limit_config() -> RateLimitConfig:
    """Inject rate limit configuration"""
    return Provide[Container.rate_limit_config]


# Context managers for dependency injection
class DIContext:
    """Context manager for dependency injection in agents"""

    def __init__(self, container: Container = None):
        self.container = container or get_container()
        self.original_providers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original providers if any were overridden
        for key, provider in self.original_providers.items():
            setattr(self.container, key, provider)

    def override_config(self, settings: AppSettings):
        """Override configuration for testing"""
        self.original_providers["config"] = self.container.config
        self.container.config.override(settings)

    def override_async_data_manager(self, mock_manager: AsyncDataManager):
        """Override async data manager for testing"""
        self.original_providers["async_data_manager"] = self.container.async_data_manager
        self.container.async_data_manager.override(mock_manager)


# Factory functions with dependency injection
@inject
def create_data_collector_agent(
    config: AppSettings = Provide[Container.config],
    async_data_manager: AsyncDataManager = Provide[Container.async_data_manager],
    logger: logging.Logger = Provide[Container.logger_factory],
):
    """Factory function for data collector agent with DI"""
    from ..agents.data_collector import AsyncDataCollectorAgent

    agent_config = {
        "collection_interval": 45,
        "health_check_interval": config.agents.health_check_interval,
        "max_memory_mb": config.agents.max_memory_mb,
    }

    agent = AsyncDataCollectorAgent(agent_config)
    agent.data_manager = async_data_manager
    agent.logger = logger.getChild("DataCollector")

    return agent


@inject
def create_health_monitor_agent(
    config: AppSettings = Provide[Container.config],
    logger: logging.Logger = Provide[Container.logger_factory],
):
    """Factory function for health monitor agent with DI"""
    from ..agents.health_monitor import HealthMonitorAgent

    agent_config = {
        "monitor_interval": config.agents.health_check_interval,
        "health_check_interval": config.agents.health_check_interval * 2,
    }

    agent = HealthMonitorAgent(agent_config)
    agent.logger = logger.getChild("HealthMonitor")

    return agent


# Helper functions for testing
def create_test_container(test_settings: AppSettings = None) -> Container:
    """Create a container for testing with optional custom settings"""
    test_container = Container()

    if test_settings:
        test_container.config.override(test_settings)

    return test_container


def inject_dependencies(func):
    """Decorator to inject dependencies into functions"""
    return inject(func)


# Configuration validation
@inject
def validate_configuration(
    config: AppSettings = Provide[Container.config],
    logger: logging.Logger = Provide[Container.logger_factory],
) -> bool:
    """Validate the current configuration"""
    try:
        # Check critical paths exist
        config.data.data_dir.mkdir(parents=True, exist_ok=True)
        config.logging.log_dir.mkdir(parents=True, exist_ok=True)
        config.ml.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Check exchange credentials
        has_exchange_access = (
            config.exchange.kraken_api_key and config.exchange.kraken_secret
        ) or (config.exchange.binance_api_key and config.exchange.binance_secret)

        if not has_exchange_access:
            logger.warning("No exchange credentials configured - using public APIs only")

        # Validate rate limits
        if config.exchange.requests_per_second > 50:
            logger.warning("High rate limit configured - may cause API throttling")

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


# Initialize container on import
configure_container()
