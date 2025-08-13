"""
Environment Manager System

Complete environment separation (Dev/Staging/Prod) with isolated keys,
quotas, feature flags, and canary deployment capabilities.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class CanaryState(Enum):
    """Canary deployment states"""
    INACTIVE = "inactive"
    STARTING = "starting"
    RUNNING = "running"
    VALIDATING = "validating"
    PROMOTING = "promoting"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    environment: Environment

    # API Keys and Credentials
    kraken_api_key: Optional[str] = None
    kraken_secret: Optional[str] = None
    openai_api_key: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Rate Limits and Quotas
    api_rate_limit_per_minute: int = 60
    openai_requests_per_day: int = 1000
    max_concurrent_orders: int = 5
    max_position_size_usd: int = 10000

    # Risk Limits (environment-specific)
    max_daily_loss_usd: int = 5000
    max_drawdown_percent: float = 10.0
    position_size_limit_percent: float = 5.0

    # Feature Flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    # Database and Storage
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    # Monitoring
    metrics_enabled: bool = True
    alerts_enabled: bool = True
    debug_logging: bool = False

@dataclass
class FeatureFlag:
    """Feature flag definition"""
    name: str
    description: str
    enabled_environments: List[Environment]
    rollout_percentage: float = 100.0  # 0-100
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class CanaryDeployment:
    """Canary deployment configuration"""
    name: str
    description: str

    # Traffic split
    canary_percentage: float = 1.0  # Start with 1% traffic

    # Duration and validation
    min_duration_hours: int = 168  # 1 week minimum
    validation_metrics: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)

    # State tracking
    state: CanaryState = CanaryState.INACTIVE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Risk controls
    max_risk_exposure_usd: float = 1000.0  # Max $1k at risk in canary
    auto_rollback_triggers: Dict[str, float] = field(default_factory=dict)

    # Metrics tracking
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    alerts_triggered: List[str] = field(default_factory=list)

class EnvironmentManager:
    """
    Comprehensive environment management system
    """

    def __init__(self, config_dir: str = "configs/environments"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Current environment
        self.current_environment = self._detect_environment()

        # Environment configurations
        self.environments: Dict[Environment, EnvironmentConfig] = {}
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.canary_deployments: Dict[str, CanaryDeployment] = {}

        # Load configurations
        self._load_environment_configs()
        self._load_feature_flags()
        self._load_canary_deployments()

        # Setup default environments if not exist
        self._setup_default_environments()

        logger.info(f"Environment Manager initialized - Current: {self.current_environment.value}")

    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()

        if env_name in ["prod", "production"]:
            return Environment.PRODUCTION
        elif env_name in ["stage", "staging"]:
            return Environment.STAGING
        else:
            return Environment.DEVELOPMENT

    def _setup_default_environments(self):
        """Setup default environment configurations"""

        if Environment.DEVELOPMENT not in self.environments:
            self.environments[Environment.DEVELOPMENT] = EnvironmentConfig(
                name="Development",
                environment=Environment.DEVELOPMENT,
                api_rate_limit_per_minute=30,
                openai_requests_per_day=500,
                max_concurrent_orders=2,
                max_position_size_usd=1000,
                max_daily_loss_usd=500,
                max_drawdown_percent=5.0,
                position_size_limit_percent=2.0,
                debug_logging=True,
                feature_flags={
                    "advanced_ml_models": True,
                    "real_trading": False,
                    "chaos_testing": True,
                    "debug_mode": True
                }
            )

        if Environment.STAGING not in self.environments:
            self.environments[Environment.STAGING] = EnvironmentConfig(
                name="Staging",
                environment=Environment.STAGING,
                api_rate_limit_per_minute=45,
                openai_requests_per_day=750,
                max_concurrent_orders=3,
                max_position_size_usd=2500,
                max_daily_loss_usd=1000,
                max_drawdown_percent=7.5,
                position_size_limit_percent=3.5,
                debug_logging=False,
                feature_flags={
                    "advanced_ml_models": True,
                    "real_trading": True,
                    "canary_deployment": True,
                    "shadow_trading": True
                }
            )

        if Environment.PRODUCTION not in self.environments:
            self.environments[Environment.PRODUCTION] = EnvironmentConfig(
                name="Production",
                environment=Environment.PRODUCTION,
                api_rate_limit_per_minute=60,
                openai_requests_per_day=1000,
                max_concurrent_orders=5,
                max_position_size_usd=10000,
                max_daily_loss_usd=5000,
                max_drawdown_percent=10.0,
                position_size_limit_percent=5.0,
                debug_logging=False,
                feature_flags={
                    "advanced_ml_models": True,
                    "real_trading": True,
                    "production_monitoring": True,
                    "strict_validation": True
                }
            )

        # Setup default feature flags
        self._setup_default_feature_flags()

    def _setup_default_feature_flags(self):
        """Setup default feature flags"""

        default_flags = [
            FeatureFlag(
                name="advanced_ml_models",
                description="Enable advanced ML models (Transformers, LSTM ensembles)",
                enabled_environments=[Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION],
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="real_trading",
                description="Enable real trading execution",
                enabled_environments=[Environment.STAGING, Environment.PRODUCTION],
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="canary_deployment",
                description="Enable canary deployment testing",
                enabled_environments=[Environment.STAGING],
                rollout_percentage=1.0  # 1% traffic for canary
            ),
            FeatureFlag(
                name="shadow_trading",
                description="Enable shadow trading for validation",
                enabled_environments=[Environment.STAGING, Environment.PRODUCTION],
                rollout_percentage=10.0  # 10% shadow trading
            ),
            FeatureFlag(
                name="chaos_testing",
                description="Enable automated chaos testing",
                enabled_environments=[Environment.DEVELOPMENT, Environment.STAGING],
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="production_monitoring",
                description="Enable production-grade monitoring",
                enabled_environments=[Environment.PRODUCTION],
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="debug_mode",
                description="Enable debug logging and detailed traces",
                enabled_environments=[Environment.DEVELOPMENT],
                rollout_percentage=100.0
            ),
            FeatureFlag(
                name="strict_validation",
                description="Enable strict data validation and safeguards",
                enabled_environments=[Environment.PRODUCTION],
                rollout_percentage=100.0
            )
        ]

        for flag in default_flags:
            if flag.name not in self.feature_flags:
                self.feature_flags[flag.name] = flag

    def get_current_config(self) -> EnvironmentConfig:
        """Get configuration for current environment"""
        return self.environments[self.current_environment]

    def get_config(self, environment: Environment) -> EnvironmentConfig:
        """Get configuration for specific environment"""
        return self.environments.get(environment)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled in current environment"""
        if feature_name not in self.feature_flags:
            logger.warning(f"Unknown feature flag: {feature_name}")
            return False

        flag = self.feature_flags[feature_name]

        # Check if enabled for current environment
        if self.current_environment not in flag.enabled_environments:
            return False

        # Check rollout percentage (simple random check)
        import random
        return random.random() * 100 <= flag.rollout_percentage

    def set_feature_flag(self, feature_name: str, enabled: bool, environments: Optional[List[Environment]] = None):
        """Set feature flag state"""
        if feature_name not in self.feature_flags:
            logger.error(f"Feature flag not found: {feature_name}")
            return

        flag = self.feature_flags[feature_name]

        if environments:
            flag.enabled_environments = environments

        flag.rollout_percentage = 100.0 if enabled else 0.0
        flag.updated_at = datetime.now()

        self._save_feature_flags()
        logger.info(f"Feature flag updated: {feature_name} = {enabled}")

    def create_canary_deployment(self,
                                name: str,
                                description: str,
                                canary_percentage: float = 1.0,
                                min_duration_hours: int = 168) -> CanaryDeployment:
        """Create new canary deployment"""

        # Validate canary percentage for safety
        if canary_percentage > 5.0:
            raise ValueError("Canary percentage cannot exceed 5% for safety")

        canary = CanaryDeployment(
            name=name,
            description=description,
            canary_percentage=canary_percentage,
            min_duration_hours=min_duration_hours,
            validation_metrics=[
                "error_rate",
                "latency_p95",
                "slippage_realized",
                "pnl_daily",
                "drawdown_current"
            ],
            success_criteria={
                "error_rate_max": 0.01,  # 1% max error rate
                "latency_p95_max": 2.0,  # 2s max latency
                "slippage_p95_max": 50.0,  # 50 bps max slippage
                "pnl_min_daily": -100.0,  # Max $100 daily loss
                "drawdown_max": 2.0  # Max 2% drawdown
            },
            max_risk_exposure_usd=canary_percentage * 100,  # $1 per 1% exposure
            auto_rollback_triggers={
                "error_rate": 0.05,  # 5% error rate triggers rollback
                "latency_p95": 5.0,  # 5s latency triggers rollback
                "daily_loss": 500.0,  # $500 loss triggers rollback
                "drawdown": 3.0  # 3% drawdown triggers rollback
            }
        )

        self.canary_deployments[name] = canary
        self._save_canary_deployments()

        logger.info(f"Created canary deployment: {name} ({canary_percentage}% traffic)")
        return canary

    def start_canary_deployment(self, name: str) -> bool:
        """Start canary deployment"""
        if name not in self.canary_deployments:
            logger.error(f"Canary deployment not found: {name}")
            return False

        canary = self.canary_deployments[name]

        if canary.state != CanaryState.INACTIVE:
            logger.warning(f"Canary deployment already active: {name}")
            return False

        # Validate environment
        if self.current_environment != Environment.STAGING:
            logger.error("Canary deployments can only be started in staging environment")
            return False

        # Start canary
        canary.state = CanaryState.STARTING
        canary.start_time = datetime.now()

        logger.info(f"🕊️ Starting canary deployment: {name}")

        # In a real implementation, this would:
        # 1. Deploy new version to canary infrastructure
        # 2. Route specified percentage of traffic
        # 3. Start monitoring and validation

        canary.state = CanaryState.RUNNING
        self._save_canary_deployments()

        return True

    def validate_canary_deployment(self, name: str, metrics: Dict[str, float]) -> bool:
        """Validate canary deployment against success criteria"""
        if name not in self.canary_deployments:
            return False

        canary = self.canary_deployments[name]

        if canary.state != CanaryState.RUNNING:
            return False

        # Record metrics
        canary.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        })

        # Check auto-rollback triggers
        rollback_triggered = False
        for trigger_name, trigger_value in canary.auto_rollback_triggers.items():
            if trigger_name in metrics and metrics[trigger_name] > trigger_value:
                logger.error(f"🚨 Canary rollback triggered: {trigger_name} = {metrics[trigger_name]} > {trigger_value}")
                canary.alerts_triggered.append(f"Auto rollback: {trigger_name}")
                rollback_triggered = True

        if rollback_triggered:
            self.rollback_canary_deployment(name)
            return False

        # Check if canary has run long enough
        if canary.start_time:
            runtime_hours = (datetime.now() - canary.start_time).total_seconds() / 3600

            if runtime_hours >= canary.min_duration_hours:
                # Check success criteria
                success = True
                for criterion, max_value in canary.success_criteria.items():
                    if criterion in metrics and metrics[criterion] > max_value:
                        success = False
                        break

                if success:
                    logger.info(f"✅ Canary deployment validation successful: {name}")
                    canary.state = CanaryState.VALIDATING
                    return True
                else:
                    logger.warning(f"⚠️ Canary deployment failed validation: {name}")
                    self.rollback_canary_deployment(name)
                    return False

        self._save_canary_deployments()
        return True

    def promote_canary_deployment(self, name: str) -> bool:
        """Promote canary deployment to production"""
        if name not in self.canary_deployments:
            return False

        canary = self.canary_deployments[name]

        if canary.state != CanaryState.VALIDATING:
            logger.error(f"Canary deployment not ready for promotion: {name}")
            return False

        canary.state = CanaryState.PROMOTING

        logger.info(f"🚀 Promoting canary deployment to production: {name}")

        # In real implementation:
        # 1. Deploy to production infrastructure
        # 2. Route 100% traffic to new version
        # 3. Decommission old version

        canary.state = CanaryState.PROMOTED
        canary.end_time = datetime.now()

        self._save_canary_deployments()
        return True

    def rollback_canary_deployment(self, name: str) -> bool:
        """Rollback canary deployment"""
        if name not in self.canary_deployments:
            return False

        canary = self.canary_deployments[name]
        canary.state = CanaryState.ROLLING_BACK

        logger.warning(f"⏪ Rolling back canary deployment: {name}")

        # In real implementation:
        # 1. Route traffic back to stable version
        # 2. Stop canary infrastructure
        # 3. Clean up resources

        canary.state = CanaryState.FAILED
        canary.end_time = datetime.now()

        self._save_canary_deployments()
        return True

    def get_environment_secrets(self) -> Dict[str, str]:
        """Get secrets for current environment"""
        config = self.get_current_config()

        secrets = {}

        # Add environment-specific secrets
        if config.kraken_api_key:
            secrets["KRAKEN_API_KEY"] = config.kraken_api_key
        if config.kraken_secret:
            secrets["KRAKEN_SECRET"] = config.kraken_secret
        if config.openai_api_key:
            secrets["OPENAI_API_KEY"] = config.openai_api_key
        if config.telegram_token:
            secrets["TELEGRAM_TOKEN"] = config.telegram_token
        if config.telegram_chat_id:
            secrets["TELEGRAM_CHAT_ID"] = config.telegram_chat_id

        return secrets

    def _load_environment_configs(self):
        """Load environment configurations from disk"""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)

                    config = EnvironmentConfig(
                        name=data.get("name", env.value.title()),
                        environment=env,
                        **{k: v for k, v in data.items() if k != "name"}
                    )
                    self.environments[env] = config

                except Exception as e:
                    logger.error(f"Failed to load config for {env.value}: {e}")

    def _save_environment_configs(self):
        """Save environment configurations to disk"""
        for env, config in self.environments.items():
            config_file = self.config_dir / f"{env.value}.json"
            try:
                # Convert to dict, excluding sensitive data
                data = {
                    "name": config.name,
                    "api_rate_limit_per_minute": config.api_rate_limit_per_minute,
                    "openai_requests_per_day": config.openai_requests_per_day,
                    "max_concurrent_orders": config.max_concurrent_orders,
                    "max_position_size_usd": config.max_position_size_usd,
                    "max_daily_loss_usd": config.max_daily_loss_usd,
                    "max_drawdown_percent": config.max_drawdown_percent,
                    "position_size_limit_percent": config.position_size_limit_percent,
                    "feature_flags": config.feature_flags,
                    "metrics_enabled": config.metrics_enabled,
                    "alerts_enabled": config.alerts_enabled,
                    "debug_logging": config.debug_logging
                }

                with open(config_file, 'w') as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save config for {env.value}: {e}")

    def _load_feature_flags(self):
        """Load feature flags from disk"""
        flags_file = self.config_dir / "feature_flags.json"
        if flags_file.exists():
            try:
                with open(flags_file, 'r') as f:
                    data = json.load(f)

                for flag_name, flag_data in data.items():
                    self.feature_flags[flag_name] = FeatureFlag(
                        name=flag_name,
                        description=flag_data.get("description", ""),
                        enabled_environments=[Environment(env) for env in flag_data.get("enabled_environments", [])],
                        rollout_percentage=flag_data.get("rollout_percentage", 100.0),
                        conditions=flag_data.get("conditions", {}),
                        metadata=flag_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(flag_data.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(flag_data.get("updated_at", datetime.now().isoformat()))
                    )

            except Exception as e:
                logger.error(f"Failed to load feature flags: {e}")

    def _save_feature_flags(self):
        """Save feature flags to disk"""
        flags_file = self.config_dir / "feature_flags.json"
        try:
            data = {}
            for flag_name, flag in self.feature_flags.items():
                data[flag_name] = {
                    "description": flag.description,
                    "enabled_environments": [env.value for env in flag.enabled_environments],
                    "rollout_percentage": flag.rollout_percentage,
                    "conditions": flag.conditions,
                    "metadata": flag.metadata,
                    "created_at": flag.created_at.isoformat(),
                    "updated_at": flag.updated_at.isoformat()
                }

            with open(flags_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")

    def _load_canary_deployments(self):
        """Load canary deployments from disk"""
        canary_file = self.config_dir / "canary_deployments.json"
        if canary_file.exists():
            try:
                with open(canary_file, 'r') as f:
                    data = json.load(f)

                for canary_name, canary_data in data.items():
                    self.canary_deployments[canary_name] = CanaryDeployment(
                        name=canary_name,
                        description=canary_data.get("description", ""),
                        canary_percentage=canary_data.get("canary_percentage", 1.0),
                        min_duration_hours=canary_data.get("min_duration_hours", 168),
                        validation_metrics=canary_data.get("validation_metrics", []),
                        success_criteria=canary_data.get("success_criteria", {}),
                        state=CanaryState(canary_data.get("state", "inactive")),
                        start_time=datetime.fromisoformat(canary_data["start_time"]) if canary_data.get("start_time") else None,
                        end_time=datetime.fromisoformat(canary_data["end_time"]) if canary_data.get("end_time") else None,
                        max_risk_exposure_usd=canary_data.get("max_risk_exposure_usd", 1000.0),
                        auto_rollback_triggers=canary_data.get("auto_rollback_triggers", {}),
                        metrics_history=canary_data.get("metrics_history", []),
                        alerts_triggered=canary_data.get("alerts_triggered", [])
                    )

            except Exception as e:
                logger.error(f"Failed to load canary deployments: {e}")

    def _save_canary_deployments(self):
        """Save canary deployments to disk"""
        canary_file = self.config_dir / "canary_deployments.json"
        try:
            data = {}
            for canary_name, canary in self.canary_deployments.items():
                data[canary_name] = {
                    "description": canary.description,
                    "canary_percentage": canary.canary_percentage,
                    "min_duration_hours": canary.min_duration_hours,
                    "validation_metrics": canary.validation_metrics,
                    "success_criteria": canary.success_criteria,
                    "state": canary.state.value,
                    "start_time": canary.start_time.isoformat() if canary.start_time else None,
                    "end_time": canary.end_time.isoformat() if canary.end_time else None,
                    "max_risk_exposure_usd": canary.max_risk_exposure_usd,
                    "auto_rollback_triggers": canary.auto_rollback_triggers,
                    "metrics_history": canary.metrics_history,
                    "alerts_triggered": canary.alerts_triggered
                }

            with open(canary_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save canary deployments: {e}")

    def get_environment_summary(self) -> Dict[str, Any]:
        """Get comprehensive environment summary"""
        current_config = self.get_current_config()

        # Active canaries
        active_canaries = [
            name for name, canary in self.canary_deployments.items()
            if canary.state in [CanaryState.RUNNING, CanaryState.VALIDATING]
        ]

        # Feature flags summary
        enabled_flags = [
            name for name, flag in self.feature_flags.items()
            if self.current_environment in flag.enabled_environments
        ]

        return {
            "current_environment": self.current_environment.value,
            "config": {
                "name": current_config.name,
                "max_position_size_usd": current_config.max_position_size_usd,
                "max_daily_loss_usd": current_config.max_daily_loss_usd,
                "max_drawdown_percent": current_config.max_drawdown_percent,
                "api_rate_limit": current_config.api_rate_limit_per_minute,
                "debug_logging": current_config.debug_logging
            },
            "feature_flags": {
                "total": len(self.feature_flags),
                "enabled": len(enabled_flags),
                "enabled_list": enabled_flags
            },
            "canary_deployments": {
                "total": len(self.canary_deployments),
                "active": len(active_canaries),
                "active_list": active_canaries
            },
            "risk_limits": {
                "max_daily_loss_usd": current_config.max_daily_loss_usd,
                "max_drawdown_percent": current_config.max_drawdown_percent,
                "position_size_limit_percent": current_config.position_size_limit_percent,
                "max_concurrent_orders": current_config.max_concurrent_orders
            }
        }
