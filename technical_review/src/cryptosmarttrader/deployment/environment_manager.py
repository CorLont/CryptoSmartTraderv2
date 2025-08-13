"""
Environment Manager

Environment separation system with dev/staging/prod isolation,
feature flags, and deployment safety controls.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class FeatureFlag(Enum):
    """Feature flags for gradual rollout"""
    CANARY_TRADING = "canary_trading"
    NEW_ML_MODEL = "new_ml_model"
    ADVANCED_RISK_MANAGEMENT = "advanced_risk_management"
    EXPERIMENTAL_EXECUTION = "experimental_execution"
    BETA_DASHBOARD = "beta_dashboard"
    HIGH_FREQUENCY_TRADING = "high_frequency_trading"


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    environment: Environment

    # API Configuration
    exchange_api_keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    api_rate_limits: Dict[str, int] = field(default_factory=dict)

    # Risk Configuration
    risk_limits: Dict[str, float] = field(default_factory=dict)
    position_size_multiplier: float = 1.0

    # Trading Configuration
    max_positions: int = 10
    trading_enabled: bool = True
    paper_trading_only: bool = False

    # Feature Flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    # Database Configuration
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    # Monitoring Configuration
    monitoring_enabled: bool = True
    log_level: str = "INFO"

    # Deployment Configuration
    deployment_id: Optional[str] = None
    deployed_at: Optional[datetime] = None

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_staging(self) -> bool:
        return self.environment == Environment.STAGING

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


class EnvironmentManager:
    """
    Environment separation and configuration management
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Current environment
        self.current_environment = self._detect_environment()
        self.config = self._load_environment_config()

        # Canary deployment tracking
        self.canary_deployments: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Environment Manager initialized for {self.current_environment.value}")

    def _detect_environment(self) -> Environment:
        """Auto-detect current environment"""

        # Check environment variable first
        env_var = os.getenv("ENVIRONMENT", "").lower()
        if env_var in [env.value for env in Environment]:
            return Environment(env_var)

        # Check for deployment markers
        if os.path.exists("/etc/production"):
            return Environment.PRODUCTION
        elif os.path.exists("/etc/staging"):
            return Environment.STAGING

        # Check hostname patterns
        hostname = os.getenv("HOSTNAME", "").lower()
        if "prod" in hostname:
            return Environment.PRODUCTION
        elif "staging" in hostname or "stage" in hostname:
            return Environment.STAGING

        # Default to development
        return Environment.DEVELOPMENT

    def _load_environment_config(self) -> EnvironmentConfig:
        """Load environment-specific configuration"""

        config_file = self.config_dir / f"{self.current_environment.value}.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)

                config = EnvironmentConfig(environment=self.current_environment)

                # Load configuration sections
                config.exchange_api_keys = data.get("exchange_api_keys", {})
                config.api_rate_limits = data.get("api_rate_limits", {})
                config.risk_limits = data.get("risk_limits", {})
                config.position_size_multiplier = data.get("position_size_multiplier", 1.0)
                config.max_positions = data.get("max_positions", 10)
                config.trading_enabled = data.get("trading_enabled", True)
                config.paper_trading_only = data.get("paper_trading_only", False)
                config.feature_flags = data.get("feature_flags", {})
                config.database_url = data.get("database_url")
                config.redis_url = data.get("redis_url")
                config.monitoring_enabled = data.get("monitoring_enabled", True)
                config.log_level = data.get("log_level", "INFO")

                return config

            except Exception as e:
                logger.error(f"Failed to load environment config: {e}")

        # Return default configuration
        return self._get_default_config()

    def _get_default_config(self) -> EnvironmentConfig:
        """Get default configuration for current environment"""

        config = EnvironmentConfig(environment=self.current_environment)

        if self.current_environment == Environment.PRODUCTION:
            # Production defaults
            config.risk_limits = {
                "daily_loss_pct": -5.0,
                "max_drawdown_pct": -10.0,
                "position_size_pct": 2.0,
                "correlation_limit_pct": 70.0
            }
            config.position_size_multiplier = 1.0
            config.max_positions = 20
            config.trading_enabled = True
            config.paper_trading_only = False
            config.feature_flags = {
                FeatureFlag.CANARY_TRADING.value: False,
                FeatureFlag.NEW_ML_MODEL.value: False,
                FeatureFlag.ADVANCED_RISK_MANAGEMENT.value: True,
                FeatureFlag.EXPERIMENTAL_EXECUTION.value: False,
                FeatureFlag.BETA_DASHBOARD.value: False,
                FeatureFlag.HIGH_FREQUENCY_TRADING.value: True
            }
            config.log_level = "WARNING"

        elif self.current_environment == Environment.STAGING:
            # Staging defaults
            config.risk_limits = {
                "daily_loss_pct": -2.0,  # Tighter limits for staging
                "max_drawdown_pct": -5.0,
                "position_size_pct": 1.0,
                "correlation_limit_pct": 50.0
            }
            config.position_size_multiplier = 0.1  # 10% of normal size
            config.max_positions = 5
            config.trading_enabled = True
            config.paper_trading_only = True  # Force paper trading in staging
            config.feature_flags = {
                FeatureFlag.CANARY_TRADING.value: True,
                FeatureFlag.NEW_ML_MODEL.value: True,
                FeatureFlag.ADVANCED_RISK_MANAGEMENT.value: True,
                FeatureFlag.EXPERIMENTAL_EXECUTION.value: True,
                FeatureFlag.BETA_DASHBOARD.value: True,
                FeatureFlag.HIGH_FREQUENCY_TRADING.value: False
            }
            config.log_level = "INFO"

        else:  # Development
            # Development defaults
            config.risk_limits = {
                "daily_loss_pct": -1.0,  # Very tight for dev
                "max_drawdown_pct": -2.0,
                "position_size_pct": 0.1,
                "correlation_limit_pct": 30.0
            }
            config.position_size_multiplier = 0.01  # 1% of normal size
            config.max_positions = 3
            config.trading_enabled = True
            config.paper_trading_only = True  # Always paper trading in dev
            config.feature_flags = {flag.value: True for flag in FeatureFlag}  # All features enabled
            config.log_level = "DEBUG"

        return config

    def is_feature_enabled(self, feature: Union[str, FeatureFlag]) -> bool:
        """Check if a feature flag is enabled"""

        if isinstance(feature, FeatureFlag):
            feature_name = feature.value
        else:
            feature_name = feature

        return self.config.feature_flags.get(feature_name, False)

    def enable_feature(self, feature: Union[str, FeatureFlag], persist: bool = True):
        """Enable a feature flag"""

        if isinstance(feature, FeatureFlag):
            feature_name = feature.value
        else:
            feature_name = feature

        self.config.feature_flags[feature_name] = True

        if persist:
            self._save_config()

        logger.info(f"Enabled feature flag: {feature_name}")

    def disable_feature(self, feature: Union[str, FeatureFlag], persist: bool = True):
        """Disable a feature flag"""

        if isinstance(feature, FeatureFlag):
            feature_name = feature.value
        else:
            feature_name = feature

        self.config.feature_flags[feature_name] = False

        if persist:
            self._save_config()

        logger.info(f"Disabled feature flag: {feature_name}")

    def get_risk_limit(self, limit_name: str) -> Optional[float]:
        """Get environment-specific risk limit"""
        return self.config.risk_limits.get(limit_name)

    def set_risk_limit(self, limit_name: str, value: float, persist: bool = True):
        """Set environment-specific risk limit"""

        self.config.risk_limits[limit_name] = value

        if persist:
            self._save_config()

        logger.info(f"Set risk limit {limit_name} to {value}")

    def scale_position_size(self, base_size: float) -> float:
        """Scale position size based on environment"""
        return base_size * self.config.position_size_multiplier

    def start_canary_deployment(self,
                               deployment_name: str,
                               risk_budget_pct: float = 1.0,
                               duration_hours: int = 168) -> str:  # 1 week default
        """Start canary deployment with limited risk budget"""

        if not self.is_feature_enabled(FeatureFlag.CANARY_TRADING):
            raise ValueError("Canary trading not enabled in current environment")

        if risk_budget_pct > 5.0:  # Max 5% risk budget for canary
            raise ValueError("Canary risk budget cannot exceed 5%")

        deployment_id = self._generate_deployment_id(deployment_name)

        canary_config = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "risk_budget_pct": risk_budget_pct,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "status": "active",
            "trades_executed": 0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "error_count": 0
        }

        self.canary_deployments[deployment_id] = canary_config

        logger.info(f"Started canary deployment: {deployment_name} ({deployment_id})")
        logger.info(f"Risk budget: {risk_budget_pct}%, Duration: {duration_hours}h")

        return deployment_id

    def update_canary_metrics(self,
                             deployment_id: str,
                             trades_executed: int,
                             pnl: float,
                             max_drawdown: float,
                             error_count: int):
        """Update canary deployment metrics"""

        if deployment_id not in self.canary_deployments:
            logger.warning(f"Unknown canary deployment: {deployment_id}")
            return

        canary = self.canary_deployments[deployment_id]

        canary.update({
            "trades_executed": trades_executed,
            "pnl": pnl,
            "max_drawdown": max_drawdown,
            "error_count": error_count,
            "last_updated": datetime.now()
        })

    def evaluate_canary_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Evaluate canary deployment performance"""

        if deployment_id not in self.canary_deployments:
            return {"status": "not_found"}

        canary = self.canary_deployments[deployment_id]

        # Calculate metrics
        duration_hours = (datetime.now() - canary["start_time"]).total_seconds() / 3600

        # Safety checks
        safety_passed = True
        safety_issues = []

        # Check if risk budget exceeded
        if abs(canary["max_drawdown"]) > canary["risk_budget_pct"]:
            safety_passed = False
            safety_issues.append("Risk budget exceeded")

        # Check error rate
        if canary["trades_executed"] > 0:
            error_rate = canary["error_count"] / canary["trades_executed"]
            if error_rate > 0.05:  # 5% error rate threshold
                safety_passed = False
                safety_issues.append("High error rate")

        # Check minimum runtime
        min_runtime_hours = 24  # Minimum 24 hours for meaningful evaluation
        if duration_hours < min_runtime_hours:
            evaluation = "insufficient_runtime"
        elif safety_passed and canary["pnl"] >= 0:
            evaluation = "ready_for_production"
        elif safety_passed:
            evaluation = "continue_monitoring"
        else:
            evaluation = "stop_deployment"

        return {
            "status": canary["status"],
            "evaluation": evaluation,
            "safety_passed": safety_passed,
            "safety_issues": safety_issues,
            "duration_hours": duration_hours,
            "metrics": {
                "trades_executed": canary["trades_executed"],
                "pnl": canary["pnl"],
                "max_drawdown": canary["max_drawdown"],
                "error_count": canary["error_count"]
            },
            "recommendation": self._get_canary_recommendation(evaluation, canary)
        }

    def _get_canary_recommendation(self, evaluation: str, canary: Dict[str, Any]) -> str:
        """Get recommendation based on canary evaluation"""

        if evaluation == "ready_for_production":
            return "Canary deployment successful - safe to promote to production"
        elif evaluation == "continue_monitoring":
            return "Continue monitoring canary - performance acceptable but inconclusive"
        elif evaluation == "stop_deployment":
            return f"Stop canary deployment - safety issues detected: {canary.get('safety_issues', [])}"
        elif evaluation == "insufficient_runtime":
            return "Continue canary deployment - insufficient runtime for evaluation"
        else:
            return "Unknown evaluation status"

    def stop_canary_deployment(self, deployment_id: str, reason: str = "manual_stop"):
        """Stop canary deployment"""

        if deployment_id not in self.canary_deployments:
            logger.warning(f"Unknown canary deployment: {deployment_id}")
            return

        canary = self.canary_deployments[deployment_id]
        canary["status"] = "stopped"
        canary["stop_reason"] = reason
        canary["stopped_at"] = datetime.now()

        logger.info(f"Stopped canary deployment: {deployment_id} - Reason: {reason}")

    def _generate_deployment_id(self, deployment_name: str) -> str:
        """Generate unique deployment ID"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{deployment_name}_{timestamp}_{self.current_environment.value}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        return f"{deployment_name}_{timestamp}_{hash_suffix}"

    def get_environment_status(self) -> Dict[str, Any]:
        """Get comprehensive environment status"""

        active_canaries = [
            canary for canary in self.canary_deployments.values()
            if canary["status"] == "active"
        ]

        return {
            "environment": self.current_environment.value,
            "config": {
                "trading_enabled": self.config.trading_enabled,
                "paper_trading_only": self.config.paper_trading_only,
                "position_size_multiplier": self.config.position_size_multiplier,
                "max_positions": self.config.max_positions,
                "risk_limits": self.config.risk_limits
            },
            "feature_flags": self.config.feature_flags,
            "canary_deployments": {
                "active": len(active_canaries),
                "total": len(self.canary_deployments),
                "details": active_canaries
            },
            "deployment_info": {
                "deployment_id": self.config.deployment_id,
                "deployed_at": self.config.deployed_at.isoformat() if self.config.deployed_at else None
            }
        }

    def _save_config(self):
        """Save current configuration to disk"""

        try:
            config_file = self.config_dir / f"{self.current_environment.value}.json"

            config_data = {
                "exchange_api_keys": self.config.exchange_api_keys,
                "api_rate_limits": self.config.api_rate_limits,
                "risk_limits": self.config.risk_limits,
                "position_size_multiplier": self.config.position_size_multiplier,
                "max_positions": self.config.max_positions,
                "trading_enabled": self.config.trading_enabled,
                "paper_trading_only": self.config.paper_trading_only,
                "feature_flags": self.config.feature_flags,
                "database_url": self.config.database_url,
                "redis_url": self.config.redis_url,
                "monitoring_enabled": self.config.monitoring_enabled,
                "log_level": self.config.log_level
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Saved configuration for {self.current_environment.value}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that required secrets are available"""

        required_secrets = {
            "KRAKEN_API_KEY": os.getenv("KRAKEN_API_KEY"),
            "KRAKEN_SECRET": os.getenv("KRAKEN_SECRET"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
        }

        if self.current_environment == Environment.PRODUCTION:
            # Additional production secrets
            required_secrets.update({
                "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_CHANNEL_ID": os.getenv("SLACK_CHANNEL_ID")
            })

        validation_results = {}

        for secret_name, secret_value in required_secrets.items():
            validation_results[secret_name] = secret_value is not None and len(secret_value.strip()) > 0

        return validation_results

    def enforce_environment_safety(self):
        """Enforce environment-specific safety rules"""

        if self.current_environment == Environment.PRODUCTION:
            # Production safety checks
            if not self.config.monitoring_enabled:
                raise ValueError("Monitoring must be enabled in production")

            secrets_valid = self.validate_secrets()
            if not all(secrets_valid.values()):
                missing_secrets = [name for name, valid in secrets_valid.items() if not valid]
                raise ValueError(f"Missing production secrets: {missing_secrets}")

        elif self.current_environment == Environment.STAGING:
            # Staging safety checks
            if not self.config.paper_trading_only:
                logger.warning("Staging should use paper trading only")

            if self.config.position_size_multiplier > 0.5:
                logger.warning("Large position size multiplier in staging")

        else:  # Development
            # Development safety checks
            if not self.config.paper_trading_only:
                self.config.paper_trading_only = True
                logger.warning("Forced paper trading in development environment")

            if self.config.position_size_multiplier > 0.1:
                self.config.position_size_multiplier = 0.01
                logger.warning("Reduced position size multiplier in development")
