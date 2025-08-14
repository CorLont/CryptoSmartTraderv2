#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - AI Feature Flag System
Enterprise-grade feature flag management voor AI/LLM features
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

from core.structured_logger import get_structured_logger


class FeatureState(Enum):
    """AI feature states"""
    DISABLED = "disabled"
    DEVELOPMENT = "development"
    TESTING = "testing" 
    CANARY = "canary"
    PRODUCTION = "production"
    EMERGENCY_DISABLED = "emergency_disabled"


class RolloutStrategy(Enum):
    """Feature rollout strategies"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    PERCENTAGE = "percentage"
    USER_SEGMENT = "user_segment"
    TIME_BASED = "time_based"


@dataclass
class FeatureConfig:
    """Configuration for AI feature"""
    name: str
    description: str
    state: FeatureState = FeatureState.DISABLED
    rollout_strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE
    rollout_percentage: float = 0.0  # 0-100
    enabled_for_segments: List[str] = field(default_factory=list)
    schedule_start: Optional[datetime] = None
    schedule_end: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Emergency controls
    emergency_disable_threshold: float = 0.1  # Error rate threshold
    max_cost_per_hour: float = 10.0
    max_requests_per_minute: int = 100
    
    # Evaluation criteria
    success_criteria: Dict[str, float] = field(default_factory=dict)
    evaluation_period_hours: int = 24


@dataclass
class FeatureMetrics:
    """Real-time metrics for feature"""
    total_checks: int = 0
    enabled_checks: int = 0
    error_count: int = 0
    total_cost: float = 0.0
    avg_response_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_reset: datetime = field(default_factory=datetime.now)


class AIFeatureFlagManager:
    """Enterprise AI feature flag management"""
    
    def __init__(self, config_path: str = "config/ai_feature_flags.json"):
        self.logger = get_structured_logger("AIFeatureFlagManager")
        self.config_path = Path(config_path)
        self.lock = threading.RLock()
        
        # Feature configurations
        self.features: Dict[str, FeatureConfig] = {}
        self.metrics: Dict[str, FeatureMetrics] = defaultdict(FeatureMetrics)
        
        # Emergency controls
        self.emergency_disabled_features = set()
        self.circuit_breakers = {}
        
        # Load configurations
        self._load_feature_configs()
        
        # Initialize default AI features
        self._initialize_default_features()
        
        self.logger.info("AI Feature Flag Manager initialized")
    
    def _load_feature_configs(self):
        """Load feature configurations from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                for feature_name, config_dict in data.items():
                    # Convert datetime strings back to datetime objects
                    if config_dict.get('created_at'):
                        config_dict['created_at'] = datetime.fromisoformat(config_dict['created_at'])
                    if config_dict.get('updated_at'):
                        config_dict['updated_at'] = datetime.fromisoformat(config_dict['updated_at'])
                    if config_dict.get('schedule_start'):
                        config_dict['schedule_start'] = datetime.fromisoformat(config_dict['schedule_start'])
                    if config_dict.get('schedule_end'):
                        config_dict['schedule_end'] = datetime.fromisoformat(config_dict['schedule_end'])
                    
                    # Convert enum strings back to enums
                    config_dict['state'] = FeatureState(config_dict['state'])
                    config_dict['rollout_strategy'] = RolloutStrategy(config_dict['rollout_strategy'])
                    
                    self.features[feature_name] = FeatureConfig(**config_dict)
                
                self.logger.info(f"Loaded {len(self.features)} feature configurations")
            else:
                self.logger.info("No existing feature config found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Failed to load feature configs: {e}")
    
    def _save_feature_configs(self):
        """Save feature configurations to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {}
            for feature_name, config in self.features.items():
                config_dict = {
                    'name': config.name,
                    'description': config.description,
                    'state': config.state.value,
                    'rollout_strategy': config.rollout_strategy.value,
                    'rollout_percentage': config.rollout_percentage,
                    'enabled_for_segments': config.enabled_for_segments,
                    'schedule_start': config.schedule_start.isoformat() if config.schedule_start else None,
                    'schedule_end': config.schedule_end.isoformat() if config.schedule_end else None,
                    'dependencies': config.dependencies,
                    'tags': config.tags,
                    'created_at': config.created_at.isoformat(),
                    'updated_at': config.updated_at.isoformat(),
                    'emergency_disable_threshold': config.emergency_disable_threshold,
                    'max_cost_per_hour': config.max_cost_per_hour,
                    'max_requests_per_minute': config.max_requests_per_minute,
                    'success_criteria': config.success_criteria,
                    'evaluation_period_hours': config.evaluation_period_hours
                }
                data[feature_name] = config_dict
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save feature configs: {e}")
    
    def _initialize_default_features(self):
        """Initialize default AI features"""
        default_features = [
            {
                "name": "news_analysis",
                "description": "AI-powered news impact analysis",
                "state": FeatureState.DEVELOPMENT,
                "tags": ["ai", "news", "analysis"],
                "max_cost_per_hour": 5.0,
                "success_criteria": {"accuracy": 0.7, "response_time_ms": 3000}
            },
            {
                "name": "sentiment_analysis", 
                "description": "Market sentiment analysis with LLM",
                "state": FeatureState.TESTING,
                "tags": ["ai", "sentiment", "market"],
                "max_cost_per_hour": 2.0,
                "success_criteria": {"accuracy": 0.8, "response_time_ms": 2000}
            },
            {
                "name": "market_prediction",
                "description": "AI market prediction and forecasting",
                "state": FeatureState.DISABLED,
                "tags": ["ai", "prediction", "experimental"],
                "max_cost_per_hour": 10.0,
                "success_criteria": {"accuracy": 0.6, "response_time_ms": 5000}
            },
            {
                "name": "risk_assessment",
                "description": "AI-powered risk assessment",
                "state": FeatureState.CANARY,
                "rollout_percentage": 10.0,
                "tags": ["ai", "risk", "safety"],
                "max_cost_per_hour": 3.0,
                "success_criteria": {"accuracy": 0.85, "response_time_ms": 1500}
            },
            {
                "name": "anomaly_detection",
                "description": "AI anomaly detection in market data",
                "state": FeatureState.TESTING,
                "tags": ["ai", "anomaly", "monitoring"],
                "max_cost_per_hour": 1.0,
                "success_criteria": {"precision": 0.9, "recall": 0.7}
            }
        ]
        
        with self.lock:
            for feature_data in default_features:
                feature_name = feature_data["name"]
                if feature_name not in self.features:
                    self.features[feature_name] = FeatureConfig(**feature_data)
            
            self._save_feature_configs()
    
    def is_feature_enabled(self, 
                          feature_name: str, 
                          user_segment: Optional[str] = None,
                          user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled for given context"""
        
        with self.lock:
            # Record the check
            self.metrics[feature_name].total_checks += 1
            
            # Check if feature exists
            if feature_name not in self.features:
                self.logger.warning(f"Unknown feature: {feature_name}")
                return False
            
            feature = self.features[feature_name]
            
            # Emergency disabled
            if feature_name in self.emergency_disabled_features:
                self.logger.info(f"Feature {feature_name} emergency disabled")
                return False
            
            # Check basic state
            if feature.state == FeatureState.DISABLED:
                return False
            elif feature.state == FeatureState.EMERGENCY_DISABLED:
                return False
            elif feature.state == FeatureState.PRODUCTION:
                enabled = True
            elif feature.state == FeatureState.DEVELOPMENT:
                # Only enabled in development environment
                enabled = self._is_development_environment()
            elif feature.state == FeatureState.TESTING:
                # Only enabled for testing
                enabled = self._is_testing_environment()
            elif feature.state == FeatureState.CANARY:
                # Use rollout strategy
                enabled = self._check_rollout_eligibility(feature, user_segment, user_id)
            else:
                enabled = False
            
            # Check dependencies
            if enabled:
                for dependency in feature.dependencies:
                    if not self.is_feature_enabled(dependency, user_segment, user_id):
                        enabled = False
                        break
            
            # Check schedule
            if enabled and feature.schedule_start and feature.schedule_end:
                now = datetime.now()
                if not (feature.schedule_start <= now <= feature.schedule_end):
                    enabled = False
            
            # Record enabled check
            if enabled:
                self.metrics[feature_name].enabled_checks += 1
            
            return enabled
    
    def _check_rollout_eligibility(self, 
                                  feature: FeatureConfig,
                                  user_segment: Optional[str],
                                  user_id: Optional[str]) -> bool:
        """Check if user is eligible for rollout"""
        
        if feature.rollout_strategy == RolloutStrategy.IMMEDIATE:
            return True
        
        elif feature.rollout_strategy == RolloutStrategy.PERCENTAGE:
            if user_id:
                # Consistent hash-based assignment
                hash_val = hash(f"{feature.name}:{user_id}") % 100
                return hash_val < feature.rollout_percentage
            else:
                # Random assignment
                import random
                return random.random() * 100 < feature.rollout_percentage
        
        elif feature.rollout_strategy == RolloutStrategy.USER_SEGMENT:
            return user_segment in feature.enabled_for_segments
        
        elif feature.rollout_strategy == RolloutStrategy.TIME_BASED:
            # Gradual rollout based on time
            if feature.schedule_start and feature.schedule_end:
                now = datetime.now()
                total_duration = (feature.schedule_end - feature.schedule_start).total_seconds()
                elapsed = (now - feature.schedule_start).total_seconds()
                progress = min(1.0, elapsed / total_duration)
                return progress * 100 >= feature.rollout_percentage
        
        return False
    
    def _is_development_environment(self) -> bool:
        """Check if running in development"""
        import os
        return os.getenv("ENVIRONMENT", "development") == "development"
    
    def _is_testing_environment(self) -> bool:
        """Check if running in testing"""
        import os
        return os.getenv("ENVIRONMENT", "development") in ["testing", "development"]
    
    def record_feature_error(self, feature_name: str, error: str, cost_usd: float = 0.0):
        """Record feature error for monitoring"""
        with self.lock:
            if feature_name not in self.metrics:
                return
            
            metrics = self.metrics[feature_name]
            metrics.error_count += 1
            metrics.total_cost += cost_usd
            metrics.last_error = error
            
            # Check emergency disable threshold
            if metrics.total_checks > 10:  # Minimum sample size
                error_rate = metrics.error_count / metrics.total_checks
                feature = self.features.get(feature_name)
                
                if feature and error_rate > feature.emergency_disable_threshold:
                    self.emergency_disable_feature(feature_name, f"Error rate {error_rate:.1%} exceeded threshold")
    
    def record_feature_success(self, 
                             feature_name: str,
                             response_time_ms: float,
                             cost_usd: float = 0.0):
        """Record successful feature usage"""
        with self.lock:
            if feature_name not in self.metrics:
                return
            
            metrics = self.metrics[feature_name]
            metrics.total_cost += cost_usd
            
            # Update average response time
            if metrics.enabled_checks > 1:
                metrics.avg_response_time_ms = (
                    (metrics.avg_response_time_ms * (metrics.enabled_checks - 1) + response_time_ms) /
                    metrics.enabled_checks
                )
            else:
                metrics.avg_response_time_ms = response_time_ms
    
    def emergency_disable_feature(self, feature_name: str, reason: str):
        """Emergency disable feature"""
        with self.lock:
            self.emergency_disabled_features.add(feature_name)
            
            if feature_name in self.features:
                self.features[feature_name].state = FeatureState.EMERGENCY_DISABLED
                self.features[feature_name].updated_at = datetime.now()
                self._save_feature_configs()
            
            self.logger.error(f"EMERGENCY DISABLED feature {feature_name}: {reason}")
    
    def enable_feature(self, feature_name: str, state: FeatureState = FeatureState.PRODUCTION):
        """Enable feature with specified state"""
        with self.lock:
            if feature_name in self.features:
                self.features[feature_name].state = state
                self.features[feature_name].updated_at = datetime.now()
                
                # Remove from emergency disabled if applicable
                self.emergency_disabled_features.discard(feature_name)
                
                self._save_feature_configs()
                self.logger.info(f"Enabled feature {feature_name} with state {state.value}")
    
    def update_rollout_percentage(self, feature_name: str, percentage: float):
        """Update rollout percentage for canary deployment"""
        with self.lock:
            if feature_name in self.features:
                self.features[feature_name].rollout_percentage = max(0.0, min(100.0, percentage))
                self.features[feature_name].updated_at = datetime.now()
                self._save_feature_configs()
                self.logger.info(f"Updated rollout for {feature_name} to {percentage}%")
    
    def get_feature_status(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive feature status"""
        with self.lock:
            if feature_name not in self.features:
                return {"error": "Feature not found"}
            
            feature = self.features[feature_name]
            metrics = self.metrics[feature_name]
            
            error_rate = 0.0
            if metrics.total_checks > 0:
                error_rate = metrics.error_count / metrics.total_checks
            
            enabled_rate = 0.0
            if metrics.total_checks > 0:
                enabled_rate = metrics.enabled_checks / metrics.total_checks
            
            return {
                "name": feature.name,
                "description": feature.description,
                "state": feature.state.value,
                "rollout_strategy": feature.rollout_strategy.value,
                "rollout_percentage": feature.rollout_percentage,
                "tags": feature.tags,
                "emergency_disabled": feature_name in self.emergency_disabled_features,
                "metrics": {
                    "total_checks": metrics.total_checks,
                    "enabled_checks": metrics.enabled_checks,
                    "enabled_rate": enabled_rate,
                    "error_count": metrics.error_count,
                    "error_rate": error_rate,
                    "total_cost": metrics.total_cost,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "last_error": metrics.last_error
                },
                "thresholds": {
                    "emergency_disable_threshold": feature.emergency_disable_threshold,
                    "max_cost_per_hour": feature.max_cost_per_hour,
                    "max_requests_per_minute": feature.max_requests_per_minute
                },
                "success_criteria": feature.success_criteria,
                "updated_at": feature.updated_at.isoformat()
            }
    
    def get_all_features_status(self) -> Dict[str, Any]:
        """Get status of all features"""
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "total_features": len(self.features),
                "emergency_disabled_count": len(self.emergency_disabled_features),
                "features": {
                    name: self.get_feature_status(name)
                    for name in self.features.keys()
                }
            }
    
    def create_feature(self, feature_config: FeatureConfig) -> bool:
        """Create new feature"""
        with self.lock:
            if feature_config.name in self.features:
                self.logger.warning(f"Feature {feature_config.name} already exists")
                return False
            
            self.features[feature_config.name] = feature_config
            self._save_feature_configs()
            self.logger.info(f"Created feature {feature_config.name}")
            return True


# Global singleton
_feature_flag_manager_instance = None

def get_ai_feature_flags() -> AIFeatureFlagManager:
    """Get singleton AI feature flag manager"""
    global _feature_flag_manager_instance
    if _feature_flag_manager_instance is None:
        _feature_flag_manager_instance = AIFeatureFlagManager()
    return _feature_flag_manager_instance


# Decorator for feature-flagged AI functions
def ai_feature_flag(feature_name: str, fallback_value: Any = None):
    """Decorator to wrap AI functions with feature flags"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            flags = get_ai_feature_flags()
            
            if flags.is_feature_enabled(feature_name):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    flags.record_feature_success(feature_name, response_time_ms)
                    return result
                    
                except Exception as e:
                    flags.record_feature_error(feature_name, str(e))
                    if fallback_value is not None:
                        return fallback_value
                    raise e
            else:
                if fallback_value is not None:
                    return fallback_value
                raise RuntimeError(f"AI feature {feature_name} is disabled")
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Basic validation
    flags = get_ai_feature_flags()
    
    # Test feature checks
    print(f"News analysis enabled: {flags.is_feature_enabled('news_analysis')}")
    print(f"Market prediction enabled: {flags.is_feature_enabled('market_prediction')}")
    
    # Show status
    status = flags.get_all_features_status()
    print(json.dumps(status, indent=2))