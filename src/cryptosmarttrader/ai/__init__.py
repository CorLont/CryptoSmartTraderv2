#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise AI Module
Production-ready AI/LLM integration with comprehensive governance
"""

from .enterprise_ai_governance import (
    get_ai_governance,
    EnterpriseAIGovernance,
    AITaskType,
    AIModelTier,
    AITaskConfig
)

from .enterprise_ai_evaluator import (
    get_ai_evaluator,
    EnterpriseAIEvaluator,
    EvaluationResult,
    ModelPerformanceSnapshot
)

from .modernized_openai_adapter import (
    get_modernized_openai_adapter,
    ModernizedOpenAIAdapter,
    NewsAnalysisResult,
    SentimentAnalysisResult
)

from .ai_feature_flags import (
    get_ai_feature_flags,
    AIFeatureFlagManager,
    FeatureState,
    RolloutStrategy,
    FeatureConfig,
    ai_feature_flag
)

__all__ = [
    # Core governance
    "get_ai_governance",
    "EnterpriseAIGovernance", 
    "AITaskType",
    "AIModelTier",
    "AITaskConfig",
    
    # Evaluation system
    "get_ai_evaluator",
    "EnterpriseAIEvaluator",
    "EvaluationResult",
    "ModelPerformanceSnapshot",
    
    # Modernized adapters
    "get_modernized_openai_adapter",
    "ModernizedOpenAIAdapter",
    "NewsAnalysisResult", 
    "SentimentAnalysisResult",
    
    # Feature flags
    "get_ai_feature_flags",
    "AIFeatureFlagManager",
    "FeatureState",
    "RolloutStrategy", 
    "FeatureConfig",
    "ai_feature_flag"
]

# Version info
__version__ = "2.0.0"
__author__ = "CryptoSmartTrader V2 Team"
__description__ = "Enterprise AI/LLM integration with comprehensive governance"