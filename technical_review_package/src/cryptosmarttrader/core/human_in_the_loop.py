#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Human-in-the-Loop Learning System
Active learning and feedback integration for model improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
from enum import Enum

# ML imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class FeedbackType(Enum):
    """Types of human feedback"""

    TRADE_QUALITY = "trade_quality"
    PREDICTION_ACCURACY = "prediction_accuracy"
    FEATURE_RELEVANCE = "feature_relevance"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_PREFERENCE = "strategy_preference"


class FeedbackValue(Enum):
    """Feedback values"""

    EXCELLENT = 5
    GOOD = 4
    NEUTRAL = 3
    POOR = 2
    TERRIBLE = 1


@dataclass
class HumanFeedback:
    """Represents human feedback on system decisions"""

    feedback_id: str
    feedback_type: FeedbackType
    feedback_value: FeedbackValue
    context: Dict[str, Any]
    explanation: str
    timestamp: datetime
    user_id: str
    confidence: float  # User's confidence in their feedback
    metadata: Dict[str, Any]


@dataclass
class ActiveLearningQuery:
    """Represents a query for active learning"""

    query_id: str
    query_type: str
    data_point: Dict[str, Any]
    uncertainty_score: float
    priority: int
    question: str
    options: List[str]
    timestamp: datetime


class FeedbackProcessor(ABC):
    """Abstract base class for feedback processors"""

    @abstractmethod
    def process_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Process human feedback and return insights"""
        pass


class TradeFeedbackProcessor(FeedbackProcessor):
    """Processes trade quality feedback"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Process trade quality feedback"""

        trade_context = feedback.context.get("trade", {})

        # Extract trade features
        features = {
            "entry_price": trade_context.get("entry_price", 0),
            "exit_price": trade_context.get("exit_price", 0),
            "holding_period": trade_context.get("holding_period_hours", 0),
            "position_size": trade_context.get("position_size", 0),
            "market_conditions": trade_context.get("market_conditions", {}),
            "sentiment_score": trade_context.get("sentiment_score", 0),
            "technical_indicators": trade_context.get("technical_indicators", {}),
            "ml_confidence": trade_context.get("ml_confidence", 0),
        }

        # Calculate trade outcome
        if trade_context.get("exit_price") and trade_context.get("entry_price"):
            actual_return = (
                trade_context["exit_price"] - trade_context["entry_price"]
            ) / trade_context["entry_price"]
        else:
            actual_return = 0

        # Analyze feedback
        feedback_score = feedback.feedback_value.value

        # Determine learning insights
        insights = {
            "feedback_score": feedback_score,
            "actual_return": actual_return,
            "features": features,
            "misalignment": self._calculate_misalignment(feedback_score, actual_return),
            "improvement_areas": self._identify_improvement_areas(feedback, features),
            "confidence_calibration": self._assess_confidence_calibration(feedback, features),
        }

        return insights

    def _calculate_misalignment(self, feedback_score: float, actual_return: float) -> float:
        """Calculate misalignment between feedback and actual performance"""

        # Normalize actual return to 1-5 scale
        if actual_return > 0.1:  # >10% return
            performance_score = 5
        elif actual_return > 0.05:  # 5-10% return
            performance_score = 4
        elif actual_return > 0:  # Positive return
            performance_score = 3
        elif actual_return > -0.05:  # Small loss
            performance_score = 2
        else:  # Large loss
            performance_score = 1

        # Calculate alignment
        misalignment = abs(feedback_score - performance_score)
        return misalignment / 4  # Normalize to 0-1

    def _identify_improvement_areas(
        self, feedback: HumanFeedback, features: Dict[str, Any]
    ) -> List[str]:
        """Identify areas for improvement based on feedback"""

        improvement_areas = []

        if feedback.feedback_value.value <= 2:  # Poor feedback
            # Check various aspects
            if features.get("ml_confidence", 0) > 0.8:
                improvement_areas.append("overconfident_predictions")

            if features.get("holding_period", 0) < 2:
                improvement_areas.append("exit_timing")

            if features.get("position_size", 0) > 0.1:
                improvement_areas.append("position_sizing")

            market_volatility = features.get("market_conditions", {}).get("volatility", 0)
            if market_volatility > 0.05:
                improvement_areas.append("volatility_handling")

        return improvement_areas

    def _assess_confidence_calibration(
        self, feedback: HumanFeedback, features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess how well model confidence aligns with outcomes"""

        ml_confidence = features.get("ml_confidence", 0)
        human_confidence = feedback.confidence
        feedback_score = feedback.feedback_value.value

        return {
            "ml_confidence": ml_confidence,
            "human_confidence": human_confidence,
            "feedback_quality": feedback_score / 5,
            "confidence_gap": abs(ml_confidence - human_confidence),
            "calibration_score": 1 - abs(ml_confidence - feedback_score / 5),
        }


class PredictionFeedbackProcessor(FeedbackProcessor):
    """Processes prediction accuracy feedback"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Process prediction accuracy feedback"""

        prediction_context = feedback.context.get("prediction", {})

        insights = {
            "prediction_type": prediction_context.get("type", "unknown"),
            "predicted_value": prediction_context.get("predicted_value", 0),
            "actual_value": prediction_context.get("actual_value", 0),
            "time_horizon": prediction_context.get("time_horizon", "1d"),
            "feedback_score": feedback.feedback_value.value,
            "model_features": prediction_context.get("features_used", []),
            "market_regime": prediction_context.get("market_regime", "unknown"),
            "accuracy_assessment": self._assess_prediction_accuracy(feedback, prediction_context),
        }

        return insights

    def _assess_prediction_accuracy(
        self, feedback: HumanFeedback, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess prediction accuracy based on feedback"""

        predicted = context.get("predicted_value", 0)
        actual = context.get("actual_value", 0)

        if actual != 0:
            error_rate = abs(predicted - actual) / abs(actual)
        else:
            error_rate = abs(predicted)

        return {
            "error_rate": error_rate,
            "human_assessment": feedback.feedback_value.value / 5,
            "direction_correct": (predicted > 0) == (actual > 0),
            "magnitude_accuracy": max(0, 1 - error_rate),
        }


class ActiveLearningEngine:
    """Engine for active learning and uncertainty sampling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pending_queries = []
        self.query_history = []

    def identify_uncertain_predictions(
        self, predictions: np.ndarray, features: np.ndarray, threshold: float = 0.3
    ) -> List[int]:
        """Identify predictions with high uncertainty for human feedback"""

        # Calculate uncertainty (simplified - can be enhanced with ensemble variance)
        if len(predictions.shape) > 1:
            # Multi-class predictions - use entropy
            uncertainties = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        else:
            # Regression predictions - use variance proxy
            uncertainties = np.abs(predictions - np.mean(predictions))

        # Normalize uncertainties
        if np.std(uncertainties) > 0:
            uncertainties = (uncertainties - np.mean(uncertainties)) / np.std(uncertainties)

        # Find highly uncertain predictions
        uncertain_indices = np.where(uncertainties > threshold)[0]

        return uncertain_indices.tolist()

    def generate_active_learning_query(
        self, data_point: Dict[str, Any], uncertainty_score: float
    ) -> ActiveLearningQuery:
        """Generate an active learning query for human feedback"""

        query_id = f"al_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.normal(0, 1)}"

        # Determine query type based on data
        if "prediction" in data_point:
            query_type = "prediction_validation"
            question = f"How accurate is this prediction: {data_point['prediction']:.3f}?"
            options = [
                "Very accurate",
                "Somewhat accurate",
                "Neutral",
                "Somewhat inaccurate",
                "Very inaccurate",
            ]
        elif "trade_signal" in data_point:
            query_type = "trade_signal_validation"
            question = (
                f"Should we {data_point['trade_signal']} {data_point.get('coin', 'this asset')}?"
            )
            options = [
                "Definitely yes",
                "Probably yes",
                "Uncertain",
                "Probably no",
                "Definitely no",
            ]
        else:
            query_type = "general_validation"
            question = "How confident are you in this analysis?"
            options = ["Very confident", "Confident", "Neutral", "Low confidence", "No confidence"]

        # Calculate priority
        priority = min(10, int(uncertainty_score * 10))

        query = ActiveLearningQuery(
            query_id=query_id,
            query_type=query_type,
            data_point=data_point,
            uncertainty_score=uncertainty_score,
            priority=priority,
            question=question,
            options=options,
            timestamp=datetime.now(),
        )

        self.pending_queries.append(query)
        return query

    def process_query_response(
        self, query_id: str, response: str, confidence: float
    ) -> HumanFeedback:
        """Process response to active learning query"""

        # Find the query
        query = None
        for q in self.pending_queries:
            if q.query_id == query_id:
                query = q
                break

        if not query:
            raise ValueError(f"Query {query_id} not found")

        # Convert response to feedback
        if query.query_type == "prediction_validation":
            feedback_type = FeedbackType.PREDICTION_ACCURACY
        elif query.query_type == "trade_signal_validation":
            feedback_type = FeedbackType.TRADE_QUALITY
        else:
            feedback_type = FeedbackType.STRATEGY_PREFERENCE

        # Map response to feedback value
        response_mapping = {
            0: FeedbackValue.EXCELLENT,
            1: FeedbackValue.GOOD,
            2: FeedbackValue.NEUTRAL,
            3: FeedbackValue.POOR,
            4: FeedbackValue.TERRIBLE,
        }

        # Find response index
        response_idx = 2  # Default to neutral
        for i, option in enumerate(query.options):
            if response.lower() in option.lower():
                response_idx = i
                break

        feedback_value = response_mapping.get(response_idx, FeedbackValue.NEUTRAL)

        feedback = HumanFeedback(
            feedback_id=f"fb_{query_id}",
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            context={"query": asdict(query), "data_point": query.data_point},
            explanation=response,
            timestamp=datetime.now(),
            user_id="human_expert",
            confidence=confidence,
            metadata={"source": "active_learning"},
        )

        # Move query to history
        self.pending_queries.remove(query)
        self.query_history.append(query)

        return feedback

    def get_priority_queries(self, limit: int = 5) -> List[ActiveLearningQuery]:
        """Get highest priority queries for human feedback"""

        # Sort by priority (descending) and timestamp
        sorted_queries = sorted(
            self.pending_queries, key=lambda q: (q.priority, q.timestamp), reverse=True
        )

        return sorted_queries[:limit]


class HumanInTheLoopSystem:
    """Main system for human-in-the-loop learning"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.feedback_processors = {
            FeedbackType.TRADE_QUALITY: TradeFeedbackProcessor(),
            FeedbackType.PREDICTION_ACCURACY: PredictionFeedbackProcessor(),
        }

        self.active_learning = ActiveLearningEngine()
        self.feedback_history = []
        self.learning_metrics = {}

    def submit_feedback(self, feedback: HumanFeedback) -> Dict[str, Any]:
        """Submit human feedback to the system"""

        self.feedback_history.append(feedback)

        # Process feedback
        if feedback.feedback_type in self.feedback_processors:
            processor = self.feedback_processors[feedback.feedback_type]
            insights = processor.process_feedback(feedback)

            # Update learning metrics
            self._update_learning_metrics(feedback, insights)

            self.logger.info(
                f"Processed feedback {feedback.feedback_id}: {feedback.feedback_type.value}"
            )

            return insights
        else:
            self.logger.warning(f"No processor for feedback type: {feedback.feedback_type}")
            return {}

    def request_feedback(
        self, data_point: Dict[str, Any], uncertainty_score: float
    ) -> ActiveLearningQuery:
        """Request human feedback on uncertain predictions"""

        query = self.active_learning.generate_active_learning_query(data_point, uncertainty_score)

        self.logger.info(f"Generated active learning query: {query.query_id}")

        return query

    def get_feedback_insights(self) -> Dict[str, Any]:
        """Get insights from all collected feedback"""

        if not self.feedback_history:
            return {"message": "No feedback collected yet"}

        insights = {
            "total_feedback": len(self.feedback_history),
            "feedback_distribution": {},
            "average_confidence": 0,
            "improvement_trends": {},
            "calibration_metrics": {},
            "last_updated": datetime.now(),
        }

        # Analyze feedback distribution
        for feedback in self.feedback_history:
            fb_type = feedback.feedback_type.value
            if fb_type not in insights["feedback_distribution"]:
                insights["feedback_distribution"][fb_type] = []
            insights["feedback_distribution"][fb_type].append(feedback.feedback_value.value)

        # Calculate average confidence
        insights["average_confidence"] = np.mean([f.confidence for f in self.feedback_history])

        # Analyze trends
        recent_feedback = [
            f for f in self.feedback_history if f.timestamp > datetime.now() - timedelta(days=7)
        ]

        if recent_feedback:
            insights["recent_feedback_quality"] = np.mean(
                [f.feedback_value.value for f in recent_feedback]
            )
            insights["recent_confidence"] = np.mean([f.confidence for f in recent_feedback])

        return insights

    def _update_learning_metrics(self, feedback: HumanFeedback, insights: Dict[str, Any]):
        """Update learning metrics based on feedback"""

        metric_key = f"{feedback.feedback_type.value}_metrics"

        if metric_key not in self.learning_metrics:
            self.learning_metrics[metric_key] = []

        metric_entry = {
            "timestamp": feedback.timestamp,
            "feedback_value": feedback.feedback_value.value,
            "confidence": feedback.confidence,
            "insights": insights,
        }

        self.learning_metrics[metric_key].append(metric_entry)

        # Keep only recent metrics (last 1000 entries)
        if len(self.learning_metrics[metric_key]) > 1000:
            self.learning_metrics[metric_key] = self.learning_metrics[metric_key][-1000:]

    def get_pending_queries(self, limit: int = 5) -> List[ActiveLearningQuery]:
        """Get pending queries for human review"""

        return self.active_learning.get_priority_queries(limit)

    def respond_to_query(self, query_id: str, response: str, confidence: float) -> HumanFeedback:
        """Respond to an active learning query"""

        feedback = self.active_learning.process_query_response(query_id, response, confidence)
        self.submit_feedback(feedback)
        return feedback

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""

        summary = {
            "feedback_stats": {
                "total_feedback": len(self.feedback_history),
                "pending_queries": len(self.active_learning.pending_queries),
                "processed_queries": len(self.active_learning.query_history),
            },
            "learning_progress": self.get_feedback_insights(),
            "active_learning_status": {
                "high_priority_queries": len(
                    [q for q in self.active_learning.pending_queries if q.priority >= 7]
                ),
                "total_pending": len(self.active_learning.pending_queries),
            },
            "last_updated": datetime.now(),
        }

        return summary


# Global instance
_hitl_system = None


def get_human_in_the_loop_system() -> HumanInTheLoopSystem:
    """Get or create human-in-the-loop system"""
    global _hitl_system

    if _hitl_system is None:
        _hitl_system = HumanInTheLoopSystem()

    return _hitl_system


def submit_trade_feedback(
    trade_context: Dict[str, Any],
    feedback_value: FeedbackValue,
    explanation: str,
    confidence: float = 0.8,
) -> Dict[str, Any]:
    """Submit feedback on trade quality"""

    system = get_human_in_the_loop_system()

    feedback = HumanFeedback(
        feedback_id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        feedback_type=FeedbackType.TRADE_QUALITY,
        feedback_value=feedback_value,
        context={"trade": trade_context},
        explanation=explanation,
        timestamp=datetime.now(),
        user_id="trader",
        confidence=confidence,
        metadata={"source": "manual_trade_feedback"},
    )

    return system.submit_feedback(feedback)


def request_prediction_feedback(
    prediction_context: Dict[str, Any], uncertainty_score: float
) -> ActiveLearningQuery:
    """Request feedback on uncertain predictions"""

    system = get_human_in_the_loop_system()

    data_point = {
        "prediction": prediction_context.get("predicted_value", 0),
        "confidence": prediction_context.get("confidence", 0),
        "features": prediction_context.get("features_used", []),
        "coin": prediction_context.get("coin", "Unknown"),
    }

    return system.request_feedback(data_point, uncertainty_score)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    system = get_human_in_the_loop_system()

    # Submit demo feedback
    trade_context = {
        "entry_price": 45000,
        "exit_price": 46800,
        "holding_period_hours": 12,
        "position_size": 0.05,
        "ml_confidence": 0.85,
        "sentiment_score": 0.7,
    }

    insights = submit_trade_feedback(
        trade_context,
        FeedbackValue.GOOD,
        "Good trade, but exit could have been timed better",
        confidence=0.8,
    )

    print("Trade feedback insights:", insights)

    # Get system summary
    summary = system.get_system_summary()
    print("System summary:", summary)
