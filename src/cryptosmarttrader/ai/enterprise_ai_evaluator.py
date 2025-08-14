#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise AI Evaluation System
Comprehensive framework voor AI model performance monitoring en evaluation
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import threading

from core.structured_logger import get_structured_logger


class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    RESPONSE_TIME = "response_time"
    COST_EFFICIENCY = "cost_efficiency" 
    ACCURACY = "accuracy"
    SCHEMA_COMPLIANCE = "schema_compliance"
    CONSISTENCY = "consistency"
    RELIABILITY = "reliability"


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance metrics"""
    model_name: str
    timestamp: datetime
    total_requests: int
    successful_requests: int
    avg_response_time_ms: float
    total_cost_usd: float
    accuracy_score: float
    schema_compliance_rate: float
    reliability_score: float
    cost_per_successful_request: float


@dataclass
class EvaluationResult:
    """Result of AI response evaluation"""
    model_name: str
    task_type: str
    response_quality_score: float  # 0.0 to 1.0
    schema_compliance: bool
    response_time_ms: float
    cost_usd: float
    accuracy_estimate: float
    issues: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ResponseQualityAnalyzer:
    """Analyzes quality of AI responses"""
    
    def __init__(self):
        self.logger = get_structured_logger("ResponseQualityAnalyzer")
    
    def analyze_news_analysis_quality(self, response: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze quality of news analysis response"""
        issues = []
        quality_score = 1.0
        
        # Check required fields
        required_fields = ["sentiment", "impact_magnitude", "confidence", "reasoning"]
        for field in required_fields:
            if field not in response:
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.2
        
        # Validate sentiment values
        valid_sentiments = ["bullish", "bearish", "neutral"]
        if response.get("sentiment") not in valid_sentiments:
            issues.append(f"Invalid sentiment: {response.get('sentiment')}")
            quality_score -= 0.3
        
        # Check numeric ranges
        impact_mag = response.get("impact_magnitude", -1)
        if not (0.0 <= impact_mag <= 1.0):
            issues.append(f"Impact magnitude out of range: {impact_mag}")
            quality_score -= 0.2
        
        confidence = response.get("confidence", -1)
        if not (0.0 <= confidence <= 1.0):
            issues.append(f"Confidence out of range: {confidence}")
            quality_score -= 0.2
        
        # Check reasoning quality
        reasoning = response.get("reasoning", "")
        if len(reasoning) < 10:
            issues.append("Reasoning too short or missing")
            quality_score -= 0.1
        
        # Consistency checks
        if impact_mag > 0.7 and confidence < 0.3:
            issues.append("High impact with low confidence - inconsistent")
            quality_score -= 0.15
        
        return max(0.0, quality_score), issues
    
    def analyze_sentiment_quality(self, response: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze quality of sentiment analysis response"""
        issues = []
        quality_score = 1.0
        
        # Check required fields
        required_fields = ["sentiment_score", "confidence"]
        for field in required_fields:
            if field not in response:
                issues.append(f"Missing required field: {field}")
                quality_score -= 0.3
        
        # Check numeric ranges
        sentiment_score = response.get("sentiment_score", -99)
        if not (-1.0 <= sentiment_score <= 1.0):
            issues.append(f"Sentiment score out of range: {sentiment_score}")
            quality_score -= 0.4
        
        confidence = response.get("confidence", -1)
        if not (0.0 <= confidence <= 1.0):
            issues.append(f"Confidence out of range: {confidence}")
            quality_score -= 0.3
        
        return max(0.0, quality_score), issues
    
    def analyze_general_quality(self, response: Dict[str, Any]) -> Tuple[float, List[str]]:
        """General quality analysis for unknown task types"""
        issues = []
        quality_score = 0.5  # Neutral baseline
        
        if not isinstance(response, dict):
            issues.append("Response is not a dictionary")
            quality_score = 0.0
        
        if len(response) == 0:
            issues.append("Empty response")
            quality_score = 0.0
        
        return quality_score, issues


class ABTestManager:
    """Manages A/B testing for AI models"""
    
    def __init__(self):
        self.logger = get_structured_logger("ABTestManager")
        self.active_tests = {}
        self.test_results = defaultdict(list)
        self.lock = threading.Lock()
    
    def start_ab_test(self, 
                     test_name: str,
                     model_a: str,
                     model_b: str,
                     traffic_split: float = 0.5,
                     duration_hours: int = 24) -> bool:
        """Start A/B test between two models"""
        with self.lock:
            if test_name in self.active_tests:
                self.logger.warning(f"A/B test {test_name} already active")
                return False
            
            self.active_tests[test_name] = {
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": traffic_split,
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(hours=duration_hours),
                "requests_a": 0,
                "requests_b": 0,
                "success_a": 0,
                "success_b": 0,
                "total_cost_a": 0.0,
                "total_cost_b": 0.0,
                "avg_latency_a": 0.0,
                "avg_latency_b": 0.0
            }
            
            self.logger.info(f"Started A/B test {test_name}: {model_a} vs {model_b}")
            return True
    
    def should_use_model_b(self, test_name: str) -> bool:
        """Determine if should use model B for this request"""
        import random
        
        with self.lock:
            if test_name not in self.active_tests:
                return False
            
            test = self.active_tests[test_name]
            
            # Check if test expired
            if datetime.now() > test["end_time"]:
                self.logger.info(f"A/B test {test_name} expired")
                return False
            
            return random.random() < test["traffic_split"]
    
    def record_ab_result(self, 
                        test_name: str, 
                        model_used: str, 
                        success: bool,
                        latency_ms: float,
                        cost_usd: float):
        """Record A/B test result"""
        with self.lock:
            if test_name not in self.active_tests:
                return
            
            test = self.active_tests[test_name]
            
            if model_used == test["model_a"]:
                test["requests_a"] += 1
                if success:
                    test["success_a"] += 1
                test["total_cost_a"] += cost_usd
                
                # Update average latency
                if test["requests_a"] > 1:
                    test["avg_latency_a"] = (
                        (test["avg_latency_a"] * (test["requests_a"] - 1) + latency_ms) / 
                        test["requests_a"]
                    )
                else:
                    test["avg_latency_a"] = latency_ms
                    
            elif model_used == test["model_b"]:
                test["requests_b"] += 1
                if success:
                    test["success_b"] += 1
                test["total_cost_b"] += cost_usd
                
                if test["requests_b"] > 1:
                    test["avg_latency_b"] = (
                        (test["avg_latency_b"] * (test["requests_b"] - 1) + latency_ms) / 
                        test["requests_b"]
                    )
                else:
                    test["avg_latency_b"] = latency_ms
    
    def get_ab_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results"""
        with self.lock:
            if test_name not in self.active_tests:
                return None
            
            test = self.active_tests[test_name]
            
            # Calculate success rates
            success_rate_a = test["success_a"] / max(1, test["requests_a"])
            success_rate_b = test["success_b"] / max(1, test["requests_b"])
            
            # Calculate cost efficiency
            cost_per_success_a = test["total_cost_a"] / max(1, test["success_a"])
            cost_per_success_b = test["total_cost_b"] / max(1, test["success_b"])
            
            return {
                "test_name": test_name,
                "model_a": test["model_a"],
                "model_b": test["model_b"],
                "status": "active" if datetime.now() <= test["end_time"] else "completed",
                "duration_hours": (datetime.now() - test["start_time"]).total_seconds() / 3600,
                "results": {
                    "model_a": {
                        "requests": test["requests_a"],
                        "success_rate": success_rate_a,
                        "avg_latency_ms": test["avg_latency_a"],
                        "total_cost": test["total_cost_a"],
                        "cost_per_success": cost_per_success_a
                    },
                    "model_b": {
                        "requests": test["requests_b"],
                        "success_rate": success_rate_b,
                        "avg_latency_ms": test["avg_latency_b"],
                        "total_cost": test["total_cost_b"],
                        "cost_per_success": cost_per_success_b
                    }
                },
                "winner": self._determine_winner(success_rate_a, success_rate_b, 
                                               cost_per_success_a, cost_per_success_b)
            }
    
    def _determine_winner(self, 
                         success_a: float, success_b: float,
                         cost_a: float, cost_b: float) -> str:
        """Determine A/B test winner based on success rate and cost"""
        
        # Weight success rate higher than cost
        score_a = success_a * 0.7 + (1.0 / max(0.001, cost_a)) * 0.3
        score_b = success_b * 0.7 + (1.0 / max(0.001, cost_b)) * 0.3
        
        if abs(score_a - score_b) < 0.05:
            return "tie"
        return "model_a" if score_a > score_b else "model_b"


class EnterpriseAIEvaluator:
    """Main enterprise AI evaluation system"""
    
    def __init__(self):
        self.logger = get_structured_logger("EnterpriseAIEvaluator")
        
        # Core components
        self.quality_analyzer = ResponseQualityAnalyzer()
        self.ab_test_manager = ABTestManager()
        
        # Metrics storage
        self.evaluation_history = deque(maxlen=10000)  # Last 10k evaluations
        self.model_metrics = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "total_response_time": 0.0,
            "total_cost": 0.0,
            "quality_scores": deque(maxlen=1000),
            "last_reset": datetime.now()
        })
        
        # Performance tracking
        self.slo_targets = {
            "max_response_time_ms": 5000,  # 5 seconds
            "min_success_rate": 0.95,      # 95%
            "max_cost_per_request": 0.05,  # $0.05
            "min_quality_score": 0.7       # 70%
        }
        
        self.lock = threading.Lock()
        self.logger.info("Enterprise AI Evaluator initialized")
    
    async def evaluate_ai_response(self,
                                 model_name: str,
                                 response: str,
                                 expected_schema: Dict[str, str],
                                 task_type: str,
                                 response_time_ms: float,
                                 cost_usd: float,
                                 ground_truth: Optional[Dict] = None) -> EvaluationResult:
        """Comprehensive evaluation of AI response"""
        
        start_time = time.time()
        issues = []
        
        try:
            # 1. Parse response
            try:
                parsed_response = json.loads(response)
                schema_compliance = self._validate_schema(parsed_response, expected_schema)
            except json.JSONDecodeError:
                parsed_response = {"error": "invalid_json"}
                schema_compliance = False
                issues.append("Invalid JSON response")
            
            # 2. Quality analysis
            if task_type == "news_analysis":
                quality_score, quality_issues = self.quality_analyzer.analyze_news_analysis_quality(parsed_response)
            elif task_type == "sentiment_analysis":
                quality_score, quality_issues = self.quality_analyzer.analyze_sentiment_quality(parsed_response)
            else:
                quality_score, quality_issues = self.quality_analyzer.analyze_general_quality(parsed_response)
            
            issues.extend(quality_issues)
            
            # 3. Accuracy estimation
            accuracy_estimate = self._estimate_accuracy(parsed_response, ground_truth, task_type)
            
            # 4. Create evaluation result
            result = EvaluationResult(
                model_name=model_name,
                task_type=task_type,
                response_quality_score=quality_score,
                schema_compliance=schema_compliance,
                response_time_ms=response_time_ms,
                cost_usd=cost_usd,
                accuracy_estimate=accuracy_estimate,
                issues=issues
            )
            
            # 5. Record metrics
            self._record_evaluation(result)
            
            # 6. Check SLO violations
            slo_violations = self._check_slo_violations(result)
            if slo_violations:
                self.logger.warning(f"SLO violations for {model_name}: {slo_violations}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                model_name=model_name,
                task_type=task_type,
                response_quality_score=0.0,
                schema_compliance=False,
                response_time_ms=response_time_ms,
                cost_usd=cost_usd,
                accuracy_estimate=0.0,
                issues=[f"Evaluation error: {str(e)}"]
            )
    
    def _validate_schema(self, response: Dict[str, Any], expected_schema: Dict[str, str]) -> bool:
        """Validate response against expected schema"""
        try:
            for field, field_type in expected_schema.items():
                if field not in response:
                    return False
                
                value = response[field]
                if field_type == "str" and not isinstance(value, str):
                    return False
                elif field_type == "float" and not isinstance(value, (int, float)):
                    return False
                elif field_type == "list" and not isinstance(value, list):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _estimate_accuracy(self, 
                          response: Dict[str, Any], 
                          ground_truth: Optional[Dict],
                          task_type: str) -> float:
        """Estimate accuracy of response"""
        
        if ground_truth is None:
            # Use heuristic accuracy estimation
            confidence = response.get("confidence", 0.0)
            if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                return confidence
            return 0.5  # Neutral estimate
        
        # Compare with ground truth if available
        try:
            if task_type == "sentiment_analysis":
                predicted = response.get("sentiment_score", 0.0)
                actual = ground_truth.get("sentiment_score", 0.0)
                error = abs(predicted - actual)
                return max(0.0, 1.0 - error)  # Accuracy based on error
            
            elif task_type == "news_analysis":
                # Compare sentiment classification
                predicted_sentiment = response.get("sentiment", "neutral")
                actual_sentiment = ground_truth.get("sentiment", "neutral")
                return 1.0 if predicted_sentiment == actual_sentiment else 0.0
            
            return 0.5  # Default for unknown comparison
            
        except Exception:
            return 0.0
    
    def _record_evaluation(self, result: EvaluationResult):
        """Record evaluation result in metrics"""
        with self.lock:
            # Add to history
            self.evaluation_history.append(result)
            
            # Update model metrics
            metrics = self.model_metrics[result.model_name]
            metrics["total_requests"] += 1
            
            if result.response_quality_score > 0.5:  # Consider success threshold
                metrics["successful_requests"] += 1
            
            metrics["total_response_time"] += result.response_time_ms
            metrics["total_cost"] += result.cost_usd
            metrics["quality_scores"].append(result.response_quality_score)
    
    def _check_slo_violations(self, result: EvaluationResult) -> List[str]:
        """Check for SLO violations"""
        violations = []
        
        if result.response_time_ms > self.slo_targets["max_response_time_ms"]:
            violations.append(f"Response time exceeded: {result.response_time_ms}ms > {self.slo_targets['max_response_time_ms']}ms")
        
        if result.cost_usd > self.slo_targets["max_cost_per_request"]:
            violations.append(f"Cost exceeded: ${result.cost_usd} > ${self.slo_targets['max_cost_per_request']}")
        
        if result.response_quality_score < self.slo_targets["min_quality_score"]:
            violations.append(f"Quality below threshold: {result.response_quality_score} < {self.slo_targets['min_quality_score']}")
        
        return violations
    
    def get_model_performance(self, model_name: str) -> Optional[ModelPerformanceSnapshot]:
        """Get performance snapshot for specific model"""
        with self.lock:
            if model_name not in self.model_metrics:
                return None
            
            metrics = self.model_metrics[model_name]
            
            if metrics["total_requests"] == 0:
                return None
            
            avg_response_time = metrics["total_response_time"] / metrics["total_requests"]
            cost_per_request = metrics["total_cost"] / max(1, metrics["successful_requests"])
            success_rate = metrics["successful_requests"] / metrics["total_requests"]
            
            quality_scores = list(metrics["quality_scores"])
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
            
            return ModelPerformanceSnapshot(
                model_name=model_name,
                timestamp=datetime.now(),
                total_requests=metrics["total_requests"],
                successful_requests=metrics["successful_requests"],
                avg_response_time_ms=avg_response_time,
                total_cost_usd=metrics["total_cost"],
                accuracy_score=avg_quality,  # Using quality as proxy for accuracy
                schema_compliance_rate=success_rate,
                reliability_score=success_rate,
                cost_per_successful_request=cost_per_request
            )
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary"""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(self.evaluation_history),
                "model_performance": {},
                "slo_targets": self.slo_targets,
                "active_ab_tests": len(self.ab_test_manager.active_tests),
                "recent_violations": []
            }
            
            # Model performance summaries
            for model_name in self.model_metrics:
                performance = self.get_model_performance(model_name)
                if performance:
                    summary["model_performance"][model_name] = {
                        "total_requests": performance.total_requests,
                        "success_rate": performance.successful_requests / max(1, performance.total_requests),
                        "avg_response_time_ms": performance.avg_response_time_ms,
                        "avg_quality_score": performance.accuracy_score,
                        "total_cost_usd": performance.total_cost_usd,
                        "cost_per_success": performance.cost_per_successful_request
                    }
            
            # Recent SLO violations
            recent_evaluations = list(self.evaluation_history)[-100:]  # Last 100
            for eval_result in recent_evaluations:
                violations = self._check_slo_violations(eval_result)
                if violations:
                    summary["recent_violations"].append({
                        "model": eval_result.model_name,
                        "timestamp": eval_result.timestamp.isoformat(),
                        "violations": violations
                    })
            
            return summary
    
    def start_model_comparison(self, 
                             model_a: str, 
                             model_b: str,
                             test_name: Optional[str] = None) -> bool:
        """Start A/B test between models"""
        if test_name is None:
            test_name = f"{model_a}_vs_{model_b}_{int(time.time())}"
        
        return self.ab_test_manager.start_ab_test(test_name, model_a, model_b)
    
    def get_ab_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results"""
        return self.ab_test_manager.get_ab_results(test_name)


# Global singleton
_ai_evaluator_instance = None

def get_ai_evaluator() -> EnterpriseAIEvaluator:
    """Get singleton AI evaluator instance"""
    global _ai_evaluator_instance
    if _ai_evaluator_instance is None:
        _ai_evaluator_instance = EnterpriseAIEvaluator()
    return _ai_evaluator_instance


if __name__ == "__main__":
    # Basic validation
    evaluator = get_ai_evaluator()
    
    # Test evaluation
    test_response = json.dumps({
        "sentiment": "bullish",
        "impact_magnitude": 0.7,
        "confidence": 0.8,
        "reasoning": "Strong institutional adoption signals"
    })
    
    async def test_evaluation():
        result = await evaluator.evaluate_ai_response(
            model_name="gpt-4o",
            response=test_response,
            expected_schema={
                "sentiment": "str",
                "impact_magnitude": "float",
                "confidence": "float"
            },
            task_type="news_analysis",
            response_time_ms=1500.0,
            cost_usd=0.01
        )
        
        print(f"Evaluation Result: {result}")
        print(f"Summary: {evaluator.get_evaluation_summary()}")
    
    asyncio.run(test_evaluation())