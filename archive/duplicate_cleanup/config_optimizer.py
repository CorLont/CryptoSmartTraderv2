# utils/config_optimizer.py
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


class ConfigOptimizer:
    """Smart configuration optimization based on system performance and usage patterns"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.optimization_rules = self._load_optimization_rules()
        self.performance_history = []
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules for different scenarios"""
        return {
            "high_cpu_usage": {
                "condition": lambda metrics: metrics.get("cpu_percent", 0) > 80,
                "optimizations": [
                    {"key": "parallel_workers", "action": "decrease", "factor": 0.8},
                    {"key": "agents.technical.update_interval", "action": "increase", "factor": 1.5},
                    {"key": "agents.sentiment.update_interval", "action": "increase", "factor": 1.3}
                ]
            },
            "high_memory_usage": {
                "condition": lambda metrics: metrics.get("memory_percent", 0) > 85,
                "optimizations": [
                    {"key": "cache_ttl_minutes", "action": "decrease", "factor": 0.7},
                    {"key": "max_coins", "action": "decrease", "factor": 0.9},
                    {"key": "data_retention_days", "action": "decrease", "factor": 0.8}
                ]
            },
            "low_performance": {
                "condition": lambda metrics: (
                    metrics.get("cpu_percent", 0) > 70 and 
                    metrics.get("memory_percent", 0) > 70
                ),
                "optimizations": [
                    {"key": "prediction_horizons", "action": "reduce_list", "keep": 3},
                    {"key": "agents.ml_predictor.update_interval", "action": "increase", "factor": 2.0},
                    {"key": "enable_gpu", "action": "set", "value": False}
                ]
            },
            "excellent_performance": {
                "condition": lambda metrics: (
                    metrics.get("cpu_percent", 0) < 30 and 
                    metrics.get("memory_percent", 0) < 50
                ),
                "optimizations": [
                    {"key": "parallel_workers", "action": "increase", "factor": 1.2, "max": 8},
                    {"key": "agents.technical.update_interval", "action": "decrease", "factor": 0.8, "min": 30},
                    {"key": "max_coins", "action": "increase", "factor": 1.1, "max": 500}
                ]
            }
        }
    
    def analyze_and_optimize(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance and suggest/apply optimizations"""
        applied_optimizations = []
        
        for rule_name, rule in self.optimization_rules.items():
            if rule["condition"](performance_metrics):
                logger.info(f"Performance rule triggered: {rule_name}")
                
                for optimization in rule["optimizations"]:
                    result = self._apply_optimization(optimization, rule_name)
                    if result:
                        applied_optimizations.append(result)
        
        if applied_optimizations:
            self._save_optimization_history(applied_optimizations, performance_metrics)
        
        return applied_optimizations
    
    def _apply_optimization(self, optimization: Dict[str, Any], rule_name: str) -> Optional[Dict[str, Any]]:
        """Apply a specific optimization"""
        try:
            key = optimization["key"]
            action = optimization["action"]
            
            current_value = self.config_manager.get(key)
            if current_value is None:
                logger.warning(f"Config key not found: {key}")
                return None
            
            new_value = self._calculate_new_value(current_value, optimization)
            if new_value == current_value:
                return None
            
            # Apply the optimization
            self.config_manager.set(key, new_value)
            
            result = {
                "rule": rule_name,
                "key": key,
                "action": action,
                "old_value": current_value,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Applied optimization: {key} {current_value} -> {new_value}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization}: {e}")
            return None
    
    def _calculate_new_value(self, current_value: Any, optimization: Dict[str, Any]) -> Any:
        """Calculate new value based on optimization rule"""
        action = optimization["action"]
        
        if action == "increase":
            factor = optimization.get("factor", 1.2)
            new_value = current_value * factor
            if "max" in optimization:
                new_value = min(new_value, optimization["max"])
            return int(new_value) if isinstance(current_value, int) else new_value
            
        elif action == "decrease":
            factor = optimization.get("factor", 0.8)
            new_value = current_value * factor
            if "min" in optimization:
                new_value = max(new_value, optimization["min"])
            return int(new_value) if isinstance(current_value, int) else new_value
            
        elif action == "set":
            return optimization["value"]
            
        elif action == "reduce_list" and isinstance(current_value, list):
            keep_count = optimization.get("keep", len(current_value) // 2)
            return current_value[:keep_count]
            
        return current_value
    
    def _save_optimization_history(self, optimizations: List[Dict[str, Any]], 
                                 metrics: Dict[str, Any]):
        """Save optimization history for analysis"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "optimizations": optimizations
        }
        
        self.performance_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Save to file
        try:
            history_file = Path("logs/optimization_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")
    
    def get_optimization_suggestions(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization suggestions without applying them"""
        suggestions = []
        
        for rule_name, rule in self.optimization_rules.items():
            if rule["condition"](performance_metrics):
                for optimization in rule["optimizations"]:
                    key = optimization["key"]
                    current_value = self.config_manager.get(key)
                    
                    if current_value is not None:
                        new_value = self._calculate_new_value(current_value, optimization)
                        
                        if new_value != current_value:
                            suggestions.append({
                                "rule": rule_name,
                                "key": key,
                                "current_value": current_value,
                                "suggested_value": new_value,
                                "action": optimization["action"],
                                "reason": self._get_optimization_reason(rule_name, optimization)
                            })
        
        return suggestions
    
    def _get_optimization_reason(self, rule_name: str, optimization: Dict[str, Any]) -> str:
        """Get human-readable reason for optimization"""
        reasons = {
            "high_cpu_usage": {
                "decrease": "Reduce system load by decreasing this value",
                "increase": "Reduce frequency to lower CPU usage"
            },
            "high_memory_usage": {
                "decrease": "Reduce memory usage by using less of this resource",
                "reduce_list": "Limit features to reduce memory consumption"
            },
            "low_performance": {
                "increase": "Reduce processing frequency to improve performance",
                "set": "Optimize for current hardware capabilities"
            },
            "excellent_performance": {
                "increase": "Take advantage of available resources",
                "decrease": "Increase processing frequency for better results"
            }
        }
        
        return reasons.get(rule_name, {}).get(
            optimization["action"], 
            "Performance optimization"
        )
    
    def revert_last_optimizations(self) -> bool:
        """Revert the last set of optimizations"""
        if not self.performance_history:
            return False
        
        try:
            last_entry = self.performance_history[-1]
            
            for optimization in reversed(last_entry["optimizations"]):
                key = optimization["key"]
                old_value = optimization["old_value"]
                self.config_manager.set(key, old_value)
                logger.info(f"Reverted {key} to {old_value}")
            
            # Remove from history
            self.performance_history.pop()
            
            logger.info("Last optimizations reverted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revert optimizations: {e}")
            return False
    
    def get_optimization_impact(self) -> Dict[str, Any]:
        """Analyze the impact of recent optimizations"""
        if len(self.performance_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        recent_metrics = recent["metrics"]
        previous_metrics = previous["metrics"]
        
        impact = {
            "cpu_change": recent_metrics.get("cpu_percent", 0) - previous_metrics.get("cpu_percent", 0),
            "memory_change": recent_metrics.get("memory_percent", 0) - previous_metrics.get("memory_percent", 0),
            "optimizations_applied": len(recent["optimizations"]),
            "improvement_score": 0
        }
        
        # Calculate improvement score (negative is better)
        improvement_score = -(impact["cpu_change"] + impact["memory_change"])
        impact["improvement_score"] = improvement_score
        impact["status"] = "improved" if improvement_score > 0 else "degraded"
        
        return impact