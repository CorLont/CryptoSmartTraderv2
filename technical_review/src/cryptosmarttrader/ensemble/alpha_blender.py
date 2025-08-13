"""
Alpha Blending System

Implements sophisticated blending strategies to combine multiple alpha sources
with dynamic weighting and risk-adjusted optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy.optimize import minimize  # type: ignore

from .base_models import ModelPrediction
from .meta_learner import EnsemblePrediction

logger = logging.getLogger(__name__)

class BlendingStrategy(Enum):
    """Alpha blending strategies"""
    EQUAL_WEIGHT = "equal_weight"           # Simple equal weighting
    PERFORMANCE_WEIGHT = "performance"     # Weight by historical performance
    VOLATILITY_ADJUSTED = "vol_adjusted"   # Risk-adjusted weighting
    SHARPE_OPTIMAL = "sharpe_optimal"      # Maximize Sharpe ratio
    KELLY_OPTIMAL = "kelly_optimal"        # Kelly criterion weighting
    ADAPTIVE = "adaptive"                  # Dynamic strategy switching


@dataclass
class BlendingConfig:
    """Configuration for alpha blending"""
    strategy: BlendingStrategy = BlendingStrategy.PERFORMANCE_WEIGHT
    lookback_days: int = 30
    min_observations: int = 20

    # Risk management
    max_weight: float = 0.6                # Maximum weight for single model
    min_weight: float = 0.05               # Minimum weight for included models
    risk_aversion: float = 2.0             # Risk aversion parameter

    # Rebalancing
    rebalance_frequency_hours: int = 6
    weight_change_threshold: float = 0.1   # Trigger rebalancing

    # Performance targets
    target_sharpe: float = 1.5
    target_max_drawdown: float = 0.1


@dataclass
class BlendedAlpha:
    """Result of alpha blending"""
    timestamp: datetime
    symbol: str

    # Blended prediction
    blended_probability: float
    blended_confidence: float
    blended_direction: str

    # Component analysis
    model_weights: Dict[str, float]
    component_contributions: Dict[str, float]

    # Risk metrics
    predicted_volatility: float
    predicted_sharpe: float
    max_component_weight: float

    # Blending metadata
    strategy_used: BlendingStrategy
    weight_stability: float        # How stable weights are
    diversification_ratio: float   # Effective number of models / actual models


class AlphaBlender:
    """
    Advanced alpha blending system with multiple optimization strategies
    """

    def __init__(self, config: BlendingConfig):
        self.config = config

        # Weight history for tracking
        self.weight_history = []
        self.performance_history = []

        # Current weights
        self.current_weights = {}
        self.last_rebalance_time = None

        # Model performance tracking
        self.model_performance = {}

    def blend_predictions(self,
                         ensemble_predictions: List[EnsemblePrediction],
                         symbol: str) -> BlendedAlpha:
        """
        Blend multiple ensemble predictions using optimal strategy

        Args:
            ensemble_predictions: List of ensemble predictions to blend
            symbol: Symbol being predicted

        Returns:
            Optimally blended alpha signal
        """
        try:
            if not ensemble_predictions:
                raise ValueError("No ensemble predictions to blend")

            # Check if rebalancing is needed
            if self._needs_rebalancing():
                self._rebalance_weights(ensemble_predictions)

            # Get current weights
            weights = self._get_current_weights(ensemble_predictions)

            # Perform blending
            blended_prob, contributions = self._blend_probabilities(
                ensemble_predictions, weights
            )

            # Calculate blended confidence
            blended_confidence = self._blend_confidences(
                ensemble_predictions, weights
            )

            # Determine direction
            blended_direction = 'up' if blended_prob > 0.5 else 'down'

            # Calculate risk metrics
            pred_volatility = self._predict_volatility(ensemble_predictions, weights)
            pred_sharpe = self._predict_sharpe(ensemble_predictions, weights)

            # Calculate metadata
            max_weight = max(weights.values()) if weights else 0.0
            weight_stability = self._calculate_weight_stability()
            diversification_ratio = self._calculate_diversification_ratio(weights)

            blended_alpha = BlendedAlpha(
                timestamp=datetime.now(),
                symbol=symbol,
                blended_probability=blended_prob,
                blended_confidence=blended_confidence,
                blended_direction=blended_direction,
                model_weights=weights,
                component_contributions=contributions,
                predicted_volatility=pred_volatility,
                predicted_sharpe=pred_sharpe,
                max_component_weight=max_weight,
                strategy_used=self.config.strategy,
                weight_stability=weight_stability,
                diversification_ratio=diversification_ratio
            )

            # Record for analysis
            self._record_blending_result(blended_alpha)

            return blended_alpha

        except Exception as e:
            logger.error(f"Alpha blending failed for {symbol}: {e}")
            raise

    def update_performance(self,
                          prediction_id: str,
                          actual_outcome: bool,
                          returns: float) -> None:
        """
        Update model performance based on actual outcomes

        Args:
            prediction_id: Unique identifier for prediction
            actual_outcome: Whether prediction was correct
            returns: Actual returns achieved
        """
        try:
            # Find the corresponding prediction
            for perf_record in self.performance_history:
                if perf_record.get('prediction_id') == prediction_id:
                    perf_record.update({
                        'actual_outcome': actual_outcome,
                        'actual_returns': returns,
                        'updated_timestamp': datetime.now()
                    })

                    # Update model-specific performance
                    model_weights = perf_record.get('model_weights', {})
                    for model_name, weight in model_weights.items():
                        if model_name not in self.model_performance:
                            self.model_performance[model_name] = {
                                'correct_predictions': 0,
                                'total_predictions': 0,
                                'total_returns': 0.0,
                                'returns_squared': 0.0
                            }

                        perf = self.model_performance[model_name]
                        perf['total_predictions'] += 1
                        perf['total_returns'] += returns * weight
                        perf['returns_squared'] += (returns * weight) ** 2

                        if actual_outcome:
                            perf['correct_predictions'] += 1

                    break

        except Exception as e:
            logger.error(f"Performance update failed: {e}")

    def _needs_rebalancing(self) -> bool:
        """Check if weights need rebalancing"""
        if self.last_rebalance_time is None:
            return True

        time_since_rebalance = datetime.now() - self.last_rebalance_time
        hours_since = time_since_rebalance.total_seconds() / 3600

        return hours_since >= self.config.rebalance_frequency_hours

    def _rebalance_weights(self, ensemble_predictions: List[EnsemblePrediction]) -> None:
        """Rebalance model weights based on strategy"""
        try:
            logger.info(f"Rebalancing weights using {self.config.strategy.value} strategy")

            if self.config.strategy == BlendingStrategy.EQUAL_WEIGHT:
                weights = self._equal_weight_strategy(ensemble_predictions)
            elif self.config.strategy == BlendingStrategy.PERFORMANCE_WEIGHT:
                weights = self._performance_weight_strategy(ensemble_predictions)
            elif self.config.strategy == BlendingStrategy.VOLATILITY_ADJUSTED:
                weights = self._volatility_adjusted_strategy(ensemble_predictions)
            elif self.config.strategy == BlendingStrategy.SHARPE_OPTIMAL:
                weights = self._sharpe_optimal_strategy(ensemble_predictions)
            elif self.config.strategy == BlendingStrategy.KELLY_OPTIMAL:
                weights = self._kelly_optimal_strategy(ensemble_predictions)
            elif self.config.strategy == BlendingStrategy.ADAPTIVE:
                weights = self._adaptive_strategy(ensemble_predictions)
            else:
                logger.warning(f"Unknown strategy: {self.config.strategy}, using equal weight")
                weights = self._equal_weight_strategy(ensemble_predictions)

            # Apply constraints
            weights = self._apply_weight_constraints(weights)

            # Update current weights
            self.current_weights = weights
            self.last_rebalance_time = datetime.now()

            # Record weight history
            self.weight_history.append({
                'timestamp': datetime.now(),
                'weights': weights.copy(),
                'strategy': self.config.strategy.value
            })

            # Keep only recent history
            if len(self.weight_history) > 1000:
                self.weight_history = self.weight_history[-1000:]

            logger.info(f"Weights rebalanced: {weights}")

        except Exception as e:
            logger.error(f"Weight rebalancing failed: {e}")
            # Fallback to equal weights
            self.current_weights = self._equal_weight_strategy(ensemble_predictions)

    def _get_current_weights(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Get current model weights"""
        if not self.current_weights:
            return self._equal_weight_strategy(ensemble_predictions)

        # Ensure all current models have weights
        model_names = set()
        for pred in ensemble_predictions:
            model_names.update(pred.base_predictions.keys())

        # Add missing models with minimum weight
        for model_name in model_names:
            if model_name not in self.current_weights:
                self.current_weights[model_name] = self.config.min_weight

        # Renormalize
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            self.current_weights = {
                k: v / total_weight for k, v in self.current_weights.items()
            }

        return self.current_weights

    def _equal_weight_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Simple equal weighting strategy"""
        model_names = set()
        for pred in ensemble_predictions:
            model_names.update(pred.base_predictions.keys())

        if not model_names:
            return {}

        equal_weight = 1.0 / len(model_names)
        return {name: equal_weight for name in model_names}

    def _performance_weight_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Weight by historical performance"""
        try:
            model_names = set()
            for pred in ensemble_predictions:
                model_names.update(pred.base_predictions.keys())

            weights = {}

            for model_name in model_names:
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]

                    # Calculate performance metrics
                    total_preds = perf['total_predictions']
                    if total_preds > 0:
                        accuracy = perf['correct_predictions'] / total_preds
                        avg_returns = perf['total_returns'] / total_preds

                        # Combined performance score
                        performance_score = 0.5 * accuracy + 0.5 * max(0, avg_returns)
                    else:
                        performance_score = 0.5  # Default for new models
                else:
                    performance_score = 0.5  # Default for new models

                weights[model_name] = performance_score

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                return self._equal_weight_strategy(ensemble_predictions)

            return weights

        except Exception as e:
            logger.error(f"Performance weighting failed: {e}")
            return self._equal_weight_strategy(ensemble_predictions)

    def _volatility_adjusted_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Risk-adjusted weighting (inverse volatility)"""
        try:
            model_names = set()
            for pred in ensemble_predictions:
                model_names.update(pred.base_predictions.keys())

            weights = {}

            for model_name in model_names:
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    total_preds = perf['total_predictions']

                    if total_preds > 1:
                        # Calculate volatility (standard deviation of returns)
                        avg_returns = perf['total_returns'] / total_preds
                        variance = (perf['returns_squared'] / total_preds) - (avg_returns ** 2)
                        volatility = np.sqrt(max(0, variance))

                        # Inverse volatility weight
                        if volatility > 0:
                            weights[model_name] = 1.0 / volatility
                        else:
                            weights[model_name] = 1.0  # Low volatility = high weight
                    else:
                        weights[model_name] = 1.0  # Default for new models
                else:
                    weights[model_name] = 1.0  # Default for new models

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                return self._equal_weight_strategy(ensemble_predictions)

            return weights

        except Exception as e:
            logger.error(f"Volatility adjustment failed: {e}")
            return self._equal_weight_strategy(ensemble_predictions)

    def _sharpe_optimal_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio"""
        try:
            model_names = list(set(
                name for pred in ensemble_predictions
                for name in pred.base_predictions.keys()
            ))

            if len(model_names) < 2:
                return self._equal_weight_strategy(ensemble_predictions)

            # Calculate expected returns and covariance
            returns_data = self._get_returns_matrix(model_names)

            if returns_data is None or len(returns_data) < self.config.min_observations:
                return self._performance_weight_strategy(ensemble_predictions)

            # Mean returns and covariance matrix
            mu = np.mean(returns_data, axis=0)
            cov = np.cov(returns_data.T)

            # Optimize for maximum Sharpe ratio
            n_assets = len(model_names)

            def neg_sharpe_ratio(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

                if portfolio_vol == 0:
                    return -999

                return -(portfolio_return / portfolio_vol)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]

            # Bounds (min and max weights)
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]

            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                neg_sharpe_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                optimal_weights = result.x
                weights_dict = {
                    model_names[i]: float(optimal_weights[i])
                    for i in range(len(model_names))
                }
                return weights_dict
            else:
                logger.warning("Sharpe optimization failed, using performance weighting")
                return self._performance_weight_strategy(ensemble_predictions)

        except Exception as e:
            logger.error(f"Sharpe optimization failed: {e}")
            return self._performance_weight_strategy(ensemble_predictions)

    def _kelly_optimal_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Kelly criterion optimal weighting"""
        try:
            model_names = set()
            for pred in ensemble_predictions:
                model_names.update(pred.base_predictions.keys())

            weights = {}

            for model_name in model_names:
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    total_preds = perf['total_predictions']

                    if total_preds > 0:
                        win_rate = perf['correct_predictions'] / total_preds
                        avg_returns = perf['total_returns'] / total_preds

                        # Simplified Kelly criterion
                        # f* = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-p
                        if avg_returns > 0:
                            # Estimate average win and loss
                            avg_win = avg_returns / win_rate if win_rate > 0 else 0
                            avg_loss = abs(avg_returns / (1 - win_rate)) if win_rate < 1 else 0.01

                            if avg_loss > 0:
                                b = avg_win / avg_loss
                                kelly_fraction = (b * win_rate - (1 - win_rate)) / b

                                # Apply fractional Kelly (25% of full Kelly)
                                weights[model_name] = max(0, kelly_fraction * 0.25)
                            else:
                                weights[model_name] = self.config.min_weight
                        else:
                            weights[model_name] = self.config.min_weight
                    else:
                        weights[model_name] = self.config.min_weight
                else:
                    weights[model_name] = self.config.min_weight

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                return self._equal_weight_strategy(ensemble_predictions)

            return weights

        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}")
            return self._performance_weight_strategy(ensemble_predictions)

    def _adaptive_strategy(self, ensemble_predictions: List[EnsemblePrediction]) -> Dict[str, float]:
        """Adaptive strategy that switches based on market conditions"""
        try:
            # Choose strategy based on recent performance and market conditions

            # If we have sufficient history, evaluate which strategy performed best recently
            if len(self.weight_history) > 10:
                # Analyze recent strategy performance
                recent_history = self.weight_history[-10:]
                strategy_performance = {}

                for record in recent_history:
                    strategy = record.get('strategy', 'equal_weight')
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []

                    # Would need actual performance data here
                    # For now, use a heuristic

                # For simplicity, choose based on market volatility
                # High volatility = use volatility adjusted
                # Low volatility = use Sharpe optimal
                recent_preds = ensemble_predictions[-1] if ensemble_predictions else None

                if recent_preds and hasattr(recent_preds, 'consensus_score'):
                    if recent_preds.consensus_score < 0.7:  # Low consensus = high uncertainty
                        return self._volatility_adjusted_strategy(ensemble_predictions)
                    else:
                        return self._sharpe_optimal_strategy(ensemble_predictions)

            # Default to performance weighting
            return self._performance_weight_strategy(ensemble_predictions)

        except Exception as e:
            logger.error(f"Adaptive strategy failed: {e}")
            return self._performance_weight_strategy(ensemble_predictions)

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints (min/max limits)"""
        try:
            constrained_weights = {}

            for model_name, weight in weights.items():
                # Apply min/max constraints
                constrained_weight = max(self.config.min_weight,
                                       min(self.config.max_weight, weight))
                constrained_weights[model_name] = constrained_weight

            # Renormalize to sum to 1
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {
                    k: v / total_weight for k, v in constrained_weights.items()
                }

            return constrained_weights

        except Exception as e:
            logger.error(f"Weight constraint application failed: {e}")
            return weights

    def _blend_probabilities(self,
                           ensemble_predictions: List[EnsemblePrediction],
                           weights: Dict[str, float]) -> tuple:
        """Blend probabilities using weights"""
        try:
            blended_prob = 0.0
            contributions = {}
            total_weight = 0.0

            for pred in ensemble_predictions:
                for model_name, base_pred in pred.base_predictions.items():
                    weight = weights.get(model_name, 0.0)
                    contribution = weight * base_pred.probability

                    blended_prob += contribution
                    contributions[model_name] = contribution
                    total_weight += weight

            # Normalize if total weight != 1
            if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
                blended_prob /= total_weight
                contributions = {k: v / total_weight for k, v in contributions.items()}

            # Ensure probability is in valid range
            blended_prob = max(0.0, min(1.0, blended_prob))

            return blended_prob, contributions

        except Exception as e:
            logger.error(f"Probability blending failed: {e}")
            return 0.5, {}

    def _blend_confidences(self,
                          ensemble_predictions: List[EnsemblePrediction],
                          weights: Dict[str, float]) -> float:
        """Blend confidences using weights"""
        try:
            blended_confidence = 0.0
            total_weight = 0.0

            for pred in ensemble_predictions:
                for model_name, base_pred in pred.base_predictions.items():
                    weight = weights.get(model_name, 0.0)
                    blended_confidence += weight * base_pred.confidence
                    total_weight += weight

            # Normalize
            if total_weight > 0:
                blended_confidence /= total_weight

            return max(0.0, min(1.0, blended_confidence))

        except Exception as e:
            logger.error(f"Confidence blending failed: {e}")
            return 0.5

    def _get_returns_matrix(self, model_names: List[str]) -> Optional[np.ndarray]:
        """Get historical returns matrix for optimization"""
        try:
            # This would extract historical returns for each model
            # For now, return None to indicate insufficient data
            return None

        except Exception as e:
            logger.error(f"Returns matrix extraction failed: {e}")
            return None

    def _predict_volatility(self,
                          ensemble_predictions: List[EnsemblePrediction],
                          weights: Dict[str, float]) -> float:
        """Predict portfolio volatility"""
        try:
            # Simplified volatility prediction
            # In practice, would use historical covariance matrix

            # Use consensus as proxy for volatility (low consensus = high vol)
            if ensemble_predictions:
                avg_consensus = np.mean([pred.consensus_score for pred in ensemble_predictions])
                predicted_vol = 0.1 + (1 - avg_consensus) * 0.2  # 10-30% volatility range
            else:
                predicted_vol = 0.15  # Default

            return predicted_vol

        except Exception as e:
            logger.error(f"Volatility prediction failed: {e}")
            return 0.15

    def _predict_sharpe(self,
                       ensemble_predictions: List[EnsemblePrediction],
                       weights: Dict[str, float]) -> float:
        """Predict portfolio Sharpe ratio"""
        try:
            # Simplified Sharpe prediction
            # Use average expected performance

            if ensemble_predictions:
                avg_expected_auc = np.mean([pred.expected_auc for pred in ensemble_predictions])
                # Convert AUC to approximate Sharpe ratio
                predicted_sharpe = (avg_expected_auc - 0.5) * 4  # Rough conversion
            else:
                predicted_sharpe = 1.0  # Default

            return max(0.0, predicted_sharpe)

        except Exception as e:
            logger.error(f"Sharpe prediction failed: {e}")
            return 1.0

    def _calculate_weight_stability(self) -> float:
        """Calculate stability of recent weights"""
        try:
            if len(self.weight_history) < 2:
                return 1.0

            # Compare recent weight changes
            recent_weights = self.weight_history[-5:]  # Last 5 rebalances

            weight_changes = []
            for i in range(1, len(recent_weights)):
                prev_weights = recent_weights[i-1]['weights']
                curr_weights = recent_weights[i]['weights']

                # Calculate weight change magnitude
                total_change = 0.0
                for model_name in set(prev_weights.keys()) | set(curr_weights.keys()):
                    prev_w = prev_weights.get(model_name, 0.0)
                    curr_w = curr_weights.get(model_name, 0.0)
                    total_change += abs(curr_w - prev_w)

                weight_changes.append(total_change)

            if weight_changes:
                avg_change = np.mean(weight_changes)
                stability = max(0.0, 1.0 - avg_change * 2)  # Scale factor
            else:
                stability = 1.0

            return stability

        except Exception as e:
            logger.error(f"Weight stability calculation failed: {e}")
            return 0.5

    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate effective diversification ratio"""
        try:
            if not weights:
                return 0.0

            # Effective number of models (Herfindahl index)
            sum_squared_weights = sum(w ** 2 for w in weights.values())
            effective_models = 1.0 / sum_squared_weights if sum_squared_weights > 0 else 0.0
            actual_models = len(weights)

            diversification_ratio = effective_models / actual_models if actual_models > 0 else 0.0

            return min(1.0, diversification_ratio)

        except Exception as e:
            logger.error(f"Diversification ratio calculation failed: {e}")
            return 0.0

    def _record_blending_result(self, blended_alpha: BlendedAlpha) -> None:
        """Record blending result for analysis"""
        try:
            result_record = {
                'timestamp': blended_alpha.timestamp,
                'symbol': blended_alpha.symbol,
                'blended_probability': blended_alpha.blended_probability,
                'blended_confidence': blended_alpha.blended_confidence,
                'model_weights': blended_alpha.model_weights,
                'strategy_used': blended_alpha.strategy_used.value,
                'predicted_sharpe': blended_alpha.predicted_sharpe,
                'diversification_ratio': blended_alpha.diversification_ratio,
                'prediction_id': f"{blended_alpha.symbol}_{blended_alpha.timestamp.timestamp()}"
            }

            self.performance_history.append(result_record)

            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        except Exception as e:
            logger.error(f"Failed to record blending result: {e}")

    def get_blending_analytics(self) -> Dict[str, Any]:
        """Get comprehensive blending analytics"""
        try:
            analytics = {
                "current_strategy": self.config.strategy.value,
                "last_rebalance": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
                "current_weights": self.current_weights,
                "total_predictions": len(self.performance_history),
                "rebalance_count": len(self.weight_history)
            }

            # Recent performance
            if self.performance_history:
                recent_preds = self.performance_history[-100:]  # Last 100 predictions
                analytics["recent_performance"] = {
                    "avg_confidence": np.mean([p['blended_confidence'] for p in recent_preds]),
                    "avg_predicted_sharpe": np.mean([p.get('predicted_sharpe', 1.0) for p in recent_preds]),
                    "avg_diversification": np.mean([p.get('diversification_ratio', 0.5) for p in recent_preds])
                }

            # Model performance summary
            if self.model_performance:
                analytics["model_performance"] = {}
                for model_name, perf in self.model_performance.items():
                    if perf['total_predictions'] > 0:
                        analytics["model_performance"][model_name] = {
                            "accuracy": perf['correct_predictions'] / perf['total_predictions'],
                            "avg_returns": perf['total_returns'] / perf['total_predictions'],
                            "total_predictions": perf['total_predictions']
                        }

            return analytics

        except Exception as e:
            logger.error(f"Blending analytics generation failed: {e}")
            return {"status": "Error", "error": str(e)}
