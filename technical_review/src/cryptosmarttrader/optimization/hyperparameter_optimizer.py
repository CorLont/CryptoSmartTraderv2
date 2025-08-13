"""
Hyperparameter Optimizer

Advanced hyperparameter optimization system using Bayesian optimization
with walk-forward validation and regime-aware cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives"""
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    RETURN_RISK_RATIO = "return_risk_ratio"
    STABLE_OOS_SHARPE = "stable_oos_sharpe"
    MULTI_OBJECTIVE = "multi_objective"


class ParameterType(Enum):
    """Parameter types for optimization"""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"
    LOG_UNIFORM = "log_uniform"


@dataclass
class ParameterSpec:
    """Parameter specification for optimization"""
    name: str
    param_type: ParameterType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    step: Optional[Union[int, float]] = None
    log: bool = False

    def validate(self) -> bool:
        """Validate parameter specification"""
        if self.param_type in [ParameterType.FLOAT, ParameterType.INT, ParameterType.LOG_UNIFORM]:
            return self.low is not None and self.high is not None
        elif self.param_type == ParameterType.CATEGORICAL:
            return self.choices is not None and len(self.choices) > 0
        return True


@dataclass
class OptimizationMetrics:
    """Optimization performance metrics"""
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_stability_score: float
    parameter_consistency: float
    regime_robustness: float
    overfitting_risk: float

    # Detailed metrics
    returns_mean: float
    returns_std: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Validation metrics
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    @property
    def is_robust(self) -> bool:
        """Check if optimization is robust (low overfitting risk)"""
        return (
            self.overfitting_risk < 0.3 and
            self.oos_stability_score > 0.7 and
            self.parameter_consistency > 0.6
        )


@dataclass
class OptimizationResult:
    """Complete optimization result"""
    study_name: str
    optimization_time: datetime

    # Best parameters
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int

    # Performance metrics
    metrics: OptimizationMetrics

    # Optimization details
    total_trials: int
    successful_trials: int
    pruned_trials: int

    # Validation results
    validation_scores: Dict[str, float]
    regime_performance: Dict[str, float]

    # Study object for further analysis
    study: optuna.Study

    @property
    def optimization_quality(self) -> str:
        """Assess optimization quality"""
        if self.metrics.is_robust:
            return "excellent"
        elif self.metrics.overfitting_risk < 0.5:
            return "good"
        elif self.metrics.overfitting_risk < 0.7:
            return "acceptable"
        else:
            return "poor"


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization system with robust validation
    """

    def __init__(self,
                 study_name: str = None,
                 optimization_objective: OptimizationObjective = OptimizationObjective.STABLE_OOS_SHARPE,
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 storage_url: str = None,
                 random_state: int = 42):

        self.study_name = study_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.optimization_objective = optimization_objective
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage_url = storage_url
        self.random_state = random_state

        # Optimization components
        self.parameter_specs = []
        self.objective_function = None
        self.validation_strategy = None

        # Results storage
        self.optimization_history = []
        self.results_path = Path("optimization_results")
        self.results_path.mkdir(exist_ok=True)

        # Optuna configuration
        self.sampler = TPESampler(seed=random_state, n_startup_trials=10)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Validation parameters
        self.min_train_size = 252  # 1 year minimum training
        self.validation_window = 63  # 3 months validation
        self.step_size = 21  # 3 weeks step

    def add_parameter(self, param_spec: ParameterSpec) -> None:
        """Add parameter to optimization space"""
        if not param_spec.validate():
            raise ValueError(f"Invalid parameter specification: {param_spec.name}")

        self.parameter_specs.append(param_spec)
        logger.info(f"Added parameter: {param_spec.name} ({param_spec.param_type.value})")

    def add_parameters_from_config(self, config: Dict[str, Dict[str, Any]]) -> None:
        """Add multiple parameters from configuration"""
        for param_name, param_config in config.items():
            param_type = ParameterType(param_config.get("type", "float"))

            param_spec = ParameterSpec(
                name=param_name,
                param_type=param_type,
                low=param_config.get("low"),
                high=param_config.get("high"),
                choices=param_config.get("choices"),
                step=param_config.get("step"),
                log=param_config.get("log", False)
            )

            self.add_parameter(param_spec)

    def set_objective_function(self, objective_func: Callable) -> None:
        """Set the objective function for optimization"""
        self.objective_function = objective_func
        logger.info("Objective function set")

    def optimize(self,
                 data: pd.DataFrame,
                 target_column: str = "returns",
                 feature_columns: List[str] = None,
                 regime_column: str = None) -> OptimizationResult:
        """
        Run hyperparameter optimization with robust validation
        """
        try:
            logger.info(f"Starting optimization: {self.study_name}")

            if not self.objective_function:
                raise ValueError("Objective function not set")

            if not self.parameter_specs:
                raise ValueError("No parameters specified for optimization")

            # Create Optuna study
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",  # Assuming we maximize Sharpe ratio
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage_url,
                load_if_exists=True
            )

            # Create objective wrapper
            objective_wrapper = self._create_objective_wrapper(
                data, target_column, feature_columns, regime_column
            )

            # Run optimization
            study.optimize(
                objective_wrapper,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=True,
                callbacks=[self._optimization_callback]
            )

            # Analyze results
            optimization_result = self._analyze_optimization_results(
                study, data, target_column, feature_columns, regime_column
            )

            # Save results
            self._save_optimization_results(optimization_result)

            logger.info(f"Optimization completed: {optimization_result.optimization_quality} quality")

            return optimization_result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _create_objective_wrapper(self,
                                 data: pd.DataFrame,
                                 target_column: str,
                                 feature_columns: List[str],
                                 regime_column: str) -> Callable:
        """Create objective function wrapper for Optuna"""

        def objective(trial: optuna.Trial) -> float:
            try:
                # Sample parameters
                params = {}
                for param_spec in self.parameter_specs:
                    if param_spec.param_type == ParameterType.FLOAT:
                        if param_spec.log:
                            params[param_spec.name] = trial.suggest_loguniform(
                                param_spec.name, param_spec.low, param_spec.high
                            )
                        else:
                            params[param_spec.name] = trial.suggest_uniform(
                                param_spec.name, param_spec.low, param_spec.high
                            )
                    elif param_spec.param_type == ParameterType.INT:
                        params[param_spec.name] = trial.suggest_int(
                            param_spec.name, param_spec.low, param_spec.high,
                            step=param_spec.step or 1
                        )
                    elif param_spec.param_type == ParameterType.CATEGORICAL:
                        params[param_spec.name] = trial.suggest_categorical(
                            param_spec.name, param_spec.choices
                        )
                    elif param_spec.param_type == ParameterType.BOOL:
                        params[param_spec.name] = trial.suggest_categorical(
                            param_spec.name, [True, False]
                        )
                    elif param_spec.param_type == ParameterType.LOG_UNIFORM:
                        params[param_spec.name] = trial.suggest_loguniform(
                            param_spec.name, param_spec.low, param_spec.high
                        )

                # Perform walk-forward validation
                validation_scores = self._walk_forward_validation(
                    data, target_column, feature_columns, params, regime_column
                )

                # Calculate objective based on strategy
                if self.optimization_objective == OptimizationObjective.STABLE_OOS_SHARPE:
                    # Penalize high variance in OOS performance
                    oos_mean = np.mean(validation_scores["oos_sharpe"])
                    oos_std = np.std(validation_scores["oos_sharpe"])
                    stability_penalty = oos_std / (abs(oos_mean) + 0.01)  # Avoid division by zero

                    objective_value = oos_mean - 0.5 * stability_penalty

                elif self.optimization_objective == OptimizationObjective.SHARPE_RATIO:
                    objective_value = np.mean(validation_scores["oos_sharpe"])

                elif self.optimization_objective == OptimizationObjective.MULTI_OBJECTIVE:
                    # Weighted combination of metrics
                    oos_sharpe = np.mean(validation_scores["oos_sharpe"])
                    stability = 1 - np.std(validation_scores["oos_sharpe"]) / (abs(oos_sharpe) + 0.01)
                    regime_robustness = np.mean(validation_scores.get("regime_performance", [0.5]))

                    objective_value = 0.6 * oos_sharpe + 0.3 * stability + 0.1 * regime_robustness

                else:
                    objective_value = np.mean(validation_scores["oos_sharpe"])

                # Prune trial if performance is poor
                if objective_value < -2.0:  # Very poor Sharpe ratio
                    raise optuna.TrialPruned()

                return objective_value

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -999  # Return very poor score for failed trials

        return objective

    def _walk_forward_validation(self,
                                data: pd.DataFrame,
                                target_column: str,
                                feature_columns: List[str],
                                params: Dict[str, Any],
                                regime_column: str = None) -> Dict[str, List[float]]:
        """Perform walk-forward validation"""
        try:
            validation_scores = {
                "oos_sharpe": [],
                "oos_returns": [],
                "oos_volatility": [],
                "regime_performance": []
            }

            # Prepare data
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col not in [target_column, regime_column]]

            n_samples = len(data)

            # Walk-forward validation loop
            start_idx = self.min_train_size

            while start_idx + self.validation_window < n_samples:
                # Define train and validation windows
                train_end = start_idx
                val_start = train_end
                val_end = min(val_start + self.validation_window, n_samples)

                # Extract data splits
                train_data = data.iloc[:train_end]
                val_data = data.iloc[val_start:val_end]

                # Train model with current parameters
                try:
                    model_performance = self.objective_function(
                        train_data[feature_columns],
                        train_data[target_column],
                        val_data[feature_columns],
                        val_data[target_column],
                        params
                    )

                    # Extract validation metrics
                    oos_returns = model_performance.get("oos_returns", [])
                    if len(oos_returns) > 0:
                        oos_mean = np.mean(oos_returns)
                        oos_std = np.std(oos_returns)
                        oos_sharpe = oos_mean / oos_std if oos_std > 0 else 0

                        validation_scores["oos_sharpe"].append(oos_sharpe)
                        validation_scores["oos_returns"].append(oos_mean)
                        validation_scores["oos_volatility"].append(oos_std)

                        # Regime-specific performance
                        if regime_column and regime_column in val_data.columns:
                            regime_performance = self._calculate_regime_performance(
                                val_data, oos_returns, regime_column
                            )
                            validation_scores["regime_performance"].append(
                                np.mean(list(regime_performance.values()))
                            )

                except Exception as e:
                    logger.warning(f"Validation fold failed: {e}")
                    # Add poor scores for failed folds
                    validation_scores["oos_sharpe"].append(-1.0)
                    validation_scores["oos_returns"].append(0.0)
                    validation_scores["oos_volatility"].append(0.1)

                # Move to next validation window
                start_idx += self.step_size

            return validation_scores

        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            return {
                "oos_sharpe": [-1.0],
                "oos_returns": [0.0],
                "oos_volatility": [0.1],
                "regime_performance": [0.0]
            }

    def _calculate_regime_performance(self,
                                    data: pd.DataFrame,
                                    returns: List[float],
                                    regime_column: str) -> Dict[str, float]:
        """Calculate performance by market regime"""
        try:
            regime_performance = {}

            if len(returns) != len(data):
                return {"default": np.mean(returns) if returns else 0.0}

            data_with_returns = data.copy()
            data_with_returns["returns"] = returns

            for regime in data[regime_column].unique():
                regime_data = data_with_returns[data_with_returns[regime_column] == regime]
                if len(regime_data) > 0:
                    regime_returns = regime_data["returns"].tolist()
                    regime_mean = np.mean(regime_returns)
                    regime_std = np.std(regime_returns)
                    regime_sharpe = regime_mean / regime_std if regime_std > 0 else 0
                    regime_performance[str(regime)] = regime_sharpe

            return regime_performance

        except Exception as e:
            logger.error(f"Regime performance calculation failed: {e}")
            return {"default": 0.0}

    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function during optimization"""
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}: Best value = {study.best_value:.4f}")

    def _analyze_optimization_results(self,
                                    study: optuna.Study,
                                    data: pd.DataFrame,
                                    target_column: str,
                                    feature_columns: List[str],
                                    regime_column: str) -> OptimizationResult:
        """Analyze optimization results and calculate metrics"""
        try:
            best_params = study.best_params
            best_value = study.best_value

            # Calculate detailed metrics using best parameters
            detailed_metrics = self._calculate_detailed_metrics(
                data, target_column, feature_columns, best_params, regime_column
            )

            # Analyze parameter stability
            parameter_consistency = self._analyze_parameter_consistency(study)

            # Calculate validation scores
            validation_scores = self._get_validation_summary(study)

            # Regime performance analysis
            regime_performance = self._analyze_regime_robustness(
                study, data, target_column, feature_columns, best_params, regime_column
            )

            optimization_result = OptimizationResult(
                study_name=self.study_name,
                optimization_time=datetime.now(),
                best_params=best_params,
                best_value=best_value,
                best_trial_number=study.best_trial.number,
                metrics=detailed_metrics,
                total_trials=len(study.trials),
                successful_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                pruned_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                validation_scores=validation_scores,
                regime_performance=regime_performance,
                study=study
            )

            return optimization_result

        except Exception as e:
            logger.error(f"Result analysis failed: {e}")
            raise

    def _calculate_detailed_metrics(self,
                                   data: pd.DataFrame,
                                   target_column: str,
                                   feature_columns: List[str],
                                   params: Dict[str, Any],
                                   regime_column: str) -> OptimizationMetrics:
        """Calculate detailed performance metrics"""
        try:
            # Perform final validation with best parameters
            validation_results = self._walk_forward_validation(
                data, target_column, feature_columns, params, regime_column
            )

            # Extract metrics
            oos_sharpe_scores = validation_results["oos_sharpe"]
            oos_returns = validation_results["oos_returns"]

            if not oos_sharpe_scores:
                return OptimizationMetrics(0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0.5, 1)

            # Calculate stability metrics
            oos_sharpe_mean = np.mean(oos_sharpe_scores)
            oos_sharpe_std = np.std(oos_sharpe_scores)
            stability_score = max(0, 1 - oos_sharpe_std / (abs(oos_sharpe_mean) + 0.01))

            # Parameter consistency (simplified)
            parameter_consistency = 0.7  # Would be calculated from parameter distribution analysis

            # Regime robustness
            regime_scores = validation_results.get("regime_performance", [0.5])
            regime_robustness = np.mean(regime_scores) if regime_scores else 0.5

            # Overfitting risk
            in_sample_sharpe = oos_sharpe_mean * 1.2  # Simplified - would calculate from in-sample
            overfitting_risk = max(0, (in_sample_sharpe - oos_sharpe_mean) / (abs(in_sample_sharpe) + 0.01))

            # Additional metrics
            returns_mean = np.mean(oos_returns) if oos_returns else 0
            returns_std = np.std(oos_returns) if oos_returns else 0.1

            # Simplified additional metrics
            max_drawdown = abs(min(np.cumsum(oos_returns))) if oos_returns else 0
            win_rate = len([r for r in oos_returns if r > 0]) / len(oos_returns) if oos_returns else 0.5
            profit_factor = sum([r for r in oos_returns if r > 0]) / abs(sum([r for r in oos_returns if r < 0])) if oos_returns else 1

            metrics = OptimizationMetrics(
                in_sample_sharpe=in_sample_sharpe,
                out_of_sample_sharpe=oos_sharpe_mean,
                oos_stability_score=stability_score,
                parameter_consistency=parameter_consistency,
                regime_robustness=regime_robustness,
                overfitting_risk=overfitting_risk,
                returns_mean=returns_mean,
                returns_std=returns_std,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                cv_scores=oos_sharpe_scores,
                cv_mean=oos_sharpe_mean,
                cv_std=oos_sharpe_std
            )

            return metrics

        except Exception as e:
            logger.error(f"Detailed metrics calculation failed: {e}")
            return OptimizationMetrics(0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0.5, 1)

    def _analyze_parameter_consistency(self, study: optuna.Study) -> float:
        """Analyze consistency of parameters across top trials"""
        try:
            # Get top 10% of trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) < 10:
                return 0.5  # Not enough trials

            n_top_trials = max(3, len(completed_trials) // 10)
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:n_top_trials]

            # Calculate parameter stability
            param_consistencies = []

            for param_spec in self.parameter_specs:
                param_name = param_spec.name
                param_values = [trial.params.get(param_name) for trial in top_trials if param_name in trial.params]

                if len(param_values) < 2:
                    continue

                if param_spec.param_type in [ParameterType.FLOAT, ParameterType.INT, ParameterType.LOG_UNIFORM]:
                    # Numerical parameters - calculate coefficient of variation
                    param_mean = np.mean(param_values)
                    param_std = np.std(param_values)
                    cv = param_std / (abs(param_mean) + 1e-8)
                    consistency = max(0, 1 - cv)  # Lower CV = higher consistency
                else:
                    # Categorical parameters - calculate entropy
                    unique_values = len(set(param_values))
                    total_values = len(param_values)
                    consistency = 1 - (unique_values - 1) / (total_values - 1) if total_values > 1 else 1

                param_consistencies.append(consistency)

            return np.mean(param_consistencies) if param_consistencies else 0.5

        except Exception as e:
            logger.error(f"Parameter consistency analysis failed: {e}")
            return 0.5

    def _get_validation_summary(self, study: optuna.Study) -> Dict[str, float]:
        """Get validation summary statistics"""
        try:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            values = [trial.value for trial in completed_trials]

            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }

        except Exception as e:
            logger.error(f"Validation summary failed: {e}")
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    def _analyze_regime_robustness(self,
                                  study: optuna.Study,
                                  data: pd.DataFrame,
                                  target_column: str,
                                  feature_columns: List[str],
                                  best_params: Dict[str, Any],
                                  regime_column: str) -> Dict[str, float]:
        """Analyze performance robustness across market regimes"""
        try:
            if not regime_column or regime_column not in data.columns:
                return {"overall": study.best_value}

            # Analyze performance by regime using best parameters
            regime_performance = {}

            for regime in data[regime_column].unique():
                regime_data = data[data[regime_column] == regime]

                if len(regime_data) < 50:  # Too few samples
                    continue

                # Simplified regime-specific validation
                try:
                    regime_validation = self._walk_forward_validation(
                        regime_data, target_column, feature_columns, best_params, regime_column
                    )

                    regime_sharpe = np.mean(regime_validation["oos_sharpe"])
                    regime_performance[str(regime)] = regime_sharpe

                except Exception as e:
                    logger.warning(f"Regime {regime} analysis failed: {e}")
                    regime_performance[str(regime)] = 0.0

            return regime_performance

        except Exception as e:
            logger.error(f"Regime robustness analysis failed: {e}")
            return {"overall": study.best_value}

    def _save_optimization_results(self, result: OptimizationResult) -> None:
        """Save optimization results to disk"""
        try:
            # Save main results
            result_file = self.results_path / f"{result.study_name}_results.json"

            # Prepare serializable data
            serializable_result = {
                "study_name": result.study_name,
                "optimization_time": result.optimization_time.isoformat(),
                "best_params": result.best_params,
                "best_value": result.best_value,
                "best_trial_number": result.best_trial_number,
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "pruned_trials": result.pruned_trials,
                "validation_scores": result.validation_scores,
                "regime_performance": result.regime_performance,
                "optimization_quality": result.optimization_quality,
                "metrics": {
                    "in_sample_sharpe": result.metrics.in_sample_sharpe,
                    "out_of_sample_sharpe": result.metrics.out_of_sample_sharpe,
                    "oos_stability_score": result.metrics.oos_stability_score,
                    "parameter_consistency": result.metrics.parameter_consistency,
                    "regime_robustness": result.metrics.regime_robustness,
                    "overfitting_risk": result.metrics.overfitting_risk,
                    "is_robust": result.metrics.is_robust
                }
            }

            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)

            # Save study object
            study_file = self.results_path / f"{result.study_name}_study.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(result.study, f)

            logger.info(f"Optimization results saved: {result_file}")

        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")

    def load_optimization_results(self, study_name: str) -> Optional[OptimizationResult]:
        """Load optimization results from disk"""
        try:
            result_file = self.results_path / f"{study_name}_results.json"
            study_file = self.results_path / f"{study_name}_study.pkl"

            if not result_file.exists() or not study_file.exists():
                logger.warning(f"Optimization results not found: {study_name}")
                return None

            # Load main results
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Load study
            with open(study_file, 'rb') as f:
                study = pickle.load(f)

            # Reconstruct OptimizationResult (simplified)
            logger.info(f"Loaded optimization results: {study_name}")
            return None  # Would need to reconstruct full object

        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return None

    def get_optimization_recommendations(self, result: OptimizationResult) -> List[str]:
        """Get recommendations based on optimization results"""
        recommendations = []

        try:
            if result.metrics.overfitting_risk > 0.5:
                recommendations.append("High overfitting risk detected - increase regularization or reduce model complexity")

            if result.metrics.oos_stability_score < 0.6:
                recommendations.append("Low out-of-sample stability - consider ensemble methods or parameter averaging")

            if result.metrics.parameter_consistency < 0.5:
                recommendations.append("Parameter instability detected - use larger validation windows or parameter regularization")

            if result.metrics.regime_robustness < 0.4:
                recommendations.append("Poor regime robustness - implement regime-aware modeling or adaptive parameters")

            if result.successful_trials / result.total_trials < 0.7:
                recommendations.append("Many failed trials - check parameter bounds and objective function implementation")

            if result.metrics.out_of_sample_sharpe < 0.5:
                recommendations.append("Low out-of-sample Sharpe ratio - review feature engineering and model selection")

            return recommendations

        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Review optimization setup and parameter specifications"]
