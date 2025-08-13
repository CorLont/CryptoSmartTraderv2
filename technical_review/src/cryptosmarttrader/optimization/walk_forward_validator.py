"""
Walk-Forward Validator

Time-series aware cross-validation system with walk-forward analysis
to prevent data leakage and ensure realistic out-of-sample validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationStrategy(Enum):
    """Validation strategies"""
    EXPANDING_WINDOW = "expanding_window"      # Expanding training window
    ROLLING_WINDOW = "rolling_window"          # Fixed-size rolling window
    ANCHORED_WALK_FORWARD = "anchored_wf"      # Anchored at start, expanding
    PURGED_WALK_FORWARD = "purged_wf"          # With purging period between train/test


@dataclass
class ValidationWindow:
    """Single validation window"""
    fold_id: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    purge_start: Optional[int] = None
    purge_end: Optional[int] = None

    @property
    def train_size(self) -> int:
        """Training window size"""
        return self.train_end - self.train_start

    @property
    def validation_size(self) -> int:
        """Validation window size"""
        return self.validation_end - self.validation_start

    @property
    def has_purge(self) -> bool:
        """Check if window has purge period"""
        return self.purge_start is not None and self.purge_end is not None


@dataclass
class ValidationMetrics:
    """Metrics for a single validation fold"""
    fold_id: int

    # Performance metrics
    returns: List[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Trade metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0

    # Timing metrics
    training_time_seconds: float = 0.0
    prediction_time_seconds: float = 0.0

    @property
    def annual_return(self) -> float:
        """Annualized return"""
        if not self.returns:
            return 0.0
        total_return = np.prod([1 + r for r in self.returns]) - 1
        periods = len(self.returns)
        return (1 + total_return) ** (252 / periods) - 1 if periods > 0 else 0.0

    @property
    def annual_volatility(self) -> float:
        """Annualized volatility"""
        return self.volatility * np.sqrt(252)


@dataclass
class ValidationResult:
    """Complete validation result"""
    strategy: ValidationStrategy
    total_folds: int

    # Individual fold results
    fold_metrics: List[ValidationMetrics] = field(default_factory=list)

    # Aggregate metrics
    mean_oos_sharpe: float = 0.0
    std_oos_sharpe: float = 0.0
    mean_oos_return: float = 0.0
    std_oos_return: float = 0.0

    # Stability metrics
    sharpe_stability_score: float = 0.0
    return_consistency: float = 0.0
    performance_decay: float = 0.0

    # Overfitting indicators
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    overfitting_ratio: float = 0.0

    @property
    def is_stable(self) -> bool:
        """Check if validation results are stable"""
        return (
            self.sharpe_stability_score > 0.6 and
            self.return_consistency > 0.5 and
            self.overfitting_ratio < 1.5
        )


class TimeSeriesCV:
    """
    Time series cross-validation generator
    """

    def __init__(self,
                 min_train_size: int = 252,      # 1 year minimum
                 validation_size: int = 63,       # 3 months validation
                 step_size: int = 21,             # 3 weeks step
                 purge_size: int = 0,             # No purge by default
                 max_folds: int = None):

        self.min_train_size = min_train_size
        self.validation_size = validation_size
        self.step_size = step_size
        self.purge_size = purge_size
        self.max_folds = max_folds

    def split(self, data: pd.DataFrame,
              strategy: ValidationStrategy = ValidationStrategy.EXPANDING_WINDOW) -> Generator[ValidationWindow, None, None]:
        """Generate validation windows"""

        n_samples = len(data)
        fold_id = 0

        # Starting position
        if strategy == ValidationStrategy.ANCHORED_WALK_FORWARD:
            train_start = 0
        else:
            train_start = 0

        current_train_end = self.min_train_size

        while True:
            # Check if we have enough data for validation
            validation_start = current_train_end + self.purge_size
            validation_end = validation_start + self.validation_size

            if validation_end > n_samples:
                break

            # Define training window based on strategy
            if strategy == ValidationStrategy.EXPANDING_WINDOW:
                # Expanding training window from start
                train_window_start = train_start
                train_window_end = current_train_end

            elif strategy == ValidationStrategy.ROLLING_WINDOW:
                # Rolling window of fixed size
                train_window_end = current_train_end
                train_window_start = max(0, train_window_end - self.min_train_size)

            elif strategy == ValidationStrategy.ANCHORED_WALK_FORWARD:
                # Anchored at start, expanding
                train_window_start = 0
                train_window_end = current_train_end

            elif strategy == ValidationStrategy.PURGED_WALK_FORWARD:
                # Expanding with purge period
                train_window_start = train_start
                train_window_end = current_train_end

            else:
                train_window_start = train_start
                train_window_end = current_train_end

            # Create validation window
            window = ValidationWindow(
                fold_id=fold_id,
                train_start=train_window_start,
                train_end=train_window_end,
                validation_start=validation_start,
                validation_end=validation_end,
                purge_start=current_train_end if self.purge_size > 0 else None,
                purge_end=validation_start if self.purge_size > 0 else None
            )

            yield window

            fold_id += 1
            current_train_end += self.step_size

            # Check max folds limit
            if self.max_folds and fold_id >= self.max_folds:
                break

    def get_fold_count(self, data: pd.DataFrame,
                       strategy: ValidationStrategy = ValidationStrategy.EXPANDING_WINDOW) -> int:
        """Get number of validation folds"""
        return len(list(self.split(data, strategy)))


class WalkForwardValidator:
    """
    Advanced walk-forward validation system for time series data
    """

    def __init__(self,
                 min_train_size: int = 252,
                 validation_size: int = 63,
                 step_size: int = 21,
                 purge_size: int = 0,
                 max_folds: int = None,
                 n_jobs: int = 1):

        self.cv = TimeSeriesCV(
            min_train_size=min_train_size,
            validation_size=validation_size,
            step_size=step_size,
            purge_size=purge_size,
            max_folds=max_folds
        )
        self.n_jobs = n_jobs

        # Results storage
        self.validation_history = []

    def validate(self,
                 model_func: callable,
                 data: pd.DataFrame,
                 target_column: str,
                 feature_columns: List[str] = None,
                 strategy: ValidationStrategy = ValidationStrategy.EXPANDING_WINDOW,
                 params: Dict[str, Any] = None) -> ValidationResult:
        """
        Perform walk-forward validation
        """
        try:
            logger.info(f"Starting walk-forward validation with {strategy.value}")

            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]

            if params is None:
                params = {}

            # Generate validation windows
            validation_windows = list(self.cv.split(data, strategy))
            logger.info(f"Generated {len(validation_windows)} validation windows")

            if not validation_windows:
                logger.warning("No validation windows generated")
                return self._create_empty_result(strategy)

            # Perform validation for each fold
            fold_metrics = []
            in_sample_sharpes = []

            for window in validation_windows:
                try:
                    fold_metric = self._validate_fold(
                        model_func, data, target_column, feature_columns, window, params
                    )
                    fold_metrics.append(fold_metric)

                    # Calculate in-sample performance for overfitting detection
                    in_sample_metric = self._calculate_in_sample_performance(
                        model_func, data, target_column, feature_columns, window, params
                    )
                    in_sample_sharpes.append(in_sample_metric)

                except Exception as e:
                    logger.warning(f"Fold {window.fold_id} validation failed: {e}")
                    # Add default metrics for failed fold
                    fold_metrics.append(ValidationMetrics(fold_id=window.fold_id))
                    in_sample_sharpes.append(0.0)

            # Calculate aggregate results
            validation_result = self._aggregate_results(
                strategy, fold_metrics, in_sample_sharpes
            )

            # Store results
            self.validation_history.append(validation_result)

            logger.info(f"Validation completed: OOS Sharpe = {validation_result.mean_oos_sharpe:.3f} ± {validation_result.std_oos_sharpe:.3f}")

            return validation_result

        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            return self._create_empty_result(strategy)

    def _validate_fold(self,
                       model_func: callable,
                       data: pd.DataFrame,
                       target_column: str,
                       feature_columns: List[str],
                       window: ValidationWindow,
                       params: Dict[str, Any]) -> ValidationMetrics:
        """Validate single fold"""

        start_time = datetime.now()

        # Extract data splits
        train_data = data.iloc[window.train_start:window.train_end]
        val_data = data.iloc[window.validation_start:window.validation_end]

        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_val = val_data[feature_columns]
        y_val = val_data[target_column]

        training_start = datetime.now()

        # Train and predict
        try:
            predictions = model_func(X_train, y_train, X_val, y_val, params)

            if isinstance(predictions, dict):
                returns = predictions.get("returns", [])
                predicted_returns = predictions.get("predictions", [])
            else:
                # Assume predictions are returns
                returns = predictions if isinstance(predictions, list) else predictions.tolist()
                predicted_returns = returns

        except Exception as e:
            logger.error(f"Model function failed: {e}")
            returns = [0.0] * len(val_data)
            predicted_returns = returns

        training_time = (datetime.now() - training_start).total_seconds()
        prediction_time = (datetime.now() - start_time).total_seconds() - training_time

        # Calculate metrics
        metrics = self._calculate_fold_metrics(
            window.fold_id, returns, predicted_returns, training_time, prediction_time
        )

        return metrics

    def _calculate_in_sample_performance(self,
                                       model_func: callable,
                                       data: pd.DataFrame,
                                       target_column: str,
                                       feature_columns: List[str],
                                       window: ValidationWindow,
                                       params: Dict[str, Any]) -> float:
        """Calculate in-sample performance for overfitting detection"""
        try:
            # Use training data for both training and testing (in-sample)
            train_data = data.iloc[window.train_start:window.train_end]

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            # Use last portion of training data as "validation" for in-sample test
            split_point = int(len(train_data) * 0.8)

            X_train_subset = X_train.iloc[:split_point]
            y_train_subset = y_train.iloc[:split_point]
            X_val_subset = X_train.iloc[split_point:]
            y_val_subset = y_train.iloc[split_point:]

            predictions = model_func(X_train_subset, y_train_subset, X_val_subset, y_val_subset, params)

            if isinstance(predictions, dict):
                returns = predictions.get("returns", [])
            else:
                returns = predictions if isinstance(predictions, list) else predictions.tolist()

            # Calculate Sharpe ratio
            if returns and len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe = mean_return / std_return if std_return > 0 else 0
                return sharpe

            return 0.0

        except Exception as e:
            logger.warning(f"In-sample performance calculation failed: {e}")
            return 0.0

    def _calculate_fold_metrics(self,
                               fold_id: int,
                               returns: List[float],
                               predictions: List[float],
                               training_time: float,
                               prediction_time: float) -> ValidationMetrics:
        """Calculate comprehensive metrics for a fold"""

        if not returns or len(returns) == 0:
            return ValidationMetrics(fold_id=fold_id)

        returns_array = np.array(returns)

        # Basic statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        # Performance ratios
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0

        # Downside deviation for Sortino ratio
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else std_return
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0

        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Calmar ratio
        annual_return = mean_return * 252  # Assuming daily returns
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Risk metrics
        var_95 = np.percentile(returns_array, 5)  # Value at Risk
        cvar_95 = np.mean(returns_array[returns_array <= var_95])  # Conditional VaR

        # Trade metrics
        positive_returns = returns_array[returns_array > 0]
        negative_returns = returns_array[returns_array < 0]

        win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0

        gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.001
        profit_factor = gross_profit / gross_loss

        return ValidationMetrics(
            fold_id=fold_id,
            returns=returns,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=std_return,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(returns),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=mean_return,
            training_time_seconds=training_time,
            prediction_time_seconds=prediction_time
        )

    def _aggregate_results(self,
                          strategy: ValidationStrategy,
                          fold_metrics: List[ValidationMetrics],
                          in_sample_sharpes: List[float]) -> ValidationResult:
        """Aggregate fold results into final validation result"""

        if not fold_metrics:
            return self._create_empty_result(strategy)

        # Extract key metrics
        oos_sharpes = [m.sharpe_ratio for m in fold_metrics]
        oos_returns = [m.avg_trade_return for m in fold_metrics]

        # Aggregate statistics
        mean_oos_sharpe = np.mean(oos_sharpes)
        std_oos_sharpe = np.std(oos_sharpes)
        mean_oos_return = np.mean(oos_returns)
        std_oos_return = np.std(oos_returns)

        # Stability metrics
        sharpe_stability = max(0, 1 - std_oos_sharpe / (abs(mean_oos_sharpe) + 0.01))
        return_consistency = max(0, 1 - std_oos_return / (abs(mean_oos_return) + 0.01))

        # Performance decay (check if performance deteriorates over time)
        if len(oos_sharpes) > 3:
            # Linear regression slope of performance over time
            x = np.arange(len(oos_sharpes))
            slope = np.polyfit(x, oos_sharpes, 1)[0]
            performance_decay = max(0, -slope)  # Positive decay means declining performance
        else:
            performance_decay = 0

        # Overfitting analysis
        mean_in_sample_sharpe = np.mean(in_sample_sharpes) if in_sample_sharpes else mean_oos_sharpe
        overfitting_ratio = mean_in_sample_sharpe / max(mean_oos_sharpe, 0.01)

        return ValidationResult(
            strategy=strategy,
            total_folds=len(fold_metrics),
            fold_metrics=fold_metrics,
            mean_oos_sharpe=mean_oos_sharpe,
            std_oos_sharpe=std_oos_sharpe,
            mean_oos_return=mean_oos_return,
            std_oos_return=std_oos_return,
            sharpe_stability_score=sharpe_stability,
            return_consistency=return_consistency,
            performance_decay=performance_decay,
            in_sample_sharpe=mean_in_sample_sharpe,
            out_of_sample_sharpe=mean_oos_sharpe,
            overfitting_ratio=overfitting_ratio
        )

    def _create_empty_result(self, strategy: ValidationStrategy) -> ValidationResult:
        """Create empty validation result for error cases"""
        return ValidationResult(
            strategy=strategy,
            total_folds=0,
            fold_metrics=[],
            mean_oos_sharpe=0.0,
            std_oos_sharpe=0.0,
            mean_oos_return=0.0,
            std_oos_return=0.0,
            sharpe_stability_score=0.0,
            return_consistency=0.0,
            performance_decay=0.0,
            in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0,
            overfitting_ratio=1.0
        )

    def analyze_validation_quality(self, result: ValidationResult) -> Dict[str, Any]:
        """Analyze validation quality and provide insights"""

        quality_score = 0
        issues = []
        recommendations = []

        # Check number of folds
        if result.total_folds < 5:
            issues.append("Too few validation folds")
            recommendations.append("Increase data period or reduce step size")
        else:
            quality_score += 20

        # Check stability
        if result.sharpe_stability_score > 0.7:
            quality_score += 30
        elif result.sharpe_stability_score < 0.5:
            issues.append("Low Sharpe ratio stability")
            recommendations.append("Consider parameter regularization or ensemble methods")

        # Check overfitting
        if result.overfitting_ratio < 1.3:
            quality_score += 25
        elif result.overfitting_ratio > 2.0:
            issues.append("High overfitting risk")
            recommendations.append("Increase regularization or reduce model complexity")

        # Check performance decay
        if result.performance_decay < 0.1:
            quality_score += 15
        else:
            issues.append("Performance decay detected")
            recommendations.append("Implement adaptive parameters or regime detection")

        # Check consistency
        if result.return_consistency > 0.6:
            quality_score += 10

        quality_assessment = "excellent" if quality_score >= 80 else \
                           "good" if quality_score >= 60 else \
                           "acceptable" if quality_score >= 40 else "poor"

        return {
            "quality_score": quality_score,
            "quality_assessment": quality_assessment,
            "issues": issues,
            "recommendations": recommendations,
            "is_stable": result.is_stable,
            "fold_consistency": np.std([m.sharpe_ratio for m in result.fold_metrics]) if result.fold_metrics else 0
        }

    def compare_strategies(self,
                          model_func: callable,
                          data: pd.DataFrame,
                          target_column: str,
                          feature_columns: List[str] = None,
                          params: Dict[str, Any] = None) -> Dict[ValidationStrategy, ValidationResult]:
        """Compare different validation strategies"""

        results = {}
        strategies = [
            ValidationStrategy.EXPANDING_WINDOW,
            ValidationStrategy.ROLLING_WINDOW,
            ValidationStrategy.ANCHORED_WALK_FORWARD
        ]

        for strategy in strategies:
            logger.info(f"Testing validation strategy: {strategy.value}")
            result = self.validate(
                model_func, data, target_column, feature_columns, strategy, params
            )
            results[strategy] = result

        return results

    def get_optimal_window_sizes(self,
                                model_func: callable,
                                data: pd.DataFrame,
                                target_column: str,
                                feature_columns: List[str] = None,
                                params: Dict[str, Any] = None) -> Dict[str, int]:
        """Find optimal window sizes through grid search"""

        train_sizes = [126, 252, 378, 504]  # 6M, 1Y, 1.5Y, 2Y
        val_sizes = [21, 42, 63, 84]        # 1M, 2M, 3M, 4M
        step_sizes = [7, 14, 21, 28]        # 1W, 2W, 3W, 4W

        best_score = -999
        best_params = {"train": 252, "val": 63, "step": 21}

        for train_size in train_sizes:
            for val_size in val_sizes:
                for step_size in step_sizes:
                    try:
                        # Create temporary validator
                        temp_validator = WalkForwardValidator(
                            min_train_size=train_size,
                            validation_size=val_size,
                            step_size=step_size
                        )

                        result = temp_validator.validate(
                            model_func, data, target_column, feature_columns,
                            ValidationStrategy.EXPANDING_WINDOW, params
                        )

                        # Score based on Sharpe stability
                        score = result.mean_oos_sharpe - 0.5 * result.std_oos_sharpe

                        if score > best_score:
                            best_score = score
                            best_params = {
                                "train": train_size,
                                "val": val_size,
                                "step": step_size
                            }

                        logger.info(f"Window sizes ({train_size}, {val_size}, {step_size}): Score = {score:.3f}")

                    except Exception as e:
                        logger.warning(f"Window size test failed: {e}")

        return best_params
