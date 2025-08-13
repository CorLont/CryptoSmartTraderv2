"""
Regime-Aware Cross-Validation

Specialized cross-validation system that considers market regimes
to ensure model robustness across different market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from sklearn.model_selection import StratifiedKFold
from collections import Counter

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


class StratificationStrategy(Enum):
    """Regime stratification strategies"""
    BALANCED = "balanced"              # Equal representation per regime
    PROPORTIONAL = "proportional"      # Maintain original proportions
    MINORITY_BOOST = "minority_boost"  # Boost minority regimes
    TEMPORAL_AWARE = "temporal_aware"  # Consider time ordering


@dataclass
class RegimeSplit:
    """Regime-aware data split"""
    fold_id: int
    train_indices: List[int]
    validation_indices: List[int]

    # Regime distribution
    train_regime_counts: Dict[str, int] = field(default_factory=dict)
    val_regime_counts: Dict[str, int] = field(default_factory=dict)

    # Time boundaries
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    val_start_date: Optional[datetime] = None
    val_end_date: Optional[datetime] = None

    @property
    def regime_balance_score(self) -> float:
        """Calculate regime balance score (0-1, 1 = perfect balance)"""
        try:
            all_regimes = set(self.train_regime_counts.keys()) | set(self.val_regime_counts.keys())

            if not all_regimes:
                return 0.0

            balance_scores = []

            for regime in all_regimes:
                train_prop = self.train_regime_counts.get(regime, 0) / max(sum(self.train_regime_counts.values()), 1)
                val_prop = self.val_regime_counts.get(regime, 0) / max(sum(self.val_regime_counts.values()), 1)

                # Balance score for this regime (1 = perfect balance)
                if train_prop + val_prop > 0:
                    regime_balance = 1 - abs(train_prop - val_prop) / (train_prop + val_prop)
                else:
                    regime_balance = 1.0

                balance_scores.append(regime_balance)

            return np.mean(balance_scores)

        except Exception as e:
            logger.error(f"Regime balance score calculation failed: {e}")
            return 0.0


@dataclass
class RegimeValidationResult:
    """Regime-aware validation result"""
    fold_id: int

    # Overall performance
    overall_sharpe: float = 0.0
    overall_return: float = 0.0
    overall_volatility: float = 0.0

    # Regime-specific performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Regime robustness metrics
    regime_consistency: float = 0.0
    worst_regime_performance: float = 0.0
    best_regime_performance: float = 0.0

    # Distribution metrics
    regime_coverage: float = 0.0  # Fraction of regimes covered
    regime_balance: float = 0.0   # How balanced the regimes are

    @property
    def regime_robustness_score(self) -> float:
        """Calculate regime robustness score"""
        if not self.regime_performance:
            return 0.0

        # Weight by consistency and worst-case performance
        consistency_weight = 0.6
        worst_case_weight = 0.4

        return (consistency_weight * self.regime_consistency +
                worst_case_weight * max(0, (self.worst_regime_performance + 2) / 4))


class RegimeAwareCV:
    """
    Regime-aware cross-validation system
    """

    def __init__(self,
                 n_splits: int = 5,
                 strategy: StratificationStrategy = StratificationStrategy.BALANCED,
                 min_regime_samples: int = 10,
                 regime_column: str = "regime",
                 time_column: str = "timestamp",
                 random_state: int = 42):

        self.n_splits = n_splits
        self.strategy = strategy
        self.min_regime_samples = min_regime_samples
        self.regime_column = regime_column
        self.time_column = time_column
        self.random_state = random_state

        # Validation history
        self.validation_history = []

    def split(self, data: pd.DataFrame,
              target_column: str = None) -> Generator[RegimeSplit, None, None]:
        """Generate regime-aware splits"""

        if self.regime_column not in data.columns:
            logger.warning(f"Regime column '{self.regime_column}' not found, falling back to random splits")
            yield from self._fallback_random_splits(data)
            return

        # Analyze regime distribution
        regime_distribution = self._analyze_regime_distribution(data)

        # Filter out regimes with too few samples
        valid_regimes = {regime: count for regime, count in regime_distribution.items()
                        if count >= self.min_regime_samples}

        if len(valid_regimes) < 2:
            logger.warning("Insufficient regime diversity, falling back to random splits")
            yield from self._fallback_random_splits(data)
            return

        # Generate splits based on strategy
        if self.strategy == StratificationStrategy.TEMPORAL_AWARE:
            yield from self._temporal_aware_splits(data, valid_regimes)
        else:
            yield from self._stratified_splits(data, valid_regimes)

    def _analyze_regime_distribution(self, data: pd.DataFrame) -> Dict[str, int]:
        """Analyze regime distribution in data"""
        try:
            regime_counts = data[self.regime_column].value_counts().to_dict()

            logger.info(f"Regime distribution: {regime_counts}")

            return regime_counts

        except Exception as e:
            logger.error(f"Regime distribution analysis failed: {e}")
            return {}

    def _stratified_splits(self, data: pd.DataFrame,
                          valid_regimes: Dict[str, int]) -> Generator[RegimeSplit, None, None]:
        """Generate stratified splits maintaining regime balance"""

        try:
            # Filter data to valid regimes
            valid_data = data[data[self.regime_column].isin(valid_regimes.keys())].copy()

            # Create stratified splits
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )

            fold_id = 0

            for train_idx, val_idx in skf.split(valid_data, valid_data[self.regime_column]):
                # Map back to original indices
                train_indices = valid_data.iloc[train_idx].index.tolist()
                val_indices = valid_data.iloc[val_idx].index.tolist()

                # Calculate regime distributions
                train_regimes = valid_data.iloc[train_idx][self.regime_column].value_counts().to_dict()
                val_regimes = valid_data.iloc[val_idx][self.regime_column].value_counts().to_dict()

                # Get time boundaries if available
                train_dates = self._get_time_boundaries(valid_data.iloc[train_idx])
                val_dates = self._get_time_boundaries(valid_data.iloc[val_idx])

                split = RegimeSplit(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    validation_indices=val_indices,
                    train_regime_counts=train_regimes,
                    val_regime_counts=val_regimes,
                    train_start_date=train_dates[0],
                    train_end_date=train_dates[1],
                    val_start_date=val_dates[0],
                    val_end_date=val_dates[1]
                )

                yield split
                fold_id += 1

        except Exception as e:
            logger.error(f"Stratified splits generation failed: {e}")
            yield from self._fallback_random_splits(data)

    def _temporal_aware_splits(self, data: pd.DataFrame,
                              valid_regimes: Dict[str, int]) -> Generator[RegimeSplit, None, None]:
        """Generate temporal-aware splits preserving time order"""

        try:
            # Sort by time if time column available
            if self.time_column in data.columns:
                data_sorted = data.sort_values(self.time_column)
            else:
                data_sorted = data.copy()

            # Filter to valid regimes
            valid_data = data_sorted[data_sorted[self.regime_column].isin(valid_regimes.keys())].copy()

            n_samples = len(valid_data)
            fold_size = n_samples // self.n_splits

            for fold_id in range(self.n_splits):
                # Create temporal split
                val_start = fold_id * fold_size
                val_end = min((fold_id + 1) * fold_size, n_samples)

                # Validation set
                val_indices = valid_data.iloc[val_start:val_end].index.tolist()

                # Training set (all data except validation)
                train_indices = (valid_data.iloc[:val_start].index.tolist() +
                               valid_data.iloc[val_end:].index.tolist())

                # Calculate regime distributions
                train_data = valid_data.loc[train_indices]
                val_data = valid_data.loc[val_indices]

                train_regimes = train_data[self.regime_column].value_counts().to_dict()
                val_regimes = val_data[self.regime_column].value_counts().to_dict()

                # Get time boundaries
                train_dates = self._get_time_boundaries(train_data)
                val_dates = self._get_time_boundaries(val_data)

                split = RegimeSplit(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    validation_indices=val_indices,
                    train_regime_counts=train_regimes,
                    val_regime_counts=val_regimes,
                    train_start_date=train_dates[0],
                    train_end_date=train_dates[1],
                    val_start_date=val_dates[0],
                    val_end_date=val_dates[1]
                )

                yield split

        except Exception as e:
            logger.error(f"Temporal aware splits generation failed: {e}")
            yield from self._fallback_random_splits(data)

    def _fallback_random_splits(self, data: pd.DataFrame) -> Generator[RegimeSplit, None, None]:
        """Fallback to random splits when regime-aware splits fail"""

        try:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            fold_id = 0

            for train_idx, val_idx in kf.split(data):
                train_indices = data.iloc[train_idx].index.tolist()
                val_indices = data.iloc[val_idx].index.tolist()

                split = RegimeSplit(
                    fold_id=fold_id,
                    train_indices=train_indices,
                    validation_indices=val_indices
                )

                yield split
                fold_id += 1

        except Exception as e:
            logger.error(f"Fallback random splits failed: {e}")

    def _get_time_boundaries(self, data: pd.DataFrame) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get time boundaries for data subset"""
        try:
            if self.time_column in data.columns:
                start_date = data[self.time_column].min()
                end_date = data[self.time_column].max()

                # Convert to datetime if needed
                if not isinstance(start_date, datetime):
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)

                return start_date, end_date

            return None, None

        except Exception as e:
            logger.error(f"Time boundary calculation failed: {e}")
            return None, None

    def validate_regime_model(self,
                             model_func: callable,
                             data: pd.DataFrame,
                             target_column: str,
                             feature_columns: List[str] = None,
                             params: Dict[str, Any] = None) -> List[RegimeValidationResult]:
        """Perform regime-aware model validation"""

        try:
            logger.info("Starting regime-aware validation")

            if feature_columns is None:
                feature_columns = [col for col in data.columns
                                 if col not in [target_column, self.regime_column, self.time_column]]

            if params is None:
                params = {}

            validation_results = []

            # Generate regime-aware splits
            for split in self.split(data, target_column):
                try:
                    result = self._validate_regime_split(
                        model_func, data, target_column, feature_columns, split, params
                    )
                    validation_results.append(result)

                except Exception as e:
                    logger.warning(f"Regime fold {split.fold_id} validation failed: {e}")
                    # Add empty result for failed fold
                    validation_results.append(RegimeValidationResult(fold_id=split.fold_id))

            # Store results
            self.validation_history.extend(validation_results)

            logger.info(f"Regime validation completed: {len(validation_results)} folds")

            return validation_results

        except Exception as e:
            logger.error(f"Regime-aware validation failed: {e}")
            return []

    def _validate_regime_split(self,
                              model_func: callable,
                              data: pd.DataFrame,
                              target_column: str,
                              feature_columns: List[str],
                              split: RegimeSplit,
                              params: Dict[str, Any]) -> RegimeValidationResult:
        """Validate single regime split"""

        # Extract data
        train_data = data.loc[split.train_indices]
        val_data = data.loc[split.validation_indices]

        X_train = train_data[feature_columns]
        y_train = train_data[target_column]
        X_val = val_data[feature_columns]
        y_val = val_data[target_column]

        # Train model and get predictions
        try:
            predictions = model_func(X_train, y_train, X_val, y_val, params)

            if isinstance(predictions, dict):
                returns = predictions.get("returns", [])
            else:
                returns = predictions if isinstance(predictions, list) else predictions.tolist()

        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            returns = [0.0] * len(val_data)

        # Calculate overall performance
        overall_metrics = self._calculate_overall_metrics(returns)

        # Calculate regime-specific performance
        regime_performance = self._calculate_regime_performance(
            val_data, returns, split.val_regime_counts
        )

        # Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(regime_performance)

        return RegimeValidationResult(
            fold_id=split.fold_id,
            overall_sharpe=overall_metrics["sharpe"],
            overall_return=overall_metrics["return"],
            overall_volatility=overall_metrics["volatility"],
            regime_performance=regime_performance,
            regime_consistency=robustness_metrics["consistency"],
            worst_regime_performance=robustness_metrics["worst"],
            best_regime_performance=robustness_metrics["best"],
            regime_coverage=robustness_metrics["coverage"],
            regime_balance=split.regime_balance_score
        )

    def _calculate_overall_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate overall performance metrics"""

        if not returns or len(returns) == 0:
            return {"sharpe": 0.0, "return": 0.0, "volatility": 0.1}

        returns_array = np.array(returns)

        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0

        return {
            "sharpe": sharpe_ratio,
            "return": mean_return,
            "volatility": std_return
        }

    def _calculate_regime_performance(self,
                                    val_data: pd.DataFrame,
                                    returns: List[float],
                                    regime_counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """Calculate performance by regime"""

        regime_performance = {}

        if len(returns) != len(val_data) or self.regime_column not in val_data.columns:
            # Fallback: distribute performance equally across regimes
            overall_metrics = self._calculate_overall_metrics(returns)
            for regime in regime_counts.keys():
                regime_performance[regime] = overall_metrics
            return regime_performance

        # Create data with returns
        val_with_returns = val_data.copy()
        val_with_returns["model_returns"] = returns

        # Calculate performance by regime
        for regime in val_with_returns[self.regime_column].unique():
            regime_data = val_with_returns[val_with_returns[self.regime_column] == regime]
            regime_returns = regime_data["model_returns"].tolist()

            regime_metrics = self._calculate_overall_metrics(regime_returns)
            regime_performance[str(regime)] = regime_metrics

        return regime_performance

    def _calculate_robustness_metrics(self,
                                    regime_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate regime robustness metrics"""

        if not regime_performance:
            return {"consistency": 0.0, "worst": 0.0, "best": 0.0, "coverage": 0.0}

        # Extract Sharpe ratios
        sharpe_ratios = [perf["sharpe"] for perf in regime_performance.values()]

        if not sharpe_ratios:
            return {"consistency": 0.0, "worst": 0.0, "best": 0.0, "coverage": 0.0}

        # Consistency (lower standard deviation = higher consistency)
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        consistency = max(0, 1 - std_sharpe / (abs(mean_sharpe) + 0.01))

        # Worst and best regime performance
        worst_performance = min(sharpe_ratios)
        best_performance = max(sharpe_ratios)

        # Coverage (fraction of regimes with positive performance)
        positive_regimes = sum(1 for sharpe in sharpe_ratios if sharpe > 0)
        coverage = positive_regimes / len(sharpe_ratios) if sharpe_ratios else 0

        return {
            "consistency": consistency,
            "worst": worst_performance,
            "best": best_performance,
            "coverage": coverage
        }

    def analyze_regime_robustness(self,
                                 validation_results: List[RegimeValidationResult]) -> Dict[str, Any]:
        """Analyze overall regime robustness"""

        if not validation_results:
            return {"status": "no_data"}

        # Aggregate metrics
        robustness_scores = [result.regime_robustness_score for result in validation_results]
        consistency_scores = [result.regime_consistency for result in validation_results]
        coverage_scores = [result.regime_coverage for result in validation_results]

        # Overall regime performance analysis
        all_regime_performance = {}

        for result in validation_results:
            for regime, performance in result.regime_performance.items():
                if regime not in all_regime_performance:
                    all_regime_performance[regime] = []
                all_regime_performance[regime].append(performance["sharpe"])

        # Calculate regime stability
        regime_stability = {}
        for regime, sharpe_values in all_regime_performance.items():
            if sharpe_values:
                mean_sharpe = np.mean(sharpe_values)
                std_sharpe = np.std(sharpe_values)
                stability = max(0, 1 - std_sharpe / (abs(mean_sharpe) + 0.01))
                regime_stability[regime] = {
                    "mean_sharpe": mean_sharpe,
                    "stability": stability,
                    "n_folds": len(sharpe_values)
                }

        return {
            "overall_robustness": {
                "mean_robustness_score": np.mean(robustness_scores),
                "robustness_consistency": 1 - np.std(robustness_scores),
                "mean_regime_consistency": np.mean(consistency_scores),
                "mean_regime_coverage": np.mean(coverage_scores)
            },
            "regime_stability": regime_stability,
            "recommendations": self._get_regime_recommendations(
                all_regime_performance, np.mean(robustness_scores)
            )
        }

    def _get_regime_recommendations(self,
                                   regime_performance: Dict[str, List[float]],
                                   overall_robustness: float) -> List[str]:
        """Generate regime-based recommendations"""

        recommendations = []

        try:
            if overall_robustness < 0.4:
                recommendations.append("Low regime robustness - consider regime-specific models or features")

            # Identify problematic regimes
            problematic_regimes = []
            for regime, sharpe_values in regime_performance.items():
                if sharpe_values:
                    mean_performance = np.mean(sharpe_values)
                    if mean_performance < -0.5:
                        problematic_regimes.append(regime)

            if problematic_regimes:
                recommendations.append(f"Poor performance in {', '.join(problematic_regimes)} - implement regime-aware parameters")

            # Check regime coverage
            positive_regimes = sum(1 for sharpe_values in regime_performance.values()
                                 if sharpe_values and np.mean(sharpe_values) > 0)
            total_regimes = len(regime_performance)

            if positive_regimes / total_regimes < 0.6:
                recommendations.append("Limited positive performance across regimes - review feature engineering")

            # Stability recommendations
            unstable_regimes = []
            for regime, sharpe_values in regime_performance.items():
                if sharpe_values and len(sharpe_values) > 1:
                    stability = 1 - np.std(sharpe_values) / (abs(np.mean(sharpe_values)) + 0.01)
                    if stability < 0.5:
                        unstable_regimes.append(regime)

            if unstable_regimes:
                recommendations.append(f"Unstable performance in {', '.join(unstable_regimes)} - consider ensemble methods")

            return recommendations

        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Review regime detection and model architecture"]
