"""
Walk-Forward Training System for CryptoSmartTrader
Enterprise rolling retraining with temporal validation and canary deployment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
from pathlib import Path

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        log_loss,
    )
    import xgboost as xgb
except ImportError:
    warnings.warn("ML libraries not fully available, some features may be limited")

from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetrics,
    ModelType,
    ModelStatus,
    TrainingConfig,
    create_model_registry,
)


class RetrainingTrigger(Enum):
    """Triggers for model retraining."""

    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    TIME_DECAY = "time_decay"


class ValidationMethod(Enum):
    """Validation methods for walk-forward analysis."""

    FIXED_WINDOW = "fixed_window"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    PURGED_CV = "purged_cv"


@dataclass
class RetrainingConfig:
    """Configuration for walk-forward retraining."""

    # Retraining schedule
    retrain_frequency_days: int = 7
    min_training_samples: int = 1000
    max_training_samples: int = 10000

    # Validation settings
    validation_method: ValidationMethod = ValidationMethod.ROLLING_WINDOW
    validation_window_days: int = 30
    test_window_days: int = 7

    # Performance thresholds
    min_accuracy: float = 0.6
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.15

    # Drift detection
    drift_threshold: float = 0.3
    performance_decay_threshold: float = 0.1

    # Risk management
    canary_risk_budget_pct: float = 1.0
    canary_duration_hours: int = 72

    # Feature engineering
    feature_selection: bool = True
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.001


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    model_id: str
    training_period: Tuple[datetime, datetime]
    validation_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]

    # Performance metrics
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics

    # Model info
    model_version: ModelVersion
    feature_importance: Dict[str, float]

    # Validation details
    cv_scores: List[float]
    oos_performance: Dict[str, float]

    # Risk assessment
    risk_score: float
    deployment_recommendation: str


class WalkForwardTrainer:
    """
    Enterprise walk-forward training system with temporal validation.

    Features:
    - Rolling/expanding window validation
    - Temporal data splits with purging
    - Performance-based retraining triggers
    - Drift detection and monitoring
    - Canary deployment integration
    - Feature selection and engineering
    - Risk-aware model deployment
    """

    def __init__(self, model_registry: ModelRegistry, config: RetrainingConfig = None):
        self.model_registry = model_registry
        self.config = config or RetrainingConfig()

        # Training history
        self.training_history: List[WalkForwardResult] = []
        self.performance_history: List[Dict[str, float]] = []

        # Current models
        self.production_model = None
        self.candidate_models: List[ModelVersion] = []

        self.logger = logging.getLogger(__name__)
        self.logger.info("WalkForwardTrainer initialized")

    def run_walk_forward_training(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        model_factory: Optional[Callable] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[WalkForwardResult]:
        """
        Run complete walk-forward training and validation.

        Args:
            data: Time series data with datetime index
            target_column: Name of target column
            feature_columns: List of feature columns (auto-detect if None)
            model_factory: Function to create model instances
            start_date: Start date for training (use data start if None)
            end_date: End date for training (use data end if None)

        Returns:
            List of WalkForwardResult objects
        """

        self.logger.info("Starting walk-forward training process")

        # Prepare data
        data = self._prepare_data(data, target_column, feature_columns)

        # Generate time splits
        time_splits = self._generate_time_splits(data, start_date, end_date)

        # Default model factory
        if model_factory is None:
            model_factory = self._default_model_factory

        results = []

        for i, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(
            time_splits
        ):
            self.logger.info(
                f"Processing fold {i + 1}/{len(time_splits)}: "
                f"Train: {train_start.date()} - {train_end.date()}, "
                f"Test: {test_start.date()} - {test_end.date()}"
            )

            # Extract data splits
            train_data = data.loc[train_start:train_end]
            val_data = data.loc[val_start:val_end] if val_start and val_end else None
            test_data = data.loc[test_start:test_end]

            # Skip if insufficient data
            if len(train_data) < self.config.min_training_samples:
                self.logger.warning(f"Insufficient training data: {len(train_data)} samples")
                continue

            # Train model
            result = self._train_and_validate_model(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                target_column=target_column,
                model_factory=model_factory,
                fold_id=i,
            )

            if result:
                results.append(result)
                self.training_history.append(result)

                # Evaluate for deployment
                self._evaluate_deployment_candidate(result)

        self.logger.info(f"Walk-forward training completed: {len(results)} models trained")
        return results

    def check_retraining_triggers(
        self, current_data: pd.DataFrame, current_performance: Dict[str, float]
    ) -> List[RetrainingTrigger]:
        """
        Check if retraining should be triggered based on various criteria.

        Args:
            current_data: Recent data for drift detection
            current_performance: Current model performance metrics

        Returns:
            List of triggered retraining reasons
        """

        triggers = []

        # Check scheduled retraining
        if self._is_scheduled_retrain_due():
            triggers.append(RetrainingTrigger.SCHEDULED)

        # Check performance degradation
        if self._detect_performance_degradation(current_performance):
            triggers.append(RetrainingTrigger.PERFORMANCE_DEGRADATION)

        # Check data drift
        if self._detect_data_drift(current_data):
            triggers.append(RetrainingTrigger.DATA_DRIFT)

        # Check time decay
        if self._detect_time_decay():
            triggers.append(RetrainingTrigger.TIME_DECAY)

        if triggers:
            self.logger.info(f"Retraining triggers detected: {[t.value for t in triggers]}")

        return triggers

    def retrain_model(
        self,
        new_data: pd.DataFrame,
        target_column: str,
        trigger: RetrainingTrigger,
        feature_columns: Optional[List[str]] = None,
    ) -> Optional[ModelVersion]:
        """
        Retrain model with new data.

        Args:
            new_data: New training data
            target_column: Target column name
            trigger: Reason for retraining
            feature_columns: Feature columns to use

        Returns:
            New model version if successful
        """

        self.logger.info(f"Starting model retraining: trigger={trigger.value}")

        try:
            # Prepare data
            prepared_data = self._prepare_data(new_data, target_column, feature_columns)

            # Create training split (use last 80% for training, 20% for validation)
            split_idx = int(len(prepared_data) * 0.8)
            train_data = prepared_data.iloc[:split_idx]
            val_data = prepared_data.iloc[split_idx:]

            # Train new model
            result = self._train_and_validate_model(
                train_data=train_data,
                val_data=val_data,
                test_data=val_data,  # Use validation data as test for retraining
                target_column=target_column,
                model_factory=self._default_model_factory,
                fold_id=len(self.training_history),
                retrain_trigger=trigger,
            )

            if result and self._validate_retrained_model(result):
                # Deploy to canary if quality is sufficient
                success = self.model_registry.deploy_to_production(
                    model_id=result.model_version.model_id,
                    version=result.model_version.version,
                    risk_budget_pct=self.config.canary_risk_budget_pct,
                    canary_duration_hours=self.config.canary_duration_hours,
                )

                if success:
                    self.logger.info(
                        f"Retrained model deployed to canary: {result.model_version.model_id}"
                    )
                    return result.model_version

            return None

        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            return None

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training history summary."""

        if not self.training_history:
            return {"message": "No training history available"}

        # Aggregate metrics
        all_test_accuracy = [r.test_metrics.accuracy for r in self.training_history]
        all_test_sharpe = [
            r.test_metrics.sharpe_ratio
            for r in self.training_history
            if r.test_metrics.sharpe_ratio
        ]

        # Feature importance analysis
        feature_importance_agg = {}
        for result in self.training_history:
            for feature, importance in result.feature_importance.items():
                if feature not in feature_importance_agg:
                    feature_importance_agg[feature] = []
                feature_importance_agg[feature].append(importance)

        # Average feature importance
        avg_feature_importance = {
            feature: np.mean(importances) for feature, importances in feature_importance_agg.items()
        }

        # Performance trends
        performance_trend = "stable"
        if len(all_test_accuracy) >= 3:
            recent_perf = np.mean(all_test_accuracy[-3:])
            early_perf = np.mean(all_test_accuracy[:3])

            if recent_perf > early_perf * 1.05:
                performance_trend = "improving"
            elif recent_perf < early_perf * 0.95:
                performance_trend = "declining"

        summary = {
            "total_models_trained": len(self.training_history),
            "training_period": {
                "start": min(r.training_period[0] for r in self.training_history),
                "end": max(r.training_period[1] for r in self.training_history),
            },
            "performance_summary": {
                "avg_test_accuracy": np.mean(all_test_accuracy),
                "std_test_accuracy": np.std(all_test_accuracy),
                "best_test_accuracy": max(all_test_accuracy),
                "worst_test_accuracy": min(all_test_accuracy),
                "avg_sharpe_ratio": np.mean(all_test_sharpe) if all_test_sharpe else None,
                "performance_trend": performance_trend,
            },
            "feature_analysis": {
                "top_features": sorted(
                    avg_feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "feature_stability": self._calculate_feature_stability(feature_importance_agg),
            },
            "deployment_stats": {
                "models_deployed": len(
                    [r for r in self.training_history if r.deployment_recommendation == "deploy"]
                ),
                "avg_risk_score": np.mean([r.risk_score for r in self.training_history]),
            },
            "validation_quality": {
                "avg_cv_score": np.mean(
                    [np.mean(r.cv_scores) for r in self.training_history if r.cv_scores]
                ),
                "oos_performance_correlation": self._calculate_oos_correlation(),
            },
        }

        return summary

    # Private methods

    def _prepare_data(
        self, data: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Prepare data for training with proper datetime index."""

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data = data.set_index("timestamp")
            elif "date" in data.columns:
                data = data.set_index("date")
            else:
                self.logger.warning("No datetime index found, using sequential index")

        # Select features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]

        # Validate required columns
        missing_cols = [col for col in feature_columns + [target_column] if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Clean data
        clean_data = data[feature_columns + [target_column]].copy()

        # Remove rows with missing target
        clean_data = clean_data.dropna(subset=[target_column])

        # Handle missing features (forward fill then drop)
        clean_data = clean_data.fillna(method="ffill").dropna()

        self.logger.info(
            f"Data prepared: {len(clean_data)} samples, {len(feature_columns)} features"
        )
        return clean_data

    def _generate_time_splits(
        self, data: pd.DataFrame, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> List[Tuple[datetime, datetime, datetime, datetime, datetime, datetime]]:
        """Generate time-based splits for walk-forward validation."""

        if start_date is None:
            start_date = data.index.min()
        if end_date is None:
            end_date = data.index.max()

        splits = []

        train_window = timedelta(days=90)  # 3 months training
        val_window = timedelta(days=self.config.validation_window_days)
        test_window = timedelta(days=self.config.test_window_days)
        step_size = timedelta(days=self.config.retrain_frequency_days)

        current_date = start_date + train_window

        while current_date + test_window <= end_date:
            # Training period
            train_start = current_date - train_window
            train_end = current_date

            # Validation period (optional gap to prevent look-ahead)
            val_start = train_end + timedelta(days=1)
            val_end = val_start + val_window

            # Test period
            test_start = val_end + timedelta(days=1)
            test_end = test_start + test_window

            # Ensure we have data for all periods
            if (
                train_start >= data.index.min()
                and test_end <= data.index.max()
                and len(data.loc[train_start:train_end]) >= self.config.min_training_samples
            ):
                splits.append((train_start, train_end, val_start, val_end, test_start, test_end))

            current_date += step_size

        self.logger.info(f"Generated {len(splits)} time splits for walk-forward validation")
        return splits

    def _default_model_factory(self) -> Any:
        """Default model factory for random forest."""

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    def _train_and_validate_model(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        test_data: pd.DataFrame,
        target_column: str,
        model_factory: Callable,
        fold_id: int,
        retrain_trigger: Optional[RetrainingTrigger] = None,
    ) -> Optional[WalkForwardResult]:
        """Train and validate a single model."""

        try:
            # Prepare features and target
            feature_columns = [col for col in train_data.columns if col != target_column]

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            if val_data is not None:
                X_val = val_data[feature_columns]
                y_val = val_data[target_column]
            else:
                X_val, y_val = None, None

            # Feature selection
            if self.config.feature_selection:
                selected_features = self._select_features(X_train, y_train, feature_columns)
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                if X_val is not None:
                    X_val = X_val[selected_features]
                feature_columns = selected_features

            # Train model
            model = model_factory()
            model.fit(X_train, y_train)

            # Calculate metrics
            train_metrics = self._calculate_metrics(model, X_train, y_train)
            test_metrics = self._calculate_metrics(model, X_test, y_test)

            if val_data is not None:
                val_metrics = self._calculate_metrics(model, X_val, y_val)
            else:
                val_metrics = test_metrics  # Use test as validation if no separate validation

            # Cross-validation scores
            cv_scores = self._calculate_cv_scores(model_factory, X_train, y_train)

            # Feature importance
            feature_importance = self._get_feature_importance(model, feature_columns)

            # Register model
            training_config = TrainingConfig(
                model_type=ModelType.RANDOM_FOREST,
                hyperparameters={
                    "n_estimators": getattr(model, "n_estimators", 100),
                    "max_depth": getattr(model, "max_depth", None),
                    "features": feature_columns,
                    "retrain_trigger": retrain_trigger.value if retrain_trigger else None,
                },
                training_script="walk_forward_trainer.py",
                training_environment={"framework": "sklearn"},
                random_seed=42,
                training_duration_seconds=60.0,  # Placeholder
                cross_validation_folds=5,
            )

            model_version = self.model_registry.register_model(
                model=model,
                model_type=ModelType.RANDOM_FOREST,
                dataset=train_data,
                target_column=target_column,
                metrics=test_metrics,
                training_config=training_config,
                tags=["walk_forward", f"fold_{fold_id}"],
            )

            # Risk assessment
            risk_score = self._assess_deployment_risk(train_metrics, val_metrics, test_metrics)
            deployment_rec = (
                "deploy"
                if risk_score < 0.5 and test_metrics.accuracy > self.config.min_accuracy
                else "hold"
            )

            result = WalkForwardResult(
                model_id=model_version.model_id,
                training_period=(train_data.index.min(), train_data.index.max()),
                validation_period=(val_data.index.min(), val_data.index.max())
                if val_data is not None
                else (None, None),
                test_period=(test_data.index.min(), test_data.index.max()),
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                model_version=model_version,
                feature_importance=feature_importance,
                cv_scores=cv_scores,
                oos_performance={"accuracy": test_metrics.accuracy},
                risk_score=risk_score,
                deployment_recommendation=deployment_rec,
            )

            return result

        except Exception as e:
            self.logger.error(f"Training failed for fold {fold_id}: {e}")
            return None

    def _calculate_metrics(self, model: Any, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Calculate comprehensive model metrics."""

        try:
            # Predictions
            y_pred = model.predict(X)

            # Probabilities (if available)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
                if y_proba.shape[1] == 2:  # Binary classification
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = np.max(y_proba, axis=1)
            else:
                y_proba_pos = y_pred

            # Basic metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

            # AUC (binary classification)
            try:
                if len(np.unique(y)) == 2:
                    auc_roc = roc_auc_score(y, y_proba_pos)
                else:
                    auc_roc = 0.5  # Multi-class placeholder
            except Exception:
                auc_roc = 0.5

            # Log loss
            try:
                if hasattr(model, "predict_proba"):
                    log_loss_val = log_loss(y, model.predict_proba(X))
                else:
                    log_loss_val = 0.0
            except Exception:
                log_loss_val = 0.0

            # Trading-specific metrics (simplified)
            returns = np.diff(y) if len(y) > 1 else np.array([0])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

            cumulative_returns = np.cumprod(1 + returns) - 1
            max_drawdown = np.min(cumulative_returns) if len(cumulative_returns) > 0 else 0.0

            win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.5

            return ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                log_loss=log_loss_val,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                information_ratio=sharpe_ratio * 0.8,  # Approximation
                prediction_stability=0.8,  # Placeholder
            )

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            # Return default metrics
            return ModelMetrics(
                accuracy=0.5, precision=0.5, recall=0.5, f1_score=0.5, auc_roc=0.5, log_loss=1.0
            )

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str]
    ) -> List[str]:
        """Select most important features."""

        if self.config.max_features and len(feature_columns) > self.config.max_features:
            # Train a quick model for feature selection
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)

            # Get feature importance
            importance = rf.feature_importances_

            # Select top features
            feature_importance = list(zip(feature_columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            selected = [
                f
                for f, imp in feature_importance[: self.config.max_features]
                if imp > self.config.feature_importance_threshold
            ]

            return selected if len(selected) > 0 else feature_columns[: self.config.max_features]

        return feature_columns

    def _calculate_cv_scores(
        self, model_factory: Callable, X: pd.DataFrame, y: pd.Series
    ) -> List[float]:
        """Calculate cross-validation scores."""

        try:
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                model = model_factory()
                model.fit(X_train_cv, y_train_cv)
                score = model.score(X_val_cv, y_val_cv)
                scores.append(score)

            return scores

        except Exception as e:
            self.logger.warning(f"CV calculation failed: {e}")
            return [0.5, 0.5, 0.5]

    def _get_feature_importance(self, model: Any, feature_columns: List[str]) -> Dict[str, float]:
        """Extract feature importance from model."""

        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                return dict(zip(feature_columns, importance))
            elif hasattr(model, "coef_"):
                # Linear models
                importance = np.abs(model.coef_).flatten()
                return dict(zip(feature_columns, importance))
            else:
                # Default uniform importance
                uniform_importance = 1.0 / len(feature_columns)
                return {col: uniform_importance for col in feature_columns}

        except Exception:
            uniform_importance = 1.0 / len(feature_columns)
            return {col: uniform_importance for col in feature_columns}

    def _assess_deployment_risk(
        self, train_metrics: ModelMetrics, val_metrics: ModelMetrics, test_metrics: ModelMetrics
    ) -> float:
        """Assess risk score for model deployment."""

        risk_factors = []

        # Performance consistency
        perf_consistency = abs(train_metrics.accuracy - test_metrics.accuracy)
        risk_factors.append(perf_consistency)

        # Overfitting detection
        if train_metrics.accuracy > test_metrics.accuracy + 0.1:
            risk_factors.append(0.3)  # High overfitting penalty

        # Minimum performance threshold
        if test_metrics.accuracy < self.config.min_accuracy:
            risk_factors.append(0.5)  # High risk for poor performance

        # Sharpe ratio check
        if test_metrics.sharpe_ratio and test_metrics.sharpe_ratio < self.config.min_sharpe_ratio:
            risk_factors.append(0.3)

        # Maximum drawdown check
        if (
            test_metrics.max_drawdown
            and abs(test_metrics.max_drawdown) > self.config.max_drawdown_threshold
        ):
            risk_factors.append(0.4)

        # Calculate overall risk score
        risk_score = np.mean(risk_factors) if risk_factors else 0.2

        return min(1.0, risk_score)

    def _evaluate_deployment_candidate(self, result: WalkForwardResult):
        """Evaluate if model should be deployed."""

        if result.deployment_recommendation == "deploy" and result.risk_score < 0.3:
            self.candidate_models.append(result.model_version)
            self.logger.info(f"Model added to deployment candidates: {result.model_id}")

    def _is_scheduled_retrain_due(self) -> bool:
        """Check if scheduled retraining is due."""

        if not self.training_history:
            return True

        last_training = self.training_history[-1].training_period[1]
        days_since_training = (datetime.utcnow() - last_training).days

        return days_since_training >= self.config.retrain_frequency_days

    def _detect_performance_degradation(self, current_performance: Dict[str, float]) -> bool:
        """Detect if model performance has degraded."""

        if not self.performance_history or "accuracy" not in current_performance:
            return False

        # Compare with recent performance
        recent_accuracy = [p.get("accuracy", 0) for p in self.performance_history[-5:]]
        if not recent_accuracy:
            return False

        baseline_accuracy = np.mean(recent_accuracy)
        current_accuracy = current_performance["accuracy"]

        degradation = (baseline_accuracy - current_accuracy) / baseline_accuracy

        return degradation > self.config.performance_decay_threshold

    def _detect_data_drift(self, current_data: pd.DataFrame) -> bool:
        """Detect data drift using model registry."""

        # Get current production model
        production_info = self.model_registry._get_production_model()
        if not production_info:
            return False

        try:
            # Calculate drift metrics
            drift_metrics = self.model_registry.detect_drift(
                model_id=production_info["model_id"],
                version=production_info["version"],
                current_data=current_data,
            )

            overall_drift = drift_metrics.get("overall_drift", 0.0)
            return overall_drift > self.config.drift_threshold

        except Exception as e:
            self.logger.warning(f"Drift detection failed: {e}")
            return False

    def _detect_time_decay(self) -> bool:
        """Detect if model is too old and needs refresh."""

        production_info = self.model_registry._get_production_model()
        if not production_info:
            return False

        try:
            model_id = production_info["model_id"]
            version = production_info["version"]
            model_version = self.model_registry.registry[model_id][version]

            model_age = (datetime.utcnow() - model_version.created_at).days

            # Consider model stale after 30 days
            return model_age > 30

        except Exception:
            return False

    def _validate_retrained_model(self, result: WalkForwardResult) -> bool:
        """Validate if retrained model meets quality standards."""

        # Minimum performance requirements
        min_requirements = [
            result.test_metrics.accuracy >= self.config.min_accuracy,
            result.risk_score < 0.5,
            result.test_metrics.f1_score >= 0.5,
        ]

        # Optional requirements
        if result.test_metrics.sharpe_ratio:
            min_requirements.append(
                result.test_metrics.sharpe_ratio >= self.config.min_sharpe_ratio
            )

        return all(min_requirements)

    def _calculate_feature_stability(self, feature_importance_agg: Dict[str, List[float]]) -> float:
        """Calculate stability of feature importance across models."""

        if not feature_importance_agg:
            return 0.0

        stabilities = []
        for feature, importances in feature_importance_agg.items():
            if len(importances) > 1:
                stability = 1.0 - (np.std(importances) / np.mean(importances))
                stabilities.append(max(0.0, stability))

        return np.mean(stabilities) if stabilities else 0.0

    def _calculate_oos_correlation(self) -> float:
        """Calculate correlation between validation and test performance."""

        if len(self.training_history) < 2:
            return 0.0

        val_scores = [r.validation_metrics.accuracy for r in self.training_history]
        test_scores = [r.test_metrics.accuracy for r in self.training_history]

        try:
            correlation = np.corrcoef(val_scores, test_scores)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0


def create_walk_forward_trainer(
    model_registry: ModelRegistry, config: Optional[RetrainingConfig] = None
) -> WalkForwardTrainer:
    """Create walk-forward trainer instance."""
    return WalkForwardTrainer(model_registry, config)
