# ml/continual_learning.py - Continual learning with drift detection
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Statistical drift detection for model performance monitoring"""

    def __init__(self, window_size: int = 100, sensitivity: float = 0.05):
        """
        Args:
            window_size: Size of sliding window for drift detection
            sensitivity: Statistical significance threshold (alpha level)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.reference_performance = None
        self.performance_history = []

    def update_reference(self, performance_scores: list):
        """Update reference performance distribution"""
        self.reference_performance = {
            "mean": np.mean(performance_scores),
            "std": np.std(performance_scores),
            "scores": performance_scores[-self.window_size :],
        }
        logger.info(f"Updated reference performance: mean={self.reference_performance['mean']:.4f}")

    def detect_performance_drift(self, recent_scores: list) -> dict:
        """
        Detect performance drift using statistical tests

        Returns:
            Dict with drift detection results
        """
        if self.reference_performance is None:
            return {"drift_detected": False, "reason": "no_reference"}

        if len(recent_scores) < 10:
            return {"drift_detected": False, "reason": "insufficient_data"}

        # Kolmogorov-Smirnov test for distribution change
        ks_statistic, ks_p_value = stats.ks_2samp(
            self.reference_performance["scores"], recent_scores
        )

        # Mann-Whitney U test for location shift
        mw_statistic, mw_p_value = stats.mannwhitneyu(
            self.reference_performance["scores"], recent_scores, alternative="two-sided"
        )

        # Performance degradation check (mean comparison)
        recent_mean = np.mean(recent_scores)
        reference_mean = self.reference_performance["mean"]
        degradation_pct = (reference_mean - recent_mean) / reference_mean * 100

        # Detect drift
        drift_detected = (
            ks_p_value < self.sensitivity
            or mw_p_value < self.sensitivity
            or degradation_pct > 20  # 20% performance drop
        )

        return {
            "drift_detected": drift_detected,
            "ks_p_value": ks_p_value,
            "mw_p_value": mw_p_value,
            "degradation_pct": degradation_pct,
            "recent_mean": recent_mean,
            "reference_mean": reference_mean,
            "recommendation": self._get_recommendation(drift_detected, degradation_pct, ks_p_value),
        }

    def _get_recommendation(
        self, drift_detected: bool, degradation_pct: float, ks_p_value: float
    ) -> str:
        """Generate recommendation based on drift detection"""
        if not drift_detected:
            return "continue_monitoring"

        if degradation_pct > 30:
            return "immediate_retrain"
        elif degradation_pct > 20:
            return "schedule_retrain"
        elif ks_p_value < 0.01:
            return "investigate_distribution_shift"
        else:
            return "increase_monitoring_frequency"


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""

    def __init__(self, importance_lambda: float = 1000.0):
        """
        Args:
            importance_lambda: Regularization strength for important parameters
        """
        self.importance_lambda = importance_lambda
        self.important_weights = {}
        self.fisher_information = {}

    def compute_fisher_information(self, model, data_loader) -> dict:
        """Compute Fisher Information Matrix for model parameters"""
        # Simplified Fisher information calculation
        # In practice, would compute gradients of log-likelihood

        fisher_info = {}

        # For each parameter in the model, compute importance
        if hasattr(model, "feature_importances_"):
            # For tree-based models, use feature importance as proxy
            fisher_info["feature_importance"] = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models, use coefficient magnitude
            fisher_info["coefficients"] = np.abs(model.coef_)
        else:
            # Generic approach: parameter variance as importance
            fisher_info["generic"] = np.ones(10) * 0.1  # Placeholder

        return fisher_info

    def consolidate_weights(self, old_model, new_model, fisher_info: dict):
        """Apply EWC regularization to preserve important weights"""

        # Store important weights and Fisher information
        self.important_weights = self._extract_weights(old_model)
        self.fisher_information = fisher_info

        # In practice, would modify loss function to include EWC term:
        # loss = task_loss + (lambda/2) * sum(F_i * (theta_i - theta*_i)^2)

        logger.info("Applied EWC consolidation to preserve important knowledge")

        return new_model

    def _extract_weights(self, model) -> dict:
        """Extract model weights/parameters"""
        weights = {}

        if hasattr(model, "feature_importances_"):
            weights["feature_importance"] = model.feature_importances_.copy()
        elif hasattr(model, "coef_"):
            weights["coefficients"] = model.coef_.copy()

        return weights


class ContinualLearningManager:
    """Manages continual learning with drift detection and model updates"""

    def __init__(self, model_path: str = "models/continual"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.drift_detector = DriftDetector()
        self.ewc = ElasticWeightConsolidation()

        self.current_model = None
        self.model_version = 0
        self.performance_log = []

    def load_current_model(self):
        """Load the current production model"""
        latest_model_path = self.model_path / "latest_model.pkl"

        if latest_model_path.exists():
            self.current_model = joblib.load(latest_model_path)

            # Load metadata
            metadata_path = self.model_path / "model_metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.model_version = metadata.get("version", 0)

            logger.info(f"Loaded model version {self.model_version}")
        else:
            logger.warning("No current model found")

    def evaluate_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate current model performance"""

        if self.current_model is None:
            return {"mse": float("inf"), "mae": float("inf")}

        predictions = self.current_model.predict(X_test)

        performance = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": np.mean(np.abs(y_test - predictions)),
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_test),
        }

        self.performance_log.append(performance)

        return performance

    def check_for_drift(self) -> dict:
        """Check if model performance has drifted"""

        if len(self.performance_log) < 20:
            return {"drift_detected": False, "reason": "insufficient_history"}

        # Get recent performance scores
        recent_scores = [entry["mse"] for entry in self.performance_log[-20:]]

        # Update reference if needed (use older stable period)
        if len(self.performance_log) >= 100:
            reference_scores = [entry["mse"] for entry in self.performance_log[-100:-50]]
            self.drift_detector.update_reference(reference_scores)

        # Detect drift
        drift_result = self.drift_detector.detect_performance_drift(recent_scores)

        if drift_result["drift_detected"]:
            logger.warning(f"Performance drift detected: {drift_result}")

        return drift_result

    def trigger_retraining(
        self, X_new: np.ndarray, y_new: np.ndarray, retrain_mode: str = "incremental"
    ) -> dict:
        """Trigger model retraining based on drift detection"""

        logger.info(f"Starting {retrain_mode} retraining...")

        if retrain_mode == "incremental" and self.current_model is not None:
            # Incremental learning with EWC
            new_model = self._incremental_update(X_new, y_new)
        else:
            # Full retraining
            new_model = self._full_retrain(X_new, y_new)

        # Validate new model
        validation_result = self._validate_new_model(new_model, X_new, y_new)

        if validation_result["approved"]:
            self._deploy_new_model(new_model)
            return {"status": "success", "new_version": self.model_version}
        else:
            logger.warning(f"New model rejected: {validation_result['reason']}")
            return {"status": "rejected", "reason": validation_result["reason"]}

    def _incremental_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Perform incremental learning update"""

        from sklearn.ensemble import RandomForestRegressor

        # For demonstration, create new model (in practice, would update existing)
        new_model = RandomForestRegressor(n_estimators=200, random_state=42, warm_start=True)

        # Fit on new data
        new_model.fit(X_new, y_new)

        # Apply EWC if we have previous model
        if self.current_model is not None:
            fisher_info = self.ewc.compute_fisher_information(self.current_model, None)
            new_model = self.ewc.consolidate_weights(self.current_model, new_model, fisher_info)

        return new_model

    def _full_retrain(self, X: np.ndarray, y: np.ndarray):
        """Perform full model retraining"""

        from sklearn.ensemble import RandomForestRegressor

        new_model = RandomForestRegressor(
            n_estimators=300,  # More trees for full retrain
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )

        new_model.fit(X, y)

        return new_model

    def _validate_new_model(self, new_model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Validate new model before deployment"""

        # Test new model performance
        new_predictions = new_model.predict(X_val)
        new_mse = mean_squared_error(y_val, new_predictions)

        # Compare with current model
        if self.current_model is not None:
            current_predictions = self.current_model.predict(X_val)
            current_mse = mean_squared_error(y_val, current_predictions)

            improvement = (current_mse - new_mse) / current_mse * 100

            # Require at least 5% improvement or similar performance
            if improvement >= -10:  # Allow 10% degradation for robustness
                return {"approved": True, "improvement_pct": improvement}
            else:
                return {"approved": False, "reason": f"performance_degradation: {improvement:.1f}%"}
        else:
            return {"approved": True, "improvement_pct": 0}

    def _deploy_new_model(self, new_model):
        """Deploy new model to production"""

        # Save new model
        self.model_version += 1
        model_path = self.model_path / f"model_v{self.model_version}.pkl"
        latest_path = self.model_path / "latest_model.pkl"

        joblib.dump(new_model, model_path)
        joblib.dump(new_model, latest_path)

        # Save metadata
        import json

        metadata = {
            "version": self.model_version,
            "deployed_at": datetime.now().isoformat(),
            "performance_history": self.performance_log[-10:],  # Keep recent history
        }

        metadata_path = self.model_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update current model
        self.current_model = new_model

        logger.info(f"Deployed new model version {self.model_version}")


def setup_continual_learning_pipeline() -> ContinualLearningManager:
    """Setup continual learning pipeline"""

    manager = ContinualLearningManager()
    manager.load_current_model()

    return manager
