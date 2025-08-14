#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Continual Learning Engine
Meta-learning, online learning, and automated retraining with drift detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from typing import Any
import hashlib
import hmac
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from copy import deepcopy

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from scipy import stats

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class LearningMode(Enum):
    BATCH_LEARNING = "batch_learning"
    ONLINE_LEARNING = "online_learning"
    META_LEARNING = "meta_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CONTINUAL_LEARNING = "continual_learning"


class DriftType(Enum):
    CONCEPT_DRIFT = "concept_drift"  # Relationship between features and target changes
    COVARIATE_SHIFT = "covariate_shift"  # Input distribution changes
    PRIOR_SHIFT = "prior_shift"  # Target distribution changes
    VIRTUAL_DRIFT = "virtual_drift"  # Performance degradation without drift


class RetrainingTrigger(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIFT_DETECTED = "drift_detected"
    SCHEDULED = "scheduled"
    NEW_COIN_ADDED = "new_coin_added"
    MANUAL = "manual"


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning system"""

    # Drift detection
    drift_detection_window: int = 100
    drift_threshold: float = 0.05
    performance_degradation_threshold: float = 0.15
    min_samples_for_drift: int = 50

    # Online learning
    online_learning_rate: float = 1e-4
    online_batch_size: int = 8
    forgetting_factor: float = 0.99

    # Meta-learning
    meta_learning_rate: float = 1e-3
    adaptation_steps: int = 5
    meta_batch_size: int = 16
    few_shot_samples: int = 10

    # Continual learning
    rehearsal_buffer_size: int = 1000
    elastic_weight_consolidation: bool = True
    ewc_lambda: float = 1000.0
    progressive_networks: bool = False

    # Automated retraining
    retraining_schedule_hours: int = 24
    min_performance_samples: int = 20
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Model ensemble for catastrophic forgetting prevention
    ensemble_size: int = 5
    knowledge_distillation: bool = True
    distillation_temperature: float = 3.0


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""

    drift_detected: bool
    drift_type: DriftType
    confidence: float
    p_value: float
    drift_score: float
    affected_features: List[str]
    recommendation: str
    timestamp: datetime


@dataclass
class RetrainingTask:
    """Automated retraining task"""

    task_id: str
    trigger: RetrainingTrigger
    model_id: str
    coin_symbols: List[str]
    priority: int
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Dict] = None


class MAMLLearner(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation"""

    def __init__(self, base_model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.base_model = base_model
        self.meta_lr = learning_rate

    def forward(self, x):
        return self.base_model(x)

    def adapt(
        self, support_x: torch.Tensor, support_y: torch.Tensor, adaptation_steps: int = 5
    ) -> nn.Module:
        """Adapt model to new task using support set"""
        adapted_model = deepcopy(self.base_model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.meta_lr)

        for step in range(adaptation_steps):
            optimizer.zero_grad()
            pred = adapted_model(support_x)
            loss = F.mse_loss(pred, support_y)
            loss.backward()
            optimizer.step()

        return adapted_model

    def meta_update(self, tasks: List[Tuple], meta_optimizer: optim.Optimizer):
        """Meta-update using multiple tasks"""
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to support set
            adapted_model = self.adapt(support_x, support_y)

            # Evaluate on query set
            query_pred = adapted_model(query_x)
            task_loss = F.mse_loss(query_pred, query_y)
            meta_loss += task_loss

        # Meta-gradient update
        meta_loss /= len(tasks)
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        return meta_loss.item()


class DriftDetector:
    """Statistical drift detection system"""

    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DriftDetector")

        # Historical data for comparison
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_performance: Dict[str, List[float]] = {}

        # Drift detection models
        self.drift_detectors: Dict[str, Any] = {}

    def update_reference_data(
        self, model_id: str, features: np.ndarray, predictions: np.ndarray, targets: np.ndarray
    ):
        """Update reference data for drift detection"""
        self.reference_data[model_id] = {
            "features": features[-self.config.drift_detection_window :],
            "predictions": predictions[-self.config.drift_detection_window :],
            "targets": targets[-self.config.drift_detection_window :],
        }

        # Calculate performance metrics
        mse = mean_squared_error(targets, predictions)

        if model_id not in self.reference_performance:
            self.reference_performance[model_id] = []

        self.reference_performance[model_id].append(mse)

        # Keep only recent performance
        self.reference_performance[model_id] = self.reference_performance[model_id][
            -self.config.drift_detection_window :
        ]

    def detect_drift(
        self,
        model_id: str,
        new_features: np.ndarray,
        new_predictions: np.ndarray,
        new_targets: np.ndarray,
    ) -> DriftDetectionResult:
        """Detect various types of drift"""
        try:
            if model_id not in self.reference_data:
                return DriftDetectionResult(
                    drift_detected=False,
                    drift_type=DriftType.VIRTUAL_DRIFT,
                    confidence=0.0,
                    p_value=1.0,
                    drift_score=0.0,
                    affected_features=[],
                    recommendation="Insufficient reference data",
                    timestamp=datetime.now(),
                )

            reference = self.reference_data[model_id]

            # 1. Covariate shift detection (input distribution change)
            covariate_result = self._detect_covariate_shift(reference["features"], new_features)

            # 2. Concept drift detection (input-output relationship change)
            concept_result = self._detect_concept_drift(
                reference["features"], reference["targets"], new_features, new_targets
            )

            # 3. Prior shift detection (target distribution change)
            prior_result = self._detect_prior_shift(reference["targets"], new_targets)

            # 4. Performance degradation detection
            performance_result = self._detect_performance_degradation(
                model_id, new_predictions, new_targets
            )

            # Combine results
            max_confidence = max(
                covariate_result["confidence"],
                concept_result["confidence"],
                prior_result["confidence"],
                performance_result["confidence"],
            )

            # Determine dominant drift type
            if performance_result["confidence"] > 0.7:
                drift_type = DriftType.VIRTUAL_DRIFT
                main_result = performance_result
            elif concept_result["confidence"] > 0.6:
                drift_type = DriftType.CONCEPT_DRIFT
                main_result = concept_result
            elif covariate_result["confidence"] > 0.6:
                drift_type = DriftType.COVARIATE_SHIFT
                main_result = covariate_result
            elif prior_result["confidence"] > 0.6:
                drift_type = DriftType.PRIOR_SHIFT
                main_result = prior_result
            else:
                drift_type = DriftType.VIRTUAL_DRIFT
                main_result = {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

            drift_detected = max_confidence > self.config.drift_threshold

            # Generate recommendation
            recommendation = self._generate_drift_recommendation(drift_type, max_confidence)

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=drift_type,
                confidence=max_confidence,
                p_value=main_result.get("p_value", 1.0),
                drift_score=max_confidence,
                affected_features=main_result.get("affected_features", []),
                recommendation=recommendation,
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Drift detection failed for {model_id}: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.VIRTUAL_DRIFT,
                confidence=0.0,
                p_value=1.0,
                drift_score=0.0,
                affected_features=[],
                recommendation="Drift detection failed",
                timestamp=datetime.now(),
            )

    def _detect_covariate_shift(
        self, ref_features: np.ndarray, new_features: np.ndarray
    ) -> Dict[str, Any]:
        """Detect covariate shift using statistical tests"""
        try:
            if not HAS_SKLEARN:
                return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

            # Use Kolmogorov-Smirnov test for each feature
            affected_features = []
            p_values = []

            min_features = min(ref_features.shape[1], new_features.shape[1])

            for i in range(min_features):
                if len(ref_features) > 0 and len(new_features) > 0:
                    ks_stat, p_val = stats.ks_2samp(ref_features[:, i], new_features[:, i])
                    p_values.append(p_val)

                    if p_val < 0.05:  # Significant difference
                        affected_features.append(f"feature_{i}")

            # Bonferroni correction for multiple testing
            if p_values:
                corrected_alpha = 0.05 / len(p_values)
                significant_shifts = sum(1 for p in p_values if p < corrected_alpha)
                confidence = significant_shifts / len(p_values)
                min_p_value = min(p_values)
            else:
                confidence = 0.0
                min_p_value = 1.0

            return {
                "confidence": confidence,
                "p_value": min_p_value,
                "affected_features": affected_features,
            }

        except Exception:
            return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

    def _detect_concept_drift(
        self,
        ref_features: np.ndarray,
        ref_targets: np.ndarray,
        new_features: np.ndarray,
        new_targets: np.ndarray,
    ) -> Dict[str, Any]:
        """Detect concept drift using model performance comparison"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error

            # Train model on reference data
            ref_model = LinearRegression()
            ref_model.fit(ref_features, ref_targets)

            # Test on both reference and new data
            ref_pred = ref_model.predict(ref_features)
            new_pred = ref_model.predict(new_features)

            ref_mse = mean_squared_error(ref_targets, ref_pred)
            new_mse = mean_squared_error(new_targets, new_pred)

            # Performance degradation indicates concept drift
            performance_change = (new_mse - ref_mse) / max(ref_mse, 1e-8)
            confidence = min(1.0, max(0.0, performance_change))

            # Statistical test for coefficient changes
            p_value = 0.5  # Simplified - would use proper statistical test

            return {
                "confidence": confidence,
                "p_value": p_value,
                "affected_features": [],  # Would identify which features changed
            }

        except Exception:
            return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

    def _detect_prior_shift(
        self, ref_targets: np.ndarray, new_targets: np.ndarray
    ) -> Dict[str, Any]:
        """Detect prior shift (target distribution change)"""
        try:
            # Kolmogorov-Smirnov test for target distribution
            ks_stat, p_val = stats.ks_2samp(ref_targets, new_targets)

            # Also check mean and variance changes
            ref_mean, ref_std = np.mean(ref_targets), np.std(ref_targets)
            new_mean, new_std = np.mean(new_targets), np.std(new_targets)

            mean_change = abs(new_mean - ref_mean) / max(abs(ref_mean), 1e-8)
            std_change = abs(new_std - ref_std) / max(ref_std, 1e-8)

            # Combine KS test with distribution moment changes
            confidence = min(1.0, (1 - p_val) + 0.3 * (mean_change + std_change))

            return {
                "confidence": confidence,
                "p_value": p_val,
                "affected_features": ["target_distribution"],
            }

        except Exception:
            return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

    def _detect_performance_degradation(
        self, model_id: str, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, Any]:
        """Detect performance degradation"""
        try:
            if model_id not in self.reference_performance:
                return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

            current_mse = mean_squared_error(targets, predictions)
            ref_performance = self.reference_performance[model_id]

            if len(ref_performance) < self.config.min_performance_samples:
                return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

            ref_mean_mse = np.mean(ref_performance)
            performance_degradation = (current_mse - ref_mean_mse) / max(ref_mean_mse, 1e-8)

            # Statistical test for performance change
            if len(ref_performance) > 1:
                t_stat, p_val = stats.ttest_1samp(ref_performance, current_mse)
                p_val = p_val / 2  # One-sided test
            else:
                p_val = 0.5

            confidence = min(1.0, max(0.0, performance_degradation))

            return {
                "confidence": confidence,
                "p_value": p_val,
                "affected_features": ["model_performance"],
            }

        except Exception:
            return {"confidence": 0.0, "p_value": 1.0, "affected_features": []}

    def _generate_drift_recommendation(self, drift_type: DriftType, confidence: float) -> str:
        """Generate recommendation based on drift type and confidence"""
        if confidence < 0.3:
            return "No action needed - drift confidence too low"
        elif confidence < 0.6:
            return "Monitor closely - potential drift detected"
        else:
            if drift_type == DriftType.CONCEPT_DRIFT:
                return "Retrain model - concept drift detected"
            elif drift_type == DriftType.COVARIATE_SHIFT:
                return "Update feature preprocessing - input distribution changed"
            elif drift_type == DriftType.PRIOR_SHIFT:
                return "Rebalance training data - target distribution changed"
            else:
                return "Investigate model performance - degradation detected"


class ContinualLearningEngine:
    """Main continual learning engine with automated retraining"""

    def __init__(self, config: Optional[ContinualLearningConfig] = None):
        self.config = config or ContinualLearningConfig()
        self.logger = logging.getLogger(f"{__name__}.ContinualLearningEngine")

        if not HAS_TORCH:
            self.logger.error("PyTorch not available - continual learning disabled")
            return

        # Core components
        self.drift_detector = DriftDetector(self.config)

        # Model management
        self.active_models: Dict[str, nn.Module] = {}
        self.model_histories: Dict[str, List] = {}
        self.rehearsal_buffers: Dict[str, List] = {}

        # Meta-learning
        self.meta_learners: Dict[str, MAMLLearner] = {}

        # Automated retraining
        self.retraining_queue: List[RetrainingTask] = []
        self.completed_tasks: List[RetrainingTask] = []

        # Performance tracking
        self.performance_history: Dict[str, List] = {}
        self.learning_curves: Dict[str, List] = {}

        # Knowledge distillation models
        self.teacher_models: Dict[str, nn.Module] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()

        # Start automated retraining scheduler
        self._start_retraining_scheduler()

        self.logger.info(f"Continual Learning Engine initialized on {self.device}")

    def register_model(self, model_id: str, model: nn.Module, enable_meta_learning: bool = True):
        """Register model for continual learning"""
        with self._lock:
            try:
                self.active_models[model_id] = model.to(self.device)
                self.model_histories[model_id] = []
                self.rehearsal_buffers[model_id] = []
                self.performance_history[model_id] = []

                if enable_meta_learning:
                    self.meta_learners[model_id] = MAMLLearner(
                        model, self.config.meta_learning_rate
                    ).to(self.device)

                self.logger.info(f"Registered model for continual learning: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to register model {model_id}: {e}")
                return False

    def online_update(
        self, model_id: str, batch_x: torch.Tensor, batch_y: torch.Tensor
    ) -> Dict[str, float]:
        """Perform online learning update"""
        with self._lock:
            try:
                if model_id not in self.active_models:
                    return {"error": "Model not registered"}

                model = self.active_models[model_id]
                model.train()

                # Online SGD update
                optimizer = optim.SGD(model.parameters(), lr=self.config.online_learning_rate)

                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = F.mse_loss(predictions, batch_y)

                # Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
                if self.config.elastic_weight_consolidation and model_id in self.model_histories:
                    ewc_loss = self._calculate_ewc_loss(model_id, model)
                    loss += self.config.ewc_lambda * ewc_loss

                loss.backward()
                optimizer.step()

                # Update rehearsal buffer
                self._update_rehearsal_buffer(model_id, batch_x, batch_y)

                # Store performance
                with torch.no_grad():
                    predictions_np = predictions.cpu().numpy()
                    targets_np = batch_y.cpu().numpy()
                    mse = np.mean((predictions_np - targets_np) ** 2)

                    self.performance_history[model_id].append(
                        {"mse": mse, "timestamp": datetime.now(), "batch_size": len(batch_x)}
                    )

                return {"loss": loss.item(), "mse": mse, "updated": True}

            except Exception as e:
                self.logger.error(f"Online update failed for {model_id}: {e}")
                return {"error": str(e)}

    def meta_learn_new_coin(
        self,
        base_model_id: str,
        new_coin_symbol: str,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> str:
        """Learn new coin using meta-learning"""
        with self._lock:
            try:
                if base_model_id not in self.meta_learners:
                    return None

                meta_learner = self.meta_learners[base_model_id]
                new_model_id = f"{base_model_id}_{new_coin_symbol}"

                # Adapt base model to new coin
                support_x, support_y = support_data
                adapted_model = meta_learner.adapt(
                    support_x.to(self.device),
                    support_y.to(self.device),
                    self.config.adaptation_steps,
                )

                # Validate on query set
                query_x, query_y = query_data
                adapted_model.eval()
                with torch.no_grad():
                    query_pred = adapted_model(query_x.to(self.device))
                    validation_loss = F.mse_loss(query_pred, query_y.to(self.device))

                # Register adapted model
                self.register_model(new_model_id, adapted_model, enable_meta_learning=False)

                self.logger.info(
                    f"Meta-learned new coin model: {new_model_id}, validation loss: {validation_loss:.4f}"
                )
                return new_model_id

            except Exception as e:
                self.logger.error(f"Meta-learning failed for {new_coin_symbol}: {e}")
                return None

    def detect_and_handle_drift(
        self, model_id: str, new_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> DriftDetectionResult:
        """Detect drift and automatically handle retraining"""
        try:
            features, predictions, targets = new_data

            # Convert to numpy for drift detection
            features_np = features.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Detect drift
            drift_result = self.drift_detector.detect_drift(
                model_id, features_np, predictions_np, targets_np
            )

            # Update reference data
            self.drift_detector.update_reference_data(
                model_id, features_np, predictions_np, targets_np
            )

            # Handle drift if detected
            if drift_result.drift_detected:
                self._handle_detected_drift(model_id, drift_result)

            return drift_result

        except Exception as e:
            self.logger.error(f"Drift detection failed for {model_id}: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.VIRTUAL_DRIFT,
                confidence=0.0,
                p_value=1.0,
                drift_score=0.0,
                affected_features=[],
                recommendation="Error in drift detection",
                timestamp=datetime.now(),
            )

    def _handle_detected_drift(self, model_id: str, drift_result: DriftDetectionResult):
        """Handle detected drift by scheduling retraining"""
        trigger_map = {
            DriftType.CONCEPT_DRIFT: RetrainingTrigger.DRIFT_DETECTED,
            DriftType.COVARIATE_SHIFT: RetrainingTrigger.DRIFT_DETECTED,
            DriftType.PRIOR_SHIFT: RetrainingTrigger.DRIFT_DETECTED,
            DriftType.VIRTUAL_DRIFT: RetrainingTrigger.PERFORMANCE_DEGRADATION,
        }

        trigger = trigger_map.get(drift_result.drift_type, RetrainingTrigger.DRIFT_DETECTED)

        # Create retraining task
        task = RetrainingTask(
            task_id=f"drift_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger=trigger,
            model_id=model_id,
            coin_symbols=[],  # Will be filled when executed
            priority=int(drift_result.confidence * 10),  # Higher confidence = higher priority
            created_at=datetime.now(),
            scheduled_for=datetime.now() + timedelta(minutes=5),  # Small delay for batching
        )

        self.retraining_queue.append(task)
        self.logger.info(
            f"Scheduled retraining for {model_id} due to {drift_result.drift_type.value}"
        )

    def _update_rehearsal_buffer(self, model_id: str, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """Update rehearsal buffer for catastrophic forgetting prevention"""
        buffer = self.rehearsal_buffers[model_id]

        # Add new samples
        for i in range(len(batch_x)):
            sample = {"x": batch_x[i].cpu(), "y": batch_y[i].cpu(), "timestamp": datetime.now()}
            buffer.append(sample)

        # Remove old samples if buffer is full
        if len(buffer) > self.config.rehearsal_buffer_size:
            # Keep diverse samples (remove older ones with some randomization)
            remove_count = len(buffer) - self.config.rehearsal_buffer_size
            indices_to_remove = np.random.choice(
                len(buffer) // 2,  # Remove from first half (older samples)
                size=remove_count,
                replace=False,
            )

            for idx in sorted(indices_to_remove, reverse=True):
                buffer.pop(idx)

    def _calculate_ewc_loss(self, model_id: str, model: nn.Module) -> torch.Tensor:
        """Calculate Elastic Weight Consolidation loss"""
        try:
            if model_id not in self.model_histories or not self.model_histories[model_id]:
                return torch.tensor(0.0, device=self.device)

            ewc_loss = torch.tensor(0.0, device=self.device)

            # Get reference parameters from model history
            ref_params = self.model_histories[model_id][-1]  # Most recent checkpoint

            for name, param in model.named_parameters():
                if name in ref_params["params"] and name in ref_params["fisher"]:
                    ref_param = ref_params["params"][name].to(self.device)
                    fisher_info = ref_params["fisher"][name].to(self.device)

                    # EWC penalty: Fisher Information * (param - ref_param)^2
                    ewc_loss += (fisher_info * (param - ref_param) ** 2).sum()

            return ewc_loss

        except Exception as e:
            self.logger.warning(f"EWC loss calculation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def save_model_checkpoint(self, model_id: str, include_fisher: bool = True):
        """Save model checkpoint with Fisher Information for EWC"""
        try:
            if model_id not in self.active_models:
                return False

            model = self.active_models[model_id]
            checkpoint = {"params": {}, "fisher": {}, "timestamp": datetime.now()}

            # Save parameters
            for name, param in model.named_parameters():
                checkpoint["params"][name] = param.data.clone()

            # Calculate Fisher Information if requested
            if include_fisher and self.rehearsal_buffers[model_id]:
                checkpoint["fisher"] = self._calculate_fisher_information(model_id, model)

            # Store in model history
            self.model_histories[model_id].append(checkpoint)

            # Limit history size
            if len(self.model_histories[model_id]) > 5:
                self.model_histories[model_id] = self.model_histories[model_id][-5:]

            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for {model_id}: {e}")
            return False

    def _calculate_fisher_information(
        self, model_id: str, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Calculate Fisher Information Matrix for EWC"""
        try:
            model.train()
            fisher_info = {}

            # Initialize Fisher Information
            for name, param in model.named_parameters():
                fisher_info[name] = torch.zeros_like(param.data)

            # Sample from rehearsal buffer
            buffer = self.rehearsal_buffers[model_id]
            if not buffer:
                return fisher_info

            sample_size = min(100, len(buffer))  # Limit for efficiency
            sampled_data = np.random.choice(buffer, size=sample_size, replace=False)

            for sample in sampled_data:
                model.zero_grad()

                x = sample["x"].unsqueeze(0).to(self.device)
                y = sample["y"].unsqueeze(0).to(self.device)

                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()

                # Accumulate squared gradients (Fisher Information approximation)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data**2

            # Normalize by number of samples
            for name in fisher_info:
                fisher_info[name] /= sample_size

            return fisher_info

        except Exception as e:
            self.logger.error(f"Fisher Information calculation failed: {e}")
            return {}

    def _start_retraining_scheduler(self):
        """Start automated retraining scheduler"""
        import threading
        import time

        def scheduler_loop():
            while True:
                try:
                    self._process_retraining_queue()
                    self._schedule_periodic_retraining()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    self.logger.error(f"Retraining scheduler error: {e}")
                    time.sleep(60)

        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        self.logger.info("Automated retraining scheduler started")

    def _process_retraining_queue(self):
        """Process pending retraining tasks"""
        with self._lock:
            current_time = datetime.now()

            # Find tasks ready for execution
            ready_tasks = [
                task
                for task in self.retraining_queue
                if task.status == "pending"
                and (task.scheduled_for is None or task.scheduled_for <= current_time)
            ]

            # Sort by priority
            ready_tasks.sort(key=lambda t: t.priority, reverse=True)

            # Execute high-priority tasks
            for task in ready_tasks[:3]:  # Limit concurrent retraining
                self._execute_retraining_task(task)

    def _execute_retraining_task(self, task: RetrainingTask):
        """Execute a retraining task"""
        try:
            task.status = "running"
            task.progress = 0.1

            model_id = task.model_id

            if model_id not in self.active_models:
                task.status = "failed"
                task.result = {"error": "Model not found"}
                return

            # Save current model as teacher for knowledge distillation
            if self.config.knowledge_distillation:
                self.teacher_models[model_id] = deepcopy(self.active_models[model_id])

            # Get training data from rehearsal buffer and recent data
            training_data = self._prepare_retraining_data(model_id)

            if not training_data:
                task.status = "failed"
                task.result = {"error": "Insufficient training data"}
                return

            task.progress = 0.3

            # Retrain model
            retrain_result = self._retrain_model(model_id, training_data, task)

            task.progress = 1.0
            task.status = "completed"
            task.result = retrain_result

            # Move to completed tasks
            self.retraining_queue.remove(task)
            self.completed_tasks.append(task)

            self.logger.info(f"Retraining completed for {model_id}: {retrain_result}")

        except Exception as e:
            task.status = "failed"
            task.result = {"error": str(e)}
            self.logger.error(f"Retraining task failed: {e}")

    def _prepare_retraining_data(self, model_id: str) -> Optional[Tuple]:
        """Prepare data for retraining"""
        try:
            buffer = self.rehearsal_buffers[model_id]

            if len(buffer) < self.config.min_samples_for_drift:
                return None

            # Combine rehearsal buffer data
            x_data = torch.stack([sample["x"] for sample in buffer])
            y_data = torch.stack([sample["y"] for sample in buffer])

            return x_data, y_data

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None

    def _retrain_model(
        self, model_id: str, training_data: Tuple, task: RetrainingTask
    ) -> Dict[str, Any]:
        """Retrain model with continual learning techniques"""
        try:
            model = self.active_models[model_id]
            x_data, y_data = training_data

            # Move to device
            x_data = x_data.to(self.device)
            y_data = y_data.to(self.device)

            # Split data
            val_size = int(len(x_data) * self.config.validation_split)
            train_x, val_x = x_data[:-val_size], x_data[-val_size:]
            train_y, val_y = y_data[:-val_size], y_data[-val_size:]

            # Setup training
            optimizer = optim.Adam(model.parameters(), lr=self.config.online_learning_rate * 10)
            best_val_loss = float("inf")
            patience = 0

            model.train()

            # Training loop
            for epoch in range(100):  # Max epochs
                optimizer.zero_grad()

                # Forward pass
                train_pred = model(train_x)
                loss = F.mse_loss(train_pred, train_y)

                # Add EWC regularization
                if self.config.elastic_weight_consolidation:
                    ewc_loss = self._calculate_ewc_loss(model_id, model)
                    loss += self.config.ewc_lambda * ewc_loss

                # Knowledge distillation
                if self.config.knowledge_distillation and model_id in self.teacher_models:
                    teacher_model = self.teacher_models[model_id]
                    teacher_model.eval()

                    with torch.no_grad():
                        teacher_pred = teacher_model(train_x)

                    # Distillation loss
                    distill_loss = F.mse_loss(
                        F.softmax(train_pred / self.config.distillation_temperature, dim=-1),
                        F.softmax(teacher_pred / self.config.distillation_temperature, dim=-1),
                    )
                    loss += 0.5 * distill_loss

                loss.backward()
                optimizer.step()

                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(val_x)
                        val_loss = F.mse_loss(val_pred, val_y)

                    model.train()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1

                    # Update progress
                    task.progress = 0.3 + 0.6 * (epoch / 100)

                    if patience >= self.config.early_stopping_patience:
                        break

            # Save checkpoint after retraining
            self.save_model_checkpoint(model_id, include_fisher=True)

            return {
                "success": True,
                "final_train_loss": loss.item(),
                "final_val_loss": best_val_loss.item(),
                "epochs_trained": epoch + 1,
            }

        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            return {"success": False, "error": str(e)}

    def _schedule_periodic_retraining(self):
        """Schedule periodic retraining tasks"""
        current_time = datetime.now()

        for model_id in self.active_models:
            # Check if we need to schedule periodic retraining
            recent_tasks = [
                task
                for task in self.completed_tasks + self.retraining_queue
                if task.model_id == model_id
                and task.trigger == RetrainingTrigger.SCHEDULED
                and task.created_at
                > current_time - timedelta(hours=self.config.retraining_schedule_hours)
            ]

            if not recent_tasks:
                # Schedule periodic retraining
                task = RetrainingTask(
                    task_id=f"scheduled_{model_id}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                    trigger=RetrainingTrigger.SCHEDULED,
                    model_id=model_id,
                    coin_symbols=[],
                    priority=1,  # Low priority for scheduled tasks
                    created_at=current_time,
                    scheduled_for=current_time
                    + timedelta(hours=self.config.retraining_schedule_hours),
                )

                self.retraining_queue.append(task)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive continual learning summary"""
        with self._lock:
            return {
                "registered_models": len(self.active_models),
                "meta_learners": len(self.meta_learners),
                "pending_retraining_tasks": len(
                    [t for t in self.retraining_queue if t.status == "pending"]
                ),
                "running_retraining_tasks": len(
                    [t for t in self.retraining_queue if t.status == "running"]
                ),
                "completed_retraining_tasks": len(self.completed_tasks),
                "rehearsal_buffer_sizes": {
                    model_id: len(buffer) for model_id, buffer in self.rehearsal_buffers.items()
                },
                "recent_drift_detections": len(
                    [
                        task
                        for task in self.completed_tasks
                        if task.trigger == RetrainingTrigger.DRIFT_DETECTED
                        and task.created_at > datetime.now() - timedelta(days=1)
                    ]
                ),
                "config": {
                    "drift_threshold": self.config.drift_threshold,
                    "online_learning_rate": self.config.online_learning_rate,
                    "retraining_schedule_hours": self.config.retraining_schedule_hours,
                    "ewc_enabled": self.config.elastic_weight_consolidation,
                },
            }


# Singleton continual learning engine
_continual_learning_engine = None
_cl_lock = threading.Lock()


def get_continual_learning_engine(
    config: Optional[ContinualLearningConfig] = None,
) -> ContinualLearningEngine:
    """Get the singleton continual learning engine"""
    global _continual_learning_engine

    with _cl_lock:
        if _continual_learning_engine is None:
            _continual_learning_engine = ContinualLearningEngine(config)
        return _continual_learning_engine


def register_model_for_continual_learning(model_id: str, model: nn.Module) -> bool:
    """Convenient function to register model for continual learning"""
    engine = get_continual_learning_engine()
    return engine.register_model(model_id, model)
