#!/usr/bin/env python3
"""
Model Registry - Version control for ML models and metadata
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ModelMetadata:
    """Model metadata container"""

    model_id: str
    version: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_data_hash: str
    feature_columns: List[str]
    target_column: str
    training_timestamp: str
    model_file_path: str
    feature_engineering_version: str
    validation_scores: Dict[str, float]
    model_size_bytes: int
    training_duration_seconds: float
    drift_baseline: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """
    Enterprise model registry for version control and metadata management

    Features:
    - Model versioning with metadata
    - Training data fingerprinting
    - Performance tracking
    - Model promotion workflow
    - Drift baseline storage
    """

    def __init__(self, registry_path: str = "./models"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.registry_path / "active").mkdir(exist_ok=True)
        (self.registry_path / "archive").mkdir(exist_ok=True)
        (self.registry_path / "staging").mkdir(exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self.logger = logging.getLogger(__name__)

        # Load existing registry
        self.models = self._load_registry()

    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)

                models = {}
                for model_id, metadata_dict in data.items():
                    models[model_id] = ModelMetadata(**metadata_dict)

                return models
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                return {}

        return {}

    def _save_registry(self):
        """Save model registry to disk"""
        try:
            data = {model_id: asdict(metadata) for model_id, metadata in self.models.items()}

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of training data for fingerprinting"""
        # Sort columns for consistent hashing
        df_sorted = df.reindex(sorted(df.columns), axis=1)

        # Convert to string representation
        data_str = df_sorted.to_string()

        # Calculate hash
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def register_model(
        self,
        model,
        model_id: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        feature_engineering_version: str = "1.0",
        validation_scores: Optional[Dict[str, float]] = None,
        training_duration_seconds: float = 0.0,
        stage: str = "staging",
    ) -> str:
        """
        Register a new model in the registry

        Args:
            model: Trained model object
            model_id: Unique model identifier
            algorithm: Algorithm name
            hyperparameters: Model hyperparameters
            performance_metrics: Training performance metrics
            training_data: Training dataset
            feature_columns: Feature column names
            target_column: Target column name
            feature_engineering_version: Feature engineering version
            validation_scores: Validation scores
            training_duration_seconds: Training duration
            stage: Model stage (staging, active, archive)

        Returns:
            Model version string
        """

        # Generate version
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Calculate data hash
        data_hash = self._calculate_data_hash(training_data)

        # Determine model file path
        model_filename = f"{model_id}_v{version}.pkl"

        if stage == "staging":
            model_file_path = self.registry_path / "staging" / model_filename
        elif stage == "active":
            model_file_path = self.registry_path / "active" / model_filename
        else:
            model_file_path = self.registry_path / "archive" / model_filename

        # Save model to disk
        try:
            with open(model_file_path, "wb") as f:
                pickle.dump(model, f)

            model_size = model_file_path.stat().st_size

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            performance_metrics=performance_metrics,
            training_data_hash=data_hash,
            feature_columns=feature_columns,
            target_column=target_column,
            training_timestamp=datetime.utcnow().isoformat(),
            model_file_path=str(model_file_path),
            feature_engineering_version=feature_engineering_version,
            validation_scores=validation_scores or {},
            model_size_bytes=model_size,
            training_duration_seconds=training_duration_seconds,
        )

        # Store in registry
        registry_key = f"{model_id}_v{version}"
        self.models[registry_key] = metadata

        # Save registry
        self._save_registry()

        self.logger.info(f"Registered model {model_id} version {version}")
        return version

    def load_model(self, model_id: str, version: Optional[str] = None):
        """
        Load model from registry

        Args:
            model_id: Model identifier
            version: Specific version (None for latest)

        Returns:
            Loaded model object
        """

        if version:
            registry_key = f"{model_id}_v{version}"
        else:
            # Find latest version
            model_versions = [key for key in self.models.keys() if key.startswith(f"{model_id}_v")]

            if not model_versions:
                raise ValueError(f"No models found for {model_id}")

            # Sort by version (timestamp) and get latest
            registry_key = sorted(model_versions)[-1]

        if registry_key not in self.models:
            raise ValueError(f"Model {registry_key} not found in registry")

        metadata = self.models[registry_key]

        try:
            with open(metadata.model_file_path, "rb") as f:
                model = pickle.load(f)

            self.logger.info(f"Loaded model {registry_key}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load model {registry_key}: {e}")
            raise

    def get_model_metadata(self, model_id: str, version: Optional[str] = None) -> ModelMetadata:
        """Get model metadata"""
        if version:
            registry_key = f"{model_id}_v{version}"
        else:
            # Find latest version
            model_versions = [key for key in self.models.keys() if key.startswith(f"{model_id}_v")]

            if not model_versions:
                raise ValueError(f"No models found for {model_id}")

            registry_key = sorted(model_versions)[-1]

        if registry_key not in self.models:
            raise ValueError(f"Model {registry_key} not found in registry")

        return self.models[registry_key]

    def promote_model(self, model_id: str, version: str) -> bool:
        """
        Promote model from staging to active

        Args:
            model_id: Model identifier
            version: Model version to promote

        Returns:
            Success status
        """

        registry_key = f"{model_id}_v{version}"

        if registry_key not in self.models:
            raise ValueError(f"Model {registry_key} not found")

        metadata = self.models[registry_key]

        # Move model file to active directory
        current_path = Path(metadata.model_file_path)
        new_path = self.registry_path / "active" / current_path.name

        try:
            # Copy file to active directory
            import shutil

            shutil.copy2(current_path, new_path)

            # Update metadata
            metadata.model_file_path = str(new_path)

            # Archive previous active model if exists
            self._archive_active_models(model_id, exclude_version=version)

            # Save registry
            self._save_registry()

            self.logger.info(f"Promoted model {model_id} version {version} to active")
            return True

        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
            return False

    def _archive_active_models(self, model_id: str, exclude_version: str):
        """Archive existing active models for the same model_id"""
        active_dir = self.registry_path / "active"
        archive_dir = self.registry_path / "archive"

        for registry_key, metadata in self.models.items():
            if (
                metadata.model_id == model_id
                and metadata.version != exclude_version
                and "active" in metadata.model_file_path
            ):
                current_path = Path(metadata.model_file_path)
                archive_path = archive_dir / current_path.name

                try:
                    import shutil

                    shutil.move(current_path, archive_path)
                    metadata.model_file_path = str(archive_path)

                except Exception as e:
                    self.logger.warning(f"Failed to archive {registry_key}: {e}")

    def list_models(
        self, model_id: Optional[str] = None, stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List models in registry

        Args:
            model_id: Filter by model ID
            stage: Filter by stage (staging, active, archive)

        Returns:
            List of model summaries
        """

        models = []

        for registry_key, metadata in self.models.items():
            # Apply filters
            if model_id and metadata.model_id != model_id:
                continue

            if stage:
                if stage == "active" and "active" not in metadata.model_file_path:
                    continue
                elif stage == "staging" and "staging" not in metadata.model_file_path:
                    continue
                elif stage == "archive" and "archive" not in metadata.model_file_path:
                    continue

            models.append(
                {
                    "registry_key": registry_key,
                    "model_id": metadata.model_id,
                    "version": metadata.version,
                    "algorithm": metadata.algorithm,
                    "training_timestamp": metadata.training_timestamp,
                    "performance_metrics": metadata.performance_metrics,
                    "stage": self._get_model_stage(metadata.model_file_path),
                }
            )

        # Sort by training timestamp
        models.sort(key=lambda x: x["training_timestamp"], reverse=True)
        return models

    def _get_model_stage(self, file_path: str) -> str:
        """Determine model stage from file path"""
        if "active" in file_path:
            return "active"
        elif "staging" in file_path:
            return "staging"
        elif "archive" in file_path:
            return "archive"
        else:
            return "unknown"

    def delete_model(self, model_id: str, version: str) -> bool:
        """
        Delete model and metadata

        Args:
            model_id: Model identifier
            version: Model version

        Returns:
            Success status
        """

        registry_key = f"{model_id}_v{version}"

        if registry_key not in self.models:
            raise ValueError(f"Model {registry_key} not found")

        metadata = self.models[registry_key]

        try:
            # Delete model file
            model_path = Path(metadata.model_file_path)
            if model_path.exists():
                model_path.unlink()

            # Remove from registry
            del self.models[registry_key]

            # Save registry
            self._save_registry()

            self.logger.info(f"Deleted model {registry_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {registry_key}: {e}")
            return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_models": len(self.models),
            "models_by_stage": {},
            "models_by_algorithm": {},
            "total_size_mb": 0,
        }

        for metadata in self.models.values():
            # Count by stage
            stage = self._get_model_stage(metadata.model_file_path)
            stats["models_by_stage"][stage] = stats["models_by_stage"].get(stage, 0) + 1

            # Count by algorithm
            algo = metadata.algorithm
            stats["models_by_algorithm"][algo] = stats["models_by_algorithm"].get(algo, 0) + 1

            # Sum sizes
            stats["total_size_mb"] += metadata.model_size_bytes / (1024 * 1024)

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats


# Global model registry
model_registry = ModelRegistry()
