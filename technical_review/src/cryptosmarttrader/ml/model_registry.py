"""
Model Registry for CryptoSmartTrader
Enterprise model versioning, tracking, and deployment management.
"""

import hashlib
import json
import pickle
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import pandas as pd


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    STAGED = "staged"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"
    CANARY = "canary"


class ModelType(Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    log_loss: float
    
    # Trading-specific metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Out-of-sample validation
    oos_accuracy: Optional[float] = None
    oos_sharpe: Optional[float] = None
    
    # Statistical metrics
    information_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    # Stability metrics
    prediction_stability: Optional[float] = None
    feature_importance_stability: Optional[float] = None


@dataclass
class DatasetInfo:
    """Dataset metadata and validation."""
    dataset_hash: str
    feature_count: int
    sample_count: int
    start_date: datetime
    end_date: datetime
    features: List[str]
    target_column: str
    preprocessing_steps: List[str]
    validation_split: float
    test_split: float
    
    # Data quality metrics
    missing_value_ratio: float
    outlier_ratio: float
    correlation_matrix_hash: str
    feature_distributions_hash: str


@dataclass
class TrainingConfig:
    """Training configuration and hyperparameters."""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    training_script: str
    training_environment: Dict[str, str]
    random_seed: int
    training_duration_seconds: float
    cross_validation_folds: int
    early_stopping_patience: Optional[int] = None
    
    # Training metadata
    git_commit: Optional[str] = None
    python_version: str = "3.11"
    dependencies: Dict[str, str] = None


@dataclass
class ModelVersion:
    """Complete model version with all metadata."""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    
    # Core components
    metrics: ModelMetrics
    dataset_info: DatasetInfo
    training_config: TrainingConfig
    
    # Model artifacts
    model_path: str
    artifacts_path: str
    
    # Deployment info
    deployment_info: Optional[Dict[str, Any]] = None
    rollback_info: Optional[Dict[str, Any]] = None
    
    # Risk management
    risk_budget_pct: float = 1.0
    canary_duration_hours: int = 72
    
    # Performance tracking
    live_metrics: Optional[Dict[str, float]] = None
    drift_metrics: Optional[Dict[str, float]] = None
    
    # Metadata
    tags: List[str] = None
    notes: str = ""
    creator: str = "system"


class ModelRegistry:
    """
    Enterprise model registry with versioning, deployment tracking, and drift detection.
    
    Features:
    - Complete model lifecycle management
    - Dataset versioning with hash validation
    - Training configuration tracking
    - Comprehensive metrics collection
    - Canary deployment support
    - Automatic rollback capabilities
    - Drift detection and monitoring
    - Performance comparison across versions
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry structure
        self.models_path = self.registry_path / "models"
        self.artifacts_path = self.registry_path / "artifacts"
        self.metadata_path = self.registry_path / "metadata"
        self.datasets_path = self.registry_path / "datasets"
        
        for path in [self.models_path, self.artifacts_path, self.metadata_path, self.datasets_path]:
            path.mkdir(exist_ok=True)
        
        # Registry database (JSON-based for simplicity)
        self.registry_db_path = self.registry_path / "registry.json"
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing registry
        self.registry: Dict[str, Dict[str, ModelVersion]] = self._load_registry()
        
        # Current deployments
        self.deployments_path = self.registry_path / "deployments.json"
        self.deployments: Dict[str, Dict[str, Any]] = self._load_deployments()
        
        self.logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def register_model(self,
                      model: Any,
                      model_type: ModelType,
                      dataset: pd.DataFrame,
                      target_column: str,
                      metrics: ModelMetrics,
                      training_config: TrainingConfig,
                      model_id: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      notes: str = "") -> ModelVersion:
        """
        Register a new model version in the registry.
        
        Args:
            model: Trained model object
            model_type: Type of model
            dataset: Training dataset for hash calculation
            target_column: Target column name
            metrics: Model performance metrics
            training_config: Training configuration
            model_id: Optional model ID (auto-generated if None)
            tags: Optional tags for organization
            notes: Optional notes
            
        Returns:
            ModelVersion with complete metadata
        """
        
        # Generate model ID and version
        if model_id is None:
            model_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate version number
        existing_versions = self.registry.get(model_id, {})
        version_num = len(existing_versions) + 1
        version = f"v{version_num:03d}"
        
        # Generate dataset information
        dataset_info = self._create_dataset_info(dataset, target_column, training_config)
        
        # Save model artifacts
        model_artifacts_path = self.artifacts_path / model_id / version
        model_artifacts_path.mkdir(parents=True, exist_ok=True)
        
        model_file_path = model_artifacts_path / "model.pkl"
        
        # Save model
        try:
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
        
        # Save additional artifacts
        self._save_artifacts(model_artifacts_path, {
            'training_config': asdict(training_config),
            'dataset_info': asdict(dataset_info),
            'metrics': asdict(metrics),
            'feature_names': dataset.columns.tolist()
        })
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            status=ModelStatus.STAGED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metrics=metrics,
            dataset_info=dataset_info,
            training_config=training_config,
            model_path=str(model_file_path),
            artifacts_path=str(model_artifacts_path),
            tags=tags or [],
            notes=notes
        )
        
        # Register in database
        if model_id not in self.registry:
            self.registry[model_id] = {}
        
        self.registry[model_id][version] = model_version
        self._save_registry()
        
        self.logger.info(f"Model registered: {model_id} {version}")
        return model_version
    
    def deploy_to_production(self,
                           model_id: str,
                           version: str,
                           risk_budget_pct: float = 1.0,
                           canary_duration_hours: int = 72) -> bool:
        """
        Deploy model version to production with canary deployment.
        
        Args:
            model_id: Model identifier
            version: Version to deploy
            risk_budget_pct: Risk budget percentage for canary
            canary_duration_hours: Canary deployment duration
            
        Returns:
            True if deployment successful
        """
        
        if model_id not in self.registry or version not in self.registry[model_id]:
            raise ValueError(f"Model {model_id} {version} not found in registry")
        
        model_version = self.registry[model_id][version]
        
        # Check if model is eligible for production
        if model_version.status not in [ModelStatus.STAGED, ModelStatus.CANARY]:
            raise ValueError(f"Model {model_id} {version} not eligible for production (status: {model_version.status})")
        
        # Start canary deployment
        if model_version.status == ModelStatus.STAGED:
            self.logger.info(f"Starting canary deployment: {model_id} {version}")
            
            # Update model status
            model_version.status = ModelStatus.CANARY
            model_version.risk_budget_pct = min(risk_budget_pct, 1.0)  # Cap at 1%
            model_version.canary_duration_hours = canary_duration_hours
            model_version.updated_at = datetime.utcnow()
            
            # Set deployment info
            model_version.deployment_info = {
                'canary_start': datetime.utcnow().isoformat(),
                'canary_end': (datetime.utcnow() + timedelta(hours=canary_duration_hours)).isoformat(),
                'risk_budget_pct': risk_budget_pct,
                'deployment_type': 'canary'
            }
        
        # Check if canary period is complete
        elif model_version.status == ModelStatus.CANARY:
            canary_start = datetime.fromisoformat(model_version.deployment_info['canary_start'])
            canary_duration = timedelta(hours=model_version.canary_duration_hours)
            
            if datetime.utcnow() >= canary_start + canary_duration:
                # Canary period complete, promote to production
                self.logger.info(f"Promoting canary to production: {model_id} {version}")
                
                # Archive current production model
                current_production = self._get_production_model()
                if current_production:
                    self._archive_model(current_production['model_id'], current_production['version'])
                
                # Promote to production
                model_version.status = ModelStatus.PRODUCTION
                model_version.risk_budget_pct = 100.0  # Full production traffic
                model_version.deployment_info.update({
                    'production_start': datetime.utcnow().isoformat(),
                    'deployment_type': 'production',
                    'promoted_from_canary': True
                })
            else:
                self.logger.info(f"Canary deployment in progress: {model_id} {version}")
                return True
        
        # Update deployments
        self.deployments['current_production'] = {
            'model_id': model_id,
            'version': version,
            'deployed_at': datetime.utcnow().isoformat(),
            'status': model_version.status.value,
            'risk_budget_pct': model_version.risk_budget_pct
        }
        
        # Save changes
        self.registry[model_id][version] = model_version
        self._save_registry()
        self._save_deployments()
        
        return True
    
    def rollback_model(self, reason: str = "Performance degradation") -> bool:
        """
        Rollback current production model to previous stable version.
        
        Args:
            reason: Reason for rollback
            
        Returns:
            True if rollback successful
        """
        
        current_production = self._get_production_model()
        if not current_production:
            self.logger.warning("No production model to rollback")
            return False
        
        # Find previous production model
        rollback_target = self._find_rollback_target()
        if not rollback_target:
            self.logger.error("No suitable rollback target found")
            return False
        
        # Perform rollback
        current_model_version = self.registry[current_production['model_id']][current_production['version']]
        rollback_model_version = self.registry[rollback_target['model_id']][rollback_target['version']]
        
        # Archive current model
        current_model_version.status = ModelStatus.ARCHIVED
        current_model_version.rollback_info = {
            'rollback_time': datetime.utcnow().isoformat(),
            'rollback_reason': reason,
            'rolled_back_to': f"{rollback_target['model_id']} {rollback_target['version']}"
        }
        
        # Restore previous model
        rollback_model_version.status = ModelStatus.PRODUCTION
        rollback_model_version.updated_at = datetime.utcnow()
        rollback_model_version.deployment_info = rollback_model_version.deployment_info or {}
        rollback_model_version.deployment_info.update({
            'rollback_deployment': datetime.utcnow().isoformat(),
            'rollback_from': f"{current_production['model_id']} {current_production['version']}",
            'rollback_reason': reason
        })
        
        # Update deployments
        self.deployments['current_production'] = {
            'model_id': rollback_target['model_id'],
            'version': rollback_target['version'],
            'deployed_at': datetime.utcnow().isoformat(),
            'status': 'production',
            'risk_budget_pct': 100.0,
            'rollback_info': {
                'rollback_from': f"{current_production['model_id']} {current_production['version']}",
                'rollback_reason': reason
            }
        }
        
        # Save changes
        self._save_registry()
        self._save_deployments()
        
        self.logger.info(f"Model rollback completed: {rollback_target['model_id']} {rollback_target['version']}")
        return True
    
    def update_live_metrics(self, model_id: str, version: str, live_metrics: Dict[str, float]):
        """Update live performance metrics for deployed model."""
        
        if model_id not in self.registry or version not in self.registry[model_id]:
            self.logger.warning(f"Model {model_id} {version} not found for metrics update")
            return
        
        model_version = self.registry[model_id][version]
        model_version.live_metrics = live_metrics
        model_version.updated_at = datetime.utcnow()
        
        self._save_registry()
    
    def detect_drift(self, model_id: str, version: str, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect data/statistical drift for deployed model.
        
        Args:
            model_id: Model identifier
            version: Model version
            current_data: Current data for drift comparison
            
        Returns:
            Drift metrics dictionary
        """
        
        if model_id not in self.registry or version not in self.registry[model_id]:
            raise ValueError(f"Model {model_id} {version} not found")
        
        model_version = self.registry[model_id][version]
        
        # Load original dataset hash and features for comparison
        original_features = model_version.dataset_info.features
        
        # Calculate drift metrics
        drift_metrics = {}
        
        # Feature drift (simple statistical tests)
        if all(col in current_data.columns for col in original_features):
            for feature in original_features:
                if feature in current_data.columns:
                    # Kolmogorov-Smirnov test simulation (simplified)
                    current_values = current_data[feature].dropna()
                    if len(current_values) > 0:
                        # Simple statistical drift detection
                        drift_score = self._calculate_drift_score(current_values)
                        drift_metrics[f"{feature}_drift"] = drift_score
        
        # Data quality drift
        current_missing_ratio = current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))
        original_missing_ratio = model_version.dataset_info.missing_value_ratio
        
        drift_metrics['missing_value_drift'] = abs(current_missing_ratio - original_missing_ratio)
        
        # Feature count drift
        feature_count_drift = abs(len(current_data.columns) - model_version.dataset_info.feature_count) / model_version.dataset_info.feature_count
        drift_metrics['feature_count_drift'] = feature_count_drift
        
        # Overall drift score
        drift_metrics['overall_drift'] = np.mean(list(drift_metrics.values()))
        
        # Update model with drift metrics
        model_version.drift_metrics = drift_metrics
        model_version.updated_at = datetime.utcnow()
        self._save_registry()
        
        return drift_metrics
    
    def get_model_comparison(self, model_id: str, versions: Optional[List[str]] = None) -> pd.DataFrame:
        """Get comparison table of model versions."""
        
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_versions = self.registry[model_id]
        if versions:
            model_versions = {v: model_versions[v] for v in versions if v in model_versions}
        
        comparison_data = []
        
        for version, model_version in model_versions.items():
            row = {
                'version': version,
                'status': model_version.status.value,
                'created_at': model_version.created_at,
                'accuracy': model_version.metrics.accuracy,
                'precision': model_version.metrics.precision,
                'recall': model_version.metrics.recall,
                'f1_score': model_version.metrics.f1_score,
                'auc_roc': model_version.metrics.auc_roc,
                'sharpe_ratio': model_version.metrics.sharpe_ratio,
                'max_drawdown': model_version.metrics.max_drawdown,
                'oos_accuracy': model_version.metrics.oos_accuracy,
                'dataset_samples': model_version.dataset_info.sample_count,
                'feature_count': model_version.dataset_info.feature_count,
                'risk_budget_pct': model_version.risk_budget_pct
            }
            
            # Add live metrics if available
            if model_version.live_metrics:
                for metric, value in model_version.live_metrics.items():
                    row[f'live_{metric}'] = value
            
            # Add drift metrics if available
            if model_version.drift_metrics:
                row['overall_drift'] = model_version.drift_metrics.get('overall_drift', 0.0)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('created_at', ascending=False)
    
    def list_models(self, status_filter: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List all models with optional status filtering."""
        
        models = []
        
        for model_id, versions in self.registry.items():
            for version, model_version in versions.items():
                if status_filter is None or model_version.status == status_filter:
                    models.append({
                        'model_id': model_id,
                        'version': version,
                        'status': model_version.status.value,
                        'created_at': model_version.created_at,
                        'model_type': model_version.model_type.value,
                        'accuracy': model_version.metrics.accuracy,
                        'sharpe_ratio': model_version.metrics.sharpe_ratio,
                        'tags': model_version.tags
                    })
        
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def load_model(self, model_id: str, version: str) -> Any:
        """Load model object from registry."""
        
        if model_id not in self.registry or version not in self.registry[model_id]:
            raise ValueError(f"Model {model_id} {version} not found")
        
        model_version = self.registry[model_id][version]
        
        try:
            with open(model_version.model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id} {version}: {e}")
            raise
    
    def get_production_model(self) -> Optional[Tuple[Any, ModelVersion]]:
        """Get current production model and its metadata."""
        
        current_production = self._get_production_model()
        if not current_production:
            return None
        
        model_id = current_production['model_id']
        version = current_production['version']
        
        model = self.load_model(model_id, version)
        model_version = self.registry[model_id][version]
        
        return model, model_version
    
    def cleanup_old_models(self, keep_versions: int = 10, archive_after_days: int = 30):
        """Cleanup old model versions and artifacts."""
        
        cleanup_count = 0
        
        for model_id, versions in self.registry.items():
            # Sort versions by creation date
            sorted_versions = sorted(
                versions.items(),
                key=lambda x: x[1].created_at,
                reverse=True
            )
            
            # Keep recent versions and production models
            for i, (version, model_version) in enumerate(sorted_versions):
                should_archive = (
                    i >= keep_versions and
                    model_version.status not in [ModelStatus.PRODUCTION, ModelStatus.CANARY] and
                    (datetime.utcnow() - model_version.created_at).days > archive_after_days
                )
                
                if should_archive:
                    self._archive_model(model_id, version)
                    cleanup_count += 1
        
        self.logger.info(f"Cleaned up {cleanup_count} old model versions")
        return cleanup_count
    
    # Private methods
    
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load registry from disk."""
        
        if not self.registry_db_path.exists():
            return {}
        
        try:
            with open(self.registry_db_path, 'r') as f:
                data = json.load(f)
            
            # Convert JSON back to ModelVersion objects
            registry = {}
            for model_id, versions in data.items():
                registry[model_id] = {}
                for version, version_data in versions.items():
                    # Convert datetime strings back to datetime objects
                    version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                    version_data['updated_at'] = datetime.fromisoformat(version_data['updated_at'])
                    version_data['dataset_info']['start_date'] = datetime.fromisoformat(version_data['dataset_info']['start_date'])
                    version_data['dataset_info']['end_date'] = datetime.fromisoformat(version_data['dataset_info']['end_date'])
                    
                    # Convert enums
                    version_data['model_type'] = ModelType(version_data['model_type'])
                    version_data['status'] = ModelStatus(version_data['status'])
                    
                    # Convert nested enum references
                    if 'training_config' in version_data and 'model_type' in version_data['training_config']:
                        version_data['training_config']['model_type'] = ModelType(version_data['training_config']['model_type'])
                    
                    # Create objects
                    version_data['metrics'] = ModelMetrics(**version_data['metrics'])
                    version_data['dataset_info'] = DatasetInfo(**version_data['dataset_info'])
                    version_data['training_config'] = TrainingConfig(**version_data['training_config'])
                    
                    registry[model_id][version] = ModelVersion(**version_data)
            
            return registry
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        
        try:
            # Convert ModelVersion objects to JSON-serializable format
            data = {}
            for model_id, versions in self.registry.items():
                data[model_id] = {}
                for version, model_version in versions.items():
                    version_dict = asdict(model_version)
                    
                    # Convert datetime objects to ISO strings
                    version_dict['created_at'] = model_version.created_at.isoformat()
                    version_dict['updated_at'] = model_version.updated_at.isoformat()
                    version_dict['dataset_info']['start_date'] = model_version.dataset_info.start_date.isoformat()
                    version_dict['dataset_info']['end_date'] = model_version.dataset_info.end_date.isoformat()
                    
                    # Convert enums to strings
                    version_dict['model_type'] = model_version.model_type.value
                    version_dict['status'] = model_version.status.value
                    
                    # Convert nested enum references
                    version_dict['training_config']['model_type'] = model_version.training_config.model_type.value
                    
                    data[model_id][version] = version_dict
            
            with open(self.registry_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def _load_deployments(self) -> Dict[str, Dict[str, Any]]:
        """Load deployment information."""
        
        if not self.deployments_path.exists():
            return {}
        
        try:
            with open(self.deployments_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load deployments: {e}")
            return {}
    
    def _save_deployments(self):
        """Save deployment information."""
        
        try:
            with open(self.deployments_path, 'w') as f:
                json.dump(self.deployments, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save deployments: {e}")
    
    def _create_dataset_info(self, dataset: pd.DataFrame, target_column: str, training_config: TrainingConfig) -> DatasetInfo:
        """Create dataset information with hash validation."""
        
        # Calculate dataset hash
        dataset_str = dataset.to_string()
        dataset_hash = hashlib.sha256(dataset_str.encode()).hexdigest()
        
        # Feature analysis
        features = [col for col in dataset.columns if col != target_column]
        missing_ratio = dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
        
        # Calculate correlation matrix hash
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = dataset[numeric_cols].corr()
            corr_hash = hashlib.sha256(str(corr_matrix.values).encode()).hexdigest()
        else:
            corr_hash = "no_numeric_features"
        
        # Feature distributions hash (simplified)
        feature_stats = dataset.describe().to_string()
        distributions_hash = hashlib.sha256(feature_stats.encode()).hexdigest()
        
        return DatasetInfo(
            dataset_hash=dataset_hash,
            feature_count=len(features),
            sample_count=len(dataset),
            start_date=datetime.utcnow() - timedelta(days=30),  # Placeholder
            end_date=datetime.utcnow(),
            features=features,
            target_column=target_column,
            preprocessing_steps=training_config.hyperparameters.get('preprocessing', []),
            validation_split=0.2,  # Default
            test_split=0.1,  # Default
            missing_value_ratio=missing_ratio,
            outlier_ratio=0.05,  # Placeholder
            correlation_matrix_hash=corr_hash,
            feature_distributions_hash=distributions_hash
        )
    
    def _save_artifacts(self, artifacts_path: Path, artifacts: Dict[str, Any]):
        """Save additional model artifacts."""
        
        for name, artifact in artifacts.items():
            artifact_path = artifacts_path / f"{name}.json"
            
            try:
                with open(artifact_path, 'w') as f:
                    json.dump(artifact, f, indent=2, default=str)
            except Exception as e:
                self.logger.warning(f"Failed to save artifact {name}: {e}")
    
    def _get_production_model(self) -> Optional[Dict[str, str]]:
        """Get current production model info."""
        
        return self.deployments.get('current_production')
    
    def _find_rollback_target(self) -> Optional[Dict[str, str]]:
        """Find suitable model for rollback."""
        
        # Look for most recent archived production model
        for model_id, versions in self.registry.items():
            for version, model_version in versions.items():
                if (model_version.status == ModelStatus.ARCHIVED and 
                    model_version.deployment_info and
                    model_version.deployment_info.get('deployment_type') == 'production'):
                    return {'model_id': model_id, 'version': version}
        
        # Fallback: find any stable model
        for model_id, versions in self.registry.items():
            for version, model_version in versions.items():
                if (model_version.status == ModelStatus.STAGED and 
                    model_version.metrics.accuracy > 0.7):  # Minimum quality threshold
                    return {'model_id': model_id, 'version': version}
        
        return None
    
    def _archive_model(self, model_id: str, version: str):
        """Archive a model version."""
        
        if model_id in self.registry and version in self.registry[model_id]:
            model_version = self.registry[model_id][version]
            model_version.status = ModelStatus.ARCHIVED
            model_version.updated_at = datetime.utcnow()
            
            self.logger.info(f"Archived model: {model_id} {version}")
    
    def _calculate_drift_score(self, current_values: pd.Series) -> float:
        """Calculate simple drift score for feature values."""
        
        # Simplified drift calculation using statistical properties
        try:
            mean_val = float(current_values.mean())
            std_val = float(current_values.std())
            
            # Simple drift score based on coefficient of variation
            if std_val > 0:
                cv = std_val / abs(mean_val) if mean_val != 0 else std_val
                drift_score = min(1.0, cv / 2.0)  # Normalize to 0-1
            else:
                drift_score = 0.0
            
            return drift_score
            
        except Exception:
            return 0.0


def create_model_registry(registry_path: str = "models/registry") -> ModelRegistry:
    """Create model registry instance."""
    return ModelRegistry(registry_path)