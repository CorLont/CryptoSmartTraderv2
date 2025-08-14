#!/usr/bin/env python3
"""
Enterprise Model Registry - Versioning, tracking en lifecycle management

Implementeert:
- Model versioning met metadata tracking
- Dataset hashing en versioning
- Evaluation metrics per model
- Drift detection en monitoring
- Canary deployment pipeline
- Rollback capabilities
"""

import hashlib
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DriftStatus(Enum):
    """Data drift severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    version: str
    name: str
    algorithm: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    
    # Training info
    training_dataset_hash: str
    feature_set_version: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float
    
    # Performance metrics
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    production_metrics: Dict[str, float]
    
    # Deployment info
    deployment_date: Optional[datetime] = None
    canary_percentage: float = 0.0
    risk_budget_used: float = 0.0
    
    # Monitoring
    drift_score: float = 0.0
    drift_status: DriftStatus = DriftStatus.NONE
    last_drift_check: Optional[datetime] = None
    
    # Metadata
    description: str = ""
    tags: List[str] = None
    author: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class DatasetVersion:
    """Dataset versioning metadata"""
    dataset_id: str
    version: str
    hash: str
    created_at: datetime
    size_rows: int
    size_bytes: int
    
    # Data quality metrics
    completeness_score: float
    consistency_score: float
    validity_score: float
    
    # Schema info
    feature_count: int
    feature_names: List[str]
    feature_types: Dict[str, str]
    
    # Lineage
    source_datasets: List[str] = None
    transformations: List[str] = None
    
    def __post_init__(self):
        if self.source_datasets is None:
            self.source_datasets = []
        if self.transformations is None:
            self.transformations = []


class ModelRegistry:
    """
    Enterprise Model Registry voor ML/AI governance
    
    Features:
    - Model versioning en metadata tracking
    - Dataset hashing en lineage tracking  
    - Performance metrics monitoring
    - Drift detection en alerting
    - Canary deployment orchestration
    - Automated rollback capabilities
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # Registry storage
        self.models_path = self.registry_path / "models"
        self.datasets_path = self.registry_path / "datasets"
        self.metrics_path = self.registry_path / "metrics"
        self.artifacts_path = self.registry_path / "artifacts"
        
        for path in [self.models_path, self.datasets_path, self.metrics_path, self.artifacts_path]:
            path.mkdir(exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("🏛️ Model Registry geïnitialiseerd")

    def register_dataset(self, data: pd.DataFrame, dataset_id: str, 
                        version: str = None, transformations: List[str] = None) -> DatasetVersion:
        """Register een nieuwe dataset versie"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Generate content hash
        data_hash = self._hash_dataframe(data)
        
        # Data quality assessment
        completeness = self._assess_completeness(data)
        consistency = self._assess_consistency(data)
        validity = self._assess_validity(data)
        
        # Create dataset version
        dataset_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            hash=data_hash,
            created_at=datetime.now(),
            size_rows=len(data),
            size_bytes=data.memory_usage(deep=True).sum(),
            completeness_score=completeness,
            consistency_score=consistency,
            validity_score=validity,
            feature_count=len(data.columns),
            feature_names=list(data.columns),
            feature_types={col: str(data[col].dtype) for col in data.columns},
            transformations=transformations or []
        )
        
        # Save dataset metadata
        self._save_dataset_metadata(dataset_version)
        
        # Save dataset artifact
        dataset_file = self.datasets_path / f"{dataset_id}_v{version}.parquet"
        data.to_parquet(dataset_file)
        
        self.logger.info(f"📊 Dataset {dataset_id} v{version} geregistreerd (hash: {data_hash[:8]})")
        return dataset_version

    def register_model(self, model: Any, model_id: str, name: str, algorithm: str,
                      training_data_hash: str, hyperparameters: Dict[str, Any],
                      validation_metrics: Dict[str, float], test_metrics: Dict[str, float],
                      version: str = None, description: str = "", tags: List[str] = None) -> ModelMetadata:
        """Register een nieuw model"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            algorithm=algorithm,
            status=ModelStatus.DEVELOPMENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            training_dataset_hash=training_data_hash,
            feature_set_version="1.0",
            hyperparameters=hyperparameters,
            training_duration_seconds=0.0,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            production_metrics={},
            description=description,
            tags=tags or [],
            author="system"
        )
        
        # Save model metadata
        self._save_model_metadata(metadata)
        
        # Save model artifact
        model_file = self.artifacts_path / f"{model_id}_v{version}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        self.logger.info(f"🤖 Model {model_id} v{version} geregistreerd")
        return metadata

    def promote_to_canary(self, model_id: str, version: str, 
                         canary_percentage: float = 1.0) -> bool:
        """Promote model to canary deployment"""
        
        if canary_percentage > 1.0:
            self.logger.error("❌ Canary percentage cannot exceed 1.0% risk budget")
            return False
            
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            self.logger.error(f"❌ Model {model_id} v{version} not found")
            return False
            
        # Update metadata
        metadata.status = ModelStatus.CANARY
        metadata.deployment_date = datetime.now()
        metadata.canary_percentage = canary_percentage
        metadata.updated_at = datetime.now()
        
        self._save_model_metadata(metadata)
        
        self.logger.info(f"🚀 Model {model_id} v{version} promoted to canary ({canary_percentage}%)")
        return True

    def promote_to_production(self, model_id: str, version: str) -> bool:
        """Promote canary model to full production"""
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata or metadata.status != ModelStatus.CANARY:
            self.logger.error(f"❌ Model {model_id} v{version} is not in canary status")
            return False
            
        # Check canary performance
        if not self._validate_canary_performance(metadata):
            self.logger.error(f"❌ Canary performance validation failed")
            return False
            
        # Promote to production
        metadata.status = ModelStatus.PRODUCTION
        metadata.canary_percentage = 100.0
        metadata.updated_at = datetime.now()
        
        self._save_model_metadata(metadata)
        
        self.logger.info(f"🎯 Model {model_id} v{version} promoted to production")
        return True

    def rollback_model(self, model_id: str, target_version: str = None) -> bool:
        """Rollback to previous stable version"""
        
        if target_version is None:
            # Find last stable production version
            versions = self.list_model_versions(model_id)
            production_versions = [v for v in versions if v.status == ModelStatus.PRODUCTION]
            
            if not production_versions:
                self.logger.error(f"❌ No stable production version found for {model_id}")
                return False
                
            target_version = max(production_versions, key=lambda x: x.created_at).version
            
        target_metadata = self.get_model_metadata(model_id, target_version)
        if not target_metadata:
            self.logger.error(f"❌ Target version {target_version} not found")
            return False
            
        # Create rollback entry
        rollback_metadata = ModelMetadata(
            model_id=model_id,
            version=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"{target_metadata.name}_rollback",
            algorithm=target_metadata.algorithm,
            status=ModelStatus.PRODUCTION,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            training_dataset_hash=target_metadata.training_dataset_hash,
            feature_set_version=target_metadata.feature_set_version,
            hyperparameters=target_metadata.hyperparameters,
            training_duration_seconds=0.0,
            validation_metrics=target_metadata.validation_metrics,
            test_metrics=target_metadata.test_metrics,
            production_metrics=target_metadata.production_metrics,
            description=f"Rollback to {target_version}",
            tags=target_metadata.tags + ["rollback"],
            author="system"
        )
        
        self._save_model_metadata(rollback_metadata)
        
        self.logger.warning(f"🔄 Model {model_id} rolled back to {target_version}")
        return True

    def check_data_drift(self, model_id: str, version: str, 
                        new_data: pd.DataFrame, reference_data: pd.DataFrame = None) -> DriftStatus:
        """Check for data drift tussen datasets"""
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            return DriftStatus.CRITICAL
            
        if reference_data is None:
            # Load reference dataset
            reference_data = self._load_reference_dataset(metadata.training_dataset_hash)
            
        if reference_data is None:
            self.logger.warning("⚠️ No reference dataset found")
            return DriftStatus.MEDIUM
            
        # Calculate drift scores
        drift_scores = self._calculate_drift_scores(reference_data, new_data)
        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        
        # Determine drift status
        if max_drift < 0.1:
            drift_status = DriftStatus.NONE
        elif max_drift < 0.3:
            drift_status = DriftStatus.LOW
        elif max_drift < 0.5:
            drift_status = DriftStatus.MEDIUM
        elif max_drift < 0.8:
            drift_status = DriftStatus.HIGH
        else:
            drift_status = DriftStatus.CRITICAL
            
        # Update metadata
        metadata.drift_score = max_drift
        metadata.drift_status = drift_status
        metadata.last_drift_check = datetime.now()
        metadata.updated_at = datetime.now()
        
        self._save_model_metadata(metadata)
        
        self.logger.info(f"🔍 Drift check: {drift_status.value} (score: {max_drift:.3f})")
        return drift_status

    def update_production_metrics(self, model_id: str, version: str, 
                                 metrics: Dict[str, float]) -> bool:
        """Update production performance metrics"""
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            return False
            
        metadata.production_metrics.update(metrics)
        metadata.updated_at = datetime.now()
        
        # Check if performance degradation triggers rollback
        if self._should_trigger_rollback(metadata):
            self.logger.warning(f"⚠️ Performance degradation detected - triggering rollback")
            self.rollback_model(model_id)
            
        self._save_model_metadata(metadata)
        return True

    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        
        metadata_file = self.models_path / f"{model_id}_v{version}.json"
        if not metadata_file.exists():
            return None
            
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            
        # Convert datetime strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('deployment_date'):
            data['deployment_date'] = datetime.fromisoformat(data['deployment_date'])
        if data.get('last_drift_check'):
            data['last_drift_check'] = datetime.fromisoformat(data['last_drift_check'])
            
        # Convert enums
        data['status'] = ModelStatus(data['status'])
        data['drift_status'] = DriftStatus(data['drift_status'])
        
        return ModelMetadata(**data)

    def list_model_versions(self, model_id: str) -> List[ModelMetadata]:
        """List all versions for a model"""
        
        versions = []
        for file_path in self.models_path.glob(f"{model_id}_v*.json"):
            version = file_path.stem.split('_v')[-1]
            metadata = self.get_model_metadata(model_id, version)
            if metadata:
                versions.append(metadata)
                
        return sorted(versions, key=lambda x: x.created_at, reverse=True)

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get registry overview"""
        
        models = list(self.models_path.glob("*.json"))
        datasets = list(self.datasets_path.glob("*.parquet"))
        
        status_counts = {}
        for model_file in models:
            model_id, version = model_file.stem.split('_v')
            metadata = self.get_model_metadata(model_id, version)
            if metadata:
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
        return {
            'total_models': len(models),
            'total_datasets': len(datasets),
            'models_by_status': status_counts,
            'registry_size_mb': sum(f.stat().st_size for f in self.registry_path.rglob('*') if f.is_file()) / 1024 / 1024
        }

    # Helper methods
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate content hash for dataframe"""
        content = df.to_string().encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness (% non-null values)"""
        return (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))

    def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency (simplified)"""
        # Simple heuristic: variance in data types and value ranges
        consistency_scores = []
        for col in df.select_dtypes(include=[np.number]).columns:
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            consistency_scores.append(1 / (1 + cv))  # Lower CV = higher consistency
        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _assess_validity(self, df: pd.DataFrame) -> float:
        """Assess data validity (simplified)"""
        # Simple heuristic: check for obvious invalid values
        invalid_count = 0
        total_values = len(df) * len(df.columns)
        
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                # Check for empty strings
                invalid_count += (df[col] == '').sum()
            else:
                # Check for infinite values
                invalid_count += np.isinf(df[col]).sum()
                
        return 1 - (invalid_count / total_values)

    def _calculate_drift_scores(self, reference: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate drift scores per feature"""
        
        drift_scores = {}
        common_cols = set(reference.columns) & set(new_data.columns)
        
        for col in common_cols:
            if reference[col].dtype in ['int64', 'float64']:
                # Numerical drift: KL divergence approximation
                ref_mean, ref_std = reference[col].mean(), reference[col].std()
                new_mean, new_std = new_data[col].mean(), new_data[col].std()
                
                if ref_std > 0 and new_std > 0:
                    # Simplified drift score based on mean/std changes
                    mean_diff = abs(new_mean - ref_mean) / ref_std
                    std_ratio = max(new_std/ref_std, ref_std/new_std) - 1
                    drift_scores[col] = min(1.0, (mean_diff + std_ratio) / 2)
                else:
                    drift_scores[col] = 0.0
            else:
                # Categorical drift: distribution comparison
                ref_dist = reference[col].value_counts(normalize=True)
                new_dist = new_data[col].value_counts(normalize=True)
                
                # Calculate total variation distance
                all_categories = set(ref_dist.index) | set(new_dist.index)
                tv_distance = 0.5 * sum(abs(ref_dist.get(cat, 0) - new_dist.get(cat, 0)) 
                                      for cat in all_categories)
                drift_scores[col] = tv_distance
                
        return drift_scores

    def _validate_canary_performance(self, metadata: ModelMetadata) -> bool:
        """Validate canary performance before production promotion"""
        
        if not metadata.production_metrics:
            return False
            
        # Check if production metrics are better than or equal to test metrics
        test_accuracy = metadata.test_metrics.get('accuracy', 0)
        prod_accuracy = metadata.production_metrics.get('accuracy', 0)
        
        return prod_accuracy >= test_accuracy * 0.95  # Allow 5% degradation

    def _should_trigger_rollback(self, metadata: ModelMetadata) -> bool:
        """Check if performance degradation should trigger rollback"""
        
        if not metadata.production_metrics or not metadata.test_metrics:
            return False
            
        # Check significant performance drop
        test_accuracy = metadata.test_metrics.get('accuracy', 0)
        prod_accuracy = metadata.production_metrics.get('accuracy', 0)
        
        return prod_accuracy < test_accuracy * 0.8  # 20% degradation triggers rollback

    def _load_reference_dataset(self, dataset_hash: str) -> Optional[pd.DataFrame]:
        """Load reference dataset by hash"""
        
        for dataset_file in self.datasets_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(dataset_file)
                if self._hash_dataframe(df) == dataset_hash:
                    return df
            except Exception:
                continue
                
        return None

    def _save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata to JSON"""
        
        metadata_dict = asdict(metadata)
        
        # Convert datetime objects to ISO strings
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_dict['updated_at'] = metadata.updated_at.isoformat()
        if metadata.deployment_date:
            metadata_dict['deployment_date'] = metadata.deployment_date.isoformat()
        if metadata.last_drift_check:
            metadata_dict['last_drift_check'] = metadata.last_drift_check.isoformat()
            
        # Convert enums to strings
        metadata_dict['status'] = metadata.status.value
        metadata_dict['drift_status'] = metadata.drift_status.value
        
        metadata_file = self.models_path / f"{metadata.model_id}_v{metadata.version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

    def _save_dataset_metadata(self, dataset_version: DatasetVersion):
        """Save dataset metadata to JSON"""
        
        metadata_dict = asdict(dataset_version)
        metadata_dict['created_at'] = dataset_version.created_at.isoformat()
        
        # Convert numpy integers to Python integers for JSON serialization
        if isinstance(metadata_dict.get('size_rows'), np.integer):
            metadata_dict['size_rows'] = int(metadata_dict['size_rows'])
        if isinstance(metadata_dict.get('size_bytes'), np.integer):
            metadata_dict['size_bytes'] = int(metadata_dict['size_bytes'])
        if isinstance(metadata_dict.get('feature_count'), np.integer):
            metadata_dict['feature_count'] = int(metadata_dict['feature_count'])
            
        metadata_file = self.datasets_path / f"{dataset_version.dataset_id}_v{dataset_version.version}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)


# Singleton instance
_model_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get shared model registry instance"""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry()
    return _model_registry_instance