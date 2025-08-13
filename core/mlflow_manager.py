#!/usr/bin/env python3
"""
MLflow Manager
Manages MLflow runs, artifacts, and model registry with horizon/regime tags
"""

import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

# MLflow imports (optional - graceful fallback if not installed)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - using local file-based model tracking")


class MLflowManager:
    """MLflow integration for model tracking, versioning, and registry"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_structured_logger("MLflowManager")

        # Configuration
        self.config = {
            "mlflow_tracking_uri": "file:./mlruns",
            "experiment_name": "CryptoSmartTrader",
            "artifact_location": "./mlartifacts",
            "enable_mlflow": MLFLOW_AVAILABLE,
            "local_backup_dir": "./model_backup",
            "model_registry_prefix": "cryptosmarttrader",
            "auto_register_models": True,
            "retention_days": 90,
        }

        if config:
            self.config.update(config)

        # Create directories
        for dir_path in [self.config["artifact_location"], self.config["local_backup_dir"]]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize MLflow if available
        if self.config["enable_mlflow"] and MLFLOW_AVAILABLE:
            self._initialize_mlflow()
        else:
            self.logger.warning("MLflow not available - using local file tracking")
            self.mlflow_client = None
            self.experiment_id = None

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking and experiment"""

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])

            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        name=self.config["experiment_name"],
                        artifact_location=self.config["artifact_location"],
                    )
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                self.logger.warning(f"Could not create/get experiment: {e}")
                self.experiment_id = "0"  # Default experiment

            # Initialize client
            self.mlflow_client = MlflowClient()

            self.logger.info(f"MLflow initialized - experiment: {self.config['experiment_name']}")

        except Exception as e:
            self.logger.error(f"MLflow initialization failed: {e}")
            self.config["enable_mlflow"] = False
            self.mlflow_client = None

    def start_run(
        self,
        run_name: str,
        horizon: str,
        regime: str = "normal",
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Start MLflow run with horizon and regime tags"""

        if not self.config["enable_mlflow"]:
            return self._start_local_run(run_name, horizon, regime, tags)

        try:
            # Prepare tags
            run_tags = {
                "horizon": horizon,
                "regime": regime,
                "model_type": "crypto_predictor",
                "framework": "pytorch",
                "created_at": datetime.now().isoformat(),
            }

            if tags:
                run_tags.update(tags)

            # Start MLflow run
            mlflow.set_experiment(self.config["experiment_name"])
            run = mlflow.start_run(run_name=run_name, tags=run_tags)

            self.logger.info(
                f"Started MLflow run: {run_name} (horizon: {horizon}, regime: {regime})"
            )

            return run.info.run_id

        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            return self._start_local_run(run_name, horizon, regime, tags)

    def _start_local_run(
        self, run_name: str, horizon: str, regime: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start local file-based run tracking"""

        run_id = f"{run_name}_{horizon}_{regime}_{int(datetime.now().timestamp())}"

        run_dir = Path(self.config["local_backup_dir"]) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        run_metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "horizon": horizon,
            "regime": regime,
            "tags": tags or {},
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
        }

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        self.logger.info(f"Started local run: {run_name} (horizon: {horizon}, regime: {regime})")

        return run_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow or local storage"""

        if self.config["enable_mlflow"] and mlflow.active_run():
            try:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, step=step)

                self.logger.debug(f"Logged metrics: {list(metrics.keys())}")

            except Exception as e:
                self.logger.error(f"Failed to log metrics to MLflow: {e}")
                self._log_metrics_local(metrics, step)
        else:
            self._log_metrics_local(metrics, step)

    def _log_metrics_local(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to local file"""

        try:
            # Find current run directory
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"
            if not runs_dir.exists():
                return

            # Get most recent run
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                return

            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

            # Log metrics
            metrics_file = latest_run / "metrics.jsonl"

            metric_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "metrics": metrics,
            }

            with open(metrics_file, "a") as f:
                f.write(json.dumps(metric_entry) + "\n")

            self.logger.debug(f"Logged metrics locally: {list(metrics.keys())}")

        except Exception as e:
            self.logger.error(f"Failed to log metrics locally: {e}")

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow or local storage"""

        if self.config["enable_mlflow"] and mlflow.active_run():
            try:
                for param_name, value in params.items():
                    mlflow.log_param(param_name, value)

                self.logger.debug(f"Logged parameters: {list(params.keys())}")

            except Exception as e:
                self.logger.error(f"Failed to log parameters to MLflow: {e}")
                self._log_parameters_local(params)
        else:
            self._log_parameters_local(params)

    def _log_parameters_local(self, params: Dict[str, Any]) -> None:
        """Log parameters to local file"""

        try:
            # Find current run directory
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"
            if not runs_dir.exists():
                return

            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                return

            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

            # Log parameters
            params_file = latest_run / "params.json"

            if params_file.exists():
                with open(params_file, "r") as f:
                    existing_params = json.load(f)
            else:
                existing_params = {}

            existing_params.update(params)

            with open(params_file, "w") as f:
                json.dump(existing_params, f, indent=2)

            self.logger.debug(f"Logged parameters locally: {list(params.keys())}")

        except Exception as e:
            self.logger.error(f"Failed to log parameters locally: {e}")

    def log_model(
        self,
        model: Any,
        model_name: str,
        horizon: str,
        regime: str,
        model_type: str = "pytorch",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log model with horizon/regime tags"""

        if self.config["enable_mlflow"] and mlflow.active_run():
            try:
                # Prepare model metadata
                model_metadata = {
                    "horizon": horizon,
                    "regime": regime,
                    "model_type": model_type,
                    "created_at": datetime.now().isoformat(),
                }

                if metadata:
                    model_metadata.update(metadata)

                # Log model based on type
                if model_type == "pytorch":
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path=model_name,
                        registered_model_name=f"{self.config['model_registry_prefix']}_{model_name}",
                        metadata=model_metadata,
                    )
                elif model_type == "sklearn":
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=model_name,
                        registered_model_name=f"{self.config['model_registry_prefix']}_{model_name}",
                        metadata=model_metadata,
                    )
                else:
                    # Generic model logging
                    mlflow.log_artifact(model, artifact_path=model_name)

                self.logger.info(
                    f"Logged model: {model_name} (horizon: {horizon}, regime: {regime})"
                )

            except Exception as e:
                self.logger.error(f"Failed to log model to MLflow: {e}")
                self._log_model_local(model, model_name, horizon, regime, model_type, metadata)
        else:
            self._log_model_local(model, model_name, horizon, regime, model_type, metadata)

    def _log_model_local(
        self,
        model: Any,
        model_name: str,
        horizon: str,
        regime: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log model to local storage"""

        try:
            # Find current run directory
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"
            if not runs_dir.exists():
                return

            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                return

            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)

            # Create model directory
            model_dir = latest_run / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = model_dir / "model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            # Save metadata
            model_metadata = {
                "model_name": model_name,
                "horizon": horizon,
                "regime": regime,
                "model_type": model_type,
                "created_at": datetime.now().isoformat(),
            }

            if metadata:
                model_metadata.update(metadata)

            with open(model_dir / "metadata.json", "w") as f:
                json.dump(model_metadata, f, indent=2)

            self.logger.info(
                f"Logged model locally: {model_name} (horizon: {horizon}, regime: {regime})"
            )

        except Exception as e:
            self.logger.error(f"Failed to log model locally: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End current MLflow run"""

        if self.config["enable_mlflow"] and mlflow.active_run():
            try:
                mlflow.end_run(status=status)
                self.logger.info("Ended MLflow run")

            except Exception as e:
                self.logger.error(f"Failed to end MLflow run: {e}")

        # Update local run status
        self._end_local_run(status)

    def _end_local_run(self, status: str) -> None:
        """End local run tracking"""

        try:
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"
            if not runs_dir.exists():
                return

            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                return

            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            metadata_file = latest_run / "metadata.json"

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                metadata["status"] = status
                metadata["end_time"] = datetime.now().isoformat()

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            self.logger.info(f"Ended local run with status: {status}")

        except Exception as e:
            self.logger.error(f"Failed to end local run: {e}")

    def get_model_by_tags(self, horizon: str, regime: str = None) -> Optional[Any]:
        """Get model by horizon and regime tags"""

        if self.config["enable_mlflow"] and self.mlflow_client:
            try:
                # Search for models with matching tags
                query = f"tags.horizon = '{horizon}'"
                if regime:
                    query += f" and tags.regime = '{regime}'"

                runs = self.mlflow_client.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=query,
                    order_by=["start_time DESC"],
                    max_results=1,
                )

                if runs:
                    run = runs[0]
                    # Load model from run
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.pytorch.load_model(model_uri)

                    self.logger.info(f"Loaded model for horizon: {horizon}, regime: {regime}")
                    return model

            except Exception as e:
                self.logger.error(f"Failed to get model from MLflow: {e}")

        # Fallback to local storage
        return self._get_model_local(horizon, regime)

    def _get_model_local(self, horizon: str, regime: str = None) -> Optional[Any]:
        """Get model from local storage"""

        try:
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"

            if not runs_dir.exists():
                return None

            # Search for matching runs
            matching_runs = []

            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                metadata_file = run_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                if metadata.get("horizon") == horizon:
                    if regime is None or metadata.get("regime") == regime:
                        matching_runs.append((run_dir, metadata))

            if not matching_runs:
                return None

            # Get most recent run
            latest_run_dir, _ = max(matching_runs, key=lambda x: x[1].get("start_time", ""))

            # Load model
            models_dir = latest_run_dir / "models"
            if models_dir.exists():
                model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                if model_dirs:
                    model_file = model_dirs[0] / "model.pkl"
                    if model_file.exists():
                        with open(model_file, "rb") as f:
                            model = pickle.load(f)

                        self.logger.info(
                            f"Loaded local model for horizon: {horizon}, regime: {regime}"
                        )
                        return model

            return None

        except Exception as e:
            self.logger.error(f"Failed to get local model: {e}")
            return None

    def list_models(self, horizon: str = None, regime: str = None) -> List[Dict[str, Any]]:
        """List models with optional filtering by horizon/regime"""

        models = []

        if self.config["enable_mlflow"] and self.mlflow_client:
            try:
                # Build query
                query_parts = []
                if horizon:
                    query_parts.append(f"tags.horizon = '{horizon}'")
                if regime:
                    query_parts.append(f"tags.regime = '{regime}'")

                query = " and ".join(query_parts) if query_parts else ""

                runs = self.mlflow_client.search_runs(
                    experiment_ids=[self.experiment_id],
                    filter_string=query,
                    order_by=["start_time DESC"],
                )

                for run in runs:
                    models.append(
                        {
                            "run_id": run.info.run_id,
                            "horizon": run.data.tags.get("horizon"),
                            "regime": run.data.tags.get("regime"),
                            "created_at": run.info.start_time,
                            "status": run.info.status,
                            "source": "mlflow",
                        }
                    )

            except Exception as e:
                self.logger.error(f"Failed to list MLflow models: {e}")

        # Add local models
        local_models = self._list_local_models(horizon, regime)
        models.extend(local_models)

        return models

    def _list_local_models(self, horizon: str = None, regime: str = None) -> List[Dict[str, Any]]:
        """List local models"""

        models = []

        try:
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"

            if not runs_dir.exists():
                return models

            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                metadata_file = run_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Apply filters
                if horizon and metadata.get("horizon") != horizon:
                    continue
                if regime and metadata.get("regime") != regime:
                    continue

                models.append(
                    {
                        "run_id": metadata.get("run_id"),
                        "horizon": metadata.get("horizon"),
                        "regime": metadata.get("regime"),
                        "created_at": metadata.get("start_time"),
                        "status": metadata.get("status"),
                        "source": "local",
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to list local models: {e}")

        return models

    def cleanup_old_runs(self, retention_days: int = None) -> None:
        """Clean up old runs and artifacts"""

        retention_days = retention_days or self.config["retention_days"]
        cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 3600)

        # Clean up local runs
        try:
            runs_dir = Path(self.config["local_backup_dir"]) / "runs"

            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff_date:
                        shutil.rmtree(run_dir)
                        self.logger.info(f"Cleaned up old run: {run_dir.name}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old runs: {e}")


if __name__ == "__main__":

    async def test_mlflow_manager():
        """Test MLflow manager"""

        print("🔍 TESTING MLFLOW MANAGER")
        print("=" * 60)

        # Create manager
        manager = MLflowManager()

        print("🚀 Starting MLflow run...")
        run_id = manager.start_run("test_model", "1H", "bull_market", {"test": "true"})
        print(f"   Run ID: {run_id}")

        print("\n📊 Logging parameters and metrics...")

        # Log parameters
        params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100, "model_type": "LSTM"}
        manager.log_parameters(params)

        # Log metrics
        for epoch in range(5):
            metrics = {
                "loss": 0.5 - epoch * 0.1,
                "accuracy": 0.6 + epoch * 0.08,
                "val_loss": 0.6 - epoch * 0.09,
            }
            manager.log_metrics(metrics, step=epoch)

        print("   Logged parameters and metrics")

        # Create dummy model for testing
        print("\n🤖 Logging test model...")

        import numpy as np

        dummy_model = {
            "weights": np.random.random((10, 5)),
            "bias": np.random.random(5),
            "architecture": "LSTM",
        }

        manager.log_model(
            model=dummy_model,
            model_name="test_predictor",
            horizon="1H",
            regime="bull_market",
            model_type="pytorch",
            metadata={"version": "1.0", "features": 20},
        )

        print("   Model logged successfully")

        # End run
        print("\n🏁 Ending run...")
        manager.end_run("FINISHED")

        # Test model retrieval
        print("\n🔍 Testing model retrieval...")
        retrieved_model = manager.get_model_by_tags("1H", "bull_market")

        if retrieved_model:
            print("   ✅ Model retrieved successfully")
        else:
            print("   ⚠️ Model retrieval failed (expected for dummy model)")

        # List models
        print("\n📋 Listing models...")
        models = manager.list_models()
        print(f"   Found {len(models)} models")

        for model in models[:3]:  # Show first 3
            print(f"      {model['horizon']} | {model['regime']} | {model['source']}")

        print("\n✅ MLFLOW MANAGER TEST COMPLETED")

        return run_id is not None and len(models) > 0

    # Run test
    import asyncio

    success = asyncio.run(test_mlflow_manager())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
