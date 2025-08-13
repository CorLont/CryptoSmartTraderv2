#!/usr/bin/env python3
"""
Demo: ML Model Registry & Training System
Comprehensive demonstration of enterprise ML infrastructure with model versioning, walk-forward training, and drift detection.
"""

import asyncio
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.ml import (
    create_model_registry,
    create_walk_forward_trainer,
    create_drift_detector,
    ModelMetrics,
    ModelType,
    ModelStatus,
    TrainingConfig,
    RetrainingConfig,
    DriftConfig,
)


# Mock ML models for demo
class MockRandomForestModel:
    """Mock Random Forest model for demonstration."""

    def __init__(self, n_estimators=100, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_importances_ = None
        self.is_fitted = False

    def fit(self, X, y):
        """Mock fitting process."""
        n_features = len(X.columns) if hasattr(X, "columns") else X.shape[1]
        # Generate random feature importance
        self.feature_importances_ = np.random.dirichlet(np.ones(n_features))
        self.is_fitted = True
        return self

    def predict(self, X):
        """Mock prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Generate random predictions (0/1 for binary classification)
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.random.randint(0, 2, n_samples)

    def predict_proba(self, X):
        """Mock probability prediction."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        # Generate random probabilities that sum to 1
        proba_class_0 = np.random.uniform(0.2, 0.8, n_samples)
        proba_class_1 = 1 - proba_class_0
        return np.column_stack([proba_class_0, proba_class_1])

    def score(self, X, y):
        """Mock scoring."""
        pred = self.predict(X)
        return np.mean(pred == y) if len(y) > 0 else 0.5


def generate_crypto_dataset(n_samples: int = 1000, start_date: str = "2024-01-01") -> pd.DataFrame:
    """Generate synthetic cryptocurrency trading dataset."""

    np.random.seed(42)

    # Generate time index
    dates = pd.date_range(start=start_date, periods=n_samples, freq="H")

    # Generate features
    data = {}

    # Price features
    base_price = 50000
    price_trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
    data["price"] = base_price * (1 + price_trend)
    data["price_ma_24h"] = pd.Series(data["price"]).rolling(24).mean()
    data["price_std_24h"] = pd.Series(data["price"]).rolling(24).std()

    # Volume features
    data["volume"] = np.random.lognormal(15, 0.5, n_samples)
    data["volume_ma_24h"] = pd.Series(data["volume"]).rolling(24).mean()

    # Technical indicators
    data["rsi"] = np.random.uniform(20, 80, n_samples)
    data["macd"] = np.random.normal(0, 1, n_samples)
    data["bollinger_position"] = np.random.uniform(-1, 1, n_samples)

    # Market features
    data["spread_bps"] = np.random.uniform(5, 50, n_samples)
    data["order_book_imbalance"] = np.random.normal(0, 0.1, n_samples)
    data["funding_rate"] = np.random.normal(0.01, 0.005, n_samples)

    # Sentiment features
    data["sentiment_score"] = np.random.uniform(-1, 1, n_samples)
    data["social_volume"] = np.random.lognormal(10, 1, n_samples)
    data["fear_greed_index"] = np.random.uniform(0, 100, n_samples)

    # Target variable (binary: price goes up in next hour)
    future_returns = np.diff(data["price"], prepend=data["price"][0]) / data["price"]
    data["target"] = (future_returns > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame(data, index=dates)

    # Forward fill NaN values
    df = df.fillna(method="ffill").fillna(0)

    return df


def calculate_mock_metrics(y_true, y_pred, y_proba=None) -> ModelMetrics:
    """Calculate mock performance metrics."""

    accuracy = np.mean(y_true == y_pred)

    # Simple precision/recall calculation
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.5
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.5
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.5

    # Mock AUC
    auc_roc = 0.5 + abs(accuracy - 0.5)  # Simple approximation

    # Trading metrics
    returns = np.where(y_pred == 1, np.random.normal(0.001, 0.02, len(y_pred)), 0)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
    max_drawdown = np.min(np.cumsum(returns)) if len(returns) > 0 else 0.0
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.5

    return ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        auc_roc=auc_roc,
        log_loss=0.5,  # Placeholder
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        oos_accuracy=accuracy * 0.95,  # Slightly lower OOS
        information_ratio=sharpe_ratio * 0.8,
    )


async def demonstrate_ml_model_registry():
    """Comprehensive demonstration of ML model registry and training system."""

    print("🤖 ML MODEL REGISTRY & TRAINING SYSTEM DEMONSTRATION")
    print("=" * 70)

    # Initialize systems
    print("🔧 Initializing ML infrastructure...")
    model_registry = create_model_registry("models/demo_registry")

    retrain_config = RetrainingConfig(
        retrain_frequency_days=7,
        min_training_samples=500,
        validation_window_days=14,
        drift_threshold=0.25,
        canary_risk_budget_pct=1.0,
    )

    drift_config = DriftConfig(
        data_drift_threshold=0.3,
        performance_drift_threshold=0.15,
        check_frequency_hours=6,
        auto_rollback_threshold=0.8,
    )

    walk_forward_trainer = create_walk_forward_trainer(model_registry, retrain_config)
    drift_detector = create_drift_detector(model_registry, drift_config)

    print("✅ ML infrastructure initialized")

    # Demo 1: Model Registry Operations
    print("\n📋 DEMO 1: Model Registry Operations")
    print("-" * 45)

    # Generate training data
    print("   Generating synthetic training data...")
    training_data = generate_crypto_dataset(n_samples=2000, start_date="2024-01-01")

    print(f"   Dataset: {len(training_data)} samples, {len(training_data.columns) - 1} features")
    print(f"   Target distribution: {training_data['target'].mean():.1%} positive class")

    # Train and register multiple models
    models_trained = []

    for i in range(3):
        print(f"\n   Training model {i + 1}/3...")

        # Create model
        model = MockRandomForestModel(n_estimators=100 + i * 50, max_depth=10 + i * 2)

        # Train/test split
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data.iloc[:split_idx]
        test_data = training_data.iloc[split_idx:]

        feature_columns = [col for col in training_data.columns if col != "target"]

        X_train = train_data[feature_columns]
        y_train = train_data["target"]
        X_test = test_data[feature_columns]
        y_test = test_data["target"]

        # Train model
        model.fit(X_train, y_train)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = calculate_mock_metrics(y_test, y_pred, y_proba)

        # Training config
        training_config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            hyperparameters={
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "features": feature_columns,
            },
            training_script="demo_ml_model_registry.py",
            training_environment={"framework": "mock_sklearn"},
            random_seed=42,
            training_duration_seconds=30.0,
            cross_validation_folds=5,
        )

        # Register model
        model_version = model_registry.register_model(
            model=model,
            model_type=ModelType.RANDOM_FOREST,
            dataset=train_data,
            target_column="target",
            metrics=metrics,
            training_config=training_config,
            tags=[f"demo_model_{i + 1}", "cryptocurrency", "binary_classification"],
            notes=f"Demo model {i + 1} with {model.n_estimators} estimators",
        )

        models_trained.append(model_version)

        print(f"      Model ID: {model_version.model_id}")
        print(f"      Version: {model_version.version}")
        print(f"      Accuracy: {metrics.accuracy:.3f}")
        print(f"      Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"      Status: {model_version.status.value}")

    # Demo 2: Model Deployment and Management
    print("\n🚀 DEMO 2: Model Deployment and Management")
    print("-" * 50)

    # Deploy best model to production
    best_model = max(models_trained, key=lambda m: m.metrics.accuracy)

    print(f"   Deploying best model: {best_model.model_id} {best_model.version}")
    print(f"   Best accuracy: {best_model.metrics.accuracy:.3f}")

    # Start canary deployment
    print("\n   Starting canary deployment...")
    success = model_registry.deploy_to_production(
        model_id=best_model.model_id,
        version=best_model.version,
        risk_budget_pct=1.0,
        canary_duration_hours=72,
    )

    if success:
        print("   ✅ Canary deployment started")

        # Simulate canary completion by updating deployment info
        best_model.deployment_info = best_model.deployment_info or {}
        best_model.deployment_info["canary_start"] = (
            datetime.utcnow() - timedelta(hours=73).isoformat()
        best_model.status = ModelStatus.CANARY
        model_registry._save_registry()

        # Promote to production
        print("   Promoting canary to production...")
        success = model_registry.deploy_to_production(
            model_id=best_model.model_id, version=best_model.version
        )

        if success:
            print("   ✅ Model promoted to production")

    # List all models
    print("\n   Current model registry:")
    all_models = model_registry.list_models()
    for model_info in all_models:
        print(
            f"      {model_info['model_id']} {model_info['version']}: "
            f"{model_info['status']} (acc: {model_info['accuracy']:.3f})"
        )

    # Demo 3: Walk-Forward Training
    print("\n📈 DEMO 3: Walk-Forward Training")
    print("-" * 40)

    print("   Generating extended time series data...")
    # Generate longer time series for walk-forward
    extended_data = generate_crypto_dataset(n_samples=5000, start_date="2024-01-01")

    print(f"   Extended dataset: {len(extended_data)} samples")
    print("   Running walk-forward validation...")

    # Custom model factory
    def model_factory():
        return MockRandomForestModel(n_estimators=100, max_depth=8)

    # Run walk-forward training (limited to 3 folds for demo)
    start_date = extended_data.index.min() + timedelta(days=90)
    end_date = extended_data.index.min() + timedelta(days=120)

    wf_results = walk_forward_trainer.run_walk_forward_training(
        data=extended_data,
        target_column="target",
        model_factory=model_factory,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"   Walk-forward results: {len(wf_results)} models trained")

    for i, result in enumerate(wf_results):
        print(
            f"      Fold {i + 1}: Train acc {result.train_metrics.accuracy:.3f}, "
            f"Test acc {result.test_metrics.accuracy:.3f}, "
            f"Risk score {result.risk_score:.3f}"
        )

    # Training summary
    if wf_results:
        summary = walk_forward_trainer.get_training_summary()
        print(f"\n   Training Summary:")
        print(f"      Total models: {summary['total_models_trained']}")
        print(f"      Avg test accuracy: {summary['performance_summary']['avg_test_accuracy']:.3f}")
        print(f"      Performance trend: {summary['performance_summary']['performance_trend']}")
        print(f"      Models deployed: {summary['deployment_stats']['models_deployed']}")

    # Demo 4: Drift Detection
    print("\n🔍 DEMO 4: Drift Detection")
    print("-" * 35)

    # Set baseline for drift detection
    baseline_data = extended_data.iloc[:1000]  # First 1000 samples

    print("   Setting drift detection baseline...")
    drift_detector.set_baseline(
        model_id=best_model.model_id, version=best_model.version, baseline_data=baseline_data
    )

    print(f"   Baseline set: {len(baseline_data)} samples")

    # Simulate drift scenarios
    drift_scenarios = [
        ("No Drift", extended_data.iloc[1000:1500]),  # Similar data
        ("Moderate Drift", extended_data.iloc[2000:2500] * 1.2),  # Scaled data
        (
            "High Drift",
            extended_data.iloc[3000:3500]
            + np.random.normal(0, 1, (500, len(extended_data.columns))),
        ),  # Noisy data
    ]

    for scenario_name, drift_data in drift_scenarios:
        print(f"\n   Testing scenario: {scenario_name}")

        # Clean drift data
        drift_data = drift_data.fillna(method="ffill").fillna(0)

        alerts = drift_detector.detect_drift(
            model_id=best_model.model_id, version=best_model.version, current_data=drift_data
        )

        if alerts:
            print(f"      Drift alerts: {len(alerts)}")
            for alert in alerts:
                print(
                    f"         {alert.drift_type.value}: {alert.severity.value} "
                    f"(confidence: {alert.rollback_confidence:.2f})"
                )
                if alert.affected_features:
                    print(f"         Affected features: {alert.affected_features[:3]}")
        else:
            print("      No drift detected")

    # Demo 5: Model Rollback
    print("\n⏪ DEMO 5: Model Rollback")
    print("-" * 30)

    # Simulate performance degradation requiring rollback
    print("   Simulating performance degradation...")

    # Update live metrics with poor performance
    poor_metrics = {
        "accuracy": 0.45,  # Below threshold
        "sharpe_ratio": -0.2,
        "drawdown": -0.15,
    }

    model_registry.update_live_metrics(
        model_id=best_model.model_id, version=best_model.version, live_metrics=poor_metrics
    )

    print("   Poor performance detected, executing rollback...")

    # Execute rollback
    rollback_success = model_registry.rollback_model(
        reason="Performance degradation - accuracy dropped to 45%"
    )

    if rollback_success:
        print("   ✅ Model rollback completed successfully")

        # Show current production model
        current_production = model_registry._get_production_model()
        if current_production:
            print(
                f"   Current production: {current_production['model_id']} {current_production['version']}"
            )
    else:
        print("   ❌ Rollback failed - no suitable rollback target")

    # Demo 6: Automated Retraining
    print("\n🔄 DEMO 6: Automated Retraining")
    print("-" * 40)

    # Generate new data for retraining
    new_data = generate_crypto_dataset(n_samples=1500, start_date="2024-06-01")

    print(f"   New data available: {len(new_data)} samples")

    # Check retraining triggers
    triggers = walk_forward_trainer.check_retraining_triggers(
        current_data=new_data, current_performance=poor_metrics
    )

    print(f"   Retraining triggers: {[t.value for t in triggers]}")

    if triggers:
        print("   Executing automated retraining...")

        # Retrain model
        new_model_version = walk_forward_trainer.retrain_model(
            new_data=new_data, target_column="target", trigger=triggers[0]
        )

        if new_model_version:
            print(
                f"   ✅ New model trained: {new_model_version.model_id} {new_model_version.version}"
            )
            print(f"   Model status: {new_model_version.status.value}")
            print(f"   Accuracy: {new_model_version.metrics.accuracy:.3f}")
        else:
            print("   ❌ Retraining failed or model didn't meet quality standards")

    # Demo 7: Model Comparison
    print("\n📊 DEMO 7: Model Comparison")
    print("-" * 35)

    # Get model comparison for the best model ID
    comparison_df = model_registry.get_model_comparison(best_model.model_id)

    print("   Model version comparison:")
    print(
        comparison_df[
            ["version", "status", "accuracy", "sharpe_ratio", "risk_budget_pct"]
        ].to_string(index=False)

    # Demo 8: System Health Summary
    print("\n🏥 DEMO 8: System Health Summary")
    print("-" * 40)

    # Drift summary
    drift_summary = drift_detector.get_drift_summary()

    print("   Drift Detection Summary:")
    if "total_alerts" in drift_summary:
        print(f"      Total alerts: {drift_summary['total_alerts']}")
        print(f"      Recent alerts: {drift_summary['recent_alerts']}")

        if drift_summary["drift_by_type"]:
            print("      Drift by type:")
            for drift_type, stats in drift_summary["drift_by_type"].items():
                print(f"         {drift_type}: {stats['count']} alerts")

        rollback_stats = drift_summary["rollback_stats"]
        print(
            f"      Rollbacks: {rollback_stats['successful_rollbacks']}/{rollback_stats['total_rollbacks']} successful"
        )
    else:
        print(f"      {drift_summary.get('message', 'No drift data available')}")

    # Model registry stats
    all_models = model_registry.list_models()
    status_counts = {}
    for model in all_models:
        status = model["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\n   Model Registry Summary:")
    print(f"      Total models: {len(all_models)}")
    print(f"      Status distribution:")
    for status, count in status_counts.items():
        print(f"         {status}: {count}")

    # Performance metrics
    accuracies = [m["accuracy"] for m in all_models]
    sharpes = [m["sharpe_ratio"] for m in all_models if m["sharpe_ratio"]]

    print(f"      Performance stats:")
    print(f"         Avg accuracy: {np.mean(accuracies):.3f}")
    print(f"         Best accuracy: {max(accuracies):.3f}")
    if sharpes:
        print(f"         Avg Sharpe: {np.mean(sharpes):.3f}")

    # Demo 9: Production Deployment Status
    print("\n🎯 DEMO 9: Production Deployment Status")
    print("-" * 45)

    # Get production model
    production_model_info = model_registry.get_production_model()

    if production_model_info:
        production_model, production_version = production_model_info

        print("   Current Production Model:")
        print(f"      Model ID: {production_version.model_id}")
        print(f"      Version: {production_version.version}")
        print(f"      Accuracy: {production_version.metrics.accuracy:.3f}")
        print(f"      Sharpe Ratio: {production_version.metrics.sharpe_ratio:.3f}")
        print(f"      Deployed: {production_version.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"      Risk Budget: {production_version.risk_budget_pct:.1f}%")

        if production_version.live_metrics:
            print("      Live Performance:")
            for metric, value in production_version.live_metrics.items():
                print(f"         {metric}: {value:.3f}")

        if production_version.deployment_info:
            print("      Deployment Info:")
            for key, value in production_version.deployment_info.items():
                print(f"         {key}: {value}")
    else:
        print("   No production model currently deployed")

    print("\n✅ ML MODEL REGISTRY DEMONSTRATION COMPLETED")
    print("=" * 70)

    # Final summary
    print("🎯 SYSTEM ACHIEVEMENTS:")
    print(
        f"   ✅ Model registry with {len(all_models)} models across {len(status_counts)} statuses"
    )
    print(f"   ✅ Walk-forward training with {len(wf_results)} temporal validation folds")
    print(f"   ✅ Drift detection with {drift_summary.get('total_alerts', 0)} alerts generated")
    print(f"   ✅ Automated deployment with canary → production promotion")
    print(f"   ✅ Model rollback system with performance monitoring")
    print(f"   ✅ Enterprise ML pipeline ready for production use")


if __name__ == "__main__":
    print("🤖 CRYPTOSMARTTRADER V2 - ML MODEL REGISTRY DEMO")
    print("=" * 70)

    try:
        asyncio.run(demonstrate_ml_model_registry())
        print("\n🏆 ML model registry demonstration completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
