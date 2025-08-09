#!/usr/bin/env python3
"""
Baseline RF-ensemble training for production deployment
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_features():
    """Load processed features from data pipeline"""
    features_file = Path("data/processed/features.csv")
    
    if not features_file.exists():
        logger.error("Features file not found - run scrape_all.py first")
        sys.exit(1)
    
    df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Validate required columns
    required_cols = ['coin', 'timestamp', 'price_change_24h', 'volume_24h']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    return df

def prepare_training_data(df):
    """Prepare features and targets for training"""
    # Feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if not col.startswith(('coin', 'timestamp', 'target_'))]
    
    X = df[feature_cols].fillna(0)
    
    # Check if we have targets, if not create synthetic ones
    has_targets = any(f'target_return_{h}' in df.columns for h in ['1h', '24h', '168h', '720h'])
    
    if not has_targets:
        logger.warning("No target columns found - creating synthetic targets for initial training")
        from ml.synthetic_targets import create_synthetic_targets
        df = create_synthetic_targets(df.copy())
    
    # Create targets for multi-horizon prediction
    targets = {}
    
    # Price return targets (regression)
    for horizon in ['1h', '24h', '168h', '720h']:
        target_col = f'target_return_{horizon}'
        if target_col in df.columns:
            targets[f'return_{horizon}'] = df[target_col].fillna(0)
    
    # Direction targets (classification) 
    for horizon in ['1h', '24h', '168h', '720h']:
        direction_col = f'target_direction_{horizon}'
        if direction_col in df.columns:
            targets[f'direction_{horizon}'] = (df[direction_col] > 0).astype(int)
    
    logger.info(f"Prepared {X.shape[1]} features and {len(targets)} targets")
    return X, targets, feature_cols

def train_baseline_models(X, targets, feature_cols):
    """Train RF ensemble for each target"""
    models = {}
    performance_metrics = {}
    
    # Split data
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, X.index, test_size=0.2, random_state=42, shuffle=True
    )
    
    for target_name, y in targets.items():
        logger.info(f"Training model for {target_name}")
        
        y_train = y.iloc[indices_train]
        y_test = y.iloc[indices_test]
        
        # Skip if insufficient positive samples
        if target_name.startswith('direction_'):
            if y_train.sum() < 10:
                logger.warning(f"Insufficient positive samples for {target_name}, skipping")
                continue
        
        try:
            # Choose model type based on target
            if target_name.startswith('direction_'):
                # Classification
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Regression
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            if target_name.startswith('direction_'):
                accuracy = accuracy_score(y_test, y_pred)
                performance_metrics[target_name] = {
                    'accuracy': accuracy,
                    'samples': len(y_test),
                    'positive_rate': y_test.mean()
                }
                logger.info(f"{target_name}: Accuracy = {accuracy:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                performance_metrics[target_name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'samples': len(y_test)
                }
                logger.info(f"{target_name}: RMSE = {np.sqrt(mse):.4f}")
            
            # Store model
            models[target_name] = model
            
        except Exception as e:
            logger.error(f"Failed to train {target_name}: {e}")
            continue
    
    return models, performance_metrics

def save_models(models, performance_metrics, feature_cols):
    """Save trained models and metadata"""
    models_dir = Path("models/baseline")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual models
    for name, model in models.items():
        model_file = models_dir / f"{name}.joblib"
        joblib.dump(model, model_file)
        logger.info(f"Saved {name} model to {model_file}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'feature_columns': feature_cols,
        'models_trained': list(models.keys()),
        'performance_metrics': performance_metrics,
        'model_type': 'RandomForest',
        'n_estimators': 200
    }
    
    metadata_file = models_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")
    return metadata

def main():
    """Main training pipeline"""
    logger.info("Starting baseline RF-ensemble training")
    
    # Load data
    df = load_features()
    
    # Prepare training data
    X, targets, feature_cols = prepare_training_data(df)
    
    if len(targets) == 0:
        logger.error("No valid targets found")
        sys.exit(1)
    
    # Train models
    models, performance_metrics = train_baseline_models(X, targets, feature_cols)
    
    if len(models) == 0:
        logger.error("No models successfully trained")
        sys.exit(1)
    
    # Save models
    metadata = save_models(models, performance_metrics, feature_cols)
    
    # Print summary
    logger.info("=== Training Summary ===")
    logger.info(f"Successfully trained {len(models)} models")
    
    for name, metrics in performance_metrics.items():
        if 'accuracy' in metrics:
            logger.info(f"{name}: {metrics['accuracy']:.3f} accuracy")
        else:
            logger.info(f"{name}: {metrics['rmse']:.4f} RMSE")
    
    logger.info("Baseline RF-ensemble training completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())