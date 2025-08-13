#!/usr/bin/env python3
"""
Production prediction pipeline - predict_all() integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import joblib
import sys
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_baseline_models():
    """Load trained baseline RF models"""
    models_dir = Path("models/baseline")
    metadata_file = models_dir / "metadata.json"

    if not metadata_file.exists():
        logger.error("Baseline models not found - run train_baseline.py first")
        sys.exit(1)

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    models = {}
    for model_name in metadata["models_trained"]:
        model_file = models_dir / f"{model_name}.joblib"
        if model_file.exists():
            models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} model")

    return models, metadata


def load_latest_features():
    """Load latest feature data for prediction"""
    features_file = Path("data/processed/features.csv")

    if not features_file.exists():
        logger.error("Features file not found - run scrape_all.py first")
        sys.exit(1)

    df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(df)} samples for prediction")

    return df


def predict_all(models: Dict, metadata: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Main prediction function - predict all coins across all horizons"""

    # Prepare features
    feature_cols = metadata["feature_columns"]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        # Add missing columns with zeros
        for col in missing_cols:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Make predictions
    predictions_data = []

    for idx, row in df.iterrows():
        coin = row.get("coin", f"COIN_{idx}")
        timestamp = row.get("timestamp", datetime.now().isoformat())

        coin_features = X.iloc[idx : idx + 1]
        coin_predictions = {"coin": coin, "timestamp": timestamp}

        # Predict for each horizon
        for horizon in ["1h", "24h", "168h", "720h"]:
            # Return prediction
            return_model_name = f"return_{horizon}"
            direction_model_name = f"direction_{horizon}"

            predicted_return = 0.0
            predicted_direction = 0
            confidence = 0.0

            # Get return prediction
            if return_model_name in models:
                try:
                    pred_return = models[return_model_name].predict(coin_features)[0]
                    predicted_return = pred_return
                except Exception as e:
                    logger.warning(f"Return prediction failed for {coin} {horizon}: {e}")

            # Get direction prediction with confidence
            if direction_model_name in models:
                try:
                    pred_direction = models[direction_model_name].predict(coin_features)[0]
                    pred_proba = models[direction_model_name].predict_proba(coin_features)[0]
                    predicted_direction = pred_direction
                    confidence = max(pred_proba) * 100  # Convert to percentage
                except Exception as e:
                    logger.warning(f"Direction prediction failed for {coin} {horizon}: {e}")

            # Store prediction
            coin_predictions.update(
                {
                    f"predicted_return_{horizon}": predicted_return,
                    f"predicted_direction_{horizon}": predicted_direction,
                    f"confidence_{horizon}": confidence,
                    f"expected_return_pct": predicted_return * 100
                    if predicted_direction == 1
                    else -abs(predicted_return) * 100,
                }
            )

        predictions_data.append(coin_predictions)

    predictions_df = pd.DataFrame(predictions_data)
    logger.info(f"Generated predictions for {len(predictions_df)} coins")

    return predictions_df


def apply_confidence_gate(
    predictions_df: pd.DataFrame, min_confidence: float = 80.0
) -> pd.DataFrame:
    """Apply 80% confidence gate with backend enforcement"""

    # Import backend enforcement
    sys.path.append(str(Path.cwd()))
    try:
        from core.backend_enforcement import apply_backend_enforcement

        # Use backend enforcement
        filtered_df, gate_result, readiness_result = apply_backend_enforcement(
            predictions_df, min_confidence
        )

        logger.info(
            f"Backend enforcement: {gate_result['filtered_count']}/{gate_result['original_count']} predictions passed"
        )
        logger.info(f"System readiness: {readiness_result['go_no_go']}")

        return filtered_df

    except ImportError:
        logger.warning("Backend enforcement not available, using basic filtering")

        # Fallback to basic filtering
        confidence_cols = [col for col in predictions_df.columns if col.startswith("confidence_")]

        if not confidence_cols:
            logger.warning("No confidence columns found")
            return predictions_df

        # Filter by confidence threshold
        filtered_predictions = []

        for _, row in predictions_df.iterrows():
            # Check if any horizon meets confidence threshold
            max_confidence = max([row.get(col, 0) for col in confidence_cols])

            if max_confidence >= min_confidence:
                filtered_predictions.append(row.to_dict())

        filtered_df = pd.DataFrame(filtered_predictions)

        logger.info(
            f"Basic confidence gate: {len(filtered_df)}/{len(predictions_df)} predictions pass ≥{min_confidence}% threshold"
        )

        return filtered_df


def enhance_predictions_with_ml(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all advanced ML enhancements"""

    logger.info("Applying advanced ML enhancements...")

    # Apply meta-labeling
    try:
        from ml.meta_labeling_active import apply_meta_labeling

        predictions_df = apply_meta_labeling(predictions_df)
        logger.info("✓ Applied meta-labeling")
    except Exception as e:
        logger.warning(f"Meta-labeling failed: {e}")

    # Apply uncertainty quantification
    try:
        from ml.uncertainty_active import apply_uncertainty_quantification

        predictions_df = apply_uncertainty_quantification(predictions_df)
        logger.info("✓ Applied uncertainty quantification")
    except Exception as e:
        logger.warning(f"Uncertainty quantification failed: {e}")

    # Apply regime detection
    try:
        from ml.regime_detection_active import apply_regime_detection

        predictions_df = apply_regime_detection(predictions_df)
        logger.info("✓ Applied regime detection")
    except Exception as e:
        logger.warning(f"Regime detection failed: {e}")

    return predictions_df


def save_predictions(predictions_df: pd.DataFrame):
    """Save predictions in multiple formats"""
    exports_dir = Path("exports/production")
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Save enhanced predictions
    enhanced_file = exports_dir / "enhanced_predictions.json"
    predictions_dict = predictions_df.to_dict("records")

    with open(enhanced_file, "w") as f:
        json.dump(predictions_dict, f, indent=2, default=str)
    logger.info(f"Saved enhanced predictions to {enhanced_file}")

    # Save as parquet (main format)
    parquet_file = exports_dir / "predictions.parquet"
    predictions_df.to_parquet(parquet_file, index=False)
    logger.info(f"Saved predictions to {parquet_file}")

    # Save as CSV for compatibility
    csv_file = exports_dir / "predictions.csv"
    predictions_df.to_csv(csv_file, index=False)
    logger.info(f"Saved predictions to {csv_file}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(predictions_df),
        "high_confidence_count": len(
            predictions_df[
                predictions_df[
                    [col for col in predictions_df.columns if col.startswith("confidence_")]
                ].max(axis=1)
                >= 80
            ]
        ),
        "horizons": ["1h", "24h", "168h", "720h"],
        "confidence_threshold": 80.0,
        "enhanced_features": {
            "meta_labeling": "meta_label_quality" in predictions_df.columns,
            "uncertainty_quantification": "epistemic_uncertainty" in predictions_df.columns,
            "regime_detection": "regime" in predictions_df.columns,
        },
    }

    metadata_file = exports_dir / "predictions_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return parquet_file


def main():
    """Main prediction pipeline"""
    logger.info("Starting production prediction pipeline")

    # Load models
    models, metadata = load_baseline_models()

    if len(models) == 0:
        logger.error("No models loaded")
        sys.exit(1)

    # Load features
    df = load_latest_features()

    # Generate predictions
    predictions_df = predict_all(models, metadata, df)

    if predictions_df.empty:
        logger.error("No predictions generated")
        sys.exit(1)

    # Enhance predictions with advanced ML
    enhanced_predictions = enhance_predictions_with_ml(predictions_df)

    # Apply confidence gate
    filtered_predictions = apply_confidence_gate(enhanced_predictions)

    # Save predictions
    output_file = save_predictions(filtered_predictions)

    # Summary
    logger.info("=== Prediction Summary ===")
    logger.info(f"Total coins processed: {len(df)}")
    logger.info(f"Predictions generated: {len(predictions_df)}")
    logger.info(f"High confidence (≥80%): {len(filtered_predictions)}")
    logger.info(f"Output file: {output_file}")

    logger.info("Production prediction pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
