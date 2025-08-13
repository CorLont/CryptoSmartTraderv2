# agents/ml_predictor.py - Multi-horizon ML prediction agent
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class MLPredictorAgent:
    """Multi-horizon ML predictions with confidence scoring"""

    def __init__(self):
        self.horizons = ["1h", "24h", "168h", "720h"]
        self.min_confidence = 0.80

    async def generate_predictions(self):
        """Generate predictions for all horizons"""
        try:
            # Load latest features
            features_path = Path("exports/features.parquet")
            if not features_path.exists():
                logger.warning("No features available for prediction")
                return []

            features_df = pd.read_parquet(features_path)

            # Generate predictions using trained models
            from ml.models.predict import predict_all
            predictions_df = predict_all(features_df)

            # Apply enterprise confidence gate
            from orchestration.strict_gate import enterprise_confidence_gate

            results = []
            for horizon in self.horizons:
                pred_col = f"pred_{horizon}"
                conf_col = f"conf_{horizon}"

                if pred_col in predictions_df.columns and conf_col in predictions_df.columns:
                    # Filter by confidence gate
                    filtered_df, gate_report = enterprise_confidence_gate(
                        predictions_df, min_threshold=self.min_confidence
                    )

                    # Convert to opportunities format
                    for _, row in filtered_df.iterrows():
                        if pd.notna(row[pred_col]) and pd.notna(row[conf_col]):
                            expected_return = row[pred_col] * 100  # Convert to percentage
                            confidence = row[conf_col] * 100

                            results.append({
                                'symbol': row['coin'],
                                'horizon': horizon,
                                'expected_return': expected_return,
                                'confidence': confidence,
                                'timestamp': datetime.utcnow().isoformat(),
                                'score': confidence,
                                'risk_level': self._calculate_risk_level(expected_return, confidence)
                            })

            # Sort by confidence * expected return (best opportunities first)
            results.sort(key=lambda x: x['confidence'] * abs(x['expected_return']), reverse=True)

            return results

        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return []

    def _calculate_risk_level(self, expected_return, confidence):
        """Calculate risk level based on return and confidence"""
        if confidence >= 90 and expected_return > 0:
            return "LOW"
        elif confidence >= 80 and expected_return > 0:
            return "MEDIUM"
        elif confidence >= 70:
            return "HIGH"
        else:
            return "EXTREME"

    async def save_predictions(self, predictions):
        """Save predictions to production files"""
        if not predictions:
            logger.warning("No predictions to save")
            return

        # Convert to DataFrame and save
        df = pd.DataFrame(predictions)

        # Ensure directories exist
        Path("exports/production").mkdir(parents=True, exist_ok=True)

        # Save as CSV for UI consumption
        df.to_csv("exports/production/predictions.csv", index=False)

        # Save as JSON for API consumption
        with open("exports/production/predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        # Create summary metrics
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_predictions': len(predictions),
            'by_horizon': df['horizon'].value_counts().to_dict(),
            'by_risk': df['risk_level'].value_counts().to_dict(),
            'avg_confidence': df['confidence'].mean(),
            'avg_expected_return': df['expected_return'].mean()
        }

        with open("exports/production/prediction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved {len(predictions)} predictions across {len(set(p['horizon'] for p in predictions))} horizons")

    async def run_continuous(self):
        """Run ML prediction continuously"""
        while True:
            try:
                logger.info("Starting prediction cycle...")

                predictions = await self.generate_predictions()
                await self.save_predictions(predictions)

                if predictions:
                    high_confidence = [p for p in predictions if p['confidence'] >= 85]
                    logger.info(f"Generated {len(predictions)} predictions, {len(high_confidence)} high-confidence")

            except Exception as e:
                logger.error(f"Prediction cycle failed: {e}")

            await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    agent = MLPredictorAgent()
    asyncio.run(agent.run_continuous())
