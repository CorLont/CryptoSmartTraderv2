#!/usr/bin/env python3
"""
Sentiment Models Ensemble - Multiple sentiment models with uncertainty quantification
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from core.structured_logger import get_structured_logger
from .model import SentimentModel

class SentimentEnsemble:
    """Ensemble of sentiment models for robust predictions"""

    def __init__(self):
        self.logger = get_structured_logger("SentimentEnsemble")
        self.models = {}
        self.weights = {}
        self.initialized = False

    async def initialize(self):
        """Initialize ensemble models"""
        try:
            self.logger.info("Initializing sentiment ensemble")

            # Initialize different models
            self.models['finbert'] = SentimentModel("ProsusAI/finbert")
            await self.models['finbert'].initialize()

            # Model weights (could be learned from validation data)
            self.weights['finbert'] = 1.0

            self.initialized = True
            self.logger.info("Sentiment ensemble initialized")

        except Exception as e:
            self.logger.error(f"Ensemble initialization failed: {e}")
            raise

    async def predict_ensemble(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate ensemble predictions"""

        if not self.initialized:
            await self.initialize()

        try:
            # Get predictions from all models
            all_predictions = {}

            for model_name, model in self.models.items():
                batch_result = await model.predict_batch(texts)
                all_predictions[model_name] = batch_result.results

            # Combine predictions
            ensemble_results = []

            for i, text in enumerate(texts):
                model_scores = []
                model_confidences = []

                for model_name, predictions in all_predictions.items():
                    if i < len(predictions):
                        model_scores.append(predictions[i].score * self.weights[model_name])
                        model_confidences.append(predictions[i].confidence)

                # Ensemble prediction
                if model_scores:
                    ensemble_score = np.mean(model_scores)
                    ensemble_confidence = np.mean(model_confidences)
                    uncertainty = np.std(model_scores) if len(model_scores) > 1 else 0.1

                    result = {
                        "text": text,
                        "ensemble_score": ensemble_score,
                        "ensemble_confidence": ensemble_confidence,
                        "uncertainty": uncertainty,
                        "model_agreement": 1.0 - uncertainty,
                        "individual_predictions": {
                            model_name: {
                                "score": preds[i].score if i < len(preds) else 0.0,
                                "confidence": preds[i].confidence if i < len(preds) else 0.0
                            }
                            for model_name, preds in all_predictions.items()
                        }
                    }
                    ensemble_results.append(result)

            return ensemble_results

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return []
