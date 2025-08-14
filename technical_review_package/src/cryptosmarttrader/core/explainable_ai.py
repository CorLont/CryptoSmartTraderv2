"""
CryptoSmartTrader V2 - Explainable AI Module
SHAP-based feature importance and prediction explanations
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    logging.warning("SHAP not available - explanations will use simplified methods")


class PredictionExplainer:
    """Explainable AI for ML predictions with SHAP integration"""

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()

        # Feature mappings for human-readable explanations
        self.feature_descriptions = {
            "price_change_1h": "Price change in last hour",
            "price_change_24h": "Price change in last 24 hours",
            "volume_spike": "Volume spike indicator",
            "rsi_14": "RSI (14-period momentum)",
            "macd_signal": "MACD signal strength",
            "bb_position": "Bollinger Bands position",
            "sentiment_score": "Social sentiment score",
            "whale_activity": "Large transaction activity",
            "market_dominance": "Market cap dominance",
            "correlation_btc": "Correlation with Bitcoin",
            "volatility_ratio": "Volatility compared to average",
            "momentum_strength": "Price momentum strength",
            "support_resistance": "Support/resistance levels",
            "onchain_activity": "On-chain activity score",
            "news_sentiment": "News sentiment impact",
        }

        self.explainers = {}  # Store SHAP explainers per model

    def explain_prediction(
        self,
        coin: str,
        prediction: float,
        horizon: str,
        features: Dict[str, float],
        model_name: str = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation for a prediction"""
        try:
            explanation = {
                "coin": coin,
                "prediction": prediction,
                "horizon": horizon,
                "confidence": self._calculate_confidence(features),
                "timestamp": datetime.now().isoformat(),
                "feature_importance": [],
                "key_factors": [],
                "risk_factors": [],
                "prediction_range": self._calculate_prediction_range(prediction, features),
                "scenario_analysis": self._generate_scenarios(features, prediction),
                "human_explanation": "",
            }

            # Get feature importance
            if SHAP_AVAILABLE and model_name and shap is not None:
                shap_explanation = self._get_shap_explanation(features, model_name, horizon)
                if shap_explanation:
                    explanation["feature_importance"] = shap_explanation["feature_importance"]
                    explanation["shap_values"] = shap_explanation["shap_values"]
                else:
                    # Fallback to rule-based importance
                    explanation["feature_importance"] = self._calculate_rule_based_importance(
                        features
                    )
            else:
                # Fallback to rule-based importance
                explanation["feature_importance"] = self._calculate_rule_based_importance(features)

            # Identify key factors
            explanation["key_factors"] = self._identify_key_factors(
                features, explanation["feature_importance"]
            )
            explanation["risk_factors"] = self._identify_risk_factors(features)

            # Generate human-readable explanation
            explanation["human_explanation"] = self._generate_human_explanation(
                coin, prediction, horizon, explanation["key_factors"], explanation["risk_factors"]
            )

            return explanation

        except Exception as e:
            self.logger.error(f"Prediction explanation failed for {coin}: {e}")
            return self._get_fallback_explanation(coin, prediction, horizon)

    def _get_shap_explanation(
        self, features: Dict[str, float], model_name: str, horizon: str
    ) -> Optional[Dict]:
        """Get SHAP-based explanation"""
        try:
            if not SHAP_AVAILABLE or shap is None:
                return None

            # Load or create SHAP explainer
            explainer_key = f"{model_name}_{horizon}"
            if explainer_key not in self.explainers:
                self.explainers[explainer_key] = self._create_shap_explainer(model_name, horizon)

            explainer = self.explainers[explainer_key]
            if not explainer:
                return None

            # Prepare feature array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            feature_names = list(features.keys())

            # Calculate SHAP values
            shap_values = explainer.shap_values(feature_array)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for binary classification

            # Create feature importance list
            feature_importance = []
            for i, (feature, value) in enumerate(features.items()):
                importance = (
                    abs(shap_values[0][i]) if len(shap_values.shape) > 1 else abs(shap_values[i])
                )
                direction = (
                    "positive"
                    if (shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]) > 0
                    else "negative"
                )

                feature_importance.append(
                    {
                        "feature": feature,
                        "description": self.feature_descriptions.get(feature, feature),
                        "value": float(value),
                        "importance": float(importance),
                        "direction": direction,
                        "shap_value": float(
                            shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]
                        ),
                    }
                )

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            return {"feature_importance": feature_importance, "shap_values": shap_values.tolist()}

        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return None

    def _create_shap_explainer(self, model_name: str, horizon: str) -> Optional[Any]:
        """Create SHAP explainer for model"""
        try:
            if not SHAP_AVAILABLE or shap is None:
                return None

            # Try to load model
            model_path = Path(f"models/{model_name}_{horizon}_model.pkl")
            if not model_path.exists():
                return None

            model = joblib.load(model_path)

            # Create appropriate explainer based on model type
            if hasattr(model, "predict_proba"):
                # Tree-based models (LightGBM, XGBoost, RandomForest)
                explainer = shap.TreeExplainer(model)
            else:
                # Linear models or others
                # Need background data for KernelExplainer
                background_data = self._get_background_data()
                if background_data is not None:
                    explainer = shap.KernelExplainer(model.predict, background_data)
                else:
                    return None

            return explainer

        except Exception as e:
            self.logger.error(f"Failed to create SHAP explainer: {e}")
            return None

    def _get_background_data(self) -> Optional[np.ndarray]:
        """Get background data for SHAP explainer"""
        try:
            # Try to get cached training data sample
            cached_data = self.cache_manager.get("training_data_sample")
            if cached_data is not None:
                return cached_data

            # Generate synthetic background data as fallback
            n_features = 15  # Adjust based on actual feature count
            background = np.random.normal(0, 1)

            self.cache_manager.set("training_data_sample", background, ttl_minutes=120)
            return background

        except Exception as e:
            self.logger.error(f"Failed to get background data: {e}")
            return None

    def _calculate_rule_based_importance(self, features: Dict[str, float]) -> List[Dict]:
        """Calculate feature importance using rule-based approach"""
        try:
            # Define importance weights for different feature types
            importance_weights = {
                "price_change_24h": 0.9,
                "volume_spike": 0.8,
                "sentiment_score": 0.7,
                "whale_activity": 0.8,
                "rsi_14": 0.6,
                "macd_signal": 0.6,
                "momentum_strength": 0.7,
                "news_sentiment": 0.6,
                "onchain_activity": 0.7,
                "volatility_ratio": 0.5,
            }

            feature_importance = []
            for feature, value in features.items():
                # Calculate importance based on value magnitude and predefined weights
                base_weight = importance_weights.get(feature, 0.3)
                value_magnitude = abs(value) if isinstance(value, (int, float)) else 0.5

                # Normalize value magnitude (assuming most values are between -5 and 5)
                normalized_magnitude = min(float(value_magnitude) / 5.0, 1.0)

                importance = base_weight * normalized_magnitude
                direction = "positive" if value > 0 else "negative"

                feature_importance.append(
                    {
                        "feature": feature,
                        "description": self.feature_descriptions.get(feature, feature),
                        "value": float(value) if isinstance(value, (int, float)) else 0.0,
                        "importance": importance,
                        "direction": direction,
                    }
                )

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            return feature_importance

        except Exception as e:
            self.logger.error(f"Rule-based importance calculation failed: {e}")
            return []

    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate prediction confidence based on feature quality"""
        try:
            confidence_factors = []

            # Data completeness
            non_zero_features = sum(
                1 for v in features.values() if isinstance(v, (int, float)) and v != 0
            )
            completeness = non_zero_features / len(features)
            confidence_factors.append(completeness)

            # Feature consistency (look for extreme values)
            extreme_values = sum(
                1 for v in features.values() if isinstance(v, (int, float)) and abs(v) > 3
            )
            consistency = max(0, 1 - (extreme_values / len(features)))
            confidence_factors.append(consistency)

            # Signal strength (higher absolute values = more confident)
            signal_strength = np.mean(
                [min(abs(v), 2) / 2 for v in features.values() if isinstance(v, (int, float))]
            )
            confidence_factors.append(signal_strength)

            # Calculate weighted average
            overall_confidence = np.mean(confidence_factors)

            return max(0.1, min(0.95, overall_confidence))  # Clamp between 10% and 95%

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _identify_key_factors(
        self, features: Dict[str, float], feature_importance: List[Dict]
    ) -> List[str]:
        """Identify key factors driving the prediction"""
        try:
            key_factors = []

            # Take top 5 most important features
            for feat in feature_importance[:5]:
                if feat["importance"] > 0.3:  # Significance threshold
                    factor_desc = f"{feat['description']}: {feat['value']:.2f}"
                    if feat["direction"] == "positive":
                        factor_desc += " (bullish)"
                    else:
                        factor_desc += " (bearish)"
                    key_factors.append(factor_desc)

            return key_factors

        except Exception as e:
            self.logger.error(f"Key factors identification failed: {e}")
            return []

    def _identify_risk_factors(self, features: Dict[str, float]) -> List[str]:
        """Identify risk factors that could affect prediction"""
        try:
            risk_factors = []

            # High volatility risk
            if features.get("volatility_ratio", 0) > 2:
                risk_factors.append("High volatility - price could swing dramatically")

            # Low volume risk
            if features.get("volume_spike", 0) < -0.5:
                risk_factors.append("Low trading volume - liquidity concerns")

            # Negative sentiment risk
            if features.get("sentiment_score", 0) < -0.5:
                risk_factors.append("Negative market sentiment")

            # Technical indicator risks
            if features.get("rsi_14", 50) > 80:
                risk_factors.append("Overbought conditions (RSI > 80)")
            elif features.get("rsi_14", 50) < 20:
                risk_factors.append("Oversold conditions (RSI < 20)")

            # Correlation risk
            if features.get("correlation_btc", 0) > 0.8:
                risk_factors.append("High Bitcoin correlation - susceptible to BTC moves")

            return risk_factors

        except Exception as e:
            self.logger.error(f"Risk factors identification failed: {e}")
            return []

    def _calculate_prediction_range(
        self, prediction: float, features: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate prediction range based on uncertainty"""
        try:
            # Base uncertainty from volatility
            volatility = abs(features.get("volatility_ratio", 1))
            base_uncertainty = min(volatility * 0.1, 0.3)  # Max 30% uncertainty

            # Adjust for data quality
            data_quality = sum(
                1 for v in features.values() if isinstance(v, (int, float)) and v != 0
            ) / len(features)
            uncertainty = base_uncertainty * (
                2 - data_quality
            )  # Lower quality = higher uncertainty

            return {
                "lower_bound": prediction * (1 - uncertainty),
                "upper_bound": prediction * (1 + uncertainty),
                "uncertainty": uncertainty,
            }

        except Exception as e:
            self.logger.error(f"Prediction range calculation failed: {e}")
            return {
                "lower_bound": prediction * 0.8,
                "upper_bound": prediction * 1.2,
                "uncertainty": 0.2,
            }

    def _generate_scenarios(self, features: Dict[str, float], prediction: float) -> Dict[str, Dict]:
        """Generate what-if scenarios"""
        try:
            scenarios = {}

            # Sentiment boost scenario
            if "sentiment_score" in features:
                scenarios["sentiment_boost"] = {
                    "description": "If sentiment improves by 50%",
                    "impact": prediction * 1.2,
                    "probability": 0.3,
                }

            # Volume surge scenario
            if "volume_spike" in features:
                scenarios["volume_surge"] = {
                    "description": "If trading volume doubles",
                    "impact": prediction * 1.15,
                    "probability": 0.25,
                }

            # Market crash scenario
            scenarios["market_crash"] = {
                "description": "If Bitcoin drops 20%",
                "impact": prediction * 0.7,
                "probability": 0.1,
            }

            # News catalyst scenario
            scenarios["positive_news"] = {
                "description": "If major positive news breaks",
                "impact": prediction * 1.5,
                "probability": 0.15,
            }

            return scenarios

        except Exception as e:
            self.logger.error(f"Scenario generation failed: {e}")
            return {}

    def _generate_human_explanation(
        self,
        coin: str,
        prediction: float,
        horizon: str,
        key_factors: List[str],
        risk_factors: List[str],
    ) -> str:
        """Generate human-readable explanation"""
        try:
            # Determine prediction direction and magnitude
            if prediction > 1.5:
                direction = "strong upward"
                confidence_phrase = "very bullish"
            elif prediction > 1.1:
                direction = "moderate upward"
                confidence_phrase = "bullish"
            elif prediction > 0.9:
                direction = "sideways"
                confidence_phrase = "neutral"
            elif prediction > 0.7:
                direction = "moderate downward"
                confidence_phrase = "bearish"
            else:
                direction = "strong downward"
                confidence_phrase = "very bearish"

            # Build explanation
            explanation = f"The model predicts a {direction} movement for {coin} over the {horizon} timeframe. "
            explanation += f"This {confidence_phrase} outlook is based on several key factors:\n\n"

            # Add key factors
            if key_factors:
                explanation += "Key supporting factors:\n"
                for i, factor in enumerate(key_factors[:3], 1):
                    explanation += f"{i}. {factor}\n"
                explanation += "\n"

            # Add risk factors
            if risk_factors:
                explanation += "Risk factors to consider:\n"
                for i, risk in enumerate(risk_factors[:3], 1):
                    explanation += f"{i}. {risk}\n"
                explanation += "\n"

            # Add disclaimer
            explanation += (
                f"Prediction confidence varies based on data quality and market conditions. "
            )
            explanation += f"Always consider multiple factors and risk management when making trading decisions."

            return explanation

        except Exception as e:
            self.logger.error(f"Human explanation generation failed: {e}")
            return f"Model predicts {prediction:.1%} change for {coin} over {horizon}. Analysis details unavailable."

    def _get_fallback_explanation(
        self, coin: str, prediction: float, horizon: str
    ) -> Dict[str, Any]:
        """Get fallback explanation when main explanation fails"""
        return {
            "coin": coin,
            "prediction": prediction,
            "horizon": horizon,
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat(),
            "feature_importance": [],
            "key_factors": ["Analysis unavailable"],
            "risk_factors": ["Standard market risks apply"],
            "human_explanation": f"Model predicts {prediction:.1%} change for {coin} over {horizon}. Detailed analysis unavailable.",
            "prediction_range": {
                "lower_bound": prediction * 0.8,
                "upper_bound": prediction * 1.2,
                "uncertainty": 0.2,
            },
            "scenario_analysis": {},
        }

    def explain_portfolio_allocation(
        self, allocations: Dict[str, float], explanations: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Explain portfolio allocation decisions"""
        try:
            portfolio_explanation = {
                "timestamp": datetime.now().isoformat(),
                "total_allocation": sum(allocations.values()),
                "top_positions": [],
                "allocation_rationale": [],
                "risk_assessment": {},
                "diversification_score": 0.0,
            }

            # Sort positions by allocation
            sorted_positions = sorted(allocations.items(), key=lambda x: x[1], reverse=True)

            # Analyze top positions
            for coin, allocation in sorted_positions[:5]:
                if coin in explanations:
                    exp = explanations[coin]
                    portfolio_explanation["top_positions"].append(
                        {
                            "coin": coin,
                            "allocation": allocation,
                            "prediction": exp.get("prediction", 0),
                            "confidence": exp.get("confidence", 0),
                            "key_reason": exp.get("key_factors", ["Unknown"])[0]
                            if exp.get("key_factors")
                            else "Unknown",
                        }
                    )

            # Calculate diversification score
            portfolio_explanation["diversification_score"] = self._calculate_diversification_score(
                allocations
            )

            # Generate allocation rationale
            portfolio_explanation["allocation_rationale"] = self._generate_allocation_rationale(
                sorted_positions, explanations
            )

            return portfolio_explanation

        except Exception as e:
            self.logger.error(f"Portfolio explanation failed: {e}")
            return {"error": str(e)}

    def _calculate_diversification_score(self, allocations: Dict[str, float]) -> float:
        """Calculate portfolio diversification score"""
        try:
            if not allocations:
                return 0.0

            # Calculate Herfindahl-Hirschman Index
            total_allocation = sum(allocations.values())
            if total_allocation == 0:
                return 0.0

            hhi = sum((allocation / total_allocation) ** 2 for allocation in allocations.values())

            # Convert to diversification score (1 = perfectly diversified, 0 = concentrated)
            diversification_score = 1 - hhi

            return diversification_score

        except Exception as e:
            self.logger.error(f"Diversification calculation failed: {e}")
            return 0.0

    def _generate_allocation_rationale(
        self, sorted_positions: List[Tuple], explanations: Dict[str, Dict]
    ) -> List[str]:
        """Generate rationale for allocation decisions"""
        try:
            rationale = []

            if not sorted_positions:
                return ["No positions allocated"]

            # Analyze concentration
            top_3_allocation = sum(allocation for _, allocation in sorted_positions[:3])
            if top_3_allocation > 0.7:
                rationale.append(
                    "High concentration in top 3 positions - higher risk/reward profile"
                )
            elif top_3_allocation < 0.4:
                rationale.append("Well-diversified allocation across multiple positions")

            # Analyze top position
            top_coin, top_allocation = sorted_positions[0]
            if top_coin in explanations:
                top_exp = explanations[top_coin]
                confidence = top_exp.get("confidence", 0.5)
                if confidence > 0.8:
                    rationale.append(f"High confidence in {top_coin} justifies largest allocation")
                elif confidence < 0.6:
                    rationale.append(
                        f"Moderate confidence in {top_coin} suggests cautious position sizing"
                    )

            # Count high-confidence positions
            high_conf_positions = sum(
                1
                for coin, _ in sorted_positions
                if coin in explanations and explanations[coin].get("confidence", 0) > 0.7
            )

            if high_conf_positions >= 3:
                rationale.append(
                    "Multiple high-confidence opportunities support aggressive allocation"
                )
            elif high_conf_positions == 0:
                rationale.append("Lower confidence levels suggest defensive positioning")

            return rationale

        except Exception as e:
            self.logger.error(f"Allocation rationale generation failed: {e}")
            return ["Allocation analysis unavailable"]
