#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Causal Inference Engine
Advanced causality discovery using Double Machine Learning and causal frameworks
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


@dataclass
class CausalEffect:
    """Causal effect structure"""

    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significance: bool
    method: str
    timestamp: datetime
    sample_size: int
    confounders: List[str]


@dataclass
class CounterfactualPrediction:
    """Counterfactual prediction structure"""

    scenario_id: str
    treatment_variable: str
    original_value: float
    counterfactual_value: float
    predicted_outcome: float
    confidence: float
    explanation: str
    timestamp: datetime


@dataclass
class GrangerCausalityResult:
    """Granger causality test result"""

    cause: str
    effect: str
    f_statistic: float
    p_value: float
    is_causal: bool
    lag_order: int
    aic_score: float
    direction: str  # 'unidirectional', 'bidirectional', 'none'


class DoubleMachineLearning:
    """Double Machine Learning for causal inference"""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # ML models for nuisance functions
        self.outcome_models = [
            RandomForestRegressor(n_estimators=100, random_state=random_state),
            GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            xgb.XGBRegressor(n_estimators=100, random_state=random_state),
        ]

        self.treatment_models = [
            RandomForestRegressor(n_estimators=100, random_state=random_state),
            GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            xgb.XGBRegressor(n_estimators=100, random_state=random_state),
        ]

        self.scaler = StandardScaler()

    def estimate_ate(
        self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, confounders: List[str]
    ) -> CausalEffect:
        """Estimate Average Treatment Effect using DML"""
        try:
            n_samples = len(Y)

            # Cross-fitting procedure
            tscv = TimeSeriesSplit(n_splits=self.n_folds)

            # Storage for cross-fitted predictions
            Y_pred = np.zeros(n_samples)
            T_pred = np.zeros(n_samples)

            # Cross-fitting loop
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                Y_train, Y_val = Y[train_idx], Y[val_idx]
                T_train, T_val = T[train_idx], T[val_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # Train outcome model
                outcome_model = self.outcome_models[fold_idx % len(self.outcome_models)]
                outcome_model.fit(X_train_scaled, Y_train)
                Y_pred[val_idx] = outcome_model.predict(X_val_scaled)

                # Train treatment model
                treatment_model = self.treatment_models[fold_idx % len(self.treatment_models)]
                treatment_model.fit(X_train_scaled, T_train)
                T_pred[val_idx] = treatment_model.predict(X_val_scaled)

            # Compute residuals
            Y_res = Y - Y_pred
            T_res = T - T_pred

            # Estimate causal effect (moment condition)
            numerator = np.mean(T_res * Y_res)
            denominator = np.mean(T_res * T_res)

            if abs(denominator) < 1e-10:
                effect_size = 0.0
                p_value = 1.0
            else:
                effect_size = numerator / denominator

                # Compute standard error and p-value
                residuals = Y_res - effect_size * T_res
                variance = np.var(residuals) / (denominator**2)
                std_error = np.sqrt(variance / n_samples)
                t_stat = effect_size / (std_error + 1e-10)
                p_value = 2 * (1 - abs(t_stat / np.sqrt(n_samples)))

            # Confidence interval (approximate)
            margin_error = (
                1.96 * np.sqrt(variance / n_samples)
                if "variance" in locals()
                else 0.1 * abs(effect_size)
            )
            confidence_interval = (effect_size - margin_error, effect_size + margin_error)

            return CausalEffect(
                treatment="treatment_variable",
                outcome="outcome_variable",
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                p_value=p_value,
                significance=p_value < 0.05,
                method="Double Machine Learning",
                timestamp=datetime.now(),
                sample_size=n_samples,
                confounders=confounders,
            )

        except Exception as e:
            self.logger.error(f"Error in DML estimation: {e}")
            return CausalEffect(
                treatment="treatment_variable",
                outcome="outcome_variable",
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                significance=False,
                method="Double Machine Learning (Failed)",
                timestamp=datetime.now(),
                sample_size=0,
                confounders=confounders,
            )


class GrangerCausalityAnalyzer:
    """Granger causality analysis for time series"""

    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.logger = logging.getLogger(__name__)

    def test_granger_causality(
        self, cause: pd.Series, effect: pd.Series, variable_names: Tuple[str, str] = ("X", "Y")
    ) -> GrangerCausalityResult:
        """Test Granger causality between two time series"""
        try:
            # Ensure same length and alignment
            min_len = min(len(cause), len(effect))
            cause = cause.iloc[-min_len:].values
            effect = effect.iloc[-min_len:].values

            # Find optimal lag order
            best_lag, best_aic = self._find_optimal_lag(cause, effect)

            if best_lag == 0:
                return GrangerCausalityResult(
                    cause=variable_names[0],
                    effect=variable_names[1],
                    f_statistic=0.0,
                    p_value=1.0,
                    is_causal=False,
                    lag_order=0,
                    aic_score=float("inf"),
                    direction="none",
                )

            # Test causality
            f_stat, p_value = self._compute_granger_test(cause, effect, best_lag)

            # Test reverse direction
            f_stat_reverse, p_value_reverse = self._compute_granger_test(effect, cause, best_lag)

            # Determine direction
            is_causal = p_value < 0.05
            is_reverse_causal = p_value_reverse < 0.05

            if is_causal and is_reverse_causal:
                direction = "bidirectional"
            elif is_causal:
                direction = "unidirectional"
            else:
                direction = "none"

            return GrangerCausalityResult(
                cause=variable_names[0],
                effect=variable_names[1],
                f_statistic=f_stat,
                p_value=p_value,
                is_causal=is_causal,
                lag_order=best_lag,
                aic_score=best_aic,
                direction=direction,
            )

        except Exception as e:
            self.logger.error(f"Error in Granger causality test: {e}")
            return GrangerCausalityResult(
                cause=variable_names[0],
                effect=variable_names[1],
                f_statistic=0.0,
                p_value=1.0,
                is_causal=False,
                lag_order=0,
                aic_score=float("inf"),
                direction="none",
            )

    def _find_optimal_lag(self, cause: np.ndarray, effect: np.ndarray) -> Tuple[int, float]:
        """Find optimal lag order using AIC"""
        best_aic = float("inf")
        best_lag = 0

        for lag in range(1, min(self.max_lags + 1, len(cause) // 4)):
            try:
                aic = self._compute_aic(cause, effect, lag)
                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except Exception:
                continue

        return best_lag, best_aic

    def _compute_aic(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        """Compute AIC for given lag order"""
        try:
            # Prepare lagged data
            n = len(effect) - lag
            y = effect[lag:]

            # Create design matrix
            X = np.column_stack(
                [effect[i : n + i] for i in range(lag)] + [cause[i : n + i] for i in range(lag)]
            )

            # Add intercept
            X = np.column_stack([np.ones(n), X])

            # Fit model
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coeffs

            # Compute residual sum of squares
            rss = np.sum((y - y_pred) ** 2)

            # AIC = 2k + n * ln(RSS/n)
            k = X.shape[1]  # number of parameters
            aic = 2 * k + n * np.log(rss / n)

            return aic

        except Exception:
            return float("inf")

    def _compute_granger_test(
        self, cause: np.ndarray, effect: np.ndarray, lag: int
    ) -> Tuple[float, float]:
        """Compute Granger causality F-test"""
        try:
            n = len(effect) - lag
            y = effect[lag:]

            # Restricted model (without cause)
            X_restricted = np.column_stack(
                [
                    np.ones(n),  # intercept
                    *[effect[i : n + i] for i in range(lag)],
                ]
            )

            # Unrestricted model (with cause)
            X_unrestricted = np.column_stack(
                [X_restricted, *[cause[i : n + i] for i in range(lag)]]
            )

            # Fit models
            coeffs_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            coeffs_unrestricted = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]

            # Compute RSS
            rss_restricted = np.sum((y - X_restricted @ coeffs_restricted) ** 2)
            rss_unrestricted = np.sum((y - X_unrestricted @ coeffs_unrestricted) ** 2)

            # F-statistic
            df_num = lag  # number of restrictions
            df_den = n - X_unrestricted.shape[1]  # degrees of freedom

            if df_den <= 0 or rss_unrestricted <= 0:
                return 0.0, 1.0

            f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_den)

            # Approximate p-value (simplified)
            p_value = max(0.001, min(0.999, 1 - (f_stat / (f_stat + df_den))))

            return f_stat, p_value

        except Exception as e:
            return 0.0, 1.0


class CounterfactualPredictor:
    """Counterfactual prediction and what-if analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit counterfactual prediction model"""
        try:
            self.feature_names = list(X.columns)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Use ensemble model for robustness
            self.model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )

            self.model.fit(X_scaled, y)

            self.logger.info("Counterfactual prediction model fitted successfully")

        except Exception as e:
            self.logger.error(f"Error fitting counterfactual model: {e}")

    def predict_counterfactual(
        self, X_original: pd.DataFrame, intervention_variable: str, intervention_value: float
    ) -> List[CounterfactualPrediction]:
        """Predict counterfactual outcomes"""
        predictions = []

        try:
            if self.model is None:
                self.logger.warning("Model not fitted")
                return predictions

            for idx in X_original.index:
                # Original prediction
                X_orig_sample = X_original.loc[idx:idx]
                X_orig_scaled = self.scaler.transform(X_orig_sample)
                original_prediction = self.model.predict(X_orig_scaled)[0]

                # Counterfactual intervention
                X_counterfactual = X_orig_sample.copy()
                original_value = X_counterfactual[intervention_variable].iloc[0]
                X_counterfactual[intervention_variable] = intervention_value

                # Counterfactual prediction
                X_cf_scaled = self.scaler.transform(X_counterfactual)
                counterfactual_prediction = self.model.predict(X_cf_scaled)[0]

                # Effect size
                effect = counterfactual_prediction - original_prediction

                # Confidence (simplified based on prediction difference)
                confidence = min(
                    0.95, max(0.1, 1.0 - abs(effect) / (abs(original_prediction) + 1e-6))
                )

                # Create explanation
                direction = "increase" if effect > 0 else "decrease"
                magnitude = abs(effect)
                explanation = f"Changing {intervention_variable} from {original_value:.4f} to {intervention_value:.4f} would {direction} outcome by {magnitude:.4f}"

                prediction = CounterfactualPrediction(
                    scenario_id=f"scenario_{idx}_{intervention_variable}",
                    treatment_variable=intervention_variable,
                    original_value=original_value,
                    counterfactual_value=intervention_value,
                    predicted_outcome=counterfactual_prediction,
                    confidence=confidence,
                    explanation=explanation,
                    timestamp=datetime.now(),
                )

                predictions.append(prediction)

        except Exception as e:
            self.logger.error(f"Error in counterfactual prediction: {e}")

        return predictions


class CausalInferenceEngine:
    """Main causal inference engine"""

    def __init__(self, config_path: str = "config/causal_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)

        # Initialize components
        self.dml = DoubleMachineLearning()
        self.granger = GrangerCausalityAnalyzer()
        self.counterfactual = CounterfactualPredictor()

        # Results storage
        self.causal_effects: List[CausalEffect] = []
        self.granger_results: List[GrangerCausalityResult] = []
        self.counterfactual_predictions: List[CounterfactualPrediction] = []

        # Load configuration
        self.config = self._load_config()

        self.logger.info("Causal Inference Engine initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load causal inference configuration"""
        default_config = {
            "dml_settings": {"n_folds": 5, "significance_level": 0.05, "min_sample_size": 100},
            "granger_settings": {"max_lags": 10, "significance_level": 0.05},
            "counterfactual_settings": {"confidence_threshold": 0.7, "max_scenarios": 100},
            "analysis_variables": {
                "price_variables": ["price_change", "volume_change", "volatility"],
                "market_variables": ["market_cap_change", "sentiment_score", "whale_activity"],
                "technical_variables": ["rsi", "macd", "bollinger_position"],
            },
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Could not load config, using defaults: {e}")

        return default_config

    def analyze_causal_relationships(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: List[str] = None,
    ) -> CausalEffect:
        """Analyze causal relationships using Double Machine Learning"""
        try:
            if len(data) < self.config["dml_settings"]["min_sample_size"]:
                self.logger.warning(f"Insufficient data for causal analysis: {len(data)} samples")
                return None

            # Prepare data
            confounders = confounders or [
                col for col in data.columns if col not in [treatment_col, outcome_col]
            ]

            Y = data[outcome_col].values
            T = data[treatment_col].values
            X = data[confounders].values

            # Remove any NaN values
            valid_idx = ~(np.isnan(Y) | np.isnan(T) | np.isnan(X).any(axis=1))
            Y = Y[valid_idx]
            T = T[valid_idx]
            X = X[valid_idx]

            if len(Y) < self.config["dml_settings"]["min_sample_size"]:
                self.logger.warning("Insufficient valid data after cleaning")
                return None

            # Estimate causal effect
            effect = self.dml.estimate_ate(Y, T, X, confounders)
            effect.treatment = treatment_col
            effect.outcome = outcome_col

            # Store result
            self.causal_effects.append(effect)

            self.logger.info(
                f"Causal effect estimated: {treatment_col} -> {outcome_col} = {effect.effect_size:.4f}"
            )

            return effect

        except Exception as e:
            self.logger.error(f"Error in causal analysis: {e}")
            return None

    def test_granger_causality_matrix(
        self, data: pd.DataFrame, variables: List[str] = None
    ) -> List[GrangerCausalityResult]:
        """Test Granger causality between all variable pairs"""
        results = []

        try:
            variables = variables or list(data.columns)

            for i, cause_var in enumerate(variables):
                for j, effect_var in enumerate(variables):
                    if i != j:  # Don't test variable with itself
                        cause_series = data[cause_var].dropna()
                        effect_series = data[effect_var].dropna()

                        if len(cause_series) > 20 and len(effect_series) > 20:
                            result = self.granger.test_granger_causality(
                                cause_series, effect_series, (cause_var, effect_var)
                            )
                            results.append(result)

            # Store results
            self.granger_results.extend(results)

            # Log significant relationships
            significant = [r for r in results if r.is_causal]
            self.logger.info(f"Found {len(significant)} significant Granger causal relationships")

            return results

        except Exception as e:
            self.logger.error(f"Error in Granger causality analysis: {e}")
            return results

    def generate_counterfactual_scenarios(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        intervention_variables: List[str] = None,
        intervention_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> List[CounterfactualPrediction]:
        """Generate counterfactual predictions for what-if scenarios"""
        predictions = []

        try:
            # Prepare features and target
            feature_cols = [col for col in data.columns if col != outcome_variable]
            X = data[feature_cols].dropna()
            y = data[outcome_variable].loc[X.index]

            # Fit counterfactual model
            self.counterfactual.fit(X, y)

            # Default intervention variables
            intervention_variables = intervention_variables or feature_cols[:3]

            # Generate scenarios
            for var in intervention_variables:
                if var not in X.columns:
                    continue

                # Determine intervention range
                if intervention_ranges and var in intervention_ranges:
                    min_val, max_val = intervention_ranges[var]
                else:
                    min_val = X[var].min()
                    max_val = X[var].max()

                # Generate intervention values
                intervention_values = np.linspace(min_val, max_val, 5)

                for intervention_value in intervention_values:
                    # Select sample for counterfactual analysis
                    sample_data = X.tail(min(10, len(X)))  # Last 10 observations

                    scenario_predictions = self.counterfactual.predict_counterfactual(
                        sample_data, var, intervention_value
                    )

                    predictions.extend(scenario_predictions)

            # Store predictions
            self.counterfactual_predictions.extend(predictions)

            # Filter high-confidence predictions
            high_confidence = [
                p
                for p in predictions
                if p.confidence > self.config["counterfactual_settings"]["confidence_threshold"]
            ]

            self.logger.info(
                f"Generated {len(predictions)} counterfactual scenarios, {len(high_confidence)} high-confidence"
            )

            return predictions

        except Exception as e:
            self.logger.error(f"Error generating counterfactual scenarios: {e}")
            return predictions

    def discover_market_causality(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive market causality discovery"""
        results = {
            "causal_effects": [],
            "granger_relationships": [],
            "counterfactual_scenarios": [],
            "summary": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Analyze price causality
            price_vars = [col for col in market_data.columns if "price" in col.lower()]
            if len(price_vars) >= 2:
                for outcome in price_vars[:2]:  # Limit to avoid excessive computation
                    for treatment in [col for col in market_data.columns if col != outcome][:3]:
                        effect = self.analyze_causal_relationships(market_data, treatment, outcome)
                        if effect:
                            results["causal_effects"].append(asdict(effect))

            # Granger causality analysis
            key_variables = [
                col
                for col in market_data.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["price", "volume", "rsi", "sentiment", "whale"]
                )
            ][:6]

            if len(key_variables) >= 2:
                granger_results = self.test_granger_causality_matrix(market_data[key_variables])
                results["granger_relationships"] = [asdict(r) for r in granger_results]

            # Counterfactual scenarios
            if price_vars:
                outcome_var = price_vars[0]
                intervention_vars = [
                    col
                    for col in market_data.columns
                    if col != outcome_var
                    and any(kw in col.lower() for kw in ["volume", "sentiment", "whale"])
                ][:3]

                if intervention_vars:
                    cf_predictions = self.generate_counterfactual_scenarios(
                        market_data, outcome_var, intervention_vars
                    )
                    results["counterfactual_scenarios"] = [asdict(p) for p in cf_predictions]

            # Generate summary
            results["summary"] = {
                "total_causal_effects": len(results["causal_effects"]),
                "significant_effects": len(
                    [e for e in results["causal_effects"] if e["significance"]]
                ),
                "granger_relationships": len(results["granger_relationships"]),
                "significant_granger": len(
                    [r for r in results["granger_relationships"] if r["is_causal"]]
                ),
                "counterfactual_scenarios": len(results["counterfactual_scenarios"]),
                "high_confidence_scenarios": len(
                    [s for s in results["counterfactual_scenarios"] if s["confidence"] > 0.7]
                ),
            }

            self.logger.info(f"Market causality discovery completed: {results['summary']}")

        except Exception as e:
            self.logger.error(f"Error in market causality discovery: {e}")
            results["error"] = str(e)

        return results

    def explain_price_movement(
        self, market_data: pd.DataFrame, target_coin: str, movement_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Explain significant price movements using causal analysis"""
        try:
            # Identify significant movements
            price_col = f"{target_coin}_price_change"
            if price_col not in market_data.columns:
                price_col = "price_change"  # fallback

            if price_col not in market_data.columns:
                return {"error": "Price data not available"}

            # Find large movements
            large_movements = market_data[abs(market_data[price_col]) > movement_threshold]

            if len(large_movements) == 0:
                return {"explanation": "No significant price movements found"}

            # Analyze causality for these movements
            explanations = []

            for idx, movement in large_movements.tail(5).iterrows():  # Last 5 movements
                movement_size = movement[price_col]
                direction = "increase" if movement_size > 0 else "decrease"

                # Find potential causes
                potential_causes = []
                for col in market_data.columns:
                    if col != price_col and not pd.isna(movement[col]):
                        if abs(movement[col]) > market_data[col].std():  # Anomalous value
                            potential_causes.append(
                                {
                                    "variable": col,
                                    "value": movement[col],
                                    "z_score": (movement[col] - market_data[col].mean())
                                    / market_data[col].std(),
                                }
                            )

                explanations.append(
                    {
                        "timestamp": idx,
                        "movement": f"{movement_size:.2%} {direction}",
                        "potential_causes": sorted(
                            potential_causes, key=lambda x: abs(x["z_score"]), reverse=True
                        )[:3],
                    }
                )

            return {
                "coin": target_coin,
                "movement_threshold": movement_threshold,
                "explanations": explanations,
                "summary": f"Analyzed {len(explanations)} significant movements",
            }

        except Exception as e:
            self.logger.error(f"Error explaining price movement: {e}")
            return {"error": str(e)}

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        try:
            return {
                "causal_effects": {
                    "total": len(self.causal_effects),
                    "significant": len([e for e in self.causal_effects if e.significance]),
                    "recent": [asdict(e) for e in self.causal_effects[-5:]],
                },
                "granger_causality": {
                    "total_tests": len(self.granger_results),
                    "significant_relationships": len(
                        [r for r in self.granger_results if r.is_causal]
                    ),
                    "bidirectional": len(
                        [r for r in self.granger_results if r.direction == "bidirectional"]
                    ),
                    "recent": [asdict(r) for r in self.granger_results[-5:]],
                },
                "counterfactual_predictions": {
                    "total_scenarios": len(self.counterfactual_predictions),
                    "high_confidence": len(
                        [p for p in self.counterfactual_predictions if p.confidence > 0.7]
                    ),
                    "recent": [asdict(p) for p in self.counterfactual_predictions[-5:]],
                },
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {e}")
            return {"error": str(e)}


# Singleton instance
_causal_inference_engine = None


def get_causal_inference_engine() -> CausalInferenceEngine:
    """Get or create causal inference engine singleton"""
    global _causal_inference_engine
    if _causal_inference_engine is None:
        _causal_inference_engine = CausalInferenceEngine()
    return _causal_inference_engine


def analyze_causality(
    data: pd.DataFrame, treatment: str, outcome: str, confounders: List[str] = None
) -> CausalEffect:
    """Convenient function for causal analysis"""
    engine = get_causal_inference_engine()
    return engine.analyze_causal_relationships(data, treatment, outcome, confounders)


def test_granger_causality(
    data: pd.DataFrame, variables: List[str] = None
) -> List[GrangerCausalityResult]:
    """Convenient function for Granger causality testing"""
    engine = get_causal_inference_engine()
    return engine.test_granger_causality_matrix(data, variables)


def predict_counterfactuals(
    data: pd.DataFrame, outcome: str, interventions: List[str] = None
) -> List[CounterfactualPrediction]:
    """Convenient function for counterfactual prediction"""
    engine = get_causal_inference_engine()
    return engine.generate_counterfactual_scenarios(data, outcome, interventions)


def discover_market_causality(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Convenient function for comprehensive market causality discovery"""
    engine = get_causal_inference_engine()
    return engine.discover_market_causality(market_data)
