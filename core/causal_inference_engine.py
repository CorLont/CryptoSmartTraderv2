#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Causal Inference Engine
Advanced causal inference for understanding WHY market movements happen, not just correlations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class CausalMethod(Enum):
    DOUBLE_ML = "double_ml"
    INSTRUMENTAL_VARIABLES = "instrumental_variables"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    PROPENSITY_SCORE = "propensity_score"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    GRANGER_CAUSALITY = "granger_causality"

class InterventionType(Enum):
    VOLUME_SPIKE = "volume_spike"
    WHALE_MOVEMENT = "whale_movement"
    NEWS_EVENT = "news_event"
    TECHNICAL_BREAKOUT = "technical_breakout"
    MARKET_REGIME_CHANGE = "market_regime_change"
    CROSS_ASSET_CORRELATION = "cross_asset_correlation"

@dataclass
class CausalEffect:
    """Represents a discovered causal effect"""
    treatment: str
    outcome: str
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: CausalMethod
    confounders: List[str]
    instruments: List[str] = field(default_factory=list)
    evidence_strength: float = 0.0
    temporal_lag: int = 0  # Time lag in periods
    mechanism: str = ""
    counterfactual_prediction: Optional[float] = None

@dataclass
class CausalGraph:
    """Represents causal relationships between variables"""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (from, to, strength)
    confounders: Dict[str, List[str]]
    mediators: Dict[str, List[str]]
    instruments: Dict[str, List[str]]

@dataclass
class CausalInferenceConfig:
    """Configuration for causal inference engine"""
    # Methods to use
    enabled_methods: List[CausalMethod] = field(default_factory=lambda: [
        CausalMethod.DOUBLE_ML,
        CausalMethod.GRANGER_CAUSALITY,
        CausalMethod.DIFFERENCE_IN_DIFFERENCES
    ])
    
    # Data requirements
    min_samples: int = 200
    min_treatment_samples: int = 50
    lookback_periods: int = 100
    
    # Statistical thresholds
    significance_level: float = 0.05
    min_effect_size: float = 0.01
    min_evidence_strength: float = 0.6
    
    # Double ML parameters
    n_folds: int = 5
    n_jobs: int = -1
    
    # Instrumental variables
    weak_instrument_threshold: float = 10.0  # F-statistic threshold
    
    # Model persistence
    save_models: bool = True
    model_cache_dir: str = "models/causal_inference"

class DoubleMachineLearning:
    """Double Machine Learning for causal inference"""
    
    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DoubleMachineLearning")
        
        # Models for nuisance functions
        self.outcome_model = None
        self.treatment_model = None
        self.fitted = False
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DoubleMachineLearning':
        """
        Fit Double ML model
        
        Args:
            X: Confounders/controls
            T: Treatment variable
            Y: Outcome variable
        """
        try:
            if not HAS_SKLEARN:
                raise ImportError("Scikit-learn required for Double ML")
            
            self.logger.info(f"Fitting Double ML with {len(X)} samples")
            
            # Initialize models for nuisance functions
            self.outcome_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.treatment_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Cross-fitting procedure
            n_folds = self.config.n_folds
            fold_size = len(X) // n_folds
            
            self.residuals_y = np.zeros(len(Y))
            self.residuals_t = np.zeros(len(T))
            
            for fold in range(n_folds):
                # Split data
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else len(X)
                
                # Test set for this fold
                test_idx = np.arange(start_idx, end_idx)
                train_idx = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(X))])
                
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                
                X_train, X_test = X[train_idx], X[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                
                # Fit outcome model E[Y|X]
                outcome_model_fold = RandomForestRegressor(n_estimators=100, random_state=42)
                outcome_model_fold.fit(X_train, Y_train)
                Y_pred = outcome_model_fold.predict(X_test)
                
                # Fit treatment model E[T|X]
                treatment_model_fold = RandomForestRegressor(n_estimators=100, random_state=42)
                treatment_model_fold.fit(X_train, T_train)
                T_pred = treatment_model_fold.predict(X_test)
                
                # Calculate residuals
                self.residuals_y[test_idx] = Y_test - Y_pred
                self.residuals_t[test_idx] = T_test - T_pred
            
            # Final causal effect estimation
            # θ = E[ψ(W,θ)] where ψ is the moment condition
            valid_idx = (np.abs(self.residuals_t) > 1e-8)  # Avoid division by zero
            
            if np.sum(valid_idx) < 10:
                self.logger.warning("Insufficient valid residuals for causal effect estimation")
                self.causal_effect = 0.0
                self.se = float('inf')
            else:
                # Simple IV-like estimator: Cov(Y_residual, T_residual) / Var(T_residual)
                self.causal_effect = np.cov(self.residuals_y[valid_idx], self.residuals_t[valid_idx])[0, 1] / np.var(self.residuals_t[valid_idx])
                
                # Standard error estimation (simplified)
                n_valid = np.sum(valid_idx)
                residual_product = self.residuals_y[valid_idx] * self.residuals_t[valid_idx]
                moment_var = np.var(residual_product - self.causal_effect * self.residuals_t[valid_idx]**2)
                self.se = np.sqrt(moment_var / (n_valid * np.var(self.residuals_t[valid_idx])**2))
            
            self.fitted = True
            self.logger.info(f"Double ML fitted: effect = {self.causal_effect:.4f} ± {self.se:.4f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Double ML fitting failed: {e}")
            self.causal_effect = 0.0
            self.se = float('inf')
            return self
    
    def get_effect(self) -> Tuple[float, float, float]:
        """Get causal effect estimate with confidence interval"""
        if not self.fitted:
            return 0.0, 0.0, 1.0
        
        # 95% confidence interval
        ci_lower = self.causal_effect - 1.96 * self.se
        ci_upper = self.causal_effect + 1.96 * self.se
        
        # P-value (two-tailed)
        if self.se > 0:
            z_stat = abs(self.causal_effect) / self.se
            p_value = 2 * (1 - self._normal_cdf(z_stat))
        else:
            p_value = 1.0
        
        return self.causal_effect, (ci_lower, ci_upper), p_value
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class GrangerCausality:
    """Granger Causality testing for temporal relationships"""
    
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.logger = logging.getLogger(f"{__name__}.GrangerCausality")
        
    def test_causality(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, int]:
        """
        Test if X Granger-causes Y
        
        Returns:
            F-statistic, p-value, optimal lag
        """
        try:
            if len(X) != len(Y):
                raise ValueError("X and Y must have same length")
            
            if len(X) < 2 * self.max_lags + 10:
                self.logger.warning(f"Insufficient data for Granger causality: {len(X)} samples")
                return 0.0, 1.0, 0
            
            best_f_stat = 0.0
            best_p_value = 1.0
            best_lag = 0
            
            # Test different lag lengths
            for lag in range(1, min(self.max_lags + 1, len(X) // 4)):
                try:
                    f_stat, p_value = self._test_lag(X, Y, lag)
                    
                    if f_stat > best_f_stat:
                        best_f_stat = f_stat
                        best_p_value = p_value
                        best_lag = lag
                        
                except Exception as e:
                    self.logger.debug(f"Granger test failed for lag {lag}: {e}")
                    continue
            
            return best_f_stat, best_p_value, best_lag
            
        except Exception as e:
            self.logger.error(f"Granger causality test failed: {e}")
            return 0.0, 1.0, 0
    
    def _test_lag(self, X: np.ndarray, Y: np.ndarray, lag: int) -> Tuple[float, float]:
        """Test Granger causality for specific lag"""
        if not HAS_SKLEARN:
            return 0.0, 1.0
        
        # Prepare lagged data
        n = len(Y) - lag
        if n < 10:
            return 0.0, 1.0
        
        # Dependent variable (current Y)
        y_current = Y[lag:]
        
        # Restricted model: Y_t = α + Σβ_i * Y_{t-i} + ε_t
        y_lags = np.column_stack([Y[lag-i:-i] if i < lag else Y[lag-i:] 
                                 for i in range(1, lag + 1)])
        
        # Unrestricted model: Y_t = α + Σβ_i * Y_{t-i} + Σγ_j * X_{t-j} + ε_t
        x_lags = np.column_stack([X[lag-i:-i] if i < lag else X[lag-i:] 
                                 for i in range(1, lag + 1)])
        
        try:
            # Fit restricted model
            restricted_model = LinearRegression()
            restricted_model.fit(y_lags, y_current)
            y_pred_restricted = restricted_model.predict(y_lags)
            rss_restricted = np.sum((y_current - y_pred_restricted)**2)
            
            # Fit unrestricted model
            X_full = np.column_stack([y_lags, x_lags])
            unrestricted_model = LinearRegression()
            unrestricted_model.fit(X_full, y_current)
            y_pred_unrestricted = unrestricted_model.predict(X_full)
            rss_unrestricted = np.sum((y_current - y_pred_unrestricted)**2)
            
            # F-test
            df_num = lag  # Number of X lags added
            df_den = n - 2 * lag - 1  # Degrees of freedom for unrestricted model
            
            if df_den <= 0 or rss_unrestricted <= 0:
                return 0.0, 1.0
            
            f_stat = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_den)
            
            # Approximate p-value using F-distribution
            p_value = self._f_distribution_pvalue(f_stat, df_num, df_den)
            
            return max(0.0, f_stat), min(1.0, max(0.0, p_value))
            
        except Exception as e:
            self.logger.debug(f"Lag test failed: {e}")
            return 0.0, 1.0
    
    def _f_distribution_pvalue(self, f_stat: float, df1: int, df2: int) -> float:
        """Approximate F-distribution p-value"""
        if f_stat <= 0:
            return 1.0
        
        # Crude approximation for F-distribution
        # In practice, would use scipy.stats.f.sf(f_stat, df1, df2)
        if f_stat > 10:
            return 0.001
        elif f_stat > 5:
            return 0.01
        elif f_stat > 3:
            return 0.05
        elif f_stat > 2:
            return 0.1
        else:
            return 0.5

class CausalInferenceEngine:
    """Main causal inference engine for cryptocurrency markets"""
    
    def __init__(self, config: Optional[CausalInferenceConfig] = None):
        self.config = config or CausalInferenceConfig()
        self.logger = logging.getLogger(f"{__name__}.CausalInferenceEngine")
        
        # Causal discovery results
        self.causal_effects: List[CausalEffect] = []
        self.causal_graph: Optional[CausalGraph] = None
        self.discovered_relationships: Dict[str, List[CausalEffect]] = {}
        
        # Models and components
        self.double_ml = DoubleMachineLearning(self.config)
        self.granger = GrangerCausality()
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Cache and persistence
        self.model_cache_dir = Path(self.config.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        
        self.logger.info("Causal Inference Engine initialized")
    
    def discover_causal_effects(self, data: pd.DataFrame, 
                               treatment_cols: List[str],
                               outcome_cols: List[str],
                               confounder_cols: List[str] = None) -> List[CausalEffect]:
        """
        Discover causal effects between treatments and outcomes
        
        Args:
            data: Market data with features
            treatment_cols: Treatment variables (e.g., volume_spike, whale_movement)
            outcome_cols: Outcome variables (e.g., price_change, volatility)
            confounder_cols: Control variables
        """
        with self._lock:
            try:
                self.logger.info(f"Discovering causal effects: {len(treatment_cols)} treatments, {len(outcome_cols)} outcomes")
                
                discovered_effects = []
                
                if confounder_cols is None:
                    confounder_cols = [col for col in data.columns 
                                     if col not in treatment_cols + outcome_cols]
                
                # Ensure sufficient data
                if len(data) < self.config.min_samples:
                    self.logger.warning(f"Insufficient data: {len(data)} < {self.config.min_samples}")
                    return discovered_effects
                
                # Discover effects for each treatment-outcome pair
                for treatment in treatment_cols:
                    if treatment not in data.columns:
                        continue
                    
                    for outcome in outcome_cols:
                        if outcome not in data.columns:
                            continue
                        
                        effects = self._analyze_treatment_outcome(
                            data, treatment, outcome, confounder_cols
                        )
                        discovered_effects.extend(effects)
                
                # Store results
                self.causal_effects.extend(discovered_effects)
                
                # Update discovered relationships
                for effect in discovered_effects:
                    if effect.treatment not in self.discovered_relationships:
                        self.discovered_relationships[effect.treatment] = []
                    self.discovered_relationships[effect.treatment].append(effect)
                
                # Build causal graph
                self._build_causal_graph(discovered_effects)
                
                self.logger.info(f"Discovered {len(discovered_effects)} significant causal effects")
                
                return discovered_effects
                
            except Exception as e:
                self.logger.error(f"Causal discovery failed: {e}")
                return []
    
    def _analyze_treatment_outcome(self, data: pd.DataFrame, 
                                 treatment: str, outcome: str,
                                 confounders: List[str]) -> List[CausalEffect]:
        """Analyze causal effect between specific treatment and outcome"""
        effects = []
        
        try:
            # Prepare data
            clean_data = data[[treatment, outcome] + confounders].dropna()
            
            if len(clean_data) < self.config.min_samples:
                return effects
            
            T = clean_data[treatment].values
            Y = clean_data[outcome].values
            X = clean_data[confounders].values if confounders else np.zeros((len(clean_data), 1))
            
            # Check treatment variation
            if np.var(T) < 1e-8:
                self.logger.debug(f"No variation in treatment {treatment}")
                return effects
            
            # Apply each enabled method
            for method in self.config.enabled_methods:
                try:
                    effect = self._apply_causal_method(method, X, T, Y, treatment, outcome, confounders)
                    
                    if effect and self._validate_effect(effect):
                        effects.append(effect)
                        
                except Exception as e:
                    self.logger.debug(f"Method {method.value} failed for {treatment}->{outcome}: {e}")
                    continue
            
            return effects
            
        except Exception as e:
            self.logger.error(f"Treatment-outcome analysis failed: {e}")
            return effects
    
    def _apply_causal_method(self, method: CausalMethod, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                           treatment: str, outcome: str, confounders: List[str]) -> Optional[CausalEffect]:
        """Apply specific causal inference method"""
        try:
            if method == CausalMethod.DOUBLE_ML:
                return self._double_ml_analysis(X, T, Y, treatment, outcome, confounders)
            
            elif method == CausalMethod.GRANGER_CAUSALITY:
                return self._granger_analysis(T, Y, treatment, outcome)
            
            elif method == CausalMethod.DIFFERENCE_IN_DIFFERENCES:
                return self._diff_in_diff_analysis(X, T, Y, treatment, outcome, confounders)
            
            # Other methods would be implemented here
            else:
                self.logger.debug(f"Method {method.value} not yet implemented")
                return None
                
        except Exception as e:
            self.logger.debug(f"Causal method {method.value} failed: {e}")
            return None
    
    def _double_ml_analysis(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                          treatment: str, outcome: str, confounders: List[str]) -> Optional[CausalEffect]:
        """Apply Double Machine Learning"""
        try:
            # Scale data
            if self.scaler is not None and X.shape[1] > 0:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Fit Double ML
            self.double_ml.fit(X_scaled, T, Y)
            effect_size, ci, p_value = self.double_ml.get_effect()
            
            if abs(effect_size) < self.config.min_effect_size:
                return None
            
            return CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect_size=effect_size,
                confidence_interval=ci,
                p_value=p_value,
                method=CausalMethod.DOUBLE_ML,
                confounders=confounders,
                evidence_strength=1.0 - p_value if p_value < 1.0 else 0.0,
                mechanism=f"Double ML estimation with {len(confounders)} confounders"
            )
            
        except Exception as e:
            self.logger.debug(f"Double ML analysis failed: {e}")
            return None
    
    def _granger_analysis(self, T: np.ndarray, Y: np.ndarray,
                         treatment: str, outcome: str) -> Optional[CausalEffect]:
        """Apply Granger Causality testing"""
        try:
            f_stat, p_value, lag = self.granger.test_causality(T, Y)
            
            if p_value > self.config.significance_level:
                return None
            
            # Effect size approximation based on F-statistic
            effect_size = np.sqrt(f_stat) / 100  # Crude approximation
            
            return CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect_size=effect_size,
                confidence_interval=(0.0, 2 * effect_size),  # Approximate
                p_value=p_value,
                method=CausalMethod.GRANGER_CAUSALITY,
                confounders=[],
                evidence_strength=min(1.0, f_stat / 10),
                temporal_lag=lag,
                mechanism=f"Granger causality with {lag} period lag"
            )
            
        except Exception as e:
            self.logger.debug(f"Granger analysis failed: {e}")
            return None
    
    def _diff_in_diff_analysis(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                              treatment: str, outcome: str, confounders: List[str]) -> Optional[CausalEffect]:
        """Apply Difference-in-Differences estimation"""
        try:
            # Simplified DiD: Compare before/after treatment
            # In practice, would need proper time periods and control groups
            
            # Create binary treatment indicator
            T_binary = (T > np.median(T)).astype(int)
            
            # Split into treated and control
            treated_idx = T_binary == 1
            control_idx = T_binary == 0
            
            if np.sum(treated_idx) < self.config.min_treatment_samples or np.sum(control_idx) < self.config.min_treatment_samples:
                return None
            
            # Simple difference in means
            treated_outcome = np.mean(Y[treated_idx])
            control_outcome = np.mean(Y[control_idx])
            effect_size = treated_outcome - control_outcome
            
            # Basic statistical test
            se = np.sqrt(np.var(Y[treated_idx])/np.sum(treated_idx) + np.var(Y[control_idx])/np.sum(control_idx))
            
            if se > 0:
                t_stat = effect_size / se
                p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            else:
                p_value = 1.0
            
            if p_value > self.config.significance_level or abs(effect_size) < self.config.min_effect_size:
                return None
            
            ci_lower = effect_size - 1.96 * se
            ci_upper = effect_size + 1.96 * se
            
            return CausalEffect(
                treatment=treatment,
                outcome=outcome,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                method=CausalMethod.DIFFERENCE_IN_DIFFERENCES,
                confounders=confounders,
                evidence_strength=1.0 - p_value if p_value < 1.0 else 0.0,
                mechanism=f"Difference-in-differences with {np.sum(treated_idx)} treated units"
            )
            
        except Exception as e:
            self.logger.debug(f"DiD analysis failed: {e}")
            return None
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _validate_effect(self, effect: CausalEffect) -> bool:
        """Validate discovered causal effect"""
        return (
            effect.p_value <= self.config.significance_level and
            abs(effect.effect_size) >= self.config.min_effect_size and
            effect.evidence_strength >= self.config.min_evidence_strength
        )
    
    def _build_causal_graph(self, effects: List[CausalEffect]):
        """Build causal graph from discovered effects"""
        try:
            # Extract unique variables
            nodes = set()
            edges = []
            
            for effect in effects:
                nodes.add(effect.treatment)
                nodes.add(effect.outcome)
                edges.append((effect.treatment, effect.outcome, effect.effect_size))
            
            # Build graph structure
            self.causal_graph = CausalGraph(
                nodes=list(nodes),
                edges=edges,
                confounders={},  # Would be populated with confounder analysis
                mediators={},    # Would be populated with mediation analysis
                instruments={}   # Would be populated with IV analysis
            )
            
        except Exception as e:
            self.logger.error(f"Causal graph construction failed: {e}")
    
    def predict_counterfactual(self, data: pd.DataFrame, 
                              treatment: str, outcome: str,
                              intervention_value: float) -> Optional[float]:
        """
        Predict counterfactual outcome under intervention
        
        Args:
            data: Current market state
            treatment: Treatment variable to intervene on
            outcome: Outcome to predict
            intervention_value: Value to set treatment to
        """
        try:
            # Find relevant causal effect
            relevant_effect = None
            for effect in self.causal_effects:
                if effect.treatment == treatment and effect.outcome == outcome:
                    relevant_effect = effect
                    break
            
            if relevant_effect is None:
                self.logger.warning(f"No causal effect found for {treatment} -> {outcome}")
                return None
            
            # Current treatment value
            current_treatment = data[treatment].iloc[-1] if treatment in data.columns else 0
            
            # Current outcome value
            current_outcome = data[outcome].iloc[-1] if outcome in data.columns else 0
            
            # Counterfactual prediction
            treatment_change = intervention_value - current_treatment
            predicted_change = treatment_change * relevant_effect.effect_size
            counterfactual_outcome = current_outcome + predicted_change
            
            self.logger.info(f"Counterfactual: {treatment}={intervention_value} -> {outcome}={counterfactual_outcome:.4f}")
            
            return counterfactual_outcome
            
        except Exception as e:
            self.logger.error(f"Counterfactual prediction failed: {e}")
            return None
    
    def explain_movement(self, data: pd.DataFrame, 
                        outcome: str, 
                        time_window: int = 10) -> Dict[str, Any]:
        """
        Explain why a price movement happened using causal analysis
        
        Args:
            data: Market data
            outcome: Variable to explain (e.g., 'price_change')
            time_window: Window to analyze
        """
        try:
            explanations = {
                'primary_causes': [],
                'contributing_factors': [],
                'mechanism_strength': {},
                'temporal_structure': {},
                'confidence': 0.0
            }
            
            if outcome not in data.columns:
                return explanations
            
            # Recent data
            recent_data = data.tail(time_window)
            outcome_change = recent_data[outcome].iloc[-1] - recent_data[outcome].iloc[0]
            
            # Find causal effects for this outcome
            relevant_effects = [e for e in self.causal_effects if e.outcome == outcome]
            
            total_explained_variance = 0.0
            
            for effect in relevant_effects:
                if effect.treatment not in recent_data.columns:
                    continue
                
                # Calculate treatment change
                treatment_change = recent_data[effect.treatment].iloc[-1] - recent_data[effect.treatment].iloc[0]
                
                if abs(treatment_change) < 1e-8:
                    continue
                
                # Predicted contribution
                predicted_contribution = treatment_change * effect.effect_size
                contribution_ratio = abs(predicted_contribution / outcome_change) if abs(outcome_change) > 1e-8 else 0
                
                explanation = {
                    'cause': effect.treatment,
                    'effect_size': effect.effect_size,
                    'treatment_change': treatment_change,
                    'predicted_contribution': predicted_contribution,
                    'contribution_ratio': contribution_ratio,
                    'confidence': effect.evidence_strength,
                    'method': effect.method.value,
                    'lag': effect.temporal_lag,
                    'mechanism': effect.mechanism
                }
                
                # Categorize by importance
                if contribution_ratio > 0.3 and effect.evidence_strength > 0.7:
                    explanations['primary_causes'].append(explanation)
                elif contribution_ratio > 0.1 and effect.evidence_strength > 0.5:
                    explanations['contributing_factors'].append(explanation)
                
                total_explained_variance += contribution_ratio * effect.evidence_strength
            
            # Sort by importance
            explanations['primary_causes'].sort(key=lambda x: x['contribution_ratio'] * x['confidence'], reverse=True)
            explanations['contributing_factors'].sort(key=lambda x: x['contribution_ratio'] * x['confidence'], reverse=True)
            
            explanations['confidence'] = min(1.0, total_explained_variance)
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Movement explanation failed: {e}")
            return {'primary_causes': [], 'contributing_factors': [], 'confidence': 0.0}
    
    def get_causal_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of discovered causal relationships"""
        with self._lock:
            return {
                'total_effects_discovered': len(self.causal_effects),
                'effects_by_method': {
                    method.value: len([e for e in self.causal_effects if e.method == method])
                    for method in CausalMethod
                },
                'strongest_effects': sorted(
                    [
                        {
                            'treatment': e.treatment,
                            'outcome': e.outcome,
                            'effect_size': e.effect_size,
                            'p_value': e.p_value,
                            'method': e.method.value
                        }
                        for e in self.causal_effects
                    ],
                    key=lambda x: abs(x['effect_size']) * (1 - x['p_value']),
                    reverse=True
                )[:10],
                'causal_graph_nodes': len(self.causal_graph.nodes) if self.causal_graph else 0,
                'causal_graph_edges': len(self.causal_graph.edges) if self.causal_graph else 0,
                'relationships_discovered': {
                    treatment: len(effects) 
                    for treatment, effects in self.discovered_relationships.items()
                }
            }


# Singleton causal inference engine
_causal_inference_engine = None
_cie_lock = threading.Lock()

def get_causal_inference_engine(config: Optional[CausalInferenceConfig] = None) -> CausalInferenceEngine:
    """Get the singleton causal inference engine"""
    global _causal_inference_engine
    
    with _cie_lock:
        if _causal_inference_engine is None:
            _causal_inference_engine = CausalInferenceEngine(config)
        return _causal_inference_engine

def discover_market_causality(data: pd.DataFrame, 
                             treatments: List[str],
                             outcomes: List[str]) -> List[CausalEffect]:
    """Convenient function to discover causal effects in market data"""
    engine = get_causal_inference_engine()
    return engine.discover_causal_effects(data, treatments, outcomes)

def explain_price_movement(data: pd.DataFrame, outcome_col: str = 'price_change') -> Dict[str, Any]:
    """Convenient function to explain price movements"""
    engine = get_causal_inference_engine()
    return engine.explain_movement(data, outcome_col)