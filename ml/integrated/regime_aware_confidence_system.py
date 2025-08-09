#!/usr/bin/env python3
"""
Regime-Aware Confidence System
Integrated system combining regime detection, calibrated confidence gates, and uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

from ml.regime.market_regime_detector import MarketRegime, RegimeState, MarketRegimeDetector, create_regime_detector
from ml.regime.regime_aware_models import RegimeRouter, create_regime_aware_system
from ml.calibration.confidence_gate_calibrated import CalibratedConfidenceGate, CalibratedPrediction, ConfidenceGateConfig
from ml.calibration.probability_calibrator import ProbabilityCalibrator, CalibrationMethod
from ml.uncertainty.bayesian_uncertainty import UncertaintyQuantifier, UncertaintyEstimate

@dataclass
class RegimeAwarePrediction:
    """Prediction with regime context, calibrated confidence, and uncertainty"""
    symbol: str
    prediction: float
    raw_confidence: float
    calibrated_confidence: float
    regime: MarketRegime
    regime_confidence: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_used: str
    regime_adjusted_confidence: float
    passes_regime_gate: bool
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())

@dataclass
class RegimeGateConfig:
    """Configuration for regime-aware confidence gating"""
    base_confidence_threshold: float = 0.80
    regime_adjustments: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.BULL_MARKET: 0.0,        # No adjustment for bull
        MarketRegime.BEAR_MARKET: 0.1,        # Higher threshold for bear
        MarketRegime.HIGH_VOLATILITY: 0.15,   # Much higher for high vol
        MarketRegime.CONSOLIDATION: -0.05,    # Lower for consolidation
        MarketRegime.LOW_VOLATILITY: -0.1,    # Lower for low vol
        MarketRegime.TREND_REVERSAL: 0.2,     # Highest for reversals
        MarketRegime.UNKNOWN: 0.1             # Conservative for unknown
    })
    uncertainty_scaling: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.BULL_MARKET: 1.0,
        MarketRegime.BEAR_MARKET: 1.2,
        MarketRegime.HIGH_VOLATILITY: 1.5,
        MarketRegime.CONSOLIDATION: 0.8,
        MarketRegime.LOW_VOLATILITY: 0.7,
        MarketRegime.TREND_REVERSAL: 1.8,
        MarketRegime.UNKNOWN: 1.3
    })

class RegimeAwareCalibrator:
    """Regime-specific probability calibration"""
    
    def __init__(self):
        self.regime_calibrators: Dict[MarketRegime, ProbabilityCalibrator] = {}
        self.fallback_calibrator = ProbabilityCalibrator()
        self.calibration_data_requirements = 100  # Min samples per regime
        
        self.logger = logging.getLogger(__name__)
    
    def fit_regime_calibrators(
        self,
        predictions_history: pd.DataFrame,
        regime_history: List[MarketRegime]
    ):
        """Fit separate calibrators for each regime"""
        
        required_columns = ['raw_confidence', 'actual_outcome']
        if not all(col in predictions_history.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        if len(regime_history) != len(predictions_history):
            raise ValueError("Regime history length must match predictions history")
        
        # Group data by regime
        regime_data = {}
        for regime in MarketRegime:
            mask = np.array([r == regime for r in regime_history])
            if np.sum(mask) >= self.calibration_data_requirements:
                regime_data[regime] = {
                    'probabilities': predictions_history.loc[mask, 'raw_confidence'].values,
                    'labels': predictions_history.loc[mask, 'actual_outcome'].values
                }
        
        # Fit calibrator for each regime with sufficient data
        for regime, data in regime_data.items():
            try:
                calibrator = ProbabilityCalibrator([
                    CalibrationMethod.PLATT_SCALING,
                    CalibrationMethod.ISOTONIC_REGRESSION
                ])
                
                calibrator.fit(data['probabilities'], data['labels'])
                self.regime_calibrators[regime] = calibrator
                
                self.logger.info(f"Fitted calibrator for {regime.value} with {len(data['probabilities'])} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to fit calibrator for {regime.value}: {e}")
        
        # Fit fallback calibrator on all data
        all_probabilities = predictions_history['raw_confidence'].values
        all_labels = predictions_history['actual_outcome'].values
        
        try:
            self.fallback_calibrator.fit(all_probabilities, all_labels)
            self.logger.info("Fitted fallback calibrator on all regime data")
        except Exception as e:
            self.logger.error(f"Failed to fit fallback calibrator: {e}")
    
    def calibrate_for_regime(
        self,
        raw_confidence: float,
        regime: MarketRegime
    ) -> float:
        """Apply regime-specific calibration"""
        
        if regime in self.regime_calibrators:
            try:
                calibrated = self.regime_calibrators[regime].predict(np.array([raw_confidence]))[0]
                return calibrated
            except Exception as e:
                self.logger.warning(f"Regime-specific calibration failed for {regime.value}: {e}")
        
        # Fallback to general calibrator
        try:
            if self.fallback_calibrator.fitted:
                return self.fallback_calibrator.predict(np.array([raw_confidence]))[0]
            else:
                return raw_confidence
        except Exception as e:
            self.logger.warning(f"Fallback calibration failed: {e}")
            return raw_confidence

class RegimeAwareConfidenceSystem:
    """Complete integrated regime-aware confidence and uncertainty system"""
    
    def __init__(
        self,
        regime_gate_config: RegimeGateConfig = None,
        confidence_gate_config: ConfidenceGateConfig = None
    ):
        self.regime_gate_config = regime_gate_config or RegimeGateConfig()
        self.confidence_gate_config = confidence_gate_config or ConfidenceGateConfig()
        
        # Initialize components
        self.regime_detector = create_regime_detector(use_hmm=True, use_rules=True)
        self.regime_router = create_regime_aware_system(self.regime_detector)
        self.confidence_gate = CalibratedConfidenceGate(self.confidence_gate_config)
        self.regime_calibrator = RegimeAwareCalibrator()
        self.uncertainty_quantifier = None
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.prediction_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_system(
        self,
        historical_market_data: pd.DataFrame,
        historical_predictions: Optional[pd.DataFrame] = None,
        feature_data: Optional[np.ndarray] = None,
        target_data: Optional[np.ndarray] = None
    ):
        """Initialize all system components with historical data"""
        
        # Fit regime detector
        self.logger.info("Fitting regime detection system...")
        self.regime_detector.fit_historical_regimes(historical_market_data)
        
        # Train regime-aware models if training data provided
        if feature_data is not None and target_data is not None:
            self.logger.info("Training regime-aware models...")
            
            # Detect regimes for training data
            regime_labels = []
            for i in range(len(historical_market_data)):
                subset_data = historical_market_data.iloc[:i+1]
                if len(subset_data) > 30:
                    regime_state = self.regime_detector.detect_current_regime(subset_data)
                    regime_labels.append(regime_state.regime)
                else:
                    regime_labels.append(MarketRegime.UNKNOWN)
            
            # Filter out unknown regimes for training
            known_mask = np.array([r != MarketRegime.UNKNOWN for r in regime_labels])
            if np.sum(known_mask) > 100:
                X_filtered = feature_data[known_mask]
                y_filtered = target_data[known_mask]
                regime_labels_filtered = [r for i, r in enumerate(regime_labels) if known_mask[i]]
                
                self.regime_router.train_regime_models(X_filtered, y_filtered, regime_labels_filtered)
        
        # Initialize confidence gate calibration if historical predictions provided
        if historical_predictions is not None:
            self.logger.info("Updating confidence gate calibration...")
            self.confidence_gate.update_calibration(historical_predictions)
            
            # Fit regime-specific calibrators
            if 'regime' in historical_predictions.columns:
                regime_history = historical_predictions['regime'].tolist()
                self.regime_calibrator.fit_regime_calibrators(historical_predictions, regime_history)
        
        # Initialize uncertainty quantifier if training data provided
        if feature_data is not None and target_data is not None:
            self.logger.info("Training uncertainty quantification system...")
            try:
                from ml.uncertainty.bayesian_uncertainty import create_uncertainty_quantifier
                
                self.uncertainty_quantifier = create_uncertainty_quantifier(
                    model_type="ensemble_mc_dropout",
                    ensemble_size=3,
                    mc_samples=50
                )
                
                self.uncertainty_quantifier.build_model(
                    input_size=feature_data.shape[1],
                    hidden_sizes=[64, 32, 16]
                )
                
                # Use subset for quick training
                train_size = min(1000, len(feature_data))
                self.uncertainty_quantifier.train(
                    feature_data[:train_size],
                    target_data[:train_size],
                    epochs=30
                )
                
            except Exception as e:
                self.logger.error(f"Failed to initialize uncertainty quantifier: {e}")
                self.uncertainty_quantifier = None
    
    def predict_with_regime_awareness(
        self,
        predictions: List[Dict[str, Any]],
        current_market_data: pd.DataFrame,
        feature_data: Optional[np.ndarray] = None
    ) -> List[RegimeAwarePrediction]:
        """Make regime-aware predictions with calibrated confidence and uncertainty"""
        
        # Detect current regime
        regime_state = self.regime_detector.detect_current_regime(current_market_data)
        self.current_regime = regime_state.regime
        self.regime_confidence = regime_state.confidence
        
        self.logger.info(f"Current regime: {self.current_regime.value} (confidence: {self.regime_confidence:.3f})")
        
        regime_aware_predictions = []
        
        for i, pred_dict in enumerate(predictions):
            try:
                # Extract prediction data
                symbol = pred_dict.get('symbol', 'unknown')
                raw_prediction = pred_dict.get('prediction', 0.0)
                raw_confidence = pred_dict.get('confidence', 0.5)
                model_name = pred_dict.get('model_name', 'unknown')
                
                # Apply regime-specific calibration
                calibrated_confidence = self.regime_calibrator.calibrate_for_regime(
                    raw_confidence, self.current_regime
                )
                
                # Calculate regime-adjusted confidence threshold
                base_threshold = self.regime_gate_config.base_confidence_threshold
                regime_adjustment = self.regime_gate_config.regime_adjustments.get(self.current_regime, 0.0)
                adjusted_threshold = base_threshold + regime_adjustment
                
                # Apply regime-adjusted confidence
                regime_adjusted_confidence = calibrated_confidence * (1.0 - regime_adjustment * 0.1)
                
                # Estimate uncertainty
                epistemic_uncertainty = 0.0
                aleatoric_uncertainty = 0.0
                total_uncertainty = 0.0
                ci_lower = raw_prediction
                ci_upper = raw_prediction
                
                if self.uncertainty_quantifier and feature_data is not None and i < len(feature_data):
                    try:
                        uncertainty_est = self.uncertainty_quantifier.predict_with_uncertainty(
                            feature_data[i:i+1]
                        )
                        
                        if isinstance(uncertainty_est, list):
                            uncertainty_est = uncertainty_est[0]
                        
                        epistemic_uncertainty = uncertainty_est.epistemic_uncertainty
                        aleatoric_uncertainty = uncertainty_est.aleatoric_uncertainty
                        total_uncertainty = uncertainty_est.total_uncertainty
                        ci_lower = uncertainty_est.confidence_interval_lower
                        ci_upper = uncertainty_est.confidence_interval_upper
                        
                        # Apply regime-specific uncertainty scaling
                        uncertainty_scale = self.regime_gate_config.uncertainty_scaling.get(self.current_regime, 1.0)
                        total_uncertainty *= uncertainty_scale
                        
                    except Exception as e:
                        self.logger.warning(f"Uncertainty estimation failed for {symbol}: {e}")
                
                # Determine if passes regime-aware gate
                passes_regime_gate = (
                    regime_adjusted_confidence >= adjusted_threshold and
                    self.regime_confidence >= 0.6 and  # Minimum regime confidence
                    total_uncertainty < 0.5  # Maximum uncertainty threshold
                )
                
                # Create regime-aware prediction
                regime_prediction = RegimeAwarePrediction(
                    symbol=symbol,
                    prediction=raw_prediction,
                    raw_confidence=raw_confidence,
                    calibrated_confidence=calibrated_confidence,
                    regime=self.current_regime,
                    regime_confidence=self.regime_confidence,
                    epistemic_uncertainty=epistemic_uncertainty,
                    aleatoric_uncertainty=aleatoric_uncertainty,
                    total_uncertainty=total_uncertainty,
                    confidence_interval_lower=ci_lower,
                    confidence_interval_upper=ci_upper,
                    model_used=model_name,
                    regime_adjusted_confidence=regime_adjusted_confidence,
                    passes_regime_gate=passes_regime_gate
                )
                
                regime_aware_predictions.append(regime_prediction)
                
            except Exception as e:
                self.logger.error(f"Failed to process prediction for {pred_dict.get('symbol', 'unknown')}: {e}")
                continue
        
        # Store prediction history
        self.prediction_history.extend(regime_aware_predictions)
        
        # Keep only recent history (last 1000 predictions)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        return regime_aware_predictions
    
    def filter_regime_confident_predictions(
        self,
        regime_predictions: List[RegimeAwarePrediction],
        min_regime_confidence: float = 0.7,
        max_uncertainty: float = 0.3
    ) -> List[RegimeAwarePrediction]:
        """Filter predictions based on regime confidence and uncertainty"""
        
        filtered_predictions = []
        
        for pred in regime_predictions:
            if (pred.passes_regime_gate and
                pred.regime_confidence >= min_regime_confidence and
                pred.total_uncertainty <= max_uncertainty):
                filtered_predictions.append(pred)
        
        self.logger.info(f"Filtered {len(filtered_predictions)}/{len(regime_predictions)} predictions "
                        f"(regime conf >= {min_regime_confidence}, uncertainty <= {max_uncertainty})")
        
        return filtered_predictions
    
    def get_regime_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance by regime"""
        
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        # Group predictions by regime
        regime_groups = {}
        for pred in self.prediction_history:
            regime = pred.regime
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(pred)
        
        # Analyze each regime
        regime_analysis = {}
        
        for regime, predictions in regime_groups.items():
            if not predictions:
                continue
            
            # Calculate regime-specific metrics
            confidences = [p.calibrated_confidence for p in predictions]
            uncertainties = [p.total_uncertainty for p in predictions]
            gate_passes = [p.passes_regime_gate for p in predictions]
            
            regime_analysis[regime.value] = {
                'total_predictions': len(predictions),
                'gate_pass_rate': np.mean(gate_passes),
                'avg_confidence': np.mean(confidences),
                'avg_uncertainty': np.mean(uncertainties),
                'confidence_std': np.std(confidences),
                'uncertainty_std': np.std(uncertainties),
                'regime_adjustments': self.regime_gate_config.regime_adjustments.get(regime, 0.0)
            }
        
        # Overall system statistics
        total_predictions = len(self.prediction_history)
        total_gate_passes = sum(1 for p in self.prediction_history if p.passes_regime_gate)
        
        return {
            'regime_analysis': regime_analysis,
            'overall_stats': {
                'total_predictions': total_predictions,
                'overall_gate_pass_rate': total_gate_passes / max(total_predictions, 1),
                'current_regime': self.current_regime.value,
                'current_regime_confidence': self.regime_confidence,
                'regimes_encountered': len(regime_groups)
            },
            'calibration_status': {
                'regime_calibrators_fitted': len(self.regime_calibrator.regime_calibrators),
                'fallback_calibrator_fitted': self.regime_calibrator.fallback_calibrator.fitted,
                'uncertainty_quantifier_available': self.uncertainty_quantifier is not None
            }
        }
    
    def update_regime_gate_config(
        self,
        regime_adjustments: Optional[Dict[MarketRegime, float]] = None,
        uncertainty_scaling: Optional[Dict[MarketRegime, float]] = None
    ):
        """Update regime gate configuration based on performance"""
        
        if regime_adjustments:
            self.regime_gate_config.regime_adjustments.update(regime_adjustments)
            self.logger.info("Updated regime confidence adjustments")
        
        if uncertainty_scaling:
            self.regime_gate_config.uncertainty_scaling.update(uncertainty_scaling)
            self.logger.info("Updated regime uncertainty scaling")

def create_integrated_confidence_system(
    base_confidence_threshold: float = 0.80,
    regime_adjustments: Optional[Dict[str, float]] = None
) -> RegimeAwareConfidenceSystem:
    """Create integrated regime-aware confidence system"""
    
    # Convert string regime keys to MarketRegime enum
    regime_adj_enum = {}
    if regime_adjustments:
        for regime_str, adjustment in regime_adjustments.items():
            try:
                regime_enum = MarketRegime(regime_str)
                regime_adj_enum[regime_enum] = adjustment
            except ValueError:
                continue
    
    regime_config = RegimeGateConfig(
        base_confidence_threshold=base_confidence_threshold,
        regime_adjustments=regime_adj_enum
    )
    
    return RegimeAwareConfidenceSystem(regime_gate_config=regime_config)

def integrate_regime_awareness(
    predictions: List[Dict[str, Any]],
    market_data: pd.DataFrame,
    historical_data: Optional[pd.DataFrame] = None,
    feature_data: Optional[np.ndarray] = None,
    target_data: Optional[np.ndarray] = None
) -> Tuple[List[RegimeAwarePrediction], Dict[str, Any]]:
    """High-level function to apply regime-aware confidence filtering"""
    
    # Create integrated system
    system = create_integrated_confidence_system()
    
    # Initialize system if historical data provided
    if historical_data is not None:
        system.initialize_system(
            historical_market_data=historical_data,
            feature_data=feature_data,
            target_data=target_data
        )
    
    # Apply regime-aware prediction
    regime_predictions = system.predict_with_regime_awareness(
        predictions, market_data, feature_data
    )
    
    # Filter high-confidence predictions
    filtered_predictions = system.filter_regime_confident_predictions(regime_predictions)
    
    # Get performance analysis
    analysis = system.get_regime_performance_analysis()
    
    return filtered_predictions, analysis