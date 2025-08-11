#!/usr/bin/env python3
"""
Calibrated Confidence Gate
Enterprise confidence gating with properly calibrated probabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from ml.calibration.probability_calibrator import ProbabilityCalibrator, CalibrationMethod, CalibrationMetrics

@dataclass
class CalibratedPrediction:
    """Single calibrated prediction with confidence metrics"""
    symbol: str
    raw_prediction: float
    raw_confidence: float
    calibrated_confidence: float
    calibration_method: str
    prediction_timestamp: datetime
    model_name: str
    features_used: List[str] = field(default_factory=list)
    calibration_quality_score: float = 0.0
    uncertainty_lower: float = 0.0
    uncertainty_upper: float = 0.0
    passes_confidence_gate: bool = False

@dataclass
class ConfidenceGateConfig:
    """Configuration for calibrated confidence gate"""
    confidence_threshold: float = 0.80
    min_calibration_quality: float = 0.95  # ECE < 0.05
    min_samples_for_calibration: int = 100
    recalibration_frequency_days: int = 7
    uncertainty_estimation: bool = True
    bootstrap_samples: int = 1000
    calibration_methods: List[str] = field(default_factory=lambda: ["platt_scaling", "isotonic_regression"])

@dataclass
class ConfidenceGateReport:
    """Report from confidence gate filtering"""
    total_predictions: int
    passed_gate: int
    failed_confidence: int
    failed_calibration_quality: int
    gate_pass_rate: float
    average_calibrated_confidence: float
    calibration_quality_score: float
    calibration_method_used: str
    uncertainty_coverage: float
    recommendations: List[str] = field(default_factory=list)

class CalibratedConfidenceGate:
    """Enterprise-grade confidence gate with proper probability calibration"""
    
    def __init__(self, config: ConfidenceGateConfig = None):
        self.config = config or ConfidenceGateConfig()
        self.probability_calibrator = None
        self.calibration_history = []
        self.last_calibration = None
        self.calibration_quality_metrics = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize calibrator
        self.probability_calibrator = ProbabilityCalibrator(
            methods=[CalibrationMethod(m) for m in self.config.calibration_methods]
        )
        
        # Statistics tracking
        self.gate_statistics = {
            'total_processed': 0,
            'total_passed': 0,
            'calibration_updates': 0,
            'average_calibration_error': [],
            'confidence_distributions': [],
            'timestamp_last_update': None
        }
    
    def update_calibration(
        self,
        historical_predictions: pd.DataFrame,
        force_update: bool = False
    ) -> bool:
        """Update probability calibration using historical data"""
        
        required_columns = ['raw_confidence', 'actual_outcome', 'prediction_timestamp']
        if not all(col in historical_predictions.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False
        
        # Check if recalibration is needed
        if not force_update and self._should_skip_calibration():
            return True
        
        # Filter recent data for calibration
        recent_data = self._filter_recent_calibration_data(historical_predictions)
        
        if len(recent_data) < self.config.min_samples_for_calibration:
            self.logger.warning(f"Insufficient data for calibration: {len(recent_data)} < {self.config.min_samples_for_calibration}")
            return False
        
        try:
            # Extract probabilities and labels
            probabilities = recent_data['raw_confidence'].values
            true_labels = recent_data['actual_outcome'].values
            
            # Validate data quality
            if not self._validate_calibration_data(probabilities, true_labels):
                return False
            
            # Fit calibration
            calibration_results = self.probability_calibrator.fit(probabilities, true_labels)
            
            # Update calibration quality metrics
            self.calibration_quality_metrics = self.probability_calibrator.get_calibration_metrics(
                probabilities, true_labels
            )
            
            # Store calibration history
            self.calibration_history.append({
                'timestamp': datetime.utcnow(),
                'data_points': len(recent_data),
                'calibration_error': self.calibration_quality_metrics.expected_calibration_error,
                'best_method': self.probability_calibrator.best_method.value,
                'quality_score': 1.0 - self.calibration_quality_metrics.expected_calibration_error
            })
            
            self.last_calibration = datetime.utcnow()
            self.gate_statistics['calibration_updates'] += 1
            self.gate_statistics['timestamp_last_update'] = self.last_calibration
            
            # Log calibration results
            best_method = self.probability_calibrator.best_method
            best_result = calibration_results[best_method]
            
            self.logger.info(f"Calibration updated: method={best_method.value}, "
                           f"ECE={best_result.calibration_error:.4f}, "
                           f"well_calibrated={best_result.is_well_calibrated}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration update failed: {e}")
            return False
    
    def apply_confidence_gate(
        self,
        predictions: List[Dict[str, Any]],
        require_calibration: bool = True
    ) -> Tuple[List[CalibratedPrediction], ConfidenceGateReport]:
        """Apply calibrated confidence gate to predictions"""
        
        # Check calibration status
        if require_calibration and not self._is_calibration_valid():
            self.logger.error("Confidence gate requires valid calibration")
            return [], self._create_empty_report()
        
        calibrated_predictions = []
        passed_gate = 0
        failed_confidence = 0
        failed_calibration_quality = 0
        
        for pred_dict in predictions:
            try:
                # Create calibrated prediction
                calibrated_pred = self._create_calibrated_prediction(pred_dict)
                calibrated_predictions.append(calibrated_pred)
                
                # Apply gate filtering
                if calibrated_pred.passes_confidence_gate:
                    passed_gate += 1
                else:
                    if calibrated_pred.calibrated_confidence < self.config.confidence_threshold:
                        failed_confidence += 1
                    if calibrated_pred.calibration_quality_score < self.config.min_calibration_quality:
                        failed_calibration_quality += 1
                
            except Exception as e:
                self.logger.error(f"Error processing prediction: {e}")
                continue
        
        # Update statistics
        self.gate_statistics['total_processed'] += len(predictions)
        self.gate_statistics['total_passed'] += passed_gate
        
        # Create report
        report = self._create_gate_report(
            total_predictions=len(predictions),
            passed_gate=passed_gate,
            failed_confidence=failed_confidence,
            failed_calibration_quality=failed_calibration_quality,
            calibrated_predictions=calibrated_predictions
        )
        
        return calibrated_predictions, report
    
    def _create_calibrated_prediction(self, pred_dict: Dict[str, Any]) -> CalibratedPrediction:
        """Create calibrated prediction from raw prediction data"""
        
        # Extract raw data
        symbol = pred_dict.get('symbol', 'unknown')
        raw_prediction = pred_dict.get('prediction', 0.0)
        raw_confidence = pred_dict.get('confidence', 0.5)
        model_name = pred_dict.get('model_name', 'unknown')
        features_used = pred_dict.get('features', [])
        
        # Apply calibration if available
        if self.probability_calibrator and self.probability_calibrator.fitted:
            try:
                calibrated_confidence = self.probability_calibrator.predict(np.array([raw_confidence]))[0]
                calibration_method = self.probability_calibrator.best_method.value
                
                # Calculate calibration quality score
                if self.calibration_quality_metrics:
                    calibration_quality_score = 1.0 - self.calibration_quality_metrics.expected_calibration_error
                else:
                    calibration_quality_score = 0.5
                
            except Exception as e:
                self.logger.warning(f"Calibration failed for {symbol}: {e}")
                calibrated_confidence = raw_confidence
                calibration_method = "none"
                calibration_quality_score = 0.0
        else:
            calibrated_confidence = raw_confidence
            calibration_method = "none" 
            calibration_quality_score = 0.0
        
        # Estimate uncertainty using bootstrap if enabled
        uncertainty_lower, uncertainty_upper = self._estimate_uncertainty(
            raw_confidence, calibrated_confidence
        )
        
        # Check if passes confidence gate
        passes_gate = (
            calibrated_confidence >= self.config.confidence_threshold and
            calibration_quality_score >= self.config.min_calibration_quality
        )
        
        return CalibratedPrediction(
            symbol=symbol,
            raw_prediction=raw_prediction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            calibration_method=calibration_method,
            prediction_timestamp=datetime.utcnow(),
            model_name=model_name,
            features_used=features_used,
            calibration_quality_score=calibration_quality_score,
            uncertainty_lower=uncertainty_lower,
            uncertainty_upper=uncertainty_upper,
            passes_confidence_gate=passes_gate
        )
    
    def _estimate_uncertainty(
        self, 
        raw_confidence: float, 
        calibrated_confidence: float
    ) -> Tuple[float, float]:
        """Estimate uncertainty bounds for calibrated confidence"""
        
        if not self.config.uncertainty_estimation:
            return calibrated_confidence, calibrated_confidence
        
        try:
            # Use calibration quality to estimate uncertainty
            if self.calibration_quality_metrics:
                calibration_error = self.calibration_quality_metrics.expected_calibration_error
                
                # Uncertainty bounds based on calibration error
                uncertainty_margin = calibration_error * 2  # 2x calibration error as margin
                
                lower_bound = max(0.0, calibrated_confidence - uncertainty_margin)
                upper_bound = min(1.0, calibrated_confidence + uncertainty_margin)
                
                return lower_bound, upper_bound
            else:
                # Fallback: use difference between raw and calibrated as uncertainty
                uncertainty_margin = abs(calibrated_confidence - raw_confidence) * 1.5
                
                lower_bound = max(0.0, calibrated_confidence - uncertainty_margin)
                upper_bound = min(1.0, calibrated_confidence + uncertainty_margin)
                
                return lower_bound, upper_bound
                
        except Exception as e:
            self.logger.warning(f"Uncertainty estimation failed: {e}")
            return calibrated_confidence, calibrated_confidence
    
    def _should_skip_calibration(self) -> bool:
        """Check if calibration update should be skipped"""
        
        if self.last_calibration is None:
            return False
        
        days_since_calibration = (datetime.utcnow() - self.last_calibration).days
        return days_since_calibration < self.config.recalibration_frequency_days
    
    def _filter_recent_calibration_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to recent period for calibration"""
        
        # Use last 30 days of data for calibration
        cutoff_date = datetime.utcnow() - pd.Timedelta(days=30)
        
        if 'prediction_timestamp' in data.columns:
            recent_data = data[data['prediction_timestamp'] >= cutoff_date]
        else:
            # If no timestamp, use most recent data
            recent_data = data.tail(self.config.min_samples_for_calibration * 2)
        
        return recent_data
    
    def _validate_calibration_data(
        self, 
        probabilities: np.ndarray, 
        labels: np.ndarray
    ) -> bool:
        """Validate data quality for calibration"""
        
        # Check for valid probability range
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            self.logger.error("Probabilities outside [0,1] range")
            return False
        
        # Check for binary labels
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2 or not set(unique_labels).issubset({0, 1}):
            self.logger.error("Labels must be binary (0/1)")
            return False
        
        # Check for sufficient diversity
        if len(unique_labels) < 2:
            self.logger.error("Need both positive and negative examples")
            return False
        
        # Check for reasonable balance
        positive_rate = np.mean(labels)
        if positive_rate < 0.05 or positive_rate > 0.95:
            self.logger.warning(f"Highly imbalanced data: {positive_rate:.1%} positive")
        
        return True
    
    def _is_calibration_valid(self) -> bool:
        """Check if current calibration is valid"""
        
        if not self.probability_calibrator or not self.probability_calibrator.fitted:
            return False
        
        if self.calibration_quality_metrics is None:
            return False
        
        # Check calibration quality
        quality_score = 1.0 - self.calibration_quality_metrics.expected_calibration_error
        if quality_score < self.config.min_calibration_quality:
            return False
        
        # Check recency
        if self.last_calibration is None:
            return False
        
        days_since_calibration = (datetime.utcnow() - self.last_calibration).days
        if days_since_calibration > self.config.recalibration_frequency_days * 2:
            return False
        
        return True
    
    def _create_gate_report(
        self,
        total_predictions: int,
        passed_gate: int,
        failed_confidence: int,
        failed_calibration_quality: int,
        calibrated_predictions: List[CalibratedPrediction]
    ) -> ConfidenceGateReport:
        """Create comprehensive gate report"""
        
        gate_pass_rate = passed_gate / max(total_predictions, 1)
        
        # Calculate average calibrated confidence
        if calibrated_predictions:
            avg_calibrated_conf = np.mean([p.calibrated_confidence for p in calibrated_predictions])
            
            # Calculate uncertainty coverage
            uncertainties = [(p.uncertainty_upper - p.uncertainty_lower) for p in calibrated_predictions]
            uncertainty_coverage = np.mean(uncertainties) if uncertainties else 0.0
        else:
            avg_calibrated_conf = 0.0
            uncertainty_coverage = 0.0
        
        # Get calibration info
        if self.calibration_quality_metrics:
            calibration_quality_score = 1.0 - self.calibration_quality_metrics.expected_calibration_error
        else:
            calibration_quality_score = 0.0
        
        calibration_method = self.probability_calibrator.best_method.value if self.probability_calibrator.fitted else "none"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            gate_pass_rate, calibration_quality_score, failed_confidence, failed_calibration_quality
        )
        
        return ConfidenceGateReport(
            total_predictions=total_predictions,
            passed_gate=passed_gate,
            failed_confidence=failed_confidence,
            failed_calibration_quality=failed_calibration_quality,
            gate_pass_rate=gate_pass_rate,
            average_calibrated_confidence=avg_calibrated_conf,
            calibration_quality_score=calibration_quality_score,
            calibration_method_used=calibration_method,
            uncertainty_coverage=uncertainty_coverage,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        gate_pass_rate: float,
        calibration_quality_score: float,
        failed_confidence: int,
        failed_calibration_quality: int
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if gate_pass_rate < 0.1:
            recommendations.append("Very low gate pass rate - consider lowering confidence threshold or improving model performance")
        
        if calibration_quality_score < 0.9:
            recommendations.append("Poor calibration quality - collect more diverse training data and retrain calibrator")
        
        if failed_confidence > failed_calibration_quality * 2:
            recommendations.append("Most failures due to low confidence - focus on improving model confidence estimation")
        
        if failed_calibration_quality > failed_confidence * 2:
            recommendations.append("Most failures due to poor calibration - focus on improving calibration quality")
        
        if len(self.calibration_history) > 1:
            recent_error = self.calibration_history[-1]['calibration_error']
            previous_error = self.calibration_history[-2]['calibration_error']
            
            if recent_error > previous_error * 1.2:
                recommendations.append("Calibration quality degrading - check for data drift or concept drift")
        
        if not recommendations:
            recommendations.append("Confidence gate operating within acceptable parameters")
        
        return recommendations
    
    def _create_empty_report(self) -> ConfidenceGateReport:
        """Create empty report for error cases"""
        
        return ConfidenceGateReport(
            total_predictions=0,
            passed_gate=0,
            failed_confidence=0,
            failed_calibration_quality=0,
            gate_pass_rate=0.0,
            average_calibrated_confidence=0.0,
            calibration_quality_score=0.0,
            calibration_method_used="none",
            uncertainty_coverage=0.0,
            recommendations=["Error in confidence gate processing"]
        )
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gate statistics"""
        
        total_processed = self.gate_statistics['total_processed']
        
        stats = {
            'total_processed': total_processed,
            'total_passed': self.gate_statistics['total_passed'],
            'overall_pass_rate': self.gate_statistics['total_passed'] / max(total_processed, 1),
            'calibration_updates': self.gate_statistics['calibration_updates'],
            'last_calibration': self.last_calibration,
            'current_calibration_method': self.probability_calibrator.best_method.value if self.probability_calibrator.fitted else "none"
        }
        
        # Add calibration quality info
        if self.calibration_quality_metrics:
            stats.update({
                'expected_calibration_error': self.calibration_quality_metrics.expected_calibration_error,
                'calibration_quality_score': 1.0 - self.calibration_quality_metrics.expected_calibration_error,
                'overconfidence_error': self.calibration_quality_metrics.overconfidence_error,
                'underconfidence_error': self.calibration_quality_metrics.underconfidence_error
            })
        
        # Add calibration history summary
        if self.calibration_history:
            recent_errors = [h['calibration_error'] for h in self.calibration_history[-5:]]
            stats['calibration_error_trend'] = {
                'recent_avg': np.mean(recent_errors),
                'recent_std': np.std(recent_errors),
                'improving': len(recent_errors) > 1 and recent_errors[-1] < recent_errors[0]
            }
        
        return stats

def create_calibrated_confidence_gate(
    confidence_threshold: float = 0.80,
    calibration_methods: List[str] = None
) -> CalibratedConfidenceGate:
    """Create calibrated confidence gate with specified configuration"""
    
    if calibration_methods is None:
        calibration_methods = ["platt_scaling", "isotonic_regression", "temperature_scaling"]
    
    config = ConfidenceGateConfig(
        confidence_threshold=confidence_threshold,
        calibration_methods=calibration_methods
    )
    
    return CalibratedConfidenceGate(config)

def apply_calibrated_gate(
    predictions: List[Dict[str, Any]],
    historical_data: pd.DataFrame,
    confidence_threshold: float = 0.80
) -> Tuple[List[CalibratedPrediction], ConfidenceGateReport]:
    """High-level function to apply calibrated confidence gate"""
    
    gate = create_calibrated_confidence_gate(confidence_threshold)
    
    # Update calibration if historical data provided
    if not historical_data.empty:
        gate.update_calibration(historical_data)
    
    # Apply gate
    return gate.apply_confidence_gate(predictions)