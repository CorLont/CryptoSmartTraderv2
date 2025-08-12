"""
Confidence-Weighted Position Sizing Integration

Combines probability calibration with Kelly sizing for optimal position sizing
based on model confidence and regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .probability_calibration import ProbabilityCalibrator
from .kelly_sizing import KellySizer, KellyMode, KellyParameters

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceWeightedSizing:
    """Result of confidence-weighted sizing calculation"""
    position_size: float        # Final position size (%)
    raw_ml_confidence: float   # Original ML model confidence
    calibrated_probability: float  # Calibrated win probability
    kelly_fraction: float      # Raw Kelly fraction
    final_size_reasoning: str  # Explanation of sizing decision
    risk_assessment: str       # Risk level assessment
    should_trade: bool        # Trading recommendation
    regime_adjustment: float   # Regime-based adjustment factor
    

class ConfidenceWeighter:
    """
    Integrates probability calibration with Kelly sizing for 
    confidence-weighted position sizing
    """
    
    def __init__(self, 
                 kelly_mode: KellyMode = KellyMode.MODERATE,
                 max_position: float = 20.0,
                 min_confidence: float = 0.6,
                 calibration_path: Optional[str] = None):
        """
        Initialize confidence-weighted sizer
        
        Args:
            kelly_mode: Kelly sizing aggressiveness
            max_position: Maximum position size (%)
            min_confidence: Minimum confidence for trading
            calibration_path: Path to saved probability calibrator
        """
        self.kelly_sizer = KellySizer(
            mode=kelly_mode,
            max_position=max_position,
            min_position=0.5,
            confidence_threshold=min_confidence
        )
        
        self.probability_calibrator = ProbabilityCalibrator()
        self.min_confidence = min_confidence
        self.max_position = max_position
        
        # Load calibrator if path provided
        if calibration_path:
            self.probability_calibrator.load_calibrator(calibration_path)
        
        # Track performance for adaptation
        self.sizing_performance = []
        
    def calculate_position_size(self,
                              ml_confidence: float,
                              predicted_direction: str,  # 'up' or 'down'
                              expected_return: float,
                              expected_loss: float,
                              regime_factor: float = 1.0,
                              market_conditions: Optional[Dict[str, float]] = None) -> ConfidenceWeightedSizing:
        """
        Calculate optimal position size using confidence-weighted Kelly
        
        Args:
            ml_confidence: Raw ML model confidence (0-1)
            predicted_direction: 'up' or 'down'
            expected_return: Expected return if trade wins (%)
            expected_loss: Expected loss if trade loses (%) - positive value
            regime_factor: Regime-based sizing adjustment (0.5-2.0)
            market_conditions: Additional market factors
            
        Returns:
            Complete sizing recommendation
        """
        try:
            # Step 1: Calibrate probability
            calibrated_prob = self._calibrate_confidence(ml_confidence)
            
            # Step 2: Calculate payoff ratio
            payoff_ratio = expected_return / expected_loss if expected_loss > 0 else 1.0
            
            # Step 3: Apply market condition adjustments
            adjusted_confidence, adjusted_regime = self._apply_market_adjustments(
                calibrated_prob, regime_factor, market_conditions
            )
            
            # Step 4: Calculate Kelly sizing
            kelly_result = self.kelly_sizer.calculate_kelly_size(
                win_probability=adjusted_confidence,
                payoff_ratio=payoff_ratio,
                confidence=ml_confidence,
                regime_factor=adjusted_regime
            )
            
            # Step 5: Generate final recommendation
            recommendation = self._generate_final_recommendation(
                kelly_result, ml_confidence, calibrated_prob, 
                adjusted_confidence, payoff_ratio, adjusted_regime
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Confidence-weighted sizing failed: {e}")
            return self._error_recommendation(str(e))
    
    def batch_size_signals(self, 
                          signals: List[Dict[str, Any]]) -> List[ConfidenceWeightedSizing]:
        """
        Calculate position sizes for multiple trading signals
        
        Args:
            signals: List of signal dicts with required keys:
                    'ml_confidence', 'direction', 'expected_return', 'expected_loss'
                    Optional: 'regime_factor', 'market_conditions'
            
        Returns:
            List of sizing recommendations
        """
        recommendations = []
        
        for i, signal in enumerate(signals):
            try:
                rec = self.calculate_position_size(
                    ml_confidence=signal.get('ml_confidence', 0.5),
                    predicted_direction=signal.get('direction', 'up'),
                    expected_return=signal.get('expected_return', 2.0),
                    expected_loss=signal.get('expected_loss', 1.0),
                    regime_factor=signal.get('regime_factor', 1.0),
                    market_conditions=signal.get('market_conditions')
                )
                recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Batch sizing failed for signal {i}: {e}")
                recommendations.append(self._error_recommendation(f"Signal {i}: {e}"))
        
        return recommendations
    
    def train_calibration(self, 
                         historical_confidences: List[float],
                         historical_outcomes: List[bool]) -> Dict[str, Any]:
        """
        Train probability calibration on historical data
        
        Args:
            historical_confidences: List of past ML confidences
            historical_outcomes: List of trade outcomes (True=win, False=loss)
            
        Returns:
            Calibration training results
        """
        try:
            # Convert to numpy arrays
            confidences = np.array(historical_confidences)
            outcomes = np.array([1 if outcome else 0 for outcome in historical_outcomes])
            
            # Train calibrator
            results = self.probability_calibrator.fit(confidences, outcomes)
            
            logger.info(f"Calibration training completed. Method: {results.get('best_method')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Calibration training failed: {e}")
            return {"error": str(e)}
    
    def optimize_kelly_parameters(self,
                                historical_signals: List[Dict[str, Any]],
                                historical_returns: List[float]) -> Dict[str, Any]:
        """
        Optimize Kelly parameters based on historical performance
        
        Args:
            historical_signals: Past signals with confidence, returns, etc.
            historical_returns: Corresponding trade returns (%)
            
        Returns:
            Optimization results and recommended parameters
        """
        try:
            # Prepare data for Kelly optimization
            predictions = []
            
            for signal in historical_signals:
                # Calculate calibrated probability
                calibrated_prob = self._calibrate_confidence(
                    signal.get('ml_confidence', 0.5)
                )
                
                # Calculate payoff ratio
                expected_return = signal.get('expected_return', 2.0)
                expected_loss = signal.get('expected_loss', 1.0)
                payoff_ratio = expected_return / expected_loss
                
                predictions.append({
                    'win_prob': calibrated_prob,
                    'payoff_ratio': payoff_ratio,
                    'confidence': signal.get('ml_confidence', 0.5),
                    'regime_factor': signal.get('regime_factor', 1.0)
                })
            
            # Optimize Kelly fraction factor
            optimization_results = self.kelly_sizer.optimize_fraction_factor(
                predictions, historical_returns
            )
            
            # Update Kelly sizer with optimal parameters
            if 'optimal_fraction' in optimization_results:
                optimal_fraction = optimization_results['optimal_fraction']
                
                # Map fraction back to Kelly mode
                if optimal_fraction <= 0.25:
                    recommended_mode = KellyMode.CONSERVATIVE
                elif optimal_fraction <= 0.4:
                    recommended_mode = KellyMode.MODERATE
                else:
                    recommended_mode = KellyMode.AGGRESSIVE
                
                optimization_results['recommended_mode'] = recommended_mode
                
                logger.info(f"Kelly optimization completed. "
                           f"Optimal fraction: {optimal_fraction:.3f}, "
                           f"Recommended mode: {recommended_mode.value}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}")
            return {"error": str(e)}
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply probability calibration to raw ML confidence"""
        try:
            if not self.probability_calibrator.is_fitted:
                logger.warning("Probability calibrator not trained, using raw confidence")
                return raw_confidence
            
            calibrated = self.probability_calibrator.calibrate(np.array([raw_confidence]))
            return float(calibrated[0])
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return raw_confidence
    
    def _apply_market_adjustments(self, 
                                 calibrated_prob: float,
                                 regime_factor: float,
                                 market_conditions: Optional[Dict[str, float]]) -> Tuple[float, float]:
        """Apply market condition adjustments to probability and regime factor"""
        try:
            adjusted_prob = calibrated_prob
            adjusted_regime = regime_factor
            
            if market_conditions:
                # Volatility adjustment
                volatility = market_conditions.get('volatility', 0.5)
                if volatility > 0.8:  # High volatility
                    adjusted_prob *= 0.9  # Reduce confidence
                    adjusted_regime *= 0.7  # Reduce position size
                elif volatility < 0.2:  # Low volatility
                    adjusted_prob *= 1.05  # Slight confidence boost
                    adjusted_regime *= 1.1  # Slight size increase
                
                # Liquidity adjustment
                liquidity = market_conditions.get('liquidity', 0.8)
                if liquidity < 0.5:  # Low liquidity
                    adjusted_regime *= 0.8  # Reduce size
                
                # Market correlation adjustment
                correlation = market_conditions.get('correlation', 0.3)
                if correlation > 0.9:  # Everything moving together
                    adjusted_regime *= 0.85  # Reduce size (systemic risk)
            
            # Ensure bounds
            adjusted_prob = max(0.05, min(0.95, adjusted_prob))
            adjusted_regime = max(0.1, min(3.0, adjusted_regime))
            
            return adjusted_prob, adjusted_regime
            
        except Exception as e:
            logger.error(f"Market adjustment failed: {e}")
            return calibrated_prob, regime_factor
    
    def _generate_final_recommendation(self,
                                     kelly_result: Dict[str, Any],
                                     raw_confidence: float,
                                     calibrated_prob: float,
                                     adjusted_confidence: float,
                                     payoff_ratio: float,
                                     regime_factor: float) -> ConfidenceWeightedSizing:
        """Generate final sizing recommendation"""
        
        try:
            position_size = kelly_result.get('position_size', 0.0)
            should_trade = kelly_result.get('should_trade', False)
            risk_level = kelly_result.get('risk_level', 'high')
            
            # Generate reasoning
            reasoning_parts = []
            
            if raw_confidence != calibrated_prob:
                reasoning_parts.append(
                    f"Confidence calibrated: {raw_confidence:.3f} → {calibrated_prob:.3f}"
                )
            
            if regime_factor != 1.0:
                reasoning_parts.append(f"Regime adjustment: {regime_factor:.2f}x")
            
            if position_size == 0:
                reasoning_parts.append("No position: " + kelly_result.get('reason', 'Unknown'))
            elif position_size >= self.max_position * 0.8:
                reasoning_parts.append("Large position: High conviction signal")
            else:
                reasoning_parts.append(
                    f"Kelly sizing: {kelly_result.get('kelly_fraction', 0):.3f} fraction"
                )
            
            final_reasoning = "; ".join(reasoning_parts)
            
            return ConfidenceWeightedSizing(
                position_size=position_size,
                raw_ml_confidence=raw_confidence,
                calibrated_probability=calibrated_prob,
                kelly_fraction=kelly_result.get('kelly_fraction', 0.0),
                final_size_reasoning=final_reasoning,
                risk_assessment=risk_level,
                should_trade=should_trade,
                regime_adjustment=regime_factor
            )
            
        except Exception as e:
            logger.error(f"Final recommendation generation failed: {e}")
            return self._error_recommendation(str(e))
    
    def _error_recommendation(self, error_msg: str) -> ConfidenceWeightedSizing:
        """Generate error recommendation"""
        return ConfidenceWeightedSizing(
            position_size=0.0,
            raw_ml_confidence=0.0,
            calibrated_probability=0.0,
            kelly_fraction=0.0,
            final_size_reasoning=f"Error: {error_msg}",
            risk_assessment="high",
            should_trade=False,
            regime_adjustment=1.0
        )
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of sizing system status"""
        try:
            kelly_analytics = self.kelly_sizer.get_sizing_analytics()
            calibration_summary = self.probability_calibrator.get_calibration_summary()
            
            return {
                "calibration_status": calibration_summary.get("status", "Not fitted"),
                "calibration_method": calibration_summary.get("best_method", "None"),
                "kelly_mode": self.kelly_sizer.mode.value,
                "kelly_fraction": self.kelly_sizer.default_fraction,
                "max_position": self.max_position,
                "min_confidence": self.min_confidence,
                "recent_recommendations": kelly_analytics.get("total_recommendations", 0),
                "avg_position_size": kelly_analytics.get("avg_position_size", 0),
                "trade_rate": kelly_analytics.get("trade_rate", 0)
            }
            
        except Exception as e:
            logger.error(f"Sizing summary generation failed: {e}")
            return {"status": "Error", "error": str(e)}