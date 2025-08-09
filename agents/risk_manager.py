# agents/risk_manager.py - Risk management and false positive detection
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskManagerAgent:
    """Risk management with false positive alert system"""
    
    def __init__(self):
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% max per position
            'max_correlation': 0.7,    # Max correlation between positions
            'max_drawdown': 0.15,      # 15% max drawdown
            'min_liquidity': 1000000   # $1M minimum daily volume
        }
        
        self.false_positive_history = []
        
    async def analyze_risk_metrics(self):
        """Analyze current risk metrics"""
        try:
            # Load current predictions
            predictions_path = Path("exports/production/predictions.json")
            if not predictions_path.exists():
                return {}
            
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
            
            if not predictions:
                return {}
            
            # Risk analysis
            risk_metrics = {
                'total_positions': len(predictions),
                'high_risk_positions': len([p for p in predictions if p['risk_level'] in ['HIGH', 'EXTREME']]),
                'avg_confidence': np.mean([p['confidence'] for p in predictions]),
                'concentration_risk': self._calculate_concentration_risk(predictions),
                'false_positive_rate': self._calculate_false_positive_rate(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Check risk limits
            risk_alerts = []
            
            if risk_metrics['high_risk_positions'] / max(1, risk_metrics['total_positions']) > 0.3:
                risk_alerts.append("HIGH_RISK_CONCENTRATION")
            
            if risk_metrics['avg_confidence'] < 75:
                risk_alerts.append("LOW_CONFIDENCE_AVERAGE")
            
            if risk_metrics['false_positive_rate'] > 0.15:
                risk_alerts.append("HIGH_FALSE_POSITIVE_RATE")
            
            risk_metrics['alerts'] = risk_alerts
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {}
    
    def _calculate_concentration_risk(self, predictions):
        """Calculate concentration risk across predictions"""
        if len(predictions) <= 1:
            return 0.0
        
        # Simple concentration based on position distribution
        position_sizes = [1 / len(predictions)] * len(predictions)  # Equal weight assumption
        hhi = sum(p**2 for p in position_sizes)  # Herfindahl-Hirschman Index
        
        # Normalize (0 = perfect diversification, 1 = full concentration)
        return (hhi - 1/len(predictions)) / (1 - 1/len(predictions))
    
    def _calculate_false_positive_rate(self):
        """Calculate false positive rate from historical data"""
        if len(self.false_positive_history) < 10:
            return 0.0  # Not enough data
        
        recent_alerts = self.false_positive_history[-50:]  # Last 50 alerts
        false_positives = sum(1 for alert in recent_alerts if alert.get('was_false_positive', False))
        
        return false_positives / len(recent_alerts)
    
    async def detect_false_positives(self, predictions):
        """Detect potential false positive signals"""
        false_positive_indicators = []
        
        for pred in predictions:
            # Check for unrealistic returns
            if pred['expected_return'] > 500:  # >500% return
                false_positive_indicators.append({
                    'symbol': pred['symbol'],
                    'reason': 'UNREALISTIC_RETURN',
                    'value': pred['expected_return'],
                    'threshold': 500
                })
            
            # Check for low volume (high slippage risk)
            # In real implementation, would check actual volume data
            if pred['confidence'] > 90 and pred['expected_return'] > 100:
                # Suspicious: very high confidence + very high return
                false_positive_indicators.append({
                    'symbol': pred['symbol'],
                    'reason': 'SUSPICIOUS_HIGH_CONFIDENCE',
                    'confidence': pred['confidence'],
                    'return': pred['expected_return']
                })
        
        return false_positive_indicators
    
    async def save_risk_report(self, risk_metrics, false_positives):
        """Save comprehensive risk report"""
        risk_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'risk_metrics': risk_metrics,
            'false_positives': false_positives,
            'risk_score': self._calculate_overall_risk_score(risk_metrics),
            'recommendations': self._generate_risk_recommendations(risk_metrics, false_positives)
        }
        
        # Save risk report
        Path("logs/risk").mkdir(parents=True, exist_ok=True)
        with open("logs/risk/latest_risk_report.json", 'w') as f:
            json.dump(risk_report, f, indent=2)
        
        # Update false positive history
        for fp in false_positives:
            self.false_positive_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'indicator': fp,
                'was_false_positive': None  # To be updated later with actual results
            })
        
        # Keep only last 100 entries
        self.false_positive_history = self.false_positive_history[-100:]
        
        return risk_report
    
    def _calculate_overall_risk_score(self, metrics):
        """Calculate overall risk score (0-100, lower is better)"""
        if not metrics:
            return 50  # Neutral when no data
        
        risk_factors = [
            metrics.get('concentration_risk', 0) * 30,
            (1 - metrics.get('avg_confidence', 80)/100) * 25,
            metrics.get('false_positive_rate', 0) * 25,
            len(metrics.get('alerts', [])) * 5
        ]
        
        return min(100, sum(risk_factors))
    
    def _generate_risk_recommendations(self, metrics, false_positives):
        """Generate actionable risk recommendations"""
        recommendations = []
        
        if metrics.get('concentration_risk', 0) > 0.5:
            recommendations.append("Reduce position concentration - diversify across more assets")
        
        if metrics.get('avg_confidence', 0) < 75:
            recommendations.append("Increase minimum confidence threshold - current predictions too uncertain")
        
        if false_positives:
            recommendations.append(f"Review {len(false_positives)} potential false positive signals")
        
        if len(metrics.get('alerts', [])) > 2:
            recommendations.append("Multiple risk alerts active - consider reducing position sizes")
        
        return recommendations
    
    async def run_continuous(self):
        """Run risk management continuously"""
        while True:
            try:
                logger.info("Running risk management cycle...")
                
                risk_metrics = await self.analyze_risk_metrics()
                
                if risk_metrics:
                    # Load predictions for false positive analysis
                    predictions_path = Path("exports/production/predictions.json")
                    if predictions_path.exists():
                        with open(predictions_path, 'r') as f:
                            predictions = json.load(f)
                        
                        false_positives = await self.detect_false_positives(predictions)
                        risk_report = await self.save_risk_report(risk_metrics, false_positives)
                        
                        # Log significant risk events
                        if risk_report['risk_score'] > 70:
                            logger.warning(f"High risk detected: Score {risk_report['risk_score']:.1f}")
                        elif false_positives:
                            logger.warning(f"Detected {len(false_positives)} potential false positives")
                
            except Exception as e:
                logger.error(f"Risk management cycle failed: {e}")
            
            await asyncio.sleep(180)  # 3 minutes

if __name__ == "__main__":
    agent = RiskManagerAgent()
    asyncio.run(agent.run_continuous())