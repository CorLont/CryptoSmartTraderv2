#!/usr/bin/env python3
"""
12-Week Training Requirement Enforcer
Blocks confidence scores and trades until models have trained for 12+ weeks
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingEnforcer:
    """Enforce 12-week minimum training requirement"""
    
    def __init__(self):
        self.required_weeks = 12
        self.model_dir = Path("models/saved")
        self.status_file = Path("training_status.json")
        
    def check_training_duration(self) -> Dict[str, Any]:
        """Check if models meet 12-week training requirement"""
        
        if not self.model_dir.exists():
            return {
                'status': 'no_models',
                'training_complete': False,
                'weeks_trained': 0,
                'message': 'No trained models found'
            }
        
        # Find oldest model (represents start of training)
        oldest_date = None
        model_count = 0
        
        for model_file in self.model_dir.glob("*.pkl"):
            model_count += 1
            file_stat = model_file.stat()
            training_date = datetime.fromtimestamp(file_stat.st_mtime)
            
            if oldest_date is None or training_date < oldest_date:
                oldest_date = training_date
        
        if oldest_date is None:
            return {
                'status': 'no_models',
                'training_complete': False,
                'weeks_trained': 0,
                'message': 'No model files found'
            }
        
        # Calculate training duration
        weeks_trained = (datetime.now() - oldest_date).days / 7
        training_complete = weeks_trained >= self.required_weeks
        
        status = {
            'status': 'training_complete' if training_complete else 'training_in_progress',
            'training_complete': training_complete,
            'weeks_trained': round(weeks_trained, 1),
            'weeks_required': self.required_weeks,
            'weeks_remaining': max(0, round(self.required_weeks - weeks_trained, 1)),
            'oldest_model_date': oldest_date.isoformat(),
            'model_count': model_count,
            'estimated_completion': (oldest_date + timedelta(weeks=self.required_weeks)).isoformat(),
            'message': self._get_status_message(training_complete, weeks_trained)
        }
        
        return status
    
    def _get_status_message(self, complete: bool, weeks: float) -> str:
        """Get human-readable status message"""
        
        if complete:
            return f"✅ Training complete ({weeks:.1f} weeks) - Confidence scores and trades ENABLED"
        else:
            remaining = self.required_weeks - weeks
            return f"⏳ Training in progress ({weeks:.1f}/{self.required_weeks} weeks) - {remaining:.1f} weeks remaining"
    
    def enforce_trading_block(self) -> bool:
        """Check if trading should be blocked due to insufficient training"""
        
        status = self.check_training_duration()
        
        if not status['training_complete']:
            logger.warning(f"🚫 TRADING BLOCKED: {status['message']}")
            return True  # Block trading
        else:
            logger.info(f"✅ TRADING ENABLED: {status['message']}")
            return False  # Allow trading
    
    def save_status(self):
        """Save training status to file"""
        
        status = self.check_training_duration()
        status['last_checked'] = datetime.now().isoformat()
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"💾 Training status saved: {self.status_file}")
        
        return status
    
    def create_training_gate(self) -> Dict[str, Any]:
        """Create comprehensive training gate for production"""
        
        status = self.save_status()
        
        # Create gate configuration
        gate_config = {
            'confidence_scores_enabled': status['training_complete'],
            'trading_enabled': status['training_complete'],
            'predictions_enabled': True,  # Basic predictions always allowed
            'advanced_features_enabled': status['training_complete'],
            'reason': status['message'],
            'enforcement_active': True
        }
        
        # Save gate config
        gate_file = Path("production_gate_config.json")
        with open(gate_file, 'w') as f:
            json.dump(gate_config, f, indent=2)
        
        return gate_config
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get status for dashboard display"""
        
        status = self.check_training_duration()
        
        dashboard_info = {
            'training_progress_percent': min(100, (status['weeks_trained'] / self.required_weeks) * 100),
            'status_color': 'green' if status['training_complete'] else 'orange',
            'status_text': status['message'],
            'features_blocked': [] if status['training_complete'] else [
                'Confidence scores',
                'Trade execution',
                'Advanced ML features'
            ],
            'estimated_completion_date': status.get('estimated_completion', 'Unknown')
        }
        
        return dashboard_info

def main():
    """Run training enforcement check"""
    
    print("🔒 12-WEEK TRAINING REQUIREMENT ENFORCER")
    print("="*50)
    
    enforcer = TrainingEnforcer()
    
    # Check current status
    status = enforcer.save_status()
    gate_config = enforcer.create_training_gate()
    
    print(f"\n📊 TRAINING STATUS:")
    print(f"Weeks trained: {status['weeks_trained']}")
    print(f"Weeks required: {status['weeks_required']}")
    print(f"Training complete: {'✅ YES' if status['training_complete'] else '❌ NO'}")
    
    if not status['training_complete']:
        print(f"Weeks remaining: {status['weeks_remaining']}")
        print(f"Estimated completion: {status['estimated_completion'][:10]}")
    
    print(f"\n🎯 FEATURE AVAILABILITY:")
    print(f"Confidence scores: {'✅ ENABLED' if gate_config['confidence_scores_enabled'] else '🚫 BLOCKED'}")
    print(f"Trading: {'✅ ENABLED' if gate_config['trading_enabled'] else '🚫 BLOCKED'}")
    print(f"Basic predictions: ✅ ENABLED")
    
    print(f"\n💡 STATUS: {status['message']}")
    
    if not status['training_complete']:
        print("\n⚠️  WARNING: System in training mode")
        print("Confidence scores and trades will be enabled after 12-week training period")

if __name__ == "__main__":
    main()