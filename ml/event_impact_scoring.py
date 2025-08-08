#!/usr/bin/env python3
"""
Event/News Impact Scoring System
LLM-based event analysis with impact scoring and half-life features
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NewsEvent:
    """News event data structure"""
    timestamp: datetime
    coin: str
    title: str
    content: str
    source: str
    event_type: str
    impact_score: float
    sentiment: float
    confidence: float
    half_life_hours: float

class EventImpactScorer:
    """
    LLM-based event impact scoring system
    """
    
    def __init__(self):
        self.event_types = {
            'listing': {'base_impact': 0.8, 'half_life': 12},
            'delisting': {'base_impact': -0.9, 'half_life': 6},
            'partnership': {'base_impact': 0.6, 'half_life': 24},
            'upgrade': {'base_impact': 0.7, 'half_life': 48},
            'hack': {'base_impact': -0.9, 'half_life': 72},
            'unlock': {'base_impact': -0.5, 'half_life': 168},
            'acquisition': {'base_impact': 0.8, 'half_life': 96},
            'regulation': {'base_impact': -0.3, 'half_life': 240},
            'staking': {'base_impact': 0.4, 'half_life': 720},
            'governance': {'base_impact': 0.2, 'half_life': 168}
        }
        
    def analyze_event_with_llm(self, event_text: str, coin: str) -> Dict[str, Any]:
        """Analyze event using LLM (OpenAI)"""
        
        try:
            from openai import OpenAI
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return self._fallback_analysis(event_text, coin)
            
            client = OpenAI(api_key=api_key)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            prompt = f"""
            Analyze this cryptocurrency event for {coin}:
            
            Event: {event_text}
            
            Provide analysis in JSON format:
            {{
                "event_type": "one of: listing, delisting, partnership, upgrade, hack, unlock, acquisition, regulation, staking, governance, other",
                "impact_score": "float between -1.0 and 1.0 (negative=bearish, positive=bullish)",
                "sentiment": "float between -1.0 and 1.0",
                "confidence": "float between 0.0 and 1.0 (how certain is this analysis)",
                "half_life_hours": "estimated hours for impact to decay by 50%",
                "key_factors": ["list", "of", "key", "impact", "factors"],
                "reasoning": "brief explanation of the analysis"
            }}
            
            Consider:
            - Market cap and liquidity of the coin
            - Historical impact of similar events
            - Current market conditions
            - Event credibility and source reliability
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_analysis(event_text, coin)
    
    def _fallback_analysis(self, event_text: str, coin: str) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        
        text_lower = event_text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ['list', 'binance', 'coinbase', 'exchange']):
            event_type = 'listing'
        elif any(word in text_lower for word in ['delist', 'remove', 'suspend']):
            event_type = 'delisting'
        elif any(word in text_lower for word in ['partner', 'collaborate', 'integrate']):
            event_type = 'partnership'
        elif any(word in text_lower for word in ['unlock', 'vest', 'release']):
            event_type = 'unlock'
        elif any(word in text_lower for word in ['hack', 'exploit', 'breach']):
            event_type = 'hack'
        elif any(word in text_lower for word in ['upgrade', 'update', 'improve']):
            event_type = 'upgrade'
        else:
            event_type = 'other'
        
        # Get base parameters
        base_params = self.event_types.get(event_type, {'base_impact': 0.1, 'half_life': 24})
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up']
        negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'fall']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        sentiment = (pos_count - neg_count) / max(1, pos_count + neg_count)
        
        return {
            'event_type': event_type,
            'impact_score': base_params['base_impact'],
            'sentiment': sentiment,
            'confidence': 0.6,  # Lower confidence for fallback
            'half_life_hours': base_params['half_life'],
            'key_factors': ['automated_analysis'],
            'reasoning': f'Fallback analysis classified as {event_type}'
        }
    
    def calculate_time_decay(self, 
                           impact_score: float, 
                           half_life_hours: float, 
                           hours_elapsed: float) -> float:
        """Calculate time-decayed impact score"""
        
        if hours_elapsed <= 0:
            return impact_score
        
        # Exponential decay: score * (1/2)^(t/half_life)
        decay_factor = np.power(0.5, hours_elapsed / half_life_hours)
        return impact_score * decay_factor
    
    def process_event(self, 
                     event_text: str, 
                     coin: str, 
                     timestamp: datetime,
                     source: str = "unknown") -> NewsEvent:
        """Process single event and create NewsEvent object"""
        
        # Analyze with LLM
        analysis = self.analyze_event_with_llm(event_text, coin)
        
        # Create NewsEvent
        event = NewsEvent(
            timestamp=timestamp,
            coin=coin,
            title=event_text[:100] + "..." if len(event_text) > 100 else event_text,
            content=event_text,
            source=source,
            event_type=analysis.get('event_type', 'other'),
            impact_score=analysis.get('impact_score', 0.0),
            sentiment=analysis.get('sentiment', 0.0),
            confidence=analysis.get('confidence', 0.5),
            half_life_hours=analysis.get('half_life_hours', 24.0)
        )
        
        return event

class EventFeatureGenerator:
    """
    Generate features from news events for ML models
    """
    
    def __init__(self):
        self.events_db = []
        
    def add_events(self, events: List[NewsEvent]):
        """Add events to the database"""
        self.events_db.extend(events)
        
        # Sort by timestamp
        self.events_db.sort(key=lambda x: x.timestamp)
    
    def generate_impact_features(self, 
                                coin: str, 
                                current_time: datetime,
                                lookback_hours: int = 168) -> Dict[str, float]:
        """Generate impact features for a specific coin and time"""
        
        # Filter relevant events
        start_time = current_time - timedelta(hours=lookback_hours)
        relevant_events = [
            event for event in self.events_db
            if event.coin == coin and start_time <= event.timestamp <= current_time
        ]
        
        if not relevant_events:
            return self._empty_features()
        
        # Calculate current impact scores
        scorer = EventImpactScorer()
        current_impacts = []
        
        for event in relevant_events:
            hours_elapsed = (current_time - event.timestamp).total_seconds() / 3600
            current_impact = scorer.calculate_time_decay(
                event.impact_score, 
                event.half_life_hours, 
                hours_elapsed
            )
            current_impacts.append(current_impact)
        
        # Aggregate features
        features = {
            'total_impact_score': sum(current_impacts),
            'positive_impact_score': sum(max(0, impact) for impact in current_impacts),
            'negative_impact_score': sum(min(0, impact) for impact in current_impacts),
            'max_impact_score': max(current_impacts) if current_impacts else 0,
            'min_impact_score': min(current_impacts) if current_impacts else 0,
            'avg_impact_score': np.mean(current_impacts) if current_impacts else 0,
            'impact_volatility': np.std(current_impacts) if len(current_impacts) > 1 else 0,
            'event_count_24h': len([e for e in relevant_events if (current_time - e.timestamp).total_seconds() / 3600 <= 24]),
            'event_count_7d': len([e for e in relevant_events if (current_time - e.timestamp).total_seconds() / 3600 <= 168]),
            'high_impact_events': len([e for e in relevant_events if abs(e.impact_score) > 0.5]),
            'recent_listing': 1 if any(e.event_type == 'listing' and (current_time - e.timestamp).total_seconds() / 3600 <= 48 for e in relevant_events) else 0,
            'recent_unlock': 1 if any(e.event_type == 'unlock' and (current_time - e.timestamp).total_seconds() / 3600 <= 72 for e in relevant_events) else 0,
            'recent_partnership': 1 if any(e.event_type == 'partnership' and (current_time - e.timestamp).total_seconds() / 3600 <= 72 for e in relevant_events) else 0
        }
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature set"""
        return {
            'total_impact_score': 0.0,
            'positive_impact_score': 0.0,
            'negative_impact_score': 0.0,
            'max_impact_score': 0.0,
            'min_impact_score': 0.0,
            'avg_impact_score': 0.0,
            'impact_volatility': 0.0,
            'event_count_24h': 0,
            'event_count_7d': 0,
            'high_impact_events': 0,
            'recent_listing': 0,
            'recent_unlock': 0,
            'recent_partnership': 0
        }

def create_sample_events() -> List[NewsEvent]:
    """Create sample events for testing"""
    
    scorer = EventImpactScorer()
    events = []
    
    sample_news = [
        ("BTC listed on new major exchange", "BTC", "listing"),
        ("ETH partnership with Microsoft announced", "ETH", "partnership"),
        ("SOL network upgrade completed successfully", "SOL", "upgrade"),
        ("ADA major token unlock scheduled", "ADA", "unlock"),
        ("MATIC hack reported on DeFi protocol", "MATIC", "hack")
    ]
    
    base_time = datetime.now() - timedelta(days=7)
    
    for i, (news, coin, event_type) in enumerate(sample_news):
        timestamp = base_time + timedelta(hours=i * 24)
        event = scorer.process_event(news, coin, timestamp, "test_source")
        events.append(event)
    
    return events

if __name__ == "__main__":
    print("📰 TESTING EVENT IMPACT SCORING SYSTEM")
    print("=" * 45)
    
    # Test event processing
    scorer = EventImpactScorer()
    
    test_event = "Bitcoin gets listed on major institutional exchange"
    event = scorer.process_event(test_event, "BTC", datetime.now())
    
    print(f"Event Analysis:")
    print(f"   Type: {event.event_type}")
    print(f"   Impact Score: {event.impact_score:.3f}")
    print(f"   Sentiment: {event.sentiment:.3f}")
    print(f"   Confidence: {event.confidence:.3f}")
    print(f"   Half-life: {event.half_life_hours}h")
    
    # Test time decay
    hours_later = 12
    decayed_impact = scorer.calculate_time_decay(
        event.impact_score, 
        event.half_life_hours, 
        hours_later
    )
    print(f"   Impact after {hours_later}h: {decayed_impact:.3f}")
    
    # Test feature generation
    feature_gen = EventFeatureGenerator()
    sample_events = create_sample_events()
    feature_gen.add_events(sample_events)
    
    features = feature_gen.generate_impact_features("BTC", datetime.now())
    print(f"\nBTC Impact Features:")
    for key, value in features.items():
        print(f"   {key}: {value}")
    
    print("✅ Event impact scoring system testing completed")