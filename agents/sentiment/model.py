# agents/sentiment/model.py (compacte, robuuste versie)
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SentimentOutput:
    score: float   # -1..1
    prob_pos: float  # 0..1 (gecalibreerd later)
    sarcasm: float

class SentimentModel:
    def __init__(self, use_llm: bool=False):
        self.use_llm = use_llm  # hook voor OpenAI later
        
        # Enhanced keyword lists for better accuracy
        self.positive_keywords = {
            "bull", "bullish", "pump", "moon", "rocket", "surge", "rally",
            "good", "great", "excellent", "amazing", "breakthrough", "up", 
            "rise", "gain", "profit", "win", "success", "boom", "breakout",
            "adopt", "adoption", "partnership", "upgrade", "improve", "strong"
        }
        
        self.negative_keywords = {
            "bear", "bearish", "dump", "crash", "drop", "fall", "down",
            "bad", "terrible", "awful", "loss", "lose", "fail", "failure",
            "rug", "scam", "hack", "exploit", "ban", "regulation", "fear",
            "panic", "sell", "exit", "decline", "weak", "concern", "worry"
        }
        
        # Intensity modifiers
        self.intensifiers = {
            "very": 1.5, "extremely": 2.0, "massive": 1.8, "huge": 1.6,
            "major": 1.4, "significant": 1.3, "strong": 1.2, "slight": 0.8,
            "minor": 0.7, "small": 0.6
        }

    def _basic_score(self, text: str) -> float:
        """Enhanced sentiment scoring with intensity modifiers"""
        if not text:
            return 0.0
            
        text = text.lower().strip()
        words = text.split()
        
        # Count positive and negative words
        pos_count = 0
        neg_count = 0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            intensity = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = self.intensifiers[words[i-1]]
            
            # Count sentiment words with intensity
            if word in self.positive_keywords:
                pos_count += intensity
            elif word in self.negative_keywords:
                neg_count += intensity
        
        # Handle negations (simple approach)
        negation_words = {"not", "no", "never", "nothing", "nobody", "nowhere", "neither", "nor"}
        for neg_word in negation_words:
            if neg_word in text:
                # Flip sentiment if negation is found (simplified)
                pos_count, neg_count = neg_count * 0.8, pos_count * 0.8
                break
        
        # Calculate normalized score
        total_sentiment = pos_count + neg_count
        if total_sentiment == 0:
            return 0.0
        
        # Tanh normalization for smooth -1 to 1 range
        raw_score = (pos_count - neg_count) / max(total_sentiment, 1)
        return float(np.tanh(raw_score))

    def _detect_sarcasm(self, text: str) -> float:
        """Simple sarcasm detection (placeholder for future enhancement)"""
        if not text:
            return 0.0
            
        text = text.lower()
        
        # Simple heuristics for sarcasm
        sarcasm_indicators = {
            "sure", "right", "oh great", "fantastic", "wonderful", 
            "perfect", "just what we needed", "brilliant"
        }
        
        # Check for mixed signals (positive words with negative context)
        has_positive = any(word in text for word in self.positive_keywords)
        has_negative = any(word in text for word in self.negative_keywords)
        
        # Check for sarcasm indicators
        sarcasm_score = sum(1 for indicator in sarcasm_indicators if indicator in text)
        
        # Mixed sentiment + sarcasm indicators = higher sarcasm probability
        if has_positive and has_negative and sarcasm_score > 0:
            return min(0.8, sarcasm_score * 0.3)
        elif sarcasm_score > 0:
            return min(0.5, sarcasm_score * 0.2)
        
        return 0.0

    def predict_single(self, text: str) -> SentimentOutput:
        """Predict sentiment for a single text"""
        
        # Basic sentiment score
        score = self._basic_score(text)
        
        # Sarcasm detection
        sarcasm = self._detect_sarcasm(text)
        
        # Adjust score based on sarcasm
        if sarcasm > 0.3:
            score *= -0.5  # Flip and reduce if high sarcasm
        
        # Convert to probability (0.5 = neutral baseline)
        prob_pos = 0.5 + 0.5 * score
        prob_pos = max(0.0, min(1.0, prob_pos))  # Clamp to [0,1]
        
        return SentimentOutput(
            score=score,
            prob_pos=prob_pos,
            sarcasm=sarcasm
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentOutput]:
        """Predict sentiment for a batch of texts"""
        if not texts:
            return []
        
        results = []
        for text in texts:
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                # Fallback for any processing errors
                print(f"Sentiment prediction error for text '{text[:50]}...': {e}")
                results.append(SentimentOutput(score=0.0, prob_pos=0.5, sarcasm=0.0))
        
        return results

def get_sentiment_model():
    """Factory function to get sentiment model"""
    return SentimentModel()

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the sentiment model"""
        return {
            "model_type": "keyword_based_enhanced",
            "use_llm": self.use_llm,
            "positive_keywords": len(self.positive_keywords),
            "negative_keywords": len(self.negative_keywords),
            "intensifiers": len(self.intensifiers),
            "features": ["intensity_modifiers", "negation_handling", "sarcasm_detection"]
        }

# Convenience functions
def analyze_sentiment(text: str, use_llm: bool = False) -> SentimentOutput:
    """Convenience function for single sentiment analysis"""
    model = SentimentModel(use_llm=use_llm)
    return model.predict_single(text)

def analyze_sentiment_batch(texts: List[str], use_llm: bool = False) -> List[SentimentOutput]:
    """Convenience function for batch sentiment analysis"""
    model = SentimentModel(use_llm=use_llm)
    return model.predict_batch(texts)

# Test the model
if __name__ == "__main__":
    print("Testing Enhanced Sentiment Model...")
    
    # Test cases
    test_texts = [
        "Bitcoin is going to the moon! This is amazing!",
        "This dump is terrible, we're all going to lose money",
        "The market is neutral today, nothing special happening",
        "Oh great, another crash. Just what we needed.",  # Sarcasm
        "Very bullish news with major partnerships announced",
        "Slight bearish pressure but nothing too concerning",
        "Not good at all, this is a disaster",  # Negation
        "",  # Empty text
        "The price is moving up significantly after the breakthrough"
    ]
    
    model = SentimentModel()
    
    print("\nSentiment Analysis Results:")
    print("-" * 80)
    
    for text in test_texts:
        result = model.predict_single(text)
        print(f"Text: {text[:50]}...")
        print(f"  Score: {result.score:.3f} | Prob_Pos: {result.prob_pos:.3f} | Sarcasm: {result.sarcasm:.3f}")
        print()
    
    # Test batch processing
    batch_results = model.predict_batch(test_texts)
    avg_score = np.mean([r.score for r in batch_results])
    avg_confidence = np.mean([abs(r.score) for r in batch_results])
    
    print(f"Batch Results Summary:")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Total Processed: {len(batch_results)}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Info: {info}")