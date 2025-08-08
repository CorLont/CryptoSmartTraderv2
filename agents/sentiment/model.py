#!/usr/bin/env python3
"""
Sentiment Model - FinBERT/Crypto-BERT Integration
Advanced sentiment analysis with transformers, sarcasm detection and calibration
"""

import torch
import numpy as np
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import pickle

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn.functional as F

# Import structured logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.structured_logger import get_structured_logger

@dataclass
class SentimentResult:
    """Individual sentiment analysis result"""
    text: str
    score: float  # -1 to 1 (negative to positive)
    prob_pos: float  # 0 to 1 (probability of positive sentiment)
    prob_neg: float  # 0 to 1 (probability of negative sentiment)
    prob_neutral: float  # 0 to 1 (probability of neutral sentiment)
    confidence: float  # 0 to 1 (calibrated confidence)
    sarcasm: int  # 0 or 1 (sarcasm flag)
    processing_time: float
    model_used: str

@dataclass
class BatchSentimentResult:
    """Batch sentiment analysis result"""
    results: List[SentimentResult]
    total_processed: int
    processing_time: float
    average_confidence: float
    sentiment_distribution: Dict[str, float]
    calibration_report: Dict[str, Any]

class SarcasmDetector:
    """Simple sarcasm detection using heuristics and patterns"""
    
    def __init__(self):
        # Sarcasm indicators
        self.sarcasm_patterns = [
            r'\b(yeah right|sure thing|of course|obviously)\b',
            r'\b(totally|absolutely|definitely)\b.*\b(not|never)\b',
            r'\b(great|wonderful|amazing|fantastic)\b.*[!]{2,}',
            r'\b(just what i needed|exactly what i wanted)\b',
            r'\b(perfect|brilliant)\b.*[.]{3,}',
            r'/s\b',  # Reddit sarcasm tag
            r'.*🙄.*',  # Eye roll emoji
            r'.*😒.*',  # Unamused emoji
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sarcasm_patterns]
        
        # Contradiction words
        self.contradiction_words = [
            'but', 'however', 'although', 'despite', 'though', 'yet', 'still'
        ]
    
    def detect_sarcasm(self, text: str) -> float:
        """Detect sarcasm probability (0-1)"""
        
        sarcasm_score = 0.0
        
        # Pattern matching
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                sarcasm_score += 0.3
        
        # Excessive punctuation
        if len(re.findall(r'[!]{2,}', text)) > 0:
            sarcasm_score += 0.2
        
        if len(re.findall(r'[.]{3,}', text)) > 0:
            sarcasm_score += 0.2
        
        # Contradiction detection
        words = text.lower().split()
        for word in self.contradiction_words:
            if word in words:
                sarcasm_score += 0.1
        
        # All caps with positive words but negative context
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if caps_words and any(word.lower() in ['great', 'amazing', 'perfect', 'wonderful'] for word in caps_words):
            sarcasm_score += 0.25
        
        # Normalize and cap at 1.0
        return min(sarcasm_score, 1.0)

class SentimentModel:
    """Advanced sentiment analysis with FinBERT/Crypto-BERT and calibration"""
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 device: Optional[str] = None,
                 enable_calibration: bool = True):
        
        self.logger = get_structured_logger("SentimentModel")
        self.model_name = model_name
        self.enable_calibration = enable_calibration
        
        # Device configuration
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.calibrator = None
        
        # Sarcasm detector
        self.sarcasm_detector = SarcasmDetector()
        
        # Calibration data
        self.calibration_data = None
        self.calibration_report = {}
        
        # Performance tracking
        self.batch_stats = {
            "total_processed": 0,
            "total_time": 0.0,
            "average_time_per_item": 0.0
        }
        
        self.logger.info(f"SentimentModel initialized", 
                        model_name=model_name, 
                        device=self.device,
                        calibration_enabled=enable_calibration)
    
    async def initialize(self):
        """Initialize the sentiment model asynchronously"""
        
        start_time = time.time()
        
        try:
            # Load tokenizer and model
            self.logger.info(f"Loading model {self.model_name}")
            
            if "finbert" in self.model_name.lower():
                # FinBERT specific loading
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertForSequenceClassification.from_pretrained(self.model_name)
            else:
                # Generic transformers loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            initialization_time = time.time() - start_time
            
            self.logger.info(f"Model initialized successfully",
                           initialization_time=initialization_time,
                           model_parameters=sum(p.numel() for p in self.model.parameters()),
                           device=self.device)
            
            # Initialize calibration if enabled
            if self.enable_calibration:
                await self._initialize_calibration()
        
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def _initialize_calibration(self):
        """Initialize probability calibration"""
        
        try:
            # Check for saved calibrator
            calibrator_path = Path(f"models/calibrator_{self.model_name.replace('/', '_')}.pkl")
            
            if calibrator_path.exists():
                with open(calibrator_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
                self.logger.info("Loaded existing calibrator")
            else:
                self.logger.info("No existing calibrator found, will create during first calibration")
        
        except Exception as e:
            self.logger.warning(f"Failed to load calibrator: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle crypto symbols and tickers
        text = re.sub(r'\$([A-Z]{2,10})', r'cryptocurrency \1', text)
        text = re.sub(r'#([A-Z]{2,10})', r'hashtag \1', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Normalize emojis (basic)
        text = re.sub(r'[😀-🙏]', ' [EMOJI] ', text)
        
        return text.strip()
    
    def _convert_to_sentiment_score(self, predictions: List[Dict]) -> Tuple[float, float, float, float]:
        """Convert model predictions to sentiment score and probabilities"""
        
        # Handle different model outputs
        if isinstance(predictions[0], list):
            # Multiple scores per prediction
            scores = predictions[0]
        else:
            scores = predictions
        
        # Extract probabilities
        prob_neg = 0.0
        prob_neutral = 0.0
        prob_pos = 0.0
        
        for score_dict in scores:
            label = score_dict['label'].upper()
            prob = score_dict['score']
            
            if 'NEG' in label or 'BEARISH' in label:
                prob_neg = prob
            elif 'POS' in label or 'BULLISH' in label:
                prob_pos = prob
            elif 'NEU' in label or 'NEUTRAL' in label:
                prob_neutral = prob
        
        # Normalize probabilities
        total_prob = prob_neg + prob_neutral + prob_pos
        if total_prob > 0:
            prob_neg /= total_prob
            prob_neutral /= total_prob
            prob_pos /= total_prob
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = prob_pos - prob_neg
        
        # Calculate confidence (max probability)
        confidence = max(prob_neg, prob_neutral, prob_pos)
        
        return sentiment_score, prob_pos, prob_neg, prob_neutral, confidence
    
    def _apply_calibration(self, confidence: float, sentiment_score: float) -> float:
        """Apply calibration to confidence score"""
        
        if self.calibrator is None:
            return confidence
        
        try:
            # Create feature vector for calibration
            features = np.array([[confidence, abs(sentiment_score)]]).reshape(1, -1)
            calibrated_prob = self.calibrator.predict_proba(features)[0, 1]  # Probability of being correct
            return float(calibrated_prob)
        except Exception as e:
            self.logger.warning(f"Calibration failed: {e}")
            return confidence
    
    async def predict_single(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text"""
        
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Detect sarcasm
        sarcasm_prob = self.sarcasm_detector.detect_sarcasm(text)
        sarcasm_flag = 1 if sarcasm_prob > 0.5 else 0
        
        # Get sentiment prediction
        predictions = self.pipeline(processed_text)
        
        # Convert to sentiment score
        sentiment_score, prob_pos, prob_neg, prob_neutral, confidence = self._convert_to_sentiment_score(predictions)
        
        # Apply sarcasm adjustment
        if sarcasm_flag:
            sentiment_score = -sentiment_score  # Flip sentiment for sarcasm
            confidence *= 0.8  # Reduce confidence for sarcastic content
        
        # Apply calibration
        calibrated_confidence = self._apply_calibration(confidence, sentiment_score)
        
        processing_time = time.time() - start_time
        
        return SentimentResult(
            text=text,
            score=sentiment_score,
            prob_pos=prob_pos,
            prob_neg=prob_neg,
            prob_neutral=prob_neutral,
            confidence=calibrated_confidence,
            sarcasm=sarcasm_flag,
            processing_time=processing_time,
            model_used=self.model_name
        )
    
    async def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a batch of texts - main interface method"""
        
        start_time = time.time()
        batch_size = len(texts)
        
        self.logger.info(f"Processing batch of {batch_size} texts")
        
        # Process in chunks for memory efficiency
        chunk_size = 100  # Process 100 texts at a time
        all_results = []
        
        for i in range(0, batch_size, chunk_size):
            chunk = texts[i:i + chunk_size]
            
            # Preprocess chunk
            processed_texts = [self._preprocess_text(text) for text in chunk]
            
            # Detect sarcasm for chunk
            sarcasm_flags = [1 if self.sarcasm_detector.detect_sarcasm(text) > 0.5 else 0 for text in chunk]
            
            # Get predictions for chunk
            predictions = self.pipeline(processed_texts)
            
            # Process results
            for j, (original_text, prediction, sarcasm_flag) in enumerate(zip(chunk, predictions, sarcasm_flags)):
                sentiment_score, prob_pos, prob_neg, prob_neutral, confidence = self._convert_to_sentiment_score([prediction])
                
                # Apply sarcasm adjustment
                if sarcasm_flag:
                    sentiment_score = -sentiment_score
                    confidence *= 0.8
                
                # Apply calibration
                calibrated_confidence = self._apply_calibration(confidence, sentiment_score)
                
                # Create result dictionary
                result = {
                    'score': sentiment_score,
                    'prob_pos': prob_pos,
                    'confidence': calibrated_confidence,
                    'sarcasm': sarcasm_flag
                }
                
                all_results.append(result)
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.batch_stats["total_processed"] += batch_size
        self.batch_stats["total_time"] += processing_time
        self.batch_stats["average_time_per_item"] = self.batch_stats["total_time"] / self.batch_stats["total_processed"]
        
        self.logger.info(f"Batch processing completed",
                        batch_size=batch_size,
                        processing_time=processing_time,
                        average_time_per_item=processing_time/batch_size,
                        throughput=batch_size/processing_time)
        
        return all_results
    
    async def predict_batch_detailed(self, texts: List[str]) -> BatchSentimentResult:
        """Predict sentiment for batch with detailed analytics"""
        
        start_time = time.time()
        
        # Get predictions using main interface
        result_dicts = await self.predict_batch(texts)
        
        # Convert to detailed results
        results = []
        for i, result_dict in enumerate(result_dicts):
            result = SentimentResult(
                text=texts[i],
                score=result_dict['score'],
                prob_pos=result_dict['prob_pos'],
                prob_neg=1.0 - result_dict['prob_pos'] - (1.0 - abs(result_dict['score'])) * 0.5,  # Estimate prob_neg
                prob_neutral=1.0 - abs(result_dict['score']),  # Estimate prob_neutral
                confidence=result_dict['confidence'],
                sarcasm=result_dict['sarcasm'],
                processing_time=0.0,  # Individual timing not tracked in batch
                model_used=self.model_name
            )
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Calculate analytics
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        
        sentiment_distribution = {
            "positive": len([s for s in scores if s > 0.1]) / len(scores),
            "negative": len([s for s in scores if s < -0.1]) / len(scores),
            "neutral": len([s for s in scores if -0.1 <= s <= 0.1]) / len(scores)
        }
        
        # Generate calibration report
        calibration_report = self._generate_calibration_report(confidences, scores)
        
        return BatchSentimentResult(
            results=results,
            total_processed=len(texts),
            processing_time=processing_time,
            average_confidence=np.mean(confidences),
            sentiment_distribution=sentiment_distribution,
            calibration_report=calibration_report
        )
    
    def _generate_calibration_report(self, confidences: List[float], scores: List[float]) -> Dict[str, Any]:
        """Generate calibration report with probability buckets"""
        
        buckets = {
            "0.8-0.9": {"count": 0, "correct": 0, "accuracy": 0.0},
            "0.9-1.0": {"count": 0, "correct": 0, "accuracy": 0.0},
            "overall": {"count": len(confidences), "average_confidence": np.mean(confidences)}
        }
        
        # Define correctness (simplified - assumes strong sentiment scores are "correct")
        for conf, score in zip(confidences, scores):
            is_correct = abs(score) > 0.3  # Strong sentiment is considered "correct"
            
            if 0.8 <= conf < 0.9:
                buckets["0.8-0.9"]["count"] += 1
                if is_correct:
                    buckets["0.8-0.9"]["correct"] += 1
            elif 0.9 <= conf <= 1.0:
                buckets["0.9-1.0"]["count"] += 1
                if is_correct:
                    buckets["0.9-1.0"]["correct"] += 1
        
        # Calculate accuracies
        for bucket in ["0.8-0.9", "0.9-1.0"]:
            if buckets[bucket]["count"] > 0:
                buckets[bucket]["accuracy"] = buckets[bucket]["correct"] / buckets[bucket]["count"]
        
        return buckets
    
    async def calibrate_model(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Calibrate the model's confidence scores"""
        
        if not self.enable_calibration:
            return {"status": "calibration_disabled"}
        
        self.logger.info(f"Starting model calibration with {len(texts)} samples")
        
        try:
            # Get predictions for calibration data
            predictions = await self.predict_batch_detailed(texts)
            
            # Prepare data for calibration
            X = np.array([[r.confidence, abs(r.score)] for r in predictions.results])
            y = np.array(labels)  # 1 for correct, 0 for incorrect
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train calibrator
            from sklearn.ensemble import RandomForestClassifier
            base_estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            
            self.calibrator = CalibratedClassifierCV(base_estimator, method='isotonic', cv=3)
            self.calibrator.fit(X_train, y_train)
            
            # Evaluate calibration
            train_score = self.calibrator.score(X_train, y_train)
            test_score = self.calibrator.score(X_test, y_test)
            
            # Save calibrator
            calibrator_path = Path("models")
            calibrator_path.mkdir(exist_ok=True)
            calibrator_file = calibrator_path / f"calibrator_{self.model_name.replace('/', '_')}.pkl"
            
            with open(calibrator_file, 'wb') as f:
                pickle.dump(self.calibrator, f)
            
            calibration_report = {
                "status": "success",
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "calibrator_saved": str(calibrator_file)
            }
            
            self.calibration_report = calibration_report
            self.logger.info("Model calibration completed", **calibration_report)
            
            return calibration_report
        
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "calibration_enabled": self.enable_calibration,
            "calibration_available": self.calibrator is not None,
            "batch_stats": self.batch_stats.copy(),
            "calibration_report": self.calibration_report.copy() if self.calibration_report else {}
        }

# Global model instance
_sentiment_model: Optional[SentimentModel] = None

async def get_sentiment_model(model_name: str = "ProsusAI/finbert", **kwargs) -> SentimentModel:
    """Get global sentiment model instance"""
    global _sentiment_model
    
    if _sentiment_model is None or _sentiment_model.model_name != model_name:
        _sentiment_model = SentimentModel(model_name=model_name, **kwargs)
        await _sentiment_model.initialize()
    
    return _sentiment_model

async def cleanup_sentiment_model():
    """Cleanup global sentiment model"""
    global _sentiment_model
    
    if _sentiment_model is not None:
        # Clear GPU memory
        if _sentiment_model.model is not None and torch.cuda.is_available():
            del _sentiment_model.model
            torch.cuda.empty_cache()
        _sentiment_model = None