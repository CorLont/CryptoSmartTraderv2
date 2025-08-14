#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise Sentiment Analysis Framework
Unified sentiment analysis met robust error handling en multiple sources
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import threading
from collections import defaultdict, deque

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        @staticmethod
        def var(x):
            if not x: return 0
            mean_val = sum(x) / len(x)
            return sum((v - mean_val) ** 2 for v in x) / len(x)
    np = MockNumpy()

from core.structured_logger import get_structured_logger
from src.cryptosmarttrader.ai import get_modernized_openai_adapter, SentimentAnalysisResult


class SentimentSource(Enum):
    """Sources of sentiment analysis"""
    LEXICON_BASED = "lexicon_based"
    LLM_ANALYSIS = "llm_analysis"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"


class SentimentStrength(Enum):
    """Sentiment strength levels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEUTRAL = "neutral"
    SLIGHTLY_POSITIVE = "slightly_positive"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class SentimentSignal:
    """Individual sentiment signal"""
    source: SentimentSource
    text_excerpt: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    strength: SentimentStrength
    keywords: List[str]
    context: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentAnalysisResult:
    """Comprehensive sentiment analysis result"""
    text_id: str
    text_length: int
    overall_sentiment_score: float  # -1.0 to 1.0
    overall_confidence: float  # 0.0 to 1.0
    sentiment_strength: SentimentStrength
    individual_signals: List[SentimentSignal]
    source_breakdown: Dict[str, float]  # Score per source
    keywords_detected: List[str]
    emotion_indicators: List[str]
    risk_indicators: List[str]
    uncertainty_level: float  # 0.0 to 1.0
    processing_time_ms: float
    data_quality_score: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    enable_llm_analysis: bool = True
    enable_lexicon_analysis: bool = True
    enable_rule_based: bool = True
    confidence_threshold: float = 0.3
    min_text_length: int = 10
    max_text_length: int = 10000
    cache_results: bool = True
    cache_ttl_hours: int = 24
    parallel_processing: bool = True
    fallback_on_error: bool = True


class CryptoSentimentLexicon:
    """Cryptocurrency-specific sentiment lexicon"""
    
    def __init__(self):
        self.logger = get_structured_logger("CryptoSentimentLexicon")
        
        # Positive crypto terms
        self.positive_terms = {
            # Price movements
            "moon": 2.0, "mooning": 2.0, "to the moon": 2.5,
            "pump": 1.5, "pumping": 1.5, "rally": 1.5,
            "bull": 1.8, "bullish": 1.8, "bull run": 2.0,
            "surge": 1.7, "soar": 1.8, "breakout": 1.6,
            "ath": 2.0, "all time high": 2.0,
            
            # Adoption & fundamentals
            "adoption": 1.5, "institutional": 1.6,
            "etf": 1.7, "approval": 1.8, "mainstream": 1.5,
            "partnership": 1.4, "collaboration": 1.3,
            "upgrade": 1.4, "innovation": 1.5,
            
            # Market sentiment
            "diamond hands": 1.8, "hodl": 1.5, "hold": 1.0,
            "buy the dip": 1.3, "accumulate": 1.4,
            "bullish divergence": 1.7, "golden cross": 1.8,
            
            # General positive
            "optimistic": 1.2, "confident": 1.3, "strong": 1.1,
            "promising": 1.4, "breakthrough": 1.6, "milestone": 1.3
        }
        
        # Negative crypto terms
        self.negative_terms = {
            # Price movements
            "crash": -2.0, "dump": -1.8, "plummet": -2.2,
            "bear": -1.8, "bearish": -1.8, "bear market": -2.0,
            "correction": -1.3, "pullback": -1.1, "decline": -1.2,
            "red": -1.0, "bleeding": -1.7, "capitulation": -2.5,
            
            # Risk & fear
            "fud": -1.8, "fear": -1.5, "uncertainty": -1.2,
            "doubt": -1.1, "panic": -2.0, "sell off": -1.6,
            "liquidation": -1.9, "margin call": -2.1,
            "death cross": -2.0, "bearish divergence": -1.7,
            
            # Market conditions
            "volatile": -0.8, "unstable": -1.4, "risky": -1.3,
            "bubble": -1.8, "speculation": -1.1, "manipulation": -2.0,
            "scam": -2.5, "ponzi": -2.5, "rugpull": -2.8,
            
            # Regulatory
            "ban": -2.2, "regulation": -1.5, "crackdown": -2.0,
            "illegal": -2.3, "prohibited": -2.1,
            
            # General negative
            "pessimistic": -1.2, "concerned": -1.0, "worried": -1.3,
            "disappointing": -1.4, "failure": -1.8, "problem": -1.1
        }
        
        # Context modifiers
        self.intensifiers = {
            "very": 1.5, "extremely": 2.0, "highly": 1.4,
            "significantly": 1.6, "massively": 1.8, "huge": 1.7,
            "massive": 1.8, "incredible": 1.9, "amazing": 1.6
        }
        
        self.diminishers = {
            "slightly": 0.5, "somewhat": 0.6, "fairly": 0.7,
            "moderately": 0.8, "mildly": 0.4, "little": 0.5
        }
        
        self.negations = {"not", "no", "never", "none", "nothing", "neither", "nor"}
    
    def analyze_text(self, text: str) -> SentimentSignal:
        """Analyze text using crypto lexicon"""
        
        if not text or len(text.strip()) < 3:
            return SentimentSignal(
                source=SentimentSource.LEXICON_BASED,
                text_excerpt=text[:50],
                sentiment_score=0.0,
                confidence=0.0,
                strength=SentimentStrength.NEUTRAL,
                keywords=[],
                context="insufficient_text"
            )
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        sentiment_scores = []
        detected_keywords = []
        
        # Process text word by word with context
        for i, word in enumerate(words):
            score = 0.0
            
            # Check positive terms
            if word in self.positive_terms:
                score = self.positive_terms[word]
                detected_keywords.append(word)
            
            # Check negative terms
            elif word in self.negative_terms:
                score = self.negative_terms[word]
                detected_keywords.append(word)
            
            if score != 0.0:
                # Apply modifiers
                # Check for intensifiers before the word
                if i > 0 and words[i-1] in self.intensifiers:
                    score *= self.intensifiers[words[i-1]]
                
                # Check for diminishers before the word
                if i > 0 and words[i-1] in self.diminishers:
                    score *= self.diminishers[words[i-1]]
                
                # Check for negations (within 3 words before)
                negated = False
                for j in range(max(0, i-3), i):
                    if words[j] in self.negations:
                        score *= -0.8  # Flip and reduce intensity
                        negated = True
                        break
                
                sentiment_scores.append(score)
        
        # Calculate overall sentiment
        if sentiment_scores:
            # Weight by frequency and normalize
            overall_score = sum(sentiment_scores) / len(words) * 10
            overall_score = max(-1.0, min(1.0, overall_score))  # Clamp to [-1, 1]
            
            # Calculate confidence based on keyword density
            keyword_density = len(detected_keywords) / len(words)
            confidence = min(1.0, keyword_density * 5)  # Scale confidence
            
        else:
            overall_score = 0.0
            confidence = 0.0
        
        # Determine sentiment strength
        strength = self._score_to_strength(overall_score)
        
        return SentimentSignal(
            source=SentimentSource.LEXICON_BASED,
            text_excerpt=text[:100],
            sentiment_score=overall_score,
            confidence=confidence,
            strength=strength,
            keywords=detected_keywords,
            context=f"lexicon_analysis_{len(detected_keywords)}_keywords"
        )
    
    def _score_to_strength(self, score: float) -> SentimentStrength:
        """Convert sentiment score to strength enum"""
        if score <= -0.6:
            return SentimentStrength.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentStrength.NEGATIVE
        elif score <= -0.05:
            return SentimentStrength.SLIGHTLY_NEGATIVE
        elif score >= 0.6:
            return SentimentStrength.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentStrength.POSITIVE
        elif score >= 0.05:
            return SentimentStrength.SLIGHTLY_POSITIVE
        else:
            return SentimentStrength.NEUTRAL


class RuleBasedSentimentAnalyzer:
    """Rule-based sentiment analysis for crypto text"""
    
    def __init__(self):
        self.logger = get_structured_logger("RuleBasedSentiment")
        
        # Define rule patterns
        self.price_patterns = {
            r'\b(?:price|btc|eth|crypto)\s+(?:up|rising|increasing|climbing)': 1.5,
            r'\b(?:price|btc|eth|crypto)\s+(?:down|falling|decreasing|dropping)': -1.5,
            r'\b(?:broke|breaking)\s+(?:resistance|support)': 1.2,
            r'\b(?:lost|losing)\s+support': -1.4,
            r'\bmoon\b|\bmooning\b': 2.0,
            r'\bcrash\b|\bcrashing\b': -2.0
        }
        
        self.volume_patterns = {
            r'\bhigh\s+volume\b': 1.0,
            r'\blow\s+volume\b': -0.5,
            r'\bvolume\s+spike\b': 1.2,
            r'\bno\s+volume\b': -0.8
        }
        
        self.adoption_patterns = {
            r'\betf\s+(?:approved|approval)': 2.2,
            r'\betf\s+(?:rejected|denial)': -2.0,
            r'\binstitutional\s+(?:adoption|buying)': 1.8,
            r'\bmainstream\s+adoption': 1.6,
            r'\b(?:ban|banned|banning)\s+(?:crypto|bitcoin)': -2.5
        }
    
    def analyze_text(self, text: str) -> SentimentSignal:
        """Analyze text using rule patterns"""
        
        if not text or len(text.strip()) < 5:
            return SentimentSignal(
                source=SentimentSource.RULE_BASED,
                text_excerpt=text[:50],
                sentiment_score=0.0,
                confidence=0.0,
                strength=SentimentStrength.NEUTRAL,
                keywords=[],
                context="insufficient_text"
            )
        
        text_lower = text.lower()
        detected_patterns = []
        sentiment_scores = []
        
        # Check all pattern categories
        pattern_categories = [
            ("price", self.price_patterns),
            ("volume", self.volume_patterns),
            ("adoption", self.adoption_patterns)
        ]
        
        for category, patterns in pattern_categories:
            for pattern, score in patterns.items():
                matches = re.findall(pattern, text_lower)
                if matches:
                    detected_patterns.extend([f"{category}:{pattern}" for _ in matches])
                    sentiment_scores.extend([score] * len(matches))
        
        # Calculate overall sentiment
        if sentiment_scores:
            # Average the scores but weight by pattern strength
            overall_score = sum(sentiment_scores) / len(sentiment_scores)
            overall_score = max(-1.0, min(1.0, overall_score / 2.0))  # Normalize and clamp
            
            # Confidence based on pattern matches
            confidence = min(1.0, len(detected_patterns) * 0.3)
            
        else:
            overall_score = 0.0
            confidence = 0.0
        
        # Determine sentiment strength
        if overall_score <= -0.6:
            strength = SentimentStrength.VERY_NEGATIVE
        elif overall_score <= -0.2:
            strength = SentimentStrength.NEGATIVE
        elif overall_score <= -0.05:
            strength = SentimentStrength.SLIGHTLY_NEGATIVE
        elif overall_score >= 0.6:
            strength = SentimentStrength.VERY_POSITIVE
        elif overall_score >= 0.2:
            strength = SentimentStrength.POSITIVE
        elif overall_score >= 0.05:
            strength = SentimentStrength.SLIGHTLY_POSITIVE
        else:
            strength = SentimentStrength.NEUTRAL
        
        return SentimentSignal(
            source=SentimentSource.RULE_BASED,
            text_excerpt=text[:100],
            sentiment_score=overall_score,
            confidence=confidence,
            strength=strength,
            keywords=detected_patterns,
            context=f"rule_based_{len(detected_patterns)}_patterns"
        )


class EnterpriseSentimentAnalyzer:
    """Unified enterprise sentiment analysis coordinator"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.logger = get_structured_logger("EnterpriseSentimentAnalyzer")
        
        # Initialize analyzers
        self.lexicon_analyzer = CryptoSentimentLexicon()
        self.rule_analyzer = RuleBasedSentimentAnalyzer()
        
        # LLM analyzer (optional)
        self.llm_analyzer = None
        if self.config.enable_llm_analysis:
            try:
                self.llm_analyzer = get_modernized_openai_adapter()
            except Exception as e:
                self.logger.warning(f"LLM analyzer not available: {e}")
        
        # Results cache
        self.cache = {} if self.config.cache_results else None
        self.cache_timestamps = {} if self.config.cache_results else None
        
        # Metrics tracking
        self.analysis_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
        self.logger.info("Enterprise Sentiment Analyzer initialized")
    
    async def analyze_text(self, 
                          text: str,
                          text_id: Optional[str] = None,
                          sources: Optional[List[SentimentSource]] = None) -> SentimentAnalysisResult:
        """Comprehensive sentiment analysis"""
        
        start_time = time.time()
        text_id = text_id or f"text_{int(time.time())}"
        
        # Input validation
        warnings = []
        errors = []
        
        if not text or not isinstance(text, str):
            errors.append("Invalid text input")
            return self._create_error_result(text_id, errors, start_time)
        
        text = text.strip()
        if len(text) < self.config.min_text_length:
            warnings.append(f"Text too short: {len(text)} < {self.config.min_text_length}")
        
        if len(text) > self.config.max_text_length:
            warnings.append(f"Text truncated: {len(text)} > {self.config.max_text_length}")
            text = text[:self.config.max_text_length]
        
        # Check cache
        if self.cache and text_id in self.cache:
            cache_time = self.cache_timestamps.get(text_id)
            if cache_time and (datetime.now() - cache_time).hours < self.config.cache_ttl_hours:
                self.logger.debug(f"Using cached result for {text_id}")
                return self.cache[text_id]
        
        # Determine sources to use
        if sources is None:
            sources = []
            if self.config.enable_lexicon_analysis:
                sources.append(SentimentSource.LEXICON_BASED)
            if self.config.enable_rule_based:
                sources.append(SentimentSource.RULE_BASED)
            if self.config.enable_llm_analysis and self.llm_analyzer:
                sources.append(SentimentSource.LLM_ANALYSIS)
        
        # Analyze with each source
        individual_signals = []
        
        try:
            if self.config.parallel_processing:
                # Parallel analysis
                tasks = []
                
                if SentimentSource.LEXICON_BASED in sources:
                    tasks.append(self._analyze_lexicon(text))
                
                if SentimentSource.RULE_BASED in sources:
                    tasks.append(self._analyze_rules(text))
                
                if SentimentSource.LLM_ANALYSIS in sources and self.llm_analyzer:
                    tasks.append(self._analyze_llm(text))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            errors.append(f"Analysis error: {str(result)}")
                        elif isinstance(result, SentimentSignal):
                            individual_signals.append(result)
                        
            else:
                # Sequential analysis
                if SentimentSource.LEXICON_BASED in sources:
                    try:
                        signal = await self._analyze_lexicon(text)
                        individual_signals.append(signal)
                    except Exception as e:
                        errors.append(f"Lexicon analysis failed: {str(e)}")
                
                if SentimentSource.RULE_BASED in sources:
                    try:
                        signal = await self._analyze_rules(text)
                        individual_signals.append(signal)
                    except Exception as e:
                        errors.append(f"Rule analysis failed: {str(e)}")
                
                if SentimentSource.LLM_ANALYSIS in sources and self.llm_analyzer:
                    try:
                        signal = await self._analyze_llm(text)
                        individual_signals.append(signal)
                    except Exception as e:
                        errors.append(f"LLM analysis failed: {str(e)}")
            
            # Aggregate results
            result = self._aggregate_signals(text_id, text, individual_signals, warnings, errors, start_time)
            
            # Cache result
            if self.cache:
                self.cache[text_id] = result
                self.cache_timestamps[text_id] = datetime.now()
            
            # Record metrics
            with self.lock:
                self.analysis_metrics["total"].append({
                    "timestamp": datetime.now(),
                    "processing_time": result.processing_time_ms,
                    "text_length": len(text),
                    "signal_count": len(individual_signals),
                    "overall_confidence": result.overall_confidence,
                    "warning_count": len(warnings),
                    "error_count": len(errors)
                })
            
            return result
            
        except Exception as e:
            errors.append(f"Analysis failed: {str(e)}")
            self.logger.error(f"Sentiment analysis error: {e}")
            return self._create_error_result(text_id, errors, start_time)
    
    async def _analyze_lexicon(self, text: str) -> SentimentSignal:
        """Async wrapper for lexicon analysis"""
        return self.lexicon_analyzer.analyze_text(text)
    
    async def _analyze_rules(self, text: str) -> SentimentSignal:
        """Async wrapper for rule analysis"""
        return self.rule_analyzer.analyze_text(text)
    
    async def _analyze_llm(self, text: str) -> SentimentSignal:
        """LLM-based sentiment analysis"""
        if not self.llm_analyzer:
            raise ValueError("LLM analyzer not available")
        
        try:
            llm_result = await self.llm_analyzer.await get_sentiment_analyzer().analyze_text(text)
            
            # Convert LLM result to SentimentSignal
            sentiment_score = llm_result.sentiment_score
            confidence = llm_result.confidence
            
            # Determine strength
            if sentiment_score <= -0.6:
                strength = SentimentStrength.VERY_NEGATIVE
            elif sentiment_score <= -0.2:
                strength = SentimentStrength.NEGATIVE
            elif sentiment_score <= -0.05:
                strength = SentimentStrength.SLIGHTLY_NEGATIVE
            elif sentiment_score >= 0.6:
                strength = SentimentStrength.VERY_POSITIVE
            elif sentiment_score >= 0.2:
                strength = SentimentStrength.POSITIVE
            elif sentiment_score >= 0.05:
                strength = SentimentStrength.SLIGHTLY_POSITIVE
            else:
                strength = SentimentStrength.NEUTRAL
            
            return SentimentSignal(
                source=SentimentSource.LLM_ANALYSIS,
                text_excerpt=text[:100],
                sentiment_score=sentiment_score,
                confidence=confidence,
                strength=strength,
                keywords=llm_result.key_phrases,
                context=f"llm_analysis_{llm_result.source}"
            )
            
        except Exception as e:
            self.logger.error(f"LLM sentiment analysis failed: {e}")
            raise e
    
    def _aggregate_signals(self, 
                          text_id: str,
                          text: str,
                          signals: List[SentimentSignal],
                          warnings: List[str],
                          errors: List[str],
                          start_time: float) -> SentimentAnalysisResult:
        """Aggregate multiple sentiment signals"""
        
        if not signals:
            return self._create_error_result(text_id, errors + ["No valid signals"], start_time)
        
        # Weight signals by confidence
        weighted_scores = []
        total_weight = 0.0
        source_breakdown = {}
        all_keywords = []
        
        for signal in signals:
            if signal.confidence >= self.config.confidence_threshold:
                weight = signal.confidence
                weighted_scores.append(signal.sentiment_score * weight)
                total_weight += weight
                
                source_breakdown[signal.source.value] = signal.sentiment_score
                all_keywords.extend(signal.keywords)
        
        # Calculate overall sentiment
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
            overall_confidence = total_weight / len(signals)  # Average confidence
        else:
            overall_score = 0.0
            overall_confidence = 0.0
            warnings.append("All signals below confidence threshold")
        
        # Determine overall strength
        if overall_score <= -0.6:
            strength = SentimentStrength.VERY_NEGATIVE
        elif overall_score <= -0.2:
            strength = SentimentStrength.NEGATIVE
        elif overall_score <= -0.05:
            strength = SentimentStrength.SLIGHTLY_NEGATIVE
        elif overall_score >= 0.6:
            strength = SentimentStrength.VERY_POSITIVE
        elif overall_score >= 0.2:
            strength = SentimentStrength.POSITIVE
        elif overall_score >= 0.05:
            strength = SentimentStrength.SLIGHTLY_POSITIVE
        else:
            strength = SentimentStrength.NEUTRAL
        
        # Extract emotion and risk indicators
        emotion_indicators = []
        risk_indicators = []
        
        for keyword in all_keywords:
            if any(emotion in keyword for emotion in ["fear", "greed", "panic", "euphoria", "anxiety"]):
                emotion_indicators.append(keyword)
            if any(risk in keyword for risk in ["risk", "volatile", "uncertain", "scam", "manipulation"]):
                risk_indicators.append(keyword)
        
        # Calculate uncertainty level
        score_variance = 0.0
        if len(signals) > 1:
            scores = [s.sentiment_score for s in signals]
            score_variance = np.var(scores) if 'np' in globals() else 0.0
        
        uncertainty_level = min(1.0, score_variance * 2.0)
        
        # Data quality score
        quality_factors = [
            len(text) / self.config.max_text_length,  # Text length factor
            len(signals) / 3.0,  # Number of signals factor
            overall_confidence,  # Confidence factor
            1.0 - (len(errors) / max(1, len(signals)))  # Error rate factor
        ]
        data_quality_score = min(1.0, sum(quality_factors) / len(quality_factors))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SentimentAnalysisResult(
            text_id=text_id,
            text_length=len(text),
            overall_sentiment_score=overall_score,
            overall_confidence=overall_confidence,
            sentiment_strength=strength,
            individual_signals=signals,
            source_breakdown=source_breakdown,
            keywords_detected=list(set(all_keywords)),
            emotion_indicators=list(set(emotion_indicators)),
            risk_indicators=list(set(risk_indicators)),
            uncertainty_level=uncertainty_level,
            processing_time_ms=processing_time_ms,
            data_quality_score=data_quality_score,
            warnings=warnings,
            errors=errors
        )
    
    def _create_error_result(self, text_id: str, errors: List[str], start_time: float) -> SentimentAnalysisResult:
        """Create error result"""
        processing_time_ms = (time.time() - start_time) * 1000
        
        return SentimentAnalysisResult(
            text_id=text_id,
            text_length=0,
            overall_sentiment_score=0.0,
            overall_confidence=0.0,
            sentiment_strength=SentimentStrength.NEUTRAL,
            individual_signals=[],
            source_breakdown={},
            keywords_detected=[],
            emotion_indicators=[],
            risk_indicators=[],
            uncertainty_level=1.0,
            processing_time_ms=processing_time_ms,
            data_quality_score=0.0,
            warnings=[],
            errors=errors
        )
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "llm_enabled": self.config.enable_llm_analysis,
                    "lexicon_enabled": self.config.enable_lexicon_analysis,
                    "rules_enabled": self.config.enable_rule_based,
                    "confidence_threshold": self.config.confidence_threshold
                },
                "metrics": {}
            }
            
            if self.analysis_metrics["total"]:
                recent_analyses = self.analysis_metrics["total"][-100:]  # Last 100
                
                summary["metrics"] = {
                    "total_analyses": len(self.analysis_metrics["total"]),
                    "avg_processing_time_ms": np.mean([a["processing_time"] for a in recent_analyses]) if 'np' in globals() else 0.0,
                    "avg_confidence": np.mean([a["overall_confidence"] for a in recent_analyses]) if 'np' in globals() else 0.0,
                    "error_rate": np.mean([a["error_count"] > 0 for a in recent_analyses]) if 'np' in globals() else 0.0,
                    "last_analysis": recent_analyses[-1]["timestamp"].isoformat() if recent_analyses else None
                }
            
            if self.cache:
                summary["cache_stats"] = {
                    "cached_results": len(self.cache),
                    "cache_hit_potential": len(self.cache) / max(1, len(self.analysis_metrics.get("total", [1])))
                }
            
            return summary


# Global singleton
_sentiment_analyzer_instance = None

def get_sentiment_analyzer(config: SentimentConfig = None) -> EnterpriseSentimentAnalyzer:
    """Get singleton sentiment analyzer"""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        _sentiment_analyzer_instance = EnterpriseSentimentAnalyzer(config)
    return _sentiment_analyzer_instance


if __name__ == "__main__":
    # Basic validation
    import asyncio
    
    async def test_analyzer():
        analyzer = get_sentiment_analyzer()
        
        # Test texts
        test_texts = [
            "Bitcoin is mooning! Great bullish momentum with institutional adoption",
            "Market crash incoming, very bearish signals and high fear",
            "Neutral market conditions, waiting for clearer direction"
        ]
        
        for i, text in enumerate(test_texts):
            result = await analyzer.analyze_text(text, f"test_{i}")
            print(f"Text {i}: Score={result.overall_sentiment_score:.2f}, "
                  f"Confidence={result.overall_confidence:.2f}, "
                  f"Strength={result.sentiment_strength.value}")
        
        # Show summary
        summary = analyzer.get_analysis_summary()
        print(f"Analysis summary: {summary}")
    
    asyncio.run(test_analyzer())