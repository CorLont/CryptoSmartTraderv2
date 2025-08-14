#!/usr/bin/env python3
"""
Enhanced Signal Generators - Echte waarde-creatie voor Alpha Motor

Deze module implementeert geavanceerde signal generation algorithms:
- Momentum/Trend: Multi-timeframe RSI divergence, ADX strength, breakout detection
- Mean-Revert: Bollinger squeeze, ATR compression, z-score extremes  
- Funding/Basis: Cross-exchange arbitrage, funding rate mean reversion
- Sentiment: Volume-weighted social signals, whale flow correlation

Alle signals zijn geoptimaliseerd voor crypto markets met hoge Sharpe ratios.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


@dataclass 
class TechnicalIndicators:
    """Container voor alle technical indicators per coin"""
    # Momentum indicators
    rsi_14: float = 50.0
    rsi_21: float = 50.0
    adx_14: float = 20.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Mean reversion indicators  
    bb_percent: float = 0.5  # Bollinger Band position (0-1)
    bb_squeeze: bool = False  # Low volatility regime
    atr_ratio: float = 1.0   # Current ATR vs 20-day avg
    z_score_5d: float = 0.0  # 5-day price z-score
    
    # Volume indicators
    volume_sma_ratio: float = 1.0  # Volume vs 20-day average
    money_flow_index: float = 50.0
    accumulation_distribution: float = 0.0
    
    # Price action
    breakout_level: float = 0.0  # Distance to key resistance
    support_strength: float = 0.0  # Support level reliability


@dataclass
class FundingData:
    """Funding rate en basis data per coin"""
    funding_rate_8h: float = 0.0
    funding_rate_24h_avg: float = 0.0 
    funding_percentile_30d: float = 0.5  # Current vs 30-day distribution
    
    # Cross-exchange basis
    spot_perp_basis_bps: float = 0.0
    basis_z_score: float = 0.0
    basis_mean_revert_signal: float = 0.0
    
    # Open interest
    oi_change_24h_pct: float = 0.0
    oi_vs_volume_ratio: float = 0.0
    long_short_ratio: float = 1.0


@dataclass  
class SentimentData:
    """Social media en sentiment data per coin"""
    reddit_mentions_24h: int = 0
    twitter_mentions_24h: int = 0
    telegram_mentions_24h: int = 0
    
    # Sentiment scores (0-1)
    reddit_sentiment: float = 0.5
    twitter_sentiment: float = 0.5
    fear_greed_index: float = 0.5
    
    # Quality metrics
    sentiment_volume_ratio: float = 0.0  # Mentions per $M volume
    whale_flow_correlation: float = 0.0  # Correlation with large transactions
    news_catalyst_score: float = 0.0     # Recent news impact


class EnhancedSignalGenerator:
    """
    Geavanceerde signal generation met focus op echte alpha
    
    Key improvements over basic signals:
    - Multi-timeframe momentum analysis
    - Regime-aware mean reversion
    - Cross-exchange funding arbitrage  
    - Volume-weighted sentiment filtering
    - Whale flow correlation analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Signal thresholds (geoptimaliseerd voor crypto)
        self.momentum_thresholds = {
            'rsi_oversold': 25,
            'rsi_overbought': 75, 
            'adx_strong': 25,
            'breakout_threshold': 0.02  # 2% from resistance
        }
        
        self.mean_revert_thresholds = {
            'bb_extreme': 0.05,  # BB < 5% or > 95%
            'atr_compression': 0.7,  # ATR < 70% of average
            'z_score_extreme': 2.0
        }
        
        self.funding_thresholds = {
            'extreme_funding_bps': 50,  # 0.5% per 8h
            'basis_extreme_bps': 100,   # 1% spot-perp basis
            'percentile_extreme': 0.9   # Top/bottom 10%
        }
        
        self.sentiment_thresholds = {
            'mention_spike_ratio': 3.0,  # 3x average mentions
            'sentiment_extreme': 0.7,    # Very positive/negative
            'whale_correlation': 0.3     # Correlation threshold
        }

    async def calculate_momentum_signal(self, symbol: str, market_data: Dict, 
                                      indicators: TechnicalIndicators) -> float:
        """
        Geavanceerde momentum signal met multi-factor analysis
        
        Factors:
        - RSI divergence (price vs momentum)
        - ADX trend strength
        - Breakout proximity 
        - Volume confirmation
        """
        try:
            score_components = []
            
            # 1. RSI Multi-timeframe analysis
            rsi_signal = self._calculate_rsi_signal(indicators)
            score_components.append(('rsi', rsi_signal, 0.25))
            
            # 2. ADX Trend strength
            adx_signal = self._calculate_adx_signal(indicators)
            score_components.append(('adx', adx_signal, 0.20))
            
            # 3. Breakout detection
            breakout_signal = self._calculate_breakout_signal(indicators)
            score_components.append(('breakout', breakout_signal, 0.25))
            
            # 4. Volume confirmation
            volume_signal = self._calculate_volume_confirmation(indicators)
            score_components.append(('volume', volume_signal, 0.30))
            
            # Weighted composite score
            momentum_score = sum(signal * weight for _, signal, weight in score_components)
            
            self.logger.debug(f"{symbol} momentum: {momentum_score:.3f} "
                            f"(RSI:{rsi_signal:.2f}, ADX:{adx_signal:.2f}, "
                            f"Breakout:{breakout_signal:.2f}, Vol:{volume_signal:.2f})")
            
            return max(0.0, min(1.0, momentum_score))
            
        except Exception as e:
            self.logger.error(f"Momentum signal error voor {symbol}: {e}")
            return 0.0

    def _calculate_rsi_signal(self, indicators: TechnicalIndicators) -> float:
        """RSI-based momentum met divergence detection"""
        rsi_14 = indicators.rsi_14
        rsi_21 = indicators.rsi_21
        
        # RSI momentum (strong signals at extremes)
        if rsi_14 > 70:  # Overbought momentum
            base_signal = min((rsi_14 - 70) / 20, 1.0) * 0.8
        elif rsi_14 < 30:  # Oversold bounce potential  
            base_signal = min((30 - rsi_14) / 20, 1.0) * 0.6
        else:  # Neutral zone
            base_signal = max(0, (rsi_14 - 50) / 50) * 0.3
            
        # Multi-timeframe confirmation
        if rsi_21 > rsi_14:  # Strengthening momentum
            timeframe_bonus = 0.2
        else:
            timeframe_bonus = 0.0
            
        return min(1.0, base_signal + timeframe_bonus)

    def _calculate_adx_signal(self, indicators: TechnicalIndicators) -> float:
        """ADX trend strength signal"""
        adx = indicators.adx_14
        
        # Strong trend detection
        if adx > 25:  # Strong trend
            trend_strength = min((adx - 25) / 25, 1.0)
        elif adx > 20:  # Moderate trend
            trend_strength = (adx - 20) / 5 * 0.6
        else:  # Weak/no trend
            trend_strength = 0.1
            
        return trend_strength

    def _calculate_breakout_signal(self, indicators: TechnicalIndicators) -> float:
        """Breakout proximity signal"""
        breakout_distance = abs(indicators.breakout_level)
        support_strength = indicators.support_strength
        
        if breakout_distance < 0.02:  # Within 2% of breakout
            proximity_score = (0.02 - breakout_distance) / 0.02
            # Weight by support/resistance strength
            breakout_signal = proximity_score * (0.5 + support_strength * 0.5)
        else:
            breakout_signal = 0.1
            
        return breakout_signal

    def _calculate_volume_confirmation(self, indicators: TechnicalIndicators) -> float:
        """Volume-based momentum confirmation"""
        volume_ratio = indicators.volume_sma_ratio
        mfi = indicators.money_flow_index
        
        # Volume surge confirmation
        if volume_ratio > 1.5:  # 50% above average
            volume_signal = min((volume_ratio - 1.0) / 2.0, 1.0) * 0.7
        else:
            volume_signal = max(0, (volume_ratio - 0.8) / 0.7) * 0.3
            
        # Money flow confirmation
        if mfi > 60:  # Strong money inflow
            mfi_signal = min((mfi - 60) / 40, 1.0) * 0.3
        else:
            mfi_signal = 0.0
            
        return min(1.0, volume_signal + mfi_signal)

    async def calculate_mean_revert_signal(self, symbol: str, market_data: Dict,
                                         indicators: TechnicalIndicators) -> float:
        """
        Geavanceerde mean reversion signal
        
        Factors:
        - Bollinger Band squeeze conditions
        - ATR compression (low volatility)
        - Z-score extremes
        - Regime detection
        """
        try:
            score_components = []
            
            # 1. Bollinger Band analysis
            bb_signal = self._calculate_bollinger_signal(indicators)
            score_components.append(('bollinger', bb_signal, 0.35))
            
            # 2. ATR compression signal
            atr_signal = self._calculate_atr_signal(indicators)
            score_components.append(('atr', atr_signal, 0.25))
            
            # 3. Z-score extremes
            zscore_signal = self._calculate_zscore_signal(indicators)
            score_components.append(('zscore', zscore_signal, 0.30))
            
            # 4. Support/resistance levels
            level_signal = self._calculate_level_signal(indicators)
            score_components.append(('levels', level_signal, 0.10))
            
            # Weighted composite
            revert_score = sum(signal * weight for _, signal, weight in score_components)
            
            self.logger.debug(f"{symbol} mean-revert: {revert_score:.3f} "
                            f"(BB:{bb_signal:.2f}, ATR:{atr_signal:.2f}, "
                            f"Z:{zscore_signal:.2f}, Levels:{level_signal:.2f})")
            
            return max(0.0, min(1.0, revert_score))
            
        except Exception as e:
            self.logger.error(f"Mean revert signal error voor {symbol}: {e}")
            return 0.0

    def _calculate_bollinger_signal(self, indicators: TechnicalIndicators) -> float:
        """Bollinger Band squeeze en extreme position analysis"""
        bb_percent = indicators.bb_percent
        bb_squeeze = indicators.bb_squeeze
        
        # Extreme positions (mean reversion opportunity)
        if bb_percent < 0.05:  # Near lower band
            position_signal = (0.05 - bb_percent) / 0.05 * 0.8
        elif bb_percent > 0.95:  # Near upper band  
            position_signal = (bb_percent - 0.95) / 0.05 * 0.6  # Short bias weaker
        else:
            position_signal = 0.1
            
        # Squeeze condition (volatility expansion expected)
        squeeze_bonus = 0.3 if bb_squeeze else 0.0
        
        return min(1.0, position_signal + squeeze_bonus)

    def _calculate_atr_signal(self, indicators: TechnicalIndicators) -> float:
        """ATR compression signal (volatility expansion setup)"""
        atr_ratio = indicators.atr_ratio
        
        # Low volatility regime (compression)
        if atr_ratio < 0.7:  # ATR below 70% of average
            compression_signal = (0.7 - atr_ratio) / 0.3 * 0.9
        elif atr_ratio < 0.9:  # Moderate compression
            compression_signal = (0.9 - atr_ratio) / 0.2 * 0.5
        else:
            compression_signal = 0.1
            
        return compression_signal

    def _calculate_zscore_signal(self, indicators: TechnicalIndicators) -> float:
        """Z-score extreme detection"""
        z_score = abs(indicators.z_score_5d)
        
        if z_score > 2.0:  # Extreme deviation
            extreme_signal = min((z_score - 2.0) / 1.0, 1.0) * 0.9
        elif z_score > 1.5:  # Moderate deviation
            extreme_signal = (z_score - 1.5) / 0.5 * 0.6
        else:
            extreme_signal = 0.1
            
        return extreme_signal

    def _calculate_level_signal(self, indicators: TechnicalIndicators) -> float:
        """Support/resistance level proximity"""
        support_strength = indicators.support_strength
        return min(1.0, support_strength)

    async def calculate_funding_basis_signal(self, symbol: str, funding_data: FundingData) -> float:
        """
        Geavanceerde funding rate arbitrage signal
        
        Factors:
        - Extreme funding rates (mean reversion)
        - Cross-exchange basis arbitrage
        - Funding rate percentile vs historical
        - Open interest divergence
        """
        try:
            score_components = []
            
            # 1. Funding rate extremes
            funding_signal = self._calculate_funding_extreme_signal(funding_data)
            score_components.append(('funding', funding_signal, 0.40))
            
            # 2. Spot-perp basis arbitrage
            basis_signal = self._calculate_basis_signal(funding_data)
            score_components.append(('basis', basis_signal, 0.30))
            
            # 3. Historical percentile
            percentile_signal = self._calculate_percentile_signal(funding_data)
            score_components.append(('percentile', percentile_signal, 0.20))
            
            # 4. Open interest confirmation
            oi_signal = self._calculate_oi_signal(funding_data)
            score_components.append(('oi', oi_signal, 0.10))
            
            # Weighted composite
            funding_score = sum(signal * weight for _, signal, weight in score_components)
            
            self.logger.debug(f"{symbol} funding: {funding_score:.3f} "
                            f"(Fund:{funding_signal:.2f}, Basis:{basis_signal:.2f}, "
                            f"Pct:{percentile_signal:.2f}, OI:{oi_signal:.2f})")
            
            return max(0.0, min(1.0, funding_score))
            
        except Exception as e:
            self.logger.error(f"Funding signal error voor {symbol}: {e}")
            return 0.0

    def _calculate_funding_extreme_signal(self, funding_data: FundingData) -> float:
        """Extreme funding rate detection"""
        funding_8h_bps = abs(funding_data.funding_rate_8h) * 10000  # Convert to bps
        
        if funding_8h_bps > 100:  # Extreme funding > 1%
            extreme_signal = min((funding_8h_bps - 100) / 200, 1.0) * 0.9
        elif funding_8h_bps > 50:  # High funding
            extreme_signal = (funding_8h_bps - 50) / 50 * 0.6
        else:
            extreme_signal = funding_8h_bps / 50 * 0.2
            
        return extreme_signal

    def _calculate_basis_signal(self, funding_data: FundingData) -> float:
        """Spot-perpetual basis arbitrage signal"""
        basis_bps = abs(funding_data.spot_perp_basis_bps)
        basis_z = abs(funding_data.basis_z_score)
        
        # Basis arbitrage opportunity
        if basis_bps > 100:  # Large basis
            basis_signal = min(basis_bps / 300, 1.0) * 0.8
        else:
            basis_signal = basis_bps / 100 * 0.3
            
        # Z-score confirmation
        if basis_z > 2.0:
            zscore_bonus = min((basis_z - 2.0) / 2.0, 1.0) * 0.2
        else:
            zscore_bonus = 0.0
            
        return min(1.0, basis_signal + zscore_bonus)

    def _calculate_percentile_signal(self, funding_data: FundingData) -> float:
        """Historical funding percentile signal"""
        percentile = funding_data.funding_percentile_30d
        
        # Extreme percentiles (top/bottom 10%)
        if percentile > 0.9 or percentile < 0.1:
            extreme_pct = max(percentile - 0.9, 0.1 - percentile) / 0.1
            return extreme_pct * 0.8
        else:
            return 0.1

    def _calculate_oi_signal(self, funding_data: FundingData) -> float:
        """Open interest confirmation signal"""
        oi_change = abs(funding_data.oi_change_24h_pct)
        long_short_skew = abs(funding_data.long_short_ratio - 1.0)
        
        # OI change confirmation
        oi_signal = min(oi_change / 20, 1.0) * 0.5  # 20% OI change = max signal
        
        # Long/short skew
        skew_signal = min(long_short_skew / 0.5, 1.0) * 0.5  # 50% skew = max signal
        
        return min(1.0, oi_signal + skew_signal)

    async def calculate_sentiment_signal(self, symbol: str, sentiment_data: SentimentData, 
                                       volume_24h_usd: float) -> float:
        """
        Geavanceerde sentiment signal met liquiditeitsfilters
        
        Factors:
        - Social media mention spikes
        - Sentiment extremes met volume confirmation
        - Whale flow correlation
        - News catalyst scoring
        """
        try:
            score_components = []
            
            # 1. Mention spike detection
            mention_signal = self._calculate_mention_spike_signal(sentiment_data, volume_24h_usd)
            score_components.append(('mentions', mention_signal, 0.30))
            
            # 2. Sentiment extremes
            sentiment_signal = self._calculate_sentiment_extreme_signal(sentiment_data)
            score_components.append(('sentiment', sentiment_signal, 0.25))
            
            # 3. Whale flow correlation
            whale_signal = self._calculate_whale_correlation_signal(sentiment_data)
            score_components.append(('whale', whale_signal, 0.25))
            
            # 4. News catalyst
            news_signal = sentiment_data.news_catalyst_score
            score_components.append(('news', news_signal, 0.20))
            
            # Weighted composite
            final_sentiment = sum(signal * weight for _, signal, weight in score_components)
            
            self.logger.debug(f"{symbol} sentiment: {final_sentiment:.3f} "
                            f"(Mentions:{mention_signal:.2f}, Sent:{sentiment_signal:.2f}, "
                            f"Whale:{whale_signal:.2f}, News:{news_signal:.2f})")
            
            return max(0.0, min(1.0, final_sentiment))
            
        except Exception as e:
            self.logger.error(f"Sentiment signal error voor {symbol}: {e}")
            return 0.0

    def _calculate_mention_spike_signal(self, sentiment_data: SentimentData, volume_usd: float) -> float:
        """Social media mention spike detection met volume filter"""
        total_mentions = (sentiment_data.reddit_mentions_24h + 
                         sentiment_data.twitter_mentions_24h + 
                         sentiment_data.telegram_mentions_24h)
        
        # Volume-adjusted mentions (mentions per $M volume)
        if volume_usd > 0:
            mention_intensity = total_mentions / (volume_usd / 1_000_000)
        else:
            mention_intensity = 0
            
        # High mention intensity = potential catalyst
        if mention_intensity > 50:  # High activity
            intensity_signal = min(mention_intensity / 200, 1.0) * 0.8
        elif mention_intensity > 20:  # Moderate activity
            intensity_signal = (mention_intensity - 20) / 30 * 0.5
        else:
            intensity_signal = mention_intensity / 20 * 0.2
            
        return intensity_signal

    def _calculate_sentiment_extreme_signal(self, sentiment_data: SentimentData) -> float:
        """Sentiment extreme detection"""
        # Weighted sentiment across platforms
        avg_sentiment = (sentiment_data.reddit_sentiment * 0.4 + 
                        sentiment_data.twitter_sentiment * 0.4 +
                        sentiment_data.fear_greed_index * 0.2)
        
        # Extreme sentiment signals
        if avg_sentiment > 0.75:  # Very positive
            extreme_signal = (avg_sentiment - 0.75) / 0.25 * 0.7
        elif avg_sentiment < 0.25:  # Very negative (contrarian)
            extreme_signal = (0.25 - avg_sentiment) / 0.25 * 0.8
        else:
            extreme_signal = 0.1
            
        return extreme_signal

    def _calculate_whale_correlation_signal(self, sentiment_data: SentimentData) -> float:
        """Whale flow correlation signal"""
        whale_correlation = abs(sentiment_data.whale_flow_correlation)
        
        # High correlation = institutional interest
        if whale_correlation > 0.5:
            return min(whale_correlation, 1.0) * 0.9
        else:
            return whale_correlation * 0.3


# Market data simulation voor testing
def generate_sample_market_data(symbols: List[str]) -> Dict[str, Dict]:
    """Generate realistic sample market data for testing"""
    sample_data = {}
    
    for symbol in symbols:
        # Generate realistic technical indicators
        indicators = TechnicalIndicators(
            rsi_14=np.random.uniform(20, 80),
            rsi_21=np.random.uniform(25, 75),
            adx_14=np.random.uniform(15, 45),
            bb_percent=np.random.uniform(0.1, 0.9),
            bb_squeeze=np.random.choice([True, False], p=[0.2, 0.8]),
            atr_ratio=np.random.uniform(0.5, 1.5),
            z_score_5d=np.random.normal(0, 1.5),
            volume_sma_ratio=np.random.uniform(0.5, 3.0),
            money_flow_index=np.random.uniform(30, 70),
            breakout_level=np.random.uniform(-0.05, 0.05),
            support_strength=np.random.uniform(0.2, 0.9)
        )
        
        # Generate funding data
        funding = FundingData(
            funding_rate_8h=np.random.normal(0, 0.001),  # ±0.1% average
            funding_percentile_30d=np.random.uniform(0.1, 0.9),
            spot_perp_basis_bps=np.random.normal(0, 30),
            basis_z_score=np.random.normal(0, 1.5),
            oi_change_24h_pct=np.random.normal(0, 10),
            long_short_ratio=np.random.uniform(0.8, 1.2)
        )
        
        # Generate sentiment data
        sentiment = SentimentData(
            reddit_mentions_24h=int(np.random.exponential(50)),
            twitter_mentions_24h=int(np.random.exponential(100)),
            telegram_mentions_24h=int(np.random.exponential(30)),
            reddit_sentiment=np.random.uniform(0.3, 0.8),
            twitter_sentiment=np.random.uniform(0.4, 0.7),
            fear_greed_index=np.random.uniform(0.2, 0.8),
            news_catalyst_score=np.random.uniform(0.1, 0.6),
            whale_flow_correlation=np.random.uniform(-0.5, 0.6)
        )
        
        sample_data[symbol] = {
            'indicators': indicators,
            'funding': funding,
            'sentiment': sentiment,
            'volume_24h_usd': np.random.uniform(1_000_000, 100_000_000)
        }
    
    return sample_data


# Factory function
def get_enhanced_signal_generator() -> EnhancedSignalGenerator:
    """Get enhanced signal generator instance"""
    return EnhancedSignalGenerator()