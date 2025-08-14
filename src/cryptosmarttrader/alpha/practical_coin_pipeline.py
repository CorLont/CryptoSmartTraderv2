#!/usr/bin/env python3
"""
Practical Coin Selection Pipeline - "Welke coins kopen voor hoge rendementen?"

Complete pipeline implementation:
1. Universe filtering (volume, spread, depth)
2. Regime detection per coin (trend/mr/chop/high-vol)  
3. Signal generation per regime
4. Ranking met weighted scoring
5. Risk/Execution validation
6. Kelly sizing met cluster caps
7. Order ticket generation
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .enhanced_signal_generators import (
    TechnicalIndicators, FundingData, SentimentData,
    generate_sample_market_data
)

logger = logging.getLogger(__name__)

class Regime(Enum):
    """Market regime classification"""
    TREND = "trend"
    MEAN_REVERT = "mean_revert" 
    CHOPPY = "choppy"
    HIGH_VOL = "high_vol"

@dataclass
class UniverseFilters:
    """Universe filtering criteria"""
    min_volume_24h_usd: float = 10_000_000  # 10M USD minimum
    max_spread_bps: float = 50              # 50 bps maximum
    min_depth_usd: float = 500_000          # 500K USD minimum depth

@dataclass 
class RegimeSignals:
    """Signal scores per regime bucket"""
    trend_score: float = 0.0
    mean_revert_score: float = 0.0
    event_sentiment_score: float = 0.0
    regime_confidence: float = 0.0

@dataclass
class CoinCandidate:
    """Enhanced coin candidate with full pipeline data"""
    symbol: str
    
    # Market data
    market_cap_usd: float
    volume_24h_usd: float
    spread_bps: float
    depth_usd: float
    
    # Regime classification
    regime: Regime
    regime_confidence: float
    
    # Technical indicators
    price_20ma: float
    price_100ma: float
    adx: float
    rsi: float
    z_score: float
    
    # Funding/basis
    funding_rate_8h: float
    oi_change_24h_pct: float
    
    # Sentiment
    sentiment_score: float
    sentiment_spike: bool
    
    # Signal scores
    signals: RegimeSignals = field(default_factory=RegimeSignals)
    
    # Final scoring
    weighted_score: float = 0.0
    liquidity_penalty: float = 0.0
    final_score: float = 0.0
    
    # Risk/Execution
    risk_approved: bool = False
    execution_approved: bool = False
    
    # Position sizing
    kelly_weight: float = 0.0
    vol_target_weight: float = 0.0
    cluster_weight: float = 0.0
    final_weight: float = 0.0
    correlation_cluster: int = 0
    
    # Order details
    client_order_id: str = ""
    time_in_force: str = "GTC"
    post_only: bool = True

class PracticalCoinPipeline:
    """
    Complete practical pipeline for coin selection
    """
    
    def __init__(self, 
                 universe_filters: Optional[UniverseFilters] = None,
                 max_positions: int = 15,
                 target_leverage: float = 0.95):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.universe_filters = universe_filters or UniverseFilters()
        self.max_positions = max_positions
        self.target_leverage = target_leverage
        
        # Signal weights per regime
        self.regime_weights = {
            Regime.TREND: {'trend': 0.6, 'mean_revert': 0.1, 'event_sentiment': 0.3},
            Regime.MEAN_REVERT: {'trend': 0.1, 'mean_revert': 0.6, 'event_sentiment': 0.3},
            Regime.CHOPPY: {'trend': 0.2, 'mean_revert': 0.3, 'event_sentiment': 0.5},
            Regime.HIGH_VOL: {'trend': 0.3, 'mean_revert': 0.2, 'event_sentiment': 0.5}
        }
        
        # Risk parameters
        self.max_single_weight = 0.15  # 15% max per position
        self.max_cluster_weight = 0.40  # 40% max per cluster
        
        self.logger.info("🎯 Practical Coin Pipeline initialized")
    
    async def run_pipeline(self, market_data: Dict) -> List[CoinCandidate]:
        """
        Execute complete pipeline from universe to final tickets
        """
        self.logger.info("🔄 Starting practical coin selection pipeline")
        
        # Step 1: Universe filtering
        universe = await self._filter_universe(market_data)
        self.logger.info(f"📊 Universe filtered: {len(universe)} coins qualified")
        
        # Step 2: Regime detection
        await self._detect_regimes(universe)
        self.logger.info("🔍 Regime detection completed")
        
        # Step 3: Signal generation per regime
        await self._generate_regime_signals(universe)
        self.logger.info("📈 Regime signals generated")
        
        # Step 4: Ranking with weighted scoring
        await self._rank_candidates(universe)
        self.logger.info("⚖️ Candidates ranked")
        
        # Step 5: Risk & Execution validation
        approved_candidates = await self._validate_risk_execution(universe)
        self.logger.info(f"🛡️ Risk validation: {len(approved_candidates)} approved")
        
        # Step 6: Position sizing
        sized_positions = await self._apply_kelly_sizing(approved_candidates)
        self.logger.info(f"💰 Position sizing: {len(sized_positions)} positions")
        
        # Step 7: Generate order tickets
        final_tickets = await self._generate_order_tickets(sized_positions)
        self.logger.info(f"🎫 Order tickets: {len(final_tickets)} generated")
        
        return final_tickets
    
    async def _filter_universe(self, market_data: Dict) -> List[CoinCandidate]:
        """
        Step 1: Filter universe based on volume, spread, depth criteria
        """
        qualified_coins = []
        
        for symbol, data in market_data.items():
            # Extract market metrics
            volume_24h = data.get('volume_24h_usd', 0)
            spread_bps = data.get('spread_bps', 999)
            depth_usd = data.get('depth_1pct_usd', 0)
            
            # Apply filters
            if (volume_24h >= self.universe_filters.min_volume_24h_usd and
                spread_bps <= self.universe_filters.max_spread_bps and
                depth_usd >= self.universe_filters.min_depth_usd):
                
                # Create candidate
                candidate = CoinCandidate(
                    symbol=symbol,
                    market_cap_usd=data.get('market_cap_usd', 0),
                    volume_24h_usd=volume_24h,
                    spread_bps=spread_bps,
                    depth_usd=depth_usd,
                    
                    # Technical data - using realistic calculations
                    price_20ma=100.0 + np.random.normal(0, 5),  # Mock MA values
                    price_100ma=98.0 + np.random.normal(0, 3),
                    adx=data.get('indicators', TechnicalIndicators()).adx_14,
                    rsi=data.get('indicators', TechnicalIndicators()).rsi_14,
                    z_score=data.get('indicators', TechnicalIndicators()).z_score_5d,
                    
                    # Funding data
                    funding_rate_8h=data.get('funding', FundingData()).funding_rate_8h,
                    oi_change_24h_pct=data.get('funding', FundingData()).oi_change_24h_pct,
                    
                    # Sentiment - calculate combined score
                    sentiment_score=(data.get('sentiment', SentimentData()).reddit_sentiment + 
                                   data.get('sentiment', SentimentData()).twitter_sentiment) / 2,
                    sentiment_spike=False,  # Will calculate later
                    
                    regime=Regime.TREND,  # Default, will detect later
                    regime_confidence=0.0
                )
                
                qualified_coins.append(candidate)
        
        return qualified_coins
    
    async def _detect_regimes(self, candidates: List[CoinCandidate]) -> None:
        """
        Step 2: Detect market regime per coin (trend/mr/chop/high-vol)
        """
        for candidate in candidates:
            regime, confidence = self._classify_regime(candidate)
            candidate.regime = regime
            candidate.regime_confidence = confidence
    
    def _classify_regime(self, candidate: CoinCandidate) -> Tuple[Regime, float]:
        """
        Classify market regime based on technical indicators
        """
        # Trend detection: 20MA > 100MA, ADX >= 25
        trend_signal = (candidate.price_20ma > candidate.price_100ma and 
                       candidate.adx >= 25)
        
        # Mean revert detection: z-score extreme, low ADX
        mr_signal = (abs(candidate.z_score) > 2.0 and 
                    candidate.adx < 20)
        
        # High vol detection: High ADX but conflicting signals
        high_vol_signal = (candidate.adx > 30 and 
                          abs(candidate.rsi - 50) > 25)
        
        # Calculate confidence and regime
        if trend_signal and not mr_signal:
            confidence = min(0.9, candidate.adx / 50.0)
            return Regime.TREND, confidence
        elif mr_signal and not trend_signal:
            confidence = min(0.9, abs(candidate.z_score) / 3.0)
            return Regime.MEAN_REVERT, confidence
        elif high_vol_signal:
            confidence = min(0.8, (candidate.adx - 20) / 30.0)
            return Regime.HIGH_VOL, confidence
        else:
            # Default to choppy with low confidence
            return Regime.CHOPPY, 0.3
    
    async def _generate_regime_signals(self, candidates: List[CoinCandidate]) -> None:
        """
        Step 3: Generate signals per regime
        """
        for candidate in candidates:
            signals = RegimeSignals()
            
            # Trend signals: 20>100, ADX≥25, HH/HL sequence
            if candidate.regime == Regime.TREND:
                signals.trend_score = self._calculate_trend_score(candidate)
            
            # Mean revert signals: z-score < -2, squeeze, low funding
            elif candidate.regime == Regime.MEAN_REVERT:
                signals.mean_revert_score = self._calculate_mr_score(candidate)
            
            # Event/Sentiment signals: score spike + liquidity
            signals.event_sentiment_score = self._calculate_sentiment_score(candidate)
            
            # Set regime confidence
            signals.regime_confidence = candidate.regime_confidence
            
            candidate.signals = signals
    
    def _calculate_trend_score(self, candidate: CoinCandidate) -> float:
        """Calculate trend signal score"""
        score = 0.0
        
        # MA cross strength
        if candidate.price_20ma > candidate.price_100ma:
            ma_strength = (candidate.price_20ma / candidate.price_100ma - 1) * 10
            score += min(0.4, ma_strength)
        
        # ADX strength
        adx_score = min(0.4, candidate.adx / 50.0)
        score += adx_score
        
        # RSI momentum
        if candidate.rsi > 60:
            rsi_score = min(0.2, (candidate.rsi - 60) / 20.0)
            score += rsi_score
        
        return min(1.0, score)
    
    def _calculate_mr_score(self, candidate: CoinCandidate) -> float:
        """Calculate mean reversion signal score"""
        score = 0.0
        
        # Z-score extremeness
        if candidate.z_score < -2:
            zscore_strength = min(0.5, abs(candidate.z_score + 2) / 2.0)
            score += zscore_strength
        
        # Low funding (contrarian)
        if candidate.funding_rate_8h < 0:
            funding_score = min(0.3, abs(candidate.funding_rate_8h) * 1000)
            score += funding_score
        
        # RSI oversold
        if candidate.rsi < 30:
            rsi_score = min(0.2, (30 - candidate.rsi) / 20.0)
            score += rsi_score
        
        return min(1.0, score)
    
    def _calculate_sentiment_score(self, candidate: CoinCandidate) -> float:
        """Calculate event/sentiment signal score"""
        score = 0.0
        
        # Sentiment strength
        if candidate.sentiment_score > 0.6:
            sent_score = min(0.5, (candidate.sentiment_score - 0.6) / 0.4)
            score += sent_score
        
        # OI increase (momentum)
        if candidate.oi_change_24h_pct > 10:
            oi_score = min(0.3, candidate.oi_change_24h_pct / 50.0)
            score += oi_score
        
        # Volume surge (from relative volume would be better)
        # Using depth as proxy for liquidity quality
        if candidate.depth_usd > 1_000_000:
            liquidity_score = min(0.2, candidate.depth_usd / 5_000_000)
            score += liquidity_score
        
        return min(1.0, score)
    
    async def _rank_candidates(self, candidates: List[CoinCandidate]) -> None:
        """
        Step 4: Rank candidates using weighted average with penalties
        """
        for candidate in candidates:
            # Get regime-specific weights
            weights = self.regime_weights[candidate.regime]
            
            # Calculate weighted score
            weighted_score = (
                weights['trend'] * candidate.signals.trend_score +
                weights['mean_revert'] * candidate.signals.mean_revert_score +
                weights['event_sentiment'] * candidate.signals.event_sentiment_score
            )
            
            # Apply regime confidence
            weighted_score *= candidate.regime_confidence
            
            # Calculate liquidity penalties
            liquidity_penalty = 0.0
            
            # Spread penalty
            if candidate.spread_bps > 20:
                spread_penalty = (candidate.spread_bps - 20) / 100.0  # 1% per 100 bps
                liquidity_penalty += min(0.3, spread_penalty)
            
            # Depth penalty
            if candidate.depth_usd < 1_000_000:
                depth_penalty = (1_000_000 - candidate.depth_usd) / 2_000_000
                liquidity_penalty += min(0.2, depth_penalty)
            
            # Final score
            final_score = max(0.0, weighted_score - liquidity_penalty)
            
            candidate.weighted_score = weighted_score
            candidate.liquidity_penalty = liquidity_penalty
            candidate.final_score = final_score
    
    async def _validate_risk_execution(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """
        Step 5: Risk & Execution validation
        """
        approved = []
        
        # Sort by final score
        sorted_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        
        for candidate in sorted_candidates:
            # RiskGuard validation (simplified)
            risk_approved = self._risk_guard_check(candidate)
            candidate.risk_approved = risk_approved
            
            if risk_approved:
                # ExecutionPolicy validation
                exec_approved = self._execution_policy_check(candidate)
                candidate.execution_approved = exec_approved
                
                if exec_approved:
                    approved.append(candidate)
                    
                    # Stop when we have enough positions
                    if len(approved) >= self.max_positions:
                        break
        
        return approved
    
    def _risk_guard_check(self, candidate: CoinCandidate) -> bool:
        """Simplified RiskGuard validation"""
        # Basic risk checks
        if candidate.final_score < 0.1:  # Minimum alpha threshold
            return False
        
        if candidate.volume_24h_usd < 5_000_000:  # Minimum liquidity
            return False
            
        if candidate.spread_bps > 100:  # Maximum spread
            return False
            
        return True
    
    def _execution_policy_check(self, candidate: CoinCandidate) -> bool:
        """ExecutionPolicy validation - slippage budget check"""
        # Estimate slippage impact
        estimated_notional = 100_000  # Assume 100K position size
        impact_bps = (estimated_notional / candidate.depth_usd) * 100
        
        # Check against slippage budget (200 bps daily budget)
        if impact_bps > 50:  # 50 bps per trade max
            return False
            
        return True
    
    async def _apply_kelly_sizing(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """
        Step 6: Apply fractional Kelly sizing with vol-targeting and cluster caps
        """
        positioned_candidates = []
        cluster_weights = {}
        
        for i, candidate in enumerate(candidates):
            # Fractional Kelly calculation (simplified)
            # Kelly = (alpha - cost) / variance
            alpha = candidate.final_score
            cost = candidate.spread_bps / 10000  # Convert to decimal
            variance = 0.04  # Assume 20% daily vol -> 4% variance
            
            kelly_fraction = max(0, (alpha - cost) / variance)
            kelly_weight = min(0.25, kelly_fraction * 0.5)  # 50% of Kelly, max 25%
            
            # Vol targeting (scale down for high vol coins)
            vol_factor = min(1.0, 0.15 / np.sqrt(variance))  # Target 15% vol
            vol_target_weight = kelly_weight * vol_factor
            
            # Apply single position cap
            position_weight = min(vol_target_weight, self.max_single_weight)
            
            # Cluster assignment (simplified)
            cluster = i % 3  # Simple clustering
            candidate.correlation_cluster = cluster
            
            # Check cluster cap
            current_cluster_weight = cluster_weights.get(cluster, 0.0)
            
            if current_cluster_weight + position_weight <= self.max_cluster_weight:
                candidate.kelly_weight = kelly_weight
                candidate.vol_target_weight = vol_target_weight
                candidate.cluster_weight = current_cluster_weight + position_weight
                candidate.final_weight = position_weight
                
                cluster_weights[cluster] = current_cluster_weight + position_weight
                positioned_candidates.append(candidate)
            else:
                # Reduce to fit cluster cap
                remaining = self.max_cluster_weight - current_cluster_weight
                if remaining > 0.01:  # Min 1% position
                    candidate.final_weight = remaining
                    positioned_candidates.append(candidate)
                    cluster_weights[cluster] = self.max_cluster_weight
        
        return positioned_candidates
    
    async def _generate_order_tickets(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """
        Step 7: Generate order tickets with COID, TIF, logging
        """
        tickets = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, candidate in enumerate(candidates):
            # Generate Client Order ID
            coid = f"CST_{timestamp}_{candidate.symbol.replace('/', '')}_{i:03d}"
            candidate.client_order_id = coid
            
            # Set execution parameters
            candidate.time_in_force = "GTC"  # Good Till Cancelled
            candidate.post_only = True       # Post-only to avoid taker fees
            
            tickets.append(candidate)
            
            # Log order details
            self.logger.info(
                f"🎫 Order ticket: {candidate.symbol} | "
                f"Weight: {candidate.final_weight:.1%} | "
                f"Score: {candidate.final_score:.3f} | "
                f"Regime: {candidate.regime.value} | "
                f"COID: {coid}"
            )
        
        return tickets

def get_practical_pipeline(**kwargs) -> PracticalCoinPipeline:
    """Factory function for practical pipeline"""
    return PracticalCoinPipeline(**kwargs)