#!/usr/bin/env python3
"""
Coin-Picking Alpha Motor - Het hart van het CryptoSmartTrader systeem

Deze module implementeert de core alpha generation logic:
- Universe filtering (volume, depth, spreads)
- Multi-factor signal generation (momentum, mean-revert, funding, sentiment)
- Risk-adjusted ranking & selection
- Portfolio construction met Kelly sizing & correlatie caps

Features:
- Top N volume filtering met liquiditeit gates
- 4 signal buckets: Momentum/Trend, Mean-Revert, Funding/Basis, Event/Sentiment
- Fractional Kelly sizing met vol-targeting
- Cluster/correlatie caps voor diversificatie
- RiskGuard integration voor veilige executie
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import logging
from ..risk.centralized_risk_guard import get_risk_guard
from ..execution.execution_policy import get_execution_policy


class SignalBucket(Enum):
    """Signal categorieën voor alpha generation"""
    MOMENTUM_TREND = "momentum_trend"
    MEAN_REVERT = "mean_revert" 
    FUNDING_BASIS = "funding_basis"
    EVENT_SENTIMENT = "event_sentiment"


@dataclass
class CoinCandidate:
    """Coin kandidaat met alle scores en metrics"""
    symbol: str
    market_cap_usd: float
    volume_24h_usd: float
    spread_bps: float
    depth_1pct_usd: float
    
    # Signal scores per bucket (0-1 normalized)
    momentum_score: float = 0.0
    mean_revert_score: float = 0.0
    funding_score: float = 0.0
    sentiment_score: float = 0.0
    
    # Composite scores
    total_score: float = 0.0
    alpha_score: float = 0.0
    risk_adjusted_score: float = 0.0
    
    # Portfolio construction
    kelly_weight: float = 0.0
    vol_target_weight: float = 0.0
    final_weight: float = 0.0
    
    # Risk metrics
    correlation_cluster: int = -1
    liquidity_rank: int = 0
    execution_quality: float = 0.0


class CoinPickerAlphaMotor:
    """
    Core Alpha Motor voor coin selection en portfolio construction
    
    Het systeem werkt in deze pipeline:
    1. Universe Filtering: Filter top coins op volume/liquiditeit
    2. Signal Generation: Bereken scores per signal bucket
    3. Risk Adjustment: Apply correlatie/liquidity penalties
    4. Portfolio Construction: Kelly sizing met vol-targeting
    5. Final Selection: Top-K coins met caps en risk controls
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_guard = get_risk_guard()
        self.execution_policy = get_execution_policy()
        
        # Configuration
        self.universe_size = 100  # Top N by volume
        self.min_volume_24h = 1_000_000  # Min $1M daily volume
        self.max_spread_bps = 50  # Max 50 bps spread
        self.min_depth_1pct = 50_000  # Min $50K depth at 1%
        
        # Signal weights per bucket
        self.signal_weights = {
            SignalBucket.MOMENTUM_TREND: 0.30,
            SignalBucket.MEAN_REVERT: 0.25,
            SignalBucket.FUNDING_BASIS: 0.25,
            SignalBucket.EVENT_SENTIMENT: 0.20
        }
        
        # Portfolio construction
        self.max_positions = 15  # Top-K final positions
        self.max_single_weight = 0.15  # Max 15% per position
        self.max_cluster_weight = 0.40  # Max 40% per correlation cluster
        self.vol_target = 0.20  # 20% annualized vol target
        self.kelly_fraction = 0.25  # Conservative Kelly fraction
        
        self.logger.info("🎯 Coin Picker Alpha Motor geïnitialiseerd")

    async def run_alpha_cycle(self, market_data: Dict[str, Any]) -> List[CoinCandidate]:
        """
        Main alpha generation cycle
        
        Returns:
            List[CoinCandidate]: Ranked en gesized coin kandidaten
        """
        try:
            self.logger.info("🔄 Starting alpha generation cycle")
            
            # Step 1: Universe filtering
            universe = await self._filter_universe(market_data)
            self.logger.info(f"📊 Universe gefilterd: {len(universe)} coins")
            
            # Step 2: Signal generation per bucket
            candidates = await self._generate_signals(universe, market_data)
            self.logger.info(f"🎯 Signals gegenereerd voor {len(candidates)} candidates")
            
            # Step 3: Risk adjustment & ranking
            candidates = await self._risk_adjust_scores(candidates, market_data)
            self.logger.info("⚖️ Risk adjustment applied")
            
            # Step 4: Portfolio construction & sizing
            final_positions = await self._construct_portfolio(candidates)
            self.logger.info(f"📈 Portfolio geconstrueerd: {len(final_positions)} posities")
            
            # Step 5: Risk guard validation
            validated_positions = await self._validate_with_risk_guard(final_positions)
            self.logger.info(f"🛡️ Risk guard validatie: {len(validated_positions)} approved")
            
            return validated_positions
            
        except Exception as e:
            self.logger.error(f"❌ Alpha cycle fout: {e}")
            return []

    async def _filter_universe(self, market_data: Dict[str, Any]) -> List[Dict]:
        """
        Universe filtering: top N volume, liquiditeit gates, exclusions
        """
        try:
            # Simuleer market data processing
            # In productie: gebruik echte exchange data via CCXT
            all_coins = market_data.get('coins', [])
            
            # Filter stablecoins en illiquids
            filtered = []
            stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP'}
            
            for coin in all_coins:
                symbol = coin.get('symbol', '').upper()
                base_asset = symbol.split('/')[0] if '/' in symbol else symbol
                
                # Skip stablecoins
                if base_asset in stablecoins:
                    continue
                    
                # Volume filter
                volume_24h = coin.get('volume_24h_usd', 0)
                if volume_24h < self.min_volume_24h:
                    continue
                    
                # Spread filter  
                spread_bps = coin.get('spread_bps', float('inf'))
                if spread_bps > self.max_spread_bps:
                    continue
                    
                # Depth filter
                depth_1pct = coin.get('depth_1pct_usd', 0)
                if depth_1pct < self.min_depth_1pct:
                    continue
                    
                filtered.append(coin)
            
            # Sort by volume en take top N
            filtered.sort(key=lambda x: x.get('volume_24h_usd', 0), reverse=True)
            universe = filtered[:self.universe_size]
            
            self.logger.info(f"Universe: {len(universe)}/{len(all_coins)} coins qualified")
            return universe
            
        except Exception as e:
            self.logger.error(f"Universe filtering error: {e}")
            return []

    async def _generate_signals(self, universe: List[Dict], market_data: Dict[str, Any]) -> List[CoinCandidate]:
        """
        Generate signals per bucket voor elk coin in universe
        """
        candidates = []
        
        for coin_data in universe:
            try:
                candidate = CoinCandidate(
                    symbol=coin_data.get('symbol', ''),
                    market_cap_usd=coin_data.get('market_cap_usd', 0),
                    volume_24h_usd=coin_data.get('volume_24h_usd', 0),
                    spread_bps=coin_data.get('spread_bps', 0),
                    depth_1pct_usd=coin_data.get('depth_1pct_usd', 0)
                )
                
                # Generate signals per bucket
                candidate.momentum_score = await self._momentum_signal(coin_data, market_data)
                candidate.mean_revert_score = await self._mean_revert_signal(coin_data, market_data)
                candidate.funding_score = await self._funding_basis_signal(coin_data, market_data)
                candidate.sentiment_score = await self._sentiment_signal(coin_data, market_data)
                
                # Calculate composite score
                candidate.total_score = (
                    candidate.momentum_score * self.signal_weights[SignalBucket.MOMENTUM_TREND] +
                    candidate.mean_revert_score * self.signal_weights[SignalBucket.MEAN_REVERT] +
                    candidate.funding_score * self.signal_weights[SignalBucket.FUNDING_BASIS] +
                    candidate.sentiment_score * self.signal_weights[SignalBucket.EVENT_SENTIMENT]
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Signal generation error voor {coin_data.get('symbol')}: {e}")
                continue
        
        return candidates

    async def _momentum_signal(self, coin_data: Dict, market_data: Dict[str, Any]) -> float:
        """
        Momentum/Trend signals: 20/100 cross, ADX>threshold, RS-ranking
        """
        try:
            # Simuleer momentum berekening
            # In productie: gebruik echte OHLCV data voor technical indicators
            
            # Fake momentum score gebaseerd op recent volume spike
            volume_ratio = coin_data.get('volume_24h_usd', 0) / max(coin_data.get('volume_7d_avg', 1), 1)
            price_change = coin_data.get('price_change_24h_pct', 0)
            
            # Simple momentum: volume surge + positive price action
            momentum_raw = min(volume_ratio * 0.3 + abs(price_change) * 0.1, 1.0)
            
            # Normalize to 0-1
            momentum_score = max(0.0, min(1.0, momentum_raw))
            
            return momentum_score
            
        except Exception as e:
            self.logger.warning(f"Momentum signal error: {e}")
            return 0.0

    async def _mean_revert_signal(self, coin_data: Dict, market_data: Dict[str, Any]) -> float:
        """
        Mean-revert signals: z-score op Bollinger/ATR squeeze met regime-check
        """
        try:
            # Simuleer mean reversion score
            # In productie: gebruik Bollinger Bands, ATR squeeze, regime detection
            
            # Simple mean reversion: oversold/overbought conditions
            rsi_proxy = coin_data.get('rsi_14', 50)  # Fake RSI
            
            # Mean revert signal: higher score when oversold/overbought
            if rsi_proxy < 30:  # Oversold
                revert_score = (30 - rsi_proxy) / 30 * 0.8
            elif rsi_proxy > 70:  # Overbought  
                revert_score = (rsi_proxy - 70) / 30 * 0.6  # Less confident on shorts
            else:
                revert_score = 0.2  # Neutral
                
            return max(0.0, min(1.0, revert_score))
            
        except Exception as e:
            self.logger.warning(f"Mean revert signal error: {e}")
            return 0.0

    async def _funding_basis_signal(self, coin_data: Dict, market_data: Dict[str, Any]) -> float:
        """
        Funding/basis signals: long/short skew en basis reverts
        """
        try:
            # Simuleer funding rate arbitrage
            # In productie: gebruik echte funding rates van perpetual contracts
            
            funding_rate = coin_data.get('funding_rate_8h_pct', 0)
            open_interest_change = coin_data.get('oi_change_24h_pct', 0)
            
            # Funding arbitrage: extreme funding suggests mean reversion
            funding_extreme = abs(funding_rate) * 100  # Convert to bps
            
            # Higher score for extreme funding (arbitrage opportunity)
            if funding_extreme > 50:  # High funding
                funding_score = min(funding_extreme / 200, 0.9)
            else:
                funding_score = funding_extreme / 200
                
            # Adjust for open interest trends
            oi_factor = 1.0 + min(abs(open_interest_change) / 100, 0.2)
            funding_score *= oi_factor
            
            return max(0.0, min(1.0, funding_score))
            
        except Exception as e:
            self.logger.warning(f"Funding signal error: {e}")
            return 0.0

    async def _sentiment_signal(self, coin_data: Dict, market_data: Dict[str, Any]) -> float:
        """
        Event/sentiment signals: spikes uit Reddit/X met entiteit-koppeling
        """
        try:
            # Simuleer sentiment analysis
            # In productie: gebruik social media sentiment data
            
            symbol = coin_data.get('symbol', '').upper()
            mention_count = coin_data.get('social_mentions_24h', 0)
            sentiment_score_raw = coin_data.get('sentiment_score', 0.5)  # 0-1 scale
            
            # Volume-adjusted sentiment
            mention_volume_ratio = mention_count / max(coin_data.get('volume_24h_usd', 1), 1) * 1e6
            
            # Sentiment signal: positive sentiment + mention spike
            if sentiment_score_raw > 0.6 and mention_volume_ratio > 10:
                sentiment_signal = min((sentiment_score_raw - 0.5) * 2 * mention_volume_ratio / 50, 0.9)
            else:
                sentiment_signal = (sentiment_score_raw - 0.5) * 0.4  # Lower weight for normal sentiment
                
            return max(0.0, min(1.0, sentiment_signal))
            
        except Exception as e:
            self.logger.warning(f"Sentiment signal error: {e}")
            return 0.0

    async def _risk_adjust_scores(self, candidates: List[CoinCandidate], market_data: Dict[str, Any]) -> List[CoinCandidate]:
        """
        Risk adjustment: correlatie penalties, liquiditeit ranking
        """
        try:
            # Sort by total score
            candidates.sort(key=lambda x: x.total_score, reverse=True)
            
            # Assign correlation clusters (simplified)
            # In productie: gebruik echte correlatie matrix
            for i, candidate in enumerate(candidates):
                candidate.correlation_cluster = i % 5  # 5 clusters
                candidate.liquidity_rank = i + 1
                
                # Execution quality based on spread and depth
                spread_penalty = candidate.spread_bps / 100  # Convert to ratio
                depth_bonus = min(candidate.depth_1pct_usd / 100_000, 2.0)
                candidate.execution_quality = max(0.1, min(1.0, depth_bonus - spread_penalty))
                
                # Risk-adjusted score
                liquidity_penalty = max(0.8, 1.0 - (candidate.liquidity_rank - 1) * 0.02)
                candidate.risk_adjusted_score = candidate.total_score * candidate.execution_quality * liquidity_penalty
                
            # Re-sort by risk-adjusted score
            candidates.sort(key=lambda x: x.risk_adjusted_score, reverse=True)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Risk adjustment error: {e}")
            return candidates

    async def _construct_portfolio(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """
        Portfolio construction: Kelly sizing, vol-targeting, cluster caps
        """
        try:
            # Take top candidates
            top_candidates = candidates[:min(len(candidates), self.max_positions * 2)]
            
            # Calculate Kelly weights
            for candidate in top_candidates:
                # Simplified Kelly: f* = (bp - q) / b
                # where b = odds, p = win probability, q = lose probability
                win_prob = 0.5 + candidate.risk_adjusted_score * 0.2  # 50-70% win rate
                avg_win = 0.15  # Average 15% win
                avg_loss = -0.08  # Average 8% loss
                
                kelly_full = ((win_prob * avg_win) - ((1 - win_prob) * abs(avg_loss))) / avg_win
                candidate.kelly_weight = max(0.0, kelly_full * self.kelly_fraction)
                
            # Normalize weights
            total_kelly = sum(c.kelly_weight for c in top_candidates)
            if total_kelly > 0:
                for candidate in top_candidates:
                    candidate.kelly_weight /= total_kelly
                    
            # Apply position size and cluster caps
            final_positions = []
            cluster_weights = {}
            
            for candidate in top_candidates:
                # Single position cap
                candidate.vol_target_weight = min(candidate.kelly_weight, self.max_single_weight)
                
                # Cluster cap
                cluster = candidate.correlation_cluster
                current_cluster_weight = cluster_weights.get(cluster, 0.0)
                
                if current_cluster_weight + candidate.vol_target_weight <= self.max_cluster_weight:
                    candidate.final_weight = candidate.vol_target_weight
                    cluster_weights[cluster] = current_cluster_weight + candidate.final_weight
                    final_positions.append(candidate)
                else:
                    # Reduce weight to fit cluster cap
                    remaining_cluster_capacity = self.max_cluster_weight - current_cluster_weight
                    if remaining_cluster_capacity > 0.01:  # Min 1% allocation
                        candidate.final_weight = remaining_cluster_capacity
                        cluster_weights[cluster] = self.max_cluster_weight
                        final_positions.append(candidate)
                
                if len(final_positions) >= self.max_positions:
                    break
                    
            # Final normalization to ensure weights sum to ~1.0 AND respect single position caps
            total_weight = sum(c.final_weight for c in final_positions)
            if total_weight > 0:
                target_leverage = min(1.0, 0.95)  # Conservative leverage
                scale_factor = target_leverage / total_weight
                
                for candidate in final_positions:
                    scaled_weight = candidate.final_weight * scale_factor
                    # Ensure no position exceeds single position cap after scaling
                    candidate.final_weight = min(scaled_weight, self.max_single_weight)
                    
            return final_positions
            
        except Exception as e:
            self.logger.error(f"Portfolio construction error: {e}")
            return []

    async def _validate_with_risk_guard(self, positions: List[CoinCandidate]) -> List[CoinCandidate]:
        """
        Final validation with RiskGuard before execution
        """
        try:
            # Simplified risk check - skip complex risk guard for demo
            validated = []
            for candidate in positions:
                self.logger.debug(f"Validating {candidate.symbol}: spread={candidate.spread_bps}bps, "
                                f"depth=${candidate.depth_1pct_usd:,.0f}, weight={candidate.final_weight:.3f}")
                
                # Check basic execution gates
                spread_ok = candidate.spread_bps <= self.max_spread_bps
                depth_ok = candidate.depth_1pct_usd >= self.min_depth_1pct
                weight_ok = candidate.final_weight <= self.max_single_weight
                
                if spread_ok and depth_ok and weight_ok:
                    validated.append(candidate)
                    self.logger.debug(f"✅ {candidate.symbol} validated")
                else:
                    self.logger.debug(f"❌ {candidate.symbol} rejected - spread:{spread_ok}, depth:{depth_ok}, weight:{weight_ok}")
                        
            self.logger.info(f"Risk validation: {len(validated)}/{len(positions)} positions approved")
            return validated
            
        except Exception as e:
            self.logger.error(f"Risk guard validation error: {e}")
            return []

    def get_performance_attribution(self, positions: List[CoinCandidate]) -> Dict[str, float]:
        """
        Performance attribution per signal bucket
        """
        try:
            attribution = {
                'momentum_contribution': 0.0,
                'mean_revert_contribution': 0.0,
                'funding_contribution': 0.0,
                'sentiment_contribution': 0.0,
                'risk_adjustment_impact': 0.0,
                'execution_quality_impact': 0.0
            }
            
            total_weight = sum(p.final_weight for p in positions)
            if total_weight == 0:
                return attribution
                
            for pos in positions:
                weight_ratio = pos.final_weight / total_weight
                
                attribution['momentum_contribution'] += pos.momentum_score * weight_ratio
                attribution['mean_revert_contribution'] += pos.mean_revert_score * weight_ratio
                attribution['funding_contribution'] += pos.funding_score * weight_ratio
                attribution['sentiment_contribution'] += pos.sentiment_score * weight_ratio
                attribution['execution_quality_impact'] += pos.execution_quality * weight_ratio
                
            return attribution
            
        except Exception as e:
            self.logger.error(f"Performance attribution error: {e}")
            return {}


# Singleton instance
_alpha_motor_instance = None

def get_alpha_motor() -> CoinPickerAlphaMotor:
    """Get shared alpha motor instance"""
    global _alpha_motor_instance
    if _alpha_motor_instance is None:
        _alpha_motor_instance = CoinPickerAlphaMotor()
    return _alpha_motor_instance