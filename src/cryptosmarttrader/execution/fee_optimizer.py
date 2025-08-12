"""
Fee Optimization Engine

Optimizes trading fees through maker/taker ratio management,
fee tier awareness, and exchange-specific fee structures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FeeType(Enum):
    """Fee type classification"""
    MAKER = "maker"
    TAKER = "taker"


class FeeTier(Enum):
    """Exchange fee tiers"""
    VIP_0 = "vip_0"      # Lowest tier
    VIP_1 = "vip_1"
    VIP_2 = "vip_2"
    VIP_3 = "vip_3"
    VIP_4 = "vip_4"
    VIP_5 = "vip_5"      # Highest tier


@dataclass
class FeeStructure:
    """Exchange fee structure"""
    exchange: str
    tier: FeeTier
    maker_fee_bp: int        # Basis points (negative = rebate)
    taker_fee_bp: int        # Basis points
    monthly_volume_usd: float  # Required volume for tier
    
    # Fee reductions
    native_token_discount: float = 0.0  # BNB, KCS, etc.
    staking_discount: float = 0.0       # Token staking discounts


@dataclass
class FeeOptimization:
    """Fee optimization recommendation"""
    recommended_fee_type: FeeType
    expected_fee_bp: float
    potential_savings_bp: float
    confidence: float
    reasoning: str
    
    # Strategy parameters
    suggested_order_type: str
    suggested_price_offset_bp: int
    expected_fill_probability: float


class FeeOptimizer:
    """
    Optimizes trading fees across exchanges and trading strategies
    """
    
    def __init__(self):
        # Exchange fee structures
        self.fee_structures = {}  # exchange -> FeeTier -> FeeStructure
        
        # Historical fee performance
        self.fee_history = []
        self.maker_taker_ratios = {}  # exchange -> ratio
        
        # Optimization targets
        self.target_maker_ratio = 0.8  # 80% maker fills
        self.max_acceptable_taker_fee_bp = 25
        
        # Initialize default fee structures
        self._initialize_fee_structures()
    
    def optimize_fee_strategy(self, 
                            exchange: str,
                            pair: str,
                            order_side: str,
                            order_size_usd: float,
                            market_data: Dict[str, Any],
                            urgency: str = "normal") -> FeeOptimization:
        """
        Optimize fee strategy for a trade
        
        Args:
            exchange: Exchange name
            pair: Trading pair
            order_side: 'buy' or 'sell'
            order_size_usd: Order size in USD
            market_data: Current market conditions
            urgency: 'low', 'normal', 'high'
            
        Returns:
            Fee optimization recommendation
        """
        try:
            # Get fee structure for exchange
            fee_structure = self._get_fee_structure(exchange, order_size_usd)
            
            if not fee_structure:
                logger.warning(f"No fee structure found for {exchange}")
                return self._get_default_optimization()
            
            # Analyze market conditions
            spread_bp = market_data.get("spread_bp", 0)
            depth = market_data.get("depth_quote", 0)
            volatility = market_data.get("volatility_1h", 0)
            
            # Calculate maker vs taker trade-offs
            maker_analysis = self._analyze_maker_strategy(
                fee_structure, spread_bp, depth, volatility, urgency
            )
            
            taker_analysis = self._analyze_taker_strategy(
                fee_structure, spread_bp, urgency
            )
            
            # Choose optimal strategy
            if maker_analysis["score"] > taker_analysis["score"]:
                optimization = FeeOptimization(
                    recommended_fee_type=FeeType.MAKER,
                    expected_fee_bp=fee_structure.maker_fee_bp,
                    potential_savings_bp=taker_analysis["expected_fee_bp"] - fee_structure.maker_fee_bp,
                    confidence=maker_analysis["confidence"],
                    reasoning=maker_analysis["reasoning"],
                    suggested_order_type="limit",
                    suggested_price_offset_bp=maker_analysis["price_offset_bp"],
                    expected_fill_probability=maker_analysis["fill_probability"]
                )
            else:
                optimization = FeeOptimization(
                    recommended_fee_type=FeeType.TAKER,
                    expected_fee_bp=fee_structure.taker_fee_bp,
                    potential_savings_bp=0,  # No savings vs taker
                    confidence=taker_analysis["confidence"],
                    reasoning=taker_analysis["reasoning"],
                    suggested_order_type="market",
                    suggested_price_offset_bp=0,
                    expected_fill_probability=0.95
                )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Fee optimization failed: {e}")
            return self._get_default_optimization()
    
    def calculate_fee_tier_benefit(self, 
                                  exchange: str,
                                  current_monthly_volume_usd: float,
                                  projected_monthly_volume_usd: float) -> Dict[str, Any]:
        """Calculate benefit of upgrading to higher fee tier"""
        try:
            current_tier = self._get_fee_tier_for_volume(exchange, current_monthly_volume_usd)
            projected_tier = self._get_fee_tier_for_volume(exchange, projected_monthly_volume_usd)
            
            if current_tier == projected_tier:
                return {
                    "upgrade_beneficial": False,
                    "current_tier": current_tier.value if current_tier else "unknown",
                    "reason": "Volume insufficient for tier upgrade"
                }
            
            # Get fee structures
            current_fees = self._get_fee_structure_by_tier(exchange, current_tier)
            projected_fees = self._get_fee_structure_by_tier(exchange, projected_tier)
            
            if not current_fees or not projected_fees:
                return {"upgrade_beneficial": False, "reason": "Fee structure not found"}
            
            # Calculate savings
            maker_savings_bp = current_fees.maker_fee_bp - projected_fees.maker_fee_bp
            taker_savings_bp = current_fees.taker_fee_bp - projected_fees.taker_fee_bp
            
            # Estimate monthly savings based on current maker/taker ratio
            current_ratio = self.maker_taker_ratios.get(exchange, 0.5)
            
            avg_savings_bp = (current_ratio * maker_savings_bp + 
                            (1 - current_ratio) * taker_savings_bp)
            
            monthly_savings_usd = projected_monthly_volume_usd * avg_savings_bp / 10000
            
            return {
                "upgrade_beneficial": monthly_savings_usd > 0,
                "current_tier": current_tier.value if current_tier else "unknown",
                "projected_tier": projected_tier.value if projected_tier else "unknown",
                "maker_savings_bp": maker_savings_bp,
                "taker_savings_bp": taker_savings_bp,
                "estimated_monthly_savings_usd": monthly_savings_usd,
                "volume_needed_usd": projected_fees.monthly_volume_usd - current_monthly_volume_usd
            }
            
        except Exception as e:
            logger.error(f"Fee tier benefit calculation failed: {e}")
            return {"upgrade_beneficial": False, "error": str(e)}
    
    def record_fee_outcome(self, 
                          exchange: str,
                          pair: str,
                          fee_type: str,
                          fee_paid_usd: float,
                          notional_usd: float,
                          fill_success: bool) -> None:
        """Record fee outcome for optimization learning"""
        try:
            fee_record = {
                "timestamp": datetime.now(),
                "exchange": exchange,
                "pair": pair,
                "fee_type": fee_type,
                "fee_paid_usd": fee_paid_usd,
                "notional_usd": notional_usd,
                "fee_bp": (fee_paid_usd / notional_usd * 10000) if notional_usd > 0 else 0,
                "fill_success": fill_success
            }
            
            self.fee_history.append(fee_record)
            
            # Keep only recent history
            if len(self.fee_history) > 10000:
                self.fee_history = self.fee_history[-10000:]
            
            # Update maker/taker ratios
            self._update_maker_taker_ratios()
            
        except Exception as e:
            logger.error(f"Failed to record fee outcome: {e}")
    
    def get_fee_analytics(self, 
                         exchange: Optional[str] = None,
                         days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive fee analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent history
            recent_fees = [
                fee for fee in self.fee_history
                if fee["timestamp"] >= cutoff_time and 
                   (exchange is None or fee["exchange"] == exchange)
            ]
            
            if not recent_fees:
                return {"status": "No recent fee data"}
            
            analytics = {
                "period_days": days_back,
                "total_trades": len(recent_fees),
                "exchanges": len(set(fee["exchange"] for fee in recent_fees))
            }
            
            # Overall metrics
            total_fees = sum(fee["fee_paid_usd"] for fee in recent_fees)
            total_notional = sum(fee["notional_usd"] for fee in recent_fees)
            
            maker_trades = [fee for fee in recent_fees if fee["fee_type"] == "maker"]
            taker_trades = [fee for fee in recent_fees if fee["fee_type"] == "taker"]
            
            analytics.update({
                "total_fees_usd": total_fees,
                "total_notional_usd": total_notional,
                "avg_fee_bp": (total_fees / total_notional * 10000) if total_notional > 0 else 0,
                "maker_ratio": len(maker_trades) / len(recent_fees),
                "maker_fee_bp": np.mean([fee["fee_bp"] for fee in maker_trades]) if maker_trades else 0,
                "taker_fee_bp": np.mean([fee["fee_bp"] for fee in taker_trades]) if taker_trades else 0
            })
            
            # Per-exchange breakdown
            if exchange is None:
                analytics["per_exchange"] = {}
                for ex in set(fee["exchange"] for fee in recent_fees):
                    ex_fees = [fee for fee in recent_fees if fee["exchange"] == ex]
                    ex_total_fees = sum(fee["fee_paid_usd"] for fee in ex_fees)
                    ex_total_notional = sum(fee["notional_usd"] for fee in ex_fees)
                    ex_maker_ratio = len([fee for fee in ex_fees if fee["fee_type"] == "maker"]) / len(ex_fees)
                    
                    analytics["per_exchange"][ex] = {
                        "trades": len(ex_fees),
                        "fees_usd": ex_total_fees,
                        "avg_fee_bp": (ex_total_fees / ex_total_notional * 10000) if ex_total_notional > 0 else 0,
                        "maker_ratio": ex_maker_ratio
                    }
            
            # Fee tier analysis
            if exchange:
                current_volume = total_notional  # Simplified - would use actual monthly volume
                tier_analysis = self.calculate_fee_tier_benefit(exchange, current_volume * 0.8, current_volume)
                analytics["fee_tier_analysis"] = tier_analysis
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate fee analytics: {e}")
            return {"status": "Error", "error": str(e)}
    
    def _initialize_fee_structures(self) -> None:
        """Initialize default fee structures for major exchanges"""
        try:
            # Kraken fee structure (example)
            kraken_fees = {
                FeeTier.VIP_0: FeeStructure("kraken", FeeTier.VIP_0, 16, 26, 0),
                FeeTier.VIP_1: FeeStructure("kraken", FeeTier.VIP_1, 14, 24, 50000),
                FeeTier.VIP_2: FeeStructure("kraken", FeeTier.VIP_2, 12, 22, 100000),
                FeeTier.VIP_3: FeeStructure("kraken", FeeTier.VIP_3, 10, 20, 250000),
                FeeTier.VIP_4: FeeStructure("kraken", FeeTier.VIP_4, 8, 18, 500000),
                FeeTier.VIP_5: FeeStructure("kraken", FeeTier.VIP_5, 6, 16, 1000000),
            }
            
            # Binance fee structure (example)
            binance_fees = {
                FeeTier.VIP_0: FeeStructure("binance", FeeTier.VIP_0, 10, 10, 0, native_token_discount=0.25),
                FeeTier.VIP_1: FeeStructure("binance", FeeTier.VIP_1, 9, 10, 100000, native_token_discount=0.25),
                FeeTier.VIP_2: FeeStructure("binance", FeeTier.VIP_2, 8, 10, 500000, native_token_discount=0.25),
                FeeTier.VIP_3: FeeStructure("binance", FeeTier.VIP_3, 7, 9, 2000000, native_token_discount=0.25),
                FeeTier.VIP_4: FeeStructure("binance", FeeTier.VIP_4, 7, 8, 5000000, native_token_discount=0.25),
                FeeTier.VIP_5: FeeStructure("binance", FeeTier.VIP_5, 5, 7, 10000000, native_token_discount=0.25),
            }
            
            self.fee_structures = {
                "kraken": kraken_fees,
                "binance": binance_fees
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize fee structures: {e}")
    
    def _get_fee_structure(self, exchange: str, volume_usd: float) -> Optional[FeeStructure]:
        """Get appropriate fee structure based on volume"""
        try:
            if exchange not in self.fee_structures:
                return None
            
            # Find appropriate tier based on monthly volume (simplified)
            # In practice, would track actual monthly volumes
            monthly_volume_estimate = volume_usd * 30  # Rough estimate
            
            tier = self._get_fee_tier_for_volume(exchange, monthly_volume_estimate)
            
            if tier and tier in self.fee_structures[exchange]:
                return self.fee_structures[exchange][tier]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get fee structure for {exchange}: {e}")
            return None
    
    def _get_fee_tier_for_volume(self, exchange: str, monthly_volume_usd: float) -> Optional[FeeTier]:
        """Determine fee tier based on monthly volume"""
        try:
            if exchange not in self.fee_structures:
                return None
            
            # Find highest tier that volume qualifies for
            qualified_tier = FeeTier.VIP_0
            
            for tier, structure in self.fee_structures[exchange].items():
                if monthly_volume_usd >= structure.monthly_volume_usd:
                    qualified_tier = tier
            
            return qualified_tier
            
        except Exception as e:
            logger.error(f"Failed to determine fee tier: {e}")
            return FeeTier.VIP_0
    
    def _get_fee_structure_by_tier(self, exchange: str, tier: FeeTier) -> Optional[FeeStructure]:
        """Get fee structure for specific tier"""
        try:
            if exchange in self.fee_structures and tier in self.fee_structures[exchange]:
                return self.fee_structures[exchange][tier]
            return None
        except Exception as e:
            logger.error(f"Failed to get fee structure by tier: {e}")
            return None
    
    def _analyze_maker_strategy(self, 
                               fee_structure: FeeStructure,
                               spread_bp: float,
                               depth_usd: float,
                               volatility: float,
                               urgency: str) -> Dict[str, Any]:
        """Analyze maker strategy viability"""
        try:
            # Base score calculation
            score = 100
            
            # Fee advantage
            fee_advantage = fee_structure.taker_fee_bp - fee_structure.maker_fee_bp
            score += fee_advantage * 2  # Weight fee savings heavily
            
            # Market conditions impact
            if spread_bp > 20:
                score -= 30  # Wide spreads reduce maker attractiveness
            elif spread_bp < 5:
                score += 20  # Tight spreads favor makers
            
            if depth_usd < 10000:
                score -= 40  # Low depth reduces fill probability
            elif depth_usd > 100000:
                score += 15  # High depth improves fill probability
            
            # Volatility impact
            if volatility > 0.05:
                score -= 25  # High volatility reduces maker fill rate
            elif volatility < 0.02:
                score += 10  # Low volatility favors makers
            
            # Urgency impact
            if urgency == "high":
                score -= 50  # High urgency reduces maker viability
            elif urgency == "low":
                score += 20  # Low urgency favors makers
            
            # Calculate fill probability
            base_fill_prob = 0.7
            if spread_bp < 10:
                base_fill_prob += 0.2
            if volatility < 0.03:
                base_fill_prob += 0.1
            if urgency == "high":
                base_fill_prob -= 0.3
            
            fill_probability = max(0.1, min(0.95, base_fill_prob))
            
            # Price offset recommendation
            if spread_bp > 15:
                price_offset_bp = max(1, int(spread_bp * 0.3))
            else:
                price_offset_bp = 1
            
            # Confidence calculation
            confidence = max(0.3, min(0.9, score / 100))
            
            reasoning = f"Maker strategy: {fee_advantage}bp fee advantage, {spread_bp:.1f}bp spread"
            
            return {
                "score": score,
                "confidence": confidence,
                "fill_probability": fill_probability,
                "price_offset_bp": price_offset_bp,
                "expected_fee_bp": fee_structure.maker_fee_bp,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Maker strategy analysis failed: {e}")
            return {"score": 0, "confidence": 0.3}
    
    def _analyze_taker_strategy(self, 
                               fee_structure: FeeStructure,
                               spread_bp: float,
                               urgency: str) -> Dict[str, Any]:
        """Analyze taker strategy viability"""
        try:
            # Base score
            score = 50  # Lower base than maker due to higher fees
            
            # Urgency boost
            if urgency == "high":
                score += 60  # High urgency favors immediate execution
            elif urgency == "normal":
                score += 20
            
            # Market conditions
            if spread_bp > 30:
                score -= 20  # Very wide spreads penalize taker
            elif spread_bp < 10:
                score += 10  # Tight spreads are OK for taker
            
            # Fee penalty
            if fee_structure.taker_fee_bp > 25:
                score -= 30  # High taker fees are penalized
            elif fee_structure.taker_fee_bp < 15:
                score += 10  # Reasonable taker fees are acceptable
            
            confidence = max(0.3, min(0.9, score / 100))
            
            reasoning = f"Taker strategy: {fee_structure.taker_fee_bp}bp fee, immediate execution"
            
            return {
                "score": score,
                "confidence": confidence,
                "expected_fee_bp": fee_structure.taker_fee_bp,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Taker strategy analysis failed: {e}")
            return {"score": 0, "confidence": 0.3}
    
    def _update_maker_taker_ratios(self) -> None:
        """Update maker/taker ratios from recent history"""
        try:
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_fees = [fee for fee in self.fee_history if fee["timestamp"] >= recent_cutoff]
            
            # Calculate per-exchange ratios
            for exchange in set(fee["exchange"] for fee in recent_fees):
                exchange_fees = [fee for fee in recent_fees if fee["exchange"] == exchange]
                
                if exchange_fees:
                    maker_count = sum(1 for fee in exchange_fees if fee["fee_type"] == "maker")
                    ratio = maker_count / len(exchange_fees)
                    self.maker_taker_ratios[exchange] = ratio
            
        except Exception as e:
            logger.error(f"Failed to update maker/taker ratios: {e}")
    
    def _get_default_optimization(self) -> FeeOptimization:
        """Get default conservative fee optimization"""
        return FeeOptimization(
            recommended_fee_type=FeeType.MAKER,
            expected_fee_bp=15,  # Conservative estimate
            potential_savings_bp=10,
            confidence=0.5,
            reasoning="Default maker strategy (insufficient market data)",
            suggested_order_type="limit",
            suggested_price_offset_bp=1,
            expected_fill_probability=0.7
        )