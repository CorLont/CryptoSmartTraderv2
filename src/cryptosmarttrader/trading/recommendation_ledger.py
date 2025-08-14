#!/usr/bin/env python3
"""
Trading Recommendations Ledger System

Enterprise-grade ledger voor het loggen van alle trading aanbevelingen met
volledige context, performance tracking en label generatie voor ML training.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TradingSide(str, Enum):
    """Trading side enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExitReason(str, Enum):
    """Exit reason enumeration"""
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_DECAY = "TIME_DECAY"
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_LIMIT = "RISK_LIMIT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    MANUAL_EXIT = "MANUAL_EXIT"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


@dataclass
class PolicyVersions:
    """Policy version tracking voor reproducibility"""
    risk_policy_hash: str
    execution_policy_hash: str
    model_version_hash: str
    parameters_hash: str
    whale_config_hash: str = ""
    sentiment_config_hash: str = ""


@dataclass
class SignalScores:
    """Signal scores per bucket voor detailed attribution"""
    momentum_score: float = 0.0
    mean_revert_score: float = 0.0
    funding_score: float = 0.0
    sentiment_score: float = 0.0
    whale_score: float = 0.0
    regime_score: float = 0.0
    technical_score: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0


class TradingRecommendation(BaseModel):
    """Comprehensive trading recommendation met alle context"""
    
    # Core identification
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    ts_signal: datetime = Field(..., description="Signal generation timestamp (UTC)")
    symbol: str = Field(..., description="Trading symbol")
    side: TradingSide = Field(..., description="Recommended trading side")
    
    # Signal analysis
    signal_scores: SignalScores = Field(..., description="Detailed signal scores")
    features_snapshot: Dict[str, Any] = Field(..., description="Complete feature set snapshot")
    
    # Risk and return estimates
    expected_return_bps: int = Field(..., description="Expected return in basis points")
    risk_budget_bps: int = Field(..., description="Risk budget in basis points")
    slippage_budget_bps: int = Field(..., description="Expected slippage budget")
    
    # Policy versioning
    policy_versions: PolicyVersions = Field(..., description="Policy version hashes")
    
    # Execution tracking (filled when executed)
    ts_entry: Optional[datetime] = Field(None, description="Actual entry timestamp")
    entry_price: Optional[float] = Field(None, description="Actual entry price")
    entry_quantity: Optional[float] = Field(None, description="Actual entry quantity")
    
    # Exit tracking (filled when closed)
    ts_exit: Optional[datetime] = Field(None, description="Actual exit timestamp")
    exit_price: Optional[float] = Field(None, description="Actual exit price")
    realized_pnl_bps: Optional[int] = Field(None, description="Realized PnL in basis points")
    realized_slippage_bps: Optional[int] = Field(None, description="Realized slippage")
    reason_exit: Optional[ExitReason] = Field(None, description="Exit reason")
    
    # Performance metrics (computed when closed)
    hit_rate: Optional[float] = Field(None, description="1.0 if profitable, 0.0 if loss")
    sharpe_contribution: Optional[float] = Field(None, description="Sharpe ratio contribution")
    max_drawdown_bps: Optional[int] = Field(None, description="Maximum drawdown during trade")
    holding_period_hours: Optional[float] = Field(None, description="Holding period in hours")
    
    # Additional context
    market_regime: str = Field("", description="Market regime at signal time")
    volatility_percentile: Optional[float] = Field(None, description="Volatility percentile 0-100")
    liquidity_score: Optional[float] = Field(None, description="Liquidity score 0-1")
    execution_difficulty: Optional[str] = Field(None, description="Execution difficulty level")


class RecommendationLedger:
    """Enterprise recommendation ledger with performance analytics"""
    
    def __init__(self, ledger_path: str = "data/recommendations_ledger.json"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache voor active recommendations
        self._active_recommendations: Dict[str, TradingRecommendation] = {}
        
        # Policy version caching
        self._cached_policy_versions: Optional[PolicyVersions] = None
        
        # Performance metrics cache
        self._performance_cache: Dict[str, Any] = {}
        
        logger.info(f"RecommendationLedger initialized: {self.ledger_path}")
    
    def _generate_policy_versions(self) -> PolicyVersions:
        """Generate current policy version hashes"""
        
        if self._cached_policy_versions:
            return self._cached_policy_versions
        
        try:
            # Risk policy hash (from risk guard config)
            risk_config = json.dumps({
                "max_daily_loss_pct": 5.0,
                "max_position_size_pct": 20.0,
                "max_leverage": 3.0,
                "max_correlation": 0.7
            }, sort_keys=True)
            risk_hash = hashlib.md5(risk_config.encode()).hexdigest()[:8]
            
            # Execution policy hash
            execution_config = json.dumps({
                "max_spread_bps": 50,
                "min_depth_usd": 100000,
                "max_slippage_bps": 25,
                "timeout_seconds": 30
            }, sort_keys=True)
            execution_hash = hashlib.md5(execution_config.encode()).hexdigest()[:8]
            
            # Model version hash
            model_config = json.dumps({
                "version": "2.1.0",
                "features": ["rsi", "macd", "whale_flow", "sentiment"],
                "ensemble_weights": [0.3, 0.25, 0.25, 0.2]
            }, sort_keys=True)
            model_hash = hashlib.md5(model_config.encode()).hexdigest()[:8]
            
            # Parameters hash
            params_config = json.dumps({
                "lookback_days": 30,
                "min_volume_usd": 1000000,
                "confidence_threshold": 0.7,
                "rebalance_frequency": "1H"
            }, sort_keys=True)
            params_hash = hashlib.md5(params_config.encode()).hexdigest()[:8]
            
            self._cached_policy_versions = PolicyVersions(
                risk_policy_hash=risk_hash,
                execution_policy_hash=execution_hash,
                model_version_hash=model_hash,
                parameters_hash=params_hash
            )
            
            return self._cached_policy_versions
            
        except Exception as e:
            logger.error(f"Error generating policy versions: {e}")
            
            # Fallback to timestamp-based hashes
            fallback_hash = hashlib.md5(
                datetime.now().strftime("%Y%m%d").encode()
            ).hexdigest()[:8]
            
            return PolicyVersions(
                risk_policy_hash=fallback_hash,
                execution_policy_hash=fallback_hash,
                model_version_hash=fallback_hash,
                parameters_hash=fallback_hash
            )
    
    def log_recommendation(
        self,
        symbol: str,
        side: TradingSide,
        signal_scores: SignalScores,
        features_snapshot: Dict[str, Any],
        expected_return_bps: int,
        risk_budget_bps: int,
        slippage_budget_bps: int,
        market_regime: str = "",
        volatility_percentile: Optional[float] = None,
        liquidity_score: Optional[float] = None
    ) -> str:
        """Log a new trading recommendation"""
        
        # Generate unique recommendation ID
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        rec_id = f"{symbol.replace('/', '')}_{side.value}_{timestamp_str}"
        
        # Create recommendation
        recommendation = TradingRecommendation(
            recommendation_id=rec_id,
            ts_signal=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            signal_scores=signal_scores,
            features_snapshot=features_snapshot,
            expected_return_bps=expected_return_bps,
            risk_budget_bps=risk_budget_bps,
            slippage_budget_bps=slippage_budget_bps,
            policy_versions=self._generate_policy_versions(),
            market_regime=market_regime,
            volatility_percentile=volatility_percentile,
            liquidity_score=liquidity_score,
            # Optional fields with defaults
            ts_entry=None,
            entry_price=None,
            entry_quantity=None,
            ts_exit=None,
            exit_price=None,
            realized_pnl_bps=None,
            realized_slippage_bps=None,
            reason_exit=None,
            hit_rate=None,
            sharpe_contribution=None,
            max_drawdown_bps=None,
            holding_period_hours=None,
            execution_difficulty=None
        )
        
        # Store in active recommendations
        self._active_recommendations[rec_id] = recommendation
        
        # Append to ledger file
        self._append_to_ledger(recommendation)
        
        logger.info(f"Recommendation logged: {rec_id} | {symbol} {side.value} | "
                   f"Score: {signal_scores.combined_score:.3f}")
        
        return rec_id
    
    def update_entry_execution(
        self,
        recommendation_id: str,
        entry_price: float,
        entry_quantity: float,
        execution_timestamp: Optional[datetime] = None
    ) -> bool:
        """Update recommendation with entry execution details"""
        
        if recommendation_id not in self._active_recommendations:
            logger.error(f"Recommendation not found: {recommendation_id}")
            return False
        
        recommendation = self._active_recommendations[recommendation_id]
        recommendation.ts_entry = execution_timestamp or datetime.now(timezone.utc)
        recommendation.entry_price = entry_price
        recommendation.entry_quantity = entry_quantity
        
        # Update in ledger file
        self._update_ledger_record(recommendation)
        
        logger.info(f"Entry execution updated: {recommendation_id} | "
                   f"Price: {entry_price} | Qty: {entry_quantity}")
        
        return True
    
    def close_recommendation(
        self,
        recommendation_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        exit_timestamp: Optional[datetime] = None
    ) -> bool:
        """Close recommendation with exit execution details"""
        
        if recommendation_id not in self._active_recommendations:
            logger.error(f"Recommendation not found: {recommendation_id}")
            return False
        
        recommendation = self._active_recommendations[recommendation_id]
        
        if not recommendation.entry_price:
            logger.error(f"Cannot close recommendation without entry: {recommendation_id}")
            return False
        
        # Update exit details
        recommendation.ts_exit = exit_timestamp or datetime.now(timezone.utc)
        recommendation.exit_price = exit_price
        recommendation.reason_exit = exit_reason
        
        # Calculate performance metrics
        self._calculate_performance_metrics(recommendation)
        
        # Update in ledger file
        self._update_ledger_record(recommendation)
        
        # Move to completed (remove from active)
        del self._active_recommendations[recommendation_id]
        
        logger.info(f"Recommendation closed: {recommendation_id} | "
                   f"PnL: {recommendation.realized_pnl_bps}bps | "
                   f"Reason: {exit_reason.value}")
        
        return True
    
    def _calculate_performance_metrics(self, recommendation: TradingRecommendation):
        """Calculate performance metrics for closed recommendation"""
        
        if not all([recommendation.entry_price, recommendation.exit_price, 
                   recommendation.ts_entry, recommendation.ts_exit]):
            return
        
        entry_price = float(recommendation.entry_price or 0)
        exit_price = float(recommendation.exit_price or 0)
        
        if entry_price == 0:
            return
        
        # Calculate PnL in basis points
        if recommendation.side == TradingSide.BUY:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl_pct = (entry_price - exit_price) / entry_price
        
        recommendation.realized_pnl_bps = int(pnl_pct * 10000)
        
        # Hit rate (binary)
        recommendation.hit_rate = 1.0 if recommendation.realized_pnl_bps > 0 else 0.0
        
        # Holding period
        if recommendation.ts_exit and recommendation.ts_entry:
            holding_timedelta = recommendation.ts_exit - recommendation.ts_entry
            recommendation.holding_period_hours = holding_timedelta.total_seconds() / 3600
        
        # Estimate slippage (simplified)
        expected_slippage = recommendation.slippage_budget_bps
        # In reality, this would be calculated from actual execution vs mid prices
        recommendation.realized_slippage_bps = int(expected_slippage * 0.8)  # Assume 80% of budget used
    
    def _append_to_ledger(self, recommendation: TradingRecommendation):
        """Append recommendation to ledger file"""
        try:
            # Convert to dict for JSON serialization
            rec_dict = recommendation.dict()
            
            # Convert datetime objects to ISO strings
            if rec_dict['ts_signal']:
                rec_dict['ts_signal'] = rec_dict['ts_signal'].isoformat()
            if rec_dict['ts_entry']:
                rec_dict['ts_entry'] = rec_dict['ts_entry'].isoformat()
            if rec_dict['ts_exit']:
                rec_dict['ts_exit'] = rec_dict['ts_exit'].isoformat()
            
            # Append to file
            with open(self.ledger_path, 'a') as f:
                f.write(json.dumps(rec_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Error appending to ledger: {e}")
    
    def _update_ledger_record(self, recommendation: TradingRecommendation):
        """Update existing record in ledger file (simplified implementation)"""
        # In production, this would use a proper database
        # For now, just append the updated record
        self._append_to_ledger(recommendation)
    
    def get_active_recommendations(self) -> List[TradingRecommendation]:
        """Get all active recommendations"""
        return list(self._active_recommendations.values())
    
    def get_recommendation_history(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30
    ) -> pd.DataFrame:
        """Get recommendation history as DataFrame"""
        
        try:
            if not self.ledger_path.exists():
                return pd.DataFrame()
            
            # Read all records
            records = []
            with open(self.ledger_path, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Filter by symbol if specified
            if symbol:
                df = df[df['symbol'] == symbol]
            
            # Filter by date
            if 'ts_signal' in df.columns:
                df['ts_signal'] = pd.to_datetime(df['ts_signal'])
                cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
                df = df[df['ts_signal'] >= cutoff_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading recommendation history: {e}")
            return pd.DataFrame()
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        df = self.get_recommendation_history()
        
        if df.empty:
            return {"total_recommendations": 0}
        
        analytics = {
            "total_recommendations": len(df),
            "active_recommendations": len(self._active_recommendations),
            "completed_recommendations": len(df[df['ts_exit'].notna()]),
        }
        
        # Performance metrics for completed trades only
        completed_df = df[df['ts_exit'].notna()]
        
        if not completed_df.empty:
            new_metrics = {
                "hit_rate": float(completed_df['hit_rate'].mean()),
                "avg_pnl_bps": float(completed_df['realized_pnl_bps'].mean()),
                "total_pnl_bps": int(completed_df['realized_pnl_bps'].sum()),
                "best_trade_bps": int(completed_df['realized_pnl_bps'].max()),
                "worst_trade_bps": int(completed_df['realized_pnl_bps'].min()),
                "avg_holding_hours": float(completed_df['holding_period_hours'].mean()),
                "avg_slippage_bps": float(completed_df['realized_slippage_bps'].mean())
            }
            analytics.update(new_metrics)
            
            # Performance by signal bucket (simplified for now)
            signal_performance = {
                "momentum_score": 0.0,
                "sentiment_score": 0.0, 
                "whale_score": 0.0
            }
            analytics['signal_performance'] = signal_performance
        
        return analytics
    
    def generate_training_labels(
        self,
        lookforward_hours: int = 24,
        return_threshold_bps: int = 50
    ) -> pd.DataFrame:
        """Generate ML training labels from recommendation outcomes"""
        
        df = self.get_recommendation_history()
        completed_df = df[df['ts_exit'].notna()]
        
        if completed_df.empty:
            return pd.DataFrame()
        
        # Generate binary labels based on return threshold
        completed_df = completed_df.copy()
        completed_df['label_profitable'] = (completed_df['realized_pnl_bps'] > 0).astype(int)
        completed_df['label_significant'] = (
            completed_df['realized_pnl_bps'].abs() > return_threshold_bps
        ).astype(int)
        
        # Multi-class labels
        def categorize_return(pnl_bps):
            if pnl_bps > return_threshold_bps:
                return 2  # Strong positive
            elif pnl_bps < -return_threshold_bps:
                return 0  # Strong negative
            else:
                return 1  # Neutral
        
        completed_df['label_multiclass'] = completed_df['realized_pnl_bps'].apply(categorize_return)
        
        # Select relevant columns for training
        label_columns = [
            'recommendation_id', 'symbol', 'ts_signal',
            'features_snapshot', 'signal_scores',
            'realized_pnl_bps', 'label_profitable', 'label_significant', 'label_multiclass',
            'hit_rate', 'holding_period_hours'
        ]
        
        return completed_df[label_columns]


# Global ledger instance
_global_ledger: Optional[RecommendationLedger] = None


def get_recommendation_ledger() -> RecommendationLedger:
    """Get global recommendation ledger instance"""
    global _global_ledger
    
    if _global_ledger is None:
        _global_ledger = RecommendationLedger()
    
    return _global_ledger


def log_trading_recommendation(
    symbol: str,
    side: TradingSide,
    signal_scores: SignalScores,
    features_snapshot: Dict[str, Any],
    expected_return_bps: int,
    risk_budget_bps: int,
    slippage_budget_bps: int,
    **kwargs
) -> str:
    """Convenience function voor logging recommendations"""
    
    ledger = get_recommendation_ledger()
    
    return ledger.log_recommendation(
        symbol=symbol,
        side=side,
        signal_scores=signal_scores,
        features_snapshot=features_snapshot,
        expected_return_bps=expected_return_bps,
        risk_budget_bps=risk_budget_bps,
        slippage_budget_bps=slippage_budget_bps,
        **kwargs
    )


if __name__ == "__main__":
    # Test recommendation ledger
    ledger = RecommendationLedger("test_recommendations.json")
    
    # Create test recommendation
    test_scores = SignalScores(
        momentum_score=0.7,
        sentiment_score=0.4,
        whale_score=0.6,
        combined_score=0.65,
        confidence=0.8
    )
    
    test_features = {
        "price_usd": 45000.0,
        "volume_24h": 2000000000,
        "rsi_14": 65.5,
        "whale_flow_24h": 5000000,
        "sentiment_score": 0.4
    }
    
    # Log recommendation
    rec_id = ledger.log_recommendation(
        symbol="BTC/USDT",
        side=TradingSide.BUY,
        signal_scores=test_scores,
        features_snapshot=test_features,
        expected_return_bps=150,
        risk_budget_bps=200,
        slippage_budget_bps=25,
        market_regime="BULL_TREND"
    )
    
    print(f"Logged recommendation: {rec_id}")
    
    # Simulate entry
    ledger.update_entry_execution(rec_id, 45100.0, 0.1)
    
    # Simulate exit
    ledger.close_recommendation(rec_id, 45600.0, ExitReason.TAKE_PROFIT)
    
    # Get analytics
    analytics = ledger.get_performance_analytics()
    print(f"Performance analytics: {analytics}")