"""
Adaptive Risk Manager

Central risk management coordinator that integrates drawdown monitoring,
kill switches, and data health to dynamically adjust trading risk.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from .drawdown_monitor import DrawdownMonitor, DrawdownLevel, RiskReduction
from .kill_switch import KillSwitch, KillSwitchLevel
from .data_health_monitor import DataHealthMonitor, HealthGate

logger = logging.getLogger(__name__)

class RiskMode(Enum):
    """Overall risk management mode"""
    NORMAL = "normal"           # Normal operations
    CONSERVATIVE = "conservative"  # Reduced risk
    DEFENSIVE = "defensive"     # Minimal risk
    EMERGENCY = "emergency"     # No new trades
    SHUTDOWN = "shutdown"       # Complete halt


class RiskAdjustment(Enum):
    """Types of risk adjustments"""
    KELLY_SCALING = "kelly_scaling"         # Scale Kelly fractions
    POSITION_LIMITS = "position_limits"     # Reduce position sizes
    EXPOSURE_LIMITS = "exposure_limits"     # Limit total exposure
    CONCENTRATION_LIMITS = "concentration_limits"  # Limit concentration
    CORRELATION_LIMITS = "correlation_limits"      # Limit correlated exposure
    VOLATILITY_SCALING = "volatility_scaling"      # Scale based on volatility
    DATA_GATING = "data_gating"             # Gate trades on data quality


@dataclass
class RiskProfile:
    """Current risk profile and constraints"""
    mode: RiskMode
    timestamp: datetime
    
    # Scaling factors (0.0 to 1.0)
    kelly_multiplier: float
    position_size_multiplier: float
    max_exposure_pct: float
    max_single_position_pct: float
    max_correlation_exposure_pct: float
    
    # Trading constraints
    allow_new_positions: bool
    allow_position_increases: bool
    force_position_reductions: bool
    paper_trading_only: bool
    
    # Data requirements
    min_data_quality_score: float
    max_data_staleness_minutes: int
    
    # Active adjustments
    active_adjustments: List[RiskAdjustment]
    
    # Source information
    triggered_by: List[str]     # What triggered this profile
    
    @property
    def effective_risk_multiplier(self) -> float:
        """Overall effective risk multiplier"""
        return min(self.kelly_multiplier, self.position_size_multiplier)
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if any trading is allowed"""
        return self.mode not in [RiskMode.SHUTDOWN] and self.allow_new_positions


@dataclass
class RiskMetrics:
    """Risk metrics and monitoring data"""
    timestamp: datetime
    
    # Portfolio metrics
    portfolio_value: float
    total_exposure_pct: float
    max_single_exposure_pct: float
    correlation_risk_score: float
    
    # Drawdown metrics
    current_drawdown_pct: float
    max_drawdown_pct: float
    drawdown_level: DrawdownLevel
    
    # Kill switch status
    kill_switch_level: KillSwitchLevel
    active_kill_switches: int
    
    # Data health
    data_health_score: float
    unhealthy_pairs_pct: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Volatility metrics
    portfolio_volatility: float
    max_individual_volatility: float


class AdaptiveRiskManager:
    """
    Central adaptive risk management system
    """
    
    def __init__(self):
        # Component managers
        self.drawdown_monitor = DrawdownMonitor()
        self.kill_switch = KillSwitch()
        self.data_health_monitor = DataHealthMonitor()
        
        # Risk state
        self.current_profile = self._create_default_profile()
        self.profile_history = []
        
        # Configuration
        self.risk_mode_thresholds = {
            RiskMode.CONSERVATIVE: {"drawdown": 0.05, "data_health": 0.8},
            RiskMode.DEFENSIVE: {"drawdown": 0.10, "data_health": 0.6},
            RiskMode.EMERGENCY: {"drawdown": 0.15, "data_health": 0.4},
            RiskMode.SHUTDOWN: {"drawdown": 0.25, "data_health": 0.2}
        }
        
        # Adaptive parameters
        self.volatility_scaling_enabled = True
        self.correlation_monitoring_enabled = True
        self.dynamic_kelly_scaling = True
        
    def update_risk_assessment(self, 
                              portfolio_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              timestamp: Optional[datetime] = None) -> RiskProfile:
        """Update comprehensive risk assessment"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Update component monitors
            drawdown_info = self.drawdown_monitor.update_equity(
                portfolio_data.get("total_value", 0), timestamp
            )
            
            kill_switch_info = self.kill_switch.monitor_portfolio_risk(
                portfolio_data, timestamp
            )
            
            # Check data health for key pairs
            data_health_summary = self._assess_data_health(market_data)
            
            # Determine new risk mode
            new_risk_mode = self._determine_risk_mode(
                drawdown_info, kill_switch_info, data_health_summary
            )
            
            # Calculate risk adjustments
            risk_adjustments = self._calculate_risk_adjustments(
                new_risk_mode, drawdown_info, kill_switch_info, data_health_summary
            )
            
            # Create new risk profile
            new_profile = self._create_risk_profile(
                new_risk_mode, risk_adjustments, timestamp
            )
            
            # Log profile changes
            if new_profile.mode != self.current_profile.mode:
                self._log_risk_mode_change(self.current_profile.mode, new_profile.mode)
            
            # Store old profile and update current
            self.profile_history.append(self.current_profile)
            self.current_profile = new_profile
            
            # Keep recent history only
            cutoff_time = timestamp - timedelta(days=7)
            self.profile_history = [
                p for p in self.profile_history if p.timestamp >= cutoff_time
            ]
            
            return self.current_profile
            
        except Exception as e:
            logger.error(f"Risk assessment update failed: {e}")
            return self.current_profile
    
    def get_position_size_limit(self, 
                               pair: str,
                               base_size: float,
                               confidence: float,
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position size limit based on current risk profile"""
        try:
            # Base size adjustment from risk profile
            adjusted_size = base_size * self.current_profile.position_size_multiplier
            
            # Kelly scaling
            if self.dynamic_kelly_scaling:
                kelly_adjusted_size = adjusted_size * self.current_profile.kelly_multiplier
            else:
                kelly_adjusted_size = adjusted_size
            
            # Confidence-based scaling
            confidence_scaling = min(1.0, confidence / 0.8)  # Scale to 80% confidence baseline
            confidence_adjusted_size = kelly_adjusted_size * confidence_scaling
            
            # Volatility scaling
            if self.volatility_scaling_enabled:
                volatility_factor = self._calculate_volatility_scaling(pair, market_data)
                volatility_adjusted_size = confidence_adjusted_size * volatility_factor
            else:
                volatility_adjusted_size = confidence_adjusted_size
            
            # Maximum position limit
            max_position_limit = self.current_profile.max_single_position_pct
            portfolio_value = market_data.get("portfolio_value", 1.0)
            max_allowed_size = (max_position_limit * portfolio_value) / market_data.get("price", 1.0)
            
            # Final size
            final_size = min(volatility_adjusted_size, max_allowed_size)
            
            # Check if position is allowed
            position_allowed = (
                self.current_profile.allow_new_positions and
                not self.current_profile.paper_trading_only and
                self._check_data_quality_gate(pair, market_data)
            )
            
            result = {
                "original_size": base_size,
                "final_size": final_size if position_allowed else 0.0,
                "position_allowed": position_allowed,
                "paper_trading_only": self.current_profile.paper_trading_only,
                "scaling_factors": {
                    "risk_profile": self.current_profile.position_size_multiplier,
                    "kelly": self.current_profile.kelly_multiplier,
                    "confidence": confidence_scaling,
                    "volatility": volatility_factor if self.volatility_scaling_enabled else 1.0
                },
                "limits_applied": {
                    "max_position_pct": max_position_limit,
                    "max_allowed_size": max_allowed_size
                },
                "risk_mode": self.current_profile.mode.value
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Position size limit calculation failed: {e}")
            return {
                "original_size": base_size,
                "final_size": 0.0,
                "position_allowed": False,
                "error": str(e)
            }
    
    def check_portfolio_constraints(self, 
                                  current_positions: List[Dict[str, Any]],
                                  proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Check if proposed trade violates portfolio constraints"""
        try:
            # Calculate current exposure
            total_exposure = sum(abs(pos.get("exposure_pct", 0)) for pos in current_positions)
            
            # Calculate proposed new exposure
            new_exposure = abs(proposed_trade.get("exposure_pct", 0))
            total_new_exposure = total_exposure + new_exposure
            
            # Check exposure limits
            max_exposure_exceeded = total_new_exposure > self.current_profile.max_exposure_pct
            
            # Check single position limit
            single_position_exceeded = new_exposure > self.current_profile.max_single_position_pct
            
            # Check correlation limits
            correlation_risk = self._check_correlation_limits(current_positions, proposed_trade)
            
            # Check concentration risk
            concentration_risk = self._check_concentration_risk(current_positions, proposed_trade)
            
            # Overall decision
            trade_allowed = not any([
                max_exposure_exceeded,
                single_position_exceeded,
                correlation_risk["exceeded"],
                concentration_risk["exceeded"],
                not self.current_profile.allow_new_positions
            ])
            
            result = {
                "trade_allowed": trade_allowed,
                "risk_mode": self.current_profile.mode.value,
                "exposure_check": {
                    "current_exposure_pct": total_exposure,
                    "new_exposure_pct": new_exposure,
                    "total_new_exposure_pct": total_new_exposure,
                    "max_allowed_pct": self.current_profile.max_exposure_pct,
                    "exceeded": max_exposure_exceeded
                },
                "position_size_check": {
                    "position_exposure_pct": new_exposure,
                    "max_allowed_pct": self.current_profile.max_single_position_pct,
                    "exceeded": single_position_exceeded
                },
                "correlation_check": correlation_risk,
                "concentration_check": concentration_risk,
                "violations": []
            }
            
            # Add violation details
            if max_exposure_exceeded:
                result["violations"].append("Maximum portfolio exposure exceeded")
            if single_position_exceeded:
                result["violations"].append("Maximum single position size exceeded")
            if correlation_risk["exceeded"]:
                result["violations"].append("Correlation limits exceeded")
            if concentration_risk["exceeded"]:
                result["violations"].append("Concentration limits exceeded")
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio constraint check failed: {e}")
            return {
                "trade_allowed": False,
                "error": str(e)
            }
    
    def get_risk_metrics(self, 
                        portfolio_data: Dict[str, Any],
                        market_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Get drawdown statistics
            drawdown_stats = self.drawdown_monitor.get_drawdown_statistics()
            
            # Get kill switch status
            kill_switch_status = self.kill_switch.get_current_status()
            
            # Calculate portfolio metrics
            portfolio_value = portfolio_data.get("total_value", 0)
            positions = portfolio_data.get("positions", [])
            
            total_exposure = sum(abs(pos.get("exposure_pct", 0)) for pos in positions)
            max_single_exposure = max((abs(pos.get("exposure_pct", 0)) for pos in positions), default=0)
            
            # Calculate correlation risk
            correlation_risk_score = self._calculate_correlation_risk_score(positions, market_data)
            
            # Data health assessment
            data_health_score = self._calculate_overall_data_health_score(market_data)
            unhealthy_pairs_pct = self._calculate_unhealthy_pairs_percentage(market_data)
            
            # Risk-adjusted return metrics
            returns_data = portfolio_data.get("returns_history", [])
            sharpe_ratio = self._calculate_sharpe_ratio(returns_data)
            calmar_ratio = drawdown_stats.calmar_ratio
            sortino_ratio = self._calculate_sortino_ratio(returns_data)
            
            # Volatility metrics
            portfolio_volatility = self._calculate_portfolio_volatility(returns_data)
            max_individual_volatility = self._calculate_max_individual_volatility(positions, market_data)
            
            metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                total_exposure_pct=total_exposure,
                max_single_exposure_pct=max_single_exposure,
                correlation_risk_score=correlation_risk_score,
                current_drawdown_pct=drawdown_stats.current_drawdown_pct,
                max_drawdown_pct=drawdown_stats.max_drawdown_pct,
                drawdown_level=self.drawdown_monitor.current_risk_level,
                kill_switch_level=self.kill_switch.current_level,
                active_kill_switches=len(self.kill_switch.active_events),
                data_health_score=data_health_score,
                unhealthy_pairs_pct=unhealthy_pairs_pct,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                portfolio_volatility=portfolio_volatility,
                max_individual_volatility=max_individual_volatility
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return self._create_empty_metrics()
    
    def force_risk_mode(self, 
                       mode: RiskMode,
                       reason: str,
                       duration_minutes: Optional[int] = None) -> bool:
        """Manually force a specific risk mode"""
        try:
            # Create forced profile
            forced_adjustments = self._get_adjustments_for_mode(mode)
            forced_profile = self._create_risk_profile(
                mode, forced_adjustments, datetime.now()
            )
            forced_profile.triggered_by = [f"Manual override: {reason}"]
            
            # Store current profile and apply forced one
            self.profile_history.append(self.current_profile)
            self.current_profile = forced_profile
            
            logger.warning(f"Risk mode manually forced: {mode.value} - {reason}")
            
            # Schedule restoration (if duration specified)
            if duration_minutes:
                # Implementation would depend on task scheduler
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to force risk mode: {e}")
            return False
    
    def get_risk_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive risk analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter recent profiles
            recent_profiles = [
                p for p in self.profile_history
                if p.timestamp >= cutoff_time
            ]
            
            if not recent_profiles:
                return {"status": "no_data"}
            
            # Mode distribution
            mode_distribution = {}
            for mode in RiskMode:
                count = sum(1 for p in recent_profiles if p.mode == mode)
                mode_distribution[mode.value] = {
                    "count": count,
                    "percentage": count / len(recent_profiles)
                }
            
            # Average scaling factors
            avg_kelly_multiplier = np.mean([p.kelly_multiplier for p in recent_profiles])
            avg_position_multiplier = np.mean([p.position_size_multiplier for p in recent_profiles])
            
            # Risk events
            drawdown_analytics = self.drawdown_monitor.get_drawdown_statistics(days_back)
            kill_switch_analytics = self.kill_switch.get_kill_switch_analytics(days_back)
            
            analytics = {
                "analysis_period_days": days_back,
                "total_profile_changes": len(recent_profiles),
                "mode_distribution": mode_distribution,
                "average_scaling_factors": {
                    "kelly_multiplier": avg_kelly_multiplier,
                    "position_multiplier": avg_position_multiplier,
                    "effective_risk": avg_kelly_multiplier * avg_position_multiplier
                },
                "drawdown_analytics": drawdown_analytics,
                "kill_switch_analytics": kill_switch_analytics,
                "risk_efficiency": self._calculate_risk_efficiency(recent_profiles)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Risk analytics failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _determine_risk_mode(self, 
                           drawdown_info: Dict[str, Any],
                           kill_switch_info: Dict[str, Any],
                           data_health_summary: Dict[str, Any]) -> RiskMode:
        """Determine appropriate risk mode based on current conditions"""
        try:
            current_dd = drawdown_info.get("current_drawdown_pct", 0)
            kill_switch_level = self.kill_switch.current_level
            data_health_score = data_health_summary.get("avg_quality_score", 1.0)
            
            # Emergency conditions
            if (kill_switch_level == KillSwitchLevel.EMERGENCY or
                current_dd >= self.risk_mode_thresholds[RiskMode.SHUTDOWN]["drawdown"]):
                return RiskMode.SHUTDOWN
            
            # Defensive conditions
            if (kill_switch_level == KillSwitchLevel.HALT or
                current_dd >= self.risk_mode_thresholds[RiskMode.EMERGENCY]["drawdown"] or
                data_health_score < self.risk_mode_thresholds[RiskMode.EMERGENCY]["data_health"]):
                return RiskMode.EMERGENCY
            
            # Conservative conditions
            if (kill_switch_level == KillSwitchLevel.WARNING or
                current_dd >= self.risk_mode_thresholds[RiskMode.DEFENSIVE]["drawdown"] or
                data_health_score < self.risk_mode_thresholds[RiskMode.DEFENSIVE]["data_health"]):
                return RiskMode.DEFENSIVE
            
            # Conservative conditions
            if (current_dd >= self.risk_mode_thresholds[RiskMode.CONSERVATIVE]["drawdown"] or
                data_health_score < self.risk_mode_thresholds[RiskMode.CONSERVATIVE]["data_health"]):
                return RiskMode.CONSERVATIVE
            
            return RiskMode.NORMAL
            
        except Exception as e:
            logger.error(f"Risk mode determination failed: {e}")
            return RiskMode.DEFENSIVE
    
    def _calculate_risk_adjustments(self, 
                                  risk_mode: RiskMode,
                                  drawdown_info: Dict[str, Any],
                                  kill_switch_info: Dict[str, Any],
                                  data_health_summary: Dict[str, Any]) -> List[RiskAdjustment]:
        """Calculate specific risk adjustments needed"""
        adjustments = []
        
        # Always apply based on mode
        adjustments.extend(self._get_adjustments_for_mode(risk_mode))
        
        # Additional adjustments based on specific conditions
        if drawdown_info.get("current_drawdown_pct", 0) > 0.1:
            adjustments.append(RiskAdjustment.KELLY_SCALING)
        
        if data_health_summary.get("unhealthy_pairs_pct", 0) > 0.2:
            adjustments.append(RiskAdjustment.DATA_GATING)
        
        # Remove duplicates
        return list(set(adjustments))
    
    def _get_adjustments_for_mode(self, mode: RiskMode) -> List[RiskAdjustment]:
        """Get standard adjustments for a risk mode"""
        mode_adjustments = {
            RiskMode.NORMAL: [],
            RiskMode.CONSERVATIVE: [
                RiskAdjustment.KELLY_SCALING,
                RiskAdjustment.POSITION_LIMITS
            ],
            RiskMode.DEFENSIVE: [
                RiskAdjustment.KELLY_SCALING,
                RiskAdjustment.POSITION_LIMITS,
                RiskAdjustment.EXPOSURE_LIMITS,
                RiskAdjustment.DATA_GATING
            ],
            RiskMode.EMERGENCY: [
                RiskAdjustment.KELLY_SCALING,
                RiskAdjustment.POSITION_LIMITS,
                RiskAdjustment.EXPOSURE_LIMITS,
                RiskAdjustment.CONCENTRATION_LIMITS,
                RiskAdjustment.DATA_GATING
            ],
            RiskMode.SHUTDOWN: [
                RiskAdjustment.KELLY_SCALING,
                RiskAdjustment.POSITION_LIMITS,
                RiskAdjustment.EXPOSURE_LIMITS,
                RiskAdjustment.CONCENTRATION_LIMITS,
                RiskAdjustment.CORRELATION_LIMITS,
                RiskAdjustment.DATA_GATING
            ]
        }
        
        return mode_adjustments.get(mode, [])
    
    def _create_risk_profile(self, 
                           mode: RiskMode,
                           adjustments: List[RiskAdjustment],
                           timestamp: datetime) -> RiskProfile:
        """Create risk profile from mode and adjustments"""
        try:
            # Base parameters by mode
            mode_params = {
                RiskMode.NORMAL: {
                    "kelly_multiplier": 1.0,
                    "position_size_multiplier": 1.0,
                    "max_exposure_pct": 0.8,
                    "max_single_position_pct": 0.15,
                    "max_correlation_exposure_pct": 0.3,
                    "allow_new_positions": True,
                    "allow_position_increases": True,
                    "force_position_reductions": False,
                    "paper_trading_only": False,
                    "min_data_quality_score": 0.7,
                    "max_data_staleness_minutes": 30
                },
                RiskMode.CONSERVATIVE: {
                    "kelly_multiplier": 0.75,
                    "position_size_multiplier": 0.8,
                    "max_exposure_pct": 0.6,
                    "max_single_position_pct": 0.1,
                    "max_correlation_exposure_pct": 0.25,
                    "allow_new_positions": True,
                    "allow_position_increases": True,
                    "force_position_reductions": False,
                    "paper_trading_only": False,
                    "min_data_quality_score": 0.8,
                    "max_data_staleness_minutes": 20
                },
                RiskMode.DEFENSIVE: {
                    "kelly_multiplier": 0.5,
                    "position_size_multiplier": 0.5,
                    "max_exposure_pct": 0.4,
                    "max_single_position_pct": 0.05,
                    "max_correlation_exposure_pct": 0.15,
                    "allow_new_positions": True,
                    "allow_position_increases": False,
                    "force_position_reductions": False,
                    "paper_trading_only": False,
                    "min_data_quality_score": 0.85,
                    "max_data_staleness_minutes": 15
                },
                RiskMode.EMERGENCY: {
                    "kelly_multiplier": 0.25,
                    "position_size_multiplier": 0.25,
                    "max_exposure_pct": 0.2,
                    "max_single_position_pct": 0.02,
                    "max_correlation_exposure_pct": 0.1,
                    "allow_new_positions": False,
                    "allow_position_increases": False,
                    "force_position_reductions": True,
                    "paper_trading_only": True,
                    "min_data_quality_score": 0.9,
                    "max_data_staleness_minutes": 10
                },
                RiskMode.SHUTDOWN: {
                    "kelly_multiplier": 0.0,
                    "position_size_multiplier": 0.0,
                    "max_exposure_pct": 0.0,
                    "max_single_position_pct": 0.0,
                    "max_correlation_exposure_pct": 0.0,
                    "allow_new_positions": False,
                    "allow_position_increases": False,
                    "force_position_reductions": True,
                    "paper_trading_only": True,
                    "min_data_quality_score": 1.0,
                    "max_data_staleness_minutes": 5
                }
            }
            
            params = mode_params[mode]
            
            profile = RiskProfile(
                mode=mode,
                timestamp=timestamp,
                kelly_multiplier=params["kelly_multiplier"],
                position_size_multiplier=params["position_size_multiplier"],
                max_exposure_pct=params["max_exposure_pct"],
                max_single_position_pct=params["max_single_position_pct"],
                max_correlation_exposure_pct=params["max_correlation_exposure_pct"],
                allow_new_positions=params["allow_new_positions"],
                allow_position_increases=params["allow_position_increases"],
                force_position_reductions=params["force_position_reductions"],
                paper_trading_only=params["paper_trading_only"],
                min_data_quality_score=params["min_data_quality_score"],
                max_data_staleness_minutes=params["max_data_staleness_minutes"],
                active_adjustments=adjustments,
                triggered_by=[]
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Risk profile creation failed: {e}")
            return self._create_default_profile()
    
    def _create_default_profile(self) -> RiskProfile:
        """Create default conservative risk profile"""
        return RiskProfile(
            mode=RiskMode.CONSERVATIVE,
            timestamp=datetime.now(),
            kelly_multiplier=0.5,
            position_size_multiplier=0.5,
            max_exposure_pct=0.3,
            max_single_position_pct=0.05,
            max_correlation_exposure_pct=0.1,
            allow_new_positions=True,
            allow_position_increases=False,
            force_position_reductions=False,
            paper_trading_only=False,
            min_data_quality_score=0.8,
            max_data_staleness_minutes=15,
            active_adjustments=[RiskAdjustment.KELLY_SCALING, RiskAdjustment.POSITION_LIMITS],
            triggered_by=["Default initialization"]
        )
    
    def _assess_data_health(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data health"""
        try:
            # Simple aggregation - would be more sophisticated in practice
            pairs_data = market_data.get("pairs_data", {})
            
            if not pairs_data:
                return {
                    "avg_quality_score": 0.0,
                    "unhealthy_pairs_pct": 1.0,
                    "total_pairs": 0
                }
            
            quality_scores = []
            unhealthy_count = 0
            
            for pair, data in pairs_data.items():
                quality_score = data.get("data_quality_score", 0.5)
                quality_scores.append(quality_score)
                
                if quality_score < 0.7:
                    unhealthy_count += 1
            
            return {
                "avg_quality_score": np.mean(quality_scores),
                "unhealthy_pairs_pct": unhealthy_count / len(pairs_data),
                "total_pairs": len(pairs_data)
            }
            
        except Exception as e:
            logger.error(f"Data health assessment failed: {e}")
            return {
                "avg_quality_score": 0.0,
                "unhealthy_pairs_pct": 1.0,
                "total_pairs": 0
            }
    
    def _check_data_quality_gate(self, pair: str, market_data: Dict[str, Any]) -> bool:
        """Check if data quality gate allows trading for this pair"""
        try:
            pair_data = market_data.get("pairs_data", {}).get(pair, {})
            
            quality_score = pair_data.get("data_quality_score", 0.0)
            staleness_minutes = pair_data.get("staleness_minutes", 999)
            
            quality_ok = quality_score >= self.current_profile.min_data_quality_score
            staleness_ok = staleness_minutes <= self.current_profile.max_data_staleness_minutes
            
            return quality_ok and staleness_ok
            
        except Exception as e:
            logger.error(f"Data quality gate check failed: {e}")
            return False
    
    def _calculate_volatility_scaling(self, pair: str, market_data: Dict[str, Any]) -> float:
        """Calculate volatility-based scaling factor"""
        try:
            pair_data = market_data.get("pairs_data", {}).get(pair, {})
            current_volatility = pair_data.get("volatility_24h", 0.02)
            baseline_volatility = 0.02  # 2% baseline
            
            # Inverse volatility scaling - reduce size for higher volatility
            volatility_ratio = current_volatility / baseline_volatility
            scaling_factor = 1.0 / (1.0 + volatility_ratio - 1.0)
            
            # Clamp between 0.1 and 1.0
            return max(0.1, min(1.0, scaling_factor))
            
        except Exception as e:
            logger.error(f"Volatility scaling calculation failed: {e}")
            return 0.5
    
    def _check_correlation_limits(self, 
                                 current_positions: List[Dict[str, Any]],
                                 proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Check correlation limits"""
        try:
            # Simplified correlation check
            proposed_pair = proposed_trade.get("pair", "")
            proposed_exposure = abs(proposed_trade.get("exposure_pct", 0))
            
            # Find correlated positions
            correlated_exposure = 0.0
            for position in current_positions:
                pair = position.get("pair", "")
                correlation = self._get_pair_correlation(proposed_pair, pair)
                
                if correlation > 0.7:  # High correlation threshold
                    correlated_exposure += abs(position.get("exposure_pct", 0))
            
            total_correlated_exposure = correlated_exposure + proposed_exposure
            limit_exceeded = total_correlated_exposure > self.current_profile.max_correlation_exposure_pct
            
            return {
                "correlated_exposure_pct": correlated_exposure,
                "proposed_exposure_pct": proposed_exposure,
                "total_correlated_pct": total_correlated_exposure,
                "limit_pct": self.current_profile.max_correlation_exposure_pct,
                "exceeded": limit_exceeded
            }
            
        except Exception as e:
            logger.error(f"Correlation limit check failed: {e}")
            return {"exceeded": False}
    
    def _check_concentration_risk(self, 
                                 current_positions: List[Dict[str, Any]],
                                 proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Check concentration risk limits"""
        try:
            # Simple concentration check - could be more sophisticated
            proposed_exposure = abs(proposed_trade.get("exposure_pct", 0))
            
            # Check if this would be the largest position
            current_exposures = [abs(pos.get("exposure_pct", 0)) for pos in current_positions]
            max_current_exposure = max(current_exposures) if current_exposures else 0
            
            would_be_largest = proposed_exposure > max_current_exposure
            concentration_risk = proposed_exposure > 0.1  # 10% concentration threshold
            
            return {
                "proposed_exposure_pct": proposed_exposure,
                "max_current_exposure_pct": max_current_exposure,
                "would_be_largest": would_be_largest,
                "concentration_threshold_pct": 0.1,
                "exceeded": concentration_risk
            }
            
        except Exception as e:
            logger.error(f"Concentration risk check failed: {e}")
            return {"exceeded": False}
    
    def _get_pair_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs (simplified)"""
        try:
            # Simplified - would use actual correlation matrix in practice
            if pair1 == pair2:
                return 1.0
            
            # Basic heuristics for crypto correlations
            if "BTC" in pair1 and "BTC" in pair2:
                return 0.8
            elif ("ETH" in pair1 and "ETH" in pair2) or ("BTC" in pair1 and "ETH" in pair2):
                return 0.7
            else:
                return 0.3  # Default moderate correlation
                
        except Exception:
            return 0.5
    
    def _calculate_correlation_risk_score(self, 
                                        positions: List[Dict[str, Any]],
                                        market_data: Dict[str, Any]) -> float:
        """Calculate overall portfolio correlation risk score"""
        try:
            if len(positions) < 2:
                return 0.0
            
            total_correlation_risk = 0.0
            pair_count = 0
            
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    pair1 = pos1.get("pair", "")
                    pair2 = pos2.get("pair", "")
                    
                    correlation = self._get_pair_correlation(pair1, pair2)
                    exposure1 = abs(pos1.get("exposure_pct", 0))
                    exposure2 = abs(pos2.get("exposure_pct", 0))
                    
                    # Risk is correlation * product of exposures
                    correlation_risk = correlation * exposure1 * exposure2
                    total_correlation_risk += correlation_risk
                    pair_count += 1
            
            # Normalize by number of pairs
            return total_correlation_risk / pair_count if pair_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Correlation risk score calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_data_health_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall data health score"""
        try:
            pairs_data = market_data.get("pairs_data", {})
            
            if not pairs_data:
                return 0.0
            
            quality_scores = [
                data.get("data_quality_score", 0.0)
                for data in pairs_data.values()
            ]
            
            return np.mean(quality_scores)
            
        except Exception as e:
            logger.error(f"Overall data health score calculation failed: {e}")
            return 0.0
    
    def _calculate_unhealthy_pairs_percentage(self, market_data: Dict[str, Any]) -> float:
        """Calculate percentage of unhealthy pairs"""
        try:
            pairs_data = market_data.get("pairs_data", {})
            
            if not pairs_data:
                return 1.0
            
            unhealthy_count = sum(
                1 for data in pairs_data.values()
                if data.get("data_quality_score", 0.0) < 0.7
            )
            
            return unhealthy_count / len(pairs_data)
            
        except Exception as e:
            logger.error(f"Unhealthy pairs percentage calculation failed: {e}")
            return 1.0
    
    def _calculate_sharpe_ratio(self, returns_data: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns_data) < 2:
                return 0.0
            
            mean_return = np.mean(returns_data)
            std_return = np.std(returns_data)
            
            if std_return == 0:
                return 0.0
            
            # Assuming 0% risk-free rate
            sharpe_ratio = mean_return / std_return
            
            # Annualize (assuming daily returns)
            return sharpe_ratio * np.sqrt(365)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns_data: List[float]) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns_data) < 2:
                return 0.0
            
            mean_return = np.mean(returns_data)
            negative_returns = [r for r in returns_data if r < 0]
            
            if not negative_returns:
                return float('inf')
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = mean_return / downside_deviation
            
            # Annualize
            return sortino_ratio * np.sqrt(365)
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self, returns_data: List[float]) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(returns_data) < 2:
                return 0.0
            
            volatility = np.std(returns_data)
            
            # Annualize
            return volatility * np.sqrt(365)
            
        except Exception as e:
            logger.error(f"Portfolio volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_max_individual_volatility(self, 
                                           positions: List[Dict[str, Any]],
                                           market_data: Dict[str, Any]) -> float:
        """Calculate maximum individual position volatility"""
        try:
            max_vol = 0.0
            
            for position in positions:
                pair = position.get("pair", "")
                pair_data = market_data.get("pairs_data", {}).get(pair, {})
                volatility = pair_data.get("volatility_24h", 0.0)
                
                max_vol = max(max_vol, volatility)
            
            return max_vol
            
        except Exception as e:
            logger.error(f"Max individual volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_risk_efficiency(self, profiles: List[RiskProfile]) -> float:
        """Calculate risk management efficiency score"""
        try:
            if not profiles:
                return 0.0
            
            # Simple efficiency metric: how often we were in appropriate risk mode
            # This would be more sophisticated with actual performance data
            appropriate_responses = sum(
                1 for profile in profiles
                if profile.mode != RiskMode.NORMAL  # Assuming market had some risk
            )
            
            return appropriate_responses / len(profiles)
            
        except Exception as e:
            logger.error(f"Risk efficiency calculation failed: {e}")
            return 0.5
    
    def _log_risk_mode_change(self, old_mode: RiskMode, new_mode: RiskMode) -> None:
        """Log risk mode changes"""
        logger.warning(f"Risk mode changed: {old_mode.value} -> {new_mode.value}")
    
    def _create_empty_metrics(self) -> RiskMetrics:
        """Create empty risk metrics"""
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=0.0,
            total_exposure_pct=0.0,
            max_single_exposure_pct=0.0,
            correlation_risk_score=0.0,
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            drawdown_level=DrawdownLevel.NORMAL,
            kill_switch_level=KillSwitchLevel.ACTIVE,
            active_kill_switches=0,
            data_health_score=0.0,
            unhealthy_pairs_pct=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            portfolio_volatility=0.0,
            max_individual_volatility=0.0
        )