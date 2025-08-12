"""
Event Detector

Unified event detection system that combines funding, OI, and basis signals
to identify high-probability trading opportunities around market events.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of market events"""
    FUNDING_FLIP = "funding_flip"
    OI_DIVERGENCE = "oi_divergence"
    BASIS_EXTREME = "basis_extreme"
    COMBINED_SIGNAL = "combined_signal"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    INSTITUTIONAL_FLOW = "institutional_flow"


class EventSeverity(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventDirection(Enum):
    """Expected price direction from event"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class MarketEvent:
    """Unified market event"""
    event_id: str
    timestamp: datetime
    pair: str
    exchange: str
    
    # Event classification
    event_type: EventType
    severity: EventSeverity
    direction: EventDirection
    confidence: float           # Overall event confidence (0-1)
    
    # Component signals
    funding_component: Optional[Dict[str, Any]] = None
    oi_component: Optional[Dict[str, Any]] = None
    basis_component: Optional[Dict[str, Any]] = None
    
    # Market context
    price_at_event: float
    volume_context: str         # "high", "normal", "low"
    volatility_context: str     # "high", "normal", "low"
    
    # Timing predictions
    expected_resolution_minutes: int    # Time to expected resolution
    signal_decay_minutes: int          # When signal loses validity
    
    # Trade implications
    suggested_direction: str           # "long", "short", "neutral"
    suggested_size_pct: float         # Suggested position size
    stop_loss_bp: int
    take_profit_bp: int
    
    # Performance tracking
    hit_rate_30min: Optional[float] = None    # Historical 30min hit rate
    hit_rate_2hr: Optional[float] = None      # Historical 2hr hit rate
    avg_pnl_bp: Optional[float] = None        # Average PnL in basis points
    
    @property
    def risk_adjusted_score(self) -> float:
        """Risk-adjusted event score"""
        return self.confidence * self.severity_multiplier * self.hit_rate_multiplier
    
    @property
    def severity_multiplier(self) -> float:
        """Multiplier based on severity"""
        multipliers = {
            EventSeverity.LOW: 0.5,
            EventSeverity.MEDIUM: 0.75,
            EventSeverity.HIGH: 1.0,
            EventSeverity.CRITICAL: 1.25
        }
        return multipliers.get(self.severity, 0.75)
    
    @property
    def hit_rate_multiplier(self) -> float:
        """Multiplier based on historical hit rates"""
        if self.hit_rate_2hr is not None and self.hit_rate_2hr > 0:
            return min(1.5, self.hit_rate_2hr / 0.6)  # Normalized to 60% baseline
        return 1.0


class EventDetector:
    """
    Unified event detection combining funding, OI, and basis analysis
    """
    
    def __init__(self, funding_analyzer, oi_tracker, basis_analyzer):
        self.funding_analyzer = funding_analyzer
        self.oi_tracker = oi_tracker
        self.basis_analyzer = basis_analyzer
        
        self.event_history = []
        self.performance_tracker = {}
        
        # Event detection thresholds
        self.confidence_thresholds = {
            EventSeverity.LOW: 0.3,
            EventSeverity.MEDIUM: 0.5,
            EventSeverity.HIGH: 0.7,
            EventSeverity.CRITICAL: 0.85
        }
        
        # Combination weights
        self.component_weights = {
            "funding": 0.35,
            "oi": 0.35,
            "basis": 0.30
        }
        
    def detect_market_events(self, 
                           exchange: str,
                           pair: str,
                           current_price: float,
                           market_context: Dict[str, Any]) -> List[MarketEvent]:
        """Detect unified market events from all components"""
        try:
            events = []
            
            # Get component signals
            funding_signals = self._get_funding_signals(exchange, pair, current_price)
            oi_signals = self._get_oi_signals(exchange, pair, current_price)
            basis_signals = self._get_basis_signals(exchange, pair, current_price)
            
            # Detect individual component events
            events.extend(self._detect_funding_events(funding_signals, market_context))
            events.extend(self._detect_oi_events(oi_signals, market_context))
            events.extend(self._detect_basis_events(basis_signals, market_context))
            
            # Detect combined events
            combined_events = self._detect_combined_events(
                funding_signals, oi_signals, basis_signals, market_context
            )
            events.extend(combined_events)
            
            # Filter and rank events
            significant_events = self._filter_significant_events(events)
            
            # Add performance context
            for event in significant_events:
                self._add_performance_context(event)
            
            # Store events
            for event in significant_events:
                self.event_history.append(event)
                logger.info(f"Market event detected: {event.event_type.value} for {pair} on {exchange}")
            
            return significant_events
            
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            return []
    
    def get_event_forecast(self, 
                          exchange: str,
                          pair: str,
                          horizon_minutes: int = 120) -> Dict[str, Any]:
        """Get event-based price forecast"""
        try:
            # Get recent events
            recent_events = self._get_recent_events(exchange, pair, hours_back=24)
            
            if not recent_events:
                return {"status": "no_recent_events"}
            
            # Active events (not yet resolved)
            active_events = [
                event for event in recent_events
                if self._is_event_active(event, horizon_minutes)
            ]
            
            if not active_events:
                return {"status": "no_active_events"}
            
            # Aggregate forecast
            forecast = self._aggregate_event_forecast(active_events, horizon_minutes)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Event forecast failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_event_analytics(self, 
                           exchange: Optional[str] = None,
                           pair: Optional[str] = None,
                           days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive event analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            # Filter events
            filtered_events = [
                event for event in self.event_history
                if (event.timestamp >= cutoff_time and
                    (exchange is None or event.exchange == exchange) and
                    (pair is None or event.pair == pair))
            ]
            
            if not filtered_events:
                return {"status": "no_events"}
            
            analytics = self._calculate_event_analytics(filtered_events)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Event analytics failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_funding_signals(self, exchange: str, pair: str, current_price: float) -> Dict[str, Any]:
        """Get funding-based signals"""
        try:
            return self.funding_analyzer.get_current_funding_signals(exchange, pair, current_price)
        except Exception as e:
            logger.error(f"Failed to get funding signals: {e}")
            return {"status": "error"}
    
    def _get_oi_signals(self, exchange: str, pair: str, current_price: float) -> Dict[str, Any]:
        """Get OI-based signals"""
        try:
            return self.oi_tracker.get_current_oi_signals(exchange, pair, current_price)
        except Exception as e:
            logger.error(f"Failed to get OI signals: {e}")
            return {"status": "error"}
    
    def _get_basis_signals(self, exchange: str, pair: str, current_price: float) -> List[Any]:
        """Get basis-based signals"""
        try:
            return self.basis_analyzer.generate_basis_signals(exchange, pair, current_price)
        except Exception as e:
            logger.error(f"Failed to get basis signals: {e}")
            return []
    
    def _detect_funding_events(self, 
                              funding_signals: Dict[str, Any],
                              market_context: Dict[str, Any]) -> List[MarketEvent]:
        """Detect funding-based events"""
        try:
            events = []
            
            if funding_signals.get("status") != "error" and "signals" in funding_signals:
                for signal in funding_signals["signals"]:
                    if signal.get("confidence", 0) > 0.5:
                        
                        # Determine severity
                        severity = self._classify_severity(signal["confidence"])
                        
                        # Map direction
                        direction = self._map_signal_direction(signal)
                        
                        event = MarketEvent(
                            event_id=f"funding_{funding_signals.get('exchange', '')}_{funding_signals.get('pair', '')}_{datetime.now().timestamp()}",
                            timestamp=datetime.now(),
                            pair=funding_signals.get("pair", ""),
                            exchange=funding_signals.get("exchange", ""),
                            event_type=EventType.FUNDING_FLIP,
                            severity=severity,
                            direction=direction,
                            confidence=signal["confidence"],
                            funding_component=signal,
                            price_at_event=market_context.get("current_price", 0),
                            volume_context=self._classify_volume_context(market_context),
                            volatility_context=self._classify_volatility_context(market_context),
                            expected_resolution_minutes=signal.get("target_hold_hours", 24) * 60,
                            signal_decay_minutes=signal.get("target_hold_hours", 24) * 60,
                            suggested_direction=signal["direction"],
                            suggested_size_pct=0.02,  # 2% default
                            stop_loss_bp=signal.get("stop_loss_bp", 150),
                            take_profit_bp=signal.get("take_profit_bp", 300)
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Funding event detection failed: {e}")
            return []
    
    def _detect_oi_events(self, 
                         oi_signals: Dict[str, Any],
                         market_context: Dict[str, Any]) -> List[MarketEvent]:
        """Detect OI-based events"""
        try:
            events = []
            
            if oi_signals.get("status") != "error" and "signals" in oi_signals:
                for signal in oi_signals["signals"]:
                    if signal.get("confidence", 0) > 0.5:
                        
                        severity = self._classify_severity(signal["confidence"])
                        direction = self._map_signal_direction(signal)
                        
                        event = MarketEvent(
                            event_id=f"oi_{oi_signals.get('exchange', '')}_{oi_signals.get('pair', '')}_{datetime.now().timestamp()}",
                            timestamp=datetime.now(),
                            pair=oi_signals.get("pair", ""),
                            exchange=oi_signals.get("exchange", ""),
                            event_type=EventType.OI_DIVERGENCE,
                            severity=severity,
                            direction=direction,
                            confidence=signal["confidence"],
                            oi_component=signal,
                            price_at_event=market_context.get("current_price", 0),
                            volume_context=self._classify_volume_context(market_context),
                            volatility_context=self._classify_volatility_context(market_context),
                            expected_resolution_minutes=signal.get("target_hold_hours", 48) * 60,
                            signal_decay_minutes=signal.get("target_hold_hours", 48) * 60,
                            suggested_direction=signal["direction"],
                            suggested_size_pct=0.03,
                            stop_loss_bp=signal.get("stop_loss_bp", 150),
                            take_profit_bp=signal.get("take_profit_bp", 300)
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"OI event detection failed: {e}")
            return []
    
    def _detect_basis_events(self, 
                            basis_signals: List[Any],
                            market_context: Dict[str, Any]) -> List[MarketEvent]:
        """Detect basis-based events"""
        try:
            events = []
            
            for signal in basis_signals:
                if signal.confidence > 0.5:
                    
                    severity = self._classify_severity(signal.confidence)
                    direction = self._map_basis_direction(signal)
                    
                    event = MarketEvent(
                        event_id=signal.signal_id,
                        timestamp=signal.timestamp,
                        pair=signal.pair,
                        exchange=signal.exchange,
                        event_type=EventType.BASIS_EXTREME,
                        severity=severity,
                        direction=direction,
                        confidence=signal.confidence,
                        basis_component=signal,
                        price_at_event=market_context.get("current_price", 0),
                        volume_context=self._classify_volume_context(market_context),
                        volatility_context=self._classify_volatility_context(market_context),
                        expected_resolution_minutes=int(signal.target_hold_hours * 60),
                        signal_decay_minutes=int(signal.target_hold_hours * 60),
                        suggested_direction=signal.direction,
                        suggested_size_pct=signal.max_position_size_pct,
                        stop_loss_bp=signal.stop_loss_bp,
                        take_profit_bp=signal.take_profit_bp
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Basis event detection failed: {e}")
            return []
    
    def _detect_combined_events(self, 
                               funding_signals: Dict[str, Any],
                               oi_signals: Dict[str, Any],
                               basis_signals: List[Any],
                               market_context: Dict[str, Any]) -> List[MarketEvent]:
        """Detect combined multi-component events"""
        try:
            events = []
            
            # Check for signal alignment
            signal_directions = {}
            signal_confidences = {}
            
            # Extract funding direction and confidence
            if funding_signals.get("signals"):
                for signal in funding_signals["signals"]:
                    signal_directions["funding"] = signal["direction"]
                    signal_confidences["funding"] = signal["confidence"]
                    break
            
            # Extract OI direction and confidence
            if oi_signals.get("signals"):
                for signal in oi_signals["signals"]:
                    signal_directions["oi"] = signal["direction"]
                    signal_confidences["oi"] = signal["confidence"]
                    break
            
            # Extract basis direction and confidence
            if basis_signals:
                for signal in basis_signals:
                    signal_directions["basis"] = signal.direction
                    signal_confidences["basis"] = signal.confidence
                    break
            
            # Check for signal alignment (2+ components agreeing)
            if len(signal_directions) >= 2:
                direction_counts = {}
                for component, direction in signal_directions.items():
                    if direction not in direction_counts:
                        direction_counts[direction] = []
                    direction_counts[direction].append(component)
                
                # Find dominant direction
                max_agreement = max(len(components) for components in direction_counts.values())
                
                if max_agreement >= 2:  # At least 2 components agree
                    dominant_direction = None
                    agreeing_components = []
                    
                    for direction, components in direction_counts.items():
                        if len(components) == max_agreement:
                            dominant_direction = direction
                            agreeing_components = components
                            break
                    
                    # Calculate combined confidence
                    combined_confidence = np.mean([
                        signal_confidences[comp] for comp in agreeing_components
                    ])
                    
                    # Weight by number of agreeing components
                    agreement_boost = 1 + (max_agreement - 1) * 0.2  # 20% boost per additional component
                    combined_confidence = min(0.95, combined_confidence * agreement_boost)
                    
                    if combined_confidence > 0.6:  # Threshold for combined signals
                        
                        severity = self._classify_severity(combined_confidence)
                        
                        event = MarketEvent(
                            event_id=f"combined_{funding_signals.get('exchange', '')}_{funding_signals.get('pair', '')}_{datetime.now().timestamp()}",
                            timestamp=datetime.now(),
                            pair=funding_signals.get("pair", ""),
                            exchange=funding_signals.get("exchange", ""),
                            event_type=EventType.COMBINED_SIGNAL,
                            severity=severity,
                            direction=self._map_combined_direction(dominant_direction),
                            confidence=combined_confidence,
                            funding_component=funding_signals.get("signals", [{}])[0] if "funding" in agreeing_components else None,
                            oi_component=oi_signals.get("signals", [{}])[0] if "oi" in agreeing_components else None,
                            basis_component=basis_signals[0] if basis_signals and "basis" in agreeing_components else None,
                            price_at_event=market_context.get("current_price", 0),
                            volume_context=self._classify_volume_context(market_context),
                            volatility_context=self._classify_volatility_context(market_context),
                            expected_resolution_minutes=90,  # 1.5 hours for combined signals
                            signal_decay_minutes=240,        # 4 hours decay
                            suggested_direction=dominant_direction,
                            suggested_size_pct=0.05 * max_agreement,  # Larger size for combined signals
                            stop_loss_bp=200,
                            take_profit_bp=400
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Combined event detection failed: {e}")
            return []
    
    def _filter_significant_events(self, events: List[MarketEvent]) -> List[MarketEvent]:
        """Filter and rank significant events"""
        try:
            # Filter by minimum confidence
            significant_events = [
                event for event in events
                if event.confidence >= self.confidence_thresholds[EventSeverity.LOW]
            ]
            
            # Remove duplicate events (same type, pair, exchange within 30 minutes)
            filtered_events = []
            for event in significant_events:
                is_duplicate = False
                for existing in filtered_events:
                    if (event.event_type == existing.event_type and
                        event.pair == existing.pair and
                        event.exchange == existing.exchange and
                        abs((event.timestamp - existing.timestamp).total_seconds()) < 1800):  # 30 minutes
                        
                        # Keep the higher confidence event
                        if event.confidence > existing.confidence:
                            filtered_events.remove(existing)
                        else:
                            is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_events.append(event)
            
            # Sort by risk-adjusted score
            filtered_events.sort(key=lambda e: e.risk_adjusted_score, reverse=True)
            
            return filtered_events
            
        except Exception as e:
            logger.error(f"Event filtering failed: {e}")
            return events
    
    def _classify_severity(self, confidence: float) -> EventSeverity:
        """Classify event severity based on confidence"""
        if confidence >= self.confidence_thresholds[EventSeverity.CRITICAL]:
            return EventSeverity.CRITICAL
        elif confidence >= self.confidence_thresholds[EventSeverity.HIGH]:
            return EventSeverity.HIGH
        elif confidence >= self.confidence_thresholds[EventSeverity.MEDIUM]:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def _map_signal_direction(self, signal: Dict[str, Any]) -> EventDirection:
        """Map signal direction to event direction"""
        direction = signal.get("direction", "")
        signal_type = signal.get("type", "")
        
        if "mean_reversion" in signal_type:
            return EventDirection.MEAN_REVERSION
        elif direction == "buy":
            return EventDirection.BULLISH
        elif direction == "sell":
            return EventDirection.BEARISH
        else:
            return EventDirection.NEUTRAL
    
    def _map_basis_direction(self, signal) -> EventDirection:
        """Map basis signal direction to event direction"""
        if hasattr(signal, 'signal_type') and "mean_reversion" in signal.signal_type.value:
            return EventDirection.MEAN_REVERSION
        elif hasattr(signal, 'direction'):
            if signal.direction == "long":
                return EventDirection.BULLISH
            elif signal.direction == "short":
                return EventDirection.BEARISH
        
        return EventDirection.NEUTRAL
    
    def _map_combined_direction(self, direction: str) -> EventDirection:
        """Map combined signal direction to event direction"""
        if direction in ["buy", "long"]:
            return EventDirection.BULLISH
        elif direction in ["sell", "short"]:
            return EventDirection.BEARISH
        else:
            return EventDirection.NEUTRAL
    
    def _classify_volume_context(self, market_context: Dict[str, Any]) -> str:
        """Classify volume context"""
        volume_ratio = market_context.get("volume_ratio_24h", 1.0)
        
        if volume_ratio > 1.5:
            return "high"
        elif volume_ratio < 0.7:
            return "low"
        else:
            return "normal"
    
    def _classify_volatility_context(self, market_context: Dict[str, Any]) -> str:
        """Classify volatility context"""
        volatility = market_context.get("volatility_24h", 0.02)
        
        if volatility > 0.05:  # 5%
            return "high"
        elif volatility < 0.02:  # 2%
            return "low"
        else:
            return "normal"
    
    def _add_performance_context(self, event: MarketEvent) -> None:
        """Add historical performance context to event"""
        try:
            # Get historical performance for this event type
            event_key = f"{event.event_type.value}_{event.pair}_{event.exchange}"
            
            if event_key in self.performance_tracker:
                perf = self.performance_tracker[event_key]
                event.hit_rate_30min = perf.get("hit_rate_30min", 0.5)
                event.hit_rate_2hr = perf.get("hit_rate_2hr", 0.5)
                event.avg_pnl_bp = perf.get("avg_pnl_bp", 0.0)
            else:
                # Default performance assumptions
                event.hit_rate_30min = 0.5
                event.hit_rate_2hr = 0.5
                event.avg_pnl_bp = 0.0
                
        except Exception as e:
            logger.error(f"Performance context addition failed: {e}")
    
    def _get_recent_events(self, exchange: str, pair: str, hours_back: int) -> List[MarketEvent]:
        """Get recent events for pair"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            return [
                event for event in self.event_history
                if (event.timestamp >= cutoff_time and
                    event.exchange == exchange and
                    event.pair == pair)
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    def _is_event_active(self, event: MarketEvent, horizon_minutes: int) -> bool:
        """Check if event is still active"""
        try:
            time_since_event = (datetime.now() - event.timestamp).total_seconds() / 60
            
            return (time_since_event < event.signal_decay_minutes and
                    time_since_event < horizon_minutes)
            
        except Exception as e:
            logger.error(f"Event activity check failed: {e}")
            return False
    
    def _aggregate_event_forecast(self, 
                                 active_events: List[MarketEvent],
                                 horizon_minutes: int) -> Dict[str, Any]:
        """Aggregate forecast from active events"""
        try:
            if not active_events:
                return {"status": "no_active_events"}
            
            # Weight events by confidence and time decay
            weighted_directions = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
            total_weight = 0.0
            
            for event in active_events:
                # Time decay factor
                time_since_event = (datetime.now() - event.timestamp).total_seconds() / 60
                time_decay = max(0.1, 1 - time_since_event / event.signal_decay_minutes)
                
                # Event weight
                weight = event.confidence * event.severity_multiplier * time_decay
                
                # Direction mapping
                if event.direction == EventDirection.BULLISH:
                    weighted_directions["bullish"] += weight
                elif event.direction == EventDirection.BEARISH:
                    weighted_directions["bearish"] += weight
                elif event.direction == EventDirection.MEAN_REVERSION:
                    # Mean reversion contributes to both directions based on current position
                    weighted_directions["bullish"] += weight * 0.5
                    weighted_directions["bearish"] += weight * 0.5
                else:
                    weighted_directions["neutral"] += weight
                
                total_weight += weight
            
            if total_weight == 0:
                return {"status": "no_weighted_signals"}
            
            # Normalize
            for direction in weighted_directions:
                weighted_directions[direction] /= total_weight
            
            # Determine forecast
            max_direction = max(weighted_directions, key=weighted_directions.get)
            forecast_confidence = weighted_directions[max_direction]
            
            # Calculate expected move
            avg_take_profit = np.mean([event.take_profit_bp for event in active_events])
            expected_move_bp = avg_take_profit * forecast_confidence
            
            forecast = {
                "horizon_minutes": horizon_minutes,
                "active_events": len(active_events),
                "forecast_direction": max_direction,
                "forecast_confidence": forecast_confidence,
                "expected_move_bp": expected_move_bp,
                "direction_probabilities": weighted_directions,
                "top_events": [
                    {
                        "event_type": event.event_type.value,
                        "confidence": event.confidence,
                        "direction": event.direction.value,
                        "minutes_since": (datetime.now() - event.timestamp).total_seconds() / 60
                    }
                    for event in sorted(active_events, key=lambda e: e.confidence, reverse=True)[:3]
                ]
            }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Event forecast aggregation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_event_analytics(self, events: List[MarketEvent]) -> Dict[str, Any]:
        """Calculate comprehensive event analytics"""
        try:
            if not events:
                return {"status": "no_events"}
            
            analytics = {
                "total_events": len(events),
                "event_types": {},
                "severity_distribution": {},
                "direction_distribution": {},
                "average_confidence": np.mean([e.confidence for e in events]),
                "events_per_day": len(events) / 30,  # Assuming 30-day period
                "component_usage": {
                    "funding_only": 0,
                    "oi_only": 0,
                    "basis_only": 0,
                    "combined": 0
                }
            }
            
            # Event type distribution
            for event_type in EventType:
                count = sum(1 for e in events if e.event_type == event_type)
                analytics["event_types"][event_type.value] = {
                    "count": count,
                    "percentage": count / len(events),
                    "avg_confidence": np.mean([e.confidence for e in events if e.event_type == event_type]) if count > 0 else 0
                }
            
            # Severity distribution
            for severity in EventSeverity:
                count = sum(1 for e in events if e.severity == severity)
                analytics["severity_distribution"][severity.value] = {
                    "count": count,
                    "percentage": count / len(events)
                }
            
            # Direction distribution
            for direction in EventDirection:
                count = sum(1 for e in events if e.direction == direction)
                analytics["direction_distribution"][direction.value] = {
                    "count": count,
                    "percentage": count / len(events)
                }
            
            # Component usage analysis
            for event in events:
                component_count = sum([
                    1 if event.funding_component else 0,
                    1 if event.oi_component else 0,
                    1 if event.basis_component else 0
                ])
                
                if component_count > 1:
                    analytics["component_usage"]["combined"] += 1
                elif event.funding_component:
                    analytics["component_usage"]["funding_only"] += 1
                elif event.oi_component:
                    analytics["component_usage"]["oi_only"] += 1
                elif event.basis_component:
                    analytics["component_usage"]["basis_only"] += 1
            
            # Performance metrics (if available)
            events_with_performance = [e for e in events if e.hit_rate_2hr is not None]
            if events_with_performance:
                analytics["performance_metrics"] = {
                    "avg_hit_rate_30min": np.mean([e.hit_rate_30min for e in events_with_performance]),
                    "avg_hit_rate_2hr": np.mean([e.hit_rate_2hr for e in events_with_performance]),
                    "avg_pnl_bp": np.mean([e.avg_pnl_bp for e in events_with_performance])
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Event analytics calculation failed: {e}")
            return {"status": "error"}