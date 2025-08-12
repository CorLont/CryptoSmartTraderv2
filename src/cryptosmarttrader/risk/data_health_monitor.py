"""
Data Health Monitor

Monitors data quality and completeness to ensure trading decisions
are based on reliable, up-to-date market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"     # >95% quality
    GOOD = "good"              # 85-95% quality
    FAIR = "fair"              # 70-85% quality
    POOR = "poor"              # 50-70% quality
    CRITICAL = "critical"      # <50% quality


class HealthGate(Enum):
    """Health gate decisions"""
    PASS = "pass"              # Allow trading
    WARN = "warn"              # Allow with warnings
    BLOCK = "block"            # Block new positions
    EMERGENCY = "emergency"    # Emergency stop


@dataclass
class DataHealthMetrics:
    """Data health metrics for a specific source/pair"""
    source: str
    pair: str
    timestamp: datetime
    
    # Completeness metrics
    expected_ticks: int
    received_ticks: int
    missing_ticks: int
    completeness_pct: float
    
    # Timeliness metrics
    last_update_time: datetime
    staleness_minutes: float
    max_gap_minutes: float
    
    # Quality metrics
    invalid_values: int
    outliers_detected: int
    duplicate_ticks: int
    quality_score: float
    
    # Consistency metrics
    price_jumps: int           # Unrealistic price movements
    volume_anomalies: int      # Volume inconsistencies
    consistency_score: float
    
    # Overall assessment
    overall_quality: DataQuality
    health_gate: HealthGate
    
    @property
    def is_healthy(self) -> bool:
        """Check if data is healthy enough for trading"""
        return self.health_gate in [HealthGate.PASS, HealthGate.WARN]
    
    @property
    def missing_data_pct(self) -> float:
        """Calculate missing data percentage"""
        return (self.missing_ticks / self.expected_ticks) if self.expected_ticks > 0 else 0.0


@dataclass
class SystemHealthSummary:
    """Overall system data health summary"""
    timestamp: datetime
    total_pairs_monitored: int
    healthy_pairs: int
    degraded_pairs: int
    critical_pairs: int
    
    # Aggregate metrics
    avg_completeness_pct: float
    avg_staleness_minutes: float
    avg_quality_score: float
    
    # System-wide gates
    overall_health_gate: HealthGate
    trading_recommendation: str
    
    # Issues summary
    top_issues: List[str]
    affected_exchanges: List[str]
    
    @property
    def health_pct(self) -> float:
        """Overall system health percentage"""
        return (self.healthy_pairs / self.total_pairs_monitored) if self.total_pairs_monitored > 0 else 0.0


class DataHealthMonitor:
    """
    Advanced data health monitoring system
    """
    
    def __init__(self):
        self.health_history = {}    # source -> pair -> List[DataHealthMetrics]
        self.alert_history = []
        
        # Configuration thresholds
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.85,
            DataQuality.FAIR: 0.70,
            DataQuality.POOR: 0.50
        }
        
        self.completeness_thresholds = {
            HealthGate.PASS: 0.95,      # >95% completeness
            HealthGate.WARN: 0.90,      # >90% completeness
            HealthGate.BLOCK: 0.80,     # >80% completeness
        }
        
        self.staleness_thresholds = {
            HealthGate.PASS: 5,         # <5 minutes stale
            HealthGate.WARN: 15,        # <15 minutes stale
            HealthGate.BLOCK: 30,       # <30 minutes stale
        }
        
        # Monitoring windows
        self.health_check_window_minutes = 60    # Check last 60 minutes
        self.trend_analysis_window_hours = 24    # Trend analysis over 24 hours
        
    def check_data_health(self, 
                         source: str,
                         pair: str,
                         raw_data: List[Dict[str, Any]],
                         expected_interval_seconds: int = 60) -> DataHealthMetrics:
        """Check data health for a specific source/pair"""
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(minutes=self.health_check_window_minutes)
            
            # Filter data to monitoring window
            recent_data = [
                tick for tick in raw_data
                if tick.get('timestamp', datetime.min) >= window_start
            ]
            
            # Calculate expected ticks
            expected_ticks = int(self.health_check_window_minutes * 60 / expected_interval_seconds)
            received_ticks = len(recent_data)
            missing_ticks = max(0, expected_ticks - received_ticks)
            completeness_pct = received_ticks / expected_ticks if expected_ticks > 0 else 0.0
            
            # Timeliness analysis
            timeliness_metrics = self._analyze_timeliness(recent_data, current_time)
            
            # Quality analysis
            quality_metrics = self._analyze_data_quality(recent_data)
            
            # Consistency analysis
            consistency_metrics = self._analyze_data_consistency(recent_data)
            
            # Overall quality score
            overall_quality_score = (
                completeness_pct * 0.4 +
                quality_metrics["quality_score"] * 0.3 +
                consistency_metrics["consistency_score"] * 0.3
            )
            
            # Classify quality level
            overall_quality = self._classify_quality_level(overall_quality_score)
            
            # Determine health gate
            health_gate = self._determine_health_gate(
                completeness_pct,
                timeliness_metrics["staleness_minutes"],
                overall_quality_score
            )
            
            # Create metrics object
            metrics = DataHealthMetrics(
                source=source,
                pair=pair,
                timestamp=current_time,
                expected_ticks=expected_ticks,
                received_ticks=received_ticks,
                missing_ticks=missing_ticks,
                completeness_pct=completeness_pct,
                last_update_time=timeliness_metrics["last_update_time"],
                staleness_minutes=timeliness_metrics["staleness_minutes"],
                max_gap_minutes=timeliness_metrics["max_gap_minutes"],
                invalid_values=quality_metrics["invalid_values"],
                outliers_detected=quality_metrics["outliers_detected"],
                duplicate_ticks=quality_metrics["duplicate_ticks"],
                quality_score=quality_metrics["quality_score"],
                price_jumps=consistency_metrics["price_jumps"],
                volume_anomalies=consistency_metrics["volume_anomalies"],
                consistency_score=consistency_metrics["consistency_score"],
                overall_quality=overall_quality,
                health_gate=health_gate
            )
            
            # Store in history
            self._store_health_metrics(source, pair, metrics)
            
            # Check for alerts
            self._check_health_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Data health check failed for {source} {pair}: {e}")
            return self._create_error_metrics(source, pair, str(e))
    
    def get_system_health_summary(self, sources: List[str], pairs: List[str]) -> SystemHealthSummary:
        """Get overall system health summary"""
        try:
            current_time = datetime.now()
            all_metrics = []
            
            # Collect latest metrics for all monitored pairs
            for source in sources:
                for pair in pairs:
                    latest_metrics = self._get_latest_metrics(source, pair)
                    if latest_metrics:
                        all_metrics.append(latest_metrics)
            
            if not all_metrics:
                return self._create_empty_summary(current_time)
            
            # Categorize pairs
            healthy_pairs = sum(1 for m in all_metrics if m.health_gate == HealthGate.PASS)
            degraded_pairs = sum(1 for m in all_metrics if m.health_gate == HealthGate.WARN)
            critical_pairs = sum(1 for m in all_metrics if m.health_gate in [HealthGate.BLOCK, HealthGate.EMERGENCY])
            
            # Calculate aggregate metrics
            avg_completeness = np.mean([m.completeness_pct for m in all_metrics])
            avg_staleness = np.mean([m.staleness_minutes for m in all_metrics])
            avg_quality = np.mean([m.quality_score for m in all_metrics])
            
            # Determine overall health gate
            overall_gate = self._determine_overall_health_gate(all_metrics)
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(overall_gate, all_metrics)
            
            # Identify top issues
            top_issues = self._identify_top_issues(all_metrics)
            
            # Identify affected exchanges
            affected_exchanges = list(set(m.source for m in all_metrics if not m.is_healthy))
            
            summary = SystemHealthSummary(
                timestamp=current_time,
                total_pairs_monitored=len(all_metrics),
                healthy_pairs=healthy_pairs,
                degraded_pairs=degraded_pairs,
                critical_pairs=critical_pairs,
                avg_completeness_pct=avg_completeness,
                avg_staleness_minutes=avg_staleness,
                avg_quality_score=avg_quality,
                overall_health_gate=overall_gate,
                trading_recommendation=trading_recommendation,
                top_issues=top_issues,
                affected_exchanges=affected_exchanges
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"System health summary failed: {e}")
            return self._create_empty_summary(datetime.now())
    
    def get_health_trends(self, 
                         source: str,
                         pair: str,
                         hours_back: int = 24) -> Dict[str, Any]:
        """Get health trends for a specific source/pair"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Get historical metrics
            historical_metrics = self._get_historical_metrics(source, pair, cutoff_time)
            
            if len(historical_metrics) < 2:
                return {"status": "insufficient_data"}
            
            # Calculate trends
            completeness_trend = self._calculate_trend([m.completeness_pct for m in historical_metrics])
            staleness_trend = self._calculate_trend([m.staleness_minutes for m in historical_metrics])
            quality_trend = self._calculate_trend([m.quality_score for m in historical_metrics])
            
            # Health deterioration analysis
            deterioration_events = self._detect_deterioration_events(historical_metrics)
            
            # Recovery analysis
            recovery_events = self._detect_recovery_events(historical_metrics)
            
            trends = {
                "source": source,
                "pair": pair,
                "analysis_period_hours": hours_back,
                "data_points": len(historical_metrics),
                "trends": {
                    "completeness": {
                        "direction": "improving" if completeness_trend > 0.01 else "declining" if completeness_trend < -0.01 else "stable",
                        "slope": completeness_trend,
                        "current_value": historical_metrics[-1].completeness_pct
                    },
                    "staleness": {
                        "direction": "improving" if staleness_trend < -0.5 else "declining" if staleness_trend > 0.5 else "stable",
                        "slope": staleness_trend,
                        "current_value": historical_metrics[-1].staleness_minutes
                    },
                    "quality": {
                        "direction": "improving" if quality_trend > 0.01 else "declining" if quality_trend < -0.01 else "stable",
                        "slope": quality_trend,
                        "current_value": historical_metrics[-1].quality_score
                    }
                },
                "deterioration_events": len(deterioration_events),
                "recovery_events": len(recovery_events),
                "stability_score": self._calculate_stability_score(historical_metrics)
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Health trends analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def force_health_gate(self, 
                         source: str,
                         pair: str,
                         gate: HealthGate,
                         reason: str,
                         duration_minutes: int = 60) -> bool:
        """Manually force a health gate decision"""
        try:
            # Create override metrics
            current_time = datetime.now()
            
            override_metrics = DataHealthMetrics(
                source=source,
                pair=pair,
                timestamp=current_time,
                expected_ticks=0,
                received_ticks=0,
                missing_ticks=0,
                completeness_pct=0.0,
                last_update_time=current_time,
                staleness_minutes=0.0,
                max_gap_minutes=0.0,
                invalid_values=0,
                outliers_detected=0,
                duplicate_ticks=0,
                quality_score=0.0,
                price_jumps=0,
                volume_anomalies=0,
                consistency_score=0.0,
                overall_quality=DataQuality.CRITICAL,
                health_gate=gate
            )
            
            # Store override
            self._store_health_metrics(source, pair, override_metrics)
            
            # Log override
            logger.warning(f"Health gate manually overridden: {source} {pair} -> {gate.value} - {reason}")
            
            # Schedule automatic restoration (if needed)
            # Implementation would depend on task scheduler
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to force health gate: {e}")
            return False
    
    def _analyze_timeliness(self, data: List[Dict[str, Any]], current_time: datetime) -> Dict[str, Any]:
        """Analyze data timeliness"""
        try:
            if not data:
                return {
                    "last_update_time": current_time - timedelta(hours=24),
                    "staleness_minutes": 1440.0,  # 24 hours
                    "max_gap_minutes": 1440.0
                }
            
            # Sort by timestamp
            sorted_data = sorted(data, key=lambda x: x.get('timestamp', datetime.min))
            
            # Last update time
            last_update = sorted_data[-1].get('timestamp', current_time - timedelta(hours=24))
            staleness_minutes = (current_time - last_update).total_seconds() / 60
            
            # Calculate gaps between consecutive ticks
            gaps = []
            for i in range(1, len(sorted_data)):
                prev_time = sorted_data[i-1].get('timestamp', datetime.min)
                curr_time = sorted_data[i].get('timestamp', datetime.min)
                gap_minutes = (curr_time - prev_time).total_seconds() / 60
                gaps.append(gap_minutes)
            
            max_gap_minutes = max(gaps) if gaps else 0.0
            
            return {
                "last_update_time": last_update,
                "staleness_minutes": staleness_minutes,
                "max_gap_minutes": max_gap_minutes
            }
            
        except Exception as e:
            logger.error(f"Timeliness analysis failed: {e}")
            return {
                "last_update_time": current_time,
                "staleness_minutes": 0.0,
                "max_gap_minutes": 0.0
            }
    
    def _analyze_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        try:
            if not data:
                return {
                    "invalid_values": 0,
                    "outliers_detected": 0,
                    "duplicate_ticks": 0,
                    "quality_score": 0.0
                }
            
            invalid_values = 0
            outliers_detected = 0
            duplicate_ticks = 0
            
            # Check for invalid values
            for tick in data:
                price = tick.get('price', 0)
                volume = tick.get('volume', 0)
                
                # Invalid price/volume
                if price <= 0 or volume < 0:
                    invalid_values += 1
                
                # Basic outlier detection (very simple heuristic)
                if price > 0:
                    # Check if price is unreasonably high/low compared to recent data
                    recent_prices = [t.get('price', 0) for t in data[-10:] if t.get('price', 0) > 0]
                    if recent_prices:
                        median_price = np.median(recent_prices)
                        if price > median_price * 10 or price < median_price * 0.1:
                            outliers_detected += 1
            
            # Check for duplicates (same timestamp)
            timestamps = [tick.get('timestamp') for tick in data]
            duplicate_ticks = len(timestamps) - len(set(timestamps))
            
            # Calculate quality score
            total_issues = invalid_values + outliers_detected + duplicate_ticks
            quality_score = max(0.0, 1.0 - total_issues / len(data))
            
            return {
                "invalid_values": invalid_values,
                "outliers_detected": outliers_detected,
                "duplicate_ticks": duplicate_ticks,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return {
                "invalid_values": 0,
                "outliers_detected": 0,
                "duplicate_ticks": 0,
                "quality_score": 0.0
            }
    
    def _analyze_data_consistency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data consistency"""
        try:
            if len(data) < 2:
                return {
                    "price_jumps": 0,
                    "volume_anomalies": 0,
                    "consistency_score": 1.0
                }
            
            price_jumps = 0
            volume_anomalies = 0
            
            # Sort by timestamp
            sorted_data = sorted(data, key=lambda x: x.get('timestamp', datetime.min))
            
            # Analyze price movements
            prices = [tick.get('price', 0) for tick in sorted_data if tick.get('price', 0) > 0]
            if len(prices) > 1:
                price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
                # Flag changes > 10% as potential jumps
                price_jumps = sum(1 for change in price_changes if change > 0.1)
            
            # Analyze volume patterns
            volumes = [tick.get('volume', 0) for tick in sorted_data]
            if len(volumes) > 1:
                avg_volume = np.mean(volumes)
                std_volume = np.std(volumes)
                # Flag volumes > 5 standard deviations from mean
                if std_volume > 0:
                    volume_anomalies = sum(1 for vol in volumes if abs(vol - avg_volume) > 5 * std_volume)
            
            # Calculate consistency score
            total_inconsistencies = price_jumps + volume_anomalies
            consistency_score = max(0.0, 1.0 - total_inconsistencies / len(data))
            
            return {
                "price_jumps": price_jumps,
                "volume_anomalies": volume_anomalies,
                "consistency_score": consistency_score
            }
            
        except Exception as e:
            logger.error(f"Data consistency analysis failed: {e}")
            return {
                "price_jumps": 0,
                "volume_anomalies": 0,
                "consistency_score": 1.0
            }
    
    def _classify_quality_level(self, quality_score: float) -> DataQuality:
        """Classify overall quality level"""
        for quality, threshold in self.quality_thresholds.items():
            if quality_score >= threshold:
                return quality
        return DataQuality.CRITICAL
    
    def _determine_health_gate(self, 
                              completeness_pct: float,
                              staleness_minutes: float,
                              quality_score: float) -> HealthGate:
        """Determine appropriate health gate"""
        # Emergency conditions
        if (completeness_pct < 0.5 or 
            staleness_minutes > 60 or 
            quality_score < 0.3):
            return HealthGate.EMERGENCY
        
        # Block conditions
        if (completeness_pct < self.completeness_thresholds[HealthGate.BLOCK] or
            staleness_minutes > self.staleness_thresholds[HealthGate.BLOCK] or
            quality_score < 0.6):
            return HealthGate.BLOCK
        
        # Warning conditions
        if (completeness_pct < self.completeness_thresholds[HealthGate.WARN] or
            staleness_minutes > self.staleness_thresholds[HealthGate.WARN] or
            quality_score < 0.8):
            return HealthGate.WARN
        
        # Pass conditions
        return HealthGate.PASS
    
    def _store_health_metrics(self, source: str, pair: str, metrics: DataHealthMetrics) -> None:
        """Store health metrics in history"""
        try:
            if source not in self.health_history:
                self.health_history[source] = {}
            
            if pair not in self.health_history[source]:
                self.health_history[source][pair] = []
            
            self.health_history[source][pair].append(metrics)
            
            # Keep only recent history (7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.health_history[source][pair] = [
                m for m in self.health_history[source][pair]
                if m.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to store health metrics: {e}")
    
    def _check_health_alerts(self, metrics: DataHealthMetrics) -> None:
        """Check for health alert conditions"""
        try:
            if metrics.health_gate in [HealthGate.BLOCK, HealthGate.EMERGENCY]:
                alert = {
                    "timestamp": metrics.timestamp,
                    "source": metrics.source,
                    "pair": metrics.pair,
                    "gate": metrics.health_gate.value,
                    "issues": []
                }
                
                if metrics.completeness_pct < 0.8:
                    alert["issues"].append(f"Low completeness: {metrics.completeness_pct:.1%}")
                
                if metrics.staleness_minutes > 30:
                    alert["issues"].append(f"Stale data: {metrics.staleness_minutes:.1f} minutes")
                
                if metrics.quality_score < 0.7:
                    alert["issues"].append(f"Low quality: {metrics.quality_score:.1%}")
                
                self.alert_history.append(alert)
                logger.warning(f"Data health alert: {metrics.source} {metrics.pair} - {alert['issues']}")
            
        except Exception as e:
            logger.error(f"Health alert check failed: {e}")
    
    def _get_latest_metrics(self, source: str, pair: str) -> Optional[DataHealthMetrics]:
        """Get latest health metrics for source/pair"""
        try:
            if (source in self.health_history and 
                pair in self.health_history[source] and 
                self.health_history[source][pair]):
                return self.health_history[source][pair][-1]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return None
    
    def _get_historical_metrics(self, 
                               source: str,
                               pair: str,
                               cutoff_time: datetime) -> List[DataHealthMetrics]:
        """Get historical metrics for source/pair"""
        try:
            if (source not in self.health_history or 
                pair not in self.health_history[source]):
                return []
            
            return [
                m for m in self.health_history[source][pair]
                if m.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return []
    
    def _determine_overall_health_gate(self, all_metrics: List[DataHealthMetrics]) -> HealthGate:
        """Determine overall system health gate"""
        emergency_count = sum(1 for m in all_metrics if m.health_gate == HealthGate.EMERGENCY)
        block_count = sum(1 for m in all_metrics if m.health_gate == HealthGate.BLOCK)
        warn_count = sum(1 for m in all_metrics if m.health_gate == HealthGate.WARN)
        
        total_pairs = len(all_metrics)
        
        # Emergency if >20% of pairs are in emergency
        if emergency_count / total_pairs > 0.2:
            return HealthGate.EMERGENCY
        
        # Block if >30% of pairs are blocked or in emergency
        if (block_count + emergency_count) / total_pairs > 0.3:
            return HealthGate.BLOCK
        
        # Warn if >50% of pairs have warnings or worse
        if (warn_count + block_count + emergency_count) / total_pairs > 0.5:
            return HealthGate.WARN
        
        return HealthGate.PASS
    
    def _generate_trading_recommendation(self, 
                                        overall_gate: HealthGate,
                                        all_metrics: List[DataHealthMetrics]) -> str:
        """Generate trading recommendation based on health status"""
        if overall_gate == HealthGate.EMERGENCY:
            return "EMERGENCY STOP: Critical data quality issues detected"
        elif overall_gate == HealthGate.BLOCK:
            return "BLOCK NEW POSITIONS: Significant data quality degradation"
        elif overall_gate == HealthGate.WARN:
            return "REDUCE RISK: Multiple data quality warnings detected"
        else:
            return "NORMAL OPERATIONS: Data quality is acceptable"
    
    def _identify_top_issues(self, all_metrics: List[DataHealthMetrics]) -> List[str]:
        """Identify top data health issues"""
        issues = []
        
        # Count issue types
        high_staleness = sum(1 for m in all_metrics if m.staleness_minutes > 15)
        low_completeness = sum(1 for m in all_metrics if m.completeness_pct < 0.9)
        low_quality = sum(1 for m in all_metrics if m.quality_score < 0.8)
        
        if high_staleness > 0:
            issues.append(f"Stale data affecting {high_staleness} pairs")
        
        if low_completeness > 0:
            issues.append(f"Missing data affecting {low_completeness} pairs")
        
        if low_quality > 0:
            issues.append(f"Quality issues affecting {low_quality} pairs")
        
        return issues[:5]  # Top 5 issues
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope from values"""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
            
        except Exception:
            return 0.0
    
    def _detect_deterioration_events(self, metrics: List[DataHealthMetrics]) -> List[Dict[str, Any]]:
        """Detect health deterioration events"""
        events = []
        
        for i in range(1, len(metrics)):
            prev_score = (metrics[i-1].completeness_pct + metrics[i-1].quality_score) / 2
            curr_score = (metrics[i].completeness_pct + metrics[i].quality_score) / 2
            
            # Significant deterioration
            if curr_score < prev_score - 0.2:  # 20% drop
                events.append({
                    "timestamp": metrics[i].timestamp,
                    "severity": curr_score - prev_score,
                    "type": "deterioration"
                })
        
        return events
    
    def _detect_recovery_events(self, metrics: List[DataHealthMetrics]) -> List[Dict[str, Any]]:
        """Detect health recovery events"""
        events = []
        
        for i in range(1, len(metrics)):
            prev_score = (metrics[i-1].completeness_pct + metrics[i-1].quality_score) / 2
            curr_score = (metrics[i].completeness_pct + metrics[i].quality_score) / 2
            
            # Significant improvement
            if curr_score > prev_score + 0.2:  # 20% improvement
                events.append({
                    "timestamp": metrics[i].timestamp,
                    "improvement": curr_score - prev_score,
                    "type": "recovery"
                })
        
        return events
    
    def _calculate_stability_score(self, metrics: List[DataHealthMetrics]) -> float:
        """Calculate stability score based on metric variations"""
        try:
            if len(metrics) < 3:
                return 0.5
            
            # Calculate coefficient of variation for key metrics
            completeness_values = [m.completeness_pct for m in metrics]
            quality_values = [m.quality_score for m in metrics]
            
            completeness_cv = np.std(completeness_values) / np.mean(completeness_values) if np.mean(completeness_values) > 0 else 1
            quality_cv = np.std(quality_values) / np.mean(quality_values) if np.mean(quality_values) > 0 else 1
            
            # Lower CV = higher stability
            stability_score = max(0.0, 1.0 - (completeness_cv + quality_cv) / 2)
            
            return stability_score
            
        except Exception:
            return 0.5
    
    def _create_error_metrics(self, source: str, pair: str, error: str) -> DataHealthMetrics:
        """Create error metrics when health check fails"""
        return DataHealthMetrics(
            source=source,
            pair=pair,
            timestamp=datetime.now(),
            expected_ticks=0,
            received_ticks=0,
            missing_ticks=0,
            completeness_pct=0.0,
            last_update_time=datetime.now(),
            staleness_minutes=0.0,
            max_gap_minutes=0.0,
            invalid_values=0,
            outliers_detected=0,
            duplicate_ticks=0,
            quality_score=0.0,
            price_jumps=0,
            volume_anomalies=0,
            consistency_score=0.0,
            overall_quality=DataQuality.CRITICAL,
            health_gate=HealthGate.EMERGENCY
        )
    
    def _create_empty_summary(self, timestamp: datetime) -> SystemHealthSummary:
        """Create empty system health summary"""
        return SystemHealthSummary(
            timestamp=timestamp,
            total_pairs_monitored=0,
            healthy_pairs=0,
            degraded_pairs=0,
            critical_pairs=0,
            avg_completeness_pct=0.0,
            avg_staleness_minutes=0.0,
            avg_quality_score=0.0,
            overall_health_gate=HealthGate.EMERGENCY,
            trading_recommendation="NO DATA: System health monitoring unavailable",
            top_issues=["No data available"],
            affected_exchanges=[]
        )