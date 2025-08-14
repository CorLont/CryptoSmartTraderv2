#!/usr/bin/env python3
"""
Confidence Gate Manager - Strict filtering based on confidence thresholds
Implements hard gates that show NOTHING when no candidates meet confidence requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings

warnings.filterwarnings("ignore")

from ..core.logging_manager import get_logger


class GateStatus(str, Enum):
    """Gate status levels"""

    OPEN = "open"  # Gate is open - candidates passed
    CLOSED = "closed"  # Gate is closed - no candidates passed
    PARTIAL = "partial"  # Gate partially open - some candidates passed
    DISABLED = "disabled"  # Gate is disabled for testing


class ConfidenceLevel(str, Enum):
    """Confidence level requirements"""

    STRICT = "strict"  # 90%+ confidence required
    HIGH = "high"  # 80%+ confidence required
    MEDIUM = "medium"  # 70%+ confidence required
    LOW = "low"  # 60%+ confidence required


@dataclass
class ConfidenceGateConfig:
    """Configuration for confidence gates"""

    minimum_confidence: float = 0.8  # 80% minimum confidence
    minimum_candidates: int = 1  # At least 1 candidate must pass
    show_empty_state: bool = True  # Show empty state when no candidates
    gate_level: ConfidenceLevel = ConfidenceLevel.HIGH
    strict_mode: bool = True  # Strict mode = hard gate enforcement


@dataclass
class CandidateResult:
    """Candidate with confidence score"""

    symbol: str
    prediction: float
    confidence: float
    horizon: str
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """Result from confidence gate filtering"""

    gate_id: str
    timestamp: datetime
    gate_status: GateStatus
    config: ConfidenceGateConfig
    total_candidates: int
    passed_candidates: int
    rejected_candidates: int
    passed_results: List[CandidateResult]
    rejection_reasons: Dict[str, int]
    confidence_distribution: Dict[str, int]


class ConfidenceGateManager:
    """Manages confidence-based filtering with strict gates"""

    def __init__(self, config: Optional[ConfidenceGateConfig] = None):
        self.config = config or ConfidenceGateConfig()
        self.logger = get_logger()

        # Gate state tracking
        self.gate_history = []
        self.current_gate_status = GateStatus.CLOSED
        self.last_gate_result = None

        # Confidence thresholds by level
        self.confidence_thresholds = {
            ConfidenceLevel.STRICT: 0.90,
            ConfidenceLevel.HIGH: 0.80,
            ConfidenceLevel.MEDIUM: 0.70,
            ConfidenceLevel.LOW: 0.60,
        }

        self.logger.info(
            "Confidence Gate Manager initialized",
            extra={
                "minimum_confidence": self.config.minimum_confidence,
                "gate_level": self.config.gate_level.value,
                "strict_mode": self.config.strict_mode,
            },
        )

    def apply_confidence_gate(
        self, candidates: List[CandidateResult], gate_id: Optional[str] = None
    ) -> GateResult:
        """Apply confidence gate - STRICT filtering with no fallbacks"""

        gate_id = gate_id or f"gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        gate_start = datetime.now()

        self.logger.info(f"Applying confidence gate: {gate_id}")

        # Initialize gate result
        gate_result = GateResult(
            gate_id=gate_id,
            timestamp=gate_start,
            gate_status=GateStatus.CLOSED,
            config=self.config,
            total_candidates=len(candidates),
            passed_candidates=0,
            rejected_candidates=0,
            passed_results=[],
            rejection_reasons={},
            confidence_distribution={},
        )

        if not candidates:
            # No candidates to evaluate
            gate_result.gate_status = GateStatus.CLOSED
            gate_result.rejection_reasons["no_candidates"] = 1

            self.logger.warning(
                f"Gate {gate_id}: No candidates provided",
                extra={"gate_id": gate_id, "status": "CLOSED"},
            )

            self._update_gate_state(gate_result)
            return gate_result

        # Apply strict confidence filtering
        passed_results = []
        rejected_count = 0
        rejection_reasons = {}
        confidence_distribution = {
            "0-20%": 0,
            "20-40%": 0,
            "40-60%": 0,
            "60-80%": 0,
            "80-90%": 0,
            "90-100%": 0,
        }

        for candidate in candidates:
            # Categorize confidence for distribution
            conf_bucket = self._get_confidence_bucket(candidate.confidence)
            confidence_distribution[conf_bucket] += 1

            # Apply confidence threshold
            if candidate.confidence >= self.config.minimum_confidence:
                # Candidate passes confidence gate
                passed_results.append(candidate)

                self.logger.debug(
                    f"Candidate PASSED gate: {candidate.symbol}",
                    extra={
                        "symbol": candidate.symbol,
                        "confidence": candidate.confidence,
                        "prediction": candidate.prediction,
                        "gate_id": gate_id,
                    },
                )
            else:
                # Candidate REJECTED - insufficient confidence
                rejected_count += 1

                reason = f"confidence_below_{int(self.config.minimum_confidence * 100)}pct"
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = 0
                rejection_reasons[reason] += 1

                self.logger.debug(
                    f"Candidate REJECTED: {candidate.symbol} - confidence {candidate.confidence:.2f} < {self.config.minimum_confidence:.2f}",
                    extra={
                        "symbol": candidate.symbol,
                        "confidence": candidate.confidence,
                        "threshold": self.config.minimum_confidence,
                        "gate_id": gate_id,
                        "action": "REJECTED",
                    },
                )

        # Determine gate status
        passed_count = len(passed_results)

        if passed_count >= self.config.minimum_candidates:
            if passed_count == len(candidates):
                gate_result.gate_status = GateStatus.OPEN
            else:
                gate_result.gate_status = GateStatus.PARTIAL
        else:
            # HARD GATE: Not enough candidates passed
            gate_result.gate_status = GateStatus.CLOSED

            if self.config.strict_mode:
                # In strict mode, clear all results when gate closes
                passed_results = []

                self.logger.warning(
                    f"STRICT GATE CLOSED: {gate_id} - only {passed_count}/{len(candidates)} candidates passed (minimum: {self.config.minimum_candidates})",
                    extra={
                        "gate_id": gate_id,
                        "passed_count": passed_count,
                        "total_candidates": len(candidates),
                        "minimum_required": self.config.minimum_candidates,
                        "action": "GATE_CLOSED",
                    },
                )

        # Finalize gate result
        gate_result.passed_candidates = len(passed_results)
        gate_result.rejected_candidates = len(candidates) - len(passed_results)
        gate_result.passed_results = passed_results
        gate_result.rejection_reasons = rejection_reasons
        gate_result.confidence_distribution = confidence_distribution

        # Log gate result
        self.logger.info(
            f"Confidence gate applied: {gate_id}",
            extra={
                "gate_id": gate_id,
                "status": gate_result.gate_status.value,
                "total_candidates": gate_result.total_candidates,
                "passed_candidates": gate_result.passed_candidates,
                "rejected_candidates": gate_result.rejected_candidates,
                "pass_rate": gate_result.passed_candidates / gate_result.total_candidates
                if gate_result.total_candidates > 0
                else 0,
            },
        )

        # Update gate state
        self._update_gate_state(gate_result)

        return gate_result

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for distribution tracking"""

        if confidence >= 0.9:
            return "90-100%"
        elif confidence >= 0.8:
            return "80-90%"
        elif confidence >= 0.6:
            return "60-80%"
        elif confidence >= 0.4:
            return "40-60%"
        elif confidence >= 0.2:
            return "20-40%"
        else:
            return "0-20%"

    def _update_gate_state(self, gate_result: GateResult):
        """Update internal gate state"""

        self.current_gate_status = gate_result.gate_status
        self.last_gate_result = gate_result

        # Store in history
        self.gate_history.append(gate_result)
        if len(self.gate_history) > 1000:  # Keep last 1000 gate results
            self.gate_history = self.gate_history[-1000:]

    def get_filtered_recommendations(
        self, candidates: List[CandidateResult], sort_by: str = "confidence"
    ) -> List[CandidateResult]:
        """Get filtered recommendations with confidence gate applied"""

        gate_result = self.apply_confidence_gate(candidates)

        if gate_result.gate_status == GateStatus.CLOSED:
            # Gate is closed - return empty list
            self.logger.info(
                f"No recommendations - confidence gate CLOSED",
                extra={
                    "total_candidates": len(candidates),
                    "minimum_confidence": self.config.minimum_confidence,
                },
            )
            return []

        # Sort passed results
        if sort_by == "confidence":
            sorted_results = sorted(
                gate_result.passed_results, key=lambda x: x.confidence, reverse=True
            )
        elif sort_by == "prediction":
            sorted_results = sorted(
                gate_result.passed_results, key=lambda x: x.prediction, reverse=True
            )
        else:
            sorted_results = gate_result.passed_results

        return sorted_results

    def is_gate_open(self) -> bool:
        """Check if confidence gate is currently open"""
        return self.current_gate_status in [GateStatus.OPEN, GateStatus.PARTIAL]

    def get_gate_summary(self) -> Dict[str, Any]:
        """Get comprehensive gate status summary"""

        if not self.last_gate_result:
            return {
                "status": "no_gate_applied",
                "message": "No confidence gate has been applied yet",
            }

        result = self.last_gate_result

        # Calculate recent trends
        recent_gates = (
            self.gate_history[-10:] if len(self.gate_history) >= 10 else self.gate_history
        )
        recent_pass_rates = [
            g.passed_candidates / g.total_candidates if g.total_candidates > 0 else 0
            for g in recent_gates
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "current_gate_status": self.current_gate_status.value,
            "is_gate_open": self.is_gate_open(),
            "last_gate_result": {
                "gate_id": result.gate_id,
                "timestamp": result.timestamp.isoformat(),
                "status": result.gate_status.value,
                "total_candidates": result.total_candidates,
                "passed_candidates": result.passed_candidates,
                "rejected_candidates": result.rejected_candidates,
                "pass_rate": result.passed_candidates / result.total_candidates
                if result.total_candidates > 0
                else 0,
                "confidence_distribution": result.confidence_distribution,
                "rejection_reasons": result.rejection_reasons,
            },
            "configuration": {
                "minimum_confidence": self.config.minimum_confidence,
                "minimum_candidates": self.config.minimum_candidates,
                "gate_level": self.config.gate_level.value,
                "strict_mode": self.config.strict_mode,
                "show_empty_state": self.config.show_empty_state,
            },
            "trends": {
                "total_gates_applied": len(self.gate_history),
                "recent_average_pass_rate": np.mean(recent_pass_rates) if recent_pass_rates else 0,
                "gates_closed_recently": len(
                    [g for g in recent_gates if g.gate_status == GateStatus.CLOSED]
                ),
            },
        }

    def get_empty_state_message(self) -> Dict[str, Any]:
        """Get empty state message when no candidates pass gate"""

        if not self.last_gate_result:
            return {
                "title": "No Analysis Available",
                "message": "No market analysis has been performed yet.",
                "action": "Run market analysis to see recommendations.",
            }

        result = self.last_gate_result

        if result.gate_status == GateStatus.CLOSED:
            # Analyze why gate is closed
            if result.total_candidates == 0:
                return {
                    "title": "No Market Data",
                    "message": "No cryptocurrency data is currently available for analysis.",
                    "action": "Check data sources and try again later.",
                    "details": {"reason": "no_data", "candidates_analyzed": 0},
                }
            else:
                # Candidates exist but none passed confidence threshold
                highest_confidence = max(
                    (
                        c.confidence
                        for c in result.passed_results
                        + [CandidateResult("", 0, 0, "", datetime.now())]
                    ),
                    default=0,
                )

                return {
                    "title": "No High-Confidence Opportunities",
                    "message": f"No cryptocurrencies meet the {self.config.minimum_confidence:.0%} confidence threshold for trading recommendations.",
                    "action": "Market conditions may not be favorable for high-confidence trades. Consider waiting for better opportunities.",
                    "details": {
                        "reason": "low_confidence",
                        "candidates_analyzed": result.total_candidates,
                        "highest_confidence": highest_confidence,
                        "required_confidence": self.config.minimum_confidence,
                        "confidence_distribution": result.confidence_distribution,
                    },
                }

        return {
            "title": "Analysis Complete",
            "message": f"{result.passed_candidates} cryptocurrencies meet confidence requirements.",
            "action": "Review the recommendations below.",
            "details": {"reason": "gate_open", "passed_candidates": result.passed_candidates},
        }


class OrchestrationGateFilter:
    """Orchestration-level confidence gate integration"""

    def __init__(self):
        self.logger = get_logger()
        self.confidence_gate = ConfidenceGateManager()

    def filter_batch_results(
        self, batch_results: List[Dict[str, Any]], confidence_threshold: float = 0.8
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter batch inference results through confidence gate"""

        # Convert batch results to candidates
        candidates = []

        for result in batch_results:
            symbol = result.get("symbol", "")
            predictions = result.get("predictions", {})
            confidence_scores = result.get("confidence_scores", {})

            # Create candidates for each horizon
            for horizon, prediction in predictions.items():
                confidence = confidence_scores.get(horizon, 0.0)

                candidate = CandidateResult(
                    symbol=symbol,
                    prediction=prediction,
                    confidence=confidence,
                    horizon=horizon,
                    timestamp=datetime.now(),
                    features=result.get("features", {}),
                    metadata={"batch_result": result},
                )

                candidates.append(candidate)

        # Update gate configuration
        self.confidence_gate.config.minimum_confidence = confidence_threshold

        # Apply confidence gate
        gate_result = self.confidence_gate.apply_confidence_gate(candidates)

        # Convert passed candidates back to batch results
        filtered_results = []
        seen_symbols = set()

        for candidate in gate_result.passed_results:
            if candidate.symbol not in seen_symbols:
                # Reconstruct result for this symbol
                original_result = candidate.metadata.get("batch_result", {})
                filtered_results.append(original_result)
                seen_symbols.add(candidate.symbol)

        # Create gate summary
        gate_summary = {
            "gate_status": gate_result.gate_status.value,
            "total_candidates": gate_result.total_candidates,
            "passed_candidates": gate_result.passed_candidates,
            "filtered_symbols": len(filtered_results),
            "confidence_threshold": confidence_threshold,
            "empty_state": self.confidence_gate.get_empty_state_message()
            if gate_result.gate_status == GateStatus.CLOSED
            else None,
        }

        self.logger.info(
            f"Batch results filtered through confidence gate",
            extra={
                "original_results": len(batch_results),
                "filtered_results": len(filtered_results),
                "gate_status": gate_result.gate_status.value,
                "confidence_threshold": confidence_threshold,
            },
        )

        return filtered_results, gate_summary


# Global instances
_confidence_gate_manager = None
_orchestration_gate_filter = None


def get_confidence_gate_manager(
    config: Optional[ConfidenceGateConfig] = None,
) -> ConfidenceGateManager:
    """Get global confidence gate manager instance"""
    global _confidence_gate_manager
    if _confidence_gate_manager is None:
        _confidence_gate_manager = ConfidenceGateManager(config)
    return _confidence_gate_manager


def get_orchestration_gate_filter() -> OrchestrationGateFilter:
    """Get global orchestration gate filter instance"""
    global _orchestration_gate_filter
    if _orchestration_gate_filter is None:
        _orchestration_gate_filter = OrchestrationGateFilter()
    return _orchestration_gate_filter
