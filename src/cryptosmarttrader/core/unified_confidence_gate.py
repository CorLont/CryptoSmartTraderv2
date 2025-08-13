#!/usr/bin/env python3
"""
Unified Confidence Gate - Single Source of Truth
Consolidates class and standalone gates into one reliable system
"""

import pandas as pd
import numpy as np
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..core.unified_structured_logger import get_unified_logger

# Import OpenAI intelligence with fallback
try:
    from integrations.openai_enhanced_intelligence import get_openai_intelligence

    OPENAI_INTEGRATION_AVAILABLE = True
except ImportError:
    OPENAI_INTEGRATION_AVAILABLE = False

    def get_openai_intelligence():
        return None


class UnifiedConfidenceGate:
    """Single reliable confidence gate - replaces both class and standalone versions"""

    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        self.logger = get_unified_logger("UnifiedConfidenceGate")
        self.ai_intelligence = get_openai_intelligence() if OPENAI_INTEGRATION_AVAILABLE else None

        # Tracking for consistency
        self.gate_applications = []
        self.rejection_counts = {"low_confidence": 0, "missing_data": 0, "invalid_predictions": 0}

    async def apply_gate_with_explanations(
        self, predictions_df: pd.DataFrame, gate_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply confidence gate with authentic AI-powered explanations"""

        if gate_id is None:
            gate_id = f"unified_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        start_time = time.time()

        try:
            self.logger.info(f"Applying unified confidence gate: {gate_id}")

            # Apply core filtering (same logic for consistency)
            filtered_df, gate_report = self._apply_core_filtering(predictions_df, gate_id)

            # Generate authentic explanations for passed candidates
            explanations = {}
            if (
                not filtered_df.empty
                and self.ai_intelligence
                and hasattr(self.ai_intelligence, "is_available")
                and self.ai_intelligence.is_available()
            ):
                explanations = await self._generate_authentic_explanations(filtered_df)
            elif not filtered_df.empty:
                # Fallback to feature-based explanations (not random)
                explanations = self._generate_feature_explanations(filtered_df)

            # Add explanations to gate report
            gate_report["explanations"] = explanations
            gate_report["processing_time"] = time.time() - start_time
            gate_report["explanation_method"] = (
                "ai_powered"
                if (
                    self.ai_intelligence
                    and hasattr(self.ai_intelligence, "is_available")
                    and self.ai_intelligence.is_available()
                )
                else "feature_based"
            )

            # Log results with explanation quality
            explanation_count = len(explanations)
            self.logger.info(
                f"Gate {gate_id}: {len(filtered_df)} candidates passed with {explanation_count} explanations"
            )

            return filtered_df, gate_report

        except Exception as e:
            self.logger.error(f"Unified confidence gate failed: {e}")
            return pd.DataFrame(), self._create_error_report(gate_id, str(e))

    def apply_gate_fast(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Fast gate application without explanations (for high-frequency usage)"""

        if predictions_df is None or predictions_df.empty:
            return pd.DataFrame()

        # Validate required columns
        required_columns = ["coin"]
        confidence_columns = [col for col in predictions_df.columns if col.startswith("conf_")]
        prediction_columns = [col for col in predictions_df.columns if col.startswith("pred_")]

        if not all(col in predictions_df.columns for col in required_columns):
            self.logger.warning("Missing required columns for fast gate")
            return pd.DataFrame()

        if not confidence_columns:
            self.logger.warning("No confidence columns found for fast gate")
            return pd.DataFrame()

        # Apply ALL confidence filters (strict AND logic)
        confidence_mask = pd.Series([True] * len(predictions_df), index=predictions_df.index)

        for conf_col in confidence_columns:
            if conf_col in predictions_df.columns:
                conf_values = pd.to_numeric(predictions_df[conf_col], errors="coerce")
                confidence_mask &= conf_values >= self.confidence_threshold

        # Apply validity filters
        validity_mask = pd.Series([True] * len(predictions_df), index=predictions_df.index)

        for pred_col in prediction_columns:
            if pred_col in predictions_df.columns:
                pred_values = pd.to_numeric(predictions_df[pred_col], errors="coerce")
                validity_mask &= (
                    pred_values.notna()
                    & np.isfinite(pred_values)
                    & (np.abs(pred_values) < 10.0)  # No extreme predictions
                )

        # Combine filters
        final_mask = confidence_mask & validity_mask

        # Apply filter and sort
        filtered_df = predictions_df[final_mask].copy()

        if not filtered_df.empty and "pred_30d" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("pred_30d", ascending=False).reset_index(
                drop=True
            )

        return filtered_df

    def _apply_core_filtering(
        self, predictions_df: pd.DataFrame, gate_id: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Core filtering logic (consistent between fast and explained versions)"""

        if predictions_df.empty:
            return pd.DataFrame(), self._create_gate_report(gate_id, 0, 0, "no_input")

        original_count = len(predictions_df)

        # Use fast gate for consistent logic
        filtered_df = self.apply_gate_fast(predictions_df)
        passed_count = len(filtered_df)

        # Calculate rejection reasons
        low_confidence_count = original_count - passed_count  # Simplified for now

        self.rejection_counts["low_confidence"] += low_confidence_count

        return filtered_df, self._create_gate_report(
            gate_id,
            original_count,
            passed_count,
            "success" if passed_count > 0 else "no_candidates",
        )

    async def _generate_authentic_explanations(self, filtered_df: pd.DataFrame) -> Dict[str, str]:
        """Generate authentic AI-powered explanations for passed candidates"""

        explanations = {}

        try:
            # Process top 5 candidates for explanations
            top_candidates = filtered_df.head(5)

            for _, row in top_candidates.iterrows():
                coin = row.get("coin", "Unknown")

                # Extract features for explanation
                features = {}
                for col in row.index:
                    if col.startswith(("pred_", "conf_", "sentiment_", "whale_")):
                        features[col] = row[col]

                # Get authentic AI explanation
                explanation = None
                if self.ai_intelligence and hasattr(
                    self.ai_intelligence, "explain_prediction_confidence"
                ):
                    explanation = await self.ai_intelligence.explain_prediction_confidence(
                        coin=coin,
                        prediction=row.get("pred_30d", 0),
                        confidence=row.get("conf_30d", 0),
                        features=features,
                    )

                if explanation:
                    explanations[coin] = explanation
                else:
                    # Fallback to feature-based if AI fails
                    explanations[coin] = self._generate_single_feature_explanation(row)

                # Rate limiting for API calls
                await asyncio.sleep(0.2)

        except Exception as e:
            self.logger.error(f"AI explanation generation failed: {e}")
            # Fallback to feature-based explanations
            return self._generate_feature_explanations(filtered_df)

        return explanations

    def _generate_feature_explanations(self, filtered_df: pd.DataFrame) -> Dict[str, str]:
        """Generate feature-based explanations (no AI required)"""

        explanations = {}

        for _, row in filtered_df.head(5).iterrows():
            coin = row.get("coin", "Unknown")
            explanations[coin] = self._generate_single_feature_explanation(row)

        return explanations

    def _generate_single_feature_explanation(self, row: pd.Series) -> str:
        """Generate explanation based on actual feature values"""

        coin = row.get("coin", "Unknown")
        pred_30d = row.get("pred_30d", 0)
        conf_30d = row.get("conf_30d", 0)

        # Build explanation based on actual features
        factors = []

        if conf_30d >= 0.9:
            factors.append("very high model confidence")
        elif conf_30d >= 0.85:
            factors.append("high model confidence")
        else:
            factors.append("adequate model confidence")

        if pred_30d > 0.5:
            factors.append("strong growth prediction")
        elif pred_30d > 0.2:
            factors.append("moderate growth prediction")
        else:
            factors.append("positive growth prediction")

        # Add sentiment/whale factors if available
        if "sentiment_score" in row and row["sentiment_score"] > 0.6:
            factors.append("positive sentiment indicators")

        if "whale_activity" in row and row["whale_activity"] > 0.7:
            factors.append("significant whale activity")

        explanation = f"{coin} passed the {self.confidence_threshold:.0%} confidence gate due to: {', '.join(factors)}. Predicted 30-day growth: {pred_30d:.1%} with {conf_30d:.1%} confidence."

        return explanation

    def _create_gate_report(
        self, gate_id: str, input_count: int, output_count: int, status: str
    ) -> Dict[str, Any]:
        """Create standardized gate report"""

        return {
            "gate_id": gate_id,
            "timestamp": datetime.now().isoformat(),
            "input_count": input_count,
            "output_count": output_count,
            "status": status,
            "confidence_threshold": self.confidence_threshold,
            "rejection_rate": (input_count - output_count) / max(input_count, 1),
            "gate_type": "unified",
        }

    def _create_error_report(self, gate_id: str, error: str) -> Dict[str, Any]:
        """Create error report"""

        return {
            "gate_id": gate_id,
            "timestamp": datetime.now().isoformat(),
            "input_count": 0,
            "output_count": 0,
            "status": "error",
            "error": error,
            "gate_type": "unified",
        }


# Global instance for consistent usage
_unified_gate: Optional[UnifiedConfidenceGate] = None


def get_unified_confidence_gate(threshold: float = 0.80) -> UnifiedConfidenceGate:
    """Get global unified confidence gate instance"""
    global _unified_gate
    if _unified_gate is None or _unified_gate.confidence_threshold != threshold:
        _unified_gate = UnifiedConfidenceGate(threshold)
    return _unified_gate


# Compatibility functions to replace old gates
def apply_strict_gate_orchestration(
    pred_df: pd.DataFrame, pred_col="pred_720h", conf_col="conf_720h", threshold=0.80
) -> pd.DataFrame:
    """Compatibility function replacing standalone gate"""
    gate = get_unified_confidence_gate(threshold)
    return gate.apply_gate_fast(pred_df)


def strict_toplist_multi_horizon(pred_df: pd.DataFrame, threshold=0.80) -> Dict[str, pd.DataFrame]:
    """Compatibility function for multi-horizon filtering"""
    gate = get_unified_confidence_gate(threshold)

    if pred_df is None or pred_df.empty:
        return {}

    results = {}
    horizons = ["1h", "24h", "168h", "720h"]  # Standard horizons

    for h in horizons:
        pred_col = f"pred_{h}"
        conf_col = f"conf_{h}"

        if pred_col in pred_df.columns and conf_col in pred_df.columns:
            # Create subset with relevant columns
            subset_cols = ["coin", pred_col, conf_col]
            subset_cols.extend(
                [col for col in pred_df.columns if col.startswith(("sentiment_", "whale_"))]
            )

            subset_df = pred_df[subset_cols].copy()
            filtered = gate.apply_gate_fast(subset_df)
            results[h] = filtered

    return results
