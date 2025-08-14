#!/usr/bin/env python3
"""
Meta-Labeling with Triple Barrier Method
Advanced signal quality filtering using Lopez de Prado's method
Filters false signals and focuses on trade quality instead of just direction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

from core.structured_logger import get_structured_logger


class TripleBarrierLabeler:
    """Meta-labeling system using triple barrier method for signal quality"""

    def __init__(
        self,
        profit_target: float = 0.02,  # 2% profit target
        stop_loss: float = 0.01,  # 1% stop loss
        max_hold_days: int = 5,
    ):  # Max 5 days holding period
        self.logger = get_structured_logger("TripleBarrierLabeler")
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days

    def apply_triple_barrier(self, price_data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier method to evaluate signal quality"""

        self.logger.info(f"Applying triple barrier to {len(signals)} signals")

        try:
            # Ensure price data is sorted by timestamp
            price_data = price_data.sort_values("timestamp").reset_index(drop=True)
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])

            labeled_signals = []

            for _, signal in signals.iterrows():
                label_result = self._evaluate_signal_quality(price_data, signal)
                if label_result:
                    labeled_signals.append(label_result)

            if labeled_signals:
                result_df = pd.DataFrame(labeled_signals)
                self.logger.info(f"Labeled {len(result_df)} signals with triple barrier")
                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Triple barrier labeling failed: {e}")
            return pd.DataFrame()

    def _evaluate_signal_quality(
        self, price_data: pd.DataFrame, signal: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Evaluate quality of a single signal using triple barrier"""

        try:
            signal_time = pd.to_datetime(signal["timestamp"])
            symbol = signal["symbol"]
            direction = signal.get("direction", "BUY")
            confidence = signal.get("confidence", 0.5)

            # Find signal entry point
            entry_idx = self._find_entry_point(price_data, signal_time, symbol)
            if entry_idx is None:
                return None

            entry_price = price_data.iloc[entry_idx]["close"]

            # Define barriers
            if direction == "BUY":
                profit_barrier = entry_price * (1 + self.profit_target)
                loss_barrier = entry_price * (1 - self.stop_loss)
            elif direction == "SELL":
                profit_barrier = entry_price * (1 - self.profit_target)
                loss_barrier = entry_price * (1 + self.stop_loss)
            else:
                return None  # Skip HOLD signals

            # Time barrier
            max_exit_time = signal_time + timedelta(days=self.max_hold_days)

            # Find exit point
            exit_result = self._find_exit_point(
                price_data, entry_idx, profit_barrier, loss_barrier, max_exit_time, symbol
            )

            if exit_result is None:
                return None

            exit_idx, exit_reason, exit_price, actual_return = exit_result

            # Calculate signal quality metrics
            quality_metrics = self._calculate_quality_metrics(
                entry_price, exit_price, actual_return, exit_reason, confidence, direction
            )

            return {
                "signal_id": f"{symbol}_{signal_time.strftime('%Y%m%d_%H%M%S')}",
                "symbol": symbol,
                "entry_time": signal_time.isoformat(),
                "entry_price": entry_price,
                "exit_time": price_data.iloc[exit_idx]["timestamp"].isoformat(),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "direction": direction,
                "actual_return": actual_return,
                "original_confidence": confidence,
                "signal_quality_score": quality_metrics["quality_score"],
                "trade_outcome": quality_metrics["outcome"],
                "hold_duration_hours": quality_metrics["hold_duration_hours"],
                "risk_adjusted_return": quality_metrics["risk_adjusted_return"],
            }

        except Exception as e:
            self.logger.error(f"Signal evaluation failed: {e}")
            return None

    def _find_entry_point(
        self, price_data: pd.DataFrame, signal_time: pd.Timestamp, symbol: str
    ) -> Optional[int]:
        """Find the entry point closest to signal time"""

        symbol_data = price_data[price_data["symbol"] == symbol].copy()
        symbol_data["time_diff"] = abs(symbol_data["timestamp"] - signal_time)

        if len(symbol_data) == 0:
            return None

        closest_idx = symbol_data["time_diff"].idxmin()
        return symbol_data.index.get_loc(closest_idx) if closest_idx in symbol_data.index else None

    def _find_exit_point(
        self,
        price_data: pd.DataFrame,
        entry_idx: int,
        profit_barrier: float,
        loss_barrier: float,
        max_exit_time: pd.Timestamp,
        symbol: str,
    ) -> Optional[Tuple]:
        """Find exit point based on triple barrier"""

        try:
            # Get subsequent price data
            future_data = price_data[entry_idx + 1 :].copy()
            future_data = future_data[future_data["symbol"] == symbol]
            future_data = future_data[future_data["timestamp"] <= max_exit_time]

            if len(future_data) == 0:
                return None

            entry_price = price_data.iloc[entry_idx]["close"]

            for idx, row in future_data.iterrows():
                current_price = row["close"]

                # Check profit barrier
                if (profit_barrier > entry_price and current_price >= profit_barrier) or (
                    profit_barrier < entry_price and current_price <= profit_barrier
                ):
                    actual_return = (current_price - entry_price) / entry_price
                    return (idx, "PROFIT_TARGET", current_price, actual_return)

                # Check loss barrier
                if (loss_barrier < entry_price and current_price <= loss_barrier) or (
                    loss_barrier > entry_price and current_price >= loss_barrier
                ):
                    actual_return = (current_price - entry_price) / entry_price
                    return (idx, "STOP_LOSS", current_price, actual_return)

            # Time barrier reached
            if len(future_data) > 0:
                last_row = future_data.iloc[-1]
                last_price = last_row["close"]
                actual_return = (last_price - entry_price) / entry_price
                return (last_row.name, "TIME_LIMIT", last_price, actual_return)

            return None

        except Exception as e:
            self.logger.error(f"Exit point finding failed: {e}")
            return None

    def _calculate_quality_metrics(
        self,
        entry_price: float,
        exit_price: float,
        actual_return: float,
        exit_reason: str,
        confidence: float,
        direction: str,
    ) -> Dict[str, Any]:
        """Calculate signal quality metrics"""

        # Basic outcome classification
        if exit_reason == "PROFIT_TARGET":
            outcome = "WIN"
            quality_boost = 0.3
        elif exit_reason == "STOP_LOSS":
            outcome = "LOSS"
            quality_boost = -0.5
        else:  # TIME_LIMIT
            if actual_return > 0:
                outcome = "SMALL_WIN"
                quality_boost = 0.1
            else:
                outcome = "SMALL_LOSS"
                quality_boost = -0.2

        # Calculate base quality score
        base_quality = confidence

        # Adjust for actual performance
        performance_factor = min(abs(actual_return) * 10, 0.4)  # Cap at 0.4
        if actual_return > 0:
            quality_score = base_quality + performance_factor + quality_boost
        else:
            quality_score = base_quality - performance_factor + quality_boost

        # Ensure quality score is in [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        # Risk-adjusted return (Sharpe-like metric)
        risk_adjusted_return = actual_return / max(abs(actual_return), 0.01)

        return {
            "quality_score": quality_score,
            "outcome": outcome,
            "hold_duration_hours": 24 * self.max_hold_days,  # Simplified
            "risk_adjusted_return": risk_adjusted_return,
        }

    def filter_high_quality_signals(
        self, labeled_signals: pd.DataFrame, min_quality_score: float = 0.7
    ) -> pd.DataFrame:
        """Filter signals based on historical quality score"""

        if len(labeled_signals) == 0:
            return labeled_signals

        high_quality = labeled_signals[labeled_signals["signal_quality_score"] >= min_quality_score]

        self.logger.info(
            f"Filtered to {len(high_quality)}/{len(labeled_signals)} high quality signals"
        )

        return high_quality

    def calculate_model_meta_score(self, labeled_signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate meta-learning scores for signal quality prediction"""

        if len(labeled_signals) == 0:
            return {"meta_accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        # Define "good" signals
        good_signals = labeled_signals["signal_quality_score"] >= 0.7
        predicted_good = labeled_signals["original_confidence"] >= 0.8

        # Calculate meta-learning metrics
        true_positives = sum(good_signals & predicted_good)
        false_positives = sum(~good_signals & predicted_good)
        false_negatives = sum(good_signals & ~predicted_good)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        # Overall meta-accuracy
        correct_predictions = sum(good_signals == predicted_good)
        meta_accuracy = correct_predictions / len(labeled_signals)

        return {
            "meta_accuracy": meta_accuracy,
            "precision": precision,
            "recall": recall,
            "total_signals_evaluated": len(labeled_signals),
            "high_quality_signals": sum(good_signals),
        }
