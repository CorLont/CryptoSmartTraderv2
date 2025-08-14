#!/usr/bin/env python3
"""
Meta-Labeling System (Lopez de Prado Triple-Barrier Method)
Filters false signals by training a secondary classifier on trade quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TripleBarrierEvent:
    """Triple barrier event result"""

    entry_time: datetime
    exit_time: datetime
    exit_reason: str  # 'profit_target', 'stop_loss', 'time_limit'
    return_pct: float
    barrier_hit: str  # 'upper', 'lower', 'time'
    days_held: float
    success: bool


@dataclass
class MetaLabel:
    """Meta-label for trade quality"""

    primary_signal: float  # Original prediction (direction/magnitude)
    primary_confidence: float
    meta_label: int  # 0 = don't trade, 1 = trade
    meta_confidence: float
    expected_hit_rate: float
    risk_reward_ratio: float


class TripleBarrierLabeler:
    """Creates labels using Lopez de Prado triple-barrier method"""

    def __init__(
        self,
        profit_target_pct: float = 0.15,  # 15% profit target
        stop_loss_pct: float = 0.08,  # 8% stop loss
        max_hold_days: int = 30,  # 30 day time limit
    ):
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.logger = logging.getLogger(__name__)

    def create_barriers(
        self, price_data: pd.DataFrame, entry_signals: pd.Series
    ) -> List[TripleBarrierEvent]:
        """Create triple-barrier events for all entry signals"""

        events = []

        for signal_idx in entry_signals.index:
            if signal_idx not in price_data.index:
                continue

            entry_price = price_data.loc[signal_idx, "price"]
            entry_time = price_data.loc[signal_idx, "timestamp"]

            # Define barriers
            upper_barrier = entry_price * (1 + self.profit_target_pct)
            lower_barrier = entry_price * (1 - self.stop_loss_pct)
            time_barrier = entry_time + timedelta(days=self.max_hold_days)

            # Find exit point
            exit_event = self._find_first_barrier_touch(
                price_data.loc[signal_idx:], upper_barrier, lower_barrier, time_barrier, entry_price
            )

            if exit_event:
                events.append(exit_event)

        return events

    def _find_first_barrier_touch(
        self,
        price_data: pd.DataFrame,
        upper_barrier: float,
        lower_barrier: float,
        time_barrier: datetime,
        entry_price: float,
    ) -> Optional[TripleBarrierEvent]:
        """Find first barrier touched"""

        for idx, row in price_data.iterrows():
            current_time = row["timestamp"]
            current_price = row["price"]

            # Check time barrier first
            if current_time >= time_barrier:
                return_pct = (current_price - entry_price) / entry_price
                return TripleBarrierEvent(
                    entry_time=price_data.iloc[0]["timestamp"],
                    exit_time=current_time,
                    exit_reason="time_limit",
                    return_pct=return_pct,
                    barrier_hit="time",
                    days_held=(current_time - price_data.iloc[0]["timestamp"]).days,
                    success=return_pct > 0,
                )

            # Check profit target
            if current_price >= upper_barrier:
                return_pct = (current_price - entry_price) / entry_price
                return TripleBarrierEvent(
                    entry_time=price_data.iloc[0]["timestamp"],
                    exit_time=current_time,
                    exit_reason="profit_target",
                    return_pct=return_pct,
                    barrier_hit="upper",
                    days_held=(current_time - price_data.iloc[0]["timestamp"]).days,
                    success=True,
                )

            # Check stop loss
            if current_price <= lower_barrier:
                return_pct = (current_price - entry_price) / entry_price
                return TripleBarrierEvent(
                    entry_time=price_data.iloc[0]["timestamp"],
                    exit_time=current_time,
                    exit_reason="stop_loss",
                    return_pct=return_pct,
                    barrier_hit="lower",
                    days_held=(current_time - price_data.iloc[0]["timestamp"]).days,
                    success=False,
                )

        return None


class MetaLabelingClassifier:
    """Secondary classifier for trade quality (meta-labeling)"""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)

        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=20, random_state=42
            )
        elif model_type == "logistic":
            self.model = LogisticRegression(random_state=42, class_weight="balanced")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_meta_features(
        self, primary_signals: pd.DataFrame, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features for meta-labeling"""

        features = pd.DataFrame(index=primary_signals.index)

        # Primary signal features
        features["primary_pred"] = primary_signals["prediction"]
        features["primary_conf"] = primary_signals["confidence"]
        features["signal_strength"] = np.abs(primary_signals["prediction"])

        # Market state features
        features["volatility_20d"] = market_data["price"].rolling(20).std() / market_data["price"]
        features["volume_ratio"] = (
            market_data["volume_24h"] / market_data["volume_24h"].rolling(7).mean()
        )
        features["price_momentum"] = market_data["price"].pct_change(5)

        # Technical features
        features["rsi_14"] = self._get_technical_analyzer().calculate_indicator("RSI", market_data["price"], 14).values
        features["macd_signal"] = self._calculate_macd_signal(market_data["price"])
        features["bollinger_position"] = self._calculate_bollinger_position(market_data["price"])

        # Cross-sectional features (market regime)
        features["market_stress"] = features["volatility_20d"] > features[
            "volatility_20d"
        ].quantile(0.8)
        features["high_volume"] = features["volume_ratio"] > 2.0
        features["strong_momentum"] = np.abs(features["price_momentum"]) > 0.05

        # Interaction features
        features["conf_x_vol"] = features["primary_conf"] * features["volatility_20d"]
        features["signal_x_momentum"] = features["signal_strength"] * features["price_momentum"]

        return features.fillna(0)

    def fit(
        self, features: pd.DataFrame, barrier_events: List[TripleBarrierEvent]
    ) -> Dict[str, Any]:
        """Train meta-labeling classifier"""

        # Create labels from barrier events
        labels = []
        feature_rows = []

        for event in barrier_events:
            if event.entry_time in features.index:
                # Label: 1 if profitable, 0 if not
                label = 1 if event.success else 0
                labels.append(label)
                feature_rows.append(features.loc[event.entry_time])

        if len(labels) < 10:
            return {"success": False, "error": "Insufficient training data"}

        X = pd.DataFrame(feature_rows).fillna(0)
        y = np.array(labels)

        # Train model
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        # Evaluate performance
        y_pred = self.model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)

        training_result = {
            "success": True,
            "samples_trained": len(labels),
            "success_rate": np.mean(labels),
            "precision": precision,
            "recall": recall,
            "feature_importance": dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

        self.logger.info(f"Meta-labeling trained: {precision:.3f} precision, {recall:.3f} recall")
        return training_result

    def predict_trade_quality(self, features: pd.DataFrame) -> List[MetaLabel]:
        """Predict whether to trade based on signal quality"""

        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Predict probabilities
        X = features.fillna(0)
        meta_probs = self.model.predict_proba(X)[:, 1]  # Probability of success
        meta_predictions = self.model.predict(X)

        meta_labels = []

        for idx, (_, row) in enumerate(features.iterrows()):
            # Calculate risk-reward based on features
            volatility = row.get("volatility_20d", 0.02)
            expected_return = row.get("primary_pred", 0)
            confidence = row.get("primary_conf", 0.5)

            # Estimate risk-reward ratio
            risk_reward = abs(expected_return) / (volatility * 2) if volatility > 0 else 1.0

            meta_label = MetaLabel(
                primary_signal=row.get("primary_pred", 0),
                primary_confidence=row.get("primary_conf", 0.5),
                meta_label=int(meta_predictions[idx]),
                meta_confidence=meta_probs[idx],
                expected_hit_rate=meta_probs[idx],
                risk_reward_ratio=risk_reward,
            )

            meta_labels.append(meta_label)

        return meta_labels

    def _get_technical_analyzer().calculate_indicator("RSI", self, prices: pd.Series, period: int = 14).values -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD signal line"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd - signal

    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        # Position: 0 = lower band, 0.5 = middle, 1 = upper band
        position = (prices - lower_band) / (upper_band - lower_band)
        return position.clip(0, 1)


class MetaLabelingSystem:
    """Complete meta-labeling system integrating triple-barrier and quality classifier"""

    def __init__(
        self,
        profit_target_pct: float = 0.15,
        stop_loss_pct: float = 0.08,
        max_hold_days: int = 30,
        min_meta_confidence: float = 0.6,
    ):
        self.labeler = TripleBarrierLabeler(profit_target_pct, stop_loss_pct, max_hold_days)
        self.classifier = MetaLabelingClassifier()
        self.min_meta_confidence = min_meta_confidence
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_stats = {
            "total_signals": 0,
            "meta_filtered": 0,
            "successful_trades": 0,
            "filter_precision": 0.0,
        }

    def train_meta_labeling(
        self, price_data: pd.DataFrame, primary_signals: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train complete meta-labeling system"""

        # Step 1: Create triple-barrier events
        entry_signals = primary_signals[primary_signals["prediction"].abs() > 0.1]
        barrier_events = self.labeler.create_barriers(price_data, entry_signals.index)

        if len(barrier_events) < 20:
            return {"success": False, "error": "Insufficient barrier events for training"}

        # Step 2: Create meta-features
        meta_features = self.classifier.create_meta_features(primary_signals, price_data)

        # Step 3: Train meta-classifier
        training_result = self.classifier.fit(meta_features, barrier_events)

        if not training_result["success"]:
            return training_result

        # Step 4: Evaluate meta-labeling effectiveness
        evaluation = self._evaluate_meta_system(primary_signals, meta_features, barrier_events)

        training_result.update(evaluation)

        self.logger.info(
            f"Meta-labeling system trained: {evaluation['meta_precision']:.3f} precision"
        )
        return training_result

    def apply_meta_filter(
        self, primary_signals: pd.DataFrame, market_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply meta-labeling filter to primary signals"""

        if not self.classifier.is_fitted:
            return primary_signals, {"status": "not_trained"}

        # Create meta-features
        meta_features = self.classifier.create_meta_features(primary_signals, market_data)

        # Get meta-labels
        meta_labels = self.classifier.predict_trade_quality(meta_features)

        # Filter signals based on meta-confidence
        filtered_signals = []

        for idx, (signal_idx, signal_row) in enumerate(primary_signals.iterrows()):
            meta_label = meta_labels[idx]

            # Apply meta-filter
            if (
                meta_label.meta_label == 1
                and meta_label.meta_confidence >= self.min_meta_confidence
            ):
                # Enhance signal with meta-information
                enhanced_signal = signal_row.copy()
                enhanced_signal["meta_confidence"] = meta_label.meta_confidence
                enhanced_signal["meta_label"] = meta_label.meta_label
                enhanced_signal["expected_hit_rate"] = meta_label.expected_hit_rate
                enhanced_signal["risk_reward_ratio"] = meta_label.risk_reward_ratio
                enhanced_signal["trade_quality_score"] = (
                    meta_label.meta_confidence * meta_label.risk_reward_ratio
                )

                filtered_signals.append(enhanced_signal)

        filtered_df = pd.DataFrame(filtered_signals)

        # Calculate filter statistics
        filter_stats = {
            "original_signals": len(primary_signals),
            "filtered_signals": len(filtered_df),
            "filter_rate": (len(primary_signals) - len(filtered_df)) / len(primary_signals),
            "avg_meta_confidence": np.mean([ml.meta_confidence for ml in meta_labels]),
            "avg_risk_reward": np.mean([ml.risk_reward_ratio for ml in meta_labels]),
            "high_quality_signals": sum(1 for ml in meta_labels if ml.meta_confidence > 0.7),
        }

        self.performance_stats["total_signals"] += len(primary_signals)
        self.performance_stats["meta_filtered"] += len(filtered_df)

        return filtered_df, filter_stats

    def _evaluate_meta_system(
        self,
        primary_signals: pd.DataFrame,
        meta_features: pd.DataFrame,
        barrier_events: List[TripleBarrierEvent],
    ) -> Dict[str, Any]:
        """Evaluate meta-labeling system performance"""

        # Get meta-predictions
        meta_labels = self.classifier.predict_trade_quality(meta_features)

        # Match predictions with outcomes
        successful_events = [e for e in barrier_events if e.success]
        total_events = len(barrier_events)

        # Calculate precision of meta-filter
        high_confidence_predictions = [
            ml
            for ml in meta_labels
            if ml.meta_label == 1 and ml.meta_confidence >= self.min_meta_confidence
        ]

        meta_precision = len(high_confidence_predictions) / len(meta_labels) if meta_labels else 0
        baseline_success_rate = len(successful_events) / total_events if total_events > 0 else 0

        evaluation = {
            "baseline_success_rate": baseline_success_rate,
            "meta_precision": meta_precision,
            "high_confidence_signals": len(high_confidence_predictions),
            "total_barrier_events": total_events,
            "avg_return_successful": np.mean([e.return_pct for e in successful_events])
            if successful_events
            else 0,
            "avg_days_held": np.mean([e.days_held for e in barrier_events])
            if barrier_events
            else 0,
            "profit_target_hit_rate": sum(
                1 for e in barrier_events if e.exit_reason == "profit_target"
            )
            / total_events
            if total_events > 0
            else 0,
        }

        return evaluation

    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""

        filter_precision = (
            self.performance_stats["meta_filtered"] / self.performance_stats["total_signals"]
            if self.performance_stats["total_signals"] > 0
            else 0
        )

        return {
            "performance_stats": self.performance_stats,
            "filter_precision": filter_precision,
            "is_trained": self.classifier.is_fitted,
            "feature_importance": self.classifier.feature_importance,
        }


def create_meta_labeling_pipeline() -> MetaLabelingSystem:
    """Create and return meta-labeling system"""
    return MetaLabelingSystem(
        profit_target_pct=0.15,  # 15% profit target
        stop_loss_pct=0.08,  # 8% stop loss
        max_hold_days=30,  # 30 day maximum hold
        min_meta_confidence=0.6,  # 60% minimum meta-confidence
    )
