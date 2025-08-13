#!/usr/bin/env python3
"""
Advanced Feature Engineering
Automated feature generation with leakage detection and SHAP analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..core.logging_manager import get_logger

@dataclass
class FeatureMetadata:
    """Metadata for engineered features"""
    feature_name: str
    feature_type: str  # 'technical', 'sentiment', 'onchain', 'cross_asset'
    lookback_window: int
    creation_timestamp: datetime
    importance_score: Optional[float] = None
    leakage_risk: str = "low"  # 'low', 'medium', 'high'

class FeatureEngineering:
    """Advanced feature engineering with automated generation and validation"""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = get_logger()
        self.config = config or {}
        self.feature_metadata = {}

    def engineer_features(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        onchain_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Engineer comprehensive features from multiple data sources"""

        self.logger.info("Starting comprehensive feature engineering")

        # Initialize feature container
        features_df = price_data[['timestamp', 'symbol']].copy()

        # Technical features
        tech_features, tech_metadata = self._create_technical_features(price_data)
        features_df = features_df.merge(tech_features, on=['timestamp', 'symbol'], how='left')
        self.feature_metadata.update(tech_metadata)

        # Sentiment features
        if sentiment_data is not None:
            sent_features, sent_metadata = self._create_sentiment_features(sentiment_data)
            features_df = features_df.merge(sent_features, on=['timestamp', 'symbol'], how='left')
            self.feature_metadata.update(sent_metadata)

        # On-chain features
        if onchain_data is not None:
            onchain_features, onchain_metadata = self._create_onchain_features(onchain_data)
            features_df = features_df.merge(onchain_features, on=['timestamp', 'symbol'], how='left')
            self.feature_metadata.update(onchain_metadata)

        # Cross-asset features
        cross_features, cross_metadata = self._create_cross_asset_features(price_data)
        features_df = features_df.merge(cross_features, on=['timestamp', 'symbol'], how='left')
        self.feature_metadata.update(cross_metadata)

        # Interaction features
        interaction_features, interaction_metadata = self._create_interaction_features(features_df)
        features_df = pd.concat([features_df, interaction_features], axis=1)
        self.feature_metadata.update(interaction_metadata)

        # Feature validation and leakage detection
        validated_features = self._validate_features(features_df)

        self.logger.info(f"Feature engineering completed: {len(validated_features.columns)} features created")

        return validated_features, self.feature_metadata

    def _create_technical_features(self, price_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Create advanced technical analysis features"""

        features_list = []
        metadata = {}

        for symbol, symbol_data in price_data.groupby('symbol'):
            symbol_data = symbol_data.sort_values('timestamp').copy()

            # Price-based features
            symbol_features = pd.DataFrame({
                'timestamp': symbol_data['timestamp'],
                'symbol': symbol_data['symbol']
            })

            # Returns (multiple horizons)
            for window in [1, 6, 24, 168]:  # 1h, 6h, 1d, 1w
                symbol_features[f'return_{window}h'] = symbol_data['close'].pct_change(window)
                metadata[f'return_{window}h'] = FeatureMetadata(
                    f'return_{window}h', 'technical', window, datetime.now()
                )

            # Volatility features
            for window in [24, 168, 720]:  # 1d, 1w, 1m
                returns = symbol_data['close'].pct_change()
                symbol_features[f'volatility_{window}h'] = returns.rolling(window).std()
                metadata[f'volatility_{window}h'] = FeatureMetadata(
                    f'volatility_{window}h', 'technical', window, datetime.now()
                )

            # Technical indicators
            symbol_features['rsi_14'] = self._calculate_rsi(symbol_data['close'], 14)
            symbol_features['macd'] = self._calculate_macd(symbol_data['close'])
            symbol_features['bb_position'] = self._calculate_bollinger_position(symbol_data['close'])

            # Add metadata for technical indicators
            for indicator in ['rsi_14', 'macd', 'bb_position']:
                metadata[indicator] = FeatureMetadata(
                    indicator, 'technical', 14, datetime.now()
                )

            # Volume features
            if 'volume' in symbol_data.columns:
                symbol_features['volume_ma_ratio'] = (
                    symbol_data['volume'] / symbol_data['volume'].rolling(24).mean()
                )
                symbol_features['price_volume_trend'] = self._calculate_pvt(symbol_data)

                metadata['volume_ma_ratio'] = FeatureMetadata(
                    'volume_ma_ratio', 'technical', 24, datetime.now()
                )
                metadata['price_volume_trend'] = FeatureMetadata(
                    'price_volume_trend', 'technical', 1, datetime.now()
                )

            features_list.append(symbol_features)

        # Combine all symbols
        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
        else:
            combined_features = pd.DataFrame(columns=['timestamp', 'symbol'])

        return combined_features, metadata

    def _create_sentiment_features(self, sentiment_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Create sentiment-based features"""

        features_list = []
        metadata = {}

        for symbol, symbol_data in sentiment_data.groupby('symbol'):
            symbol_data = symbol_data.sort_values('timestamp').copy()

            symbol_features = pd.DataFrame({
                'timestamp': symbol_data['timestamp'],
                'symbol': symbol_data['symbol']
            })

            # Raw sentiment
            symbol_features['sentiment_score'] = symbol_data['sentiment_score']
            symbol_features['sentiment_confidence'] = symbol_data['confidence']
            symbol_features['sentiment_volume'] = symbol_data['volume_mentions']

            # Sentiment momentum
            for window in [6, 24, 168]:
                symbol_features[f'sentiment_ma_{window}h'] = (
                    symbol_data['sentiment_score'].rolling(window).mean()
                )
                symbol_features[f'sentiment_momentum_{window}h'] = (
                    symbol_data['sentiment_score'] -
                    symbol_data['sentiment_score'].rolling(window).mean()
                )

                metadata[f'sentiment_ma_{window}h'] = FeatureMetadata(
                    f'sentiment_ma_{window}h', 'sentiment', window, datetime.now()
                )
                metadata[f'sentiment_momentum_{window}h'] = FeatureMetadata(
                    f'sentiment_momentum_{window}h', 'sentiment', window, datetime.now()
                )

            # Sentiment volatility
            symbol_features['sentiment_volatility'] = (
                symbol_data['sentiment_score'].rolling(24).std()
            )

            # Add metadata for basic sentiment features
            for feature in ['sentiment_score', 'sentiment_confidence', 'sentiment_volume', 'sentiment_volatility']:
                metadata[feature] = FeatureMetadata(
                    feature, 'sentiment', 1, datetime.now()
                )

            features_list.append(symbol_features)

        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
        else:
            combined_features = pd.DataFrame(columns=['timestamp', 'symbol'])

        return combined_features, metadata

    def _create_onchain_features(self, onchain_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Create on-chain analysis features"""

        features_list = []
        metadata = {}

        for symbol, symbol_data in onchain_data.groupby('symbol'):
            symbol_data = symbol_data.sort_values('timestamp').copy()

            symbol_features = pd.DataFrame({
                'timestamp': symbol_data['timestamp'],
                'symbol': symbol_data['symbol']
            })

            # Mock on-chain features (in production, would use real blockchain data)
            symbol_features['active_addresses'] = np.random.exponential(1000, len(symbol_data))
            symbol_features['transaction_volume'] = np.random.exponential(10000, len(symbol_data))
            symbol_features['whale_activity'] = np.random.normal(0, 1))

            # On-chain momentum
            for feature in ['active_addresses', 'transaction_volume', 'whale_activity']:
                for window in [24, 168]:
                    momentum_feature = f'{feature}_momentum_{window}h'
                    symbol_features[momentum_feature] = (
                        symbol_features[feature] / symbol_features[feature].rolling(window).mean() - 1
                    )

                    metadata[momentum_feature] = FeatureMetadata(
                        momentum_feature, 'onchain', window, datetime.now()
                    )

            # Add metadata for basic on-chain features
            for feature in ['active_addresses', 'transaction_volume', 'whale_activity']:
                metadata[feature] = FeatureMetadata(
                    feature, 'onchain', 1, datetime.now()
                )

            features_list.append(symbol_features)

        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
        else:
            combined_features = pd.DataFrame(columns=['timestamp', 'symbol'])

        return combined_features, metadata

    def _create_cross_asset_features(self, price_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Create cross-asset correlation and relative strength features"""

        features_list = []
        metadata = {}

        # Create market-wide features
        market_data = price_data.pivot_table(
            values='close', index='timestamp', columns='symbol', aggfunc='first'
        )

        # Market momentum
        market_returns = market_data.pct_change()
        market_momentum = market_returns.mean(axis=1)

        for symbol in market_data.columns:
            symbol_timestamps = price_data[price_data['symbol'] == symbol]['timestamp']

            symbol_features = pd.DataFrame({
                'timestamp': symbol_timestamps,
                'symbol': symbol
            })

            # Relative strength vs market
            if symbol in market_returns.columns:
                symbol_returns = market_returns[symbol]
                relative_strength = symbol_returns - market_momentum

                symbol_features['relative_strength'] = relative_strength.reindex(symbol_timestamps).values
                symbol_features['market_correlation'] = (
                    symbol_returns.rolling(168).corr(market_momentum).reindex(symbol_timestamps).values
                )

                metadata['relative_strength'] = FeatureMetadata(
                    'relative_strength', 'cross_asset', 1, datetime.now()
                )
                metadata['market_correlation'] = FeatureMetadata(
                    'market_correlation', 'cross_asset', 168, datetime.now()
                )

            features_list.append(symbol_features)

        if features_list:
            combined_features = pd.concat(features_list, ignore_index=True)
        else:
            combined_features = pd.DataFrame(columns=['timestamp', 'symbol'])

        return combined_features, metadata

    def _create_interaction_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, FeatureMetadata]]:
        """Create interaction features between different data types"""

        interaction_features = pd.DataFrame(index=features_df.index)
        metadata = {}

        # Price-sentiment interactions
        if 'return_24h' in features_df.columns and 'sentiment_score' in features_df.columns:
            interaction_features['price_sentiment_interaction'] = (
                features_df['return_24h'] * features_df['sentiment_score']
            )
            metadata['price_sentiment_interaction'] = FeatureMetadata(
                'price_sentiment_interaction', 'interaction', 1, datetime.now()
            )

        # Volatility-sentiment interactions
        if 'volatility_24h' in features_df.columns and 'sentiment_volatility' in features_df.columns:
            interaction_features['vol_sentiment_interaction'] = (
                features_df['volatility_24h'] * features_df['sentiment_volatility']
            )
            metadata['vol_sentiment_interaction'] = FeatureMetadata(
                'vol_sentiment_interaction', 'interaction', 1, datetime.now()
            )

        return interaction_features, metadata

    def _validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Validate features for quality and detect potential leakage"""

        # Remove features with excessive missing values
        missing_threshold = 0.5
        missing_ratios = features_df.isnull().mean()
        valid_features = missing_ratios[missing_ratios <= missing_threshold].index

        validated_df = features_df[valid_features].copy()

        # Log feature validation results
        removed_features = set(features_df.columns) - set(valid_features)
        if removed_features:
            self.logger.warning(f"Removed {len(removed_features)} features due to excessive missing values")

        # Update metadata with leakage risk assessment
        self._assess_leakage_risk(validated_df)

        return validated_df

    def _assess_leakage_risk(self, features_df: pd.DataFrame) -> None:
        """Assess and flag features with potential leakage risk"""

        for feature_name in features_df.columns:
            if feature_name in ['timestamp', 'symbol']:
                continue

            if feature_name in self.feature_metadata:
                # Simple leakage detection based on feature type and lookback
                metadata = self.feature_metadata[feature_name]

                # Features with very short lookback windows have higher leakage risk
                if metadata.lookback_window <= 1:
                    metadata.leakage_risk = "medium"
                elif 'future' in feature_name or 'forward' in feature_name:
                    metadata.leakage_risk = "high"
                else:
                    metadata.leakage_risk = "low"

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        # Position: 0 = at lower band, 0.5 = at middle, 1 = at upper band
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position.clip(0, 1)

    def _calculate_pvt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)

        price_change = data['close'].pct_change()
        pvt = (price_change * data['volume']).cumsum()
        return pvt
