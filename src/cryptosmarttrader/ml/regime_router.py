# ml/regime_router.py - Mixture-of-Experts regime-aware routing
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketRegimeRouter:
    """Mixture-of-Experts router for different market regimes"""

    def __init__(self, n_regimes=4):
        """
        Args:
            n_regimes: Number of market regimes (default: 4 for bull/bear/sideways/volatile)
        """
        self.n_regimes = n_regimes
        self.regime_model = None
        self.regime_experts = {}
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "bull_trending",
            1: "bear_trending",
            2: "sideways_low_vol",
            3: "volatile_choppy"
        }

    def extract_regime_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features that characterize market regimes"""

        # Calculate returns
        price_data['return_1d'] = price_data['close'].pct_change(1)
        price_data['return_5d'] = price_data['close'].pct_change(5)
        price_data['return_20d'] = price_data['close'].pct_change(20)

        # Volatility measures
        price_data['volatility_5d'] = price_data['return_1d'].rolling(5).std()
        price_data['volatility_20d'] = price_data['return_1d'].rolling(20).std()

        # Trend strength
        price_data['sma_20'] = price_data['close'].rolling(20).mean()
        price_data['sma_50'] = price_data['close'].rolling(50).mean()
        price_data['trend_strength'] = (price_data['close'] - price_data['sma_20']) / price_data['sma_20']

        # Mean reversion indicator
        price_data['mean_reversion'] = (price_data['close'] - price_data['sma_20']) / price_data['volatility_20d']

        # Volume-based features (if available)
        if 'volume' in price_data.columns:
            price_data['volume_sma'] = price_data['volume'].rolling(20).mean()
            price_data['volume_ratio'] = price_data['volume'] / price_data['volume_sma']
        else:
            price_data['volume_ratio'] = 1.0

        # Momentum indicators
        price_data['rsi'] = self._calculate_rsi(price_data['close'])
        price_data['momentum'] = price_data['return_20d']

        # Select regime features
        regime_features = [
            'return_5d', 'return_20d', 'volatility_5d', 'volatility_20d',
            'trend_strength', 'mean_reversion', 'volume_ratio', 'rsi', 'momentum'
        ]

        return price_data[regime_features].dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fit_regime_detector(self, features: pd.DataFrame) -> None:
        """Fit Gaussian Mixture Model for regime detection"""

        # Scale features
        scaled_features = self.scaler.fit_transform(features.dropna())

        # Fit GMM
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )

        self.regime_model.fit(scaled_features)

        # Predict regimes for training data
        regime_predictions = self.regime_model.predict(scaled_features)

        # Analyze regime characteristics
        self._analyze_regimes(features.dropna(), regime_predictions)

        logger.info(f"Fitted regime detector with {self.n_regimes} regimes")

    def _analyze_regimes(self, features: pd.DataFrame, regimes: np.ndarray) -> None:
        """Analyze and characterize detected regimes"""

        regime_stats = {}

        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id
            regime_data = features[regime_mask]

            if len(regime_data) == 0:
                continue

            stats = {
                'count': len(regime_data),
                'avg_volatility': regime_data['volatility_20d'].mean(),
                'avg_return': regime_data['return_20d'].mean(),
                'avg_trend_strength': regime_data['trend_strength'].mean(),
                'avg_rsi': regime_data['rsi'].mean()
            }

            regime_stats[regime_id] = stats

            # Characterize regime based on stats
            if stats['avg_return'] > 0.02 and stats['avg_trend_strength'] > 0:
                self.regime_names[regime_id] = "bull_trending"
            elif stats['avg_return'] < -0.02 and stats['avg_trend_strength'] < 0:
                self.regime_names[regime_id] = "bear_trending"
            elif abs(stats['avg_return']) < 0.01 and stats['avg_volatility'] < 0.02:
                self.regime_names[regime_id] = "sideways_low_vol"
            else:
                self.regime_names[regime_id] = "volatile_choppy"

        logger.info(f"Regime analysis: {regime_stats}")

    def predict_regime(self, current_features: pd.DataFrame) -> dict:
        """Predict current market regime"""

        if self.regime_model is None:
            raise ValueError("Regime detector not fitted. Call fit_regime_detector first.")

        # Scale features
        scaled_features = self.scaler.transform(current_features.dropna())

        # Get regime probabilities
        regime_probs = self.regime_model.predict_proba(scaled_features)
        regime_pred = self.regime_model.predict(scaled_features)

        # Return most recent regime prediction
        latest_regime = regime_pred[-1]
        latest_probs = regime_probs[-1]

        return {
            'regime_id': latest_regime,
            'regime_name': self.regime_names[latest_regime],
            'confidence': latest_probs[latest_regime],
            'probabilities': {
                self.regime_names[i]: prob
                for i, prob in enumerate(latest_probs)
            }
        }

    def train_regime_experts(self, features: pd.DataFrame, targets: pd.DataFrame, regimes: np.ndarray) -> None:
        """Train specialized models for each regime"""

        from sklearn.ensemble import RandomForestRegressor

        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id

            if regime_mask.sum() < 50:  # Need minimum samples
                logger.warning(f"Insufficient data for regime {regime_id}: {regime_mask.sum()} samples")
                continue

            # Get regime-specific data
            regime_features = features[regime_mask]
            regime_targets = targets[regime_mask]

            # Train expert model for this regime
            expert_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            expert_model.fit(regime_features, regime_targets)

            self.regime_experts[regime_id] = expert_model

            logger.info(f"Trained expert for regime {self.regime_names[regime_id]} on {len(regime_features)} samples")

    def predict_with_routing(self, features: pd.DataFrame) -> dict:
        """Make predictions using regime-appropriate expert"""

        # Detect current regime
        regime_info = self.predict_regime(features)
        current_regime = regime_info['regime_id']

        # Get expert model for this regime
        expert_model = self.regime_experts.get(current_regime)

        if expert_model is None:
            # Fallback to most confident expert
            logger.warning(f"No expert for regime {current_regime}, using fallback")
            expert_model = list(self.regime_experts.values())[0]
            current_regime = list(self.regime_experts.keys())[0]

        # Make prediction with regime expert
        prediction = expert_model.predict(features.dropna())

        # Calculate prediction confidence based on regime confidence
        base_confidence = 0.7  # Base model confidence
        regime_confidence = regime_info['confidence']

        # Combine confidences (geometric mean)
        combined_confidence = np.sqrt(base_confidence * regime_confidence)

        return {
            'prediction': prediction[-1],  # Most recent prediction
            'confidence': combined_confidence,
            'regime_used': regime_info['regime_name'],
            'regime_confidence': regime_confidence,
            'expert_model': f"regime_{current_regime}_expert"
        }

    def save_router(self, filepath: str) -> None:
        """Save trained regime router"""

        router_data = {
            'regime_model': self.regime_model,
            'regime_experts': self.regime_experts,
            'scaler': self.scaler,
            'regime_names': self.regime_names,
            'n_regimes': self.n_regimes
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(router_data, filepath)

        logger.info(f"Saved regime router to {filepath}")

    def load_router(self, filepath: str) -> None:
        """Load trained regime router"""

        router_data = joblib.load(filepath)

        self.regime_model = router_data['regime_model']
        self.regime_experts = router_data['regime_experts']
        self.scaler = router_data['scaler']
        self.regime_names = router_data['regime_names']
        self.n_regimes = router_data['n_regimes']

        logger.info(f"Loaded regime router from {filepath}")

def create_regime_aware_predictions(price_data: pd.DataFrame, target_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create regime-aware predictions using mixture of experts

    Args:
        price_data: Historical price data
        target_data: Target returns to predict

    Returns:
        DataFrame with regime-aware predictions
    """

    router = MarketRegimeRouter(n_regimes=4)

    # Extract regime features
    regime_features = router.extract_regime_features(price_data)

    # Fit regime detector
    router.fit_regime_detector(regime_features)

    # Get regime predictions for training data
    scaled_features = router.scaler.transform(regime_features)
    regimes = router.regime_model.predict(scaled_features)

    # Train regime experts
    router.train_regime_experts(regime_features, target_data, regimes)

    # Generate regime-aware predictions
    results = []

    for i in range(len(regime_features)):
        current_features = regime_features.iloc[i:i+1]

        try:
            prediction_result = router.predict_with_routing(current_features)

            results.append({
                'timestamp': price_data.index[i] if hasattr(price_data.index, '__getitem__') else i,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'regime': prediction_result['regime_used'],
                'regime_confidence': prediction_result['regime_confidence']
            })
        except Exception:
            continue

    # Save router for future use
    router.save_router("models/regime_router.pkl")

    return pd.DataFrame(results)
