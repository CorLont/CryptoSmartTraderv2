#!/usr/bin/env python3
"""
Regime Router (Mixture-of-Experts)
Routes predictions to best model per market regime to prevent collapse during stress
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeClassification:
    """Market regime classification result"""
    regime_id: int
    regime_name: str  # 'bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol'
    confidence: float
    regime_features: Dict[str, float]
    duration_days: int
    stability_score: float

@dataclass
class ExpertModelPerformance:
    """Performance metrics for expert model in specific regime"""
    regime_id: int
    model_name: str
    mae: float
    mse: float
    sharpe_ratio: float
    hit_rate: float
    sample_count: int
    confidence_score: float

class MarketRegimeClassifier:
    """Classifies market regimes using multiple indicators"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.regime_model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.is_fitted = False
        self.regime_names = ['bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol']
        self.logger = logging.getLogger(__name__)
        
    def extract_regime_features(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Extract features for regime classification"""
        
        features = pd.DataFrame(index=df.index)
        
        # Price momentum features
        returns_1d = df[price_col].pct_change(24)
        returns_7d = df[price_col].pct_change(168)
        returns_30d = df[price_col].pct_change(720)
        
        features['momentum_1d'] = returns_1d
        features['momentum_7d'] = returns_7d
        features['momentum_30d'] = returns_30d
        
        # Volatility features
        features['volatility_7d'] = returns_1d.rolling(168, min_periods=168).std()
        features['volatility_30d'] = returns_1d.rolling(720, min_periods=720).std()
        features['volatility_ratio'] = features['volatility_7d'] / features['volatility_30d']
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_trend'] = df['volume'].pct_change(168)
            features['volume_volatility'] = df['volume'].rolling(168, min_periods=168).std()
        
        # Market stress indicators
        features['drawdown'] = self._calculate_drawdown(df[price_col])
        features['volatility_spike'] = (features['volatility_7d'] > features['volatility_30d'] * 1.5).astype(float)
        
        # Trend strength
        sma_20 = df[price_col].rolling(480, min_periods=480).mean()  # 20-day SMA
        sma_50 = df[price_col].rolling(1200, min_periods=1200).mean()  # 50-day SMA
        features['trend_strength'] = (sma_20 - sma_50) / sma_50
        
        return features.fillna(0)
    
    def fit(self, regime_features: pd.DataFrame) -> Dict[str, Any]:
        """Fit regime classification model"""
        
        # Select most important features for regime classification
        feature_cols = ['momentum_7d', 'momentum_30d', 'volatility_7d', 'volatility_30d', 
                       'drawdown', 'trend_strength']
        
        available_cols = [col for col in feature_cols if col in regime_features.columns]
        
        if len(available_cols) < 3:
            return {"success": False, "error": "Insufficient features for regime classification"}
        
        X = regime_features[available_cols].dropna()
        
        if len(X) < 100:
            return {"success": False, "error": "Insufficient data for regime fitting"}
        
        # Fit Gaussian Mixture Model
        self.regime_model.fit(X)
        self.is_fitted = True
        
        # Assign regime names based on cluster characteristics
        regime_centers = self.regime_model.means_
        regime_assignments = self._assign_regime_names(regime_centers, available_cols)
        
        result = {
            "success": True,
            "n_regimes": self.n_regimes,
            "features_used": available_cols,
            "data_points": len(X),
            "regime_centers": regime_centers.tolist(),
            "regime_assignments": regime_assignments
        }
        
        self.logger.info(f"Regime classifier fitted with {len(X)} data points")
        return result
    
    def predict_regime(self, regime_features: pd.DataFrame) -> List[RegimeClassification]:
        """Predict market regime for given features"""
        
        if not self.is_fitted:
            raise ValueError("Regime classifier not fitted yet")
        
        feature_cols = ['momentum_7d', 'momentum_30d', 'volatility_7d', 'volatility_30d', 
                       'drawdown', 'trend_strength']
        
        available_cols = [col for col in feature_cols if col in regime_features.columns]
        X = regime_features[available_cols].fillna(0)
        
        # Predict regimes and probabilities
        regime_predictions = self.regime_model.predict(X)
        regime_probabilities = self.regime_model.predict_proba(X)
        
        classifications = []
        
        for i, (regime_id, probs) in enumerate(zip(regime_predictions, regime_probabilities)):
            confidence = probs[regime_id]
            
            # Extract regime features for this observation
            regime_features_dict = X.iloc[i].to_dict()
            
            classification = RegimeClassification(
                regime_id=int(regime_id),
                regime_name=self.regime_names[regime_id % len(self.regime_names)],
                confidence=float(confidence),
                regime_features=regime_features_dict,
                duration_days=1,  # Will be calculated separately
                stability_score=float(confidence)
            )
            
            classifications.append(classification)
        
        return classifications
    
    def _assign_regime_names(self, centers: np.ndarray, feature_cols: List[str]) -> Dict[int, str]:
        """Assign meaningful names to regime clusters based on characteristics"""
        
        assignments = {}
        
        # Find momentum and volatility indices
        momentum_idx = next((i for i, col in enumerate(feature_cols) if 'momentum' in col), 0)
        volatility_idx = next((i for i, col in enumerate(feature_cols) if 'volatility' in col), 1)
        
        for i, center in enumerate(centers):
            momentum = center[momentum_idx] if momentum_idx < len(center) else 0
            volatility = center[volatility_idx] if volatility_idx < len(center) else 0
            
            # Classify based on momentum and volatility
            if momentum > 0:
                if volatility > np.median(centers[:, volatility_idx]):
                    assignments[i] = 'bull_high_vol'
                else:
                    assignments[i] = 'bull_low_vol'
            else:
                if volatility > np.median(centers[:, volatility_idx]):
                    assignments[i] = 'bear_high_vol'
                else:
                    assignments[i] = 'bear_low_vol'
        
        return assignments
    
    def _calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        
        running_max = prices.rolling(720, min_periods=720).max()  # 30-day rolling max
        drawdown = (prices - running_max) / running_max
        
        return drawdown

class ExpertModelManager:
    """Manages multiple expert models for different regimes"""
    
    def __init__(self):
        self.expert_models = {}
        self.model_performance = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize different model types
        self.model_types = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': Ridge(alpha=1.0),
            'ensemble': None  # Will be ensemble of above
        }
    
    def train_expert_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_labels: List[int]
    ) -> Dict[str, Any]:
        """Train expert models for each regime"""
        
        regime_labels_array = np.array(regime_labels)
        unique_regimes = np.unique(regime_labels_array)
        
        training_results = {}
        
        for regime_id in unique_regimes:
            regime_mask = regime_labels_array == regime_id
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 20:  # Minimum samples per regime
                continue
            
            # Train multiple models for this regime
            regime_models = {}
            regime_performance = {}
            
            for model_name, model in self.model_types.items():
                if model_name == 'ensemble':
                    continue
                
                try:
                    # Clone and train model
                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_regime, y_regime)
                    
                    # Evaluate performance
                    y_pred = model_clone.predict(X_regime)
                    
                    mae = mean_absolute_error(y_regime, y_pred)
                    mse = mean_squared_error(y_regime, y_pred)
                    
                    # Calculate hit rate (correct direction prediction)
                    hit_rate = np.mean(np.sign(y_regime) == np.sign(y_pred))
                    
                    # Calculate Sharpe-like ratio
                    sharpe_ratio = np.mean(y_pred) / (np.std(y_pred) + 1e-6)
                    
                    regime_models[model_name] = model_clone
                    regime_performance[model_name] = ExpertModelPerformance(
                        regime_id=regime_id,
                        model_name=model_name,
                        mae=mae,
                        mse=mse,
                        sharpe_ratio=sharpe_ratio,
                        hit_rate=hit_rate,
                        sample_count=len(y_regime),
                        confidence_score=hit_rate * (1 - mae)  # Combined metric
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name} for regime {regime_id}: {e}")
            
            # Select best model for this regime
            if regime_performance:
                best_model_name = max(regime_performance.keys(), 
                                    key=lambda k: regime_performance[k].confidence_score)
                
                self.expert_models[regime_id] = regime_models[best_model_name]
                self.model_performance[regime_id] = regime_performance[best_model_name]
                
                training_results[regime_id] = {
                    'best_model': best_model_name,
                    'performance': regime_performance[best_model_name],
                    'models_trained': list(regime_performance.keys())
                }
        
        result = {
            "success": True,
            "regimes_trained": list(training_results.keys()),
            "total_expert_models": len(self.expert_models),
            "training_details": training_results
        }
        
        self.logger.info(f"Trained expert models for {len(self.expert_models)} regimes")
        return result
    
    def predict_with_expert(
        self, 
        X: pd.DataFrame, 
        regime_predictions: List[RegimeClassification]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using regime-specific expert models"""
        
        predictions = np.zeros(len(X))
        confidences = np.zeros(len(X))
        
        for i, regime_class in enumerate(regime_predictions):
            regime_id = regime_class.regime_id
            
            if regime_id in self.expert_models:
                # Use regime-specific expert
                expert_model = self.expert_models[regime_id]
                pred = expert_model.predict(X.iloc[i:i+1])[0]
                
                # Confidence based on regime confidence and model performance
                model_performance = self.model_performance[regime_id]
                conf = regime_class.confidence * model_performance.confidence_score
                
                predictions[i] = pred
                confidences[i] = conf
            else:
                # Fallback to ensemble if no expert available
                predictions[i] = self._ensemble_predict(X.iloc[i:i+1])
                confidences[i] = 0.5  # Lower confidence for fallback
        
        return predictions, confidences
    
    def _ensemble_predict(self, X: pd.DataFrame) -> float:
        """Fallback ensemble prediction when no regime expert available"""
        
        if not self.expert_models:
            return 0.0
        
        # Use average of all available expert models
        predictions = []
        
        for expert_model in self.expert_models.values():
            try:
                pred = expert_model.predict(X)[0]
                predictions.append(pred)
            except:
                continue
        
        return np.mean(predictions) if predictions else 0.0

class RegimeRouterSystem:
    """Complete regime-routing system combining classification and expert models"""
    
    def __init__(self, n_regimes: int = 4):
        self.regime_classifier = MarketRegimeClassifier(n_regimes)
        self.expert_manager = ExpertModelManager()
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        
        # Performance tracking
        self.system_performance = {
            "regime_accuracy": 0.0,
            "prediction_improvement": 0.0,
            "stability_score": 0.0
        }
    
    def train_complete_system(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        feature_cols: List[str],
        price_col: str = 'price'
    ) -> Dict[str, Any]:
        """Train complete regime-routing system"""
        
        # Step 1: Extract regime features
        regime_features = self.regime_classifier.extract_regime_features(df, price_col)
        
        # Step 2: Fit regime classifier
        classifier_result = self.regime_classifier.fit(regime_features)
        
        if not classifier_result["success"]:
            return classifier_result
        
        # Step 3: Predict regimes for training data
        regime_classifications = self.regime_classifier.predict_regime(regime_features)
        regime_labels = [r.regime_id for r in regime_classifications]
        
        # Step 4: Prepare features and targets
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Align data (remove NaN rows)
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        regime_labels_clean = [regime_labels[i] for i in range(len(regime_labels)) if valid_indices.iloc[i]]
        
        # Step 5: Train expert models
        expert_result = self.expert_manager.train_expert_models(X_clean, y_clean, regime_labels_clean)
        
        if not expert_result["success"]:
            return expert_result
        
        # Step 6: Evaluate system performance
        system_evaluation = self._evaluate_system_performance(
            X_clean, y_clean, regime_features[valid_indices], regime_labels_clean
        )
        
        self.is_trained = True
        
        result = {
            "success": True,
            "regime_classifier": classifier_result,
            "expert_models": expert_result,
            "system_evaluation": system_evaluation,
            "training_samples": len(X_clean)
        }
        
        self.logger.info(f"Regime router system trained successfully with {len(X_clean)} samples")
        return result
    
    def predict_with_routing(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        price_col: str = 'price'
    ) -> Tuple[np.ndarray, np.ndarray, List[RegimeClassification]]:
        """Make predictions using regime routing"""
        
        if not self.is_trained:
            raise ValueError("Regime router system not trained yet")
        
        # Extract regime features
        regime_features = self.regime_classifier.extract_regime_features(df, price_col)
        
        # Classify regimes
        regime_classifications = self.regime_classifier.predict_regime(regime_features)
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        
        # Get expert predictions
        predictions, confidences = self.expert_manager.predict_with_expert(X, regime_classifications)
        
        return predictions, confidences, regime_classifications
    
    def _evaluate_system_performance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_features: pd.DataFrame,
        regime_labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate overall system performance"""
        
        # Baseline: simple linear model
        from sklearn.linear_model import LinearRegression
        baseline_model = LinearRegression()
        baseline_model.fit(X, y)
        baseline_pred = baseline_model.predict(X)
        baseline_mae = mean_absolute_error(y, baseline_pred)
        
        # Regime-routed predictions
        regime_classifications = self.regime_classifier.predict_regime(regime_features)
        routed_pred, routed_conf = self.expert_manager.predict_with_expert(X, regime_classifications)
        routed_mae = mean_absolute_error(y, routed_pred)
        
        # Calculate improvement
        improvement = (baseline_mae - routed_mae) / baseline_mae if baseline_mae > 0 else 0
        
        # Regime stability (how often regime changes)
        regime_changes = sum(1 for i in range(1, len(regime_labels)) 
                           if regime_labels[i] != regime_labels[i-1])
        stability_score = 1 - (regime_changes / len(regime_labels))
        
        evaluation = {
            "baseline_mae": baseline_mae,
            "routed_mae": routed_mae,
            "improvement_pct": improvement * 100,
            "stability_score": stability_score,
            "avg_confidence": np.mean(routed_conf),
            "regime_distribution": {str(i): regime_labels.count(i) for i in set(regime_labels)}
        }
        
        # Update system performance
        self.system_performance.update({
            "regime_accuracy": stability_score,
            "prediction_improvement": improvement,
            "stability_score": stability_score
        })
        
        return evaluation
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance"""
        
        return {
            "is_trained": self.is_trained,
            "n_regimes": self.regime_classifier.n_regimes,
            "expert_models_count": len(self.expert_manager.expert_models),
            "performance": self.system_performance,
            "regime_names": self.regime_classifier.regime_names
        }

def create_regime_router_system(n_regimes: int = 4) -> RegimeRouterSystem:
    """Create and return regime router system"""
    return RegimeRouterSystem(n_regimes)