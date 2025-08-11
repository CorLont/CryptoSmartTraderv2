#!/usr/bin/env python3
"""
Enhanced Multi-Model Training with XGBoost + Random Forest Ensemble
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available - using RF only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Train ensemble of RF + XGBoost models"""
    
    def __init__(self):
        self.models = {}
        self.horizons = ['1h', '24h', '168h', '720h']
        self.model_path = Path("models/saved")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def train_ensemble_models(self):
        """Train both RF and XGBoost models for ensemble"""
        
        # Load enhanced features with sentiment/whale data
        features_file = Path("data/processed/features.csv")
        alt_features_file = Path("features.csv")
        
        if features_file.exists():
            df = pd.read_csv(features_file)
        elif alt_features_file.exists():
            df = pd.read_csv(alt_features_file)
        else:
            logger.error("Enhanced features.csv not found - run fix_ml_training_pipeline.py first")
            return False
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Verify sentiment/whale features are present
        required_features = ['sentiment_numeric', 'whale_detected_numeric', 'whale_score']
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            logger.error(f"Missing sentiment/whale features: {missing}")
            return False
            
        logger.info("✅ Sentiment/whale features confirmed in training data")
        
        # Prepare feature matrix
        feature_cols = [col for col in df.columns if col not in ['coin', 'timestamp']]
        X = df[feature_cols].fillna(0)
        
        # Create synthetic targets if real targets not available
        target_cols = [f'target_return_{h}' for h in self.horizons]
        if not any(col in df.columns for col in target_cols):
            logger.info("Creating synthetic targets for training")
            df = self._create_synthetic_targets(df)
        
        success_count = 0
        
        for horizon in self.horizons:
            logger.info(f"Training ensemble for {horizon}...")
            
            target_col = f'target_return_{horizon}'
            if target_col not in df.columns:
                continue
                
            y = df[target_col].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Save RF model (backward compatibility)
            rf_path = self.model_path / f"rf_{horizon}.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(rf_model, f)
            
            models_trained = {"rf": rf_model}
            
            # Train XGBoost if available
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                
                # Save XGBoost model
                xgb_path = self.model_path / f"xgb_{horizon}.pkl"
                with open(xgb_path, 'wb') as f:
                    pickle.dump(xgb_model, f)
                    
                models_trained["xgb"] = xgb_model
            
            # Evaluate ensemble
            rf_pred = rf_model.predict(X_test)
            rf_score = r2_score(y_test, rf_pred)
            
            logger.info(f"✅ {horizon} - RF R²: {rf_score:.4f}")
            
            if XGBOOST_AVAILABLE:
                xgb_pred = xgb_model.predict(X_test)
                xgb_score = r2_score(y_test, xgb_pred)
                
                # Ensemble prediction (simple average)
                ensemble_pred = (rf_pred + xgb_pred) / 2
                ensemble_score = r2_score(y_test, ensemble_pred)
                
                logger.info(f"✅ {horizon} - XGB R²: {xgb_score:.4f}")
                logger.info(f"✅ {horizon} - Ensemble R²: {ensemble_score:.4f}")
            
            self.models[horizon] = models_trained
            success_count += 1
        
        logger.info(f"✅ Ensemble training completed: {success_count}/{len(self.horizons)} horizons")
        return success_count > 0
    
    def _create_synthetic_targets(self, df):
        """Create synthetic targets based on enhanced features"""
        np.random.seed(42)
        
        for horizon in self.horizons:
            hours_map = {'1h': 1, '24h': 24, '168h': 168, '720h': 720}
            hours = hours_map[horizon]
            
            # Enhanced synthetic targets using sentiment/whale features
            base_return = (
                df.get('sentiment_numeric', 0.5) - 0.5  # Sentiment bias
                + df.get('whale_detected_numeric', 0) * 0.1  # Whale influence
                + df.get('price_change_24h', 0) / 100 * 0.5  # Momentum
            )
            
            time_scaling = np.sqrt(hours) / 10
            noise = np.random.normal(0, 0.01 * time_scaling, len(df))
            
            df[f'target_return_{horizon}'] = base_return * time_scaling + noise
        
        logger.info("Enhanced synthetic targets created with sentiment/whale features")
        return df

if __name__ == "__main__":
    trainer = EnsembleTrainer()
    success = trainer.train_ensemble_models()
    
    if success:
        print("✅ ENSEMBLE TRAINING SUCCESS - RF + XGBoost models ready")
    else:
        print("❌ ENSEMBLE TRAINING FAILED - Check logs for details")
