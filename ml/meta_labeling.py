#!/usr/bin/env python3
"""
Meta-Labeling with Triple-Barrier Method (Lopez de Prado)
Filters false signals and creates quality labels instead of directional labels
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TripleBarrierLabeling:
    """
    Triple-barrier method for creating meta-labels
    """
    
    def __init__(self, 
                 profit_target: float = 0.02,  # 2% profit target
                 stop_loss: float = 0.01,      # 1% stop loss
                 max_holding_period: int = 24): # 24 hours max hold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        
    def create_barriers(self, 
                       df: pd.DataFrame, 
                       price_col: str = 'close') -> pd.DataFrame:
        """Create triple barriers for each signal"""
        
        results = []
        
        for idx, row in df.iterrows():
            entry_price = row[price_col]
            entry_time = row['timestamp']
            
            # Define barriers
            upper_barrier = entry_price * (1 + self.profit_target)
            lower_barrier = entry_price * (1 - self.stop_loss)
            
            # Get future prices within holding period
            future_mask = (df['timestamp'] > entry_time) & \
                         (df['timestamp'] <= entry_time + timedelta(hours=self.max_holding_period))
            
            future_prices = df[future_mask][price_col]
            future_times = df[future_mask]['timestamp']
            
            if len(future_prices) == 0:
                # No future data available
                label = 0  # Neutral/uncertain
                exit_reason = 'no_data'
                exit_time = entry_time
                exit_price = entry_price
            else:
                # Check which barrier is hit first
                profit_hits = future_prices >= upper_barrier
                loss_hits = future_prices <= lower_barrier
                
                profit_hit_times = future_times[profit_hits]
                loss_hit_times = future_times[loss_hits]
                
                first_profit = profit_hit_times.min() if len(profit_hit_times) > 0 else pd.Timestamp.max
                first_loss = loss_hit_times.min() if len(loss_hit_times) > 0 else pd.Timestamp.max
                max_time = entry_time + timedelta(hours=self.max_holding_period)
                
                # Determine which barrier was hit first
                if first_profit < first_loss and first_profit < max_time:
                    label = 1  # Profit target hit
                    exit_reason = 'profit'
                    exit_time = first_profit
                    exit_price = future_prices[future_times == first_profit].iloc[0]
                elif first_loss < first_profit and first_loss < max_time:
                    label = -1  # Stop loss hit
                    exit_reason = 'stop_loss'
                    exit_time = first_loss
                    exit_price = future_prices[future_times == first_loss].iloc[0]
                else:
                    # Time barrier hit
                    label = 0  # Neutral
                    exit_reason = 'time_limit'
                    exit_time = max_time
                    exit_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
            
            results.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'label': label,
                'exit_reason': exit_reason,
                'return': (exit_price - entry_price) / entry_price,
                'holding_hours': (exit_time - entry_time).total_seconds() / 3600
            })
        
        return pd.DataFrame(results)

class MetaClassifier(nn.Module):
    """
    Meta-classifier to predict probability of hitting profit/stop targets
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3),  # 3 classes: profit, loss, neutral
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class MetaLabelingSystem:
    """
    Complete meta-labeling system combining barriers and classifier
    """
    
    def __init__(self):
        self.barrier_labeler = None
        self.meta_classifier = None
        self.feature_scaler = None
        self.is_trained = False
        
    def create_meta_labels(self, 
                          df: pd.DataFrame,
                          primary_signals: pd.Series,
                          profit_target: float = 0.02,
                          stop_loss: float = 0.01) -> pd.DataFrame:
        """Create meta-labels for primary model signals"""
        
        # Filter data to only signals
        signal_data = df[primary_signals == 1].copy()
        
        if len(signal_data) == 0:
            return pd.DataFrame()
        
        # Create triple barriers
        self.barrier_labeler = TripleBarrierLabeling(
            profit_target=profit_target,
            stop_loss=stop_loss
        )
        
        barriers = self.barrier_labeler.create_barriers(signal_data)
        
        # Add original features
        signal_data = signal_data.reset_index(drop=True)
        barriers = barriers.reset_index(drop=True)
        
        meta_data = pd.concat([signal_data, barriers], axis=1)
        
        return meta_data
    
    def prepare_meta_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for meta-classifier"""
        
        feature_cols = [
            'volume_momentum', 'whale_activity_score', 'sentiment_score',
            'technical_rsi', 'volatility_24h', 'market_cap_rank',
            'price_momentum_1h', 'price_momentum_24h'
        ]
        
        # Use available features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) == 0:
            # Create basic features if none available
            df['price_momentum_1h'] = df['close'].pct_change(1)
            df['volatility_24h'] = df['close'].rolling(24).std()
            available_cols = ['price_momentum_1h', 'volatility_24h']
        
        features = df[available_cols].fillna(0).values
        
        return features
    
    def train_meta_classifier(self, 
                             training_data: pd.DataFrame,
                             epochs: int = 100) -> Dict[str, float]:
        """Train the meta-classifier"""
        
        # Prepare features and labels
        features = self.prepare_meta_features(training_data)
        labels = training_data['label'].values
        
        # Convert labels to classification format
        # -1 (loss) -> 0, 0 (neutral) -> 1, 1 (profit) -> 2
        labels_mapped = labels + 1
        
        # Initialize model
        input_size = features.shape[1]
        self.meta_classifier = MetaClassifier(input_size)
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels_mapped)
        
        # Training setup
        optimizer = torch.optim.Adam(self.meta_classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.meta_classifier.train()
        train_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.meta_classifier(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        
        # Calculate training metrics
        self.meta_classifier.eval()
        with torch.no_grad():
            outputs = self.meta_classifier(X)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y).float().mean().item()
        
        return {
            'final_loss': train_losses[-1],
            'training_accuracy': accuracy,
            'epochs_trained': epochs
        }
    
    def predict_signal_quality(self, 
                              df: pd.DataFrame,
                              confidence_threshold: float = 0.7) -> pd.DataFrame:
        """Predict signal quality using meta-classifier"""
        
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained first")
        
        features = self.prepare_meta_features(df)
        X = torch.FloatTensor(features)
        
        self.meta_classifier.eval()
        with torch.no_grad():
            probabilities = self.meta_classifier(X)
            
            # Get probabilities for each class
            loss_prob = probabilities[:, 0].numpy()      # P(stop loss)
            neutral_prob = probabilities[:, 1].numpy()   # P(neutral)
            profit_prob = probabilities[:, 2].numpy()    # P(profit target)
            
            # Calculate signal quality score
            quality_score = profit_prob - loss_prob  # Higher is better
            
            # Apply confidence threshold
            max_prob = torch.max(probabilities, dim=1)[0].numpy()
            high_confidence = max_prob >= confidence_threshold
        
        results = pd.DataFrame({
            'loss_probability': loss_prob,
            'neutral_probability': neutral_prob,
            'profit_probability': profit_prob,
            'quality_score': quality_score,
            'max_confidence': max_prob,
            'high_confidence': high_confidence
        })
        
        return results

def create_meta_labeling_pipeline(df: pd.DataFrame,
                                 primary_signals: pd.Series,
                                 train_ratio: float = 0.8) -> Tuple[MetaLabelingSystem, Dict[str, float]]:
    """Create complete meta-labeling pipeline"""
    
    # Initialize system
    meta_system = MetaLabelingSystem()
    
    # Create meta-labels
    meta_data = meta_system.create_meta_labels(df, primary_signals)
    
    if len(meta_data) == 0:
        return meta_system, {'error': 'No training data available'}
    
    # Split training/validation
    train_size = int(len(meta_data) * train_ratio)
    train_data = meta_data[:train_size]
    val_data = meta_data[train_size:]
    
    # Train meta-classifier
    train_metrics = meta_system.train_meta_classifier(train_data)
    
    # Validate on test set
    if len(val_data) > 0:
        val_predictions = meta_system.predict_signal_quality(val_data)
        
        # Calculate validation metrics
        val_quality_scores = val_predictions['quality_score'].values
        val_actual_returns = val_data['return'].values
        
        correlation = np.corrcoef(val_quality_scores, val_actual_returns)[0, 1]
        train_metrics['validation_correlation'] = correlation
    
    return meta_system, train_metrics

if __name__ == "__main__":
    print("🎯 TESTING META-LABELING SYSTEM")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='1H'),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume_momentum': np.random.randn(n_samples),
        'whale_activity_score': np.random.uniform(0, 1, n_samples),
        'sentiment_score': np.random.uniform(-1, 1, n_samples),
        'technical_rsi': np.random.uniform(0, 100, n_samples)
    })
    
    # Create random primary signals
    primary_signals = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]))
    
    # Test meta-labeling pipeline
    meta_system, metrics = create_meta_labeling_pipeline(sample_data, primary_signals)
    
    print("Meta-labeling Pipeline Results:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Test signal quality prediction
    if meta_system.is_trained:
        quality_predictions = meta_system.predict_signal_quality(sample_data[:100])
        
        high_quality_signals = quality_predictions[quality_predictions['high_confidence']].shape[0]
        print(f"   High quality signals identified: {high_quality_signals}/100")
    
    print("✅ Meta-labeling system testing completed")