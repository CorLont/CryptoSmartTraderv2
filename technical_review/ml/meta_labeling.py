# ml/meta_labeling.py - Lopez de Prado Triple-Barrier Method
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """Implementation of Lopez de Prado's Triple-Barrier Method for signal quality"""
    
    def __init__(self, pt_sl_ratio=2.0, min_ret=0.01, max_hold_days=7):
        """
        Args:
            pt_sl_ratio: Profit target / Stop loss ratio (2.0 = 2:1 ratio)
            min_ret: Minimum return threshold for positive labels
            max_hold_days: Maximum holding period in days
        """
        self.pt_sl_ratio = pt_sl_ratio
        self.min_ret = min_ret
        self.max_hold_days = max_hold_days
    
    def compute_daily_volatility(self, close: pd.Series, window: int = 20) -> pd.Series:
        """Compute daily volatility using close-to-close returns"""
        returns = close.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def get_vertical_barriers(self, timestamps: pd.Series) -> pd.Series:
        """Get vertical barriers (time-based exits)"""
        return timestamps + pd.Timedelta(days=self.max_hold_days)
    
    def get_horizontal_barriers(self, prices: pd.Series, volatility: pd.Series) -> tuple:
        """Get horizontal barriers (profit target and stop loss)"""
        # Dynamic barriers based on volatility
        pt = volatility * self.min_ret * self.pt_sl_ratio  # Profit target
        sl = -volatility * self.min_ret  # Stop loss (negative)
        
        return pt, sl
    
    def apply_triple_barrier(self, price_data: pd.DataFrame, signal_timestamps: pd.Series) -> pd.DataFrame:
        """
        Apply triple-barrier method to generate labels
        
        Args:
            price_data: DataFrame with 'timestamp', 'close' columns
            signal_timestamps: Series of timestamps when signals were generated
            
        Returns:
            DataFrame with labels and barrier hit information
        """
        results = []
        
        # Compute volatility
        volatility = self.compute_daily_volatility(price_data['close'])
        
        for signal_time in signal_timestamps:
            try:
                # Find signal price
                signal_idx = price_data[price_data['timestamp'] >= signal_time].index[0]
                signal_price = price_data.loc[signal_idx, 'close']
                signal_vol = volatility.iloc[signal_idx]
                
                # Define barriers
                vertical_barrier = signal_time + pd.Timedelta(days=self.max_hold_days)
                pt_threshold, sl_threshold = self.get_horizontal_barriers(
                    price_data['close'], pd.Series([signal_vol])
                )
                
                pt_price = signal_price * (1 + pt_threshold.iloc[0])
                sl_price = signal_price * (1 + sl_threshold.iloc[0])
                
                # Find future prices after signal
                future_data = price_data[
                    (price_data['timestamp'] > signal_time) & 
                    (price_data['timestamp'] <= vertical_barrier)
                ]
                
                if future_data.empty:
                    continue
                
                # Track which barrier is hit first
                hit_pt = future_data['close'] >= pt_price
                hit_sl = future_data['close'] <= sl_price
                
                label = 0  # Default: no clear outcome
                barrier_hit = 'vertical'  # Default: time exit
                exit_time = vertical_barrier
                exit_price = future_data['close'].iloc[-1] if not future_data.empty else signal_price
                
                # Check profit target
                if hit_pt.any():
                    pt_idx = hit_pt.idxmax() if hit_pt.any() else None
                    if pt_idx is not None:
                        label = 1  # Positive outcome
                        barrier_hit = 'profit_target'
                        exit_time = future_data.loc[pt_idx, 'timestamp']
                        exit_price = future_data.loc[pt_idx, 'close']
                
                # Check stop loss (only if PT not hit first)
                if label == 0 and hit_sl.any():
                    sl_idx = hit_sl.idxmax() if hit_sl.any() else None
                    if sl_idx is not None:
                        # Check if SL hit before PT
                        pt_time = future_data.loc[hit_pt.idxmax(), 'timestamp'] if hit_pt.any() else pd.Timestamp.max
                        sl_time = future_data.loc[sl_idx, 'timestamp']
                        
                        if sl_time < pt_time:
                            label = -1  # Negative outcome
                            barrier_hit = 'stop_loss'
                            exit_time = sl_time
                            exit_price = future_data.loc[sl_idx, 'close']
                
                # Calculate actual return
                actual_return = (exit_price - signal_price) / signal_price
                
                results.append({
                    'signal_time': signal_time,
                    'signal_price': signal_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'actual_return': actual_return,
                    'label': label,
                    'barrier_hit': barrier_hit,
                    'volatility': signal_vol,
                    'pt_threshold': pt_threshold.iloc[0],
                    'sl_threshold': sl_threshold.iloc[0]
                })
                
            except Exception as e:
                logger.error(f"Triple barrier failed for signal {signal_time}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def compute_meta_labels(self, predictions: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute meta-labels for prediction quality filtering
        
        Args:
            predictions: DataFrame with 'timestamp', 'prediction', 'confidence' columns
            price_data: Historical price data for validation
            
        Returns:
            DataFrame with meta-labels indicating prediction quality
        """
        # Apply triple-barrier method
        barrier_results = self.apply_triple_barrier(
            price_data, 
            predictions['timestamp']
        )
        
        # Merge with original predictions
        merged = predictions.merge(
            barrier_results, 
            left_on='timestamp', 
            right_on='signal_time',
            how='left'
        )
        
        # Compute meta-label features
        merged['signal_quality'] = merged.apply(self._compute_signal_quality, axis=1)
        merged['should_trade'] = merged['signal_quality'] > 0.5
        
        # Add additional quality metrics
        merged['hit_rate'] = self._compute_rolling_hit_rate(merged)
        merged['avg_return'] = self._compute_rolling_avg_return(merged)
        merged['sharpe_ratio'] = self._compute_rolling_sharpe(merged)
        
        return merged
    
    def _compute_signal_quality(self, row) -> float:
        """Compute signal quality score (0-1)"""
        base_score = 0.5
        
        # Adjust based on outcome
        if pd.notna(row.get('label')):
            if row['label'] == 1:  # Profit target hit
                base_score += 0.3
            elif row['label'] == -1:  # Stop loss hit
                base_score -= 0.3
        
        # Adjust based on confidence
        if pd.notna(row.get('confidence')):
            confidence_bonus = (row['confidence'] - 0.5) * 0.2
            base_score += confidence_bonus
        
        # Adjust based on actual return vs prediction
        if pd.notna(row.get('actual_return')) and pd.notna(row.get('prediction')):
            prediction_accuracy = 1 - abs(row['actual_return'] - row['prediction'])
            base_score += prediction_accuracy * 0.2
        
        return np.clip(base_score, 0, 1)
    
    def _compute_rolling_hit_rate(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling hit rate (successful trades / total trades)"""
        if 'label' not in df.columns:
            return pd.Series([0.5] * len(df))
        
        successful_trades = (df['label'] == 1).astype(int)
        return successful_trades.rolling(window=window, min_periods=1).mean()
    
    def _compute_rolling_avg_return(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling average return"""
        if 'actual_return' not in df.columns:
            return pd.Series([0.0] * len(df))
        
        return df['actual_return'].rolling(window=window, min_periods=1).mean()
    
    def _compute_rolling_sharpe(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling Sharpe ratio"""
        if 'actual_return' not in df.columns:
            return pd.Series([0.0] * len(df))
        
        rolling_mean = df['actual_return'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['actual_return'].rolling(window=window, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.fillna(0.01)  
        rolling_std = rolling_std.replace(0, 0.01)
        
        return rolling_mean / rolling_std * np.sqrt(252)  # Annualized

def apply_meta_labeling_filter(predictions: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Apply meta-labeling filter to predictions
    
    Args:
        predictions: Model predictions with timestamps
        price_history: Historical price data for validation
        
    Returns:
        Filtered predictions with quality scores
    """
    labeler = TripleBarrierLabeler()
    
    # Apply meta-labeling
    meta_labeled = labeler.compute_meta_labels(predictions, price_history)
    
    # Filter based on quality thresholds
    quality_threshold = 0.6
    hit_rate_threshold = 0.55
    
    filtered = meta_labeled[
        (meta_labeled['signal_quality'] >= quality_threshold) &
        (meta_labeled['hit_rate'] >= hit_rate_threshold)
    ]
    
    logger.info(f"Meta-labeling filter: {len(filtered)}/{len(predictions)} predictions passed quality gates")
    
    return filtered