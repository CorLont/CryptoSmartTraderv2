#!/usr/bin/env python3
"""
Daily Evaluator - Comprehensive metrics & attribution analysis

Dagelijkse batch analysis van trading performance:
- Hit rate, precision@K, average PnL, Information Ratio, Sharpe
- Calibration analysis (Brier score, NLL voor probabilistic scores)
- Regret analysis per strategy arm (voor bandit algorithms)
- Signal attribution & performance decomposition
- Risk-adjusted returns & drawdown analysis
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score

from ..trading.recommendation_ledger import RecommendationLedger, get_recommendation_ledger

logger = logging.getLogger(__name__)

class DailyEvaluator:
    """Comprehensive daily evaluation van trading performance"""
    
    def __init__(self, 
                 output_dir: str = "data/evaluation",
                 lookback_days: int = 30,
                 min_samples: int = 10):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        
        self.ledger = get_recommendation_ledger()
        
        logger.info(f"DailyEvaluator initialized with {lookback_days}d lookback")
    
    def run_daily_evaluation(self) -> Dict:
        """Run complete daily evaluation analysis"""
        
        logger.info(f"📊 Starting daily evaluation for last {self.lookback_days} days")
        
        # Load completed recommendations with labels
        completed_recs = self._load_completed_recommendations()
        
        if len(completed_recs) < self.min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(completed_recs),
                "min_required": self.min_samples
            }
        
        logger.info(f"Analyzing {len(completed_recs)} completed recommendations")
        
        # Run alle evaluation components
        evaluation_results = {}
        
        try:
            # 1. Basic Performance Metrics
            evaluation_results['basic_metrics'] = self._calculate_basic_metrics(completed_recs)
            
            # 2. Hit Rate & Precision Analysis
            evaluation_results['precision_metrics'] = self._calculate_precision_metrics(completed_recs)
            
            # 3. Risk-Adjusted Returns
            evaluation_results['risk_metrics'] = self._calculate_risk_adjusted_metrics(completed_recs)
            
            # 4. Calibration Analysis
            evaluation_results['calibration'] = self._calculate_calibration_metrics(completed_recs)
            
            # 5. Signal Attribution
            evaluation_results['signal_attribution'] = self._analyze_signal_attribution(completed_recs)
            
            # 6. Regret Analysis
            evaluation_results['regret_analysis'] = self._calculate_regret_analysis(completed_recs)
            
            # 7. Temporal Performance
            evaluation_results['temporal_analysis'] = self._analyze_temporal_performance(completed_recs)
            
            # 8. Execution Quality
            evaluation_results['execution_quality'] = self._analyze_execution_quality(completed_recs)
            
        except Exception as e:
            logger.error(f"Error in evaluation analysis: {e}")
            evaluation_results['error'] = str(e)
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.output_dir / f"daily_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate summary report
        summary = self._generate_evaluation_summary(evaluation_results)
        
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Daily evaluation completed: {results_file.name}")
        
        return {
            "status": "success",
            "samples_analyzed": len(completed_recs),
            "files_created": [results_file.name, summary_file.name],
            "evaluation_results": evaluation_results,
            "summary": summary
        }
    
    def _load_completed_recommendations(self) -> pd.DataFrame:
        """Load completed recommendations with outcome data"""
        
        # Get recommendation history
        history_df = self.ledger.get_recommendation_history(days_back=self.lookback_days)
        
        if history_df.empty:
            return pd.DataFrame()
        
        # Filter to completed recommendations only
        completed = history_df[history_df['ts_exit'].notna()].copy()
        
        # Calculate performance metrics
        if not completed.empty:
            completed['realized_pnl_bps'] = completed.get('realized_pnl_bps', 0)
            completed['label_profitable'] = (completed['realized_pnl_bps'] > 0).astype(int)
            completed['holding_hours'] = completed.get('holding_period_hours', 0)
            
            # Add combined score for analysis
            completed['combined_score'] = completed.get('combined_score', 0.5)
            completed['confidence'] = completed.get('confidence', 0.5)
        
        logger.info(f"Loaded {len(completed)} completed recommendations")
        return completed
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics"""
        
        if df.empty:
            return {}
        
        metrics = {
            'total_trades': len(df),
            'profitable_trades': (df['realized_pnl_bps'] > 0).sum(),
            'hit_rate': (df['realized_pnl_bps'] > 0).mean(),
            'avg_pnl_bps': df['realized_pnl_bps'].mean(),
            'median_pnl_bps': df['realized_pnl_bps'].median(),
            'total_pnl_bps': df['realized_pnl_bps'].sum(),
            'best_trade_bps': df['realized_pnl_bps'].max(),
            'worst_trade_bps': df['realized_pnl_bps'].min(),
            'pnl_std_bps': df['realized_pnl_bps'].std(),
            'avg_holding_hours': df['holding_hours'].mean(),
            'median_holding_hours': df['holding_hours'].median()
        }
        
        # Win/Loss analysis
        wins = df[df['realized_pnl_bps'] > 0]['realized_pnl_bps']
        losses = df[df['realized_pnl_bps'] < 0]['realized_pnl_bps']
        
        if len(wins) > 0:
            metrics['avg_win_bps'] = wins.mean()
            metrics['largest_win_bps'] = wins.max()
        
        if len(losses) > 0:
            metrics['avg_loss_bps'] = losses.mean()
            metrics['largest_loss_bps'] = losses.min()
        
        # Win/Loss ratio
        if len(losses) > 0 and metrics.get('avg_loss_bps'):
            metrics['win_loss_ratio'] = abs(metrics.get('avg_win_bps', 0) / metrics['avg_loss_bps'])
        
        return metrics
    
    def _calculate_precision_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate precision@K and related metrics"""
        
        if df.empty:
            return {}
        
        # Sort by combined score (descending)
        df_sorted = df.sort_values('combined_score', ascending=False).copy()
        
        precision_metrics = {}
        
        # Calculate precision@K for different K values
        k_values = [5, 10, 20, 50]
        
        for k in k_values:
            if len(df_sorted) >= k:
                top_k = df_sorted.head(k)
                precision_k = (top_k['realized_pnl_bps'] > 0).mean()
                precision_metrics[f'precision_at_{k}'] = precision_k
                
                # Average return van top K
                precision_metrics[f'avg_return_top_{k}_bps'] = top_k['realized_pnl_bps'].mean()
        
        # Precision and Recall voor binary classification
        if len(df) > 0:
            # Use combined_score > 0.6 as positive prediction threshold
            y_pred = (df['combined_score'] > 0.6).astype(int)
            y_true = df['label_profitable']
            
            if len(np.unique(y_true)) > 1:  # Need both positive and negative samples
                precision_metrics['binary_precision'] = precision_score(y_true, y_pred, zero_division=0)
                precision_metrics['binary_recall'] = recall_score(y_true, y_pred, zero_division=0)
                
                # F1 Score
                prec = precision_metrics['binary_precision']
                recall = precision_metrics['binary_recall']
                if prec + recall > 0:
                    precision_metrics['f1_score'] = 2 * (prec * recall) / (prec + recall)
        
        return precision_metrics
    
    def _calculate_risk_adjusted_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        
        if df.empty or len(df) < 3:
            return {}
        
        returns_bps = df['realized_pnl_bps']
        
        risk_metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns_bps),
            'information_ratio': self._calculate_information_ratio(returns_bps),
            'sortino_ratio': self._calculate_sortino_ratio(returns_bps),
            'max_drawdown_bps': self._calculate_max_drawdown(returns_bps),
            'calmar_ratio': self._calculate_calmar_ratio(returns_bps),
            'volatility_bps': returns_bps.std()
        }
        
        # Risk-adjusted return
        avg_return = returns_bps.mean()
        volatility = returns_bps.std()
        
        if volatility > 0:
            risk_metrics['return_volatility_ratio'] = avg_return / volatility
        
        return risk_metrics
    
    def _calculate_calibration_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate calibration metrics voor probabilistic predictions"""
        
        if df.empty:
            return {}
        
        # Use confidence scores as probabilities
        y_prob = df['confidence'].values
        y_true = df['label_profitable'].values
        
        calibration_metrics = {}
        
        try:
            # Brier Score (lower is better)
            brier_score = brier_score_loss(y_true, y_prob)
            calibration_metrics['brier_score'] = brier_score
            
            # Decompose Brier score
            reliability, resolution, uncertainty = self._decompose_brier_score(y_true, y_prob)
            calibration_metrics['brier_reliability'] = reliability
            calibration_metrics['brier_resolution'] = resolution
            calibration_metrics['brier_uncertainty'] = uncertainty
            
            # Log Loss / Negative Log-Likelihood
            # Clip probabilities to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            log_loss_score = log_loss(y_true, y_prob_clipped)
            calibration_metrics['log_loss'] = log_loss_score
            
            # Calibration slope (perfect calibration has slope = 1)
            if len(np.unique(y_prob)) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_prob, y_true)
                calibration_metrics['calibration_slope'] = slope
                calibration_metrics['calibration_intercept'] = intercept
                calibration_metrics['calibration_r_squared'] = r_value ** 2
            
        except Exception as e:
            logger.warning(f"Error calculating calibration metrics: {e}")
            calibration_metrics['error'] = str(e)
        
        return calibration_metrics
    
    def _analyze_signal_attribution(self, df: pd.DataFrame) -> Dict:
        """Analyze performance attribution per signal bucket"""
        
        if df.empty:
            return {}
        
        attribution = {}
        
        # Signal columns to analyze
        signal_columns = [
            'momentum_score', 'mean_revert_score', 'funding_score',
            'sentiment_score', 'whale_score', 'technical_score'
        ]
        
        for signal in signal_columns:
            if signal in df.columns:
                # Correlation with realized returns
                if df[signal].std() > 0:
                    correlation = df[signal].corr(df['realized_pnl_bps'])
                    attribution[f'{signal}_correlation'] = correlation
                    
                    # Top vs bottom quintile performance
                    quintiles = pd.qcut(df[signal], q=5, labels=False, duplicates='drop')
                    
                    if len(np.unique(quintiles)) > 1:
                        top_quintile_return = df[quintiles == 4]['realized_pnl_bps'].mean()
                        bottom_quintile_return = df[quintiles == 0]['realized_pnl_bps'].mean()
                        
                        attribution[f'{signal}_top_quintile_bps'] = top_quintile_return
                        attribution[f'{signal}_bottom_quintile_bps'] = bottom_quintile_return
                        attribution[f'{signal}_spread_bps'] = top_quintile_return - bottom_quintile_return
        
        # Combined score analysis
        if 'combined_score' in df.columns and df['combined_score'].std() > 0:
            # Correlation met realized returns
            combined_corr = df['combined_score'].corr(df['realized_pnl_bps'])
            attribution['combined_score_correlation'] = combined_corr
            
            # Decile analysis
            deciles = pd.qcut(df['combined_score'], q=10, labels=False, duplicates='drop')
            
            if len(np.unique(deciles)) > 1:
                decile_returns = []
                for i in range(10):
                    decile_mask = (deciles == i)
                    if decile_mask.sum() > 0:
                        decile_return = df[decile_mask]['realized_pnl_bps'].mean()
                        decile_returns.append(decile_return)
                
                attribution['combined_score_decile_returns'] = decile_returns
                
                if len(decile_returns) >= 2:
                    attribution['top_bottom_decile_spread_bps'] = decile_returns[-1] - decile_returns[0]
        
        return attribution
    
    def _calculate_regret_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate regret analysis voor bandit-style evaluation"""
        
        if df.empty:
            return {}
        
        regret_analysis = {}
        
        # Simple regret = difference tussen best possible and actual performance
        best_possible_return = df['realized_pnl_bps'].max()
        actual_avg_return = df['realized_pnl_bps'].mean()
        
        regret_analysis['simple_regret_bps'] = best_possible_return - actual_avg_return
        
        # Cumulative regret over time
        df_sorted = df.sort_values('ts_signal').copy()
        df_sorted['cumulative_return'] = df_sorted['realized_pnl_bps'].cumsum()
        
        # Best possible cumulative return (always picking best trade)
        best_cumulative = df_sorted['realized_pnl_bps'].max() * len(df_sorted)
        actual_cumulative = df_sorted['cumulative_return'].iloc[-1]
        
        regret_analysis['cumulative_regret_bps'] = best_cumulative - actual_cumulative
        
        # Strategy arm regret (by symbol/side)
        if 'symbol' in df.columns:
            symbol_performance = df.groupby('symbol')['realized_pnl_bps'].agg(['mean', 'count'])
            symbol_performance = symbol_performance[symbol_performance['count'] >= 3]  # Min 3 trades
            
            if not symbol_performance.empty:
                best_symbol_return = symbol_performance['mean'].max()
                
                symbol_regrets = {}
                for symbol in symbol_performance.index:
                    symbol_return = symbol_performance.loc[symbol, 'mean']
                    symbol_regrets[symbol] = best_symbol_return - symbol_return
                
                regret_analysis['symbol_regrets'] = symbol_regrets
        
        return regret_analysis
    
    def _analyze_temporal_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance over time"""
        
        if df.empty:
            return {}
        
        df['ts_signal'] = pd.to_datetime(df['ts_signal'])
        df_sorted = df.sort_values('ts_signal').copy()
        
        temporal_analysis = {}
        
        # Rolling performance metrics
        window_sizes = [7, 14, 30]  # days
        
        for window in window_sizes:
            if len(df_sorted) >= window:
                rolling_returns = df_sorted['realized_pnl_bps'].rolling(window=window).mean()
                rolling_hit_rate = (df_sorted['realized_pnl_bps'] > 0).rolling(window=window).mean()
                
                temporal_analysis[f'rolling_{window}d_avg_return'] = rolling_returns.iloc[-1]
                temporal_analysis[f'rolling_{window}d_hit_rate'] = rolling_hit_rate.iloc[-1]
        
        # Daily performance breakdown
        df_sorted['date'] = df_sorted['ts_signal'].dt.date
        daily_stats = df_sorted.groupby('date')['realized_pnl_bps'].agg(['mean', 'count', 'sum'])
        
        if not daily_stats.empty:
            temporal_analysis['avg_daily_return_bps'] = daily_stats['mean'].mean()
            temporal_analysis['best_day_bps'] = daily_stats['sum'].max()
            temporal_analysis['worst_day_bps'] = daily_stats['sum'].min()
            temporal_analysis['avg_trades_per_day'] = daily_stats['count'].mean()
        
        return temporal_analysis
    
    def _analyze_execution_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze execution quality metrics"""
        
        execution_analysis = {}
        
        # Slippage analysis
        if 'slippage_bps' in df.columns:
            slippage_data = df[df['slippage_bps'].notna()]
            
            if not slippage_data.empty:
                execution_analysis['avg_slippage_bps'] = slippage_data['slippage_bps'].mean()
                execution_analysis['median_slippage_bps'] = slippage_data['slippage_bps'].median()
                execution_analysis['max_slippage_bps'] = slippage_data['slippage_bps'].max()
                execution_analysis['slippage_std_bps'] = slippage_data['slippage_bps'].std()
        
        # Time to execution
        if 'ts_entry' in df.columns and 'ts_signal' in df.columns:
            df['time_to_execution'] = (
                pd.to_datetime(df['ts_entry']) - pd.to_datetime(df['ts_signal'])
            ).dt.total_seconds() / 60  # minutes
            
            execution_data = df[df['time_to_execution'].notna()]
            
            if not execution_data.empty:
                execution_analysis['avg_time_to_execution_min'] = execution_data['time_to_execution'].mean()
                execution_analysis['median_time_to_execution_min'] = execution_data['time_to_execution'].median()
        
        return execution_analysis
    
    # Helper methods voor risk metrics
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std()
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_return: float = 0.0) -> float:
        """Calculate Information Ratio"""
        active_returns = returns - benchmark_return
        if active_returns.std() == 0:
            return 0.0
        return active_returns.mean() / active_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (using downside deviation)"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        max_dd = abs(self._calculate_max_drawdown(returns))
        if max_dd == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        return returns.mean() / max_dd
    
    def _decompose_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, float]:
        """Decompose Brier score into reliability, resolution, and uncertainty"""
        
        # Bin probabilities
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0
        resolution = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Average predicted probability in this bin
                avg_predicted_prob = y_prob[in_bin].mean()
                
                # Average true outcome in this bin
                avg_true_outcome = y_true[in_bin].mean()
                
                # Contribution to reliability (calibration error)
                reliability += prop_in_bin * (avg_predicted_prob - avg_true_outcome) ** 2
                
                # Contribution to resolution (discrimination)
                base_rate = y_true.mean()
                resolution += prop_in_bin * (avg_true_outcome - base_rate) ** 2
        
        # Uncertainty (inherent in the outcomes)
        base_rate = y_true.mean()
        uncertainty = base_rate * (1 - base_rate)
        
        return reliability, resolution, uncertainty
    
    def _generate_evaluation_summary(self, evaluation_results: Dict) -> Dict:
        """Generate concise summary van evaluation results"""
        
        summary = {
            'evaluation_timestamp': datetime.now(timezone.utc),
            'lookback_days': self.lookback_days
        }
        
        # Extract key metrics
        if 'basic_metrics' in evaluation_results:
            basic = evaluation_results['basic_metrics']
            summary['key_performance'] = {
                'total_trades': basic.get('total_trades', 0),
                'hit_rate': basic.get('hit_rate', 0),
                'avg_pnl_bps': basic.get('avg_pnl_bps', 0),
                'total_pnl_bps': basic.get('total_pnl_bps', 0),
                'sharpe_ratio': evaluation_results.get('risk_metrics', {}).get('sharpe_ratio', 0)
            }
        
        # Risk assessment
        if 'risk_metrics' in evaluation_results:
            risk = evaluation_results['risk_metrics']
            summary['risk_assessment'] = {
                'max_drawdown_bps': risk.get('max_drawdown_bps', 0),
                'volatility_bps': risk.get('volatility_bps', 0),
                'information_ratio': risk.get('information_ratio', 0)
            }
        
        # Signal quality
        if 'signal_attribution' in evaluation_results:
            attribution = evaluation_results['signal_attribution']
            summary['signal_quality'] = {
                'combined_score_correlation': attribution.get('combined_score_correlation', 0),
                'top_bottom_spread_bps': attribution.get('top_bottom_decile_spread_bps', 0)
            }
        
        # Model calibration
        if 'calibration' in evaluation_results:
            calibration = evaluation_results['calibration']
            summary['model_calibration'] = {
                'brier_score': calibration.get('brier_score', 0),
                'log_loss': calibration.get('log_loss', 0),
                'calibration_slope': calibration.get('calibration_slope', 0)
            }
        
        return summary

def run_daily_evaluation() -> Dict:
    """Main entry point voor daily evaluation"""
    
    evaluator = DailyEvaluator()
    return evaluator.run_daily_evaluation()

if __name__ == "__main__":
    # Run als standalone script
    result = run_daily_evaluation()
    print(f"Daily evaluation completed: {result['status']}")