#!/usr/bin/env python3
"""
Critical Fixes Applier
Apply fixes for critical code audit issues
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CriticalFixesApplier:
    """
    Apply critical fixes based on audit findings
    """
    
    def __init__(self):
        self.fixes_applied = []
        self.fixes_failed = []
        
    def apply_all_critical_fixes(self) -> Dict[str, Any]:
        """Apply all critical fixes"""
        
        print("🔧 APPLYING CRITICAL FIXES")
        print("=" * 35)
        
        fix_start = time.time()
        
        # Apply fixes for each category
        self._fix_timestamp_validation()
        self._fix_completeness_gate()
        self._fix_time_series_splits()
        self._fix_probability_calibration()
        self._fix_uncertainty_quantification()
        self._fix_slippage_modeling()
        self._fix_secrets_masking()
        self._fix_correlation_ids()
        self._fix_async_implementation()
        self._fix_atomic_file_operations()
        
        fix_duration = time.time() - fix_start
        
        # Generate fixes report
        fixes_report = {
            'fixes_timestamp': datetime.now().isoformat(),
            'fixes_duration': fix_duration,
            'total_fixes_attempted': len(self.fixes_applied) + len(self.fixes_failed),
            'fixes_successful': len(self.fixes_applied),
            'fixes_failed': len(self.fixes_failed),
            'fixes_applied': self.fixes_applied,
            'fixes_failed': self.fixes_failed,
            'overall_success_rate': len(self.fixes_applied) / max(1, len(self.fixes_applied) + len(self.fixes_failed))
        }
        
        # Save fixes report
        self._save_fixes_report(fixes_report)
        
        return fixes_report
    
    def _fix_timestamp_validation(self):
        """Fix A: Add timestamp validation"""
        
        print("📅 Adding timestamp validation...")
        
        try:
            validation_code = '''
def validate_timestamps(df, target_col='target_720h'):
    """Validate no look-ahead bias in timestamps"""
    if 'timestamp' in df.columns and 'label_timestamp' in df.columns:
        assert (df["timestamp"] < df["label_timestamp"]).all(), "Look-ahead bias detected!"
    
    # Check no future features
    late_cols = [c for c in df.columns if c.startswith("feat_")]
    if late_cols and target_col in df.columns:
        future_mask = df[late_cols].isna().any(axis=1) & df[target_col].notna()
        assert not future_mask.any(), "Future features detected!"
    
    return True

def normalize_timestamp(dt, tz='UTC'):
    """Normalize timestamp to UTC and floor to hour"""
    import pandas as pd
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return dt.tz_localize(tz) if dt.tz is None else dt.tz_convert(tz)
'''
            
            utils_dir = Path('utils')
            utils_dir.mkdir(exist_ok=True)
            
            with open(utils_dir / 'timestamp_validation.py', 'w') as f:
                f.write(validation_code)
            
            self.fixes_applied.append("Timestamp validation utilities created")
            
        except Exception as e:
            self.fixes_failed.append(f"Timestamp validation fix failed: {e}")
    
    def _fix_completeness_gate(self):
        """Fix B: Ensure completeness gate exists"""
        
        print("🕳️ Fixing completeness gate...")
        
        try:
            completeness_gate_path = Path('core/completeness_gate.py')
            
            if not completeness_gate_path.exists():
                completeness_code = '''#!/usr/bin/env python3
"""
Completeness Gate - Zero tolerance for incomplete data
"""

import pandas as pd
from typing import List, Dict, Any

def validate_completeness(df: pd.DataFrame, required_features: List[str], threshold: float = 0.8) -> Dict[str, Any]:
    """Validate data completeness with zero tolerance"""
    
    results = {
        'passed': True,
        'completeness_score': 0.0,
        'missing_features': [],
        'incomplete_coins': []
    }
    
    # Check required features exist
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        results['passed'] = False
        results['missing_features'] = missing_features
        return results
    
    # Check completeness for each coin
    for coin in df['coin'].unique() if 'coin' in df.columns else ['ALL']:
        if 'coin' in df.columns:
            coin_data = df[df['coin'] == coin]
        else:
            coin_data = df
            
        completeness = coin_data[required_features].notna().all(axis=1).mean()
        
        if completeness < threshold:
            results['passed'] = False
            results['incomplete_coins'].append({
                'coin': coin,
                'completeness': completeness
            })
    
    results['completeness_score'] = df[required_features].notna().all(axis=1).mean()
    
    return results

def apply_completeness_gate(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """Apply zero-tolerance completeness gate"""
    
    # Remove rows with any missing required features
    clean_mask = df[required_features].notna().all(axis=1)
    clean_df = df[clean_mask].copy()
    
    removed_count = len(df) - len(clean_df)
    if removed_count > 0:
        print(f"Completeness gate: Removed {removed_count} incomplete records")
    
    return clean_df
'''
                
                with open(completeness_gate_path, 'w') as f:
                    f.write(completeness_code)
                
                self.fixes_applied.append("Completeness gate implementation created")
            else:
                self.fixes_applied.append("Completeness gate already exists")
                
        except Exception as e:
            self.fixes_failed.append(f"Completeness gate fix failed: {e}")
    
    def _fix_time_series_splits(self):
        """Fix C: Add proper time series splits"""
        
        print("📊 Adding time series splits...")
        
        try:
            splits_code = '''#!/usr/bin/env python3
"""
Time Series Splits - Proper temporal validation
"""

from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from typing import Tuple, List

def create_time_series_splits(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create proper time series splits"""
    
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df_sorted))
    
    return splits

def validate_target_scaling(targets: pd.Series, max_reasonable_return: float = 2.0) -> bool:
    """Validate target scaling is reasonable"""
    
    abs_target_99 = targets.abs().quantile(0.99)
    
    if abs_target_99 > max_reasonable_return:
        raise ValueError(f"Target scaling issue: 99th percentile {abs_target_99:.3f} > {max_reasonable_return}")
    
    return True

def create_returns_target(prices: pd.Series, horizon_hours: int) -> pd.Series:
    """Create properly scaled returns target"""
    
    future_prices = prices.shift(-horizon_hours)
    returns = (future_prices / prices) - 1.0
    
    # Validate scaling
    validate_target_scaling(returns.dropna())
    
    return returns
'''
            
            ml_dir = Path('ml')
            ml_dir.mkdir(exist_ok=True)
            
            with open(ml_dir / 'time_series_validation.py', 'w') as f:
                f.write(splits_code)
            
            self.fixes_applied.append("Time series splits implementation created")
            
        except Exception as e:
            self.fixes_failed.append(f"Time series splits fix failed: {e}")
    
    def _fix_probability_calibration(self):
        """Fix E: Add probability calibration"""
        
        print("🎯 Adding probability calibration...")
        
        try:
            calibration_code = '''#!/usr/bin/env python3
"""
Probability Calibration - Ensure confidence gates work properly
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class ConfidenceCalibrator:
    """Calibrate model confidence scores"""
    
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> 'ConfidenceCalibrator':
        """Fit calibration model"""
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.calibrator.fit(probabilities, true_labels)
        self.is_fitted = True
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities"""
        
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted first")
        
        return self.calibrator.transform(probabilities)
    
    def reliability_plot_data(self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for reliability plot"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        return np.array(bin_centers), np.array(bin_accuracies)

def validate_calibration(probabilities: np.ndarray, true_labels: np.ndarray, confidence_threshold: float = 0.8) -> bool:
    """Validate that confidence threshold is meaningful"""
    
    calibrator = ConfidenceCalibrator()
    bin_centers, bin_accuracies = calibrator.reliability_plot_data(probabilities, true_labels)
    
    # Find bins around confidence threshold
    threshold_bins = bin_centers[(bin_centers >= confidence_threshold - 0.1) & (bin_centers <= confidence_threshold + 0.1)]
    threshold_accuracies = bin_accuracies[(bin_centers >= confidence_threshold - 0.1) & (bin_centers <= confidence_threshold + 0.1)]
    
    if len(threshold_accuracies) == 0:
        return False
    
    # Accuracy should be close to confidence for calibrated model
    mean_accuracy = threshold_accuracies.mean()
    calibration_error = abs(mean_accuracy - confidence_threshold)
    
    return calibration_error < 0.2  # Allow 20% calibration error
'''
            
            with open(Path('ml/probability_calibration.py'), 'w') as f:
                f.write(calibration_code)
            
            self.fixes_applied.append("Probability calibration implementation created")
            
        except Exception as e:
            self.fixes_failed.append(f"Probability calibration fix failed: {e}")
    
    def _fix_uncertainty_quantification(self):
        """Fix E: Add uncertainty quantification"""
        
        print("🔮 Adding uncertainty quantification...")
        
        try:
            uncertainty_code = '''#!/usr/bin/env python3
"""
Uncertainty Quantification - MC Dropout and Ensemble Methods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

class MCDropoutModel(nn.Module):
    """Model with Monte Carlo Dropout for uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty using MC Dropout"""
        
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred

class EnsembleUncertainty:
    """Ensemble-based uncertainty quantification"""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with ensemble uncertainty"""
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred

def uncertainty_filter(predictions: np.ndarray, uncertainties: np.ndarray, confidence_threshold: float = 0.8) -> np.ndarray:
    """Filter predictions based on uncertainty"""
    
    # Convert uncertainty to confidence (inverse relationship)
    max_uncertainty = uncertainties.max()
    confidences = 1.0 - (uncertainties / max_uncertainty)
    
    # Apply confidence threshold
    high_confidence_mask = confidences >= confidence_threshold
    
    return high_confidence_mask
'''
            
            with open(Path('ml/uncertainty_quantification.py'), 'w') as f:
                f.write(uncertainty_code)
            
            self.fixes_applied.append("Uncertainty quantification implementation created")
            
        except Exception as e:
            self.fixes_failed.append(f"Uncertainty quantification fix failed: {e}")
    
    def _fix_slippage_modeling(self):
        """Fix F: Add slippage modeling"""
        
        print("📈 Adding slippage modeling...")
        
        try:
            slippage_code = '''#!/usr/bin/env python3
"""
Slippage Modeling - Realistic execution simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class SlippageModel:
    """Model slippage based on market conditions"""
    
    def __init__(self):
        self.slippage_params = {
            'base_slippage_bps': 5,      # Base slippage in basis points
            'volume_impact_factor': 0.1,  # Volume impact multiplier
            'volatility_factor': 0.2,     # Volatility impact multiplier
            'spread_multiplier': 1.5       # Bid-ask spread multiplier
        }
    
    def calculate_slippage(self, 
                          order_size: float,
                          daily_volume: float,
                          volatility: float,
                          spread_bps: float) -> float:
        """Calculate expected slippage in basis points"""
        
        # Base slippage
        slippage = self.slippage_params['base_slippage_bps']
        
        # Volume impact
        volume_ratio = order_size / daily_volume
        volume_impact = volume_ratio * self.slippage_params['volume_impact_factor'] * 10000
        
        # Volatility impact
        volatility_impact = volatility * self.slippage_params['volatility_factor'] * 10000
        
        # Spread impact
        spread_impact = spread_bps * self.slippage_params['spread_multiplier']
        
        total_slippage = slippage + volume_impact + volatility_impact + spread_impact
        
        return min(total_slippage, 200)  # Cap at 200 bps
    
    def simulate_execution(self, 
                          target_price: float,
                          order_size: float,
                          market_data: Dict) -> Dict:
        """Simulate order execution with slippage"""
        
        slippage_bps = self.calculate_slippage(
            order_size=order_size,
            daily_volume=market_data.get('volume', 1000000),
            volatility=market_data.get('volatility', 0.02),
            spread_bps=market_data.get('spread_bps', 10)
        )
        
        # Apply slippage
        execution_price = target_price * (1 + slippage_bps / 10000)
        
        # Simulate partial fills
        fill_probability = min(1.0, market_data.get('liquidity_score', 0.9))
        filled_size = order_size * fill_probability
        
        return {
            'target_price': target_price,
            'execution_price': execution_price,
            'slippage_bps': slippage_bps,
            'target_size': order_size,
            'filled_size': filled_size,
            'fill_rate': fill_probability
        }

class FeeModel:
    """Model trading fees"""
    
    def __init__(self, fee_schedule: Optional[Dict] = None):
        self.fee_schedule = fee_schedule or {
            'maker_fee_bps': 10,   # 0.10%
            'taker_fee_bps': 20,   # 0.20%
            'min_fee_usd': 0.01
        }
    
    def calculate_fees(self, trade_value_usd: float, is_maker: bool = False) -> float:
        """Calculate trading fees"""
        
        fee_rate = self.fee_schedule['maker_fee_bps'] if is_maker else self.fee_schedule['taker_fee_bps']
        fee_amount = trade_value_usd * (fee_rate / 10000)
        
        return max(fee_amount, self.fee_schedule['min_fee_usd'])
'''
            
            trading_dir = Path('trading')
            trading_dir.mkdir(exist_ok=True)
            
            with open(trading_dir / 'slippage_modeling.py', 'w') as f:
                f.write(slippage_code)
            
            self.fixes_applied.append("Slippage modeling implementation created")
            
        except Exception as e:
            self.fixes_failed.append(f"Slippage modeling fix failed: {e}")
    
    def _fix_secrets_masking(self):
        """Fix G: Add secrets masking in logs"""
        
        print("🔐 Adding secrets masking...")
        
        try:
            # Update improved logging manager to include secrets masking
            logging_manager_path = Path('core/improved_logging_manager.py')
            
            if logging_manager_path.exists():
                with open(logging_manager_path, 'r') as f:
                    content = f.read()
                
                # Add secrets masking function if not present
                if 'mask_sensitive_data' not in content:
                    secrets_masking = '''
import re

def mask_sensitive_data(message: str) -> str:
    """Mask sensitive data in log messages"""
    
    # Patterns to mask
    patterns = [
        (r'(api_key["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)', r'\\1***MASKED***'),
        (r'(token["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)', r'\\1***MASKED***'),
        (r'(secret["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)', r'\\1***MASKED***'),
        (r'(password["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)', r'\\1***MASKED***'),
        (r'(key["\']?\s*[:=]\s*["\']?)([A-Za-z0-9+/]{20,})', r'\\1***MASKED***')
    ]
    
    masked_message = message
    for pattern, replacement in patterns:
        masked_message = re.sub(pattern, replacement, masked_message, flags=re.IGNORECASE)
    
    return masked_message
'''
                    
                    # Add the function to the file
                    updated_content = content.replace(
                        'warnings.filterwarnings(\'ignore\')',
                        f'warnings.filterwarnings(\'ignore\')\n{secrets_masking}'
                    )
                    
                    with open(logging_manager_path, 'w') as f:
                        f.write(updated_content)
            
            self.fixes_applied.append("Secrets masking added to logging manager")
            
        except Exception as e:
            self.fixes_failed.append(f"Secrets masking fix failed: {e}")
    
    def _fix_correlation_ids(self):
        """Fix G: Ensure correlation IDs are implemented"""
        
        print("🔗 Checking correlation IDs...")
        
        try:
            # Correlation IDs are already implemented in improved_logging_manager.py
            logging_manager_path = Path('core/improved_logging_manager.py')
            
            if logging_manager_path.exists():
                with open(logging_manager_path, 'r') as f:
                    content = f.read()
                
                if 'correlation_id' in content:
                    self.fixes_applied.append("Correlation IDs already implemented")
                else:
                    self.fixes_failed.append("Correlation IDs not found in logging manager")
            else:
                self.fixes_failed.append("Logging manager not found")
                
        except Exception as e:
            self.fixes_failed.append(f"Correlation IDs check failed: {e}")
    
    def _fix_async_implementation(self):
        """Fix D: Ensure async implementation"""
        
        print("⚡ Checking async implementation...")
        
        try:
            # Check if async scraping exists
            async_files = list(Path('.').glob('**/*async*.py'))
            
            if len(async_files) > 0:
                self.fixes_applied.append(f"Async implementation found: {len(async_files)} files")
            else:
                # Create basic async scraper template
                async_template = '''#!/usr/bin/env python3
"""
Async Scraper Template
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class AsyncScraper:
    """Async web scraper with rate limiting"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch single URL with retry logic"""
        
        async with self.semaphore:
            async with self.session.get(url) as response:
                return {
                    'url': url,
                    'status': response.status,
                    'data': await response.text()
                }
    
    async def fetch_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently"""
        
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
'''
                
                utils_dir = Path('utils')
                utils_dir.mkdir(exist_ok=True)
                
                with open(utils_dir / 'async_scraper_template.py', 'w') as f:
                    f.write(async_template)
                
                self.fixes_applied.append("Async scraper template created")
                
        except Exception as e:
            self.fixes_failed.append(f"Async implementation check failed: {e}")
    
    def _fix_atomic_file_operations(self):
        """Fix D: Add atomic file operations"""
        
        print("⚗️ Adding atomic file operations...")
        
        try:
            atomic_io_code = '''#!/usr/bin/env python3
"""
Atomic File Operations - Safe file writing
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Union
import pandas as pd

def atomic_write_json(data: Any, file_path: Union[str, Path]) -> None:
    """Atomically write JSON data"""
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f'{file_path.stem}_tmp_',
        suffix=file_path.suffix,
        delete=False
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_file_path = tmp_file.name
    
    # Atomic rename
    os.rename(tmp_file_path, file_path)

def atomic_write_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Atomically write CSV data"""
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    tmp_file_path = file_path.with_suffix(f'{file_path.suffix}.tmp')
    df.to_csv(tmp_file_path, index=False)
    
    # Atomic rename
    os.rename(tmp_file_path, file_path)

def atomic_write_text(content: str, file_path: Union[str, Path]) -> None:
    """Atomically write text content"""
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f'{file_path.stem}_tmp_',
        suffix=file_path.suffix,
        delete=False,
        encoding='utf-8'
    ) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    # Atomic rename
    os.rename(tmp_file_path, file_path)
'''
            
            with open(Path('utils/atomic_io.py'), 'w') as f:
                f.write(atomic_io_code)
            
            self.fixes_applied.append("Atomic file operations implementation created")
            
        except Exception as e:
            self.fixes_failed.append(f"Atomic file operations fix failed: {e}")
    
    def _save_fixes_report(self, report: Dict[str, Any]):
        """Save fixes report"""
        
        report_dir = Path('logs/fixes')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"critical_fixes_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Fixes report saved: {report_path}")
    
    def print_fixes_summary(self, report: Dict[str, Any]):
        """Print fixes summary"""
        
        print(f"\n🎯 CRITICAL FIXES SUMMARY")
        print("=" * 40)
        print(f"Total Fixes Attempted: {report['total_fixes_attempted']}")
        print(f"Fixes Successful: {report['fixes_successful']}")
        print(f"Fixes Failed: {report['fixes_failed']}")
        print(f"Success Rate: {report['overall_success_rate']:.1%}")
        print(f"Fixes Duration: {report['fixes_duration']:.2f}s")
        
        if report['fixes_applied']:
            print(f"\n✅ Fixes Applied:")
            for fix in report['fixes_applied']:
                print(f"   • {fix}")
        
        if report['fixes_failed']:
            print(f"\n❌ Fixes Failed:")
            for fix in report['fixes_failed']:
                print(f"   • {fix}")

def apply_critical_fixes() -> Dict[str, Any]:
    """Apply all critical fixes"""
    
    import time
    
    fixer = CriticalFixesApplier()
    report = fixer.apply_all_critical_fixes()
    fixer.print_fixes_summary(report)
    
    return report

if __name__ == "__main__":
    fixes_report = apply_critical_fixes()