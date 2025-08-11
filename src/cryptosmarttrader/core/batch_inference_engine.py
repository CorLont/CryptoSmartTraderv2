#!/usr/bin/env python3
"""
Multi-Horizon Batch Inference Engine
Implements uniform batch processing for all coins across all horizons with atomic operations
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger
from core.data_quality_manager import get_data_quality_manager
from core.hard_data_filter import get_hard_data_filter
from core.async_data_manager import get_async_data_manager
from core.ml_slo_monitor import get_slo_monitor

class InferenceStatus(str, Enum):
    """Batch inference status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class HorizonType(str, Enum):
    """Time horizons for predictions"""
    H1 = "1h"
    H4 = "4h"
    H24 = "24h"
    D7 = "7d"
    D30 = "30d"

@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference"""
    horizons: List[HorizonType] = field(default_factory=lambda: [HorizonType.H1, HorizonType.H24, HorizonType.D7])
    batch_size: int = 100
    max_parallel_coins: int = 50
    model_timeout_seconds: int = 30
    feature_extraction_timeout: int = 60
    atomic_write_enabled: bool = True
    retry_attempts: int = 3
    completeness_threshold: float = 0.8

@dataclass
class CoinPrediction:
    """Single coin prediction across all horizons"""
    symbol: str
    timestamp: datetime
    predictions: Dict[HorizonType, float]  # horizon -> prediction
    confidence_scores: Dict[HorizonType, float]  # horizon -> confidence
    features: Dict[str, float]  # unified feature set
    model_versions: Dict[HorizonType, str]  # horizon -> model version
    inference_latency_ms: float
    data_completeness: float

@dataclass
class BatchInferenceResult:
    """Complete batch inference result"""
    batch_id: str
    timestamp: datetime
    config: BatchInferenceConfig
    status: InferenceStatus
    total_coins: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[CoinPrediction]
    execution_time_seconds: float
    average_latency_ms: float
    completeness_stats: Dict[str, float]
    error_summary: Dict[str, int]

class FeatureEngineering:
    """Unified feature engineering for all horizons"""
    
    def __init__(self):
        self.logger = get_logger()
        self.feature_cache = {}
        
    def extract_unified_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract unified feature set that works for all horizons"""
        
        features = {}
        
        try:
            # Price-based features
            features.update(self._extract_price_features(coin_data))
            
            # Volume-based features
            features.update(self._extract_volume_features(coin_data))
            
            # Technical indicator features
            features.update(self._extract_technical_features(coin_data))
            
            # Market microstructure features
            features.update(self._extract_microstructure_features(coin_data))
            
            # Cross-coin features
            features.update(self._extract_cross_coin_features(coin_data))
            
            # Validate features
            features = self._validate_and_normalize_features(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            features = self._get_fallback_features()
        
        return features
    
    def _extract_price_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract price-based features"""
        
        features = {}
        
        # Current price metrics
        current_price = coin_data.get("price", 0)
        bid = coin_data.get("bid", 0)
        ask = coin_data.get("ask", 0)
        
        if current_price > 0:
            features["price_normalized"] = np.log(current_price)
            
            if bid > 0 and ask > 0:
                spread_bps = ((ask - bid) / current_price) * 10000
                features["spread_bps"] = min(spread_bps, 1000)  # Cap at 1000 bps
                features["mid_price_deviation"] = (current_price - (bid + ask) / 2) / current_price
        
        # OHLCV-based price features
        ohlcv_data = coin_data.get("ohlcv", {})
        if ohlcv_data:
            for horizon, data in ohlcv_data.items():
                if data and len(data) >= 20:  # At least 20 periods
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Price momentum features
                    returns = df['close'].pct_change().dropna()
                    features[f"returns_mean_{horizon}"] = returns.mean()
                    features[f"returns_std_{horizon}"] = returns.std()
                    features[f"returns_skew_{horizon}"] = returns.skew()
                    features[f"returns_kurt_{horizon}"] = returns.kurtosis()
                    
                    # Price level features
                    features[f"price_vs_high_{horizon}"] = (current_price - df['high'].max()) / df['high'].max()
                    features[f"price_vs_low_{horizon}"] = (current_price - df['low'].min()) / df['low'].min()
                    
                    # Volatility features
                    high_low_vol = ((df['high'] - df['low']) / df['close']).mean()
                    features[f"hl_volatility_{horizon}"] = high_low_vol
        
        return features
    
    def _extract_volume_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volume-based features"""
        
        features = {}
        
        current_volume = coin_data.get("volume", 0)
        if current_volume > 0:
            features["volume_log"] = np.log(current_volume)
        
        # OHLCV volume features
        ohlcv_data = coin_data.get("ohlcv", {})
        if ohlcv_data:
            for horizon, data in ohlcv_data.items():
                if data and len(data) >= 10:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Volume statistics
                    vol_mean = df['volume'].mean()
                    vol_std = df['volume'].std()
                    
                    features[f"volume_mean_{horizon}"] = np.log(vol_mean) if vol_mean > 0 else 0
                    features[f"volume_cv_{horizon}"] = vol_std / vol_mean if vol_mean > 0 else 0
                    features[f"volume_trend_{horizon}"] = np.corrcoef(range(len(df)), df['volume'])[0, 1]
                    
                    # Volume-price relationship
                    if len(df) > 1:
                        price_change = df['close'].pct_change().dropna()
                        volume_change = df['volume'].pct_change().dropna()
                        
                        if len(price_change) > 1 and len(volume_change) > 1:
                            min_len = min(len(price_change), len(volume_change))
                            corr = np.corrcoef(price_change[-min_len:], volume_change[-min_len:])[0, 1]
                            features[f"price_volume_corr_{horizon}"] = corr if not np.isnan(corr) else 0
        
        return features
    
    def _extract_technical_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical indicator features"""
        
        features = {}
        
        # Basic technical indicators from ticker
        if coin_data.get("high") and coin_data.get("low") and coin_data.get("price"):
            high = coin_data["high"]
            low = coin_data["low"]
            close = coin_data["price"]
            
            features["daily_range"] = (high - low) / close if close > 0 else 0
            features["close_vs_high"] = (close - high) / high if high > 0 else 0
            features["close_vs_low"] = (close - low) / low if low > 0 else 0
        
        # Advanced technical indicators from OHLCV
        ohlcv_data = coin_data.get("ohlcv", {})
        if ohlcv_data:
            for horizon, data in ohlcv_data.items():
                if data and len(data) >= 50:  # Need enough data for indicators
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # RSI
                    rsi = self._calculate_rsi(df['close'], 14)
                    features[f"rsi_{horizon}"] = rsi
                    
                    # Moving averages
                    sma_20 = df['close'].rolling(20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    if sma_20 > 0 and sma_50 > 0:
                        features[f"price_vs_sma20_{horizon}"] = (current_price - sma_20) / sma_20
                        features[f"price_vs_sma50_{horizon}"] = (current_price - sma_50) / sma_50
                        features[f"sma20_vs_sma50_{horizon}"] = (sma_20 - sma_50) / sma_50
                    
                    # Bollinger Bands
                    sma = df['close'].rolling(20).mean()
                    std = df['close'].rolling(20).std()
                    upper_band = sma + (std * 2)
                    lower_band = sma - (std * 2)
                    
                    bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
                    features[f"bb_position_{horizon}"] = bb_position if not np.isnan(bb_position) else 0.5
        
        return features
    
    def _extract_microstructure_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market microstructure features"""
        
        features = {}
        
        # Order book features
        order_book = coin_data.get("order_book", {})
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if bids and asks:
                # Bid-ask spread
                best_bid = bids[0][0] if bids[0] else 0
                best_ask = asks[0][0] if asks[0] else 0
                
                if best_bid > 0 and best_ask > 0:
                    spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
                    features["bid_ask_spread"] = spread
                
                # Order book depth
                bid_depth = sum(bid[1] for bid in bids[:5] if len(bid) >= 2)
                ask_depth = sum(ask[1] for ask in asks[:5] if len(ask) >= 2)
                
                features["bid_depth"] = np.log(bid_depth) if bid_depth > 0 else 0
                features["ask_depth"] = np.log(ask_depth) if ask_depth > 0 else 0
                features["order_book_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        return features
    
    def _extract_cross_coin_features(self, coin_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cross-coin correlation features"""
        
        features = {}
        
        # Market cap tier (estimated from volume)
        volume = coin_data.get("volume", 0)
        if volume > 100000000:  # $100M+
            features["market_tier"] = 1.0  # Large cap
        elif volume > 10000000:  # $10M+
            features["market_tier"] = 0.7  # Mid cap
        elif volume > 1000000:  # $1M+
            features["market_tier"] = 0.4  # Small cap
        else:
            features["market_tier"] = 0.1  # Micro cap
        
        # Market activity level
        change = coin_data.get("change", 0)
        if abs(change) > 10:
            features["volatility_regime"] = 1.0  # High volatility
        elif abs(change) > 5:
            features["volatility_regime"] = 0.5  # Medium volatility
        else:
            features["volatility_regime"] = 0.1  # Low volatility
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0  # Neutral RSI
    
    def _validate_and_normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize features"""
        
        validated_features = {}
        
        for key, value in features.items():
            # Handle NaN and infinite values
            if np.isnan(value) or np.isinf(value):
                validated_features[key] = 0.0
            else:
                # Clip extreme values
                validated_features[key] = np.clip(value, -10.0, 10.0)
        
        return validated_features
    
    def _get_fallback_features(self) -> Dict[str, float]:
        """Get fallback features when extraction fails"""
        
        return {
            "price_normalized": 0.0,
            "volume_log": 0.0,
            "returns_mean_1d": 0.0,
            "returns_std_1d": 0.01,
            "market_tier": 0.1,
            "volatility_regime": 0.1
        }

class ModelInference:
    """Multi-horizon model inference engine"""
    
    def __init__(self):
        self.logger = get_logger()
        self.models = {}
        self.model_versions = {}
        
    async def load_models(self, horizons: List[HorizonType]) -> bool:
        """Load models for all required horizons"""
        
        try:
            for horizon in horizons:
                model_path = Path(f"models/ml_model_{horizon.value}.pkl")
                
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[horizon] = pickle.load(f)
                        self.model_versions[horizon] = f"v1.0_{datetime.now().strftime('%Y%m%d')}"
                    
                    self.logger.info(f"Model loaded for horizon {horizon.value}")
                else:
                    # Create simple mock model for demonstration
                    self.models[horizon] = self._create_mock_model()
                    self.model_versions[horizon] = f"mock_v1.0_{datetime.now().strftime('%Y%m%d')}"
                    
                    self.logger.warning(f"Using mock model for horizon {horizon.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def predict_all_horizons(
        self, 
        features: Dict[str, float], 
        horizons: List[HorizonType]
    ) -> Tuple[Dict[HorizonType, float], Dict[HorizonType, float]]:
        """Generate predictions for all horizons simultaneously"""
        
        predictions = {}
        confidence_scores = {}
        
        # Convert features to array
        feature_array = np.array([features.get(f"returns_mean_1d", 0), 
                                 features.get("volume_log", 0), 
                                 features.get("market_tier", 0.1)])
        
        for horizon in horizons:
            try:
                model = self.models.get(horizon)
                if model:
                    # Generate prediction
                    pred = model.predict([feature_array])[0]
                    
                    # Generate confidence (simplified)
                    confidence = min(0.5 + abs(pred) * 0.3, 0.95)
                    
                    predictions[horizon] = float(pred)
                    confidence_scores[horizon] = float(confidence)
                else:
                    predictions[horizon] = 0.0
                    confidence_scores[horizon] = 0.1
                    
            except Exception as e:
                self.logger.warning(f"Prediction failed for horizon {horizon.value}: {e}")
                predictions[horizon] = 0.0
                confidence_scores[horizon] = 0.1
        
        return predictions, confidence_scores
    
    def _create_mock_model(self):
        """Create mock model for demonstration"""
        from sklearn.linear_model import LinearRegression
        
        # Train simple model on random data
        X = np.random.randn(100, 3)
        y = np.random.randn(100) * 0.1
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model

class BatchInferenceEngine:
    """Main batch inference engine"""
    
    def __init__(self, config: Optional[BatchInferenceConfig] = None):
        self.config = config or BatchInferenceConfig()
        self.logger = get_logger()
        self.data_quality_manager = get_data_quality_manager()
        self.hard_data_filter = get_hard_data_filter()
        self.slo_monitor = get_slo_monitor()
        
        # Components
        self.feature_engineering = FeatureEngineering()
        self.model_inference = ModelInference()
        
        # State tracking
        self.current_batch = None
        self.batch_history = []
        
    async def run_batch_inference(self, target_coins: Optional[Set[str]] = None) -> BatchInferenceResult:
        """Run complete batch inference for all coins across all horizons"""
        
        batch_start = datetime.now()
        batch_id = f"batch_{batch_start.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting batch inference: {batch_id}")
        
        try:
            # Initialize models
            models_loaded = await self.model_inference.load_models(self.config.horizons)
            if not models_loaded:
                raise Exception("Failed to load required models")
            
            # Get filtered coin data
            coin_data = await self._get_filtered_coin_data(target_coins)
            
            if not coin_data:
                raise Exception("No valid coin data available")
            
            # Process coins in batches
            predictions = []
            successful_count = 0
            failed_count = 0
            error_summary = {}
            latencies = []
            
            total_coins = len(coin_data)
            
            # Process coins in parallel batches
            coin_symbols = list(coin_data.keys())
            
            for i in range(0, len(coin_symbols), self.config.batch_size):
                batch_symbols = coin_symbols[i:i + self.config.batch_size]
                
                # Process batch
                batch_predictions = await self._process_coin_batch(
                    {symbol: coin_data[symbol] for symbol in batch_symbols}
                )
                
                for prediction in batch_predictions:
                    if prediction.predictions:
                        predictions.append(prediction)
                        successful_count += 1
                        latencies.append(prediction.inference_latency_ms)
                    else:
                        failed_count += 1
                        error_summary["prediction_failed"] = error_summary.get("prediction_failed", 0) + 1
                
                # Log progress
                self.logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(coin_symbols) + self.config.batch_size - 1)//self.config.batch_size}")
            
            # Calculate statistics
            execution_time = (datetime.now() - batch_start).total_seconds()
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            completeness_scores = [p.data_completeness for p in predictions]
            completeness_stats = {
                "mean": np.mean(completeness_scores) if completeness_scores else 0.0,
                "median": np.median(completeness_scores) if completeness_scores else 0.0,
                "min": np.min(completeness_scores) if completeness_scores else 0.0,
                "max": np.max(completeness_scores) if completeness_scores else 0.0
            }
            
            # Determine status
            if successful_count == total_coins:
                status = InferenceStatus.COMPLETED
            elif successful_count > 0:
                status = InferenceStatus.PARTIAL
            else:
                status = InferenceStatus.FAILED
            
            # Create result
            result = BatchInferenceResult(
                batch_id=batch_id,
                timestamp=batch_start,
                config=self.config,
                status=status,
                total_coins=total_coins,
                successful_predictions=successful_count,
                failed_predictions=failed_count,
                predictions=predictions,
                execution_time_seconds=execution_time,
                average_latency_ms=avg_latency,
                completeness_stats=completeness_stats,
                error_summary=error_summary
            )
            
            # Store result atomically
            await self._store_batch_result_atomic(result)
            
            # Record SLO metrics
            await self._record_slo_metrics(result)
            
            # Update batch history
            self.batch_history.append(result)
            if len(self.batch_history) > 100:  # Keep last 100 batches
                self.batch_history = self.batch_history[-100:]
            
            self.current_batch = result
            
            self.logger.info(
                f"Batch inference completed: {batch_id}",
                extra={
                    "batch_id": batch_id,
                    "status": status.value,
                    "successful_predictions": successful_count,
                    "failed_predictions": failed_count,
                    "execution_time_seconds": execution_time,
                    "average_latency_ms": avg_latency
                }
            )
            
            return result
            
        except Exception as e:
            # Create failed result
            execution_time = (datetime.now() - batch_start).total_seconds()
            
            failed_result = BatchInferenceResult(
                batch_id=batch_id,
                timestamp=batch_start,
                config=self.config,
                status=InferenceStatus.FAILED,
                total_coins=0,
                successful_predictions=0,
                failed_predictions=0,
                predictions=[],
                execution_time_seconds=execution_time,
                average_latency_ms=0.0,
                completeness_stats={},
                error_summary={"batch_failure": 1}
            )
            
            self.batch_history.append(failed_result)
            self.current_batch = failed_result
            
            self.logger.error(f"Batch inference failed: {batch_id} - {e}")
            
            return failed_result
    
    async def _get_filtered_coin_data(self, target_coins: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Get filtered coin data using hard data filter"""
        
        # Get raw coin data
        async_data_manager = await get_async_data_manager()
        raw_data = await async_data_manager.batch_collect_all_exchanges()
        
        # Extract coin data from exchanges
        all_coin_data = {}
        
        for exchange_name, exchange_data in raw_data.get("exchanges", {}).items():
            if exchange_data.get("tickers"):
                for symbol, ticker_data in exchange_data["tickers"].items():
                    # Add OHLCV and order book data if available
                    ticker_data["ohlcv"] = exchange_data.get("ohlcv", {}).get(symbol, {})
                    ticker_data["order_book"] = exchange_data.get("order_books", {}).get(symbol, {})
                    
                    all_coin_data[symbol] = ticker_data
        
        # Filter by target coins if specified
        if target_coins:
            all_coin_data = {symbol: data for symbol, data in all_coin_data.items() if symbol in target_coins}
        
        # Apply hard data filter
        filtered_data, filter_stats = self.hard_data_filter.apply_hard_filter(all_coin_data)
        
        self.logger.info(
            f"Data filtering completed: {filter_stats.coins_passed}/{filter_stats.total_coins_processed} coins passed",
            extra={
                "total_processed": filter_stats.total_coins_processed,
                "passed": filter_stats.coins_passed,
                "blocked": filter_stats.coins_blocked,
                "pass_rate": filter_stats.coins_passed / filter_stats.total_coins_processed if filter_stats.total_coins_processed > 0 else 0
            }
        )
        
        return filtered_data
    
    async def _process_coin_batch(self, coin_batch: Dict[str, Any]) -> List[CoinPrediction]:
        """Process batch of coins for predictions"""
        
        predictions = []
        
        # Process coins in parallel
        tasks = [
            self._process_single_coin(symbol, coin_data)
            for symbol, coin_data in coin_batch.items()
        ]
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_parallel_coins)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Collect results
        for symbol, result in zip(coin_batch.keys(), results):
            if isinstance(result, Exception):
                self.logger.warning(f"Failed to process coin {symbol}: {result}")
            elif result:
                predictions.append(result)
        
        return predictions
    
    async def _process_single_coin(self, symbol: str, coin_data: Dict[str, Any]) -> Optional[CoinPrediction]:
        """Process single coin for multi-horizon predictions"""
        
        start_time = datetime.now()
        
        try:
            # Extract unified features
            features = self.feature_engineering.extract_unified_features(coin_data)
            
            # Generate predictions for all horizons
            predictions, confidence_scores = self.model_inference.predict_all_horizons(
                features, self.config.horizons
            )
            
            # Calculate data completeness
            data_completeness = self._calculate_coin_completeness(coin_data)
            
            # Calculate inference latency
            inference_latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return CoinPrediction(
                symbol=symbol,
                timestamp=start_time,
                predictions=predictions,
                confidence_scores=confidence_scores,
                features=features,
                model_versions={h: self.model_inference.model_versions.get(h, "unknown") for h in self.config.horizons},
                inference_latency_ms=inference_latency,
                data_completeness=data_completeness
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process coin {symbol}: {e}")
            return None
    
    def _calculate_coin_completeness(self, coin_data: Dict[str, Any]) -> float:
        """Calculate data completeness score for coin"""
        
        components = ["price", "volume", "ohlcv", "order_book"]
        score = 0.0
        
        if coin_data.get("price", 0) > 0:
            score += 0.3
        
        if coin_data.get("volume", 0) > 0:
            score += 0.3
        
        if coin_data.get("ohlcv") and any(coin_data["ohlcv"].values()):
            score += 0.3
        
        if coin_data.get("order_book"):
            score += 0.1
        
        return score
    
    async def _store_batch_result_atomic(self, result: BatchInferenceResult):
        """Store batch result atomically"""
        
        try:
            # Prepare data for storage
            storage_data = {
                "batch_id": result.batch_id,
                "timestamp": result.timestamp.isoformat(),
                "status": result.status.value,
                "total_coins": result.total_coins,
                "successful_predictions": result.successful_predictions,
                "failed_predictions": result.failed_predictions,
                "execution_time_seconds": result.execution_time_seconds,
                "average_latency_ms": result.average_latency_ms,
                "completeness_stats": result.completeness_stats,
                "error_summary": result.error_summary,
                "predictions": [
                    {
                        "symbol": p.symbol,
                        "timestamp": p.timestamp.isoformat(),
                        "predictions": {h.value: pred for h, pred in p.predictions.items()},
                        "confidence_scores": {h.value: conf for h, conf in p.confidence_scores.items()},
                        "features": p.features,
                        "model_versions": {h.value: ver for h, ver in p.model_versions.items()},
                        "inference_latency_ms": p.inference_latency_ms,
                        "data_completeness": p.data_completeness
                    }
                    for p in result.predictions
                ]
            }
            
            # Create directories
            batch_dir = Path("data/batch_inference")
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first (atomic operation)
            temp_file = batch_dir / f"{result.batch_id}.tmp"
            final_file = batch_dir / f"{result.batch_id}.json"
            
            with open(temp_file, 'w') as f:
                json.dump(storage_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(final_file)
            
            # Also store latest batch
            latest_file = batch_dir / "latest_batch.json"
            temp_latest = batch_dir / "latest_batch.tmp"
            
            with open(temp_latest, 'w') as f:
                json.dump(storage_data, f, indent=2)
            
            temp_latest.rename(latest_file)
            
            self.logger.info(f"Batch result stored atomically: {final_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store batch result: {e}")
            raise
    
    async def _record_slo_metrics(self, result: BatchInferenceResult):
        """Record SLO metrics for batch inference"""
        
        try:
            # Record batch-level metrics
            for prediction in result.predictions:
                for horizon, pred_value in prediction.predictions.items():
                    confidence = prediction.confidence_scores.get(horizon, 0.5)
                    
                    # Record individual prediction
                    self.slo_monitor.record_performance(
                        horizon=horizon.value,
                        model_version=prediction.model_versions.get(horizon, "unknown"),
                        predictions=[pred_value],
                        actuals=[0.0],  # Would be populated with actual returns later
                        confidence_scores=[confidence]
                    )
            
        except Exception as e:
            self.logger.warning(f"Failed to record SLO metrics: {e}")
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get comprehensive batch inference summary"""
        
        if not self.current_batch:
            return {"error": "No batch data available"}
        
        recent_batches = self.batch_history[-10:] if len(self.batch_history) >= 10 else self.batch_history
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_batch": {
                "batch_id": self.current_batch.batch_id,
                "status": self.current_batch.status.value,
                "total_coins": self.current_batch.total_coins,
                "successful_predictions": self.current_batch.successful_predictions,
                "success_rate": self.current_batch.successful_predictions / self.current_batch.total_coins if self.current_batch.total_coins > 0 else 0,
                "average_latency_ms": self.current_batch.average_latency_ms,
                "completeness_stats": self.current_batch.completeness_stats
            },
            "trends": {
                "total_batches_run": len(self.batch_history),
                "recent_success_rate": np.mean([b.successful_predictions / b.total_coins for b in recent_batches if b.total_coins > 0]),
                "recent_avg_latency": np.mean([b.average_latency_ms for b in recent_batches]),
                "recent_avg_coins": np.mean([b.total_coins for b in recent_batches])
            },
            "configuration": {
                "horizons": [h.value for h in self.config.horizons],
                "batch_size": self.config.batch_size,
                "max_parallel_coins": self.config.max_parallel_coins,
                "completeness_threshold": self.config.completeness_threshold
            }
        }

# Global instance
_batch_inference_engine = None

def get_batch_inference_engine(config: Optional[BatchInferenceConfig] = None) -> BatchInferenceEngine:
    """Get global batch inference engine instance"""
    global _batch_inference_engine
    if _batch_inference_engine is None:
        _batch_inference_engine = BatchInferenceEngine(config)
    return _batch_inference_engine