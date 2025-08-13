#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Meta-Learning Coordinator
Coordination layer for meta-learning across multiple trading strategies and markets
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from copy import deepcopy
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .continual_learning_engine import get_continual_learning_engine

class TaskType(Enum):
    NEW_COIN_ADAPTATION = "new_coin_adaptation"
    MARKET_REGIME_ADAPTATION = "market_regime_adaptation"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    CROSS_EXCHANGE_TRANSFER = "cross_exchange_transfer"
    TIMEFRAME_ADAPTATION = "timeframe_adaptation"

class MetaObjective(Enum):
    FAST_ADAPTATION = "fast_adaptation"         # Learn new tasks quickly
    ROBUST_GENERALIZATION = "robust_generalization"  # Generalize across markets
    CATASTROPHIC_FORGETTING_PREVENTION = "forgetting_prevention"
    MULTI_TASK_PERFORMANCE = "multi_task_performance"

@dataclass
class MetaTask:
    """Meta-learning task definition"""
    task_id: str
    task_type: TaskType
    support_set: Tuple[torch.Tensor, torch.Tensor]  # (features, targets)
    query_set: Tuple[torch.Tensor, torch.Tensor]
    task_metadata: Dict[str, Any]
    created_at: datetime
    difficulty_score: float = 0.5
    priority: int = 1

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning coordinator"""
    # MAML settings
    inner_learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-4
    inner_steps: int = 5
    meta_batch_size: int = 8

    # Task generation
    min_support_samples: int = 10
    max_support_samples: int = 50
    min_query_samples: int = 20

    # Portfolio meta-learning
    portfolio_lookback_days: int = 30
    strategy_ensemble_size: int = 5
    adaptation_validation_split: float = 0.3

    # Online meta-learning
    online_task_buffer_size: int = 100
    meta_update_frequency: int = 10
    experience_replay_ratio: float = 0.3

    # Cross-domain adaptation
    domain_similarity_threshold: float = 0.7
    transfer_learning_weight: float = 0.5

    # Performance tracking
    adaptation_time_limit_seconds: int = 30
    min_adaptation_improvement: float = 0.05

@dataclass
class AdaptationResult:
    """Result of meta-learning adaptation"""
    task_id: str
    adapted_model_id: str
    adaptation_steps: int
    final_loss: float
    improvement_ratio: float
    adaptation_time_seconds: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class PortfolioMetaLearner(nn.Module):
    """Meta-learner for portfolio optimization across different market conditions"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Portfolio allocation head
        self.portfolio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Single asset allocation
            nn.Sigmoid()
        )

        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Market regime head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 market regimes
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.feature_encoder(features)

        return {
            'allocation': self.portfolio_head(encoded),
            'risk': self.risk_head(encoded),
            'regime': self.regime_head(encoded)
        }

    def adapt_to_market(self, support_x: torch.Tensor, support_y: torch.Tensor,
                       adaptation_steps: int = 5, learning_rate: float = 1e-3) -> 'PortfolioMetaLearner':
        """Adapt portfolio strategy to new market conditions"""
        adapted_model = deepcopy(self)
        optimizer = optim.SGD(adapted_model.parameters(), lr=learning_rate)

        for step in range(adaptation_steps):
            optimizer.zero_grad()

            outputs = adapted_model(support_x)

            # Multi-objective loss
            allocation_loss = F.mse_loss(outputs['allocation'], support_y[:, 0:1])
            risk_loss = F.mse_loss(outputs['risk'], support_y[:, 1:2])

            if support_y.shape[1] > 2:
                regime_loss = F.cross_entropy(outputs['regime'], support_y[:, 2].long())
                total_loss = allocation_loss + 0.5 * risk_loss + 0.3 * regime_loss
            else:
                total_loss = allocation_loss + 0.5 * risk_loss

            total_loss.backward()
            optimizer.step()

        return adapted_model

class TaskGenerator:
    """Generate meta-learning tasks from historical trading data"""

    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TaskGenerator")

        # Historical data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.strategy_results: Dict[str, List] = {}

    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """Add market data for task generation"""
        self.market_data[symbol] = data.copy()

    def generate_new_coin_tasks(self, new_coin: str, reference_coins: List[str]) -> List[MetaTask]:
        """Generate tasks for adapting to new cryptocurrency"""
        tasks = []

        try:
            if new_coin not in self.market_data:
                return tasks

            new_coin_data = self.market_data[new_coin]

            # Generate tasks based on similar coins
            for ref_coin in reference_coins:
                if ref_coin in self.market_data:
                    ref_data = self.market_data[ref_coin]

                    # Find similar market periods
                    similar_periods = self._find_similar_market_periods(new_coin_data, ref_data)

                    for period_start, period_end in similar_periods:
                        task = self._create_adaptation_task(
                            f"new_coin_{new_coin}_{ref_coin}_{period_start}",
                            TaskType.NEW_COIN_ADAPTATION,
                            new_coin_data.iloc[period_start:period_end],
                            ref_coin
                        )
                        if task:
                            tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"New coin task generation failed: {e}")
            return []

    def generate_regime_adaptation_tasks(self, lookback_days: int = 30) -> List[MetaTask]:
        """Generate tasks for market regime adaptation"""
        tasks = []

        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)

            for symbol, data in self.market_data.items():
                if data.empty:
                    continue

                # Identify different market regimes in historical data
                regimes = self._identify_market_regimes(data)

                # Create tasks for each regime transition
                for i in range(len(regimes) - 1):
                    current_regime = regimes[i]
                    next_regime = regimes[i + 1]

                    task = self._create_regime_transition_task(
                        symbol, current_regime, next_regime, data
                    )
                    if task:
                        tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"Regime adaptation task generation failed: {e}")
            return []

    def generate_strategy_optimization_tasks(self) -> List[MetaTask]:
        """Generate tasks for strategy parameter optimization"""
        tasks = []

        try:
            for symbol in self.market_data:
                # Different strategy configurations to test
                strategy_configs = [
                    {'lookback': 20, 'threshold': 0.02},
                    {'lookback': 50, 'threshold': 0.05},
                    {'lookback': 100, 'threshold': 0.1}
                ]

                for config in strategy_configs:
                    task = self._create_strategy_optimization_task(symbol, config)
                    if task:
                        tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"Strategy optimization task generation failed: {e}")
            return []

    def _find_similar_market_periods(self, target_data: pd.DataFrame,
                                   reference_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Find periods with similar market characteristics"""
        similar_periods = []

        try:
            # Calculate market features for comparison
            target_features = self._calculate_market_features(target_data)
            ref_features = self._calculate_market_features(reference_data)

            if target_features is None or ref_features is None:
                return similar_periods

            # Find similar periods using correlation
            window_size = min(50, len(target_features), len(ref_features))

            for i in range(len(ref_features) - window_size):
                ref_window = ref_features[i:i+window_size]

                # Calculate similarity with target
                for j in range(len(target_features) - window_size):
                    target_window = target_features[j:j+window_size]

                    correlation = np.corrcoef(ref_window.flatten(), target_window.flatten())[0, 1]

                    if correlation > 0.7 and not np.isnan(correlation):
                        similar_periods.append((i, i + window_size))
                        break

            return similar_periods[:5]  # Limit to top 5

        except Exception as e:
            self.logger.error(f"Similar period detection failed: {e}")
            return []

    def _calculate_market_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate market features for similarity comparison"""
        try:
            if 'close' not in data.columns and 'price' not in data.columns:
                return None

            price_col = 'close' if 'close' in data.columns else 'price'
            prices = data[price_col].values

            if len(prices) < 20:
                return None

            # Calculate features
            returns = np.diff(prices) / prices[:-1]

            features = []
            window_sizes = [5, 10, 20]

            for window in window_sizes:
                if len(returns) >= window:
                    # Rolling statistics
                    rolling_mean = pd.Series(returns).rolling(window).mean().fillna(0).values
                    rolling_std = pd.Series(returns).rolling(window).std().fillna(0).values
                    rolling_skew = pd.Series(returns).rolling(window).skew().fillna(0).values

                    features.append(rolling_mean)
                    features.append(rolling_std)
                    features.append(rolling_skew)

            if features:
                return np.column_stack(features)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Market feature calculation failed: {e}")
            return None

    def _identify_market_regimes(self, data: pd.DataFrame) -> List[Dict]:
        """Identify different market regimes in historical data"""
        regimes = []

        try:
            if 'close' not in data.columns and 'price' not in data.columns:
                return regimes

            price_col = 'close' if 'close' in data.columns else 'price'
            prices = data[price_col].values

            if len(prices) < 50:
                return regimes

            # Calculate regime indicators
            returns = pd.Series(prices).pct_change().fillna(0)
            volatility = returns.rolling(20).std()
            trend = pd.Series(prices).rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

            # Simple regime classification
            for i in range(20, len(prices), 20):  # 20-period regimes
                period_vol = volatility.iloc[i-20:i].mean()
                period_trend = trend.iloc[i-20:i].mean()

                if period_trend > 0.01 and period_vol < 0.02:
                    regime_type = "bull_low_vol"
                elif period_trend > 0.01 and period_vol >= 0.02:
                    regime_type = "bull_high_vol"
                elif period_trend < -0.01 and period_vol < 0.02:
                    regime_type = "bear_low_vol"
                elif period_trend < -0.01 and period_vol >= 0.02:
                    regime_type = "bear_high_vol"
                else:
                    regime_type = "sideways"

                regimes.append({
                    'start_idx': i-20,
                    'end_idx': i,
                    'type': regime_type,
                    'volatility': period_vol,
                    'trend': period_trend
                })

            return regimes

        except Exception as e:
            self.logger.error(f"Regime identification failed: {e}")
            return []

    def _create_adaptation_task(self, task_id: str, task_type: TaskType,
                              data: pd.DataFrame, reference_symbol: str) -> Optional[MetaTask]:
        """Create adaptation task from market data"""
        try:
            if len(data) < self.config.min_support_samples + self.config.min_query_samples:
                return None

            # Extract features and targets
            features, targets = self._extract_features_targets(data)

            if features is None or targets is None:
                return None

            # Split into support and query sets
            support_size = min(self.config.max_support_samples,
                             max(self.config.min_support_samples, len(features) // 3))

            support_x = torch.FloatTensor(features[:support_size])
            support_y = torch.FloatTensor(targets[:support_size])
            query_x = torch.FloatTensor(features[support_size:])
            query_y = torch.FloatTensor(targets[support_size:])

            return MetaTask(
                task_id=task_id,
                task_type=task_type,
                support_set=(support_x, support_y),
                query_set=(query_x, query_y),
                task_metadata={'reference_symbol': reference_symbol},
                created_at=datetime.now(),
                difficulty_score=np.std(targets)  # Higher volatility = harder task
            )

        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            return None

    def _create_regime_transition_task(self, symbol: str, current_regime: Dict,
                                     next_regime: Dict, data: pd.DataFrame) -> Optional[MetaTask]:
        """Create task for regime transition adaptation"""
        try:
            # Extract data for both regimes
            current_data = data.iloc[current_regime['start_idx']:current_regime['end_idx']]
            next_data = data.iloc[next_regime['start_idx']:next_regime['end_idx']]

            # Use current regime as support, next regime as query
            support_features, support_targets = self._extract_features_targets(current_data)
            query_features, query_targets = self._extract_features_targets(next_data)

            if any(x is None for x in [support_features, support_targets, query_features, query_targets]):
                return None

            task_id = f"regime_{symbol}_{current_regime['type']}_to_{next_regime['type']}"

            return MetaTask(
                task_id=task_id,
                task_type=TaskType.MARKET_REGIME_ADAPTATION,
                support_set=(torch.FloatTensor(support_features), torch.FloatTensor(support_targets)),
                query_set=(torch.FloatTensor(query_features), torch.FloatTensor(query_targets)),
                task_metadata={
                    'symbol': symbol,
                    'from_regime': current_regime['type'],
                    'to_regime': next_regime['type']
                },
                created_at=datetime.now(),
                difficulty_score=abs(current_regime['volatility'] - next_regime['volatility'])
            )

        except Exception as e:
            self.logger.error(f"Regime transition task creation failed: {e}")
            return None

    def _create_strategy_optimization_task(self, symbol: str, config: Dict) -> Optional[MetaTask]:
        """Create strategy optimization task"""
        try:
            if symbol not in self.market_data:
                return None

            data = self.market_data[symbol]
            features, targets = self._extract_features_targets(data, strategy_config=config)

            if features is None or targets is None:
                return None

            # Split data
            split_idx = len(features) // 2
            support_x = torch.FloatTensor(features[:split_idx])
            support_y = torch.FloatTensor(targets[:split_idx])
            query_x = torch.FloatTensor(features[split_idx:])
            query_y = torch.FloatTensor(targets[split_idx:])

            task_id = f"strategy_{symbol}_{config['lookback']}_{config['threshold']}"

            return MetaTask(
                task_id=task_id,
                task_type=TaskType.STRATEGY_OPTIMIZATION,
                support_set=(support_x, support_y),
                query_set=(query_x, query_y),
                task_metadata={'symbol': symbol, 'config': config},
                created_at=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Strategy optimization task creation failed: {e}")
            return None

    def _extract_features_targets(self, data: pd.DataFrame,
                                strategy_config: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract features and targets from market data"""
        try:
            if data.empty:
                return None, None

            price_col = 'close' if 'close' in data.columns else 'price'
            if price_col not in data.columns:
                return None, None

            prices = data[price_col].values
            if len(prices) < 20:
                return None, None

            # Calculate features
            returns = pd.Series(prices).pct_change().fillna(0).values

            # Rolling features
            features = []
            for window in [5, 10, 20]:
                if len(prices) >= window:
                    rolling_mean = pd.Series(returns).rolling(window).mean().fillna(0).values
                    rolling_std = pd.Series(returns).rolling(window).std().fillna(0).values

                    features.append(rolling_mean)
                    features.append(rolling_std)

            if not features:
                return None, None

            feature_matrix = np.column_stack(features)

            # Calculate targets (future returns)
            targets = np.roll(returns, -1)[:-1]  # Next period return
            feature_matrix = feature_matrix[:-1]  # Remove last row

            # Apply strategy config if provided
            if strategy_config:
                lookback = strategy_config.get('lookback', 20)
                threshold = strategy_config.get('threshold', 0.02)

                # Filter based on signal strength
                signal_strength = np.abs(feature_matrix[:, 0])  # Use first feature as signal
                mask = signal_strength > threshold

                feature_matrix = feature_matrix[mask]
                targets = targets[mask]

            if len(feature_matrix) == 0:
                return None, None

            return feature_matrix, targets.reshape(-1, 1)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None, None

class MetaLearningCoordinator:
    """Main coordinator for meta-learning across trading strategies"""

    def __init__(self, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        self.logger = logging.getLogger(f"{__name__}.MetaLearningCoordinator")

        if not HAS_TORCH:
            self.logger.error("PyTorch not available - meta-learning disabled")
            return

        # Core components
        self.task_generator = TaskGenerator(self.config)
        self.continual_engine = get_continual_learning_engine()

        # Meta-learning models
        self.meta_models: Dict[str, nn.Module] = {}
        self.portfolio_meta_learner: Optional[PortfolioMetaLearner] = None

        # Task management
        self.task_queue: List[MetaTask] = []
        self.completed_tasks: List[MetaTask] = []
        self.adaptation_results: List[AdaptationResult] = []

        # Performance tracking
        self.meta_performance: Dict[str, List] = {
            'adaptation_times': [],
            'improvement_ratios': [],
            'success_rates': []
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._lock = threading.RLock()

        self.logger.info(f"Meta-Learning Coordinator initialized on {self.device}")

    def initialize_portfolio_meta_learner(self, feature_dim: int):
        """Initialize portfolio meta-learner"""
        with self._lock:
            try:
                self.portfolio_meta_learner = PortfolioMetaLearner(feature_dim).to(self.device)
                self.logger.info(f"Portfolio meta-learner initialized with {feature_dim} features")
                return True
            except Exception as e:
                self.logger.error(f"Portfolio meta-learner initialization failed: {e}")
                return False

    def add_market_data_for_tasks(self, symbol: str, data: pd.DataFrame):
        """Add market data for task generation"""
        self.task_generator.add_market_data(symbol, data)

    def generate_adaptation_tasks(self, new_coin: str, reference_coins: List[str]) -> int:
        """Generate adaptation tasks for new cryptocurrency"""
        with self._lock:
            try:
                # Generate different types of tasks
                new_coin_tasks = self.task_generator.generate_new_coin_tasks(new_coin, reference_coins)
                regime_tasks = self.task_generator.generate_regime_adaptation_tasks()
                strategy_tasks = self.task_generator.generate_strategy_optimization_tasks()

                # Add to queue
                all_tasks = new_coin_tasks + regime_tasks + strategy_tasks
                self.task_queue.extend(all_tasks)

                # Sort by priority and difficulty
                self.task_queue.sort(key=lambda t: (t.priority, -t.difficulty_score), reverse=True)

                self.logger.info(f"Generated {len(all_tasks)} meta-learning tasks for {new_coin}")
                return len(all_tasks)

            except Exception as e:
                self.logger.error(f"Task generation failed: {e}")
                return 0

    def execute_meta_learning_batch(self, batch_size: Optional[int] = None) -> List[AdaptationResult]:
        """Execute a batch of meta-learning tasks"""
        with self._lock:
            try:
                batch_size = batch_size or self.config.meta_batch_size

                if len(self.task_queue) < batch_size:
                    return []

                # Select tasks for batch
                batch_tasks = self.task_queue[:batch_size]
                self.task_queue = self.task_queue[batch_size:]

                results = []

                for task in batch_tasks:
                    result = self._execute_single_task(task)
                    if result:
                        results.append(result)
                        self.adaptation_results.append(result)

                    self.completed_tasks.append(task)

                # Update meta-learning performance metrics
                self._update_meta_performance_metrics(results)

                return results

            except Exception as e:
                self.logger.error(f"Meta-learning batch execution failed: {e}")
                return []

    def _execute_single_task(self, task: MetaTask) -> Optional[AdaptationResult]:
        """Execute single meta-learning task"""
        try:
            start_time = datetime.now()

            # Get base model for adaptation
            base_model = self._get_base_model_for_task(task)
            if base_model is None:
                return None

            # Perform adaptation
            adapted_model, final_loss, steps = self._adapt_model_to_task(base_model, task)

            # Evaluate adaptation
            improvement = self._evaluate_adaptation(base_model, adapted_model, task)

            adaptation_time = (datetime.now() - start_time).total_seconds()

            # Register adapted model if successful
            adapted_model_id = None
            if improvement > self.config.min_adaptation_improvement:
                adapted_model_id = f"adapted_{task.task_id}_{int(start_time.timestamp())}"
                self.continual_engine.register_model(adapted_model_id, adapted_model)

            success = improvement > self.config.min_adaptation_improvement

            result = AdaptationResult(
                task_id=task.task_id,
                adapted_model_id=adapted_model_id,
                adaptation_steps=steps,
                final_loss=final_loss,
                improvement_ratio=improvement,
                adaptation_time_seconds=adaptation_time,
                success=success,
                metadata={
                    'task_type': task.task_type.value,
                    'difficulty_score': task.difficulty_score,
                    'base_model_performance': self._get_baseline_performance(base_model, task)
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Task execution failed for {task.task_id}: {e}")
            return None

    def _get_base_model_for_task(self, task: MetaTask) -> Optional[nn.Module]:
        """Get appropriate base model for task type"""
        try:
            if task.task_type == TaskType.NEW_COIN_ADAPTATION:
                # Use portfolio meta-learner for new coin adaptation
                if self.portfolio_meta_learner is not None:
                    return deepcopy(self.portfolio_meta_learner)

            elif task.task_type == TaskType.MARKET_REGIME_ADAPTATION:
                # Use regime-specific model if available
                reference_symbol = task.task_metadata.get('symbol', 'default')
                if reference_symbol in self.meta_models:
                    return deepcopy(self.meta_models[reference_symbol])
                elif self.portfolio_meta_learner is not None:
                    return deepcopy(self.portfolio_meta_learner)

            elif task.task_type == TaskType.STRATEGY_OPTIMIZATION:
                # Use strategy-specific model
                if 'strategy' in self.meta_models:
                    return deepcopy(self.meta_models['strategy'])
                elif self.portfolio_meta_learner is not None:
                    return deepcopy(self.portfolio_meta_learner)

            # Fallback to portfolio meta-learner
            if self.portfolio_meta_learner is not None:
                return deepcopy(self.portfolio_meta_learner)

            return None

        except Exception as e:
            self.logger.error(f"Base model selection failed: {e}")
            return None

    def _adapt_model_to_task(self, base_model: nn.Module, task: MetaTask) -> Tuple[nn.Module, float, int]:
        """Adapt model to specific task"""
        try:
            adapted_model = deepcopy(base_model).to(self.device)
            optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_learning_rate)

            support_x, support_y = task.support_set
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)

            final_loss = 0.0

            # Adaptation steps
            for step in range(self.config.inner_steps):
                optimizer.zero_grad()

                if isinstance(adapted_model, PortfolioMetaLearner):
                    outputs = adapted_model(support_x)
                    loss = F.mse_loss(outputs['allocation'], support_y)
                else:
                    outputs = adapted_model(support_x)
                    loss = F.mse_loss(outputs, support_y)

                loss.backward()
                optimizer.step()

                final_loss = loss.item()

            return adapted_model, final_loss, self.config.inner_steps

        except Exception as e:
            self.logger.error(f"Model adaptation failed: {e}")
            return base_model, float('inf'), 0

    def _evaluate_adaptation(self, base_model: nn.Module, adapted_model: nn.Module, task: MetaTask) -> float:
        """Evaluate adaptation improvement"""
        try:
            query_x, query_y = task.query_set
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Evaluate base model
            base_model.eval()
            with torch.no_grad():
                if isinstance(base_model, PortfolioMetaLearner):
                    base_outputs = base_model(query_x)
                    base_loss = F.mse_loss(base_outputs['allocation'], query_y)
                else:
                    base_outputs = base_model(query_x)
                    base_loss = F.mse_loss(base_outputs, query_y)

            # Evaluate adapted model
            adapted_model.eval()
            with torch.no_grad():
                if isinstance(adapted_model, PortfolioMetaLearner):
                    adapted_outputs = adapted_model(query_x)
                    adapted_loss = F.mse_loss(adapted_outputs['allocation'], query_y)
                else:
                    adapted_outputs = adapted_model(query_x)
                    adapted_loss = F.mse_loss(adapted_outputs, query_y)

            # Calculate improvement ratio
            if base_loss.item() > 0:
                improvement = (base_loss.item() - adapted_loss.item()) / base_loss.item()
            else:
                improvement = 0.0

            return improvement

        except Exception as e:
            self.logger.error(f"Adaptation evaluation failed: {e}")
            return 0.0

    def _get_baseline_performance(self, model: nn.Module, task: MetaTask) -> float:
        """Get baseline performance before adaptation"""
        try:
            query_x, query_y = task.query_set
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            model.eval()
            with torch.no_grad():
                if isinstance(model, PortfolioMetaLearner):
                    outputs = model(query_x)
                    loss = F.mse_loss(outputs['allocation'], query_y)
                else:
                    outputs = model(query_x)
                    loss = F.mse_loss(outputs, query_y)

            return loss.item()

        except Exception:
            return 0.0

    def _update_meta_performance_metrics(self, results: List[AdaptationResult]):
        """Update meta-learning performance metrics"""
        try:
            if not results:
                return

            # Collect metrics
            adaptation_times = [r.adaptation_time_seconds for r in results]
            improvement_ratios = [r.improvement_ratio for r in results]
            success_rate = sum(1 for r in results if r.success) / len(results)

            # Update history
            self.meta_performance['adaptation_times'].extend(adaptation_times)
            self.meta_performance['improvement_ratios'].extend(improvement_ratios)
            self.meta_performance['success_rates'].append(success_rate)

            # Limit history size
            max_history = 1000
            for key in self.meta_performance:
                if len(self.meta_performance[key]) > max_history:
                    self.meta_performance[key] = self.meta_performance[key][-max_history:]

        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")

    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning summary"""
        with self._lock:
            summary = {
                'meta_models_registered': len(self.meta_models),
                'portfolio_meta_learner_available': self.portfolio_meta_learner is not None,
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.completed_tasks),
                'successful_adaptations': sum(1 for r in self.adaptation_results if r.success),
                'total_adaptations': len(self.adaptation_results)
            }

            # Performance metrics
            if self.meta_performance['adaptation_times']:
                summary['average_adaptation_time'] = np.mean(self.meta_performance['adaptation_times'])
                summary['average_improvement_ratio'] = np.mean(self.meta_performance['improvement_ratios'])
                summary['recent_success_rate'] = np.mean(self.meta_performance['success_rates'][-10:]) if self.meta_performance['success_rates'] else 0.0

            # Task type breakdown
            task_types = {}
            for task in self.completed_tasks:
                task_type = task.task_type.value
                task_types[task_type] = task_types.get(task_type, 0) + 1
            summary['completed_task_types'] = task_types

            return summary


# Singleton meta-learning coordinator
_meta_learning_coordinator = None
_meta_lock = threading.Lock()

def get_meta_learning_coordinator(config: Optional[MetaLearningConfig] = None) -> MetaLearningCoordinator:
    """Get the singleton meta-learning coordinator"""
    global _meta_learning_coordinator

    with _meta_lock:
        if _meta_learning_coordinator is None:
            _meta_learning_coordinator = MetaLearningCoordinator(config)
        return _meta_learning_coordinator

def adapt_to_new_coin(new_coin: str, reference_coins: List[str]) -> int:
    """Convenient function to adapt models to new cryptocurrency"""
    coordinator = get_meta_learning_coordinator()
    return coordinator.generate_adaptation_tasks(new_coin, reference_coins)
