#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Reinforcement Learning Portfolio Allocator
Dynamic portfolio allocation using RL agents instead of fixed formulas
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import pickle
import json
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import sharpe_ratio
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class ActionSpace(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"

class RewardMetric(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    KELLY_CRITERION = "kelly_criterion"

class RLAlgorithm(Enum):
    PPO = "ppo"  # Proximal Policy Optimization
    SAC = "sac"  # Soft Actor-Critic
    A3C = "a3c"  # Asynchronous Advantage Actor-Critic
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient
    TD3 = "td3"  # Twin Delayed DDPG

@dataclass
class PortfolioState:
    """Current portfolio state for RL agent"""
    allocations: np.ndarray  # Current allocations per asset
    returns: np.ndarray     # Recent returns history
    volatilities: np.ndarray  # Recent volatility history
    correlations: np.ndarray  # Correlation matrix
    market_features: np.ndarray  # Market indicators
    cash_ratio: float = 0.0
    total_value: float = 100000.0
    drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    time_step: int = 0

@dataclass
class PortfolioAction:
    """Portfolio action from RL agent"""
    new_allocations: np.ndarray
    rebalance_strength: float = 1.0
    risk_level: float = 0.5
    confidence: float = 0.0

@dataclass
class RLPortfolioConfig:
    """Configuration for RL portfolio allocator"""
    # Asset universe
    max_assets: int = 20
    min_allocation: float = 0.0
    max_allocation: float = 0.3
    cash_buffer: float = 0.05
    
    # RL parameters
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    action_space: ActionSpace = ActionSpace.CONTINUOUS
    state_features: int = 50
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 3e-4
    
    # Training parameters
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 100
    target_update_frequency: int = 1000
    
    # Reward configuration
    primary_reward: RewardMetric = RewardMetric.SHARPE_RATIO
    reward_weights: Dict[RewardMetric, float] = field(default_factory=lambda: {
        RewardMetric.SHARPE_RATIO: 0.4,
        RewardMetric.MAX_DRAWDOWN: 0.3,
        RewardMetric.TOTAL_RETURN: 0.2,
        RewardMetric.RISK_ADJUSTED_RETURN: 0.1
    })
    
    # Risk management
    max_drawdown_threshold: float = 0.15
    risk_free_rate: float = 0.02
    rebalance_threshold: float = 0.05
    
    # Model persistence
    save_models: bool = True
    model_cache_dir: str = "models/rl_portfolio"

class PPOActor(nn.Module):
    """PPO Actor network for portfolio allocation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers for mean and std of action distribution
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = self.network(state)
        
        # Mean of action distribution (raw logits)
        mean = self.mean_layer(x)
        
        # Standard deviation (ensure positive)
        std = F.softplus(self.std_layer(x)) + 1e-6
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """Get action from current policy"""
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            # Sample from normal distribution
            dist = Normal(mean, std)
            action = dist.sample()
        
        # Apply softmax to ensure allocations sum to 1
        allocation_logits = action
        allocations = F.softmax(allocation_logits, dim=-1)
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return allocations, log_prob

class PPOCritic(nn.Module):
    """PPO Critic network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_layers: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Single value output
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        return self.network(state)

class PortfolioEnvironment:
    """Portfolio trading environment for RL"""
    
    def __init__(self, data: pd.DataFrame, config: RLPortfolioConfig):
        self.data = data
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PortfolioEnvironment")
        
        # Environment state
        self.current_step = 0
        self.portfolio_value = 100000.0
        self.initial_value = 100000.0
        self.cash = 0.0
        self.positions = {}
        
        # Asset information
        self.asset_columns = [col for col in data.columns if col.endswith('_close')]
        self.n_assets = min(len(self.asset_columns), config.max_assets)
        
        if self.n_assets == 0:
            # Fallback: use any numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            self.asset_columns = numeric_columns[:config.max_assets]
            self.n_assets = len(self.asset_columns)
        
        # History tracking
        self.portfolio_history = []
        self.allocation_history = []
        self.return_history = []
        
        self.logger.info(f"Portfolio environment initialized with {self.n_assets} assets")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.cash = self.config.cash_buffer * self.portfolio_value
        
        # Initialize equal allocations
        equal_allocation = (1.0 - self.config.cash_buffer) / self.n_assets
        self.positions = {asset: equal_allocation for asset in self.asset_columns[:self.n_assets]}
        
        # Clear history
        self.portfolio_history = [self.portfolio_value]
        self.allocation_history = []
        self.return_history = []
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        try:
            # Validate action
            if len(action) != self.n_assets:
                action = np.ones(self.n_assets) / self.n_assets
            
            # Normalize allocations to sum to 1
            allocations = np.clip(action, self.config.min_allocation, self.config.max_allocation)
            allocations = allocations / allocations.sum()
            
            # Move to next time step
            self.current_step += 1
            
            # Check if episode is done
            done = self.current_step >= len(self.data) - 1
            
            if done:
                return self._get_state(), 0.0, done, {}
            
            # Calculate returns for this step
            returns = self._calculate_returns()
            
            # Update portfolio based on allocations and returns
            old_value = self.portfolio_value
            self._update_portfolio(allocations, returns)
            
            # Calculate reward
            reward = self._calculate_reward(old_value)
            
            # Update history
            self.allocation_history.append(allocations.copy())
            self.portfolio_history.append(self.portfolio_value)
            
            if len(self.portfolio_history) > 1:
                period_return = (self.portfolio_value - old_value) / old_value
                self.return_history.append(period_return)
            
            # Create info dict
            info = {
                'portfolio_value': self.portfolio_value,
                'period_return': self.return_history[-1] if self.return_history else 0.0,
                'allocations': allocations.copy(),
                'drawdown': self._calculate_drawdown()
            }
            
            return self._get_state(), reward, done, info
            
        except Exception as e:
            self.logger.error(f"Environment step failed: {e}")
            return self._get_state(), -1.0, True, {}
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        try:
            if self.current_step >= len(self.data):
                return np.zeros(self.config.state_features)
            
            # Current data point
            current_data = self.data.iloc[self.current_step]
            
            # Asset prices (normalized)
            asset_prices = []
            for asset in self.asset_columns[:self.n_assets]:
                if asset in current_data:
                    asset_prices.append(current_data[asset])
                else:
                    asset_prices.append(0.0)
            
            # Price features
            price_features = np.array(asset_prices)
            if len(price_features) > 0:
                price_features = price_features / (np.max(price_features) + 1e-8)
            
            # Portfolio features
            current_allocations = np.array([self.positions.get(asset, 0.0) for asset in self.asset_columns[:self.n_assets]])
            
            # Recent returns
            recent_returns = []
            lookback = min(10, self.current_step)
            
            if lookback > 0 and len(self.return_history) >= lookback:
                recent_returns = self.return_history[-lookback:]
            else:
                recent_returns = [0.0] * 10
            
            # Pad or trim to exact size
            recent_returns = (recent_returns + [0.0] * 10)[:10]
            
            # Market indicators
            market_indicators = [
                self._calculate_drawdown(),
                self._calculate_sharpe_ratio(),
                len(self.return_history),
                self.portfolio_value / self.initial_value - 1.0  # Total return
            ]
            
            # Combine all features
            state_features = np.concatenate([
                price_features,
                current_allocations,
                recent_returns,
                market_indicators
            ])
            
            # Ensure fixed size
            if len(state_features) > self.config.state_features:
                state_features = state_features[:self.config.state_features]
            elif len(state_features) < self.config.state_features:
                padding = np.zeros(self.config.state_features - len(state_features))
                state_features = np.concatenate([state_features, padding])
            
            return state_features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"State calculation failed: {e}")
            return np.zeros(self.config.state_features)
    
    def _calculate_returns(self) -> Dict[str, float]:
        """Calculate asset returns for current period"""
        returns = {}
        
        if self.current_step == 0:
            return {asset: 0.0 for asset in self.asset_columns[:self.n_assets]}
        
        try:
            current_data = self.data.iloc[self.current_step]
            previous_data = self.data.iloc[self.current_step - 1]
            
            for asset in self.asset_columns[:self.n_assets]:
                if asset in current_data and asset in previous_data:
                    current_price = current_data[asset]
                    previous_price = previous_data[asset]
                    
                    if previous_price > 0:
                        returns[asset] = (current_price - previous_price) / previous_price
                    else:
                        returns[asset] = 0.0
                else:
                    returns[asset] = 0.0
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Return calculation failed: {e}")
            return {asset: 0.0 for asset in self.asset_columns[:self.n_assets]}
    
    def _update_portfolio(self, allocations: np.ndarray, returns: Dict[str, float]):
        """Update portfolio based on allocations and returns"""
        try:
            # Calculate new portfolio value
            portfolio_return = 0.0
            
            for i, asset in enumerate(self.asset_columns[:self.n_assets]):
                if i < len(allocations):
                    allocation = allocations[i]
                    asset_return = returns.get(asset, 0.0)
                    portfolio_return += allocation * asset_return
                    
                    # Update position
                    self.positions[asset] = allocation
            
            # Update portfolio value
            self.portfolio_value *= (1.0 + portfolio_return)
            
            # Update cash
            self.cash = self.config.cash_buffer * self.portfolio_value
            
        except Exception as e:
            self.logger.error(f"Portfolio update failed: {e}")
    
    def _calculate_reward(self, old_value: float) -> float:
        """Calculate reward for current step"""
        try:
            rewards = {}
            
            # Period return
            period_return = (self.portfolio_value - old_value) / old_value if old_value > 0 else 0.0
            
            # Sharpe ratio reward
            if len(self.return_history) >= 10:
                returns_array = np.array(self.return_history[-10:])
                if np.std(returns_array) > 0:
                    sharpe = (np.mean(returns_array) - self.config.risk_free_rate / 252) / np.std(returns_array)
                    rewards[RewardMetric.SHARPE_RATIO] = sharpe
                else:
                    rewards[RewardMetric.SHARPE_RATIO] = 0.0
            else:
                rewards[RewardMetric.SHARPE_RATIO] = 0.0
            
            # Drawdown penalty
            drawdown = self._calculate_drawdown()
            rewards[RewardMetric.MAX_DRAWDOWN] = -drawdown * 10  # Penalty for drawdown
            
            # Total return reward
            total_return = self.portfolio_value / self.initial_value - 1.0
            rewards[RewardMetric.TOTAL_RETURN] = total_return
            
            # Risk-adjusted return
            if len(self.return_history) >= 5:
                returns_std = np.std(self.return_history[-5:])
                if returns_std > 0:
                    risk_adjusted = period_return / returns_std
                    rewards[RewardMetric.RISK_ADJUSTED_RETURN] = risk_adjusted
                else:
                    rewards[RewardMetric.RISK_ADJUSTED_RETURN] = period_return
            else:
                rewards[RewardMetric.RISK_ADJUSTED_RETURN] = period_return
            
            # Combine rewards based on weights
            total_reward = 0.0
            for metric, weight in self.config.reward_weights.items():
                if metric in rewards:
                    total_reward += weight * rewards[metric]
            
            # Additional penalties
            if drawdown > self.config.max_drawdown_threshold:
                total_reward -= 5.0  # Large penalty for excessive drawdown
            
            return float(total_reward)
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            return 0.0
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        peak = max(self.portfolio_history)
        current = self.portfolio_history[-1]
        
        return (peak - current) / peak if peak > 0 else 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.return_history) < 10:
            return 0.0
        
        returns_array = np.array(self.return_history)
        excess_returns = returns_array - self.config.risk_free_rate / 252
        
        if np.std(excess_returns) > 0:
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            return 0.0

class PPOPortfolioAgent:
    """PPO agent for portfolio allocation"""
    
    def __init__(self, config: RLPortfolioConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PPOPortfolioAgent")
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for RL portfolio allocation")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = PPOActor(
            config.state_features,
            config.max_assets,
            config.hidden_layers
        ).to(self.device)
        
        self.critic = PPOCritic(
            config.state_features,
            config.hidden_layers
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Training data
        self.memory = []
        self.training_step = 0
        
        self.logger.info(f"PPO Portfolio Agent initialized on {self.device}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from agent"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                allocations, log_prob = self.actor.get_action(state_tensor, deterministic)
            
            return allocations.cpu().numpy()[0], log_prob.cpu().item()
            
        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            # Return equal allocations as fallback
            n_assets = self.config.max_assets
            return np.ones(n_assets) / n_assets, 0.0
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, 
                        done: bool, log_prob: float):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
        
        # Keep memory size manageable
        if len(self.memory) > self.config.buffer_size:
            self.memory = self.memory[-self.config.buffer_size:]
    
    def update(self):
        """Update agent networks"""
        try:
            if len(self.memory) < self.config.batch_size:
                return
            
            self.logger.debug(f"Updating PPO agent with {len(self.memory)} transitions")
            
            # Convert memory to tensors
            states = torch.FloatTensor([t['state'] for t in self.memory]).to(self.device)
            actions = torch.FloatTensor([t['action'] for t in self.memory]).to(self.device)
            rewards = torch.FloatTensor([t['reward'] for t in self.memory]).to(self.device)
            next_states = torch.FloatTensor([t['next_state'] for t in self.memory]).to(self.device)
            dones = torch.BoolTensor([t['done'] for t in self.memory]).to(self.device)
            old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.memory]).to(self.device)
            
            # Calculate advantages and returns
            with torch.no_grad():
                values = self.critic(states).squeeze()
                next_values = self.critic(next_states).squeeze()
                
                # Calculate returns using GAE
                returns = torch.zeros_like(rewards)
                advantages = torch.zeros_like(rewards)
                
                gae = 0
                gamma = 0.99
                lam = 0.95
                
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_value = 0 if dones[t] else next_values[t]
                    else:
                        next_value = values[t + 1]
                    
                    delta = rewards[t] + gamma * next_value - values[t]
                    gae = delta + gamma * lam * gae
                    advantages[t] = gae
                    returns[t] = advantages[t] + values[t]
            
            # Normalize advantages
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            clip_epsilon = 0.2
            
            for _ in range(4):  # Multiple epochs
                # Sample batch
                batch_indices = torch.randint(0, len(states), (self.config.batch_size,))
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Actor update
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                
                # Calculate log probabilities for batch actions
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Policy ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic update
                batch_values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(batch_values, batch_returns)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
            
            # Clear memory after update
            self.memory = []
            self.training_step += 1
            
        except Exception as e:
            self.logger.error(f"PPO update failed: {e}")

class ReinforcementPortfolioAllocator:
    """Main reinforcement learning portfolio allocator"""
    
    def __init__(self, config: Optional[RLPortfolioConfig] = None):
        self.config = config or RLPortfolioConfig()
        self.logger = logging.getLogger(f"{__name__}.ReinforcementPortfolioAllocator")
        
        # Core components
        self.agent: Optional[PPOPortfolioAgent] = None
        self.environment: Optional[PortfolioEnvironment] = None
        
        # State tracking
        self.current_allocations: Optional[np.ndarray] = None
        self.training_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Model persistence
        self.model_cache_dir = Path(self.config.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        
        self.logger.info("Reinforcement Learning Portfolio Allocator initialized")
    
    def initialize_agent(self, data: pd.DataFrame) -> bool:
        """Initialize RL agent and environment"""
        with self._lock:
            try:
                self.logger.info("Initializing RL agent and environment")
                
                # Create environment
                self.environment = PortfolioEnvironment(data, self.config)
                
                # Update config based on actual assets
                self.config.max_assets = self.environment.n_assets
                
                # Create agent
                if self.config.algorithm == RLAlgorithm.PPO:
                    self.agent = PPOPortfolioAgent(self.config)
                else:
                    raise NotImplementedError(f"Algorithm {self.config.algorithm.value} not yet implemented")
                
                # Try to load existing model
                self._load_model()
                
                self.logger.info(f"RL agent initialized with {self.environment.n_assets} assets")
                return True
                
            except Exception as e:
                self.logger.error(f"Agent initialization failed: {e}")
                return False
    
    def train(self, data: pd.DataFrame, episodes: int = 1000) -> Dict[str, Any]:
        """Train the RL agent"""
        with self._lock:
            try:
                if not self.initialize_agent(data):
                    return {'success': False, 'error': 'Agent initialization failed'}
                
                self.logger.info(f"Training RL agent for {episodes} episodes")
                
                training_results = {
                    'episode_rewards': [],
                    'episode_returns': [],
                    'episode_sharpe_ratios': [],
                    'final_allocations': [],
                    'best_episode': 0,
                    'best_reward': float('-inf')
                }
                
                for episode in range(episodes):
                    try:
                        episode_result = self._run_episode(training=True)
                        
                        # Track results
                        training_results['episode_rewards'].append(episode_result['total_reward'])
                        training_results['episode_returns'].append(episode_result['total_return'])
                        training_results['episode_sharpe_ratios'].append(episode_result['sharpe_ratio'])
                        training_results['final_allocations'].append(episode_result['final_allocations'])
                        
                        # Update best episode
                        if episode_result['total_reward'] > training_results['best_reward']:
                            training_results['best_reward'] = episode_result['total_reward']
                            training_results['best_episode'] = episode
                        
                        # Periodic logging
                        if episode % 100 == 0:
                            recent_rewards = training_results['episode_rewards'][-10:]
                            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                            
                            self.logger.info(
                                f"Episode {episode}: avg_reward={avg_reward:.4f}, "
                                f"best_reward={training_results['best_reward']:.4f}"
                            )
                        
                        # Update agent
                        if episode % self.config.update_frequency == 0:
                            self.agent.update()
                        
                    except Exception as e:
                        self.logger.error(f"Episode {episode} failed: {e}")
                        continue
                
                # Save trained model
                if self.config.save_models:
                    self._save_model()
                
                # Calculate final metrics
                training_results['success'] = True
                training_results['final_avg_reward'] = np.mean(training_results['episode_rewards'][-100:]) if training_results['episode_rewards'] else 0
                training_results['final_avg_return'] = np.mean(training_results['episode_returns'][-100:]) if training_results['episode_returns'] else 0
                training_results['final_avg_sharpe'] = np.mean(training_results['episode_sharpe_ratios'][-100:]) if training_results['episode_sharpe_ratios'] else 0
                
                self.training_history.append(training_results)
                
                self.logger.info(f"Training completed: avg_reward={training_results['final_avg_reward']:.4f}")
                
                return training_results
                
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def _run_episode(self, training: bool = True) -> Dict[str, Any]:
        """Run single episode"""
        try:
            state = self.environment.reset()
            total_reward = 0.0
            steps = 0
            
            episode_allocations = []
            episode_rewards = []
            
            while True:
                # Get action from agent
                action, log_prob = self.agent.get_action(state, deterministic=not training)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store transition for training
                if training:
                    self.agent.store_transition(state, action, reward, next_state, done, log_prob)
                
                # Update tracking
                total_reward += reward
                episode_allocations.append(action.copy())
                episode_rewards.append(reward)
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Calculate episode metrics
            final_value = self.environment.portfolio_value
            total_return = (final_value - self.environment.initial_value) / self.environment.initial_value
            sharpe_ratio = self.environment._calculate_sharpe_ratio()
            max_drawdown = max([self.environment._calculate_drawdown()] + [0.0])
            
            return {
                'total_reward': total_reward,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value,
                'steps': steps,
                'final_allocations': episode_allocations[-1] if episode_allocations else np.array([]),
                'episode_allocations': episode_allocations,
                'episode_rewards': episode_rewards
            }
            
        except Exception as e:
            self.logger.error(f"Episode execution failed: {e}")
            return {
                'total_reward': -10.0,
                'total_return': -0.1,
                'sharpe_ratio': -1.0,
                'max_drawdown': 0.5,
                'final_value': 50000.0,
                'steps': 0,
                'final_allocations': np.array([]),
                'episode_allocations': [],
                'episode_rewards': []
            }
    
    def get_optimal_allocation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get optimal portfolio allocation using trained agent"""
        with self._lock:
            try:
                if self.agent is None:
                    if not self.initialize_agent(data):
                        return {'success': False, 'error': 'Agent not available'}
                
                # Get current state
                if self.environment is None:
                    self.environment = PortfolioEnvironment(data, self.config)
                
                # Reset environment to get current state
                state = self.environment.reset()
                
                # Get action from trained agent
                allocations, confidence = self.agent.get_action(state, deterministic=True)
                
                # Validate allocations
                allocations = np.clip(allocations, self.config.min_allocation, self.config.max_allocation)
                allocations = allocations / allocations.sum()  # Ensure sum to 1
                
                # Map to asset names
                asset_allocations = {}
                for i, asset in enumerate(self.environment.asset_columns[:self.environment.n_assets]):
                    if i < len(allocations):
                        asset_allocations[asset] = float(allocations[i])
                
                # Add cash allocation
                cash_allocation = self.config.cash_buffer
                asset_allocations['cash'] = cash_allocation
                
                # Calculate expected metrics
                expected_return, expected_volatility = self._estimate_portfolio_metrics(data, allocations)
                
                self.current_allocations = allocations
                
                return {
                    'success': True,
                    'allocations': asset_allocations,
                    'expected_return': expected_return,
                    'expected_volatility': expected_volatility,
                    'confidence': confidence,
                    'rebalance_recommended': self._should_rebalance(allocations),
                    'risk_level': np.std(allocations),  # Concentration as risk proxy
                    'diversification_score': 1.0 - np.sum(allocations**2)  # Herfindahl index
                }
                
            except Exception as e:
                self.logger.error(f"Optimal allocation calculation failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def _estimate_portfolio_metrics(self, data: pd.DataFrame, allocations: np.ndarray) -> Tuple[float, float]:
        """Estimate portfolio return and volatility"""
        try:
            if len(self.environment.asset_columns) == 0:
                return 0.0, 0.0
            
            # Calculate historical returns for each asset
            asset_returns = {}
            for asset in self.environment.asset_columns[:len(allocations)]:
                if asset in data.columns:
                    prices = data[asset].dropna()
                    if len(prices) > 1:
                        returns = prices.pct_change().dropna()
                        asset_returns[asset] = returns.mean() if len(returns) > 0 else 0.0
                    else:
                        asset_returns[asset] = 0.0
                else:
                    asset_returns[asset] = 0.0
            
            # Portfolio expected return
            expected_return = sum(
                allocations[i] * asset_returns.get(asset, 0.0)
                for i, asset in enumerate(self.environment.asset_columns[:len(allocations)])
            )
            
            # Simplified volatility estimation (without correlation matrix)
            asset_volatilities = {}
            for asset in self.environment.asset_columns[:len(allocations)]:
                if asset in data.columns:
                    prices = data[asset].dropna()
                    if len(prices) > 1:
                        returns = prices.pct_change().dropna()
                        asset_volatilities[asset] = returns.std() if len(returns) > 0 else 0.0
                    else:
                        asset_volatilities[asset] = 0.0
                else:
                    asset_volatilities[asset] = 0.0
            
            # Portfolio volatility (assuming independence - simplified)
            expected_volatility = np.sqrt(sum(
                (allocations[i] * asset_volatilities.get(asset, 0.0))**2
                for i, asset in enumerate(self.environment.asset_columns[:len(allocations)])
            ))
            
            return float(expected_return), float(expected_volatility)
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics estimation failed: {e}")
            return 0.0, 0.0
    
    def _should_rebalance(self, new_allocations: np.ndarray) -> bool:
        """Determine if portfolio should be rebalanced"""
        if self.current_allocations is None:
            return True
        
        # Calculate allocation drift
        drift = np.abs(new_allocations - self.current_allocations).max()
        return drift > self.config.rebalance_threshold
    
    def _save_model(self):
        """Save trained model"""
        try:
            if self.agent is None:
                return
            
            model_path = self.model_cache_dir / "rl_portfolio_agent.pt"
            
            torch.save({
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
                'config': self.config,
                'training_step': self.agent.training_step
            }, model_path)
            
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
    
    def _load_model(self):
        """Load trained model"""
        try:
            model_path = self.model_cache_dir / "rl_portfolio_agent.pt"
            
            if not model_path.exists():
                self.logger.info("No saved model found")
                return
            
            checkpoint = torch.load(model_path, map_location=self.agent.device)
            
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.agent.training_step = checkpoint.get('training_step', 0)
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive allocation summary"""
        with self._lock:
            return {
                'algorithm': self.config.algorithm.value,
                'max_assets': self.config.max_assets,
                'current_allocations': self.current_allocations.tolist() if self.current_allocations is not None else [],
                'training_episodes': len(self.training_history),
                'last_training_performance': self.training_history[-1] if self.training_history else {},
                'performance_metrics': self.performance_metrics,
                'model_trained': self.agent is not None,
                'config': {
                    'primary_reward': self.config.primary_reward.value,
                    'max_drawdown_threshold': self.config.max_drawdown_threshold,
                    'rebalance_threshold': self.config.rebalance_threshold,
                    'learning_rate': self.config.learning_rate
                }
            }


# Singleton RL portfolio allocator
_rl_portfolio_allocator = None
_rpa_lock = threading.Lock()

def get_rl_portfolio_allocator(config: Optional[RLPortfolioConfig] = None) -> ReinforcementPortfolioAllocator:
    """Get the singleton RL portfolio allocator"""
    global _rl_portfolio_allocator
    
    with _rpa_lock:
        if _rl_portfolio_allocator is None:
            _rl_portfolio_allocator = ReinforcementPortfolioAllocator(config)
        return _rl_portfolio_allocator

def optimize_portfolio_allocation(data: pd.DataFrame) -> Dict[str, Any]:
    """Convenient function to get optimal portfolio allocation"""
    allocator = get_rl_portfolio_allocator()
    return allocator.get_optimal_allocation(data)

def train_portfolio_agent(data: pd.DataFrame, episodes: int = 1000) -> Dict[str, Any]:
    """Convenient function to train portfolio RL agent"""
    allocator = get_rl_portfolio_allocator()
    return allocator.train(data, episodes)