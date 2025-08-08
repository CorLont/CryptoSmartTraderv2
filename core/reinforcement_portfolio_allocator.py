#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Reinforcement Learning Portfolio Allocator
Advanced RL-based dynamic portfolio allocation using PPO and continuous action spaces
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RL and ML imports
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

@dataclass
class PortfolioState:
    """Portfolio state representation"""
    timestamp: datetime
    prices: np.ndarray
    returns: np.ndarray
    volatilities: np.ndarray
    correlations: np.ndarray
    market_features: np.ndarray
    current_allocation: np.ndarray
    portfolio_value: float
    cash_ratio: float

@dataclass
class AllocationAction:
    """Portfolio allocation action"""
    timestamp: datetime
    target_allocation: np.ndarray
    rebalance_amount: np.ndarray
    transaction_cost: float
    confidence: float
    reasoning: str

@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics"""
    timestamp: datetime
    portfolio_value: float
    returns: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_transactions: int
    transaction_costs: float

class PPOActor(nn.Module):
    """PPO Actor Network for continuous portfolio allocation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through actor network"""
        # Get raw allocation logits
        allocation_logits = self.network(state)
        
        # Apply softmax to ensure allocations sum to 1
        allocation_probs = F.softmax(allocation_logits, dim=-1)
        
        return allocation_probs

class PPOCritic(nn.Module):
    """PPO Critic Network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through critic network"""
        return self.network(state)

class TradingEnvironment:
    """Cryptocurrency trading environment for RL training"""
    
    def __init__(self, market_data: pd.DataFrame, initial_capital: float = 100000.0, 
                 transaction_cost: float = 0.001):
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Environment state
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.holdings = {}
        self.allocation_history = []
        self.performance_history = []
        
        # Market features
        self.coin_columns = [col for col in market_data.columns if 'price' in col.lower()]
        self.n_assets = len(self.coin_columns)
        
        # Preprocessors
        self.scaler = StandardScaler()
        self._preprocess_data()
        
        self.logger = logging.getLogger(__name__)
    
    def _preprocess_data(self):
        """Preprocess market data for RL training"""
        # Calculate returns
        for col in self.coin_columns:
            self.market_data[f"{col}_return"] = self.market_data[col].pct_change()
            self.market_data[f"{col}_volatility"] = self.market_data[f"{col}_return"].rolling(24).std()
        
        # Calculate market features
        self.market_data['market_volatility'] = self.market_data[[f"{col}_return" for col in self.coin_columns]].std(axis=1)
        self.market_data['market_momentum'] = self.market_data[[f"{col}_return" for col in self.coin_columns]].mean(axis=1)
        
        # Drop NaN values
        self.market_data = self.market_data.dropna()
        
        # Normalize features
        feature_columns = [col for col in self.market_data.columns if 'return' in col or 'volatility' in col or 'momentum' in col]
        self.market_data[feature_columns] = self.scaler.fit_transform(self.market_data[feature_columns])
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = {coin: 0.0 for coin in self.coin_columns}
        self.allocation_history = []
        self.performance_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        if self.current_step >= len(self.market_data):
            self.current_step = len(self.market_data) - 1
        
        current_data = self.market_data.iloc[self.current_step]
        
        # Price features
        prices = np.array([current_data[col] for col in self.coin_columns])
        
        # Return features
        returns = np.array([current_data[f"{col}_return"] for col in self.coin_columns])
        
        # Volatility features
        volatilities = np.array([current_data[f"{col}_volatility"] for col in self.coin_columns])
        
        # Market features
        market_features = np.array([
            current_data['market_volatility'],
            current_data['market_momentum']
        ])
        
        # Portfolio features
        current_allocation = np.array([self.holdings.get(coin, 0.0) / max(self.portfolio_value, 1.0) 
                                     for coin in self.coin_columns])
        
        cash_ratio = self.cash / max(self.portfolio_value, 1.0)
        
        # Combine all features
        state = np.concatenate([
            prices,
            returns,
            volatilities,
            market_features,
            current_allocation,
            [cash_ratio]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        # Ensure action is valid allocation (sums to 1)
        action = action / (np.sum(action) + 1e-8)
        action = np.clip(action, 0.0, 1.0)
        
        # Calculate current portfolio value
        current_prices = np.array([self.market_data.iloc[self.current_step][col] for col in self.coin_columns])
        
        # Calculate target allocation in dollars
        target_allocation_dollars = action * self.portfolio_value
        
        # Calculate rebalancing needed
        current_allocation_dollars = np.array([self.holdings.get(coin, 0.0) * current_prices[i] 
                                             for i, coin in enumerate(self.coin_columns)])
        
        rebalance_amounts = target_allocation_dollars - current_allocation_dollars
        
        # Calculate transaction costs
        transaction_cost_total = np.sum(np.abs(rebalance_amounts)) * self.transaction_cost
        
        # Execute rebalancing
        for i, coin in enumerate(self.coin_columns):
            if current_prices[i] > 0:
                self.holdings[coin] = target_allocation_dollars[i] / current_prices[i]
        
        # Update cash
        self.cash = max(0.0, self.portfolio_value - np.sum(target_allocation_dollars) - transaction_cost_total)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.market_data):
            new_prices = np.array([self.market_data.iloc[self.current_step][col] for col in self.coin_columns])
            new_portfolio_value = np.sum([self.holdings.get(coin, 0.0) * new_prices[i] 
                                        for i, coin in enumerate(self.coin_columns)]) + self.cash
        else:
            new_portfolio_value = self.portfolio_value
        
        # Calculate reward
        reward = self._calculate_reward(new_portfolio_value, transaction_cost_total, action)
        
        # Update portfolio value
        previous_value = self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Check if done
        done = self.current_step >= len(self.market_data) - 1
        
        # Create info
        info = {
            'portfolio_value': self.portfolio_value,
            'transaction_cost': transaction_cost_total,
            'allocation': action.copy(),
            'return': (new_portfolio_value - previous_value) / previous_value if previous_value > 0 else 0.0
        }
        
        # Store allocation history
        self.allocation_history.append({
            'timestamp': self.market_data.index[self.current_step - 1],
            'allocation': action.copy(),
            'portfolio_value': self.portfolio_value,
            'transaction_cost': transaction_cost_total
        })
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, new_portfolio_value: float, transaction_cost: float, 
                         allocation: np.ndarray) -> float:
        """Calculate reward for the current action"""
        # Portfolio return
        portfolio_return = (new_portfolio_value - self.portfolio_value) / max(self.portfolio_value, 1.0)
        
        # Risk penalty (concentration penalty)
        concentration_penalty = np.sum(allocation ** 2)  # Herfindahl index
        
        # Transaction cost penalty
        cost_penalty = transaction_cost / max(self.portfolio_value, 1.0)
        
        # Combine rewards
        reward = portfolio_return - 0.1 * concentration_penalty - 10.0 * cost_penalty
        
        # Bonus for positive returns
        if portfolio_return > 0:
            reward += 0.1 * portfolio_return
        
        return reward

class PPOPortfolioAgent:
    """PPO-based portfolio allocation agent"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.epsilon = 0.2
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.logger = logging.getLogger(__name__)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """Get action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        if training:
            # Sample from policy during training
            dist = torch.distributions.Categorical(action_probs)
            action_indices = dist.sample()
            log_prob = dist.log_prob(action_indices).sum()
            
            # Convert to allocation weights
            action = action_probs.squeeze().numpy()
        else:
            # Use deterministic policy during evaluation
            action = action_probs.squeeze().numpy()
            log_prob = 0.0
        
        return action, log_prob
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store transition for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_i = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value_i = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value_i * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return advantages, returns
    
    def update_policy(self, n_epochs: int = 10):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        for epoch in range(n_epochs):
            # Forward pass
            action_probs = self.actor(states)
            values = self.critic(states).squeeze()
            
            # Calculate new log probabilities
            # For continuous actions, use MSE loss
            action_loss = F.mse_loss(action_probs, actions)
            
            # Value loss
            value_loss = F.mse_loss(values, returns_tensor)
            
            # Entropy bonus
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()
            
            # Total loss
            total_loss = action_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update actor
            self.actor_optimizer.zero_grad()
            action_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        self.logger.info(f"PPO update completed - Action Loss: {action_loss:.4f}, Value Loss: {value_loss:.4f}")

class ReinforcementPortfolioAllocator:
    """Main RL-based portfolio allocation system"""
    
    def __init__(self, config_path: str = "config/rl_portfolio_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)
        
        # Components
        self.environment = None
        self.agent = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = []
        
        # Load configuration
        self.config = self._load_config()
        
        self.logger.info("Reinforcement Learning Portfolio Allocator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load RL configuration"""
        default_config = {
            "training": {
                "episodes": 1000,
                "learning_rate": 3e-4,
                "update_frequency": 10,
                "batch_size": 64
            },
            "environment": {
                "initial_capital": 100000.0,
                "transaction_cost": 0.001,
                "lookback_window": 24
            },
            "agent": {
                "hidden_dim": 256,
                "epsilon": 0.2,
                "gamma": 0.99,
                "lambda_gae": 0.95
            },
            "risk_management": {
                "max_allocation_per_asset": 0.4,
                "min_allocation_per_asset": 0.02,
                "rebalance_threshold": 0.05
            }
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Could not load config, using defaults: {e}")
        
        return default_config
    
    def setup_environment(self, market_data: pd.DataFrame):
        """Setup trading environment with market data"""
        try:
            self.environment = TradingEnvironment(
                market_data,
                initial_capital=self.config["environment"]["initial_capital"],
                transaction_cost=self.config["environment"]["transaction_cost"]
            )
            
            # Setup agent
            state_dim = len(self.environment._get_state())
            action_dim = self.environment.n_assets
            
            self.agent = PPOPortfolioAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.config["training"]["learning_rate"]
            )
            
            self.logger.info(f"Environment setup: {action_dim} assets, {state_dim} state dimensions")
            
        except Exception as e:
            self.logger.error(f"Error setting up environment: {e}")
            raise
    
    def train_agent(self, n_episodes: int = None) -> Dict[str, Any]:
        """Train the RL agent"""
        if self.environment is None or self.agent is None:
            raise ValueError("Environment and agent must be setup first")
        
        n_episodes = n_episodes or self.config["training"]["episodes"]
        update_frequency = self.config["training"]["update_frequency"]
        
        training_results = {
            "episodes": [],
            "rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": []
        }
        
        try:
            for episode in range(n_episodes):
                state = self.environment.reset()
                episode_reward = 0
                episode_returns = []
                
                while True:
                    # Get action from agent
                    action, log_prob = self.agent.get_action(state, training=True)
                    
                    # Get value estimate
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        value = self.agent.critic(state_tensor).item()
                    
                    # Execute action
                    next_state, reward, done, info = self.environment.step(action)
                    
                    # Store transition
                    self.agent.store_transition(state, action, reward, value, log_prob, done)
                    
                    episode_reward += reward
                    episode_returns.append(info['return'])
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Update policy
                if episode % update_frequency == 0:
                    self.agent.update_policy()
                
                # Calculate episode metrics
                final_portfolio_value = self.environment.portfolio_value
                episode_sharpe = self._calculate_sharpe_ratio(episode_returns)
                
                # Store results
                training_results["episodes"].append(episode)
                training_results["rewards"].append(episode_reward)
                training_results["portfolio_values"].append(final_portfolio_value)
                training_results["sharpe_ratios"].append(episode_sharpe)
                
                # Log progress
                if episode % 100 == 0:
                    self.logger.info(f"Episode {episode}: Reward={episode_reward:.4f}, "
                                   f"Portfolio Value=${final_portfolio_value:.2f}, "
                                   f"Sharpe Ratio={episode_sharpe:.4f}")
            
            self.is_trained = True
            self.training_history = training_results
            
            self.logger.info(f"Training completed: {n_episodes} episodes")
            
            return {
                "success": True,
                "episodes_trained": n_episodes,
                "final_portfolio_value": training_results["portfolio_values"][-1],
                "final_sharpe_ratio": training_results["sharpe_ratios"][-1],
                "average_reward": np.mean(training_results["rewards"][-100:]),  # Last 100 episodes
                "training_results": training_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return {"success": False, "error": str(e)}
    
    def get_optimal_allocation(self, current_state: Dict[str, Any]) -> AllocationAction:
        """Get optimal portfolio allocation for current market state"""
        try:
            if not self.is_trained or self.agent is None:
                # Fallback to equal weight allocation
                n_assets = len(current_state.get('prices', []))
                if n_assets == 0:
                    n_assets = 5  # Default
                
                allocation = np.ones(n_assets) / n_assets
                
                return AllocationAction(
                    timestamp=datetime.now(),
                    target_allocation=allocation,
                    rebalance_amount=np.zeros(n_assets),
                    transaction_cost=0.0,
                    confidence=0.5,
                    reasoning="Equal weight allocation (agent not trained)"
                )
            
            # Prepare state vector
            state_vector = self._prepare_state_vector(current_state)
            
            # Get action from trained agent
            allocation, _ = self.agent.get_action(state_vector, training=False)
            
            # Apply risk management constraints
            allocation = self._apply_risk_constraints(allocation)
            
            # Calculate rebalancing needed
            current_allocation = np.array(current_state.get('current_allocation', allocation))
            rebalance_amount = allocation - current_allocation
            
            # Estimate transaction cost
            transaction_cost = np.sum(np.abs(rebalance_amount)) * self.config["environment"]["transaction_cost"]
            
            # Calculate confidence based on action certainty
            confidence = self._calculate_confidence(allocation)
            
            # Generate reasoning
            reasoning = self._generate_allocation_reasoning(allocation, current_state)
            
            return AllocationAction(
                timestamp=datetime.now(),
                target_allocation=allocation,
                rebalance_amount=rebalance_amount,
                transaction_cost=transaction_cost,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error getting optimal allocation: {e}")
            # Return safe equal weight allocation
            n_assets = 5
            allocation = np.ones(n_assets) / n_assets
            
            return AllocationAction(
                timestamp=datetime.now(),
                target_allocation=allocation,
                rebalance_amount=np.zeros(n_assets),
                transaction_cost=0.0,
                confidence=0.1,
                reasoning=f"Error in allocation calculation: {e}"
            )
    
    def _prepare_state_vector(self, current_state: Dict[str, Any]) -> np.ndarray:
        """Prepare state vector from current market state"""
        # Extract features from current state
        prices = np.array(current_state.get('prices', [1.0] * 5))
        returns = np.array(current_state.get('returns', [0.0] * 5))
        volatilities = np.array(current_state.get('volatilities', [0.1] * 5))
        market_features = np.array(current_state.get('market_features', [0.0, 0.0]))
        current_allocation = np.array(current_state.get('current_allocation', [0.2] * 5))
        cash_ratio = current_state.get('cash_ratio', 0.0)
        
        # Combine all features
        state_vector = np.concatenate([
            prices,
            returns,
            volatilities,
            market_features,
            current_allocation,
            [cash_ratio]
        ])
        
        return state_vector.astype(np.float32)
    
    def _apply_risk_constraints(self, allocation: np.ndarray) -> np.ndarray:
        """Apply risk management constraints to allocation"""
        max_allocation = self.config["risk_management"]["max_allocation_per_asset"]
        min_allocation = self.config["risk_management"]["min_allocation_per_asset"]
        
        # Apply constraints
        allocation = np.clip(allocation, min_allocation, max_allocation)
        
        # Renormalize to sum to 1
        allocation = allocation / np.sum(allocation)
        
        return allocation
    
    def _calculate_confidence(self, allocation: np.ndarray) -> float:
        """Calculate confidence in allocation decision"""
        # Higher concentration = higher confidence in specific assets
        concentration = np.sum(allocation ** 2)  # Herfindahl index
        
        # Convert to confidence score (0 to 1)
        # More concentrated allocations get higher confidence
        confidence = min(0.95, max(0.1, concentration * 2))
        
        return confidence
    
    def _generate_allocation_reasoning(self, allocation: np.ndarray, 
                                     current_state: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for allocation"""
        # Find top allocations
        top_indices = np.argsort(allocation)[-3:][::-1]  # Top 3
        
        reasoning_parts = []
        
        coin_names = current_state.get('coin_names', [f'Asset_{i}' for i in range(len(allocation))])
        
        for i in top_indices:
            percentage = allocation[i] * 100
            if percentage > 5:  # Only mention significant allocations
                coin_name = coin_names[i] if i < len(coin_names) else f'Asset_{i}'
                reasoning_parts.append(f"{coin_name}: {percentage:.1f}%")
        
        if len(reasoning_parts) > 0:
            reasoning = f"Recommended allocation - {', '.join(reasoning_parts)}"
        else:
            reasoning = "Diversified allocation across all assets"
        
        # Add market context
        returns = current_state.get('returns', [])
        if len(returns) > 0:
            avg_return = np.mean(returns)
            if avg_return > 0.02:
                reasoning += " (Bullish market conditions)"
            elif avg_return < -0.02:
                reasoning += " (Bearish market conditions)"
            else:
                reasoning += " (Neutral market conditions)"
        
        return reasoning
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for returns"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe = (mean_return * 365) / (std_return * np.sqrt(365))
        
        return sharpe
    
    def evaluate_performance(self, market_data: pd.DataFrame) -> PortfolioPerformance:
        """Evaluate portfolio performance on historical data"""
        try:
            if not self.is_trained or self.environment is None:
                self.logger.warning("Agent not trained or environment not setup")
                return None
            
            # Reset environment for evaluation
            state = self.environment.reset()
            portfolio_values = [self.environment.initial_capital]
            returns = []
            transaction_costs = []
            allocations = []
            
            while True:
                # Get action (deterministic)
                action, _ = self.agent.get_action(state, training=False)
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store metrics
                portfolio_values.append(info['portfolio_value'])
                returns.append(info['return'])
                transaction_costs.append(info['transaction_cost'])
                allocations.append(info['allocation'])
                
                state = next_state
                
                if done:
                    break
            
            # Calculate performance metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Calculate max drawdown
            portfolio_values_array = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_values_array)
            drawdown = (portfolio_values_array - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Win rate
            positive_returns = len([r for r in returns if r > 0])
            win_rate = positive_returns / len(returns) if len(returns) > 0 else 0.0
            
            performance = PortfolioPerformance(
                timestamp=datetime.now(),
                portfolio_value=portfolio_values[-1],
                returns=total_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=abs(max_drawdown),
                win_rate=win_rate,
                total_transactions=len(transaction_costs),
                transaction_costs=sum(transaction_costs)
            )
            
            self.performance_metrics.append(performance)
            
            self.logger.info(f"Performance evaluation: Return={total_return:.2%}, "
                           f"Sharpe={sharpe_ratio:.4f}, Max DD={abs(max_drawdown):.2%}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance: {e}")
            return None
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get comprehensive allocation summary"""
        try:
            summary = {
                "training_status": self.is_trained,
                "training_episodes": len(self.training_history.get("episodes", [])) if self.training_history else 0,
                "performance_evaluations": len(self.performance_metrics),
                "last_updated": datetime.now().isoformat()
            }
            
            # Add training results if available
            if self.training_history:
                recent_rewards = self.training_history["rewards"][-100:] if self.training_history["rewards"] else []
                recent_values = self.training_history["portfolio_values"][-100:] if self.training_history["portfolio_values"] else []
                
                summary["training_results"] = {
                    "average_recent_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                    "best_portfolio_value": max(recent_values) if recent_values else 0.0,
                    "final_portfolio_value": recent_values[-1] if recent_values else 0.0
                }
            
            # Add latest performance if available
            if self.performance_metrics:
                latest_performance = self.performance_metrics[-1]
                summary["latest_performance"] = asdict(latest_performance)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating allocation summary: {e}")
            return {"error": str(e)}


# Singleton instance
_rl_portfolio_allocator = None

def get_rl_portfolio_allocator() -> ReinforcementPortfolioAllocator:
    """Get or create RL portfolio allocator singleton"""
    global _rl_portfolio_allocator
    if _rl_portfolio_allocator is None:
        _rl_portfolio_allocator = ReinforcementPortfolioAllocator()
    return _rl_portfolio_allocator

def train_rl_allocator(market_data: pd.DataFrame, n_episodes: int = 500) -> Dict[str, Any]:
    """Convenient function to train RL allocator"""
    allocator = get_rl_portfolio_allocator()
    allocator.setup_environment(market_data)
    return allocator.train_agent(n_episodes)

def get_optimal_allocation(current_state: Dict[str, Any]) -> AllocationAction:
    """Convenient function to get optimal allocation"""
    allocator = get_rl_portfolio_allocator()
    return allocator.get_optimal_allocation(current_state)

def evaluate_rl_performance(market_data: pd.DataFrame) -> PortfolioPerformance:
    """Convenient function to evaluate RL performance"""
    allocator = get_rl_portfolio_allocator()
    return allocator.evaluate_performance(market_data)