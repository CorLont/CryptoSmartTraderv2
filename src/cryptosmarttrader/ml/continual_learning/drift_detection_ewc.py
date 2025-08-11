#!/usr/bin/env python3
"""
Continual Learning with Drift Detection and Elastic Weight Consolidation (EWC)
Maintains stable model performance over time while adapting to new market conditions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import pickle
import json

from core.structured_logger import get_structured_logger

class DataDriftDetector:
    """Detects distribution drift in input features and target variables"""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.1):
        self.logger = get_structured_logger("DriftDetector")
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Reference statistics
        self.reference_stats = {}
        self.drift_history = []
        
    def fit_reference(self, X: np.ndarray, y: np.ndarray = None):
        """Fit reference distribution statistics"""
        
        self.logger.info(f"Fitting reference statistics on {len(X)} samples")
        
        try:
            # Feature statistics
            self.reference_stats['feature_means'] = np.mean(X, axis=0)
            self.reference_stats['feature_stds'] = np.std(X, axis=0)
            self.reference_stats['feature_mins'] = np.min(X, axis=0)
            self.reference_stats['feature_maxs'] = np.max(X, axis=0)
            
            # Target statistics (if provided)
            if y is not None:
                self.reference_stats['target_mean'] = np.mean(y)
                self.reference_stats['target_std'] = np.std(y)
                self.reference_stats['target_min'] = np.min(y)
                self.reference_stats['target_max'] = np.max(y)
            
            # Correlation structure
            if X.shape[1] > 1:
                correlation_matrix = np.corrcoef(X.T)
                self.reference_stats['correlation_matrix'] = correlation_matrix
            
            self.logger.info("Reference statistics fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Reference fitting failed: {e}")
            raise
    
    def detect_drift(self, X_new: np.ndarray, y_new: np.ndarray = None) -> Dict[str, Any]:
        """Detect drift in new data compared to reference"""
        
        try:
            if not self.reference_stats:
                raise ValueError("Reference statistics not fitted. Call fit_reference first.")
            
            drift_results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'feature_drifts': {},
                'target_drift': None,
                'drift_timestamp': datetime.utcnow().isoformat()
            }
            
            # Feature drift detection
            feature_drift_scores = []
            
            for i, feature_name in enumerate([f'feature_{i}' for i in range(X_new.shape[1])]):
                # Calculate distribution shift
                ref_mean = self.reference_stats['feature_means'][i]
                ref_std = self.reference_stats['feature_stds'][i]
                
                new_mean = np.mean(X_new[:, i])
                new_std = np.std(X_new[:, i])
                
                # Normalized difference in means
                mean_shift = abs(new_mean - ref_mean) / (ref_std + 1e-8)
                
                # Ratio of standard deviations
                std_ratio = max(new_std, ref_std) / (min(new_std, ref_std) + 1e-8)
                
                # Combined drift score
                feature_drift = (mean_shift + np.log(std_ratio)) / 2
                
                drift_results['feature_drifts'][feature_name] = {
                    'drift_score': feature_drift,
                    'mean_shift': mean_shift,
                    'std_ratio': std_ratio,
                    'ref_mean': ref_mean,
                    'new_mean': new_mean
                }
                
                feature_drift_scores.append(feature_drift)
            
            # Overall feature drift
            avg_feature_drift = np.mean(feature_drift_scores)
            max_feature_drift = np.max(feature_drift_scores)
            
            # Target drift (if available)
            target_drift_score = 0.0
            if y_new is not None and 'target_mean' in self.reference_stats:
                ref_target_mean = self.reference_stats['target_mean']
                ref_target_std = self.reference_stats['target_std']
                
                new_target_mean = np.mean(y_new)
                new_target_std = np.std(y_new)
                
                target_mean_shift = abs(new_target_mean - ref_target_mean) / (ref_target_std + 1e-8)
                target_std_ratio = max(new_target_std, ref_target_std) / (min(new_target_std, ref_target_std) + 1e-8)
                
                target_drift_score = (target_mean_shift + np.log(target_std_ratio)) / 2
                
                drift_results['target_drift'] = {
                    'drift_score': target_drift_score,
                    'mean_shift': target_mean_shift,
                    'std_ratio': target_std_ratio
                }
            
            # Overall drift decision
            overall_drift_score = max(avg_feature_drift, target_drift_score)
            drift_results['drift_score'] = overall_drift_score
            drift_results['drift_detected'] = overall_drift_score > self.drift_threshold
            
            # Store in history
            self.drift_history.append({
                'timestamp': datetime.utcnow(),
                'drift_score': overall_drift_score,
                'drift_detected': drift_results['drift_detected']
            })
            
            # Maintain history size
            if len(self.drift_history) > 100:
                self.drift_history = self.drift_history[-100:]
            
            self.logger.info(f"Drift detection: {'DETECTED' if drift_results['drift_detected'] else 'NONE'} "
                           f"(score: {overall_drift_score:.3f})")
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'error': str(e),
                'drift_timestamp': datetime.utcnow().isoformat()
            }

class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for continual learning without catastrophic forgetting"""
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.logger = get_structured_logger("EWC")
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Store previous task parameters and Fisher information
        self.previous_params = {}
        self.fisher_information = {}
        self.task_count = 0
        
    def calculate_fisher_information(self, data_loader, device: str = 'cpu'):
        """Calculate Fisher Information Matrix for current task"""
        
        self.logger.info("Calculating Fisher Information Matrix")
        
        try:
            # Initialize Fisher information
            fisher = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    fisher[name] = torch.zeros_like(param.data)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Calculate Fisher information
            num_samples = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = self.model(data)
                loss = nn.MSELoss()(output.squeeze(), target)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                
                num_samples += len(data)
                
                # Limit samples for efficiency
                if num_samples >= 1000:
                    break
            
            # Normalize Fisher information
            for name in fisher:
                fisher[name] /= num_samples
            
            self.fisher_information = fisher
            self.logger.info(f"Fisher Information calculated from {num_samples} samples")
            
        except Exception as e:
            self.logger.error(f"Fisher Information calculation failed: {e}")
            raise
    
    def store_previous_params(self):
        """Store current model parameters as previous task parameters"""
        
        self.previous_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.previous_params[name] = param.data.clone()
        
        self.task_count += 1
        self.logger.info(f"Stored parameters for task {self.task_count}")
    
    def ewc_loss(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Calculate EWC loss to prevent catastrophic forgetting"""
        
        if not self.previous_params or not self.fisher_information:
            return current_loss
        
        try:
            ewc_penalty = 0.0
            
            for name, param in self.model.named_parameters():
                if name in self.previous_params and name in self.fisher_information:
                    # EWC penalty: F * (θ - θ*)^2
                    penalty = self.fisher_information[name] * (param - self.previous_params[name]) ** 2
                    ewc_penalty += penalty.sum()
            
            total_loss = current_loss + (self.ewc_lambda / 2) * ewc_penalty
            return total_loss
            
        except Exception as e:
            self.logger.error(f"EWC loss calculation failed: {e}")
            return current_loss
    
    def save_ewc_data(self, filepath: str):
        """Save EWC data for persistence"""
        
        try:
            ewc_data = {
                'previous_params': {name: param.cpu().numpy() for name, param in self.previous_params.items()},
                'fisher_information': {name: fisher.cpu().numpy() for name, fisher in self.fisher_information.items()},
                'task_count': self.task_count,
                'ewc_lambda': self.ewc_lambda
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(ewc_data, f)
            
            self.logger.info(f"EWC data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"EWC data saving failed: {e}")
    
    def load_ewc_data(self, filepath: str):
        """Load EWC data from file"""
        
        try:
            with open(filepath, 'rb') as f:
                ewc_data = pickle.load(f)
            
            # Convert back to tensors
            self.previous_params = {name: torch.tensor(param) for name, param in ewc_data['previous_params'].items()}
            self.fisher_information = {name: torch.tensor(fisher) for name, fisher in ewc_data['fisher_information'].items()}
            self.task_count = ewc_data['task_count']
            self.ewc_lambda = ewc_data['ewc_lambda']
            
            self.logger.info(f"EWC data loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"EWC data loading failed: {e}")

class ReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, buffer_size: int = 10000, sampling_strategy: str = 'random'):
        self.logger = get_structured_logger("ReplayBuffer")
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        
        self.buffer = deque(maxlen=buffer_size)
        self.task_boundaries = []
        
    def add_samples(self, X: np.ndarray, y: np.ndarray, task_id: int = 0):
        """Add samples to replay buffer"""
        
        for i in range(len(X)):
            sample = {
                'features': X[i],
                'target': y[i],
                'task_id': task_id,
                'timestamp': datetime.utcnow()
            }
            self.buffer.append(sample)
        
        self.logger.info(f"Added {len(X)} samples to replay buffer (total: {len(self.buffer)})")
    
    def sample_batch(self, batch_size: int, current_task_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Sample batch from replay buffer"""
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.sampling_strategy == 'random':
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
        elif self.sampling_strategy == 'balanced':
            # Balance between current and previous tasks
            current_task_samples = int(batch_size * current_task_ratio)
            previous_task_samples = batch_size - current_task_samples
            
            # Get current task samples (most recent)
            recent_samples = list(self.buffer)[-1000:] if len(self.buffer) > 1000 else list(self.buffer)
            current_indices = np.random.choice(len(recent_samples), 
                                             min(current_task_samples, len(recent_samples)), 
                                             replace=False)
            current_samples = [recent_samples[i] for i in current_indices]
            
            # Get previous task samples
            if len(self.buffer) > len(recent_samples):
                old_samples = list(self.buffer)[:-1000]
                old_indices = np.random.choice(len(old_samples),
                                             min(previous_task_samples, len(old_samples)),
                                             replace=False)
                old_samples_selected = [old_samples[i] for i in old_indices]
                samples = current_samples + old_samples_selected
            else:
                samples = current_samples
        else:
            # Default to random
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
        
        # Convert to arrays
        X_batch = np.array([sample['features'] for sample in samples])
        y_batch = np.array([sample['target'] for sample in samples])
        
        return X_batch, y_batch

class ContinualLearningManager:
    """Main manager for continual learning with drift detection and EWC"""
    
    def __init__(self, model: nn.Module, drift_threshold: float = 0.1, 
                 ewc_lambda: float = 1000.0, replay_buffer_size: int = 10000):
        
        self.logger = get_structured_logger("ContinualLearningManager")
        
        # Initialize components
        self.drift_detector = DataDriftDetector(drift_threshold=drift_threshold)
        self.ewc = ElasticWeightConsolidation(model, ewc_lambda=ewc_lambda)
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size)
        
        self.model = model
        self.device = next(model.parameters()).device
        
        # Training state
        self.is_initialized = False
        self.last_drift_check = None
        self.retrain_triggered = False
        
    def initialize_system(self, X_init: np.ndarray, y_init: np.ndarray):
        """Initialize the continual learning system with initial data"""
        
        self.logger.info("Initializing continual learning system")
        
        try:
            # Fit reference distribution
            self.drift_detector.fit_reference(X_init, y_init)
            
            # Add initial samples to replay buffer
            self.replay_buffer.add_samples(X_init, y_init, task_id=0)
            
            # Train initial model (simplified)
            self._train_model_on_data(X_init, y_init)
            
            # Store initial parameters for EWC
            self.ewc.store_previous_params()
            
            self.is_initialized = True
            self.logger.info("Continual learning system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def process_new_data(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Process new data with drift detection and potential retraining"""
        
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize_system first.")
        
        try:
            self.logger.info(f"Processing {len(X_new)} new samples")
            
            # Detect drift
            drift_results = self.drift_detector.detect_drift(X_new, y_new)
            
            # Add new samples to replay buffer
            current_task_id = self.ewc.task_count
            self.replay_buffer.add_samples(X_new, y_new, task_id=current_task_id)
            
            # Decide on retraining
            retrain_decision = self._should_retrain(drift_results, X_new, y_new)
            
            processing_results = {
                'drift_detected': drift_results['drift_detected'],
                'drift_score': drift_results['drift_score'],
                'retrain_triggered': retrain_decision['retrain'],
                'retrain_reason': retrain_decision['reason'],
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            # Retrain if necessary
            if retrain_decision['retrain']:
                retraining_results = self._perform_continual_retraining(X_new, y_new)
                processing_results['retraining_results'] = retraining_results
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"New data processing failed: {e}")
            return {
                'error': str(e),
                'processing_timestamp': datetime.utcnow().isoformat()
            }
    
    def _should_retrain(self, drift_results: Dict[str, Any], 
                       X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Decide whether retraining is necessary"""
        
        # Criterion 1: Significant drift detected
        if drift_results['drift_detected']:
            return {'retrain': True, 'reason': 'significant_drift_detected'}
        
        # Criterion 2: Performance degradation (simplified check)
        # In practice, this would involve model performance evaluation
        current_performance = self._evaluate_current_performance(X_new, y_new)
        if current_performance < 0.6:  # Example threshold
            return {'retrain': True, 'reason': 'performance_degradation'}
        
        # Criterion 3: Time-based retraining
        if self.last_drift_check is None:
            self.last_drift_check = datetime.utcnow()
        
        time_since_last = datetime.utcnow() - self.last_drift_check
        if time_since_last > timedelta(days=7):  # Weekly retraining
            return {'retrain': True, 'reason': 'scheduled_retraining'}
        
        return {'retrain': False, 'reason': 'no_retraining_needed'}
    
    def _perform_continual_retraining(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Perform continual learning retraining with EWC"""
        
        try:
            self.logger.info("Starting continual retraining with EWC")
            
            # Calculate Fisher information for current task
            # (This would need a proper DataLoader in practice)
            
            # Sample from replay buffer
            X_replay, y_replay = self.replay_buffer.sample_batch(
                batch_size=min(1000, len(self.replay_buffer.buffer))
            )
            
            # Combine new data with replay data
            X_combined = np.vstack([X_new, X_replay])
            y_combined = np.hstack([y_new, y_replay])
            
            # Train with EWC loss
            training_loss = self._train_model_with_ewc(X_combined, y_combined)
            
            # Update EWC parameters for next task
            self.ewc.store_previous_params()
            
            self.last_drift_check = datetime.utcnow()
            
            return {
                'training_loss': training_loss,
                'samples_used': len(X_combined),
                'replay_samples': len(X_replay),
                'new_samples': len(X_new),
                'task_id': self.ewc.task_count
            }
            
        except Exception as e:
            self.logger.error(f"Continual retraining failed: {e}")
            return {'error': str(e)}
    
    def _train_model_on_data(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train model on given data (simplified)"""
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
    
    def _train_model_with_ewc(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> float:
        """Train model with EWC regularization"""
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            base_loss = criterion(outputs.squeeze(), y_tensor)
            
            # Add EWC penalty
            total_loss_with_ewc = self.ewc.ewc_loss(base_loss)
            
            total_loss_with_ewc.backward()
            optimizer.step()
            
            total_loss += total_loss_with_ewc.item()
        
        return total_loss / epochs
    
    def _evaluate_current_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate current model performance (simplified)"""
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            predictions = self.model(X_tensor).squeeze()
            mse = torch.mean((predictions - y_tensor) ** 2)
            
            # Convert to performance score (higher is better)
            performance = 1.0 / (1.0 + mse.item())
            return performance
    
    def save_state(self, filepath: str):
        """Save complete continual learning state"""
        
        try:
            # Save EWC data
            ewc_filepath = filepath.replace('.pkl', '_ewc.pkl')
            self.ewc.save_ewc_data(ewc_filepath)
            
            # Save other state
            state = {
                'drift_history': self.drift_detector.drift_history,
                'reference_stats': self.drift_detector.reference_stats,
                'is_initialized': self.is_initialized,
                'last_drift_check': self.last_drift_check.isoformat() if self.last_drift_check else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Continual learning state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"State saving failed: {e}")
    
    def load_state(self, filepath: str):
        """Load continual learning state"""
        
        try:
            # Load EWC data
            ewc_filepath = filepath.replace('.pkl', '_ewc.pkl')
            self.ewc.load_ewc_data(ewc_filepath)
            
            # Load other state
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.drift_detector.drift_history = state['drift_history']
            self.drift_detector.reference_stats = state['reference_stats']
            self.is_initialized = state['is_initialized']
            
            if state['last_drift_check']:
                self.last_drift_check = datetime.fromisoformat(state['last_drift_check'])
            
            self.logger.info(f"Continual learning state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"State loading failed: {e}")

# Example usage
def example_continual_learning():
    """Example of continual learning system usage"""
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    
    # Initialize continual learning manager
    cl_manager = ContinualLearningManager(model)
    
    # Generate initial data
    X_init = np.random.randn(1000, 10)
    y_init = np.random.randn(1000)
    
    # Initialize system
    cl_manager.initialize_system(X_init, y_init)
    
    # Simulate new data with drift
    X_new = np.random.randn(100, 10) + 0.5  # Shifted distribution
    y_new = np.random.randn(100) + 0.3
    
    # Process new data
    results = cl_manager.process_new_data(X_new, y_new)
    
    return results

if __name__ == "__main__":
    results = example_continual_learning()
    print("Continual learning results:")
    for key, value in results.items():
        print(f"{key}: {value}")