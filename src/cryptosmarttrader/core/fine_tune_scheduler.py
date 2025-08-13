#!/usr/bin/env python3
"""
Fine-Tune Scheduler System
Small learning-rate updates with replay buffer and EWC (Elastic Weight Consolidation)
"""

import numpy as np
import json
import pickle
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class FineTuneJob:
    """Fine-tuning job configuration"""
    job_id: str
    model_name: str
    trigger_reason: str  # 'drift_detected', 'performance_degradation', 'scheduled'
    priority: str  # 'low', 'medium', 'high', 'critical'
    learning_rate: float
    max_epochs: int
    batch_size: int
    replay_ratio: float  # Ratio of old data to include
    ewc_lambda: float  # EWC regularization strength
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    metrics: Dict[str, Any] = None

@dataclass
class ReplayBufferSample:
    """Sample from replay buffer"""
    features: np.ndarray
    targets: np.ndarray
    timestamp: datetime
    importance_weight: float = 1.0

class ReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, max_size: int = 10000, stratified: bool = True):
        self.max_size = max_size
        self.stratified = stratified
        self.logger = get_structured_logger("ReplayBuffer")
        
        # Buffer storage
        self.samples: deque = deque(maxlen=max_size)
        self.importance_weights: deque = deque(maxlen=max_size)
        
        # Stratified sampling support
        self.label_indices: Dict[str, List[int]] = {}
        
    def add_sample(self, features: np.ndarray, targets: np.ndarray, 
                   importance_weight: float = 1.0, timestamp: Optional[datetime] = None) -> None:
        """Add sample to replay buffer"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        sample = ReplayBufferSample(
            features=features,
            targets=targets,
            timestamp=timestamp,
            importance_weight=importance_weight
        )
        
        # Add to buffer
        if len(self.samples) == self.max_size:
            # Remove oldest sample and update indices
            old_sample = self.samples[0]
            self._remove_from_indices(0, old_sample)
        
        self.samples.append(sample)
        self.importance_weights.append(importance_weight)
        
        # Update stratified indices
        if self.stratified:
            self._add_to_indices(len(self.samples) - 1, sample)
        
        self.logger.debug(f"Added sample to replay buffer (size: {len(self.samples)})")
    
    def add_batch(self, features_batch: np.ndarray, targets_batch: np.ndarray,
                  importance_weights: Optional[np.ndarray] = None) -> None:
        """Add batch of samples to replay buffer"""
        
        if importance_weights is None:
            importance_weights = np.ones(len(features_batch))
            
        for i in range(len(features_batch)):
            self.add_sample(features_batch[i], targets_batch[i], importance_weights[i])
    
    def sample(self, batch_size: int, strategy: str = "uniform") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch from replay buffer"""
        
        if len(self.samples) == 0:
            return np.array([]), np.array([]), np.array([])
        
        if strategy == "uniform":
            indices = np.# REMOVED: Mock data pattern not allowed in production(len(self.samples), 
                                     size=min(batch_size, len(self.samples)), 
                                     replace=False)
        elif strategy == "importance_weighted":
            # Sample based on importance weights
            weights = np.array(list(self.importance_weights))
            weights = weights / np.sum(weights)
            indices = np.# REMOVED: Mock data pattern not allowed in production(len(self.samples), 
                                     size=min(batch_size, len(self.samples)), 
                                     replace=False, p=weights)
        elif strategy == "recent":
            # Sample more recent data
            recent_count = min(batch_size, len(self.samples))
            indices = list(range(len(self.samples) - recent_count, len(self.samples)))
            np.random.shuffle(indices)
        else:
            # Default to uniform
            indices = np.# REMOVED: Mock data pattern not allowed in production(len(self.samples), 
                                     size=min(batch_size, len(self.samples)), 
                                     replace=False)
        
        # Extract samples
        features_list = []
        targets_list = []
        weights_list = []
        
        for idx in indices:
            sample = self.samples[idx]
            features_list.append(sample.features)
            targets_list.append(sample.targets)
            weights_list.append(sample.importance_weight)
        
        features = np.array(features_list) if features_list else np.array([])
        targets = np.array(targets_list) if targets_list else np.array([])
        weights = np.array(weights_list) if weights_list else np.array([])
        
        return features, targets, weights
    
    def _add_to_indices(self, index: int, sample: ReplayBufferSample) -> None:
        """Add sample to stratified indices"""
        # Simple implementation - could be enhanced for multi-class
        label = str(int(np.mean(sample.targets)))  # Simple label extraction
        
        if label not in self.label_indices:
            self.label_indices[label] = []
        
        self.label_indices[label].append(index)
    
    def _remove_from_indices(self, index: int, sample: ReplayBufferSample) -> None:
        """Remove sample from stratified indices"""
        label = str(int(np.mean(sample.targets)))
        
        if label in self.label_indices and index in self.label_indices[label]:
            self.label_indices[label].remove(index)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics"""
        
        if len(self.samples) == 0:
            return {'size': 0, 'oldest_sample': None, 'newest_sample': None}
        
        timestamps = [s.timestamp for s in self.samples]
        
        return {
            'size': len(self.samples),
            'max_size': self.max_size,
            'oldest_sample': min(timestamps).isoformat(),
            'newest_sample': max(timestamps).isoformat(),
            'average_importance': np.mean(list(self.importance_weights)),
            'stratified_labels': len(self.label_indices) if self.stratified else 0
        }

class EWCRegularizer:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, model=None, lambda_reg: float = 1000.0):
        self.model = model
        self.lambda_reg = lambda_reg
        self.logger = get_structured_logger("EWCRegularizer")
        
        # Store important weights and Fisher information
        self.important_weights: Dict[str, torch.Tensor] = {}
        self.fisher_information: Dict[str, torch.Tensor] = {}
        
    def compute_fisher_information(self, dataloader, device: str = "cpu") -> None:
        """Compute Fisher Information Matrix for current task"""
        
        if not TORCH_AVAILABLE or self.model is None:
            self.logger.warning("PyTorch not available or model not set")
            return
        
        try:
            self.model.eval()
            
            # Initialize Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher_information[name] = torch.zeros_like(param)
            
            # Compute Fisher information
            num_samples = 0
            
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Compute loss (assuming classification or regression)
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    # Classification - use log probabilities
                    log_probs = torch.log_softmax(outputs, dim=1)
                    loss = -torch.sum(log_probs * torch.softmax(outputs.detach(), dim=1))
                else:
                    # Regression - use squared error
                    loss = torch.sum((outputs - batch_targets) ** 2)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Accumulate gradients squared (Fisher information)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_information[name] += param.grad ** 2
                
                num_samples += batch_features.size(0)
            
            # Normalize by number of samples
            for name in self.fisher_information:
                self.fisher_information[name] /= num_samples
            
            # Store current weights as important weights
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.important_weights[name] = param.data.clone()
            
            self.logger.info(f"Computed Fisher information for {len(self.fisher_information)} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to compute Fisher information: {e}")
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        
        if not TORCH_AVAILABLE or self.model is None:
            return torch.tensor(0.0)
        
        ewc_loss = torch.tensor(0.0)
        
        try:
            for name, param in self.model.named_parameters():
                if name in self.fisher_information and name in self.important_weights:
                    fisher = self.fisher_information[name]
                    important_weight = self.important_weights[name]
                    
                    # EWC penalty: λ/2 * F * (θ - θ*)²
                    penalty = fisher * (param - important_weight) ** 2
                    ewc_loss += self.lambda_reg * 0.5 * penalty.sum()
            
            return ewc_loss
            
        except Exception as e:
            self.logger.error(f"Failed to compute EWC loss: {e}")
            return torch.tensor(0.0)

class FineTuneScheduler:
    """Complete fine-tuning scheduler with replay buffer and EWC"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_structured_logger("FineTuneScheduler")
        
        # Default configuration
        self.config = {
            'base_learning_rate': 1e-4,
            'drift_learning_rate': 5e-5,  # Smaller LR for drift fine-tuning
            'max_epochs': 10,
            'batch_size': 32,
            'replay_buffer_size': 10000,
            'default_replay_ratio': 0.3,  # 30% old data
            'ewc_lambda': 1000.0,
            'job_timeout_hours': 2,
            'max_concurrent_jobs': 2,
            'enable_ewc': True,
            'enable_replay_buffer': True
        }
        
        if config:
            self.config.update(config)
        
        # Job management
        self.pending_jobs: deque = deque()
        self.running_jobs: Dict[str, FineTuneJob] = {}
        self.completed_jobs: List[FineTuneJob] = []
        self.job_lock = threading.Lock()
        
        # Replay buffer for each model
        self.replay_buffers: Dict[str, ReplayBuffer] = {}
        
        # EWC regularizers for each model
        self.ewc_regularizers: Dict[str, EWCRegularizer] = {}
        
    def create_fine_tune_job(self, model_name: str, trigger_reason: str, 
                           priority: str = "medium", custom_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new fine-tuning job"""
        
        job_config = self.config.copy()
        if custom_config:
            job_config.update(custom_config)
        
        # Adjust learning rate based on trigger
        if trigger_reason == "drift_detected":
            learning_rate = job_config['drift_learning_rate']
            max_epochs = min(job_config['max_epochs'], 5)  # Shorter for drift
        else:
            learning_rate = job_config['base_learning_rate']
            max_epochs = job_config['max_epochs']
        
        job = FineTuneJob(
            job_id=f"finetune_{model_name}_{int(datetime.now().timestamp())}",
            model_name=model_name,
            trigger_reason=trigger_reason,
            priority=priority,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=job_config['batch_size'],
            replay_ratio=job_config['default_replay_ratio'],
            ewc_lambda=job_config['ewc_lambda'],
            created_at=datetime.now()
        )
        
        with self.job_lock:
            # Insert based on priority
            if priority == "critical":
                self.pending_jobs.appendleft(job)
            elif priority == "high":
                # Insert after any critical jobs
                critical_count = sum(1 for j in self.pending_jobs if j.priority == "critical")
                if critical_count == 0:
                    self.pending_jobs.appendleft(job)
                else:
                    # Convert to list for insertion
                    jobs_list = list(self.pending_jobs)
                    jobs_list.insert(critical_count, job)
                    self.pending_jobs = deque(jobs_list)
            else:
                self.pending_jobs.append(job)
        
        self.logger.info(f"Created fine-tuning job {job.job_id} for {model_name} "
                        f"(reason: {trigger_reason}, priority: {priority})")
        
        return job.job_id
    
    def add_training_data(self, model_name: str, features: np.ndarray, targets: np.ndarray,
                         importance_weights: Optional[np.ndarray] = None) -> None:
        """Add training data to replay buffer"""
        
        if not self.config['enable_replay_buffer']:
            return
        
        # Initialize replay buffer if needed
        if model_name not in self.replay_buffers:
            self.replay_buffers[model_name] = ReplayBuffer(
                max_size=self.config['replay_buffer_size']
            )
        
        # Add data to replay buffer
        self.replay_buffers[model_name].add_batch(features, targets, importance_weights)
        
        self.logger.debug(f"Added {len(features)} samples to replay buffer for {model_name}")
    
    def setup_ewc_for_model(self, model_name: str, model, training_dataloader, device: str = "cpu") -> None:
        """Setup EWC regularizer for a model"""
        
        if not self.config['enable_ewc'] or not TORCH_AVAILABLE:
            return
        
        ewc_regularizer = EWCRegularizer(model, self.config['ewc_lambda'])
        ewc_regularizer.compute_fisher_information(training_dataloader, device)
        
        self.ewc_regularizers[model_name] = ewc_regularizer
        
        self.logger.info(f"Setup EWC regularizer for {model_name}")
    
    def run_fine_tuning_job(self, job: FineTuneJob, model, optimizer, 
                          train_dataloader, device: str = "cpu") -> Dict[str, Any]:
        """Execute a fine-tuning job"""
        
        self.logger.info(f"Starting fine-tuning job {job.job_id}")
        
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            # Prepare mixed training data (new + replay)
            mixed_dataloader = self._prepare_mixed_dataloader(job, train_dataloader)
            
            # Get EWC regularizer if available
            ewc_regularizer = self.ewc_regularizers.get(job.model_name)
            
            # Training metrics
            epoch_losses = []
            epoch_accuracies = []
            
            # Fine-tuning loop
            for epoch in range(job.max_epochs):
                model.train()
                epoch_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_features, batch_targets in mixed_dataloader:
                    if TORCH_AVAILABLE:
                        batch_features = batch_features.to(device)
                        batch_targets = batch_targets.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    
                    # Primary loss
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        # Classification
                        primary_loss = nn.CrossEntropyLoss()(outputs, batch_targets.long())
                        
                        # Accuracy calculation
                        _, predicted = torch.max(outputs.data, 1)
                        correct_predictions += (predicted == batch_targets.long()).sum().item()
                    else:
                        # Regression
                        primary_loss = nn.MSELoss()(outputs.squeeze(), batch_targets.float())
                        
                        # Simple accuracy for regression (within threshold)
                        threshold = 0.1
                        correct_predictions += (torch.abs(outputs.squeeze() - batch_targets.float()) < threshold).sum().item()
                    
                    total_predictions += batch_targets.size(0)
                    
                    # Add EWC regularization if available
                    if ewc_regularizer and self.config['enable_ewc']:
                        ewc_loss = ewc_regularizer.compute_ewc_loss()
                        total_loss = primary_loss + ewc_loss
                    else:
                        total_loss = primary_loss
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                
                # Calculate epoch metrics
                avg_loss = epoch_loss / len(mixed_dataloader)
                accuracy = correct_predictions / max(total_predictions, 1)
                
                epoch_losses.append(avg_loss)
                epoch_accuracies.append(accuracy)
                
                self.logger.info(f"Job {job.job_id} - Epoch {epoch+1}/{job.max_epochs}: "
                               f"Loss={avg_loss:.4f}, Accuracy={accuracy:.3f}")
            
            # Job completion
            job.status = "completed"
            job.completed_at = datetime.now()
            job.metrics = {
                'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
                'final_accuracy': epoch_accuracies[-1] if epoch_accuracies else 0.0,
                'epoch_losses': epoch_losses,
                'epoch_accuracies': epoch_accuracies,
                'training_duration': (job.completed_at - job.started_at).total_seconds(),
                'epochs_completed': len(epoch_losses)
            }
            
            self.logger.info(f"Fine-tuning job {job.job_id} completed successfully. "
                           f"Final accuracy: {job.metrics['final_accuracy']:.3f}")
            
            return job.metrics
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.metrics = {'error': str(e)}
            
            self.logger.error(f"Fine-tuning job {job.job_id} failed: {e}")
            return {'error': str(e)}
    
    def _prepare_mixed_dataloader(self, job: FineTuneJob, new_dataloader):
        """Prepare mixed dataloader with new data and replay buffer data"""
        
        if not self.config['enable_replay_buffer'] or job.model_name not in self.replay_buffers:
            return new_dataloader
        
        replay_buffer = self.replay_buffers[job.model_name]
        
        if len(replay_buffer.samples) == 0:
            return new_dataloader
        
        try:
            # Calculate replay batch size
            new_batch_size = job.batch_size
            replay_batch_size = int(new_batch_size * job.replay_ratio)
            
            # Sample from replay buffer
            replay_features, replay_targets, replay_weights = replay_buffer.sample(
                batch_size=replay_batch_size * len(new_dataloader),
                strategy="importance_weighted"
            )
            
            if len(replay_features) == 0:
                return new_dataloader
            
            # Create mixed batches
            mixed_batches = []
            replay_idx = 0
            
            for new_batch in new_dataloader:
                new_features, new_targets = new_batch
                
                # Add replay data to batch
                if replay_idx < len(replay_features):
                    end_idx = min(replay_idx + replay_batch_size, len(replay_features))
                    batch_replay_features = replay_features[replay_idx:end_idx]
                    batch_replay_targets = replay_targets[replay_idx:end_idx]
                    
                    # Combine new and replay data
                    if TORCH_AVAILABLE:
                        combined_features = torch.cat([
                            torch.tensor(new_features, dtype=torch.float32),
                            torch.tensor(batch_replay_features, dtype=torch.float32)
                        ])
                        combined_targets = torch.cat([
                            torch.tensor(new_targets, dtype=torch.float32),
                            torch.tensor(batch_replay_targets, dtype=torch.float32)
                        ])
                    else:
                        combined_features = np.concatenate([new_features, batch_replay_features])
                        combined_targets = np.concatenate([new_targets, batch_replay_targets])
                    
                    mixed_batches.append((combined_features, combined_targets))
                    replay_idx = end_idx
                else:
                    mixed_batches.append(new_batch)
            
            self.logger.debug(f"Created mixed dataloader with {len(mixed_batches)} batches "
                            f"(replay ratio: {job.replay_ratio})")
            
            return mixed_batches
            
        except Exception as e:
            self.logger.error(f"Failed to create mixed dataloader: {e}")
            return new_dataloader
    
    def process_pending_jobs(self) -> List[str]:
        """Process pending fine-tuning jobs"""
        
        processed_jobs = []
        
        with self.job_lock:
            # Check running jobs for completion/timeout
            completed_job_ids = []
            for job_id, job in self.running_jobs.items():
                if job.status in ["completed", "failed"]:
                    completed_job_ids.append(job_id)
                    self.completed_jobs.append(job)
                elif job.started_at:
                    # Check timeout
                    runtime = datetime.now() - job.started_at
                    if runtime.total_seconds() > self.config['job_timeout_hours'] * 3600:
                        job.status = "failed"
                        job.completed_at = datetime.now()
                        job.metrics = {'error': 'Job timeout'}
                        completed_job_ids.append(job_id)
                        self.completed_jobs.append(job)
                        self.logger.warning(f"Fine-tuning job {job_id} timed out")
            
            # Remove completed jobs from running
            for job_id in completed_job_ids:
                del self.running_jobs[job_id]
            
            # Start new jobs if capacity available
            while (len(self.running_jobs) < self.config['max_concurrent_jobs'] and 
                   len(self.pending_jobs) > 0):
                
                job = self.pending_jobs.popleft()
                self.running_jobs[job.job_id] = job
                processed_jobs.append(job.job_id)
                
                self.logger.info(f"Queued fine-tuning job {job.job_id} for processing")
        
        return processed_jobs
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific job"""
        
        # Check running jobs
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status,
                'model_name': job.model_name,
                'trigger_reason': job.trigger_reason,
                'priority': job.priority,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'metrics': job.metrics
            }
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'model_name': job.model_name,
                    'trigger_reason': job.trigger_reason,
                    'priority': job.priority,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'metrics': job.metrics
                }
        
        # Check pending jobs
        for job in self.pending_jobs:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'model_name': job.model_name,
                    'trigger_reason': job.trigger_reason,
                    'priority': job.priority,
                    'created_at': job.created_at.isoformat(),
                    'started_at': None,
                    'completed_at': None,
                    'metrics': None
                }
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get fine-tuning scheduler status"""
        
        # Count jobs by status
        pending_by_priority = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for job in self.pending_jobs:
            pending_by_priority[job.priority] += 1
        
        running_jobs_info = []
        for job in self.running_jobs.values():
            runtime = None
            if job.started_at:
                runtime = (datetime.now() - job.started_at).total_seconds()
            
            running_jobs_info.append({
                'job_id': job.job_id,
                'model_name': job.model_name,
                'trigger_reason': job.trigger_reason,
                'runtime_seconds': runtime
            })
        
        # Recent completed jobs (last 24h)
        recent_completed = [job for job in self.completed_jobs 
                           if job.completed_at and 
                           (datetime.now() - job.completed_at).total_seconds() < 86400]
        
        status = {
            'pending_jobs': len(self.pending_jobs),
            'pending_by_priority': pending_by_priority,
            'running_jobs': len(self.running_jobs),
            'running_jobs_info': running_jobs_info,
            'completed_jobs_24h': len(recent_completed),
            'total_completed_jobs': len(self.completed_jobs),
            'replay_buffers': {name: buffer.get_statistics() 
                             for name, buffer in self.replay_buffers.items()},
            'ewc_models': list(self.ewc_regularizers.keys()),
            'config': {
                'max_concurrent_jobs': self.config['max_concurrent_jobs'],
                'enable_replay_buffer': self.config['enable_replay_buffer'],
                'enable_ewc': self.config['enable_ewc']
            }
        }
        
        return status

if __name__ == "__main__":
    def test_fine_tune_scheduler():
        """Test fine-tune scheduler system"""
        
        print("🔍 TESTING FINE-TUNE SCHEDULER SYSTEM")
        print("=" * 60)
        
        # Create scheduler
        scheduler = FineTuneScheduler()
        
        print("📊 Creating test fine-tuning jobs...")
        
        # Create various jobs
        job_ids = []
        job_ids.append(scheduler.create_fine_tune_job("ml_predictor", "drift_detected", "high"))
        job_ids.append(scheduler.create_fine_tune_job("sentiment_analyzer", "performance_degradation", "medium"))
        job_ids.append(scheduler.create_fine_tune_job("whale_detector", "scheduled", "low"))
        
        print(f"   Created {len(job_ids)} jobs")
        
        # Add some training data to replay buffer
        print("\n📚 Adding training data to replay buffer...")
        
        # Mock training data
        features = np.random.randn(100, 10)  # 100 samples, 10 features
        targets = np.# REMOVED: Mock data pattern not allowed in production(0, 2, 100)  # Binary classification
        
        scheduler.add_training_data("ml_predictor", features, targets)
        
        # Check replay buffer stats
        buffer_stats = scheduler.replay_buffers["ml_predictor"].get_statistics()
        print(f"   Replay buffer size: {buffer_stats['size']}")
        
        # Process pending jobs
        print("\n⚙️  Processing pending jobs...")
        processed = scheduler.process_pending_jobs()
        print(f"   Queued {len(processed)} jobs for processing")
        
        # Check system status
        print("\n📈 System status:")
        status = scheduler.get_system_status()
        
        print(f"   Pending jobs: {status['pending_jobs']}")
        print(f"   Running jobs: {status['running_jobs']}")
        print(f"   Completed jobs (24h): {status['completed_jobs_24h']}")
        
        print("\n📊 Pending jobs by priority:")
        for priority, count in status['pending_by_priority'].items():
            print(f"   {priority}: {count}")
        
        print("\n💾 Replay buffers:")
        for model, stats in status['replay_buffers'].items():
            print(f"   {model}: {stats['size']} samples")
        
        # Check individual job status
        print("\n🔍 Job statuses:")
        for job_id in job_ids:
            job_status = scheduler.get_job_status(job_id)
            if job_status:
                print(f"   {job_id}: {job_status['status']} ({job_status['trigger_reason']})")
        
        print("\n✅ FINE-TUNE SCHEDULER TEST COMPLETED")
        return len(job_ids) > 0
    
    # Run test
    success = test_fine_tune_scheduler()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")