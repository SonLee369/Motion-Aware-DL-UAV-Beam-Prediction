"""
Custom samplers for balanced training
"""

import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SpeedStratifiedSampler(Sampler):
    """Sampler for balanced speed category representation"""
    
    def __init__(
        self,
        speed_labels: np.ndarray,
        target_distribution: Optional[Dict[int, float]] = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize stratified sampler
        
        Args:
            speed_labels: [N] array of speed categories (0, 1, 2)
            target_distribution: Target ratio for each category
                Default: {0: 0.33, 1: 0.33, 2: 0.34}
            shuffle: Whether to shuffle within each category
            seed: Random seed
        """
        self.speed_labels = speed_labels
        self.num_samples = len(speed_labels)
        self.shuffle = shuffle
        
        if target_distribution is None:
            target_distribution = {0: 0.33, 1: 0.33, 2: 0.34}
        
        self.target_dist = target_distribution
        
        # Group indices by speed category
        self.speed_indices = {
            speed: np.where(speed_labels == speed)[0].tolist()
            for speed in [0, 1, 2]
        }
        
        # Log distribution
        logger.info("Speed category distribution:")
        for speed in [0, 1, 2]:
            count = len(self.speed_indices[speed])
            logger.info(f"  Category {speed}: {count} samples ({count/self.num_samples*100:.1f}%)")
        
        # Compute sampling weights
        self.weights = self._compute_weights()
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
    
    def _compute_weights(self) -> np.ndarray:
        """Compute per-sample weights for balanced sampling"""
        weights = np.zeros(self.num_samples)
        
        for speed in [0, 1, 2]:
            indices = self.speed_indices[speed]
            current_ratio = len(indices) / self.num_samples
            target_ratio = self.target_dist[speed]
            
            # Weight inversely proportional to current frequency
            if current_ratio > 0:
                weight = target_ratio / current_ratio
            else:
                weight = 0
            
            weights[indices] = weight
        
        return weights
    
    def __iter__(self):
        """Generate sample indices"""
        # Sample with replacement according to weights
        indices = self.rng.choice(
            self.num_samples,
            size=self.num_samples,
            replace=True,
            p=self.weights / self.weights.sum()
        )
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has balanced speed categories
    """
    
    def __init__(
        self,
        speed_labels: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize balanced batch sampler
        
        Args:
            speed_labels: [N] array of speed categories
            batch_size: Batch size (should be divisible by 3)
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.speed_labels = speed_labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        if batch_size % 3 != 0:
            logger.warning(f"batch_size={batch_size} not divisible by 3, may have unbalanced batches")
        
        # Samples per category per batch
        self.samples_per_category = batch_size // 3
        
        # Group indices by speed
        self.speed_indices = {
            speed: np.where(speed_labels == speed)[0].tolist()
            for speed in [0, 1, 2]
        }
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
    
    def __iter__(self):
        """Generate batches"""
        # Shuffle indices within each category
        if self.shuffle:
            for speed in [0, 1, 2]:
                self.rng.shuffle(self.speed_indices[speed])
        
        # Create balanced batches
        batches = []
        
        # Determine number of batches
        min_samples = min(len(self.speed_indices[s]) for s in [0, 1, 2])
        num_batches = min_samples // self.samples_per_category
        
        for batch_idx in range(num_batches):
            batch = []
            
            for speed in [0, 1, 2]:
                start_idx = batch_idx * self.samples_per_category
                end_idx = start_idx + self.samples_per_category
                batch.extend(self.speed_indices[speed][start_idx:end_idx])
            
            # Shuffle within batch
            if self.shuffle:
                self.rng.shuffle(batch)
            
            batches.append(batch)
        
        # Handle remaining samples
        if not self.drop_last:
            remaining = []
            for speed in [0, 1, 2]:
                start_idx = num_batches * self.samples_per_category
                remaining.extend(self.speed_indices[speed][start_idx:])
            
            if len(remaining) > 0:
                if self.shuffle:
                    self.rng.shuffle(remaining)
                batches.append(remaining)
        
        # Flatten and return
        for batch in batches:
            yield batch
    
    def __len__(self):
        """Number of batches"""
        min_samples = min(len(self.speed_indices[s]) for s in [0, 1, 2])
        num_batches = min_samples // self.samples_per_category
        
        if not self.drop_last:
            # Check if there are remaining samples
            total_used = num_batches * self.samples_per_category * 3
            if total_used < len(self.speed_labels):
                num_batches += 1
        
        return num_batches


class WeightedRandomSampler(Sampler):
    """
    Weighted random sampler with custom weights
    """
    
    def __init__(
        self,
        weights: np.ndarray,
        num_samples: int,
        replacement: bool = True,
        seed: int = 42
    ):
        """
        Initialize weighted sampler
        
        Args:
            weights: [N] sampling weights
            num_samples: Number of samples to draw
            replacement: Sample with replacement
            seed: Random seed
        """
        self.weights = weights / weights.sum()  # Normalize
        self.num_samples = num_samples
        self.replacement = replacement
        self.rng = np.random.RandomState(seed)
    
    def __iter__(self):
        """Generate samples"""
        indices = self.rng.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class StratifiedNoReplaceSampler(Sampler):
    """
    Stratified sampling WITHOUT replacement.

    Epoch size = 3 × n_minority — each class contributes exactly n_minority
    unique samples per epoch.  No sample is ever repeated within an epoch,
    eliminating the memorisation caused by heavy oversampling with replacement.
    """

    def __init__(self, speed_labels: np.ndarray,
                 shuffle: bool = True, seed: int = 42):
        self.speed_labels = speed_labels
        self.shuffle = shuffle
        self.speed_indices = {
            speed: np.where(speed_labels == speed)[0]
            for speed in [0, 1, 2]
        }
        counts = {s: len(v) for s, v in self.speed_indices.items()}
        self.n_per_class = min(counts.values())
        self.epoch_size = 3 * self.n_per_class
        self.rng = np.random.RandomState(seed)

        logger.info("StratifiedNoReplaceSampler initialised:")
        for speed in [0, 1, 2]:
            n = counts[speed]
            logger.info(f"  Category {speed}: {n} available, "
                        f"using {self.n_per_class} per epoch")
        logger.info(f"  Epoch size: {self.epoch_size} "
                    f"(original dataset: {len(speed_labels)})")

    def __iter__(self):
        indices = []
        for speed in [0, 1, 2]:
            pool = self.speed_indices[speed]
            chosen = self.rng.choice(pool, size=self.n_per_class, replace=False)
            indices.extend(chosen.tolist())
        if self.shuffle:
            self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.epoch_size


# Test code
if __name__ == "__main__":
    print("Testing Samplers\n" + "="*50)
    
    # Create dummy speed labels (imbalanced)
    speed_labels = np.concatenate([
        np.zeros(1000),      # Slow: 1000 samples
        np.ones(500),        # Medium: 500 samples
        np.full(200, 2)      # Fast: 200 samples
    ])
    
    print(f"Total samples: {len(speed_labels)}")
    print(f"Distribution: Slow={1000}, Medium={500}, Fast={200}")
    
    # Test stratified sampler
    print("\n1. Testing SpeedStratifiedSampler")
    sampler = SpeedStratifiedSampler(speed_labels, shuffle=True, seed=42)
    
    # Sample and check distribution
    sampled_indices = list(sampler)
    sampled_labels = speed_labels[sampled_indices]
    
    unique, counts = np.unique(sampled_labels, return_counts=True)
    print("Sampled distribution:")
    for label, count in zip(unique, counts):
        print(f"  Category {int(label)}: {count} ({count/len(sampled_labels)*100:.1f}%)")
    
    # Test balanced batch sampler
    print("\n2. Testing BalancedBatchSampler")
    batch_sampler = BalancedBatchSampler(
        speed_labels,
        batch_size=30,
        shuffle=True,
        seed=42
    )
    
    print(f"Number of batches: {len(batch_sampler)}")
    
    # Check first few batches
    for i, batch_indices in enumerate(batch_sampler):
        if i >= 3:  # Only show first 3 batches
            break
        
        batch_labels = speed_labels[batch_indices]
        unique, counts = np.unique(batch_labels, return_counts=True)
        
        print(f"\nBatch {i+1} (size={len(batch_indices)}):")
        for label, count in zip(unique, counts):
            print(f"  Category {int(label)}: {count}")
