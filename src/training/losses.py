"""
Custom loss functions for beam prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SpeedWeightedLoss(nn.Module):
    """Loss function with speed-dependent weighting"""
    
    def __init__(
        self,
        speed_weights: Optional[Dict[int, float]] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize speed-weighted loss
        
        Args:
            speed_weights: Weight for each speed category
                Default: {0: 1.0, 1: 1.5, 2: 2.0}
            reduction: Reduction method ('mean' or 'sum')
        """
        super().__init__()
        
        if speed_weights is None:
            speed_weights = {0: 1.0, 1: 1.5, 2: 2.0}
        
        self.speed_weights = speed_weights
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        speed_categories: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch, num_predictions, num_beams]
            targets: [batch, num_predictions]
            speed_categories: [batch]
        
        Returns:
            weighted_loss: scalar
        """
        batch_size, num_steps, num_beams = predictions.shape
        
        total_loss = 0
        
        for step in range(num_steps):
            # Compute per-sample loss
            step_loss = self.ce_loss(
                predictions[:, step, :],
                targets[:, step]
            )  # [batch]
            
            # Apply speed-specific weights
            weights = torch.tensor([
                self.speed_weights[cat.item()] 
                for cat in speed_categories
            ], device=predictions.device, dtype=torch.float32)
            
            weighted_loss = step_loss * weights
            
            if self.reduction == 'mean':
                weighted_loss = weighted_loss.mean()
            elif self.reduction == 'sum':
                weighted_loss = weighted_loss.sum()
            
            total_loss += weighted_loss
        
        return total_loss / num_steps


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Weighting factor in [0, 1]
            gamma: Focusing parameter (gamma >= 0)
            reduction: Reduction method
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch, num_predictions, num_beams] logits
            targets: [batch, num_predictions] target indices
        
        Returns:
            loss: scalar
        """
        batch_size, num_steps, num_beams = predictions.shape
        
        total_loss = 0
        
        for step in range(num_steps):
            # Get predictions and targets for this step
            pred = predictions[:, step, :]  # [batch, num_beams]
            target = targets[:, step]  # [batch]
            
            # Compute softmax probabilities
            p = F.softmax(pred, dim=1)
            
            # Get probability of true class
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)  # [batch]
            
            # Compute focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # Compute cross entropy
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            
            # Apply focal weight and alpha
            loss = self.alpha * focal_weight * ce_loss
            
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            
            total_loss += loss
        
        return total_loss / num_steps


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for regularization
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1
    ):
        """
        Initialize label smoothing loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor in [0, 1]
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch, num_predictions, num_beams] logits
            targets: [batch, num_predictions] target indices
        
        Returns:
            loss: scalar
        """
        batch_size, num_steps, num_beams = predictions.shape
        
        total_loss = 0
        
        for step in range(num_steps):
            pred = predictions[:, step, :]  # [batch, num_beams]
            target = targets[:, step]  # [batch]
            
            # Compute log probabilities
            log_probs = F.log_softmax(pred, dim=1)
            
            # Create smoothed labels
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Compute KL divergence
            loss = -torch.sum(true_dist * log_probs, dim=1).mean()
            
            total_loss += loss
        
        return total_loss / num_steps


class PowerLoss(nn.Module):
    """
    Power-based loss using beam power values
    """
    
    def __init__(
        self,
        temperature: float = 1.0
    ):
        """
        Initialize power loss
        
        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        beam_powers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch, num_predictions, num_beams] logits
            targets: [batch, num_predictions] target indices
            beam_powers: [batch, num_predictions, num_beams] power values
        
        Returns:
            loss: scalar
        """
        batch_size, num_steps, num_beams = predictions.shape
        
        total_loss = 0
        
        for step in range(num_steps):
            pred = predictions[:, step, :]  # [batch, num_beams]
            target = targets[:, step]  # [batch]
            powers = beam_powers[:, step, :]  # [batch, num_beams]
            
            # Compute predicted probabilities
            probs = F.softmax(pred / self.temperature, dim=1)
            
            # Compute expected power
            expected_power = torch.sum(probs * powers, dim=1)
            
            # Get optimal power
            optimal_power = powers.gather(1, target.unsqueeze(1)).squeeze(1)
            
            # Power loss (negative because we want to maximize)
            loss = (optimal_power - expected_power).mean()
            
            total_loss += loss
        
        return total_loss / num_steps


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross-entropy + Power loss
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        power_weight: float = 0.1,
        num_beams: int = 32
    ):
        """
        Initialize combined loss
        
        Args:
            ce_weight: Weight for cross-entropy loss
            power_weight: Weight for power loss
            num_beams: Number of beams
        """
        super().__init__()
        
        self.ce_weight = ce_weight
        self.power_weight = power_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.power_loss = PowerLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        beam_powers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            predictions: [batch, num_predictions, num_beams]
            targets: [batch, num_predictions]
            beam_powers: [batch, num_predictions, num_beams] (optional)
        
        Returns:
            loss: scalar
        """
        # Cross-entropy loss
        ce = 0
        for step in range(predictions.size(1)):
            ce += self.ce_loss(predictions[:, step, :], targets[:, step])
        ce /= predictions.size(1)
        
        total_loss = self.ce_weight * ce
        
        # Add power loss if beam powers provided
        if beam_powers is not None:
            power = self.power_loss(predictions, targets, beam_powers)
            total_loss += self.power_weight * power
        
        return total_loss


# Test code
if __name__ == "__main__":
    print("Testing Loss Functions\n" + "="*50)
    
    # Create dummy data
    batch_size = 8
    num_predictions = 4
    num_beams = 32
    
    predictions = torch.randn(batch_size, num_predictions, num_beams)
    targets = torch.randint(0, num_beams, (batch_size, num_predictions))
    speed_cats = torch.randint(0, 3, (batch_size,))
    beam_powers = torch.randn(batch_size, num_predictions, num_beams)
    
    # Test speed-weighted loss
    print("\n1. Speed-Weighted Loss")
    loss_fn = SpeedWeightedLoss()
    loss = loss_fn(predictions, targets, speed_cats)
    print(f"Loss: {loss.item():.4f}")
    
    # Test focal loss
    print("\n2. Focal Loss")
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test label smoothing
    print("\n3. Label Smoothing Loss")
    loss_fn = LabelSmoothingLoss(num_classes=num_beams, smoothing=0.1)
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")
    
    # Test power loss
    print("\n4. Power Loss")
    loss_fn = PowerLoss(temperature=1.0)
    loss = loss_fn(predictions, targets, beam_powers)
    print(f"Loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n5. Combined Loss")
    loss_fn = CombinedLoss(ce_weight=1.0, power_weight=0.1)
    loss = loss_fn(predictions, targets, beam_powers)
    print(f"Loss: {loss.item():.4f}")
