"""
Training utilities for beam prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for beam prediction models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        device: str = 'cuda',
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        num_epochs: int = 20,
        lr_schedule: List[int] = [12, 18],
        lr_gamma: float = 0.1,
        gradient_clip: float = 0.0,
        save_dir: str = 'checkpoints',
        log_dir: str = 'runs',
        early_stopping_patience: int = 999,
        min_epochs: int = 0
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function (default: CrossEntropyLoss)
            device: Device for training
            learning_rate: Initial learning rate
            weight_decay: Weight decay for Adam optimizer (paper: 0)
            num_epochs: Number of training epochs
            lr_schedule: Epochs for learning rate decay
            lr_gamma: Learning rate decay factor
            gradient_clip: Gradient clipping value
            save_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            early_stopping_patience: Patience for early stopping (only after min_epochs)
            min_epochs: Minimum epochs to train before early stopping is allowed
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.min_epochs = min_epochs
        
        # Loss function
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=lr_schedule,
            gamma=lr_gamma
        )
        
        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def _parse_batch(self, batch_data):
        """
        Parse a batch tuple into (features, beams, speed_cats, beam_powers).

        Batch formats supported (from BeamPredictionDataset):
          2-element: (features, beams)
          3-element: (features, beams, speed_cats)  OR  (features, beams, beam_powers)
          4-element: (features, beams, speed_cats, beam_powers)

        3-element disambiguation: beam_powers tensor is 3-D [B, V, M];
        speed_cats is scalar/1-D.
        """
        if len(batch_data) == 2:
            features, beams = batch_data
            speed_cats, beam_powers = None, None
        elif len(batch_data) == 3:
            features, beams, third = batch_data
            if third.ndim == 3:          # beam_powers: [B, V, M]
                speed_cats, beam_powers = None, third
            else:                        # speed_cats: scalar or [B]
                speed_cats, beam_powers = third, None
        elif len(batch_data) == 4:
            features, beams, speed_cats, beam_powers = batch_data
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")

        features = features.to(self.device)
        beams = beams.to(self.device)
        if speed_cats is not None:
            speed_cats = speed_cats.to(self.device)
        if beam_powers is not None:
            beam_powers = beam_powers.to(self.device)
        return features, beams, speed_cats, beam_powers

    def _forward(self, features, speed_cats=None):
        """
        Run model forward pass.

        Passes speed_cats to the model only when it exposes a speed_embedding
        attribute (i.e. SpeedConditionedBeamPredictor). Unwraps tuple outputs
        (e.g. attention-returning EnhancedBeamPredictor).
        """
        if speed_cats is not None and hasattr(self.model, 'speed_embedding'):
            output = self.model(features, speed_cats)
        else:
            output = self.model(features)
        return output[0] if isinstance(output, tuple) else output

    def _compute_loss(self, predictions, beams, speed_cats=None, beam_powers=None):
        """
        Dispatch loss computation based on criterion class name.

        All custom losses (SpeedWeightedLoss, FocalLoss, LabelSmoothingLoss,
        PowerLoss, CombinedLoss) accept full [B, T, M] / [B, T] tensors and
        handle the step loop internally.  Plain CrossEntropyLoss is handled by
        reshaping to [B*T, M] / [B*T].
        """
        loss_cls = type(self.criterion).__name__
        if loss_cls == 'SpeedWeightedLoss':
            return self.criterion(predictions, beams, speed_cats)
        elif loss_cls in ('PowerLoss', 'CombinedLoss'):
            return self.criterion(predictions, beams, beam_powers)
        elif loss_cls in ('FocalLoss', 'LabelSmoothingLoss'):
            return self.criterion(predictions, beams)
        else:
            # CrossEntropyLoss: reshape across time steps
            B, T, M = predictions.shape
            return self.criterion(predictions.reshape(B * T, M), beams.reshape(B * T))

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

        for batch_idx, batch_data in enumerate(pbar):
            features, beams, speed_cats, beam_powers = self._parse_batch(batch_data)

            # Forward pass
            predictions = self._forward(features, speed_cats)

            # Compute loss (dispatch based on criterion type)
            loss = self._compute_loss(predictions, beams, speed_cats, beam_powers)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate model

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc='Validation'):
                features, beams, speed_cats, beam_powers = self._parse_batch(batch_data)

                # Forward pass
                predictions = self._forward(features, speed_cats)

                # Loss
                loss = self._compute_loss(predictions, beams, speed_cats, beam_powers)
                total_loss += loss.item()
                num_batches += 1

                # Accuracy (average across ALL prediction steps, not just v=0)
                for v in range(predictions.size(1)):
                    pred_beams = predictions[:, v, :].argmax(dim=1)
                    correct += (pred_beams == beams[:, v]).sum().item()
                total += beams.size(0) * predictions.size(1)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: str = None
    ):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print statistics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! Accuracy: {self.best_val_acc:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping (only after min_epochs)
            if epoch + 1 >= self.min_epochs and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate schedule
            self.scheduler.step()
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        logger.info(f">>> Best epoch: {self.best_epoch+1}/{self.num_epochs}  |  Val Acc: {self.best_val_acc*100:.2f}%")
        logger.info(f"{'='*60}")
        
        self.writer.close()
    
    def get_training_history(self) -> Dict:
        """Get training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }


# Example usage
if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.baseline import BaselineBeamPredictor
    
    # Create dummy data
    train_features = torch.randn(1000, 8, 5)
    train_beams = torch.randint(0, 32, (1000, 4))
    train_dataset = TensorDataset(train_features, train_beams)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_features = torch.randn(200, 8, 5)
    val_beams = torch.randint(0, 32, (200, 4))
    val_dataset = TensorDataset(val_features, val_beams)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = BaselineBeamPredictor(input_dim=5, hidden_dim=128)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=5
    )
    
    # Train
    trainer.train()
    
    # Get history
    history = trainer.get_training_history()
    print(f"\nBest accuracy: {history['best_val_acc']:.4f}")
