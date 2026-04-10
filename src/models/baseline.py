"""
Baseline beam prediction model (position-only features)
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaselineBeamPredictor(nn.Module):
    """
    Baseline model from paper using position-only features
    
    Architecture:
        - 1D CNN for temporal feature extraction
        - GRU encoder-decoder
        - Linear classifier for beam prediction
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_beams: int = 32,
        num_predictions: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize baseline model
        
        Args:
            input_dim: Input feature dimension (5 for position-only)
            hidden_dim: Hidden dimension for GRU
            num_beams: Number of beams (output classes)
            num_predictions: Number of future predictions (V+1)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_beams = num_beams
        self.num_predictions = num_predictions
        
        # Feature extraction (1D CNN) — teacher's forward pass:
        #   Conv1(5→128) → ReLU → Conv2(128→128) → BN(128) → ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0  # No dropout for single layer
        )
        
        # Decoder GRU (input = replicated context vector, dim = hidden_dim)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Classifier head — Paper: "two FC layers with softmax"
        #   Linear(128→64) → ReLU → Linear(64→32)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_beams)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, speed_cats: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, window_size, input_dim] input features
            speed_cats: ignored (accepted for API compatibility with Trainer)
        
        Returns:
            predictions: [batch, num_predictions, num_beams] logits
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction via 1D CNN
        x_cnn = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        features = self.feature_extractor(x_cnn)  # [batch, 128, seq_len]
        features = features.transpose(1, 2)  # [batch, seq_len, 128]
        
        # Encode sequence
        encoder_output, hidden = self.encoder(features)
        # encoder_output: [batch, seq_len, hidden_dim]
        # hidden: [1, batch, hidden_dim]
        
        # Context vector = encoder's final hidden state (paper Section II.C.2)
        context_vector = hidden[-1]  # [batch, hidden_dim]
        
        # Replicate context vector V times as decoder input sequence
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.num_predictions, 1)
        # decoder_input: [batch, num_predictions, hidden_dim]
        
        # Decode: h0 = encoder hidden state (carries information flow)
        decoder_output, _ = self.decoder(decoder_input, hidden)
        # decoder_output: [batch, num_predictions, hidden_dim]
        
        # Classify each prediction step
        predictions = self.classifier(decoder_output)
        # predictions: [batch, num_predictions, num_beams]
        
        return predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ImprovedBaselineBeamPredictor(nn.Module):
    """
    Improved baseline with proper decoder architecture
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_beams: int = 32,
        num_predictions: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_beams = num_beams
        self.num_predictions = num_predictions
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Projection for decoder input
        self.decoder_input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_beams)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved decoder
        
        Args:
            x: [batch, window_size, input_dim]
        
        Returns:
            predictions: [batch, num_predictions, num_beams]
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        x_cnn = x.transpose(1, 2)
        features = self.feature_extractor(x_cnn)
        features = features.transpose(1, 2)
        
        # Encode
        encoder_output, hidden = self.encoder(features)
        
        # Prepare decoder input
        context = encoder_output[:, -1, :]  # Last encoder output
        decoder_input = self.decoder_input_proj(context)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.num_predictions, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, hidden)
        
        # Classify
        predictions = self.classifier(decoder_output)
        
        return predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    # Test baseline model
    model = BaselineBeamPredictor(input_dim=5, hidden_dim=128, num_beams=32)
    
    # Create dummy input
    batch_size = 8
    window_size = 8
    input_dim = 5
    
    dummy_input = torch.randn(batch_size, window_size, input_dim)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [8, 4, 32]")
    print(f"Number of parameters: {model.get_num_parameters():,}")
    
    # Test improved model
    print("\n" + "="*50)
    model_improved = ImprovedBaselineBeamPredictor(input_dim=5, hidden_dim=128)
    output_improved = model_improved(dummy_input)
    print(f"Improved model output shape: {output_improved.shape}")
    print(f"Number of parameters: {model_improved.get_num_parameters():,}")
