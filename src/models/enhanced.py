"""
Enhanced beam prediction models with motion features and advanced architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale 1D CNN for temporal feature extraction"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_sizes: list = [3, 5, 7]
    ):
        """
        Initialize multi-scale CNN
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension (split across scales)
            kernel_sizes: List of kernel sizes for parallel convolutions
        """
        super().__init__()
        
        self.num_scales = len(kernel_sizes)
        self.output_dim_per_scale = output_dim // self.num_scales
        
        # Create parallel convolutional branches
        self.conv_branches = nn.ModuleList()
        
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    self.output_dim_per_scale,
                    kernel_size=k,
                    padding=k//2
                ),
                nn.BatchNorm1d(self.output_dim_per_scale),
                nn.ReLU(inplace=True)
            )
            self.conv_branches.append(branch)
        
        # Adjust output dimension if needed
        total_out_dim = self.output_dim_per_scale * self.num_scales
        if total_out_dim != output_dim:
            self.adjust = nn.Conv1d(total_out_dim, output_dim, kernel_size=1)
        else:
            self.adjust = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, input_dim, seq_len]
        
        Returns:
            out: [batch, output_dim, seq_len]
        """
        # Apply each branch in parallel
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)
        
        # Adjust dimension if needed
        if self.adjust is not None:
            out = self.adjust(out)
        
        return out


class MotionAttentionModule(nn.Module):
    """Attention mechanism for motion features"""
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize motion attention
        
        Args:
            feature_dim: Dimension of features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
    
    def forward(
        self, 
        position_features: torch.Tensor,
        motion_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            position_features: [batch, seq_len, feature_dim]
            motion_features: [batch, seq_len, feature_dim]
        
        Returns:
            output: [batch, seq_len, feature_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = position_features.shape
        
        # Compute Q, K, V
        Q = self.query(position_features)
        K = self.key(motion_features)
        V = self.value(motion_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)
        
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.feature_dim)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        # Residual connection
        output = position_features + output
        
        return output, attention_weights


class SpeedConditionedBeamPredictor(nn.Module):
    """Model with speed category conditioning"""
    
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_beams: int = 32,
        num_predictions: int = 4,
        speed_embedding_dim: int = 16,
        dropout: float = 0.1
    ):
        """
        Initialize speed-conditioned model
        
        Args:
            input_dim: Input feature dimension (11 for full features)
            hidden_dim: Hidden dimension
            num_beams: Number of beams
            num_predictions: Number of predictions
            speed_embedding_dim: Dimension of speed embedding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_beams = num_beams
        self.num_predictions = num_predictions
        
        # Speed embedding (3 categories: slow, medium, fast)
        self.speed_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=speed_embedding_dim
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer (combine features with speed embedding)
        self.fusion = nn.Sequential(
            nn.Linear(128 + speed_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_beams)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        speed_category: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, window_size, input_dim]
            speed_category: [batch] speed categories (0, 1, or 2)
        
        Returns:
            predictions: [batch, num_predictions, num_beams]
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        x_cnn = x.transpose(1, 2)
        features = self.feature_extractor(x_cnn)
        features = features.transpose(1, 2)  # [batch, seq_len, 128]
        
        # Speed embedding
        speed_emb = self.speed_embedding(speed_category)  # [batch, speed_emb_dim]
        speed_emb = speed_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse features with speed information
        features_concat = torch.cat([features, speed_emb], dim=-1)
        features_fused = self.fusion(features_concat)
        
        # Encoder
        encoder_output, hidden = self.encoder(features_fused)
        
        # Decoder input
        decoder_input = encoder_output[:, -1:, :].repeat(1, self.num_predictions, 1)
        
        # Decoder
        decoder_output, _ = self.decoder(decoder_input, hidden)
        
        # Classify
        predictions = self.classifier(decoder_output)
        
        return predictions

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedBeamPredictor(nn.Module):
    """
    Enhanced model with multi-scale CNN and motion attention
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_beams: int = 32,
        num_predictions: int = 4,
        use_multiscale: bool = True,
        use_attention: bool = True,
        position_dim: int = 5,
        motion_dim: int = 6,
        dropout: float = 0.1
    ):
        """
        Initialize enhanced model
        
        Args:
            input_dim: Total input dimension (position + motion)
            hidden_dim: Hidden dimension
            num_beams: Number of beams
            num_predictions: Number of predictions
            use_multiscale: Whether to use multi-scale CNN
            use_attention: Whether to use motion attention
            position_dim: Dimension of position features
            motion_dim: Dimension of motion features
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_beams = num_beams
        self.num_predictions = num_predictions
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        self.position_dim = position_dim
        self.motion_dim = motion_dim
        
        # Feature extraction
        if use_multiscale:
            self.feature_extractor = nn.Sequential(
                MultiScaleTemporalConv(input_dim, 64, kernel_sizes=[3, 5, 7]),
                MultiScaleTemporalConv(64, 128, kernel_sizes=[3, 5, 7]),
                MultiScaleTemporalConv(128, hidden_dim, kernel_sizes=[3, 5, 7])
            )
        else:
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
        
        # Motion attention (if enabled)
        if use_attention:
            self.position_proj = nn.Linear(position_dim, hidden_dim)
            self.motion_proj = nn.Linear(motion_dim, hidden_dim)
            self.attention = MotionAttentionModule(
                feature_dim=hidden_dim,
                num_heads=4,
                dropout=dropout
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_beams)
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: [batch, window_size, input_dim]
        
        Returns:
            predictions: [batch, num_predictions, num_beams]
            attention_weights: [batch, num_heads, seq_len, seq_len] (if use_attention)
        """
        batch_size, seq_len, _ = x.shape
        attention_weights = None
        
        # Always run feature extractor (multi-scale CNN or standard CNN)
        x_cnn = x.transpose(1, 2)
        cnn_features = self.feature_extractor(x_cnn)
        cnn_features = cnn_features.transpose(1, 2)  # [B, T, hidden_dim]

        if self.use_attention:
            # Split raw input into position and motion features for attention
            position_features = x[:, :, :self.position_dim]
            motion_features = x[:, :, self.position_dim:]

            # Project to common hidden dimension
            pos_proj = self.position_proj(position_features)
            motion_proj = self.motion_proj(motion_features)

            # Apply cross-attention between position and motion projections
            attn_out, attention_weights = self.attention(pos_proj, motion_proj)

            # Fuse CNN features with attention output via residual + layer norm
            features = self.layer_norm(cnn_features + attn_out)
        else:
            features = cnn_features
        
        # Encoder
        encoder_output, hidden = self.encoder(features)
        
        # Decoder input
        decoder_input = encoder_output[:, -1:, :].repeat(1, self.num_predictions, 1)
        
        # Decoder
        decoder_output, _ = self.decoder(decoder_input, hidden)
        
        # Classify
        predictions = self.classifier(decoder_output)
        
        return predictions, attention_weights
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    print("Testing Enhanced Models\n" + "="*50)
    
    # Test multi-scale CNN
    print("\n1. Testing MultiScaleTemporalConv")
    multiscale = MultiScaleTemporalConv(input_dim=11, output_dim=128)
    x = torch.randn(8, 11, 8)
    out = multiscale(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Test motion attention
    print("\n2. Testing MotionAttentionModule")
    attention = MotionAttentionModule(feature_dim=128, num_heads=4)
    pos_feat = torch.randn(8, 8, 128)
    motion_feat = torch.randn(8, 8, 128)
    out, attn_weights = attention(pos_feat, motion_feat)
    print(f"Output: {out.shape}, Attention: {attn_weights.shape}")
    
    # Test speed-conditioned model
    print("\n3. Testing SpeedConditionedBeamPredictor")
    model_speed = SpeedConditionedBeamPredictor(input_dim=11, hidden_dim=128)
    x = torch.randn(8, 8, 11)
    speed_cat = torch.randint(0, 3, (8,))
    out = model_speed(x, speed_cat)
    print(f"Output: {out.shape}")
    print(f"Parameters: {model_speed.get_num_parameters():,}")
    
    # Test enhanced model
    print("\n4. Testing EnhancedBeamPredictor")
    model_enhanced = EnhancedBeamPredictor(
        input_dim=11,
        hidden_dim=128,
        use_multiscale=True,
        use_attention=True
    )
    x = torch.randn(8, 8, 11)
    out, attn = model_enhanced(x)
    print(f"Output: {out.shape}")
    if attn is not None:
        print(f"Attention: {attn.shape}")
    print(f"Parameters: {model_enhanced.get_num_parameters():,}")
