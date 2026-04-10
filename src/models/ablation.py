"""
Configurable beam predictor for E4 architecture ablation.

One class (AblationBeamPredictor) supports all 8 variants (E4.0-E4.7)
by toggling CNN type, attention, encoder type, GRU layers, hidden dim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class _MultiScaleCNN(nn.Module):
    """Parallel Conv1d branches (k=3,5,7) → cat → mixing conv. Same 2-stage depth as baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Split hidden_dim across 3 branches: 42 + 43 + 43 = 128
        d1 = hidden_dim // 3
        d2 = (hidden_dim - d1) // 2
        d3 = hidden_dim - d1 - d2

        self.branch3 = nn.Sequential(
            nn.Conv1d(input_dim, d1, kernel_size=3, padding=1),
            nn.BatchNorm1d(d1),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(input_dim, d2, kernel_size=5, padding=2),
            nn.BatchNorm1d(d2),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(input_dim, d3, kernel_size=7, padding=3),
            nn.BatchNorm1d(d3),
            nn.ReLU(inplace=True),
        )
        # Mixing conv (same as baseline's second conv layer)
        self.mix = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T] → [B, hidden_dim, T]"""
        out = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        return self.mix(out)


class _BaselineCNN(nn.Module):
    """Teacher's forward: Conv1(in→128) → ReLU → Conv2(128→128) → BN → ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        return x + self.pe[:, :x.size(1), :]


class AblationBeamPredictor(nn.Module):
    """
    Configurable beam predictor for clean architecture ablation.

    Supports:
        cnn_type:       'baseline' | 'multiscale'
        use_attention:  bool  (cross-attention on position↔motion)
        encoder_type:   'gru' | 'bigru' | 'transformer'
        num_encoder_layers: int
        hidden_dim:     int (128 or 256)
        dropout:        float

    Forward returns single tensor [B, V, M] (no tuple).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_beams: int = 32,
        num_predictions: int = 4,
        cnn_type: str = 'baseline',
        use_attention: bool = False,
        position_dim: int = 5,
        motion_dim: int = 3,
        encoder_type: str = 'gru',
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_beams = num_beams
        self.num_predictions = num_predictions
        self.encoder_type = encoder_type
        self.use_attention = use_attention
        self.position_dim = position_dim
        self.motion_dim = motion_dim

        # --- CNN feature extractor ---
        if cnn_type == 'multiscale':
            self.cnn = _MultiScaleCNN(input_dim, hidden_dim)
        else:
            self.cnn = _BaselineCNN(input_dim, hidden_dim)

        # --- Attention (optional) ---
        if use_attention:
            try:
                from src.models.enhanced import MotionAttentionModule
            except ModuleNotFoundError:
                from enhanced import MotionAttentionModule
            self.position_proj = nn.Linear(position_dim, hidden_dim)
            self.motion_proj = nn.Linear(motion_dim, hidden_dim)
            self.attention = MotionAttentionModule(
                feature_dim=hidden_dim, num_heads=4, dropout=dropout
            )
            self.attn_layer_norm = nn.LayerNorm(hidden_dim)

        # --- Encoder ---
        if encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_encoder_layers,
                batch_first=True,
                dropout=dropout if num_encoder_layers > 1 else 0,
            )
        elif encoder_type == 'bigru':
            self.encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_encoder_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_encoder_layers > 1 else 0,
            )
            # Project concatenated fwd+bwd hidden → decoder hidden
            self.bigru_hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        elif encoder_type == 'transformer':
            self.pos_encoding = _SinusoidalPE(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # --- Decoder (always 1-layer GRU) ---
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # --- Classifier (Paper spec: Linear→ReLU→Linear) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_beams),
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self):
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, speed_cats: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, W, input_dim]
            speed_cats: ignored (API compat with Trainer)
        Returns:
            predictions: [B, V, num_beams]
        """
        batch_size = x.size(0)

        # 1) CNN feature extraction
        x_cnn = x.transpose(1, 2)            # [B, C, T]
        cnn_features = self.cnn(x_cnn)        # [B, hidden, T]
        cnn_features = cnn_features.transpose(1, 2)  # [B, T, hidden]

        # 2) Optional attention fusion
        if self.use_attention:
            pos_feat = x[:, :, :self.position_dim]
            mot_feat = x[:, :, self.position_dim:]
            pos_proj = self.position_proj(pos_feat)
            mot_proj = self.motion_proj(mot_feat)
            attn_out, _ = self.attention(pos_proj, mot_proj)
            features = self.attn_layer_norm(cnn_features + attn_out)
        else:
            features = cnn_features

        # 3) Encode
        if self.encoder_type == 'transformer':
            enc_out = self.encoder(self.pos_encoding(features))  # [B, T, H]
            context = enc_out[:, -1, :].contiguous()             # [B, H]
            dec_hidden = context.unsqueeze(0).contiguous()       # [1, B, H]
        elif self.encoder_type == 'bigru':
            _, hidden = self.encoder(features)
            # hidden: [2*num_layers, B, H//2]  — interleaved (fwd_L0, bwd_L0, ...)
            # Take last layer fwd+bwd, concatenate
            fwd = hidden[-2]   # last-layer forward
            bwd = hidden[-1]   # last-layer backward
            context = self.bigru_hidden_proj(torch.cat([fwd, bwd], dim=-1))  # [B, H]
            dec_hidden = context.unsqueeze(0)  # [1, B, H]
        else:  # gru
            _, hidden = self.encoder(features)
            # hidden: [num_layers, B, H]
            dec_hidden = hidden[-1:, :, :]  # last layer → [1, B, H]

        # 4) Decode
        context_vec = dec_hidden.squeeze(0)  # [B, H]
        decoder_input = context_vec.unsqueeze(1).repeat(1, self.num_predictions, 1)
        decoder_output, _ = self.decoder(decoder_input, dec_hidden)

        # 5) Classify
        predictions = self.classifier(decoder_output)  # [B, V, M]
        return predictions

    # ------------------------------------------------------------------
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Self-test
# ======================================================================
if __name__ == '__main__':
    configs = [
        ('E4.0 Baseline',           dict(cnn_type='baseline',   use_attention=False, encoder_type='gru',         num_encoder_layers=1, hidden_dim=128)),
        ('E4.1 MultiScale CNN',     dict(cnn_type='multiscale', use_attention=False, encoder_type='gru',         num_encoder_layers=1, hidden_dim=128)),
        ('E4.2 Attention',          dict(cnn_type='baseline',   use_attention=True,  encoder_type='gru',         num_encoder_layers=1, hidden_dim=128)),
        ('E4.3 MultiScale+Attn',    dict(cnn_type='multiscale', use_attention=True,  encoder_type='gru',         num_encoder_layers=1, hidden_dim=128)),
        ('E4.4 Hidden=256',         dict(cnn_type='baseline',   use_attention=False, encoder_type='gru',         num_encoder_layers=1, hidden_dim=256)),
        ('E4.5 2-layer GRU',        dict(cnn_type='baseline',   use_attention=False, encoder_type='gru',         num_encoder_layers=2, hidden_dim=128)),
        ('E4.6 BiGRU',              dict(cnn_type='baseline',   use_attention=False, encoder_type='bigru',       num_encoder_layers=1, hidden_dim=128)),
        ('E4.7 Transformer',        dict(cnn_type='baseline',   use_attention=False, encoder_type='transformer', num_encoder_layers=2, hidden_dim=128)),
    ]

    B, W, M, V = 4, 8, 32, 4
    input_dim_8d = 8
    position_dim = 5

    print(f"{'Variant':<25} {'Params':>10} {'Output Shape':>15} {'OK':>4}")
    print('-' * 60)

    for name, cfg in configs:
        model = AblationBeamPredictor(
            input_dim=input_dim_8d,
            num_beams=M,
            num_predictions=V,
            position_dim=position_dim,
            motion_dim=input_dim_8d - position_dim,
            dropout=0.1,
            **cfg,
        )
        x = torch.randn(B, W, input_dim_8d)
        out = model(x)
        ok = out.shape == (B, V, M)
        print(f"{name:<25} {model.get_num_parameters():>10,} {str(tuple(out.shape)):>15} {'OK' if ok else 'FAIL':>4}")
        assert ok, f"Shape mismatch for {name}: {out.shape}"

    print("\nAll variants passed.")
