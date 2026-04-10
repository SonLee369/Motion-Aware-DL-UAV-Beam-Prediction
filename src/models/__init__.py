"""
Model architectures
"""

from .baseline import BaselineBeamPredictor, ImprovedBaselineBeamPredictor
from .enhanced import (
    MultiScaleTemporalConv,
    MotionAttentionModule,
    SpeedConditionedBeamPredictor,
    EnhancedBeamPredictor
)
