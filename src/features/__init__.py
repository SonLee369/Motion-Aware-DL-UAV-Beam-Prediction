"""
Feature engineering modules
"""

from .preprocessing import (
    DeepSenseDataLoader,
    adjusted_splitting,
    normalize_geodetic,
    geodetic_to_ecef,
    compute_ue_bs_unit_vector,
    create_baseline_features,
    save_processed_data
)
from .velocity import (
    add_velocity_features,
    normalize_velocity_features,
    compute_geodetic_velocity,
    compute_ecef_velocity,
    compute_relative_velocity
)
from .acceleration import (
    add_acceleration_features,
    normalize_acceleration_features,
    compute_smoothed_acceleration
)
from .angular_rate import (
    add_angular_rate_features,
    add_directed_tangential_features,
    normalize_angular_rate_features,
    normalize_directed_tangential_features
)
