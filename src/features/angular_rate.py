"""
Angular rate and directed tangential vector computation from GPS data.

Angular rate:   ω = ‖û(t) × û(t−1)‖ / Δt   (rad/s)
Directed tangential velocity: v_tan_vec = v_ue − (v_ue · û) · û   (ECEF, m/s)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────── #
#  Core computation helpers                                                #
# ──────────────────────────────────────────────────────────────────────── #

def _compute_angular_rate(
    u_ue_bs: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Compute angular rate from UE→BS unit‐vector time series.

    ω(t) = arcsin(‖û(t) × û(t−1)‖) / Δt   ≈  ‖û(t) × û(t−1)‖ / Δt  for small angles

    Args:
        u_ue_bs: [N, 3] unit vectors (ECEF)
        timestamps: [N] timestamps in seconds

    Returns:
        angular_rate: [N-1] angular rate (rad/s), always ≥ 0
    """
    dt = np.diff(timestamps)
    dt = np.where(dt <= 0, 1.0, dt)  # guard zero/negative Δt

    # Cross product of consecutive unit vectors
    cross = np.cross(u_ue_bs[:-1], u_ue_bs[1:])        # [N-1, 3]
    sin_angle = np.linalg.norm(cross, axis=1)           # |sin θ|

    # Clip for numerical safety (unit vectors may not be exactly unit)
    sin_angle = np.clip(sin_angle, 0.0, 1.0)
    angle = np.arcsin(sin_angle)                        # θ ∈ [0, π/2]

    angular_rate = angle / dt                           # rad/s
    return angular_rate


def _compute_directed_tangential(
    pos_ue: np.ndarray,
    pos_bs: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """
    Compute the directed tangential velocity vector in ECEF.

    v_tan = v_ue − (v_ue · û) · û

    where û = (r_UE − r_BS) / ‖r_UE − r_BS‖.

    Args:
        pos_ue: [N, 3] UE ECEF positions
        pos_bs: [N, 3] BS ECEF positions
        timestamps: [N] timestamps in seconds

    Returns:
        v_tan_vec: [N-1, 3] tangential velocity vector (ECEF, m/s)
    """
    # UE velocity via finite difference
    dt = np.diff(timestamps)
    dt = np.where(dt <= 0, 1.0, dt)
    v_ue = np.diff(pos_ue, axis=0) / dt[:, np.newaxis]   # [N-1, 3]

    # UE→BS unit vector (use earlier timestep for alignment)
    r = pos_ue[:-1] - pos_bs[:-1]
    dist = np.linalg.norm(r, axis=1, keepdims=True)
    dist = np.where(dist < 1e-10, 1.0, dist)
    u = r / dist                                          # [N-1, 3]

    # Radial projection and tangential remainder
    v_radial_scalar = np.sum(v_ue * u, axis=1, keepdims=True)  # [N-1, 1]
    v_tan_vec = v_ue - v_radial_scalar * u                     # [N-1, 3]

    return v_tan_vec


# ──────────────────────────────────────────────────────────────────────── #
#  DataFrame‐level feature adders (per‐sequence loop)                      #
# ──────────────────────────────────────────────────────────────────────── #

def add_angular_rate_features(
    df: pd.DataFrame,
    smooth: bool = False,
    window_size: int = 5,
    poly_order: int = 2,
) -> pd.DataFrame:
    """
    Add angular rate feature to dataframe.

    Output column: ``angular_rate`` (rad/s, ≥ 0).
    NaN for the first sample of each sequence.
    """
    df = df.copy()
    df['angular_rate'] = np.nan

    logger.info("Computing angular rate features...")

    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_df = df[seq_mask].sort_values('sample_idx')

        if len(seq_df) < 2:
            continue

        lat = seq_df['lat_ue'].values
        lon = seq_df['lon_ue'].values
        alt = seq_df['height_ue'].values if 'height_ue' in seq_df.columns else seq_df['altitude_ue'].values
        timestamps = seq_df['timestamp'].values

        if smooth and len(seq_df) >= window_size:
            lat = savgol_filter(lat, window_size, poly_order)
            lon = savgol_filter(lon, window_size, poly_order)
            alt = savgol_filter(alt, window_size, poly_order)

        from src.features.preprocessing import geodetic_to_ecef

        ue_x, ue_y, ue_z = geodetic_to_ecef(lat, lon, alt)
        pos_ue = np.column_stack([ue_x, ue_y, ue_z])

        bs_x, bs_y, bs_z = geodetic_to_ecef(
            seq_df['lat_bs'].values,
            seq_df['lon_bs'].values,
            np.zeros_like(lat),
        )
        pos_bs = np.column_stack([bs_x, bs_y, bs_z])

        # UE→BS unit vector
        r = pos_ue - pos_bs
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        dist = np.where(dist < 1e-10, 1.0, dist)
        u_ue_bs = r / dist

        try:
            omega = _compute_angular_rate(u_ue_bs, timestamps)
            indices = seq_df.index[1:]
            if len(indices) == len(omega):
                df.loc[indices, 'angular_rate'] = omega
        except Exception as e:
            logger.error(f"Angular rate failed for seq {seq_idx}: {e}")

    valid = df['angular_rate'].notna().sum()
    logger.info(f"Angular rate: {valid}/{len(df)} samples computed")
    if valid > 0:
        logger.info(
            f"  range: {df['angular_rate'].min():.6f} – "
            f"{df['angular_rate'].max():.6f} rad/s"
        )

    return df


def add_directed_tangential_features(
    df: pd.DataFrame,
    smooth: bool = False,
    window_size: int = 5,
    poly_order: int = 2,
) -> pd.DataFrame:
    """
    Add directed tangential velocity vector to dataframe.

    Output columns: ``v_tan_x``, ``v_tan_y``, ``v_tan_z`` (ECEF, m/s).
    NaN for the first sample of each sequence.
    """
    df = df.copy()
    for c in ('v_tan_x', 'v_tan_y', 'v_tan_z'):
        df[c] = np.nan

    logger.info("Computing directed tangential velocity features...")

    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_df = df[seq_mask].sort_values('sample_idx')

        if len(seq_df) < 2:
            continue

        lat = seq_df['lat_ue'].values
        lon = seq_df['lon_ue'].values
        alt = seq_df['height_ue'].values if 'height_ue' in seq_df.columns else seq_df['altitude_ue'].values
        timestamps = seq_df['timestamp'].values

        if smooth and len(seq_df) >= window_size:
            lat = savgol_filter(lat, window_size, poly_order)
            lon = savgol_filter(lon, window_size, poly_order)
            alt = savgol_filter(alt, window_size, poly_order)

        from src.features.preprocessing import geodetic_to_ecef

        ue_x, ue_y, ue_z = geodetic_to_ecef(lat, lon, alt)
        pos_ue = np.column_stack([ue_x, ue_y, ue_z])

        bs_x, bs_y, bs_z = geodetic_to_ecef(
            seq_df['lat_bs'].values,
            seq_df['lon_bs'].values,
            np.zeros_like(lat),
        )
        pos_bs = np.column_stack([bs_x, bs_y, bs_z])

        try:
            v_tan = _compute_directed_tangential(pos_ue, pos_bs, timestamps)
            indices = seq_df.index[1:]
            if len(indices) == len(v_tan):
                df.loc[indices, 'v_tan_x'] = v_tan[:, 0]
                df.loc[indices, 'v_tan_y'] = v_tan[:, 1]
                df.loc[indices, 'v_tan_z'] = v_tan[:, 2]
        except Exception as e:
            logger.error(f"Directed tangential failed for seq {seq_idx}: {e}")

    valid = df['v_tan_x'].notna().sum()
    logger.info(f"Directed tangential: {valid}/{len(df)} samples computed")

    return df


# ──────────────────────────────────────────────────────────────────────── #
#  Normalization                                                           #
# ──────────────────────────────────────────────────────────────────────── #

def normalize_angular_rate_features(
    df: pd.DataFrame,
    omega_max: float = 0.1,
    fit_on: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clip‐normalize angular rate to [0, 1].

    Args:
        df: DataFrame with ``angular_rate`` column
        omega_max: Saturation value (rad/s). Default 0.1 ≈ 5.7°/s
        fit_on: Optional reference DataFrame (for val/test)

    Returns:
        (normalized_df, {'omega_max': float})
    """
    df = df.copy()
    if omega_max is None:
        ref = fit_on if fit_on is not None else df
        omega_max = float(ref['angular_rate'].max())
        logger.info(f"Using computed omega_max: {omega_max:.6f} rad/s")

    df['angular_rate_norm'] = np.clip(df['angular_rate'] / omega_max, 0, 1)
    return df, {'omega_max': omega_max}


def normalize_directed_tangential_features(
    df: pd.DataFrame,
    v_max: float = 30.0,
    fit_on: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clip‐normalize directed tangential velocity to [-1, 1] per component.

    Uses the same ``v_max`` convention as velocity normalization.

    Args:
        df: DataFrame with ``v_tan_x/y/z`` columns
        v_max: Saturation speed (m/s). Default 30.
        fit_on: Optional reference DataFrame

    Returns:
        (normalized_df, {'v_tan_max': float})
    """
    df = df.copy()
    for c in ('v_tan_x', 'v_tan_y', 'v_tan_z'):
        df[f'{c}_norm'] = np.clip(df[c] / v_max, -1, 1)
    return df, {'v_tan_max': v_max}


# ──────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    print("Angular rate module loaded successfully")
