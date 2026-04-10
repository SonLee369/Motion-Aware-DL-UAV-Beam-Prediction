"""
Acceleration computation from GPS data with smoothing
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_smoothed_acceleration(
    positions: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 5,
    poly_order: int = 2
) -> np.ndarray:
    """
    Compute acceleration with Savitzky-Golay smoothing
    
    Args:
        positions: [N, 3] position array (ECEF coordinates)
        timestamps: [N] timestamps in seconds
        window_size: Savitzky-Golay window size (must be odd)
        poly_order: Polynomial order for smoothing
    
    Returns:
        acceleration: [N-2, 3] acceleration (m/s^2)
        
    Raises:
        ValueError: If insufficient samples or invalid parameters
    """
    if len(positions) < window_size:
        raise ValueError(f"Need at least {window_size} samples for smoothing")
    
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    if poly_order >= window_size:
        raise ValueError("poly_order must be less than window_size")
    
    # Smooth positions along each axis
    pos_smooth = savgol_filter(positions, window_size, poly_order, axis=0)
    
    # Compute velocity from smoothed positions
    dt = np.diff(timestamps)
    
    # Handle zero or negative time differences
    if np.any(dt <= 0):
        logger.warning("Found zero or negative time differences")
        dt = np.where(dt <= 0, 1.0, dt)
    
    velocity = np.diff(pos_smooth, axis=0) / dt[:, np.newaxis]
    
    # Compute acceleration from velocity
    dt2 = dt[:-1]
    acceleration = np.diff(velocity, axis=0) / dt2[:, np.newaxis]
    
    return acceleration


def add_acceleration_features(
    df: pd.DataFrame,
    window_size: int = 5,
    poly_order: int = 2
) -> pd.DataFrame:
    """
    Add acceleration features to dataframe
    
    Args:
        df: Input dataframe with GPS coordinates
        window_size: Savitzky-Golay window size
        poly_order: Polynomial order
    
    Returns:
        Dataframe with added acceleration columns:
        - a_radial, a_tangential, a_mag (m/s^2)
    """
    df = df.copy()
    
    # Initialize acceleration columns
    df['a_radial'] = np.nan
    df['a_tangential'] = np.nan
    df['a_mag'] = np.nan
    
    logger.info("Computing acceleration features...")
    
    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_df = df[seq_mask].sort_values('sample_idx')
        
        if len(seq_df) < window_size:
            logger.debug(f"Sequence {seq_idx} has < {window_size} samples, skipping")
            continue
        
        # Get positions and timestamps
        lat = seq_df['lat_ue'].values
        lon = seq_df['lon_ue'].values
        alt = seq_df['height_ue'].values
        timestamps = seq_df['timestamp'].values
        
        # Convert to ECEF
        from src.features.preprocessing import geodetic_to_ecef
        
        ue_x, ue_y, ue_z = geodetic_to_ecef(lat, lon, alt)
        pos_ue = np.column_stack([ue_x, ue_y, ue_z])
        
        bs_x, bs_y, bs_z = geodetic_to_ecef(
            seq_df['lat_bs'].values,
            seq_df['lon_bs'].values,
            np.zeros_like(lat)
        )
        pos_bs = np.column_stack([bs_x, bs_y, bs_z])
        
        try:
            # Compute smoothed acceleration
            accel_ue = compute_smoothed_acceleration(
                pos_ue, timestamps, window_size, poly_order
            )
            
            # Compute radial and tangential components
            # Skip first 2 samples due to differentiation
            r_ue_bs = pos_ue[2:] - pos_bs[2:]
            distance = np.linalg.norm(r_ue_bs, axis=1, keepdims=True)
            distance = np.where(distance < 1e-10, 1.0, distance)
            u_ue_bs = r_ue_bs / distance
            
            # Radial acceleration
            a_radial = np.sum(accel_ue * u_ue_bs, axis=1)
            
            # Tangential acceleration
            a_mag = np.linalg.norm(accel_ue, axis=1)
            a_tangential = np.sqrt(np.maximum(a_mag**2 - a_radial**2, 0))
            
            # Assign to dataframe (skip first 2 samples)
            indices = seq_df.index[2:]
            if len(indices) == len(a_radial):
                df.loc[indices, 'a_radial'] = a_radial
                df.loc[indices, 'a_tangential'] = a_tangential
                df.loc[indices, 'a_mag'] = a_mag
                
        except Exception as e:
            logger.error(f"Failed to compute acceleration for sequence {seq_idx}: {e}")
            continue
    
    # Report statistics
    valid_samples = df['a_mag'].notna().sum()
    logger.info(f"Computed acceleration for {valid_samples}/{len(df)} samples")
    
    if valid_samples > 0:
        logger.info(f"Acceleration range: {df['a_mag'].min():.2f} - {df['a_mag'].max():.2f} m/s^2")
    
    return df


def normalize_acceleration_features(
    df: pd.DataFrame,
    a_max: float = 10.0,  # m/s^2
    fit_on: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize acceleration features to [-1, 1]
    
    Args:
        df: Dataframe with acceleration features
        a_max: Maximum expected acceleration magnitude (m/s^2)
        fit_on: Optional dataframe to compute normalization from
    
    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    df = df.copy()
    
    # Determine reference dataframe
    ref_df = fit_on if fit_on is not None else df
    
    # Use either specified a_max or computed max
    if a_max is None:
        a_max = ref_df['a_mag'].abs().max()
        logger.info(f"Using computed a_max: {a_max:.2f} m/s^2")
    
    accel_cols = ['a_radial', 'a_tangential', 'a_mag']
    
    for col in accel_cols:
        if col in df.columns:
            df[f'{col}_norm'] = np.clip(df[col] / a_max, -1, 1)
    
    norm_params = {'a_max': a_max}
    
    return df, norm_params


# Example usage
if __name__ == "__main__":
    print("Acceleration module loaded successfully")
