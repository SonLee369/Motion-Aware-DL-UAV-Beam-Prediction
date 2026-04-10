"""
Velocity computation from GPS data
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_geodetic_velocity(
    lat: np.ndarray,
    lon: np.ndarray,
    timestamps: np.ndarray,
    earth_radius: float = 6371000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute velocity from geodetic coordinates
    
    Args:
        lat: [N] latitude in degrees
        lon: [N] longitude in degrees
        timestamps: [N] timestamps in seconds
        earth_radius: Earth radius in meters
    
    Returns:
        v_lat_m: [N-1] latitude velocity (m/s)
        v_lon_m: [N-1] longitude velocity (m/s)
        v_mag: [N-1] velocity magnitude (m/s)
    """
    if len(lat) < 2:
        raise ValueError("Need at least 2 samples to compute velocity")
    
    # Time differences
    dt = np.diff(timestamps)
    
    # Handle zero or negative time differences
    if np.any(dt <= 0):
        logger.warning("Found zero or negative time differences, replacing with 1.0")
        dt = np.where(dt <= 0, 1.0, dt)
    
    # Velocity in degrees/s
    v_lat = np.diff(lat) / dt
    v_lon = np.diff(lon) / dt
    
    # Convert to m/s
    lat_mid = (lat[:-1] + lat[1:]) / 2
    v_lat_m = v_lat * (np.pi / 180) * earth_radius
    v_lon_m = v_lon * (np.pi / 180) * earth_radius * np.cos(np.radians(lat_mid))
    
    # Magnitude
    v_mag = np.sqrt(v_lat_m**2 + v_lon_m**2)
    
    return v_lat_m, v_lon_m, v_mag


def compute_ecef_velocity(
    ecef_positions: np.ndarray,
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Compute velocity in ECEF frame
    
    Args:
        ecef_positions: [N, 3] ECEF coordinates (x, y, z)
        timestamps: [N] timestamps in seconds
    
    Returns:
        v_ecef: [N-1, 3] velocity in ECEF (m/s)
    """
    if len(ecef_positions) < 2:
        raise ValueError("Need at least 2 samples to compute velocity")
    
    dt = np.diff(timestamps)
    
    # Handle zero or negative time differences
    if np.any(dt <= 0):
        logger.warning("Found zero or negative time differences")
        dt = np.where(dt <= 0, 1.0, dt)
    
    v_ecef = np.diff(ecef_positions, axis=0) / dt[:, np.newaxis]
    
    return v_ecef


def compute_relative_velocity(
    pos_ue: np.ndarray,
    pos_bs: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial and tangential velocity components
    
    Args:
        pos_ue: [N, 3] UE ECEF positions
        pos_bs: [N, 3] BS ECEF positions
        timestamps: [N] timestamps
    
    Returns:
        v_radial: [N-1] radial velocity (m/s) - toward/away from BS
        v_tangential: [N-1] tangential velocity (m/s) - perpendicular to radial
    """
    if len(pos_ue) < 2:
        raise ValueError("Need at least 2 samples to compute velocity")
    
    # UE-BS unit vector at each time step
    r_ue_bs = pos_ue - pos_bs
    distance = np.linalg.norm(r_ue_bs, axis=1, keepdims=True)
    
    # Avoid division by zero
    distance = np.where(distance < 1e-10, 1.0, distance)
    u_ue_bs = r_ue_bs / distance
    
    # UE velocity
    v_ue = compute_ecef_velocity(pos_ue, timestamps)
    
    # Radial component (dot product with unit vector at t+1)
    # Use unit vector from next time step for alignment
    v_radial = np.sum(v_ue * u_ue_bs[:-1], axis=1)
    
    # Tangential component (perpendicular to radial)
    v_mag = np.linalg.norm(v_ue, axis=1)
    v_tangential = np.sqrt(np.maximum(v_mag**2 - v_radial**2, 0))
    
    return v_radial, v_tangential


def add_velocity_features(
    df: pd.DataFrame,
    smooth: bool = False,
    window_size: int = 5,
    poly_order: int = 2
) -> pd.DataFrame:
    """
    Add velocity features to dataframe
    
    Args:
        df: Input dataframe with GPS coordinates
        smooth: Whether to apply Savitzky-Golay smoothing
        window_size: Window size for smoothing
        poly_order: Polynomial order for smoothing
    
    Returns:
        Dataframe with added velocity columns:
        - v_radial, v_tangential, v_mag (m/s)
    """
    df = df.copy()
    
    # Initialize velocity columns
    df['v_radial'] = np.nan
    df['v_tangential'] = np.nan
    df['v_mag'] = np.nan
    
    logger.info("Computing velocity features...")
    
    for seq_idx in df['sequence_idx'].unique():
        seq_mask = df['sequence_idx'] == seq_idx
        seq_df = df[seq_mask].sort_values('sample_idx')
        
        if len(seq_df) < 2:
            logger.warning(f"Sequence {seq_idx} has < 2 samples, skipping velocity computation")
            continue
        
        # Get positions
        lat = seq_df['lat_ue'].values
        lon = seq_df['lon_ue'].values
        alt = seq_df['height_ue'].values
        timestamps = seq_df['timestamp'].values
        
        # Apply smoothing if requested
        if smooth and len(seq_df) >= window_size:
            lat = savgol_filter(lat, window_size, poly_order)
            lon = savgol_filter(lon, window_size, poly_order)
            alt = savgol_filter(alt, window_size, poly_order)
        
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
            # Compute velocities
            v_radial, v_tangential = compute_relative_velocity(
                pos_ue, pos_bs, timestamps
            )
            _, _, v_mag = compute_geodetic_velocity(lat, lon, timestamps)
            
            # Assign to dataframe (skip first sample)
            indices = seq_df.index[1:]
            if len(indices) == len(v_radial):
                df.loc[indices, 'v_radial'] = v_radial
                df.loc[indices, 'v_tangential'] = v_tangential
                df.loc[indices, 'v_mag'] = v_mag
        except Exception as e:
            logger.error(f"Failed to compute velocity for sequence {seq_idx}: {e}")
            continue
    
    # Report statistics
    valid_samples = df['v_mag'].notna().sum()
    logger.info(f"Computed velocity for {valid_samples}/{len(df)} samples")
    logger.info(f"Velocity range: {df['v_mag'].min():.2f} - {df['v_mag'].max():.2f} m/s")
    
    return df


def normalize_velocity_features(
    df: pd.DataFrame,
    v_max: float = 30.0,  # m/s (approximately 67 mph)
    fit_on: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize velocity features to [0, 1]
    
    Args:
        df: Dataframe with velocity features
        v_max: Maximum expected velocity (m/s)
        fit_on: Optional dataframe to compute normalization from
    
    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    df = df.copy()
    
    # Determine reference dataframe for computing max
    ref_df = fit_on if fit_on is not None else df
    
    # Use either specified v_max or computed max
    if v_max is None:
        v_max = ref_df['v_mag'].max()
        logger.info(f"Using computed v_max: {v_max:.2f} m/s")
    
    velocity_cols = ['v_radial', 'v_tangential', 'v_mag']
    
    for col in velocity_cols:
        if col in df.columns:
            if col == 'v_radial':
                # Radial range: [-1, 1]
                df[f'{col}_norm'] = np.clip(df[col] / v_max, -1, 1)
            else:
                # Tangential & Magnitude range: [0, 1] (vì luôn dương)
                df[f'{col}_norm'] = np.clip(df[col] / v_max, 0, 1)
    
    norm_params = {'v_max': v_max}
    
    return df, norm_params


# Example usage
if __name__ == "__main__":
    # Assuming df is loaded
    print("Velocity module loaded successfully")
