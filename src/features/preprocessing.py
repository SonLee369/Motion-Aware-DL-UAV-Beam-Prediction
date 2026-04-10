"""
Data preprocessing utilities for DeepSense 6G Scenario 23
Includes data loading, splitting, and feature extraction
"""

import numpy as np
import pandas as pd
import ast
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSenseDataLoader:
    """Load and parse DeepSense 6G Scenario 23 data"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data directory or CSV file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.data_dir = None  # Will be set when parsing DeepSense format
        self.csv_file_path = None  # Will be set when CSV is found
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    def load_raw_data(self, filename: str = "scenario23.csv") -> pd.DataFrame:
        """
        Load raw data from CSV files with error handling
        
        Args:
            filename: Name of CSV file to load (default: scenario23.csv from DeepSense 6G)
            
        Returns:
            DataFrame with standardized columns:
            - sequence_idx: Trip identifier (from seq_index)
            - sample_idx: Time step within trip (from index)
            - lat_bs, lon_bs: BS GPS coordinates (parsed from unit1_loc)
            - lat_ue, lon_ue: UE GPS coordinates (parsed from unit2_loc)
            - altitude_ue: UE absolute altitude in meters (from unit2_altitude)
            - height_ue: UE relative height above ground in meters (from unit2_height)  <-- THÊM DÒNG NÀY
            - beam_idx: Optimal beam index 0-31 (from unit1_beam_index)
            - beam_power: Received power for each beam, 32 values (parsed from unit1_pwr_60ghz)
            - timestamp: Unix timestamp (from time_stamp[UTC])
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        # Try multiple possible paths for the CSV file
        possible_paths = [
            self.data_path / filename,
            self.data_path / "scenario23_dev_w_resources" / "scenario23_dev" / filename,
            self.data_path / "scenario23_dev" / filename,
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(
                f"Data file not found. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths)
            )
        
        # Store actual CSV file path for loading external data files
        self.csv_file_path = file_path
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Map DeepSense 6G column names to expected names
        logger.info("Mapping DeepSense 6G column names to standard format...")
        
        # Check if data is in DeepSense format or already preprocessed
        if 'seq_index' in df.columns:
            # DeepSense 6G format - need to parse and map
            df = self._parse_deepsense_format(df)
        elif 'sequence_idx' in df.columns:
            # Already in expected format
            logger.info("Data already in expected format")
        else:
            raise ValueError(
                f"Unrecognized data format. Available columns: {list(df.columns)}"
            )
        
        # Validate required columns after mapping
        required_cols = [
            'sequence_idx', 'sample_idx', 'lat_bs', 'lon_bs',
            'lat_ue', 'lon_ue', 'altitude_ue', 'height_ue', 'beam_idx', 'beam_power'
        ]        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns after mapping: {missing_cols}")
        
        self.raw_data = df
        logger.info(f"Loaded {len(df)} samples from {df['sequence_idx'].nunique()} sequences")
        
        return df
    
    def _parse_deepsense_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse DeepSense 6G CSV format to standard format
        
        Column mapping:
        - seq_index → sequence_idx
        - index → sample_idx
        - unit2_loc → lat_ue, lon_ue (parse string/list)
        - unit1_loc → lat_bs, lon_bs (parse string/list)
        - unit2_altitude → altitude_ue
        - unit1_beam_index → beam_idx
        - unit1_pwr_60ghz → beam_power (parse array)
        - time_stamp[UTC] → timestamp
        
        Args:
            df: Raw DeepSense 6G dataframe
            
        Returns:
            Dataframe with standardized column names
        """
        logger.info("Parsing DeepSense 6G format...")
        df = df.copy()
        
        # Store the CSV file directory for loading beam power files
        # Use the actual CSV location, not the initial data_path
        self.data_dir = self.csv_file_path.parent
        
        # Simple column renames
        rename_map = {
            'seq_index': 'sequence_idx',
            'index': 'sample_idx',
            'unit2_altitude': 'altitude_ue',
            'unit1_beam_index': 'beam_idx',
            'time_stamp[UTC]': 'timestamp'
        }
        df = df.rename(columns=rename_map)
        
        # Parse unit2_loc (UE location) → lat_ue, lon_ue (from GPS files)
        logger.info("Loading UE location from GPS files (unit2_loc)...")
        ue_locs = df['unit2_loc'].apply(lambda x: self._parse_location(x, self.data_dir))
        df['lat_ue'] = ue_locs.apply(lambda x: x[0])
        df['lon_ue'] = ue_locs.apply(lambda x: x[1])
        
        # Parse unit1_loc (BS location) → lat_bs, lon_bs (from GPS files)
        logger.info("Loading BS location from GPS files (unit1_loc)...")
        bs_locs = df['unit1_loc'].apply(lambda x: self._parse_location(x, self.data_dir))
        df['lat_bs'] = bs_locs.apply(lambda x: x[0])
        df['lon_bs'] = bs_locs.apply(lambda x: x[1])
        
        # Load altitude from file (single value per sample)
        logger.info("Loading altitude from files...")
        df['altitude_ue'] = df['altitude_ue'].apply(
            lambda x: self._load_from_file(x, self.data_dir)
        )
        
        # Load height above ground (for computing BS ground elevation)
        if 'unit2_height' in df.columns:
            logger.info("Loading UE height above ground from files...")
            df['height_ue'] = df['unit2_height'].apply(
                lambda x: self._load_from_file(x, self.data_dir)
            )
        
        # Parse unit1_pwr_60ghz → beam_power array (load from files)
        logger.info("Loading beam power data from files...")
        df['beam_power'] = df['unit1_pwr_60ghz'].apply(
            lambda path: self._parse_beam_power(path, self.data_dir)
        )
        
        # Validate beam power dimensions (DeepSense 6G has 64 beams)
        invalid_powers = df['beam_power'].apply(lambda x: len(x) not in [32, 64])
        if invalid_powers.any():
            logger.warning(f"Found {invalid_powers.sum()} samples with invalid beam power dimensions")
        
        # Log beam power array size
        if len(df) > 0:
            sample_power_len = len(df['beam_power'].iloc[0])
            logger.info(f"Beam power array size: {sample_power_len}")
        
        # Load UE speed from files (m/s)
        if 'unit2_speed' in df.columns:
            logger.info("Loading UE speed from files (unit2_speed)...")
            df['speed_mps'] = df['unit2_speed'].apply(
                lambda x: self._load_from_file(x, self.data_dir)
            )
            logger.info(f"Speed range: {df['speed_mps'].min():.2f} - "
                        f"{df['speed_mps'].max():.2f} m/s")

        # Load 3D velocity components from IMU/GPS-fusion sensor (reliable, direct measurements)
        # unit2_x-speed, unit2_y-speed, unit2_z-speed contain file paths to per-sample velocity files
        sensor_vel_axes = [('vx_mps', 'unit2_x-speed'), ('vy_mps', 'unit2_y-speed'), ('vz_mps', 'unit2_z-speed')]
        if all(col in df.columns for _, col in sensor_vel_axes):
            logger.info("Loading UE 3D velocity components from sensor files...")
            for out_col, raw_col in sensor_vel_axes:
                df[out_col] = df[raw_col].apply(lambda x: self._load_from_file(x, self.data_dir))
            sensor_3d = np.sqrt(df['vx_mps']**2 + df['vy_mps']**2 + df['vz_mps']**2)
            logger.info(f"Sensor 3D speed range: {sensor_3d.min():.2f} - {sensor_3d.max():.2f} m/s")
        
        # Parse timestamp: DeepSense format "['HH-MM-SS-millis']" → seconds
        logger.info("Parsing timestamps...")
        df['timestamp'] = df['timestamp'].apply(self._parse_timestamp)
        
        logger.info("DeepSense format parsed successfully")
        return df
    
    @staticmethod
    def _load_from_file(file_path_or_value, data_dir: Path = None) -> float:
        """
        Load a single float value from file path or return value directly
        
        Args:
            file_path_or_value: Either a file path string or numeric value
            data_dir: Directory containing the CSV (parent of the file paths)
            
        Returns:
            Float value
        """
        try:
            if isinstance(file_path_or_value, str) and (file_path_or_value.startswith('./') or file_path_or_value.startswith('.\\')):
                # It's a file path - load from file
                if data_dir is None:
                    logger.warning("data_dir not provided for file path")
                    return 0.0
                
                # Construct full path
                full_path = data_dir / file_path_or_value.lstrip('./')
                
                # Read single value from file
                with open(full_path, 'r') as f:
                    value = float(f.read().strip())
                
                return value
            else:
                # Direct value
                return float(file_path_or_value)
        except Exception as e:
            logger.warning(f"Failed to load value: {e}")
            return 0.0
    
    @staticmethod
    def _parse_timestamp(ts_value) -> float:
        """
        Parse DeepSense 6G timestamp to seconds.
        
        Format: "['HH-MM-SS-millis']" → total seconds from midnight.
        Falls back to float conversion for already-numeric values.
        
        Args:
            ts_value: Timestamp string or numeric value
            
        Returns:
            Timestamp in seconds (float)
        """
        try:
            if isinstance(ts_value, (int, float)):
                return float(ts_value)
            s = str(ts_value).strip()
            # Strip list brackets: "['HH-MM-SS-ms']" → "HH-MM-SS-ms"
            if s.startswith('[') or s.startswith("'") or s.startswith('"'):
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    s = str(parsed[0])
                else:
                    s = str(parsed)
            parts = s.split('-')
            if len(parts) == 4:
                h, m, sec, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                return h * 3600.0 + m * 60.0 + sec + ms / 1000.0
            # Fallback
            return float(s)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{ts_value}': {e}")
            return 0.0
    
    @staticmethod
    def _parse_location(loc_path_or_str, data_dir: Path = None) -> Tuple[float, float]:
        """
        Parse location from file path or string to (lat, lon) tuple
        
        In DeepSense 6G, unit1_loc and unit2_loc contain file paths to GPS data.
        
        Args:
            loc_path_or_str: File path string or location string/list
            data_dir: Directory containing the CSV (parent of the file paths)
            
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            if isinstance(loc_path_or_str, str) and (loc_path_or_str.startswith('./') or loc_path_or_str.startswith('.\\')):
                # It's a file path - load GPS from file
                if data_dir is None:
                    logger.warning("data_dir not provided for file path")
                    return (0.0, 0.0)
                
                # Construct full path
                full_path = data_dir / loc_path_or_str.lstrip('./')
                
                # Read GPS data from file (format: two lines with lat and lon)
                with open(full_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        lat = float(lines[0].strip())
                        lon = float(lines[1].strip())
                        return (lat, lon)
                    else:
                        logger.warning(f"GPS file has fewer than 2 lines: {full_path}")
                        return (0.0, 0.0)
            elif isinstance(loc_path_or_str, str):
                # String format "[lat, lon]"
                loc_str = loc_path_or_str.strip('[]() ')
                parts = loc_str.split(',')
                if len(parts) >= 2:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    return (lat, lon)
                else:
                    return (0.0, 0.0)
            elif isinstance(loc_path_or_str, (list, tuple)):
                if len(loc_path_or_str) >= 2:
                    return (float(loc_path_or_str[0]), float(loc_path_or_str[1]))
                else:
                    return (0.0, 0.0)
            else:
                return (0.0, 0.0)
        except Exception as e:
            logger.warning(f"Failed to parse location: {e}")
            return (0.0, 0.0)
    
    @staticmethod
    def _parse_beam_power(file_path_or_array, data_dir: Path = None) -> np.ndarray:
        """
        Safely parse beam power from file path or array
        
        In DeepSense 6G, unit1_pwr_60ghz contains file paths like:
        './unit1/mmWave_data/mmWave_power_1.txt'
        
        Each file contains 64 lines with one power value per line (64 beams).
        
        Args:
            file_path_or_array: Either a file path string or beam power array
            data_dir: Directory containing the CSV (parent of the file paths)
            
        Returns:
            Numpy array of beam powers (32 values)
        """
        try:
            if isinstance(file_path_or_array, str):
                # Check if it's a file path
                if file_path_or_array.startswith('./') or file_path_or_array.startswith('.\\'):
                    # It's a file path - load from file
                    if data_dir is None:
                        logger.warning("data_dir not provided for file path")
                        return np.zeros(32, dtype=np.float32)
                    
                    # Construct full path (handle both ./ and .\ prefixes)
                    clean_path = file_path_or_array.lstrip('./').lstrip('.\\')
                    full_path = data_dir / clean_path
                    
                    # Read beam power values from file
                    with open(full_path, 'r') as f:
                        values = [float(line.strip()) for line in f.readlines()]
                    
                    # DeepSense 6G has 64 beam power measurements per sample
                    if len(values) not in [32, 64]:
                        logger.warning(f"Unexpected beam power count: {len(values)} from {full_path}")
                        # Pad or truncate to 64
                        if len(values) < 64:
                            values.extend([0.0] * (64 - len(values)))
                        else:
                            values = values[:64]
                    
                    return np.array(values, dtype=np.float32)
                else:
                    # Try to parse as array string
                    return np.array(ast.literal_eval(file_path_or_array), dtype=np.float32)
                    
            elif isinstance(file_path_or_array, (list, np.ndarray)):
                return np.array(file_path_or_array, dtype=np.float32)
            else:
                logger.warning(f"Unexpected beam_power type: {type(file_path_or_array)}")
                return np.zeros(64, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to parse beam_power: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        stats = {
            'total_samples': len(self.raw_data),
            'num_sequences': self.raw_data['sequence_idx'].nunique(),
            'beam_distribution': self.raw_data['beam_idx'].value_counts().to_dict(),
            'lat_range': (
                float(self.raw_data['lat_ue'].min()), 
                float(self.raw_data['lat_ue'].max())
            ),
            'lon_range': (
                float(self.raw_data['lon_ue'].min()), 
                float(self.raw_data['lon_ue'].max())
            ),
            'altitude_range': (
                float(self.raw_data['altitude_ue'].min()), 
                float(self.raw_data['altitude_ue'].max())
            ),
            'samples_per_sequence': {
                'mean': float(self.raw_data.groupby('sequence_idx').size().mean()),
                'std': float(self.raw_data.groupby('sequence_idx').size().std()),
                'min': int(self.raw_data.groupby('sequence_idx').size().min()),
                'max': int(self.raw_data.groupby('sequence_idx').size().max())
            }
        }
        
        return stats
    
    def validate_data(self) -> bool:
        """
        Validate loaded data for common issues
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        df = self.raw_data
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        
        # Check beam_idx range (DeepSense 6G has beams 0-60, 56 unique indices)
        min_beam = df['beam_idx'].min()
        max_beam = df['beam_idx'].max()
        if min_beam < 0 or max_beam > 60:
            raise ValueError(
                f"Invalid beam indices found: range [{min_beam}, {max_beam}]. Expected [0, 60]"
            )
        logger.info(f"Beam index range: [{min_beam}, {max_beam}]")
        
        # Check GPS coordinates validity
        if not (df['lat_ue'].between(-90, 90).all()):
            raise ValueError("Invalid latitude values found")
        if not (df['lon_ue'].between(-180, 180).all()):
            raise ValueError("Invalid longitude values found")
        
        # Check for duplicate samples
        duplicates = df.duplicated(subset=['sequence_idx', 'sample_idx'])
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate samples")
        
        logger.info("Data validation passed")
        return True


def adjusted_splitting(
    df: pd.DataFrame,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    chunk_sizes: List[int] = None,
    random_seed: int = 42,
    min_samples_per_split: int = 100,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Adjusted splitting matching teacher's algorithm exactly (gpsbeam DataPrep._split_general_case).

    Algorithm:
    1. Try chunk sizes as percentages of total samples [1%,2%,3%,4%,5%,10%,100%],
       filtered to keep only those > min sequence length.
    2. For each chunk size, split the ENTIRE dataset sequentially into chunks.
       Within each chunk: first train_ratio fraction → train, next val_ratio → val,
       rest → test (no randomness — deterministic sequential assignment).
    3. Select the chunk size with the lowest label-distribution divergence score.
    4. Per-class re-splitting ("adjusted" step): for each beam class, collect all
       samples of that class from all three splits and re-split sequentially by ratio.
       This ensures each class has exactly the right proportion in each split.
    5. Sort each split by sample_idx ascending.

    Args:
        df: Raw dataframe with 'beam_idx', 'sequence_idx', 'sample_idx' columns
        train_ratio: Training set ratio (default 0.65)
        val_ratio:   Validation set ratio (default 0.15)
        test_ratio:  Test set ratio (default 0.20)
        chunk_sizes: Candidate chunk sizes to try; if None, use percentage-based sizes
        random_seed: Unused (kept for API compatibility; algorithm is deterministic)
        min_samples_per_split: Minimum samples required per split
        debug: Whether to log extra info

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    total_samples = len(df)

    # Step 1: Build candidate chunk sizes as percentages of total, filtered by min seq length
    if chunk_sizes is None:
        seq_counts = df['sequence_idx'].value_counts()
        min_seq_len = int(seq_counts.min())
        pct_sizes = [int(total_samples * p) for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 1.0]]
        chunk_sizes = [s for s in pct_sizes if s > min_seq_len]
        if not chunk_sizes:
            chunk_sizes = [total_samples]  # fallback: single chunk
        logger.info(f"Auto chunk sizes (pct-based, min_seq={min_seq_len}): {chunk_sizes}")

    label_col = 'beam_idx'
    class_distribution = df[label_col].value_counts(normalize=True)

    best_score = float('inf')
    best_splits = None
    best_chunk_size = None

    # Step 2: Try each chunk size
    for chunk_size in chunk_sizes:
        train_df = pd.DataFrame()
        val_df_tmp = pd.DataFrame()
        test_df = pd.DataFrame()

        # Chunk the ENTIRE dataset sequentially
        for i in range(0, total_samples, chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            n = len(chunk)
            chunk_train_size = int(train_ratio * n)
            chunk_val_size = int(val_ratio * n)

            chunk_train = chunk.iloc[:chunk_train_size]
            chunk_val = chunk.iloc[chunk_train_size:chunk_train_size + chunk_val_size]
            chunk_test = chunk.iloc[chunk_train_size + chunk_val_size:]

            train_df = pd.concat([train_df, chunk_train])
            val_df_tmp = pd.concat([val_df_tmp, chunk_val])
            test_df = pd.concat([test_df, chunk_test])

        # Skip if any split is empty or too small
        if (len(train_df) < min_samples_per_split or
                len(val_df_tmp) < min_samples_per_split or
                len(test_df) < min_samples_per_split):
            logger.warning(f"chunk_size={chunk_size} resulted in insufficient samples, skipping")
            continue

        # Calculate distribution divergence score (lower = better)
        train_dist = train_df[label_col].value_counts(normalize=True)
        val_dist = val_df_tmp[label_col].value_counts(normalize=True)
        test_dist = test_df[label_col].value_counts(normalize=True)

        dist_score = (
            (class_distribution - train_dist).abs().sum() +
            (class_distribution - val_dist).abs().sum() +
            (class_distribution - test_dist).abs().sum()
        )
        logger.info(f"chunk_size={chunk_size}: score={dist_score:.4f}, "
                    f"train={len(train_df)}, val={len(val_df_tmp)}, test={len(test_df)}")

        if dist_score < best_score:
            best_score = dist_score
            best_chunk_size = chunk_size
            best_splits = (train_df.copy(), val_df_tmp.copy(), test_df.copy())

    if best_splits is None:
        raise ValueError("Could not create valid splits. Check data or chunk_sizes.")

    train_df, val_df, test_df = best_splits
    logger.info(f"Best chunk_size: {best_chunk_size}, score: {best_score:.4f}")

    # Step 3: Per-class re-splitting ("adjusted" step)
    # For each beam class, collect all its samples from across splits and re-split by ratio.
    train_parts, val_parts, test_parts = [], [], []
    for class_label in class_distribution.index:
        train_cls = train_df[train_df[label_col] == class_label]
        val_cls = val_df[val_df[label_col] == class_label]
        test_cls = test_df[test_df[label_col] == class_label]

        class_data = pd.concat([train_cls, val_cls, test_cls])
        n_cls = len(class_data)

        cls_train_size = int(train_ratio * n_cls)
        cls_val_size = int(val_ratio * n_cls)

        train_parts.append(class_data.iloc[:cls_train_size])
        val_parts.append(class_data.iloc[cls_train_size:cls_train_size + cls_val_size])
        test_parts.append(class_data.iloc[cls_train_size + cls_val_size:])

    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)
    test_df = pd.concat(test_parts)

    # Step 4: Sort by sample_idx ascending
    train_df = train_df.sort_values(by='sample_idx', ascending=True).reset_index(drop=True)
    val_df = val_df.sort_values(by='sample_idx', ascending=True).reset_index(drop=True)
    test_df = test_df.sort_values(by='sample_idx', ascending=True).reset_index(drop=True)

    logger.info(f"Final split: train={len(train_df)} ({len(train_df)/total_samples:.3f}), "
                f"val={len(val_df)} ({len(val_df)/total_samples:.3f}), "
                f"test={len(test_df)} ({len(test_df)/total_samples:.3f})")

    return train_df, val_df, test_df


def normalize_geodetic(
    df: pd.DataFrame,
    fit_on: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Min-max normalization of UE geodetic coordinates
    
    Args:
        df: Dataframe to normalize
        fit_on: Optional dataframe to compute normalization parameters from
                (use for val/test sets to avoid data leakage)
    
    Returns:
        Tuple of (normalized_df, normalization_params)
    """
    df = df.copy()
    
    # Determine which dataframe to compute stats from
    ref_df = fit_on if fit_on is not None else df
    
    lat_min, lat_max = ref_df['lat_ue'].min(), ref_df['lat_ue'].max()
    lon_min, lon_max = ref_df['lon_ue'].min(), ref_df['lon_ue'].max()
    
    # Handle edge case: all values are the same
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    
    if lat_range < 1e-10:
        logger.warning("Latitude range too small, setting normalized values to 0.5")
        df['lat_ue_norm'] = 0.5
    else:
        df['lat_ue_norm'] = (df['lat_ue'] - lat_min) / lat_range
    
    if lon_range < 1e-10:
        logger.warning("Longitude range too small, setting normalized values to 0.5")
        df['lon_ue_norm'] = 0.5
    else:
        df['lon_ue_norm'] = (df['lon_ue'] - lon_min) / lon_range
    
    # Store normalization parameters
    norm_params = {
        'lat': (lat_min, lat_max),
        'lon': (lon_min, lon_max)
    }
    
    return df, norm_params


def geodetic_to_ecef(
    lat: np.ndarray, 
    lon: np.ndarray, 
    alt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic coordinates to ECEF (Earth-Centered Earth-Fixed)
    
    Uses WGS-84 ellipsoid model
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters above ellipsoid
    
    Returns:
        Tuple of (x, y, z) in ECEF frame (meters)
    """
    # WGS-84 parameters
    a = 6378137.0  # Semi-major axis (meters)
    e2 = 0.00669437999014  # First eccentricity squared
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Radius of curvature in prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    
    return x, y, z


def compute_ue_bs_unit_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UE-BS unit direction vector in ECEF frame following paper convention.
    
    Paper convention:
    1. BS altitude is set to 0.
    2. UE altitude uses 'height_ue' (relative height from ground/BS plane), not absolute altitude.
    
    Args:
        df: Dataframe with GPS coordinates and height_ue
    
    Returns:
        Dataframe with added columns: u_ue_bs_x, u_ue_bs_y, u_ue_bs_z
    """
    df = df.copy()
    
    # Check if height_ue exists (crucial for this paper's method)
    if 'height_ue' not in df.columns:
        # Fallback if strictly necessary, but ideally should raise error for this specific dataset
        logger.warning("'height_ue' missing, falling back to 'altitude_ue' (May be incorrect per paper spec)")
        ue_alt_input = df['altitude_ue'].values
    else:
        # CORRECT: Use relative height as altitude per paper eq (12)
        ue_alt_input = df['height_ue'].values

    # Convert UE position to ECEF
    # Important: We feed 'height_ue' into the altitude parameter
    ue_x, ue_y, ue_z = geodetic_to_ecef(
        df['lat_ue'].values,
        df['lon_ue'].values,
        ue_alt_input 
    )
    
    # Convert BS position to ECEF 
    # Important: BS altitude is strictly 0 per paper
    bs_x, bs_y, bs_z = geodetic_to_ecef(
        df['lat_bs'].values,
        df['lon_bs'].values,
        np.zeros_like(df['lat_bs'].values) # Altitude = 0.0
    )
    
    # Compute UE-BS vector (r_UE-BS = r_UE - r_BS)
    vec_x = ue_x - bs_x
    vec_y = ue_y - bs_y
    vec_z = ue_z - bs_z
    
    # Normalize to unit vector
    norm = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
    
    # Avoid division by zero
    norm = np.where(norm < 1e-10, 1.0, norm)
    
    df['u_ue_bs_x'] = vec_x / norm
    df['u_ue_bs_y'] = vec_y / norm
    df['u_ue_bs_z'] = vec_z / norm
    
    return df


def remap_beam_to_M32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap beam indices from N=64 oversampled codebook to M=32 codebook.
    
    Paper (Section III): "codebook comprising 32 pre-defined beams (M=32)"
    DeepSense 6G Scenario 23 uses N=64 oversampled codebook.
    Downsampling: take every other beam -> powers_32 = powers_64[::2]
    New beam index: argmax(powers_32), range [0, 31]
    
    Args:
        df: DataFrame with 'beam_power' (64-element arrays) and 'beam_idx' columns
    
    Returns:
        DataFrame with updated 'beam_idx' (0-indexed, range [0, 31])
        and 'beam_power' (32-element arrays)
    """
    df = df.copy()
    
    new_beam_idx = []
    new_beam_power = []
    
    for i in range(len(df)):
        powers_64 = np.array(df.iloc[i]['beam_power'])
        powers_32 = powers_64[::2]  # Downsample: take every other beam
        beam_idx_32 = int(np.argmax(powers_32))  # 0-indexed in [0, 31]
        new_beam_idx.append(beam_idx_32)
        new_beam_power.append(powers_32)
    
    df['beam_idx'] = new_beam_idx
    df['beam_power'] = new_beam_power
    
    logger.info(f"Remapped beams: N=64 -> M=32")
    logger.info(f"  New beam_idx range: [{df['beam_idx'].min()}, {df['beam_idx'].max()}]")
    logger.info(f"  Unique beams: {df['beam_idx'].nunique()}")
    
    return df


def create_baseline_features(
    df: pd.DataFrame,
    norm_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create baseline feature set (position-only features)
    
    Features:
        - lat_ue_norm, lon_ue_norm: Normalized UE position
        - u_ue_bs_x, u_ue_bs_y, u_ue_bs_z: UE-BS unit vector
    
    Args:
        df: Input dataframe
        norm_params: Pre-computed normalization parameters (for val/test)
    
    Returns:
        Dataframe with baseline features
    """
    logger.info("Creating baseline features...")
    
    # Normalize geodetic coordinates
    if norm_params is None:
        df, norm_params = normalize_geodetic(df)
    else:
        # Apply existing normalization
        lat_min, lat_max = norm_params['lat']
        lon_min, lon_max = norm_params['lon']
        
        df = df.copy()
        df['lat_ue_norm'] = (df['lat_ue'] - lat_min) / (lat_max - lat_min)
        df['lon_ue_norm'] = (df['lon_ue'] - lon_min) / (lon_max - lon_min)
    
    # Compute UE-BS unit vector
    df = compute_ue_bs_unit_vector(df)
    
    logger.info("Baseline features created")
    
    return df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    norm_params: Optional[Dict] = None
):
    """
    Save processed data splits to disk
    
    Args:
        train_df, val_df, test_df: Data splits
        output_dir: Output directory
        norm_params: Normalization parameters to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_path}")
    
    train_df.to_pickle(output_path / 'train.pkl')
    val_df.to_pickle(output_path / 'val.pkl')
    test_df.to_pickle(output_path / 'test.pkl')
    
    if norm_params is not None:
        import json
        with open(output_path / 'norm_params.json', 'w') as f:
            json.dump(norm_params, f, indent=2)
    
    logger.info("Data saved successfully")


# Example usage
if __name__ == "__main__":
    # Load data
    loader = DeepSenseDataLoader("data/raw/scenario23")
    df = loader.load_raw_data()
    
    # Validate
    loader.validate_data()
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of sequences: {stats['num_sequences']}")
    
    # Split data
    train_df, val_df, test_df = adjusted_splitting(df, random_seed=42)
    
    # Create features
    train_df = create_baseline_features(train_df)
    val_df = create_baseline_features(val_df)
    test_df = create_baseline_features(test_df)
    
    # Save
    save_processed_data(train_df, val_df, test_df, "data/processed")
