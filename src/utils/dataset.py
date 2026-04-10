"""
PyTorch Dataset classes for beam prediction
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class BeamPredictionDataset(Dataset):
    """
    Dataset for beam prediction with sliding window
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        window_size: int = 8,
        num_predictions: int = 4,
        include_beam_powers: bool = False,
        include_speed_category: bool = False,
        pad_initial: bool = True,
    ):
        """
        Initialize dataset

        Args:
            df: Dataframe with features and targets
            feature_columns: List of feature column names
            window_size: Input window size (W)
            num_predictions: Number of future predictions (V+1)
            include_beam_powers: Whether to include beam power values
            include_speed_category: Whether to include speed category
            pad_initial: If True, zero-pad the first W-1 input timesteps of
                each sequence so that every position with V future targets is
                included (matches author's zero_pad_nonconsecutive=True).
                If False (default), only fully-consecutive windows are used.
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.num_predictions = num_predictions
        self.include_beam_powers = include_beam_powers
        self.include_speed_category = include_speed_category
        self.pad_initial = pad_initial
        self.pad_value = 0.0
        
        # Create samples
        self.samples = self._create_samples()
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
        logger.info(f"Feature dimension: {len(feature_columns)}")
        logger.info(f"Window size: {window_size}, Predictions: {num_predictions}")
    
    def _create_samples(self) -> List[Dict]:
        """
        Create sliding window samples from sequences
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        W = self.window_size
        V = self.num_predictions
        
        for seq_idx in self.df['sequence_idx'].unique():
            seq_df = self.df[self.df['sequence_idx'] == seq_idx].sort_values('sample_idx')
            L = len(seq_df)
            # Original requirement: need at least W + V - 1 samples to form
            # a full (unpadded) input window ending at t and V future targets.
            # If pad_initial is enabled we instead allow initial windows to be
            # zero-padded; sequences must still have at least V samples to
            # produce targets (paper drops sequences lacking future samples).
            if not self.pad_initial:
                if L < W + V - 1:
                    continue
                # Sliding windows without padding
                sample_indices = seq_df['sample_idx'].values
                for i in range(L - W - V + 2):
                    # Input window: [t-W+1, ..., t]
                    window_df = seq_df.iloc[i:i+W]

                    # Targets: v=0 = current beam (t), v=1..V-1 = future
                    future_df = seq_df.iloc[i+W-1:i+W-1+V]

                    # Skip windows spanning non-consecutive sample indices
                    # (can occur at split boundaries after adjusted splitting)
                    window_indices = sample_indices[i:i+W+V-1]
                    if not np.all(np.diff(window_indices) == 1):
                        continue

                    features = window_df[self.feature_columns].values.astype(np.float32)
                    beams = future_df['beam_idx'].values.astype(np.int64)

                    sample = {
                        'features': features,
                        'beams': beams,
                        'sequence_idx': seq_idx,
                        'start_idx': i
                    }

                    # Add beam powers if requested
                    if self.include_beam_powers:
                        beam_powers = np.stack(future_df['beam_power'].values).astype(np.float32)
                        sample['beam_powers'] = beam_powers

                    # Add speed category if requested
                    if self.include_speed_category:
                        if 'speed_category' in window_df.columns:
                            sample['speed_category'] = int(window_df.iloc[-1]['speed_category'])
                        elif 'speed_mps' in window_df.columns:
                            # Use last (most recent) speed in window, matching author's approach
                            v_mag = window_df.iloc[-1]['speed_mps']
                            if pd.isna(v_mag):
                                v_mag = 0.0
                            speed_mph = v_mag * 2.23694
                            if speed_mph <= 10:
                                sample['speed_category'] = 0
                            elif speed_mph <= 20:
                                sample['speed_category'] = 1
                            else:
                                sample['speed_category'] = 2
                        elif 'v_mag' in window_df.columns:
                            v_mag = window_df.iloc[-1]['v_mag']
                            speed_mph = v_mag * 2.23694
                            if speed_mph <= 10:
                                sample['speed_category'] = 0
                            elif speed_mph <= 20:
                                sample['speed_category'] = 1
                            else:
                                sample['speed_category'] = 2
                        else:
                            sample['speed_category'] = 0

                    samples.append(sample)
            else:
                # Pad-initial mode: allow initial windows by zero-padding missing
                # earlier timesteps. Require at least V samples to provide
                # future targets.
                if L < V:
                    continue
                feature_dim = len(self.feature_columns)
                sample_indices = seq_df['sample_idx'].values
                for t in range(0, L - V + 1):
                    # OUTPUT must be consecutive (labels must be correct)
                    out_indices = sample_indices[t:t + V]
                    if not np.all(np.diff(out_indices) == 1):
                        continue

                    # Find the start of the consecutive run in the INPUT ending
                    # at t. Scan backwards up to W-1 steps; stop at any gap.
                    limit = max(0, t - W + 1)
                    real_start = t
                    while real_start > limit:
                        if sample_indices[real_start] - sample_indices[real_start - 1] != 1:
                            break
                        real_start -= 1

                    real_count = t - real_start + 1
                    pad_count = W - real_count

                    real_part = seq_df.iloc[real_start:t + 1][self.feature_columns].values.astype(np.float32)
                    if pad_count > 0:
                        pad_array = np.full((pad_count, feature_dim), self.pad_value, dtype=np.float32)
                        features = np.vstack([pad_array, real_part])
                    else:
                        features = real_part

                    future_df = seq_df.iloc[t:t + V]
                    beams = future_df['beam_idx'].values.astype(np.int64)

                    sample = {
                        'features': features,
                        'beams': beams,
                        'sequence_idx': seq_idx,
                        'start_idx': max(0, t - W + 1)
                    }

                    if self.include_beam_powers:
                        beam_powers = np.stack(future_df['beam_power'].values).astype(np.float32)
                        sample['beam_powers'] = beam_powers

                    if self.include_speed_category:
                        # Use current (last real) element in window, matching author's approach
                        if 'speed_category' in seq_df.columns:
                            sample['speed_category'] = int(seq_df.iloc[t]['speed_category'])
                        elif 'speed_mps' in seq_df.columns:
                            v_mag = seq_df.iloc[t]['speed_mps']
                            if pd.isna(v_mag):
                                v_mag = 0.0
                            speed_mph = v_mag * 2.23694
                            if speed_mph <= 10:
                                sample['speed_category'] = 0
                            elif speed_mph <= 20:
                                sample['speed_category'] = 1
                            else:
                                sample['speed_category'] = 2
                        elif 'v_mag' in seq_df.columns:
                            v_mag = seq_df.iloc[t]['v_mag']
                            speed_mph = v_mag * 2.23694
                            if speed_mph <= 10:
                                sample['speed_category'] = 0
                            elif speed_mph <= 20:
                                sample['speed_category'] = 1
                            else:
                                sample['speed_category'] = 2
                        else:
                            sample['speed_category'] = 0

                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of tensors (features, beams, [speed_category], [beam_powers])
        """
        sample = self.samples[idx]
        
        features = torch.from_numpy(sample['features'])
        beams = torch.from_numpy(sample['beams'])
        
        outputs = [features, beams]
        
        if self.include_speed_category:
            speed_cat = torch.tensor(sample['speed_category'], dtype=torch.long)
            outputs.append(speed_cat)
        
        if self.include_beam_powers:
            beam_powers = torch.from_numpy(sample['beam_powers'])
            outputs.append(beam_powers)
        
        return tuple(outputs)
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        return {
            'sequence_idx': self.samples[idx]['sequence_idx'],
            'start_idx': self.samples[idx]['start_idx']
        }


class SequenceBeamDataset(Dataset):
    """
    Dataset that returns full sequences (for sequence-level evaluation)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        min_sequence_length: int = 20
    ):
        """
        Initialize sequence dataset
        
        Args:
            df: Dataframe with features
            feature_columns: Feature column names
            min_sequence_length: Minimum sequence length to include
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.min_sequence_length = min_sequence_length
        
        # Get valid sequences
        self.sequences = self._get_sequences()
        
        logger.info(f"Created sequence dataset with {len(self.sequences)} sequences")
    
    def _get_sequences(self) -> List[int]:
        """Get list of valid sequence IDs"""
        sequences = []
        
        for seq_idx in self.df['sequence_idx'].unique():
            seq_df = self.df[self.df['sequence_idx'] == seq_idx]
            
            if len(seq_df) >= self.min_sequence_length:
                sequences.append(seq_idx)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a full sequence
        
        Returns:
            features: [seq_len, feature_dim]
            beams: [seq_len]
        """
        seq_idx = self.sequences[idx]
        seq_df = self.df[self.df['sequence_idx'] == seq_idx].sort_values('sample_idx')
        
        features = torch.from_numpy(
            seq_df[self.feature_columns].values.astype(np.float32)
        )
        beams = torch.from_numpy(
            seq_df['beam_idx'].values.astype(np.int64)
        )
        
        return features, beams


def build_speed_labels(
    df: pd.DataFrame,
    dataset: BeamPredictionDataset,
    window_size: int = 8,
    num_predictions: int = 4,
    pad_initial: bool = True,
) -> np.ndarray:
    """
    Build speed labels aligned with the sliding-window dataset.

    Replicates the EXACT same sliding-window iteration as
    BeamPredictionDataset._create_samples (respecting pad_initial), so the
    resulting label array is guaranteed to have the same length as the dataset.

    Speed is taken from the LAST (most recent) real element in each window,
    matching the author's approach (input_speed[-1] * 2.23694).

    Thresholds (matching author — convert to mph, then use <=):
        Slow:   speed_mph <= 10
        Medium: 10 < speed_mph <= 20
        Fast:   speed_mph > 20

    Args:
        df: DataFrame used to create the dataset (must have speed_mps or v_mag)
        dataset: The BeamPredictionDataset instance (used only for length check)
        window_size: Input window size (W)
        num_predictions: Number of future predictions (V)
        pad_initial: Must match the pad_initial value used when creating the dataset.

    Returns:
        np.ndarray of int64 speed labels (0=Slow, 1=Medium, 2=Fast)
    """
    speed_col = None
    if 'speed_mps' in df.columns:
        speed_col = 'speed_mps'
    elif 'v_mag' in df.columns:
        speed_col = 'v_mag'

    if speed_col is None:
        logger.warning("No speed column found (speed_mps or v_mag). "
                       "Speed category evaluation skipped.")
        return None

    speed_labels = []
    W = window_size
    V = num_predictions

    for seq_idx in df['sequence_idx'].unique():
        seq_df = df[df['sequence_idx'] == seq_idx].sort_values('sample_idx')
        L = len(seq_df)

        if pad_initial:
            # Mirror _create_samples pad_initial branch exactly:
            # output must be consecutive; input is padded at gaps.
            if L < V:
                continue
            sample_indices = seq_df['sample_idx'].values
            for t in range(0, L - V + 1):
                # Skip if output samples are not consecutive (same check as _create_samples)
                out_indices = sample_indices[t:t + V]
                if not np.all(np.diff(out_indices) == 1):
                    continue
                v_mag = seq_df.iloc[t][speed_col]
                if pd.isna(v_mag):
                    v_mag = 0.0
                speed_mph = v_mag * 2.23694
                if speed_mph <= 10:
                    speed_labels.append(0)
                elif speed_mph <= 20:
                    speed_labels.append(1)
                else:
                    speed_labels.append(2)
        else:
            # Mirror _create_samples non-padded branch:
            # only fully-consecutive windows of length W+V-1.
            if L < W + V - 1:
                continue
            sample_indices = seq_df['sample_idx'].values
            for i in range(L - W - V + 2):
                # Skip non-consecutive windows (same check as _create_samples)
                window_indices = sample_indices[i:i + W + V - 1]
                if not np.all(np.diff(window_indices) == 1):
                    continue

                # Last real input is at position i + W - 1
                v_mag = seq_df.iloc[i + W - 1][speed_col]
                if pd.isna(v_mag):
                    v_mag = 0.0

                speed_mph = v_mag * 2.23694
                if speed_mph <= 10:
                    speed_labels.append(0)
                elif speed_mph <= 20:
                    speed_labels.append(1)
                else:
                    speed_labels.append(2)

    speed_labels = np.array(speed_labels, dtype=np.int64)

    # Verify alignment with dataset
    expected_len = len(dataset)
    if len(speed_labels) != expected_len:
        logger.warning(
            f"Speed labels count ({len(speed_labels)}) != "
            f"dataset size ({expected_len}). "
            f"Truncating/padding to match."
        )
        if len(speed_labels) > expected_len:
            speed_labels = speed_labels[:expected_len]
        else:
            speed_labels = np.pad(
                speed_labels,
                (0, expected_len - len(speed_labels)),
                constant_values=0
            )

    logger.info(
        f"Speed label distribution: "
        f"Slow={np.sum(speed_labels == 0)}, "
        f"Medium={np.sum(speed_labels == 1)}, "
        f"Fast={np.sum(speed_labels == 2)}"
    )
    return speed_labels


# Test code
if __name__ == "__main__":
    print("Dataset module loaded successfully")
