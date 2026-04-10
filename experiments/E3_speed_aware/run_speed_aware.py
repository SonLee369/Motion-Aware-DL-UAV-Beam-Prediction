"""
E3: Speed-Aware Training Comparison
Compare different speed-aware training strategies
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from src.features.preprocessing import DeepSenseDataLoader, adjusted_splitting
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.models.enhanced import SpeedConditionedBeamPredictor
from src.models.baseline import BaselineBeamPredictor
from src.training.trainer import Trainer
from src.training.losses import SpeedWeightedLoss
from src.training.samplers import SpeedStratifiedSampler, BalancedBatchSampler, StratifiedNoReplaceSampler
from src.evaluation.metrics import evaluate_model, evaluate_by_speed_category
from src.evaluation.report import generate_e3_report
from src.utils.dataset import BeamPredictionDataset
from torch.utils.data import Subset

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E3_speed_aware.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _mean_std(values: List[float]) -> tuple:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) > 0 else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_speed_results(speed_results_list: List[Dict]) -> Dict:
    if not speed_results_list:
        return None

    aggregated = {}
    for speed_name in ['slow', 'medium', 'fast']:
        per_seed = [m for m in speed_results_list if m and speed_name in m]
        if not per_seed:
            continue

        aggregated[speed_name] = {}
        v_keys = [k for k in per_seed[0].keys() if k.startswith('v')]
        for v_key in v_keys:
            aggregated[speed_name][v_key] = {}
            for metric_name in per_seed[0][v_key].keys():
                vals = [m[speed_name][v_key][metric_name] for m in per_seed]
                mean, std = _mean_std(vals)
                aggregated[speed_name][v_key][metric_name] = mean
                aggregated[speed_name][v_key][f'{metric_name}_std'] = std

        aggregated[speed_name]['num_samples'] = int(np.mean([
            m[speed_name].get('num_samples', 0) for m in per_seed
        ]))

        aggregated[speed_name]['overall'] = {}
        for metric_name in per_seed[0][speed_name]['overall'].keys():
            vals = [m[speed_name]['overall'][metric_name] for m in per_seed]
            mean, std = _mean_std(vals)
            aggregated[speed_name]['overall'][metric_name] = mean
            aggregated[speed_name]['overall'][f'{metric_name}_std'] = std

    return aggregated


def aggregate_strategy_results(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    base = seed_results[0]
    aggregated = {
        'strategy': base.get('strategy'),
        'seed_stats': {
            'seeds': seeds,
            'count': len(seeds)
        },
        'training_history': base.get('training_history'),
        'training_histories': [r.get('training_history') for r in seed_results]
    }

    v_keys = [k for k in base.keys() if k.startswith('v')]
    for v_key in v_keys:
        aggregated[v_key] = {}
        for metric_name in base[v_key].keys():
            vals = [r[v_key][metric_name] for r in seed_results]
            mean, std = _mean_std(vals)
            aggregated[v_key][metric_name] = mean
            aggregated[v_key][f'{metric_name}_std'] = std

    aggregated['overall'] = {}
    for metric_name in base['overall'].keys():
        vals = [r['overall'][metric_name] for r in seed_results]
        mean, std = _mean_std(vals)
        aggregated['overall'][metric_name] = mean
        aggregated['overall'][f'{metric_name}_std'] = std

    aggregated['by_speed'] = aggregate_speed_results([r.get('by_speed') for r in seed_results])

    return aggregated


def compute_speed_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute speed categories for samples
    
    Speed categories:
    - 0 (Slow): v < 4.47 m/s (< 10 mph)
    - 1 (Medium): 4.47 <= v < 8.94 m/s (10-20 mph)
    - 2 (Fast): v >= 8.94 m/s (>= 20 mph)
    
    Args:
        df: Dataframe with velocity features
    
    Returns:
        Dataframe with speed_category column
    """
    df = df.copy()
    
    def categorize_speed(v_mag):
        if pd.isna(v_mag):
            return 0  # Default to slow
        elif v_mag < 4.47:
            return 0  # Slow
        elif v_mag < 8.94:
            return 1  # Medium
        else:
            return 2  # Fast
    
    df['speed_category'] = df['v_mag'].apply(categorize_speed)
    
    # Log distribution
    dist = df['speed_category'].value_counts().sort_index()
    logger.info("Speed category distribution:")
    for cat, count in dist.items():
        pct = count / len(df) * 100
        logger.info(f"  Category {cat}: {count} samples ({pct:.1f}%)")
    
    return df


def prepare_data_with_speed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple:
    """
    Prepare data with speed categories and velocity features
    
    Returns:
        train_df, val_df, test_df with speed categories
    """
    logger.info("="*70)
    logger.info("Preparing Data with Speed Categories")
    logger.info("="*70)
    
    # Add velocity features
    logger.info("\nComputing velocity features...")
    train_df = add_velocity_features(train_df, smooth=True, window_size=5, poly_order=2)
    val_df = add_velocity_features(val_df, smooth=True, window_size=5, poly_order=2)
    test_df = add_velocity_features(test_df, smooth=True, window_size=5, poly_order=2)
    
    # Normalize velocity
    train_df, v_norm_params = normalize_velocity_features(train_df, v_max=30.0)
    val_df, _ = normalize_velocity_features(val_df, v_max=v_norm_params['v_max'], fit_on=train_df)
    test_df, _ = normalize_velocity_features(test_df, v_max=v_norm_params['v_max'], fit_on=train_df)
    
    # Compute speed categories
    logger.info("\nComputing speed categories...")
    train_df = compute_speed_categories(train_df)
    val_df = compute_speed_categories(val_df)
    test_df = compute_speed_categories(test_df)
    
    return train_df, val_df, test_df


def _make_loaders(train_dataset, val_dataset, test_dataset,
                  batch_size=8, sampler=None):
    """Create DataLoaders (num_workers=0 to avoid Windows paging file issues)."""
    train_kw = dict(batch_size=batch_size, num_workers=0,
                    pin_memory=True)
    if sampler is not None:
        train_kw['sampler'] = sampler
    else:
        train_kw['shuffle'] = True
    eval_kw = dict(batch_size=1024, shuffle=False, num_workers=0,
                   pin_memory=True)
    return (DataLoader(train_dataset, **train_kw),
            DataLoader(val_dataset, **eval_kw),
            DataLoader(test_dataset, **eval_kw))


def _get_speed_labels_from_dataset(dataset) -> np.ndarray:
    """Extract speed_category labels from dataset samples."""
    return np.array([s['speed_category'] for s in dataset.samples], dtype=np.int64)


def run_E3_0_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.0: Baseline training — standard training, no speed-aware techniques
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.0: Baseline Training (No Speed Awareness)")
    logger.info("="*70)

    set_random_seed(seed)
    
    # Feature columns (position + velocity)
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets (include_speed_category for per-speed evaluation)
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8)
    
    # Create model (same BaselineBeamPredictor as E1 — isolate training strategy effects)
    model = BaselineBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    save_dir = f'checkpoints/E3_speed_aware/E3_0_baseline/seed_{seed}'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_0_baseline/seed_{seed}',
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    
    # Evaluate overall
    results = evaluate_model(model, test_loader, device=device)
    
    # Evaluate by speed category
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.0: Baseline'
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def run_E3_3_weighted_loss(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.3: Speed-weighted loss training — higher loss weight for fast-speed samples
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.3: Speed-Weighted Loss Training")
    logger.info("="*70)

    set_random_seed(seed)
    
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets (with speed category)
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8)
    
    # Create model (same BaselineBeamPredictor as E1)
    model = BaselineBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    # Speed-weighted loss
    # Higher weight for faster speeds (more challenging)
    speed_weights = {0: 1.0, 1: 1.5, 2: 2.0}
    criterion = SpeedWeightedLoss(speed_weights=speed_weights)
    
    logger.info(f"Speed weights: {speed_weights}")
    
    # Custom trainer to handle speed categories in loss
    class SpeedWeightedTrainer(Trainer):
        def train_epoch(self, epoch: int) -> float:
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            from tqdm import tqdm
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch_idx, batch_data in enumerate(pbar):
                features, beams, speed_cats = batch_data
                features = features.to(self.device)
                beams = beams.to(self.device)
                speed_cats = speed_cats.to(self.device)
                
                # Forward
                output = self.model(features)
                if isinstance(output, tuple):
                    predictions, _ = output
                else:
                    predictions = output
                
                # Loss with speed weighting
                loss = self.criterion(predictions, beams, speed_cats)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            return total_loss / num_batches

        def validate(self, epoch: int):
            """Override validate to pass speed_cats to SpeedWeightedLoss."""
            self.model.eval()
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_data in self.val_loader:
                    features, beams, speed_cats = batch_data
                    features = features.to(self.device)
                    beams = beams.to(self.device)
                    speed_cats = speed_cats.to(self.device)

                    output = self.model(features, speed_cats)
                    if isinstance(output, tuple):
                        predictions, _ = output
                    else:
                        predictions = output

                    loss = self.criterion(predictions, beams, speed_cats)
                    total_loss += loss.item()
                    num_batches += 1

                    for v in range(predictions.size(1)):
                        pred_beams = predictions[:, v, :].argmax(dim=1)
                        correct += (pred_beams == beams[:, v]).sum().item()
                    total += beams.size(0) * predictions.size(1)

            avg_loss = total_loss / num_batches
            accuracy = correct / total
            return avg_loss, accuracy
    
    # Train
    save_dir = f'checkpoints/E3_speed_aware/E3_3_weighted_loss/seed_{seed}'
    trainer = SpeedWeightedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_3_weighted_loss/seed_{seed}',
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)
    
    # Evaluate by speed
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.3: Weighted Loss'
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def run_E3_1_stratified_sampling(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.1: Stratified sampling — balanced sampling across speed categories
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.1: Stratified Sampling Training")
    logger.info("="*70)

    set_random_seed(seed)
    
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Get speed labels for training samples
    train_speed_labels = np.array([sample['speed_category'] for sample in train_dataset.samples])
    
    # Create stratified sampler
    sampler = SpeedStratifiedSampler(
        speed_labels=train_speed_labels,
        target_distribution={0: 0.33, 1: 0.33, 2: 0.34},
        shuffle=True,
        seed=seed
    )
    
    # Create dataloaders with stratified sampler
    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8, sampler=sampler)
    
    # Create model (same BaselineBeamPredictor as E1)
    model = BaselineBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    # Train
    save_dir = f'checkpoints/E3_speed_aware/E3_1_stratified/seed_{seed}'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_1_stratified/seed_{seed}',
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)
    
    # Evaluate by speed
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.1: Stratified Sampling'
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def run_E3_2_speed_conditioned(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.2: Speed-conditioned model — speed embedding concatenated to features
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.2: Speed-Conditioned Model Training")
    logger.info("="*70)

    set_random_seed(seed)
    
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8)
    
    # Create speed-conditioned model
    model = SpeedConditionedBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        speed_embedding_dim=16,
        dropout=0.1
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    save_dir = f'checkpoints/E3_speed_aware/E3_2_conditioned/seed_{seed}'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_2_conditioned/seed_{seed}',
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)
    
    # Evaluate by speed
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.2: Speed-Conditioned'
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def run_E3_4_curriculum_learning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.4: Curriculum learning — train slow → medium → all speeds
    
    Phases:
        Phase 1 (epochs 0..n1-1):  slow samples only   (category 0)
        Phase 2 (epochs n1..n2-1): slow + medium        (categories 0,1)
        Phase 3 (epochs n2..end):  all speeds            (categories 0,1,2)
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.4: Curriculum Learning (slow → medium → all)")
    logger.info("="*70)

    set_random_seed(seed)
    
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Full training dataset (with speed category)
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Build speed labels and phase-specific subsets
    train_speed = _get_speed_labels_from_dataset(train_dataset)
    slow_idx = np.where(train_speed == 0)[0].tolist()
    slow_med_idx = np.where(train_speed <= 1)[0].tolist()
    all_idx = list(range(len(train_dataset)))
    
    logger.info(f"Curriculum phases: slow={len(slow_idx)}, slow+med={len(slow_med_idx)}, all={len(all_idx)}")
    
    # Phase boundaries (split num_epochs into 3 roughly equal phases)
    n1 = num_epochs // 3
    n2 = 2 * num_epochs // 3
    phase_plan = [
        (0,  n1, slow_idx,     "Phase 1: slow only"),
        (n1, n2, slow_med_idx, "Phase 2: slow + medium"),
        (n2, num_epochs, all_idx, "Phase 3: all speeds"),
    ]
    
    # Shared val/test loaders (num_workers=0 to avoid Windows paging file issues)
    eval_kw = dict(batch_size=1024, shuffle=False, num_workers=0,
                   pin_memory=True)
    val_loader = DataLoader(val_dataset, **eval_kw)
    test_loader = DataLoader(test_dataset, **eval_kw)
    
    # Create model (same BaselineBeamPredictor as E1)
    model = BaselineBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    save_dir = f'checkpoints/E3_speed_aware/E3_4_curriculum/seed_{seed}'
    log_dir = f'runs/E3_speed_aware/E3_4_curriculum/seed_{seed}'
    
    # Train each phase, carrying over model weights
    all_train_losses = []
    all_val_losses = []
    all_val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    for phase_start, phase_end, indices, phase_name in phase_plan:
        phase_epochs = phase_end - phase_start
        if phase_epochs <= 0:
            continue
        logger.info(f"\n--- {phase_name} (epochs {phase_start+1}-{phase_end}) ---")
        
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            subset, batch_size=8, shuffle=True,
            num_workers=0, pin_memory=True
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            learning_rate=5e-4,
            num_epochs=phase_epochs,
            save_dir=save_dir,
            log_dir=log_dir,
            early_stopping_patience=999  # No early stop within phase
        )
        # Carry over best tracking
        trainer.best_val_acc = best_val_acc
        trainer.best_epoch = best_epoch
        
        trainer.train()
        
        history = trainer.get_training_history()
        all_train_losses.extend(history['train_losses'])
        all_val_losses.extend(history['val_losses'])
        all_val_accs.extend(history['val_accuracies'])
        
        if history['best_val_acc'] > best_val_acc:
            best_val_acc = history['best_val_acc']
            best_epoch = phase_start + history['best_epoch']
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)
    
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.4: Curriculum Learning'
    results['training_history'] = {
        'train_losses': all_train_losses,
        'val_losses': all_val_losses,
        'val_accuracies': all_val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }
    results['seed'] = seed
    
    return results


def run_E3_5_combined(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.5: Combined strategy — all techniques together
    
    Combines:
        - SpeedConditionedBeamPredictor  (architecture, speed embedding)
        - SpeedStratifiedSampler         (balanced sampling)
        - SpeedWeightedLoss              (higher weight for fast-speed)
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("E3.5: Combined (all strategies)")
    logger.info("="*70)

    set_random_seed(seed)
    
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets (with speed category for both model & loss)
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Stratified sampler (E3.1)
    train_speed_labels = _get_speed_labels_from_dataset(train_dataset)
    sampler = SpeedStratifiedSampler(
        speed_labels=train_speed_labels,
        target_distribution={0: 0.33, 1: 0.33, 2: 0.34},
        shuffle=True,
        seed=seed
    )
    
    # Create dataloaders with stratified sampler
    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8, sampler=sampler)
    
    # Speed-conditioned model (E3.2)
    model = SpeedConditionedBeamPredictor(
        input_dim=len(feature_columns),
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        speed_embedding_dim=16,
        dropout=0.1
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Speed-weighted loss (E3.3)
    speed_weights = {0: 1.0, 1: 1.5, 2: 2.0}
    criterion = SpeedWeightedLoss(speed_weights=speed_weights)
    
    # Custom trainer: speed-conditioned model + speed-weighted loss
    # Trainer already passes speed_cats to model when present (3-element batch),
    # but SpeedWeightedLoss expects the full 3D tensor, so override train_epoch.
    class CombinedTrainer(Trainer):
        def train_epoch(self, epoch: int) -> float:
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            from tqdm import tqdm
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch_idx, batch_data in enumerate(pbar):
                features, beams, speed_cats = batch_data
                features = features.to(self.device)
                beams = beams.to(self.device)
                speed_cats = speed_cats.to(self.device)
                
                # Forward (SpeedConditionedBeamPredictor uses speed_cats)
                predictions = self.model(features, speed_cats)
                
                # Loss (SpeedWeightedLoss uses speed_cats)
                loss = self.criterion(predictions, beams, speed_cats)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            return total_loss / num_batches

        def validate(self, epoch: int):
            """Override validate to pass speed_cats to SpeedWeightedLoss."""
            self.model.eval()
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_data in self.val_loader:
                    features, beams, speed_cats = batch_data
                    features = features.to(self.device)
                    beams = beams.to(self.device)
                    speed_cats = speed_cats.to(self.device)

                    predictions = self.model(features, speed_cats)

                    loss = self.criterion(predictions, beams, speed_cats)
                    total_loss += loss.item()
                    num_batches += 1

                    for v in range(predictions.size(1)):
                        pred_beams = predictions[:, v, :].argmax(dim=1)
                        correct += (pred_beams == beams[:, v]).sum().item()
                    total += beams.size(0) * predictions.size(1)

            avg_loss = total_loss / num_batches
            accuracy = correct / total
            return avg_loss, accuracy
    
    # Train
    save_dir = f'checkpoints/E3_speed_aware/E3_5_combined/seed_{seed}'
    trainer = CombinedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_5_combined/seed_{seed}',
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)
    
    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    
    results['by_speed'] = results_by_speed
    results['strategy'] = 'E3.5: Combined'
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def run_E3_6_no_replace(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    device: str = 'cuda',
    num_epochs: int = 20
) -> Dict:
    """
    E3.6: Stratified Sampling Without Replacement

    Fix for E3.1's root cause: SpeedStratifiedSampler with replace=True causes
    the minority (fast) class to be repeated ~11x per epoch, leading to
    memorisation instead of generalisation.

    This variant uses StratifiedNoReplaceSampler:
      - epoch_size = 3 × n_fast  (no sample repeated within an epoch)
      - num_epochs scaled so total gradient steps ≈ baseline E3.0
    """
    logger.info("\n" + "="*70)
    logger.info("E3.6: Stratified Sampling Without Replacement")
    logger.info("="*70)

    set_random_seed(seed)

    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']

    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean   = val_df.dropna(subset=feature_columns)
    test_df_clean  = test_df.dropna(subset=feature_columns)

    train_dataset = BeamPredictionDataset(
        train_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4, include_speed_category=True
    )
    val_dataset = BeamPredictionDataset(
        val_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4, include_speed_category=True
    )
    test_dataset = BeamPredictionDataset(
        test_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4, include_speed_category=True
    )

    train_speed_labels = _get_speed_labels_from_dataset(train_dataset)
    sampler = StratifiedNoReplaceSampler(
        speed_labels=train_speed_labels, shuffle=True, seed=seed
    )

    # Scale epochs so total gradient steps ≈ standard training
    original_epoch_size = len(train_dataset)
    reduced_epoch_size  = len(sampler)          # 3 × n_minority
    scale = original_epoch_size / max(reduced_epoch_size, 1)
    scaled_epochs = min(int(round(num_epochs * scale)), 300)
    logger.info(f"Epoch scaling: {original_epoch_size} → {reduced_epoch_size} samples/epoch "
                f"→ running {scaled_epochs} epochs (target steps ≈ {num_epochs}×baseline)")

    train_loader, val_loader, test_loader = _make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=8, sampler=sampler
    )

    model = BaselineBeamPredictor(
        input_dim=len(feature_columns), hidden_dim=128,
        num_beams=32, num_predictions=4, dropout=0.1
    )
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    save_dir = f'checkpoints/E3_speed_aware/E3_6_no_replace/seed_{seed}'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        num_epochs=scaled_epochs,
        save_dir=save_dir,
        log_dir=f'runs/E3_speed_aware/E3_6_no_replace/seed_{seed}',
        early_stopping_patience=999
    )

    trainer.train()

    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    results = evaluate_model(model, test_loader, device=device)

    speed_labels = _get_speed_labels_from_dataset(test_dataset)
    results_by_speed = evaluate_by_speed_category(model, test_loader, speed_labels, device)

    results['by_speed']        = results_by_speed
    results['strategy']        = 'E3.6: No-Replace Stratified'
    results['training_history'] = trainer.get_training_history()
    results['seed']            = seed
    results['scaled_epochs']   = scaled_epochs

    return results


def compare_strategies(all_results: Dict) -> pd.DataFrame:
    """Compare results across strategies"""
    
    logger.info("\n" + "="*70)
    logger.info("Comparing Speed-Aware Strategies")
    logger.info("="*70)
    
    comparison_data = []
    
    for strategy_name, results in all_results.items():
        row = {
            'Strategy': results['strategy']
        }
        
        # Overall metrics
        row['Overall_Top1'] = results['overall']['top1_acc'] * 100
        row['Overall_Top3'] = results['overall']['top3_acc'] * 100
        row['Overall_Top1_std'] = results['overall'].get('top1_acc_std', 0.0) * 100
        row['Overall_Top3_std'] = results['overall'].get('top3_acc_std', 0.0) * 100
        
        # By speed category
        if 'by_speed' in results:
            for speed_name in ['slow', 'medium', 'fast']:
                if speed_name in results['by_speed']:
                    row[f'{speed_name.capitalize()}_Top1'] = results['by_speed'][speed_name]['overall']['top1_acc'] * 100
                    row[f'{speed_name.capitalize()}_Top1_std'] = results['by_speed'][speed_name]['overall'].get('top1_acc_std', 0.0) * 100
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate improvement over baseline
    baseline_overall = df.iloc[0]['Overall_Top1']
    df['Overall_Improvement'] = df['Overall_Top1'] - baseline_overall
    
    # Calculate speed category improvements
    for speed in ['Slow', 'Medium', 'Fast']:
        col = f'{speed}_Top1'
        if col in df.columns:
            baseline_val = df.iloc[0][col]
            df[f'{speed}_Improvement'] = df[col] - baseline_val
    
    logger.info("\n" + df.to_string())
    
    return df


def generate_latex_table(comparison_df: pd.DataFrame) -> str:
    """Generate LaTeX comparison table"""
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{Speed-Aware Training Strategy Comparison}
\label{tab:e3_speed_aware}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Strategy}} & \multicolumn{3}{c}{\textbf{Top-1 Accuracy by Speed (\%)}} & \multirow{2}{*}{\textbf{Overall}} & \multirow{2}{*}{\textbf{Improvement}} \\
\cmidrule(lr){2-4}
& \textbf{Slow} & \textbf{Medium} & \textbf{Fast} & & \\
\midrule
"""
    
    def fmt(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}"

    for _, row in comparison_df.iterrows():
        strategy = row['Strategy'].replace('E3.0:', '').replace('E3.1:', '').replace('E3.2:', '').replace('E3.3:', '').replace('E3.4:', '').replace('E3.5:', '').strip()
        latex += f"{strategy} & "
        
        if 'Slow_Top1' in row:
            latex += (
                f"{fmt(row['Slow_Top1'], row.get('Slow_Top1_std', 0.0))} & "
                f"{fmt(row['Medium_Top1'], row.get('Medium_Top1_std', 0.0))} & "
                f"{fmt(row['Fast_Top1'], row.get('Fast_Top1_std', 0.0))} & "
            )
        else:
            latex += "-- & -- & -- & "
        
        latex += f"{fmt(row['Overall_Top1'], row.get('Overall_Top1_std', 0.0))} & "
        latex += f"{row['Overall_Improvement']:+.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def plot_comparison(comparison_df: pd.DataFrame, save_dir: Path):
    """Plot comparison visualizations"""
    
    # Plot 1: Overall accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = comparison_df['Strategy'].str.replace(r'E3\.\d+:\s*', '', regex=True)
    x = np.arange(len(strategies))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Overall_Top1'], width, 
           label='Top-1', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Overall_Top3'], width,
           label='Top-3', color='coral', alpha=0.8)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Speed-Aware Training: Overall Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E3_overall_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E3_overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Accuracy by speed category
    if all(col in comparison_df.columns for col in ['Slow_Top1', 'Medium_Top1', 'Fast_Top1']):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(strategies))
        width = 0.25
        
        ax.bar(x - width, comparison_df['Slow_Top1'], width, 
               label='Slow', color='lightblue', alpha=0.8)
        ax.bar(x, comparison_df['Medium_Top1'], width,
               label='Medium', color='orange', alpha=0.8)
        ax.bar(x + width, comparison_df['Fast_Top1'], width,
               label='Fast', color='red', alpha=0.8)
        
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('Speed-Aware Training: Accuracy by Speed Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'E3_by_speed.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'E3_by_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Improvement heatmap
    if all(col in comparison_df.columns for col in ['Slow_Improvement', 'Medium_Improvement', 'Fast_Improvement']):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        heatmap_data = comparison_df[['Slow_Improvement', 'Medium_Improvement', 'Fast_Improvement', 'Overall_Improvement']].values
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    xticklabels=['Slow', 'Medium', 'Fast', 'Overall'],
                    yticklabels=strategies,
                    cbar_kws={'label': 'Improvement (%)'},
                    ax=ax)
        
        ax.set_title('Speed-Aware Training: Improvement over Baseline', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'E3_improvement_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'E3_improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def main():
    """Main execution function"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategies', type=str, default=None,
        help='Comma-separated strategy keys to run, e.g. E3_0,E3_6 (default: all)'
    )
    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("E3: SPEED-AWARE TRAINING COMPARISON")
    logger.info("="*70 + "\n")

    set_random_seed(42)

    config = {
        'data_path': 'data/raw/scenario23',
        'num_epochs': 20,
        'seeds': [42, 77, 78, 79, 80],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Mapping: key → (function, display name)
    strategy_registry = [
        ('E3_0', run_E3_0_baseline,            'E3.0: Baseline'),
        ('E3_1', run_E3_1_stratified_sampling,  'E3.1: Stratified Sampling'),
        ('E3_2', run_E3_2_speed_conditioned,    'E3.2: Speed-Conditioned'),
        ('E3_3', run_E3_3_weighted_loss,         'E3.3: Weighted Loss'),
        ('E3_4', run_E3_4_curriculum_learning,   'E3.4: Curriculum Learning'),
        ('E3_5', run_E3_5_combined,              'E3.5: Combined'),
        ('E3_6', run_E3_6_no_replace,            'E3.6: No-Replace Stratified'),
    ]

    if args.strategies is not None:
        allowed = set(args.strategies.split(','))
        strategy_registry = [(k, fn, name) for k, fn, name in strategy_registry if k in allowed]
        logger.info(f"Running subset: {[k for k, _, _ in strategy_registry]}")

    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        train_df = pd.read_pickle('data/processed/E1_baseline/train.pkl')
        val_df = pd.read_pickle('data/processed/E1_baseline/val.pkl')
        test_df = pd.read_pickle('data/processed/E1_baseline/test.pkl')
        
        # Prepare data with speed categories
        train_df, val_df, test_df = prepare_data_with_speed(train_df, val_df, test_df)
        
        # Run all strategies
        all_results = {}
        
        for key, run_fn, display_name in strategy_registry:
            logger.info("\n" + "="*70)
            logger.info(f"Running {display_name}")
            logger.info("="*70)
            seed_results = []
            for seed in config['seeds']:
                seed_results.append(
                    run_fn(
                        train_df, val_df, test_df,
                        seed=seed,
                        device=config['device'],
                        num_epochs=config['num_epochs']
                    )
                )
            all_results[key] = aggregate_strategy_results(seed_results, config['seeds'])
        
        # Compare results
        comparison_df = compare_strategies(all_results)
        
        # Save results
        save_dir = Path('results/E3_speed_aware')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(save_dir / 'E3_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save comparison CSV
        comparison_df.to_csv(save_dir / 'E3_comparison.csv', index=False)
        
        # Generate LaTeX table
        latex_table = generate_latex_table(comparison_df)
        with open(save_dir / 'E3_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        # Generate plots
        plot_comparison(comparison_df, save_dir)
        
        # Generate PDF report
        generate_e3_report(
            all_results=all_results,
            comparison_df=comparison_df,
            save_dir=str(save_dir),
            train_df=train_df,
        )
        
        logger.info("\n" + "="*70)
        logger.info("E3: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
