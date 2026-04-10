"""
E2: Feature Ablation Study
Compare different feature sets to demonstrate contribution of motion features
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from src.features.preprocessing import DeepSenseDataLoader, adjusted_splitting
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.features.acceleration import add_acceleration_features, normalize_acceleration_features
from src.features.angular_rate import (
    add_angular_rate_features, add_directed_tangential_features,
    normalize_angular_rate_features, normalize_directed_tangential_features
)
from src.models.baseline import BaselineBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, evaluate_by_speed_category
from src.evaluation.report import generate_e2_report
from src.utils.dataset import BeamPredictionDataset, build_speed_labels

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E2_feature_ablation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seed(seed=77):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_feature_sets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Dict]:
    """
    Prepare different feature sets for ablation study.

    IMPORTANT: Velocity and acceleration features are computed on the FULL
    concatenated dataset before splitting back, to avoid Savitzky-Golay
    smoothing artifacts at chunk boundaries that inflate velocity/acceleration
    by 70+ m/s and 4000+ m/s^2 respectively.

    Returns:
        (feature_sets, train_df, val_df, test_df) with motion features added.
    """
    logger.info("="*70)
    logger.info("Preparing Feature Sets")
    logger.info("="*70)

    # ── Step 1: Concat splits so every sequence is complete ───────────────
    # This eliminates SG smoothing artifacts at chunk boundaries.
    logger.info("\nConcatenating splits for full-sequence motion feature computation...")
    train_df = train_df.copy(); train_df['_split'] = 'train'
    val_df   = val_df.copy();   val_df['_split']   = 'val'
    test_df  = test_df.copy();  test_df['_split']  = 'test'
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Full dataset: {len(full_df)} samples across {full_df['sequence_idx'].nunique()} sequences")

    # ── Step 2: Compute velocity/acceleration on full sequences ───────────
    logger.info("\nComputing velocity features (full dataset)...")
    full_df = add_velocity_features(full_df, smooth=True, window_size=5, poly_order=2)

    logger.info("\nComputing acceleration features (full dataset)...")
    full_df = add_acceleration_features(full_df, window_size=5, poly_order=2)

    logger.info("\nComputing angular rate features (full dataset)...")
    full_df = add_angular_rate_features(full_df, smooth=True, window_size=5, poly_order=2)

    logger.info("\nComputing directed tangential velocity features (full dataset)...")
    full_df = add_directed_tangential_features(full_df, smooth=True, window_size=5, poly_order=2)

    # ── Step 3: Split back ────────────────────────────────────────────────
    train_df = full_df[full_df['_split'] == 'train'].drop(columns='_split').reset_index(drop=True)
    val_df   = full_df[full_df['_split'] == 'val'].drop(columns='_split').reset_index(drop=True)
    test_df  = full_df[full_df['_split'] == 'test'].drop(columns='_split').reset_index(drop=True)
    logger.info(f"Split restored: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Step 4: Normalize GPS-derived velocity (fit on train only) ────────
    # v_max=30 m/s — physically motivated by sensor max speed (26.4 m/s).
    # Clips the 2 GPS outlier spikes (53 m/s) that remain on the full data.
    train_df, v_norm_params = normalize_velocity_features(train_df, v_max=30.0)
    val_df,  _ = normalize_velocity_features(val_df,  v_max=v_norm_params['v_max'], fit_on=train_df)
    test_df, _ = normalize_velocity_features(test_df, v_max=v_norm_params['v_max'], fit_on=train_df)

    # ── Step 5: Normalize GPS-derived acceleration (fit on train only) ────
    # a_max=50 m/s^2 (~5g) — typical aggressive UAV maneuver limit.
    # Values above this are GPS differentiation noise and are clipped to +-1.
    train_df, a_norm_params = normalize_acceleration_features(train_df, a_max=50.0)
    val_df,  _ = normalize_acceleration_features(val_df,  a_max=a_norm_params['a_max'], fit_on=train_df)
    test_df, _ = normalize_acceleration_features(test_df, a_max=a_norm_params['a_max'], fit_on=train_df)

    # ── Step 6: Normalize angular rate (fit on train only) ────────────────
    # omega_max=0.1 rad/s ≈ 5.7°/s — generous upper bound for UAV.
    train_df, omega_params = normalize_angular_rate_features(train_df, omega_max=0.1)
    val_df,  _ = normalize_angular_rate_features(val_df,  omega_max=omega_params['omega_max'], fit_on=train_df)
    test_df, _ = normalize_angular_rate_features(test_df, omega_max=omega_params['omega_max'], fit_on=train_df)

    # ── Step 7: Normalize directed tangential velocity (fit on train only) ─
    # Same v_max=30 m/s as velocity normalization.
    train_df, vtan_params = normalize_directed_tangential_features(train_df, v_max=30.0)
    val_df,  _ = normalize_directed_tangential_features(val_df,  v_max=vtan_params['v_tan_max'], fit_on=train_df)
    test_df, _ = normalize_directed_tangential_features(test_df, v_max=vtan_params['v_tan_max'], fit_on=train_df)

    # ── Step 8: Normalize sensor velocity if available ────────────────────
    # sensor vx/vy/vz are direct IMU/GPS-fusion measurements — much more
    # reliable than GPS-derived velocity. S_MAX=30 covers the sensor range (0-26.4 m/s).
    SENSOR_V_MAX = 30.0
    has_sensor_vel = all(c in train_df.columns for c in ['vx_mps', 'vy_mps', 'vz_mps'])
    if has_sensor_vel:
        for df_ in [train_df, val_df, test_df]:
            df_['vx_mps_norm'] = np.clip(df_['vx_mps'] / SENSOR_V_MAX, -1, 1)
            df_['vy_mps_norm'] = np.clip(df_['vy_mps'] / SENSOR_V_MAX, -1, 1)
            df_['vz_mps_norm'] = np.clip(df_['vz_mps'] / SENSOR_V_MAX, -1, 1)
        logger.info(f"Sensor 3D velocity normalized to [-1,1] with v_max={SENSOR_V_MAX} m/s")
    else:
        logger.warning("Sensor velocity columns (vx_mps, vy_mps, vz_mps) not found in data. "
                       "Re-run E1 preprocessing to generate them. E2.10/E2.11 will be skipped.")

    has_speed_mps = 'speed_mps' in train_df.columns
    if has_speed_mps:
        for df_ in [train_df, val_df, test_df]:
            df_['speed_mps_norm'] = np.clip(df_['speed_mps'] / SENSOR_V_MAX, 0, 1)

    # ── Feature column groups ─────────────────────────────────────────────
    base_pos     = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    v_mag        = ['v_mag_norm']
    v_radial     = ['v_radial_norm']
    v_tangential = ['v_tangential_norm']
    all_velocity = ['v_radial_norm', 'v_tangential_norm', 'v_mag_norm']
    a_mag        = ['a_mag_norm']
    a_radial     = ['a_radial_norm']
    a_tangential = ['a_tangential_norm']
    all_accel    = ['a_radial_norm', 'a_tangential_norm', 'a_mag_norm']
    angular_rate = ['angular_rate_norm']
    dir_tan_vec  = ['v_tan_x_norm', 'v_tan_y_norm', 'v_tan_z_norm']
    sensor_3d_v  = ['vx_mps_norm', 'vy_mps_norm', 'vz_mps_norm']
    sensor_speed = ['speed_mps_norm']

    # ── E2.0–E2.9: GPS-derived feature ablation ───────────────────────────
    feature_sets = {
        'E2_0_baseline': {
            'name': 'E2.0 Pos only',
            'features': base_pos,
            'description': 'Baseline: Position only (5 dims)'
        },
        'E2_1_plus_vmag': {
            'name': 'E2.1 + GPS v_mag',
            'features': base_pos + v_mag,
            'description': 'Position + GPS velocity magnitude (6 dims)'
        },
        'E2_2_plus_vradial': {
            'name': 'E2.2 + GPS v_radial',
            'features': base_pos + v_radial,
            'description': 'Position + GPS radial velocity (6 dims)'
        },
        'E2_3_plus_vtangential': {
            'name': 'E2.3 + GPS v_tan',
            'features': base_pos + v_tangential,
            'description': 'Position + GPS tangential velocity (6 dims)'
        },
        'E2_4_plus_vradial_vtangential': {
            'name': 'E2.4 + GPS v_rad+v_tan',
            'features': base_pos + v_radial + v_tangential,
            'description': 'Position + GPS radial & tangential velocity (7 dims)'
        },
        'E2_5_plus_all_velocity': {
            'name': 'E2.5 + all GPS vel',
            'features': base_pos + all_velocity,
            'description': 'Position + all GPS velocity (8 dims)'
        },
        'E2_6_allvel_plus_amag': {
            'name': 'E2.6 + GPS vel + a_mag',
            'features': base_pos + all_velocity + a_mag,
            'description': 'Position + all GPS velocity + acceleration magnitude (9 dims)'
        },
        'E2_7_allvel_plus_aradial': {
            'name': 'E2.7 + GPS vel + a_rad',
            'features': base_pos + all_velocity + a_radial,
            'description': 'Position + all GPS velocity + radial acceleration (9 dims)'
        },
        'E2_8_allvel_plus_atangential': {
            'name': 'E2.8 + GPS vel + a_tan',
            'features': base_pos + all_velocity + a_tangential,
            'description': 'Position + all GPS velocity + tangential acceleration (9 dims)'
        },
        'E2_9_all_features': {
            'name': 'E2.9 + all GPS vel+acc',
            'features': base_pos + all_velocity + all_accel,
            'description': 'Position + all GPS velocity + all GPS acceleration (11 dims)'
        },
    }

    # ── E2.12–E2.14: Angular rate & directed tangential features ──────────
    feature_sets['E2_12_plus_angular_rate'] = {
        'name': 'E2.12 + angular rate',
        'features': base_pos + angular_rate,
        'description': 'Position + angular rate of UE-BS direction (6 dims)'
    }
    feature_sets['E2_13_plus_dir_tangential'] = {
        'name': 'E2.13 + dir v_tan',
        'features': base_pos + dir_tan_vec,
        'description': 'Position + directed tangential velocity 3D (8 dims)'
    }
    feature_sets['E2_14_allvel_plus_angular_rate'] = {
        'name': 'E2.14 + all vel + omega',
        'features': base_pos + all_velocity + angular_rate,
        'description': 'Position + all GPS velocity + angular rate (9 dims)'
    }

    # ── E2.10–E2.11: Sensor-measured velocity (direct comparison) ─────────
    if has_speed_mps:
        feature_sets['E2_10_sensor_speed'] = {
            'name': 'E2.10 + sensor speed',
            'features': base_pos + sensor_speed,
            'description': 'Position + device horizontal speed scalar (6 dims) [sensor]'
        }
    if has_sensor_vel:
        feature_sets['E2_11_sensor_3d_vel'] = {
            'name': 'E2.11 + sensor 3D vel',
            'features': base_pos + sensor_3d_v,
            'description': 'Position + device 3D velocity (vx,vy,vz) (8 dims) [sensor]'
        }

    # Log feature sets
    for key, config in feature_sets.items():
        logger.info(f"  {key}: {config['name']} ({len(config['features'])}D)")

    return feature_sets, train_df, val_df, test_df



def _mean_std(values: List[float]) -> tuple:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) > 0 else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_speed_metrics(speed_metrics_list: List[Dict]) -> Dict:
    if not speed_metrics_list:
        return None

    aggregated = {}
    for speed_name in ['slow', 'medium', 'fast']:
        per_seed = [m for m in speed_metrics_list if m and speed_name in m]
        if not per_seed:
            continue
        aggregated[speed_name] = {}

        sample_counts = [m[speed_name].get('num_samples', 0) for m in per_seed]
        aggregated[speed_name]['num_samples'] = int(np.mean(sample_counts)) if sample_counts else 0

        v_keys = [k for k in per_seed[0].get(speed_name, {}) if k.startswith('v')]
        for v_key in v_keys:
            aggregated[speed_name][v_key] = {}
            for metric_name in per_seed[0][speed_name][v_key].keys():
                vals = [m[speed_name][v_key][metric_name] for m in per_seed]
                mean, std = _mean_std(vals)
                aggregated[speed_name][v_key][metric_name] = mean
                aggregated[speed_name][v_key][f'{metric_name}_std'] = std

        overall_vals = per_seed[0][speed_name].get('overall', {})
        aggregated[speed_name]['overall'] = {}
        for metric_name in overall_vals.keys():
            vals = [m[speed_name]['overall'][metric_name] for m in per_seed]
            mean, std = _mean_std(vals)
            aggregated[speed_name]['overall'][metric_name] = mean
            aggregated[speed_name]['overall'][f'{metric_name}_std'] = std

    return aggregated


def aggregate_results_across_seeds(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    base = seed_results[0]
    aggregated = {
        'feature_set': base.get('feature_set'),
        'num_features': base.get('num_features'),
        'num_parameters': base.get('num_parameters'),
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

    speed_metrics_list = [r.get('speed_metrics') for r in seed_results if r.get('speed_metrics')]
    aggregated['speed_metrics'] = aggregate_speed_metrics(speed_metrics_list)

    return aggregated


def run_ablation_experiment(
    feature_set_name: str,
    feature_config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    batch_size: int = 64,
    num_epochs: int = 20,
    device: str = 'cuda'
) -> Dict:
    """
    Run experiment for a single feature set
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info(f"Running Experiment: {feature_set_name}")
    logger.info(f"Feature Set: {feature_config['name']}")
    logger.info(f"Seed: {seed}")
    logger.info("="*70)

    set_random_seed(seed)
    
    # Create datasets
    feature_columns = feature_config['features']
    
    # Drop rows with NaN in required features
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    logger.info(f"Clean samples: train={len(train_df_clean)}, val={len(val_df_clean)}, test={len(test_df_clean)}")
    
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4
    )
    
    # Create dataloaders (aligned with E1: batch=8, val/test=1024, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    input_dim = len(feature_columns)
    model = BaselineBeamPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    logger.info(f"Model input dimension: {input_dim}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    save_dir = f'checkpoints/E2_feature_ablation/{feature_set_name}/seed_{seed}'
    log_dir = f'runs/E2_feature_ablation/{feature_set_name}/seed_{seed}'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=5e-4,
        weight_decay=0.0,
        num_epochs=num_epochs,
        gradient_clip=0.0,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping_patience=999
    )
    
    trainer.train()
    
    # Load best model
    best_model_path = Path(save_dir) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device=device)

    # Speed category evaluation
    speed_metrics = None
    try:
        speed_labels = build_speed_labels(
            df=test_df_clean,
            dataset=test_dataset,
            window_size=8,
            num_predictions=4,
            pad_initial=True,
        )
        if speed_labels is None:
            logger.warning("Speed labels unavailable. Skipping speed-category evaluation.")
        else:
            speed_metrics = evaluate_by_speed_category(
                model,
                test_loader,
                speed_labels,
                device=device
            )
    except Exception as e:
        logger.warning(f"Speed category evaluation failed: {e}")
        speed_metrics = None
    
    # Add training history
    results['training_history'] = trainer.get_training_history()
    results['feature_set'] = feature_config['name']
    results['num_features'] = input_dim
    results['num_parameters'] = model.get_num_parameters()
    results['speed_metrics'] = speed_metrics
    results['seed'] = seed
    
    return results


def compare_results(all_results: Dict) -> pd.DataFrame:
    """
    Compare results across feature sets
    
    Returns:
        DataFrame with comparison
    """
    logger.info("\n" + "="*70)
    logger.info("Comparing Results")
    logger.info("="*70)
    
    comparison_data = []
    
    for feature_set_name, results in all_results.items():
        row = {
            'Feature Set': results['feature_set'],
            'Num Features': results['num_features'],
            'Num Parameters': results['num_parameters']
        }
        
        # Add metrics for each prediction step
        for v in range(4):
            row[f'v{v}_top1'] = results[f'v{v}']['top1_acc'] * 100
            row[f'v{v}_top3'] = results[f'v{v}']['top3_acc'] * 100
            row[f'v{v}_top1_std'] = results[f'v{v}'].get('top1_acc_std', 0.0) * 100
            row[f'v{v}_top3_std'] = results[f'v{v}'].get('top3_acc_std', 0.0) * 100
        
        # Overall metrics
        row['Overall_top1'] = results['overall']['top1_acc'] * 100
        row['Overall_top3'] = results['overall']['top3_acc'] * 100
        row['Overall_top1_std'] = results['overall'].get('top1_acc_std', 0.0) * 100
        row['Overall_top3_std'] = results['overall'].get('top3_acc_std', 0.0) * 100
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate improvement over baseline
    baseline_acc = df.iloc[0]['Overall_top1']
    df['Improvement (%)'] = df['Overall_top1'] - baseline_acc
    
    logger.info("\n" + df.to_string())
    
    return df


def plot_comparison(comparison_df: pd.DataFrame, save_dir: Path):
    """Plot comparison visualizations for 10 ablation experiments"""
    
    # Plot 1: Bar chart of overall accuracy
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Overall_top1'], width, 
           label='Top-1', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Overall_top3'], width,
           label='Top-3', color='coral', alpha=0.8)
    
    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Feature Ablation: Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Feature Set'], rotation=35, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E2_overall_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E2_overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Line plot by prediction step
    fig, ax = plt.subplots(figsize=(12, 7))
    
    steps = [0, 1, 2, 3]
    for idx, row in comparison_df.iterrows():
        accs = [row[f'v{v}_top1'] for v in steps]
        ax.plot(steps, accs, marker='o', linewidth=2, markersize=6, 
                label=row['Feature Set'])
    
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Feature Ablation: Accuracy by Prediction Step', fontsize=14, fontweight='bold')
    ax.set_xticks(steps)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E2_by_step.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E2_by_step.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Improvement heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = comparison_df[[f'v{v}_top1' for v in range(4)]].values
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=[f'v={v}' for v in range(4)],
                yticklabels=comparison_df['Feature Set'],
                cbar_kws={'label': 'Top-1 Accuracy (%)'},
                ax=ax)
    
    ax.set_title('Feature Ablation: Accuracy Heatmap', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E2_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def main():
    """Main execution function"""
    
    logger.info("\n" + "="*70)
    logger.info("E2: FEATURE ABLATION STUDY")
    logger.info("="*70 + "\n")
    
    set_random_seed(77)
    
    config = {
        'data_path': 'data/raw/scenario23',
        'batch_size': 8,
        'num_epochs': 20,
        'seeds': [42, 77, 78, 79, 80],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        # Load data
        logger.info("Loading preprocessed data...")
        train_df = pd.read_pickle('data/processed/E1_baseline/train.pkl')
        val_df = pd.read_pickle('data/processed/E1_baseline/val.pkl')
        test_df = pd.read_pickle('data/processed/E1_baseline/test.pkl')
        
        # Prepare feature sets
        feature_sets, train_df, val_df, test_df = prepare_feature_sets(
            train_df, val_df, test_df
        )

        # Save velocity-enriched splits so E4/E5/E8 can load them directly
        vel_dir = Path('data/processed/E2_feature_ablation')
        vel_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_pickle(vel_dir / 'train_with_velocity.pkl')
        val_df.to_pickle(vel_dir / 'val_with_velocity.pkl')
        test_df.to_pickle(vel_dir / 'test_with_velocity.pkl')
        logger.info(f"Saved velocity-enriched splits to {vel_dir}")

        # Run experiments for each feature set
        all_results = {}
        
        for feature_set_name, feature_config in feature_sets.items():
            seed_results = []
            for seed in config['seeds']:
                results = run_ablation_experiment(
                    feature_set_name,
                    feature_config,
                    train_df,
                    val_df,
                    test_df,
                    seed=seed,
                    batch_size=config['batch_size'],
                    num_epochs=config['num_epochs'],
                    device=config['device']
                )
                seed_results.append(results)

            all_results[feature_set_name] = aggregate_results_across_seeds(
                seed_results,
                config['seeds']
            )
        
        # Compare results
        comparison_df = compare_results(all_results)
        
        # Save results & generate report
        save_dir = Path('results/E2_feature_ablation')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_comparison(comparison_df, save_dir)
        
        # Generate PDF report
        generate_e2_report(
            all_results=all_results,
            comparison_df=comparison_df,
            save_dir=str(save_dir),
        )
        
        logger.info("\n" + "="*70)
        logger.info("E2: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
