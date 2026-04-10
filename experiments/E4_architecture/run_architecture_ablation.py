"""
E4: Architecture Ablation Study (Redesigned)

8 architectures (E4.0-E4.7) × 2 feature sets (8D, 6D) × 5 seeds.
Each variant changes exactly ONE variable from the baseline, enabling
clean ablation of: multi-scale CNN, attention, hidden dim, GRU depth,
bidirectional GRU, and Transformer encoder.

Attention variants (E4.2, E4.3) are skipped for 6D because motion_dim=1
degenerates cross-attention to a scalar gate.
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
from collections import OrderedDict

from src.models.ablation import AblationBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.dataset import BeamPredictionDataset
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.features.acceleration import add_acceleration_features, normalize_acceleration_features

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E4_architecture_ablation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ======================================================================
# Data loading
# ======================================================================

def _load_or_build_velocity_data():
    """Load E2 velocity pkl if it exists, otherwise build from E1 baseline pkl and cache."""
    PKL_DIR = Path('data/processed/E2_feature_ablation')
    TRAIN_PKL = PKL_DIR / 'train_with_velocity.pkl'

    if TRAIN_PKL.exists():
        logger.info("Loading preprocessed velocity data from E2 cache...")
        train_df = pd.read_pickle(TRAIN_PKL)
        val_df   = pd.read_pickle(PKL_DIR / 'val_with_velocity.pkl')
        test_df  = pd.read_pickle(PKL_DIR / 'test_with_velocity.pkl')
        return train_df, val_df, test_df

    logger.warning("E2 velocity pkl not found — building from E1 baseline data (run E2 first for full feature set).")
    train_df = pd.read_pickle('data/processed/E1_baseline/train.pkl')
    val_df   = pd.read_pickle('data/processed/E1_baseline/val.pkl')
    test_df  = pd.read_pickle('data/processed/E1_baseline/test.pkl')

    for df_ in [train_df, val_df, test_df]:
        df_['_split'] = ''
    train_df['_split'] = 'train'; val_df['_split'] = 'val'; test_df['_split'] = 'test'
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = add_velocity_features(full_df, smooth=True, window_size=5, poly_order=2)
    full_df = add_acceleration_features(full_df, window_size=5, poly_order=2)
    train_df = full_df[full_df['_split'] == 'train'].drop(columns='_split').reset_index(drop=True)
    val_df   = full_df[full_df['_split'] == 'val'].drop(columns='_split').reset_index(drop=True)
    test_df  = full_df[full_df['_split'] == 'test'].drop(columns='_split').reset_index(drop=True)

    train_df, v_norm = normalize_velocity_features(train_df, v_max=30.0)
    val_df,  _       = normalize_velocity_features(val_df,  v_max=v_norm['v_max'], fit_on=train_df)
    test_df, _       = normalize_velocity_features(test_df, v_max=v_norm['v_max'], fit_on=train_df)
    train_df, a_norm = normalize_acceleration_features(train_df, a_max=50.0)
    val_df,  _       = normalize_acceleration_features(val_df,  a_max=a_norm['a_max'], fit_on=train_df)
    test_df, _       = normalize_acceleration_features(test_df, a_max=a_norm['a_max'], fit_on=train_df)

    PKL_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(TRAIN_PKL)
    val_df.to_pickle(PKL_DIR / 'val_with_velocity.pkl')
    test_df.to_pickle(PKL_DIR / 'test_with_velocity.pkl')
    logger.info(f"Cached velocity-enriched splits to {PKL_DIR}")
    return train_df, val_df, test_df


# ======================================================================
# Reproducibility
# ======================================================================

def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Architecture definitions (E4.0 – E4.7)
# ======================================================================

def define_architectures() -> OrderedDict:
    """Return ordered dict of 8 architecture configs."""
    return OrderedDict([
        ('E4.0_baseline', {
            'name': 'Baseline CNN-GRU',
            'cnn_type': 'baseline',
            'use_attention': False,
            'encoder_type': 'gru',
            'num_encoder_layers': 1,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Paper baseline: 2-Conv + 1-layer GRU (control)',
        }),
        ('E4.1_multiscale_cnn', {
            'name': 'Multi-Scale CNN',
            'cnn_type': 'multiscale',
            'use_attention': False,
            'encoder_type': 'gru',
            'num_encoder_layers': 1,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Multi-scale CNN (k=3,5,7) + 1-layer GRU',
        }),
        ('E4.2_attention', {
            'name': 'Motion Attention',
            'cnn_type': 'baseline',
            'use_attention': True,
            'encoder_type': 'gru',
            'num_encoder_layers': 1,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Baseline CNN + cross-attention (pos<->motion)',
        }),
        ('E4.3_multiscale_attention', {
            'name': 'MultiScale + Attention',
            'cnn_type': 'multiscale',
            'use_attention': True,
            'encoder_type': 'gru',
            'num_encoder_layers': 1,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Multi-scale CNN + cross-attention (combined)',
        }),
        ('E4.4_larger_hidden', {
            'name': 'Hidden=256',
            'cnn_type': 'baseline',
            'use_attention': False,
            'encoder_type': 'gru',
            'num_encoder_layers': 1,
            'hidden_dim': 256,
            'dropout': 0.1,
            'description': 'Baseline with hidden_dim=256 (param scaling control)',
        }),
        ('E4.5_deeper_gru', {
            'name': '2-Layer GRU',
            'cnn_type': 'baseline',
            'use_attention': False,
            'encoder_type': 'gru',
            'num_encoder_layers': 2,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Baseline with 2-layer GRU encoder (depth control)',
        }),
        ('E4.6_bigru', {
            'name': 'Bidirectional GRU',
            'cnn_type': 'baseline',
            'use_attention': False,
            'encoder_type': 'bigru',
            'num_encoder_layers': 1,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Bidirectional GRU encoder (1-layer)',
        }),
        ('E4.7_transformer', {
            'name': 'Transformer Encoder',
            'cnn_type': 'baseline',
            'use_attention': False,
            'encoder_type': 'transformer',
            'num_encoder_layers': 2,
            'hidden_dim': 128,
            'dropout': 0.1,
            'description': 'Transformer encoder (2L, 4H) + sinusoidal PE',
        }),
    ])


# ======================================================================
# Feature set definitions
# ======================================================================

FEATURE_SETS = OrderedDict([
    ('8D', [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'v_radial_norm', 'v_tangential_norm', 'v_mag_norm',
    ]),
    ('6D_vmag', [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'v_mag_norm',
    ]),
    ('6D_vtan', [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'v_tangential_norm',
    ]),
    ('6D_sensor', [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'speed_mps_norm',
    ]),
])

POSITION_DIM = 5  # first 5 columns are always position


# ======================================================================
# Aggregation helpers
# ======================================================================

def _mean_std(values: List[float]) -> tuple:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) > 0 else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_seed_results(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    base = seed_results[0]
    aggregated = {
        'architecture': base.get('architecture'),
        'description': base.get('description'),
        'num_parameters': base.get('num_parameters'),
        'feature_set': base.get('feature_set'),
        'seed_stats': {'seeds': seeds, 'count': len(seeds)},
        'training_history': base.get('training_history'),
        'training_histories': [r.get('training_history') for r in seed_results],
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

    return aggregated


# ======================================================================
# Single experiment run
# ======================================================================

def run_architecture_experiment(
    arch_name: str,
    arch_config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    feature_set_name: str,
    seed: int,
    batch_size: int = 8,
    num_epochs: int = 20,
    device: str = 'cuda',
) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"Architecture: {arch_name} | Features: {feature_set_name} | Seed: {seed}")
    logger.info(f"Config: {arch_config['description']}")
    logger.info("=" * 70)

    set_random_seed(seed)

    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)

    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4,
    )
    val_dataset = BeamPredictionDataset(
        val_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4,
    )
    test_dataset = BeamPredictionDataset(
        test_df_clean, feature_columns=feature_columns,
        window_size=8, num_predictions=4,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    # Create model
    input_dim = len(feature_columns)
    motion_dim = input_dim - POSITION_DIM

    model = AblationBeamPredictor(
        input_dim=input_dim,
        hidden_dim=arch_config['hidden_dim'],
        num_beams=32,
        num_predictions=4,
        cnn_type=arch_config['cnn_type'],
        use_attention=arch_config['use_attention'],
        position_dim=POSITION_DIM,
        motion_dim=motion_dim,
        encoder_type=arch_config['encoder_type'],
        num_encoder_layers=arch_config['num_encoder_layers'],
        dropout=arch_config['dropout'],
    )

    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Train
    save_dir = f'checkpoints/E4_architecture/{feature_set_name}/{arch_name}/seed_{seed}'
    log_dir = f'runs/E4_architecture/{feature_set_name}/{arch_name}/seed_{seed}'

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping_patience=999,
    )

    trainer.train()

    # Load best model
    best_model_path = Path(save_dir) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)

    # Evaluate
    results = evaluate_model(model, test_loader, device=device)

    results['architecture'] = arch_config['name']
    results['description'] = arch_config['description']
    results['num_parameters'] = model.get_num_parameters()
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    results['feature_set'] = feature_set_name

    return results


# ======================================================================
# Comparison / reporting
# ======================================================================

def compare_architectures(all_results: Dict, feature_set_name: str) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info(f"Comparing Architectures — Feature Set: {feature_set_name}")
    logger.info("=" * 70)

    rows = []
    for arch_name, results in all_results.items():
        row = {
            'Architecture': results['architecture'],
            'Description': results['description'],
            'Parameters': results['num_parameters'],
            'Feature_Set': feature_set_name,
        }
        for v in range(4):
            row[f'v{v}_top1'] = results[f'v{v}']['top1_acc'] * 100
            row[f'v{v}_top3'] = results[f'v{v}']['top3_acc'] * 100
            row[f'v{v}_top1_std'] = results[f'v{v}'].get('top1_acc_std', 0.0) * 100
            row[f'v{v}_top3_std'] = results[f'v{v}'].get('top3_acc_std', 0.0) * 100

        row['Overall_top1'] = results['overall']['top1_acc'] * 100
        row['Overall_top3'] = results['overall']['top3_acc'] * 100
        row['Overall_top1_std'] = results['overall'].get('top1_acc_std', 0.0) * 100
        row['Overall_top3_std'] = results['overall'].get('top3_acc_std', 0.0) * 100

        if 'training_history' in results:
            history = results['training_history']
            row['Best_Epoch'] = history.get('best_epoch', -1) + 1
            row['Best_Val_Acc'] = history.get('best_val_acc', 0) * 100

        rows.append(row)

    df = pd.DataFrame(rows)

    # Improvement over E4.0 baseline
    baseline_acc = df.iloc[0]['Overall_top1']
    df['Improvement (%)'] = df['Overall_top1'] - baseline_acc

    baseline_params = df.iloc[0]['Parameters']
    df['Param_Ratio'] = df['Parameters'] / baseline_params

    logger.info("\n" + df[['Architecture', 'Parameters', 'Overall_top1',
                           'Overall_top1_std', 'Improvement (%)']].to_string())
    return df


def generate_latex_table(comparison_df: pd.DataFrame, feature_set_name: str) -> str:
    def fmt(mean_val, std_val):
        return f"{mean_val:.2f} $\\pm$ {std_val:.2f}"

    latex = rf"""
\begin{{table*}}[t]
\centering
\caption{{E4 Architecture Ablation — {feature_set_name} Features}}
\label{{tab:e4_{feature_set_name.lower()}}}
\begin{{tabular}}{{lrccccc}}
\toprule
\textbf{{Architecture}} & \textbf{{Params}} & \textbf{{Top-1 (\%)}} & \textbf{{Top-3 (\%)}} & \textbf{{$\Delta$ Top-1}} & \textbf{{Param Ratio}} \\
\midrule
"""
    for _, row in comparison_df.iterrows():
        p = row['Parameters']
        params_str = f"{p / 1e6:.2f}M" if p > 1e6 else f"{p / 1e3:.1f}K"
        latex += f"{row['Architecture']} & {params_str} & "
        latex += f"{fmt(row['Overall_top1'], row.get('Overall_top1_std', 0.0))} & "
        latex += f"{fmt(row['Overall_top3'], row.get('Overall_top3_std', 0.0))} & "
        latex += f"{row['Improvement (%)']:+.2f} & {row['Param_Ratio']:.2f}$\\times$ \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


# ======================================================================
# Plotting
# ======================================================================

def plot_comparison(comparison_df: pd.DataFrame, save_dir: Path, tag: str):
    # 1) Bar chart top-1 & top-3 with error bars
    fig, ax = plt.subplots(figsize=(12, 6))
    archs = comparison_df['Architecture']
    x = np.arange(len(archs))
    w = 0.35
    ax.bar(x - w / 2, comparison_df['Overall_top1'], w, label='Top-1',
           yerr=comparison_df['Overall_top1_std'], capsize=4, color='steelblue', alpha=0.85)
    ax.bar(x + w / 2, comparison_df['Overall_top3'], w, label='Top-3',
           yerr=comparison_df['Overall_top3_std'], capsize=4, color='coral', alpha=0.85)
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'E4 Architecture Ablation ({tag}): Overall Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=25, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'E4_{tag}_bar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Accuracy by prediction step
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in comparison_df.iterrows():
        accs = [row[f'v{v}_top1'] for v in range(4)]
        ax.plot(range(4), accs, marker='o', linewidth=2, markersize=8, label=row['Architecture'])
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(f'E4 Architecture Ablation ({tag}): Accuracy by Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'E4_{tag}_by_step.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved: {save_dir}/E4_{tag}_*.png")


# ======================================================================
# Grand summary
# ======================================================================

def _print_grand_summary(grand_results: Dict, architectures: OrderedDict, save_dir: Path):
    """Print a 2-D pivot table: rows=architectures, cols=feature sets."""
    feat_names = list(FEATURE_SETS.keys())
    arch_names = list(architectures.keys())

    # Build rows: one per architecture
    rows = []
    for arch_name in arch_names:
        arch_display = architectures[arch_name]['name']
        row = {'Architecture': arch_display}
        for feat in feat_names:
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                r = grand_results[key]
                mean = r['overall']['top1_acc'] * 100
                std  = r['overall'].get('top1_acc_std', 0.0) * 100
                row[feat] = f"{mean:.2f}±{std:.2f}"
                row[f"{feat}_mean"] = mean
            else:
                row[feat] = "—"
                row[f"{feat}_mean"] = float('nan')
        rows.append(row)

    df_display = pd.DataFrame(rows).set_index('Architecture')

    # Best per column
    best_row = {'Architecture': '>>> BEST'}
    for feat in feat_names:
        mean_col = [r[f"{feat}_mean"] for r in rows]
        valid = [(v, rows[i]['Architecture']) for i, v in enumerate(mean_col) if not np.isnan(v)]
        if valid:
            best_val, best_arch = max(valid)
            best_row[feat] = f"{best_val:.2f} ({best_arch})"
        else:
            best_row[feat] = "—"

    # Log
    logger.info("\n" + "=" * 70)
    logger.info("E4 GRAND SUMMARY — Top-1 Accuracy (%) per Architecture × Feature Set")
    logger.info("=" * 70)
    logger.info(df_display[feat_names].to_string())
    logger.info("-" * 70)
    for feat in feat_names:
        logger.info(f"  Best [{feat}]: {best_row[feat]}")

    # Find overall best combo
    best_mean = -1.0
    best_combo = ""
    for arch_name in arch_names:
        for feat in feat_names:
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                v = grand_results[key]['overall']['top1_acc'] * 100
                if v > best_mean:
                    best_mean = v
                    best_combo = f"{architectures[arch_name]['name']} × {feat}"
    logger.info(f"\n  >>> OVERALL BEST: {best_combo}  →  {best_mean:.2f}%")
    logger.info("=" * 70 + "\n")

    # Save summary CSV (mean only)
    mean_rows = []
    for arch_name in arch_names:
        r = {'Architecture': architectures[arch_name]['name']}
        for feat in feat_names:
            key = f"{feat}_{arch_name}"
            if key in grand_results:
                r[f"{feat}_top1"] = grand_results[key]['overall']['top1_acc'] * 100
                r[f"{feat}_std"]  = grand_results[key]['overall'].get('top1_acc_std', 0.0) * 100
            else:
                r[f"{feat}_top1"] = float('nan')
                r[f"{feat}_std"]  = float('nan')
        mean_rows.append(r)

    summary_df = pd.DataFrame(mean_rows)
    summary_df.to_csv(save_dir / 'E4_grand_summary.csv', index=False)
    logger.info(f"Grand summary saved to {save_dir / 'E4_grand_summary.csv'}")

    _plot_grand_summary(summary_df, feat_names, save_dir)


def _plot_grand_summary(summary_df: pd.DataFrame, feat_names: List[str], save_dir: Path):
    """Two grand summary plots for paper: heatmap + grouped bar (arch × feature set)."""

    arch_labels = summary_df['Architecture'].tolist()
    top1_cols = [f"{f}_top1" for f in feat_names]
    std_cols  = [f"{f}_std"  for f in feat_names]

    # ── 1) Heatmap: arch (rows) × feature set (cols)  ──────────────────
    heat_data = summary_df[top1_cols].values  # shape (n_arch, n_feat)

    fig, ax = plt.subplots(figsize=(max(7, len(feat_names) * 2), max(5, len(arch_labels) * 0.7 + 1)))
    im = sns.heatmap(
        heat_data, annot=True, fmt='.2f', cmap='YlGnBu',
        xticklabels=feat_names,
        yticklabels=arch_labels,
        cbar_kws={'label': 'Top-1 Accuracy (%)'},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title('E4 Architecture × Feature Set — Top-1 Accuracy (%)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Feature Set', fontsize=11)
    ax.set_ylabel('Architecture', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / 'E4_grand_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Grand heatmap saved: {save_dir / 'E4_grand_heatmap.png'}")

    # ── 2) Grouped bar: arch on x-axis, grouped by feature set  ─────────
    n_arch = len(arch_labels)
    n_feat = len(feat_names)
    x = np.arange(n_arch)
    total_w = 0.75
    w = total_w / n_feat
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_feat))

    fig, ax = plt.subplots(figsize=(max(10, n_arch * 1.4), 6))
    for i, (feat, col, scol, color) in enumerate(zip(feat_names, top1_cols, std_cols, colors)):
        means = summary_df[col].values
        stds  = summary_df[scol].values
        offset = (i - n_feat / 2 + 0.5) * w
        ax.bar(x + offset, means, w, label=feat,
               yerr=np.where(np.isnan(stds), 0, stds),
               capsize=3, color=color, alpha=0.85)
        for j, m in enumerate(means):
            if np.isnan(m):
                ax.text(x[j] + offset, ax.get_ylim()[0] + 0.5, '—',
                        ha='center', va='bottom', fontsize=8, color='gray')

    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('E4 Architecture Ablation: Overall Top-1 per Feature Set',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=25, ha='right', fontsize=9)
    ax.legend(title='Feature Set', fontsize=10, title_fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    all_vals = summary_df[top1_cols].values.flatten()
    valid = all_vals[~np.isnan(all_vals)]
    if len(valid) > 0:
        ax.set_ylim(max(0, valid.min() - 2), valid.max() + 2)

    plt.tight_layout()
    plt.savefig(save_dir / 'E4_grand_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Grand bar chart saved: {save_dir / 'E4_grand_bar.png'}")


# ======================================================================
# Main
# ======================================================================

def main():
    logger.info("\n" + "=" * 70)
    logger.info("E4: ARCHITECTURE ABLATION STUDY (Redesigned)")
    logger.info("=" * 70 + "\n")

    config = {
        'batch_size': 8,
        'num_epochs': 30,
        'min_epochs': 20,
        'seeds': [42, 77, 78, 79, 80],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'early_stopping_patience': 5,
    }

    # Load data
    logger.info("Loading preprocessed data...")
    train_df, val_df, test_df = _load_or_build_velocity_data()

    architectures = define_architectures()
    save_dir = Path('results/E4_architecture')
    save_dir.mkdir(parents=True, exist_ok=True)

    grand_results = {}          # key = "feat_arch", value = aggregated results
    feat_comparison_dfs = {}    # key = feat_name, value = comparison DataFrame

    for feat_name, feature_columns in FEATURE_SETS.items():
        motion_dim = len(feature_columns) - POSITION_DIM
        logger.info(f"\n{'#' * 70}")
        logger.info(f"Feature set: {feat_name} ({len(feature_columns)}D, motion_dim={motion_dim})")
        logger.info(f"{'#' * 70}")

        feat_results = OrderedDict()

        for arch_name, arch_config in architectures.items():
            # Skip attention variants for 6D (motion_dim=1 → degenerate)
            if arch_config['use_attention'] and motion_dim < 2:
                logger.info(f"Skipping {arch_name} for {feat_name} (need >=2 motion dims for attention)")
                continue

            seed_results = []
            for seed in config['seeds']:
                result = run_architecture_experiment(
                    arch_name=arch_name,
                    arch_config=arch_config,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    feature_columns=feature_columns,
                    feature_set_name=feat_name,
                    seed=seed,
                    batch_size=config['batch_size'],
                    num_epochs=config['num_epochs'],
                    device=config['device'],
                )
                seed_results.append(result)

            agg = aggregate_seed_results(seed_results, config['seeds'])
            feat_results[arch_name] = agg
            grand_results[f"{feat_name}_{arch_name}"] = agg

        # Per-feature-set comparison
        comparison_df = compare_architectures(feat_results, feat_name)
        comparison_df.to_csv(save_dir / f'E4_{feat_name}_comparison.csv', index=False)
        feat_comparison_dfs[feat_name] = comparison_df

        latex = generate_latex_table(comparison_df, feat_name)
        with open(save_dir / f'E4_{feat_name}_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex)

        plot_comparison(comparison_df, save_dir, feat_name)

    # Save all results JSON
    with open(save_dir / 'E4_all_results.json', 'w') as f:
        json.dump(grand_results, f, indent=2, default=str)

    # Grand summary (log + PNG plots)
    _print_grand_summary(grand_results, architectures, save_dir)

    # PDF report
    from src.evaluation.report import generate_e4_report
    generate_e4_report(
        grand_results=grand_results,
        feat_comparison_dfs=feat_comparison_dfs,
        architectures=architectures,
        save_dir=str(save_dir),
    )

    logger.info("\n" + "=" * 70)
    logger.info("E4: EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to {save_dir}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
