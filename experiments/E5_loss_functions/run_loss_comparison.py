"""
E5: Loss Function Comparison
Compare different loss functions for beam prediction
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

from src.models.baseline import BaselineBeamPredictor
from src.training.trainer import Trainer
from src.training.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    PowerLoss,
    CombinedLoss
)
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
        logging.FileHandler('logs/E5_loss_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mean_std(values: List[float]) -> tuple:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) > 0 else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_loss_results(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    base = seed_results[0]
    aggregated = {
        'loss_function': base.get('loss_function'),
        'description': base.get('description'),
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

    return aggregated


def run_loss_experiment(
    loss_name: str,
    loss_config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    seed: int,
    batch_size: int = 8,
    num_epochs: int = 20,
    device: str = 'cuda'
) -> Dict:
    """
    Run experiment with specific loss function
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info(f"Running Loss Function: {loss_name}")
    logger.info(f"Configuration: {loss_config['description']}")
    logger.info(f"Seed: {seed}")
    logger.info("="*70)

    set_random_seed(seed)
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create datasets
    include_beam_powers = loss_config.get('requires_power', False)
    
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=include_beam_powers
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=include_beam_powers
    )
    
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=include_beam_powers
    )
    
    # Create dataloaders
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
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create loss function
    loss_type = loss_config['type']
    
    if loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(
            num_classes=32,
            smoothing=loss_config.get('smoothing', 0.1)
        )
    elif loss_type == 'combined':
        criterion = CombinedLoss(
            ce_weight=loss_config.get('ce_weight', 1.0),
            power_weight=loss_config.get('power_weight', 0.1)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    logger.info(f"Loss function: {criterion.__class__.__name__}")
    
    # Train
    save_dir = f'checkpoints/E5_loss_functions/{loss_name}/seed_{seed}'
    log_dir = f'runs/E5_loss_functions/{loss_name}/seed_{seed}'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
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
    results = evaluate_model(
        model, 
        test_loader, 
        device=device,
        compute_power_loss=include_beam_powers
    )
    
    # Add metadata
    results['loss_function'] = loss_config['name']
    results['description'] = loss_config['description']
    results['training_history'] = trainer.get_training_history()
    results['seed'] = seed
    
    return results


def define_loss_functions() -> Dict:
    """
    Define loss function configurations
    
    Returns:
        Dictionary of loss configurations
    """
    loss_functions = {
        'L1_cross_entropy': {
            'name': 'Cross-Entropy',
            'type': 'cross_entropy',
            'description': 'Standard cross-entropy loss',
            'requires_power': False
        },
        'L2_focal_loss': {
            'name': 'Focal Loss',
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 2.0,
            'description': 'Focal loss (α=0.25, γ=2.0) for class imbalance',
            'requires_power': False
        },
        'L3_focal_high_gamma': {
            'name': 'Focal Loss (γ=3)',
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 3.0,
            'description': 'Focal loss with higher gamma (γ=3.0)',
            'requires_power': False
        },
        'L4_label_smoothing': {
            'name': 'Label Smoothing',
            'type': 'label_smoothing',
            'smoothing': 0.1,
            'description': 'Label smoothing (ε=0.1) for regularization',
            'requires_power': False
        },
        'L5_label_smoothing_high': {
            'name': 'Label Smoothing (ε=0.2)',
            'type': 'label_smoothing',
            'smoothing': 0.2,
            'description': 'Label smoothing with higher epsilon (ε=0.2)',
            'requires_power': False
        },
        'L6_combined': {
            'name': 'Combined (CE + Power)',
            'type': 'combined',
            'ce_weight': 1.0,
            'power_weight': 0.1,
            'description': 'Combined CE + Power loss (1.0 + 0.1)',
            'requires_power': True
        },
        'L7_combined_balanced': {
            'name': 'Combined (Balanced)',
            'type': 'combined',
            'ce_weight': 0.7,
            'power_weight': 0.3,
            'description': 'Balanced combined loss (0.7 + 0.3)',
            'requires_power': True
        }
    }
    
    return loss_functions


def compare_loss_functions(all_results: Dict) -> pd.DataFrame:
    """Compare results across loss functions"""
    
    logger.info("\n" + "="*70)
    logger.info("Comparing Loss Functions")
    logger.info("="*70)
    
    comparison_data = []
    
    for loss_name, results in all_results.items():
        row = {
            'Loss Function': results['loss_function'],
            'Description': results['description']
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
        
        # Power loss (if available)
        if 'mean_power_loss_db' in results['overall']:
            row['Power_Loss_dB'] = results['overall']['mean_power_loss_db']
            row['Power_Loss_dB_std'] = results['overall'].get('mean_power_loss_db_std', 0.0)
        
        # Training metrics
        if 'training_history' in results:
            history = results['training_history']
            row['Best_Epoch'] = history.get('best_epoch', -1) + 1
            row['Best_Val_Acc'] = history.get('best_val_acc', 0) * 100
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate improvement over baseline
    baseline_acc = df.iloc[0]['Overall_top1']
    df['Improvement (%)'] = df['Overall_top1'] - baseline_acc
    
    logger.info("\n" + df[['Loss Function', 'Overall_top1', 'Overall_top3', 'Improvement (%)']].to_string())
    
    return df


def generate_latex_table(comparison_df: pd.DataFrame) -> str:
    """Generate LaTeX comparison table"""
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{Loss Function Comparison Results}
\label{tab:e5_loss_functions}
\begin{tabular}{lcccc}
\toprule
\textbf{Loss Function} & \textbf{Top-1 (\%)} & \textbf{Top-3 (\%)} & \textbf{Improvement} & \textbf{Best Epoch} \\
\midrule
"""

    def fmt(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}"
    
    for _, row in comparison_df.iterrows():
        latex += f"{row['Loss Function']} & "
        latex += f"{fmt(row['Overall_top1'], row.get('Overall_top1_std', 0.0))} & "
        latex += f"{fmt(row['Overall_top3'], row.get('Overall_top3_std', 0.0))} & "
        latex += f"{row['Improvement (%)']:+.2f} & "
        latex += f"{int(row['Best_Epoch'])} \\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def plot_comparison(comparison_df: pd.DataFrame, save_dir: Path):
    """Plot comparison visualizations"""
    
    # Plot 1: Overall accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    loss_funcs = comparison_df['Loss Function']
    x = np.arange(len(loss_funcs))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Overall_top1'], width,
           label='Top-1', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Overall_top3'], width,
           label='Top-3', color='coral', alpha=0.8)
    
    ax.set_xlabel('Loss Function', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Loss Function Comparison: Overall Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(loss_funcs, rotation=25, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E5_overall_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E5_overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Improvement bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if x >= 0 else 'red' for x in comparison_df['Improvement (%)']]
    ax.barh(loss_funcs, comparison_df['Improvement (%)'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_ylabel('Loss Function', fontsize=12)
    ax.set_title('Loss Function Comparison: Improvement over Cross-Entropy', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E5_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E5_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Accuracy by prediction step
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = [0, 1, 2, 3]
    for idx, row in comparison_df.iterrows():
        accs = [row[f'v{v}_top1'] for v in steps]
        ax.plot(steps, accs, marker='o', linewidth=2, markersize=8,
                label=row['Loss Function'])
    
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Loss Function Comparison: Accuracy by Prediction Step', fontsize=14, fontweight='bold')
    ax.set_xticks(steps)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E5_by_step.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E5_by_step.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Training convergence comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(comparison_df['Best_Epoch'], comparison_df['Overall_top1'],
               s=200, c=comparison_df.index, cmap='viridis', alpha=0.7,
               edgecolors='black', linewidth=1.5)
    
    for idx, row in comparison_df.iterrows():
        ax.annotate(row['Loss Function'],
                   (row['Best_Epoch'], row['Overall_top1']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Best Epoch', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Loss Function Comparison: Convergence Speed vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E5_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E5_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def main():
    """Main execution function"""
    
    logger.info("\n" + "="*70)
    logger.info("E5: LOSS FUNCTION COMPARISON")
    logger.info("="*70 + "\n")
    
    set_random_seed(42)
    
    config = {
        'batch_size': 8,
        'num_epochs': 20,
        'seeds': [42, 77, 78, 79, 80],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Feature columns (position-only, same as E1 — isolate loss function effect)
    feature_columns = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']

    try:
        # Load E1 baseline pkl directly — no velocity features needed
        logger.info("Loading preprocessed data...")
        train_df = pd.read_pickle('data/processed/E1_baseline/train.pkl')
        val_df   = pd.read_pickle('data/processed/E1_baseline/val.pkl')
        test_df  = pd.read_pickle('data/processed/E1_baseline/test.pkl')

        # Define loss functions
        loss_functions = define_loss_functions()
        
        logger.info("\nLoss functions to test:")
        for loss_name, loss_config in loss_functions.items():
            logger.info(f"  {loss_name}: {loss_config['description']}")
        
        # Run experiments for each loss function
        all_results = {}
        
        for loss_name, loss_config in loss_functions.items():
            seed_results = []
            for seed in config['seeds']:
                seed_results.append(
                    run_loss_experiment(
                        loss_name,
                        loss_config,
                        train_df,
                        val_df,
                        test_df,
                        feature_columns,
                        seed=seed,
                        batch_size=config['batch_size'],
                        num_epochs=config['num_epochs'],
                        device=config['device']
                    )
                )

            all_results[loss_name] = aggregate_loss_results(seed_results, config['seeds'])
        
        # Compare results
        comparison_df = compare_loss_functions(all_results)
        
        # Save results
        save_dir = Path('results/E5_loss_functions')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(save_dir / 'E5_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save comparison CSV
        comparison_df.to_csv(save_dir / 'E5_comparison.csv', index=False)
        
        # Generate LaTeX table
        latex_table = generate_latex_table(comparison_df)
        with open(save_dir / 'E5_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        # Generate plots
        plot_comparison(comparison_df, save_dir)
        
        logger.info("\n" + "="*70)
        logger.info("E5: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
