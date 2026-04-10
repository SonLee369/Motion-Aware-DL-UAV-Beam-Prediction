"""
E8: Final Comprehensive Evaluation
Complete evaluation of best model with detailed analysis
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
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report

from src.models.enhanced import EnhancedBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    evaluate_model,
    evaluate_by_speed_category,
    compute_confusion_matrix,
    compute_top_k_accuracy
)
from src.utils.dataset import BeamPredictionDataset
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.features.acceleration import add_acceleration_features, normalize_acceleration_features

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E8_final_evaluation.log'),
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

        aggregated[speed_name]['overall'] = {}
        for metric_name in per_seed[0][speed_name]['overall'].keys():
            vals = [m[speed_name]['overall'][metric_name] for m in per_seed]
            mean, std = _mean_std(vals)
            aggregated[speed_name]['overall'][metric_name] = mean
            aggregated[speed_name]['overall'][f'{metric_name}_std'] = std

    return aggregated


def aggregate_e8_results(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    base = seed_results[0]
    aggregated = {
        'seed_stats': {'seeds': seeds, 'count': len(seeds)},
        'confusion_matrices': base.get('confusion_matrices'),
        'error_analysis': base.get('error_analysis')
    }

    aggregated['overall'] = {}
    for v_key in [k for k in base['overall'].keys() if k.startswith('v')]:
        aggregated['overall'][v_key] = {}
        for metric_name in base['overall'][v_key].keys():
            vals = [r['overall'][v_key][metric_name] for r in seed_results]
            mean, std = _mean_std(vals)
            aggregated['overall'][v_key][metric_name] = mean
            aggregated['overall'][v_key][f'{metric_name}_std'] = std

    aggregated['overall']['overall'] = {}
    for metric_name in base['overall']['overall'].keys():
        vals = [r['overall']['overall'][metric_name] for r in seed_results]
        mean, std = _mean_std(vals)
        aggregated['overall']['overall'][metric_name] = mean
        aggregated['overall']['overall'][f'{metric_name}_std'] = std

    aggregated['topk'] = {}
    for v_key in [k for k in base['topk'].keys() if k.startswith('v')]:
        aggregated['topk'][v_key] = {}
        for metric_name in base['topk'][v_key].keys():
            vals = [r['topk'][v_key][metric_name] for r in seed_results]
            mean, std = _mean_std(vals)
            aggregated['topk'][v_key][metric_name] = mean
            aggregated['topk'][v_key][f'{metric_name}_std'] = std

    aggregated['topk']['overall'] = {}
    for metric_name in base['topk']['overall'].keys():
        vals = [r['topk']['overall'][metric_name] for r in seed_results]
        mean, std = _mean_std(vals)
        aggregated['topk']['overall'][metric_name] = mean
        aggregated['topk']['overall'][f'{metric_name}_std'] = std

    aggregated['by_speed'] = aggregate_speed_results([r.get('by_speed') for r in seed_results])

    return aggregated


def train_best_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: List[str],
    best_config: Dict,
    seed: int,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Train the best model with optimal configuration
    
    Returns:
        Trained model and training history
    """
    logger.info("="*70)
    logger.info("Training Best Model Configuration")
    logger.info("="*70)
    logger.info(f"Configuration: {best_config}")

    set_random_seed(seed)
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    val_df_clean = val_df.dropna(subset=feature_columns)
    
    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_df_clean,
        feature_columns=feature_columns,
        window_size=best_config['window_size'],
        num_predictions=4
    )
    
    val_dataset = BeamPredictionDataset(
        val_df_clean,
        feature_columns=feature_columns,
        window_size=best_config['window_size'],
        num_predictions=4
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_config['batch_size']*2,
        shuffle=False,
        num_workers=4
    )
    
    # Create best model
    input_dim = len(feature_columns)
    model = EnhancedBeamPredictor(
        input_dim=input_dim,
        hidden_dim=best_config['hidden_dim'],
        num_beams=32,
        num_predictions=4,
        use_multiscale=True,
        use_attention=True,
        position_dim=5,
        motion_dim=input_dim - 5,
        dropout=best_config['dropout']
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    save_dir = f'checkpoints/E8_final_evaluation/seed_{seed}'
    log_dir = f'runs/E8_final_evaluation/seed_{seed}'

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=best_config['learning_rate'],
        num_epochs=best_config['num_epochs'],
        lr_schedule=best_config['lr_schedule'],
        lr_gamma=best_config['lr_gamma'],
        gradient_clip=best_config['gradient_clip'],
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping_patience=5
    )
    
    trainer.train()
    
    # Load best model
    best_model_path = Path(save_dir) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)
    
    training_history = trainer.get_training_history()
    
    return model, training_history


def comprehensive_evaluation(
    model: nn.Module,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    batch_size: int = 128,
    device: str = 'cuda'
) -> Dict:
    """
    Perform comprehensive evaluation
    
    Returns:
        Dictionary with all evaluation results
    """
    logger.info("\n" + "="*70)
    logger.info("Comprehensive Evaluation")
    logger.info("="*70)
    
    # Clean data
    test_df_clean = test_df.dropna(subset=feature_columns)
    
    # Create dataset
    test_dataset = BeamPredictionDataset(
        test_df_clean,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_speed_category=True
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    results = {}
    
    # 1. Overall evaluation
    logger.info("\n1. Overall Evaluation")
    overall_results = evaluate_model(model, test_loader, device=device)
    results['overall'] = overall_results
    
    # 2. Evaluation by speed category
    logger.info("\n2. Evaluation by Speed Category")
    speed_labels = np.array([sample['speed_category'] for sample in test_dataset.samples])
    speed_results = evaluate_by_speed_category(model, test_loader, speed_labels, device)
    results['by_speed'] = speed_results
    
    # 3. Confusion matrices
    logger.info("\n3. Computing Confusion Matrices")
    confusion_matrices = compute_confusion_matrices(model, test_loader, device)
    results['confusion_matrices'] = confusion_matrices
    
    # 4. Top-K accuracies
    logger.info("\n4. Computing Top-K Accuracies")
    topk_results = compute_topk_accuracies(model, test_loader, device, k_values=[1, 3, 5, 10])
    results['topk'] = topk_results
    
    # 5. Error analysis
    logger.info("\n5. Error Analysis")
    error_analysis = analyze_prediction_errors(model, test_loader, device)
    results['error_analysis'] = error_analysis
    
    return results


def compute_confusion_matrices(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict:
    """Compute confusion matrices for each prediction step"""
    
    model.eval()
    
    all_predictions = {v: [] for v in range(4)}
    all_targets = {v: [] for v in range(4)}
    
    with torch.no_grad():
        for batch_data in dataloader:
            features = batch_data[0].to(device)
            beams = batch_data[1].to(device)
            
            # Forward
            output = model(features)
            if isinstance(output, tuple):
                predictions, _ = output
            else:
                predictions = output
            
            # Collect predictions and targets for each step
            for v in range(4):
                pred_beams = predictions[:, v, :].argmax(dim=1).cpu().numpy()
                target_beams = beams[:, v].cpu().numpy()
                
                all_predictions[v].extend(pred_beams)
                all_targets[v].extend(target_beams)
    
    # Compute confusion matrices
    confusion_matrices = {}
    for v in range(4):
        cm = confusion_matrix(
            all_targets[v],
            all_predictions[v],
            labels=list(range(32))
        )
        confusion_matrices[f'v{v}'] = cm.tolist()
    
    return confusion_matrices


def compute_topk_accuracies(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """Compute top-k accuracies for different k values"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            features = batch_data[0].to(device)
            beams = batch_data[1].to(device)
            
            # Forward
            output = model(features)
            if isinstance(output, tuple):
                predictions, _ = output
            else:
                predictions = output
            
            all_predictions.append(predictions.cpu())
            all_targets.append(beams.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute top-k for each prediction step
    topk_results = {}
    
    for v in range(4):
        preds_v = all_predictions[:, v, :]
        targets_v = all_targets[:, v]
        
        topk_results[f'v{v}'] = {}
        for k in k_values:
            acc = compute_top_k_accuracy(preds_v, targets_v, k=k)
            topk_results[f'v{v}'][f'top{k}'] = acc * 100
    
    # Overall
    topk_results['overall'] = {}
    for k in k_values:
        avg_acc = np.mean([topk_results[f'v{v}'][f'top{k}'] for v in range(4)])
        topk_results['overall'][f'top{k}'] = avg_acc
    
    return topk_results


def analyze_prediction_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Dict:
    """Analyze prediction errors"""
    
    model.eval()
    
    errors_by_step = {v: [] for v in range(4)}
    correct_by_step = {v: 0 for v in range(4)}
    total_by_step = {v: 0 for v in range(4)}
    
    with torch.no_grad():
        for batch_data in dataloader:
            features = batch_data[0].to(device)
            beams = batch_data[1].to(device)
            
            # Forward
            output = model(features)
            if isinstance(output, tuple):
                predictions, _ = output
            else:
                predictions = output
            
            # Analyze each step
            for v in range(4):
                pred_beams = predictions[:, v, :].argmax(dim=1)
                target_beams = beams[:, v]
                
                # Error magnitude (beam index difference)
                errors = torch.abs(pred_beams - target_beams).cpu().numpy()
                errors_by_step[v].extend(errors)
                
                # Accuracy
                correct = (pred_beams == target_beams).sum().item()
                correct_by_step[v] += correct
                total_by_step[v] += len(target_beams)
    
    # Compute statistics
    error_analysis = {}
    
    for v in range(4):
        errors = np.array(errors_by_step[v])
        
        error_analysis[f'v{v}'] = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'max_error': int(np.max(errors)),
            'accuracy': correct_by_step[v] / total_by_step[v] * 100
        }
    
    return error_analysis


def generate_comprehensive_report(
    results: Dict,
    training_history: Dict,
    save_dir: Path
):
    """Generate comprehensive evaluation report"""
    
    logger.info("\n" + "="*70)
    logger.info("Generating Comprehensive Report")
    logger.info("="*70)
    
    # 1. Summary statistics
    generate_summary_statistics(results, training_history, save_dir)
    
    # 2. LaTeX tables
    generate_all_latex_tables(results, save_dir)
    
    # 3. Visualizations
    generate_all_visualizations(results, training_history, save_dir)
    
    # 4. Detailed text report
    generate_detailed_report(results, training_history, save_dir)


def generate_summary_statistics(results: Dict, training_history: Dict, save_dir: Path):
    """Generate summary statistics"""
    
    summary = {
        'overall_performance': {
            'top1_accuracy': results['overall']['overall']['top1_acc'] * 100,
            'top3_accuracy': results['overall']['overall']['top3_acc'] * 100,
            'top5_accuracy': results['overall']['overall']['top5_acc'] * 100,
            'top1_accuracy_std': results['overall']['overall'].get('top1_acc_std', 0.0) * 100,
            'top3_accuracy_std': results['overall']['overall'].get('top3_acc_std', 0.0) * 100,
            'top5_accuracy_std': results['overall']['overall'].get('top5_acc_std', 0.0) * 100
        },
        'by_prediction_step': {},
        'by_speed_category': {},
        'training': {
            'best_epoch': training_history['best_epoch'] + 1,
            'best_val_acc': training_history['best_val_acc'] * 100,
            'total_epochs': len(training_history['train_losses'])
        }
    }
    
    # By prediction step
    for v in range(4):
        summary['by_prediction_step'][f'v{v}'] = {
            'top1_acc': results['overall'][f'v{v}']['top1_acc'] * 100,
            'top3_acc': results['overall'][f'v{v}']['top3_acc'] * 100,
            'top1_acc_std': results['overall'][f'v{v}'].get('top1_acc_std', 0.0) * 100,
            'top3_acc_std': results['overall'][f'v{v}'].get('top3_acc_std', 0.0) * 100
        }
    
    # By speed category
    if 'by_speed' in results:
        for speed in ['slow', 'medium', 'fast']:
            if speed in results['by_speed']:
                summary['by_speed_category'][speed] = {
                    'top1_acc': results['by_speed'][speed]['overall']['top1_acc'] * 100,
                    'top1_acc_std': results['by_speed'][speed]['overall'].get('top1_acc_std', 0.0) * 100
                }
    
    # Save summary
    with open(save_dir / 'E8_summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary statistics saved to {save_dir / 'E8_summary_statistics.json'}")


def generate_all_latex_tables(results: Dict, save_dir: Path):
    """Generate all LaTeX tables"""
    
    def fmt(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}"
    
    # Table 1: Overall performance
    table1 = r"""
\begin{table}[h]
\centering
\caption{Final Model: Overall Performance}
\label{tab:e8_overall}
\begin{tabular}{lcccc}
	oprule
	extbf{Step} & \textbf{Top-1 (\%)} & \textbf{Top-3 (\%)} & \textbf{Top-5 (\%)} & \textbf{Top-10 (\%)} \\
\midrule
"""
    
    for v in range(4):
        topk = results['topk'][f'v{v}']
        table1 += (
            f"$v={v}$ & "
            f"{fmt(topk['top1'], topk.get('top1_std', 0.0))} & "
            f"{fmt(topk['top3'], topk.get('top3_std', 0.0))} & "
            f"{fmt(topk['top5'], topk.get('top5_std', 0.0))} & "
            f"{fmt(topk['top10'], topk.get('top10_std', 0.0))} \\\n"
        )
    
    overall_topk = results['topk']['overall']
    table1 += r"\midrule" + "\n"
    table1 += (
        f"\\textbf{{Overall}} & "
        f"{fmt(overall_topk['top1'], overall_topk.get('top1_std', 0.0))} & "
        f"{fmt(overall_topk['top3'], overall_topk.get('top3_std', 0.0))} & "
        f"{fmt(overall_topk['top5'], overall_topk.get('top5_std', 0.0))} & "
        f"{fmt(overall_topk['top10'], overall_topk.get('top10_std', 0.0))} \\\n"
    )
    
    table1 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_dir / 'E8_table_overall.tex', 'w', encoding='utf-8') as f:
        f.write(table1)
    
    # Table 2: By speed category
    if 'by_speed' in results:
        table2 = r"""
\begin{table}[h]
\centering
\caption{Final Model: Performance by Speed Category}
\label{tab:e8_by_speed}
\begin{tabular}{lcccc}
	oprule
	extbf{Speed Category} & \textbf{Top-1 (\%)} & \textbf{Top-3 (\%)} & \textbf{Top-5 (\%)} \\
\midrule
"""
        
        for speed in ['slow', 'medium', 'fast']:
            if speed in results['by_speed']:
                speed_results = results['by_speed'][speed]['overall']
                table2 += (
                    f"{speed.capitalize()} & "
                    f"{fmt(speed_results['top1_acc']*100, speed_results.get('top1_acc_std', 0.0)*100)} & "
                    f"{fmt(speed_results['top3_acc']*100, speed_results.get('top3_acc_std', 0.0)*100)} & "
                    f"{fmt(speed_results['top5_acc']*100, speed_results.get('top5_acc_std', 0.0)*100)} \\\n"
                )
        
        table2 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(save_dir / 'E8_table_by_speed.tex', 'w', encoding='utf-8') as f:
            f.write(table2)
    
    logger.info(f"LaTeX tables saved to {save_dir}")


def generate_all_visualizations(results: Dict, training_history: Dict, save_dir: Path):
    """Generate all visualization plots"""
    
    # Plot 1: Training curves
    plot_training_curves(training_history, save_dir)
    
    # Plot 2: Top-K accuracy comparison
    plot_topk_comparison(results['topk'], save_dir)
    
    # Plot 3: Confusion matrix
    plot_confusion_matrix(results['confusion_matrices'], save_dir)
    
    # Plot 4: Error analysis
    plot_error_analysis(results['error_analysis'], save_dir)
    
    # Plot 5: Speed category comparison
    if 'by_speed' in results:
        plot_speed_comparison(results['by_speed'], save_dir)
    
    logger.info(f"All visualizations saved to {save_dir}")


def plot_training_curves(history: Dict, save_dir: Path):
    """Plot training curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['val_accuracies'], 'g-', linewidth=2)
    ax2.axhline(y=history['best_val_acc'], color='r', linestyle='--',
                label=f"Best: {history['best_val_acc']:.4f}")
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E8_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E8_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_topk_comparison(topk_results: Dict, save_dir: Path):
    """Plot top-k accuracy comparison"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = [0, 1, 2, 3]
    k_values = [1, 3, 5, 10]
    
    for k in k_values:
        accs = [topk_results[f'v{v}'][f'top{k}'] for v in steps]
        ax.plot(steps, accs, marker='o', linewidth=2, markersize=8, label=f'Top-{k}')
    
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Top-K Accuracy by Prediction Step', fontsize=14, fontweight='bold')
    ax.set_xticks(steps)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E8_topk_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E8_topk_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(confusion_matrices: Dict, save_dir: Path):
    """Plot confusion matrices"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for v in range(4):
        cm = np.array(confusion_matrices[f'v{v}'])
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[v]
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
        
        ax.set_xlabel('Predicted Beam', fontsize=11)
        ax.set_ylabel('True Beam', fontsize=11)
        ax.set_title(f'Confusion Matrix (v={v})', fontsize=12, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E8_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E8_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_analysis(error_analysis: Dict, save_dir: Path):
    """Plot error analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = [0, 1, 2, 3]
    
    # Mean error
    mean_errors = [error_analysis[f'v{v}']['mean_error'] for v in steps]
    std_errors = [error_analysis[f'v{v}']['std_error'] for v in steps]
    
    ax1.bar(steps, mean_errors, yerr=std_errors, capsize=5, color='coral', alpha=0.7)
    ax1.set_xlabel('Prediction Step (v)', fontsize=12)
    ax1.set_ylabel('Mean Error (beam indices)', fontsize=12)
    ax1.set_title('Mean Prediction Error by Step', fontsize=14, fontweight='bold')
    ax1.set_xticks(steps)
    ax1.grid(axis='y', alpha=0.3)
    
    # Accuracy
    accuracies = [error_analysis[f'v{v}']['accuracy'] for v in steps]
    ax2.plot(steps, accuracies, marker='o', linewidth=2, markersize=10, color='steelblue')
    ax2.set_xlabel('Prediction Step (v)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy by Prediction Step', fontsize=14, fontweight='bold')
    ax2.set_xticks(steps)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E8_error_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E8_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_speed_comparison(speed_results: Dict, save_dir: Path):
    """Plot speed category comparison"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    speeds = ['slow', 'medium', 'fast']
    steps = [0, 1, 2, 3]
    
    for speed in speeds:
        if speed in speed_results:
            accs = [speed_results[speed][f'v{v}']['top1_acc'] * 100 for v in steps]
            ax.plot(steps, accs, marker='o', linewidth=2, markersize=8, label=speed.capitalize())
    
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Performance by Speed Category', fontsize=14, fontweight='bold')
    ax.set_xticks(steps)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E8_speed_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E8_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_detailed_report(results: Dict, training_history: Dict, save_dir: Path):
    """Generate detailed text report"""
    
    report = f"""
{'='*70}
E8: FINAL COMPREHENSIVE EVALUATION REPORT
{'='*70}

1. TRAINING SUMMARY
{'='*70}
Total Epochs: {len(training_history['train_losses'])}
Best Epoch: {training_history['best_epoch'] + 1}
Best Validation Accuracy: {training_history['best_val_acc']*100:.2f}%

2. OVERALL PERFORMANCE
{'='*70}
"""
    
    def fmt(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}%"

    overall = results['topk']['overall']
    report += f"Top-1 Accuracy: {fmt(overall['top1'], overall.get('top1_std', 0.0))}\n"
    report += f"Top-3 Accuracy: {fmt(overall['top3'], overall.get('top3_std', 0.0))}\n"
    report += f"Top-5 Accuracy: {fmt(overall['top5'], overall.get('top5_std', 0.0))}\n"
    report += f"Top-10 Accuracy: {fmt(overall['top10'], overall.get('top10_std', 0.0))}\n"
    
    report += f"\n3. PERFORMANCE BY PREDICTION STEP\n{'='*70}\n"
    for v in range(4):
        topk = results['topk'][f'v{v}']
        report += f"\nStep v={v}:\n"
        report += f"  Top-1: {fmt(topk['top1'], topk.get('top1_std', 0.0))}\n"
        report += f"  Top-3: {fmt(topk['top3'], topk.get('top3_std', 0.0))}\n"
        report += f"  Top-5: {fmt(topk['top5'], topk.get('top5_std', 0.0))}\n"
    
    if 'by_speed' in results:
        report += f"\n4. PERFORMANCE BY SPEED CATEGORY\n{'='*70}\n"
        for speed in ['slow', 'medium', 'fast']:
            if speed in results['by_speed']:
                speed_overall = results['by_speed'][speed]['overall']
                report += (
                    f"{speed.capitalize()}: "
                    f"{fmt(speed_overall['top1_acc']*100, speed_overall.get('top1_acc_std', 0.0)*100)}\n"
                )
    
    report += f"\n5. ERROR ANALYSIS\n{'='*70}\n"
    for v in range(4):
        error = results['error_analysis'][f'v{v}']
        report += f"\nStep v={v}:\n"
        report += f"  Mean Error: {error['mean_error']:.2f} beam indices\n"
        report += f"  Std Error: {error['std_error']:.2f}\n"
        report += f"  Max Error: {error['max_error']} beam indices\n"
        report += f"  Accuracy: {error['accuracy']:.2f}%\n"
    
    report += f"\n{'='*70}\n"
    report += "END OF REPORT\n"
    report += f"{'='*70}\n"
    
    # Save report
    with open(save_dir / 'E8_detailed_report.txt', 'w') as f:
        f.write(report)
    
    logger.info(report)
    logger.info(f"Detailed report saved to {save_dir / 'E8_detailed_report.txt'}")


def main():
    """Main execution function"""
    
    logger.info("\n" + "="*70)
    logger.info("E8: FINAL COMPREHENSIVE EVALUATION")
    logger.info("="*70 + "\n")
    
    set_random_seed(42)
    
    # Best configuration (from E6 hyperparameter tuning)
    best_config = {
        'hidden_dim': 256,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'dropout': 0.15,
        'window_size': 8,
        'num_epochs': 25,
        'lr_schedule': [15, 22],
        'lr_gamma': 0.1,
        'gradient_clip': 1.0
    }
    
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_config': best_config,
        'seeds': [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    }
    
    # Feature columns (all features)
    feature_columns = [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'v_radial_norm', 'v_tangential_norm', 'v_mag_norm'
    ]
    
    try:
        # Load preprocessed data (falls back to building from E1 if E2 cache missing)
        logger.info("Loading preprocessed data...")
        train_df, val_df, test_df = _load_or_build_velocity_data()

        # Add speed categories
        def categorize_speed(v_mag):
            if pd.isna(v_mag):
                return 0
            elif v_mag < 4.47:
                return 0
            elif v_mag < 8.94:
                return 1
            else:
                return 2
        
        test_df['speed_category'] = test_df['v_mag'].apply(categorize_speed)
        
        seed_results = []
        seed_histories = []
        for seed in config['seeds']:
            model, training_history = train_best_model(
                train_df,
                val_df,
                feature_columns,
                best_config,
                seed=seed,
                device=config['device']
            )

            seed_histories.append(training_history)
            seed_results.append(
                comprehensive_evaluation(
                    model,
                    test_df,
                    feature_columns,
                    batch_size=128,
                    device=config['device']
                )
            )

        results = aggregate_e8_results(seed_results, config['seeds'])
        training_history = seed_histories[0] if seed_histories else {}
        
        # Save results
        save_dir = Path('results/E8_final_evaluation')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all results as JSON
        all_results = {
            'config': config,
            'results': results,
            'training_history': training_history,
            'seed_results': seed_results,
            'seed_histories': seed_histories
        }
        with open(save_dir / 'E8_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate comprehensive report
        generate_comprehensive_report(results, training_history, save_dir)
        
        logger.info("\n" + "="*70)
        logger.info("E8: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
