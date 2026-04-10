"""
E1: Baseline Reproduction Experiment
Reproduce baseline results from paper using position-only features
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

from src.features.preprocessing import (
    DeepSenseDataLoader,
    adjusted_splitting,
    normalize_geodetic,
    compute_ue_bs_unit_vector,
    remap_beam_to_M32,
    save_processed_data
)
from src.models.baseline import BaselineBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, evaluate_by_speed_category
from src.evaluation.report import generate_e1_report
from src.utils.dataset import BeamPredictionDataset, build_speed_labels

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E1_baseline.log'),
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
    torch.backends.cudnn.benchmark = False


def load_and_preprocess_data(data_path: str, random_seed: int = 42):
    """
    Load and preprocess data following paper's pipeline:
    1. Load raw data
    2. Remap beams N=64 -> M=32 (paper Section III)
    3. Normalize on FULL dataset BEFORE splitting (paper Section II-C)
    4. Compute UE-BS unit vectors
    5. Split into train/val/test
    
    Args:
        data_path: Path to raw data
        random_seed: Random seed
    
    Returns:
        train_df, val_df, test_df, norm_params
    """
    logger.info("="*70)
    logger.info("STEP 1: Loading and Preprocessing Data")
    logger.info("="*70)
    
    # 1. Load raw data
    loader = DeepSenseDataLoader(data_path)
    df = loader.load_raw_data()
    loader.validate_data()
    
    stats = loader.get_statistics()
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Number of sequences: {stats['num_sequences']}")
    logger.info(f"Latitude range: {stats['lat_range']}")
    logger.info(f"Longitude range: {stats['lon_range']}")
    
    # 2. Remap beams: N=64 oversampled -> M=32 (paper Section III)
    logger.info("\nRemapping beams N=64 -> M=32...")
    df = remap_beam_to_M32(df)
    
    # 3. Normalize on FULL dataset BEFORE splitting (paper Section II-C)
    logger.info("\nNormalizing geodetic coordinates on full dataset...")
    df, norm_params = normalize_geodetic(df)
    
    # 4. Compute UE-BS unit vector on full dataset
    logger.info("Computing UE-BS unit vectors...")
    df = compute_ue_bs_unit_vector(df)
    
    # 5. Split AFTER feature computation
    logger.info("\nSplitting data...")
    train_df, val_df, test_df = adjusted_splitting(
        df,
        train_ratio=0.65,
        val_ratio=0.15,
        test_ratio=0.20,
        random_seed=random_seed
    )
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    # Save processed data
    save_dir = Path('data/processed/E1_baseline')
    save_processed_data(train_df, val_df, test_df, save_dir, norm_params)
    
    return train_df, val_df, test_df, norm_params


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64
):
    """
    Create PyTorch dataloaders
    
    Args:
        train_df, val_df, test_df: Dataframes
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("="*70)
    logger.info("STEP 2: Creating Dataloaders")
    logger.info("="*70)
    
    # Baseline feature columns (position-only)
    feature_columns = [
        'lat_ue_norm',
        'lon_ue_norm',
        'u_ue_bs_x',
        'u_ue_bs_y',
        'u_ue_bs_z'
    ]
    
    logger.info(f"Feature columns: {feature_columns}")
    logger.info(f"Feature dimension: {len(feature_columns)}")
    
    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_df,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=False,
        include_speed_category=False,
        pad_initial=True,
    )

    val_dataset = BeamPredictionDataset(
        val_df,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=False,
        include_speed_category=False,
        pad_initial=True,
    )

    test_dataset = BeamPredictionDataset(
        test_df,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=False,
        include_speed_category=False,
        pad_initial=True,
    )

    # Test dataset WITH beam powers (for power loss computation)
    test_dataset_power = BeamPredictionDataset(
        test_df,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4,
        include_beam_powers=True,
        include_speed_category=False,
        pad_initial=True,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,  # Paper: val/test batch_size=1024
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,  # Paper: val/test batch_size=1024
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader_power = DataLoader(
        test_dataset_power,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, test_loader_power


def train_baseline_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    num_epochs: int = 20
):
    """
    Train baseline model
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device for training
        num_epochs: Number of epochs
    
    Returns:
        model, trainer
    """
    logger.info("="*70)
    logger.info("STEP 3: Training Baseline Model")
    logger.info("="*70)
    
    # Create model (paper: 3-layer CNN + 1-layer GRU, M=32 beams, ~0.99MB)
    model = BaselineBeamPredictor(
        input_dim=5,
        hidden_dim=128,
        num_beams=32,
        num_predictions=4,
        dropout=0.1
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer (paper: weight_decay=0, 20 fixed epochs)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        weight_decay=0.0,
        num_epochs=num_epochs,
        lr_schedule=[12, 18],
        lr_gamma=0.1,
        gradient_clip=0.0,  # Paper: no gradient clipping mentioned
        save_dir='checkpoints/E1_baseline',
        log_dir='runs/E1_baseline',
        early_stopping_patience=999
    )
    
    # Train
    trainer.train()
    
    # Load best model
    best_model_path = Path('checkpoints/E1_baseline/best_model.pth')
    if best_model_path.exists():
        logger.info(f"\nLoading best model from {best_model_path}")
        trainer.load_checkpoint(best_model_path)
    
    return model, trainer


def evaluate_baseline(
    model: nn.Module,
    test_loader: DataLoader,
    test_loader_power: DataLoader,
    test_df: pd.DataFrame,
    device: str = 'cuda'
):
    """
    Evaluate baseline model with power loss and speed category breakdown
    
    Args:
        model: Trained model
        test_loader: Test dataloader (no beam powers)
        test_loader_power: Test dataloader (with beam powers for power loss)
        test_df: Test dataframe (for speed category extraction)
        device: Device
    
    Returns:
        results: Dictionary with evaluation metrics
        speed_results: Dictionary with per-speed-category metrics
    """
    logger.info("="*70)
    logger.info("STEP 4: Evaluating Baseline Model")
    logger.info("="*70)
    
    # --- 4a. Standard evaluation with power loss ---
    results = evaluate_model(
        model=model,
        dataloader=test_loader_power,
        device=device,
        compute_power_loss=True
    )
    
    # --- 4b. Speed category evaluation ---
    logger.info("\nEvaluating by speed category...")
    speed_results = None
    try:
        # Use pre-loaded speed_mps column (m/s) from DeepSense unit2_speed files.
        # This is more reliable than GPS-derived velocity which requires
        # custom timestamp parsing and has NaN for first sample per sequence.
        if 'speed_mps' not in test_df.columns:
            logger.warning("'speed_mps' column not found in test_df. "
                           "Speed category evaluation skipped.")
            raise KeyError("speed_mps")
        
        # Log speed statistics
        valid_spd = test_df['speed_mps'].dropna()
        logger.info(f"Speed stats (m/s): min={valid_spd.min():.3f}, "
                    f"max={valid_spd.max():.3f}, mean={valid_spd.mean():.3f}")
        logger.info(f"Speed stats (mph): min={valid_spd.min()*2.23694:.2f}, "
                    f"max={valid_spd.max()*2.23694:.2f}, "
                    f"mean={valid_spd.mean()*2.23694:.2f}")
        
        # Build speed labels using shared utility (handles consecutiveness
        # check and last-element speed, matching author's approach)
        speed_labels = build_speed_labels(
            df=test_df,
            dataset=test_loader.dataset,
            window_size=8,
            num_predictions=4,
            pad_initial=True,
        )
        if speed_labels is None:
            raise ValueError("Could not build speed labels")
        
        speed_results = evaluate_by_speed_category(
            model=model,
            dataloader=test_loader,
            speed_labels=speed_labels,
            device=device
        )
    except Exception as e:
        logger.warning(f"Speed category evaluation failed: {e}")
        speed_results = None
    
    return results, speed_results


def generate_report(
    results: dict,
    trainer_history: dict,
    save_dir: str = 'results/E1_baseline',
    speed_results: dict = None,
    dataset_stats: dict = None,
    config: dict = None
):
    """
    Generate experiment report (PDF + plots)
    
    Args:
        results: Evaluation results
        trainer_history: Training history
        save_dir: Directory to save results
        speed_results: Per-speed-category evaluation results
        dataset_stats: Dataset split statistics
        config: Training configuration dict
    """
    logger.info("="*70)
    logger.info("STEP 5: Generating Report")
    logger.info("="*70)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots (needed for PDF embedding)
    plot_training_curves(trainer_history, save_path)
    plot_accuracy_by_step(results, save_path)
    
    # Generate PDF report (replaces .json / .txt / .tex)
    generate_e1_report(
        results=results,
        trainer_history=trainer_history,
        save_dir=save_dir,
        speed_results=speed_results,
        dataset_stats=dataset_stats,
        config=config,
    )
    logger.info(f"Report saved to {save_path / 'E1_report.pdf'}")


def generate_latex_table(results: dict) -> str:
    """Generate LaTeX table for results"""
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Baseline Model Performance (Position-Only Features)}
\label{tab:e1_baseline}
\begin{tabular}{lccc}
\toprule
\textbf{Prediction Step} & \textbf{Top-1 Acc (\%)} & \textbf{Top-3 Acc (\%)} & \textbf{Top-5 Acc (\%)} \\
\midrule
"""
    
    for v in range(4):
        metrics = results[f'v{v}']
        latex += f"$v={v}$ & {metrics['top1_acc']*100:.2f} & {metrics['top3_acc']*100:.2f} & {metrics['top5_acc']*100:.2f} \\\\\n"
    
    # Overall
    overall = results['overall']
    latex += r"\midrule" + "\n"
    latex += f"\\textbf{{Overall}} & \\textbf{{{overall['top1_acc']*100:.2f}}} & \\textbf{{{overall['top3_acc']*100:.2f}}} & \\textbf{{{overall['top5_acc']*100:.2f}}} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def plot_training_curves(history: dict, save_dir: Path):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history['val_accuracies'], 'g-', linewidth=2)
    ax2.axhline(y=history['best_val_acc'], color='r', linestyle='--', 
                label=f"Best: {history['best_val_acc']:.4f}")
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E1_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E1_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_dir}")


def plot_accuracy_by_step(results: dict, save_dir: Path):
    """Plot accuracy by prediction step"""
    
    steps = [0, 1, 2, 3]
    top1_accs = [results[f'v{v}']['top1_acc'] * 100 for v in steps]
    top3_accs = [results[f'v{v}']['top3_acc'] * 100 for v in steps]
    top5_accs = [results[f'v{v}']['top5_acc'] * 100 for v in steps]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(steps))
    width = 0.25
    
    ax.bar(x - width, top1_accs, width, label='Top-1', color='steelblue', alpha=0.8)
    ax.bar(x, top3_accs, width, label='Top-3', color='coral', alpha=0.8)
    ax.bar(x + width, top5_accs, width, label='Top-5', color='lightgreen', alpha=0.8)
    
    ax.set_xlabel('Prediction Step (v)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Baseline Model: Accuracy by Prediction Step', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'v={v}' for v in steps])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E1_accuracy_by_step.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E1_accuracy_by_step.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accuracy plot saved to {save_dir}")


def main():
    """Main execution function"""
    
    logger.info("\n" + "="*70)
    logger.info("E1: BASELINE REPRODUCTION EXPERIMENT")
    logger.info("="*70 + "\n")
    
    # Set random seed
    set_random_seed(42)
    
    # Configuration
    config = {
        'data_path': 'data/raw/scenario23',
        'batch_size': 8,  # Paper: train batch_size=8
        'num_epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': 42  # teacher uses seednum=42 for data splitting
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Step 1: Load and preprocess data
        train_df, val_df, test_df, norm_params = load_and_preprocess_data(
            config['data_path'],
            config['random_seed']
        )
        
        # Step 2: Create dataloaders
        train_loader, val_loader, test_loader, test_loader_power = create_dataloaders(
            train_df, val_df, test_df,
            batch_size=config['batch_size']
        )
        
        # Step 3: Train model
        model, trainer = train_baseline_model(
            train_loader,
            val_loader,
            device=config['device'],
            num_epochs=config['num_epochs']
        )
        
        # Step 4: Evaluate (with power loss + speed breakdown)
        results, speed_results = evaluate_baseline(
            model,
            test_loader,
            test_loader_power,
            test_df,
            device=config['device']
        )
        
        # Dataset statistics for the report
        dataset_stats = {
            'total': len(train_df) + len(val_df) + len(test_df),
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
        
        # Step 5: Generate report
        trainer_history = trainer.get_training_history()
        generate_report(results, trainer_history,
                        speed_results=speed_results,
                        dataset_stats=dataset_stats,
                        config=config)
        
        logger.info("\n" + "="*70)
        logger.info("E1: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
