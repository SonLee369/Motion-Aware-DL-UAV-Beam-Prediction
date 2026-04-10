"""
E7: Generalization Test Across Scenarios
Test model generalization across different deployment scenarios
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

from src.features.preprocessing import DeepSenseDataLoader, create_baseline_features
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.models.baseline import ImprovedBaselineBeamPredictor
from src.models.enhanced import EnhancedBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.dataset import BeamPredictionDataset

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E7_generalization.log'),
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


def _mean_std(values: List[float]) -> tuple:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) > 0 else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def aggregate_scenario_results(seed_results: List[Dict], seeds: List[int]) -> Dict:
    if not seed_results:
        return {}

    aggregated = {'seed_stats': {'seeds': seeds, 'count': len(seeds)}}
    scenarios = set()
    for seed_res in seed_results:
        scenarios.update(seed_res.keys())

    for scenario in scenarios:
        per_seed = [r[scenario] for r in seed_results if scenario in r]
        if not per_seed:
            continue

        aggregated[scenario] = {}
        aggregated[scenario]['num_samples'] = int(np.mean([
            r.get('num_samples', 0) for r in per_seed
        ]))

        aggregated[scenario]['overall'] = {}
        for metric_name in per_seed[0]['overall'].keys():
            vals = [r['overall'][metric_name] for r in per_seed]
            mean, std = _mean_std(vals)
            aggregated[scenario]['overall'][metric_name] = mean
            aggregated[scenario]['overall'][f'{metric_name}_std'] = std

    return aggregated


def load_scenario_data(scenario_path: str, scenario_name: str) -> pd.DataFrame:
    """
    Load and preprocess data from a specific scenario
    
    Args:
        scenario_path: Path to scenario data
        scenario_name: Name of the scenario
    
    Returns:
        Preprocessed dataframe
    """
    logger.info(f"\nLoading scenario: {scenario_name}")
    logger.info(f"Path: {scenario_path}")
    
    # Load raw data
    loader = DeepSenseDataLoader(scenario_path)
    df = loader.load_raw_data()
    
    # Validate
    loader.validate_data()
    
    # Get statistics
    stats = loader.get_statistics()
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Number of sequences: {stats['num_sequences']}")
    
    # Create features
    df = create_baseline_features(df)
    
    # Add velocity features
    df = add_velocity_features(df, smooth=True, window_size=5, poly_order=2)
    
    # Normalize (will be done globally later)
    df['scenario'] = scenario_name
    
    return df


def prepare_cross_scenario_data(
    train_scenarios: List[str],
    test_scenarios: List[str],
    data_root: str = 'data/raw'
) -> tuple:
    """
    Prepare data for cross-scenario evaluation
    
    Args:
        train_scenarios: List of training scenario names
        test_scenarios: List of test scenario names
        data_root: Root directory for data
    
    Returns:
        train_df, test_dfs (dict of test dataframes)
    """
    logger.info("="*70)
    logger.info("Preparing Cross-Scenario Data")
    logger.info("="*70)
    
    # Load training scenarios
    train_dfs = []
    for scenario in train_scenarios:
        scenario_path = Path(data_root) / scenario
        if scenario_path.exists():
            df = load_scenario_data(str(scenario_path), scenario)
            train_dfs.append(df)
        else:
            logger.warning(f"Scenario not found: {scenario_path}")
    
    # Combine training data
    if len(train_dfs) > 0:
        train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"\nCombined training data: {len(train_df)} samples")
    else:
        raise ValueError("No training data loaded!")
    
    # Normalize training data
    from src.features.preprocessing import normalize_geodetic
    train_df, norm_params = normalize_geodetic(train_df)
    train_df, v_norm_params = normalize_velocity_features(train_df, v_max=30.0)
    
    # Load and normalize test scenarios
    test_dfs = {}
    for scenario in test_scenarios:
        scenario_path = Path(data_root) / scenario
        if scenario_path.exists():
            df = load_scenario_data(str(scenario_path), scenario)
            
            # Apply same normalization as training
            df, _ = normalize_geodetic(df, fit_on=train_df)
            df, _ = normalize_velocity_features(df, v_max=v_norm_params['v_max'], fit_on=train_df)
            
            test_dfs[scenario] = df
        else:
            logger.warning(f"Test scenario not found: {scenario_path}")
    
    return train_df, test_dfs, norm_params, v_norm_params


def train_model_on_scenarios(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    model_type: str = 'baseline',
    seed: int = 42,
    batch_size: int = 64,
    num_epochs: int = 20,
    device: str = 'cuda'
) -> nn.Module:
    """
    Train model on combined scenario data
    
    Args:
        train_df: Training dataframe
        feature_columns: Feature column names
        model_type: 'baseline' or 'enhanced'
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Device for training
    
    Returns:
        Trained model
    """
    logger.info("\n" + "="*70)
    logger.info(f"Training {model_type.upper()} Model on Combined Scenarios")
    logger.info("="*70)

    set_random_seed(seed)
    
    # Clean data
    train_df_clean = train_df.dropna(subset=feature_columns)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(train_df_clean)),
        test_size=0.15,
        random_state=seed
    )
    
    train_subset = train_df_clean.iloc[train_indices].reset_index(drop=True)
    val_subset = train_df_clean.iloc[val_indices].reset_index(drop=True)
    
    logger.info(f"Train samples: {len(train_subset)}")
    logger.info(f"Val samples: {len(val_subset)}")
    
    # Create datasets
    train_dataset = BeamPredictionDataset(
        train_subset,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4
    )
    
    val_dataset = BeamPredictionDataset(
        val_subset,
        feature_columns=feature_columns,
        window_size=8,
        num_predictions=4
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=4)
    
    # Create model
    input_dim = len(feature_columns)
    
    if model_type == 'baseline':
        model = ImprovedBaselineBeamPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_beams=32,
            num_predictions=4,
            dropout=0.1
        )
    elif model_type == 'enhanced':
        model = EnhancedBeamPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_beams=32,
            num_predictions=4,
            use_multiscale=True,
            use_attention=True,
            position_dim=5,
            motion_dim=input_dim - 5,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train
    save_dir = f'checkpoints/E7_generalization/{model_type}'
    log_dir = f'runs/E7_generalization/{model_type}'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        learning_rate=5e-4,
        num_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping_patience=5
    )
    
    trainer.train()
    
    # Load best model
    best_model_path = Path(save_dir) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)
    
    return model


def evaluate_on_scenarios(
    model: nn.Module,
    test_dfs: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    batch_size: int = 128,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model on multiple test scenarios
    
    Args:
        model: Trained model
        test_dfs: Dictionary of test dataframes
        feature_columns: Feature column names
        batch_size: Batch size
        device: Device
    
    Returns:
        Dictionary with results for each scenario
    """
    logger.info("\n" + "="*70)
    logger.info("Evaluating on Test Scenarios")
    logger.info("="*70)
    
    all_results = {}
    
    for scenario_name, test_df in test_dfs.items():
        logger.info(f"\nEvaluating on: {scenario_name}")
        
        # Clean data
        test_df_clean = test_df.dropna(subset=feature_columns)
        
        # Create dataset
        test_dataset = BeamPredictionDataset(
            test_df_clean,
            feature_columns=feature_columns,
            window_size=8,
            num_predictions=4
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate
        results = evaluate_model(model, test_loader, device=device)
        results['scenario'] = scenario_name
        results['num_samples'] = len(test_dataset)
        
        all_results[scenario_name] = results
        
        logger.info(f"Top-1 Accuracy: {results['overall']['top1_acc']*100:.2f}%")
        logger.info(f"Top-3 Accuracy: {results['overall']['top3_acc']*100:.2f}%")
    
    return all_results


def compare_generalization(
    baseline_results: Dict,
    enhanced_results: Dict
) -> pd.DataFrame:
    """Compare generalization across scenarios"""
    
    logger.info("\n" + "="*70)
    logger.info("Comparing Generalization Performance")
    logger.info("="*70)
    
    comparison_data = []
    
    scenarios = set(baseline_results.keys()) | set(enhanced_results.keys())
    
    for scenario in scenarios:
        if scenario == 'seed_stats':
            continue
        row = {'Scenario': scenario}
        
        # Baseline results
        if scenario in baseline_results:
            baseline = baseline_results[scenario]
            row['Baseline_Top1'] = baseline['overall']['top1_acc'] * 100
            row['Baseline_Top3'] = baseline['overall']['top3_acc'] * 100
            row['Baseline_Top1_std'] = baseline['overall'].get('top1_acc_std', 0.0) * 100
            row['Baseline_Top3_std'] = baseline['overall'].get('top3_acc_std', 0.0) * 100
            row['Baseline_Samples'] = baseline['num_samples']
        
        # Enhanced results
        if scenario in enhanced_results:
            enhanced = enhanced_results[scenario]
            row['Enhanced_Top1'] = enhanced['overall']['top1_acc'] * 100
            row['Enhanced_Top3'] = enhanced['overall']['top3_acc'] * 100
            row['Enhanced_Top1_std'] = enhanced['overall'].get('top1_acc_std', 0.0) * 100
            row['Enhanced_Top3_std'] = enhanced['overall'].get('top3_acc_std', 0.0) * 100
            row['Enhanced_Samples'] = enhanced['num_samples']
        
        # Improvement
        if 'Baseline_Top1' in row and 'Enhanced_Top1' in row:
            row['Improvement_Top1'] = row['Enhanced_Top1'] - row['Baseline_Top1']
            row['Improvement_Top3'] = row['Enhanced_Top3'] - row['Baseline_Top3']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + df.to_string())
    
    return df


def generate_latex_table(comparison_df: pd.DataFrame) -> str:
    """Generate LaTeX comparison table"""
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{Cross-Scenario Generalization Results}
\label{tab:e7_generalization}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Scenario}} & \multicolumn{2}{c}{\textbf{Baseline}} & \multicolumn{2}{c}{\textbf{Enhanced}} & \multicolumn{2}{c}{\textbf{Improvement}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-1} & \textbf{Top-3} \\
\midrule
"""
    
    def fmt(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.2f} +/- {std_val:.2f}"

    for _, row in comparison_df.iterrows():
        latex += f"{row['Scenario']} & "
        
        if 'Baseline_Top1' in row:
            latex += (
                f"{fmt(row['Baseline_Top1'], row.get('Baseline_Top1_std', 0.0))} & "
                f"{fmt(row['Baseline_Top3'], row.get('Baseline_Top3_std', 0.0))} & "
            )
        else:
            latex += "-- & -- & "
        
        if 'Enhanced_Top1' in row:
            latex += (
                f"{fmt(row['Enhanced_Top1'], row.get('Enhanced_Top1_std', 0.0))} & "
                f"{fmt(row['Enhanced_Top3'], row.get('Enhanced_Top3_std', 0.0))} & "
            )
        else:
            latex += "-- & -- & "
        
        if 'Improvement_Top1' in row:
            latex += f"{row['Improvement_Top1']:+.2f} & {row['Improvement_Top3']:+.2f} \\\\\n"
        else:
            latex += "-- & -- \\\\\n"
    
    # Average
    if 'Baseline_Top1' in comparison_df.columns and 'Enhanced_Top1' in comparison_df.columns:
        avg_baseline_top1 = comparison_df['Baseline_Top1'].mean()
        avg_baseline_top3 = comparison_df['Baseline_Top3'].mean()
        avg_enhanced_top1 = comparison_df['Enhanced_Top1'].mean()
        avg_enhanced_top3 = comparison_df['Enhanced_Top3'].mean()
        avg_imp_top1 = avg_enhanced_top1 - avg_baseline_top1
        avg_imp_top3 = avg_enhanced_top3 - avg_baseline_top3
        
        latex += r"\midrule" + "\n"
        latex += f"\\textbf{{Average}} & {avg_baseline_top1:.2f} & {avg_baseline_top3:.2f} & "
        latex += f"{avg_enhanced_top1:.2f} & {avg_enhanced_top3:.2f} & "
        latex += f"{avg_imp_top1:+.2f} & {avg_imp_top3:+.2f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def plot_generalization_comparison(comparison_df: pd.DataFrame, save_dir: Path):
    """Plot generalization comparison visualizations"""
    
    # Plot 1: Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = comparison_df['Scenario']
    x = np.arange(len(scenarios))
    width = 0.35
    
    if 'Baseline_Top1' in comparison_df.columns:
        ax.bar(x - width/2, comparison_df['Baseline_Top1'], width,
               label='Baseline', color='steelblue', alpha=0.8)
    
    if 'Enhanced_Top1' in comparison_df.columns:
        ax.bar(x + width/2, comparison_df['Enhanced_Top1'], width,
               label='Enhanced', color='coral', alpha=0.8)
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Scenario Generalization: Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'E7_scenario_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'E7_scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Improvement heatmap
    if 'Improvement_Top1' in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        improvements = comparison_df[['Improvement_Top1', 'Improvement_Top3']].values
        
        sns.heatmap(improvements, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    xticklabels=['Top-1', 'Top-3'],
                    yticklabels=comparison_df['Scenario'],
                    cbar_kws={'label': 'Improvement (%)'},
                    ax=ax)
        
        ax.set_title('Cross-Scenario Improvement: Enhanced vs Baseline', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'E7_improvement_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'E7_improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Radar chart
    if 'Baseline_Top1' in comparison_df.columns and 'Enhanced_Top1' in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = comparison_df['Scenario'].tolist()
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        baseline_values = comparison_df['Baseline_Top1'].tolist()
        baseline_values += baseline_values[:1]
        
        enhanced_values = comparison_df['Enhanced_Top1'].tolist()
        enhanced_values += enhanced_values[:1]
        
        ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='steelblue')
        ax.fill(angles, baseline_values, alpha=0.25, color='steelblue')
        
        ax.plot(angles, enhanced_values, 'o-', linewidth=2, label='Enhanced', color='coral')
        ax.fill(angles, enhanced_values, alpha=0.25, color='coral')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title('Cross-Scenario Generalization (Top-1 Accuracy)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'E7_radar_chart.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'E7_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {save_dir}")


def main():
    """Main execution function"""
    
    logger.info("\n" + "="*70)
    logger.info("E7: CROSS-SCENARIO GENERALIZATION TEST")
    logger.info("="*70 + "\n")
    
    set_random_seed(42)
    
    config = {
        'data_root': 'data/raw',
        'train_scenarios': ['scenario23'],  # Training scenario
        'test_scenarios': ['scenario24', 'scenario25', 'scenario26'],  # Test scenarios
        'batch_size': 64,
        'num_epochs': 20,
        'seeds': [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Feature columns
    feature_columns = [
        'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
        'v_radial_norm', 'v_tangential_norm', 'v_mag_norm'
    ]
    
    try:
        # Prepare cross-scenario data
        train_df, test_dfs, norm_params, v_norm_params = prepare_cross_scenario_data(
            config['train_scenarios'],
            config['test_scenarios'],
            config['data_root']
        )
        
        # Train and evaluate baseline model (multi-seed)
        logger.info("\n" + "="*70)
        logger.info("Training Baseline Model")
        logger.info("="*70)
        baseline_seed_results = []
        for seed in config['seeds']:
            baseline_model = train_model_on_scenarios(
                train_df,
                feature_columns,
                model_type='baseline',
                seed=seed,
                batch_size=config['batch_size'],
                num_epochs=config['num_epochs'],
                device=config['device']
            )

            baseline_seed_results.append(
                evaluate_on_scenarios(
                    baseline_model,
                    test_dfs,
                    feature_columns,
                    batch_size=128,
                    device=config['device']
                )
            )

        baseline_results = aggregate_scenario_results(baseline_seed_results, config['seeds'])
        
        # Train and evaluate enhanced model (multi-seed)
        logger.info("\n" + "="*70)
        logger.info("Training Enhanced Model")
        logger.info("="*70)
        enhanced_seed_results = []
        for seed in config['seeds']:
            enhanced_model = train_model_on_scenarios(
                train_df,
                feature_columns,
                model_type='enhanced',
                seed=seed,
                batch_size=config['batch_size'],
                num_epochs=config['num_epochs'],
                device=config['device']
            )

            enhanced_seed_results.append(
                evaluate_on_scenarios(
                    enhanced_model,
                    test_dfs,
                    feature_columns,
                    batch_size=128,
                    device=config['device']
                )
            )

        enhanced_results = aggregate_scenario_results(enhanced_seed_results, config['seeds'])
        
        # Compare results
        comparison_df = compare_generalization(baseline_results, enhanced_results)
        
        # Save results
        save_dir = Path('results/E7_generalization')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        all_results = {
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'baseline_seeds': baseline_seed_results,
            'enhanced_seeds': enhanced_seed_results,
            'config': config
        }
        with open(save_dir / 'E7_all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save comparison CSV
        comparison_df.to_csv(save_dir / 'E7_comparison.csv', index=False)
        
        # Generate LaTeX table
        latex_table = generate_latex_table(comparison_df)
        with open(save_dir / 'E7_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        # Generate plots
        plot_generalization_comparison(comparison_df, save_dir)
        
        logger.info("\n" + "="*70)
        logger.info("E7: EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
