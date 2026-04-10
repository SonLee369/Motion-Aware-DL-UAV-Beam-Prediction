"""
E2-Finalist: Rerun 4 finalist configs with 10 seeds + significance tests.
Configs: E2.0, E2.2, E2.13, E2.14
Outputs: CSV, JSON, logits, significance tests
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List
from scipy import stats

from src.features.preprocessing import adjusted_splitting
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.features.acceleration import add_acceleration_features, normalize_acceleration_features
from src.features.angular_rate import (
    add_angular_rate_features, add_directed_tangential_features,
    normalize_angular_rate_features, normalize_directed_tangential_features
)
from src.models.baseline import BaselineBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, evaluate_by_speed_category
from src.utils.dataset import BeamPredictionDataset, build_speed_labels

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E2_finalist.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_and_features():
    """Load E1 pkl, compute all motion features, return (feature_sets, train, val, test)."""
    logger.info("Loading E1 preprocessed data...")
    train_df = pd.read_pickle('data/processed/E1_baseline/train.pkl')
    val_df   = pd.read_pickle('data/processed/E1_baseline/val.pkl')
    test_df  = pd.read_pickle('data/processed/E1_baseline/test.pkl')

    # Concat for full-sequence feature computation (avoids SG boundary artifacts)
    train_df = train_df.copy(); train_df['_split'] = 'train'
    val_df   = val_df.copy();   val_df['_split']   = 'val'
    test_df  = test_df.copy();  test_df['_split']  = 'test'
    full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)

    full_df = add_velocity_features(full_df, smooth=True, window_size=5, poly_order=2)
    full_df = add_acceleration_features(full_df, window_size=5, poly_order=2)
    full_df = add_angular_rate_features(full_df, smooth=True, window_size=5, poly_order=2)
    full_df = add_directed_tangential_features(full_df, smooth=True, window_size=5, poly_order=2)

    train_df = full_df[full_df['_split'] == 'train'].drop(columns='_split').reset_index(drop=True)
    val_df   = full_df[full_df['_split'] == 'val'].drop(columns='_split').reset_index(drop=True)
    test_df  = full_df[full_df['_split'] == 'test'].drop(columns='_split').reset_index(drop=True)

    # Normalize (fit on train)
    train_df, v_p = normalize_velocity_features(train_df, v_max=30.0)
    val_df,  _    = normalize_velocity_features(val_df,  v_max=v_p['v_max'], fit_on=train_df)
    test_df, _    = normalize_velocity_features(test_df, v_max=v_p['v_max'], fit_on=train_df)

    train_df, a_p = normalize_acceleration_features(train_df, a_max=50.0)
    val_df,  _    = normalize_acceleration_features(val_df,  a_max=a_p['a_max'], fit_on=train_df)
    test_df, _    = normalize_acceleration_features(test_df, a_max=a_p['a_max'], fit_on=train_df)

    train_df, o_p = normalize_angular_rate_features(train_df, omega_max=0.1)
    val_df,  _    = normalize_angular_rate_features(val_df,  omega_max=o_p['omega_max'], fit_on=train_df)
    test_df, _    = normalize_angular_rate_features(test_df, omega_max=o_p['omega_max'], fit_on=train_df)

    train_df, vt_p = normalize_directed_tangential_features(train_df, v_max=30.0)
    val_df,  _     = normalize_directed_tangential_features(val_df,  v_max=vt_p['v_tan_max'], fit_on=train_df)
    test_df, _     = normalize_directed_tangential_features(test_df, v_max=vt_p['v_tan_max'], fit_on=train_df)

    # Define 4 finalist feature sets
    base_pos = ['lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z']
    feature_sets = {
        'E2_0_baseline': {
            'name': 'E2.0 Pos only',
            'features': base_pos,
            'description': 'Baseline: Position only (5D)'
        },
        'E2_2_plus_vradial': {
            'name': 'E2.2 + GPS v_radial',
            'features': base_pos + ['v_radial_norm'],
            'description': 'Position + GPS radial velocity (6D)'
        },
        'E2_13_plus_dir_tangential': {
            'name': 'E2.13 + dir v_tan',
            'features': base_pos + ['v_tan_x_norm', 'v_tan_y_norm', 'v_tan_z_norm'],
            'description': 'Position + directed tangential velocity 3D (8D)'
        },
        'E2_14_allvel_plus_angular_rate': {
            'name': 'E2.14 + all vel + omega',
            'features': base_pos + ['v_radial_norm', 'v_tangential_norm', 'v_mag_norm', 'angular_rate_norm'],
            'description': 'Position + all GPS velocity + angular rate (9D)'
        },
    }

    return feature_sets, train_df, val_df, test_df


def run_single(feature_config, train_df, val_df, test_df, seed, device):
    """Train one model, return results dict + logits tensor."""
    set_random_seed(seed)
    feature_columns = feature_config['features']
    input_dim = len(feature_columns)

    train_clean = train_df.dropna(subset=feature_columns)
    val_clean   = val_df.dropna(subset=feature_columns)
    test_clean  = test_df.dropna(subset=feature_columns)

    train_ds = BeamPredictionDataset(train_clean, feature_columns=feature_columns, window_size=8, num_predictions=4)
    val_ds   = BeamPredictionDataset(val_clean,   feature_columns=feature_columns, window_size=8, num_predictions=4)
    test_ds  = BeamPredictionDataset(test_clean,  feature_columns=feature_columns, window_size=8, num_predictions=4)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    model = BaselineBeamPredictor(input_dim=input_dim, hidden_dim=128, num_beams=32, num_predictions=4, dropout=0.1)
    save_dir = f'checkpoints/E2_finalist/{feature_config["name"]}/seed_{seed}'

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, learning_rate=5e-4, weight_decay=0.0, num_epochs=20,
        gradient_clip=0.0, save_dir=save_dir, log_dir=f'runs/E2_finalist/seed_{seed}',
        early_stopping_patience=999
    )
    trainer.train()

    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    # Evaluate
    results = evaluate_model(model, test_loader, device=device)

    # Speed evaluation
    try:
        speed_labels = build_speed_labels(df=test_clean, dataset=test_ds, window_size=8, num_predictions=4, pad_initial=True)
        if speed_labels is not None:
            results['speed_metrics'] = evaluate_by_speed_category(model, test_loader, speed_labels, device=device)
    except Exception as e:
        logger.warning(f"Speed eval failed: {e}")

    # Export logits [N, V=4, M=32]
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            features, beams = batch[0].to(device), batch[1]
            output = model(features)
            if isinstance(output, tuple):
                output = output[0]
            all_logits.append(output.cpu())
            all_targets.append(beams)
    logits = torch.cat(all_logits, dim=0)     # [N, V, M]
    targets = torch.cat(all_targets, dim=0)   # [N, V]

    results['seed'] = seed
    results['num_parameters'] = model.get_num_parameters()
    results['best_epoch'] = trainer.best_epoch if hasattr(trainer, 'best_epoch') else -1

    return results, logits, targets


def bootstrap_ci(a, b, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.sort(diffs)
    lo = diffs[int(n_boot * alpha / 2)]
    hi = diffs[int(n_boot * (1 - alpha / 2))]
    return float(lo), float(hi)


def significance_tests(per_seed_results: Dict[str, List[Dict]]):
    """Paired t-test + bootstrap CI for E2.2/E2.13/E2.14 vs E2.0."""
    baseline_key = 'E2_0_baseline'
    comparisons = [
        ('E2_2_plus_vradial', 'E2.2 vs E2.0'),
        ('E2_13_plus_dir_tangential', 'E2.13 vs E2.0'),
        ('E2_14_allvel_plus_angular_rate', 'E2.14 vs E2.0'),
    ]

    baseline_accs = np.array([r['overall']['top1_acc'] for r in per_seed_results[baseline_key]])
    sig_results = []

    for key, label in comparisons:
        accs = np.array([r['overall']['top1_acc'] for r in per_seed_results[key]])
        delta = accs - baseline_accs
        mean_delta = float(np.mean(delta))

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(accs, baseline_accs)

        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(accs, baseline_accs)

        # Speed breakdown
        speed_deltas = {}
        for speed_cat in ['slow', 'medium', 'fast']:
            try:
                base_speed = np.array([r['speed_metrics'][speed_cat]['overall']['top1_acc']
                                       for r in per_seed_results[baseline_key]])
                comp_speed = np.array([r['speed_metrics'][speed_cat]['overall']['top1_acc']
                                       for r in per_seed_results[key]])
                speed_deltas[speed_cat] = float(np.mean(comp_speed - base_speed))
            except (KeyError, TypeError):
                speed_deltas[speed_cat] = None

        sig_results.append({
            'comparison': label,
            'delta_mean': mean_delta * 100,
            'delta_std': float(np.std(delta, ddof=1)) * 100,
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'ci_95_lo': ci_lo * 100,
            'ci_95_hi': ci_hi * 100,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
            'speed_delta_slow': speed_deltas.get('slow'),
            'speed_delta_medium': speed_deltas.get('medium'),
            'speed_delta_fast': speed_deltas.get('fast'),
        })

    return sig_results


def main():
    SEEDS = [42, 77, 78, 79, 80, 81, 82, 83, 84, 85]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path('results/E2_finalist')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("E2-FINALIST: 4 configs x 10 seeds")
    logger.info("="*70)

    feature_sets, train_df, val_df, test_df = prepare_data_and_features()

    per_seed_results = {}   # key -> list of per-seed result dicts
    all_logits = {}         # key -> list of logit tensors per seed

    for fs_key, fs_config in feature_sets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"CONFIG: {fs_config['name']} ({len(fs_config['features'])}D)")
        logger.info(f"{'='*60}")

        per_seed_results[fs_key] = []
        all_logits[fs_key] = []

        for seed in SEEDS:
            logger.info(f"\n--- Seed {seed} ---")
            results, logits, targets = run_single(fs_config, train_df, val_df, test_df, seed, DEVICE)
            per_seed_results[fs_key].append(results)
            all_logits[fs_key].append(logits)

            logger.info(f"  Overall Top-1: {results['overall']['top1_acc']*100:.2f}%")

    # ── Save logits ──
    logits_dir = save_dir / 'logits'
    logits_dir.mkdir(exist_ok=True)
    # Save targets once (same test set for all)
    torch.save(targets, logits_dir / 'targets.pt')
    for fs_key in feature_sets:
        torch.save(all_logits[fs_key], logits_dir / f'{fs_key}_logits.pt')
    logger.info(f"Logits saved to {logits_dir}")

    # ── Aggregate results ──
    rows = []
    for fs_key, fs_config in feature_sets.items():
        seed_list = per_seed_results[fs_key]
        row = {'Config': fs_config['name'], 'Dim': len(fs_config['features'])}

        for metric_scope in ['overall'] + [f'v{v}' for v in range(4)]:
            vals = np.array([r[metric_scope]['top1_acc'] for r in seed_list]) * 100
            prefix = 'Overall' if metric_scope == 'overall' else metric_scope
            row[f'{prefix}_top1_mean'] = float(np.mean(vals))
            row[f'{prefix}_top1_std'] = float(np.std(vals, ddof=1))
            n = len(vals)
            ci = stats.t.ppf(0.975, n-1) * np.std(vals, ddof=1) / np.sqrt(n)
            row[f'{prefix}_top1_ci95'] = float(ci)

            vals3 = np.array([r[metric_scope]['top3_acc'] for r in seed_list]) * 100
            row[f'{prefix}_top3_mean'] = float(np.mean(vals3))

        # Speed breakdown
        for speed_cat in ['slow', 'medium', 'fast']:
            try:
                vals = np.array([r['speed_metrics'][speed_cat]['overall']['top1_acc']
                                 for r in seed_list]) * 100
                row[f'{speed_cat}_top1_mean'] = float(np.mean(vals))
                row[f'{speed_cat}_top1_std'] = float(np.std(vals, ddof=1))
            except (KeyError, TypeError):
                row[f'{speed_cat}_top1_mean'] = None
                row[f'{speed_cat}_top1_std'] = None

        row['Params'] = seed_list[0]['num_parameters']
        rows.append(row)

    df = pd.DataFrame(rows)
    # Improvement over baseline
    baseline_top1 = df.iloc[0]['Overall_top1_mean']
    df['Improvement'] = df['Overall_top1_mean'] - baseline_top1
    df.to_csv(save_dir / 'E2_finalist_comparison.csv', index=False)
    logger.info(f"\nComparison CSV saved.")
    logger.info("\n" + df.to_string())

    # ── Per-seed detail CSV ──
    detail_rows = []
    for fs_key, fs_config in feature_sets.items():
        for i, r in enumerate(per_seed_results[fs_key]):
            detail_rows.append({
                'Config': fs_config['name'],
                'Seed': SEEDS[i],
                'Overall_top1': r['overall']['top1_acc'] * 100,
                'Overall_top3': r['overall']['top3_acc'] * 100,
                'v0_top1': r['v0']['top1_acc'] * 100,
                'v1_top1': r['v1']['top1_acc'] * 100,
                'v2_top1': r['v2']['top1_acc'] * 100,
                'v3_top1': r['v3']['top1_acc'] * 100,
            })
    pd.DataFrame(detail_rows).to_csv(save_dir / 'E2_finalist_per_seed.csv', index=False)

    # ── Significance tests ──
    sig_results = significance_tests(per_seed_results)
    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(save_dir / 'E2_finalist_significance.csv', index=False)
    logger.info("\n--- Significance Tests ---")
    for s in sig_results:
        star = "***" if s['significant_001'] else ("*" if s['significant_005'] else "n.s.")
        logger.info(f"  {s['comparison']}: "
                     f"Δ={s['delta_mean']:+.2f}% (p={s['p_value']:.4f}) "
                     f"95%CI=[{s['ci_95_lo']:+.2f}, {s['ci_95_hi']:+.2f}] {star}")

    # ── Save full JSON ──
    json_out = {
        'seeds': SEEDS,
        'num_seeds': len(SEEDS),
        'configs': {},
        'significance': sig_results,
    }
    for fs_key, fs_config in feature_sets.items():
        config_data = {
            'name': fs_config['name'],
            'dim': len(fs_config['features']),
            'features': fs_config['features'],
            'per_seed': [],
        }
        for r in per_seed_results[fs_key]:
            seed_data = {
                'seed': r['seed'],
                'overall': r['overall'],
                'params': r['num_parameters'],
            }
            for v in range(4):
                seed_data[f'v{v}'] = r[f'v{v}']
            if r.get('speed_metrics'):
                # Convert speed metrics — extract overall top1 per category
                speed_summary = {}
                for cat in ['slow', 'medium', 'fast']:
                    if cat in r['speed_metrics']:
                        speed_summary[cat] = r['speed_metrics'][cat].get('overall', {})
                seed_data['speed'] = speed_summary
            config_data['per_seed'].append(seed_data)
        json_out['configs'][fs_key] = config_data

    with open(save_dir / 'E2_finalist_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {save_dir}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
