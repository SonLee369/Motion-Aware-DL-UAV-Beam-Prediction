"""
E4-Bridge: 3 architectures × {6D_vradial, 8D} feature sets
Answers: does feature choice matter more than architecture?

Usage:
    python run_e4_bridge.py             # 6D_vradial (default, already done)
    python run_e4_bridge.py --8d        # 8D rerun with fixed code
    python run_e4_bridge.py --all       # both

Uses same AblationBeamPredictor as E4, same training settings.
Outputs: CSV, JSON, logits, per-seed detail.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import OrderedDict
from scipy import stats

from src.models.ablation import AblationBeamPredictor
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, evaluate_by_speed_category, measure_inference_time
from src.utils.dataset import BeamPredictionDataset, build_speed_labels
from src.features.velocity import add_velocity_features, normalize_velocity_features
from src.features.acceleration import add_acceleration_features, normalize_acceleration_features

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/E4_bridge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

POSITION_DIM = 5
FEATURE_6D_VRADIAL = [
    'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
    'v_radial_norm',
]
FEATURE_8D = [
    'lat_ue_norm', 'lon_ue_norm', 'u_ue_bs_x', 'u_ue_bs_y', 'u_ue_bs_z',
    'v_radial_norm', 'v_tangential_norm', 'v_mag_norm',
]

FEATURE_SETS = {
    '6D_vradial': FEATURE_6D_VRADIAL,
    '8D': FEATURE_8D,
}

BRIDGE_ARCHS = OrderedDict([
    ('Bridge_Baseline', {
        'name': 'Baseline CNN-GRU',
        'cnn_type': 'baseline',
        'use_attention': False,
        'encoder_type': 'gru',
        'num_encoder_layers': 1,
        'hidden_dim': 128,
        'dropout': 0.1,
        'description': 'Paper baseline: 2-Conv + 1-layer GRU',
    }),
    ('Bridge_2L_GRU', {
        'name': '2-Layer GRU',
        'cnn_type': 'baseline',
        'use_attention': False,
        'encoder_type': 'gru',
        'num_encoder_layers': 2,
        'hidden_dim': 128,
        'dropout': 0.1,
        'description': '2-layer GRU encoder',
    }),
    ('Bridge_Transformer', {
        'name': 'Transformer Encoder',
        'cnn_type': 'baseline',
        'use_attention': False,
        'encoder_type': 'transformer',
        'num_encoder_layers': 2,
        'hidden_dim': 128,
        'dropout': 0.1,
        'description': 'Transformer encoder (2L, 4H)',
    }),
])


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load E2 velocity pkl (already has v_radial_norm)."""
    PKL_DIR = Path('data/processed/E2_feature_ablation')
    train_df = pd.read_pickle(PKL_DIR / 'train_with_velocity.pkl')
    val_df   = pd.read_pickle(PKL_DIR / 'val_with_velocity.pkl')
    test_df  = pd.read_pickle(PKL_DIR / 'test_with_velocity.pkl')
    logger.info(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def run_single(arch_key, arch_config, train_df, val_df, test_df, seed, device,
               feature_columns=None, run_tag='E4_bridge'):
    """Train one architecture, return results + logits."""
    set_random_seed(seed)
    if feature_columns is None:
        feature_columns = FEATURE_6D_VRADIAL

    train_clean = train_df.dropna(subset=feature_columns)
    val_clean   = val_df.dropna(subset=feature_columns)
    test_clean  = test_df.dropna(subset=feature_columns)

    train_ds = BeamPredictionDataset(train_clean, feature_columns=feature_columns, window_size=8, num_predictions=4)
    val_ds   = BeamPredictionDataset(val_clean,   feature_columns=feature_columns, window_size=8, num_predictions=4)
    test_ds  = BeamPredictionDataset(test_clean,  feature_columns=feature_columns, window_size=8, num_predictions=4)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

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

    save_dir = f'checkpoints/{run_tag}/{arch_key}/seed_{seed}'
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, learning_rate=5e-4, weight_decay=0.0, num_epochs=20,
        gradient_clip=0.0, save_dir=save_dir,
        log_dir=f'runs/{run_tag}/{arch_key}/seed_{seed}',
        early_stopping_patience=999,
    )
    trainer.train()

    best_path = Path(save_dir) / 'best_model.pth'
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    results = evaluate_model(model, test_loader, device=device)

    # Speed eval
    try:
        speed_labels = build_speed_labels(df=test_clean, dataset=test_ds, window_size=8, num_predictions=4, pad_initial=True)
        if speed_labels is not None:
            results['speed_metrics'] = evaluate_by_speed_category(model, test_loader, speed_labels, device=device)
    except Exception as e:
        logger.warning(f"Speed eval failed: {e}")

    # Inference time
    inference_ms = measure_inference_time(model, input_dim=input_dim, window_size=8, device=device)

    # Export logits
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            features, beams = batch[0].to(device), batch[1]
            output = model(features)
            if isinstance(output, tuple):
                output = output[0]
            all_logits.append(output.cpu())
            all_targets.append(beams)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    results['seed'] = seed
    results['num_parameters'] = model.get_num_parameters()
    results['inference_ms'] = inference_ms
    results['architecture'] = arch_config['name']

    return results, logits, targets


def run_feature_set(feature_set_name, feature_columns, seeds, device, train_df, val_df, test_df):
    """Run all 3 architectures for a given feature set."""
    run_tag = 'E4_bridge' if feature_set_name == '6D_vradial' else f'E4_bridge_{feature_set_name}'
    save_dir = Path(f'results/{run_tag}')
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info(f"E4-BRIDGE: {feature_set_name} x 3 architectures x {len(seeds)} seeds")
    logger.info("="*70)

    per_seed_results = {}
    all_logits = {}

    for arch_key, arch_config in BRIDGE_ARCHS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ARCHITECTURE: {arch_config['name']} | FEATURES: {feature_set_name}")
        logger.info(f"{'='*60}")

        per_seed_results[arch_key] = []
        all_logits[arch_key] = []

        for seed in seeds:
            logger.info(f"\n--- Seed {seed} ---")
            results, logits, targets = run_single(
                arch_key, arch_config, train_df, val_df, test_df, seed, device,
                feature_columns=feature_columns, run_tag=run_tag,
            )
            per_seed_results[arch_key].append(results)
            all_logits[arch_key].append(logits)
            logger.info(f"  Overall Top-1: {results['overall']['top1_acc']*100:.2f}%  |  Inference: {results['inference_ms']:.2f}ms")

    # Save logits
    logits_dir = save_dir / 'logits'
    logits_dir.mkdir(exist_ok=True)
    torch.save(targets, logits_dir / 'targets.pt')
    for arch_key in BRIDGE_ARCHS:
        torch.save(all_logits[arch_key], logits_dir / f'{arch_key}_logits.pt')

    # Aggregate CSV
    rows = []
    for arch_key, arch_config in BRIDGE_ARCHS.items():
        seed_list = per_seed_results[arch_key]
        row = {
            'Architecture': arch_config['name'],
            'Params': seed_list[0]['num_parameters'],
            'Inference_ms': float(np.mean([r['inference_ms'] for r in seed_list])),
        }
        for scope in ['overall'] + [f'v{v}' for v in range(4)]:
            vals = np.array([r[scope]['top1_acc'] for r in seed_list]) * 100
            prefix = 'Overall' if scope == 'overall' else scope
            row[f'{prefix}_top1_mean'] = float(np.mean(vals))
            row[f'{prefix}_top1_std'] = float(np.std(vals, ddof=1))
            vals3 = np.array([r[scope]['top3_acc'] for r in seed_list]) * 100
            row[f'{prefix}_top3_mean'] = float(np.mean(vals3))

        for speed_cat in ['slow', 'medium', 'fast']:
            try:
                vals = np.array([r['speed_metrics'][speed_cat]['overall']['top1_acc']
                                 for r in seed_list]) * 100
                row[f'{speed_cat}_top1_mean'] = float(np.mean(vals))
                row[f'{speed_cat}_top1_std'] = float(np.std(vals, ddof=1))
            except (KeyError, TypeError):
                row[f'{speed_cat}_top1_mean'] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    baseline_top1 = df.iloc[0]['Overall_top1_mean']
    df['Improvement'] = df['Overall_top1_mean'] - baseline_top1
    prefix = run_tag.replace('/', '_')
    df.to_csv(save_dir / f'{prefix}_comparison.csv', index=False)

    # Per-seed detail
    detail_rows = []
    for arch_key, arch_config in BRIDGE_ARCHS.items():
        for i, r in enumerate(per_seed_results[arch_key]):
            detail_rows.append({
                'Architecture': arch_config['name'],
                'Seed': seeds[i],
                'Overall_top1': r['overall']['top1_acc'] * 100,
                'v0_top1': r['v0']['top1_acc'] * 100,
                'v1_top1': r['v1']['top1_acc'] * 100,
                'v2_top1': r['v2']['top1_acc'] * 100,
                'v3_top1': r['v3']['top1_acc'] * 100,
                'Inference_ms': r['inference_ms'],
            })
    pd.DataFrame(detail_rows).to_csv(save_dir / f'{prefix}_per_seed.csv', index=False)

    # JSON
    json_out = {'seeds': seeds, 'feature_set': feature_set_name, 'configs': {}}
    for arch_key, arch_config in BRIDGE_ARCHS.items():
        config_data = {
            'name': arch_config['name'],
            'params': per_seed_results[arch_key][0]['num_parameters'],
            'per_seed': [],
        }
        for r in per_seed_results[arch_key]:
            sd = {'seed': r['seed'], 'overall': r['overall'], 'inference_ms': r['inference_ms']}
            for v in range(4):
                sd[f'v{v}'] = r[f'v{v}']
            if r.get('speed_metrics'):
                speed_summary = {}
                for cat in ['slow', 'medium', 'fast']:
                    if cat in r['speed_metrics']:
                        speed_summary[cat] = r['speed_metrics'][cat].get('overall', {})
                sd['speed'] = speed_summary
            config_data['per_seed'].append(sd)
        json_out['configs'][arch_key] = config_data

    with open(save_dir / f'{prefix}_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, default=str)

    logger.info("\n" + "="*70)
    logger.info(f"{run_tag} RESULTS")
    logger.info("="*70)
    logger.info("\n" + df.to_string())
    logger.info(f"\nAll results saved to {save_dir}")
    logger.info("DONE.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--8d', dest='run_8d', action='store_true', help='Run 8D feature set')
    parser.add_argument('--6d', dest='run_6d', action='store_true', help='Run 6D_vradial feature set')
    parser.add_argument('--all', dest='run_all', action='store_true', help='Run both feature sets')
    args = parser.parse_args()

    # Default: if nothing specified, run 6D
    if not args.run_8d and not args.run_6d and not args.run_all:
        args.run_6d = True

    SEEDS = [42, 77, 78, 79, 80]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_df, val_df, test_df = load_data()

    if args.run_6d or args.run_all:
        run_feature_set('6D_vradial', FEATURE_6D_VRADIAL, SEEDS, DEVICE, train_df, val_df, test_df)

    if args.run_8d or args.run_all:
        run_feature_set('8D', FEATURE_8D, SEEDS, DEVICE, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
