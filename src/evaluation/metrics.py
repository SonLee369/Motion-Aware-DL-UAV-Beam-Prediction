"""
Evaluation metrics for beam prediction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1
) -> float:
    """
    Compute top-k accuracy
    
    Args:
        predictions: [N, num_beams] logits or probabilities
        targets: [N] target beam indices
        k: Top-k parameter
    
    Returns:
        accuracy: Top-k accuracy in [0, 1]
    """
    with torch.no_grad():
        # Get top-k predictions
        _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True)
        
        # Check if target is in top-k
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1)
        
        accuracy = correct.float().mean().item()
    
    return accuracy


def compute_mean_power_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beam_powers: torch.Tensor
) -> float:
    """
    Compute mean power loss (dB)
    
    Args:
        predictions: [N, num_beams] logits
        targets: [N] target beam indices
        beam_powers: [N, num_beams] beam power values (linear scale)
    
    Returns:
        mean_power_loss: Mean power loss in dB
    """
    with torch.no_grad():
        # Get predicted beam
        pred_beams = predictions.argmax(dim=1)
        
        # Get predicted and optimal powers
        pred_powers = beam_powers.gather(1, pred_beams.unsqueeze(1)).squeeze(1)
        optimal_powers = beam_powers.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute power loss in dB
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        power_loss_db = 10 * torch.log10((optimal_powers + epsilon) / (pred_powers + epsilon))
        
        mean_power_loss = power_loss_db.mean().item()
    
    return mean_power_loss


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_beams: int = 32
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        predictions: [N, num_beams] logits
        targets: [N] target indices
        num_beams: Number of beams
    
    Returns:
        cm: [num_beams, num_beams] confusion matrix
    """
    with torch.no_grad():
        pred_beams = predictions.argmax(dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        cm = confusion_matrix(
            targets_np,
            pred_beams,
            labels=list(range(num_beams))
        )
    
    return cm


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    compute_power_loss: bool = False
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device for computation
        compute_power_loss: Whether to compute power loss (requires beam_powers)
    
    Returns:
        results: Dictionary with metrics for each prediction step
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_beam_powers = [] if compute_power_loss else None
    
    logger.info("Evaluating model...")
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpack batch
            if len(batch_data) == 2:
                features, beams = batch_data
                speed_cats = None
                beam_powers = None
            elif len(batch_data) == 3:
                if compute_power_loss:
                    features, beams, beam_powers = batch_data
                    speed_cats = None
                else:
                    features, beams, speed_cats = batch_data
                    beam_powers = None
            elif len(batch_data) == 4:
                features, beams, speed_cats, beam_powers = batch_data
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
            
            features = features.to(device)
            beams = beams.to(device)
            if speed_cats is not None:
                speed_cats = speed_cats.to(device)
            if beam_powers is not None:
                beam_powers = beam_powers.to(device)
            
            # Forward pass
            if speed_cats is not None:
                predictions = model(features, speed_cats)
            else:
                output = model(features)
                if isinstance(output, tuple):
                    predictions, _ = output
                else:
                    predictions = output
            
            all_predictions.append(predictions.cpu())
            all_targets.append(beams.cpu())
            
            if compute_power_loss and beam_powers is not None:
                all_beam_powers.append(beam_powers.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if compute_power_loss and all_beam_powers:
        all_beam_powers = torch.cat(all_beam_powers, dim=0)
    
    # Compute metrics for each prediction step
    num_predictions = all_predictions.size(1)
    results = {}
    
    for v in range(num_predictions):
        preds_v = all_predictions[:, v, :]
        targets_v = all_targets[:, v]
        
        metrics = {
            'top1_acc': compute_top_k_accuracy(preds_v, targets_v, k=1),
            'top3_acc': compute_top_k_accuracy(preds_v, targets_v, k=3),
            'top5_acc': compute_top_k_accuracy(preds_v, targets_v, k=5),
        }
        
        # Power loss (if available)
        if compute_power_loss and all_beam_powers is not None:
            powers_v = all_beam_powers[:, v, :]
            metrics['mean_power_loss_db'] = compute_mean_power_loss(
                preds_v, targets_v, powers_v
            )
        
        results[f'v{v}'] = metrics
        
        logger.info(f"Step v={v}: Top-1={metrics['top1_acc']:.4f}, "
                   f"Top-3={metrics['top3_acc']:.4f}, Top-5={metrics['top5_acc']:.4f}")
    
    # Overall metrics (average across steps)
    results['overall'] = {
        'top1_acc': np.mean([results[f'v{v}']['top1_acc'] for v in range(num_predictions)]),
        'top3_acc': np.mean([results[f'v{v}']['top3_acc'] for v in range(num_predictions)]),
        'top5_acc': np.mean([results[f'v{v}']['top5_acc'] for v in range(num_predictions)]),
    }
    
    if compute_power_loss and all_beam_powers is not None:
        results['overall']['mean_power_loss_db'] = np.mean([
            results[f'v{v}']['mean_power_loss_db'] for v in range(num_predictions)
        ])
    
    logger.info(f"\nOverall: Top-1={results['overall']['top1_acc']:.4f}, "
               f"Top-3={results['overall']['top3_acc']:.4f}")
    
    return results


def evaluate_by_speed_category(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    speed_labels: np.ndarray,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model separately for each speed category
    
    Args:
        model: PyTorch model
        dataloader: Test dataloader
        speed_labels: [N] speed category for each sample
        device: Device for computation
    
    Returns:
        results: Dictionary with metrics per speed category
    """
    model.eval()
    
    # Collect predictions and targets by speed
    results_by_speed = {
        'slow': {'predictions': [], 'targets': []},
        'medium': {'predictions': [], 'targets': []},
        'fast': {'predictions': [], 'targets': []}
    }
    
    speed_map = {0: 'slow', 1: 'medium', 2: 'fast'}
    
    logger.info("Evaluating by speed category...")
    
    with torch.no_grad():
        sample_idx = 0
        
        for batch_data in dataloader:
            # Unpack batch
            if len(batch_data) == 2:
                features, beams = batch_data
                speed_cats = None
            elif len(batch_data) == 3:
                features, beams, speed_cats = batch_data
            else:
                raise ValueError(f"Unexpected batch format")
            
            features = features.to(device)
            beams = beams.to(device)
            if speed_cats is not None:
                speed_cats = speed_cats.to(device)
            
            # Forward pass
            if speed_cats is not None:
                predictions = model(features, speed_cats)
            else:
                output = model(features)
                if isinstance(output, tuple):
                    predictions, _ = output
                else:
                    predictions = output
            
            # Group by speed category
            batch_size = features.size(0)
            for i in range(batch_size):
                speed_cat = speed_labels[sample_idx]
                speed_name = speed_map[speed_cat]
                
                results_by_speed[speed_name]['predictions'].append(
                    predictions[i].cpu()
                )
                results_by_speed[speed_name]['targets'].append(
                    beams[i].cpu()
                )
                
                sample_idx += 1
    
    # Compute metrics for each speed category
    results = {}
    
    for speed_name in ['slow', 'medium', 'fast']:
        if len(results_by_speed[speed_name]['predictions']) == 0:
            logger.warning(f"No samples for speed category: {speed_name}")
            continue
        
        preds = torch.stack(results_by_speed[speed_name]['predictions'])
        targets = torch.stack(results_by_speed[speed_name]['targets'])
        
        results[speed_name] = {}
        
        for v in range(preds.size(1)):
            preds_v = preds[:, v, :]
            targets_v = targets[:, v]
            
            results[speed_name][f'v{v}'] = {
                'top1_acc': compute_top_k_accuracy(preds_v, targets_v, k=1),
                'top3_acc': compute_top_k_accuracy(preds_v, targets_v, k=3),
                'top5_acc': compute_top_k_accuracy(preds_v, targets_v, k=5),
            }
        
        # Overall for this speed
        num_steps = preds.size(1)
        results[speed_name]['num_samples'] = preds.size(0)
        results[speed_name]['overall'] = {
            'top1_acc': np.mean([results[speed_name][f'v{v}']['top1_acc'] for v in range(num_steps)]),
            'top3_acc': np.mean([results[speed_name][f'v{v}']['top3_acc'] for v in range(num_steps)]),
        }
        
        logger.info(f"{speed_name.capitalize()}: Top-1={results[speed_name]['overall']['top1_acc']:.4f}")
    
    return results


# ---------------------------------------------------------------------------
# Overhead savings helpers (E10)
# ---------------------------------------------------------------------------

def _unpack_model_output(output):
    """Unpack model output — handles (logits, aux) tuple or plain logits."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _parse_batch(batch):
    """Parse batch tuple into (features, beams, speed_cats, beam_powers).

    Supports 2-, 3-, or 4-element batches. Returns None for missing elements.
    """
    if len(batch) == 2:
        return batch[0], batch[1], None, None
    if len(batch) == 3:
        features, beams, third = batch
        if torch.is_tensor(third) and third.ndim == 1:
            return features, beams, third, None
        if torch.is_tensor(third) and third.ndim == 3:
            return features, beams, None, third
        raise ValueError(f"Unexpected 3rd element shape: {getattr(third, 'shape', None)}")
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError(f"Unexpected batch format: {len(batch)} elements")


def compute_overhead_savings(
    predicted_probs: torch.Tensor,
    targets: torch.Tensor,
    reliability: float = 0.9,
) -> float:
    """Overhead savings for a single reliability target R.

    Args:
        predicted_probs: [N, M] softmax probabilities.
        targets: [N] true beam indices.
        reliability: Target reliability R in (0, 1].

    Returns:
        Overhead savings = 1 - b^(R) / M.
    """
    sorted_idx = torch.argsort(predicted_probs, dim=-1, descending=True)  # [N, M]
    ranks = (sorted_idx == targets.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1  # 1..M
    ranks_np = ranks.detach().cpu().numpy()
    b_required = int(np.quantile(ranks_np, reliability, method="higher"))
    m = predicted_probs.size(-1)
    return float(1.0 - (b_required / m))


def compute_min_beams_for_reliability(
    predicted_probs: torch.Tensor,
    targets: torch.Tensor,
    reliability: float = 0.9,
) -> int:
    """Return the minimum number of beams b^{(R)} to satisfy reliability R.

    Args:
        predicted_probs: [N, M] softmax probabilities.
        targets: [N] true beam indices.
        reliability: Desired reliability level R in (0, 1].

    Returns:
        b_required in [1, M].
    """
    sorted_idx = torch.argsort(predicted_probs, dim=-1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1
    ranks_np = ranks.detach().cpu().numpy()
    return int(np.quantile(ranks_np, reliability, method="higher"))


@torch.no_grad()
def collect_step_probabilities(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect softmax probabilities and targets for a given prediction step v.

    Args:
        model: Trained beam prediction model.
        dataloader: DataLoader (2-, 3-, or 4-tuple batches).
        device: Torch device string.
        step: Prediction step index (0 = next-step, 1 = 2-step, ...).

    Returns:
        probs: [N, M] softmax probabilities.
        targets: [N] true beam indices.
    """
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in dataloader:
        features, beams, speed_cats, _beam_powers = _parse_batch(batch)
        features = features.to(device)
        beams = beams.to(device)
        if speed_cats is not None:
            speed_cats = speed_cats.to(device)

        out = model(features, speed_categories=speed_cats) if speed_cats is not None else model(features)
        logits = _unpack_model_output(out)  # [B, T, M]
        probs = torch.softmax(logits[:, step, :], dim=-1)

        all_probs.append(probs.detach().cpu())
        all_targets.append(beams[:, step].detach().cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def compute_overhead_curve(
    predicted_probs: torch.Tensor,
    targets: torch.Tensor,
    reliabilities: List[float],
) -> Dict[str, Dict[str, float]]:
    """Compute overhead savings curve for a list of reliability levels.

    Args:
        predicted_probs: [N, M] softmax probabilities.
        targets: [N] true beam indices.
        reliabilities: List of R values, e.g. [0.8, 0.9, 0.95, 0.99].

    Returns:
        Dict keyed by str(R) with entries:
          {"reliability": R, "b_required": b, "overhead_savings": 1 - b/M}
    """
    m = predicted_probs.size(-1)
    out: Dict[str, Dict[str, float]] = {}
    for r in reliabilities:
        b_req = compute_min_beams_for_reliability(predicted_probs, targets, reliability=r)
        out[str(r)] = {
            "reliability": float(r),
            "b_required": float(b_req),
            "overhead_savings": float(1.0 - (b_req / m)),
        }
    return out


def compute_power_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beam_powers: torch.Tensor,
) -> Dict:
    """Compute power loss across all prediction horizons from saved logits.

    Args:
        logits: [N, V, M] raw model output.
        targets: [N, V] ground truth beam indices.
        beam_powers: [N, V, M] per-beam power values (linear scale).

    Returns:
        Dict with 'per_horizon' (list of PL per v) and 'overall' (mean PL).
    """
    V = logits.size(1)
    per_horizon = []
    for v in range(V):
        pl = compute_mean_power_loss(logits[:, v, :], targets[:, v], beam_powers[:, v, :])
        per_horizon.append(pl)
    return {'per_horizon': per_horizon, 'overall': float(np.mean(per_horizon))}


def measure_inference_time(
    model: torch.nn.Module,
    input_dim: int,
    window_size: int = 8,
    device: str = 'cuda',
    warmup: int = 20,
    repeats: int = 100,
) -> float:
    """Measure single-sample inference time in milliseconds.

    Args:
        model: Model in eval mode.
        input_dim: Feature dimension.
        window_size: Input sequence length (W).
        device: Torch device.
        warmup: Number of warmup runs.
        repeats: Number of timed runs.

    Returns:
        Average inference time in milliseconds.
    """
    import time
    model.eval()
    dummy = torch.randn(1, window_size, input_dim).to(device)
    use_cuda = device != 'cpu' and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


# Test code
if __name__ == "__main__":
    print("Testing Evaluation Metrics\n" + "="*50)
    
    # Create dummy data
    N = 1000
    num_beams = 32
    
    predictions = torch.randn(N, num_beams)
    targets = torch.randint(0, num_beams, (N,))
    beam_powers = torch.rand(N, num_beams)
    
    # Test top-k accuracy
    print("\n1. Top-k Accuracy")
    for k in [1, 3, 5]:
        acc = compute_top_k_accuracy(predictions, targets, k=k)
        print(f"Top-{k} Accuracy: {acc:.4f}")
    
    # Test power loss
    print("\n2. Mean Power Loss")
    power_loss = compute_mean_power_loss(predictions, targets, beam_powers)
    print(f"Mean Power Loss: {power_loss:.4f} dB")
    
    # Test confusion matrix
    print("\n3. Confusion Matrix")
    cm = compute_confusion_matrix(predictions, targets, num_beams=num_beams)
    print(f"Shape: {cm.shape}")
    print(f"Diagonal sum (correct predictions): {np.trace(cm)}")
