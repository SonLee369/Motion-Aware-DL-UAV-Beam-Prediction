# UAV GPS 5G Beam Prediction

> DeepSense 6G Scenario 23 — CNN-GRU beam predictor with motion-aware features.

---

## Results Summary (2026-03-26)

### Key Finding: Feature Choice Dominates Architecture

| Model | Top-1 (%) | Std | PL (dB) | Params | p vs Baseline |
|-------|-----------|-----|---------|--------|---------------|
| E2.0 Position only (5D) | 73.74 | 0.95 | 0.177 | 260K | — |
| **E2.2 + v_radial (6D)** | **74.60** | **0.41** | **0.167** | **260K** | **0.032 *** |
| E2.13 + dir v_tan (8D) | 74.54 | 1.09 | 0.173 | 260K | 0.132 n.s. |
| E2.14 + all vel + omega (9D) | 73.94 | 1.15 | 0.178 | 260K | 0.686 n.s. |
| Bridge-Transformer (6D_vrad) | 75.09 | 0.98 | 0.162 | 558K | 0.491 n.s. vs E2.2 |

- **E2.2 (+v_radial) is the only statistically significant improvement** (paired t-test, 10 seeds)
- Architecture changes within same feature set: ANOVA p=0.720 (not significant)
- 6D_vradial beats 8D across all 3 tested architectures (+0.67~1.07%)

### Experiment Status

| Exp | Status | Seeds | Key Result |
|-----|--------|-------|------------|
| E1 | Done | 5 | Baseline reproduction (bug fixes applied) |
| E2 | Done | 5 (original) + 10 (finalist) | v_radial = only significant feature (p=0.032) |
| E3 | Done | 5 | Standard training beats all speed-aware strategies |
| E4 | Done | 5 | Transformer best arch, but feature matters more |
| E4-Bridge | Done | 5 | 6D_vradial × 3 archs confirms feature dominance |
| E5 | Done | 5 | Label Smoothing ε=0.2 → +0.37% |
| E6 | Not started | — | Hyperparameter grid search |
| E7 | Blocked | — | Needs Scenario 24/25/26 data |
| E8 | Not started | — | Final evaluation |

---

## E2 Feature Ablation (10 seeds, finalist)

4 finalist configs × 10 seeds [42, 77-85]. Script: `experiments/E2_feature_ablation/run_e2_finalist.py`

### Per-Horizon Top-1 (%)

| Config | v0 | v1 | v2 | v3 |
|--------|----|----|----|----|
| E2.0 (5D) | 74.21 | 74.64 | 74.24 | 71.86 |
| **E2.2 (6D)** | **75.19** | **75.73** | **74.86** | **72.61** |
| E2.13 (8D) | 75.43 | 75.49 | 74.41 | 72.82 |
| E2.14 (9D) | 75.34 | 75.05 | 73.57 | 71.81 |

### Per-Speed Top-1 (%)

| Config | Slow | Medium | Fast |
|--------|------|--------|------|
| E2.0 | 78.36 | 65.12 | 62.15 |
| **E2.2** | **79.39** | **66.45** | 61.71 |
| E2.14 | 77.94 | **67.93** | **62.40** |

### Significance Tests

| Comparison | Delta | p-value | 95% Bootstrap CI | Significant? |
|------------|-------|---------|-------------------|-------------|
| E2.2 vs E2.0 | +0.86% | **0.032** | [+0.20, +1.45] | **Yes** |
| E2.13 vs E2.0 | +0.80% | 0.132 | [-0.15, +1.62] | No |
| E2.14 vs E2.0 | +0.20% | 0.686 | [-0.67, +1.14] | No |

---

## E4-Bridge: Feature vs Architecture (5 seeds)

6D_vradial × 3 architectures. Script: `experiments/E4_architecture/run_e4_bridge.py`

| Architecture | Top-1 (%) | Std | PL (dB) | v3 | Inference (ms) | Params |
|-------------|-----------|-----|---------|-----|---------------|--------|
| Baseline CNN-GRU | 74.70 | 0.54 | 0.167 | 72.57 | 1.07 | 260K |
| 2-Layer GRU | 74.91 | 0.64 | 0.170 | 72.52 | 1.11 | 360K |
| Transformer | 75.09 | 0.98 | 0.162 | 73.50 | 1.83 | 558K |

**ANOVA p=0.720** — architecture differences NOT significant within same feature set.

### 8D Rerun (fixed code, 5 seeds)

| Architecture | Top-1 (%) | Std | PL (dB) | v3 | Inference (ms) | Params |
|-------------|-----------|-----|---------|-----|---------------|--------|
| Baseline CNN-GRU | 73.43 | 1.06 | 0.178 | 71.68 | 1.14 | 261K |
| 2-Layer GRU | 74.11 | 0.96 | 0.185 | 71.74 | 1.15 | 360K |
| Transformer | 74.18 | 0.37 | 0.174 | 72.55 | 1.96 | 559K |

### Comparison: Same Architecture, Different Features (6D_vrad vs 8D)

| Architecture | 6D_vrad | 8D | Delta |
|-------------|---------|-----|-------|
| Baseline CNN-GRU | 74.70 | 73.43 | +1.27 |
| 2-Layer GRU | 74.91 | 74.11 | +0.80 |
| Transformer | 75.09 | 74.18 | +0.91 |

---

## E3: Speed-Aware Training (5 seeds)

Standard CrossEntropy beats all speed-aware strategies.

| Strategy | Top-1 (%) | Delta |
|----------|-----------|-------|
| E3.0 Baseline | 74.20 | — |
| E3.3 Weighted Loss | 74.29 | +0.09 |
| E3.1 Stratified | 71.90 | -2.30 |
| E3.4 Curriculum | 67.97 | -6.23 |

## E4: Architecture Ablation (5 seeds, 8 archs × 4 feature sets, pre-bugfix)

> Note: E4 original results used code before 3 bug fixes. E4-Bridge and 8D Rerun above use fixed code.

| Architecture | 8D | 6D_vmag | 6D_vtan | 6D_sensor |
|---|---|---|---|---|
| Baseline CNN-GRU | 73.85 | 71.91 | 72.59 | 72.92 |
| Multi-Scale CNN | 73.29 | 73.11 | 72.61 | 72.71 |
| 2-Layer GRU | 74.24 | 72.62 | 73.31 | 72.60 |
| **Transformer** | **74.02** | **73.05** | **73.52** | **73.49** |

## E5: Loss Function (5 seeds)

| Loss | Top-1 (%) | vs CE |
|------|-----------|-------|
| CrossEntropy | 69.54 | — |
| **Label Smoothing ε=0.2** | **69.92** | **+0.37** |
| Focal Loss γ=2 | 69.36 | -0.18 |

---

## System-Level Metrics

### Power Loss by Horizon (dB, lower is better)

| Config | Overall | v0 | v1 | v2 | v3 |
|--------|---------|----|----|----|----|
| E2.0 (5D) | 0.177 | 0.168 | 0.157 | 0.167 | 0.216 |
| **E2.2 (6D)** | **0.167** | 0.158 | **0.141** | 0.160 | 0.210 |
| Bridge-Trans | **0.162** | 0.153 | 0.144 | **0.155** | **0.197** |

### Overhead Saving

All models achieve MinB=2 at R=0.90 (93.8% overhead saving) and MinB=3 at R=0.95 (90.6%).

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (tested with CUDA 12.8)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download data

Download **Scenario 23** from [DeepSense 6G](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23) and place it in:

```
data/raw/scenario23/
```

### 4. Run experiments (in order)

Each experiment depends on the previous one's outputs (pkl files).

```bash
# E1: Baseline — generates data/processed/E1_baseline/*.pkl
python experiments/E1_baseline/run_baseline.py

# E2: Feature ablation — generates data/processed/E2_feature_ablation/*_with_velocity.pkl
python experiments/E2_feature_ablation/run_feature_ablation.py

# E2 finalist (10 seeds, requires E1 pkl)
python experiments/E2_feature_ablation/run_e2_finalist.py

# E3: Speed-aware training (requires E1 pkl)
python experiments/E3_speed_aware/run_speed_aware.py

# E4: Architecture ablation (tries E2 pkl, falls back to E1 pkl)
python experiments/E4_architecture/run_architecture_ablation.py

# E4 bridge (requires E2 pkl)
python experiments/E4_architecture/run_e4_bridge.py

# E5: Loss comparison (tries E2 pkl, falls back to E1 pkl)
python experiments/E5_loss_functions/run_loss_comparison.py
```

Results are saved to `results/` and logs to `logs/`.

**Check logs after each run:**

```bash
# View latest log (each experiment writes its own log file)
cat logs/E1_baseline.log          # E1
cat logs/E2_feature_ablation.log  # E2
cat logs/E2_finalist.log          # E2 finalist
cat logs/E3_speed_aware.log       # E3
cat logs/E4_architecture.log      # E4
cat logs/E4_bridge.log            # E4 bridge
cat logs/E5_loss_comparison.log   # E5

# Quick check: look for final accuracy and "DONE" at end of log
tail -20 logs/E2_finalist.log
```

### Dependency chain

```
E1 (raw → E1 pkl) → E2 (E1 pkl → E2 pkl)
                  → E2 finalist (E1 pkl, computes features internally)
                  → E3 (E1 pkl, computes features internally)
                  → E4 ablation, E5, E8 (E2 pkl preferred, fallback E1 pkl)
                  → E4 bridge (E2 pkl required)
```

---

## Project Structure

```
src/
  features/    preprocessing, velocity, acceleration
  models/      baseline.py, ablation.py, enhanced.py
  training/    trainer.py, losses.py, samplers.py
  evaluation/  metrics.py, report.py
  utils/       dataset.py

experiments/
  E1_baseline/           run_baseline.py
  E2_feature_ablation/   run_feature_ablation.py, run_e2_finalist.py
  E3_speed_aware/        run_speed_aware.py
  E4_architecture/       run_architecture_ablation.py, run_e4_bridge.py
  E5_loss_comparison/    run_loss_comparison.py

data/processed/
  E1_baseline/           {train,val,test}.pkl
  E2_feature_ablation/   {train,val,test}_with_velocity.pkl

results/
  E2_finalist/    CSV, JSON, logits/
  E4_bridge/      CSV, JSON, logits/
```

### Training Config (Paper Spec)

| Param | Value |
|-------|-------|
| Window (W) | 8 |
| Predictions (V) | 4 |
| Beams (M) | 32 |
| Batch size | 8 |
| Optimizer | Adam (LR=5e-4, wd=0) |
| Scheduler | MultiStepLR [12,18], γ=0.1 |
| Epochs | 20 (fixed) |
| Architecture | CNN-GRU, 260K params |

### Speed Categories

| Category | Range (mph) | Range (m/s) |
|----------|-------------|-------------|
| Slow | ≤ 10 | ≤ 4.47 |
| Medium | 10–20 | 4.47–8.94 |
| Fast | > 20 | > 8.94 |

### Data

11,387 samples, 35 sequences, DeepSense 6G Scenario 23.
Split: 65% train / 15% val / 20% test (chunk-based adjusted splitting, seed=42).
