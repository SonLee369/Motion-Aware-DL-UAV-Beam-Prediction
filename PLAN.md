# UAV Beam Prediction — Experiment Plan & Progress Tracker

> Last updated: 2026-04-11

---

## Overview

**Goal**: Prove that motion-aware features (velocity/acceleration) improve GPS-aided beam prediction for UAVs.  
**Dataset**: DeepSense 6G Scenario 23 — 11,387 samples, 35 sequences  
**Key claim to validate**: v_radial is the most impactful feature (statistically significant, p < 0.05)

---

## Dependency Chain

```
Raw Data (scenario23/)
    └── M1: E1 Baseline  →  data/processed/E1_baseline/*.pkl
            ├── M2: E2 Feature Ablation  →  data/processed/E2_feature_ablation/*.pkl
            │       └── M3: E2 Finalist (10 seeds, significance tests)
            ├── M4: E3 Speed-Aware Training
            ├── M5: E4 Architecture Ablation  (uses E2 pkl, fallback E1 pkl)
            │       └── M6: E4 Bridge (6D_vradial × 3 archs)
            └── M7: E5 Loss Functions
                    └── M8: E8 Final Evaluation  ← best config from all above
```

---

## Module Tracker

### M1 — E1: Baseline Reproduction
- **Status**: `[ ] Not started`
- **Script**: `experiments/E1_baseline/run_baseline.py`
- **What it does**: Train position-only (5D) CNN-GRU baseline; generate processed pkl files for downstream experiments
- **Expected runtime**: ~5–10 min
- **Expected results**:
  - Overall Top-1: ~73–74%
  - Power loss: ~0.17–0.18 dB
- **Acceptance criteria**:
  - Top-1 within ±2% of 73.74%
  - pkl files created at `data/processed/E1_baseline/`
- **Outputs**:
  - `data/processed/E1_baseline/{train,val,test}.pkl`
  - `checkpoints/E1_baseline/best_model.pth`
  - `results/E1_baseline/E1_report.pdf`
  - `logs/E1_baseline.log`
- **Notes**: —

---

### M2 — E2: Feature Ablation Study
- **Status**: `[ ] Blocked (waiting for M1)`
- **Script**: `experiments/E2_feature_ablation/run_feature_ablation.py`
- **What it does**: Train 10 feature configurations (E2.0–E2.9) × 5 seeds; identify best feature set
- **Expected runtime**: ~60–90 min
- **Feature configs**:

  | ID   | Features                          | Dim |
  |------|-----------------------------------|-----|
  | E2.0 | Position only (baseline)         | 5   |
  | E2.1 | + v_mag                           | 6   |
  | E2.2 | + v_radial                        | 6   |
  | E2.3 | + v_tangential                    | 6   |
  | E2.4 | + v_radial + v_tangential         | 7   |
  | E2.5 | + All velocity                    | 8   |
  | E2.6 | + All velocity + a_mag            | 9   |
  | E2.7 | + All velocity + a_radial         | 9   |
  | E2.8 | + All velocity + a_tangential     | 9   |
  | E2.9 | + All velocity + all acceleration | 11  |

- **Expected results**: E2.2 (v_radial) should be top performer
- **Outputs**:
  - `data/processed/E2_feature_ablation/{train,val,test}_with_velocity.pkl`
  - `results/E2_feature_ablation/`
  - `logs/E2_feature_ablation.log`
- **Notes**: —

---

### M3 — E2 Finalist: Statistical Significance
- **Status**: `[ ] Blocked (waiting for M2)`
- **Script**: `experiments/E2_feature_ablation/run_e2_finalist.py`
- **What it does**: Re-run top 4 feature configs × 10 seeds (seeds 42, 77–85); compute paired t-test and bootstrap CI
- **Expected runtime**: ~40–60 min
- **Expected results**:

  | Comparison     | Delta   | p-value | Significant? |
  |----------------|---------|---------|--------------|
  | E2.2 vs E2.0   | +0.86%  | 0.032   | Yes          |
  | E2.13 vs E2.0  | +0.80%  | 0.132   | No           |
  | E2.14 vs E2.0  | +0.20%  | 0.686   | No           |

- **Outputs**:
  - `results/E2_finalist/`
  - `logs/E2_finalist.log`
- **Notes**: —

---

### M4 — E3: Speed-Aware Training
- **Status**: `[ ] Blocked (waiting for M1)`
- **Script**: `experiments/E3_speed_aware/run_speed_aware.py`
- **What it does**: Compare 5 speed-aware training strategies vs standard CrossEntropy
- **Expected runtime**: ~30–50 min
- **Strategies**:

  | ID   | Strategy            |
  |------|---------------------|
  | E3.0 | Standard CE (baseline) |
  | E3.1 | Stratified Sampling |
  | E3.2 | Speed-Conditioned   |
  | E3.3 | Weighted Loss       |
  | E3.4 | Curriculum Learning |

- **Expected results**: Standard CE wins; stratified −2.30%, curriculum −6.23%
- **Outputs**:
  - `results/E3_speed_aware/`
  - `logs/E3_speed_aware.log`
- **Notes**: —

---

### M5 — E4: Architecture Ablation
- **Status**: `[ ] Blocked (waiting for M2)`
- **Script**: `experiments/E4_architecture/run_architecture_ablation.py`
- **What it does**: Compare 8 architectures × 4 feature sets × 5 seeds
- **Expected runtime**: ~2–3 hr
- **Architectures**: Baseline CNN-GRU, Multi-Scale CNN, 2-Layer GRU, Transformer, + variants
- **Expected results**: Architecture differences NOT significant (ANOVA p ≈ 0.720)
- **Outputs**:
  - `results/E4_architecture/`
  - `logs/E4_architecture.log`
- **Notes**: Previous E4 results used pre-bugfix code — watch for differences

---

### M6 — E4 Bridge: Feature vs Architecture
- **Status**: `[ ] Blocked (waiting for M2)`
- **Script**: `experiments/E4_architecture/run_e4_bridge.py`
- **What it does**: 6D_vradial × 3 architectures × 5 seeds to confirm feature > architecture
- **Expected runtime**: ~30–40 min
- **Expected results**:

  | Architecture    | Top-1 (%) | PL (dB) |
  |-----------------|-----------|---------|
  | Baseline CNN-GRU | 74.70    | 0.167   |
  | 2-Layer GRU     | 74.91     | 0.170   |
  | Transformer     | 75.09     | 0.162   |

- **Outputs**:
  - `results/E4_bridge/`
  - `logs/E4_bridge.log`
- **Notes**: —

---

### M7 — E5: Loss Function Comparison
- **Status**: `[ ] Blocked (waiting for M1)`
- **Script**: `experiments/E5_loss_functions/run_loss_comparison.py`
- **What it does**: Compare CrossEntropy vs Label Smoothing (ε=0.2) vs Focal Loss (γ=2)
- **Expected runtime**: ~20–30 min
- **Expected results**:

  | Loss              | Top-1 (%) | vs CE |
  |-------------------|-----------|-------|
  | CrossEntropy      | ~69.54    | —     |
  | Label Smoothing   | ~69.92    | +0.37 |
  | Focal Loss        | ~69.36    | −0.18 |

- **Outputs**:
  - `results/E5_loss_functions/`
  - `logs/E5_loss_comparison.log`
- **Notes**: —

---

### M8 — E8: Final Evaluation
- **Status**: `[ ] Blocked (waiting for M3, M6)`
- **Script**: `experiments/E8_final_evaluation/` (TBD)
- **What it does**: End-to-end evaluation of best config (6D_vradial + Transformer + Label Smoothing); generate paper-ready tables and figures
- **Expected runtime**: ~20–30 min
- **Outputs**:
  - Final comparison table (all models)
  - Per-speed breakdown
  - Power loss by horizon
  - `results/E8_final_evaluation/`
- **Notes**: —

---

## Deferred / Blocked

| Module | Reason |
|--------|--------|
| E6: Feature Importance (permutation) | Not started — add after M3 |
| E7: GPS Noise Robustness | **BLOCKED** — requires Scenario 24/25/26 data (not downloaded) |

---

## Key Metrics to Track

| Model | Top-1 (%) | Power Loss (dB) | Params | Significant? |
|-------|-----------|-----------------|--------|--------------|
| E1: Baseline (5D) | — | — | 260K | — |
| E2.2: + v_radial (6D) | — | — | 260K | — |
| E3 best strategy | — | — | — | — |
| E4 Bridge: Transformer (6D) | — | — | 558K | — |
| E5 best loss | — | — | — | — |
| **M8: Final best** | — | — | — | — |

*(Fill in as experiments complete)*

---

## Training Configuration Reference

| Parameter | Value |
|-----------|-------|
| Window size (W) | 8 |
| Predictions (V) | 4 |
| Beams (M) | 32 |
| Batch size | 8 |
| Optimizer | Adam (LR=5e-4, wd=0) |
| Scheduler | MultiStepLR [12, 18], γ=0.1 |
| Epochs | 20 (fixed) |
| Seeds (standard) | 5 (42, 77, 78, 79, 80) |
| Seeds (finalist) | 10 (42, 77–85) |

---

## How to Use This File

- Update `Status` field after each module completes: `[ ]` → `[x]`
- Fill in actual results in the **Key Metrics** table
- Add notes about any anomalies or deviations from expected values
- Commit this file after each module so progress is tracked in git
