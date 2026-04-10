# 1 Tổng Quan
## 1.1 Mục Tiêu Nghiên Cứu

Nghiên cứu này nhằm chứng minh rằng việc bổ sung velocity và acceleration features vào
mô hình GPS-aided beam prediction có thể cải thiện đáng kể hiệu năng, đặc biệt cho UAV tốc
độ cao.

## 1.2 Các Đóng Góp Cần Chứng Minh
1. Motion Feature Engineering: Velocity và acceleration features cải thiện accuracy
2. Speed-Aware Training: Các chiến lược training giúp cân bằng performance across speeds
3. Enhanced Architecture: Multi-scale CNN và attention mechanism hiệu quả hơn
4. Theoretical Validation: Kết quả thực nghiệm phù hợp với phân tích lý thuyết

## 1.3 Kịch Bản Thực Nghiệm

Bảng 1: Tổng quan các kịch bản thực nghiệm
E1 Baseline Reproduction 
E2 Feature Ablation Study 
E3 Speed-Aware Training Comparison
E4 Architecture Ablation Study 
E5 Per-Speed Category Analysis 
E6 Feature Importance Analysis 
E7 Robustness to GPS Noise 
E8 Computational Complexity 

# 2 Chuẩn Bị Môi Trường
## 2.1 Cài Đặt Thư Viện

## 2.2 Cấu Trúc Thư Mục

## 2.3 Download Dataset

# 3 Kịch Bản E1: Baseline Reproduction
## 3.1 Mục Tiêu

Tái tạo kết quả baseline từ paper [?] để:
1. Xác minh implementation đúng
2. Thiết lập performance benchmark
3. Đảm bảo data preprocessing nhất quán

## 3.2 Expected Results

Từ paper gốc, baseline model (position-only) đạt:
• Overall Top-1 accuracy: ∼70.5%
• Slow speed: 75.7–80.3%
• Medium speed: 64.5–71.4%
• Fast speed: 55.6–65.3%
• Mean power loss: ∼0.59 dB

Acceptance Criteria
Kết quả được chấp nhận nếu:
• Overall Top-1 accuracy: 70.5% ± 2%
• Per-speed accuracy trong khoảng reported ranges
• Mean power loss: 0.59 dB ± 0.1 dB

## 3.3 Implementation Steps
### 3.3.1 Step 1: Data Loading
### 3.3.2 Step 2: Adjusted Data Splitting
### 3.3.3 Step 3: Feature Extraction (Baseline)
### 3.3.4 Step 4: Dataset Creation
### 3.3.5 Step 5: Baseline Model Implementation
### 3.3.6 Step 6: Training Loop
### 3.3.7 Step 7: Evaluation Metrics

## 3.4 Deliverables

E1 Deliverables
1. Code: Hoàn chỉnh implementation trong experiments/E1_baseline/
2. Results: File E1_results.json chứa metrics
3. Report: Document E1_report.pdf (2-3 trang) bao gồm:
	• Dataset statistics
	• Training curves (loss, accuracy)
	• Test results comparison với paper
	• Analysis nếu có discrepancy
4. Checkpoint: Saved model E1_baseline_best.pth

# 4 Kịch Bản E2: Feature Ablation Study
## 4.1 Mục Tiêu

Chứng minh rằng velocity và acceleration features cải thiện performance, và xác định feature
nào quan trọng nhất.

## 4.2 Hypothesis

1. Velocity features (đặc biệt vtangential) cải thiện accuracy đáng kể
2. Acceleration features cung cấp thêm improvement cho non-linear trajectories
3. Improvement lớn nhất ở fast-speed category

## 4.3 Experimental Design

Bảng 2: Feature Ablation Experiments

| Exp ID | Feature Set                                    | Input Dim |
| ------ | ---------------------------------------------- | --------- |
| E2.0   | Baseline (position only)                       | 5         |
| E2.1   | + v_mag                                        | 6         |
| E2.2   | + v_radial                                     | 6         |
| E2.3   | + v_tangential                                 | 6         |
| E2.4   | + v_radial + v_tangential                      | 7         |
| E2.5   | + All velocity (v_radial, v_tangential, v_mag) | 8         |
| E2.6   | + All velocity + a_mag                         | 9         |
| E2.7   | + All velocity + a_radial                      | 9         |
| E2.8   | + All velocity + a_tangential                  | 9         |
| E2.9   | + All velocity + all acceleration              | 11        |

## 4.4 Implementation
### 4.4.1 Step 1: Velocity Computation
### 4.4.2 Step 2: Acceleration Computation
### 4.4.3 Step 3: Feature Normalization
### 4.4.4 Step 4: Run Ablation Experiments
## 4.5 Analysis Tasks

E2 Analysis Tasks
1. Quantify improvements: Tính absolute và relative improvement cho mỗi feature
set
2. Statistical significance: Chạy multiple seeds (3-5 lần) và tính confidence intervals
3. Per-speed analysis: Breakdown results theo slow/medium/fast categories
4. Feature importance ranking: Rank features theo contribution
5. Visualization: Plot accuracy vs feature set, improvement heatmap

## 4.6 Expected Observations

Dựa trên phân tích lý thuyết, ta kỳ vọng:
1. vtangential có improvement lớn nhất (trực tiếp liên quan đến beam angle change rate)
2.
2. vradial có improvement nhỏ hơn (chủ yếu ảnh hưởng path loss)
3. Acceleration features cung cấp modest improvement (giúp với non-linear trajectories)
4. Combined features đạt best performance nhờ complementary information
5. Improvement tăng theo speed: slow < medium < fast

## 4.7 Deliverables

E2 Deliverables
1. Code: Implementation trong experiments/E2_feature_ablation/
2. Results:
	• Individual results: E2.X_results.json
	• Combined results: E2_ablation_all_results.json
	• LaTeX table: E2_ablation_table.tex
3. Figures:
	• Bar chart: Accuracy vs feature set
	• Heatmap: Improvement by feature and speed category
	• Line plot: Accuracy across prediction steps (v=0,1,2,3)
4. Report: E2_report.pdf (3-4 trang) bao gồm:
	• Ablation results table
	• Statistical analysis (mean, std, confidence intervals)
	• Per-speed category breakdown
	• Discussion về feature importance
	• Comparison với theoretical predictions
5. Checkpoints: Saved models cho mỗi configuration

# 5 Kịch Bản E3: Speed-Aware Training Comparison
## 5.1 Mục Tiêu

Chứng minh rằng speed-aware training strategies cải thiện performance, đặc biệt cho high-speed
UAVs.
## 5.2 Hypothesis
1. Stratified sampling giảm bias toward slow-speed samples
2. Speed-conditioned architecture học được speed-specific patterns
3. Speed-weighted loss cải thiện performance trên hard cases (fast speed)
4. Curriculum learning cung cấp better initialization
5. Combined strategies đạt best overall performance
## 5.3 Experimental Design

Bảng 3: Speed-Aware Training Experiments

| Exp ID | Strategy            | Description                                  |
| ------ | ------------------- | -------------------------------------------- |
| E3.0   | Baseline            | Standard training, no speed-aware techniques |
| E3.1   | Stratified Sampling | Balanced sampling across speed categories    |
| E3.2   | Speed-Conditioned   | Speed embedding concatenated to features     |
| E3.3   | Weighted Loss       | Higher loss weight for fast-speed samples    |
| E3.4   | Curriculum Learning | Train slow → medium → all speeds             |
| E3.5   | Combined            | All strategies together                      |


## 5.4 Implementation
### 5.4.1 Step 1: Speed Categorization
### 5.4.2 Step 2: Stratified Sampler
### 5.4.3 Step 3: Speed-Conditioned Model
### 5.4.4 Step 4: Speed-Weighted Loss
### 5.4.5 Step 5: Curriculum Learning
### 5.4.6 Step 6: Run All Speed-Aware Experiments
## 5.5 Analysis Tasks

E3 Analysis Tasks
1. Overall comparison: So sánh các strategies về overall accuracy
2. Per-speed breakdown: Phân tích improvement cho từng speed category
3. Statistical testing: T-test để xác định significant differences
4. Training dynamics: Plot learning curves cho mỗi strategy
5. Speed distribution: Visualize effective speed distribution trong training
6. Ablation of combined: Xác định contribution của từng component trong E3.5

## 5.6 Expected Observations
1. Stratified sampling cải thiện fast-speed performance nhưng có thể giảm nhẹ slow-speed
2. Speed-conditioned model học được adaptive behaviors
3. Weighted loss có largest impact trên fast-speed category
4. Curriculum learning cung cấp stable training nhưng modest final improvement
5. Combined strategy đạt best balance across all speeds
6. Fast-speed improvement lớn hơn slow-speed improvement

## 5.7 Deliverables

E3 Deliverables
1. Code: Implementation trong experiments/E3_speed_aware/
2. Results:
	• Individual results: E3.X_results.json
	• Combined results: E3_speed_aware_all_results.json
	• LaTeX tables: E3_overall_table.tex, E3_per_speed_table.tex
3. Figures:
	• Bar chart: Accuracy by strategy and speed category
	• Learning curves: Training/validation loss for each strategy
	• Heatmap: Improvement matrix (strategy × speed category)
	• Box plot: Accuracy distribution across speeds
4. Report: E3_report.pdf (4-5 trang) bao gồm:
	• Strategy comparison tables
	• Per-speed category analysis
	• Statistical significance tests
	• Training dynamics analysis
	• Discussion về best strategy và trade-offs
5. Checkpoints: Saved models cho mỗi strategy

# 6 Kịch Bản E4: Architecture Ablation Study
## 6.1 Mục Tiêu
Chứng minh rằng multi-scale CNN và motion attention mechanism cải thiện model capacity.
## 6.2 Hypothesis
1. Multi-scale CNN captures temporal patterns better than single-scale
2. Motion attention adaptively weights motion features
3. Combined architecture achieves best performance
4. Architecture improvements complement feature improvements
## 6.3 Experimental Design

Bảng 4: Architecture Ablation Experiments

| Exp ID | Architecture       | Description                                                |
| ------ | ------------------ | ---------------------------------------------------------- |
| E4.0   | Baseline           | Single-scale CNN (kernel=3) + GRU                          |
| E4.1   | Multi-Scale CNN    | Parallel CNNs (kernels=3,5,7) + GRU                        |
| E4.2   | Single + Attention | Single-scale CNN + Motion Attention +GRU                   |
| E4.3   | Multi + Attention  | Multi + Attention Multi-scale CNN + Motion Attention + GRU |

## 6.4 Implementation
### 6.4.1 Step 1: Multi-Scale CNN Module
### 6.4.2 Step 2: Motion Attention Module
### 6.4.3 Step 3: Enhanced Model with Full Architecture
### 6.4.4 Step 4: Run Architecture Ablation
## 6.5 Analysis Tasks

E4 Analysis Tasks
1. Performance vs complexity trade-off: Plot accuracy vs model size
2. Attention visualization: Visualize attention weights for different speeds
3. Feature map analysis: Visualize multi-scale CNN outputs
4. Inference time measurement: Measure actual inference time on GPU
5. Ablation contribution: Quantify individual contribution of each component
## 6.6 Expected Observations
1. Multi-scale CNN improves accuracy by 1-2% with modest parameter increase
2. Attention mechanism provides adaptive feature weighting
3. Combined architecture achieves best performance
4. Attention weights higher for motion features at high speeds
5. Inference time increase acceptable (< 30%)
## 6.7 Deliverables

E4 Deliverables
1. Code: Implementation trong experiments/E4_architecture/
2. Results:
	• Individual results: E4.X_results.json
	• Combined results: E4_architecture_all_results.json
	• LaTeX table: E4_architecture_table.tex
3. Figures:
	• Scatter plot: Accuracy vs model size
	• Heatmap: Attention weights visualization
	• Feature maps: Multi-scale CNN outputs
	• Bar chart: Inference time comparison
4. Report: E4_report.pdf (3-4 trang)
5. Checkpoints: Saved models cho mỗi architecture

# 7 Kịch Bản E5-E8: Các Thực Nghiệm Bổ Sung
## 7.1 E5: Per-Speed Category Analysis
### 7.1.1 Mục tiêu
Phân tích chi tiết performance breakdown theo speed categories.
### 7.1.2 Tasks
1. Evaluate best model (E3.5 + E4.3) trên từng speed category
2. Compute metrics: Top-1/3/5 accuracy, mean power loss
3. Statistical analysis: Confidence intervals, significance tests
4. Visualization: Performance curves across speeds
### 7.1.3 Deliverables
	• E5_per_speed_results.json
	• E5_per_speed_table.tex
	• Figures: Bar charts, box plots
	• Report: 2-3 trang
## 7.2 E6: Feature Importance Analysis
### 7.2.1 Mục tiêu
Quantify importance của từng feature using permutation importance.
### 7.2.2 Implementation
### 7.2.3 Deliverables
	• E6_feature_importance.json
	• Bar chart: Feature importance ranking
	• Heatmap: Importance by feature and speed category
	• Report: 2 trang
## 7.3 E7: Robustness to GPS Noise
### 7.3.1 Mục tiêu
Test model robustness under GPS measurement noise.
### 7.3.2 Implementation

### 7.3.3 Expected Observations
1. Enhanced model more robust due to Savitzky-Golay filtering
2. Baseline degrades faster with increasing noise
3. At typical GPS noise (5-10m), enhanced model maintains advantage
4. Velocity/acceleration features less sensitive to noise after smoothing
### 7.3.4 Deliverables

E7 Deliverables
1. Code: experiments/E7_robustness/
2. Results: E7_robustness_results.json
3. Table: E7_robustness_table.tex
4. Figures:
	• Line plot: Accuracy vs noise level
	• Bar chart: Accuracy drop comparison
5. Report: E7_report.pdf (2 trang)
## 7.4 E8: Computational Complexity Analysis
### 7.4.1 Mục tiêu

Đo lường computational cost và inference time của các models.
### 7.4.2 Implementation
### 7.4.3 Expected Results

Bảng 5: Expected Complexity Metrics (Approximate)

| Model         | Params | Size (MB) | Inference (ms) | Overhead |
| ------------- | ------ | --------- | -------------- | -------- |
| Baseline      | 308K   | 0.99      | 6-8            |      -   |
| + Multi-scale | 350K   | 1.12      | 7-9            | +15%     |
| + Attention   | 340K   | 1.09      | 8-10           | +25%     |
| + Both        | 1.18   | 1.18      | 9-11           | +30%     |

### 7.4.4 Deliverables

E8 Deliverables
1. Code: experiments/E8_complexity/
2. Results: E8_complexity_results.json
3. Table: E8_complexity_table.tex
4. Figures: Bar charts comparing metrics
5. Report: E8_report.pdf (2 trang)

# 8 Tổng Hợp và Báo Cáo Cuối Cùng
## 8.1 Checklist Hoàn Thành

Bảng 6: Experiment Completion Checklist

| ID  | Experiment               | Status | Verified |
| --- | ------------------------ | ------ | -------- |
| E1  | Baseline Reproduction    |        |          |
| E2  | Feature Ablation         |        |          |
| E3  | Speed-Aware Training     |        |          |
| E4  | Architecture Ablation    |        |          |
| E5  | Per-Speed Analysis       |        |          |
| E6  | Feature Importance       |        |          |
| E7  | GPS Noise Robustness     |        |          |
| E8  | Computational Complexity |        |          |


## 8.2 Final Report Structure

## 8.3 Key Tables for Paper
### 8.3.1 Table 1: Overall Performance Comparison
### 8.3.2 Table 2: Per-Speed Category Results
## 8.4 Key Figures for Paper
1. Figure 1: System architecture diagram
2. Figure 2: Feature extraction pipeline flowchart
3. Figure 3: Bar chart - Accuracy by feature set (E2)
4. Figure 4: Heatmap - Improvement by speed category and strategy (E3)
5. Figure 5: Line plot - Accuracy vs prediction step (v=0,1,2,3)
6. Figure 6: Bar chart - Feature importance ranking (E6)
7. Figure 7: Line plot - Robustness to GPS noise (E7)
8. Figure 8: Scatter plot - Accuracy vs model size trade-off (E4, E8)
## 8.5 Statistical Validation

## 8.6 Reproducibility Guidelines
Đảm Bảo Reproducibility
1. Random seeds: Set seeds cho numpy, torch, random
```python
def set_random_seed ( seed =42) :
np . random . seed ( seed )
torch . manual_seed ( seed )
torch . cuda . manual_seed_all ( seed )
random . seed ( seed )
torch . backends . cudnn . deterministic = True
torch . backends . cudnn . benchmark = False
```
2. Environment: Save environment info
```
pip freeze > requirements . txt
python -- version > python_version . txt
nvidia - smi > gpu_info . txt
```
3. Hyperparameters: Document all hyperparameters trong config file
4. Data splits: Save train/val/test indices
5. Checkpoints: Save model checkpoints at key epochs
6. Logs: Save training logs với timestamps

## 8.7 Timeline và Milestones

Bảng 7: Detailed Timeline

Week Tasks Deliverables Status
1 Setup + E1 (Baseline) Code, results, report □
2 E2 (Feature Ablation) Code, results, report □
3 E3 (Speed-Aware) Code, results, report □
4 E4 (Architecture) Code, results, report □
5 E5-E8 (Additional) Code, results, reports □
6 Statistical validation Multiple runs, tests □
7 Paper writing Draft paper □
8 Revision Final paper □

# 9 Appendix: Useful Code Snippets
## 9.1 Visualization Functions
## 9.2 Logging and Monitoring
# 10 Conclusion

Tài liệu này cung cấp hướng dẫn chi tiết để thực hiện đầy đủ các thực nghiệm chứng minh đóng
góp của nghiên cứu. Các điểm chính:

1. 8 kịch bản thực nghiệm (E1-E8) được thiết kế systematic
2. Implementation code đầy đủ và ready-to-run
3. Expected results dựa trên phân tích lý thuyết
4. Statistical validation để đảm bảo significance
5. Reproducibility guidelines để đảm bảo kết quả nhất quán
6. Visualization và reporting tools

Lưu Ý Quan Trọng
	• Chạy mỗi experiment với multiple seeds (3-5 lần)
	• Document tất cả hyperparameters
	• Save checkpoints và logs
	• Verify results statistically significant
	• Compare với baseline paper để ensure reproduction

