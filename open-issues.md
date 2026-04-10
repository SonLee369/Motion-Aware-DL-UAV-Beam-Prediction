## XI. Khuyến Nghị Cách Báo Cáo Tiến Độ cho Nghiên Cứu Sinh

Mỗi checkpoint nên có một biên bản ngắn gồm năm dòng: mục tiêu, file đã sửa, artifact đã sinh, điều kiện pass đã kiểm, và một vấn đề mở lớn nhất. Không nên báo cáo theo kiểu kể lại quá trình suy nghĩ dài dòng. Điều người hướng dẫn cần là: checkpoint đã qua hay chưa qua, nếu chưa qua thì đang vướng ở dữ liệu, do metric, hay do logic điều khiển.

**Một khuôn báo cáo tối giản có thể là:**

```
Checkpoint: CP3
Files touched: src/utils/dataset.py, src/control/state_builder.py
Artifacts: control_state.parquet, temperature.json
Pass/Fail: Pass
Open issue: pred_beam_v3 có cơ do lệch nhẹ ở 12 sample đầu sequence
```

Khuôn này đủ kỹ thuật để lần lại lỗi.

---