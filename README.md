**Child mind Institute**

**1. Giới thiệu**

Vì không có nhiều kinh nghiệm trong lĩnh vực tâm lý, đặc biệt là tâm lý trẻ em, nên việc hiểu và phân tích dữ liệu gặp rất nhiều khó khăn.

**Một số đặc điểm cần lưu ý:**

\_ Dữ liệu được cung cấp rất mất cân bằng và chứa nhiều giá trị bị khuyết.

\_ Dữ liệu test công khai có thể khác với dữ liệu test private, do đó cần đảm bảo mô hình hoạt động ổn định trên cả hai tập dữ liệu.

\_ Nên áp dụng các phương pháp xử lý hoặc tránh các vấn đề liên quan đến dữ liệu mất cân bằng.

**2. Số điểm đạt được trong cuộc thi**

Hai notebook dưới đây có một số khác biệt:

1. Public: 0.498 → Private: 0.407
2. Public: 0.434 → Private: 0.441

**3. Cải tiến các notebook có sẵn**

### Notebook 1: CMI| Tuning | [[Ensemble of solutions cd0c0c)]](https://www.kaggle.com/code/hovuan/cmi-tuning-ensemble-of-solutions-cd0c0c)
\ | Public: 0.498 → Private: 0.407

#### Các cải tiến đạt được:
- Điểm Public tăng nhẹ (0.497–0.498), điểm Private giữ nguyên (0.407).
- Hướng tiếp cận: Xây dựng mô hình phân loại phụ để dự đoán nhãn 3 nhằm giải quyết vấn đề dữ liệu mất cân bằng. Mô hình này sử dụng phương pháp tổ hợp (VotingClassifier) với ba mô hình cơ sở:
  
  1. **XGBoost Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `use_label_encoder=False`: Tắt bộ mã hóa nhãn do nhãn là các số.
     - `eval_metric='logloss'`: Sử dụng hàm mất mát log loss.
     - `scale_pos_weight=116.33`: Điều chỉnh trọng số cho lớp dương.

  2. **CatBoost Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `verbose=0`: Tắt thông báo trong quá trình huấn luyện.
     - `class_weights={0: 1, 1: 117.33}`: Điều chỉnh trọng số để xử lý mất cân bằng.

  3. **LightGBM Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `is_unbalance=True`: Xử lý mất cân bằng trong quá trình huấn luyện.

#### Phương pháp:
- Sử dụng phương pháp tổ hợp soft voting.
- Tối ưu hóa ngưỡng dựa trên precision score trên tập validation.
- Kết hợp dự đoán của mô hình hồi quy và mô hình nhãn 3.

#### Khó khăn:
- Tìm ngưỡng tối ưu cho mô hình nhãn 3.
- Chỉ dựa vào precision score mà không xem xét accuracy.
- Overfitting trên tập dữ liệu public.

### Notebook 2: CMIL_PCIAT-PCIAT

#### Các cải tiến đạt được:
- Điểm Public giảm xuống 0.434, điểm Private tăng lên 0.441.
- Notebook này không sử dụng trực tiếp cột "sii" mà chuyển đổi từ cột "pca_columns."

#### Phương pháp:
- Điều chỉnh `threshold_Rounder` thành 6 ngưỡng tương ứng với các nhãn 0-5.
- Cập nhật giá trị khởi tạo cho các ngưỡng trong `KappaOptimizer`.
- Triển khai hàm `classify_score` để chuyển tổng điểm thành nhãn "sii".

#### Khó khăn:
- Điểm Public giảm đáng kể, gây khó khăn trong việc đánh giá hiệu quả mô hình.
- Thay đổi ngưỡng giúp giảm overfitting trên tập public và cải thiện điểm private.

## Sửa lỗi majority_vote

Cả hai notebook sử dụng kết hợp kết quả từ ba submission:
- **sii1**: LightGBM, XGBoost, CatBoost, TabNet.
- **sii2**: LightGBM, XGBoost, CatBoost.
- **sii3**: LGBMRegressor, XGBRegressor, CatBoostRegressor, RandomForestRegressor, GradientBoostingRegressor.

Hàm majority vote được sửa để chọn giá trị trung vị trong trường hợp tranh chấp.

```python
def majority_vote(row):
    if row['sii_1'] != row['sii_2'] and row['sii_1'] != row['sii_3']:
        return int(row.median()) 
    else:
        return row.mode()[0]
```

## Kết luận

Các điểm quan trọng:
- Giải quyết dữ liệu mất cân bằng bằng cách tránh sử dụng trực tiếp "sii" cải thiện hiệu suất.
- Giảm overfitting trên tập public giúp cải thiện sự ổn định trên tập private.

