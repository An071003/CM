**Child mind Institute**

Tên: Hồ Vũ An.  

Tài khoản kaggle: Ho Vu AN.  

Cuộc thi: [Child mind Institute](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).  

Thứ hạng cuộc thi: 🥉 289.  


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

**Notebook 1: CMI| Tuning | [[Ensemble of solutions cd0c0c]](https://www.kaggle.com/code/hovuan/cmi-tuning-ensemble-of-solutions-cd0c0c)
\ | Public: 0.498 → Private: 0.407**

**Cải tiến (Public score tăng lên  0.497- 0.498, Private score giữ nguyên 0.407)**
- Điểm Public tăng nhẹ (0.497–0.498), điểm Private giữ nguyên (0.407).
- Hướng tiếp cận ở notebook này là sẽ xây dựng thêm mô hình phân loại phụ để dự đoán cho nhãn 3 để giải quyết vấn đề imbalanced data do các hàm phân loại có những tính năng giải quyết các tính năng để xử lý trên bộ dữ liệu imbalanced.

Tạo đầu vào của mô hình:  
  Chuyển các dòng dữ liệu có ‘sii’ = 3 thành 1 và những số còn lại thành 0.

Mô hình này sử dụng phương pháp tổ hợp (VotingClassifier) với ba mô hình cơ sở:

  1. **XGBoost Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `use_label_encoder=False`: Tắt bộ mã hóa nhãn do nhãn là các số.
     - `eval_metric='logloss'`: Sử dụng hàm mất mát log loss.
     - `scale_pos_weight=116.33`: Điều chỉnh trọng số cho lớp dương. Được tính bằng $$\frac{\text{Số lượng nhãn cao hơn}}{\text{Số lượng nhãn thấp hơn}}$$

  2. **CatBoost Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `verbose=0`: Tắt thông báo trong quá trình huấn luyện.
     - `class_weights={0: 1, 1: 117.33}`: Điều chỉnh trọng số để xử lý mất cân bằng.

  3. **LightGBM Classifier:**
     - `random_state=SEED`: Đảm bảo tính tái lập.
     - `is_unbalance=True`: Xử lý mất cân bằng trong quá trình huấn luyện.

Sử dụng phương pháp tổ hợp soft voting.

Tối ưu hóa ngưỡng dựa trên precision score trên tập validation.

Kết hợp dự đoán của mô hình hồi quy và mô hình nhãn 3.

**Khó khăn:**
-	Khó khăn lớn nhất trong thực nghiệm này là phần chọn ngưỡng phân loại cho mô hình phân loại nhãn 3 là bao nhiêu là phù hợp để mô hình vừa có thể đưa ra được kết quả nhãn 3 và vừa có điểm precision cao mà còn.
-	Việc chỉ dựa vào thông số precision_score và không kiểm tra tới accuracy_score có thể làm cho việc tìm ngưỡng chưa thật sự tối ưu.
-	Ngoài ra việc điểm public khá cao nhưng điểm trên private được cao lắm chứng tỏ mô hình đang bị overfitting trên tập public. 

**Notebook 2: [[CMIL_PCIAT-PCIAT)]]([https://www.kaggle.com/code/hovuan/cmi-tuning-ensemble-of-solutions-cd0c0c](https://www.kaggle.com/code/hovuan/cmil-pciat-pciat))
\| Public: 0.434  → Private: 0.441**

**Cải tiến (Public score giảm còn 0.434, private tăng lên 0.441):**
Ở Notebook 2, nhóm chủ yếu không sử dụng cột “sii” có trong tập dữ liệu được cung cấp mà chuyển đổi giá trị từ các cột “PCIAT-PCIAT” sang giá trị cho cột “sii”. Cột “PCIAT-PCIAT” là các cột được tạo thêm nhằm lưu trữ điểm các câu hỏi từ “PCIAT-PCIAT_1” đến “PCIAT-PCIAT-20” sau đó tính tổng lại trong cột “PCIAT-PCIAT-Total” rồi chuyển đổi sang thành cột “sii”. Lý do cho hướng đi trên là do dữ liệu chuyển đổi từ các cột điểm PCIAT-PCIAT ít bị mất cân bằng hơn cột “sii” được cung cấp.

- Điều chỉnh `threshold_Rounder` thành 6 ngưỡng tương ứng với các nhãn từ 0-5 trong cột “PCIAT-PCIAT.
```python
def threshold_Rounder(oof_non_rounded, thresholds):
         return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2,
                                      np.where(oof_non_rounded < thresholds[3], 3,
                                               np.where(oof_non_rounded < thresholds[4], 4, 5)))))
```
- Cập nhật giá trị khởi tạo cho các ngưỡng trong `KappaOptimizer`.
```python
KappaOPtimizer = minimize(evaluate_predictions,
                                  x0=[0.5, 1, 1.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                                  method='Nelder-Mead')
```
- Triển khai hàm `classify_score` để chuyển tổng điểm thành nhãn "sii".
```python
def classify_score(score):
    if 0 <= score <= 30:
        return 0
    elif 31 <= score <= 49:
        return 1
    elif 50 <= score <= 79:
        return 2
    elif 80 <= score <= 100:
        return 3
```
**Khó khăn:**
-	Việc tiến hành thực nghiệm này gặp khó khăn trong giai đoạn cuộc thi vì kết quả public đã giảm xuống còn 0.434 nên là không biết mô hình có thật sự hoạt động tốt cho cả hai tập dữ liệu hay không.
-	Việc sử dụng ngưỡng [0.5, 1, 1.5, 1.5, 2.5] thay vì [0.5, 1.5, 2.5, 3.5, 4.5] trong KappaOPtimizer sẽ làm cho điểm ở tập public từ 0.470 xuống còn 0.434 nhưng điểm ở private thì tăng từ 0.436 thành 0.442. Điều này có nghĩa là việc sử dụng ngưỡng trên đã giúp giảm overfitting ở tập public và tăng hiệu quả mô hình ở tập private. 

**4. Sửa lỗi majority_vote**

Notebook 1 và 2 sử dụng kết hợp 3 kết quả submission cho 3 trường hợp như sau:
+	sii1 sử dụng Light, XGB_Model, CatBoost_Model, TabNet_Model
+	sii2 sử dụng Light, XGB_Model, CatBoost_Model
+	sii3 sử dụng LGBMRegressor, XGBRegressor, CatBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
Sau đó sẽ sử dụng lấy kết quả dựa trên mode qua hàm majority_vote.

Tuy nhiên ở notebook gốc thì hàm majority_vote đang gặp vấn đề khi sử dụng mode là khi cả 3 submission có giá trị khác nhau thì sẽ luôn lấy theo giá trị đầu. Tuy nhiên trong thực tế khi có sự tranh chấp về các giá trị thì ta sẽ thường lấy giá trị ở giữa điều đó làm có hàm majority_vote đang chưa hoạt động đúng.

```python
def majority_vote(row):
    if row['sii_1'] != row['sii_2'] and row['sii_1'] != row['sii_3']:
        return int(row.median()) 
    else:
        return row.mode()[0]
```

**5. Kết luận**

Việc giải quyết tình trạng imbalance data cho nhãn 3 bằng việc tránh sử dụng trực tiếp sii làm đầu ra dữ liệu và việc giảm overfitting trên tập public là các bước quan trọng nhất trong cuộc thi này.

