from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 현재 파일의 경로 (prediction.py의 위치)
current_file = Path(__file__)

# 최상위 디렉토리로 이동
parent_dir = current_file.parent.parent

# train.csv 파일의 경로
train_csv_path = parent_dir / "train_disease.csv"

# 데이터 로드
train_data = pd.read_csv(train_csv_path)

important_features = [
            "itching", "joint_pain", "stomach_pain", "vomiting", "fatigue",
            "high_fever", "dark_urine", "nausea", "loss_of_appetite",
            "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes",
            "chest_pain", "muscle_weakness", "muscle_pain", "altered_sensorium",
            "family_history", "mucoid_sputum", "lack_of_concentration"
        ]

# 중요 피처 정의
# important_features = [
#     "fatigue", "vomiting", "high_fever", "loss_of_appetite", "nausea",
#     "headache", "abdominal_pain", "yellowish_skin", "yellowing_of_eyes",
#     "chills", "skin_rash", "malaise", "chest_pain", "joint_pain", "sweating"
# ]

# 학습 데이터 준비
X_train = train_data[important_features]
y_train = train_data['prognosis']

# 결측값 처리
X_train = X_train.fillna(0)

# 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 모델 저장
joblib.dump(rf_model, "rf_model.joblib")
print("모델이 저장되었습니다: rf_model.joblib")