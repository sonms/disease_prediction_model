import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 학습 데이터 로드 및 모델 학습 (초기화)
train_data = pd.read_csv("train_disease.csv")
important_features = [
    "itching", "joint_pain", "stomach_pain", "vomiting", "fatigue",
    "high_fever", "dark_urine", "nausea", "loss_of_appetite",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes",
    "chest_pain", "muscle_weakness", "muscle_pain", "altered_sensorium",
    "family_history", "mucoid_sputum", "lack_of_concentration"
]
# important_features = [
#     "fatigue", "vomiting", "high_fever", "loss_of_appetite", "nausea",
#     "headache", "abdominal_pain", "yellowish_skin", "yellowing_of_eyes",
#     "chills", "skin_rash", "malaise", "chest_pain", "joint_pain", "sweating"
# ]

X_train = train_data[important_features]
y_train = train_data['prognosis']

# 결측값 처리
X_train = X_train.fillna(0)

# 모델 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)