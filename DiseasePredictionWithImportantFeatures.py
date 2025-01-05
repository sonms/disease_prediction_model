import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 학습 데이터 로드 및 모델 학습 (초기화)
# train_data = pd.read_csv("train_disease_ko.csv", encoding='utf-8-sig')

# CSV 파일을 읽을 때 발생할 수 있는 인코딩 오류를 처리하기 위해 다양한 인코딩 시도
try:
    train_data = pd.read_csv("train_disease_ko.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        train_data = pd.read_csv("train_disease_ko.csv", encoding='utf-8-sig')
    except UnicodeDecodeError:
        train_data = pd.read_csv("train_disease_ko.csv", encoding='euc-kr')


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