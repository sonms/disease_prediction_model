from pathlib import Path

import pandas as pd
import joblib


def load_model():
    """
    학습된 모델과 중요 피처 리스트를 로드하는 함수
    """
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

    # 현재 파일의 경로
    current_file = Path(__file__).resolve()

    # 최상위 디렉토리로 이동
    parent_dir = current_file.parent.parent

    # 모델 파일 경로 (모델 파일이 `a` 디렉토리에 있다고 가정)
    model_path = parent_dir / "disease_prediction" / "rf_model.joblib"

    # 학습된 모델 로드
    model = joblib.load(model_path)
    return model, important_features


def predict_disease(model, important_features, input_data):
    """
    예측 수행 함수
    :param model: 학습된 모델
    :param important_features: 중요 피처 리스트
    :param input_data: 사용자 입력 데이터
    :return: 예측된 질병 이름
    """
    # 입력 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([input_data], columns=important_features)

    # 예측 수행
    prediction = model.predict(input_df)[0]
    return prediction