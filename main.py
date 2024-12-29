from fastapi import FastAPI

from GetDiseaseFeatures import get_disease_features
from FeaturesWithHighImportance import get_features_with_high_importance
from model.UserInputFeatureData import UserInput
from disease_prediction.PredictionResultModel import load_model, predict_disease

app = FastAPI(
    title="Disease Prediction API",
    description="질병 예측을 위해 사용하는 API",
    version="1.0.0"
)

@app.get("/", tags=["Features List"])
async def root():
    return {"message": "Hello World2"}


@app.get("/important-features", tags=["Features List"])
async def get_important_features(): #코루틴
    """
    질병과 중요 피처 데이터를 반환하는 API
    """
    data = get_disease_features()
    return data


@app.get("/features-with-high-importance", tags=["Features List"])
async def get_highest_feature_among_features():
    """
    피처 중에서도 높은 중요도를 가진 피처
    """
    data = get_features_with_high_importance()
    return data


@app.post("/predict-disease", tags=["Disease Predict"])
async def predict(input_data: UserInput):
    # 입력 데이터를 딕셔너리로 변환
    input_dict = input_data.model_dump()

    # 디버깅: 입력 데이터 확인
    print("Input data:", input_dict)

    # 중요 특성(feature) 로딩
    model, important_features = load_model()

    # 예측 수행
    predicted_disease = predict_disease(model, important_features, input_dict)

    # 결과 반환
    return {"predicted_disease": predicted_disease}