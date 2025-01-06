from fastapi import FastAPI, HTTPException

from GetDiseaseFeatures import get_disease_features
from FeaturesWithHighImportance import get_features_with_high_importance
from model.DiseaseRequest import DiseaseRequestData
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
    # 입력 데이터 디버깅
    print("Received POST request with data:", input_data)

    input_dict = input_data.model_dump()

    # 모든 입력값이 0인지 확인
    if all(value == 0 for value in input_dict.values()):
        raise HTTPException(
            status_code=400,
            detail="예측된 결과가 존재하지 않습니다. 다시 시도해주세요."
        )

    # 중요 특성(feature) 로딩
    model, important_features = load_model()

    # 예측 수행
    predicted_disease = predict_disease(model, important_features, input_dict)

    # 결과 반환
    return {"predicted_disease": predicted_disease}


# 브라우저에서 접근 시 안내 메시지를 반환
@app.get("/predict-disease", tags=["Disease Predict"])
async def handle_get():
    return {"detail": "This endpoint only supports POST requests. Please use POST method with appropriate data."}


@app.post("/disease-important-features", tags=["Recommend Medicine"])
async def disease_important_features(request : DiseaseRequestData):
    """
    예측된 질병을 받아 해당 질병의 중요 피처를 반환
    """
    # 질병 데이터 로드
    disease_features = get_disease_features()

    # 입력된 질병의 중요 피처 가져오기
    important_features = disease_features.get(request.disease)

    # 중요 피처가 없거나 데이터가 비어 있으면 에러 메시지 반환
    if important_features is None or len(important_features) == 0:
        return {"message": f"No important features found for disease: {request.disease}"}

    # 중요 피처 반환
    return important_features
