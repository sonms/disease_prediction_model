import io
from tokenize import String

from starlette.responses import JSONResponse

import PillNameSearch
from GetDiseaseFeatures import get_disease_features
from FeaturesWithHighImportance import get_features_with_high_importance
from model.DiseaseRequest import DiseaseRequestData
from model.PillInfoData import PillInfoDataRequest
from model.UserInputFeatureData import UserInput
from disease_prediction.PredictionResultModel import load_model, predict_disease
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
import pandas as pd

from PillDataPrediction import PillClassifierPrediction

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


# # 이미지 예측 API
# @app.post("/pill-predict", tags=["Disease Predict"])
# async def predict_image(file: UploadFile = File(...)):
#
#     try:
#         # 이미지 열기
#         image = Image.open(file.file).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
#
#     try:
#         # 예측 수행
#         prediction_idx = classifier.predict(image)
#         prediction_label = label_decoder.get(prediction_idx, "Unknown")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
#
#     return JSONResponse(content={"prediction": prediction_label})

# POST 요청 처리
@app.post("/pill-predict", tags=["Pill Predict"])
async def pill_predict(file: UploadFile = File(...)):
    # train.csv 파일의 경로
    train_csv_path = "some_of_drug1.csv"

    try:
        data = pd.read_csv(train_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(train_csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            data = pd.read_csv(train_csv_path, encoding='euc-kr')


    # 업로드된 파일을 이미지로 변환
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 모델 로드
    model_path = "pill_classifier.pth"
    num_classes = len(data['품목명'].unique())  # 실제 클래스 수
    classifier = PillClassifierPrediction(model_path, num_classes)

    # 이미지 예측
    try:
        prediction = classifier.predict_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction_pill_name": prediction}


@app.post("/pill-info", tags=["Pill Predict"])
async def pill_info(pillName: PillInfoDataRequest):
    train_csv_path = "some_of_drug1.csv"

    try:
        drug_data = pd.read_csv(train_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            drug_data = pd.read_csv(train_csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            drug_data = pd.read_csv(train_csv_path, encoding='euc-kr')

    # 품목 정보 검색
    infoData = PillNameSearch.print_details_for_item(drug_data, pillName.pillName)

    # 데이터가 없을 경우 처리
    if infoData is None:
        raise HTTPException(status_code=404, detail="Pill information not found")

    # 데이터 반환
    return {"Business_Name": infoData[0], "Classification_Name": infoData[1], "Open_Date": infoData[2]}
