import io

import pandas as pd
from torchvision.transforms import transforms

from GetDiseaseFeatures import get_disease_features
from FeaturesWithHighImportance import get_features_with_high_importance
from model.DiseaseRequest import DiseaseRequestData
from model.UserInputFeatureData import UserInput
from disease_prediction.PredictionResultModel import load_model, predict_disease
from fastapi import FastAPI, HTTPException, File, UploadFile, Path
from torchvision import models
import torch
from PIL import Image
from io import BytesIO

from pill_prediction.PillDataPreProcessing import image_transform
from pill_prediction.PillDataTrain import num_classes

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












# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
num_classes = len(pd.read_csv("some_of_drug1.csv")['품목명'].unique())
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model_path = Path("pill_prediction/pill_classifier.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 클래스 이름 로드
data = pd.read_csv("some_of_drug1.csv")
class_to_pill_name = {idx: name for idx, name in enumerate(data['품목명'].unique())}

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # 이미지 전처리
    image = preprocess(image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # 예측 결과 (알약 이름)
    predicted_pill_name = class_to_pill_name[predicted.item()]

    return {"predicted_pill_name": predicted_pill_name}