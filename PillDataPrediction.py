import pickle

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path

# 모델 로드
class PillClassifierPrediction:
    def __init__(self, model_path, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # label_decoder 로드
        with open('label_decoder.pkl', 'rb') as f:
            label_decoder = pickle.load(f)

        self.label_decoder = label_decoder

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 예측을 수행하는 함수
    def predict_image(self, image: Image.Image):
        # 이미지를 모델 입력 크기로 변환
        image = self.transform(image).unsqueeze(0)  # 배치 차원을 추가
        image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 모델 예측
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted_idx = torch.max(outputs, 1)

        # 예측된 레이블을 품목명으로 변환
        predicted_label = self.label_decoder[predicted_idx.item()]
        return predicted_label

    # def predict(self, image: Image.Image):
    #     image = self.transform(image).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(image)
    #         _, predicted = torch.max(outputs, 1)
    #     return predicted.item()
