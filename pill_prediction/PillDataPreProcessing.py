import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

from torchvision.models import ResNet50_Weights

# 이미지 폴더와 CSV 경로
image_folder = Path("pill_images")
# 현재 파일의 경로 (prediction.py의 위치)
current_file = Path(__file__)

# 최상위 디렉토리로 이동
parent_dir = current_file.parent.parent

# train.csv 파일의 경로
train_csv_path = parent_dir / "some_of_drug1.csv"

# CSV 데이터 로드
try:
    data = pd.read_csv(train_csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        data = pd.read_csv(train_csv_path, encoding='euc-kr')

# 데이터셋 정의
class PillDataset(Dataset):
    def __init__(self, data, image_folder, transform):
        self.data = data
        self.image_folder = Path(image_folder)
        self.transform = transform
        self.label_encoder = {name: idx for idx, name in enumerate(data['품목명'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pill_name = row['품목명']
        image_name = f"{pill_name}_{row.name}.jpg"  # 이미지 이름은 index와 일치
        image_path = self.image_folder / image_name

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros((3, 224, 224))  # 기본 이미지

        label = self.label_encoder[pill_name]
        return image, torch.tensor(label, dtype=torch.long)

# 전처리 및 데이터 증강
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 평균값과 표준편차
])

# 데이터셋 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 데이터셋 생성
train_dataset = PillDataset(train_data, image_folder, image_transform)
val_dataset = PillDataset(val_data, image_folder, image_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 정의 (ResNet50을 전이 학습으로 사용)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT) #`weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights warnings.warn(msg)

# 마지막 분류층 수정
num_classes = len(data['품목명'].unique())  # 클래스 수
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 모델을 GPU로 전송
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduling
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 학습 루프
epochs = 15
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0 # 누적 손실
    correct = 0 # 맞춘 예측의 수
    total = 0 # 전체 예측 수

    for images, labels in train_loader: #학습 train_loader=데이터로드 배치단위
        images, labels = images.to(device), labels.to(device) #데이터 gpu또는 cpu에서 학습
        optimizer.zero_grad() #계산된 기울기 초기화

        # 모델 출력
        outputs = model(images) #예측진행
        loss = criterion(outputs, labels) #예측결과와 실제 레이블간 손실계산
        loss.backward() #역전파로 기울기 계산
        optimizer.step() #가중치 업데이트

        total_loss += loss.item() #손실 값들을 누적하여 손실 구함

        # 예측 정확도 계산
        _, predicted = torch.max(outputs, 1) #예측결과에서 가장 큰 값
        total += labels.size(0)
        correct += (predicted == labels).sum().item() #맞춘예측수 누적

    train_accuracy = 100 * correct / total # 훈련정확도
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # 검증
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Validation Loss 개선 시 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "pill_classifier.pth")

    # Learning Rate Scheduler Step
    scheduler.step()

# 모델 저장 완료 후 학습 종료
print("Training finished, best model saved.")