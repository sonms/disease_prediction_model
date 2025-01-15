from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from pill_prediction.PillDataClassifier import PillClassifier
from pill_prediction.PillDataPreProcessing import PillDataset, image_folder, image_transform


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


# 데이터 분할
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 데이터셋 및 DataLoader
train_dataset = PillDataset(train_data, image_folder, image_transform)
val_dataset = PillDataset(val_data, image_folder, image_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 모델 초기화
num_classes = len(data['품목명'].unique())
model = PillClassifier(num_classes)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 검증 루프
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

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total:.2f}%")
