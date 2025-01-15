import pandas as pd
import requests
from pathlib import Path

# CSV 파일 로드
# train.csv 파일의 경로
train_csv_path = "some_of_drug1.csv"

try:
    data = pd.read_csv(train_csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        data = pd.read_csv(train_csv_path, encoding='euc-kr')

# 이미지 저장 폴더 생성
image_folder = Path("pill_images")
image_folder.mkdir(parents=True, exist_ok=True)

# 이미지 다운로드
for index, row in data.iterrows():
    image_url = row['큰제품이미지']
    pill_name = row['품목명']
    image_name = f"{pill_name}_{index}.jpg"  # 파일 이름을 pill_name_{index}.jpg로 설정

    image_path = image_folder / image_name  # Path 객체로 경로 생성

    try:
        response = requests.get(image_url, timeout=15)
        with open(image_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {image_path}")
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")