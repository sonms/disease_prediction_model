import pandas as pd

# CSV 파일 경로 설정
csv_file_path = "train_disease.csv"

# 1. 데이터 로드
df = pd.read_csv(csv_file_path)

# 2. 'prognosis' 열 제외하고 피처 사용 빈도 계산
feature_usage = df.drop(columns=["prognosis"]).sum()

# 3. 사용도가 높은 피처 정렬
feature_usage_sorted = feature_usage.sort_values(ascending=False)

# 4. 상위 피처를 문자열로 변환
top_features_string = ", ".join(feature_usage_sorted.index)

# 결과 출력
print("사용도가 높은 피처들 (정렬된 순서):")
print(top_features_string)

# 상위 N개의 피처만 표시하려면 (예: 10개)
top_n = 15
top_features_limited = feature_usage_sorted.head(top_n)
print(f"\n상위 {top_n}개의 피처:")
print(", ".join(top_features_limited.index))