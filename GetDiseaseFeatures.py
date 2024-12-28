import pandas as pd

def get_disease_features():
    """
    질병과 중요 피처 데이터를 반환하는 함수
    """
    # 데이터 로드
    train_data = pd.read_csv("train_disease.csv")

    # 'prognosis' 컬럼을 제외한 나머지 증상 데이터 (피처들)
    X = train_data.drop(columns=['prognosis'])

    # 'prognosis' 컬럼을 포함한 데이터
    y = train_data['prognosis']

    # 각 prognosis에 대해 1을 많이 가지는 피처를 추출
    disease_features = {}

    # 모든 고유 질병에 대해 계산
    for disease in y.unique():
        # 해당 질병만 필터링
        disease_data = train_data[train_data['prognosis'] == disease]

        # 'prognosis' 컬럼을 제외한 증상 피처들
        X_disease = disease_data.drop(columns=['prognosis'])

        # 1의 값을 가진 피처들의 개수 계산
        feature_ones_count = X_disease.sum(axis=0)

        # 1의 값이 있는 피처들만 필터링 (0의 값을 가지는 피처는 제외)
        non_zero_features = feature_ones_count[feature_ones_count > 0]

        # 1의 값을 가진 개수를 기준으로 내림차순 정렬
        sorted_features = non_zero_features.sort_values(ascending=False)

        # 상위 20개 피처 추출
        top_20_features = sorted_features.head(20)

        # 각 질병의 중요 피처를 딕셔너리에 저장
        disease_features[disease] = top_20_features

    return disease_features