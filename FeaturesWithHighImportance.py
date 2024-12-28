import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def get_features_with_high_importance():
    # 데이터 로드
    train_data = pd.read_csv("train_disease.csv")
    test_data = pd.read_csv("test_disease.csv")

    # 'prognosis' 컬럼 확인
    if 'prognosis' not in train_data.columns or 'prognosis' not in test_data.columns:
        print("Error: 'prognosis' column is missing.")
        return

    # X와 y 분리
    X_train = train_data.drop(columns=['prognosis'])
    X_test = test_data.drop(columns=['prognosis'])
    y_train = train_data['prognosis']
    y_test = test_data['prognosis']

    # 결측값 처리
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # 랜덤 포레스트 모델 학습
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # 피처 중요도 추출
    feature_importances = rf_model.feature_importances_

    # 피처 중요도를 데이터프레임으로 변환
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # 상위 20개 중요한 피처 출력
    top_20_features = importance_df.head(20)

    return top_20_features

    # print("상위 20개 피처 중요도:")
    # print(top_20_features[['Feature', 'Importance']])

# get_features_with_high_importance()