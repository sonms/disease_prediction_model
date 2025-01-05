import pandas as pd

# 매칭 단어의 사전 정의
translation_dict = {
    "Fungal infection": "곰팡이 감염",
    "Allergy": "알레르기",
    "GERD": "위식도 역류병",
    "Chronic cholestasis": "만성 담즙정체",
    "Drug Reaction": "약물반응",
    "Peptic ulcer diseae": "소화성 궤양 질환",
    "AIDS": "에이즈",
    "Diabetes ": "당뇨병",
    "Gastroenteritis": "위장염",
    "Bronchial Asthma": "기관지천식",
    "Hypertension ": "고혈압",
    "Migraine": "편두통",
    "Cervical spondylosis": "척추증",
    "Paralysis (brain hemorrhage)": "마비(뇌출혈)",
    "Jaundice": "황달",
    "Malaria": "말라리아",
    "Chicken pox": "수두",
    "Dengue": "뎅기",
    "Typhoid": "장티푸스",
    "hepatitis A": "A형 간염",
    "Hepatitis B": "B형 간염",
    "Hepatitis C": "C형 간염",
    "Hepatitis D": "D형 간염",
    "Hepatitis E": "E형 간염",
    "Alcoholic hepatitis": "알코올성 간염",
    "Tuberculosis": "결핵",
    "Common Cold": "감기",
    "Pneumonia": "폐렴",
    "Dimorphic hemmorhoids(piles)": "이중형 치질(치핵)",
    "Heart attack": "심장마비",
    "Varicose veins": "정맥류",
    "Hypothyroidism": "갑상샘저하증",
    "Hypoglycemia": "저혈당",
    "Osteoarthristis": "골관절염",
    "Arthritis": "관절염",
    "(vertigo) Paroymsal  Positional Vertigo": "(현기증)발작성 체위현기",
    "Acne": "여드름",
    "Urinary tract infection": "요로 감염",
    "Psoriasis": "건선",
    "Impetigo": "농가진",
    "Hyperthyroidism" : "갑상샘과다증"
}

# CSV 파일 경로 (원본)
input_csv_path = ".csv"  # 경로는 문자열이어야 합니다.

# 수정된 데이터를 저장할 CSV 파일 경로
output_csv_path = ".csv"

# 특정 열 이름
target_column = "prognosis"

# CSV 파일 읽기
df = pd.read_csv(input_csv_path, encoding="euc-kr")  # CSV 파일 경로와 인코딩 설정

# 열 데이터 치환
if target_column in df.columns:  # 대상 열이 있는지 확인
    df[target_column] = df[target_column].replace(translation_dict)  # 치환

# 수정된 데이터를 새 CSV로 저장
df.to_csv(output_csv_path, index=False, encoding="utf-8")

print(f"CSV 파일이 수정되어 저장되었습니다: {output_csv_path}")
