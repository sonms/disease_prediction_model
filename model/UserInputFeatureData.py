from pydantic import BaseModel

# 입력 데이터 구조 정의 이 클래스를 상속하여 데이터 모델을 정의하고, 해당 모델의 필드와 유효성 검사 규칙을 설정할 수 있다.
class UserInput(BaseModel):
    itching: int
    joint_pain: int
    stomach_pain: int
    vomiting: int
    fatigue: int
    high_fever: int
    dark_urine: int
    nausea: int
    loss_of_appetite: int
    abdominal_pain: int
    diarrhoea: int
    mild_fever: int
    yellowing_of_eyes: int
    chest_pain: int
    muscle_weakness: int
    muscle_pain: int
    altered_sensorium: int
    family_history: int
    mucoid_sputum: int
    lack_of_concentration: int

# class UserInput(BaseModel):
#     fatigue: int
#     vomiting: int
#     high_fever: int
#     loss_of_appetite: int
#     nausea: int
#     headache: int
#     abdominal_pain: int
#     yellowish_skin: int
#     yellowing_of_eyes: int
#     chills: int
#     skin_rash: int
#     malaise: int
#     chest_pain: int
#     joint_pain: int
#     sweating: int