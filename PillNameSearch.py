import pandas as pd

# train_csv_path = "some_of_drug1.csv"
#
# try:
#     data = pd.read_csv(train_csv_path, encoding='utf-8')
# except UnicodeDecodeError:
#     try:
#         data = pd.read_csv(train_csv_path, encoding='utf-8-sig')
#     except UnicodeDecodeError:
#         data = pd.read_csv(train_csv_path, encoding='euc-kr')


# 품목명 기준으로 업소명과 분류명 출력
def print_details_for_item(data, target_name):
    for _, row in data.iterrows():  # DataFrame을 행별로 순회
        if row["품목명"] == target_name:
            return row["업소명"], row["분류명"], row["품목허가일자"]
    return None
