import pandas as pd
from konlpy.tag import Okt

filepath_train = './fakenews_train.csv'
filepath_test = './fakenews_test.csv'

# 데이터 프레임 만들기
fakenews_train_pd = pd.read_csv(filepath_train)
fakenews_test_pd = pd.read_csv(filepath_test)

okt = Okt()

def preprocess_and_save(df, stopwords_path, output_csv_path, text_columns=['title', 'content']):
    """
    1. 불용어 txt 불러오기
    2. Okt로 명사 + 동사 추출 (불용어 제거)
    3. 기존 컬럼 덮어쓰기 (1000행마다 진행 상황 출력)
    4. CSV로 저장
    """
    # 불용어 읽기
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]

    # 전처리 함수 정의
    def extract(text):
        tagged = okt.pos(str(text), stem=True)
        tokens = [word for word, pos in tagged if pos in ['Noun', 'Verb'] and word not in stopwords]
        return tokens

    # 전처리 수행 (1000행마다 출력)
    for col in text_columns:
        print(f"🔧 '{col}' 전처리 중...")
        processed_col = []
        for i, text in enumerate(df[col]):
            tokens = extract(text)
            processed_col.append(tokens) 
        df[col] = processed_col  

    # 저장
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n저장 → {output_csv_path}")

    return df

stopwords_file = './stopword.txt'
output_file_train = 'fakenews_preprocessed_trian2.csv'
output_file_test = 'fakenews_preprocessed_test2.csv'

# 함수 실행
fakenews_train_pd = preprocess_and_save(fakenews_train_pd, stopwords_path=stopwords_file, output_csv_path=output_file_train)
fakenews_test_pd = preprocess_and_save(fakenews_test_pd, stopwords_path=stopwords_file, output_csv_path=output_file_test)