# 모듈 로딩
import os
import cv2                      
import numpy as np            
import pandas as pd
import matplotlib.pyplot as plt 
import koreanize_matplotlib
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# 데이터 준비
IMG_DIR = './kfood_prac/'
FILE_CSV = './kfood1.csv'

# 이미지 파일 리스트 불러오기
for path, _, file in os.walk(IMG_DIR):
	if path == './kfood_prac/main_dish':
		main = file
	elif path == './kfood_prac/side_dish':
		side = file
	elif path == './kfood_prac/dessert':
		dessert = file

# 이미지 전처리(리사이즈)
imglists = [main, side, dessert]
angle = [180]

for imglist in imglists:
	if imglist == main:
		path = './kfood_prac/main_dish/'
	elif imglist == side:
		path = './kfood_prac/side_dish/'
	elif imglist == dessert:
		path = './kfood_prac/dessert/'

	for img in imglist:
		for a in angle:
			image_path = path+img
			if not os.path.exists(image_path):
				print(f"이미지 파일이 존재하지 않습니다: {image_path}")
			else:
				colorImg = cv2.imread(image_path, 1)

				if colorImg is None:
					print(f"이미지 로드 실패: {image_path}")
				else:
					colorImg = cv2.imread(image_path, cv2.IMREAD_COLOR)
					downImg = cv2.resize(colorImg, (64, 64), interpolation=cv2.INTER_AREA)

					# 이미지 저장
					SAVE_FILENAME = image_path
					ret = cv2.imwrite(SAVE_FILENAME, downImg)

# 이미지 파일 리스트 업데이트
for path, _, file in os.walk(IMG_DIR):
	if path == './kfood_prac/main_dish':
		main = file
	elif path == './kfood_prac/side_dish':
		side = file
	elif path == './kfood_prac/dessert':
		dessert = file

# 이미지 벡터화 사용자 함수
def img_read_flatten(path):
    img = cv2.imread(path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.flatten()
    return img

# CSV 파일 저장
imglist = [(main, 0, 'main_dish'), (side, 1, 'side_dish'), (dessert, 2, 'dessert')]

with open(FILE_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    for imagelist, label, name in imglist:
        for path in imagelist:
            try:
                img_vector = img_read_flatten(IMG_DIR+name+'/'+path)
                row = list(img_vector) + [label]
                writer.writerow(row)
            except Exception as e:
                print(f"오류 발생: {path}, {e}")

# CSV 파일 불러오기
kfood_df = pd.read_csv(FILE_CSV, header=None)
kfood_df.head()

# 독립변수(X), 종속변수(y) 분리
X = kfood_df.drop(columns=[12288])  
y = kfood_df[12288]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# 로지스틱 회귀 모델
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))

# 랜덤포레스트 모델
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# 랜덤포레스트 하이퍼파라미터 튜닝
rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [100, 200],  # 트리 개수
    'max_depth': [None, 10],      # 최대 깊이
    'min_samples_split': [2, 5]   # 분할을 위한 최소 샘플 수
}

# RandomizedSearchCV 설정 (5번 샘플링)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters found: ", random_search.best_params_)

# 최적 하이퍼파라미터로 새로운 모델 생성
best_rf = RandomForestClassifier(**random_search.best_params_, random_state=42)

# 최적 하이퍼파라미터로 모델 학습
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))

# 라벨 인코더 (0=주식, 1=부식, 2=후식)
label_encoder = LabelEncoder()
label_encoder.fit([0, 1, 2])  # 0, 1, 2 모두 포함하도록 수정

# 모델 저장
with open("model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

# (선택) label_encoder도 있다면
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)