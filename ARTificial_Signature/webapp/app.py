# ====================================================
# 개선 사항 목록
# 1. 모델 로딩 로직을 함수로 분리하여 중복 제거 및 재사용성 향상
# 2. DB 접속 정보(user, password 등)를 .env 파일로 분리해 보안 강화
# 3. 라벨 리스트(labels)와 모델 리스트(model_list)를 config.py로 분리해 관리 용이성 확보
# 4. 모델별 예측 로직을 별도 함수로 분리하여 구조화
# ====================================================

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import pymysql
import datetime

# 현재 파일 위치 기준 BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask 앱 초기화 및 업로드 폴더 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')

# 예측 라벨 목록 정의
labels = ['Vincent van Gogh', 'Edgar Degas', 'Pablo Picasso', 'Pierre-Auguste Renoir', 'Albrecht Durer']

# 이미지 전처리 파이프라인 정의
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])


# 모델 불러오기
# resnet18
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(512, 5)
resnet_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'resnet18.pth'), map_location='cpu'))
resnet_model.eval()

# densenet121
densenet_model = models.densenet121(pretrained=False)
densenet_model.classifier = nn.Linear(1024, 5)
densenet_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'densenet121.pth'), map_location='cpu'))
densenet_model.eval()

# efficientnetb0
efficientnet_model = models.efficientnet_b0(pretrained=False)
efficientnet_model.classifier[1] = nn.Linear(1280, 5)
efficientnet_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'efficientnetb0.pth'), map_location='cpu'))
efficientnet_model.eval()

# mobilenetv2
mobilenet_model = models.mobilenet_v2(pretrained=False)
mobilenet_model.classifier[1] = nn.Linear(1280, 5)
mobilenet_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'mobilenetv2.pth'), map_location='cpu'))
mobilenet_model.eval()

# vgg16
vgg_model = models.vgg16(pretrained=False)
vgg_model.classifier[6] = nn.Linear(4096, 5)
vgg_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'vgg16.pth'), map_location='cpu'))
vgg_model.eval()

# DB 연결 함수 정의
def get_db_connection():
    conn = pymysql.connect(
        host='localhost',
        user='yujung',
        password='1234',
        db='13_web_flask',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

# 메인 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 처리
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    return redirect(url_for('result', filename=filename))

# 예측 결과 페이지 처리
@app.route('/result/<filename>')
def result(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    predictions = []

    # 모델 리스트 정의
    model_list = [
        ('ResNet18', resnet_model),
        ('DenseNet121', densenet_model),
        ('EfficientNetB0', efficientnet_model),
        ('MobileNetV2', mobilenet_model),
        ('VGG16', vgg_model)
    ]

    # 각 모델로 예측
    for model_name, model in model_list:
        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            label = labels[preds.item()]
            confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100
            predictions.append({
                'model': model_name,
                'label': label,
                'confidence': round(confidence, 2)
            })

    # 가장 높은 confidence 예측에 highlight 추가
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    for i, pred in enumerate(predictions):
        pred['highlight'] = (i == 0)

    return render_template('result.html', filename=filename, predictions=predictions)

# 게시판 공통 처리 함수
def handle_board(table_name, board_name, artist_wiki_url):
    if request.method == 'POST':
        content = request.form['content']
        created_at = datetime.datetime.now()

        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = f"INSERT INTO {table_name} (content, created_at) VALUES (%s, %s)"
            cursor.execute(sql, (content, created_at))
            conn.commit()
        conn.close()

        return redirect(request.url)

    conn = get_db_connection()
    with conn.cursor() as cursor:
        sql = f"SELECT * FROM {table_name} ORDER BY created_at DESC"
        cursor.execute(sql)
        comments = cursor.fetchall()
    conn.close()

    return render_template('board.html', comments=comments, board_name=board_name, artist_wiki_url=artist_wiki_url)

# 각 화가별 게시판 라우팅 처리
@app.route('/board/van_gogh', methods=['GET', 'POST'])
def van_gogh_board():
    return handle_board('vincent_comments', 'Vincent van Gogh Comment', 'https://ko.wikipedia.org/wiki/%EB%B9%88%EC%84%BC%ED%8A%B8_%EB%B0%98_%EA%B3%A0%ED%9D%90')

@app.route('/board/edgar_degas', methods=['GET', 'POST'])
def edgar_degas_board():
    return handle_board('edgar_comments', 'Edgar Degas Comment', 'https://ko.wikipedia.org/wiki/%EC%97%90%EB%93%9C%EA%B0%80_%EB%93%9C%EA%B0%80')

@app.route('/board/pablo_picasso', methods=['GET', 'POST'])
def pablo_picasso_board():
    return handle_board('pablo_comments', 'Pablo Picasso Comment', 'https://ko.wikipedia.org/wiki/%ED%8C%8C%EB%B8%94%EB%A1%9C_%ED%94%BC%EC%B9%B4%EC%86%8C')

@app.route('/board/renoir', methods=['GET', 'POST'])
def renoir_board():
    return handle_board('pierre_comments', 'Pierre-Auguste Renoir Comment', 'https://ko.wikipedia.org/wiki/%EC%98%A4%EA%B7%80%EC%8A%A4%ED%8A%B8_%EB%A5%B4%EB%88%84%EC%95%84%EB%A5%B4')

@app.route('/board/albrecht_durer', methods=['GET', 'POST'])
def albrecht_durer_board():
    return handle_board('albrecht_comments', 'Albrecht Durer Comment', 'https://ko.wikipedia.org/wiki/%EC%95%8C%EB%B8%8C%EB%A0%88%ED%9E%88%ED%8A%B8_%EB%92%A4%EB%9F%AC')

# 서버 실행 설정
if __name__ == '__main__':
    # 업로드 폴더가 없으면 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
