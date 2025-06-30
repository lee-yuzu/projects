# ====================================================
# 개선 사항 목록
# 1. 전처리 및 모델 정의와 추론 코드가 app.py에 모두 혼재되어 있어 유지보수 어려움
# ====================================================

from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 허용 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# 예측 함수
def predict_image(image_path, brand_model, hyundai_model, kg_model):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)

    # 1단계: 브랜드 예측
    with torch.no_grad():
        brand_output = brand_model(image)
        brand_prob = brand_output.item()

    # 데이터 편향 조정
    brand_prob = max(0, min(1, brand_prob - 0.2))
    if brand_prob < 0.3:
        brand = "Hyundai"
        car_model = hyundai_model
    elif brand_prob > 0.7:
        brand = "KG"
        car_model = kg_model
    else:
        return None, None

    # 2단계: 차종 예측
    with torch.no_grad():
        car_output = car_model(image)
        car_prediction = torch.argmax(F.softmax(car_output, dim=1), dim=1).item()

    hyundai_classes = ["Avante", "Casper", "Genesis", "Grandeur", "Santafe", "Sonata", "Starex", "Tucson"]
    kg_classes = ["Korando", "Rexton", "Tivoli"]

    car_type = hyundai_classes[car_prediction] if brand == "Hyundai" else kg_classes[car_prediction]
    return brand, car_type

# 모델 클래스 정의
# 브랜드
class BrandClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
# 상세 차종
class CarTypeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 모델 로드
model_brand = BrandClassifier()
model_hyundai = CarTypeClassifier(8)
model_kg = CarTypeClassifier(3)

model_brand.load_state_dict(torch.load("best_brand_model2.pt", map_location='cpu'))
model_hyundai.load_state_dict(torch.load("best_hyundai_model2.pt", map_location='cpu'))
model_kg.load_state_dict(torch.load("best_kg_model2.pt", map_location='cpu'))

model_brand.eval()
model_hyundai.eval()
model_kg.eval()

# 라우터
@app.route('/')
def home():
    return render_template('index.html',
                           uploaded_image=None,
                           predicted_brand=None,
                           predicted_car_type=None,
                           error=None,
                           show_upload_box=False)

@app.route('/reset')
def reset():
    return render_template('index.html',
                           uploaded_image=None,
                           predicted_brand=None,
                           predicted_car_type=None,
                           error=None,
                           show_upload_box=True)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        brand, car_type = predict_image(filepath, model_brand, model_hyundai, model_kg)
        if brand is None:
            return render_template('index.html', error="브랜드를 명확히 판단할 수 없습니다.", uploaded_image=filename)
        return render_template('index.html', uploaded_image=filename, predicted_brand=brand, predicted_car_type=car_type)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
