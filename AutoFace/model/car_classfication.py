# ====================================================
# 개선 사항 목록
# 1. 모델 구조를 외부 모듈로 분리하여 코드 메인 파일 가독성 향상
# 2. 훈련 진행 상황을 tqdm 등으로 시각적으로 보여주는 기능 추가
# 3. 평가 지표 시각화
# ====================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image


# 데이터 전처리
# Train: 흑백 변환, 리사이즈, 데이터 증강, 텐서 변환
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Test: 증강 제외
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# 데이터셋 로딩(ImageFolder를 통해 라벨 자동 지정)
dataset = datasets.ImageFolder(root="./car", transform=transform_train)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 브랜드 분류 CNN 모델
# 출력 1개: 시그모이드 사용 (이진 분류)
# 구조: Conv-BN-ReLU-MaxPool ×3 + FC → Sigmoid
class BrandClassifier(nn.Module):
    def __init__(self):
        super(BrandClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 100x100 -> 100x100
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 100x100 -> 50x50

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 50x50 -> 50x50
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 50x50 -> 25x25

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 25x25 -> 25x25
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 25x25 -> 12x12

            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 차종 분류 CNN 모델
# 출력 클래스 수는 인자로 받아 다중 분류
# 구조: Conv-ReLU-Pool ×2 + FC → Softmax
class CarTypeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CarTypeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델, 손실 함수, 옵티마이저 정의
# 이진 분류: BCELoss
# 다중 분류: CrossEntropyLoss
model_brand = BrandClassifier()
criterion_brand = nn.BCELoss()
optimizer_brand = optim.Adam(model_brand.parameters(), lr=0.0001, weight_decay=1e-4)

model_hyundai = CarTypeClassifier(num_classes=8)
model_kg = CarTypeClassifier(num_classes=3)

criterion_type = nn.CrossEntropyLoss()
optimizer_hyundai = optim.Adam(model_hyundai.parameters(), lr=0.0001, weight_decay=1e-4)
optimizer_kg = optim.Adam(model_kg.parameters(), lr=0.0001, weight_decay=1e-4)

# 모델 학습 함수
# 최적 F1-score 기준으로 best model 저장
# 일정 에폭 이상 향상 없으면 EarlyStopping
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, is_brand=False, model_name="model.pt"):
    best_f1 = 0.0
    patience = 3
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            if is_brand:
                labels = labels.view(-1, 1).float()
            else:
                labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # 모델 평가 → 테스트 데이터 기준 성능 저장
        if is_brand:
            f1 = evaluate_brand_model(model, test_loader, return_f1=True)
        else:
            f1 = evaluate_car_model(model, test_loader, return_f1=True)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_name)
            print(f"새로운 best 모델 저장: {model_name} (Test F1-score: {best_f1:.2f}%)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early Stopping - 학습 중단")
                break

# 브랜드 분류 모델 평가 함수
def evaluate_brand_model(model, test_loader, return_f1=False):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = (outputs > 0.5).float().squeeze()
            labels = labels.view(-1, 1).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, zero_division=1) * 100
    recall = recall_score(all_labels, all_predictions, zero_division=1) * 100
    f1 = f1_score(all_labels, all_predictions, zero_division=1) * 100

    print(f"\n브랜드 분류 모델 평가 결과")
    print(f"정확도(Accuracy): {accuracy:.2f}%")
    print(f"정밀도(Precision): {precision:.2f}%")
    print(f"재현율(Recall): {recall:.2f}%")
    print(f"F1-score: {f1:.2f}%\n")

    if return_f1:
        return f1

# 차종 분류 모델 평가 함수
def evaluate_car_model(model, test_loader, return_f1=False):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=1) * 100
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=1) * 100
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=1) * 100

    print(f"\n차종 분류 모델 평가 결과")
    print(f"정확도(Accuracy): {accuracy:.2f}%")
    print(f"정밀도(Precision): {precision:.2f}%")
    print(f"재현율(Recall): {recall:.2f}%")
    print(f"F1-score: {f1:.2f}%\n")

    if return_f1:
        return f1

# 모델 로드 함수
def load_trained_model(model_class, model_path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# (해당 파일에서만) 학습 실행
if __name__ == "__main__":
    train_model(model_brand, train_loader, test_loader, criterion_brand, optimizer_brand, num_epochs=30, is_brand=True, model_name="best_brand_model2.pt")
    train_model(model_hyundai, train_loader, test_loader, criterion_type, optimizer_hyundai, num_epochs=30, model_name="best_hyundai_model2.pt")
    train_model(model_kg, train_loader, test_loader, criterion_type, optimizer_kg, num_epochs=30, model_name="best_kg_model2.pt")

    print("best 모델 저장")

