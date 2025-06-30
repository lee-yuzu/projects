import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import os
import pickle
import ast
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

# 뉴스 제목을 이진 분류하는 LSTM 기반 분류 모델 정의
class SentenceClassifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim=128,
        embedding_dim=100,
        n_layers=2,
        dropout=0.6,
        bidirectional=True,
        model_type="lstm"
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        if model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )

        feature_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(feature_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        pooled = torch.mean(output, dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# 데이터세트 불러오기
train = pd.read_csv('./fakenews_preprocessed_train2.csv').sample(frac=0.9, random_state=42)
test = pd.read_csv('./fakenews_preprocessed_test2.csv')


# 데이터 토큰화 및 단어 사전 구축
def build_vocab(corpus, n_vocab, special_tokens, save_path=None, save_path_txt=None):
    counter = Counter()
    for i, tokens in enumerate(corpus):
        counter.update(tokens)
        if (i + 1) % 10000 == 0:
            print(f"{i + 1}개의 문장을 처리했습니다.")
    vocab = special_tokens[:]
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocab pkl 저장: {save_path}")

    if save_path_txt:
        with open(save_path_txt, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(f"{token}\n")
        print(f"Vocab txt 저장: {save_path_txt}")

    print(f"최종 Vocab 크기: {len(vocab)}개")
    return vocab

# content 컬럼의 문자열을 리스트로 변환
train_tokens = [ast.literal_eval(line) for line in train['content']]
test_tokens = [ast.literal_eval(line) for line in test['content']]

# 단어 사전 로딩 or 생성
VOCAB_PATH = "vocab2.pkl"

if os.path.exists(VOCAB_PATH):
    print("Vocab 불러오는 중...")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
else:
    print("Vocab 생성 중...")
    vocab = build_vocab(
        corpus=train_tokens,
        n_vocab=80000,
        special_tokens=["<PAD>", "<UNK>"],
        save_path=VOCAB_PATH,
        save_path_txt="vocab.txt"
    )

# 단어-인덱스 매핑 딕셔너리 생성
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

# 정수 인코딩 및 패딩
def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)

# 토큰 시퀀스를 정수 인코딩
unk_id = token_to_id["<UNK>"]
train_ids = [[token_to_id.get(token, unk_id) for token in review] for review in train_tokens]
test_ids = [[token_to_id.get(token, unk_id) for token in review] for review in test_tokens]

# 고정 길이로 패딩 처리
max_length = 32
pad_id = token_to_id["<PAD>"]
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)

# 데이터로더
train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

train_labels = torch.tensor(train.label.values, dtype=torch.float64)
test_labels = torch.tensor(test.label.values, dtype=torch.float64)

train_dataset = TensorDataset(train_ids, train_labels)
test_dataset = TensorDataset(test_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

# 손실 함수와 최적화 함수 정의
n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = SentenceClassifier(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers
).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001, weight_decay=1e-4)

# 모델 성능 평가 함수
def test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        yhat = torch.sigmoid(logits) > 0.5
        corrects.extend(torch.eq(yhat, labels).cpu().tolist())

    val_loss = np.mean(losses)
    val_accuracy = np.mean(corrects)
    print(f"Val Loss : {val_loss}, Val Accuracy : {val_accuracy}")
    return val_accuracy

# 모델 학습 및 평가
epochs = 10          
interval = 100

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    classifier.train()
    losses = []
    corrects = 0
    total = 0
    processed_samples = 0  

    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = classifier(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 예측값 구하기
        yhat = torch.sigmoid(logits) > 0.5 

        # 정확도 계산
        corrects += torch.sum(yhat == labels).item()
        total += labels.size(0)

        # 배치마다 처리된 데이터 수 추적
        processed_samples += labels.size(0)

        # 100번마다 출력
        if step % interval == 0:
            accuracy = corrects / total  
            print(f"Step {step}, Train Loss: {np.mean(losses)}, Train Accuracy: {accuracy * 100:.2f}%")

    # 에포크 끝날 때 처리된 총 데이터 수 출력
    print(f"Epoch {epoch+1} finished. Total samples processed: {processed_samples}/{len(train_loader.dataset)}")

    # 모델 테스트 후 정확도 계산
    accuracy = test(classifier, test_loader, criterion, device)

    # 모델 저장 (.pt 파일, 파일명에 에포크 번호와 정확도 포함)
    save_path = f"model_epoch_{epoch+1}_accuracy_{accuracy:.4f}.pt"
    torch.save(classifier.state_dict(), save_path)
    print(f"Model saved to {save_path}")