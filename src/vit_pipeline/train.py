# train.py

# 핵심 동작
# 1. transforms.Compose로 ViT 표준 규격인 224 x 224 크기 변환 및 정규화
# 2. 전체 데이터를 Train(80%)과 Validation(20%)으로 자동 분할
# 3. vit_tiny_patch16_224 백본 가중치 고정(Freeze)
# 4. 마지막 헤드만 교체하여 5 Epoch 미세조정 후 최고 성능 가중치 저장

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

def main():
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    # 1. ViT 전용 전처리 정의 (224x224 고정 필수)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 로드 및 분할 (Train 80% / Val 20%)
    dataset_path = os.path.join('dataset', 'rps')
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)
    classes = full_dataset.classes
    print(f"발견된 클래스: {classes} (총 {len(full_dataset)}장)")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. 경량 ViT 모델 로드 및 백본 고정 (Freeze)
    print("Pretrained ViT 로딩 중...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # 사전 학습 백본 가중치 고정 (CPU 환경에서도 빠른 학습 보장)
    for param in model.parameters():
        param.requires_grad = False

    # 분류 헤드만 우리 클래스 수(3개)로 교체
    num_classes = len(classes)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)

    # 4. 손실함수 및 옵티마이저 (헤드 파라미터만 학습)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    # 5. 미세조정(Fine-Tuning) 루프 (5 Epoch)
    epochs = 5
    best_acc = 0.0
    os.makedirs('weights', exist_ok=True)
    save_path = os.path.join('weights', 'vit_gesture.pth')

    print("\n--- 학습 시작 ---")
    for epoch in range(epochs):
        # [Train]
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = train_correct / train_size

        # [Validation]
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_correct / val_size

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.1f}%")

        # 최고 성능일 때 가중치 저장
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f" -> Best 모델 저장 완료 ({save_path})")

    print(f"\n최종 완료! 최고 검증 정확도: {best_acc*100:.2f}%")

if __name__ == '__main__':
    main()