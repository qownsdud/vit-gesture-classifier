# train_robust.py

# 강건한 train 모델

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

def main():
    device = torch.device('cpu')
    print(f"사용 디바이스: {device}")

    # 1. 강건성(Robustness) 향상을 위한 전처리 및 데이터 증강 파이프라인
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 조명/그림자 변화 대응
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        # 손의 위치 및 각도 편차 대응
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        # 배경/가려짐 노이즈 강건성 (일부 패치 가리기)
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), value='random'),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 검증셋은 순수 평가용이므로 왜곡 없이 표준 변환만 적용
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 로드
    dataset_path = os.path.join('dataset', 'rps')
    full_dataset = datasets.ImageFolder(root=dataset_path)
    classes = full_dataset.classes
    print(f"클래스: {classes} (총 {len(full_dataset)}장)")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    # 각각에 맞는 transform 적용
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

    # 3. 모델 로드 및 미세조정 세팅
    print("Pretrained ViT 로딩 및 레이어 세팅 중...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # 백본 기본 Freeze
    for param in model.parameters():
        param.requires_grad = False

    # 강건성을 위해 ViT의 마지막 트랜스포머 블록(blocks[-1])만 추가 언프리즈
    for param in model.blocks[-1].parameters():
        param.requires_grad = True

    # 분류 헤드 교체
    model.head = nn.Linear(model.head.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 학습 가능한 레이어만 Optimizer에 등록
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-2)

    # 4. 학습 루프
    epochs = 5
    best_acc = 0.0
    save_path = os.path.join('weights', 'vit_gesture_robust.pth')

    print("\n--- 강건성 강화 학습 시작 ---")
    for epoch in range(epochs):
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

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f" -> Best 강건 모델 저장 완료 ({save_path})")

    print(f"\n최종 완료! 최고 정확도: {best_acc*100:.2f}%")

if __name__ == '__main__':
    main()