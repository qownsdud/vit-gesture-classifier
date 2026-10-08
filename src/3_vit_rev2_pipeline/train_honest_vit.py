# train_honest_vit.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def get_honest_transforms():
    # 꼼수를 원천 차단하는 데이터 증강(Augmentation) 파이프라인
    train_transform = transforms.Compose([
        # 1. 스케일 및 비율 무작위 왜곡: 캔버스 내 점유 면적으로 클래스를 찍지 못하게 방지
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        # 2. 다양한 손 각도 반영
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        # 3. 조명/피부톤 편향 방지
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 4. 핵심 특징 강제 학습: 손가락 일부가 가려져도 나머지 손 모양을 보도록 Random Erasing 적용
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), value='random')
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_honest_vit(data_dir="dataset/rps", epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    train_tf, val_tf = get_honest_transforms()

    # 데이터셋 로드 (기존 rps 데이터셋 디렉터리 경로)
    train_dir = os.path.join(data_dir, "rps") if os.path.exists(os.path.join(data_dir, "rps")) else data_dir
    if not os.path.exists(train_dir):
        print(f"[!] 데이터셋 경로를 찾을 수 없습니다: {train_dir}")
        print("[*] src/vit_pipeline/download_dataset.py를 먼저 실행해 주세요.")
        return

    dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # ViT-tiny 모델 불러오기
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 과적합 방지 라벨 스무딩
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/vit_gesture_honest.pth"

    print("[*] 정직한 ViT 학습 시작...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # 검증
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best Model 가중치 갱신 저장: {save_path} (Val Acc: {val_acc*100:.2f}%)")

    print(f"[*] 학습 완료! 최종 가중치: {save_path}")

if __name__ == "__main__":
    train_honest_vit()