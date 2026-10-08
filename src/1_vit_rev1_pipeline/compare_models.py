# compare_models.py

# 일반 train 모델인 train.py와 강건한 train 모델인 train_robust.py의 성능을
# 트럼프 사진을 분석한 결과로 비교해보기

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

CLASSES = ['paper', 'rock', 'scissors']

def get_model(weights_path):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    return model

def main():
    img_path = 'tests\my_hand.jpg'
    if not os.path.exists(img_path):
        print(f"오류: {img_path} 파일이 없습니다! 프로젝트 폴더에 실제 손 사진(my_hand.jpg)을 넣어주세요.")
        return

    # 추론용 전처리
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)

    # 1. 기본 모델 추론
    base_model = get_model('weights/vit_gesture.pth')
    with torch.no_grad():
        base_out = base_model(input_tensor)
        base_prob = torch.softmax(base_out, dim=1)[0]
        base_conf, base_idx = torch.max(base_prob, dim=0)

    # 2. 강건 모델 추론
    robust_model = get_model('weights/vit_gesture_robust.pth')
    with torch.no_grad():
        rob_out = robust_model(input_tensor)
        rob_prob = torch.softmax(rob_out, dim=1)[0]
        rob_conf, rob_idx = torch.max(rob_prob, dim=0)

    print("\n================ [실제 사진 모델 비교 결과] ================")
    print(f"테스트 이미지: {img_path}")
    print(f"1. 기본 모델       예측: {CLASSES[base_idx]:<8} | 신뢰도: {base_conf.item()*100:.2f}%")
    print(f"2. 강건 모델(증강) 예측: {CLASSES[rob_idx]:<8} | 신뢰도: {rob_conf.item()*100:.2f}%")
    print("============================================================")

if __name__ == '__main__':
    main()