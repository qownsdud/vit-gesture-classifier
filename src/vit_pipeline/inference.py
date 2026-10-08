# inference.py

# 동작 방식
# 1. 학습된 vit_gesture.pth 가중치 로드
# 2. 웹캠 화면 중심 영역($224 \times 224$) 크롭 및 정규화
# 3. ViT 추론 후 Softmax 확률 계산
# 4. 예측 클래스, 신뢰도(Confidence Score), 실시간 FPS를 화면에 오버레이
# 5. 추후 ROS2 노드로 바로 바꿀 수 있도록 추론 결과를 표준 딕셔너리({"class": ..., "score": ...}) 형태로 추출

import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

CLASSES = ['paper', 'rock', 'scissors']

def load_model(weights_path='weights/vit_gesture.pth', device='cpu'):
    # 학습할 때와 동일한 구조로 모델 생성
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASSES))
    
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device('cpu')
    print("모델 로딩 중...")
    model = load_model('weights/vit_gesture.pth', device)
    print("모델 로드 완료! 웹캠을 시작합니다.")

    # ViT 전처리 파이프라인
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        box_size = 224
        x1, y1 = cx - box_size // 2, cy - box_size // 2
        x2, y2 = cx + box_size // 2, cy + box_size // 2

        # 1. 중심 관심 영역(ROI) 추출 및 전처리
        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_roi)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # 2. ViT 추론
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probs, dim=0)

        pred_class = CLASSES[pred_idx.item()]
        conf_score = confidence.item()

        # (추후 ROS2 Publisher 토픽으로 발행할 수 있는 표준 데이터 형태)
        inference_result = {
            "class": pred_class,
            "confidence": round(conf_score, 4)
        }

        # 3. FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (prev_time != 0) else 0
        prev_time = curr_time

        # 4. 시각화 화면 구성
        display = frame.copy()
        # 초록색 박스 표시
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 텍스트 출력: 클래스, 확률, FPS
        result_text = f"Class: {pred_class.upper()} ({conf_score*100:.1f}%)"
        fps_text = f"FPS: {fps:.1f}"

        # 신뢰도가 70% 이상일 때만 초록색 글씨, 낮으면 주황색 글씨
        text_color = (0, 255, 0) if conf_score >= 0.7 else (0, 165, 255)

        cv2.putText(display, result_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(display, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ViT Gesture Classifier (Press 'q' to exit)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()