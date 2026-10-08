# inference_video.py

# 고정된 인식 범위로 동영상 인식해보기

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import timm

def main():
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference on: {device}")

    classes = ["PAPER", "ROCK", "SCISSORS"]
    class_colors = {
        "PAPER": (255, 150, 0),    # 주황빛 하늘색
        "ROCK": (0, 165, 255),     # 주황
        "SCISSORS": (0, 255, 0)    # 초록
    }

    # 1. 정직한 ViT 가중치 로드
    weights_path = "weights/vit_gesture_honest.pth"
    if not os.path.exists(weights_path):
        print(f"[!] 가중치 파일을 찾을 수 없습니다: {weights_path}")
        print("[*] 먼저 train_honest_vit.py 학습을 완료해 주세요.")
        return

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[*] 모델 가중치 로드 완료: {weights_path}")

    # 2. 비디오 소스 설정
    video_path = "tests/test_video.mp4"
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[*] 비디오 추론 시작 (종료: 'q' 키)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 영상 반복 재생
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        # 중앙 손 영역 ROI 크기 지정 (400x400)
        box_size = 400
        x1 = max(0, w // 2 - box_size // 2)
        y1 = max(0, h // 2 - box_size // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ROI 전처리 및 텐서 변환
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # 모델 추론
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        confidence = probs[pred_idx] * 100

        # UI 오버레이: 상단 대시보드 박스
        cv2.rectangle(frame, (20, 20), (380, 170), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, 20), (380, 170), (70, 70, 70), 2)

        # 타이틀
        cv2.putText(frame, "Honest ViT Inference", (35, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3개 클래스 확률 바(Bar) 차트 표시
        for i, (cls_name, prob) in enumerate(zip(classes, probs)):
            y_offset = 80 + i * 28
            bar_w = int(prob * 140)
            color = (0, 255, 0) if i == pred_idx else (120, 120, 120)

            # 라벨 및 백분율 텍스트
            cv2.putText(frame, f"{cls_name:8s}: {prob*100:5.1f}%", (35, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            # 확률 바
            cv2.rectangle(frame, (190, y_offset), (190 + bar_w, y_offset + 14), color, -1)
            cv2.rectangle(frame, (190, y_offset), (330, y_offset + 14), (80, 80, 80), 1)

        # ROI 테두리 및 예측 텍스트 표시
        box_color = class_colors.get(pred_label, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        cv2.imshow("Honest ViT - Real-Time Inference", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()