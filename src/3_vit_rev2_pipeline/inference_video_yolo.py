# inference_video_yolo.py

# Yolo를 활용한 인식범위 자율조정 + 동영상 인식해보기

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
from ultralytics import YOLO

def load_hand_detector():
    """YOLOv8 손 검출 모델 로드 (없을 시 경량 hand 가중치 자동 다운로드/활용)"""
    model_name = "yolov8n.pt"
    # Hugging Face 또는 공개 저장소의 손 전용 가중치가 로컬에 있으면 우선 사용
    if os.path.exists("weights/yolov8n-hand.pt"):
        model_name = "weights/yolov8n-hand.pt"
    
    print(f"[*] Loading Detector: {model_name}")
    detector = YOLO(model_name)
    return detector

def main():
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] ViT Inference Device: {device}")

    classes = ["PAPER", "ROCK", "SCISSORS"]
    class_colors = {
        "PAPER": (255, 150, 0),
        "ROCK": (0, 165, 255),
        "SCISSORS": (0, 255, 0)
    }

    # 1. ViT 분류 모델 로드
    weights_path = "weights/vit_gesture_honest.pth"
    if not os.path.exists(weights_path):
        weights_path = "weights/vit_gesture.pth"

    vit_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3)
    vit_model.load_state_dict(torch.load(weights_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    # 2. YOLOv8 검출 모델 로드
    detector = load_hand_detector()

    # 3. 비디오 로드
    video_path = "tests/test_video.mp4"
    cap = cv2.VideoCapture(video_path if os.path.exists(video_path) else 0)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[*] YOLO + ViT 2-Stage 추론 시작 (종료: 'q' 키)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        
        # 1-Stage: YOLO 탐지 (conf 임계값 0.25)
        results = detector(frame, conf=0.25, verbose=False, device='cpu')
        boxes = results[0].boxes

        pred_label = "NO HAND"
        confidence = 0.0
        probs = [0.0, 0.0, 0.0]

        best_box = None
        if len(boxes) > 0:
            # 가장 신뢰도(conf)가 높은 박스 선택
            best_idx = int(torch.argmax(boxes.conf))
            xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # 정사각 비율 및 20% 여백(Padding) 추가 (손가락 끝 잘림 방지)
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half_side = int(max(bw, bh) * 0.6)

            crop_x1 = max(0, cx - half_side)
            crop_y1 = max(0, cy - half_side)
            crop_x2 = min(w, cx + half_side)
            crop_y2 = min(h, cy + half_side)

            roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # 2-Stage: ViT 제스처 분류
            if roi.shape[0] > 20 and roi.shape[1] > 20:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(roi_rgb)
                tensor = preprocess(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = vit_model(tensor)
                    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

                idx = np.argmax(probs)
                pred_label = classes[idx]
                confidence = probs[idx] * 100
                best_box = (crop_x1, crop_y1, crop_x2, crop_y2)

        # 시각화 박스 렌더링
        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            box_color = class_colors.get(pred_label, (0, 255, 0))
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(frame, f"[YOLO+ViT] {pred_label} ({confidence:.1f}%)",
                        (bx1, max(30, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # UI 대시보드
        cv2.rectangle(frame, (20, 20), (370, 160), (20, 20, 20), -1)
        cv2.putText(frame, "2-Stage (YOLO + ViT)", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        for i, (cls_name, prob) in enumerate(zip(classes, probs)):
            y_offset = 80 + i * 25
            bar_w = int(prob * 130)
            color = (0, 255, 0) if (pred_label == cls_name) else (100, 100, 100)
            cv2.putText(frame, f"{cls_name:8s}: {prob*100:5.1f}%", (35, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.rectangle(frame, (180, y_offset), (180 + bar_w, y_offset + 12), color, -1)
            cv2.rectangle(frame, (180, y_offset), (310, y_offset + 12), (70, 70, 70), 1)

        cv2.imshow("YOLOv8 + ViT Pipeline", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()