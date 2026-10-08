# 정직한 ViT 재구축과 YOLOv8 기반 2-Stage 제스처 인식 파이프라인

## 1. 배경 및 문제 제기

이전 파이프라인에서 MediaPipe를 도입하여 손 관절 기반의 실시간 분류를 안정화했으나, 딥러닝 백본 연구 관점에서 ViT의 근본적인 문제점인 **지름길 학습(Shortcut Learning)**을 극복하는 '정직한 ViT' 모델을 구현하고자 했습니다.

초기 모델(`vit_gesture.pth`)은 배경 여백과 프레임 내 살색 픽셀 점유율만으로 클래스를 예측하는 편향을 보였습니다. 이를 해결하기 위해 두 가지 핵심 과제를 정의했습니다:
1. **형태 기반 특징 학습**: 배경 및 살색 점유율에 얽매이지 않고 손가락 윤곽 형태를 온전히 학습하도록 데이터 증강 파이프라인 재설계.
2. **동적 프레이밍(Dynamic Framing)**: 수동 고정 ROI(400x400)의 한계를 벗어나, MediaPipe에 의존하지 않고 화면 내 손의 위치를 스스로 추적하여 잘라내는 엔드투엔드 2-Stage 파이프라인 구축.

---

## 2. 정직한 ViT(Honest ViT) 학습 파이프라인 설계

### 2.1 편향 차단 데이터 증강 (Augmentation Strategy)

모델이 손바닥 전체 면적이 아닌 돌출된 손가락 끝(Fingertip)의 펼침 여부와 경계선(Edge)에 주목하도록 강제하는 증강 파이프라인을 구축했습니다 (`src/vit_rev2_pipeline/train_honest_vit.py`).

* **`RandomResizedCrop(224, scale=(0.5, 1.0))`**: 손의 크기와 프레임 내 점유율을 매 에포크마다 무작위로 변경하여 '면적이 크면 보자기(PAPER)'라는 지름길 학습을 차단.
* **`RandomRotation(±30°)` 및 `RandomHorizontalFlip`**: 카메라 각도 및 좌우 손 방향 다양성 확보.
* **`ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)`**: 특정 피부 톤과 배경 단색(흰색)에 대한 과적합 방지.
* **`RandomErasing(p=0.5, scale=(0.02, 0.25))`**: 손바닥 중심부나 손가락 일부를 무작위로 마스킹하여, 가려지지 않은 나머지 손가락 윤곽만을 보고 형태를 추론하도록 유도.

### 2.2 최적화 설정

* **Backbone**: `vit_tiny_patch16_224` (사전 학습 가중치 로드)
* **손실 함수**: CrossEntropyLoss (Label Smoothing 0.1 적용으로 과신 방지)
* **옵티마이저 & 스케줄러**: AdamW (lr=$1\times 10^{-4}$, Weight Decay=$1\times 10^{-2}$), Cosine Annealing LR

---

## 3. 고정 ROI의 한계와 자동 프레이밍(Auto-Framing)의 필연성

초기 비디오 추론(`inference_video.py`)에서는 화면 중앙에 고정 크기 박스(280x280 $\rightarrow$ 400x400)를 배치하고 손을 그 안에 맞추도록 설계했습니다.

### 고정 박스의 한계점:
* 사용자의 손 위치가 화면 중심에서 벗어날 경우 손가락이 잘려 오분류 발생.
* 카메라와의 거리에 따라 박스 내 배경 노이즈가 과도하게 포함되어 ViT 입력 품질 저하.
* **결론**: 실세계 인터랙션 환경에서는 손의 위치를 실시간으로 추적하여 유효 영역만 크롭(Crop)해 전달하는 **Object Detection 단계가 선행되어야 함**.

---

## 4. MediaPipe 대안 탐색: YOLOv8 기반 2-Stage 아키텍처

관절 랜드마크 추출 라이브러리(MediaPipe)에 의존하지 않는 독립적인 딥러닝 비전 파이프라인을 구축하기 위해, 범용 객체 탐지기인 **YOLOv8-nano**를 전단에 배치하는 2-Stage 시스템을 도입했습니다.

```
[입력 프레임 (Webcam / Video)]
          │
          ▼
┌───────────────────────────────────────────────┐
│ Stage 1: Detector (YOLOv8n)                   │
│ - 화면 내 손/객체 영역 실시간 바운딩 박스 검출  │
│ - Center-point 기반 정사각 비율 보정 + 20% 마진│
└───────────────────────────────────────────────┘
          │
          │ [정사각 크롭된 Hand ROI]
          ▼
┌───────────────────────────────────────────────┐
│ Stage 2: Classifier (Honest ViT-tiny)         │
│ - 224x224 리사이즈 및 정규화                   │
│ - 손가락 윤곽 기반 ROCK / PAPER / SCISSORS 분류│
└───────────────────────────────────────────────┘
          │
          ▼
[최종 실시간 시각화 오버레이 (Bounding Box + Confidence Bar)]
```

### 4.1 정사각 마진 크롭 알고리즘
검출된 박스($w, h$)가 직사각형일 경우, ViT 입력($224 \times 224$)으로 왜곡 없이 전달하기 위해 더 긴 변을 기준으로 정사각형 확장 및 20% 여백(Padding)을 적용했습니다:

$$\mathrm{half\_side} = \lfloor \max(w, h) \times 0.6 \rfloor$$

이를 통해 손가락 끝 마디가 잘리지 않고 온전한 제스처 형태가 ViT로 전달되도록 보장했습니다.

---

## 5. 트러블슈팅: 최신 하드웨어(RTX 50 시리즈) 아키텍처 호환성 문제

### 5.1 발생 문제
YOLO 추론 스크립트 실행 중 PyTorch CUDA 런타임 에러 발생:
```text
UserWarning: NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### 5.2 원인 분석
* 사용 중인 그래픽카드(NVIDIA GeForce RTX 5060 Ti)는 최신 아키텍처(Blackwell, Compute Capability `sm_120`)를 사용함.
* 당시 로컬에 설치된 공식 PyTorch 릴리즈 빌드는 최대 `sm_90`(Hopper)까지만 컴파일된 CUDA 커널을 내장하고 있어, YOLOv8이 기본 GPU 텐서 연산을 시도할 때 바이너리 비호환성으로 커널 실행이 실패함.

### 5.3 해결 방안
YOLOv8-nano 모델은 모바일 구동이 가능할 정도로 매우 경량화(파라미터 약 300만 개)되어 있어 CPU 연산만으로도 실시간 처리가 가능합니다.  
추론 옵션에 `device='cpu'`를 명시하여 비호환 GPU 런타임 할당을 우회하고 안정적인 실시간 30+ FPS 처리를 확보했습니다.
```python
results = detector(frame, conf=0.25, verbose=False, device='cpu')
```

---

## 6. 최종 파이프라인 비교 평가

| 파이프라인 구분 | Rev 1 (초기 ViT) | Rev 2 (MediaPipe) | Rev 3 (YOLOv8 + Honest ViT) |
| :--- | :--- | :--- | :--- |
| **아키텍처** | Single-stage ViT | Landmark + Rule-based | 2-Stage (Detector + ViT) |
| **추론 방식** | 전체 프레임 / 고정 ROI | 관절 좌표 (21 Keypoints) | 동적 바운딩 박스 크롭 |
| **배경 민감도** | 극도로 취약 (살색 면적 편향) | 영향 없음 (배경 무시) | 강건함 (손 영역만 분리 전달) |
| **외부 의존성** | PyTorch, timm | MediaPipe | Ultralytics, PyTorch, timm |
| **주요 의의** | 문제점(Shortcut) 규명 | 즉시 상용화 가능한 해결책 | 딥러닝 기반 정밀 컴퓨터 비전 파이프라인 완성 |