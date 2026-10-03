# 🤖 경량 Vision Transformer(ViT) 기반 실시간 상태/제스처 분류기

> 로보틱스 비전 파이프라인 및 ROS2 노드 확장을 염두에 둔 5시간 스프린트 프로젝트

---

## 📌 1. 프로젝트 개요
* **목표**: 사전 학습된 경량 ViT(`vit_tiny_patch16_224`)를 미세조정(Transfer Learning)하여 실시간 객체/제스처 분류기 구현
* **타겟**: 로봇 제어 명령(Stop / Go / Neutral)과 연동 가능한 비전 입력단 설계
* **개발 기간**: 5시간 스프린트

---

## 🛠️ 2. 기술 스택
* **Language**: Python 3.x
* **Deep Learning**: PyTorch, timm, torchvision
* **Computer Vision**: OpenCV
* **VCS**: Git, GitHub

---

## 🧱 3. 파이프라인 구조
1. **Input**: 웹캠 스트림 영상 캡처
2. **Preprocess**: 224x224 Resize + ImageNet 정규화
3. **Backbone**: `timm` vit_tiny_patch16_224 (사전 학습 가중치 Freeze)
4. **Classifier Head**: Linear Layer 미세조정 (3개 클래스 분류)
5. **Output**: 실시간 예측 라벨, 신뢰도(Confidence Score), 처리 FPS 표시

---

## 🧱 3-1. 파이프라인 세부 구조
  ## 🔄 End-to-End 작업 흐름 (Workflow)
  
  ```mermaid
  flowchart TD
      subgraph DataPrep ["1. 데이터 준비"]
          A[웹캠 스트림] -->|ROI 224x224 캡처| B[src/collect_data.py]
          B -->|8:2 자동 분할| C[(dataset/train, val)]
      end
  
      subgraph Training ["2. 전처리 & 모델 학습"]
          C -->|Resize & Normalize| D[PyTorch DataLoader]
          E[Pretrained ViT Backbone] -->|Backbone Freeze| F[Classifier Head 교체]
          D --> G[src/train.py 학습 루프]
          F --> G
          G -->|Best 가중치 저장| H[(weights/vit_gesture.pth)]
      end
  
      subgraph Inference ["3. 실시간 추론 & 확장"]
          I[웹캠 실시간 입력] --> J[src/inference.py]
          H -.->|가중치 로드| J
          J -->|추론 결과 포맷팅| K["{'class': 'stop', 'score': 0.96}"]
          K --> L[화면 FPS/라벨 오버레이]
          K -.->|향후 확장| M[ROS2 Topic Publisher Node]
      end
  ```
  
  ---
  
  ## 🚀 빠른 시작 가이드 (Quick Start)
  
  방문자가 코드를 바로 따라 해볼 수 있도록 실행 순서를 명시합니다:
  
  ### 1. 환경 설치
  ```bash
  pip install torch torchvision timm opencv-python matplotlib
  ```
  
  ### 2. 제스처 데이터 수집 (5분)
  키보드 `0`(neutral), `1`(stop), `2`(go)를 눌러 각 40~50장 캡처 후 `q`로 종료:
  ```bash
  python src/collect_data.py
  ```
  
  ### 3. ViT 파인튜닝 (10분)
  사전 학습된 백본을 고정하고 헤드만 미세조정하여 가중치 저장:
  ```bash
  python src/train.py
  ```
  
  ### 4. 실시간 웹캠 추론 실행
  ```bash
  python src/inference.py
  ```

---

## 📂 4. 디렉토리 구조 (예정)
```text
├── dataset/              # train / val 이미지 폴더 (.gitignore 대상)
├── weights/              # 학습 완료된 가중치 파일 (.pth)
├── src/
│   ├── collect_data.py   # 웹캠 기반 빠른 캡처 수집기
│   ├── train.py          # ViT 미세조정 학습 스크립트
│   └── inference.py      # OpenCV 실시간 웹캠 추론기
├── .gitignore
└── README.md
