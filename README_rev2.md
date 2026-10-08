## 1. 프로젝트 개요
    * **목표**: 비전 모델 기반 실시간 가위바위보(Rock-Paper-Scissors) 제스처 분류 파이프라인 구축
    * **핵심 과제**: Clean 가상 데이터셋에서 학습된 모델의 실세계 영상(Real-World Video) 배포 시 발생하는 **도메인 시프트(Domain Shift)** 분석 및 해결
    * **최종 성과**:
    * Vision Transformer(ViT-tiny) 전이학습 및 Shortcut Learning(지름길 학습) 원인 규명
    * 기하학적 랜드마크(MediaPipe Hands 21-Keypoint) 기반 무결점 파이프라인 구현
    * 유지보수 및 협업을 고려한 2원화 디렉터리 아키텍처 리팩토링

---

## 2. 문제 해결 및 엔지니어링 과정

    ### [Step 1] ViT 전이학습 및 실시간 추론 구축
    * timm 라이브러리의 vit_tiny_patch16_224 백본 모델 채택
    * TensorFlow RPS 데이터셋(가위/바위/보) 기반 전이학습 수행 (검증 정확도 100% 달성)
    * OpenCV 기반 프레임 처리 파이프라인 및 클래스별 신뢰도 막대그래프 HUD 대시보드 구축

    ### [Step 2] 실세계 비디오 추론 시 도메인 시프트(Domain Shift) 직면
    * 실제 촬영 동영상(test_video.mp4) 추론 시, 주먹이나 가위 동작에서도 **PAPER로 예측 결과가 편향/수렴**하는 현상 발생
    * 고정 바운딩 박스(280x280)로 인한 손 크기/해상도 불일치 의심 -> 화면 해상도 비례 동적 스케일링 기능 도입

    ### [Step 3] 지름길 학습(Shortcut Learning) 디버깅 및 원인 규명
    * 모델 입력 프레임 실시간 시각화 창(What ViT Sees)을 연동하여 모델 입력값 직접 검증
    * **근본 원인 분석**:
    1. **합성 데이터셋의 한계**: 학습 데이터셋(TensorFlow RPS)은 3D 렌더링 손과 결함 없는 순백색 단색 배경([255, 255, 255])으로 구성됨.
    2. **여백 비율 편향**: 데이터셋 특성상 PAPER는 손가락이 펼쳐져 전체 캔버스를 꽉 채우고, ROCK과 SCISSORS는 흰색 여백 면적이 큼.
    3. **지름길 학습(Shortcut Learning)**: ViT의 Self-Attention 메커니즘이 손가락 형태적 특징 대신 **프레임 내 살색 픽셀의 점유 면적 및 여백 비율**을 핵심 피처로 오인함.

    ### [Step 4] 정규화 전처리 실험
    * 손바닥 타이트 크롭(Tight Crop) 및 비율 유지 캔버스 패딩(make_square_letterbox) 적용
    * 스케일 계수(0.75 -> 0.90)에 따라 PAPER 소멸 혹은 과예측이 발생하는 현상을 통해 **픽셀 수준 여백 정보에 모델이 종속되어 있음**을 엔지니어링적으로 입증

    ### [Step 5] MediaPipe 기반 기하학적 랜드마크 파이프라인 도입
    * 픽셀 색상/여백 의존성을 원천 배제하기 위해 구글 **MediaPipe Hands (21-Point Keypoint Detection)**로 아키텍처 전환
    * 21개 관절 3차원 상대 좌표를 기반으로 손가락 접힘/펼침을 계산하는 수학적 규칙 엔진(classify_rps) 설계:
    * **ROCK**: 엄지를 제외한 4개 손가락 끝(Tip)이 마디(PIP)보다 아래 위치 (모두 접힘)
    * **SCISSORS**: 검지·중지 Tip만 PIP보다 위 위치 (2개 펼침)
    * **PAPER**: 4개 손가락 Tip이 모두 PIP보다 위 위치 (모두 펼침)
    * 배경 노이즈, 조명, 여백 비율과 100% 무관하게 실시간 무결점 분류 달성

---

## 3. 솔루션 비교 분석

| 평가 항목 | Vision Transformer (vit_pipeline) | MediaPipe 랜드마크 (mediapipe_pipeline) |
| :--- | :--- | :--- |
| **입력 데이터** | RGB 픽셀 텐서 (224 x 224 x 3) | 21개 관절 3D 상대 좌표 (21 x 3) |
| **판단 기준** | 패치 임베딩 및 Attention 맵 | 기하학적 관절 벡터/좌표 조건문 |
| **배경/여백 영향** | **취약함** (배경 노이즈·여백 면적에 민감) | **완전 무관** (관절 뼈대만 추출하여 계산) |
| **조명/피부색 민감도** | 조명 변화에 따른 색조 왜곡 영향 큼 | 내부 검출기 수준에서 정규화 완료 |
| **적합 활용 분야** | 손 표면 결함 검사 등 픽셀 세부 분석 | 로봇 그리퍼 제어, 실시간 HCI 인터페이스 |

---

## 4. 프로젝트 디렉터리 구조

```text
vit-gesture-classifier/
├── src/
│   ├── vit_pipeline/               # 1. ViT 기반 픽셀 분류 파이프라인 (실험/검증)
│   │   ├── download_dataset.py     # RPS 데이터셋 다운로드
│   │   ├── train.py                # 기본 ViT 전이학습
│   │   ├── train_robust.py         # 증강 기법 적용 모델 학습
│   │   ├── compare_models.py       # 단일 이미지 추론 비교
│   │   └── inference_video.py      # 비디오 실시간 추론 + HUD + 마스크
│   │
│   └── mediapipe_pipeline/         # 2. 랜드마크 기반 기하학적 분류 파이프라인 (최종 프로덕션)
│       └── inference_mediapipe.py  # 21개 관절 기반 실시간 판별
│
├── tests/                          # 테스트 미디어 데이터
│   ├── my_hand.jpg
│   └── test_video.mp4
│
├── weights/                        # 사전 학습된 ViT 체크포인트
│   ├── vit_gesture.pth
│   └── vit_gesture_robust.pth
│
├── .gitignore                      # 대용량/불필요 파일 제외 목록
├── requirements.txt                # 종속성 패키지 목록
├── README_rev1.md                  # 이전 버전 문서
└── README_rev2.md                  # 최종 기술 보고서