# download_dataset.py

import os
import urllib.request
import zipfile

DATASET_DIR = "dataset"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
ZIP_PATH = "rps.zip"

print("1. 오픈 손 제스처 데이터셋 다운로드 중...")
urllib.request.urlretrieve(URL, ZIP_PATH)

print("2. 압축 해제 중...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)

# 임시 zip 파일 삭제
os.remove(ZIP_PATH)

# 다운로드된 폴더(rps) 이름을 ViT 학습 구조에 맞게 정리
# rps 폴더 내부에 rock, paper, scissors가 자동 생성됩니다.
print("완료! dataset/rps 폴더에 데이터가 준비되었습니다.")