# 빠른 시작 가이드 (한국어)

## 🚀 5분 안에 시작하기

### 1단계: 환경 설정 (1분)

```bash
# 저장소 클론
git clone https://github.com/yooha1003/med-ViT-LLM.git
cd med-ViT-LLM

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2단계: 예제 실행해보기 (2분)

```bash
# LLM 해석 예제 (CPU로도 가능)
python examples/llm_model_examples.py
```

### 3단계: 내 데이터로 훈련하기 (2분)

#### 방법 A: Segmentation 훈련

```bash
# 1. 데이터 준비
mkdir -p data/training/images data/training/labels

# 2. 내 MRI 파일 복사
cp 내_이미지들/*.nii.gz data/training/images/
cp 내_라벨들/*.nii.gz data/training/labels/

# 3. metadata.json 작성 (예제 참고)
cp data/examples/segmentation_metadata.json data/training/metadata.json
# 내용 수정: 내 데이터에 맞게

# 4. 훈련 시작!
python src/training/seg_trainer.py \
    --train-dir data/training \
    --val-dir data/validation \
    --epochs 10  # 테스트용은 10 epoch
```

#### 방법 B: LLM Fine-tuning

```bash
# 1. 진단 데이터 작성 (JSON)
cat > my_training_data.json << 'EOF'
[
  {
    "patient_id": "P001",
    "patient_info": {"age": 70, "sex": "여성"},
    "structure": "해마",
    "features": {
      "volume_cm3": 2.5,
      "sphericity": 0.68
    },
    "diagnosis": "해마 부피가 경도로 감소되어 있습니다..."
  }
]
EOF

# 2. 훈련 시작!
python src/training/llm_trainer.py \
    --model microsoft/phi-2 \
    --data-file my_training_data.json \
    --format feature_diagnosis \
    --epochs 3 \
    --use-lora
```

---

## 📊 훈련 데이터 준비 치트시트

### Segmentation용 데이터

```
필요한 것:
✅ MRI 영상 파일 (.nii.gz)
✅ 분할 마스크 파일 (.nii.gz) - 0=배경, 1=목표
✅ metadata.json

폴더 구조:
data/training/
  ├── images/
  │   └── *.nii.gz
  ├── labels/
  │   └── *.nii.gz
  └── metadata.json
```

### LLM Fine-tuning용 데이터

```json
// 파일명: training_data.json
[
  {
    "patient_id": "P001",
    "patient_info": {
      "age": 65,
      "sex": "남성"
    },
    "structure": "해마",
    "features": {
      "volume_cm3": 2.3,
      "sphericity": 0.65
    },
    "diagnosis": "전문의가 작성한 진단 내용..."
  }
]
```

---

## 💡 자주 하는 질문

### Q1: GPU가 없어도 되나요?
**A**: 작은 모델은 CPU로도 가능합니다.
```bash
# CPU 사용
--device cpu

# 작은 모델 선택
--model microsoft/phi-2
```

### Q2: 데이터가 몇 개나 필요한가요?
**A**:
- Segmentation: 최소 30-50개
- LLM: 최소 100개 (더 많을수록 좋음)

### Q3: 훈련이 너무 오래 걸려요
**A**:
```bash
# Epochs 줄이기
--epochs 10

# Batch size 늘리기 (GPU 여유 있으면)
--batch-size 4

# 양자화 사용
--load-in-8bit
```

### Q4: 메모리 오류가 나요
**A**:
```bash
# Batch size 줄이기
--batch-size 1

# 양자화 사용
--load-in-4bit

# Gradient accumulation
--gradient-accumulation-steps 8
```

---

## 🎯 다음 단계

1. **테스트**: 훈련된 모델로 새 데이터 테스트
2. **평가**: 성능 측정 (Dice score, Accuracy 등)
3. **개선**: 하이퍼파라미터 튜닝
4. **배포**: API 서버로 배포

자세한 내용은:
- 📚 [전체 훈련 가이드](TRAINING_GUIDE_KO.md)
- 📖 [데이터 형식 가이드](TRAINING_DATA_FORMAT.md)
- 💻 [예제 코드](../examples/)

---

## 📞 도움이 필요하신가요?

- GitHub Issues: [문제 보고](https://github.com/yooha1003/med-ViT-LLM/issues)
- 예제 확인: `examples/` 폴더
- 샘플 데이터: `data/examples/` 폴더
