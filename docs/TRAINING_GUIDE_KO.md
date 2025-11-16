# MRI 진단 AI 훈련 가이드 (한국어)

## 📋 목차

1. [데이터 준비하기](#1-데이터-준비하기)
2. [Segmentation 모델 훈련](#2-segmentation-모델-훈련)
3. [LLM Fine-tuning](#3-llm-fine-tuning)
4. [전체 파이프라인 사용](#4-전체-파이프라인-사용)

---

## 1. 데이터 준비하기

### 1.1 Segmentation 훈련 데이터

#### 📁 폴더 구조

```
data/
├── training/
│   ├── images/           # MRI 영상 파일들
│   │   ├── patient001_T1.nii.gz
│   │   ├── patient002_T1.nii.gz
│   │   └── ...
│   ├── labels/           # 분할 마스크 (정답 라벨)
│   │   ├── patient001_label.nii.gz
│   │   ├── patient002_label.nii.gz
│   │   └── ...
│   └── metadata.json     # 메타데이터 파일
├── validation/
│   ├── images/
│   ├── labels/
│   └── metadata.json
└── test/
    ├── images/
    ├── labels/
    └── metadata.json
```

#### 📄 metadata.json 작성법

```json
{
  "dataset_name": "해마 분할 데이터셋",
  "structure": "Hippocampus",
  "num_classes": 2,
  "class_names": ["배경", "해마"],
  "spacing": [1.0, 1.0, 1.0],
  "modality": "T1-weighted MRI",
  "samples": [
    {
      "patient_id": "patient001",
      "image": "images/patient001_T1.nii.gz",
      "label": "labels/patient001_label.nii.gz",
      "age": 65,
      "sex": "M",
      "diagnosis": "알츠하이머병"
    },
    {
      "patient_id": "patient002",
      "image": "images/patient002_T1.nii.gz",
      "label": "labels/patient002_label.nii.gz",
      "age": 58,
      "sex": "F",
      "diagnosis": "정상"
    }
  ]
}
```

#### ⚠️ 중요 사항

1. **이미지 형식**: NIfTI 형식 (.nii 또는 .nii.gz)
2. **라벨 값**:
   - 배경 = 0
   - 목표 구조물 = 1 (또는 다중 클래스의 경우 1, 2, 3, ...)
3. **파일명 매칭**: 이미지와 라벨 파일이 서로 대응되어야 함

---

### 1.2 LLM Fine-tuning 데이터

#### 형식 1: Feature-Diagnosis (추천!)

**파일**: `llm_training_data.json`

```json
[
  {
    "patient_id": "P001",
    "patient_info": {
      "age": 72,
      "sex": "여성",
      "chief_complaint": "기억력 저하",
      "clinical_history": "2년간 점진적인 인지 저하"
    },
    "structure": "Hippocampus",
    "features": {
      "volume_cm3": 2.3,
      "surface_area_mm2": 1450.5,
      "sphericity": 0.65,
      "elongation": 0.72,
      "intensity_mean": 145.3,
      "intensity_std": 23.8
    },
    "diagnosis": "해마의 부피가 2.3 cm³으로 나이에 비해 유의하게 감소되어 있습니다. 구형도 감소(0.65)와 증가된 신장도(0.72)는 위축과 일치하는 구조적 변형을 시사합니다. 환자의 나이, 점진적인 인지 증상, 임상 병력을 종합하면 알츠하이머병 병리를 강력히 시사합니다. 권고사항: (1) 신경심리검사와의 임상적 상관관계, (2) CSF 바이오마커 분석 고려, (3) 12개월 후 추적 MRI로 진행 평가."
  },
  {
    "patient_id": "P002",
    "patient_info": {
      "age": 45,
      "sex": "남성",
      "chief_complaint": "두통과 시야 변화"
    },
    "structure": "뇌하수체",
    "features": {
      "volume_cm3": 1.2,
      "sphericity": 0.82,
      "intensity_mean": 178.5
    },
    "diagnosis": "뇌하수체가 1.2 cm³로 정상 범위(0.4-0.8 cm³)를 초과하여 비대되어 있습니다. 높은 구형도(0.82)는 경계가 명확한 종괴를 시사하며, 크기상 뇌하수체 거대선종 가능성이 높습니다. 환자의 두통 및 시야 결손 증상은 시신경교차 압박을 의미합니다. 권고사항: (1) 호르몬 검사 (프로락틴, GH, ACTH 등), (2) 시야 검사, (3) 조영증강 뇌하수체 전용 MRI."
  }
]
```

#### 형식 2: Instruction 형식

**파일**: `llm_training_instruction.jsonl`

```jsonl
{"instruction": "MRI 분석 결과를 바탕으로 임상적 해석을 제공하세요.", "input": "환자 정보:\n- 나이: 72세\n- 성별: 여성\n\n해부학적 구조: 해마\n\n정량적 소견:\n- 부피: 2.3 cm³\n- 구형도: 0.65", "output": "해마 부피 2.3 cm³는 나이에 비해 감소되어 있어 해마 위축을 시사합니다. 알츠하이머병 초기 단계 가능성이 있으며, 인지 평가와의 임상적 상관관계가 권장됩니다."}
{"instruction": "MRI 분석 결과를 바탕으로 임상적 해석을 제공하세요.", "input": "환자 정보:\n- 나이: 45세\n- 성별: 남성\n\n해부학적 구조: 뇌하수체\n\n정량적 소견:\n- 부피: 1.2 cm³\n- 구형도: 0.82", "output": "뇌하수체가 정상 범위를 초과하여 비대되어 있습니다(1.2 cm³). 뇌하수체 선종 가능성이 있으며, 호르몬 검사와 시야 검사가 필요합니다."}
```

#### 💡 데이터 작성 팁

1. **충분한 샘플**: 최소 100개 이상 권장 (더 많을수록 좋음)
2. **다양성**: 정상, 비정상, 다양한 연령대 포함
3. **전문가 검증**: 방사선과 전문의가 작성한 진단 사용
4. **개인정보 제거**: 모든 환자 식별 정보 제거 필수

---

## 2. Segmentation 모델 훈련

### 2.1 데이터 확인

```bash
# 데이터 구조 확인
ls data/training/images/
ls data/training/labels/
cat data/training/metadata.json
```

### 2.2 훈련 실행

#### 기본 훈련

```bash
python src/training/seg_trainer.py \
    --train-dir data/training \
    --val-dir data/validation \
    --output-dir output/segmentation_training \
    --epochs 100 \
    --batch-size 2 \
    --lr 0.0001 \
    --num-classes 2
```

#### GPU 선택

```bash
# GPU 0번 사용
CUDA_VISIBLE_DEVICES=0 python src/training/seg_trainer.py \
    --train-dir data/training \
    --val-dir data/validation \
    --device cuda

# CPU만 사용
python src/training/seg_trainer.py \
    --train-dir data/training \
    --val-dir data/validation \
    --device cpu
```

### 2.3 훈련 모니터링

훈련 중 출력 예시:
```
Epoch 1/100
Learning rate: 0.000100
Training: 100%|████████| 50/50 [05:23<00:00]
Train - Loss: 0.3245, Dice: 0.7821
Validation: 100%|████████| 10/10 [00:54<00:00]
Val   - Loss: 0.2876, Dice: 0.8156
✓ New best model!
```

### 2.4 훈련 결과

저장되는 파일들:
- `output/segmentation_training/best_model.pth`: 최고 성능 모델
- `output/segmentation_training/checkpoint_epoch_N.pth`: 주기적 체크포인트
- `output/segmentation_training/training_history.json`: 훈련 이력

---

## 3. LLM Fine-tuning

### 3.1 데이터 준비 확인

```python
# 데이터 로드 테스트
import json

with open('data/llm_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"총 샘플 수: {len(data)}")
print(f"첫 번째 샘플:")
print(f"  환자 ID: {data[0]['patient_id']}")
print(f"  구조: {data[0]['structure']}")
print(f"  진단 길이: {len(data[0]['diagnosis'])} 글자")
```

### 3.2 훈련 실행

#### 작은 모델 (CPU 가능)

```bash
python src/training/llm_trainer.py \
    --model microsoft/phi-2 \
    --data-file data/llm_training_data.json \
    --format feature_diagnosis \
    --output-dir output/llm_finetuned \
    --epochs 3 \
    --batch-size 4 \
    --lr 0.0002 \
    --use-lora
```

#### 큰 모델 (GPU 필요, 양자화 사용)

```bash
python src/training/llm_trainer.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --data-file data/llm_training_data.json \
    --format feature_diagnosis \
    --output-dir output/llm_finetuned \
    --epochs 3 \
    --batch-size 2 \
    --use-lora \
    --load-in-8bit
```

#### 초대형 모델 (4-bit 양자화)

```bash
python src/training/llm_trainer.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --data-file data/llm_training_data.json \
    --format feature_diagnosis \
    --output-dir output/llm_finetuned \
    --epochs 3 \
    --batch-size 1 \
    --use-lora \
    --load-in-4bit
```

### 3.3 LoRA 파라미터 조정

더 세밀한 제어가 필요한 경우:

```python
from training.llm_trainer import LLMFineTuner
from training.data_loaders import MedicalDiagnosisDataset
from transformers import AutoTokenizer

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# 데이터셋 준비
train_dataset = MedicalDiagnosisDataset(
    data_file="data/llm_training_data.json",
    format_type="feature_diagnosis",
    tokenizer=tokenizer,
    max_length=512
)

# Fine-tuner 생성 (LoRA 파라미터 커스터마이징)
finetuner = LLMFineTuner(
    model_name="microsoft/phi-2",
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # 실제로는 별도 eval 데이터 사용
    use_lora=True,
    lora_r=16,              # LoRA rank (높을수록 성능↑, 메모리↑)
    lora_alpha=32,          # LoRA alpha
    lora_dropout=0.05,      # Dropout
    learning_rate=2e-4,
    num_epochs=5,
    batch_size=4
)

# 훈련 시작
finetuner.train()
```

### 3.4 훈련 모니터링

훈련 중 로그:
```
Epoch 1/3
Training: 100%|████████| 25/25 [10:23<00:00]
{'loss': 1.234, 'learning_rate': 0.0002}

Evaluating: 100%|████████| 5/5 [00:54<00:00]
{'eval_loss': 1.156}

✓ Model saved to output/llm_finetuned
```

TensorBoard로 확인:
```bash
tensorboard --logdir output/llm_finetuned/logs
```

---

## 4. 전체 파이프라인 사용

### 4.1 훈련된 모델로 진단 실행

#### Segmentation + Feature + LLM 통합

```python
from pipeline.main_pipeline import MRIDiagnosisPipeline

# 파이프라인 초기화 (훈련된 모델 사용)
pipeline = MRIDiagnosisPipeline(
    segmentation_checkpoint="output/segmentation_training/best_model.pth",
    llm_model_name="output/llm_finetuned",  # Fine-tuned 모델
    use_llm=True
)

# 환자 정보
patient_info = {
    'age': 68,
    'sex': '여성',
    'chief_complaint': '기억력 감퇴',
    'clinical_history': '1년간 점진적 악화'
}

# MRI 분석 실행
results = pipeline.run(
    image_path="data/test/patient_new.nii.gz",
    structure_name="Hippocampus",
    patient_info=patient_info,
    clinical_context="치매 평가",
    output_dir="output/patient_new_analysis"
)

# 결과 확인
print("진단 보고서:")
print(results['interpretation'])
```

### 4.2 배치 처리

```python
# 여러 환자 한번에 처리
image_list = [
    "data/test/patient001.nii.gz",
    "data/test/patient002.nii.gz",
    "data/test/patient003.nii.gz"
]

results_list = pipeline.batch_process(
    image_paths=image_list,
    structure_name="Hippocampus",
    output_dir="output/batch_analysis"
)

# 결과 요약
for i, result in enumerate(results_list):
    if result['success']:
        print(f"환자 {i+1}: 분석 완료")
        print(f"  부피: {result['features']['volume_cm3']:.2f} cm³")
    else:
        print(f"환자 {i+1}: 오류 - {result['error']}")
```

---

## 5. 팁과 권장사항

### 5.1 Segmentation 훈련

| 항목 | 권장값 | 설명 |
|------|--------|------|
| 훈련 샘플 수 | 50개 이상 | 최소한의 성능 보장 |
| Batch size | 1-2 | 3D 데이터는 메모리 많이 사용 |
| Learning rate | 1e-4 | 일반적으로 안정적 |
| Epochs | 100-200 | Early stopping 사용 권장 |

### 5.2 LLM Fine-tuning

| 모델 크기 | 추천 설정 | GPU 메모리 |
|----------|----------|-----------|
| 2-3B | Full precision | 12GB |
| 7B | 8-bit quantization + LoRA | 16GB |
| 13B+ | 4-bit quantization + LoRA | 24GB+ |

### 5.3 데이터 품질

✅ **좋은 데이터**:
- 전문가가 검증한 라벨/진단
- 다양한 연령대와 병리
- 일관된 촬영 프로토콜
- 개인정보 완전 제거

❌ **피해야 할 것**:
- 라벨링 오류
- 편향된 데이터 (예: 환자만, 정상인 없음)
- 개인정보 포함
- 불완전한 메타데이터

---

## 6. 문제 해결

### GPU 메모리 부족
```bash
# Batch size 줄이기
--batch-size 1

# Gradient accumulation 사용
--gradient-accumulation-steps 8

# 양자화 사용
--load-in-8bit
# 또는
--load-in-4bit
```

### 훈련이 너무 느림
```python
# 데이터 캐싱 활성화 (작은 데이터셋)
dataset = SegmentationDataset(
    data_dir="data/training",
    cache=True  # 메모리에 캐싱
)

# Worker 수 증가
train_loader = DataLoader(
    dataset,
    num_workers=8  # CPU 코어 수에 맞게
)
```

### 과적합(Overfitting)
- Dropout 증가
- 더 많은 데이터 수집
- Data augmentation 사용
- Early stopping patience 감소

---

## 7. 다음 단계

1. ✅ 데이터 준비 완료
2. ✅ 모델 훈련 완료
3. 📊 성능 평가
4. 🔄 모델 개선
5. 🚀 실제 환경 배포

더 자세한 정보는:
- 영문 문서: `docs/TRAINING_DATA_FORMAT.md`
- 예제 데이터: `data/examples/`
- API 문서: `README.md`
