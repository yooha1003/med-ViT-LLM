# Training Data Examples

This directory contains example training data in various formats.

## Files

### Segmentation Training Data

- `segmentation_metadata.json`: Example metadata file for segmentation training
  - Format: JSON
  - Contains: Patient info, image paths, label paths, demographics
  - Use with: `SegmentationDataset` class

### LLM Training Data

#### Feature-Diagnosis Format
- `llm_training_feature_diagnosis.json`: Feature-diagnosis pairs
  - Format: JSON array
  - Each entry contains: patient_info, structure, features, diagnosis
  - Use with: `MedicalDiagnosisDataset(format_type="feature_diagnosis")`

#### Instruction Format
- `llm_training_instruction_format.jsonl`: Instruction-tuning format
  - Format: JSONL (one JSON object per line)
  - Each entry contains: instruction, input, output
  - Use with: `MedicalDiagnosisDataset(format_type="instruction")`

## Usage Examples

### Loading Segmentation Data

```python
from training.data_loaders import SegmentationDataset

dataset = SegmentationDataset(
    data_dir="data/examples",
    metadata_file="segmentation_metadata.json"
)

image, label, info = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Patient: {info['patient_id']}")
```

### Loading LLM Training Data (Feature-Diagnosis)

```python
from training.data_loaders import MedicalDiagnosisDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

dataset = MedicalDiagnosisDataset(
    data_file="data/examples/llm_training_feature_diagnosis.json",
    format_type="feature_diagnosis",
    tokenizer=tokenizer
)

sample = dataset[0]
print(f"Prompt: {sample['raw_prompt'][:200]}...")
print(f"Response: {sample['raw_response'][:200]}...")
```

### Loading LLM Training Data (Instruction Format)

```python
dataset = MedicalDiagnosisDataset(
    data_file="data/examples/llm_training_instruction_format.jsonl",
    format_type="instruction",
    tokenizer=tokenizer
)

sample = dataset[0]
print(f"Instruction: {sample['raw_prompt'][:200]}...")
```

## Creating Your Own Training Data

### For Segmentation

1. Organize your MRI images and labels:
   ```
   data/
   ├── training/
   │   ├── images/
   │   │   └── *.nii.gz
   │   ├── labels/
   │   │   └── *.nii.gz
   │   └── metadata.json
   ```

2. Create metadata.json following the example format

3. Use with `SegmentationDataset` and `SegmentationTrainer`

### For LLM Fine-Tuning

#### Option 1: Feature-Diagnosis Format (Recommended)

```json
[
  {
    "patient_id": "P001",
    "patient_info": {
      "age": 65,
      "sex": "M"
    },
    "structure": "Hippocampus",
    "features": {
      "volume_cm3": 2.5,
      "sphericity": 0.70
    },
    "diagnosis": "Your expert diagnosis here..."
  }
]
```

#### Option 2: Instruction Format

```jsonl
{"instruction": "Interpret this MRI finding.", "input": "Patient data...", "output": "Diagnosis..."}
{"instruction": "Interpret this MRI finding.", "input": "Patient data...", "output": "Diagnosis..."}
```

#### Option 3: Chat Format

```jsonl
{"messages": [{"role": "system", "content": "You are a radiologist."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Data Quality Guidelines

1. **De-identification**: Remove all PHI (Protected Health Information)
2. **Consistency**: Use consistent terminology and units
3. **Completeness**: Include all relevant features and context
4. **Expert Review**: Have diagnoses reviewed by qualified radiologists
5. **Diversity**: Include varied cases (normal, abnormal, different demographics)

## Training Scripts

Use these example datasets with the training scripts:

### Segmentation Training
```bash
python src/training/seg_trainer.py \
    --train-dir data/training \
    --val-dir data/validation \
    --epochs 100 \
    --batch-size 2
```

### LLM Fine-Tuning
```bash
python src/training/llm_trainer.py \
    --model microsoft/phi-2 \
    --data-file data/examples/llm_training_feature_diagnosis.json \
    --format feature_diagnosis \
    --epochs 3 \
    --use-lora
```

## License and Ethics

- These are **example** data for demonstration only
- Real patient data must follow:
  - HIPAA compliance
  - IRB approval
  - Proper consent
  - De-identification protocols
