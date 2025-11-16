# Training Data Format Guide

## Overview

This document describes the data formats for training both the segmentation model and the LLM interpreter.

## 1. Segmentation Model Training Data

### Directory Structure

```
data/
├── training/
│   ├── images/
│   │   ├── patient001_T1.nii.gz
│   │   ├── patient002_T1.nii.gz
│   │   └── ...
│   ├── labels/
│   │   ├── patient001_label.nii.gz
│   │   ├── patient002_label.nii.gz
│   │   └── ...
│   └── metadata.json
├── validation/
│   ├── images/
│   ├── labels/
│   └── metadata.json
└── test/
    ├── images/
    ├── labels/
    └── metadata.json
```

### Metadata Format (JSON)

```json
{
  "dataset_name": "Hippocampus Segmentation Dataset",
  "structure": "Hippocampus",
  "num_classes": 2,
  "class_names": ["background", "hippocampus"],
  "spacing": [1.0, 1.0, 1.0],
  "modality": "T1-weighted MRI",
  "samples": [
    {
      "patient_id": "patient001",
      "image": "images/patient001_T1.nii.gz",
      "label": "labels/patient001_label.nii.gz",
      "age": 65,
      "sex": "M",
      "diagnosis": "Alzheimer's disease",
      "scanner": "Siemens 3T",
      "acquisition_date": "2024-01-15"
    },
    {
      "patient_id": "patient002",
      "image": "images/patient002_T1.nii.gz",
      "label": "labels/patient002_label.nii.gz",
      "age": 58,
      "sex": "F",
      "diagnosis": "Normal",
      "scanner": "GE 3T",
      "acquisition_date": "2024-02-20"
    }
  ]
}
```

## 2. LLM Fine-Tuning Data

### Format 1: Instruction-Tuning (JSONL)

**File**: `data/llm_training/medical_diagnosis_train.jsonl`

Each line is a separate JSON object:

```jsonl
{"instruction": "Based on the following MRI analysis, provide a clinical interpretation.", "input": "PATIENT INFORMATION:\n- Age: 72\n- Sex: Female\n\nANATOMICAL STRUCTURE: Hippocampus\n\nQUANTITATIVE FINDINGS:\n- Volume: 2.3 cm³\n- Sphericity: 0.65\n- Mean Intensity: 145.3", "output": "The hippocampal volume of 2.3 cm³ is below the normal range for this age group, suggesting hippocampal atrophy. This finding, combined with the patient's age and sex, is consistent with neurodegenerative changes. The reduced sphericity (0.65) indicates some structural irregularity. These findings may be associated with early-stage Alzheimer's disease or age-related cognitive decline. Clinical correlation with cognitive assessments is recommended."}
{"instruction": "Based on the following MRI analysis, provide a clinical interpretation.", "input": "PATIENT INFORMATION:\n- Age: 45\n- Sex: Male\n\nANATOMICAL STRUCTURE: Pituitary Gland\n\nQUANTITATIVE FINDINGS:\n- Volume: 1.2 cm³\n- Sphericity: 0.82\n- Mean Intensity: 178.5", "output": "The pituitary gland demonstrates increased volume at 1.2 cm³, which is above the normal range (0.4-0.8 cm³). The high sphericity (0.82) suggests a well-defined, rounded structure. This finding may indicate pituitary hyperplasia or a small adenoma. Clinical correlation with endocrine laboratory values (prolactin, growth hormone, ACTH) and visual field testing is essential. Further evaluation with dedicated pituitary MRI sequences is recommended."}
```

### Format 2: Feature-Diagnosis Pairs (JSON)

**File**: `data/llm_training/feature_diagnosis_pairs.json`

```json
[
  {
    "patient_id": "P001",
    "patient_info": {
      "age": 72,
      "sex": "Female",
      "chief_complaint": "Memory loss",
      "clinical_history": "Progressive cognitive decline over 2 years"
    },
    "structure": "Hippocampus",
    "features": {
      "volume_cm3": 2.3,
      "surface_area_mm2": 1450.5,
      "sphericity": 0.65,
      "elongation": 0.72,
      "flatness": 0.58,
      "intensity_mean": 145.3,
      "intensity_std": 23.8,
      "intensity_entropy": 3.2
    },
    "diagnosis": "Hippocampal atrophy consistent with early-stage Alzheimer's disease. The volume of 2.3 cm³ is below the normal range for age and sex. Recommend clinical correlation with cognitive assessments and consider follow-up imaging in 12 months.",
    "radiologist_name": "Dr. Smith",
    "report_date": "2024-01-15"
  },
  {
    "patient_id": "P002",
    "patient_info": {
      "age": 45,
      "sex": "Male",
      "chief_complaint": "Headaches and vision changes"
    },
    "structure": "Pituitary Gland",
    "features": {
      "volume_cm3": 1.2,
      "surface_area_mm2": 892.3,
      "sphericity": 0.82,
      "intensity_mean": 178.5,
      "intensity_std": 15.2
    },
    "diagnosis": "Pituitary enlargement, possibly representing microadenoma or hyperplasia. Endocrine workup including prolactin, growth hormone, and ACTH levels recommended. Visual field testing advised.",
    "radiologist_name": "Dr. Johnson",
    "report_date": "2024-02-20"
  }
]
```

### Format 3: Chat Format (JSONL)

**File**: `data/llm_training/chat_format.jsonl`

```jsonl
{"messages": [{"role": "system", "content": "You are an expert radiologist assistant. Provide accurate clinical interpretations based on MRI findings."}, {"role": "user", "content": "Patient: 72-year-old female\nStructure: Hippocampus\nVolume: 2.3 cm³\nSphericity: 0.65\n\nWhat is your interpretation?"}, {"role": "assistant", "content": "The hippocampal volume of 2.3 cm³ is reduced for this patient's age, indicating atrophy. This is consistent with neurodegenerative changes, possibly early Alzheimer's disease. Recommend clinical correlation."}]}
{"messages": [{"role": "system", "content": "You are an expert radiologist assistant. Provide accurate clinical interpretations based on MRI findings."}, {"role": "user", "content": "Patient: 45-year-old male\nStructure: Pituitary Gland\nVolume: 1.2 cm³\nSphericity: 0.82\n\nWhat is your interpretation?"}, {"role": "assistant", "content": "The pituitary gland shows enlargement (1.2 cm³ vs normal 0.4-0.8 cm³), suggesting possible adenoma or hyperplasia. Endocrine evaluation and visual field testing recommended."}]}
```

### Format 4: Structured Medical Report (JSON)

**File**: `data/llm_training/structured_reports.json`

```json
[
  {
    "case_id": "CASE001",
    "imaging_date": "2024-01-15",
    "patient": {
      "id": "P001",
      "age": 72,
      "sex": "F",
      "clinical_indication": "Memory impairment"
    },
    "examination": {
      "type": "Brain MRI",
      "protocol": "T1-weighted MPRAGE",
      "structure_analyzed": "Hippocampus"
    },
    "quantitative_analysis": {
      "volume_cm3": 2.3,
      "volume_percentile": 5,
      "surface_area_mm2": 1450.5,
      "sphericity": 0.65,
      "morphology_score": "irregular",
      "intensity_metrics": {
        "mean": 145.3,
        "std": 23.8,
        "entropy": 3.2
      }
    },
    "interpretation": {
      "findings": "Bilateral hippocampal atrophy with volume measuring 2.3 cm³, which is in the 5th percentile for age and sex.",
      "impression": "Hippocampal volume loss consistent with neurodegenerative process, likely representing early-stage Alzheimer's disease.",
      "recommendations": [
        "Clinical correlation with cognitive assessment",
        "Consider CSF biomarkers",
        "Follow-up MRI in 12 months to assess progression"
      ],
      "differential_diagnoses": [
        "Alzheimer's disease (most likely)",
        "Vascular dementia",
        "Normal age-related changes (less likely given degree of atrophy)"
      ]
    },
    "radiologist": {
      "name": "Dr. Emily Smith",
      "credentials": "MD, FRCR",
      "specialty": "Neuroradiology"
    }
  }
]
```

## 3. CSV Format (Alternative for Simple Cases)

**File**: `data/llm_training/simple_training.csv`

```csv
patient_id,age,sex,structure,volume_cm3,sphericity,intensity_mean,diagnosis
P001,72,F,Hippocampus,2.3,0.65,145.3,"Hippocampal atrophy consistent with early Alzheimer's disease"
P002,45,M,Pituitary,1.2,0.82,178.5,"Pituitary enlargement, possible microadenoma"
P003,58,F,Hippocampus,3.5,0.78,152.1,"Normal hippocampal volume for age"
```

## 4. Dataset Statistics File

**File**: `data/llm_training/dataset_stats.json`

```json
{
  "dataset_name": "Medical MRI Diagnosis Training Set",
  "version": "1.0",
  "creation_date": "2024-11-15",
  "total_samples": 1000,
  "split": {
    "train": 800,
    "validation": 100,
    "test": 100
  },
  "structures": {
    "Hippocampus": 500,
    "Pituitary Gland": 300,
    "Brain Tumor": 200
  },
  "demographics": {
    "age_range": [18, 95],
    "age_mean": 62.5,
    "sex_distribution": {
      "M": 450,
      "F": 550
    }
  },
  "diagnoses": {
    "Normal": 300,
    "Alzheimer's disease": 250,
    "Pituitary adenoma": 200,
    "Brain tumor": 150,
    "Other": 100
  },
  "data_quality": {
    "complete_features": 980,
    "missing_diagnosis": 0,
    "verified_by_radiologist": 1000
  }
}
```

## Best Practices

### 1. Data Organization
- Keep images and labels in separate directories
- Use consistent naming conventions
- Include metadata files for each split

### 2. JSON/JSONL Guidelines
- Use JSONL for large datasets (one sample per line)
- Use JSON for smaller, structured datasets
- Include all relevant metadata

### 3. Quality Control
- Validate all JSON files before training
- Check for missing values
- Ensure label consistency
- Verify patient privacy (de-identification)

### 4. Data Versioning
- Track dataset versions
- Document any changes or updates
- Include dataset statistics

### 5. Privacy and Ethics
- Remove all PHI (Protected Health Information)
- Use anonymized patient IDs
- Follow HIPAA/GDPR guidelines
- Obtain appropriate IRB approval

## Example Usage in Training

```python
import json

# Load training data
with open('data/llm_training/feature_diagnosis_pairs.json', 'r') as f:
    training_data = json.load(f)

# Or for JSONL
training_samples = []
with open('data/llm_training/medical_diagnosis_train.jsonl', 'r') as f:
    for line in f:
        training_samples.append(json.loads(line))
```

## Recommended Format for This Project

**Primary Format**: Feature-Diagnosis Pairs (JSON)

This format provides:
- Structured feature data
- Patient context
- Expert diagnosis/interpretation
- Metadata for tracking

It's ideal for fine-tuning LLMs on medical interpretation tasks.
