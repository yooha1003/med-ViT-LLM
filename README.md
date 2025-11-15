# MRI Diagnosis Pipeline

End-to-end AI system for MRI image analysis: from image segmentation to clinical interpretation.

## 🎯 Overview

This system provides automated analysis of MRI images through three main stages:

1. **Segmentation**: Automatically segment anatomical structures using deep learning (3D U-Net)
2. **Feature Extraction**: Extract quantitative features (volume, morphology, intensity, texture)
3. **Clinical Interpretation**: Generate clinical reports using Large Language Models

## 🏗️ System Architecture

```
MRI Image → Segmentation → Feature Extraction → LLM Interpretation → Clinical Report
```

### Key Features

- **Deep Learning Segmentation**: 3D U-Net architecture for medical image segmentation
- **Comprehensive Feature Analysis**: Volumetric, morphological, intensity, and texture features
- **Universal LLM Support**: Compatible with ANY open-source LLM from HuggingFace
- **Flexible API**: FastAPI-based REST API for integration
- **Visualization Tools**: Built-in visualization for results
- **Quantization Support**: 4-bit and 8-bit quantization for running large models efficiently

## 📁 Project Structure

```
med-ViT-LLM/
├── src/
│   ├── segmentation/          # Segmentation models
│   │   ├── unet_3d.py        # 3D U-Net implementation
│   │   └── model.py          # Segmentation wrapper
│   ├── feature_extraction/    # Feature extraction
│   │   └── extractor.py      # Feature extractor
│   ├── interpretation/        # LLM interpretation
│   │   └── llm_interpreter.py
│   ├── pipeline/              # Main pipeline
│   │   └── main_pipeline.py
│   └── utils/                 # Utilities
│       ├── config_loader.py
│       └── visualization.py
├── api/
│   └── fastapi_server.py     # REST API server
├── configs/
│   └── config.yaml           # Configuration file
├── examples/
│   └── run_diagnosis.py      # Usage examples
├── data/
│   ├── raw/                  # Raw MRI data
│   ├── processed/            # Processed data
│   └── models/               # Model checkpoints
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yooha1003/med-ViT-LLM.git
cd med-ViT-LLM
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models (optional):
```bash
# Download segmentation model checkpoint
# Place in data/models/
```

## 💻 Usage

### Command Line Interface

Basic usage:
```bash
python src/pipeline/main_pipeline.py \
    --image data/raw/patient_mri.nii.gz \
    --structure "Hippocampus" \
    --output output/patient_001
```

With all options:
```bash
python src/pipeline/main_pipeline.py \
    --image data/raw/patient_mri.nii.gz \
    --structure "Hippocampus" \
    --output output/patient_001 \
    --checkpoint data/models/unet3d_best.pth \
    --llm "microsoft/phi-2" \
    --no-gpu  # Use CPU only
```

### Python API

```python
from pipeline.main_pipeline import MRIDiagnosisPipeline

# Initialize pipeline
pipeline = MRIDiagnosisPipeline(
    segmentation_model_type="unet3d",
    llm_model_name="microsoft/phi-2",
    use_gpu=True
)

# Patient information
patient_info = {
    'age': 65,
    'sex': 'Male',
    'chief_complaint': 'Memory loss'
}

# Run analysis
results = pipeline.run(
    image_path="data/raw/patient_mri.nii.gz",
    structure_name="Hippocampus",
    patient_info=patient_info,
    output_dir="output/patient_001"
)

# Access results
print(f"Volume: {results['features']['volume_cm3']:.2f} cm³")
print(f"Interpretation: {results['interpretation']}")
```

### REST API

Start the API server:
```bash
python api/fastapi_server.py
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

Example API call:
```bash
curl -X POST "http://localhost:8000/analyze" \
    -F "file=@patient_mri.nii.gz" \
    -F "structure_name=Hippocampus" \
    -F "patient_age=65" \
    -F "patient_sex=Male"
```

## 📊 Examples

See `examples/run_diagnosis.py` for detailed examples:

1. Basic usage
2. Batch processing
3. Custom configuration
4. Visualization
5. Feature analysis
6. LLM interpretation

Run examples:
```bash
python examples/run_diagnosis.py
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

- Segmentation model parameters
- Feature extraction settings
- LLM configuration
- Device settings (GPU/CPU)
- Output options

## 📈 Supported Anatomical Structures

The pipeline can be configured to analyze various brain structures:

- **Hippocampus**: Memory and cognitive function
- **Pituitary Gland**: Endocrine system
- **Brain Tumors**: Tumor characterization
- **Ventricles**: CSF and brain atrophy
- **Corpus Callosum**: White matter integrity

## 🧪 Model Training

To train your own segmentation model:

```python
# Training script coming soon
# See notebooks/ for training examples
```

## 📊 Data Format

**Input**: NIfTI format (.nii, .nii.gz)
- 3D MRI volumes
- T1-weighted, T2-weighted, FLAIR, etc.

**Output**:
- Segmentation masks (NIfTI)
- Feature JSON files
- Clinical reports (TXT, JSON)
- Visualizations (PNG)

## 🗄️ Datasets

Recommended public datasets:

- [ADNI](http://adni.loni.usc.edu/) - Alzheimer's Disease Neuroimaging
- [HCP](https://www.humanconnectome.org/) - Human Connectome Project
- [TCIA](https://www.cancerimagingarchive.net/) - Cancer Imaging Archive
- [OpenNeuro](https://openneuro.org/) - Open neuroscience data

## 🤖 LLM Compatibility

### Universal HuggingFace Support

This pipeline is compatible with **ANY** open-source LLM from HuggingFace! The system automatically detects the model type and applies the appropriate prompt template.

### Supported Model Families

#### General-Purpose LLMs
- **Llama family**: Llama-2, Llama-3, CodeLlama
- **Mistral family**: Mistral-7B, Mixtral-8x7B, Zephyr
- **Phi family**: Phi-2, Phi-3
- **Google Gemma**: gemma-2b, gemma-7b
- **Qwen**: Qwen2-7B, Qwen2-72B
- **Yi**: Yi-6B, Yi-34B
- And any other causal language model!

#### Medical-Specific LLMs
- **Meditron**: `epfl-llm/meditron-7b` - Medical knowledge
- **BioMistral**: `BioMistral/BioMistral-7B` - Biomedical NLP
- **Clinical-Llama**: Medical domain fine-tuned
- **OpenBioLLM**: `aaditya/Llama3-OpenBioLLM-8B` - Biomedical tasks

### Example Usage with Different Models

```python
from interpretation.llm_interpreter import LLMInterpreter

# Example 1: Phi-2 (small, efficient)
interpreter = LLMInterpreter(
    model_name="microsoft/phi-2",
    use_gpu=False  # Can run on CPU
)

# Example 2: Llama-3 (high quality)
interpreter = LLMInterpreter(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=True  # Quantization for efficiency
)

# Example 3: Medical-specific
interpreter = LLMInterpreter(
    model_name="epfl-llm/meditron-7b",
    temperature=0.5  # Lower for factual medical output
)

# Example 4: Custom configuration
interpreter = LLMInterpreter(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype="float16",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95
)
```

### Quantization Support

Run large models efficiently with quantization:

```python
# 8-bit quantization (saves ~50% memory)
interpreter = LLMInterpreter(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=True
)

# 4-bit quantization (saves ~75% memory)
interpreter = LLMInterpreter(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    load_in_4bit=True
)
```

### Automatic Prompt Template Detection

The system automatically detects and applies the correct prompt format for each model:
- Llama-2 Chat format
- Mistral Instruct format
- ChatML format (Qwen, Yi)
- Gemma format
- Phi formats
- Or uses model's built-in chat template

See `examples/llm_model_examples.py` for more examples!

## 🔬 Technical Details

### Segmentation
- Architecture: 3D U-Net
- Input size: Variable (automatically padded)
- Output: Multi-class segmentation mask

### Feature Extraction
- **Volumetric**: Volume, surface area
- **Morphological**: Sphericity, elongation, flatness, compactness
- **Intensity**: Mean, std, min, max, entropy
- **Texture** (PyRadiomics): GLCM, GLRLM, GLSZM features

### LLM Interpretation
- **Universal compatibility**: ANY HuggingFace causal LM or Seq2Seq model
- **Automatic prompt formatting**: Detects model type and applies correct template
- **Quantization support**: 4-bit and 8-bit for efficient inference
- **Medical-specific models**: Optimized for medical domain
- **Rule-based fallback**: Works without LLM if needed

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- 3D U-Net: Çiçek et al. (2016)
- PyRadiomics: van Griethuysen et al. (2017)
- MONAI: Medical Open Network for AI
- Hugging Face Transformers

## 📞 Contact

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/yooha1003/med-ViT-LLM/issues)
- Email: [uschoi@kmedihub.re.kr]

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{mri_diagnosis_pipeline,
  title={MRI Diagnosis Pipeline: End-to-end Medical Image Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yooha1003/med-ViT-LLM}
}
```

## 🗺️ Roadmap

- [ ] Support for DICOM format
- [ ] Pre-trained model zoo
- [ ] Multi-modal image fusion
- [ ] Interactive web interface
- [ ] Docker container
- [ ] Cloud deployment guide
- [ ] Integration with PACS systems
