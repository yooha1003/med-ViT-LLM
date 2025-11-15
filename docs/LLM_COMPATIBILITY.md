# LLM Compatibility Guide

## Overview

The MRI Diagnosis Pipeline now supports **ANY** open-source LLM from HuggingFace! The system automatically detects the model type and applies the appropriate prompt template.

## Supported Models

### 1. General-Purpose LLMs

#### Llama Family
```python
# Llama-2 (Requires HuggingFace authentication)
interpreter = LLMInterpreter("meta-llama/Llama-2-7b-chat-hf")

# Llama-3 (Requires HuggingFace authentication)
interpreter = LLMInterpreter("meta-llama/Meta-Llama-3-8B-Instruct")
```

#### Mistral Family
```python
# Mistral 7B Instruct
interpreter = LLMInterpreter("mistralai/Mistral-7B-Instruct-v0.2")

# Mixtral 8x7B (Large model - use quantization)
interpreter = LLMInterpreter(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    load_in_4bit=True
)

# Zephyr (Mistral fine-tune)
interpreter = LLMInterpreter("HuggingFaceH4/zephyr-7b-beta")
```

#### Phi Family
```python
# Phi-2 (Good for CPU)
interpreter = LLMInterpreter("microsoft/phi-2", use_gpu=False)

# Phi-3 Mini
interpreter = LLMInterpreter("microsoft/Phi-3-mini-4k-instruct")
```

#### Google Gemma
```python
# Gemma 2B
interpreter = LLMInterpreter("google/gemma-2b-it")

# Gemma 7B
interpreter = LLMInterpreter("google/gemma-7b-it")
```

#### Qwen
```python
# Qwen2 7B
interpreter = LLMInterpreter("Qwen/Qwen2-7B-Instruct")

# Qwen2 72B (Use quantization)
interpreter = LLMInterpreter(
    "Qwen/Qwen2-72B-Instruct",
    load_in_4bit=True
)
```

#### Yi
```python
# Yi 6B Chat
interpreter = LLMInterpreter("01-ai/Yi-6B-Chat")

# Yi 34B Chat
interpreter = LLMInterpreter("01-ai/Yi-34B-Chat", load_in_8bit=True)
```

### 2. Medical-Specific LLMs

These models are specifically trained or fine-tuned on medical/biomedical data:

#### Meditron
```python
# Meditron 7B - Trained on medical literature
interpreter = LLMInterpreter(
    "epfl-llm/meditron-7b",
    temperature=0.5  # Lower temp for factual medical output
)
```

#### BioMistral
```python
# BioMistral 7B - Biomedical NLP
interpreter = LLMInterpreter("BioMistral/BioMistral-7B")
```

#### OpenBioLLM
```python
# Llama-3 fine-tuned on biomedical data
interpreter = LLMInterpreter("aaditya/Llama3-OpenBioLLM-8B")
```

#### Clinical Llama
```python
# Medical domain fine-tuned
interpreter = LLMInterpreter("johnsnowlabs/JSL-MedLlama-3-8B-v2.0")
```

## Features

### 1. Automatic Prompt Template Detection

The system automatically detects and applies the correct prompt format:

```python
# Automatically uses Llama-2 chat format
interpreter = LLMInterpreter("meta-llama/Llama-2-7b-chat-hf")

# Automatically uses Mistral instruct format
interpreter = LLMInterpreter("mistralai/Mistral-7B-Instruct-v0.2")

# Automatically uses ChatML format
interpreter = LLMInterpreter("Qwen/Qwen2-7B-Instruct")

# Uses model's built-in chat template if available
interpreter = LLMInterpreter(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_chat_template=True
)
```

### 2. Quantization Support

Save memory by using quantized models:

```python
# 8-bit quantization (~50% memory reduction)
interpreter = LLMInterpreter(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_8bit=True
)

# 4-bit quantization (~75% memory reduction)
interpreter = LLMInterpreter(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    load_in_4bit=True
)
```

**Note**: Quantization requires the `bitsandbytes` library:
```bash
pip install bitsandbytes
```

### 3. Custom Generation Parameters

Fine-tune the generation behavior:

```python
interpreter = LLMInterpreter(
    model_name="microsoft/phi-2",
    max_new_tokens=512,        # Maximum tokens to generate
    temperature=0.7,           # 0.0 = deterministic, 1.0+ = creative
    top_p=0.95,               # Nucleus sampling
    top_k=50,                 # Top-k sampling
    repetition_penalty=1.15,  # Prevent repetition
    torch_dtype="float16"     # Data type (auto, float16, bfloat16, float32)
)
```

### 4. Medical-Optimized Settings

For medical applications, consider these settings:

```python
# More factual, less creative
interpreter = LLMInterpreter(
    model_name="epfl-llm/meditron-7b",
    temperature=0.3,           # Lower temperature
    repetition_penalty=1.2,    # Higher penalty
    max_new_tokens=400         # Concise outputs
)
```

## Configuration File

Update `configs/config.yaml` to change the default model:

```yaml
interpretation:
  use_llm: true
  model_name: "microsoft/phi-2"  # Change to any HuggingFace model
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.95
  top_k: 50
  repetition_penalty: 1.15

  # Quantization options
  load_in_8bit: false
  load_in_4bit: false

  # Advanced
  torch_dtype: "auto"
  trust_remote_code: true
  use_chat_template: true
```

## Model Selection Guide

### For CPU Usage
- `microsoft/phi-2` (2.7B)
- `google/gemma-2b-it` (2B)

### For GPU with Limited Memory
- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `google/gemma-7b-it` (7B) with 8-bit
- `mistralai/Mistral-7B-Instruct-v0.2` (7B) with 8-bit

### For High Quality
- `meta-llama/Meta-Llama-3-8B-Instruct` (8B)
- `Qwen/Qwen2-7B-Instruct` (7B)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (47B) with 4-bit

### For Medical Domain
- `epfl-llm/meditron-7b` (Medical literature)
- `BioMistral/BioMistral-7B` (Biomedical NLP)
- `aaditya/Llama3-OpenBioLLM-8B` (Biomedical tasks)

## Memory Requirements

Approximate VRAM requirements (without quantization):

| Model Size | FP16 | 8-bit | 4-bit |
|-----------|------|-------|-------|
| 2-3B      | 6GB  | 3GB   | 2GB   |
| 7B        | 14GB | 7GB   | 4GB   |
| 13B       | 26GB | 13GB  | 7GB   |
| 34B       | 68GB | 34GB  | 17GB  |
| 70B       | 140GB| 70GB  | 35GB  |

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"
**Solution**: Use quantization
```python
interpreter = LLMInterpreter(model_name, load_in_8bit=True)
# or
interpreter = LLMInterpreter(model_name, load_in_4bit=True)
```

### Issue: "Token 'pad_token' not found"
**Solution**: This is automatically handled. The system sets pad_token to eos_token.

### Issue: Model requires authentication
**Solution**: Login to HuggingFace
```bash
huggingface-cli login
```

### Issue: Slow generation on CPU
**Solution**: Use a smaller model
```python
interpreter = LLMInterpreter("microsoft/phi-2", use_gpu=False)
```

## Examples

See `examples/llm_model_examples.py` for comprehensive examples including:
1. Phi-2 (CPU usage)
2. Llama-3 (high quality)
3. Mistral (balanced)
4. Medical LLMs
5. Gemma (lightweight)
6. Qwen (multilingual)
7. Custom generation config
8. Model comparison
9. Rule-based fallback

## Adding New Models

To use any HuggingFace model not listed here:

1. **Find the model on HuggingFace**: Browse https://huggingface.co/models
2. **Copy the model identifier**: e.g., `username/model-name`
3. **Use it directly**:
```python
interpreter = LLMInterpreter("username/model-name")
```

The system will automatically:
- Detect if it's a causal LM or seq2seq model
- Apply the appropriate prompt template
- Set up tokenizer and generation config

That's it! No code changes needed.
