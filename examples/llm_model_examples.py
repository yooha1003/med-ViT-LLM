"""
Examples of using different LLM models from HuggingFace
Demonstrates compatibility with various open-source models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from interpretation.llm_interpreter import LLMInterpreter, SimpleLLMInterpreter


def example_features():
    """Get example medical features for testing"""
    return {
        'volume_cm3': 2.3,
        'surface_area_mm2': 1450.5,
        'sphericity': 0.65,
        'elongation': 0.72,
        'flatness': 0.58,
        'intensity_mean': 145.3,
        'intensity_std': 23.8,
        'intensity_entropy': 3.2
    }


def example_patient_info():
    """Get example patient info"""
    return {
        'age': 72,
        'sex': 'Female',
        'chief_complaint': 'Memory impairment',
        'symptoms': 'Progressive cognitive decline, difficulty with word finding'
    }


def example_with_phi2():
    """Example with Microsoft Phi-2 (small, efficient model)"""
    print("\n" + "=" * 60)
    print("Example 1: Microsoft Phi-2 (2.7B)")
    print("Good for: CPU usage, quick testing")
    print("=" * 60)

    interpreter = LLMInterpreter(
        model_name="microsoft/phi-2",
        use_gpu=False,  # Can run on CPU
        max_new_tokens=256,
        temperature=0.7
    )

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    print("\nInterpretation:")
    print(interpretation)


def example_with_llama3():
    """Example with Meta Llama-3 (requires authentication)"""
    print("\n" + "=" * 60)
    print("Example 2: Meta Llama-3-8B-Instruct")
    print("Good for: High-quality medical interpretations")
    print("Note: Requires HuggingFace authentication")
    print("=" * 60)

    try:
        interpreter = LLMInterpreter(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            use_gpu=True,
            load_in_8bit=True,  # Use quantization to save memory
            max_new_tokens=512,
            temperature=0.6
        )

        interpretation = interpreter.generate_interpretation(
            features=example_features(),
            structure_name="Hippocampus",
            patient_info=example_patient_info(),
            clinical_context="Evaluation for Alzheimer's disease"
        )

        print("\nInterpretation:")
        print(interpretation)

    except Exception as e:
        print(f"\nNote: Llama-3 requires authentication. Error: {e}")
        print("To use Llama models, run: huggingface-cli login")


def example_with_mistral():
    """Example with Mistral-7B-Instruct"""
    print("\n" + "=" * 60)
    print("Example 3: Mistral-7B-Instruct-v0.2")
    print("Good for: Balanced performance and quality")
    print("=" * 60)

    interpreter = LLMInterpreter(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        use_gpu=True,
        torch_dtype="float16",
        max_new_tokens=400,
        temperature=0.7
    )

    report = interpreter.generate_report(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    formatted = interpreter.format_report(report)
    print("\nFull Report:")
    print(formatted)


def example_with_medical_llm():
    """Example with medical-specific LLM (Meditron)"""
    print("\n" + "=" * 60)
    print("Example 4: Meditron-7B (Medical-specific model)")
    print("Good for: Medical domain knowledge")
    print("=" * 60)

    interpreter = LLMInterpreter(
        model_name="epfl-llm/meditron-7b",
        use_gpu=True,
        load_in_4bit=True,  # 4-bit quantization for large model
        max_new_tokens=512,
        temperature=0.5  # Lower temp for more factual medical output
    )

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info(),
        clinical_context="Neurodegenerative disease assessment"
    )

    print("\nMedical Interpretation:")
    print(interpretation)


def example_with_gemma():
    """Example with Google Gemma"""
    print("\n" + "=" * 60)
    print("Example 5: Google Gemma-2B-IT")
    print("Good for: Lightweight, instruction-following")
    print("=" * 60)

    interpreter = LLMInterpreter(
        model_name="google/gemma-2b-it",
        use_gpu=True,
        max_new_tokens=300,
        temperature=0.7
    )

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    print("\nInterpretation:")
    print(interpretation)


def example_with_qwen():
    """Example with Qwen2-7B-Instruct"""
    print("\n" + "=" * 60)
    print("Example 6: Qwen2-7B-Instruct")
    print("Good for: Multilingual support, strong reasoning")
    print("=" * 60)

    interpreter = LLMInterpreter(
        model_name="Qwen/Qwen2-7B-Instruct",
        use_gpu=True,
        torch_dtype="bfloat16",
        max_new_tokens=512,
        temperature=0.7
    )

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    print("\nInterpretation:")
    print(interpretation)


def example_with_custom_generation_config():
    """Example with custom generation configuration"""
    print("\n" + "=" * 60)
    print("Example 7: Custom Generation Configuration")
    print("=" * 60)

    # Define custom generation config
    custom_config = {
        'max_new_tokens': 256,
        'temperature': 0.3,  # Very low for factual output
        'top_p': 0.9,
        'top_k': 40,
        'repetition_penalty': 1.2,
        'do_sample': True
    }

    interpreter = LLMInterpreter(
        model_name="microsoft/phi-2",
        use_gpu=False,
        generation_config=custom_config
    )

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    print("\nInterpretation (with custom config):")
    print(interpretation)


def example_comparison():
    """Compare outputs from different models"""
    print("\n" + "=" * 60)
    print("Example 8: Model Comparison")
    print("=" * 60)

    models = [
        ("microsoft/phi-2", {"use_gpu": False}),
        ("google/gemma-2b-it", {"use_gpu": True}),
    ]

    features = example_features()
    patient_info = example_patient_info()

    for model_name, kwargs in models:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_name}")
        print(f"{'=' * 40}")

        try:
            interpreter = LLMInterpreter(
                model_name=model_name,
                max_new_tokens=200,
                temperature=0.7,
                **kwargs
            )

            interpretation = interpreter.generate_interpretation(
                features=features,
                structure_name="Hippocampus",
                patient_info=patient_info
            )

            print(interpretation[:500] + "..." if len(interpretation) > 500 else interpretation)

        except Exception as e:
            print(f"Error: {e}")


def example_rule_based_fallback():
    """Example using rule-based interpreter (no LLM required)"""
    print("\n" + "=" * 60)
    print("Example 9: Rule-based Interpreter (Fallback)")
    print("No LLM required - works offline")
    print("=" * 60)

    interpreter = SimpleLLMInterpreter()

    interpretation = interpreter.generate_interpretation(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    print("\nRule-based Interpretation:")
    print(interpretation)

    # Generate full report
    report = interpreter.generate_report(
        features=example_features(),
        structure_name="Hippocampus",
        patient_info=example_patient_info()
    )

    formatted = interpreter.format_report(report)
    print("\n" + formatted)


if __name__ == "__main__":
    print("\n🏥 LLM Model Examples for MRI Diagnosis")
    print("=" * 60)
    print("\nThese examples demonstrate compatibility with various")
    print("open-source LLMs from HuggingFace.")
    print("\nNote: Some examples require GPU and/or HuggingFace authentication")
    print("=" * 60)

    # Choose which examples to run
    # Uncomment the ones you want to try

    # Small models (can run on CPU)
    # example_with_phi2()
    # example_with_gemma()

    # Larger models (require GPU)
    # example_with_mistral()
    # example_with_llama3()
    # example_with_qwen()

    # Medical-specific models
    # example_with_medical_llm()

    # Advanced usage
    # example_with_custom_generation_config()
    # example_comparison()

    # No LLM required
    example_rule_based_fallback()

    print("\n" + "=" * 60)
    print("✓ Examples completed!")
    print("=" * 60)
