"""
LLM-based Clinical Interpretation Generator
Uses open-source LLMs to generate clinical interpretations from extracted features
"""

import torch
from typing import Dict, Optional, List
import logging
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. LLM functionality will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterpreter:
    """
    Generate clinical interpretations using Large Language Models

    Supports:
    - Open-source models (LLama2, Mistral, Phi, etc.)
    - Custom prompting
    - Medical context integration
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = None,
        max_length: int = 512,
        temperature: float = 0.7,
        use_gpu: bool = True
    ):
        """
        Initialize LLM Interpreter

        Args:
            model_name: Hugging Face model name or local path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            use_gpu: Whether to use GPU if available
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for LLM functionality")

        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature

        # Set device
        if device is None:
            self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing LLM: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            raise

    def create_prompt(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for clinical interpretation

        Args:
            features: Dictionary of extracted features
            structure_name: Name of the anatomical structure
            patient_info: Optional patient metadata (age, sex, symptoms, etc.)
            clinical_context: Optional additional clinical context

        Returns:
            Formatted prompt string
        """
        # Start with instruction
        prompt = "You are an expert radiologist assistant. Based on the MRI findings below, provide a clinical interpretation.\n\n"

        # Add patient information if available
        if patient_info:
            prompt += "Patient Information:\n"
            for key, value in patient_info.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
            prompt += "\n"

        # Add structure information
        prompt += f"Target Structure: {structure_name}\n\n"

        # Add clinical context if provided
        if clinical_context:
            prompt += f"Clinical Context: {clinical_context}\n\n"

        # Add quantitative features
        prompt += "Quantitative Findings:\n"

        # Volumetric features
        if 'volume_cm3' in features:
            prompt += f"- Volume: {features['volume_cm3']:.2f} cm³\n"
        if 'surface_area_mm2' in features:
            prompt += f"- Surface Area: {features['surface_area_mm2']:.2f} mm²\n"

        # Morphological features
        if 'sphericity' in features:
            sphericity_desc = "spherical" if features['sphericity'] > 0.8 else \
                            "moderately spherical" if features['sphericity'] > 0.6 else "irregular"
            prompt += f"- Shape: {sphericity_desc} (sphericity: {features['sphericity']:.3f})\n"

        if 'elongation' in features:
            if features['elongation'] < 0.5:
                prompt += f"- Elongated structure (elongation: {features['elongation']:.3f})\n"

        if 'flatness' in features:
            if features['flatness'] < 0.5:
                prompt += f"- Flattened structure (flatness: {features['flatness']:.3f})\n"

        # Intensity features
        if 'intensity_mean' in features:
            prompt += f"- Mean Intensity: {features['intensity_mean']:.2f}\n"
            prompt += f"- Intensity Std Dev: {features['intensity_std']:.2f}\n"

        # Add instruction for output
        prompt += "\nPlease provide:\n"
        prompt += "1. A description of the findings\n"
        prompt += "2. Clinical significance\n"
        prompt += "3. Potential differential diagnoses or recommendations (if applicable)\n\n"
        prompt += "Clinical Interpretation:\n"

        return prompt

    def generate_interpretation(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate clinical interpretation using LLM

        Args:
            features: Dictionary of extracted features
            structure_name: Name of the anatomical structure
            patient_info: Optional patient metadata
            clinical_context: Optional clinical context
            custom_prompt: Use custom prompt instead of auto-generated one

        Returns:
            Generated clinical interpretation text
        """
        # Create or use custom prompt
        if custom_prompt is None:
            prompt = self.create_prompt(
                features=features,
                structure_name=structure_name,
                patient_info=patient_info,
                clinical_context=clinical_context
            )
        else:
            prompt = custom_prompt

        logger.info("Generating clinical interpretation...")
        logger.debug(f"Prompt:\n{prompt}")

        try:
            # Generate text
            outputs = self.pipe(
                prompt,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15
            )

            # Extract generated text
            generated_text = outputs[0]['generated_text']

            # Remove the prompt from output
            interpretation = generated_text[len(prompt):].strip()

            logger.info("Interpretation generated successfully")

            return interpretation

        except Exception as e:
            logger.error(f"Error generating interpretation: {e}")
            return f"Error generating interpretation: {str(e)}"

    def generate_report(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate a structured clinical report

        Args:
            features: Dictionary of extracted features
            structure_name: Name of the anatomical structure
            patient_info: Optional patient metadata
            clinical_context: Optional clinical context

        Returns:
            Dictionary containing report sections
        """
        report = {}

        # Generate main interpretation
        interpretation = self.generate_interpretation(
            features=features,
            structure_name=structure_name,
            patient_info=patient_info,
            clinical_context=clinical_context
        )

        # Add sections
        report['patient_info'] = patient_info or {}
        report['structure'] = structure_name
        report['clinical_context'] = clinical_context or "Not provided"

        # Summarize key features
        key_features = {}
        if 'volume_cm3' in features:
            key_features['volume'] = f"{features['volume_cm3']:.2f} cm³"
        if 'sphericity' in features:
            key_features['sphericity'] = f"{features['sphericity']:.3f}"
        if 'intensity_mean' in features:
            key_features['mean_intensity'] = f"{features['intensity_mean']:.2f}"

        report['key_features'] = key_features
        report['interpretation'] = interpretation

        return report

    def format_report(self, report: Dict[str, str]) -> str:
        """
        Format report as human-readable text

        Args:
            report: Report dictionary

        Returns:
            Formatted report string
        """
        formatted = "=" * 60 + "\n"
        formatted += "MEDICAL IMAGE ANALYSIS REPORT\n"
        formatted += "=" * 60 + "\n\n"

        # Patient info
        if report.get('patient_info'):
            formatted += "PATIENT INFORMATION:\n"
            for key, value in report['patient_info'].items():
                formatted += f"  {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"

        # Structure
        formatted += f"EXAMINED STRUCTURE: {report.get('structure', 'N/A')}\n\n"

        # Clinical context
        if report.get('clinical_context') and report['clinical_context'] != "Not provided":
            formatted += f"CLINICAL CONTEXT:\n  {report['clinical_context']}\n\n"

        # Key features
        if report.get('key_features'):
            formatted += "KEY QUANTITATIVE FINDINGS:\n"
            for key, value in report['key_features'].items():
                formatted += f"  {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"

        # Interpretation
        formatted += "CLINICAL INTERPRETATION:\n"
        formatted += report.get('interpretation', 'No interpretation available')
        formatted += "\n\n"

        formatted += "=" * 60 + "\n"
        formatted += "Note: This is an AI-generated analysis and should be reviewed by a qualified radiologist.\n"
        formatted += "=" * 60 + "\n"

        return formatted


class SimpleLLMInterpreter:
    """
    Simplified interpreter using rule-based approach when LLM is not available
    """

    def __init__(self):
        logger.info("Using rule-based interpreter (LLM not available)")

    def generate_interpretation(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate basic rule-based interpretation"""

        interpretation = f"Analysis of {structure_name}:\n\n"

        # Volume assessment
        if 'volume_cm3' in features:
            vol = features['volume_cm3']
            interpretation += f"The structure has a volume of {vol:.2f} cm³. "

            # Add volume-based assessment (these are example thresholds)
            if vol < 2.0:
                interpretation += "This appears to be reduced in volume, which may indicate atrophy. "
            elif vol > 10.0:
                interpretation += "This shows increased volume, which may warrant further investigation. "
            else:
                interpretation += "Volume appears within normal range. "

            interpretation += "\n\n"

        # Shape assessment
        if 'sphericity' in features:
            sph = features['sphericity']
            interpretation += f"Shape analysis reveals a sphericity of {sph:.3f}. "

            if sph > 0.8:
                interpretation += "The structure is highly spherical, suggesting normal morphology. "
            elif sph < 0.5:
                interpretation += "The structure shows irregular morphology, which may indicate pathological changes. "

            interpretation += "\n\n"

        # Intensity assessment
        if 'intensity_mean' in features and 'intensity_std' in features:
            mean_int = features['intensity_mean']
            std_int = features['intensity_std']
            interpretation += f"Signal intensity analysis shows mean intensity of {mean_int:.2f} "
            interpretation += f"with standard deviation of {std_int:.2f}. "

            if std_int / mean_int > 0.3:
                interpretation += "High intensity variation suggests heterogeneous tissue composition. "

            interpretation += "\n\n"

        interpretation += "RECOMMENDATION: These findings should be correlated with clinical presentation "
        interpretation += "and reviewed by a qualified radiologist for definitive interpretation."

        return interpretation

    def generate_report(self, *args, **kwargs):
        """Generate basic report"""
        interpretation = self.generate_interpretation(*args, **kwargs)

        return {
            'patient_info': kwargs.get('patient_info', {}),
            'structure': kwargs.get('structure_name', 'brain region'),
            'clinical_context': kwargs.get('clinical_context', 'Not provided'),
            'key_features': {},
            'interpretation': interpretation
        }

    def format_report(self, report: Dict[str, str]) -> str:
        """Format report"""
        formatted = "=" * 60 + "\n"
        formatted += "MEDICAL IMAGE ANALYSIS REPORT (Rule-based)\n"
        formatted += "=" * 60 + "\n\n"
        formatted += report.get('interpretation', '')
        formatted += "\n" + "=" * 60 + "\n"

        return formatted
