"""
Universal LLM-based Clinical Interpretation Generator
Compatible with any HuggingFace open-source LLM model
"""

import torch
from typing import Dict, Optional, List, Union
import logging
from pathlib import Path
import re

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        GenerationConfig,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. LLM functionality will be disabled.")

from .prompt_templates import get_prompt_template, create_medical_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterpreter:
    """
    Universal LLM Interpreter for Clinical Interpretations

    Compatible with any HuggingFace model including:
    - Llama family (Llama-2, Llama-3, CodeLlama)
    - Mistral family (Mistral, Mixtral, Zephyr)
    - Phi family (Phi-2, Phi-3)
    - Qwen family
    - Yi family
    - Gemma family
    - Medical models (Meditron, BioMistral, Clinical-Llama, OpenBioLLM)
    - And any other causal LM on HuggingFace
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        use_gpu: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
        custom_prompt_template: Optional[str] = None,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize Universal LLM Interpreter

        Args:
            model_name: HuggingFace model name or local path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            use_gpu: Whether to use GPU if available
            load_in_8bit: Load model in 8-bit precision (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit precision (requires bitsandbytes)
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16', 'float32')
            trust_remote_code: Whether to trust remote code
            use_chat_template: Whether to use tokenizer's chat template if available
            custom_prompt_template: Custom prompt template name
            generation_config: Custom generation configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for LLM functionality")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.use_chat_template = use_chat_template

        # Set device
        if device is None:
            self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing LLM: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Determine torch dtype
        if torch_dtype == "auto":
            if self.device == "cuda":
                # Use float16 for GPU by default
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # Setup quantization if requested
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
                )
                logger.info(f"Using {'4-bit' if load_in_4bit else '8-bit'} quantization")
            except Exception as e:
                logger.warning(f"Quantization setup failed: {e}. Loading in full precision.")
                quantization_config = None

        # Load tokenizer
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added new pad_token: [PAD]")

            # Check if tokenizer has chat template
            self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and \
                                    self.tokenizer.chat_template is not None

        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

        # Load model
        try:
            logger.info("Loading model...")

            # Determine model type (causal LM or seq2seq)
            model_config = self._get_model_config(model_name)
            self.is_seq2seq = self._is_seq2seq_model(model_config)

            if self.is_seq2seq:
                logger.info("Detected Seq2Seq model")
                model_class = AutoModelForSeq2SeqLM
            else:
                logger.info("Detected Causal LM model")
                model_class = AutoModelForCausalLM

            # Load model
            self.model = model_class.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" and quantization_config is None else None,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True
            )

            # Move to device if not using device_map
            if self.device == "cpu" or quantization_config is not None:
                if not load_in_8bit and not load_in_4bit:
                    self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"Model loaded successfully (dtype: {self.torch_dtype})")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Setup prompt template
        if custom_prompt_template:
            self.prompt_template = custom_prompt_template
        else:
            self.prompt_template = get_prompt_template(
                model_name,
                use_chat_template=use_chat_template and self.has_chat_template
            )

        logger.info(f"Using prompt template: {self.prompt_template.__class__.__name__}")

        # Setup generation config
        if generation_config:
            self.generation_config = GenerationConfig(**generation_config)
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

    def _get_model_config(self, model_name: str):
        """Get model configuration"""
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    def _is_seq2seq_model(self, config) -> bool:
        """Check if model is seq2seq"""
        model_type = getattr(config, 'model_type', '').lower()
        return model_type in ['t5', 'bart', 'pegasus', 'mbart', 'mt5', 'led']

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
            patient_info: Optional patient metadata
            clinical_context: Optional additional clinical context

        Returns:
            Formatted prompt string
        """
        # Create medical prompt content
        content = create_medical_prompt(
            features=features,
            structure_name=structure_name,
            patient_info=patient_info,
            clinical_context=clinical_context
        )

        # Format with model-specific template
        if self.has_chat_template and self.use_chat_template:
            # Use tokenizer's chat template
            try:
                messages = [
                    {"role": "system", "content": self.prompt_template.get_default_system_prompt()},
                    {"role": "user", "content": content}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}. Using fallback.")

        # Use custom template
        formatted = self.prompt_template.format_prompt(content)

        # If template returns dict (for generic chat template), convert it
        if isinstance(formatted, dict) and 'messages' in formatted:
            # Fallback: just use the content
            return f"{self.prompt_template.get_default_system_prompt()}\n\n{content}"

        return formatted

    def generate_interpretation(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate clinical interpretation using LLM

        Args:
            features: Dictionary of extracted features
            structure_name: Name of the anatomical structure
            patient_info: Optional patient metadata
            clinical_context: Optional clinical context
            custom_prompt: Use custom prompt instead of auto-generated one
            **generation_kwargs: Additional generation parameters

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
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Merge generation configs
            gen_config = self.generation_config
            if generation_kwargs:
                gen_config = GenerationConfig(**{
                    **self.generation_config.to_dict(),
                    **generation_kwargs
                })

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract only the new generated text (remove prompt)
            if not self.is_seq2seq:
                # For causal LM, remove the prompt
                if generated_text.startswith(prompt):
                    interpretation = generated_text[len(prompt):].strip()
                else:
                    # Try to find where the response starts
                    # Look for common response markers
                    markers = ['Assistant:', 'Response:', 'Output:', '\n\n']
                    interpretation = generated_text
                    for marker in markers:
                        if marker in generated_text:
                            parts = generated_text.split(marker, 1)
                            if len(parts) > 1:
                                interpretation = parts[1].strip()
                                break
            else:
                # For seq2seq, the output is already just the response
                interpretation = generated_text.strip()

            # Clean up the interpretation
            interpretation = self._clean_output(interpretation)

            logger.info("Interpretation generated successfully")
            logger.debug(f"Generated {len(interpretation)} characters")

            return interpretation

        except Exception as e:
            logger.error(f"Error generating interpretation: {e}")
            return f"Error generating interpretation: {str(e)}"

    def _clean_output(self, text: str) -> str:
        """Clean up generated output"""
        # Remove potential repetitions at the end
        text = text.strip()

        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if sentences and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'

        return text

    def generate_report(
        self,
        features: Dict[str, float],
        structure_name: str = "brain region",
        patient_info: Optional[Dict[str, any]] = None,
        clinical_context: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, str]:
        """
        Generate a structured clinical report

        Args:
            features: Dictionary of extracted features
            structure_name: Name of the anatomical structure
            patient_info: Optional patient metadata
            clinical_context: Optional clinical context
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary containing report sections
        """
        report = {}

        # Generate main interpretation
        interpretation = self.generate_interpretation(
            features=features,
            structure_name=structure_name,
            patient_info=patient_info,
            clinical_context=clinical_context,
            **generation_kwargs
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
        report['model_used'] = self.model_name

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

        # Model info
        if report.get('model_used'):
            formatted += f"Analysis performed using: {report['model_used']}\n\n"

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
            'interpretation': interpretation,
            'model_used': 'Rule-based system'
        }

    def format_report(self, report: Dict[str, str]) -> str:
        """Format report"""
        formatted = "=" * 60 + "\n"
        formatted += "MEDICAL IMAGE ANALYSIS REPORT (Rule-based)\n"
        formatted += "=" * 60 + "\n\n"
        formatted += report.get('interpretation', '')
        formatted += "\n" + "=" * 60 + "\n"

        return formatted
