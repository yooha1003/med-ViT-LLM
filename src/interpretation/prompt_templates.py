"""
Prompt Templates for Various LLM Models
Supports different prompt formats for HuggingFace models
"""

from typing import Dict, Optional
import re


class PromptTemplate:
    """Base class for prompt templates"""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self.get_default_system_prompt()

    def get_default_system_prompt(self) -> str:
        """Get default system prompt for medical analysis"""
        return (
            "You are an expert radiologist assistant. "
            "Provide accurate, concise clinical interpretations based on MRI findings. "
            "Focus on clinically relevant observations and recommendations."
        )

    def format_prompt(
        self,
        user_content: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt according to model's expected format

        Args:
            user_content: User message content
            system_prompt: Optional custom system prompt

        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format_prompt")


class Llama2ChatTemplate(PromptTemplate):
    """Template for Llama-2-Chat models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""<s>[INST] <<SYS>>
{sys_prompt}
<</SYS>>

{user_content} [/INST]"""


class MistralInstructTemplate(PromptTemplate):
    """Template for Mistral-Instruct models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        # Mistral doesn't use explicit system prompt in the same way
        sys_prompt = system_prompt or self.system_prompt
        return f"""<s>[INST] {sys_prompt}

{user_content} [/INST]"""


class ChatMLTemplate(PromptTemplate):
    """Template for ChatML format (used by many models like Qwen, Yi, etc.)"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""<|im_start|>system
{sys_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


class Phi2Template(PromptTemplate):
    """Template for Phi-2 (base model, no special format)"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""### System:
{sys_prompt}

### User:
{user_content}

### Assistant:
"""


class Phi3InstructTemplate(PromptTemplate):
    """Template for Phi-3-Instruct models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""<|system|>
{sys_prompt}<|end|>
<|user|>
{user_content}<|end|>
<|assistant|>
"""


class ZephyrTemplate(PromptTemplate):
    """Template for Zephyr models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""<|system|>
{sys_prompt}</s>
<|user|>
{user_content}</s>
<|assistant|>
"""


class VicunaTemplate(PromptTemplate):
    """Template for Vicuna models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

SYSTEM: {sys_prompt}

USER: {user_content}
ASSISTANT:"""


class AlpacaTemplate(PromptTemplate):
    """Template for Alpaca-style models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sys_prompt}

{user_content}

### Response:
"""


class GemmaInstructTemplate(PromptTemplate):
    """Template for Gemma-Instruct models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""<start_of_turn>user
{sys_prompt}

{user_content}<end_of_turn>
<start_of_turn>model
"""


class GenericChatTemplate(PromptTemplate):
    """Generic template for models with built-in chat template"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        # This will be handled by the tokenizer's chat template
        # Return a dict that can be used with apply_chat_template
        return {
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": user_content}
            ]
        }


class BaseModelTemplate(PromptTemplate):
    """Simple template for base (non-instruct) models"""

    def format_prompt(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        sys_prompt = system_prompt or self.system_prompt
        return f"""{sys_prompt}

{user_content}

"""


# Model name patterns to template mapping
MODEL_TEMPLATES = {
    # Llama family
    r"llama-2.*chat": Llama2ChatTemplate,
    r"llama-3.*instruct": ChatMLTemplate,
    r"codellama.*instruct": Llama2ChatTemplate,

    # Mistral family
    r"mistral.*instruct": MistralInstructTemplate,
    r"mixtral.*instruct": MistralInstructTemplate,

    # Phi family
    r"phi-2": Phi2Template,
    r"phi-3.*instruct": Phi3InstructTemplate,

    # Qwen family
    r"qwen.*chat": ChatMLTemplate,

    # Yi family
    r"yi.*chat": ChatMLTemplate,

    # Zephyr
    r"zephyr": ZephyrTemplate,

    # Vicuna
    r"vicuna": VicunaTemplate,

    # Alpaca
    r"alpaca": AlpacaTemplate,

    # Gemma
    r"gemma.*instruct": GemmaInstructTemplate,

    # Medical-specific models
    r"meditron": MistralInstructTemplate,
    r"biomistral": MistralInstructTemplate,
    r"clinical.*llama": Llama2ChatTemplate,

    # OpenBioLLM and other bio models
    r"openbio": Llama2ChatTemplate,
}


def get_prompt_template(model_name: str, use_chat_template: bool = True) -> PromptTemplate:
    """
    Automatically detect and return appropriate prompt template for a model

    Args:
        model_name: HuggingFace model name or identifier
        use_chat_template: If True, prefer using tokenizer's chat template

    Returns:
        PromptTemplate instance
    """
    model_name_lower = model_name.lower()

    # First, try to match against known patterns
    for pattern, template_class in MODEL_TEMPLATES.items():
        if re.search(pattern, model_name_lower):
            return template_class()

    # Check if it's likely an instruct/chat model
    if any(keyword in model_name_lower for keyword in ['instruct', 'chat', 'it']):
        # Default to generic chat template which will use tokenizer's template
        return GenericChatTemplate()

    # Default to base model template
    return BaseModelTemplate()


def create_medical_prompt(
    features: Dict[str, float],
    structure_name: str = "brain region",
    patient_info: Optional[Dict] = None,
    clinical_context: Optional[str] = None
) -> str:
    """
    Create medical analysis prompt content (model-agnostic)

    Args:
        features: Extracted image features
        structure_name: Anatomical structure name
        patient_info: Patient metadata
        clinical_context: Clinical context

    Returns:
        Prompt content string
    """
    content = "Based on the following MRI analysis, provide a clinical interpretation.\n\n"

    # Patient info
    if patient_info:
        content += "PATIENT INFORMATION:\n"
        for key, value in patient_info.items():
            content += f"- {key.replace('_', ' ').title()}: {value}\n"
        content += "\n"

    # Structure
    content += f"ANATOMICAL STRUCTURE: {structure_name}\n\n"

    # Clinical context
    if clinical_context:
        content += f"CLINICAL CONTEXT: {clinical_context}\n\n"

    # Features
    content += "QUANTITATIVE FINDINGS:\n"

    # Volumetric
    if 'volume_cm3' in features:
        content += f"- Volume: {features['volume_cm3']:.2f} cm³\n"
    if 'surface_area_mm2' in features:
        content += f"- Surface Area: {features['surface_area_mm2']:.2f} mm²\n"

    # Morphological
    if 'sphericity' in features:
        sphericity_desc = "spherical" if features['sphericity'] > 0.8 else \
                        "moderately spherical" if features['sphericity'] > 0.6 else "irregular"
        content += f"- Shape: {sphericity_desc} (sphericity: {features['sphericity']:.3f})\n"

    if 'elongation' in features:
        if features['elongation'] < 0.5:
            content += f"- Elongated structure (elongation: {features['elongation']:.3f})\n"

    if 'flatness' in features:
        if features['flatness'] < 0.5:
            content += f"- Flattened structure (flatness: {features['flatness']:.3f})\n"

    # Intensity
    if 'intensity_mean' in features:
        content += f"- Mean Intensity: {features['intensity_mean']:.2f}\n"
        content += f"- Intensity Std Dev: {features['intensity_std']:.2f}\n"

    # Instructions
    content += "\nProvide:\n"
    content += "1. Description of the findings\n"
    content += "2. Clinical significance\n"
    content += "3. Differential diagnoses or recommendations (if applicable)\n\n"
    content += "Keep the response concise and clinically focused."

    return content
