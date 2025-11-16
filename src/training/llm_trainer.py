"""
LLM Fine-tuning for medical diagnosis interpretation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import logging
from tqdm import tqdm
import json
from datetime import datetime

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers and/or peft not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFineTuner:
    """
    Fine-tune LLM for medical diagnosis interpretation

    Supports:
    - Full fine-tuning
    - LoRA (Parameter-Efficient Fine-Tuning)
    - QLoRA (Quantized LoRA)
    """

    def __init__(
        self,
        model_name: str,
        train_dataset,
        eval_dataset,
        output_dir: str = 'output/llm_finetuning',
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 512,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize LLM Fine-Tuner

        Args:
            model_name: HuggingFace model name
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            learning_rate: Learning rate
            num_epochs: Number of epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            max_length: Maximum sequence length
            load_in_8bit: Load model in 8-bit
            load_in_4bit: Load model in 4-bit
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft libraries required for fine-tuning")

        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_lora = use_lora
        self.max_length = max_length

        logger.info(f"Initializing fine-tuner for: {model_name}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        logger.info("Loading model...")
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if load_in_8bit or load_in_4bit else None,
            trust_remote_code=True
        )

        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common attention modules
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to=["tensorboard"],
            push_to_hub=False
        )

        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

    def train(self):
        """Train the model"""
        logger.info("Starting training...")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator
        )

        # Train
        train_result = trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Evaluate
        logger.info("Evaluating...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        logger.info(f"Training completed! Model saved to: {self.output_dir}")

        return train_result

    def generate_sample(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate sample output"""
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return generated


def prepare_dataset_for_training(
    data_file: str,
    tokenizer,
    format_type: str = "feature_diagnosis",
    max_length: int = 512,
    train_split: float = 0.8
):
    """
    Prepare and split dataset for training

    Args:
        data_file: Path to data file
        tokenizer: Tokenizer
        format_type: Data format type
        max_length: Max sequence length
        train_split: Train/eval split ratio

    Returns:
        train_dataset, eval_dataset
    """
    from data_loaders import MedicalDiagnosisDataset

    # Load full dataset
    full_dataset = MedicalDiagnosisDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        format_type=format_type,
        max_length=max_length
    )

    # Split
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size]
    )

    logger.info(f"Dataset split - Train: {train_size}, Eval: {eval_size}")

    return train_dataset, eval_dataset


def main():
    """Example fine-tuning script"""
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune LLM for Medical Diagnosis')
    parser.add_argument('--model', type=str, default='microsoft/phi-2', help='Model name')
    parser.add_argument('--data-file', type=str, required=True, help='Training data file')
    parser.add_argument('--format', type=str, default='feature_diagnosis',
                       choices=['feature_diagnosis', 'instruction', 'chat'],
                       help='Data format')
    parser.add_argument('--output-dir', type=str, default='output/llm_finetuning',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit')

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset_for_training(
        data_file=args.data_file,
        tokenizer=tokenizer,
        format_type=args.format
    )

    # Create fine-tuner
    finetuner = LLMFineTuner(
        model_name=args.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )

    # Train
    finetuner.train()

    # Test generation
    test_prompt = "Based on MRI analysis, the hippocampus shows volume of 2.3 cm³ in a 72-year-old female. Interpret:"
    generated = finetuner.generate_sample(test_prompt)

    print("\n" + "=" * 60)
    print("Test Generation:")
    print("=" * 60)
    print(f"Prompt: {test_prompt}")
    print(f"\nGenerated: {generated}")
    print("=" * 60)


if __name__ == "__main__":
    main()
