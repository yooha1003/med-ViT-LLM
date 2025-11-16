"""
Data loaders for training segmentation models and LLM fine-tuning
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """
    Dataset for 3D MRI segmentation training

    Loads MRI images and corresponding segmentation masks
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: str = "metadata.json",
        transform: Optional[callable] = None,
        augmentation: Optional[callable] = None,
        normalize: bool = True,
        cache: bool = False
    ):
        """
        Initialize Segmentation Dataset

        Args:
            data_dir: Directory containing images/ and labels/ subdirectories
            metadata_file: Name of metadata JSON file
            transform: Optional transform to apply to images
            augmentation: Optional augmentation to apply
            normalize: Whether to normalize images to [0, 1]
            cache: Whether to cache data in memory (for small datasets)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augmentation = augmentation
        self.normalize = normalize
        self.cache = cache

        # Load metadata
        metadata_path = self.data_dir / metadata_file
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.samples = self.metadata['samples']
        self.num_classes = self.metadata.get('num_classes', 2)
        self.class_names = self.metadata.get('class_names', ['background', 'foreground'])

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
        logger.info(f"Number of classes: {self.num_classes}")

        # Cache data if requested
        self.cache_data = {}
        if cache:
            logger.info("Caching dataset in memory...")
            for idx in range(len(self.samples)):
                self.cache_data[idx] = self._load_sample(idx)
            logger.info("Caching complete!")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load a single sample"""
        sample_info = self.samples[idx]

        # Load image
        image_path = self.data_dir / sample_info['image']
        image_nifti = nib.load(str(image_path))
        image = image_nifti.get_fdata().astype(np.float32)

        # Load label
        label_path = self.data_dir / sample_info['label']
        label_nifti = nib.load(str(label_path))
        label = label_nifti.get_fdata().astype(np.int64)

        # Normalize image
        if self.normalize and image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())

        return image, label, sample_info

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample

        Returns:
            image: Tensor of shape (1, D, H, W)
            label: Tensor of shape (D, H, W)
            info: Dictionary with sample metadata
        """
        # Load from cache or disk
        if self.cache and idx in self.cache_data:
            image, label, info = self.cache_data[idx]
        else:
            image, label, info = self._load_sample(idx)

        # Apply augmentation
        if self.augmentation is not None:
            image, label = self.augmentation(image, label)

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).long()

        return image, label, info


class MedicalDiagnosisDataset(Dataset):
    """
    Dataset for LLM fine-tuning on medical diagnosis

    Supports multiple input formats:
    - Feature-diagnosis pairs (JSON)
    - Instruction-tuning format (JSONL)
    - Chat format (JSONL)
    """

    def __init__(
        self,
        data_file: Union[str, Path],
        format_type: str = "feature_diagnosis",
        tokenizer = None,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize Medical Diagnosis Dataset

        Args:
            data_file: Path to data file (JSON or JSONL)
            format_type: Type of data format
                - "feature_diagnosis": Feature-diagnosis pairs
                - "instruction": Instruction-tuning format
                - "chat": Chat format
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Optional custom prompt template
        """
        self.data_file = Path(data_file)
        self.format_type = format_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

        # Load data
        self.samples = self._load_data()

        logger.info(f"Loaded {len(self.samples)} samples from {data_file}")
        logger.info(f"Format type: {format_type}")

    def _load_data(self) -> List[Dict]:
        """Load data from file"""
        if self.data_file.suffix == '.jsonl':
            # Load JSONL (one JSON per line)
            samples = []
            with open(self.data_file, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))
            return samples
        else:
            # Load JSON
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]

    def __len__(self) -> int:
        return len(self.samples)

    def _format_feature_diagnosis(self, sample: Dict) -> Tuple[str, str]:
        """Format feature-diagnosis pair into prompt and response"""
        # Build prompt
        prompt = "Based on the following MRI analysis, provide a clinical interpretation.\n\n"

        # Patient info
        if 'patient_info' in sample:
            prompt += "PATIENT INFORMATION:\n"
            for key, value in sample['patient_info'].items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
            prompt += "\n"

        # Structure
        prompt += f"ANATOMICAL STRUCTURE: {sample.get('structure', 'Unknown')}\n\n"

        # Features
        prompt += "QUANTITATIVE FINDINGS:\n"
        features = sample['features']
        if 'volume_cm3' in features:
            prompt += f"- Volume: {features['volume_cm3']:.2f} cm³\n"
        if 'sphericity' in features:
            prompt += f"- Sphericity: {features['sphericity']:.3f}\n"
        if 'intensity_mean' in features:
            prompt += f"- Mean Intensity: {features['intensity_mean']:.2f}\n"

        # Response
        response = sample['diagnosis']

        return prompt, response

    def _format_instruction(self, sample: Dict) -> Tuple[str, str]:
        """Format instruction-tuning sample"""
        prompt = sample['instruction'] + "\n\n" + sample['input']
        response = sample['output']
        return prompt, response

    def _format_chat(self, sample: Dict) -> Tuple[str, str]:
        """Format chat sample"""
        messages = sample['messages']

        # Find user and assistant messages
        prompt_parts = []
        response = ""

        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.insert(0, msg['content'])
            elif msg['role'] == 'user':
                prompt_parts.append(msg['content'])
            elif msg['role'] == 'assistant':
                response = msg['content']

        prompt = "\n\n".join(prompt_parts)

        return prompt, response

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample

        Returns:
            Dictionary with:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - labels: Tokenized output (for training)
                - raw_prompt: Original prompt text
                - raw_response: Original response text
        """
        sample = self.samples[idx]

        # Format based on type
        if self.format_type == "feature_diagnosis":
            prompt, response = self._format_feature_diagnosis(sample)
        elif self.format_type == "instruction":
            prompt, response = self._format_instruction(sample)
        elif self.format_type == "chat":
            prompt, response = self._format_chat(sample)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

        # Apply custom prompt template if provided
        if self.prompt_template:
            prompt = self.prompt_template.format(prompt=prompt)

        result = {
            'raw_prompt': prompt,
            'raw_response': response
        }

        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            # Tokenize prompt
            prompt_encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            # Tokenize response
            response_encoded = self.tokenizer(
                response,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            result['input_ids'] = prompt_encoded['input_ids'].squeeze(0)
            result['attention_mask'] = prompt_encoded['attention_mask'].squeeze(0)
            result['labels'] = response_encoded['input_ids'].squeeze(0)

        return result


def create_segmentation_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    batch_size: int = 2,
    num_workers: int = 4,
    cache: bool = False,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for segmentation

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        cache: Whether to cache data in memory
        **kwargs: Additional arguments for SegmentationDataset

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SegmentationDataset(
        train_dir,
        cache=cache,
        **kwargs
    )

    val_dataset = SegmentationDataset(
        val_dir,
        cache=cache,
        normalize=train_dataset.normalize,
        transform=train_dataset.transform
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_llm_dataloader(
    data_file: Union[str, Path],
    tokenizer,
    format_type: str = "feature_diagnosis",
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for LLM fine-tuning

    Args:
        data_file: Path to training data file
        tokenizer: HuggingFace tokenizer
        format_type: Data format type
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for MedicalDiagnosisDataset

    Returns:
        DataLoader
    """
    dataset = MedicalDiagnosisDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        format_type=format_type,
        **kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=None  # Use default collate
    )

    return dataloader
