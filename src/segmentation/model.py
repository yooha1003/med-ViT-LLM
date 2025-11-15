"""
Segmentation Model Wrapper
Provides high-level interface for MRI segmentation
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

from .unet_3d import UNet3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationModel:
    """
    High-level wrapper for MRI segmentation models

    This class handles:
    - Model loading and initialization
    - Image preprocessing
    - Inference
    - Post-processing
    """

    def __init__(
        self,
        model_type: str = "unet3d",
        num_classes: int = 2,
        device: str = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the segmentation model

        Args:
            model_type: Type of model ('unet3d', 'nnunet', etc.)
            num_classes: Number of segmentation classes
            device: Device to run model on ('cuda' or 'cpu')
            checkpoint_path: Path to pretrained model weights
        """
        self.model_type = model_type
        self.num_classes = num_classes

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = self._create_model()

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _create_model(self):
        """Create the segmentation model based on type"""
        if self.model_type.lower() == "unet3d":
            return UNet3D(in_channels=1, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        logger.info("Checkpoint loaded successfully")

    def save_checkpoint(self, save_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess MRI image for model input

        Args:
            image: Input MRI image as numpy array

        Returns:
            Preprocessed image tensor
        """
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min())

        # Add batch and channel dimensions
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        return image.to(self.device)

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to segmentation mask

        Args:
            output: Model output tensor

        Returns:
            Segmentation mask as numpy array
        """
        # Get class predictions
        mask = torch.argmax(output, dim=1)

        # Convert to numpy
        mask = mask.cpu().numpy().squeeze()

        return mask.astype(np.uint8)

    def segment(
        self,
        image_path: Union[str, Path] = None,
        image_array: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform segmentation on MRI image

        Args:
            image_path: Path to NIfTI file
            image_array: Or provide image as numpy array

        Returns:
            Tuple of (segmentation_mask, probability_map)
        """
        # Load image
        if image_path is not None:
            nifti_img = nib.load(str(image_path))
            image = nifti_img.get_fdata()
        elif image_array is not None:
            image = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")

        logger.info(f"Image shape: {image.shape}")

        # Preprocess
        image_tensor = self.preprocess(image)

        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        # Postprocess
        mask = self.postprocess(output)
        probs = probabilities.cpu().numpy().squeeze()

        logger.info(f"Segmentation completed. Unique labels: {np.unique(mask)}")

        return mask, probs

    def segment_and_save(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        save_probabilities: bool = False
    ):
        """
        Segment image and save result

        Args:
            image_path: Path to input NIfTI file
            output_path: Path to save segmentation mask
            save_probabilities: Whether to save probability maps
        """
        # Load original image to preserve affine and header
        nifti_img = nib.load(str(image_path))

        # Perform segmentation
        mask, probs = self.segment(image_path=image_path)

        # Save mask
        mask_nifti = nib.Nifti1Image(mask, nifti_img.affine, nifti_img.header)
        nib.save(mask_nifti, str(output_path))
        logger.info(f"Segmentation saved to {output_path}")

        # Save probabilities if requested
        if save_probabilities:
            prob_path = str(output_path).replace('.nii', '_prob.nii')
            prob_nifti = nib.Nifti1Image(probs, nifti_img.affine, nifti_img.header)
            nib.save(prob_nifti, prob_path)
            logger.info(f"Probabilities saved to {prob_path}")
