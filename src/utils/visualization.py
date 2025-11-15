"""
Visualization utilities for medical images and segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
import nibabel as nib


def visualize_segmentation(
    image: Union[np.ndarray, str, Path],
    mask: Union[np.ndarray, str, Path],
    slice_idx: Optional[int] = None,
    axis: int = 2,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (15, 5),
    alpha: float = 0.5
):
    """
    Visualize MRI image with segmentation overlay

    Args:
        image: MRI image array or path to NIfTI file
        mask: Segmentation mask array or path to NIfTI file
        slice_idx: Slice index to visualize (if None, uses middle slice)
        axis: Axis along which to slice (0, 1, or 2)
        save_path: Path to save visualization
        figsize: Figure size
        alpha: Transparency of overlay
    """
    # Load images if paths provided
    if isinstance(image, (str, Path)):
        image = nib.load(str(image)).get_fdata()
    if isinstance(mask, (str, Path)):
        mask = nib.load(str(mask)).get_fdata()

    # Select slice
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2

    # Get slices
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(mask_slice.T, cmap='jet', origin='lower')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_slice.T, cmap='gray', origin='lower')
    masked = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
    axes[2].imshow(masked, cmap='jet', alpha=alpha, origin='lower')
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_3d_slices(
    image: Union[np.ndarray, str, Path],
    mask: Optional[Union[np.ndarray, str, Path]] = None,
    num_slices: int = 9,
    axis: int = 2,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (15, 10)
):
    """
    Visualize multiple slices of 3D volume

    Args:
        image: MRI image array or path to NIfTI file
        mask: Optional segmentation mask
        num_slices: Number of slices to show
        axis: Axis along which to slice
        save_path: Path to save visualization
        figsize: Figure size
    """
    # Load images if paths provided
    if isinstance(image, (str, Path)):
        image = nib.load(str(image)).get_fdata()
    if mask is not None and isinstance(mask, (str, Path)):
        mask = nib.load(str(mask)).get_fdata()

    # Calculate slice indices
    depth = image.shape[axis]
    slice_indices = np.linspace(depth // 10, depth - depth // 10, num_slices, dtype=int)

    # Create grid
    cols = 3
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, slice_idx in enumerate(slice_indices):
        # Get slice
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :] if mask is not None else None
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :] if mask is not None else None
        else:
            img_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx] if mask is not None else None

        # Display
        axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
        if mask_slice is not None:
            masked = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
            axes[i].imshow(masked, cmap='jet', alpha=0.5, origin='lower')
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()
